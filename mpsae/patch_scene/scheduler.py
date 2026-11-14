"""Spawned GPU workers; the coordinator alone aggregates tables and exports."""
import contextlib
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback

from ..io import atomic_json
from .artifacts import Hub
from .execution import TaskRunner, preparation_tasks, evaluation_tasks, record_result


def _worker(index, device, config, sources, workspace, log_dir, inbox, outbox, smoke, cpu_threads):
    try:
        import torch
        torch.set_num_threads(cpu_threads)
        if not smoke:
            torch.cuda.set_device(device)
            from ..scene.server import doctor
            with (Path(log_dir)/f'worker-{index}-startup.log').open('a', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    doctor(device)
        runner = TaskRunner(config, sources, Hub(workspace), device, smoke=smoke)
        outbox.put(('ready', index, None))
        while True:
            task = inbox.get()
            if task is None:
                return
            with (Path(log_dir)/f'{task.name}.log').open('a', encoding='utf-8', buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    print(f'[worker {index} {device}] {task.name}', flush=True)
                    runner.run(task, lambda row: outbox.put(('row', index, row)))
            outbox.put(('done', index, None))
    except BaseException:
        outbox.put(('error', index, traceback.format_exc()))


def validate_devices(devices, *, smoke=False):
    if not devices:
        raise ValueError('provide at least one device')
    import torch
    canonical = []
    for name in devices:
        device = torch.device(name)
        if smoke:
            if device.type != 'cpu':
                raise ValueError('parallel smoke uses CPU worker slots only')
            canonical.append('cpu')
        else:
            if device.type != 'cuda':
                raise ValueError('parallel real experiments require CUDA devices')
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                raise ValueError(f'cuda:{index} is unavailable; visible GPUs: {torch.cuda.device_count()}')
            canonical.append(f'cuda:{index}')
    if not smoke and len(set(canonical)) != len(canonical):
        raise ValueError('each worker needs a distinct GPU')
    return canonical


def run_parallel(config, sources, hub, campaign, encoders, datasets, devices, *, smoke=False, cpu_threads=2):
    """A prepare task unlocks its dense reference and independent train-K fits."""
    if hub.repo or hub.publish:
        raise ValueError('parallel execution currently uses local artifacts; omit --repo and --publish')
    devices = validate_devices(devices, smoke=smoke)
    if cpu_threads < 1:
        raise ValueError('cpu_threads must be positive')
    log_dir = hub.root/f'campaigns/{campaign}/logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context('spawn')
    outbox = context.Queue()
    inboxes = [context.Queue() for _ in devices]
    processes = [context.Process(target=_worker, args=(i, d, config, sources, str(hub.root), str(log_dir),
                 inboxes[i], outbox, smoke, cpu_threads), name=f'patch-scene-{i}') for i, d in enumerate(devices)]
    pending = preparation_tasks(encoders, datasets)
    busy, ready, completed = {}, set(), 0
    total = len(pending)*(2+len(config.training_ks)*3)  # prepare + dense + fits
    status_path = hub.root/f'campaigns/{campaign}/scheduler.json'
    started, heartbeat = time.monotonic(), time.monotonic()

    def status(stage, **extra):
        atomic_json(status_path, {'stage': stage, 'campaign': campaign, 'devices': devices,
            'completed_jobs': completed, 'total_jobs': total, 'queued_jobs': len(pending),
            'active': {str(i): asdict(task) for i, task in busy.items()},
            'elapsed_seconds': time.monotonic()-started, **extra})

    try:
        for process in processes:
            process.start()
        status('starting_workers')
        while pending or busy or len(ready) < len(devices):
            try:
                kind, index, payload = outbox.get(timeout=1)
            except queue.Empty:
                for i, process in enumerate(processes):
                    if process.exitcode is not None:
                        raise RuntimeError(f'worker {i} exited unexpectedly ({process.exitcode}); logs: {log_dir}')
            else:
                if kind == 'ready':
                    ready.add(index)
                elif kind == 'row':
                    record_result(hub, campaign, payload)
                elif kind == 'done':
                    task = busy.pop(index)
                    completed += 1
                    if task.stage == 'prepare':
                        pending.extend(evaluation_tasks(config, task))
                    print(f'[scheduler] {devices[index]} finished {task.name}; jobs {completed}/{total}', flush=True)
                elif kind == 'error':
                    raise RuntimeError(f'worker {index} ({devices[index]}) failed:\n{payload}\nLogs: {log_dir}')
                else:
                    raise RuntimeError(f'unknown worker event: {kind}')
            for i in sorted(ready):
                if i not in busy and pending:
                    # Prefer this worker's previous cell when available, reducing
                    # CPU-bank reloads, while still allowing work stealing.
                    previous = getattr(processes[i], '_previous_cell', None)
                    position = next((j for j, t in enumerate(pending)
                                     if t.stage != 'prepare' and (t.encoder, t.dataset) == previous), 0)
                    task = pending.pop(position)
                    busy[i] = task
                    processes[i]._previous_cell = task.encoder, task.dataset
                    inboxes[i].put(task)
                    print(f'[scheduler] {devices[i]} -> {task.name}', flush=True)
                    status('running')
            now = time.monotonic()
            if now-heartbeat >= 20:
                active = '; '.join(f'{devices[i]}: {t.name}' for i, t in sorted(busy.items()))
                print(f'[scheduler] jobs {completed}/{total}, elapsed {now-started:.0f}s | {active}', flush=True)
                status('running')
                heartbeat = now
        status('complete')
    except BaseException:
        status('interrupted_or_failed')
        raise
    finally:
        for inbox in inboxes:
            inbox.put(None)
        for i, process in enumerate(processes):
            if process.pid is None:
                continue
            if i in busy and process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for channel in [*inboxes, outbox]:
            channel.cancel_join_thread()
            channel.close()
