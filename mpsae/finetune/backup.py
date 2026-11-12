"""Bounded synchronous uploads of durable artifacts; no images or feature caches."""
import time
import math


class IncrementalBackup:
    def __init__(self, session, interval=1800):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("backup interval must be positive")
        self.session, self.interval = session, interval
        self.last_checkpoint = time.monotonic()

    def checkpoint(self, directory, step, total):
        now = time.monotonic()
        # Completed fits are uploaded together with their summary by on_task.
        if step == total or now - self.last_checkpoint < self.interval:
            return
        self.last_checkpoint = now  # throttle retries too, on a failed network
        try:
            self.session.ensure_repository()
            self.session.upload(directory)
        except Exception as error:
            self.session.warn("incremental fine-tuning checkpoint", error)

    def results(self):
        try:
            self.session.ensure_repository()
            for name in ("evaluations", "campaigns"):
                self.session.upload(self.session.root / name)
        except Exception as error:
            self.session.warn("incremental fine-tuning results", error)


def sync(config, root):
    from ..remote import RemoteSession
    from ..io import exclusive_lock, read_json
    with exclusive_lock(root / "campaign.lock", timeout=0):
        session = RemoteSession(config, root)
        session.remote_prefix = f"finetune-v1/{config.name}/"
        session.start()
        manifest = read_json(root / "reports" / config.plan()["plan_id"] / "manifest.json") or {}
        if not session.finish(scientific_success=bool(manifest.get("complete"))):
            raise RuntimeError(f"HF synchronization incomplete; inspect {session.directory / 'status.json'}")
