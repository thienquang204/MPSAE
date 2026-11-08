"""Logging, metric streams and stage bookkeeping.

Every stage writes three things:

* a human log line to the console and to ``logs/<run-name>.log``;
* a machine record to a JSONL stream, so charts and resume logic never parse
  prose;
* a stage receipt under ``state/`` recording success, duration and digests, so
  re-running the pipeline skips finished work.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

from .io import append_jsonl, atomic_json, read_json

LOGGER_NAME = "mpsae"
_CONFIGURED = False
METRIC_SINK: ContextVar[Any] = ContextVar("mpsae_metric_sink", default=None)


class _Formatter(logging.Formatter):
    """Compact, aligned, colourised when the stream is a terminal."""

    COLOURS = {"DEBUG": "\033[38;5;245m", "INFO": "", "WARNING": "\033[38;5;214m",
               "ERROR": "\033[38;5;203m", "CRITICAL": "\033[1;38;5;203m"}
    RESET = "\033[0m"

    def __init__(self, *, colour: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        stage = getattr(record, "stage", "")
        prefix = f"[{stage}] " if stage else ""
        body = f"{self.formatTime(record, self.datefmt)} {record.levelname[0]} {prefix}{record.getMessage()}"
        if record.exc_info:
            body = f"{body}\n{self.formatException(record.exc_info)}"
        if self.colour and (tint := self.COLOURS.get(record.levelname)):
            return f"{tint}{body}{self.RESET}"
        return body


def setup_logging(log_dir: str | Path, *, name: str = "pipeline",
                  level: str = "INFO", quiet: bool = False) -> Path:
    """Attach console and file handlers exactly once. Returns the log path."""
    global _CONFIGURED
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}.log"
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    if not _CONFIGURED:
        logger.propagate = False
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.WARNING if quiet else getattr(logging, level.upper(), logging.INFO))
        console.setFormatter(_Formatter(colour=sys.stderr.isatty() and not os.environ.get("NO_COLOR")))
        logger.addHandler(console)
        _CONFIGURED = True
    if not any(isinstance(handler, logging.FileHandler) and
               Path(handler.baseFilename) == path.resolve() for handler in logger.handlers):
        stream = logging.FileHandler(path, encoding="utf-8")
        stream.setLevel(logging.DEBUG)
        stream.setFormatter(_Formatter(colour=False))
        logger.addHandler(stream)
    return path


def get_logger(stage: str | None = None) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger(LOGGER_NAME), {"stage": stage or ""})


def log_environment(stage: str = "env") -> dict[str, Any]:
    """Record the facts needed to explain a number months later."""
    import platform

    facts: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        facts.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                     cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            facts["gpus"] = [
                {"index": index, "name": torch.cuda.get_device_name(index),
                 "capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
                 "total_memory": torch.cuda.get_device_properties(index).total_memory}
                for index in range(torch.cuda.device_count())]
    except Exception as error:  # torch is absent on planning machines
        facts["torch_error"] = f"{type(error).__name__}: {error}"
    for module in ("numpy", "timm", "ffcv", "matplotlib", "huggingface_hub", "datasets"):
        try:
            facts[module] = __import__(module).__version__
        except Exception:
            facts[module] = None
    log = get_logger(stage)
    log.info("python %s | torch %s | cuda %s", facts["python"], facts.get("torch"), facts.get("cuda"))
    for gpu in facts.get("gpus", []):
        log.info("gpu %s: %s (sm_%s, %.1f GiB)", gpu["index"], gpu["name"],
                 gpu["capability"].replace(".", ""), gpu["total_memory"] / 2**30)
    return facts


# ------------------------------------------------------------------ metrics

@dataclass
class MetricWriter:
    """Append-only metric stream: JSONL on disk, optional Weights & Biases."""

    path: Path
    stage: str = ""
    wandb_run: Any = None
    _count: int = field(default=0, init=False)

    def log(self, record: dict[str, Any], *, step: int | None = None,
            console: str | None = None) -> None:
        payload = {"wall": time.time(), **record}
        if step is not None:
            payload["step"] = step
        append_jsonl(self.path, payload)
        sink = METRIC_SINK.get()
        if sink is not None:
            try:
                sink(self.stage, record, step)
            except Exception:
                get_logger(self.stage).warning("Remote metric logging failed; local JSONL was saved")
        self._count += 1
        if console:
            get_logger(self.stage).info("%s", console)
        if self.wandb_run is not None:
            numeric = {key: value for key, value in record.items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)}
            self.wandb_run.log(numeric, step=step)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


def open_wandb(*, project: str | None, name: str, config: dict[str, Any],
               group: str | None = None, enabled: bool = True) -> Any:
    """Return a W&B run or ``None``. Never fails the pipeline."""
    if not enabled or not project or not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb

        return wandb.init(project=project, name=name, group=group, config=config,
                          reinit=True, settings=wandb.Settings(start_method="thread"))
    except Exception as error:
        get_logger("wandb").warning("W&B disabled (%s: %s)", type(error).__name__, error)
        return None


# ------------------------------------------------------------------- stages

@dataclass
class StageResult:
    name: str
    skipped: bool
    seconds: float
    detail: dict[str, Any]


@contextmanager
def stage(name: str, state_dir: str | Path, *, force: bool = False,
          identity: dict[str, Any] | None = None,
          description: str = "") -> Iterator[dict[str, Any]]:
    """Run a pipeline stage once.

    Yields a mutable ``detail`` dict; whatever the body puts there is stored in
    the receipt. If a receipt already records success for the same identity the
    body is skipped and ``detail["skipped"]`` is set, letting the caller reuse
    the previous outcome.
    """
    from .io import canonical_hash

    state_dir = Path(state_dir)
    receipt_path = state_dir / f"{name}.json"
    signature = canonical_hash(identity or {})
    previous = read_json(receipt_path)
    log = get_logger(name)
    detail: dict[str, Any] = {}
    if previous and previous.get("ok") and previous.get("identity") == signature and not force:
        log.info("skip (already done %s, %.1fs)", previous.get("finished", "?"), previous.get("seconds", 0))
        detail.update(previous.get("detail", {}))
        detail["skipped"] = True
        yield detail
        return
    if description:
        log.info("%s", description)
    started = time.monotonic()
    try:
        yield detail
    except BaseException as error:
        atomic_json(receipt_path, {"ok": False, "identity": signature,
                                   "error": f"{type(error).__name__}: {error}",
                                   "seconds": time.monotonic() - started,
                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        log.error("failed after %.1fs: %s: %s", time.monotonic() - started,
                  type(error).__name__, error)
        raise
    elapsed = time.monotonic() - started
    detail.pop("skipped", None)
    atomic_json(receipt_path, {"ok": True, "identity": signature, "seconds": elapsed,
                               "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "detail": detail})
    log.info("done in %s", format_duration(elapsed))


# ----------------------------------------------------------------- progress

def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 90:
        return f"{int(minutes)}m{int(rest):02d}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m"


class Progress:
    """Throttled progress logging with a rate and an ETA."""

    def __init__(self, total: int, *, stage: str, what: str = "items",
                 every_seconds: float = 30.0, every_items: int | None = None,
                 done: int = 0) -> None:
        self.total, self.done, self.what = int(total), int(done), what
        self.log = get_logger(stage)
        self.every_seconds, self.every_items = every_seconds, every_items
        self.started = time.monotonic()
        self._last_report = 0.0
        self._start_done = self.done

    def advance(self, count: int = 1, *, extra: str = "") -> None:
        self.done += count
        now = time.monotonic()
        due = (now - self._last_report >= self.every_seconds
               or self.done >= self.total
               or (self.every_items and self.done % self.every_items == 0))
        if not due:
            return
        self._last_report = now
        elapsed = max(1e-9, now - self.started)
        rate = (self.done - self._start_done) / elapsed
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        share = 100.0 * self.done / max(1, self.total)
        tail = f" | {extra}" if extra else ""
        self.log.info("%s/%s %s (%.1f%%) | %.1f/s | eta %s%s",
                      f"{self.done:,}", f"{self.total:,}", self.what, share, rate,
                      format_duration(remaining), tail)


def banner(title: str, *, stage: str = "", width: int = 72) -> None:
    log = get_logger(stage)
    log.info("%s", "=" * width)
    log.info("%s", title)
    log.info("%s", "=" * width)


def dump_config(path: str | Path, payload: dict[str, Any], *, stage: str = "") -> None:
    atomic_json(path, payload)
    get_logger(stage).debug("config -> %s", path)
    get_logger(stage).debug("%s", json.dumps(payload, sort_keys=True, default=str)[:2000])
