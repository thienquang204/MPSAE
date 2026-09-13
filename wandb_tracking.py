"""Small, optional Weights & Biases integration shared by all training runners."""

from __future__ import annotations

import argparse
import os
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_ENTITY = "tdnthienquang-home"
DEFAULT_PROJECT = "MPSAE"
_active_run: Optional[Any] = None
_defined_metric_namespaces: set[tuple[str, str]] = set()


def _default_mode() -> str:
    configured = os.environ.get("WANDB_MODE", "").strip().lower()
    if configured in {"dryrun", "offline"}:
        return "offline"
    if configured in {"disabled", "off", "false", "0"}:
        return "disabled"
    if configured in {"online", "run"}:
        return "online"
    return "online" if os.environ.get("WANDB_API_KEY") else "disabled"


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    """Add consistent W&B options without importing the optional SDK."""
    tracking = parser.add_argument_group("Weights & Biases monitoring")
    tracking.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default=_default_mode(),
        help="online when WANDB_API_KEY is set, otherwise disabled",
    )
    tracking.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
    )
    tracking.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
    )
    tracking.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_NAME", ""),
    )
    tracking.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP", ""),
    )
    tracking.add_argument(
        "--wandb-tags",
        default=os.environ.get("WANDB_TAGS", ""),
        help="comma-separated tags",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def init_wandb(
    args: argparse.Namespace,
    *,
    default_name: str,
    default_group: str,
    extra_config: Optional[Mapping[str, Any]] = None,
    tags: Sequence[str] = (),
) -> Optional[Any]:
    """Initialize tracking, resuming the run ID stored beside local results."""
    global _active_run
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B monitoring is enabled but the 'wandb' package is unavailable. "
            "Install requirements.txt or run: pip install wandb"
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id_path = output_dir / "wandb_run_id.txt"
    run_id = (
        run_id_path.read_text(encoding="utf-8").strip()
        if run_id_path.is_file()
        else ""
    )
    config: Dict[str, Any] = {
        key: _json_safe(value) for key, value in vars(args).items()
    }
    if extra_config:
        config.update(_json_safe(extra_config))
    configured_tags = [
        item.strip() for item in args.wandb_tags.split(",") if item.strip()
    ]
    init_options: Dict[str, Any] = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": args.wandb_run_name or default_name,
        "group": args.wandb_group or default_group,
        "tags": list(dict.fromkeys((*tags, *configured_tags))),
        "config": config,
        "mode": args.wandb_mode,
        "dir": str(output_dir),
    }
    if run_id and args.wandb_mode == "online":
        init_options.update({"id": run_id, "resume": "allow"})
    _active_run = wandb.init(**init_options)
    if _active_run is not None:
        run_id_path.write_text(str(_active_run.id) + "\n", encoding="utf-8")
        run_url = getattr(_active_run, "url", None)
        destination = (
            run_url
            or f"{args.wandb_entity}/{args.wandb_project}/{_active_run.id}"
        )
        print(f"W&B monitoring: {destination}", flush=True)
    return _active_run


def _as_scalar(value: Any) -> Optional[Real]:
    if isinstance(value, Real):
        return value
    if hasattr(value, "numel") and hasattr(value, "detach"):
        try:
            if value.numel() == 1:
                return value.detach().item()
        except (RuntimeError, TypeError, ValueError):
            return None
    return None


def _flatten_metrics(
    values: Mapping[str, Any], prefix: str = ""
) -> Dict[str, Real]:
    flattened: Dict[str, Real] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_metrics(value, name))
            continue
        scalar = _as_scalar(value)
        if scalar is not None:
            flattened[name] = scalar
    return flattened


def log_wandb_metrics(
    namespace: str,
    values: Mapping[str, Any],
    *,
    step_metric: Optional[str] = None,
) -> None:
    """Log numeric values under a namespace with an optional custom x-axis."""
    if _active_run is None:
        return
    payload = {
        f"{namespace}/{key}": value
        for key, value in _flatten_metrics(values).items()
    }
    if not payload:
        return
    if step_metric is not None and step_metric in values:
        definition = (namespace, step_metric)
        if definition not in _defined_metric_namespaces:
            step_name = f"{namespace}/{step_metric}"
            _active_run.define_metric(step_name)
            _active_run.define_metric(f"{namespace}/*", step_metric=step_name)
            _defined_metric_namespaces.add(definition)
    _active_run.log(payload)


def update_wandb_summary(values: Mapping[str, Any]) -> None:
    if _active_run is None:
        return
    for key, value in _flatten_metrics(values).items():
        _active_run.summary[key] = value


def finish_wandb(exit_code: int = 0) -> None:
    global _active_run
    if _active_run is not None:
        _active_run.finish(exit_code=exit_code)
        _active_run = None
        _defined_metric_namespaces.clear()
