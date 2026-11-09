"""Strict, torch-free configuration for an ImageNet + transfer campaign.

Training presets remain in mpsae.config. Campaign options select work; adding
a benchmark or changing report options never changes a training recipe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from ..config import BACKBONES, METHODS, SEEDS, Cell, grid, training_preset
from ..io import canonical_hash

DATASET_NAMES = ("eurosat", "flowers102", "sop")
IMAGENET_DEFAULTS = {"enabled": True, "prepare": True, "train_k": None,
                     "e1": True, "e2": True, "linear_probe": True, "timing": True}
DOWNSTREAM_DEFAULTS = {"enabled": True, "datasets": ["eurosat", "sop"],
                      "train_k": None, "eval_k": None, "mode": "frozen", "revisions": {},
                      "backbones": {}}
EXECUTION_DEFAULTS = {"train_missing": True, "continue_on_error": True,
                     "retain_codes": False, "max_steps": 0,
                     "storage_mode": "legacy", "cleanup_raw": False, "precomputed_only": False}


def _section(value, defaults, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {name} options: {sorted(unknown)}")
    result = {**defaults, **value}
    for key, default in defaults.items():
        if isinstance(default, bool) and not isinstance(result[key], bool):
            raise ValueError(f"{name}.{key} must be true or false")
    return result


def _names(value, available, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    if len(set(value)) != len(value) or any(v not in available for v in value):
        raise ValueError(f"invalid or duplicate {name}: {value}; choose from {list(available)}")
    return tuple(value)


def _integers(value, name, *, minimum=1):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in value):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    if len(set(value)) != len(value):
        raise ValueError(f"duplicate values in {name}")
    return tuple(value)


@dataclass(frozen=True)
class Campaign:
    name: str = "thesis"
    backbones: tuple[str, ...] = ("ff2048", "rn18")
    methods: tuple[str, ...] = METHODS
    seeds: tuple[int, ...] = SEEDS
    imagenet: dict[str, Any] = field(default_factory=lambda: dict(IMAGENET_DEFAULTS))
    downstream: dict[str, Any] = field(default_factory=lambda: dict(DOWNSTREAM_DEFAULTS))
    execution: dict[str, Any] = field(default_factory=lambda: dict(EXECUTION_DEFAULTS))

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "Campaign":
        if not isinstance(payload, dict):
            raise ValueError("campaign must be a mapping")
        unknown = set(payload) - {"name", "backbones", "methods", "seeds", "imagenet",
                                  "downstream", "execution"}
        if unknown:
            raise ValueError(f"unknown campaign options: {sorted(unknown)}; fine-tuning is not supported")
        name = payload.get("name", "thesis")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("name must be a safe directory name (letters, digits, '_' or '-')")
        result = cls(name, _names(payload.get("backbones", ["ff2048", "rn18"]), BACKBONES, "backbones"),
                     _names(payload.get("methods", list(METHODS)), METHODS, "methods"),
                     _integers(payload.get("seeds", list(SEEDS)), "seeds", minimum=0),
                     _section(payload.get("imagenet", {}), IMAGENET_DEFAULTS, "imagenet"),
                     _section(payload.get("downstream", {}), DOWNSTREAM_DEFAULTS, "downstream"),
                     _section(payload.get("execution", {}), EXECUTION_DEFAULTS, "execution"))
        if result.downstream["mode"] != "frozen":
            raise ValueError("only downstream.mode: frozen is supported; no target-data training")
        if result.execution["storage_mode"] not in ("legacy", "direct_v1"):
            raise ValueError("execution.storage_mode must be legacy or direct_v1")
        if result.execution["cleanup_raw"] and result.execution["storage_mode"] != "direct_v1":
            raise ValueError("cleanup_raw requires storage_mode: direct_v1")
        if result.execution["precomputed_only"] and result.execution["storage_mode"] != "direct_v1":
            raise ValueError("precomputed_only requires storage_mode: direct_v1")
        if result.imagenet["enabled"] and result.imagenet["e2"] and not result.imagenet["e1"]:
            raise ValueError("imagenet.e2 requires e1: true (the anchor measurement is reused)")
        if not result.imagenet["enabled"] and not result.downstream["enabled"]:
            raise ValueError("enable at least one experiment family")
        _names(result.downstream["datasets"], DATASET_NAMES, "downstream.datasets")
        targets = result.downstream["backbones"]
        if not isinstance(targets, dict) or set(targets) - set(result.downstream["datasets"]):
            raise ValueError("downstream.backbones must map selected datasets to backbone lists")
        for dataset, names in targets.items():
            _names(names, result.backbones, f"downstream.backbones.{dataset}")
        revisions = result.downstream["revisions"]
        if not isinstance(revisions, dict) or set(revisions) - set(DATASET_NAMES):
            raise ValueError("downstream.revisions must map dataset names to HF revisions")
        if any(not isinstance(v, str) or not v.strip() for v in revisions.values()):
            raise ValueError("dataset revisions must be nonempty strings")
        maximum = result.execution["max_steps"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError("execution.max_steps must be a nonnegative integer")
        for backbone in result.backbones:
            result.budgets("imagenet", "train_k", backbone)
            result.budgets("downstream", "train_k", backbone)
            result.budgets("downstream", "eval_k", backbone)
        return result

    def budgets(self, section: str, key: str, backbone: str) -> tuple[int, ...]:
        spec = BACKBONES[backbone]
        value = getattr(self, section)[key]
        if isinstance(value, dict):
            if set(value) - set(self.backbones) or backbone not in value:
                raise ValueError(f"{section}.{key} mapping needs exactly the selected backbones")
            value = value[backbone]
        if value is None:
            if section == "imagenet":
                return spec.training_budgets
            if key == "eval_k":
                return spec.k_list[:6]
            return (8,) if 8 in spec.k_list else (spec.e2_anchors[0],)
        values = _integers(value, f"{section}.{key}")
        if any(k > spec.dim for k in values):
            raise ValueError(f"{section}.{key} exceeds {backbone}'s input width {spec.dim}")
        if key == "train_k" and not set(values) <= set(spec.training_budgets):
            raise ValueError(f"{section}.{key} for {backbone} must be in {spec.training_budgets}")
        return values

    def source_cells(self) -> list[Cell]:
        cells = []
        for backbone in self.backbones:
            budgets = set(self.budgets("imagenet", "train_k", backbone)) if self.imagenet["enabled"] else set()
            if any(backbone in self.target_backbones(name) for name in self.downstream["datasets"]):
                budgets.update(self.budgets("downstream", "train_k", backbone))
            cells.extend(grid((backbone,), self.methods, self.seeds, budgets))
        return cells

    def target_backbones(self, dataset: str) -> tuple[str, ...]:
        """Selected backbones for one target; omitted entries use all backbones."""
        if not self.downstream["enabled"] or dataset not in self.downstream["datasets"]:
            return ()
        return tuple(self.downstream.get("backbones", {}).get(dataset, self.backbones))

    def downstream_cells(self, dataset: str | None = None) -> list[Cell]:
        if not self.downstream["enabled"]:
            return []
        selected = set(self.target_backbones(dataset)) if dataset is not None else {
            backbone for name in self.downstream["datasets"] for backbone in self.target_backbones(name)}
        return [cell for cell in self.source_cells()
                if cell.backbone in selected
                and cell.k in self.budgets("downstream", "train_k", cell.backbone)]

    def resolved(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        if not payload["downstream"].get("backbones"):
            payload["downstream"].pop("backbones", None)
        if not payload["execution"]["precomputed_only"]:
            payload["execution"].pop("precomputed_only")
        # Preserve historical plan IDs when the new storage path is not selected.
        if payload["execution"]["storage_mode"] == "legacy" and not payload["execution"]["cleanup_raw"]:
            payload["execution"].pop("storage_mode")
            payload["execution"].pop("cleanup_raw")
        payload["presets"] = {name: asdict(BACKBONES[name].preset) for name in self.backbones}
        payload["method_presets"] = {
            name: {method: asdict(training_preset(method, BACKBONES[name].preset))
                   for method in self.methods}
            for name in self.backbones}
        payload["resolved_budgets"] = {name: {
            "imagenet_train_k": self.budgets("imagenet", "train_k", name),
            "downstream_train_k": self.budgets("downstream", "train_k", name),
            "downstream_eval_k": self.budgets("downstream", "eval_k", name)}
            for name in self.backbones}
        return payload

    def plan(self) -> dict[str, Any]:
        sparse = sum(len(self.budgets("downstream", "eval_k", c.backbone))
                     for name in self.downstream["datasets"] for c in self.downstream_cells(name))
        dense = sum(len(self.target_backbones(name)) for name in self.downstream["datasets"])
        plan = {"name": self.name, "mode": "frozen", "plan_id": canonical_hash(self.resolved()),
                "source_cells": len(self.source_cells()), "downstream_sparse_points": sparse,
                "downstream_dense_points": dense, "config": self.resolved()}
        if self.execution.get("storage_mode") == "direct_v1":
            rows = 1_281_167 + 50_000
            feature_bytes = {n: rows * (BACKBONES[n].dim * 4 + 8) for n in self.backbones}
            checkpoints = sum((BACKBONES[c.backbone].dim * BACKBONES[c.backbone].preset.hidden_dim
                               + BACKBONES[c.backbone].dim + BACKBONES[c.backbone].preset.hidden_dim) * 12
                              for c in self.source_cells())
            plan["storage_estimate"] = {"imagenet_features_bytes": feature_bytes,
                "total_imagenet_features_bytes": sum(feature_bytes.values()),
                "active_writer_upper_bytes": max(feature_bytes.values()),
                "model_and_adam_tensor_bytes": checkpoints, "reserve_bytes": 10 << 30,
                "excluded": "raw images, download/Arrow caches, model weights, environment, "
                            "downstream features, sparse evaluation cache, checkpoint overhead"}
        return plan


def load_campaign(path: str | Path) -> Campaign:
    path = Path(path)
    contents = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(contents)
    else:
        import yaml

        payload = yaml.safe_load(contents)
    return Campaign.from_mapping(payload)
