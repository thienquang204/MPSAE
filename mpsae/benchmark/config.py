"""Strict, torch-free configuration for an ImageNet + transfer campaign.

Training presets remain in mpsae.config. Campaign options select work; adding
a benchmark or changing report options never changes a training recipe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any

from ..config import (BACKBONES, INFERENCE_BUDGETS, METHODS, METHOD_VARIANTS, SEEDS,
                      Cell, grid, inference_budgets, training_preset,
                      validate_experiment_methods, cell_objective_overrides)
from ..io import canonical_hash
from ..timing_config import DEFAULT_ROUNDS, DEFAULT_WARMUP, timing_config

DATASET_NAMES = ("eurosat", "flowers102", "sop")
DEFAULT_TASKS = {"eurosat": "knn_classification", "flowers102": "knn_classification",
                 "sop": "retrieval"}
KNN_DEFAULTS = {"n_neighbors": [1, 20], "temperature": 0.07}
IMAGENET_DEFAULTS = {"enabled": True, "prepare": True, "train_k": None, "eval_k": None,
                     "e1": True, "e2": True, "linear_probe": True, "timing": True,
                     "probe_k": None,
                     "timing_query_batch": 512, "timing_warmup": DEFAULT_WARMUP,
                     "timing_rounds": DEFAULT_ROUNDS,
                     "timing_gallery_chunk": 32768, "timing_build_entry_cap": 4194304,
                     "timing_retry_skipped": False}
DOWNSTREAM_DEFAULTS = {"enabled": True, "datasets": ["eurosat", "sop"],
                      "train_k": None, "eval_k": None, "mode": "frozen", "revisions": {},
                      "backbones": {}, "tasks": {}, "knn": KNN_DEFAULTS}
EXECUTION_DEFAULTS = {"train_missing": True, "continue_on_error": True,
                     "retain_codes": False, "max_steps": 0,
                     "storage_mode": "legacy", "cleanup_raw": False, "precomputed_only": False,
                     "train_log_every": 10, "checkpoint_every": 100, "feature_residency": "auto"}


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


def knn_settings(value=None) -> dict[str, Any]:
    """Classifier choices are independent of the SAE sparsity budget."""
    result = _section({} if value is None else value, KNN_DEFAULTS, "downstream.knn")
    result["n_neighbors"] = sorted(_integers(result["n_neighbors"], "downstream.knn.n_neighbors"))
    temperature = result["temperature"]
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("downstream.knn.temperature must be finite and positive")
    result["temperature"] = float(temperature)
    return result


@dataclass(frozen=True)
class Campaign:
    name: str = "thesis"
    backbones: tuple[str, ...] = ("ff2048", "rn18")
    methods: tuple[str, ...] = METHODS
    seeds: tuple[int, ...] = SEEDS
    imagenet: dict[str, Any] = field(default_factory=lambda: dict(IMAGENET_DEFAULTS))
    downstream: dict[str, Any] = field(default_factory=lambda: dict(DOWNSTREAM_DEFAULTS))
    execution: dict[str, Any] = field(default_factory=lambda: dict(EXECUTION_DEFAULTS))

    def __post_init__(self) -> None:
        validate_experiment_methods(self.methods)

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
        tasks = result.downstream["tasks"]
        if not isinstance(tasks, dict) or set(tasks) - set(result.downstream["datasets"]):
            raise ValueError("downstream.tasks must map selected datasets to tasks")
        for dataset in result.downstream["datasets"]:
            result.task_for(dataset)
        result.downstream["knn"] = knn_settings(result.downstream["knn"])
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
        timing_config(**{name: result.imagenet["timing_" + name] for name in
                         ("query_batch", "warmup", "rounds", "gallery_chunk", "build_entry_cap")})
        for name in ("train_log_every", "checkpoint_every"):
            value = result.execution[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"execution.{name} must be a positive integer")
        if result.execution["feature_residency"] not in ("auto", "cuda", "cpu", "mmap"):
            raise ValueError("execution.feature_residency must be auto, cuda, cpu or mmap")
        for backbone in result.backbones:
            result.budgets("imagenet", "train_k", backbone)
            result.budgets("imagenet", "eval_k", backbone)
            result.budgets("downstream", "train_k", backbone)
            result.budgets("downstream", "eval_k", backbone)
            result.probe_budgets(backbone)
        return result

    def probe_budgets(self, backbone: str) -> tuple[int, ...] | None:
        """None preserves all-K probing; a mapping opts only named K values in.

        An omitted backbone or empty list in an explicit mapping disables its
        probes. Selection refers to inference K, independently of train K.
        The existing linear_probe switch remains the global master switch.
        """
        selection = self.imagenet.get("probe_k")
        if selection is None:
            return None
        if not isinstance(selection, dict) or set(selection) - set(self.backbones):
            raise ValueError("imagenet.probe_k must map selected backbones to K lists")
        values = selection.get(backbone, [])
        if isinstance(values, (list, tuple)) and not values:
            return ()
        values = _integers(values, f"imagenet.probe_k.{backbone}")
        evaluated = set(self.budgets("imagenet", "eval_k", backbone)) if self.imagenet["e2"] else set()
        if self.imagenet["e1"]:
            evaluated.update(self.budgets("imagenet", "train_k", backbone))
        if not set(values) <= evaluated:
            raise ValueError(f"imagenet.probe_k.{backbone} must be a subset of evaluated K values")
        return values

    def wants_probe(self, backbone: str, k: int) -> bool:
        selected = self.probe_budgets(backbone)
        return bool(self.imagenet["enabled"] and self.imagenet["linear_probe"]
                    and (selected is None or k in selected))

    def task_for(self, dataset: str) -> str:
        task = self.downstream.get("tasks", {}).get(dataset, DEFAULT_TASKS[dataset])
        if task not in ("retrieval", "knn_classification"):
            raise ValueError(f"invalid downstream task {task!r} for {dataset}")
        if dataset == "sop" and task != "retrieval":
            raise ValueError("SOP uses disjoint train/test product IDs; keep its retrieval protocol")
        return task

    def evaluation_options(self, dataset: str) -> dict[str, Any]:
        task = self.task_for(dataset)
        return {"task": task, "knn": knn_settings(self.downstream.get("knn"))
                if task == "knn_classification" else None}

    def budgets(self, section: str, key: str, backbone: str) -> tuple[int, ...]:
        spec = BACKBONES[backbone]
        value = getattr(self, section)[key]
        if isinstance(value, dict):
            if set(value) - set(self.backbones) or backbone not in value:
                raise ValueError(f"{section}.{key} mapping needs exactly the selected backbones")
            value = value[backbone]
        if value is None:
            if key == "eval_k" and backbone in INFERENCE_BUDGETS:
                return inference_budgets(backbone)
            if section == "imagenet":
                return spec.k_list if key == "eval_k" else spec.training_budgets
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
        # The optional selector must not invalidate old plans when unused.
        if payload["imagenet"].get("probe_k") is None:
            payload["imagenet"].pop("probe_k", None)
        payload["downstream"]["tasks"] = {
            name: self.task_for(name) for name in self.downstream["datasets"]}
        payload["downstream"]["knn"] = knn_settings(self.downstream.get("knn"))
        variants = {method: METHOD_VARIANTS[method] for method in self.methods
                    if method in METHOD_VARIANTS}
        if variants:
            payload["method_variants"] = variants
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
        overrides = cell_objective_overrides(self.source_cells())
        if overrides:
            payload["cell_objective_overrides"] = overrides
        payload["resolved_budgets"] = {name: {
            "imagenet_train_k": self.budgets("imagenet", "train_k", name),
            "imagenet_eval_k": self.budgets("imagenet", "eval_k", name),
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
        e1 = set()
        e2 = set()
        if self.imagenet["enabled"]:
            for cell in self.source_cells():
                if cell.k not in self.budgets("imagenet", "train_k", cell.backbone):
                    continue
                if self.imagenet["e1"]:
                    e1.add((cell, cell.k))
                if self.imagenet["e2"] and cell.k in BACKBONES[cell.backbone].e2_anchors:
                    e2.update((cell, k) for k in self.budgets("imagenet", "eval_k", cell.backbone))
        plan.update(imagenet_e1_points=len(e1), imagenet_e2_points=len(e2),
                    imagenet_shared_points=len(e1 & e2),
                    imagenet_unique_points=len(e1 | e2),
                    imagenet_probe_points=sum(self.wants_probe(c.backbone, k) for c, k in e1 | e2))
        if self.execution.get("storage_mode") == "direct_v1":
            rows = 1_281_167 + 50_000
            feature_bytes = ({n: rows * (BACKBONES[n].dim * 4 + 8) for n in self.backbones}
                             if self.imagenet["enabled"] else {})
            checkpoints = sum((BACKBONES[c.backbone].dim * BACKBONES[c.backbone].preset.hidden_dim
                               + BACKBONES[c.backbone].dim + BACKBONES[c.backbone].preset.hidden_dim) * 12
                              for c in self.source_cells())
            plan["storage_estimate"] = {"imagenet_features_bytes": feature_bytes,
                "total_imagenet_features_bytes": sum(feature_bytes.values()),
                "active_writer_upper_bytes": max(feature_bytes.values(), default=0),
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
