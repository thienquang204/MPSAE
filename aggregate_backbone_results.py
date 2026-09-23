#!/usr/bin/env python3
"""Build a controlled Matryoshka and CSR/MPSAE v1/v2 architecture ablation.

The input directory must contain one subdirectory per requested backbone, each
with the ``summary.json`` written by ``csr_vs_mmpot_imagenet.py``. The output
files validate all five arms; the aggregate effect tables compare the three
primary Matryoshka/CSRv2/MPSAEv2 arms at every representation budget and across
backbones. A compact ZIP excludes feature caches for convenient transfer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SUPPORTED_BACKBONES = ("resnet18", "resnet50")
DEFAULT_ABLATION_BACKBONES = ("resnet18", "resnet50")
MATRYOSHKA = "matryoshka"
CSR_V1 = "csr"
MP_SAE_V1 = "mpsae"
CSR = "csrv2"
MP_SAE = "mpsaev2"
METHODS = (MATRYOSHKA, CSR, MP_SAE)
ALL_METHODS = (MATRYOSHKA, CSR_V1, MP_SAE_V1, CSR, MP_SAE)
CONTROLLED_CONFIG_KEYS = (
    "data_root",
    "data_backend",
    "hf_dataset_id",
    "hf_revision",
    "max_train",
    "max_val",
    "feature_batch_size",
    "workers",
    "prefetch_factor",
    "sparse_extra_topk",
    "train_k",
    "v1_train_k",
    "anneal_start_k",
    "anneal_fraction",
    "k_aux",
    "dead_steps",
    "mrl_classification_weight",
    "csrv2_main_recon_weight",
    "csrv2_multi_topk_recon_weight",
    "csrv2_aux_recon_weight",
    "csrv2_contrastive_weight",
    "mpsaev2_main_recon_weight",
    "mpsaev2_nested_recon_weight",
    "mpsaev2_aux_recon_weight",
    "mpsaev2_mmpot_weight",
    "epochs",
    "mpsaev2_extra_epochs",
    "batch_size",
    "mpsaev2_lr",
    "csrv2_lr",
    "weight_decay",
    "mrl_lr",
    "mrl_momentum",
    "ot_mass",
    "ot_eta",
    "ot_iters",
    "ot_tol",
    "ot_microbatch",
    "amp",
    "channels_last",
    "tf32",
    "device",
    "seed",
    "knn_batch_size",
    "knn_query_batch",
    "sparse_knn_query_batch",
    "knn_normalize",
    "faiss_gpu",
    "faiss_gpu_device",
)
METHOD_RESULT_FILES = (
    "summary.json",
    "comparison.csv",
    "comparison_table.md",
    "comparison_table.tex",
    "matryoshka/results.json",
    "matryoshka/history.json",
    "csr/results.json",
    "csr/history.json",
    "mpsae/results.json",
    "mpsae/history.json",
    "csrv2/results.json",
    "csrv2/history.json",
    "mpsaev2/results.json",
    "mpsaev2/history.json",
)
PORTABLE_SUFFIXES = {".csv", ".json", ".log", ".md", ".pdf", ".png", ".tex"}


def parse_backbones(value: str) -> List[str]:
    names = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = [name for name in names if name not in SUPPORTED_BACKBONES]
    if not names or invalid:
        suffix = f"; invalid: {', '.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(
            f"expected comma-separated {', '.join(SUPPORTED_BACKBONES)}{suffix}"
        )
    return names


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_backbone_summary(results_root: Path, backbone: str) -> Dict[str, Any]:
    path = results_root / backbone / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed result for {backbone}: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc

    recorded = summary.get("backbone", {}).get("name")
    if recorded != backbone:
        raise RuntimeError(
            f"{path} records backbone={recorded!r}, expected {backbone!r}"
        )
    results = summary.get("results")
    if not isinstance(results, dict):
        raise RuntimeError(f"{path} contains no benchmark results")
    missing_methods = [method for method in ALL_METHODS if method not in results]
    if missing_methods:
        raise RuntimeError(
            f"{path} is incomplete for an architecture ablation; missing "
            f"{', '.join(missing_methods)} (run with --method all)"
        )
    return summary


def validate_controlled_protocol(
    backbones: Sequence[str], summaries: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Require every non-architectural experimental variable to be matched."""
    reference_name = backbones[0]
    reference = summaries[reference_name].get("config", {})
    if reference.get("method") != "all":
        raise RuntimeError(
            f"{reference_name} was not run with --method all; architecture "
            "ablation requires all five comparison arms"
        )
    mismatches: List[str] = []
    for name in backbones[1:]:
        config = summaries[name].get("config", {})
        if config.get("method") != "all":
            mismatches.append(f"{name}.method={config.get('method')!r}")
        for key in CONTROLLED_CONFIG_KEYS:
            if config.get(key) != reference.get(key):
                mismatches.append(
                    f"{key}: {reference_name}={reference.get(key)!r}, "
                    f"{name}={config.get(key)!r}"
                )
    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "architecture ablation configurations are not matched: " + details
        )
    weight_recipes = {
        str(summaries[name].get("backbone", {}).get("weights", "")).rsplit(".", 1)[-1]
        for name in backbones
    }
    if len(weight_recipes) != 1 or "" in weight_recipes:
        raise RuntimeError(
            "architecture ablation uses unmatched pretrained-weight recipes: "
            + ", ".join(
                f"{name}={summaries[name].get('backbone', {}).get('weights')!r}"
                for name in backbones
            )
        )
    width_multipliers: Dict[str, float] = {}
    for name in backbones:
        config = summaries[name].get("config", {})
        feature_dim = summaries[name].get("backbone", {}).get("feature_dim")
        hidden_dim = config.get("hidden_dim")
        if not isinstance(feature_dim, (int, float)) or not isinstance(
            hidden_dim, (int, float)
        ):
            raise RuntimeError(f"{name} is missing feature_dim or hidden_dim")
        width_multipliers[name] = float(hidden_dim) / float(feature_dim)
    hidden_dims = [summaries[name].get("config", {}).get("hidden_dim") for name in backbones]
    all_same_hidden_dim = len(set(hidden_dims)) == 1 and hidden_dims[0] is not None
    if not all_same_hidden_dim and max(width_multipliers.values()) - min(width_multipliers.values()) > 1e-12:
        raise RuntimeError(
            "architecture ablation must keep the CSRv2/MPSAEv2 width multiplier "
            "constant: "
            + ", ".join(
                f"{name}={multiplier:g}x"
                for name, multiplier in width_multipliers.items()
            )
        )
    protocol = {key: reference.get(key) for key in CONTROLLED_CONFIG_KEYS}
    protocol["evaluation_budgets_by_backbone"] = {
        name: summaries[name].get("config", {}).get("topk") for name in backbones
    }
    protocol["pretrained_weight_recipe"] = next(iter(weight_recipes))
    protocol["sae_width_multiplier"] = (
        next(iter(width_multipliers.values())) if not all_same_hidden_dim else float(hidden_dims[0])
    )
    return protocol


def method_metrics(summary: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    return (
        summary.get("results", {})
        .get(method, {})
        .get("knn", {})
        .get("per_topk", {})
    )


def metric_value(metrics: Mapping[str, Any], budget: int, name: str) -> Optional[float]:
    value = metrics.get(str(budget), {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def extract_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    backbone = summary["backbone"]
    config = summary.get("config", {})
    method_results = summary.get("results", {})
    per_method = {method: method_metrics(summary, method) for method in METHODS}
    v1_metrics = {
        CSR_V1: method_metrics(summary, CSR_V1),
        MP_SAE_V1: method_metrics(summary, MP_SAE_V1),
    }
    budget_sets = {
        method: {int(budget) for budget in metrics}
        for method, metrics in per_method.items()
    }
    if any(not budgets for budgets in budget_sets.values()):
        raise RuntimeError(f"{backbone['name']} has no per-budget 1-NN results")
    v1_budget_sets = {
        method: {int(budget) for budget in metrics}
        for method, metrics in v1_metrics.items()
    }
    expected_sparse_budgets = budget_sets[CSR]
    if any(budgets != expected_sparse_budgets for budgets in v1_budget_sets.values()):
        raise RuntimeError(
            f"{backbone['name']} has unmatched v1/v2 sparse budgets: "
            + ", ".join(
                f"{method}={sorted(budgets)}" for method, budgets in v1_budget_sets.items()
            )
        )
    if budget_sets[CSR] != budget_sets[MP_SAE] or not budget_sets[MATRYOSHKA].issubset(
        budget_sets[CSR]
    ):
        raise RuntimeError(
            f"{backbone['name']} has invalid method-specific representation budgets: "
            + ", ".join(
                f"{method}={sorted(budgets)}"
                for method, budgets in budget_sets.items()
            )
        )
    budgets = sorted(budget_sets[MATRYOSHKA])

    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        mrl_top1 = metric_value(per_method[MATRYOSHKA], budget, "top1")
        csr_top1 = metric_value(per_method[CSR], budget, "top1")
        mp_top1 = metric_value(per_method[MP_SAE], budget, "top1")
        if mrl_top1 is None or csr_top1 is None or mp_top1 is None:
            raise RuntimeError(
                f"{backbone['name']} budget {budget} is missing a numeric top1 "
                "metric for one or more comparison arms"
            )
        effect = (
            mp_top1 - mrl_top1
            if mrl_top1 is not None and mp_top1 is not None
            else None
        )
        relative_error_reduction = (
            100.0 * effect / (100.0 - mrl_top1)
            if effect is not None and mrl_top1 is not None and mrl_top1 < 100.0
            else None
        )
        rows.append(
            {
                "backbone": backbone["name"],
                "backbone_display_name": backbone["display_name"],
                "feature_dim": int(backbone["feature_dim"]),
                "sae_hidden_dim": config.get("hidden_dim"),
                "matryoshka_trainable_parameters": method_results.get(
                    MATRYOSHKA, {}
                ).get("trainable_parameters"),
                "csrv2_trainable_parameters": method_results.get(CSR, {}).get(
                    "trainable_parameters"
                ),
                "mpsaev2_trainable_parameters": method_results.get(MP_SAE, {}).get(
                    "trainable_parameters"
                ),
                "sae_width_multiplier": (
                    float(config["hidden_dim"]) / float(backbone["feature_dim"])
                    if isinstance(config.get("hidden_dim"), (int, float))
                    else None
                ),
                "representation_budget": budget,
                "mrl_classification_weight": config.get(
                    "mrl_classification_weight"
                ),
                "csrv2_main_recon_weight": config.get("csrv2_main_recon_weight"),
                "csrv2_multi_topk_recon_weight": config.get(
                    "csrv2_multi_topk_recon_weight"
                ),
                "csrv2_aux_recon_weight": config.get("csrv2_aux_recon_weight"),
                "csrv2_contrastive_weight": config.get("csrv2_contrastive_weight"),
                "mpsaev2_main_recon_weight": config.get(
                    "mpsaev2_main_recon_weight"
                ),
                "mpsaev2_nested_recon_weight": config.get(
                    "mpsaev2_nested_recon_weight"
                ),
                "mpsaev2_aux_recon_weight": config.get("mpsaev2_aux_recon_weight"),
                "mpsaev2_mmpot_weight": config.get("mpsaev2_mmpot_weight"),
                "annealing_schedule": (
                    method_results.get(CSR, {})
                    .get("topk_annealing", {})
                    .get("schedule")
                ),
                "anneal_start_k": config.get("anneal_start_k"),
                "anneal_target_k": config.get("train_k"),
                "anneal_fraction": config.get("anneal_fraction"),
                "matryoshka_1nn_top1": mrl_top1,
                "csrv2_1nn_top1": csr_top1,
                "mpsaev2_1nn_top1": mp_top1,
                "method_effect_csrv2_minus_matryoshka_pp": csr_top1 - mrl_top1,
                "method_effect_mpsaev2_minus_matryoshka_pp": effect,
                "method_effect_mpsaev2_minus_csrv2_pp": mp_top1 - csr_top1,
                "relative_error_reduction_pct": relative_error_reduction,
                "mpsaev2_wins_at_budget": (
                    int(effect > 0.0) if effect is not None else None
                ),
                "delta_mpsaev2_minus_matryoshka": effect,
                "delta_csrv2_minus_matryoshka": csr_top1 - mrl_top1,
                "delta_mpsaev2_minus_csrv2": mp_top1 - csr_top1,
                "matryoshka_mean_neighbor_l2_squared": metric_value(
                    per_method[MATRYOSHKA], budget, "mean_neighbor_l2_squared"
                ),
                "mpsaev2_mean_neighbor_l2_squared": metric_value(
                    per_method[MP_SAE], budget, "mean_neighbor_l2_squared"
                ),
                "csrv2_mean_neighbor_l2_squared": metric_value(
                    per_method[CSR], budget, "mean_neighbor_l2_squared"
                ),
                "matryoshka_mean_retrieval_ms_per_query": metric_value(
                    per_method[MATRYOSHKA], budget,
                    "mean_retrieval_milliseconds_per_query",
                ),
                "csrv2_mean_retrieval_ms_per_query": metric_value(
                    per_method[CSR], budget, "mean_retrieval_milliseconds_per_query"
                ),
                "mpsaev2_mean_retrieval_ms_per_query": metric_value(
                    per_method[MP_SAE], budget, "mean_retrieval_milliseconds_per_query"
                ),
            }
        )
    return rows


def numeric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.fmean(values) if values else None


def numeric_median(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.median(values) if values else None


def numeric_min(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return min(values) if values else None


def numeric_max(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return max(values) if values else None


def build_aggregate(
    results_root: Path,
    backbones: Sequence[str],
    summaries: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    controlled_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    aggregate_backbones: Dict[str, Any] = {}
    effect_key = "method_effect_mpsaev2_minus_matryoshka_pp"
    csr_effect_key = "method_effect_csrv2_minus_matryoshka_pp"
    mp_vs_csr_key = "method_effect_mpsaev2_minus_csrv2_pp"
    for name in backbones:
        summary = summaries[name]
        backbone_rows = [row for row in rows if row["backbone"] == name]
        effect_values = [
            float(row[effect_key])
            for row in backbone_rows
            if isinstance(row.get(effect_key), (int, float))
        ]
        csr_effect_values = [float(row[csr_effect_key]) for row in backbone_rows]
        mp_vs_csr_values = [float(row[mp_vs_csr_key]) for row in backbone_rows]
        aggregate_backbones[name] = {
            **summary["backbone"],
            "source_summary": str(Path(name) / "summary.json"),
            "configuration": summary.get("config", {}),
            "topk_annealing": {
                CSR: summary["results"][CSR].get("topk_annealing"),
                MP_SAE: summary["results"][MP_SAE].get("topk_annealing"),
            },
            "matryoshka_trainable_parameters": summary["results"][MATRYOSHKA].get(
                "trainable_parameters"
            ),
            "csrv2_trainable_parameters": summary["results"][CSR].get(
                "trainable_parameters"
            ),
            "mpsaev2_trainable_parameters": summary["results"][MP_SAE].get(
                "trainable_parameters"
            ),
            "mean_matryoshka_1nn_top1": numeric_mean(
                backbone_rows, "matryoshka_1nn_top1"
            ),
            "mean_csrv2_1nn_top1": numeric_mean(backbone_rows, "csrv2_1nn_top1"),
            "mean_mpsaev2_1nn_top1": numeric_mean(backbone_rows, "mpsaev2_1nn_top1"),
            "mean_matryoshka_retrieval_ms_per_query": numeric_mean(
                backbone_rows, "matryoshka_mean_retrieval_ms_per_query"
            ),
            "mean_csrv2_retrieval_ms_per_query": numeric_mean(
                backbone_rows, "csrv2_mean_retrieval_ms_per_query"
            ),
            "mean_mpsaev2_retrieval_ms_per_query": numeric_mean(
                backbone_rows, "mpsaev2_mean_retrieval_ms_per_query"
            ),
            "mean_delta_csrv2_minus_matryoshka": numeric_mean(
                backbone_rows, "delta_csrv2_minus_matryoshka"
            ),
            "mean_delta_mpsaev2_minus_matryoshka": numeric_mean(
                backbone_rows, "delta_mpsaev2_minus_matryoshka"
            ),
            "method_effect": {
                "definition": "MPSAEv2 top-1 minus Matryoshka top-1, in percentage points",
                "mean_pp": numeric_mean(backbone_rows, effect_key),
                "median_pp": numeric_median(backbone_rows, effect_key),
                "minimum_pp": numeric_min(backbone_rows, effect_key),
                "maximum_pp": numeric_max(backbone_rows, effect_key),
                "standard_deviation_pp": (
                    statistics.pstdev(effect_values) if effect_values else None
                ),
                "positive_budget_count": sum(value > 0.0 for value in effect_values),
                "evaluated_budget_count": len(effect_values),
                "positive_budget_fraction": (
                    sum(value > 0.0 for value in effect_values) / len(effect_values)
                    if effect_values
                    else None
                ),
                "mean_relative_error_reduction_pct": numeric_mean(
                    backbone_rows, "relative_error_reduction_pct"
                ),
            },
            "csrv2_effect": {
                "definition": "CSRv2 top-1 minus Matryoshka top-1, in percentage points",
                "mean_pp": numeric_mean(backbone_rows, csr_effect_key),
                "median_pp": numeric_median(backbone_rows, csr_effect_key),
                "minimum_pp": numeric_min(backbone_rows, csr_effect_key),
                "maximum_pp": numeric_max(backbone_rows, csr_effect_key),
                "standard_deviation_pp": statistics.pstdev(csr_effect_values),
                "positive_budget_count": sum(value > 0.0 for value in csr_effect_values),
                "evaluated_budget_count": len(csr_effect_values),
            },
            "mpsaev2_vs_csrv2_effect": {
                "definition": "MPSAEv2 top-1 minus CSRv2 top-1, in percentage points",
                "mean_pp": numeric_mean(backbone_rows, mp_vs_csr_key),
                "median_pp": numeric_median(backbone_rows, mp_vs_csr_key),
                "minimum_pp": numeric_min(backbone_rows, mp_vs_csr_key),
                "maximum_pp": numeric_max(backbone_rows, mp_vs_csr_key),
                "standard_deviation_pp": statistics.pstdev(mp_vs_csr_values),
                "positive_budget_count": sum(value > 0.0 for value in mp_vs_csr_values),
                "evaluated_budget_count": len(mp_vs_csr_values),
            },
            "per_budget": {
                str(row["representation_budget"]): {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "backbone",
                        "backbone_display_name",
                        "feature_dim",
                        "sae_hidden_dim",
                        "matryoshka_trainable_parameters",
                        "csrv2_trainable_parameters",
                        "mpsaev2_trainable_parameters",
                        "sae_width_multiplier",
                        "representation_budget",
                    }
                }
                for row in backbone_rows
            },
        }
    backbone_mean_effects = [
        float(entry["method_effect"]["mean_pp"])
        for entry in aggregate_backbones.values()
        if isinstance(entry["method_effect"]["mean_pp"], (int, float))
    ]
    sensitivity: Dict[str, Any] = {
        "mean_effect_across_backbones_pp": (
            statistics.fmean(backbone_mean_effects) if backbone_mean_effects else None
        ),
        "effect_standard_deviation_across_backbones_pp": (
            statistics.pstdev(backbone_mean_effects) if backbone_mean_effects else None
        ),
        "effect_range_across_backbones_pp": (
            max(backbone_mean_effects) - min(backbone_mean_effects)
            if backbone_mean_effects
            else None
        ),
    }
    if "resnet18" in aggregate_backbones and "resnet50" in aggregate_backbones:
        sensitivity["resnet50_minus_resnet18_mean_effect_pp"] = (
            aggregate_backbones["resnet50"]["method_effect"]["mean_pp"]
            - aggregate_backbones["resnet18"]["method_effect"]["mean_pp"]
        )
    return {
        "experiment": "Matryoshka_CSR_MPSAE_v1_v2_architecture_ablation_ImageNet",
        "ablation_factor": "frozen_and_matryoshka_backbone_architecture",
        "comparison_arms": list(ALL_METHODS),
        "effect_definitions": {
            "csrv2_minus_matryoshka": "CSRv2 top-1 minus Matryoshka top-1, in percentage points",
            "mpsaev2_minus_matryoshka": "MPSAEv2 top-1 minus Matryoshka top-1, in percentage points",
            "mpsaev2_minus_csrv2": "MPSAEv2 top-1 minus CSRv2 top-1, in percentage points",
        },
        "inference_scope": (
            "Descriptive matched-seed architecture ablation; uncertainty across "
            "independent random seeds is not estimated."
        ),
        "completed_at_utc": utc_now(),
        "results_root": str(results_root),
        "backbone_order": list(backbones),
        "controlled_protocol": dict(controlled_protocol),
        "architecture_sensitivity": sensitivity,
        "backbones": aggregate_backbones,
    }


def effect_summary_rows(
    aggregate: Mapping[str, Any], backbones: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in backbones:
        backbone = aggregate["backbones"][name]
        effect = backbone["method_effect"]
        rows.append(
            {
                "backbone": name,
                "backbone_display_name": backbone["display_name"],
                "feature_dim": backbone["feature_dim"],
                "matryoshka_trainable_parameters": backbone.get(
                    "matryoshka_trainable_parameters"
                ),
                "csrv2_trainable_parameters": backbone.get(
                    "csrv2_trainable_parameters"
                ),
                "mpsaev2_trainable_parameters": backbone.get(
                    "mpsaev2_trainable_parameters"
                ),
                "mean_csrv2_minus_matryoshka_pp": backbone["csrv2_effect"]["mean_pp"],
                "mean_mpsaev2_minus_csrv2_pp": backbone["mpsaev2_vs_csrv2_effect"]["mean_pp"],
                "mean_method_effect_pp": effect["mean_pp"],
                "median_method_effect_pp": effect["median_pp"],
                "minimum_method_effect_pp": effect["minimum_pp"],
                "maximum_method_effect_pp": effect["maximum_pp"],
                "effect_standard_deviation_pp": effect["standard_deviation_pp"],
                "positive_budget_count": effect["positive_budget_count"],
                "evaluated_budget_count": effect["evaluated_budget_count"],
                "positive_budget_fraction": effect["positive_budget_fraction"],
                "mean_relative_error_reduction_pct": effect[
                    "mean_relative_error_reduction_pct"
                ],
            }
        )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty aggregate CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def format_metric(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "--"


def write_markdown(
    rows: Sequence[Mapping[str, Any]], backbones: Sequence[str], path: Path
) -> None:
    lines = [
        "# ImageNet Matryoshka / CSRv2 / MPSAEv2 architecture ablation",
        "",
        "The ablation changes only the backbone architecture and reports all pairwise "
        "method effects under a matched protocol.",
        "",
        "| Backbone | Feature dim | SAE dim | K | Matryoshka | CSRv2 | MPSAEv2 | CSRv2-M | MPSAEv2-M | MPSAEv2-CSRv2 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in backbones:
        selected = [row for row in rows if row["backbone"] == name]
        for row in selected:
            lines.append(
                f"| {row['backbone_display_name']} | {row['feature_dim']} | "
                f"{row['sae_hidden_dim']} | {row['representation_budget']} | "
                f"{format_metric(row['matryoshka_1nn_top1'])} | "
                f"{format_metric(row['csrv2_1nn_top1'])} | "
                f"{format_metric(row['mpsaev2_1nn_top1'])} | "
                f"{format_metric(row['method_effect_csrv2_minus_matryoshka_pp'])} | "
                f"{format_metric(row['method_effect_mpsaev2_minus_matryoshka_pp'])} | "
                f"{format_metric(row['method_effect_mpsaev2_minus_csrv2_pp'])} |"
            )
        lines.append(
            f"| **{selected[0]['backbone_display_name']} mean** |  |  |  | "
            f"**{format_metric(numeric_mean(selected, 'matryoshka_1nn_top1'))}** | "
            f"**{format_metric(numeric_mean(selected, 'csrv2_1nn_top1'))}** | "
            f"**{format_metric(numeric_mean(selected, 'mpsaev2_1nn_top1'))}** | "
            f"**{format_metric(numeric_mean(selected, 'method_effect_csrv2_minus_matryoshka_pp'))}** | "
            f"**{format_metric(numeric_mean(selected, 'method_effect_mpsaev2_minus_matryoshka_pp'))}** | "
            f"**{format_metric(numeric_mean(selected, 'method_effect_mpsaev2_minus_csrv2_pp'))}** |"
        )
    lines.extend(
        [
            "",
            "Top-1 values and deltas are percentage points on ImageNet validation using unit-normalized exact L2 1-NN.",
            "Tables contain the matched K>=8 budgets; sparse-only K=1,2,4 points remain in each backbone summary and plot.",
            "K is the Matryoshka prefix dimension or the number of active CSRv2/MPSAEv2 latents.",
            "The effect-summary CSV additionally records the effect range, variability, and fraction of budgets won.",
            "This is a descriptive matched-seed study; it does not estimate uncertainty across independent seeds.",
        ]
    )
    atomic_text("\n".join(lines) + "\n", path)


def write_latex(
    rows: Sequence[Mapping[str, Any]], backbones: Sequence[str], path: Path
) -> None:
    slash = chr(92)
    row_end = slash * 2
    lines = [
        f"{slash}begin{{table*}}[t]",
        f"{slash}centering",
        f"{slash}caption{{Architecture ablation of Matryoshka, CSRv2, and MPSAEv2 on ImageNet using unit-normalized exact L2 1-NN top-1 accuracy.}}",
        f"{slash}label{{tab:three-method-architecture-ablation}}",
        f"{slash}small",
        f"{slash}begin{{tabular}}{{lrrrrrrrrr}}",
        f"{slash}toprule",
        f"Backbone & Feature dim & SAE dim & K & Matryoshka & CSRv2 & MPSAEv2 & CSRv2-M & MPSAEv2-M & MPSAEv2-CSRv2 {row_end}",
        f"{slash}midrule",
    ]
    for backbone_index, name in enumerate(backbones):
        selected = [row for row in rows if row["backbone"] == name]
        for row in selected:
            lines.append(
                f"{row['backbone_display_name']} & {row['feature_dim']} & "
                f"{row['sae_hidden_dim']} & {row['representation_budget']} & "
                f"{format_metric(row['matryoshka_1nn_top1'])} & "
                f"{format_metric(row['csrv2_1nn_top1'])} & "
                f"{format_metric(row['mpsaev2_1nn_top1'])} & "
                f"{format_metric(row['method_effect_csrv2_minus_matryoshka_pp'])} & "
                f"{format_metric(row['method_effect_mpsaev2_minus_matryoshka_pp'])} & "
                f"{format_metric(row['method_effect_mpsaev2_minus_csrv2_pp'])} {row_end}"
            )
        lines.append(
            f"{slash}textbf{{{selected[0]['backbone_display_name']} mean}} & & & & "
            f"{format_metric(numeric_mean(selected, 'matryoshka_1nn_top1'))} & "
            f"{format_metric(numeric_mean(selected, 'csrv2_1nn_top1'))} & "
            f"{format_metric(numeric_mean(selected, 'mpsaev2_1nn_top1'))} & "
            f"{format_metric(numeric_mean(selected, 'method_effect_csrv2_minus_matryoshka_pp'))} & "
            f"{format_metric(numeric_mean(selected, 'method_effect_mpsaev2_minus_matryoshka_pp'))} & "
            f"{format_metric(numeric_mean(selected, 'method_effect_mpsaev2_minus_csrv2_pp'))} {row_end}"
        )
        if backbone_index != len(backbones) - 1:
            lines.append(f"{slash}midrule")
    lines.extend(
        [
            f"{slash}bottomrule",
            f"{slash}end{{tabular}}",
            f"{slash}end{{table*}}",
        ]
    )
    atomic_text("\n".join(lines) + "\n", path)


def configure_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_comparison(
    rows: Sequence[Mapping[str, Any]], backbones: Sequence[str], results_root: Path
) -> List[Path]:
    configure_plot_style()
    import matplotlib.pyplot as plt

    colors = {"resnet18": "#0072B2", "resnet50": "#009E73"}
    fig, axis = plt.subplots(figsize=(5.4, 3.1), constrained_layout=True)
    mean_csr_effects: List[float] = []
    mean_mp_effects: List[float] = []
    display_names: List[str] = []

    for name in backbones:
        selected = [row for row in rows if row["backbone"] == name]
        display_name = str(selected[0]["backbone_display_name"])
        display_names.append(display_name)
        mean_csr_effects.append(
            float(numeric_mean(selected, "method_effect_csrv2_minus_matryoshka_pp"))
        )
        mean_mp_effects.append(
            float(numeric_mean(selected, "method_effect_mpsaev2_minus_matryoshka_pp"))
        )

    positions = list(range(len(backbones)))
    bar_colors = [colors.get(name, "#333333") for name in backbones]
    axis.bar(
        [position - 0.18 for position in positions], mean_csr_effects,
        width=0.36, color=bar_colors, alpha=0.55, label="CSRv2 - Matryoshka",
    )
    axis.bar(
        [position + 0.18 for position in positions], mean_mp_effects,
        width=0.36, color=bar_colors, alpha=0.95, label="MPSAEv2 - Matryoshka",
    )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(positions, display_names)
    axis.set_title("Mean method effect by backbone", loc="left", fontweight="bold")
    axis.set_ylabel("Method - Matryoshka (pp)")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#B8B8B8", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    output_paths = [
        results_root / "architecture_ablation_effect.pdf",
        results_root / "architecture_ablation_effect.png",
    ]
    fig.savefig(output_paths[0], bbox_inches="tight")
    fig.savefig(output_paths[1], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_artifacts(
    results_root: Path, backbones: Sequence[str], aggregate_paths: Iterable[Path]
) -> List[Path]:
    paths = [path for path in aggregate_paths if path.is_file()]
    for name in backbones:
        for relative in METHOD_RESULT_FILES:
            path = results_root / name / relative
            if path.is_file():
                paths.append(path)
        for path in (results_root / name).glob("ablation_*"):
            if path.is_file() and path.suffix.lower() in PORTABLE_SUFFIXES:
                paths.append(path)
        log_dir = results_root / name / "logs"
        if log_dir.is_dir():
            paths.extend(path for path in log_dir.glob("*.log") if path.is_file())
    root_log_dir = results_root / "logs"
    if root_log_dir.is_dir():
        paths.extend(path for path in root_log_dir.glob("*.log") if path.is_file())
    return sorted(set(paths), key=lambda path: path.as_posix())


def write_bundle(results_root: Path, artifacts: Sequence[Path], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            archive.write(artifact, artifact.relative_to(results_root).as_posix())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a controlled Matryoshka and CSR/MPSAE v1/v2 architecture ablation from "
            "csr_vs_mmpot_imagenet.py outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_root", type=Path)
    parser.add_argument(
        "--backbones", type=parse_backbones,
        default=list(DEFAULT_ABLATION_BACKBONES),
        help="backbone directories that must be present",
    )
    parser.add_argument(
        "--bundle-name", default="imagenet_architecture_ablation_results.zip"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_name = Path(args.bundle_name)
    if bundle_name.name != args.bundle_name or bundle_name.suffix.lower() != ".zip":
        raise ValueError("bundle-name must be a plain filename ending in .zip")
    results_root = args.results_root.expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    summaries = {
        name: load_backbone_summary(results_root, name) for name in args.backbones
    }
    rows = [
        row
        for name in args.backbones
        for row in extract_rows(summaries[name])
    ]
    controlled_protocol = validate_controlled_protocol(args.backbones, summaries)
    aggregate = build_aggregate(
        results_root, args.backbones, summaries, rows, controlled_protocol
    )
    effect_rows = effect_summary_rows(aggregate, args.backbones)

    aggregate_summary_path = results_root / "architecture_ablation_summary.json"
    csv_path = results_root / "architecture_ablation_per_budget.csv"
    effect_csv_path = results_root / "architecture_ablation_effect_summary.csv"
    markdown_path = results_root / "architecture_ablation_results.md"
    latex_path = results_root / "architecture_ablation_results.tex"
    atomic_json(aggregate, aggregate_summary_path)
    write_csv(rows, csv_path)
    write_csv(effect_rows, effect_csv_path)
    write_markdown(rows, args.backbones, markdown_path)
    write_latex(rows, args.backbones, latex_path)
    plot_paths = plot_comparison(rows, args.backbones, results_root)

    aggregate_paths = [
        aggregate_summary_path,
        csv_path,
        effect_csv_path,
        markdown_path,
        latex_path,
        *plot_paths,
    ]
    artifacts = result_artifacts(results_root, args.backbones, aggregate_paths)
    manifest_path = results_root / "artifact_manifest.json"
    manifest = {
        "created_at_utc": utc_now(),
        "backbones": args.backbones,
        "artifacts": [
            {
                "path": artifact.relative_to(results_root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
            for artifact in artifacts
        ],
    }
    atomic_json(manifest, manifest_path)
    artifacts.append(manifest_path)

    bundle_path = results_root / bundle_name
    write_bundle(results_root, artifacts, bundle_path)
    completion_path = results_root / "RUN_COMPLETE.json"
    atomic_json(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "backbones": args.backbones,
            "bundle": bundle_path.name,
            "bundle_bytes": bundle_path.stat().st_size,
            "bundle_sha256": sha256(bundle_path),
            "study": "Matryoshka_CSR_MPSAE_v1_v2_architecture_ablation_ImageNet",
            "primary_plot_png": "architecture_ablation_effect.png",
            "primary_plot_pdf": "architecture_ablation_effect.pdf",
            "per_budget_table": "architecture_ablation_per_budget.csv",
            "effect_summary_table": "architecture_ablation_effect_summary.csv",
        },
        completion_path,
    )

    print("\nFive-arm run complete; primary v2 effects (exact L2 1-NN top-1)")
    print(
        f"{'backbone':<12} {'Matryoshka':>12} {'CSRv2':>12} "
        f"{'MPSAEv2':>12} {'MPSAEv2-CSRv2':>14}"
    )
    for name in args.backbones:
        entry = aggregate["backbones"][name]
        print(
            f"{entry['display_name']:<12} "
            f"{format_metric(entry['mean_matryoshka_1nn_top1']):>12} "
            f"{format_metric(entry['mean_csrv2_1nn_top1']):>12} "
            f"{format_metric(entry['mean_mpsaev2_1nn_top1']):>12} "
            f"{format_metric(entry['mpsaev2_vs_csrv2_effect']['mean_pp']):>14}"
        )
    print(f"\nResults bundle: {bundle_path}")
    print(f"Completion marker: {completion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
