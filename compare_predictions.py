"""Paired, within-class query bootstrap conditional on the supplied fitted models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from repair_data import atomic_json, sha256_file


def read_predictions(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"seed", "query_id", "label", "prediction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} requires columns {sorted(required)}")
        for row in reader:
            seed = int(row["seed"])
            query = row["query_id"]
            if not query:
                raise ValueError("Empty query ID")
            key = (seed, query)
            if key in rows:
                raise ValueError(f"Duplicate seed/query ID {key} in {path}")
            rows[key] = (int(row["label"]), int(row["prediction"]))
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def compare_predictions(baseline, candidate, *, replicates=1000, seed=2026):
    """CSV paths or mappings -> JSON-compatible paired accuracy report.

    Every draw resamples query IDs within each class, with the same multiplicity
    for both methods and every fitted seed. This is not a seed bootstrap.
    """
    import numpy as np

    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    base_path = Path(baseline) if isinstance(baseline, (str, Path)) else None
    cand_path = Path(candidate) if isinstance(candidate, (str, Path)) else None
    base = read_predictions(base_path) if base_path is not None else baseline
    cand = read_predictions(cand_path) if cand_path is not None else candidate
    if not base or set(base) != set(cand):
        raise ValueError("Methods must contain exactly matching seed/query IDs")
    seeds = sorted({key[0] for key in base})
    queries = sorted({key[1] for key in base})
    if len(base) != len(seeds) * len(queries):
        raise ValueError("Every supplied seed must predict every query")
    labels = np.empty(len(queries), dtype=np.int64)
    baseline_correct = np.empty((len(seeds), len(queries)), dtype=np.float64)
    candidate_correct = np.empty_like(baseline_correct)
    for col, query in enumerate(queries):
        expected_label = base[(seeds[0], query)][0]
        labels[col] = expected_label
        for row, fit_seed in enumerate(seeds):
            target, prediction_base = base[(fit_seed, query)]
            target_cand, prediction_cand = cand[(fit_seed, query)]
            if target != expected_label or target_cand != expected_label:
                raise ValueError(f"Labels disagree across methods or seeds for query {query}")
            baseline_correct[row, col] = prediction_base == target
            candidate_correct[row, col] = prediction_cand == target
    # Averaging across fitted seeds before a paired draw is algebraically
    # identical to sharing the same draw across seeds and then averaging.
    paired = (candidate_correct - baseline_correct).mean(axis=0)
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    # Bounded batches avoid replicates x all-query allocations on full data.
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        totals = np.zeros(count, dtype=np.float64)
        for members in strata:
            samples = generator.integers(0, len(members), size=(count, len(members)))
            totals += paired[members[samples]].sum(axis=1)
        bootstrap[start:start + count] = totals * (100.0 / len(queries))
    interval = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    report = {
        "baseline_top1": float(100 * baseline_correct.mean()),
        "candidate_top1": float(100 * candidate_correct.mean()),
        "delta_percentage_points": float(100 * paired.mean()),
        "ci95_percentage_points": interval,
        "ci_lower_percentage_points": interval[0], "ci_upper_percentage_points": interval[1],
        "seeds": seeds, "queries": len(queries), "classes": len(strata),
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "bootstrap_method": "paired within-class query resampling; all fitted seeds share each draw",
        "uncertainty_scope": "Query sampling conditional on fitted model pairs; excludes training-seed and parent-pretraining uncertainty and does not correct cheap-set selection.",
        "per_seed": [{"seed": fit_seed, "baseline_correct": int(baseline_correct[row].sum()), "candidate_correct": int(candidate_correct[row].sum()), "total": len(queries), "delta_percentage_points": float(100 * (candidate_correct[row] - baseline_correct[row]).mean())} for row, fit_seed in enumerate(seeds)],
    }
    if base_path is not None:
        report["baseline_sha256"] = sha256_file(base_path)
    if cand_path is not None:
        report["candidate_sha256"] = sha256_file(cand_path)
    return report


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    report = compare_predictions(args.baseline, args.candidate, replicates=args.replicates, seed=args.seed)
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
