#!/usr/bin/env python3
"""Validate and bundle every experiment produced by the Docker suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_EXPERIMENTS = ("csr_vs_mpsae", "mmpot_proxy", "mmpot_true")
PORTABLE_SUFFIXES = {".csv", ".json", ".log", ".md", ".pdf", ".png", ".tex", ".txt"}
EXCLUDED_PARTS = {"cache", "feature_cache", "weights", "__pycache__"}


def parse_names(value: str) -> List[str]:
    names = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = sorted(set(names) - set(DEFAULT_EXPERIMENTS))
    if not names or invalid:
        suffix = f"; invalid: {', '.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(
            f"expected comma-separated {', '.join(DEFAULT_EXPERIMENTS)}{suffix}"
        )
    return names


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_files(root: Path, experiment_dirs: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for directory in experiment_dirs:
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and path.suffix.lower() in PORTABLE_SUFFIXES
                and not any(part in EXCLUDED_PARTS for part in relative.parts)
                and not path.name.endswith(".tmp")
                and path.name != "all_experiments_results.zip"
            ):
                files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def summarize_experiment(root: Path, name: str) -> Dict[str, Any]:
    directory = root / name
    if not directory.is_dir():
        raise FileNotFoundError(f"missing experiment output directory: {directory}")
    summaries = sorted(directory.rglob("summary.json"))
    if not summaries:
        raise RuntimeError(f"experiment {name!r} produced no summary.json")
    completion_markers = sorted(directory.rglob("RUN_COMPLETE.json"))
    return {
        "directory": name,
        "summary_files": [path.relative_to(root).as_posix() for path in summaries],
        "completion_markers": [
            path.relative_to(root).as_posix() for path in completion_markers
        ],
    }


def write_bundle(root: Path, paths: Sequence[Path], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    os.replace(temporary, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one portable bundle and completion marker for the full experiment suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("suite_root", type=Path)
    parser.add_argument("--experiments", type=parse_names, default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--bundle-name", default="all_experiments_results.zip")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_name = Path(args.bundle_name)
    if bundle_name.name != args.bundle_name or bundle_name.suffix.lower() != ".zip":
        raise ValueError("bundle-name must be a plain filename ending in .zip")
    suite_root = args.suite_root.expanduser().resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    experiment_summaries = {
        name: summarize_experiment(suite_root, name) for name in args.experiments
    }
    experiment_dirs = [suite_root / name for name in args.experiments]

    suite_summary_path = suite_root / "all_experiments_summary.json"
    atomic_json(
        {
            "suite": "graduate_thesis_all_experiments",
            "status": "complete",
            "completed_at_utc": utc_now(),
            "experiments": experiment_summaries,
        },
        suite_summary_path,
    )

    portable_roots = list(experiment_dirs)
    if (suite_root / "logs").is_dir():
        portable_roots.append(suite_root / "logs")
    artifacts = portable_files(suite_root, portable_roots)
    suite_run_manifest = suite_root / "suite_run_manifest.txt"
    if suite_run_manifest.is_file():
        artifacts.append(suite_run_manifest)
    artifacts.append(suite_summary_path)
    manifest_path = suite_root / "all_experiments_artifact_manifest.json"
    atomic_json(
        {
            "created_at_utc": utc_now(),
            "experiments": args.experiments,
            "artifacts": [
                {
                    "path": path.relative_to(suite_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in artifacts
            ],
        },
        manifest_path,
    )
    artifacts.append(manifest_path)

    bundle_path = suite_root / bundle_name
    write_bundle(suite_root, artifacts, bundle_path)
    completion_path = suite_root / "ALL_EXPERIMENTS_COMPLETE.json"
    atomic_json(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "experiments": args.experiments,
            "bundle": bundle_path.name,
            "bundle_bytes": bundle_path.stat().st_size,
            "bundle_sha256": sha256(bundle_path),
            "artifact_manifest": manifest_path.name,
        },
        completion_path,
    )
    print(f"Full-suite bundle: {bundle_path}")
    print(f"Completion marker: {completion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
