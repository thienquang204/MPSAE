#!/usr/bin/env python3
"""Fetch the published baseline fits into the run directory before comparing.

Each top-level folder of the Hub repository holding a trial.json is one
completed fit in the mmrsae_archive.py layout. Folders missing from
--output-dir are downloaded to a staging folder, verified against the
repository's manifest.json (size and SHA-256), then moved into place, so the
run directory only ever holds complete baseline folders. Folders already
present are left untouched.

    python mmrsae_baselines.py --output-dir runs/mmrsae --repo Lukita/almost_cvpr
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from repair_data import sha256_file

DEFAULT_REPO = "Lukita/almost_cvpr"


def fetch(output_dir, repo, revision="main"):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    token = os.environ.get("HF_TOKEN") or None
    output_dir = Path(output_dir)
    files = HfApi().list_repo_files(repo, revision=revision, token=token)
    trials = sorted({name.split("/")[0] for name in files if name.count("/") == 1 and name.endswith("/trial.json")})
    missing = [trial for trial in trials if not (output_dir / trial / "trial.json").exists()]
    if not missing:
        return trials, []
    manifest = json.loads(Path(hf_hub_download(repo, "manifest.json", revision=revision, token=token))
                          .read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest}
    staging = output_dir / ".baselines_download"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        snapshot_download(repo, revision=revision, token=token, local_dir=staging,
                          allow_patterns=[f"{trial}/*" for trial in missing])
        for trial in missing:
            wanted = {path for path in expected if path.startswith(trial + "/")}
            present = {path.relative_to(staging).as_posix() for path in (staging / trial).rglob("*") if path.is_file()}
            if wanted != present:
                raise RuntimeError(f"{repo}:{trial} files differ from manifest.json")
            for path in sorted(wanted):
                entry, local = expected[path], staging / path
                if local.stat().st_size != entry["size"] or sha256_file(local) != entry["sha256"]:
                    raise RuntimeError(f"{repo}:{path} fails its manifest size/SHA-256 check")
            os.replace(staging / trial, output_dir / trial)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return trials, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mmrsae"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials, fetched = fetch(args.output_dir, args.repo, args.revision)
    print(f"MMRSAE baselines ({args.repo}): {len(trials)} published, "
          f"{'downloaded ' + ', '.join(fetched) if fetched else 'all already present'} in {args.output_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
