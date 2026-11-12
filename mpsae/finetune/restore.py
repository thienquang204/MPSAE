"""Restore only the nine frozen source fits and their provenance, not embeddings."""
from pathlib import Path
import shutil

from ..io import sha256_file


def restore_sources(config, workspace, repo, *, revision="main", token=None):
    from huggingface_hub import HfApi, hf_hub_download
    from .run import preflight_sources
    root = Path(workspace).resolve()
    info = HfApi(token=token).repo_info(repo, repo_type="dataset", revision=revision)
    commit = info.sha
    names = HfApi(token=token).list_repo_files(repo, repo_type="dataset", revision=commit)
    required = {f"direct-v1/runs/{c.slug}/{filename}" for c in config.cells()
                for filename in ("checkpoint.pt", "config.json", "summary.json")}
    missing = required - set(names)
    if missing:
        raise FileNotFoundError(f"HF snapshot lacks required source files: {sorted(missing)[:5]}")
    for name in names:
        if name.startswith("direct-v1/embeddings/") and name.endswith(".json") and any(
                f"1K_{split}_{backbone}." in name
                for backbone in config.campaign.backbones for split in ("train", "val")):
            required.add(name)
    for name in sorted(required):
        cached = Path(hf_hub_download(repo, name, repo_type="dataset", revision=commit, token=token))
        destination = (root / name).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("unexpected source artifact path")
        if destination.exists():
            if sha256_file(cached) != sha256_file(destination):
                raise RuntimeError(f"existing source differs from HF snapshot: {destination}; use a separate workspace")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(cached, temporary)
        temporary.replace(destination)
    preflight_sources(config, root / "direct-v1")
    return {"repository": repo, "revision": commit, "files": len(required), "source_root": str(root / "direct-v1")}
