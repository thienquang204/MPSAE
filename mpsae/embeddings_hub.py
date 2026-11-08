"""Embeddings only: build each (dataset, backbone) once, keep them in one HF repo.

Every embedding set lives in one folder named ``<dataset>_<backbone>_<dim>``
(``imagenet_ff2048_2048``, ``sop_rn18_512``, ``eurosat_dinov2_vits14_384``),
both locally under ``<workspace>/embeddings-hub/`` and in one private
Hugging Face dataset repository (default ``<HF account>/MPSAE-embeddings``).

``sync`` makes every selected folder present on this machine AND on the Hub:

    local  Hub
    yes    yes   -> use the local folder
    yes    no    -> push it (local hashes checked first, remote hashes after)
    no     yes   -> pull it (every SHA-256 checked)
    no     no    -> build it with the campaign's own extractors, push it, and
                    only then delete the raw data

A local folder is complete when ``folder.json`` exists and every file it lists
has its size; a Hub folder when its ``folder.json`` is there (uploaded last).

The extraction is exactly the campaign's (``storage_mode: direct_v1``):

* ImageNet: ``data.imagenet.prepare_imagenet`` then ``embed.direct.extract_source``
  (recipe ``imagenet-direct-pil-v1``). ImageNet JPEGs are deleted only after
  every selected backbone's folder is built, uploaded and verified, and after
  the dense 1-NN gate of backbones with a published value (ff2048: 71.19).
* Downstream (``benchmark.data.DATASETS``): ``load_dataset_bundle`` then
  ``benchmark.extract.extract_bundle`` (recipe ``frozen-transfer-pil-bilinear-v1``),
  with the backbone weights pinned to the ones recorded in that backbone's
  ImageNet folder. One dataset at a time: download, embed every backbone,
  upload and verify, delete the raw download, next dataset.

    python -m mpsae.embeddings_hub status
    python -m mpsae.embeddings_hub sync
    python -m mpsae.embeddings_hub sync --datasets imagenet,sop --backbones ff2048,rn18
    python -m mpsae.embeddings_hub sync --config configs/thesis.yaml   # what that campaign uses

With ``--config``, the selection comes from the campaign and, after the sync,
the folders are hard-linked where the campaign reads them (no extra disk):
ImageNet archives into ``<campaign workspace>/embeddings/<backbone>/`` (the
campaign's own source verification still checks recipe, rows, IDs and hash),
downstream folders into ``<campaign workspace>/benchmarks/<dataset>/hub/<backbone>/``
(used by ``benchmark.storage.run_targets`` only when their backbone weights
equal the source checkpoint's). all_in_one.sh runs this before the campaign.

HF_TOKEN needs read access to ILSVRC/imagenet-1k (terms accepted) and write
access for the repository. The repository is private: ImageNet-derived
features inherit ImageNet's non-commercial terms.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

from .io import atomic_json, read_json, sha256_file

FORMAT = "mpsae-embedding-folder-v1"
FOLDER_MANIFEST = "folder.json"
REPOSITORY_NAME = "MPSAE-embeddings"
IMAGENET = "imagenet"
DENSE_GATE = 0.3       # percentage points, as the campaign's source gate


# ---- names and selection ------------------------------------------------------

def folder_name(dataset: str, backbone: str) -> str:
    """``<dataset>_<backbone>_<dim>``, e.g. ``imagenet_ff2048_2048``."""
    from .config import BACKBONES
    return f"{dataset}_{backbone}_{BACKBONES[backbone].dim}"


def all_datasets() -> list[str]:
    from .benchmark.data import DATASETS
    return [IMAGENET, *DATASETS]


def all_backbones() -> list[str]:
    from .config import BACKBONES
    return list(BACKBONES)


def selection(datasets=None, backbones=None) -> list[tuple[str, str]]:
    """(dataset, backbone) pairs, ImageNet first (downstream needs its weight record)."""
    datasets = list(datasets or all_datasets())
    backbones = list(backbones or all_backbones())
    unknown = sorted(set(datasets) - set(all_datasets())) + sorted(set(backbones) - set(all_backbones()))
    if unknown:
        raise ValueError(f"unknown datasets/backbones: {', '.join(unknown)}")
    ordered = sorted(datasets, key=lambda name: name != IMAGENET)
    return [(dataset, backbone) for dataset in ordered for backbone in backbones]


# ---- local folders -------------------------------------------------------------

def local_complete(root: Path, folder: str, *, verify_hashes: bool = False) -> bool:
    """Every file listed in folder.json is present with its size (and hash if asked)."""
    manifest = read_json(root / folder / FOLDER_MANIFEST)
    if not manifest or manifest.get("format") != FORMAT or not manifest.get("files"):
        return False
    for name, entry in manifest["files"].items():
        path = root / folder / name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return False
        if verify_hashes and sha256_file(path) != entry["sha256"]:
            return False
    return True


def write_folder_manifest(root: Path, folder: str, *, dataset: str, backbone: str, recipe: str,
                          weights: dict, extra: dict | None = None) -> dict:
    """Hash every file of the folder; written last, so it marks a complete folder."""
    from .config import BACKBONES
    directory = root / folder
    files = {path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
             for path in sorted(directory.iterdir())
             if path.is_file() and path.name != FOLDER_MANIFEST and not path.name.endswith(".partial")}
    if not files:
        raise RuntimeError(f"{directory}: nothing to describe")
    manifest = {"format": FORMAT, "folder": folder, "dataset": dataset, "backbone": backbone,
                "dim": BACKBONES[backbone].dim, "recipe": recipe, "weights": weights,
                "files": files, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
    atomic_json(directory / FOLDER_MANIFEST, manifest)
    return manifest


# ---- the Hub -------------------------------------------------------------------

class Hub:
    """The one repository: list, pull a folder, push a folder and verify it."""

    def __init__(self, api, repo: str, *, token: str | None = None):
        self.api, self.repo, self.token = api, repo, token
        self._files: set[str] | None = None

    @classmethod
    def connect(cls, repo: str | None = None):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Set HF_TOKEN (read ImageNet, write the embeddings repository)")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        if repo is None:
            account = api.whoami().get("name")
            if not account:
                raise RuntimeError("HF_TOKEN did not resolve to a Hugging Face account")
            repo = f"{account}/{REPOSITORY_NAME}"
        return cls(api, repo, token=token)

    def files(self) -> set[str]:
        if self._files is None:
            from huggingface_hub.utils import RepositoryNotFoundError
            try:
                self._files = set(self.api.list_repo_files(self.repo, repo_type="dataset"))
            except RepositoryNotFoundError:
                self._files = set()
        return self._files

    def complete(self, folder: str) -> bool:
        return f"{folder}/{FOLDER_MANIFEST}" in self.files()

    def manifest(self, folder: str) -> dict | None:
        """Only the small folder.json, e.g. to read a backbone's weight record."""
        if not self.complete(folder):
            return None
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(self.repo, f"{folder}/{FOLDER_MANIFEST}", repo_type="dataset", token=self.token)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def pull(self, folder: str, root: Path) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo, repo_type="dataset", allow_patterns=[f"{folder}/*"],
                          local_dir=str(root), token=self.token)
        if not local_complete(root, folder, verify_hashes=True):
            raise RuntimeError(f"{root / folder}: downloaded files do not match {FOLDER_MANIFEST}")

    def push(self, folder: str, root: Path) -> None:
        """Data files first, folder.json last, then check every remote hash."""
        self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
        if not self.api.repo_info(self.repo, repo_type="dataset").private:
            raise RuntimeError(f"{self.repo} is public; make it private before uploading ImageNet-derived features")
        manifest = read_json(root / folder / FOLDER_MANIFEST)
        data = sorted(manifest["files"])
        self.api.upload_folder(repo_id=self.repo, repo_type="dataset", folder_path=str(root / folder),
                               path_in_repo=folder, allow_patterns=data,
                               commit_message=f"{folder}: embeddings")
        self.api.upload_file(path_or_fileobj=str(root / folder / FOLDER_MANIFEST),
                             path_in_repo=f"{folder}/{FOLDER_MANIFEST}", repo_id=self.repo,
                             repo_type="dataset", commit_message=f"{folder}: complete")
        self.verify(folder, root)
        self.files().update(f"{folder}/{name}" for name in [*data, FOLDER_MANIFEST])

    def verify(self, folder: str, root: Path) -> None:
        """Remote LFS SHA-256 (or git blob SHA-1) of every file equals the local one."""
        names = [*read_json(root / folder / FOLDER_MANIFEST)["files"], FOLDER_MANIFEST]
        revision = self.api.repo_info(self.repo, repo_type="dataset").sha
        entries = {entry.path: entry for entry in self.api.get_paths_info(
            self.repo, [f"{folder}/{name}" for name in names], repo_type="dataset", revision=revision)}
        for name in names:
            path, entry = root / folder / name, entries.get(f"{folder}/{name}")
            if entry is None or getattr(entry, "size", None) != path.stat().st_size:
                raise RuntimeError(f"{self.repo}: {folder}/{name} missing or of the wrong size")
            lfs = getattr(entry, "lfs", None)
            if lfs:
                expected = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
                observed = sha256_file(path)
            else:
                expected = entry.blob_id
                digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 << 20), b""):
                        digest.update(chunk)
                observed = digest.hexdigest()
            if observed != expected:
                raise RuntimeError(f"{self.repo}: {folder}/{name} hash differs from the local file")


# ---- building ------------------------------------------------------------------

def _dense_gate(root: Path, folder: str, backbone: str, cache: Path, device: str) -> dict | None:
    """The campaign's source gate: dense 1-NN within 0.3 pp of the published value."""
    from .config import BACKBONES
    spec = BACKBONES[backbone]
    if spec.reference_dense_top1 is None:
        return None
    from .evaluate.knn import dense_top1
    from .io import open_npz_memmap
    arrays = {}
    for split in ("train", "val"):
        path = root / folder / f"1K_{split}_{backbone}.npz"
        sidecar = read_json(path.with_name(path.name + ".json")) or {}
        arrays[split] = open_npz_memmap(path, cache / "features", expected_sha256=sidecar.get("sha256"))
    try:
        result = dense_top1(arrays["train"][0], arrays["train"][1], arrays["val"][0], arrays["val"][1],
                            device=device, dim=spec.dim)
    finally:
        for features, _, _ in arrays.values():
            mmap = getattr(features, "_mmap", None)
            if mmap is not None:
                mmap.close()
    if abs(result["top1"] - spec.reference_dense_top1) > DENSE_GATE:
        raise RuntimeError(f"{backbone}: dense 1-NN {result['top1']:.2f} is more than {DENSE_GATE} pp from "
                           f"the published {spec.reference_dense_top1}; ImageNet images are kept")
    return {"top1": result["top1"], "reference": spec.reference_dense_top1}


def build_imagenet(backbones, root: Path, workspace: Path, hub: Hub, *, device, workers, batch_size,
                   keep_raw: bool) -> None:
    """Download ImageNet once, extract every missing backbone, push, then delete the JPEGs."""
    from .config import BACKBONES
    from .data.imagenet import EXPECTED_ROWS, imagenet_folder_stats, prepare_imagenet
    from .embed.direct import RECIPE, extract_source, preflight_backbones
    from .benchmark.storage import remove_owned

    cache = root / ".cache"
    image_root = workspace / IMAGENET
    preflight_backbones(backbones, cache, device)     # fail before a 150 GB download
    rows = sum(EXPECTED_ROWS.values())
    reserve = sum(rows * (BACKBONES[name].dim * 4 + 8) * 2 for name in backbones) + (10 << 30)
    provenance = prepare_imagenet(image_root, token=os.environ.get("HF_TOKEN"), reserve_bytes=reserve)
    for split, key in (("train", "train"), ("val", "validation")):
        stats = imagenet_folder_stats(image_root / split)
        if stats["images"] != EXPECTED_ROWS[key] or stats["classes"] != 1000:
            raise RuntimeError(f"{image_root / split}: incomplete ImageNet; nothing extracted")
    for backbone in backbones:
        folder = folder_name(IMAGENET, backbone)
        results = extract_source(BACKBONES[backbone], image_root, root / folder, cache,
                                 device=device, workers=workers, batch_size=batch_size)
        dense = _dense_gate(root, folder, backbone, cache, device)
        write_folder_manifest(root, folder, dataset=IMAGENET, backbone=backbone, recipe=RECIPE,
                              weights=results["train"]["weights"],
                              extra={"source": {"repository": provenance["repository"],
                                                "revision": provenance["revision"]},
                                     "splits": {"train": "1K_train", "val": "1K_val"}, "dense": dense})
        hub.push(folder, root)
        print(f"{folder}: built, uploaded and verified", flush=True)
    if not keep_raw:
        for split in ("train", "val"):
            remove_owned(image_root / split, image_root)
        remove_owned(image_root / "_mpsae" / "hub", image_root)
        print(f"{image_root}: ImageNet images deleted (every selected folder is on {hub.repo})", flush=True)


def _imagenet_weights(backbone: str, root: Path, hub: Hub) -> dict:
    """The backbone weights recorded for its ImageNet embeddings (local, else the Hub)."""
    folder = folder_name(IMAGENET, backbone)
    manifest = read_json(root / folder / FOLDER_MANIFEST) or hub.manifest(folder)
    if not manifest:
        raise RuntimeError(f"{backbone}: build {folder} first; downstream features must use the "
                           "same backbone weights as the ImageNet embeddings")
    return manifest["weights"]


def build_downstream(dataset: str, backbones, root: Path, hub: Hub, *, device, workers, batch_size,
                     keep_raw: bool) -> None:
    """Download one dataset, extract every missing backbone, push, then delete its raw copy."""
    from .config import BACKBONES
    from .io import canonical_hash
    from .benchmark.data import DATASETS, load_dataset_bundle
    from .benchmark.extract import RECIPE, extract_bundle
    from .benchmark.storage import remove_owned

    cache = root / ".cache"
    raw = root / ".raw" / dataset
    staging = root / ".staging"
    weights = {backbone: _imagenet_weights(backbone, root, hub) for backbone in backbones}
    spec = DATASETS[dataset]
    rows = sum(spec.expected_rows[split] for split in {spec.gallery_split, spec.query_split})
    reserve = (10 << 30) + 2 * rows * sum(BACKBONES[name].dim * 4 + 8 for name in backbones)
    source = load_dataset_bundle(dataset, raw, token=os.environ.get("HF_TOKEN"), reserve_bytes=reserve)
    try:
        for backbone in backbones:
            folder = folder_name(dataset, backbone)
            bundle = extract_bundle(BACKBONES[backbone], source, staging, cache, device=device,
                                    batch_size=batch_size, workers=workers, token=os.environ.get("HF_TOKEN"),
                                    expected_weights=weights[backbone], reclaim_sink=True)
            directory = root / folder
            directory.mkdir(parents=True, exist_ok=True)
            splits = {}
            for split in {id(bundle.gallery): bundle.gallery, id(bundle.query): bundle.query}.values():
                mmap = getattr(split.features, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                os.replace(split.path, directory / f"{split.name}.npz")
                os.replace(split.path.with_name(split.path.name + ".json"), directory / f"{split.name}.npz.json")
                atomic_json(directory / f"{split.name}.ids.json", split.ids)
                splits[split.name] = {"rows": split.rows, "ids_sha256": canonical_hash(split.ids)}
            atomic_json(directory / "bundle.json", {
                "dataset": dataset, "protocol": bundle.protocol, "backbone": backbone,
                "gallery": bundle.gallery.name, "query": bundle.query.name,
                "exclude_self": bundle.exclude_self, "identity": bundle.identity, "manifest": bundle.manifest})
            write_folder_manifest(root, folder, dataset=dataset, backbone=backbone, recipe=RECIPE,
                                  weights=weights[backbone],
                                  extra={"source": {"repository": spec.repository,
                                                    "revision": source.identity["revision"]},
                                         "protocol": bundle.protocol, "splits": splits})
            hub.push(folder, root)
            print(f"{folder}: built, uploaded and verified", flush=True)
    finally:
        del source
        shutil.rmtree(staging, ignore_errors=True)
    if not keep_raw:
        remove_owned(raw, root)
        print(f"{dataset}: raw download deleted", flush=True)


# ---- orchestration -------------------------------------------------------------

PLAN = {(True, True): "use", (True, False): "push", (False, True): "pull", (False, False): "build"}
PLAN_TEXT = {"use": "local and Hub (use local)", "push": "local only (push)",
             "pull": "Hub only (pull)", "build": "missing (build, push)"}


def plan(pairs, root: Path, hub: Hub) -> dict[str, str]:
    """{folder: use | push | pull | build} without changing anything."""
    state = {}
    for dataset, backbone in pairs:
        folder = folder_name(dataset, backbone)
        state[folder] = PLAN[local_complete(root, folder), hub.complete(folder)]
    return state


def sync(pairs, root: Path, workspace: Path, hub: Hub, *, device="cuda:0", workers=16, batch_size=256,
         keep_raw=False, builders=None) -> dict[str, str]:
    """Make every selected folder complete locally and on the Hub; return {folder: what was done}."""
    builders = builders or {"imagenet": build_imagenet, "downstream": build_downstream}
    root.mkdir(parents=True, exist_ok=True)
    state = plan(pairs, root, hub)
    done = {}
    for folder, action in state.items():
        if action == "use":
            done[folder] = "used local"
        elif action == "push":
            if not local_complete(root, folder, verify_hashes=True):
                raise RuntimeError(f"{root / folder}: a file differs from {FOLDER_MANIFEST}; not pushing it")
            hub.push(folder, root)
            done[folder] = "pushed"
        elif action == "pull":
            hub.pull(folder, root)
            done[folder] = "pulled"
    missing: dict[str, list[str]] = {}
    for dataset, backbone in pairs:
        if state[folder_name(dataset, backbone)] == "build":
            missing.setdefault(dataset, []).append(backbone)
    options = {"device": device, "workers": workers, "batch_size": batch_size, "keep_raw": keep_raw}
    for dataset, backbones in missing.items():
        if dataset == IMAGENET:
            builders["imagenet"](backbones, root, workspace, hub, **options)
        else:
            builders["downstream"](dataset, backbones, root, hub, **options)
        for backbone in backbones:
            done[folder_name(dataset, backbone)] = "built"
    return done


def campaign_selection(campaign) -> tuple[list[str], list[str]]:
    """(datasets, backbones) a campaign reads: ImageNet if enabled, then its downstream sets."""
    datasets = [IMAGENET] if campaign.imagenet.get("enabled", True) else []
    if campaign.downstream.get("enabled", False):
        datasets += list(campaign.downstream["datasets"])
    return datasets, list(campaign.backbones)


def _link(source: Path, target: Path) -> None:
    """Hard link (same disk, no copy); an identical existing file is kept."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        if target.stat().st_size == source.stat().st_size and sha256_file(target) == sha256_file(source):
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:              # another filesystem
        shutil.copy2(source, target)


def hub_folder_in_campaign(campaign_root: Path, dataset: str, backbone: str) -> Path:
    return Path(campaign_root) / "benchmarks" / dataset / "hub" / backbone


def link_campaign(root: Path, campaign_root: Path, pairs) -> list[str]:
    """Place complete folders where the campaign reads them; return the linked folders."""
    linked = []
    for dataset, backbone in pairs:
        folder = folder_name(dataset, backbone)
        if not local_complete(root, folder):
            continue
        names = read_json(root / folder / FOLDER_MANIFEST)["files"]
        if dataset == IMAGENET:
            target = Path(campaign_root) / "embeddings" / backbone
            names = [name for name in names if name.startswith("1K_")]
        else:
            target = hub_folder_in_campaign(campaign_root, dataset, backbone)
            names = [*names, FOLDER_MANIFEST]
        for name in names:
            _link(root / folder / name, target / name)
        linked.append(folder)
    return linked


def load_campaign_hub_bundle(campaign_root: Path, dataset: str, backbone: str, weights: dict | None,
                             cache_dir: Path):
    """The linked downstream folder as a FeatureBundle, or None.

    None unless the folder is complete and its backbone weights (repository,
    revision, file, SHA-256) equal ``weights`` -- the source checkpoint's. The
    campaign's verify_feature_provenance checks the extraction record again.
    """
    from .io import canonical_hash, open_npz_memmap
    from .benchmark.extract import FeatureBundle, FeatureSplit
    directory = hub_folder_in_campaign(campaign_root, dataset, backbone)
    manifest = read_json(directory / FOLDER_MANIFEST)
    payload = read_json(directory / "bundle.json")
    if not manifest or not payload or not weights:
        return None
    keys = ("repository", "revision", "file", "sha256")
    if any(manifest.get("weights", {}).get(key) != weights.get(key) for key in keys):
        return None
    for name, entry in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return None
    splits = {}
    for split in dict.fromkeys((payload["gallery"], payload["query"])):
        archive = directory / f"{split}.npz"
        sidecar = read_json(archive.with_name(archive.name + ".json"))
        ids = read_json(directory / f"{split}.ids.json")
        features, labels, _ = open_npz_memmap(archive, cache_dir / "benchmark_features",
                                              expected_sha256=sidecar["sha256"])
        if len(ids) != len(labels) or canonical_hash(ids) != sidecar.get("ids_sha256"):
            raise RuntimeError(f"{archive}: sample IDs disagree with the archive")
        splits[split] = FeatureSplit(split, features, labels, ids, sidecar, archive)
    return FeatureBundle(dataset, payload["protocol"], backbone, splits[payload["gallery"]],
                         splits[payload["query"]], payload["exclude_self"], payload["identity"],
                         payload["manifest"])


def _names(value):
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def main(argv=None) -> int:
    from .cli import load_dotenv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "sync"))
    parser.add_argument("--config", help="campaign YAML: take its datasets/backbones and link the folders for it")
    parser.add_argument("--datasets", help=f"comma-separated, default all: {','.join(all_datasets())}")
    parser.add_argument("--backbones", help=f"comma-separated, default all: {','.join(all_backbones())}")
    parser.add_argument("--workspace", default=os.environ.get("MPSAE_WORKSPACE", "work"))
    parser.add_argument("--repo", default=os.environ.get("MPSAE_EMBEDDINGS_REPO") or None,
                        help=f"owner/name (or MPSAE_EMBEDDINGS_REPO); default <HF account>/{REPOSITORY_NAME}, "
                             "created private if missing")
    parser.add_argument("--device", default=os.environ.get("MPSAE_DEVICE", "cuda:0"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MPSAE_WORKERS", "16")))
    parser.add_argument("--batch-size", "--embed-batch-size", dest="batch_size", type=int,
                        default=int(os.environ.get("MPSAE_EMBED_BATCH", "256")))
    parser.add_argument("--keep-raw", action="store_true", help="do not delete downloaded images after upload")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    load_dotenv(args.env_file)
    workspace = Path(args.workspace)
    root = workspace / "embeddings-hub"
    campaign_root = None
    datasets, backbones = _names(args.datasets), _names(args.backbones)
    if args.config:
        from .benchmark.config import load_campaign
        from .benchmark.run import campaign_paths
        campaign = load_campaign(args.config)
        selected = campaign_selection(campaign)
        datasets, backbones = datasets or selected[0], backbones or selected[1]
        campaign_root = campaign_paths(campaign, workspace)["workspace"]
    pairs = selection(datasets, backbones)
    hub = Hub.connect(args.repo)
    print(f"repository: https://huggingface.co/datasets/{hub.repo}\nlocal: {root}")
    if args.command == "status":
        for folder, action in plan(pairs, root, hub).items():
            print(f"  {folder:34s} {PLAN_TEXT[action]}")
        return 0
    for folder, result in sync(pairs, root, workspace, hub, device=args.device, workers=args.workers,
                               batch_size=args.batch_size, keep_raw=args.keep_raw).items():
        print(f"  {folder:34s} {result}")
    if campaign_root is not None:
        linked = link_campaign(root, campaign_root, pairs)
        print(f"linked {len(linked)} folders into {campaign_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
