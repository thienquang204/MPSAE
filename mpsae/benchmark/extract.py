"""Frozen, deterministic backbone features shared by all retrieval methods.

This is an evaluation-only image path. The historical ImageNet extraction
recipes remain unchanged, including their different augmentation policies.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any
import zlib

from ..config import Backbone
from ..io import canonical_hash, open_npz_memmap
from .data import DATASETS, DatasetBundle, DatasetSplit, open_rgb

RECIPE = "frozen-transfer-pil-bilinear-v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class FeatureSplit:
    name: str
    features: Any
    labels: Any
    ids: list[str]
    manifest: dict[str, Any]
    path: Path | None = None

    @property
    def rows(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])


@dataclass
class FeatureBundle:
    name: str
    protocol: str
    backbone: str
    gallery: FeatureSplit
    query: FeatureSplit
    exclude_self: bool
    identity: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def dim(self) -> int:
        return self.gallery.dim


class _ImageRows:
    """Picklable DataLoader dataset; labels stay aligned with source order."""

    def __init__(self, split: DatasetSplit, transform, start: int = 0):
        self.split, self.transform, self.start = split, transform, start

    def __len__(self):
        return self.split.rows - self.start

    def __getitem__(self, index):
        position = self.start + index
        image = open_rgb(self.split.dataset[position]["image"])
        return self.transform(image), self.split.labels[position]


def _expected_weights(spec: Backbone, expected: dict[str, Any] | None) -> dict[str, str] | None:
    if expected is None:
        return None
    if not isinstance(expected, dict):
        raise TypeError("expected_weights must be the original ImageNet archive weights mapping")
    fields = ("repository", "revision", "file", "sha256")
    if any(not isinstance(expected.get(key), str) or not expected[key] for key in fields):
        raise ValueError("source ImageNet manifest needs weights repository, revision, file and sha256")
    pinned = {key: expected[key] for key in fields}
    if not _COMMIT.fullmatch(pinned["revision"]) or not _SHA256.fullmatch(pinned["sha256"]):
        raise ValueError("source backbone weights need an immutable 40-hex revision and 64-hex SHA256")
    repository, _, filename = spec.source.partition(":") if spec.recipe == "mrl" else \
        (spec.source if spec.source.startswith("timm/") else f"timm/{spec.source}", "", "model.safetensors")
    if pinned["repository"] != repository or pinned["file"] != filename:
        raise ValueError(f"source checkpoint weights do not belong to backbone {spec.name}: "
                         f"{pinned['repository']}:{pinned['file']}")
    return pinned


def _check_weights(observed: dict[str, Any], expected: dict[str, str] | None) -> None:
    if expected is not None and any(observed.get(key) != value for key, value in expected.items()):
        raise RuntimeError("frozen transfer backbone weights differ from the source ImageNet "
                           f"checkpoint provenance: expected {expected}, observed {observed}")


def _network(spec: Backbone, cache_dir: Path, device, token: str | None,
             expected_weights: dict[str, Any] | None = None):
    from ..embed.archive import hub_file, pinned_release

    expected = _expected_weights(spec, expected_weights)
    revision = expected["revision"] if expected is not None else "main"
    if spec.recipe == "mrl":
        from ..embed.mrl_ff2048 import _load_model

        repository, _, filename = spec.source.partition(":")
        release = pinned_release(repository, revision=revision, cache_dir=cache_dir, token=token)
        weights, digest = hub_file(release, filename, cache_dir=cache_dir,
                                   token=token, require_lfs=True)
        provenance = {"repository": repository, "revision": release["revision"],
                      "file": filename, "sha256": digest,
                      "feature": "resnet50_blurpool_avgpool"}
        _check_weights(provenance, expected)
        model = _load_model(weights, device)
        return model, {}, provenance
    from ..embed.csrv2_timm import WEIGHT_FILE, _timm_model

    repository = spec.source if spec.source.startswith("timm/") else f"timm/{spec.source}"
    release = pinned_release(repository, revision=revision, cache_dir=cache_dir, token=token)
    _, digest = hub_file(release, WEIGHT_FILE, cache_dir=cache_dir, token=token, require_lfs=True)
    provenance = {"repository": repository, "revision": release["revision"],
                  "file": WEIGHT_FILE, "sha256": digest,
                  "feature": "forward_head(forward_features(x),pre_logits=True)"}
    _check_weights(provenance, expected)
    model, config, tag, override = _timm_model(spec, release, device)
    return model, config, {**provenance, "timm_tag": tag, "img_size_override": override}


def evaluation_transform(spec: Backbone, pretrained_cfg: dict[str, Any] | None = None):
    """One deterministic RGB transform for gallery and queries, with provenance.

    timm normalisation follows the same registry override as ImageNet. The
    evaluation crop ratio is the released CSRv2 ratio, not a newly inferred
    timm crop percentage. PIL interpolation is a recorded downstream recipe.
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    config = pretrained_cfg or {}
    normalization = spec.normalize
    if normalization is None:
        if not config.get("mean") or not config.get("std"):
            raise ValueError(f"{spec.name}: no mean/std in registry or pretrained config")
        normalization = tuple(config["mean"]), tuple(config["std"])
    mean, std = normalization
    resize = 256 if spec.recipe == "mrl" else math.floor(spec.input_size / (224 / 256))
    transform = transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(spec.input_size), transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
    identity = {"color": "RGB", "resize_shorter_side": resize, "center_crop": spec.input_size,
                "interpolation": "PIL_bilinear_antialias", "mean": list(mean), "std": list(std),
                "augmentation": "none", "shuffle": False, "drop_last": False,
                "input_feature_l2_normalization": False}
    return transform, identity


def _read_split(path: Path, source: DatasetSplit, manifest: dict[str, Any],
                cache_dir: Path) -> FeatureSplit:
    import numpy as np

    features, labels, metadata = open_npz_memmap(
        path, cache_dir / "benchmark_features", expected_sha256=manifest["sha256"])
    if metadata["rows"] != source.rows:
        raise RuntimeError(f"{path}: cached features no longer align with source sample IDs")
    if not np.array_equal(labels, np.asarray(source.labels, dtype=np.int64)):
        raise RuntimeError(f"{path}: cached labels no longer align with source sample IDs")
    return FeatureSplit(source.name, features, labels, source.ids, manifest, path)


def extract_bundle(spec: Backbone, bundle: DatasetBundle, out_dir: str | Path,
                   cache_dir: str | Path, *, device: str = "cuda:0", batch_size: int = 256,
                   workers: int = 0, token: str | None = None,
                   expected_weights: dict[str, Any] | None = None,
                   reclaim_sink: bool = False) -> FeatureBundle:
    """Extract each active split once, then share it across methods and budgets."""
    import torch
    from torch.utils.data import DataLoader

    from ..embed.archive import FeatureSink, finalize_archive, library_versions, load_current_archive
    from ..logs import Progress

    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} is not available")
    expected = _expected_weights(spec, expected_weights)
    if target.type == "cuda" and expected is None:
        raise ValueError("production frozen transfer needs original ImageNet backbone weight "
                         "provenance; expected_weights is missing")
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    network, config, weights = _network(spec, cache_dir, target, token, expected)
    _check_weights(weights, expected)
    network.eval().requires_grad_(False)
    transform, preprocessing = evaluation_transform(spec, config)
    # Extraction is deterministic float32 on CPU (offline tests); production
    # uses the baseline's fp16-compute/fp32-storage contract on CUDA.
    precision = "autocast_fp16_stored_float32" if target.type == "cuda" else "cpu_float32"
    identity = {"recipe": RECIPE, "dataset": bundle.identity, "backbone": spec.name,
                "dim": spec.dim, "weights": weights, "preprocessing": preprocessing,
                "precision": precision, "batch_size": batch_size,
                "source_weights_verified": expected is not None,
                "tf32": False, "libraries": library_versions()}
    directory = out_dir / bundle.name / bundle.protocol / spec.name / canonical_hash(identity)
    loaded: dict[str, FeatureSplit] = {}
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for split_name in dict.fromkeys((bundle.gallery_split, bundle.query_split)):
            source = bundle.splits[split_name]
            split_identity = {**identity, "split": split_name,
                              "ids_sha256": canonical_hash(source.ids),
                              "labels_sha256": canonical_hash(source.labels)}
            path = directory / f"{split_name}.npz"
            existing = load_current_archive(path, identity=split_identity)
            if existing is None:
                with FeatureSink(cache_dir, rows=source.rows, dim=spec.dim,
                                 identity=split_identity, stage=f"benchmark {bundle.name}/{spec.name}") as sink:
                    if sink.remaining:
                        loader = DataLoader(_ImageRows(source, transform, start=sink.cursor),
                                            batch_size=batch_size, num_workers=workers,
                                            shuffle=False, drop_last=False,
                                            pin_memory=target.type == "cuda")
                        progress = Progress(source.rows, done=sink.cursor,
                                            stage=f"benchmark {bundle.name}/{spec.name}/{split_name}",
                                            what="images")
                        autocast = torch.autocast("cuda", dtype=torch.float16) \
                            if target.type == "cuda" else nullcontext()
                        with torch.inference_mode(), autocast:
                            for images, labels in loader:
                                if reclaim_sink and target.type == "cuda":
                                    from ..embed.direct import forward_batches
                                    features = forward_batches(network, images, spec, target)
                                    sink.write(features, labels.cpu().numpy())
                                    progress.advance(len(images))
                                    continue
                                images = images.to(target, non_blocking=True)
                                features = network(images) if spec.recipe == "mrl" else \
                                    network.forward_head(network.forward_features(images), pre_logits=True)
                                if features.ndim != 2 or features.shape[1] != spec.dim:
                                    raise RuntimeError(f"{spec.name}: expected [batch,{spec.dim}], "
                                                       f"got {tuple(features.shape)}")
                                sink.write(features.float().cpu().numpy(), labels.cpu().numpy())
                                progress.advance(len(images))
                    features, labels = sink.publish()
                existing = finalize_archive(path, features, labels, manifest={
                    **split_identity, "identity": canonical_hash(split_identity),
                    "training_on_target": False,
                    "deviations": ["downstream evaluation uses deterministic PIL bilinear RGB "
                                   "preprocessing for every split; no FFCV train augmentation"],
                })
                if reclaim_sink:
                    from ..embed.direct import release_sink
                    release_sink(sink, features, labels)
            loaded[split_name] = _read_split(path, source, existing, cache_dir)
            if reclaim_sink:
                from ..embed.direct import reclaim_verified_writer
                reclaim_verified_writer(cache_dir, existing)
    finally:
        del network
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
        if target.type == "cuda":
            torch.cuda.empty_cache()
    feature_identity = {"extraction": canonical_hash(identity), "dataset": bundle.identity,
                        "backbone": spec.name,
                        "splits": {name: {"sha256": split.manifest["sha256"],
                                          "ids_sha256": canonical_hash(split.ids)}
                                   for name, split in loaded.items()}}
    return FeatureBundle(bundle.name, bundle.protocol, spec.name, loaded[bundle.gallery_split],
                         loaded[bundle.query_split], bundle.exclude_self, feature_identity,
                         {"dataset": bundle.manifest, "extraction": identity,
                          "signature": canonical_hash(feature_identity)})


def make_synthetic_feature_bundle(spec: Backbone, dataset_name: str, out_dir: str | Path,
                                  cache_dir: str | Path, *, seed: int = 1227) -> FeatureBundle:
    """Small class-centred arrays for an explicitly non-scientific smoke run."""
    import numpy as np

    from ..embed.archive import finalize_archive, load_current_archive

    dataset = DATASETS[dataset_name]
    classes, gallery_per_class, query_per_class = 5, 12, 6
    generator = np.random.default_rng([seed, zlib.crc32(dataset_name.encode()),
                                       zlib.crc32(spec.name.encode())])
    centres = generator.normal(size=(classes, spec.dim)).astype(np.float32)
    identity = {"recipe": "synthetic-smoke-frozen-transfer-v1", "dataset": dataset_name,
                "backbone": spec.name, "dim": spec.dim, "seed": seed,
                "classes": classes, "gallery_per_class": gallery_per_class,
                "query_per_class": query_per_class, "scientific": False}
    directory = Path(out_dir) / dataset_name / "synthetic_smoke_v1" / spec.name / canonical_hash(identity)
    loaded = {}
    split_names = ("test",) if dataset.exclude_self else ("train", "test")
    for name in split_names:
        count = gallery_per_class if name == "train" or dataset.exclude_self else query_per_class
        labels = np.repeat(np.arange(classes), count).astype(np.int64)
        features = centres[labels] + generator.normal(scale=0.25, size=(len(labels), spec.dim)).astype(np.float32)
        ids = [f"synthetic:{dataset_name}:{name}:{index}" for index in range(len(labels))]
        source = DatasetSplit(name, None, ids, labels.tolist())
        split_identity = {**identity, "split": name, "ids_sha256": canonical_hash(ids)}
        path = directory / f"{name}.npz"
        manifest = load_current_archive(path, identity=split_identity)
        if manifest is None:
            manifest = finalize_archive(path, features, labels, manifest={
                **split_identity, "identity": canonical_hash(split_identity),
                "precision": "synthetic_float32", "training_on_target": False,
                "deviations": ["SYNTHETIC DATA: smoke rehearsal, not a retrieval benchmark"],
            })
        loaded[name] = _read_split(path, source, manifest, Path(cache_dir))
    gallery = loaded["test" if dataset.exclude_self else "train"]
    query = loaded["test"]
    return FeatureBundle(dataset_name, "synthetic_smoke_v1", spec.name, gallery, query,
                         dataset.exclude_self, identity,
                         {**identity, "protocol": dataset.protocol, "scientific": False})
