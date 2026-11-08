"""timm embeddings with the CSRv2 extraction recipe.

Follows ``CSRv2/image/pretrained_embeddings.py``: the database is read through
FFCV with ``RandomResizedCrop`` plus horizontal flip in random order with
``drop_last=True``, the query set with a centre crop in sequential order, the
feature is ``forward_head(forward_features(x), pre_logits=True)`` under fp16
autocast, stored as float32.

Three plumbing bugs of the released pipeline are fixed, not three recipe
changes: the labels travel in the same batch as their features (the released
``stack_emb.py`` concatenated feature chunks in ``os.listdir`` order and dropped
the labels), the rows are written once in loader order, and the archive key is
``labels``. Two real deviations are unavoidable for ViT/DINOv2 and are recorded
in the manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Backbone
from ..io import canonical_hash, read_json
from ..logs import Progress, get_logger
from .archive import (PRECISION, FeatureSink, finalize_archive, hub_file, library_versions,
                      load_current_archive, pinned_release)

RECIPE = "csrv2-pretrained_embeddings-v1"
SPLIT_FOLDER = {"train": "train", "validation": "val", "val": "val"}
#: ``DEFAULT_CROP_RATIO`` of the released script and of the FFCV ImageNet recipe.
CROP_RATIO = 224 / 256
WEIGHT_FILE = "model.safetensors"

FFCV_MISSING = ("ffcv is not importable. The CSRv2 recipe reads its dataset through FFCV; "
                "run 'bash scripts/setup_server.sh', then 'source scripts/activate_server.sh'.")


def _deviations(spec: Backbone, mean, std, source: str) -> list[str]:
    """Departures from the released script, each with the reason it is forced."""
    notes = [
        f"val centre crop side is {spec.input_size} (the model input size), not the hard-coded "
        "256 of the released script: timm ViT-B/16 is fixed at 224 and DINOv2 uses patch 14, so "
        "256 either errors or silently changes the model",
        "the released script set no seed for the FFCV random order and augmentation; the Loader "
        "seed is pinned and recorded so the train rows are reproducible",
    ]
    if tuple(mean) != (0.5, 0.5, 0.5) or tuple(std) != (0.5, 0.5, 0.5):
        notes.insert(1, f"normalisation mean={list(mean)} std={list(std)} taken from {source} "
                        "instead of the hard-coded 0.5: ConvNeXt fb_in1k and DINOv2 were "
                        "pretrained with ImageNet statistics")
    return notes


def _timm_model(spec: Backbone, release: dict[str, Any], device):
    """Create the timm model at the pinned revision and return it with its config."""
    import timm

    tag = spec.source.split("/", 1)[1] if spec.source.startswith("timm/") else spec.source
    default = timm.get_pretrained_cfg(tag)
    default_side = int(default.input_size[-1])
    # Only pass img_size when the checkpoint's native resolution differs, so
    # timm interpolates the position embedding (DINOv2: 518 -> 224) and the
    # fixed-resolution models stay untouched.
    override = spec.input_size if default_side != spec.input_size else None
    extra = {"img_size": override} if override else {}
    # timm reads "<repo>@<revision>" as a pinned Hub revision, which is the only
    # way to stop ``pretrained=True`` from following a moving branch.
    model = timm.create_model(tag, pretrained=True, num_classes=1000,
                              pretrained_cfg_overlay={"hf_hub_id": f"{release['repository']}@"
                                                                   f"{release['revision']}"},
                              **extra)
    config = dict(getattr(model, "pretrained_cfg", {}) or {})
    reported = config.get("input_size")
    if override and reported and int(reported[-1]) != spec.input_size:
        raise RuntimeError(f"{tag}: asked for img_size={spec.input_size} but the model reports "
                           f"input_size={reported}")
    return model.eval().requires_grad_(False).to(device), config, tag, override


def _loader(ffcv_path: Path, *, spec: Backbone, train: bool, batch_size: int, workers: int,
            seed: int, device, mean, std):
    """The released train/val FFCV pipelines, parameterised only where it must be."""
    import numpy as np

    try:
        from ffcv.fields.basics import IntDecoder
        from ffcv.fields.rgb_image import (CenterCropRGBImageDecoder,
                                           RandomResizedCropRGBImageDecoder)
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import (NormalizeImage, RandomHorizontalFlip, Squeeze, ToDevice,
                                     ToTensor, ToTorchImage)
    except ImportError as error:
        raise RuntimeError(FFCV_MISSING) from error

    side = spec.input_size
    scale = np.array(mean, dtype=np.float64) * 255, np.array(std, dtype=np.float64) * 255
    if train:
        decoder = RandomResizedCropRGBImageDecoder((side, side))
        image_pipeline = [decoder, RandomHorizontalFlip(), ToTensor(),
                          ToDevice(device, non_blocking=True), ToTorchImage(),
                          NormalizeImage(scale[0], scale[1], np.float32)]
    else:
        decoder = CenterCropRGBImageDecoder((side, side), ratio=CROP_RATIO)
        image_pipeline = [decoder, ToTensor(), ToDevice(device, non_blocking=True),
                          ToTorchImage(), NormalizeImage(scale[0], scale[1], np.float32)]
    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(device, non_blocking=True)]
    options = {"os_cache": 1} if train else {}
    return Loader(str(ffcv_path), batch_size=batch_size, num_workers=workers,
                  order=OrderOption.RANDOM if train else OrderOption.SEQUENTIAL,
                  drop_last=train, seed=seed,
                  pipelines={"image": image_pipeline, "label": label_pipeline}, **options)


def _rows(loader, *, batch_size: int, train: bool) -> int:
    """Row count of one pass, taken from the FFCV file rather than assumed."""
    batches = len(loader)
    if batches < 1:
        raise RuntimeError("the FFCV loader yields no batches")
    if train:
        # drop_last=True: the tail that does not fill a batch is never seen,
        # which is why the train archive has 1,281,024 rows at batch 256.
        return batches * batch_size
    samples = getattr(getattr(loader, "reader", None), "num_samples", None)
    if samples is None:
        raise RuntimeError("this ffcv version exposes no reader.num_samples; cannot size the "
                           "sequential pass without guessing")
    return int(samples)


def extract_timm(spec: Backbone, ffcv_path: str | Path, out_dir: str | Path, *, split: str,
                 cache_dir: str | Path, device: str = "cuda:0", batch_size: int = 256,
                 workers: int = 16, seed: int = 1227, download: bool = True) -> dict[str, Any]:
    """Extract one split into ``<out_dir>/1K_{train,val}_<backbone>.npz``.

    Returns the archive manifest, which is also written next to the archive.
    """
    import torch

    if spec.recipe != "csrv2":
        raise ValueError(f"{spec.name} uses the {spec.recipe} recipe, not csrv2")
    if split not in SPLIT_FOLDER:
        raise ValueError(f"split must be one of {sorted(SPLIT_FOLDER)}, got {split!r}")
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    folder = SPLIT_FOLDER[split]
    train = folder == "train"
    side = spec.input_size
    ffcv_path, cache_dir, out_dir = Path(ffcv_path), Path(cache_dir), Path(out_dir)
    if not ffcv_path.is_file():
        raise FileNotFoundError(f"{ffcv_path} does not exist; write it with data.ffcv_write first")
    log = get_logger(f"{spec.name}:{folder}")
    out_path = out_dir / f"1K_{folder}_{spec.name}.npz"

    repo = spec.source if spec.source.startswith("timm/") else f"timm/{spec.source}"
    release = pinned_release(repo, repo_type="model", revision="main", cache_dir=cache_dir,
                             download=download, token=None)
    # Hash the weight file ourselves rather than trusting what timm happens to
    # have in its cache; timm loads the same pinned revision below.
    _, weights_sha = hub_file(release, WEIGHT_FILE, cache_dir=cache_dir, download=download,
                              require_lfs=True)
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"the CSRv2 recipe extracts under CUDA fp16 autocast; {device} is "
                           "unavailable")

    model, config, tag, img_size_override = _timm_model(spec, release, target)
    if spec.normalize is not None:
        mean, std = spec.normalize
        normalize_source = "the backbone registry"
    else:
        if not config.get("mean") or not config.get("std"):
            raise RuntimeError(f"{tag} reports no mean/std in its pretrained_cfg; pin them in "
                               "the backbone registry instead")
        mean, std = tuple(config["mean"]), tuple(config["std"])
        normalize_source = f"the {tag} pretrained_cfg"

    dataset = read_json(ffcv_path.with_name(ffcv_path.name + ".json")) or {}
    status = ffcv_path.stat()
    weight_provenance = {"repository": release["repository"], "file": WEIGHT_FILE,
                         "revision": release["revision"], "sha256": weights_sha, "timm_tag": tag,
                         "feature": "forward_head(forward_features(x), pre_logits=True)",
                         "img_size_override": img_size_override}
    identity = {"recipe": RECIPE, "backbone": spec.name, "split": folder, "dim": spec.dim,
                "input_size": spec.input_size, "batch_size": batch_size, "seed": seed,
                "order": "ffcv_random" if train else "ffcv_sequential", "drop_last": train,
                "augmentation": "random_resized_crop+hflip" if train
                else f"center_crop(ratio={CROP_RATIO:.6f})",
                "normalize": {"mean": list(mean), "std": list(std), "source": normalize_source},
                "precision": PRECISION, "weights": weight_provenance,
                # Size and the writer's own source fingerprint, not the mtime:
                # copying the FFCV file to another disk must not invalidate an
                # archive, changing its contents must.
                "ffcv": {"name": ffcv_path.name, "bytes": status.st_size,
                         "identity": dataset.get("identity"),
                         "images": dataset.get("images")}}
    if (existing := load_current_archive(out_path, identity=identity)) is not None:
        log.info("skip %s (archive matches its manifest)", out_path.name)
        return existing

    loader = _loader(ffcv_path, spec=spec, train=train, batch_size=batch_size, workers=workers,
                     seed=seed, device=target, mean=mean, std=std)
    rows = _rows(loader, batch_size=batch_size, train=train)
    if dataset.get("images") and not train and rows != dataset["images"]:
        raise RuntimeError(f"{ffcv_path.name} holds {dataset['images']:,} images but the "
                           f"sequential loader reports {rows:,}")
    log.info("%s: %s rows, %s batches of %s, order=%s", out_path.name, f"{rows:,}",
             f"{len(loader):,}", batch_size, "random" if train else "sequential")

    with FeatureSink(cache_dir, rows=rows, dim=spec.dim, identity=identity,
                     stage=f"{spec.name}:{folder}") as sink:
        if sink.remaining:
            if sink.cursor % batch_size:
                raise RuntimeError(f"cursor {sink.cursor} is not a whole number of batches; "
                                   "the FFCV pass can only be resumed on a batch boundary")
            skip = sink.cursor // batch_size
            if skip:
                # The order is a deterministic function of (seed, epoch 0), so a
                # resumed pass replays it and discards what is already stored.
                log.info("resuming: replaying and discarding %s batches", f"{skip:,}")
            progress = Progress(rows, stage=f"{spec.name}:{folder}", what="images",
                                done=sink.cursor)
            with torch.inference_mode():
                for index, (images, labels) in enumerate(loader):
                    if index < skip:
                        continue
                    with torch.autocast("cuda", dtype=torch.float16):
                        features = model.forward_head(model.forward_features(images),
                                                      pre_logits=True)
                    features = features.float()
                    if features.ndim != 2 or features.shape[1] != spec.dim:
                        raise RuntimeError(f"{tag}: expected (n, {spec.dim}) features, got "
                                           f"{tuple(features.shape)}")
                    sink.write(features.cpu().numpy(), labels.cpu().numpy())
                    progress.advance(len(features))
            del model
            torch.cuda.empty_cache()
        features, labels = sink.publish()

    manifest = {
        **identity,
        "identity": canonical_hash(identity),
        "recipe_reference": "CSRv2/image/pretrained_embeddings.py; "
                            "CSRv2/image/utils.py generate_pretrained_embed",
        "preprocessing": {
            "decoder": (f"RandomResizedCropRGBImageDecoder(({side},{side}))" if train
                        else f"CenterCropRGBImageDecoder(({side},{side}), ratio={CROP_RATIO:.6f})"),
            "augmentation": "RandomHorizontalFlip" if train else "none",
            "tensor_pipeline": "ToTensor -> ToDevice -> ToTorchImage -> NormalizeImage(mean*255, "
                               "std*255, float32)",
            "order": "OrderOption.RANDOM, os_cache=1" if train else "OrderOption.SEQUENTIAL",
            "drop_last": train, "seed": seed,
        },
        "autocast": "torch.autocast('cuda', float16)",
        "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "pretrained_cfg": {key: config.get(key) for key in
                           ("architecture", "tag", "input_size", "crop_pct", "interpolation",
                            "mean", "std", "num_classes", "hf_hub_id")},
        "ffcv_dataset_manifest": dataset or None,
        "ffcv_path": str(ffcv_path.resolve()),
        "ffcv_mtime_ns": status.st_mtime_ns,
        "expected_dense_top1": spec.reference_dense_top1,
        "deviations": _deviations(spec, mean, std, normalize_source),
        "libraries": library_versions(),
    }
    return finalize_archive(out_path, features, labels, manifest=manifest)
