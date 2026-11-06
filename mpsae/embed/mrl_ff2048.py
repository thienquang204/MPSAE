"""FF2048 embeddings with the MRL retrieval recipe.

Follows ``MRL/inference/pytorch_inference.py --retrieval`` and its ``utils.py``:
a BlurPool ResNet-50 fixed-feature checkpoint, the *test* transform for both the
retrieval database and the query set, source order, every image kept, the
2048-d avgpool activation taken under fp16 autocast and stored as float32.

This is the backbone of CSR Table 4, so the published dense 1-NN top-1 (71.19)
is the gate that tells us the recipe was reproduced; the gate itself runs in the
evaluation stage, this module only records what it is expected to be.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ..config import Backbone
from ..io import canonical_hash
from ..logs import Progress, get_logger
from .archive import (PRECISION, FeatureSink, finalize_archive, hub_file, library_versions,
                      load_current_archive, pinned_release)

RECIPE = "mrl-pytorch_inference-retrieval-v1"
#: MRL's ``IMG_SIZE``: the shorter side the image is resized to before cropping.
RESIZE_SIZE = 256
#: ``Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize``. MRL uses this
#: for the database as well as the queries, so train and val rows are comparable.
TRANSFORM = "resize256_shorter_side;center_crop224;to_tensor;normalize(mean,std);no_augmentation"
SPLIT_FOLDER = {"train": "train", "validation": "val", "val": "val"}

#: Differences from the released script that are bookkeeping, not recipe.
DEVIATIONS = (
    "labels stored as int64 instead of the released float16 (exact for 0..999, but a float "
    "label key is a trap for the contrastive loss)",
    "one NPZ {data, labels} per split instead of the released separate -X.npy/-y.npy, so "
    "features and labels can never be paired up in the wrong order",
)


def blurpool_class():
    """Build MRL's BlurPool wrapper. Defined lazily so importing needs no torch."""
    import torch
    from torch import nn
    import torch.nn.functional as functional

    class _BlurPoolConv2d(nn.Module):
        """FFCV/MRL anti-aliasing blur applied before every strided convolution.

        Reference: libffcv/ffcv-imagenet ``train_imagenet.py`` and MRL
        ``utils.py``. The checkpoint only loads strictly if the wrapping matches,
        so this is what makes ``strict=True`` meaningful.
        """

        def __init__(self, conv: "nn.Conv2d") -> None:
            super().__init__()
            self.conv = conv
            kernel = torch.tensor([1.0, 2.0, 1.0])
            kernel = torch.outer(kernel, kernel).reshape(1, 1, 3, 3) / 16
            self.register_buffer("blur_filter", kernel.repeat(conv.in_channels, 1, 1, 1))

        def forward(self, x):
            blurred = functional.conv2d(x, self.blur_filter, padding=1,
                                        groups=self.conv.in_channels)
            return self.conv(blurred)

    return _BlurPoolConv2d


def apply_blurpool(module) -> None:
    """Wrap every strided convolution with >=16 input channels, as MRL does."""
    from torch import nn

    wrapper = blurpool_class()

    def walk(parent) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d) and max(child.stride) > 1 and child.in_channels >= 16:
                setattr(parent, name, wrapper(child))
            else:
                walk(child)

    walk(module)


def _clean_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Strip DDP/Sequential wrappers and the classifier, keeping everything else.

    Never ``strict=False``: an unexpected or missing feature tensor means the
    architecture is not the one the weights were trained in.
    """
    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        while key.split(".", 1)[0] in ("module", "model", "0") and "." in key:
            key = key.split(".", 1)[1]
        if key.startswith("fc."):
            continue
        if key in cleaned:
            raise RuntimeError(f"duplicate tensor {key} after removing wrapper prefixes")
        cleaned[key] = value
    if not cleaned:
        raise RuntimeError("MRL checkpoint carries no feature tensors")
    return cleaned


def _load_model(weights: Path, device):
    import torch
    from torch import nn
    from torchvision import models

    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    network = models.resnet50(weights=None)
    apply_blurpool(network)
    network.fc = nn.Identity()  # the 2048-d avgpool output becomes the model output
    network.load_state_dict(_clean_state_dict(state), strict=True)
    return network.eval().requires_grad_(False).to(device)


def _dataset(directory: Path, spec: Backbone):
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    if spec.normalize is None:
        raise ValueError(f"{spec.name}: the MRL recipe has no pretrained_cfg to fall back on; "
                         "pin mean/std in the backbone registry")
    mean, std = spec.normalize
    transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(spec.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
    dataset = ImageFolder(str(directory), transform=transform)
    if len(dataset.classes) != 1000 or any(not re.fullmatch(r"n[0-9]{8}", name)
                                           for name in dataset.classes):
        raise RuntimeError(f"{directory} must hold all 1,000 n######## synset folders, found "
                           f"{len(dataset.classes)}")
    return dataset, (mean, std)


def extract_ff2048(spec: Backbone, imagenet_root: str | Path, out_dir: str | Path, *,
                   split: str, cache_dir: str | Path, device: str = "cuda:0",
                   batch_size: int = 256, workers: int = 8, download: bool = True,
                   hf_token: str | None = None) -> dict[str, Any]:
    """Extract one split into ``<out_dir>/1K_{train,val}_<backbone>.npz``.

    Returns the archive manifest, which is also written next to the archive.
    """
    import torch

    if spec.recipe != "mrl":
        raise ValueError(f"{spec.name} uses the {spec.recipe} recipe, not mrl")
    if split not in SPLIT_FOLDER:
        raise ValueError(f"split must be one of {sorted(SPLIT_FOLDER)}, got {split!r}")
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    repo, _, filename = spec.source.partition(":")
    if not repo or not filename:
        raise ValueError(f"{spec.name}: source must be '<hf repo>:<file>', got {spec.source!r}")

    from ..data.imagenet import imagenet_folder_stats

    folder = SPLIT_FOLDER[split]
    log = get_logger(f"ff2048:{folder}")
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    directory = Path(imagenet_root) / folder
    out_path = out_dir / f"1K_{folder}_{spec.name}.npz"

    release = pinned_release(repo, repo_type="model", revision="main", cache_dir=cache_dir,
                             download=download, token=hf_token)
    weights, weights_sha = hub_file(release, filename, cache_dir=cache_dir, download=download,
                                    token=hf_token, require_lfs=True)
    source = imagenet_folder_stats(directory)
    weight_provenance = {"repository": repo, "file": filename, "revision": release["revision"],
                         "sha256": weights_sha, "architecture": "torchvision.resnet50+blurpool",
                         "classifier": "fc=Identity", "state_dict_load": "strict"}
    identity = {"recipe": RECIPE, "backbone": spec.name, "split": folder, "dim": spec.dim,
                "transform": TRANSFORM, "normalize": spec.normalize,
                "input_size": spec.input_size, "batch_size": batch_size,
                "precision": PRECISION, "order": "imagefolder_sorted_no_shuffle",
                "drop_last": False, "weights": weight_provenance,
                "imagenet": {key: source[key] for key in ("classes", "images", "fingerprint")}}
    if (existing := load_current_archive(out_path, identity=identity)) is not None:
        log.info("skip %s (archive matches its manifest)", out_path.name)
        return existing

    dataset, (mean, std) = _dataset(directory, spec)
    rows = len(dataset)
    if rows != source["images"]:
        raise RuntimeError(f"{directory}: ImageFolder sees {rows:,} images, the folder scan saw "
                           f"{source['images']:,}")
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"the MRL recipe extracts under CUDA fp16 autocast; {device} is "
                           "unavailable")

    with FeatureSink(cache_dir, rows=rows, dim=spec.dim, identity=identity,
                     stage=f"ff2048:{folder}") as sink:
        if sink.remaining:
            from torch.utils.data import DataLoader, Subset

            network = _load_model(weights, target)
            loader = DataLoader(Subset(dataset, range(sink.cursor, rows)), shuffle=False,
                                batch_size=batch_size, num_workers=workers, pin_memory=True,
                                drop_last=False,
                                **({"prefetch_factor": 2} if workers else {}))
            progress = Progress(rows, stage=f"ff2048:{folder}", what="images", done=sink.cursor)
            log.info("extracting %s rows from %s (%s classes) with %s workers",
                     f"{sink.remaining:,}", directory, source["classes"], workers)
            with torch.inference_mode():
                for images, labels in loader:
                    images = images.to(target, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        features = network(images)
                    features = features.float()
                    if features.ndim != 2 or features.shape[1] != spec.dim:
                        raise RuntimeError(f"expected (n, {spec.dim}) features, got "
                                           f"{tuple(features.shape)}")
                    sink.write(features.cpu().numpy(), labels.cpu().numpy())
                    progress.advance(len(images))
            del network
            torch.cuda.empty_cache()
        features, labels = sink.publish()

    manifest = {
        **identity,
        "identity": canonical_hash(identity),
        "recipe_reference": "MRL/inference/pytorch_inference.py --retrieval; MRL/utils.py "
                            "generate_retrieval_data",
        "preprocessing": {"resize": RESIZE_SIZE, "crop": spec.input_size,
                          "interpolation": "PIL bilinear", "mean": list(mean), "std": list(std),
                          "augmentation": "none", "shuffle": False, "drop_last": False,
                          "order": "ImageFolder sorted classes then sorted files"},
        "autocast": "torch.autocast('cuda', float16)",
        "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "seed": None,  # nothing in this recipe is random
        "imagenet": {key: value for key, value in source.items() if key != "per_class"},
        "expected_dense_top1": spec.reference_dense_top1,
        "deviations": list(DEVIATIONS),
        "libraries": library_versions(),
    }
    return finalize_archive(out_path, features, labels, manifest=manifest)
