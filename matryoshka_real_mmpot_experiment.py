#!/usr/bin/env python3
"""Train and benchmark vanilla MRL against Matryoshka-MMPOT.

This is a self-contained PyTorch experiment runner derived from the local
``MRL.html``, ``CSR.html``, and ``Matryoshka_plan.md`` files.  The primary
reproduction target is ResNet-50 on ImageNet-1K with nested dimensions
8,16,...,2048.

This runner solves the true multi-marginal objective.  ``--ot-marginals 3``
(the default) builds the rank-3 cost tensor and runs the multi-marginal partial
OT solver, so experiment metadata may legitimately claim a multi-marginal
solve.  ``--ot-marginals 2`` falls back to the averaged B x B pairwise cost with
a two-marginal partial-OT solver, which the plan calls the "pairwise MMPOT
proxy"; that is the objective the older ``matryoshka_mmpot_experiment.py``
implements.

This is the script the Docker pipeline runs (see ``container_pipeline.sh``).

Examples
--------
Full comparison on ImageNet downloaded/cached from Hugging Face:

    python matryoshka_real_mmpot_experiment.py \
      --dataset imagenet --data-root /datasets/huggingface-cache --method both \
      --output-dir runs/imagenet_rn50 --benchmark head,knn,linear

Download and prepare the gated ImageNet train/validation splits only:

    python matryoshka_real_mmpot_experiment.py \
      --dataset imagenet --data-root /datasets/huggingface-cache \
      --prepare-data-only

Quick end-to-end smoke test (still requires a CUDA GPU):

    python matryoshka_real_mmpot_experiment.py \
      --dataset fake --architecture tiny_cnn --nested-dims 8,16,32 \
      --method both --epochs 1 --batch-size 16 --workers 0 \
      --fake-train-size 64 --fake-val-size 32 \
      --benchmark head,knn,linear --probe-epochs 1 \
      --output-dir /tmp/matryoshka_smoke

Resume each arm of a comparison from its ``last.pt`` checkpoint:

    python matryoshka_real_mmpot_experiment.py ... --resume auto

Notes
-----
* ``head`` evaluates the jointly trained Matryoshka classifiers.
* ``--dataset imagenet`` loads ``ILSVRC/imagenet-1k`` through Hugging Face
  Datasets. ImageNet access is gated: accept its terms on the dataset page and
  authenticate with ``hf auth login`` or set ``HF_TOKEN`` before running.
  ``--dataset imagefolder`` remains available for an existing train/val tree.
  The dedicated downloader caches train/validation/test; supervised training
  uses train and all reported metrics use the labelled validation split. The
  unlabelled official test split is not used to compute accuracy.
* ``linear`` resets all classifiers, freezes the encoder, trains fresh probes
  on the training split, and evaluates them on validation.
* ``knn`` uses the training split as the gallery and validation as queries,
  with exact chunked L2 search for every prefix. Features stay on the GPU, so
  full ImageNet extraction needs roughly 10 GiB of VRAM on top of the model for
  float32 ResNet-50 features; use ``--knn-max-train``/``--knn-max-val`` to cap
  it. ``knn`` is not part of the default ``--benchmark head``.
* This runner is CUDA-only: ``--device`` defaults to ``cuda`` and rejects
  ``cpu``/``mps``. Dataset decoding in the DataLoader workers still runs on the
  host CPU, which PyTorch cannot avoid.
* Training defaults to 5 epochs (``--epochs``) and the linear probe to 5 epochs
  (``--probe-epochs``).
* The OT solver runs per local mini-batch in float32. ``envelope`` gradients
  are the memory-efficient default: the converged plan is detached while the
  transport cost remains differentiable with respect to the cost matrix.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

from experiment_reporting import generate_matryoshka_report


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METHODS = ("mrl", "mmpot")


@dataclass
class DatasetBundle:
    train: Dataset
    train_eval: Dataset
    val: Dataset
    num_classes: int
    class_to_idx: Dict[str, int]


class HuggingFaceImageDataset(Dataset):
    """Adapt a Hugging Face map-style image split to torchvision transforms."""

    def __init__(self, split: Any, transform: Any) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.split[index]
        image = sample["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return self.transform(image), int(sample["label"])


@dataclass
class EpochStats:
    loss: float
    mrl_loss: float
    ot_loss: float
    ot_mass: float
    ot_cap_violation: float
    samples: int
    seconds: float
    samples_per_second: float


def parse_csv_ints(value: str) -> List[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def parse_csv_strings(value: str) -> List[str]:
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    if "all" in parsed:
        return ["head", "knn", "linear"]
    invalid = sorted(set(parsed) - {"head", "knn", "linear", "none"})
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown benchmark(s): {', '.join(invalid)}")
    if "none" in parsed and len(parsed) != 1:
        raise argparse.ArgumentTypeError("'none' cannot be combined with other benchmarks")
    return [] if parsed == ["none"] else list(dict.fromkeys(parsed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fair MRL versus Matryoshka-MMPOT training and representation benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset", choices=("imagenet", "imagefolder", "cifar10", "cifar100", "fake"), default="imagenet")
    data.add_argument("--data-root", type=Path, default=Path("data"), help="Hugging Face cache root, or ImageFolder root containing train/ and val/")
    data.add_argument("--download", action="store_true", help="allow torchvision to download CIFAR data")
    data.add_argument("--prepare-data-only", action="store_true", help="download/cache and validate the selected dataset, then exit")
    data.add_argument("--hf-dataset-id", default="ILSVRC/imagenet-1k", help="Hugging Face dataset used by --dataset imagenet")
    data.add_argument("--hf-revision", default="main")
    data.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding a gated-dataset access token; cached hf login is also used")
    data.add_argument("--image-size", type=int, default=0, help="0 selects 224 for ImageNet/Fake and 32 for CIFAR")
    data.add_argument(
        "--require-validated-data",
        action="store_true",
        help="refuse ImageNet training unless the downloader manifest proves every train/validation image decoded",
    )
    data.add_argument("--fake-train-size", type=int, default=1024)
    data.add_argument("--fake-val-size", type=int, default=256)
    data.add_argument("--fake-num-classes", type=int, default=10)

    model = parser.add_argument_group("model and objective")
    model.add_argument("--method", choices=("mrl", "mmpot", "both"), default="both")
    model.add_argument("--architecture", choices=("resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16", "tiny_cnn"), default="resnet50")
    model.add_argument("--pretrained", action="store_true", help="initialize the encoder with torchvision DEFAULT weights")
    model.add_argument("--nested-dims", type=parse_csv_ints, default=None, help="auto: RN50=8,...,2048; ViT-B/16=12,...,768")
    model.add_argument("--tiny-feature-dim", type=int, default=64)
    model.add_argument("--head-style", choices=("independent", "shared"), default="independent", help="shared implements MRL-E weight tying")
    model.add_argument("--mrl-loss-reduction", choices=("mean", "sum"), default="mean", help="mean matches the plan code; sum matches c_m=1 paper notation")
    model.add_argument("--label-smoothing", type=float, default=0.0)

    ot = parser.add_argument_group("pairwise MMPOT proxy")
    ot.add_argument("--ot-lambda", type=float, default=0.5)
    ot.add_argument("--ot-mass", type=float, default=0.8)
    ot.add_argument("--ot-eta", type=float, default=0.1)
    ot.add_argument("--ot-iters", type=int, default=50)
    ot.add_argument("--ot-tol", type=float, default=1e-4)
    ot.add_argument("--ot-grad", choices=("envelope", "unrolled"), default="envelope")
    ot.add_argument("--ot-solver-mode", choices=("cyclic", "greedy"), default="cyclic", help="solver iteration mode: cyclic (looping through all constraints) or greedy")
    ot.add_argument("--ot-marginals", type=int, choices=(2, 3), default=3, help="number of marginals for MMPOT solver: 2 (pairwise proxy) or 3 (3-marginal tensor MMPOT)")
    ot.add_argument("--ot-scale-pairs", choices=("all", "adjacent"), default="all")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--prefetch-factor", type=int, default=2, help="batches prefetched by each DataLoader worker")
    train.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    train.add_argument("--lr", type=float, default=0.1)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--scheduler", choices=("cosine", "step", "none"), default="cosine")
    train.add_argument("--step-milestones", type=parse_csv_ints, default=[30, 60, 80])
    train.add_argument("--step-gamma", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=0.0)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--channels-last", action="store_true")
    train.add_argument("--device", default="cuda", help="cuda or cuda:N; this runner is CUDA-only")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--deterministic", action="store_true")
    train.add_argument("--print-freq", type=int, default=50)
    train.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show tqdm progress, throughput, and ETA for long-running loops",
    )
    train.add_argument("--max-train-batches", type=int, default=0, help="0 means the full epoch")
    train.add_argument("--max-val-batches", type=int, default=0, help="0 means the full validation split")

    output = parser.add_argument_group("checkpointing and output")
    output.add_argument("--output-dir", type=Path, default=Path("runs/matryoshka_comparison"))
    output.add_argument("--resume", default="", help="checkpoint path for one method, or 'auto' to use each method's last.pt")
    output.add_argument("--save-every", type=int, default=0, help="also save epoch_NNNN.pt every N epochs; 0 disables")
    output.add_argument("--selection-metric", choices=("mean_top1", "full_top1"), default="mean_top1")
    output.add_argument("--benchmark-checkpoint", choices=("best", "last"), default="best")

    bench = parser.add_argument_group("benchmarks")
    bench.add_argument("--benchmark", type=parse_csv_strings, default=["head"], help="comma-separated: head,knn,linear,all,none")
    bench.add_argument("--knn-query-chunk", type=int, default=256)
    bench.add_argument("--knn-gallery-chunk", type=int, default=16384)
    bench.add_argument("--knn-normalize", action="store_true", help="L2-normalize each prefix before exact L2 search")
    bench.add_argument("--knn-max-train", type=int, default=0, help="limit gallery samples for development; 0 means all")
    bench.add_argument("--knn-max-val", type=int, default=0, help="limit query samples for development; 0 means all")
    bench.add_argument("--probe-epochs", type=int, default=5)
    bench.add_argument("--probe-lr", type=float, default=0.1)
    bench.add_argument("--probe-weight-decay", type=float, default=0.0)
    bench.add_argument("--probe-max-train-batches", type=int, default=0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 0 or args.batch_size < 1 or args.workers < 0 or args.prefetch_factor < 1:
        raise ValueError("epochs must be non-negative; batch-size/prefetch-factor positive; workers non-negative")
    if args.image_size < 0 or args.fake_train_size < 1 or args.fake_val_size < 1:
        raise ValueError("image and fake dataset sizes must be positive (or image-size=0 for auto)")
    if args.fake_num_classes < 2:
        raise ValueError("fake-num-classes must be at least 2")
    if not 0.0 < args.ot_mass <= 1.0:
        raise ValueError("ot-mass must be in (0, 1]")
    if args.ot_eta <= 0.0 or args.ot_lambda < 0.0 or args.ot_iters < 1 or args.ot_tol < 0.0:
        raise ValueError("OT eta/iters must be positive; lambda/tolerance non-negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label-smoothing must be in [0, 1)")
    if args.lr <= 0.0 or args.weight_decay < 0.0 or args.probe_lr <= 0.0:
        raise ValueError("learning rates must be positive and weight decay non-negative")
    if not 0.0 <= args.momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must be in [0, 1)")
    if args.probe_epochs < 1 and "linear" in args.benchmark:
        raise ValueError("probe-epochs must be positive when linear benchmarking is enabled")
    if args.resume and args.resume != "auto" and args.method == "both":
        raise ValueError("an explicit --resume path is ambiguous with --method both; use --resume auto")


def resolve_image_size(args: argparse.Namespace) -> int:
    if args.image_size:
        return args.image_size
    return 32 if args.dataset in {"cifar10", "cifar100"} else 224


def select_device(spec: str) -> torch.device:
    """Resolve --device to a CUDA device. CPU and MPS execution are rejected."""
    if spec == "auto":
        spec = "cuda"
    device = torch.device(spec)
    if device.type != "cuda":
        raise RuntimeError(
            f"this runner is CUDA-only; --device={spec!r} is not supported. "
            "Pass 'cuda' or 'cuda:N' and run on a machine with an NVIDIA GPU."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled torch "
            "build and start the container with `--gpus all`."
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device.index} was requested but only "
            f"{torch.cuda.device_count()} device(s) are visible"
        )
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def worker_seed_fn(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def classification_transforms(dataset_name: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if dataset_name in {"cifar10", "cifar100"}:
        train_ops: List[Any] = []
        eval_ops: List[Any] = []
        if image_size == 32:
            train_ops.append(transforms.RandomCrop(32, padding=4))
        else:
            train_ops.append(transforms.RandomResizedCrop(image_size))
            eval_ops.append(transforms.Resize((image_size, image_size)))
        train_ops.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        eval_ops.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(train_ops), transforms.Compose(eval_ops)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    resize_size = int(round(image_size * 256 / 224))
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_datasets(args: argparse.Namespace, image_size: int) -> DatasetBundle:
    train_transform, eval_transform = classification_transforms(args.dataset, image_size)
    root = args.data_root.expanduser().resolve()
    if args.dataset == "imagenet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "--dataset imagenet requires `pip install datasets`; install it in the active environment"
            ) from exc
        root.mkdir(parents=True, exist_ok=True)
        token_value = os.environ.get(args.hf_token_env)
        token: Any = token_value if token_value else True
        load_kwargs = {
            "path": args.hf_dataset_id,
            "cache_dir": str(root),
            "revision": args.hf_revision,
            "token": token,
        }
        try:
            train_split = load_dataset(split="train", **load_kwargs)
            val_split = load_dataset(split="validation", **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"could not load gated Hugging Face dataset {args.hf_dataset_id!r}. "
                "Accept its access terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k, "
                f"then run `hf auth login` or set {args.hf_token_env}. Original error: {exc}"
            ) from exc
        if args.require_validated_data:
            manifest_path = root / "imagenet_download_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"validated-data gate failed: missing {manifest_path}; run download_imagenet.py --check-samples all"
                )
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            failures = []
            if not manifest.get("validation_passed"):
                failures.append("manifest is not marked validation_passed")
            for split_name, split in (("train", train_split), ("validation", val_split)):
                result = manifest.get("splits", {}).get(split_name, {})
                if result.get("validation_status") != "passed":
                    failures.append(f"{split_name} status is not passed")
                if not result.get("fully_checked") or result.get("decoded_samples_checked") != len(split):
                    failures.append(f"{split_name} was not completely decoded")
                if result.get("fingerprint") != getattr(split, "_fingerprint", None):
                    failures.append(f"{split_name} fingerprint changed after validation")
            if failures:
                raise RuntimeError("validated-data gate failed: " + "; ".join(failures))
            print(f"validated-data gate passed: {manifest_path}", flush=True)

            # Known malformed EXIF metadata does not affect decoded RGB pixels.
            # It was counted by the complete scan, so avoid repeating the same
            # warning from every DataLoader worker during training.
            warnings.filterwarnings(
                "ignore",
                message=r"Corrupt EXIF data\..*",
                category=UserWarning,
                module=r"PIL\.TiffImagePlugin",
            )
        label_feature = train_split.features["label"]
        classes = list(getattr(label_feature, "names", []) or [])
        if not classes:
            num_classes = int(getattr(label_feature, "num_classes", 1000))
            classes = [str(index) for index in range(num_classes)]
        train = HuggingFaceImageDataset(train_split, train_transform)
        train_eval = HuggingFaceImageDataset(train_split, eval_transform)
        val = HuggingFaceImageDataset(val_split, eval_transform)
        class_to_idx = {name: index for index, name in enumerate(classes)}
        return DatasetBundle(train, train_eval, val, len(classes), class_to_idx)
    if args.dataset == "imagefolder":
        train_root, val_root = root / "train", root / "val"
        if not train_root.is_dir() or not val_root.is_dir():
            raise FileNotFoundError(f"expected ImageFolder directories {train_root} and {val_root}")
        train = datasets.ImageFolder(train_root, transform=train_transform)
        train_eval = datasets.ImageFolder(train_root, transform=eval_transform)
        val = datasets.ImageFolder(val_root, transform=eval_transform)
        if train.class_to_idx != val.class_to_idx:
            raise ValueError("train and val ImageFolders have different class_to_idx mappings")
        return DatasetBundle(train, train_eval, val, len(train.classes), dict(train.class_to_idx))
    if args.dataset == "cifar10":
        train = datasets.CIFAR10(root, train=True, transform=train_transform, download=args.download)
        train_eval = datasets.CIFAR10(root, train=True, transform=eval_transform, download=args.download)
        val = datasets.CIFAR10(root, train=False, transform=eval_transform, download=args.download)
        classes = list(train.classes)
    elif args.dataset == "cifar100":
        train = datasets.CIFAR100(root, train=True, transform=train_transform, download=args.download)
        train_eval = datasets.CIFAR100(root, train=True, transform=eval_transform, download=args.download)
        val = datasets.CIFAR100(root, train=False, transform=eval_transform, download=args.download)
        classes = list(train.classes)
    else:
        train = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, image_size, image_size),
            num_classes=args.fake_num_classes,
            transform=train_transform,
            random_offset=0,
        )
        train_eval = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, image_size, image_size),
            num_classes=args.fake_num_classes,
            transform=eval_transform,
            random_offset=0,
        )
        val = datasets.FakeData(
            size=args.fake_val_size,
            image_size=(3, image_size, image_size),
            num_classes=args.fake_num_classes,
            transform=eval_transform,
            random_offset=1_000_000,
        )
        classes = [str(index) for index in range(args.fake_num_classes)]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    return DatasetBundle(train, train_eval, val, len(classes), class_to_idx)


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + seed_offset)
    loader_options: Dict[str, Any] = {}
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed_fn,
        generator=generator,
        drop_last=False,
        **loader_options,
    )


class TinyConvEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def build_encoder(args: argparse.Namespace, image_size: int) -> Tuple[nn.Module, int]:
    if args.architecture == "tiny_cnn":
        return TinyConvEncoder(args.tiny_feature_dim), args.tiny_feature_dim
    builder = getattr(models, args.architecture)
    weights = None
    if args.pretrained:
        weights = models.get_model_weights(args.architecture).DEFAULT
    kwargs: Dict[str, Any] = {"weights": weights}
    if args.architecture == "vit_b_16" and not args.pretrained:
        kwargs["image_size"] = image_size
    encoder = builder(**kwargs)
    if args.architecture.startswith("resnet"):
        feature_dim = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
    elif args.architecture == "vit_b_16":
        if hasattr(encoder.heads, "head"):
            feature_dim = int(encoder.heads.head.in_features)
        else:
            feature_dim = int(encoder.hidden_dim)
        encoder.heads = nn.Identity()
    else:  # pragma: no cover - choices above make this defensive only
        raise ValueError(f"unsupported architecture: {args.architecture}")
    return encoder, feature_dim


def default_nested_dims(architecture: str, feature_dim: int) -> List[int]:
    if architecture == "vit_b_16":
        return [dim for dim in (12, 24, 48, 96, 192, 384, 768) if dim <= feature_dim]
    return [dim for dim in (8, 16, 32, 64, 128, 256, 512, 1024, 2048) if dim <= feature_dim]


def resolve_nested_dims(args: argparse.Namespace, feature_dim: int) -> List[int]:
    dims = list(args.nested_dims) if args.nested_dims is not None else default_nested_dims(args.architecture, feature_dim)
    if dims != sorted(set(dims)):
        raise ValueError("nested-dims must be strictly increasing and unique")
    if dims[0] <= 0 or dims[-1] > feature_dim:
        raise ValueError(f"nested-dims must be in [1, {feature_dim}], got {dims}")
    if dims[-1] != feature_dim:
        print(f"warning: largest nested dimension {dims[-1]} is smaller than encoder output {feature_dim}", file=sys.stderr)
    return dims


class IndependentHeads(nn.Module):
    def __init__(self, dims: Sequence[int], num_classes: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, num_classes) for dim in dims])

    def forward(self, features: torch.Tensor, dims: Sequence[int]) -> List[torch.Tensor]:
        return [head(features[:, :dim]) for head, dim in zip(self.heads, dims)]


class SharedHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.bias = nn.Parameter(torch.empty(num_classes))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(feature_dim)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: torch.Tensor, dims: Sequence[int]) -> List[torch.Tensor]:
        return [F.linear(features[:, :dim], self.weight[:, :dim], self.bias) for dim in dims]


class ScaleCostBuilder(nn.Module):
    """M3-inspired cost tensor builder supporting 2-marginal matrix and 3-marginal tensor costs."""

    def __init__(self, nested_dims: Sequence[int], feature_dim: int, num_marginals: int = 3, pairs: str = "all") -> None:
        super().__init__()
        self.nested_dims = tuple(nested_dims)
        self.feature_dim = feature_dim
        self.num_marginals = num_marginals
        if num_marginals == 2:
            if pairs == "adjacent":
                self.scale_pairs = tuple((i, i + 1) for i in range(len(nested_dims) - 1))
            else:
                self.scale_pairs = tuple(
                    (left, right)
                    for left in range(len(nested_dims))
                    for right in range(left + 1, len(nested_dims))
                )
            if not self.scale_pairs:
                raise ValueError("2-marginal MMPOT requires at least two nested dimensions")
        elif num_marginals == 3:
            if pairs == "adjacent":
                self.scale_triples = tuple((i, i + 1, i + 2) for i in range(len(nested_dims) - 2))
            else:
                self.scale_triples = tuple(
                    (a, b, c)
                    for a in range(len(nested_dims))
                    for b in range(a + 1, len(nested_dims))
                    for c in range(b + 1, len(nested_dims))
                )
            if not self.scale_triples:
                raise ValueError("3-marginal MMPOT requires at least three nested dimensions")
        else:
            raise ValueError(f"unsupported num_marginals: {num_marginals}; expected 2 or 3")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features32 = features.float()
        normalized: List[torch.Tensor] = []
        for dim in self.nested_dims:
            prefix = F.normalize(features32[:, :dim], p=2, dim=1, eps=1e-12)
            normalized.append(F.pad(prefix, (0, self.feature_dim - dim)))
        
        N = features32.shape[0]
        if self.num_marginals == 2:
            cost = features32.new_zeros((N, N))
            for left, right in self.scale_pairs:
                cost = cost + (1.0 - normalized[left] @ normalized[right].T)
            return (cost / len(self.scale_pairs)).clamp(min=0.0, max=2.0)
        else:
            cost = features32.new_zeros((N, N, N))
            for a, b, c in self.scale_triples:
                cost_ab = 1.0 - normalized[a] @ normalized[b].T
                cost_bc = 1.0 - normalized[b] @ normalized[c].T
                cost_ac = 1.0 - normalized[a] @ normalized[c].T
                cost_abc = cost_ab[:, :, None] + cost_bc[None, :, :] + cost_ac[:, None, :]
                cost = cost + cost_abc
            return (cost / len(self.scale_triples)).clamp(min=0.0, max=6.0)


PairwiseScaleCost = ScaleCostBuilder


class GreenkhornMMPOTSolver(nn.Module):
    """Stabilized 2-marginal and 3-marginal entropic partial-OT Greenkhorn solver."""

    def __init__(
        self,
        mass: float,
        eta: float,
        max_iters: int,
        tol: float,
        gradient_mode: str,
        solver_mode: str = "cyclic",
    ) -> None:
        super().__init__()
        self.mass = float(mass)
        self.eta = float(eta)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.gradient_mode = gradient_mode
        self.solver_mode = solver_mode
        self.eps = 1e-8
        self.scale_min = 1e-8
        self.scale_max = 1e8

    @staticmethod
    def _rho(target: torch.Tensor, value: torch.Tensor, eps: float) -> torch.Tensor:
        target_safe = target.clamp_min(eps)
        value_safe = value.clamp_min(eps)
        return (value - target + target_safe * torch.log(target_safe / value_safe)).sum()

    def _solve2d(self, cost: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n = cost.shape[0]
        marginal = torch.full((n,), 1.0 / n, dtype=cost.dtype, device=cost.device)
        kernel = torch.exp((-cost / self.eta).clamp(min=-60.0, max=0.0))
        v1 = torch.ones_like(marginal)
        v2 = torch.ones_like(marginal)
        weight = torch.ones((), dtype=cost.dtype, device=cost.device)
        target_mass = cost.new_tensor(self.mass)
        normalizer_mass = self.mass + 2.0 * (1.0 - self.mass)
        last_error = cost.new_tensor(float("inf"))
        iterations = 0

        for iteration in range(self.max_iters):
            if self.solver_mode == "cyclic":
                scaled = kernel * torch.outer(v1, v2)
                partition = (weight * scaled.sum() + v1.sum() + v2.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                r1 = factor * (weight * scaled.sum(dim=1) + v1)
                v1 = (v1 * marginal / r1.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

                scaled = kernel * torch.outer(v1, v2)
                partition = (weight * scaled.sum() + v1.sum() + v2.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                r2 = factor * (weight * scaled.sum(dim=0) + v2)
                v2 = (v2 * marginal / r2.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

                scaled = kernel * torch.outer(v1, v2)
                partition = (weight * scaled.sum() + v1.sum() + v2.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                current_mass = factor * weight * scaled.sum()
                weight = (weight * target_mass / current_mass.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

                errors = torch.stack(
                    [
                        self._rho(marginal, r1, self.eps),
                        self._rho(marginal, r2, self.eps),
                        self._rho(target_mass.reshape(1), current_mass.reshape(1), self.eps),
                    ]
                )
                last_error = errors.max()
                iterations = iteration + 1
                if self.tol and float(last_error.detach()) <= self.tol:
                    break
            else:
                scaled = kernel * torch.outer(v1, v2)
                partition = (weight * scaled.sum() + v1.sum() + v2.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                r1 = factor * (weight * scaled.sum(dim=1) + v1)
                r2 = factor * (weight * scaled.sum(dim=0) + v2)
                current_mass = factor * weight * scaled.sum()
                errors = torch.stack(
                    [
                        self._rho(marginal, r1, self.eps),
                        self._rho(marginal, r2, self.eps),
                        self._rho(target_mass.reshape(1), current_mass.reshape(1), self.eps),
                    ]
                )
                last_error = errors.max()
                iterations = iteration + 1
                if self.tol and float(last_error.detach()) <= self.tol:
                    break
                worst = int(errors.detach().argmax())
                if worst == 0:
                    v1 = (v1 * marginal / r1.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)
                elif worst == 1:
                    v2 = (v2 * marginal / r2.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)
                else:
                    weight = (weight * target_mass / current_mass.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

        scaled = kernel * torch.outer(v1, v2)
        partition = (weight * scaled.sum() + v1.sum() + v2.sum()).clamp_min(self.eps)
        plan = (normalizer_mass / partition) * weight * scaled
        row_mass, col_mass = plan.sum(dim=1), plan.sum(dim=0)
        cap_violation = torch.maximum(
            (row_mass - marginal).clamp_min(0).max(),
            (col_mass - marginal).clamp_min(0).max(),
        )
        diagnostics = {
            "mass": plan.sum(),
            "cap_violation": cap_violation,
            "constraint_error": last_error,
            "iterations": cost.new_tensor(float(iterations)),
        }
        return plan, diagnostics

    def _solve3d(self, cost: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n = cost.shape[0]
        marginal = torch.full((n,), 1.0 / n, dtype=cost.dtype, device=cost.device)
        kernel = torch.exp((-cost / self.eta).clamp(min=-60.0, max=0.0))
        v1 = torch.ones_like(marginal)
        v2 = torch.ones_like(marginal)
        v3 = torch.ones_like(marginal)
        weight = torch.ones((), dtype=cost.dtype, device=cost.device)
        target_mass = cost.new_tensor(self.mass)
        normalizer_mass = self.mass + 3.0 * (1.0 - self.mass)
        last_error = cost.new_tensor(float("inf"))
        iterations = 0

        for iteration in range(self.max_iters):
            if self.solver_mode == "cyclic":
                scaled = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
                partition = (weight * scaled.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                r1 = factor * (weight * scaled.sum(dim=(1, 2)) + v1)
                v1 = (v1 * marginal / r1.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

                scaled = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
                partition = (weight * scaled.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                r2 = factor * (weight * scaled.sum(dim=(0, 2)) + v2)
                v2 = (v2 * marginal / r2.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

                scaled = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
                partition = (weight * scaled.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                r3 = factor * (weight * scaled.sum(dim=(0, 1)) + v3)
                v3 = (v3 * marginal / r3.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

                scaled = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
                partition = (weight * scaled.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                current_mass = factor * weight * scaled.sum()
                weight = (weight * target_mass / current_mass.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

                errors = torch.stack(
                    [
                        self._rho(marginal, r1, self.eps),
                        self._rho(marginal, r2, self.eps),
                        self._rho(marginal, r3, self.eps),
                        self._rho(target_mass.reshape(1), current_mass.reshape(1), self.eps),
                    ]
                )
                last_error = errors.max()
                iterations = iteration + 1
                if self.tol and float(last_error.detach()) <= self.tol:
                    break
            else:
                scaled = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
                partition = (weight * scaled.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(self.eps)
                factor = normalizer_mass / partition
                r1 = factor * (weight * scaled.sum(dim=(1, 2)) + v1)
                r2 = factor * (weight * scaled.sum(dim=(0, 2)) + v2)
                r3 = factor * (weight * scaled.sum(dim=(0, 1)) + v3)
                current_mass = factor * weight * scaled.sum()
                errors = torch.stack(
                    [
                        self._rho(marginal, r1, self.eps),
                        self._rho(marginal, r2, self.eps),
                        self._rho(marginal, r3, self.eps),
                        self._rho(target_mass.reshape(1), current_mass.reshape(1), self.eps),
                    ]
                )
                last_error = errors.max()
                iterations = iteration + 1
                if self.tol and float(last_error.detach()) <= self.tol:
                    break
                worst = int(errors.detach().argmax())
                if worst == 0:
                    v1 = (v1 * marginal / r1.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)
                elif worst == 1:
                    v2 = (v2 * marginal / r2.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)
                elif worst == 2:
                    v3 = (v3 * marginal / r3.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)
                else:
                    weight = (weight * target_mass / current_mass.clamp_min(self.eps)).clamp(self.scale_min, self.scale_max)

        scaled = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
        partition = (weight * scaled.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(self.eps)
        plan = (normalizer_mass / partition) * weight * scaled
        m1_mass = plan.sum(dim=(1, 2))
        m2_mass = plan.sum(dim=(0, 2))
        m3_mass = plan.sum(dim=(0, 1))
        cap_violation = torch.maximum(
            torch.maximum(
                (m1_mass - marginal).clamp_min(0).max(),
                (m2_mass - marginal).clamp_min(0).max(),
            ),
            (m3_mass - marginal).clamp_min(0).max(),
        )
        diagnostics = {
            "mass": plan.sum(),
            "cap_violation": cap_violation,
            "constraint_error": last_error,
            "iterations": cost.new_tensor(float(iterations)),
        }
        return plan, diagnostics

    def _solve(self, cost: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if cost.ndim == 2:
            return self._solve2d(cost)
        elif cost.ndim == 3:
            return self._solve3d(cost)
        else:
            raise ValueError(f"unsupported cost dimension: {cost.ndim}; expected 2 or 3")

    def forward(self, cost: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cost32 = cost.float()
        if self.gradient_mode == "envelope":
            with torch.no_grad():
                plan, diagnostics = self._solve(cost32.detach())
            loss = (cost32 * plan).sum()
        else:
            plan, diagnostics = self._solve(cost32)
            loss = (cost32 * plan).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite partial-OT loss; try increasing --ot-eta")
        return loss, diagnostics


GreenkhornPartialOT = GreenkhornMMPOTSolver


class MatryoshkaClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        nested_dims: Sequence[int],
        num_classes: int,
        method: str,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.nested_dims = tuple(nested_dims)
        self.num_classes = num_classes
        self.method = method
        self.loss_reduction = args.mrl_loss_reduction
        self.label_smoothing = args.label_smoothing
        self.ot_lambda = args.ot_lambda
        if args.head_style == "independent":
            self.classifier = IndependentHeads(nested_dims, num_classes)
        else:
            self.classifier = SharedHead(feature_dim, num_classes)
        if method == "mmpot":
            num_marginals = getattr(args, "ot_marginals", 3)
            self.cost_builder: Optional[ScaleCostBuilder] = ScaleCostBuilder(
                nested_dims, feature_dim, num_marginals, args.ot_scale_pairs
            )
            self.ot_solver: Optional[GreenkhornMMPOTSolver] = GreenkhornMMPOTSolver(
                args.ot_mass, args.ot_eta, args.ot_iters, args.ot_tol, args.ot_grad, getattr(args, "ot_solver_mode", "cyclic")
            )
        else:
            self.cost_builder = None
            self.ot_solver = None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(f"encoder returned {features.shape[1]} features, expected {self.feature_dim}")
        return features

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = self.encode(images)
        return features, self.classifier(features, self.nested_dims)

    def losses(
        self, features: torch.Tensor, logits: Sequence[torch.Tensor], labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        head_losses = torch.stack(
            [F.cross_entropy(head_logits, labels, label_smoothing=self.label_smoothing) for head_logits in logits]
        )
        mrl_loss = head_losses.mean() if self.loss_reduction == "mean" else head_losses.sum()
        zero = mrl_loss.new_zeros(())
        diagnostics = {"mass": zero, "cap_violation": zero, "constraint_error": zero, "iterations": zero}
        if self.method == "mmpot":
            assert self.cost_builder is not None and self.ot_solver is not None
            with torch.autocast(device_type=features.device.type, enabled=False):
                cost = self.cost_builder(features)
                ot_loss, diagnostics = self.ot_solver(cost)
            total = mrl_loss + self.ot_lambda * ot_loss
        else:
            ot_loss = zero
            total = mrl_loss
        return total, mrl_loss, ot_loss, diagnostics


def build_model(
    args: argparse.Namespace,
    image_size: int,
    num_classes: int,
    method: str,
) -> Tuple[MatryoshkaClassifier, int, List[int]]:
    encoder, feature_dim = build_encoder(args, image_size)
    dims = resolve_nested_dims(args, feature_dim)
    req_dims = getattr(args, "ot_marginals", 3) if method == "mmpot" else 1
    if method == "mmpot" and len(dims) < req_dims:
        raise ValueError(f"mmpot with {req_dims} marginals requires at least {req_dims} nested dimensions")
    return MatryoshkaClassifier(encoder, feature_dim, dims, num_classes, method, args), feature_dim, dims


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> Optional[Any]:
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step_milestones, gamma=args.step_gamma)
    return None


def build_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_context(device: torch.device, enabled: bool) -> Any:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def progress_loader(
    loader: DataLoader,
    description: str,
    *,
    enabled: bool,
    limit: int = 0,
) -> Any:
    total = min(len(loader), limit) if limit > 0 else len(loader)
    iterable = itertools.islice(loader, total) if limit > 0 else loader
    return tqdm(
        iterable,
        total=total,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        disable=not enabled,
    )


def train_one_epoch(
    model: MatryoshkaClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> EpochStats:
    model.train()
    sums = {"loss": 0.0, "mrl": 0.0, "ot": 0.0, "mass": 0.0, "cap": 0.0}
    samples = 0
    amp_enabled = bool(args.amp and device.type == "cuda")
    start = time.time()
    progress = progress_loader(
        loader,
        f"train epoch {epoch + 1:03d}",
        enabled=args.progress,
        limit=args.max_train_batches,
    )
    for batch_index, (images, labels) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp_enabled):
            features, logits = model(images)
            loss, mrl_loss, ot_loss, diagnostics = model.losses(features, logits, labels)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        count = labels.shape[0]
        samples += count
        sums["loss"] += float(loss.detach()) * count
        sums["mrl"] += float(mrl_loss.detach()) * count
        sums["ot"] += float(ot_loss.detach()) * count
        sums["mass"] += float(diagnostics["mass"].detach()) * count
        sums["cap"] += float(diagnostics["cap_violation"].detach()) * count
        if args.print_freq > 0 and (batch_index % args.print_freq == 0):
            print(
                f"epoch={epoch + 1:03d} batch={batch_index:05d} "
                f"loss={float(loss.detach()):.4f} mrl={float(mrl_loss.detach()):.4f} "
                f"ot={float(ot_loss.detach()):.4f} elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    if samples == 0:
        raise RuntimeError("training loader produced no samples")
    seconds = time.time() - start
    return EpochStats(
        sums["loss"] / samples,
        sums["mrl"] / samples,
        sums["ot"] / samples,
        sums["mass"] / samples,
        sums["cap"] / samples,
        samples,
        seconds,
        samples / max(seconds, 1e-12),
    )


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    predictions = logits.topk(k, dim=1).indices
    return int(predictions.eq(labels[:, None]).any(dim=1).sum())


@torch.inference_mode()
def evaluate_heads(
    model: MatryoshkaClassifier,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    description: str = "evaluate heads",
) -> Dict[str, Any]:
    model.eval()
    correct1 = [0] * len(model.nested_dims)
    correct5 = [0] * len(model.nested_dims)
    loss_sums = [0.0] * len(model.nested_dims)
    samples = 0
    amp_enabled = bool(args.amp and device.type == "cuda")
    progress = progress_loader(
        loader,
        description,
        enabled=args.progress,
        limit=args.max_val_batches,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_context(device, amp_enabled):
            _, logits = model(images)
        count = labels.shape[0]
        samples += count
        for index, head_logits in enumerate(logits):
            loss_sums[index] += float(F.cross_entropy(head_logits, labels)) * count
            correct1[index] += topk_correct(head_logits, labels, 1)
            correct5[index] += topk_correct(head_logits, labels, 5)
    if samples == 0:
        raise RuntimeError("validation loader produced no samples")
    per_dim = {
        str(dim): {
            "loss": loss_sums[index] / samples,
            "top1": 100.0 * correct1[index] / samples,
            "top5": 100.0 * correct5[index] / samples,
        }
        for index, dim in enumerate(model.nested_dims)
    }
    top1_values = [entry["top1"] for entry in per_dim.values()]
    return {
        "samples": samples,
        "per_dim": per_dim,
        "mean_top1": sum(top1_values) / len(top1_values),
        "full_top1": per_dim[str(model.nested_dims[-1])]["top1"],
    }


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def objective_name(method: str, args: argparse.Namespace) -> str:
    if method == "mrl":
        return "mrl_cross_entropy"
    if args.ot_marginals >= 3:
        return f"mrl_cross_entropy_plus_true_{args.ot_marginals}_marginal_partial_ot"
    return "mrl_cross_entropy_plus_pairwise_partial_ot_proxy"


def checkpoint_payload(
    model: MatryoshkaClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Any,
    epoch: int,
    best_metric: float,
    method: str,
    args: argparse.Namespace,
    class_to_idx: Mapping[str, int],
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "method": method,
        "objective": objective_name(method, args),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
        "feature_dim": model.feature_dim,
        "nested_dims": list(model.nested_dims),
        "num_classes": model.num_classes,
        "class_to_idx": dict(class_to_idx),
        "rng_state": rng_state(),
    }


def resolve_resume_path(args: argparse.Namespace, method_dir: Path) -> Optional[Path]:
    if not args.resume:
        return None
    candidate = method_dir / "last.pt" if args.resume == "auto" else Path(args.resume).expanduser().resolve()
    if not candidate.is_file():
        if args.resume == "auto":
            print(f"resume=auto: no checkpoint at {candidate}; starting from scratch")
            return None
        raise FileNotFoundError(candidate)
    return candidate


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Upsert an epoch record so checkpoint resumes cannot duplicate history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_epoch: Dict[int, Dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing = json.loads(line)
                    by_epoch[int(existing["epoch"])] = existing
    by_epoch[int(record["epoch"])] = dict(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for epoch in sorted(by_epoch):
            handle.write(json.dumps(by_epoch[epoch], sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def extract_features(
    model: MatryoshkaClassifier,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    description: str,
    progress_enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    seen = 0
    max_batches = math.ceil(max_samples / loader.batch_size) if max_samples else 0
    progress = progress_loader(
        loader,
        description,
        enabled=progress_enabled,
        limit=max_batches,
    )
    for images, labels in progress:
        if max_samples and seen >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        features = model.encode(images).float()
        labels = labels.to(device, non_blocking=True)
        if max_samples:
            remaining = max_samples - seen
            features, labels = features[:remaining], labels[:remaining]
        all_features.append(features)
        all_labels.append(labels)
        seen += labels.shape[0]
    if not all_features:
        raise RuntimeError("feature loader produced no samples")
    return torch.cat(all_features), torch.cat(all_labels)


@torch.inference_mode()
def exact_l2_knn(
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    dims: Sequence[int],
    query_chunk: int,
    gallery_chunk: int,
    normalize: bool,
    progress_enabled: bool,
) -> Dict[str, Any]:
    if query_chunk < 1 or gallery_chunk < 1:
        raise ValueError("kNN chunk sizes must be positive")
    results: Dict[str, Any] = {}
    for dim in dims:
        gallery_dim = gallery[:, :dim]
        query_dim = queries[:, :dim]
        if normalize:
            gallery_dim = F.normalize(gallery_dim, dim=1)
            query_dim = F.normalize(query_dim, dim=1)
        correct = 0
        query_starts = range(0, query_dim.shape[0], query_chunk)
        query_progress = tqdm(
            query_starts,
            total=len(query_starts),
            desc=f"1-NN dim {dim}",
            unit="chunk",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not progress_enabled,
        )
        for query_start in query_progress:
            query_batch = query_dim[query_start : query_start + query_chunk]
            best_distance = torch.full(
                (query_batch.shape[0],), float("inf"), device=query_batch.device
            )
            best_label = torch.empty(
                (query_batch.shape[0],),
                dtype=gallery_labels.dtype,
                device=gallery_labels.device,
            )
            query_norm = (query_batch * query_batch).sum(dim=1)
            for gallery_start in range(0, gallery_dim.shape[0], gallery_chunk):
                gallery_batch = gallery_dim[gallery_start : gallery_start + gallery_chunk]
                distances = (
                    query_norm[:, None]
                    + (gallery_batch * gallery_batch).sum(dim=1)[None, :]
                    - 2.0 * query_batch @ gallery_batch.T
                ).clamp_min_(0.0)
                chunk_distance, chunk_index = distances.min(dim=1)
                update = chunk_distance < best_distance
                best_distance[update] = chunk_distance[update]
                best_label[update] = gallery_labels[gallery_start + chunk_index[update]]
            targets = query_labels[query_start : query_start + query_batch.shape[0]]
            correct += int(best_label.eq(targets).sum())
        results[str(dim)] = {"top1": 100.0 * correct / query_labels.shape[0]}
        print(f"1-NN dim={dim:4d} top1={results[str(dim)]['top1']:.3f}", flush=True)
    return {
        "protocol": "exact_l2_train_gallery_val_query",
        "normalized": normalize,
        "gallery_samples": gallery.shape[0],
        "query_samples": queries.shape[0],
        "per_dim": results,
    }


class LinearProbeBank(nn.Module):
    def __init__(self, dims: Sequence[int], num_classes: int) -> None:
        super().__init__()
        self.dims = tuple(dims)
        self.heads = nn.ModuleList([nn.Linear(dim, num_classes) for dim in dims])

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        return [head(features[:, :dim]) for head, dim in zip(self.heads, self.dims)]


def run_linear_probe(
    model: MatryoshkaClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # Freeze encoder for linear evaluation
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    seed_everything(args.seed + 10_000, args.deterministic)
    probes = LinearProbeBank(model.nested_dims, model.num_classes).to(device)
    optimizer = torch.optim.SGD(
        probes.parameters(), lr=args.probe_lr, momentum=0.9, weight_decay=args.probe_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)
    probe_history: List[Dict[str, Any]] = []
    try:
        for epoch in range(args.probe_epochs):
            epoch_lr = optimizer.param_groups[0]["lr"]
            probes.train()
            total_loss, samples = 0.0, 0
            progress = progress_loader(
                train_loader,
                f"linear probe epoch {epoch + 1:03d}/{args.probe_epochs:03d}",
                enabled=args.progress,
                limit=args.probe_max_train_batches,
            )
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.no_grad():
                    features = model.encode(images).detach().float()
                logits = probes(features)
                loss = torch.stack([F.cross_entropy(item, labels) for item in logits]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * labels.shape[0]
                samples += labels.shape[0]
                progress.set_postfix(loss=f"{total_loss / samples:.4f}", refresh=False)
            if samples == 0:
                raise RuntimeError("linear-probe training loader produced no samples")
            probe_history.append(
                {"epoch": epoch + 1, "loss": total_loss / samples, "learning_rate": epoch_lr}
            )
            scheduler.step()
            print(f"linear-probe epoch={epoch + 1:03d} loss={total_loss / samples:.4f}", flush=True)

        probes.eval()
        correct1 = [0] * len(model.nested_dims)
        correct5 = [0] * len(model.nested_dims)
        val_samples = 0
        with torch.inference_mode():
            progress = progress_loader(
                val_loader,
                "linear probe validation",
                enabled=args.progress,
                limit=args.max_val_batches,
            )
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                features = model.encode(images).float()
                logits = probes(features)
                val_samples += labels.shape[0]
                for index, item in enumerate(logits):
                    correct1[index] += topk_correct(item, labels, 1)
                    correct5[index] += topk_correct(item, labels, 5)
        if val_samples == 0:
            raise RuntimeError("linear-probe validation loader produced no samples")
        per_dim = {
            str(dim): {
                "top1": 100.0 * correct1[idx] / val_samples,
                "top5": 100.0 * correct5[idx] / val_samples,
            }
            for idx, dim in enumerate(model.nested_dims)
        }
    finally:
        # Always restore encoder to trainable + train mode regardless of errors
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
        model.encoder.train()
    return {
        "protocol": "fresh_linear_heads_frozen_encoder_train_to_val",
        "preset_is_paper_exact": False,
        "epochs": args.probe_epochs,
        "lr": args.probe_lr,
        "weight_decay": args.probe_weight_decay,
        "samples": val_samples,
        "training_history": probe_history,
        "per_dim": per_dim,
    }


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def train_and_benchmark_method(
    method: str,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    image_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    method_dir = args.output_dir.expanduser().resolve() / method
    method_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, args.deterministic)
    model, feature_dim, dims = build_model(args, image_size, bundle.num_classes, method)
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = build_scaler(bool(args.amp and device.type == "cuda"))
    train_loader = make_loader(bundle.train, args, shuffle=True, seed_offset=0)
    val_loader = make_loader(bundle.val, args, shuffle=False, seed_offset=1)
    start_epoch, best_metric = 0, -float("inf")

    resume_path = resolve_resume_path(args, method_dir)
    history_path = method_dir / "history.jsonl"
    if resume_path is None:
        history_path.unlink(missing_ok=True)
    if resume_path is not None:
        checkpoint = safe_torch_load(resume_path, device)
        if checkpoint.get("method") != method:
            raise ValueError(f"checkpoint method {checkpoint.get('method')} does not match {method}")
        if checkpoint.get("nested_dims") != dims or checkpoint.get("num_classes") != bundle.num_classes:
            raise ValueError("checkpoint dimensions/classes do not match the current experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        restore_rng_state(checkpoint.get("rng_state", {}))
        print(f"resumed {method} from {resume_path} at epoch {start_epoch}")

    print(
        f"\n=== {method.upper()} | device={device} architecture={args.architecture} "
        f"feature_dim={feature_dim} dims={dims} classes={bundle.num_classes} ===",
        flush=True,
    )
    final_head: Optional[Dict[str, Any]] = None
    for epoch in range(start_epoch, args.epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        head_stats = evaluate_heads(model, val_loader, device, args)
        final_head = head_stats
        if scheduler is not None:
            scheduler.step()
        selected = float(head_stats[args.selection_metric])
        improved = selected > best_metric
        best_metric = max(best_metric, selected)
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, best_metric, method, args, bundle.class_to_idx
        )
        atomic_torch_save(payload, method_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, method_dir / "best.pt")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            atomic_torch_save(payload, method_dir / f"epoch_{epoch + 1:04d}.pt")
        record = {
            "epoch": epoch + 1,
            "lr": epoch_lr,
            "train": asdict(train_stats),
            "head": head_stats,
            "selection_metric": args.selection_metric,
            "selected_value": selected,
            "best_metric": best_metric,
        }
        append_jsonl(history_path, record)
        print(
            f"epoch={epoch + 1:03d} train_loss={train_stats.loss:.4f} "
            f"mean_top1={head_stats['mean_top1']:.3f} full_top1={head_stats['full_top1']:.3f} "
            f"best={best_metric:.3f}",
            flush=True,
        )

    selected_checkpoint = method_dir / f"{args.benchmark_checkpoint}.pt"
    if selected_checkpoint.is_file():
        loaded = load_model_weights(model, selected_checkpoint, device)
        best_metric = float(loaded.get("best_metric", best_metric))
        print(f"benchmarking {method} from {selected_checkpoint}")
    elif args.epochs == 0 and resume_path is None:
        print("warning: benchmarking randomly initialized weights because epochs=0 and no checkpoint was loaded")

    results: Dict[str, Any] = {
        "method": method,
        "objective": objective_name(method, args),
        "feature_dim": feature_dim,
        "nested_dims": dims,
        "best_metric": best_metric,
        "selection_metric": args.selection_metric,
        "checkpoint": str(selected_checkpoint) if selected_checkpoint.is_file() else None,
    }
    if "head" in args.benchmark:
        results["head"] = evaluate_heads(model, val_loader, device, args, "head benchmark")
    elif final_head is not None:
        results["last_epoch_head"] = final_head
    if "knn" in args.benchmark:
        gallery_loader = make_loader(bundle.train_eval, args, shuffle=False, seed_offset=2)
        query_loader = make_loader(bundle.val, args, shuffle=False, seed_offset=3)
        try:
            gallery, gallery_labels = extract_features(
                model,
                gallery_loader,
                device,
                args.knn_max_train,
                "kNN gallery features",
                args.progress,
            )
            queries, query_labels = extract_features(
                model,
                query_loader,
                device,
                args.knn_max_val,
                "kNN query features",
                args.progress,
            )
            results["knn"] = exact_l2_knn(
                gallery,
                gallery_labels,
                queries,
                query_labels,
                dims,
                args.knn_query_chunk,
                args.knn_gallery_chunk,
                args.knn_normalize,
                args.progress,
            )
        finally:
            # Explicitly delete loaders to release persistent workers (important on Windows)
            del gallery_loader, query_loader
        del gallery, gallery_labels, queries, query_labels
    if "linear" in args.benchmark:
        # Use eval-transform (no augmentation) loader for reproducible linear probing
        probe_train_loader = make_loader(bundle.train_eval, args, shuffle=True, seed_offset=4)
        probe_val_loader = make_loader(bundle.val, args, shuffle=False, seed_offset=5)
        try:
            results["linear"] = run_linear_probe(
                model, probe_train_loader, probe_val_loader, device, args
            )
        finally:
            del probe_train_loader, probe_val_loader
    atomic_json_dump(results, method_dir / "results.json")
    return results


def comparison_rows(results: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not all(method in results for method in METHODS):
        return []
    baseline, novel = results["mrl"], results["mmpot"]
    rows: List[Dict[str, Any]] = []
    dims = baseline["nested_dims"]
    for benchmark in ("head", "knn", "linear"):
        if benchmark not in baseline or benchmark not in novel:
            continue
        base_per_dim = baseline[benchmark].get("per_dim", {})
        novel_per_dim = novel[benchmark].get("per_dim", {})
        for dim in dims:
            dim_key = str(dim)
            if dim_key not in base_per_dim or dim_key not in novel_per_dim:
                continue
            base_metrics = base_per_dim[dim_key]
            novel_metrics = novel_per_dim[dim_key]
            for metric in sorted(set(base_metrics) & set(novel_metrics)):
                if isinstance(base_metrics[metric], (int, float)):
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "dimension": dim,
                            "metric": metric,
                            "mrl": base_metrics[metric],
                            "mmpot": novel_metrics[metric],
                            "delta_mmpot_minus_mrl": novel_metrics[metric] - base_metrics[metric],
                        }
                    )
    return rows


def write_comparison_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ["benchmark", "dimension", "metric", "mrl", "mmpot", "delta_mmpot_minus_mrl"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def json_safe_args(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    image_size = resolve_image_size(args)
    device = select_device(args.device)
    seed_everything(args.seed, args.deterministic)
    bundle = build_datasets(args, image_size)
    if args.prepare_data_only:
        print(
            f"Prepared dataset={args.dataset} at cache/root={args.data_root.expanduser().resolve()} "
            f"train={len(bundle.train)} val={len(bundle.val)} classes={bundle.num_classes}",
            flush=True,
        )
        return 0
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(METHODS) if args.method == "both" else [args.method]
    print(
        f"dataset={args.dataset} train={len(bundle.train)} val={len(bundle.val)} "
        f"classes={bundle.num_classes} image_size={image_size} methods={methods}",
        flush=True,
    )
    method_results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        method_results[method] = train_and_benchmark_method(
            method, args, bundle, image_size, device
        )
    rows = comparison_rows(method_results)
    write_comparison_csv(rows, args.output_dir / "comparison.csv")
    # Report what was actually solved: this runner does a true multi-marginal
    # tensor solve at --ot-marginals 3 and only degrades to the plan's pairwise
    # proxy at 2, so the old hard-coded "proxy" label would misreport a 3-marginal run.
    fidelity = (
        f"true_rank_{args.ot_marginals}_multi_marginal_partial_OT_tensor"
        if args.ot_marginals >= 3
        else "pairwise_BxB_partial_OT_proxy_from_Matryoshka_plan_not_full_rank_m_tensor"
    )
    summary = {
        "experiment": "Matryoshka-MMPOT_vs_MRL",
        "mmpot_fidelity": fidelity,
        "ot_marginals": args.ot_marginals,
        "dataset": {
            "name": args.dataset,
            "train_samples": len(bundle.train),
            "val_samples": len(bundle.val),
            "num_classes": bundle.num_classes,
            "image_size": image_size,
        },
        "config": json_safe_args(args),
        "methods": method_results,
        "comparison": rows,
    }
    summary["reporting"] = generate_matryoshka_report(
        args.output_dir,
        summary,
        filename_prefix="true_mmpot",
    )
    atomic_json_dump(summary, args.output_dir / "summary.json")
    print(f"\nCompleted. Summary: {args.output_dir / 'summary.json'}", flush=True)
    if rows:
        print(f"Comparison table: {args.output_dir / 'comparison.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
