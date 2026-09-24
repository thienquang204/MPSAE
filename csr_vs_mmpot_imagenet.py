#!/usr/bin/env python3
"""ImageNet ablation: Matryoshka and CSR/MPSAE v1/v2 representations.

The controlled study runs all five arms with ResNet-18 and ResNet-50.
Every arm starts from the matching ImageNet-1K V1 pretrained backbone recipe.

Pipeline
--------
1. Download/cache torchvision's pretrained backbone weights locally.
2. Freeze the backbone and cache deterministic ImageNet train/validation features.
3. Fine-tune the backbone end-to-end with Matryoshka Representation Learning
   (MRL), using a classifier at every requested feature-prefix dimension and
   at the full representation dimension.
4. Train fixed-K CSR v1 and MPSAE v1 sparse autoencoders on the same frozen
   features.
5. Train CSRv2 and MPSAEv2 with the corresponding objectives and progressive
   cosine Top-K curricula. Every objective coefficient is explicit and
   configurable while retaining paper-aligned defaults.
6. Encode the train split as the gallery and validation split as queries.
7. Unit-normalize every evaluated embedding and run exact L2 1-NN: dense
   Matryoshka prefixes use FAISS ``IndexFlatL2`` while all four sparse arms use
   exact chunked SciPy CSR products (equivalent to L2 ranking after normalization).
8. Record search-only retrieval latency and save histories, JSON results,
   publication tables, and figures. Model weights and optimizer checkpoints
   are deliberately not saved.

The proposed Multimarginal Presentation with Sparse Autoencoder v2 (MPSAEv2) arm
treats Top-K, Top-2K, and Top-4K codes of the same frozen image embedding as
three aligned views.  Its multiway cost is the circular-variance cost of Piran
et al. (2024), and its reference partial polymatching is ``s J``.  This is a
research extension: ``v2`` specifically denotes applying CSRv2's progressive
Top-K annealing to the existing MPSAE objective.

Expected ImageNet layout
------------------------
    /path/to/imagenet/
      train/n01440764/*.JPEG
      ...
      val/n01440764/*.JPEG
      ...

The complete two-backbone setting is launched by
``run_all_experiments_docker.sh``; this Python entry point executes one
backbone unit within that fixed study.

Dependencies: torch, torchvision, numpy, scipy, and FAISS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from wandb_tracking import (
    add_wandb_arguments,
    finish_wandb,
    init_wandb,
    log_wandb_metrics,
    update_wandb_summary,
)


MATRYOSHKA = "matryoshka"
CSR_V1 = "csr"
MP_SAE_V1 = "mpsae"
CSR = "csrv2"
MP_SAE = "mpsaev2"
SPARSE_METHODS = (CSR_V1, MP_SAE_V1, CSR, MP_SAE)
METHODS = (MATRYOSHKA, *SPARSE_METHODS)
DEFAULT_MPSAE_MMPOT_WEIGHT = 1.3
IMAGENET_CLASSES = 1000
DEFAULT_EVALUATION_BUDGETS = {
    "resnet18": (8, 16, 32, 64, 128, 256, 512),
    "resnet50": (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
}
DEFAULT_SPARSE_EXTRA_BUDGETS = (1, 2, 4)


@dataclass(frozen=True)
class BackboneSpec:
    """Everything needed to expose a torchvision classifier as a feature model."""

    name: str
    display_name: str
    output_dim: int
    constructor: Any
    weights: Any
    classifier_attribute: str

    @property
    def weights_id(self) -> str:
        return f"{self.weights.__class__.__name__}.{self.weights.name}"


BACKBONE_SPECS = {
    "resnet18": BackboneSpec(
        name="resnet18",
        display_name="ResNet-18",
        output_dim=512,
        constructor=models.resnet18,
        weights=models.ResNet18_Weights.IMAGENET1K_V1,
        classifier_attribute="fc",
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        display_name="ResNet-50",
        output_dim=2048,
        constructor=models.resnet50,
        # Match ResNet-18's ImageNet-1K V1 pretraining recipe so the default
        # architecture ablation does not confound backbone depth with a newer
        # weights recipe. The weights ID is also part of the cache signature.
        weights=models.ResNet50_Weights.IMAGENET1K_V1,
        classifier_attribute="fc",
    ),
}
BACKBONES = tuple(BACKBONE_SPECS)


def method_labels(backbone: BackboneSpec) -> Dict[str, str]:
    return {
        MATRYOSHKA: f"Matryoshka {backbone.display_name}",
        CSR_V1: f"CSR v1 ({backbone.display_name})",
        MP_SAE_V1: f"MPSAE v1 ({backbone.display_name})",
        CSR: f"CSRv2 ({backbone.display_name})",
        MP_SAE: f"MPSAEv2 ({backbone.display_name})",
    }


def matryoshka_training_dims(
    backbone: BackboneSpec, evaluation_budgets: Sequence[int]
) -> List[int]:
    """Paper-style power-of-two MRL heads plus every evaluated budget."""
    dimensions = set(evaluation_budgets)
    dimension = 8
    while dimension < backbone.output_dim:
        dimensions.add(dimension)
        dimension *= 2
    dimensions.add(backbone.output_dim)
    return sorted(dimensions)


def parse_ints(value: str) -> List[int]:
    try:
        result = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return sorted(set(result))


def cosine_annealed_k(
    step: int,
    total_steps: int,
    start_k: int,
    target_k: int,
    anneal_fraction: float,
) -> int:
    """Half-cosine Top-K curriculum followed by a fixed target-K phase."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    anneal_steps = max(1, math.ceil(total_steps * anneal_fraction))
    if anneal_steps == 1 or step >= anneal_steps - 1:
        return target_k
    progress = step / (anneal_steps - 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    value = target_k + (start_k - target_k) * cosine_decay
    return min(start_k, max(target_k, int(round(value))))


def annealing_metadata(
    args: argparse.Namespace, total_steps: int
) -> Dict[str, Any]:
    return {
        "schedule": "half_cosine_then_hold",
        "start_k": args.anneal_start_k,
        "target_k": args.train_k,
        "anneal_fraction": args.anneal_fraction,
        "total_steps": total_steps,
        "anneal_steps": max(1, math.ceil(total_steps * args.anneal_fraction)),
    }


def fixed_k_metadata(k: int, total_steps: int) -> Dict[str, Any]:
    return {
        "schedule": "fixed",
        "start_k": k,
        "target_k": k,
        "anneal_fraction": 0.0,
        "total_steps": total_steps,
        "anneal_steps": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ImageNet: matched Matryoshka and CSR/MPSAE v1/v2 representation ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("ImageNet and feature cache")
    data.add_argument("--data-root", type=Path, required=True, help="ImageFolder root or Hugging Face cache directory")
    data.add_argument("--data-backend", choices=("imagefolder", "hf"), default="imagefolder")
    data.add_argument(
        "--backbone", choices=BACKBONES, default="resnet18",
        help="torchvision ImageNet backbone used by all five experiment arms",
    )
    data.add_argument("--hf-dataset-id", default="ILSVRC/imagenet-1k")
    data.add_argument("--hf-revision", default="main")
    data.add_argument("--hf-token-env", default="HF_TOKEN")
    data.add_argument("--cache-dir", type=Path, default=Path("runs/three_method_ablation/cache"))
    data.add_argument("--weights-cache", type=Path, default=Path("weights"), help="local TORCH_HOME for pretrained weights")
    data.add_argument("--rebuild-cache", action="store_true")
    data.add_argument("--feature-batch-size", type=int, default=512)
    data.add_argument("--workers", type=int, default=8)
    data.add_argument("--prefetch-factor", type=int, default=2, help="batches prefetched by each data-loading worker")
    data.add_argument("--max-train", type=int, default=0, help="deterministic subset; 0 uses all training images")
    data.add_argument("--max-val", type=int, default=0, help="deterministic subset; 0 uses all validation images")

    model = p.add_argument_group("sparse representations")
    model.add_argument(
        "--hidden-dim", type=int, default=8196,
        help="CSR/MPSAE v1/v2 latent dimension (default: 8196)",
    )
    model.add_argument(
        "--topk", type=parse_ints, default=None,
        help="matched budgets; defaults to 8..512 (ResNet-18) or 8..2048 (ResNet-50)",
    )
    model.add_argument(
        "--sparse-extra-topk", type=parse_ints,
        default=list(DEFAULT_SPARSE_EXTRA_BUDGETS),
        help="additional lower budgets evaluated for every CSR/MPSAE sparse arm",
    )
    model.add_argument("--train-k", type=int, default=2)
    model.add_argument(
        "--v1-train-k", type=int, default=32,
        help="fixed training support for CSR v1 and MPSAE v1",
    )
    model.add_argument(
        "--anneal-start-k", type=int, default=64,
        help="initial broad Top-K support for CSRv2 and MPSAEv2 training",
    )
    model.add_argument(
        "--anneal-fraction", type=float, default=0.7,
        help="fraction of each sparse method's optimization steps used for cosine K annealing",
    )
    model.add_argument("--k-aux", type=int, default=512)
    model.add_argument("--dead-steps", type=int, default=1000)

    loss = p.add_argument_group("loss component weights")
    loss.add_argument(
        "--mrl-classification-weight", type=float, default=1.0,
        help="weight on Matryoshka's summed nested classification loss",
    )
    loss.add_argument(
        "--csrv2-main-recon-weight", "--csr-main-recon-weight",
        dest="csr_main_recon_weight", type=float, default=1.0,
        help="weight on CSRv2's annealed Top-K reconstruction loss",
    )
    loss.add_argument(
        "--csrv2-multi-topk-recon-weight", "--csr-multi-topk-recon-weight",
        dest="csr_multi_topk_recon_weight", type=float, default=1.0 / 8.0,
        help="weight on CSRv2's annealed Top-4K reconstruction loss",
    )
    loss.add_argument(
        "--csrv2-aux-recon-weight", "--csr-aux-recon-weight",
        dest="csr_aux_recon_weight", type=float, default=1.0 / 32.0,
        help="weight on CSRv2's dead-latent auxiliary reconstruction loss",
    )
    loss.add_argument(
        "--csrv2-contrastive-weight", "--csr-contrastive-weight",
        dest="csr_contrastive_weight", type=float, default=1.0,
        help="gamma multiplying CSRv2's non-negative contrastive loss",
    )
    loss.add_argument(
        "--mpsaev2-main-recon-weight", "--mpsae-main-recon-weight",
        dest="mpsae_main_recon_weight", type=float, default=1.0,
        help="weight on MPSAEv2's annealed Top-K reconstruction loss",
    )
    loss.add_argument(
        "--mpsaev2-nested-recon-weight", "--mpsae-nested-recon-weight",
        dest="mpsae_nested_recon_weight", type=float, default=1.0 / 8.0,
        help="weight on MPSAEv2's annealed mean Top-2K/Top-4K reconstruction loss",
    )
    loss.add_argument(
        "--mpsaev2-aux-recon-weight", "--mpsae-aux-recon-weight",
        dest="mpsae_aux_recon_weight", type=float, default=1.0 / 32.0,
        help="weight on MPSAEv2's dead-latent auxiliary reconstruction loss",
    )
    loss.add_argument(
        "--mpsaev2-mmpot-weight", "--mpsae-mmpot-weight",
        dest="mpsae_mmpot_weight", type=float, default=DEFAULT_MPSAE_MMPOT_WEIGHT,
        help="weight on MPSAEv2's multimarginal partial-transport gap",
    )

    train = p.add_argument_group("training")
    train.add_argument("--method", choices=(*METHODS, "all"), default="all")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument(
        "--mpsaev2-extra-epochs", "--mpsae-extra-epochs",
        dest="mpsae_extra_epochs", type=int, default=4,
        help="additional MPSAE v1/v2 epochs beyond the common epoch count; fixed to 4 for this study",
    )
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--mpsaev2-lr", "--lr", dest="lr", type=float, default=4e-5)
    train.add_argument("--csrv2-lr", "--csr-lr", dest="csr_lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--mrl-lr", type=float, default=1e-2, help="backbone MRL fine-tuning learning rate")
    train.add_argument("--mrl-momentum", type=float, default=0.9)
    train.add_argument("--ot-mass", type=float, default=0.8)
    train.add_argument("--ot-eta", type=float, default=0.2, help="M3G recommendation for circular variance")
    train.add_argument("--ot-iters", type=int, default=100)
    train.add_argument("--ot-tol", type=float, default=1e-4)
    train.add_argument("--ot-microbatch", type=int, default=32, help="B^3 cost makes small OT groups necessary")
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow fast TF32 CUDA matrix operations")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--print-freq", type=int, default=50)
    train.add_argument("--output-dir", type=Path, default=Path("runs/three_method_ablation"))

    knn = p.add_argument_group("exact dense/sparse 1-NN")
    knn.add_argument("--knn-batch-size", type=int, default=4096)
    knn.add_argument("--knn-query-batch", type=int, default=4096)
    knn.add_argument(
        "--sparse-knn-query-batch", type=int, default=32,
        help="query batch for exact chunked SciPy CSR sparse retrieval",
    )
    knn.add_argument(
        "--knn-normalize", action=argparse.BooleanOptionalAction, default=True,
        help="unit-normalize gallery and query embeddings (required by this study)",
    )
    knn.add_argument(
        "--faiss-gpu", action=argparse.BooleanOptionalAction, default=False,
        help="optional CUDA FAISS override; CPU is the fair default beside chunked SciPy CSR",
    )
    knn.add_argument("--faiss-gpu-device", type=int, default=None,
                     help="CUDA device for FAISS (defaults to the model CUDA device, otherwise 0)")
    knn.add_argument("--faiss-temp-memory-mib", type=int, default=512,
                     help="temporary GPU memory reserved by FAISS; 0 disables its allocation stack")
    add_wandb_arguments(p)
    return p


def validate_args(a: argparse.Namespace) -> None:
    if a.data_backend == "imagefolder":
        for split in ("train", "val"):
            if not (a.data_root / split).is_dir():
                raise FileNotFoundError(f"missing ImageNet directory: {a.data_root / split}")
    elif not a.data_root.is_dir():
        raise FileNotFoundError(f"missing Hugging Face cache directory: {a.data_root}")
    backbone = BACKBONE_SPECS[a.backbone]
    if a.topk is None:
        a.topk = list(DEFAULT_EVALUATION_BUDGETS[a.backbone])
    if a.method in (MATRYOSHKA, "all") and max(a.topk) > backbone.output_dim:
        raise ValueError(
            f"Matryoshka prefix dimensions cannot exceed the {backbone.display_name} "
            f"feature dimension ({backbone.output_dim})"
        )
    if a.hidden_dim < 0:
        raise ValueError("hidden-dim must be positive, or 0 to select 8196")
    if min(a.train_k, a.v1_train_k, a.anneal_start_k, a.dead_steps) < 1 or a.k_aux < 0:
        raise ValueError(
            "train-k/v1-train-k/anneal-start-k/dead-steps must be positive "
            "and k-aux must be non-negative"
        )
    if a.anneal_start_k < a.train_k:
        raise ValueError("anneal-start-k must be greater than or equal to train-k")
    if not 0.0 < a.anneal_fraction <= 1.0:
        raise ValueError("anneal-fraction must be in (0,1]")
    loss_weights = {
        "mrl-classification-weight": a.mrl_classification_weight,
        "csrv2-main-recon-weight": a.csr_main_recon_weight,
        "csrv2-multi-topk-recon-weight": a.csr_multi_topk_recon_weight,
        "csrv2-aux-recon-weight": a.csr_aux_recon_weight,
        "csrv2-contrastive-weight": a.csr_contrastive_weight,
        "mpsaev2-main-recon-weight": a.mpsae_main_recon_weight,
        "mpsaev2-nested-recon-weight": a.mpsae_nested_recon_weight,
        "mpsaev2-aux-recon-weight": a.mpsae_aux_recon_weight,
        "mpsaev2-mmpot-weight": a.mpsae_mmpot_weight,
    }
    invalid_loss_weights = [
        name for name, value in loss_weights.items()
        if not math.isfinite(value) or value < 0
    ]
    if invalid_loss_weights:
        raise ValueError(
            "loss weights must be finite and non-negative: "
            + ", ".join(invalid_loss_weights)
        )
    if a.mrl_classification_weight == 0:
        raise ValueError("mrl-classification-weight must be positive")
    if (
        a.csr_main_recon_weight
        + a.csr_multi_topk_recon_weight
        + a.csr_aux_recon_weight
        + a.csr_contrastive_weight
        == 0
    ):
        raise ValueError("at least one CSR-family loss component weight must be positive")
    if (
        a.mpsae_main_recon_weight
        + a.mpsae_nested_recon_weight
        + a.mpsae_aux_recon_weight
        + a.mpsae_mmpot_weight
        == 0
    ):
        raise ValueError("at least one MPSAE-family loss component weight must be positive")
    if a.hidden_dim == 0:
        a.hidden_dim = 8196
    if a.method in (*SPARSE_METHODS, "all") and a.hidden_dim < max(
        max((*a.topk, *a.sparse_extra_topk)),
        4 * a.anneal_start_k,
        4 * a.v1_train_k,
    ):
        raise ValueError(
            "hidden-dim must be >= every sparse evaluation budget, "
            ">= 4*anneal-start-k, and >= 4*v1-train-k"
        )
    if not 0.0 < a.ot_mass <= 1.0:
        raise ValueError("ot-mass must be in (0,1]")
    if a.ot_eta <= 0 or a.ot_iters < 1 or a.ot_tol <= 0:
        raise ValueError("ot-eta/ot-tol must be positive and ot-iters must be at least 1")
    if min(a.epochs, a.batch_size, a.feature_batch_size, a.ot_microbatch, a.prefetch_factor) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if a.mpsae_extra_epochs != 4:
        raise ValueError("mpsaev2-extra-epochs is fixed at 4 for both MPSAE arms")
    if a.csr_lr <= 0:
        raise ValueError("CSR-family lr must be positive")
    if a.ot_microbatch < 2:
        raise ValueError("ot-microbatch must be at least 2")
    if a.faiss_gpu_device is not None and a.faiss_gpu_device < 0:
        raise ValueError("faiss-gpu-device must be non-negative")
    if a.faiss_temp_memory_mib < 0:
        raise ValueError("faiss-temp-memory-mib must be non-negative")
    if a.mrl_lr <= 0 or not 0 <= a.mrl_momentum < 1:
        raise ValueError("mrl-lr must be positive and mrl-momentum must be in [0,1)")
    if a.lr <= 0 or a.weight_decay < 0:
        raise ValueError("MPSAE-family lr must be positive and weight-decay must be non-negative")
    if a.knn_batch_size < 1 or a.knn_query_batch < 1 or a.sparse_knn_query_batch < 1:
        raise ValueError("1-NN batch sizes must be positive")
    if not a.knn_normalize:
        raise ValueError("knn-normalize must remain enabled for the controlled experiment")
    if a.max_train < 0 or a.max_val < 0:
        raise ValueError("max-train/max-val must be non-negative")
    if a.prefetch_factor < 1 or a.workers < 0 or a.print_freq < 0:
        raise ValueError("prefetch-factor must be positive; workers/print-freq must be non-negative")


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = args.tf32


def loader_options(workers: int, prefetch_factor: int) -> Dict[str, Any]:
    if workers <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }


def runtime_metadata(device: torch.device, started_at: str, elapsed_seconds: float) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata["cuda_device_name"] = torch.cuda.get_device_name(index)
        metadata["cuda_device_total_memory_bytes"] = torch.cuda.get_device_properties(index).total_memory
    return metadata


def atomic_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def build_feature_network(backbone: BackboneSpec, weights_cache: Path) -> nn.Module:
    """Load official weights and replace the architecture-specific classifier."""
    os.environ["TORCH_HOME"] = str(weights_cache.expanduser().resolve())
    network = backbone.constructor(weights=backbone.weights)
    setattr(network, backbone.classifier_attribute, nn.Identity())
    return network


class FrozenBackbone(nn.Module):
    def __init__(self, weights_cache: Path, backbone: BackboneSpec) -> None:
        super().__init__()
        network = build_feature_network(backbone, weights_cache)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self.network = network
        self.transform = backbone.weights.transforms()
        self.output_dim = backbone.output_dim
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class MatryoshkaBackbone(nn.Module):
    """A torchvision backbone trained with classifiers on nested prefixes."""

    def __init__(
        self, weights_cache: Path, backbone: BackboneSpec, nested_dims: Sequence[int]
    ) -> None:
        super().__init__()
        self.network = build_feature_network(backbone, weights_cache)
        self.output_dim = backbone.output_dim
        self.backbone = backbone
        self.nested_dims = tuple(sorted(set((*nested_dims, self.output_dim))))
        if self.nested_dims[0] <= 0 or self.nested_dims[-1] > self.output_dim:
            raise ValueError(f"Matryoshka dimensions must be in [1, {self.output_dim}]")
        self.heads = nn.ModuleDict({
            str(dimension): nn.Linear(dimension, IMAGENET_CLASSES)
            for dimension in self.nested_dims
        })

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def classification_losses(
        self, features: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        per_dimension = {
            str(dimension): F.cross_entropy(
                self.heads[str(dimension)](features[:, :dimension]), target
            )
            for dimension in self.nested_dims
        }
        # Standard MRL uses c_m=1 for every nested classifier.
        return torch.stack(tuple(per_dimension.values())).sum(), per_dimension


def matryoshka_transforms(backbone: BackboneSpec) -> Tuple[Any, Any]:
    evaluation_transform = backbone.weights.transforms()
    crop_size = evaluation_transform.crop_size
    image_size = crop_size[0] if isinstance(crop_size, (list, tuple)) else crop_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, interpolation=evaluation_transform.interpolation
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(evaluation_transform.mean, evaluation_transform.std),
    ])
    return train_transform, evaluation_transform


class HuggingFaceImages(Dataset):
    """Apply torchvision preprocessing to a cached Hugging Face split."""

    def __init__(self, split: Any, transform: Any) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.split[index]
        image = sample["image"]
        if image is None:
            raise RuntimeError(f"ImageNet sample {index} has no decoded image")
        return self.transform(image.convert("RGB")), int(sample["label"])


def deterministic_subset(dataset: Dataset, maximum: int, seed: int) -> Dataset:
    if maximum <= 0 or maximum >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def build_image_dataset(
    split: str,
    root: Path,
    transform: Any,
    data_backend: str,
    hf_dataset_id: str,
    hf_revision: str,
    hf_token_env: str,
) -> Tuple[Dataset, str]:
    """Build an ImageNet split without coupling it to either experiment arm."""
    if data_backend == "imagefolder":
        return datasets.ImageFolder(root / split, transform=transform), split
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face backend requires: pip install datasets") from exc
    source_split = "validation" if split == "val" else "train"
    token_value = os.environ.get(hf_token_env)
    hf_split = load_dataset(
        path=hf_dataset_id,
        split=source_split,
        cache_dir=str(root),
        revision=hf_revision,
        token=token_value if token_value else True,
    )
    return HuggingFaceImages(hf_split, transform), source_split


def cache_paths(cache_dir: Path, split: str) -> Tuple[Path, Path, Path]:
    return (
        cache_dir / f"{split}_features.f16.npy",
        cache_dir / f"{split}_labels.npy",
        cache_dir / f"{split}_meta.json",
    )


def compatible_feature_cache(
    feature_path: Path,
    label_path: Path,
    meta_path: Path,
    expected: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return metadata only when all files belong to the requested experiment."""
    if not all(path.is_file() for path in (feature_path, label_path, meta_path)):
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        features = np.load(feature_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        valid = (
            features.ndim == 2
            and features.shape[1] == expected["feature_dim"]
            and features.dtype == np.float16
            and labels.dtype == np.int64
            and len(features) == len(labels) == metadata.get("samples")
        )
        del features, labels
        return metadata if valid else None
    except (OSError, ValueError, EOFError, KeyError, json.JSONDecodeError):
        return None


@torch.inference_mode()
def cache_split(
    split: str,
    root: Path,
    backbone: FrozenBackbone,
    device: torch.device,
    cache_dir: Path,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    channels_last: bool,
    maximum: int,
    seed: int,
    rebuild: bool,
    data_backend: str,
    hf_dataset_id: str,
    hf_revision: str,
    hf_token_env: str,
) -> Dict[str, Any]:
    feature_path, label_path, meta_path = cache_paths(cache_dir, split)
    subset_seed = seed + (0 if split == "train" else 1)
    expected_metadata = {
        "cache_format": 2,
        "split": split,
        "feature_dim": backbone.output_dim,
        "dtype": "float16",
        "backbone": backbone.backbone.name,
        "weights": backbone.backbone.weights_id,
        "data_backend": data_backend,
        "data_root": str(root.expanduser().resolve()),
        "hf_dataset_id": hf_dataset_id if data_backend == "hf" else None,
        "hf_revision": hf_revision if data_backend == "hf" else None,
        "maximum": maximum,
        "subset_seed": subset_seed,
    }
    if not rebuild:
        cached = compatible_feature_cache(
            feature_path, label_path, meta_path, expected_metadata
        )
        if cached is not None:
            print(
                f"cache {split}: reusing {backbone.backbone.display_name} features "
                f"from {feature_path}",
                flush=True,
            )
            return cached
        if any(path.exists() for path in (feature_path, label_path, meta_path)):
            print(
                f"cache {split}: existing files are incompatible with "
                f"{backbone.backbone.display_name}; rebuilding",
                flush=True,
            )

    dataset, source_split = build_image_dataset(
        split, root, backbone.transform, data_backend, hf_dataset_id, hf_revision, hf_token_env
    )
    dataset = deterministic_subset(dataset, maximum, subset_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        **loader_options(workers, prefetch_factor),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_tmp = feature_path.with_suffix(feature_path.suffix + ".tmp")
    label_tmp = label_path.with_suffix(label_path.suffix + ".tmp")
    features = np.lib.format.open_memmap(
        feature_tmp, mode="w+", dtype=np.float16, shape=(len(dataset), backbone.output_dim)
    )
    labels = np.lib.format.open_memmap(label_tmp, mode="w+", dtype=np.int64, shape=(len(dataset),))
    backbone.eval()
    offset, started = 0, time.time()
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            batch_features = backbone(images).float()
        if batch_features.ndim != 2 or batch_features.shape[1] != backbone.output_dim:
            raise RuntimeError(
                f"{backbone.backbone.display_name} returned feature shape "
                f"{tuple(batch_features.shape)}; expected [batch, {backbone.output_dim}]"
            )
        count = images.shape[0]
        features[offset : offset + count] = batch_features.to(
            dtype=torch.float16
        ).cpu().numpy()
        labels[offset : offset + count] = target.numpy()
        offset += count
        if offset % (batch_size * 100) < count:
            print(f"cache {split}: {offset}/{len(dataset)}", flush=True)
    features.flush()
    labels.flush()
    del features, labels
    os.replace(feature_tmp, feature_path)
    os.replace(label_tmp, label_path)
    metadata = {
        **expected_metadata,
        "samples": len(dataset),
        "source_split": source_split,
        "seconds": time.time() - started,
    }
    atomic_json(metadata, meta_path)
    return metadata


class CachedFeatures(Dataset):
    def __init__(self, feature_path: Path, label_path: Path) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        if len(self.features) != len(self.labels):
            raise RuntimeError("feature and label caches have different lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Copy avoids PyTorch's warning about read-only numpy memmaps.
        return torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)), int(self.labels[index])

    def __getitems__(self, indices: Sequence[int]) -> List[Tuple[torch.Tensor, int]]:
        """Convert a requested batch in one NumPy operation instead of per row."""
        feature_batch = torch.from_numpy(
            np.array(self.features[list(indices)], dtype=np.float32, copy=True)
        )
        label_batch = np.asarray(self.labels[list(indices)], dtype=np.int64)
        return [
            (feature_batch[offset], int(label_batch[offset]))
            for offset in range(len(indices))
        ]


def cached_feature_batches(
    dataset: CachedFeatures, batch_size: int
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        features = torch.from_numpy(
            np.array(dataset.features[start:stop], dtype=np.float32, copy=True)
        )
        labels = torch.from_numpy(
            np.array(dataset.labels[start:stop], dtype=np.int64, copy=True)
        )
        yield features, labels


def estimate_feature_mean(
    features: np.ndarray, maximum: int = 100_000, chunk_size: int = 8192
) -> torch.Tensor:
    """Compute the SAE pre-bias without materializing a large float32 cache slice."""
    sample_count = min(len(features), maximum)
    if sample_count == 0:
        raise RuntimeError("cannot initialize MPSAEv2 from an empty feature cache")
    total = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, sample_count, chunk_size):
        batch = np.asarray(
            features[start : min(start + chunk_size, sample_count)], dtype=np.float32
        )
        total += batch.sum(axis=0, dtype=np.float64)
    return torch.from_numpy((total / sample_count).astype(np.float32))


class TopKSAE(nn.Module):
    """Tied Top-K sparse autoencoder with dead-latent auxiliary tracking."""

    def __init__(self, input_dim: int, hidden_dim: int = 8196, dead_steps: int = 1000) -> None:
        super().__init__()
        decoder = torch.empty(hidden_dim, input_dim)
        nn.init.kaiming_uniform_(decoder, a=math.sqrt(5))
        decoder = F.normalize(decoder, dim=1)
        self.decoder = nn.Parameter(decoder)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("inactive_steps", torch.zeros(hidden_dim, dtype=torch.long))
        self.dead_steps = int(dead_steps)

    @staticmethod
    def keep_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, pre.shape[1])
        values, indices = torch.topk(pre, k=k, dim=1, sorted=False)
        values = F.relu(values)
        result = torch.zeros_like(pre)
        return result.scatter(1, indices, values)

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.pre_bias) @ self.decoder.T + self.encoder_bias

    def encode(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self.keep_topk(self.preactivations(x), k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.decoder + self.pre_bias

    def update_activity(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            active = z.gt(0).any(dim=0)
            self.inactive_steps.add_(1)
            self.inactive_steps[active] = 0

    def reconstruction_losses(
        self, x: torch.Tensor, k: int, k_aux: int, main_weight: float,
        nested_weight: float, aux_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pre = self.preactivations(x)
        z1 = self.keep_topk(pre, k)
        z2 = self.keep_topk(pre, min(2 * k, self.decoder.shape[0]))
        z4 = self.keep_topk(pre, min(4 * k, self.decoder.shape[0]))
        recon1 = self.decode(z1)
        recon2 = self.decode(z2)
        recon4 = self.decode(z4)
        main = F.mse_loss(recon1, x)
        nested = 0.5 * (F.mse_loss(recon2, x) + F.mse_loss(recon4, x))

        dead = self.inactive_steps >= self.dead_steps
        if dead.any() and k_aux > 0:
            masked = pre.masked_fill(~dead[None, :], -torch.inf)
            aux_k = min(k_aux, int(dead.sum()))
            aux_z = self.keep_topk(masked, aux_k)
            residual = (x - recon1).detach()
            aux = F.mse_loss(aux_z @ self.decoder, residual)
        else:
            aux = main.new_zeros(())
        total = main_weight * main + nested_weight * nested + aux_weight * aux
        self.update_activity(z1)
        stats = {
            "recon": main,
            "nested_recon": nested,
            "aux": aux,
            "dead_fraction": dead.float().mean(),
        }
        return total, stats, (z1, z2, z4)

    def csr_losses(
        self, x: torch.Tensor, k: int, k_aux: int, main_weight: float,
        multi_weight: float, aux_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """CSR reconstruction terms evaluated at the current annealed K."""
        pre = self.preactivations(x)
        z = self.keep_topk(pre, k)
        z4 = self.keep_topk(pre, min(4 * k, self.decoder.shape[0]))
        reconstruction = self.decode(z)
        main = F.mse_loss(reconstruction, x)
        multi = F.mse_loss(self.decode(z4), x)

        dead = self.inactive_steps >= self.dead_steps
        if dead.any() and k_aux > 0:
            masked = pre.masked_fill(~dead[None, :], -torch.inf)
            aux_k = min(k_aux, int(dead.sum()))
            aux_z = self.keep_topk(masked, aux_k)
            residual = (x - reconstruction).detach()
            auxiliary = F.mse_loss(aux_z @ self.decoder, residual)
        else:
            auxiliary = main.new_zeros(())
        total = main_weight * main + multi_weight * multi + aux_weight * auxiliary
        self.update_activity(z)
        return total, {
            "recon": main,
            "multi_topk_recon": multi,
            "aux": auxiliary,
            "dead_fraction": dead.float().mean(),
        }, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.copy_(F.normalize(self.decoder, dim=1))


def nonnegative_contrastive_loss(representations: torch.Tensor) -> torch.Tensor:
    """CSRv2 NCL: identify each non-negative sparse code against batch negatives."""
    similarities = representations.float() @ representations.float().T
    targets = torch.arange(similarities.shape[0], device=similarities.device)
    return F.cross_entropy(similarities, targets)


def circular_variance_cost(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
    """Exact M3G circular variance for three unit-normalized representation views."""
    a = F.normalize(z1.float(), dim=1, eps=1e-8)
    b = F.normalize(z2.float(), dim=1, eps=1e-8)
    c = F.normalize(z3.float(), dim=1, eps=1e-8)
    d12 = 1.0 - a @ b.T
    d13 = 1.0 - a @ c.T
    d23 = 1.0 - b @ c.T
    return ((2.0 / 9.0) * (d12[:, :, None] + d13[:, None, :] + d23[None, :, :])).clamp(0.0, 1.0)


def kl_constraint(target: torch.Tensor, value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    t = target.clamp_min(eps)
    v = value.clamp_min(eps)
    return (v - target + t * torch.log(t / v)).sum()


@torch.no_grad()
def greenkhorn_mmpot(
    cost: torch.Tensor, mass: float, eta: float, max_iters: int, tol: float
) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, float]]:
    """Greedy 3-marginal partial OT solver from the slack-variable dual."""
    if cost.ndim != 3 or len(set(cost.shape)) != 1:
        raise ValueError("expected a cubic [B,B,B] cost tensor")
    n = cost.shape[0]
    p = torch.full((n,), 1.0 / n, device=cost.device, dtype=torch.float32)
    kernel = torch.exp((-cost.float() / eta).clamp(min=-60.0, max=0.0))
    scales = [torch.ones_like(p) for _ in range(3)]
    w = cost.new_ones((), dtype=torch.float32)
    target_mass = cost.new_tensor(mass, dtype=torch.float32)
    D = mass + 3.0 * (1.0 - mass)
    error = float("inf")

    for iteration in range(max_iters):
        v1, v2, v3 = scales
        B = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
        partition = (w * B.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(1e-12)
        factor = D / partition
        estimates = [
            factor * (w * B.sum(dim=(1, 2)) + v1),
            factor * (w * B.sum(dim=(0, 2)) + v2),
            factor * (w * B.sum(dim=(0, 1)) + v3),
        ]
        current_mass = factor * w * B.sum()
        errors = torch.stack(
            [kl_constraint(p, r) for r in estimates]
            + [kl_constraint(target_mass[None], current_mass[None])]
        )
        error = float(errors.max())
        if error <= tol:
            break
        worst = int(errors.argmax())
        if worst < 3:
            scales[worst].mul_(p / estimates[worst].clamp_min(1e-12))
            scales[worst].clamp_(1e-12, 1e12)
        else:
            w.mul_(target_mass / current_mass.clamp_min(1e-12)).clamp_(1e-12, 1e12)

    v1, v2, v3 = scales
    B = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
    partition = (w * B.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(1e-12)
    factor = D / partition
    plan = factor * w * B
    marginals = [plan.sum((1, 2)), plan.sum((0, 2)), plan.sum((0, 1))]
    slacks = [(p - r).clamp_min(0.0) for r in marginals]
    cap = max(float((r - p).clamp_min(0).max()) for r in marginals)
    return plan, slacks, {
        "mass": float(plan.sum()),
        "mass_error": abs(float(plan.sum()) - mass),
        "cap_violation": cap,
        "constraint_error": error,
        "iterations": iteration + 1,
    }


def entropy_term(x: torch.Tensor) -> torch.Tensor:
    positive = x > 0
    return (x[positive] * (torch.log(x[positive]) - 1.0)).sum()


def partial_matching_gap(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
    mass: float,
    eta: float,
    max_iters: int,
    tol: float,
    microbatch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Partial M3G, averaged over independent OT microbatches.

    The solver is detached (Danskin/envelope differentiation).  The returned
    scalar has the true regularized gap value while its gradient is sJ-X*.
    """
    losses: List[torch.Tensor] = []
    diagnostics: List[Dict[str, float]] = []
    for start in range(0, z1.shape[0], microbatch):
        stop = min(start + microbatch, z1.shape[0])
        if stop - start < 2:
            continue
        cost = circular_variance_cost(z1[start:stop], z2[start:stop], z3[start:stop])
        with torch.no_grad():
            plan, slacks, diag = greenkhorn_mmpot(cost.detach(), mass, eta, max_iters, tol)
        n = cost.shape[0]
        diagonal = torch.arange(n, device=cost.device)
        reference_transport = mass * cost[diagonal, diagonal, diagonal].mean()
        optimum_transport = (cost * plan).sum()
        gradient_gap = reference_transport - optimum_transport

        # Entropic values make this the actual optimality gap. They are detached;
        # the envelope gradient through C remains sJ-X*.
        p = cost.new_full((n,), 1.0 / n)
        ref_plan_values = cost.new_full((n,), mass / n)
        ref_slack = (1.0 - mass) * p
        ref_entropy = entropy_term(ref_plan_values) + 3.0 * entropy_term(ref_slack)
        opt_entropy = entropy_term(plan) + sum(entropy_term(q) for q in slacks)
        true_value = gradient_gap.detach() + eta * (ref_entropy - opt_entropy)
        loss = gradient_gap + (true_value - gradient_gap.detach())
        losses.append(loss)
        diagnostics.append(diag)
    if not losses:
        raise RuntimeError("no valid MMPOT microbatch")
    summary = {
        key: sum(item[key] for item in diagnostics) / len(diagnostics)
        for key in diagnostics[0]
    }
    return torch.stack(losses).mean(), summary


@dataclass
class EpochResult:
    total: float
    reconstruction: float
    reconstruction_main: float
    reconstruction_nested: float
    reconstruction_auxiliary: float
    weighted_main_reconstruction: float
    weighted_nested_reconstruction: float
    weighted_auxiliary_reconstruction: float
    mmpot_regularizer: float
    weighted_mmpot_regularizer: float
    dead_fraction: float
    ot_mass_error: float
    ot_capacity_violation: float
    ot_constraint_error: float
    ot_iterations: float
    learning_rate: float
    samples: int
    seconds: float
    samples_per_second: float
    annealed_k_start: int
    annealed_k_end: int
    annealed_k_mean: float


def train_matryoshka_backbone(
    device: torch.device, args: argparse.Namespace
) -> Tuple[MatryoshkaBackbone, List[Dict[str, Any]]]:
    """Fine-tune the selected backbone with the standard nested-prefix MRL loss."""
    backbone = BACKBONE_SPECS[args.backbone]
    nested_dims = matryoshka_training_dims(backbone, args.topk)
    model = MatryoshkaBackbone(args.weights_cache, backbone, nested_dims).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.mrl_lr, momentum=args.mrl_momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    method_dir = args.output_dir / MATRYOSHKA
    history: List[Dict[str, Any]] = []

    train_transform, _ = matryoshka_transforms(backbone)
    dataset, _ = build_image_dataset(
        "train", args.data_root, train_transform, args.data_backend,
        args.hf_dataset_id, args.hf_revision, args.hf_token_env,
    )
    dataset = deterministic_subset(dataset, args.max_train, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        drop_last=False,
        **loader_options(args.workers, args.prefetch_factor),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    for epoch in range(args.epochs):
        model.train()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        objective_sum, classification_sum, samples, started = (
            0.0, 0.0, 0, time.time()
        )
        dimension_loss_sums = {str(dimension): 0.0 for dimension in model.nested_dims}
        for step, (images, target) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                features = model(images)
                classification, per_dimension_losses = model.classification_losses(
                    features, target
                )
                objective = args.mrl_classification_weight * classification
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()
            count = images.shape[0]
            samples += count
            objective_sum += float(objective.detach()) * count
            classification_sum += float(classification.detach()) * count
            for dimension, dimension_loss in per_dimension_losses.items():
                dimension_loss_sums[dimension] += float(dimension_loss.detach()) * count
            if args.print_freq > 0 and step % args.print_freq == 0:
                log_wandb_metrics(
                    f"{MATRYOSHKA}/batch",
                    {
                        "step": epoch * len(loader) + step,
                        "epoch": epoch + 1,
                        "total": objective.detach(),
                        "classification": classification.detach(),
                        "weighted_classification": objective.detach(),
                        "learning_rate": epoch_learning_rate,
                        "per_dimension_classification": {
                            dimension: loss.detach()
                            for dimension, loss in per_dimension_losses.items()
                        },
                    },
                    step_metric="step",
                )
                print(
                    f"{MATRYOSHKA}/{args.backbone} epoch={epoch + 1} "
                    f"step={step}/{len(loader)} "
                    f"loss={float(objective):.5f} "
                    f"mrl_ce={float(classification):.5f}", flush=True,
                )
        scheduler.step()
        seconds = time.time() - started
        record = {
            "epoch": epoch + 1,
            "total": objective_sum / samples,
            "classification": classification_sum / samples,
            "weighted_classification": objective_sum / samples,
            "classification_weight": args.mrl_classification_weight,
            "learning_rate": epoch_learning_rate,
            "seconds": seconds,
            "samples": samples,
            "samples_per_second": samples / max(seconds, 1e-12),
            "per_dimension_classification": {
                dimension: value / samples
                for dimension, value in dimension_loss_sums.items()
            },
        }
        history.append(record)
        log_wandb_metrics(MATRYOSHKA, record, step_metric="epoch")
        atomic_json(
            {
                "method": MATRYOSHKA,
                "backbone": args.backbone,
                "nested_dims": model.nested_dims,
                "loss_weights": {
                    "classification": args.mrl_classification_weight,
                },
                "history": history,
            },
            method_dir / "history.json",
        )
    return model, history


@torch.inference_mode()
def cache_matryoshka_split(
    split: str,
    model: MatryoshkaBackbone,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Cache deterministic features from the fine-tuned MRL backbone once."""
    cache_dir = args.cache_dir / "matryoshka_finetuned"
    feature_path, label_path, meta_path = cache_paths(cache_dir, split)
    maximum = args.max_train if split == "train" else args.max_val
    subset_seed = args.seed + (0 if split == "train" else 1)
    expected_metadata = {
        "cache_format": 4,
        "split": split,
        "feature_dim": model.output_dim,
        "dtype": "float16",
        "backbone": model.backbone.name,
        "weights": model.backbone.weights_id,
        "fine_tuned_with": "matryoshka_nested_cross_entropy",
        "nested_dims": list(model.nested_dims),
        "training_epochs": args.epochs,
        "training_seed": args.seed,
        "training_max_train": args.max_train,
        "training_batch_size": args.batch_size,
        "training_learning_rate": args.mrl_lr,
        "training_momentum": args.mrl_momentum,
        "training_weight_decay": args.weight_decay,
        "training_classification_weight": args.mrl_classification_weight,
        "model_weights_saved": False,
        "data_backend": args.data_backend,
        "data_root": str(args.data_root),
        "hf_dataset_id": args.hf_dataset_id if args.data_backend == "hf" else None,
        "hf_revision": args.hf_revision if args.data_backend == "hf" else None,
        "maximum": maximum,
        "subset_seed": subset_seed,
    }
    # This cache comes from the just-trained in-memory model. With no model
    # checkpoint to bind it to, rebuilding is the only safe behavior.
    print(f"cache {MATRYOSHKA} {split}: rebuilding from in-memory model", flush=True)
    _, evaluation_transform = matryoshka_transforms(model.backbone)
    dataset, source_split = build_image_dataset(
        split, args.data_root, evaluation_transform, args.data_backend,
        args.hf_dataset_id, args.hf_revision, args.hf_token_env,
    )
    dataset = deterministic_subset(dataset, maximum, subset_seed)
    loader = DataLoader(
        dataset, batch_size=args.feature_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        **loader_options(args.workers, args.prefetch_factor),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_tmp = feature_path.with_suffix(feature_path.suffix + ".tmp")
    label_tmp = label_path.with_suffix(label_path.suffix + ".tmp")
    features = np.lib.format.open_memmap(
        feature_tmp, mode="w+", dtype=np.float16, shape=(len(dataset), model.output_dim)
    )
    labels = np.lib.format.open_memmap(
        label_tmp, mode="w+", dtype=np.int64, shape=(len(dataset),)
    )
    model.eval()
    offset, started = 0, time.time()
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            batch_features = model(images).float()
        if batch_features.ndim != 2 or batch_features.shape[1] != model.output_dim:
            raise RuntimeError(
                f"{model.backbone.display_name} returned feature shape "
                f"{tuple(batch_features.shape)}; expected [batch, {model.output_dim}]"
            )
        count = images.shape[0]
        features[offset : offset + count] = batch_features.to(
            dtype=torch.float16
        ).cpu().numpy()
        labels[offset : offset + count] = target.numpy()
        offset += count
        if offset % (args.feature_batch_size * 100) < count:
            print(f"cache {MATRYOSHKA} {split}: {offset}/{len(dataset)}", flush=True)
    features.flush()
    labels.flush()
    del features, labels
    os.replace(feature_tmp, feature_path)
    os.replace(label_tmp, label_path)
    metadata = {
        **expected_metadata,
        "samples": len(dataset),
        "source_split": source_split,
        "seconds": time.time() - started,
    }
    atomic_json(metadata, meta_path)
    return metadata


def train_csr(
    initial_state: Mapping[str, torch.Tensor],
    dataset: CachedFeatures,
    device: torch.device,
    args: argparse.Namespace,
    method: str = CSR,
    use_annealing: bool = True,
) -> Tuple[TopKSAE, List[Dict[str, Any]], Dict[str, Any]]:
    """Train fixed-K CSR v1 or cosine-annealed CSRv2."""
    input_dim = int(dataset.features.shape[1])
    model = TopKSAE(input_dim, args.hidden_dim, args.dead_steps).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.csr_lr, weight_decay=args.weight_decay
    )
    method_dir = args.output_dir / method
    history: List[Dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        # Match MPSAEv2's mini-batches exactly during the common epochs. MPSAEv2
        # needs groups of at least two for the multi-marginal OT objective.
        drop_last=len(dataset) > args.batch_size,
        **loader_options(args.workers, args.prefetch_factor),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total_steps = args.epochs * len(loader)
    fixed_k = args.train_k if use_annealing else args.v1_train_k
    schedule = (
        annealing_metadata(args, total_steps)
        if use_annealing
        else fixed_k_metadata(fixed_k, total_steps)
    )

    for epoch in range(args.epochs):
        model.train()
        sums = {
            "total": 0.0,
            "reconstruction": 0.0,
            "main": 0.0,
            "multi": 0.0,
            "aux": 0.0,
            "contrastive": 0.0,
            "dead": 0.0,
        }
        samples, started = 0, time.time()
        epoch_k_values: List[int] = []
        for step, (features, _) in enumerate(loader):
            global_step = epoch * len(loader) + step
            current_k = (
                cosine_annealed_k(
                    global_step, total_steps, args.anneal_start_k,
                    args.train_k, args.anneal_fraction,
                )
                if use_annealing else fixed_k
            )
            epoch_k_values.append(current_k)
            features = features.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=args.amp and device.type == "cuda"
            ):
                reconstruction, reconstruction_stats, sparse_codes = model.csr_losses(
                    features,
                    current_k,
                    args.k_aux,
                    args.csr_main_recon_weight,
                    args.csr_multi_topk_recon_weight,
                    args.csr_aux_recon_weight,
                )
                with torch.autocast(device_type=device.type, enabled=False):
                    contrastive = nonnegative_contrastive_loss(sparse_codes.float())
                objective = reconstruction + args.csr_contrastive_weight * contrastive
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()
            model.normalize_decoder()

            count = features.shape[0]
            samples += count
            sums["total"] += float(objective.detach()) * count
            sums["reconstruction"] += float(reconstruction.detach()) * count
            sums["main"] += float(reconstruction_stats["recon"].detach()) * count
            sums["multi"] += float(
                reconstruction_stats["multi_topk_recon"].detach()
            ) * count
            sums["aux"] += float(reconstruction_stats["aux"].detach()) * count
            sums["contrastive"] += float(contrastive.detach()) * count
            sums["dead"] += float(reconstruction_stats["dead_fraction"]) * count
            if args.print_freq > 0 and step % args.print_freq == 0:
                log_wandb_metrics(
                    f"{method}/batch",
                    {
                        "step": epoch * len(loader) + step,
                        "epoch": epoch + 1,
                        "annealed_k": current_k,
                        "total": objective.detach(),
                        "reconstruction": reconstruction.detach(),
                        "reconstruction_main": reconstruction_stats["recon"].detach(),
                        "reconstruction_multi_topk": reconstruction_stats[
                            "multi_topk_recon"
                        ].detach(),
                        "reconstruction_auxiliary": reconstruction_stats["aux"].detach(),
                        "contrastive": contrastive.detach(),
                        "dead_fraction": reconstruction_stats["dead_fraction"].detach(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step_metric="step",
                )
                print(
                    f"{method}/{args.backbone} epoch={epoch + 1} "
                    f"step={step}/{len(loader)} k={current_k} "
                    f"loss={float(objective):.5f} "
                    f"recon={float(reconstruction):.5f} ncl={float(contrastive):.5f}",
                    flush=True,
                )
        seconds = time.time() - started
        record = {
            "epoch": epoch + 1,
            "total": sums["total"] / samples,
            "reconstruction": sums["reconstruction"] / samples,
            "reconstruction_main": sums["main"] / samples,
            "weighted_main_reconstruction": (
                args.csr_main_recon_weight * sums["main"] / samples
            ),
            "reconstruction_multi_topk": sums["multi"] / samples,
            "weighted_multi_topk_reconstruction": (
                args.csr_multi_topk_recon_weight * sums["multi"] / samples
            ),
            "reconstruction_auxiliary": sums["aux"] / samples,
            "weighted_auxiliary_reconstruction": (
                args.csr_aux_recon_weight * sums["aux"] / samples
            ),
            "contrastive": sums["contrastive"] / samples,
            "weighted_contrastive": (
                args.csr_contrastive_weight * sums["contrastive"] / samples
            ),
            "dead_fraction": sums["dead"] / samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "samples": samples,
            "seconds": seconds,
            "samples_per_second": samples / max(seconds, 1e-12),
            "annealed_k_start": epoch_k_values[0],
            "annealed_k_end": epoch_k_values[-1],
            "annealed_k_mean": float(np.mean(epoch_k_values)),
        }
        history.append(record)
        log_wandb_metrics(method, record, step_metric="epoch")
        atomic_json(
            {
                "method": method,
                "backbone": args.backbone,
                "topk_annealing": schedule,
                "loss_weights": {
                    "main_reconstruction": args.csr_main_recon_weight,
                    "multi_topk_reconstruction": args.csr_multi_topk_recon_weight,
                    "auxiliary_reconstruction": args.csr_aux_recon_weight,
                    "nonnegative_contrastive": args.csr_contrastive_weight,
                },
                "history": history,
            },
            method_dir / "history.json",
        )
    return model, history, schedule


def train_mp_sae(
    initial_state: Mapping[str, torch.Tensor],
    dataset: CachedFeatures,
    device: torch.device,
    args: argparse.Namespace,
    method: str = MP_SAE,
    use_annealing: bool = True,
) -> Tuple[TopKSAE, List[Dict[str, Any]], Dict[str, Any]]:
    input_dim = int(dataset.features.shape[1])
    model = TopKSAE(input_dim, args.hidden_dim, args.dead_steps).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=6.25e-10)
    method_dir = args.output_dir / method
    history: List[Dict[str, Any]] = []

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) > args.batch_size,
        **loader_options(args.workers, args.prefetch_factor),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    mpsae_epochs = args.epochs + args.mpsae_extra_epochs
    total_steps = mpsae_epochs * len(loader)
    fixed_k = args.train_k if use_annealing else args.v1_train_k
    schedule = (
        annealing_metadata(args, total_steps)
        if use_annealing
        else fixed_k_metadata(fixed_k, total_steps)
    )
    for epoch in range(mpsae_epochs):
        model.train()
        sums = {
            "total": 0.0,
            "recon": 0.0,
            "main": 0.0,
            "nested": 0.0,
            "aux": 0.0,
            "repr": 0.0,
            "dead": 0.0,
            "mass_error": 0.0,
            "cap_violation": 0.0,
            "constraint_error": 0.0,
            "ot_iterations": 0.0,
        }
        samples, started = 0, time.time()
        epoch_k_values: List[int] = []
        for step, (features, _) in enumerate(loader):
            global_step = epoch * len(loader) + step
            current_k = (
                cosine_annealed_k(
                    global_step, total_steps, args.anneal_start_k,
                    args.train_k, args.anneal_fraction,
                )
                if use_annealing else fixed_k
            )
            epoch_k_values.append(current_k)
            features = features.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                recon_loss, recon_stats, views = model.reconstruction_losses(
                    features,
                    current_k,
                    args.k_aux,
                    args.mpsae_main_recon_weight,
                    args.mpsae_nested_recon_weight,
                    args.mpsae_aux_recon_weight,
                )
                # Force the numerically sensitive OT path to float32.
                with torch.autocast(device_type=device.type, enabled=False):
                    repr_loss, ot_diag = partial_matching_gap(
                        views[0].float(), views[1].float(), views[2].float(),
                        args.ot_mass, args.ot_eta, args.ot_iters, args.ot_tol, args.ot_microbatch,
                    )
                objective = recon_loss + args.mpsae_mmpot_weight * repr_loss
                mass_error = ot_diag["mass_error"]
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()
            model.normalize_decoder()
            count = features.shape[0]
            samples += count
            sums["total"] += float(objective.detach()) * count
            sums["recon"] += float(recon_loss.detach()) * count
            sums["main"] += float(recon_stats["recon"].detach()) * count
            sums["nested"] += float(recon_stats["nested_recon"].detach()) * count
            sums["aux"] += float(recon_stats["aux"].detach()) * count
            sums["repr"] += float(repr_loss.detach()) * count
            sums["dead"] += float(recon_stats["dead_fraction"]) * count
            sums["mass_error"] += mass_error * count
            sums["cap_violation"] += ot_diag["cap_violation"] * count
            sums["constraint_error"] += ot_diag["constraint_error"] * count
            sums["ot_iterations"] += ot_diag["iterations"] * count
            if args.print_freq > 0 and step % args.print_freq == 0:
                log_wandb_metrics(
                    f"{method}/batch",
                    {
                        "step": epoch * len(loader) + step,
                        "epoch": epoch + 1,
                        "annealed_k": current_k,
                        "total": objective.detach(),
                        "reconstruction": recon_loss.detach(),
                        "reconstruction_main": recon_stats["recon"].detach(),
                        "reconstruction_nested": recon_stats[
                            "nested_recon"
                        ].detach(),
                        "reconstruction_auxiliary": recon_stats["aux"].detach(),
                        "mmpot_regularizer": repr_loss.detach(),
                        "dead_fraction": recon_stats["dead_fraction"].detach(),
                        "ot_mass_error": mass_error,
                        "ot_capacity_violation": ot_diag["cap_violation"],
                        "ot_constraint_error": ot_diag["constraint_error"],
                        "ot_iterations": ot_diag["iterations"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step_metric="step",
                )
                print(
                    f"{method}/{args.backbone} epoch={epoch+1} step={step}/{len(loader)} "
                    f"k={current_k} "
                    f"loss={float(objective):.5f} recon={float(recon_loss):.5f} repr={float(repr_loss):.5f}",
                    flush=True,
                )
        seconds = time.time() - started
        result = EpochResult(
            total=sums["total"] / samples,
            reconstruction=sums["recon"] / samples,
            reconstruction_main=sums["main"] / samples,
            reconstruction_nested=sums["nested"] / samples,
            reconstruction_auxiliary=sums["aux"] / samples,
            weighted_main_reconstruction=(
                args.mpsae_main_recon_weight * sums["main"] / samples
            ),
            weighted_nested_reconstruction=(
                args.mpsae_nested_recon_weight * sums["nested"] / samples
            ),
            weighted_auxiliary_reconstruction=(
                args.mpsae_aux_recon_weight * sums["aux"] / samples
            ),
            mmpot_regularizer=sums["repr"] / samples,
            weighted_mmpot_regularizer=(
                args.mpsae_mmpot_weight * sums["repr"] / samples
            ),
            dead_fraction=sums["dead"] / samples,
            ot_mass_error=sums["mass_error"] / samples,
            ot_capacity_violation=sums["cap_violation"] / samples,
            ot_constraint_error=sums["constraint_error"] / samples,
            ot_iterations=sums["ot_iterations"] / samples,
            learning_rate=optimizer.param_groups[0]["lr"],
            samples=samples,
            seconds=seconds,
            samples_per_second=samples / max(seconds, 1e-12),
            annealed_k_start=epoch_k_values[0],
            annealed_k_end=epoch_k_values[-1],
            annealed_k_mean=float(np.mean(epoch_k_values)),
        )
        epoch_record = {"epoch": epoch + 1, **asdict(result)}
        history.append(epoch_record)
        log_wandb_metrics(method, epoch_record, step_metric="epoch")
        atomic_json(
            {
                "method": method,
                "backbone": args.backbone,
                "training_epochs": mpsae_epochs,
                "topk_annealing": schedule,
                "loss_weights": {
                    "main_reconstruction": args.mpsae_main_recon_weight,
                    "nested_reconstruction": args.mpsae_nested_recon_weight,
                    "auxiliary_reconstruction": args.mpsae_aux_recon_weight,
                    "mmpot_regularizer": args.mpsae_mmpot_weight,
                },
                "history": history,
            },
            method_dir / "history.json",
        )
    return model, history, schedule


@torch.inference_mode()
def encode_for_benchmark(
    method: str,
    model: Optional[TopKSAE],
    features: torch.Tensor,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    features = features.to(device, non_blocking=True)
    if method == MATRYOSHKA:
        return features[:, :k].float()
    if model is None:
        raise ValueError(f"{method} benchmarking requires a trained sparse autoencoder")
    return model.encode(features, k).float()


def unit_normalize_embeddings(embeddings: torch.Tensor, method: str, k: int) -> torch.Tensor:
    """Normalize every row and fail loudly if a learned code is all zero."""
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
    zero_rows = norms.squeeze(1) <= torch.finfo(embeddings.dtype).eps
    if zero_rows.any():
        raise RuntimeError(
            f"cannot unit-normalize {int(zero_rows.sum().item())} zero {method} "
            f"embedding(s) at K={k}"
        )
    return embeddings / norms


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def add_gallery_to_faiss(
    index: Any, method: str, model: Optional[TopKSAE], dataset: CachedFeatures,
    k: int, batch_size: int, model_device: torch.device,
    index_device: torch.device, normalize: bool,
) -> torch.Tensor:
    labels: List[torch.Tensor] = []
    if model is not None:
        model.eval()
    for features, target in cached_feature_batches(dataset, batch_size):
        z = encode_for_benchmark(method, model, features, k, model_device)
        if normalize:
            z = unit_normalize_embeddings(z, method, k)
        index.add(z.to(index_device).contiguous())
        labels.append(target)
    return torch.cat(labels).to(index_device)


@torch.inference_mode()
def search_queries(
    index: Any,
    gallery_labels: torch.Tensor,
    method: str,
    model: Optional[TopKSAE],
    dataset: CachedFeatures,
    k: int,
    batch_size: int,
    model_device: torch.device,
    index_device: torch.device,
    normalize: bool,
) -> Tuple[float, float, int, float, int]:
    correct, total, distance_sum = 0, 0, 0.0
    retrieval_seconds, retrieval_batches = 0.0, 0
    if model is not None:
        model.eval()
    for features, target in cached_feature_batches(dataset, batch_size):
        z = encode_for_benchmark(method, model, features, k, model_device)
        if normalize:
            z = unit_normalize_embeddings(z, method, k)
        query = z.to(index_device).contiguous()
        synchronize_device(index_device)
        started = time.perf_counter()
        distances, indices = index.search(query, 1)
        synchronize_device(index_device)
        retrieval_seconds += time.perf_counter() - started
        retrieval_batches += 1
        predictions = gallery_labels[indices[:, 0]]
        truth = target.to(index_device, non_blocking=True)
        correct += int((predictions == truth).sum().item())
        total += truth.numel()
        distance_sum += float(distances[:, 0].sum().item())
    return (
        100.0 * correct / total,
        distance_sum / total,
        total,
        retrieval_seconds,
        retrieval_batches,
    )


def scipy_sparse_module() -> Any:
    try:
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for exact sparse CSR retrieval; install requirements.txt"
        ) from exc
    return sparse


def dense_codes_to_csr(codes: torch.Tensor) -> Any:
    """Convert a normalized Top-K batch to CSR without densifying it again."""
    sparse = scipy_sparse_module()
    codes_cpu = codes.detach().to(device="cpu", dtype=torch.float32).contiguous()
    nonzero = codes_cpu.ne(0)
    counts = nonzero.sum(dim=1).numpy().astype(np.int32, copy=False)
    coordinates = nonzero.nonzero(as_tuple=False)
    indices = coordinates[:, 1].numpy().astype(np.int32, copy=False)
    data = codes_cpu[nonzero].numpy()
    indptr = np.empty(codes_cpu.shape[0] + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return sparse.csr_matrix(
        (data, indices, indptr), shape=tuple(codes_cpu.shape), dtype=np.float32
    )


@torch.inference_mode()
def encode_sparse_gallery(
    method: str,
    model: TopKSAE,
    dataset: CachedFeatures,
    k: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[Tuple[Any, ...], np.ndarray, float]:
    """Encode the gallery as bounded CSR chunks with at most K entries per row."""
    model.eval()
    gallery_chunks: List[Any] = []
    label_chunks: List[np.ndarray] = []
    total_nnz = 0
    active_width = max(1, min(k, int(model.decoder.shape[0])))
    max_chunk_rows = max(1, np.iinfo(np.int32).max // active_width)
    csr_batch_size = min(batch_size, max_chunk_rows)

    for features, target in cached_feature_batches(dataset, csr_batch_size):
        codes = encode_for_benchmark(method, model, features, k, device)
        codes = unit_normalize_embeddings(codes, method, k)
        gallery_chunk = dense_codes_to_csr(codes)
        gallery_chunks.append(gallery_chunk)
        label_chunks.append(
            target.detach().to(device="cpu", dtype=torch.long).contiguous().numpy()
        )
        total_nnz += int(gallery_chunk.nnz)

    labels = np.concatenate(label_chunks)
    return tuple(gallery_chunks), labels, total_nnz / len(labels)


def sparse_row_argmax(similarities: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact row maxima, treating implicit sparse entries as zero."""
    similarities = similarities.tocsr()
    row_count = similarities.shape[0]
    best_indices = np.zeros(row_count, dtype=np.int64)
    best_scores = np.zeros(row_count, dtype=np.float32)
    for row in range(row_count):
        start, end = similarities.indptr[row:row + 2]
        if start == end:
            continue
        values = similarities.data[start:end]
        maximum = float(values.max())
        if maximum <= 0.0:
            continue
        columns = similarities.indices[start:end]
        best_indices[row] = int(columns[values == maximum].min())
        best_scores[row] = maximum
    return best_indices, best_scores


@torch.inference_mode()
def search_sparse_queries(
    gallery_chunks: Sequence[Any],
    gallery_labels: np.ndarray,
    method: str,
    model: TopKSAE,
    dataset: CachedFeatures,
    k: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, int, float, int, float]:
    """Exact sparse 1-NN using unit-vector dot products as L2 rankings."""
    model.eval()
    gallery_transposes = tuple(
        gallery_chunk.transpose().tocsc(copy=False)
        for gallery_chunk in gallery_chunks
    )
    correct, total, distance_sum = 0, 0, 0.0
    retrieval_seconds, retrieval_batches, query_nnz = 0.0, 0, 0

    for features, target in cached_feature_batches(dataset, batch_size):
        codes = encode_for_benchmark(method, model, features, k, device)
        codes = unit_normalize_embeddings(codes, method, k)
        queries = dense_codes_to_csr(codes)
        query_nnz += int(queries.nnz)

        started = time.perf_counter()
        indices = np.zeros(queries.shape[0], dtype=np.int64)
        scores = np.zeros(queries.shape[0], dtype=np.float32)
        gallery_offset = 0
        for gallery_chunk, gallery_transpose in zip(
            gallery_chunks, gallery_transposes
        ):
            similarities = queries @ gallery_transpose
            chunk_indices, chunk_scores = sparse_row_argmax(similarities)
            update = chunk_scores > scores
            indices[update] = gallery_offset + chunk_indices[update]
            scores[update] = chunk_scores[update]
            gallery_offset += gallery_chunk.shape[0]
        retrieval_seconds += time.perf_counter() - started
        retrieval_batches += 1

        truth = (
            target.detach()
            .to(device="cpu", dtype=torch.long)
            .contiguous()
            .numpy()
        )
        correct += int((gallery_labels[indices] == truth).sum())
        total += int(truth.size)
        distance_sum += float(np.maximum(0.0, 2.0 - 2.0 * scores).sum())

    return (
        100.0 * correct / total,
        distance_sum / total,
        total,
        retrieval_seconds,
        retrieval_batches,
        query_nnz / total,
    )


def make_faiss_index(
    dimension: int, use_gpu: bool, gpu_device: int, temp_memory_mib: int
) -> Tuple[Any, Optional[Any]]:
    try:
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401 - registers PyTorch tensor interop
    except ImportError as exc:
        package = "faiss-gpu-cu12" if use_gpu else "faiss-cpu"
        raise RuntimeError(f"FAISS is required. Install with: pip install {package}") from exc
    cpu_index = faiss.IndexFlatL2(dimension)
    if not use_gpu:
        return cpu_index, None
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("GPU FAISS requested, but the installed package is CPU-only. "
                           "Install faiss-gpu-cu12, or pass --no-faiss-gpu.")
    if gpu_device >= torch.cuda.device_count():
        raise RuntimeError(f"FAISS GPU {gpu_device} requested, but only "
                           f"{torch.cuda.device_count()} CUDA device(s) are visible")
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(temp_memory_mib * 1024**2)
    return faiss.index_cpu_to_gpu(resources, gpu_device, cpu_index), resources


def benchmark_method(
    method: str,
    model: Optional[TopKSAE],
    train_data: CachedFeatures,
    val_data: CachedFeatures,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    faiss_device = args.faiss_gpu_device
    if faiss_device is None:
        faiss_device = device.index if device.type == "cuda" and device.index is not None else 0
    index_device = torch.device(f"cuda:{faiss_device}") if args.faiss_gpu else torch.device("cpu")
    budgets = (
        list(args.topk)
        if method == MATRYOSHKA
        else sorted(set((*args.sparse_extra_topk, *args.topk)))
    )
    backend = (
        "faiss_index_flat_l2"
        if method == MATRYOSHKA
        else "scipy_chunked_csr_exact_l2"
    )

    for k in budgets:
        if method == MATRYOSHKA:
            print(f"FAISS exact L2: method={method} k={k} device={index_device}", flush=True)
            index, gpu_resources = make_faiss_index(
                k, args.faiss_gpu, faiss_device, args.faiss_temp_memory_mib
            )
            gallery_labels = add_gallery_to_faiss(
                index, method, model, train_data, k, args.knn_batch_size,
                device, index_device, args.knn_normalize
            )
            accuracy, mean_distance, queries, retrieval_seconds, retrieval_batches = (
                search_queries(
                    index, gallery_labels, method, model, val_data, k,
                    args.knn_query_batch, device, index_device, args.knn_normalize
                )
            )
            average_gallery_nnz = float(k)
            average_query_nnz = float(k)
            gallery_samples = len(gallery_labels)
            del index, gpu_resources, gallery_labels
        else:
            if model is None:
                raise ValueError(f"{method} sparse retrieval requires a trained model")
            print(
                f"SciPy chunked CSR exact L2: method={method} k={k} device=cpu",
                flush=True,
            )
            gallery_chunks, gallery_labels, average_gallery_nnz = (
                encode_sparse_gallery(
                    method, model, train_data, k, args.knn_batch_size, device
                )
            )
            (
                accuracy,
                mean_distance,
                queries,
                retrieval_seconds,
                retrieval_batches,
                average_query_nnz,
            ) = search_sparse_queries(
                gallery_chunks, gallery_labels, method, model, val_data, k,
                args.sparse_knn_query_batch, device
            )
            gallery_samples = len(gallery_labels)
            del gallery_chunks, gallery_labels

        results[str(k)] = {
            "top1": accuracy,
            "mean_neighbor_l2_squared": mean_distance,
            "gallery_samples": gallery_samples,
            "query_samples": queries,
            "retrieval_search_seconds": retrieval_seconds,
            "retrieval_query_batches": retrieval_batches,
            "mean_retrieval_seconds_per_query": retrieval_seconds / queries,
            "mean_retrieval_milliseconds_per_query": 1000.0 * retrieval_seconds / queries,
            "average_gallery_nonzero_coordinates": average_gallery_nnz,
            "average_query_nonzero_coordinates": average_query_nnz,
            "retrieval_backend": backend,
        }
        log_wandb_metrics(
            f"{method}/knn",
            {"budget": k, **results[str(k)]},
            step_metric="budget",
        )
    return {
        "protocol": "unit_normalized_exact_L2_train_gallery_validation_queries_1NN",
        "retrieval_backend": backend,
        "device": str(index_device) if method == MATRYOSHKA else "cpu",
        "normalized": args.knn_normalize,
        "representation": "prefix_dimension" if method == MATRYOSHKA else "topk_sparse_latents",
        "timing_scope": "search_only_excludes_encoding_normalization_and_index_construction",
        "per_topk": results,
    }


def comparison_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not all(method in results for method in METHODS):
        return []
    baseline = results[MATRYOSHKA]["knn"]["per_topk"]
    csr = results[CSR]["knn"]["per_topk"]
    proposed = results[MP_SAE]["knn"]["per_topk"]
    mrl_weights = results[MATRYOSHKA].get("loss_weights", {})
    csr_weights = results[CSR].get("loss_weights", {})
    mpsae_weights = results[MP_SAE].get("loss_weights", {})
    csr_annealing = results[CSR].get("topk_annealing", {})
    mpsae_annealing = results[MP_SAE].get("topk_annealing", {})
    rows: List[Dict[str, Any]] = []
    for k in sorted(int(value) for value in baseline):
        mrl_metrics = baseline[str(k)]
        csr_metrics = csr[str(k)]
        mp_sae_metrics = proposed[str(k)]
        rows.append({
            "backbone": results[MATRYOSHKA].get("backbone", "unknown"),
            "representation_budget": k,
            "matryoshka_prefix_dim": k,
            "csrv2_active_latents": k,
            "mpsaev2_active_latents": k,
            "matryoshka_1nn_top1": mrl_metrics["top1"],
            "csrv2_1nn_top1": csr_metrics["top1"],
            "mpsaev2_1nn_top1": mp_sae_metrics["top1"],
            "delta_csrv2_minus_matryoshka": csr_metrics["top1"] - mrl_metrics["top1"],
            "delta_mpsaev2_minus_matryoshka": mp_sae_metrics["top1"] - mrl_metrics["top1"],
            "delta_mpsaev2_minus_csrv2": mp_sae_metrics["top1"] - csr_metrics["top1"],
            "matryoshka_mean_neighbor_l2_squared": mrl_metrics["mean_neighbor_l2_squared"],
            "csrv2_mean_neighbor_l2_squared": csr_metrics["mean_neighbor_l2_squared"],
            "mpsaev2_mean_neighbor_l2_squared": mp_sae_metrics["mean_neighbor_l2_squared"],
            "matryoshka_mean_retrieval_ms_per_query": mrl_metrics[
                "mean_retrieval_milliseconds_per_query"
            ],
            "csrv2_mean_retrieval_ms_per_query": csr_metrics[
                "mean_retrieval_milliseconds_per_query"
            ],
            "mpsaev2_mean_retrieval_ms_per_query": mp_sae_metrics[
                "mean_retrieval_milliseconds_per_query"
            ],
            "mrl_classification_weight": mrl_weights.get("classification"),
            "annealing_schedule": csr_annealing.get("schedule"),
            "anneal_start_k": csr_annealing.get("start_k"),
            "anneal_target_k": csr_annealing.get("target_k"),
            "anneal_fraction": csr_annealing.get("anneal_fraction"),
            "csrv2_anneal_steps": csr_annealing.get("anneal_steps"),
            "mpsaev2_anneal_steps": mpsae_annealing.get("anneal_steps"),
            "csrv2_main_recon_weight": csr_weights.get("main_reconstruction"),
            "csrv2_multi_topk_recon_weight": csr_weights.get(
                "multi_topk_reconstruction"
            ),
            "csrv2_aux_recon_weight": csr_weights.get("auxiliary_reconstruction"),
            "csrv2_contrastive_weight": csr_weights.get("nonnegative_contrastive"),
            "mpsaev2_main_recon_weight": mpsae_weights.get("main_reconstruction"),
            "mpsaev2_nested_recon_weight": mpsae_weights.get(
                "nested_reconstruction"
            ),
            "mpsaev2_aux_recon_weight": mpsae_weights.get(
                "auxiliary_reconstruction"
            ),
            "mpsaev2_mmpot_weight": mpsae_weights.get("mmpot_regularizer"),
        })
    return rows


def write_comparison_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_markdown_table(
    rows: Sequence[Mapping[str, Any]], path: Path, backbone: BackboneSpec
) -> None:
    if not rows:
        return
    lines = [
        f"| Budget K | Matryoshka {backbone.display_name} | CSRv2 | MPSAEv2 | CSRv2 - Matryoshka | MPSAEv2 - Matryoshka | MPSAEv2 - CSRv2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['representation_budget']} | {row['matryoshka_1nn_top1']:.2f} | "
            f"{row['csrv2_1nn_top1']:.2f} | {row['mpsaev2_1nn_top1']:.2f} | "
            f"{row['delta_csrv2_minus_matryoshka']:+.2f} | "
            f"{row['delta_mpsaev2_minus_matryoshka']:+.2f} | "
            f"{row['delta_mpsaev2_minus_csrv2']:+.2f} |"
        )
    mrl_mean = float(np.mean([row["matryoshka_1nn_top1"] for row in rows]))
    csr_mean = float(np.mean([row["csrv2_1nn_top1"] for row in rows]))
    mp_sae_mean = float(np.mean([row["mpsaev2_1nn_top1"] for row in rows]))
    lines.append(
        f"| **Mean** | **{mrl_mean:.2f}** | **{csr_mean:.2f}** | "
        f"**{mp_sae_mean:.2f}** | **{csr_mean - mrl_mean:+.2f}** | "
        f"**{mp_sae_mean - mrl_mean:+.2f}** | **{mp_sae_mean - csr_mean:+.2f}** |"
    )
    lines.extend([
        "",
        "Values are ImageNet validation unit-normalized exact L2 1-NN top-1 accuracy (%).",
        "This matched table starts at K=8; sparse-only K=1,2,4 results remain in summary.json and the plots.",
        "K denotes prefix dimension for Matryoshka and active latents for CSRv2 and MPSAEv2; configured loss weights and annealing settings are recorded in comparison.csv and summary.json.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(
    rows: Sequence[Mapping[str, Any]], path: Path, backbone: BackboneSpec
) -> None:
    if not rows:
        return
    command, row_end = chr(92), chr(92) * 2

    def emphasized(value: float, best: float) -> str:
        formatted = f"{value:.2f}"
        return f"{command}textbf{{{formatted}}}" if value >= best else formatted

    lines = [
        f"{command}begin{{table}}[t]",
        f"{command}centering",
        f"{command}caption{{ImageNet validation unit-normalized exact L2 1-NN top-1 accuracy ({command}%). "
        f"The budget K is the {backbone.display_name} prefix dimension for Matryoshka and the number "
        "of active sparse latents for CSRv2 and MPSAEv2. Configured loss weights and annealing settings are recorded with the results.}",
        f"{command}label{{tab:{backbone.name}-matryoshka-csrv2-mpsaev2}}",
        f"{command}small",
        f"{command}begin{{tabular}}{{rrrrrrr}}",
        f"{command}toprule",
        f"K & Matryoshka & CSRv2 & MPSAEv2 & CSRv2-M & MPSAEv2-M & MPSAEv2-CSRv2 {row_end}",
        f"{command}midrule",
    ]
    for row in rows:
        mrl = float(row["matryoshka_1nn_top1"])
        csr = float(row["csrv2_1nn_top1"])
        mp_sae = float(row["mpsaev2_1nn_top1"])
        best = max(mrl, csr, mp_sae)
        lines.append(
            f"{row['representation_budget']} & {emphasized(mrl, best)} & "
            f"{emphasized(csr, best)} & {emphasized(mp_sae, best)} & "
            f"{csr - mrl:+.2f} & {mp_sae - mrl:+.2f} & {mp_sae - csr:+.2f} {row_end}"
        )
    mrl_mean = float(np.mean([row["matryoshka_1nn_top1"] for row in rows]))
    csr_mean = float(np.mean([row["csrv2_1nn_top1"] for row in rows]))
    mp_sae_mean = float(np.mean([row["mpsaev2_1nn_top1"] for row in rows]))
    best_mean = max(mrl_mean, csr_mean, mp_sae_mean)
    lines.extend([
        f"{command}midrule",
        f"Mean & {emphasized(mrl_mean, best_mean)} & {emphasized(csr_mean, best_mean)} & "
        f"{emphasized(mp_sae_mean, best_mean)} & {csr_mean - mrl_mean:+.2f} & "
        f"{mp_sae_mean - mrl_mean:+.2f} & {mp_sae_mean - csr_mean:+.2f} {row_end}",
        f"{command}bottomrule",
        f"{command}end{{tabular}}",
        f"{command}end{{table}}",
    ])
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def configure_publication_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot_publication_comparison(results: Mapping[str, Any], output_dir: Path) -> None:
    rows = comparison_rows(results)
    if not rows:
        return
    configure_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    common_budgets = [row["representation_budget"] for row in rows]
    per_method = {
        method: results[method]["knn"]["per_topk"] for method in METHODS
    }
    method_budgets = {
        method: sorted(int(value) for value in metrics)
        for method, metrics in per_method.items()
    }
    all_budgets = sorted(
        {budget for budgets in method_budgets.values() for budget in budgets}
    )
    mrl = [per_method[MATRYOSHKA][str(k)]["top1"] for k in method_budgets[MATRYOSHKA]]
    csr = [per_method[CSR][str(k)]["top1"] for k in method_budgets[CSR]]
    mp_sae = [per_method[MP_SAE][str(k)]["top1"] for k in method_budgets[MP_SAE]]
    csr_delta = [row["delta_csrv2_minus_matryoshka"] for row in rows]
    mp_delta = [row["delta_mpsaev2_minus_matryoshka"] for row in rows]
    blue, green, orange = "#0072B2", "#009E73", "#D55E00"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)

    axes[0].plot(
        method_budgets[MATRYOSHKA], mrl, color=blue, marker="o",
        label=results[MATRYOSHKA]["display_name"],
    )
    axes[0].plot(
        method_budgets[CSR_V1],
        [per_method[CSR_V1][str(k)]["top1"] for k in method_budgets[CSR_V1]],
        color="#56B4E9", marker="^", linestyle="--", label="CSR v1",
    )
    axes[0].plot(method_budgets[CSR], csr, color=green, marker="^", label="CSRv2")
    axes[0].plot(
        method_budgets[MP_SAE_V1],
        [per_method[MP_SAE_V1][str(k)]["top1"] for k in method_budgets[MP_SAE_V1]],
        color="#E69F00", marker="s", linestyle="--", label="MPSAE v1",
    )
    axes[0].plot(
        method_budgets[MP_SAE], mp_sae, color=orange, marker="s", label="MPSAEv2"
    )
    axes[0].set_title("(a) Exact 1-NN accuracy", loc="left", fontweight="bold")
    axes[0].set_ylabel("ImageNet val. top-1 accuracy (%)")
    axes[0].legend(frameon=False, handlelength=2.2)

    axes[1].plot(
        common_budgets, csr_delta, color=green, marker="^", label="CSRv2 - Matryoshka"
    )
    axes[1].plot(
        common_budgets, mp_delta, color=orange, marker="s", label="MPSAEv2 - Matryoshka"
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_title("(b) Effect relative to Matryoshka", loc="left", fontweight="bold")
    axes[1].set_ylabel("Delta top-1 accuracy (pp)")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xlabel("Representation budget K")
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(FixedLocator(all_budgets))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(axis="y", color="#B8B8B8", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = results[MATRYOSHKA].get(
        "backbone", results[MP_SAE].get("backbone", "backbone")
    )
    save_figure_formats(
        fig, output_dir, f"ablation_{backbone_name}_representation_accuracy_comparison"
    )
    plt.close(fig)


def plot_retrieval_time(results: Mapping[str, Any], output_dir: Path) -> None:
    """Plot search-only mean retrieval time for each backbone-specific run."""
    if not all(method in results for method in METHODS):
        return
    configure_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    colors = {
        MATRYOSHKA: "#0072B2", CSR_V1: "#56B4E9", CSR: "#009E73",
        MP_SAE_V1: "#E69F00", MP_SAE: "#D55E00",
    }
    markers = {
        MATRYOSHKA: "o", CSR_V1: "v", CSR: "^", MP_SAE_V1: "D", MP_SAE: "s"
    }
    labels = {
        MATRYOSHKA: "Matryoshka", CSR_V1: "CSR v1", CSR: "CSRv2",
        MP_SAE_V1: "MPSAE v1", MP_SAE: "MPSAEv2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    averages: List[float] = []
    all_budgets: set[int] = set()
    common_budgets = sorted(
        set.intersection(
            *[
                {int(value) for value in results[method]["knn"]["per_topk"]}
                for method in METHODS
            ]
        )
    )

    for method in METHODS:
        metrics = results[method]["knn"]["per_topk"]
        budgets = sorted(int(value) for value in metrics)
        milliseconds = [
            metrics[str(k)]["mean_retrieval_milliseconds_per_query"] for k in budgets
        ]
        all_budgets.update(budgets)
        averages.append(
            float(np.mean([
                metrics[str(k)]["mean_retrieval_milliseconds_per_query"]
                for k in common_budgets
            ]))
        )
        axes[0].plot(
            budgets, milliseconds, color=colors[method], marker=markers[method],
            label=labels[method],
        )

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].xaxis.set_major_locator(FixedLocator(sorted(all_budgets)))
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].set_xlabel("Representation budget K")
    axes[0].set_ylabel("Mean search time per query (ms)")
    axes[0].set_title("(a) Retrieval time by budget", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    positions = np.arange(len(METHODS))
    axes[1].bar(positions, averages, color=[colors[method] for method in METHODS])
    axes[1].set_xticks(positions, [labels[method] for method in METHODS], rotation=15)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean search time per query (ms)")
    axes[1].set_title("(b) Average across matched K", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#B8B8B8", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    backbone_name = results[MATRYOSHKA].get("backbone", "backbone")
    save_figure_formats(
        fig, output_dir, f"ablation_{backbone_name}_retrieval_time_comparison"
    )
    plt.close(fig)


def plot_training_diagnostics(results: Mapping[str, Any], output_dir: Path) -> None:
    if not all(method in results and results[method].get("history") for method in METHODS):
        return
    configure_publication_style()
    import matplotlib.pyplot as plt

    mrl_history = results[MATRYOSHKA]["history"]
    csr_history = results[CSR]["history"]
    mp_sae_history = results[MP_SAE]["history"]
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.75), constrained_layout=True)

    axes[0].plot(
        [row["epoch"] for row in mrl_history],
        [row["weighted_classification"] for row in mrl_history],
        color=blue, marker="o",
    )
    axes[0].set_title(
        f"(a) {results[MATRYOSHKA]['display_name']}", loc="left", fontweight="bold"
    )
    axes[0].set_ylabel("Weighted nested cross-entropy")

    csr_epochs = [row["epoch"] for row in csr_history]
    axes[1].plot(csr_epochs, [row["total"] for row in csr_history], color=green, marker="^", label="Total")
    axes[1].plot(
        csr_epochs, [row["reconstruction"] for row in csr_history],
        color="#56B4E9", marker="o", linestyle="--", label="Reconstruction",
    )
    axes[1].plot(
        csr_epochs, [row["weighted_contrastive"] for row in csr_history],
        color="#CC79A7", marker="s", linestyle=":", label="NCL",
    )
    axes[1].set_title("(b) CSRv2", loc="left", fontweight="bold")
    axes[1].set_ylabel("Training loss")
    axes[1].legend(frameon=False)

    epochs = [row["epoch"] for row in mp_sae_history]
    axes[2].plot(epochs, [row["total"] for row in mp_sae_history], color=orange, marker="s", label="Total")
    axes[2].plot(
        epochs, [row["reconstruction"] for row in mp_sae_history],
        color=green, marker="o", linestyle="--", label="Reconstruction",
    )
    axes[2].plot(
        epochs, [row["weighted_mmpot_regularizer"] for row in mp_sae_history],
        color="#CC79A7", marker="^", linestyle=":", label="Weighted MMPOT",
    )
    axes[2].set_title("(c) MPSAEv2", loc="left", fontweight="bold")
    axes[2].set_ylabel("Training loss")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", color="#B8B8B8", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    backbone_name = results[MATRYOSHKA].get(
        "backbone", results[MP_SAE].get("backbone", "backbone")
    )
    save_figure_formats(fig, output_dir, f"ablation_{backbone_name}_training_loss_curves")
    plt.close(fig)


def plot_topk_annealing(
    results: Mapping[str, Any], output_dir: Path, backbone: BackboneSpec
) -> None:
    """Plot fixed v1 and annealed v2 training supports for all sparse methods."""
    histories = {
        method: results.get(method, {}).get("history", [])
        for method in SPARSE_METHODS
        if results.get(method, {}).get("history")
    }
    if not histories:
        return
    configure_publication_style()
    import matplotlib.pyplot as plt

    colors = {
        CSR_V1: "#56B4E9", MP_SAE_V1: "#E69F00",
        CSR: "#009E73", MP_SAE: "#D55E00",
    }
    fig, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    for method, history in histories.items():
        epochs = [row["epoch"] for row in history]
        starts = [row["annealed_k_start"] for row in history]
        ends = [row["annealed_k_end"] for row in history]
        means = [row["annealed_k_mean"] for row in history]
        label = results[method]["display_name"]
        axis.plot(epochs, means, color=colors[method], marker="o", label=label)
        axis.fill_between(epochs, ends, starts, color=colors[method], alpha=0.14)
    axis.set_title("Fixed v1 and annealed v2 Top-K training", loc="left", fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Active training K")
    axis.grid(axis="y", color="#B8B8B8", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    save_figure_formats(
        fig, output_dir, f"ablation_{backbone.name}_topk_annealing_schedule"
    )
    plt.close(fig)


def training_loss_records(
    results: Mapping[str, Any], args: argparse.Namespace
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return long-form epoch records and start-to-finish component impacts."""
    records: List[Dict[str, Any]] = []
    impacts: List[Dict[str, Any]] = []
    component_series: Dict[Tuple[str, str], List[Tuple[int, float, float, float]]] = {}

    for method in METHODS:
        history = results.get(method, {}).get("history", [])
        for index, row in enumerate(history):
            epoch = int(row["epoch"])
            total = float(row["total"])
            components: List[Tuple[str, str, float, float]] = [
                ("optimized_total", "objective", total, 1.0)
            ]
            if method == MATRYOSHKA:
                per_dimension = row.get("per_dimension_classification", {})
                components.extend(
                    (
                        f"classification_dim_{dimension}",
                        "objective_component",
                        float(value),
                        args.mrl_classification_weight,
                    )
                    for dimension, value in sorted(
                        per_dimension.items(), key=lambda item: int(item[0])
                    )
                )
            elif method in (CSR_V1, CSR):
                components.extend(
                    [
                        (
                            "reconstruction_main",
                            "objective_component",
                            float(row["reconstruction_main"]),
                            args.csr_main_recon_weight,
                        ),
                        (
                            "reconstruction_multi_topk",
                            "objective_component",
                            float(row["reconstruction_multi_topk"]),
                            args.csr_multi_topk_recon_weight,
                        ),
                        (
                            "reconstruction_auxiliary",
                            "objective_component",
                            float(row["reconstruction_auxiliary"]),
                            args.csr_aux_recon_weight,
                        ),
                        (
                            "nonnegative_contrastive",
                            "objective_component",
                            float(row["contrastive"]),
                            args.csr_contrastive_weight,
                        ),
                    ]
                )
            else:
                components.extend(
                    [
                        (
                            "reconstruction_main",
                            "objective_component",
                            float(row.get("reconstruction_main", row["reconstruction"])),
                            args.mpsae_main_recon_weight,
                        ),
                        (
                            "reconstruction_nested",
                            "objective_component",
                            float(row.get("reconstruction_nested", 0.0)),
                            args.mpsae_nested_recon_weight,
                        ),
                        (
                            "reconstruction_auxiliary",
                            "objective_component",
                            float(row.get("reconstruction_auxiliary", 0.0)),
                            args.mpsae_aux_recon_weight,
                        ),
                        (
                            "mmpot_regularizer",
                            "objective_component",
                            float(row["mmpot_regularizer"]),
                            args.mpsae_mmpot_weight,
                        ),
                    ]
                )

            for component, role, raw_value, objective_weight in components:
                weighted_value = raw_value * objective_weight
                key = (method, component)
                previous = component_series.get(key, [])
                initial = previous[0][2] if previous else weighted_value
                previous_value = previous[-1][2] if previous else None
                component_series.setdefault(key, []).append(
                    (epoch, raw_value, weighted_value, total)
                )
                records.append(
                    {
                        "method": method,
                        "epoch": epoch,
                        "loss_component": component,
                        "role": role,
                        "raw_value": raw_value,
                        "objective_weight": objective_weight,
                        "weighted_value": weighted_value,
                        "decrease_from_previous_epoch": (
                            previous_value - weighted_value
                            if previous_value is not None
                            else None
                        ),
                        "decrease_from_first_epoch": initial - weighted_value,
                        "objective_contribution_percent": (
                            100.0 * weighted_value / total
                            if role == "objective_component" and abs(total) > 1e-12
                            else None
                        ),
                        "learning_rate": row.get("learning_rate"),
                        "epoch_seconds": row.get("seconds"),
                        "samples_per_second": row.get("samples_per_second"),
                        "dead_latent_fraction": row.get("dead_fraction"),
                        "ot_mass_error": row.get("ot_mass_error"),
                        "ot_capacity_violation": row.get("ot_capacity_violation"),
                        "ot_constraint_error": row.get("ot_constraint_error"),
                        "mean_ot_iterations": row.get("ot_iterations"),
                        "annealed_k_start": row.get("annealed_k_start"),
                        "annealed_k_end": row.get("annealed_k_end"),
                        "annealed_k_mean": row.get("annealed_k_mean"),
                    }
                )

    for (method, component), series in component_series.items():
        initial_raw, final_raw = series[0][1], series[-1][1]
        initial_weighted, final_weighted = series[0][2], series[-1][2]
        final_total = series[-1][3]
        impacts.append(
            {
                "method": method,
                "loss_component": component,
                "role": "objective" if component == "optimized_total" else "objective_component",
                "epochs_recorded": len(series),
                "initial_raw_value": initial_raw,
                "final_raw_value": final_raw,
                "minimum_raw_value": min(item[1] for item in series),
                "objective_weight": (
                    final_weighted / final_raw if abs(final_raw) > 1e-12 else None
                ),
                "initial_weighted_value": initial_weighted,
                "final_weighted_value": final_weighted,
                "absolute_decrease": initial_weighted - final_weighted,
                "percent_decrease_from_initial": (
                    100.0 * (initial_weighted - final_weighted) / abs(initial_weighted)
                    if abs(initial_weighted) > 1e-12
                    else None
                ),
                "decreased_from_initial": final_weighted < initial_weighted,
                "final_objective_contribution_percent": (
                    100.0 * final_weighted / final_total
                    if component != "optimized_total" and abs(final_total) > 1e-12
                    else None
                ),
            }
        )
    return records, impacts


def write_training_loss_analysis(
    results: Mapping[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    backbone: BackboneSpec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records, impacts = training_loss_records(results, args)
    prefix = f"ablation_{backbone.name}"
    history_path = output_dir / f"{prefix}_training_loss_history.csv"
    impact_path = output_dir / f"{prefix}_loss_component_impact.csv"
    write_comparison_csv(records, history_path)
    write_comparison_csv(impacts, impact_path)
    atomic_json(
        {
            "impact_definition": (
                "Measured weighted contribution to the optimized objective and "
                "observed start-to-finish decrease; not a causal ablation estimate."
            ),
            "loss_weights": {
                MATRYOSHKA: {
                    "classification": args.mrl_classification_weight,
                },
                CSR: {
                    "main_reconstruction": args.csr_main_recon_weight,
                    "multi_topk_reconstruction": args.csr_multi_topk_recon_weight,
                    "auxiliary_reconstruction": args.csr_aux_recon_weight,
                    "nonnegative_contrastive": args.csr_contrastive_weight,
                },
                MP_SAE: {
                    "main_reconstruction": args.mpsae_main_recon_weight,
                    "nested_reconstruction": args.mpsae_nested_recon_weight,
                    "auxiliary_reconstruction": args.mpsae_aux_recon_weight,
                    "mmpot_regularizer": args.mpsae_mmpot_weight,
                },
            },
            "components": impacts,
        },
        output_dir / f"{prefix}_loss_component_impact.json",
    )
    return records, impacts


def save_figure_formats(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_training_procedure(
    results: Mapping[str, Any],
    impacts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    backbone: BackboneSpec,
) -> None:
    histories = {
        method: results.get(method, {}).get("history", [])
        for method in METHODS
        if results.get(method, {}).get("history")
    }
    if not histories:
        return
    configure_publication_style()
    import matplotlib.pyplot as plt

    prefix = f"ablation_{backbone.name}"
    colors = {
        MATRYOSHKA: "#0072B2", CSR_V1: "#56B4E9", CSR: "#009E73",
        MP_SAE_V1: "#E69F00", MP_SAE: "#D55E00",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)
    for method, history in histories.items():
        epochs = [row["epoch"] for row in history]
        label = results[method]["display_name"]
        axes[0, 0].plot(
            epochs, [row["total"] for row in history],
            color=colors[method], marker="o", label=label,
        )
        axes[1, 0].plot(
            epochs, [row.get("learning_rate", math.nan) for row in history],
            color=colors[method], marker="o", label=label,
        )
        axes[1, 1].plot(
            epochs, [row.get("samples_per_second", math.nan) for row in history],
            color=colors[method], marker="o", label=label,
        )
        if method == MATRYOSHKA:
            classification_weight = results[MATRYOSHKA].get(
                "loss_weights", {}
            ).get("classification", 1.0)
            for dimension in sorted(
                history[-1].get("per_dimension_classification", {}), key=int
            ):
                axes[0, 1].plot(
                    epochs,
                    [
                        classification_weight
                        * row.get("per_dimension_classification", {}).get(
                            dimension, math.nan
                        )
                        for row in history
                    ],
                    alpha=0.55,
                    linewidth=1.0,
                    label=f"MRL CE dim {dimension}",
                )
        elif method in (CSR_V1, CSR):
            method_name = "CSR v1" if method == CSR_V1 else "CSRv2"
            component_specs = (
                ("weighted_main_reconstruction", f"{method_name} weighted main reconstruction", "#009E73"),
                ("weighted_multi_topk_reconstruction", f"{method_name} weighted 4K reconstruction", "#56B4E9"),
                ("weighted_auxiliary_reconstruction", f"{method_name} weighted auxiliary", "#E69F00"),
                ("weighted_contrastive", f"{method_name} weighted NCL", "#CC79A7"),
            )
            for key, label_name, color in component_specs:
                axes[0, 1].plot(
                    epochs, [row.get(key, math.nan) for row in history],
                    marker=".", color=color, linestyle="--", label=label_name,
                )
        else:
            component_specs = (
                ("weighted_main_reconstruction", "Weighted main reconstruction", "#009E73"),
                ("weighted_nested_reconstruction", "Weighted nested reconstruction", "#56B4E9"),
                ("weighted_auxiliary_reconstruction", "Weighted auxiliary", "#E69F00"),
                ("weighted_mmpot_regularizer", "Weighted MMPOT", "#CC79A7"),
            )
            for key, label_name, color in component_specs:
                axes[0, 1].plot(
                    epochs, [row.get(key, math.nan) for row in history],
                    marker=".", color=color, label=label_name,
                )

    titles = (
        "(a) Optimized objectives",
        "(b) Every objective component",
        "(c) Learning-rate schedule",
        "(d) Training throughput",
    )
    ylabels = ("Mean loss", "Mean loss contribution", "Learning rate", "Samples / second")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        axis.legend(frameon=False, ncol=2 if axis is axes[0, 1] else 1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_figure_formats(fig, output_dir, f"{prefix}_training_procedure_overview")
    plt.close(fig)

    component_impacts = [
        row for row in impacts if row.get("role") == "objective_component"
    ]
    if component_impacts:
        labels = [
            f"{row['method']}\n{str(row['loss_component']).replace('_', ' ')}"
            for row in component_impacts
        ]
        decreases = [
            row["percent_decrease_from_initial"]
            if row["percent_decrease_from_initial"] is not None else 0.0
            for row in component_impacts
        ]
        contributions = [
            row["final_objective_contribution_percent"]
            if row["final_objective_contribution_percent"] is not None else 0.0
            for row in component_impacts
        ]
        positions = np.arange(len(labels))
        fig, axes = plt.subplots(2, 1, figsize=(max(7.0, 0.85 * len(labels)), 5.2), constrained_layout=True)
        axes[0].bar(positions, decreases, color="#0072B2")
        axes[0].axhline(0.0, color="#333333", linewidth=0.8)
        axes[0].set_ylabel("Decrease from first epoch (%)")
        axes[0].set_title("(a) Observed loss decrease", loc="left", fontweight="bold")
        axes[1].bar(positions, contributions, color="#D55E00")
        axes[1].axhline(0.0, color="#333333", linewidth=0.8)
        axes[1].set_ylabel("Final objective contribution (%)")
        axes[1].set_title("(b) Weighted objective impact", loc="left", fontweight="bold")
        for axis in axes:
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            axis.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        save_figure_formats(fig, output_dir, f"{prefix}_loss_component_impact")
        plt.close(fig)


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    public_names = {
        "csr_main_recon_weight": "csrv2_main_recon_weight",
        "csr_multi_topk_recon_weight": "csrv2_multi_topk_recon_weight",
        "csr_aux_recon_weight": "csrv2_aux_recon_weight",
        "csr_contrastive_weight": "csrv2_contrastive_weight",
        "mpsae_main_recon_weight": "mpsaev2_main_recon_weight",
        "mpsae_nested_recon_weight": "mpsaev2_nested_recon_weight",
        "mpsae_aux_recon_weight": "mpsaev2_aux_recon_weight",
        "mpsae_mmpot_weight": "mpsaev2_mmpot_weight",
        "mpsae_extra_epochs": "mpsaev2_extra_epochs",
        "lr": "mpsaev2_lr",
        "csr_lr": "csrv2_lr",
    }
    for internal_name, public_name in public_names.items():
        payload[public_name] = payload.pop(internal_name)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started = time.time()
    args = build_parser().parse_args(argv)
    validate_args(args)
    backbone = BACKBONE_SPECS[args.backbone]
    labels = method_labels(backbone)
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.weights_cache = args.weights_cache.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = choose_device(args.device)
    configure_runtime(args, device)
    init_wandb(
        args,
        default_name=f"five-method-{backbone.name}-{args.method}",
        default_group="five-method-architecture-ablation",
        extra_config={
            "experiment_family": "matryoshka_csr_v1_v2_mpsae_v1_v2",
            "backbone_display_name": backbone.display_name,
            "backbone_weights": backbone.weights_id,
            "public_experiment_config": serializable_args(args),
        },
        tags=(
            "matryoshka", "csr", "mpsae", "csrv2", "mpsaev2",
            "topk-annealing", args.backbone, args.method,
        ),
    )
    print(
        f"backbone={backbone.name} feature_dim={backbone.output_dim} "
        f"sae_hidden_dim={args.hidden_dim} device={device} output={args.output_dir}",
        flush=True,
    )

    results: Dict[str, Any] = {}
    dataset_metadata: Dict[str, Any] = {}

    if args.method in (MATRYOSHKA, "all"):
        seed_all(args.seed)
        matryoshka_model, history = train_matryoshka_backbone(device, args)
        nested_dims = list(matryoshka_model.nested_dims)
        mrl_train_meta = cache_matryoshka_split("train", matryoshka_model, device, args)
        mrl_val_meta = cache_matryoshka_split("val", matryoshka_model, device, args)
        mrl_cache_dir = args.cache_dir / "matryoshka_finetuned"
        mrl_train_features, mrl_train_labels, _ = cache_paths(mrl_cache_dir, "train")
        mrl_val_features, mrl_val_labels, _ = cache_paths(mrl_cache_dir, "val")
        mrl_train_data = CachedFeatures(mrl_train_features, mrl_train_labels)
        mrl_val_data = CachedFeatures(mrl_val_features, mrl_val_labels)
        knn = benchmark_method(
            MATRYOSHKA, None, mrl_train_data, mrl_val_data, device, args
        )
        results[MATRYOSHKA] = {
            "display_name": labels[MATRYOSHKA],
            "backbone": backbone.name,
            "feature_dim": backbone.output_dim,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in matryoshka_model.parameters()
                if parameter.requires_grad
            ),
            "training_protocol": f"end_to_end_{backbone.name}_mrl_sum_cross_entropy",
            "training_epochs": args.epochs,
            "nested_dims": nested_dims,
            "loss_weights": {
                "classification": args.mrl_classification_weight,
            },
            "history": history,
            "knn": knn,
        }
        dataset_metadata[MATRYOSHKA] = {
            "train": mrl_train_meta, "validation": mrl_val_meta
        }
        atomic_json(
            results[MATRYOSHKA], args.output_dir / MATRYOSHKA / "results.json"
        )
        del matryoshka_model, mrl_train_data, mrl_val_data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.method in (*SPARSE_METHODS, "all"):
        frozen_backbone = FrozenBackbone(args.weights_cache, backbone).to(device)
        if args.channels_last:
            frozen_backbone = frozen_backbone.to(memory_format=torch.channels_last)
        frozen_train_meta = cache_split(
            "train", args.data_root, frozen_backbone, device, args.cache_dir,
            args.feature_batch_size, args.workers, args.prefetch_factor, args.channels_last,
            args.max_train, args.seed,
            args.rebuild_cache, args.data_backend, args.hf_dataset_id,
            args.hf_revision, args.hf_token_env,
        )
        frozen_val_meta = cache_split(
            "val", args.data_root, frozen_backbone, device, args.cache_dir,
            args.feature_batch_size, args.workers, args.prefetch_factor, args.channels_last,
            args.max_val, args.seed,
            args.rebuild_cache, args.data_backend, args.hf_dataset_id,
            args.hf_revision, args.hf_token_env,
        )
        del frozen_backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_features, train_labels, _ = cache_paths(args.cache_dir, "train")
        val_features, val_labels, _ = cache_paths(args.cache_dir, "val")
        train_data = CachedFeatures(train_features, train_labels)
        val_data = CachedFeatures(val_features, val_labels)
        if train_data.features.shape[1] != backbone.output_dim:
            raise RuntimeError(
                f"cached feature dimension is {train_data.features.shape[1]}, "
                f"expected {backbone.output_dim} for {backbone.display_name}"
            )
        seed_all(args.seed)
        template = TopKSAE(backbone.output_dim, args.hidden_dim, args.dead_steps)
        template.pre_bias.data.copy_(estimate_feature_mean(train_data.features))
        initial_state = {key: value.clone() for key, value in template.state_dict().items()}
        del template

        for method, use_annealing in ((CSR_V1, False), (CSR, True)):
            if args.method not in (method, "all"):
                continue
            seed_all(args.seed)
            csr_model, csr_history, csr_schedule = train_csr(
                initial_state, train_data, device, args, method, use_annealing
            )
            csr_knn = benchmark_method(
                method, csr_model, train_data, val_data, device, args
            )
            schedule_name = "cosine_annealed" if use_annealing else "fixed_k"
            results[method] = {
                "display_name": labels[method],
                "backbone": backbone.name,
                "feature_dim": backbone.output_dim,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in csr_model.parameters()
                    if parameter.requires_grad
                ),
                "training_protocol": (
                    f"frozen_{backbone.name}_{schedule_name}_topk_sae_reconstruction_plus_ncl"
                ),
                "training_epochs": args.epochs,
                "topk_annealing": csr_schedule,
                "loss_weights": {
                    "main_reconstruction": args.csr_main_recon_weight,
                    "multi_topk_reconstruction": args.csr_multi_topk_recon_weight,
                    "auxiliary_reconstruction": args.csr_aux_recon_weight,
                    "nonnegative_contrastive": args.csr_contrastive_weight,
                },
                "history": csr_history,
                "knn": csr_knn,
            }
            dataset_metadata[method] = {
                "train": frozen_train_meta, "validation": frozen_val_meta
            }
            atomic_json(results[method], args.output_dir / method / "results.json")
            del csr_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for method, use_annealing in ((MP_SAE_V1, False), (MP_SAE, True)):
            if args.method not in (method, "all"):
                continue
            seed_all(args.seed)
            mp_model, mp_history, mp_schedule = train_mp_sae(
                initial_state, train_data, device, args, method, use_annealing
            )
            mp_knn = benchmark_method(
                method, mp_model, train_data, val_data, device, args
            )
            schedule_name = "cosine_annealed" if use_annealing else "fixed_k"
            results[method] = {
                "display_name": labels[method],
                "backbone": backbone.name,
                "feature_dim": backbone.output_dim,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in mp_model.parameters()
                    if parameter.requires_grad
                ),
                "training_protocol": (
                    f"frozen_{backbone.name}_{schedule_name}_topk_sae_plus_mmpot"
                ),
                "training_epochs": args.epochs + args.mpsae_extra_epochs,
                "topk_annealing": mp_schedule,
                "loss_weights": {
                    "main_reconstruction": args.mpsae_main_recon_weight,
                    "nested_reconstruction": args.mpsae_nested_recon_weight,
                    "auxiliary_reconstruction": args.mpsae_aux_recon_weight,
                    "mmpot_regularizer": args.mpsae_mmpot_weight,
                },
                "history": mp_history,
                "knn": mp_knn,
            }
            dataset_metadata[method] = {
                "train": frozen_train_meta, "validation": frozen_val_meta
            }
            atomic_json(results[method], args.output_dir / method / "results.json")
            del mp_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del train_data, val_data, initial_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "experiment": f"Matryoshka_vs_CSR_v1_v2_vs_MPSAE_v1_v2_{backbone.name}",
        "study_role": "architecture_ablation_backbone_unit",
        "ablation_variable": "backbone",
        "backbone": {
            "name": backbone.name,
            "display_name": backbone.display_name,
            "feature_dim": backbone.output_dim,
            "weights": backbone.weights_id,
        },
        "method_labels": labels,
        "comparison_protocol": {
            "dataset": "ImageNet-1K",
            "metric": "unit_normalized_exact_L2_1NN_top1",
            "gallery": "training_split",
            "queries": "validation_split",
            "unit_normalized": True,
            "retrieval_timing_scope": (
                "search_only_excludes_encoding_normalization_and_index_construction"
            ),
            "retrieval_backends": {
                MATRYOSHKA: "FAISS IndexFlatL2 dense exact search",
                **{
                    method: "SciPy chunked CSR sparse exact search"
                    for method in SPARSE_METHODS
                },
            },
            "evaluation_budgets": {
                MATRYOSHKA: list(args.topk),
                **{
                    method: sorted(set((*args.sparse_extra_topk, *args.topk)))
                    for method in SPARSE_METHODS
                },
            },
            "budget_definition": {
                MATRYOSHKA: f"{backbone.display_name} feature-prefix dimension",
                **{
                    method: "number of retained Top-K sparse latents at evaluation"
                    for method in SPARSE_METHODS
                },
            },
            "training_epochs": {
                MATRYOSHKA: args.epochs,
                CSR_V1: args.epochs,
                CSR: args.epochs,
                MP_SAE_V1: args.epochs + args.mpsae_extra_epochs,
                MP_SAE: args.epochs + args.mpsae_extra_epochs,
            },
            "topk_annealing": {
                method: results.get(method, {}).get("topk_annealing")
                for method in SPARSE_METHODS
            },
            "model_weights_saved": False,
            "loss_weights": {
                MATRYOSHKA: {
                    "classification": args.mrl_classification_weight,
                },
                **{
                    method: {
                        "main_reconstruction": args.csr_main_recon_weight,
                        "multi_topk_reconstruction": args.csr_multi_topk_recon_weight,
                        "auxiliary_reconstruction": args.csr_aux_recon_weight,
                        "nonnegative_contrastive": args.csr_contrastive_weight,
                    }
                    for method in (CSR_V1, CSR)
                },
                **{
                    method: {
                        "main_reconstruction": args.mpsae_main_recon_weight,
                        "nested_reconstruction": args.mpsae_nested_recon_weight,
                        "auxiliary_reconstruction": args.mpsae_aux_recon_weight,
                        "mmpot_regularizer": args.mpsae_mmpot_weight,
                    }
                    for method in (MP_SAE_V1, MP_SAE)
                },
            },
        },
        "dataset": dataset_metadata,
        "config": serializable_args(args),
        "results": results,
    }
    rows = comparison_rows(results)
    write_comparison_csv(rows, args.output_dir / "comparison.csv")
    write_markdown_table(rows, args.output_dir / "comparison_table.md", backbone)
    write_latex_table(rows, args.output_dir / "comparison_table.tex", backbone)
    plot_publication_comparison(results, args.output_dir)
    plot_retrieval_time(results, args.output_dir)
    plot_training_diagnostics(results, args.output_dir)
    plot_topk_annealing(results, args.output_dir, backbone)
    loss_records, loss_impacts = write_training_loss_analysis(
        results, args, args.output_dir, backbone
    )
    plot_training_procedure(results, loss_impacts, args.output_dir, backbone)
    artifact_prefix = f"ablation_{backbone.name}"
    summary["training_loss_analysis"] = {
        "impact_definition": (
            "Measured weighted contribution to the optimized objective and "
            "observed start-to-finish decrease; not a causal ablation estimate."
        ),
        "epoch_component_records": len(loss_records),
        "component_impact_records": len(loss_impacts),
        "history_csv": f"{artifact_prefix}_training_loss_history.csv",
        "impact_csv": f"{artifact_prefix}_loss_component_impact.csv",
        "impact_json": f"{artifact_prefix}_loss_component_impact.json",
        "loss_curves": [
            f"{artifact_prefix}_training_loss_curves.png",
            f"{artifact_prefix}_training_loss_curves.pdf",
        ],
        "training_procedure_plots": [
            f"{artifact_prefix}_training_procedure_overview.png",
            f"{artifact_prefix}_training_procedure_overview.pdf",
            f"{artifact_prefix}_loss_component_impact.png",
            f"{artifact_prefix}_loss_component_impact.pdf",
            f"{artifact_prefix}_topk_annealing_schedule.png",
            f"{artifact_prefix}_topk_annealing_schedule.pdf",
        ],
    }
    summary["retrieval_analysis"] = {
        "timing_scope": "search_only_excludes_encoding_normalization_and_index_construction",
        "unit_normalized": True,
        "accuracy_plots": [
            f"{artifact_prefix}_representation_accuracy_comparison.png",
            f"{artifact_prefix}_representation_accuracy_comparison.pdf",
        ],
        "timing_plots": [
            f"{artifact_prefix}_retrieval_time_comparison.png",
            f"{artifact_prefix}_retrieval_time_comparison.pdf",
        ],
    }
    summary["runtime"] = runtime_metadata(
        device, run_started_at, time.time() - run_started
    )
    atomic_json(summary, args.output_dir / "summary.json")
    update_wandb_summary(
        {"runtime": summary["runtime"], "results": results}
    )
    finish_wandb()
    print(f"complete: {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
