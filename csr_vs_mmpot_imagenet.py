#!/usr/bin/env python3
"""ImageNet comparison: paper-faithful FF2048, ResNet-18, and ablations.

The default paper_ff2048 protocol reproduces the visual setting in
Wen et al. (2025): it consumes the authors' released FF2048 ResNet-50
embeddings, reports the published MRL ResNet-50 curve, and trains one CSR or
MPSAE sparse autoencoder independently for every reported K. This distinction
is important: FF2048 is a particular ResNet-50 checkpoint released with MRL;
it is not torchvision's ImageNet-1K V1/V2 ResNet-50, and no ResNet-18 FF2048
or MRL baseline is reported in either paper.

The separate paper_rn18_study protocol follows the CSR paper's Appendix E.1
ResNet-18 backbone study and extends it with a matched MPSAE arm. It uses the
paper's timm ResNet-18 model and reports a measured dense FF512 reference.
Feature extraction is a documented torchvision translation of the released
FFCV pipeline, not a bit-for-bit replay. It does not fabricate an MRL result,
because neither cited paper reports a matched ResNet-18 MRL model. The former
torchvision ResNet-18/50 experiment remains available as the explicitly
non-reproduction torchvision_ablation protocol.

Pipeline
--------
1. Load the released FF2048 train/validation embeddings (default), or cache
   torchvision features only in the explicitly selected ablation protocol.
2. Use the published MRL 1-NN curve in the paper protocol; the legacy local
   fine-tuning implementation is never presented as a paper reproduction.
3. Train fixed-K CSR v1 and MPSAE v1 sparse autoencoders on the same frozen
   features, with a separate model for each K in paper mode.
5. Train CSRv2 and MPSAEv2 with the corresponding objectives and progressive
   cosine Top-K curricula. Every objective coefficient is explicit and
   configurable while retaining paper-aligned defaults.
6. Encode the train split as the gallery and validation split as queries.
7. Unit-normalize every evaluated embedding and run exact L2 1-NN:
   dense embeddings use GPU FAISS while sparse arms use CUDA feature postings
   (exhaustive sparse dot products, equivalent to L2 ranking after normalization).
8. Record repeated search-only CUDA and wall timings and save histories, JSON results,
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

Dependencies: torch, torchvision, timm, numpy, GPU FAISS, and a CUDA compiler.
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

from retrieval_timing import CudaSearchTimer, TIMING_SCOPE
from sparse_gpu_retrieval import (
    CudaSparseIndex, check_cuda_memory_budget, preload_sparse_retrieval,
)

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
DEFAULT_MPSAE_MMPOT_WEIGHT = 2.0
IMAGENET_CLASSES = 1000
PAPER_FF2048_PROTOCOL = "paper_ff2048"
PAPER_RN18_STUDY_PROTOCOL = "paper_rn18_study"
TORCHVISION_ABLATION_PROTOCOL = "torchvision_ablation"
PAPER_FIXED_K_PROTOCOLS = (PAPER_FF2048_PROTOCOL, PAPER_RN18_STUDY_PROTOCOL)
PROTOCOLS = (*PAPER_FIXED_K_PROTOCOLS, TORCHVISION_ABLATION_PROTOCOL)
PAPER_FF2048_BACKBONE_ID = "RAIVNLab/MRL:r50_mrl0_e0_ff2048.pt"
PAPER_FF2048_DATASET_ID = "W1nd-navigator/CSR-precompute-embeds"
PAPER_FF2048_TRAIN_SAMPLES = 1_281_167
PAPER_FF2048_VAL_SAMPLES = 50_000
PAPER_FF2048_BUDGETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
PAPER_RN18_MODEL_NAME = "resnet18.a1_in1k"
PAPER_RN18_BACKBONE_ID = "timm/resnet18.a1_in1k"
PAPER_RN18_BUDGETS = (8, 16, 64, 256)
RN18_STUDY_SEEDS = (1227, 1228, 1229)
PAPER_MRL_1NN_TOP1 = {
    8: 62.19,
    16: 67.91,
    32: 69.46,
    64: 70.17,
    128: 70.52,
    256: 70.62,
    512: 70.82,
    1024: 70.89,
    2048: 70.97,
}
PAPER_CSR_1NN_TOP1 = {
    8: 67.78,
    16: 69.17,
    32: 70.15,
    64: 70.94,
    128: 70.99,
    256: 71.31,
    512: 71.29,
    1024: 71.30,
    2048: 71.20,
}
PAPER_FF2048_FULL_1NN_TOP1 = 71.19
PAPER_FF2048_INTEGRITY_TOLERANCE_PP = 0.05
DENSE = "dense"
DEFAULT_EVALUATION_BUDGETS = {
    "resnet18": (8, 16, 32, 64, 128, 256, 512),
    "resnet50": (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
}
DEFAULT_SPARSE_EXTRA_BUDGETS: Tuple[int, ...] = ()


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


def experiment_backbone_weights(
    protocol: str, backbone: BackboneSpec
) -> str:
    if protocol == PAPER_FF2048_PROTOCOL:
        return PAPER_FF2048_BACKBONE_ID
    if protocol == PAPER_RN18_STUDY_PROTOCOL:
        return PAPER_RN18_BACKBONE_ID
    return backbone.weights_id


def selected_methods(args: argparse.Namespace) -> Tuple[str, ...]:
    """Resolve the all shorthand without implying a ResNet-18 MRL arm."""
    if args.method != "all":
        return (args.method,)
    if args.protocol == PAPER_RN18_STUDY_PROTOCOL:
        return (CSR_V1, MP_SAE_V1)
    return METHODS


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
    data.add_argument(
        "--protocol", choices=PROTOCOLS, default=PAPER_FF2048_PROTOCOL,
        help=(
            "paper_ff2048 uses the authors' released ResNet-50 FF2048 "
            "embeddings; paper_rn18_study follows CSR Appendix E.1 with "
            "timm ResNet-18; torchvision_ablation retains the old non-reproduction study"
        ),
    )
    data.add_argument(
        "--data-root", type=Path, required=True,
        help=(
            "directory containing FF2048_RN50_Embeds in paper mode, or the "
            "ImageFolder/Hugging Face cache root in ResNet-18/ablation modes"
        ),
    )
    data.add_argument("--data-backend", choices=("imagefolder", "hf"), default="imagefolder")
    data.add_argument(
        "--backbone", choices=BACKBONES, default="resnet50",
        help="paper_ff2048 requires ResNet-50; paper_rn18_study requires ResNet-18",
    )
    data.add_argument(
        "--ff2048-train-embeddings", type=Path, default=None,
        help="official 1K_train_ff2048.npz (inferred below data-root when omitted)",
    )
    data.add_argument(
        "--ff2048-val-embeddings", type=Path, default=None,
        help="official 1K_val_ff2048.npz (inferred below data-root when omitted)",
    )
    data.add_argument(
        "--paper-strict", action=argparse.BooleanOptionalAction, default=True,
        help="reject settings or sample counts outside the selected paper protocol",
    )
    data.add_argument("--hf-dataset-id", default="ILSVRC/imagenet-1k")
    data.add_argument("--hf-revision", default="main")
    data.add_argument("--hf-token-env", default="HF_TOKEN")
    data.add_argument("--cache-dir", type=Path, default=Path("runs/three_method_ablation/cache"))
    data.add_argument("--weights-cache", type=Path, default=Path("weights"), help="local TORCH_HOME for pretrained weights")
    data.add_argument("--rebuild-cache", action="store_true")
    data.add_argument(
        "--feature-batch-size", type=int, default=None,
        help=(
            "frozen-feature extraction batch size; the ResNet-18 study "
            "defaults to the released extractor's batch size of 256"
        ),
    )
    data.add_argument("--workers", type=int, default=8)
    data.add_argument("--prefetch-factor", type=int, default=2, help="batches prefetched by each data-loading worker")
    data.add_argument("--max-train", type=int, default=0, help="deterministic subset; 0 uses all training images")
    data.add_argument("--max-val", type=int, default=0, help="deterministic subset; 0 uses all validation images")

    model = p.add_argument_group("sparse representations")
    model.add_argument(
        "--hidden-dim", type=int, default=8192,
        help=(
            "CSR/MPSAE latent dimension; both paper protocols use h=8192 "
            "(4d for FF2048, 16d for ResNet-18)"
        ),
    )
    model.add_argument(
        "--topk", type=parse_ints, default=None,
        help=(
            "matched budgets; Appendix E.1 ResNet-18 defaults to 8,16,64,256 "
            "and FF2048 defaults to every power of two from 8 through 2048"
        ),
    )
    model.add_argument(
        "--sparse-extra-topk", type=parse_ints,
        default=list(DEFAULT_SPARSE_EXTRA_BUDGETS),
        help="additional lower budgets evaluated for every CSR/MPSAE sparse arm",
    )
    model.add_argument("--train-k", type=int, default=2)
    model.add_argument(
        "--v1-train-k", type=int, default=32,
        help=(
            "legacy single-model fixed support and representative history K; "
            "paper mode trains a separate v1 model at every reported K"
        ),
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
    model.add_argument(
        "--dead-steps", type=int, default=30,
        help="official CSR dead-latent threshold in Top-K forward-counter updates",
    )

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
        dest="csr_contrastive_weight", type=float, default=None,
        help="gamma multiplying NCL (FF2048: 0.1; Appendix E.1 ResNet-18: 1.0)",
    )
    loss.add_argument(
        "--mpsaev2-main-recon-weight", "--mpsae-main-recon-weight",
        dest="mpsae_main_recon_weight", type=float, default=1.0,
        help="weight on MPSAEv2's annealed Top-K reconstruction loss",
    )
    loss.add_argument(
        "--mpsaev2-nested-recon-weight", "--mpsae-nested-recon-weight",
        dest="mpsae_nested_recon_weight", type=float, default=1.0 / 8.0,
        help="weight on MPSAE's Top-4K reconstruction loss; Top-2K is used only by MMPOT",
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
        dest="mpsae_extra_epochs", type=int, default=0,
        help="optional non-paper ablation; zero keeps CSR and MPSAE training budgets matched",
    )
    train.add_argument(
        "--batch-size", type=int, default=None,
        help="sparse-training batch size (FF2048: 4096; ResNet-18 study: 1024)",
    )
    train.add_argument("--mpsaev2-lr", "--lr", dest="lr", type=float, default=4e-5)
    train.add_argument("--csrv2-lr", "--csr-lr", dest="csr_lr", type=float, default=4e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--adam-eps", type=float, default=6.25e-10)
    train.add_argument(
        "--contrastive-temperature", type=float, default=0.2,
        help="temperature used by the released CSR implementation",
    )
    train.add_argument("--mrl-lr", type=float, default=1e-2, help="backbone MRL fine-tuning learning rate")
    train.add_argument("--mrl-momentum", type=float, default=0.9)
    train.add_argument("--ot-mass", type=float, default=0.9)
    train.add_argument("--ot-eta", type=float, default=0.2, help="M3G recommendation for circular variance")
    train.add_argument("--ot-iters", type=int, default=100)
    train.add_argument("--ot-tol", type=float, default=1e-4)
    train.add_argument("--ot-microbatch", type=int, default=32, help="B^3 cost makes small OT groups necessary")
    train.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False,
        help="the released CSR visual implementation trains in float32",
    )
    train.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "allow reduced-precision TF32 CUDA matrix operations; disabled by "
            "the strict paper protocol"
        ),
    )
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=1227)
    train.add_argument(
        "--study-seeds", type=parse_ints, default=None,
        help=(
            "independent sparse-model seeds; Appendix E.1 reports a "
            "three-seed mean but does not publish the seed values"
        ),
    )
    train.add_argument("--print-freq", type=int, default=50)
    train.add_argument("--output-dir", type=Path, default=Path("runs/three_method_ablation"))

    knn = p.add_argument_group("exact dense/sparse 1-NN")
    knn.add_argument("--knn-batch-size", type=int, default=4096)
    knn.add_argument("--knn-query-batch", type=int, default=32)
    knn.add_argument(
        "--sparse-knn-query-batch", type=int, default=32,
        help="CUDA sparse query batch; must match --knn-query-batch",
    )
    knn.add_argument("--sparse-gallery-chunk", type=int, default=65536,
                     help="maximum rows per GPU posting chunk; also bounded by build/score memory")
    knn.add_argument("--retrieval-warmup", type=int, default=3,
                     help="untimed searches per distinct query shape, including the tail batch")
    knn.add_argument("--retrieval-repeats", type=int, default=5,
                     help="timed searches of each resident query batch")
    knn.add_argument("--sparse-score-buffer-mib", type=int, default=256)
    knn.add_argument("--sparse-gpu-memory-fraction", type=float, default=0.8,
                     help="resident index/build/workspace memory guard as a fraction of GPU capacity")
    knn.add_argument(
        "--knn-normalize", action=argparse.BooleanOptionalAction, default=True,
        help="unit-normalize gallery and query embeddings (required by this study)",
    )
    knn.add_argument(
        "--faiss-gpu", action=argparse.BooleanOptionalAction, default=True,
        help="required: both dense and sparse retrieval run on the same CUDA device",
    )
    knn.add_argument("--faiss-gpu-device", type=int, default=None,
                     help="shared CUDA retrieval device (defaults to the model CUDA device, otherwise 0)")
    knn.add_argument("--faiss-temp-memory-mib", type=int, default=512,
                     help="temporary GPU memory reserved by FAISS; 0 disables its allocation stack")
    add_wandb_arguments(p)
    return p


def validate_args(a: argparse.Namespace) -> None:
    backbone = BACKBONE_SPECS[a.backbone]
    if a.feature_batch_size is None:
        a.feature_batch_size = (
            256 if a.protocol == PAPER_RN18_STUDY_PROTOCOL else 512
        )
    if a.batch_size is None:
        a.batch_size = (
            1024 if a.protocol == PAPER_RN18_STUDY_PROTOCOL else 4096
        )
    if a.csr_contrastive_weight is None:
        a.csr_contrastive_weight = (
            1.0 if a.protocol == PAPER_RN18_STUDY_PROTOCOL else 0.1
        )
    if a.study_seeds is None:
        a.study_seeds = (
            list(RN18_STUDY_SEEDS)
            if a.protocol == PAPER_RN18_STUDY_PROTOCOL
            else [a.seed]
        )
    if a.topk is None:
        if a.protocol == PAPER_FF2048_PROTOCOL:
            a.topk = list(PAPER_FF2048_BUDGETS)
        elif a.protocol == PAPER_RN18_STUDY_PROTOCOL:
            a.topk = list(PAPER_RN18_BUDGETS)
        else:
            a.topk = list(DEFAULT_EVALUATION_BUDGETS[a.backbone])
    if a.protocol == PAPER_FF2048_PROTOCOL:
        if a.backbone != "resnet50":
            raise ValueError(
                "paper_ff2048 is ResNet-50 only: neither paper reports an "
                "FF2048 or MRL ResNet-18 baseline"
            )
        default_embed_dir = a.data_root / "FF2048_RN50_Embeds"
        if a.ff2048_train_embeddings is None:
            a.ff2048_train_embeddings = default_embed_dir / "1K_train_ff2048.npz"
        if a.ff2048_val_embeddings is None:
            a.ff2048_val_embeddings = default_embed_dir / "1K_val_ff2048.npz"
        for description, path in (
            ("FF2048 training embeddings", a.ff2048_train_embeddings),
            ("FF2048 validation embeddings", a.ff2048_val_embeddings),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing official {description}: {path}. Download it from "
                    f"https://huggingface.co/datasets/{PAPER_FF2048_DATASET_ID}"
                )
    else:
        if a.protocol == PAPER_RN18_STUDY_PROTOCOL:
            if a.backbone != "resnet18":
                raise ValueError(
                    "paper_rn18_study requires backbone=resnet18 and the "
                    "timm/resnet18.a1_in1k weights used in CSR Appendix E.1"
                )
            if a.method == MATRYOSHKA:
                raise ValueError(
                    "paper_rn18_study has no matched published MRL baseline; "
                    "use CSR/MPSAE or the explicitly non-paper torchvision ablation"
                )
            if a.paper_strict and a.method not in ("all", CSR_V1, MP_SAE_V1):
                raise ValueError(
                    "strict paper_rn18_study supports only fixed-K CSR and "
                    "matched fixed-K MPSAE; annealed v2 arms require --no-paper-strict"
                )
        if a.data_backend == "imagefolder":
            for split in ("train", "val"):
                if not (a.data_root / split).is_dir():
                    raise FileNotFoundError(
                        f"missing ImageNet directory: {a.data_root / split}"
                    )
        elif not a.data_root.is_dir():
            raise FileNotFoundError(
                f"missing Hugging Face cache directory: {a.data_root}"
            )
    if MATRYOSHKA in selected_methods(a) and max(a.topk) > backbone.output_dim:
        raise ValueError(
            f"Matryoshka prefix dimensions cannot exceed the {backbone.display_name} "
            f"feature dimension ({backbone.output_dim})"
        )
    if a.hidden_dim < 0:
        raise ValueError("hidden-dim must be positive, or 0 to select 4 * input dimension")
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
        a.hidden_dim = 4 * backbone.output_dim
    enabled_sparse_methods = set(selected_methods(a)).intersection(SPARSE_METHODS)
    if enabled_sparse_methods:
        required_hidden_dim = max((*a.topk, *a.sparse_extra_topk))
        if enabled_sparse_methods.intersection((CSR, MP_SAE)):
            required_hidden_dim = max(
                required_hidden_dim, 4 * a.anneal_start_k
            )
        if enabled_sparse_methods.intersection((CSR_V1, MP_SAE_V1)):
            fixed_training_k = (
                max(a.topk)
                if a.protocol in PAPER_FIXED_K_PROTOCOLS
                else a.v1_train_k
            )
            required_hidden_dim = max(
                required_hidden_dim, 4 * fixed_training_k
            )
        if a.hidden_dim < required_hidden_dim:
            raise ValueError(
                "hidden-dim must cover every evaluated sparse budget and "
                f"every 4K training view (requires >= {required_hidden_dim})"
            )
    if not 0.0 < a.ot_mass <= 1.0:
        raise ValueError("ot-mass must be in (0,1]")
    if a.ot_eta <= 0 or a.ot_iters < 1 or a.ot_tol <= 0:
        raise ValueError("ot-eta/ot-tol must be positive and ot-iters must be at least 1")
    if min(a.epochs, a.batch_size, a.feature_batch_size, a.ot_microbatch, a.prefetch_factor) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if a.mpsae_extra_epochs < 0:
        raise ValueError("mpsaev2-extra-epochs must be non-negative")
    if a.csr_lr <= 0:
        raise ValueError("CSR-family lr must be positive")
    if a.ot_microbatch < 2:
        raise ValueError("ot-microbatch must be at least 2")
    if a.faiss_gpu_device is not None and a.faiss_gpu_device < 0:
        raise ValueError("faiss-gpu-device must be non-negative")
    if a.faiss_temp_memory_mib < 0:
        raise ValueError("faiss-temp-memory-mib must be non-negative")
    if not a.faiss_gpu:
        raise ValueError("this GPU retrieval study requires --faiss-gpu; CPU timing is not comparable")
    if a.knn_query_batch != a.sparse_knn_query_batch:
        raise ValueError("dense and sparse retrieval must use the same query batch size")
    if min(a.sparse_gallery_chunk, a.retrieval_warmup, a.retrieval_repeats,
           a.sparse_score_buffer_mib) < 1:
        raise ValueError("retrieval chunk, warmup, repeats, and score-buffer MiB must be positive")
    if not math.isfinite(a.sparse_gpu_memory_fraction) or not 0 < a.sparse_gpu_memory_fraction <= 1:
        raise ValueError("sparse-gpu-memory-fraction must be in (0, 1]")
    if a.knn_query_batch * 4 > a.sparse_score_buffer_mib * 1024**2:
        raise ValueError("score buffer cannot hold even one gallery row for this query batch")
    if a.mrl_lr <= 0 or not 0 <= a.mrl_momentum < 1:
        raise ValueError("mrl-lr must be positive and mrl-momentum must be in [0,1)")
    if a.lr <= 0 or a.weight_decay < 0 or a.adam_eps <= 0:
        raise ValueError(
            "MPSAE-family lr/adam-eps must be positive and weight-decay must be non-negative"
        )
    if a.contrastive_temperature <= 0:
        raise ValueError("contrastive-temperature must be positive")
    if a.knn_batch_size < 1 or a.knn_query_batch < 1 or a.sparse_knn_query_batch < 1:
        raise ValueError("1-NN batch sizes must be positive")
    if not a.knn_normalize:
        raise ValueError("knn-normalize must remain enabled for the controlled experiment")
    if a.max_train < 0 or a.max_val < 0:
        raise ValueError("max-train/max-val must be non-negative")
    if a.prefetch_factor < 1 or a.workers < 0 or a.print_freq < 0:
        raise ValueError("prefetch-factor must be positive; workers/print-freq must be non-negative")
    if a.protocol in PAPER_FIXED_K_PROTOCOLS and a.paper_strict:
        expected_batch_size = (
            1024 if a.protocol == PAPER_RN18_STUDY_PROTOCOL else 4096
        )
        expected_gamma = (
            1.0 if a.protocol == PAPER_RN18_STUDY_PROTOCOL else 0.1
        )
        expected_budgets = (
            PAPER_RN18_BUDGETS
            if a.protocol == PAPER_RN18_STUDY_PROTOCOL
            else PAPER_FF2048_BUDGETS
        )
        expected_seeds = (
            RN18_STUDY_SEEDS
            if a.protocol == PAPER_RN18_STUDY_PROTOCOL
            else (1227,)
        )
        exact_settings = {
            "hidden_dim": (a.hidden_dim, 8192),
            "epochs": (a.epochs, 10),
            "batch_size": (a.batch_size, expected_batch_size),
            "k_aux": (a.k_aux, 512),
            "dead_steps": (a.dead_steps, 30),
            "csr_lr": (a.csr_lr, 4e-5),
            "mpsaev2_lr": (a.lr, 4e-5),
            "weight_decay": (a.weight_decay, 1e-4),
            "adam_eps": (a.adam_eps, 6.25e-10),
            "csr_main_recon_weight": (a.csr_main_recon_weight, 1.0),
            "csr_multi_topk_recon_weight": (a.csr_multi_topk_recon_weight, 1.0 / 8.0),
            "csr_aux_recon_weight": (a.csr_aux_recon_weight, 1.0 / 32.0),
            "csr_contrastive_weight": (a.csr_contrastive_weight, expected_gamma),
            "mpsae_main_recon_weight": (a.mpsae_main_recon_weight, 1.0),
            "mpsae_nested_recon_weight": (a.mpsae_nested_recon_weight, 1.0 / 8.0),
            "mpsae_aux_recon_weight": (a.mpsae_aux_recon_weight, 1.0 / 32.0),
            "contrastive_temperature": (a.contrastive_temperature, 0.2),
            "mpsae_extra_epochs": (a.mpsae_extra_epochs, 0),
            "max_train": (a.max_train, 0),
            "max_val": (a.max_val, 0),
            "seed": (a.seed, 1227),
            "study_seeds (predeclared; paper values are unpublished)": (
                tuple(a.study_seeds), expected_seeds
            ),
            "amp": (a.amp, False),
            "tf32": (a.tf32, False),
        }
        if a.protocol == PAPER_RN18_STUDY_PROTOCOL:
            # This is a released-extractor control rather than a Table 9 field.
            exact_settings["feature_batch_size"] = (
                a.feature_batch_size, 256
            )
        deviations = [
            f"{name}={actual!r} (required: {expected!r})"
            for name, (actual, expected) in exact_settings.items()
            if actual != expected
        ]
        if list(a.topk) != list(expected_budgets):
            deviations.append(
                f"topk={a.topk!r} (required: {list(expected_budgets)!r})"
            )
        if a.sparse_extra_topk:
            deviations.append(
                f"sparse_extra_topk={a.sparse_extra_topk!r} (required: [])"
            )
        if deviations:
            raise ValueError(
                f"{a.protocol} strict mode rejected protocol deviations: "
                + "; ".join(deviations)
                + ". Pass --no-paper-strict only for a clearly labelled ablation."
            )


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


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
        torch.backends.cudnn.deterministic = (
            args.protocol in PAPER_FIXED_K_PROTOCOLS and args.paper_strict
        )
        torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic
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


def build_feature_network(
    backbone: BackboneSpec, weights_cache: Path, protocol: str
) -> nn.Module:
    """Load the protocol-specific frozen feature network."""
    os.environ["TORCH_HOME"] = str(weights_cache.expanduser().resolve())
    if protocol == PAPER_RN18_STUDY_PROTOCOL:
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "paper_rn18_study requires timm; install requirements.txt"
            ) from exc
        network = timm.create_model(
            PAPER_RN18_MODEL_NAME, pretrained=True, num_classes=0
        )
        return network
    network = backbone.constructor(weights=backbone.weights)
    setattr(network, backbone.classifier_attribute, nn.Identity())
    return network


class FFCVCenterCropTranslation:
    """Translate FFCV's ratio crop followed by resize for PIL inputs."""

    def __init__(self, output_size: int = 256, crop_ratio: float = 224 / 256) -> None:
        self.output_size = int(output_size)
        self.crop_ratio = float(crop_ratio)

    def __call__(self, image: Any) -> Any:
        width, height = transforms.functional.get_image_size(image)
        crop_size = int(self.crop_ratio * min(height, width))
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        image = transforms.functional.crop(
            image, top, left, crop_size, crop_size
        )
        return transforms.functional.resize(
            image,
            [self.output_size, self.output_size],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"crop_ratio={self.crop_ratio:g})"
        )


def csr_rn18_feature_transforms() -> Tuple[Any, Any]:
    """Semantic torchvision translation of the released CSR FFCV pipeline."""
    normalization = transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization,
    ])
    evaluation_transform = transforms.Compose([
        # Released code uses CenterCropRGBImageDecoder((256, 256),
        # ratio=224/256): crop first, then resize the crop to 256 square.
        FFCVCenterCropTranslation(output_size=256, crop_ratio=224 / 256),
        transforms.ToTensor(),
        normalization,
    ])
    return train_transform, evaluation_transform


class FrozenBackbone(nn.Module):
    def __init__(
        self, weights_cache: Path, backbone: BackboneSpec, protocol: str
    ) -> None:
        super().__init__()
        network = build_feature_network(backbone, weights_cache, protocol)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self.network = network
        if protocol == PAPER_RN18_STUDY_PROTOCOL:
            self.train_transform, self.evaluation_transform = (
                csr_rn18_feature_transforms()
            )
            self.weights_id = PAPER_RN18_BACKBONE_ID
            self.preprocessing_id = (
                "CSR_official_FFCV_translation_RRC224_flip_train_"
                "center_ratio224of256_resize256_val_mean0.5_std0.5_"
                "semantic_not_bit_exact"
            )
            self.preprocessing_equivalence = (
                "semantic_torchvision_translation_of_released_FFCV_"
                "pipeline_not_bit_exact"
            )
        else:
            self.train_transform = backbone.weights.transforms()
            self.evaluation_transform = backbone.weights.transforms()
            self.weights_id = backbone.weights_id
            self.preprocessing_id = backbone.weights_id + ".transforms"
            self.preprocessing_equivalence = "native_torchvision_weight_transform"
        self.output_dim = backbone.output_dim
        self.backbone = backbone
        self.protocol = protocol

    def transform_for_split(self, split: str) -> Any:
        return (
            self.train_transform if split == "train" else self.evaluation_transform
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class MatryoshkaBackbone(nn.Module):
    """A torchvision backbone trained with classifiers on nested prefixes."""

    def __init__(
        self, weights_cache: Path, backbone: BackboneSpec, nested_dims: Sequence[int]
    ) -> None:
        super().__init__()
        self.network = build_feature_network(
            backbone, weights_cache, TORCHVISION_ABLATION_PROTOCOL
        )
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
        "cache_format": 3,
        "split": split,
        "feature_dim": backbone.output_dim,
        "dtype": "float16",
        "backbone": backbone.backbone.name,
        "weights": backbone.weights_id,
        "preprocessing": backbone.preprocessing_id,
        "preprocessing_equivalence": backbone.preprocessing_equivalence,
        "feature_extraction_batch_size": batch_size,
        "feature_extraction_workers": workers,
        "feature_extraction_prefetch_factor": (
            prefetch_factor if workers > 0 else None
        ),
        "feature_extraction_channels_last": channels_last,
        "feature_extraction_compute": (
            "cuda_autocast" if device.type == "cuda" else "cpu_float32"
        ),
        "local_feature_extractor_drop_last": False,
        "protocol": backbone.protocol,
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
        split,
        root,
        backbone.transform_for_split(split),
        data_backend,
        hf_dataset_id,
        hf_revision,
        hf_token_env,
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


class ReleasedNpzFeatures(Dataset):
    """The CSR authors' released FF2048 NPZ format (keys: data, label)."""

    LABEL_KEYS = ("label", "labels", "target", "targets")

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.archive = np.load(self.path, allow_pickle=False)
        if not isinstance(self.archive, np.lib.npyio.NpzFile):
            raise ValueError(f"expected an NPZ archive, got: {self.path}")
        if "data" not in self.archive.files:
            raise ValueError(f"{self.path} has no 'data' embedding array")
        label_key = next(
            (key for key in self.LABEL_KEYS if key in self.archive.files), None
        )
        if label_key is None:
            raise ValueError(
                f"{self.path} has none of the supported label keys: "
                + ", ".join(self.LABEL_KEYS)
            )
        self.features = self.archive["data"]
        self.labels = self.archive[label_key]
        if self.features.ndim != 2:
            raise ValueError(
                f"{self.path} embeddings must be rank two, got {self.features.shape}"
            )
        if self.labels.ndim != 1:
            self.labels = self.labels.reshape(-1)
        if len(self.features) != len(self.labels):
            raise RuntimeError(
                f"{self.path} has {len(self.features)} embeddings but "
                f"{len(self.labels)} labels"
            )
        if not np.issubdtype(self.features.dtype, np.floating):
            raise TypeError(
                f"{self.path} embeddings must be floating point, got {self.features.dtype}"
            )
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise TypeError(
                f"{self.path} labels must be integer-valued, got {self.labels.dtype}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        feature = np.array(self.features[index], dtype=np.float32, copy=True)
        return torch.from_numpy(feature), int(self.labels[index])

    def __getitems__(self, indices: Sequence[int]) -> List[Tuple[torch.Tensor, int]]:
        feature_batch = torch.from_numpy(
            np.array(self.features[list(indices)], dtype=np.float32, copy=True)
        )
        label_batch = np.asarray(self.labels[list(indices)], dtype=np.int64)
        return [
            (feature_batch[offset], int(label_batch[offset]))
            for offset in range(len(indices))
        ]

    def metadata(self, split: str) -> Dict[str, Any]:
        return {
            "source": "authors_released_precomputed_embeddings",
            "dataset_id": PAPER_FF2048_DATASET_ID,
            "split": split,
            "path": str(self.path),
            "samples": len(self),
            "feature_dim": int(self.features.shape[1]),
            "stored_dtype": str(self.features.dtype),
            "training_dtype": "float32",
            "backbone_weights": PAPER_FF2048_BACKBONE_ID,
        }


def cached_feature_batches(
    dataset: Dataset, batch_size: int
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


class TopKSAE(nn.Module):
    """Paper-faithful tied Top-K sparse autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 8192, dead_steps: int = 30) -> None:
        super().__init__()
        decoder = torch.empty(hidden_dim, input_dim)
        # Match nn.Linear's initialization in the released CSR implementation.
        nn.init.kaiming_uniform_(decoder, a=math.sqrt(5))
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
            active = z.gt(1e-5).any(dim=0)
            # Released CSR resets active counters, then increments every
            # counter; an active latent therefore ends this call at one.
            self.inactive_steps.mul_((~active).to(self.inactive_steps.dtype))
            self.inactive_steps.add_(1)

    @staticmethod
    def normalized_auxiliary_mse(
        auxiliary_reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """OpenAI/CSR auxiliary MSE normalized by the mean-predictor MSE."""
        baseline = target.mean(dim=0, keepdim=True).expand_as(target)
        denominator = F.mse_loss(baseline, target)
        return (
            F.mse_loss(auxiliary_reconstruction, target) / denominator
        ).nan_to_num(0.0)

    def reconstruction_losses(
        self, x: torch.Tensor, k: int, k_aux: int, main_weight: float,
        nested_weight: float, aux_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pre = self.preactivations(x)
        z1 = self.keep_topk(pre, k)
        z2 = self.keep_topk(pre, min(2 * k, self.decoder.shape[0]))
        z4 = self.keep_topk(pre, min(4 * k, self.decoder.shape[0]))
        recon1 = self.decode(z1)
        recon4 = self.decode(z4)
        main = F.mse_loss(recon1, x)
        # Keep the CSR base objective identical across CSR and MPSAE. The 2K
        # view is an MMPOT view, not an extra reconstruction advantage.
        nested = F.mse_loss(recon4, x)

        self.update_activity(z1)
        dead = self.inactive_steps > self.dead_steps
        if k_aux > 0:
            # The released model multiplies non-dead preactivations by zero
            # and still evaluates AuxK every batch, including before any
            # latent has crossed the dead threshold.
            masked = pre * dead[None, :].to(pre.dtype)
            aux_z = self.keep_topk(
                masked, min(k_aux, self.decoder.shape[0])
            )
            residual_target = x - recon1.detach() + self.pre_bias.detach()
            aux = self.normalized_auxiliary_mse(
                self.decode(aux_z), residual_target
            )
        else:
            aux = main.new_zeros(())
        total = main_weight * main + nested_weight * nested + aux_weight * aux
        # CSR's released code invokes TopK again for the 4K reconstruction,
        # then a third time for its duplicate NCL view. MPSAE uses the 2K
        # transport view for that third counter update so both objectives
        # advance the released dead-latent statistic three times per batch.
        self.update_activity(z4)
        self.update_activity(z2)
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

        self.update_activity(z)
        dead = self.inactive_steps > self.dead_steps
        if k_aux > 0:
            masked = pre * dead[None, :].to(pre.dtype)
            aux_z = self.keep_topk(
                masked, min(k_aux, self.decoder.shape[0])
            )
            residual_target = x - reconstruction.detach() + self.pre_bias.detach()
            auxiliary = self.normalized_auxiliary_mse(
                self.decode(aux_z), residual_target
            )
        else:
            auxiliary = main.new_zeros(())
        total = main_weight * main + multi_weight * multi + aux_weight * auxiliary
        self.update_activity(z4)
        return total, {
            "recon": main,
            "multi_topk_recon": multi,
            "aux": auxiliary,
            "dead_fraction": dead.float().mean(),
        }, z


def initial_sae_state(
    input_dim: int, hidden_dim: int, dead_steps: int, seed: int
) -> Dict[str, torch.Tensor]:
    """Create a repeatable initialization shared by CSR and MPSAE."""
    seed_all(seed)
    template = TopKSAE(input_dim, hidden_dim, dead_steps)
    state = {
        key: value.clone() for key, value in template.state_dict().items()
    }
    del template
    return state


def nonnegative_contrastive_loss(
    representations: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """Released CSR two-view NCL, including L2 normalization and temperature."""
    batch = representations.shape[0]
    identities = torch.arange(batch, device=representations.device).repeat(2)
    codes = F.normalize(
        torch.cat((representations.float(), representations.float()), dim=0),
        dim=-1,
    )
    logits = (codes @ codes.T) / temperature
    same_sample = identities[:, None].eq(identities[None, :])
    same_sample.fill_diagonal_(False)
    negatives = ~identities[:, None].eq(identities[None, :])
    positive_logits = logits.masked_fill(~same_sample, -torch.inf)
    denominator_logits = logits.masked_fill(
        ~(same_sample | negatives), -torch.inf
    )
    return -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(denominator_logits, dim=1)
    ).mean()


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
    dataset: Dataset,
    device: torch.device,
    args: argparse.Namespace,
    method: str = CSR,
    use_annealing: bool = True,
    fixed_k_override: Optional[int] = None,
    artifact_seed: Optional[int] = None,
) -> Tuple[TopKSAE, List[Dict[str, Any]], Dict[str, Any]]:
    """Train fixed-K CSR v1 or cosine-annealed CSRv2."""
    input_dim = int(dataset.features.shape[1])
    model = TopKSAE(input_dim, args.hidden_dim, args.dead_steps).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.csr_lr, weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    method_dir = args.output_dir / method
    if artifact_seed is not None:
        method_dir = method_dir / f"seed_{artifact_seed}"
    history: List[Dict[str, Any]] = []
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        # The released CSR loader keeps the final incomplete ImageNet batch.
        drop_last=False,
        **loader_options(args.workers, args.prefetch_factor),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total_steps = args.epochs * len(loader)
    fixed_k = (
        args.train_k
        if use_annealing
        else (
            fixed_k_override
            if fixed_k_override is not None
            else args.v1_train_k
        )
    )
    if fixed_k_override is not None:
        method_dir = method_dir / f"k_{fixed_k}"
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
                    contrastive = nonnegative_contrastive_loss(
                        sparse_codes.float(), args.contrastive_temperature
                    )
                # The released code obtains the positive view with a second
                # deterministic forward pass, which updates activity again.
                model.update_activity(sparse_codes)
                objective = reconstruction + args.csr_contrastive_weight * contrastive
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()

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
    dataset: Dataset,
    device: torch.device,
    args: argparse.Namespace,
    method: str = MP_SAE,
    use_annealing: bool = True,
    fixed_k_override: Optional[int] = None,
    artifact_seed: Optional[int] = None,
) -> Tuple[TopKSAE, List[Dict[str, Any]], Dict[str, Any]]:
    input_dim = int(dataset.features.shape[1])
    model = TopKSAE(input_dim, args.hidden_dim, args.dead_steps).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    method_dir = args.output_dir / method
    if artifact_seed is not None:
        method_dir = method_dir / f"seed_{artifact_seed}"
    history: List[Dict[str, Any]] = []

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        # Match the CSR data exposure exactly; the final ImageNet batch is
        # still much larger than the minimum MMPOT group size.
        drop_last=False,
        **loader_options(args.workers, args.prefetch_factor),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    mpsae_epochs = args.epochs + args.mpsae_extra_epochs
    total_steps = mpsae_epochs * len(loader)
    fixed_k = (
        args.train_k
        if use_annealing
        else (
            fixed_k_override
            if fixed_k_override is not None
            else args.v1_train_k
        )
    )
    if fixed_k_override is not None:
        method_dir = method_dir / f"k_{fixed_k}"
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
    if method in (MATRYOSHKA, DENSE):
        return features[:, :k].float()
    raise ValueError(f"{method} must use compact sparse encoding for GPU retrieval")


def unit_normalize_embeddings(embeddings: torch.Tensor, method: str, k: int) -> torch.Tensor:
    """Normalize every row and fail loudly if a learned code is all zero."""
    if not bool(torch.isfinite(embeddings).all().item()):
        raise RuntimeError(f"non-finite {method} embedding at K={k}")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all().item()):
        raise RuntimeError(f"non-finite {method} embedding norm at K={k}")
    zero_rows = norms.squeeze(1) <= torch.finfo(embeddings.dtype).eps
    if zero_rows.any():
        raise RuntimeError(
            f"cannot unit-normalize {int(zero_rows.sum().item())} zero {method} "
            f"embedding(s) at K={k}"
        )
    return embeddings / norms


@torch.inference_mode()
def encode_sparse_for_benchmark(
    method: str, model: TopKSAE, features: torch.Tensor, k: int,
    model_device: torch.device, index_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep Top-K coordinates/values without scattering into [batch, hidden_dim].

    Top-K indices are unique and in range by construction. Checking the whole
    preactivation batch prevents NaNs/Infs being silently discarded by Top-K.
    Validation and normalization are setup, never timed sparse search work.
    """
    if not 0 < k <= model.decoder.shape[0]:
        raise ValueError(f"invalid sparse K={k} for dictionary width {model.decoder.shape[0]}")
    pre = model.preactivations(features.to(model_device, non_blocking=True)).float()
    if not bool(torch.isfinite(pre).all().item()):
        raise RuntimeError(f"non-finite {method} preactivations at K={k}")
    values, indices = torch.topk(pre, k, dim=1, sorted=False)
    values = unit_normalize_embeddings(F.relu(values), method, k)
    return (
        indices.to(device=index_device, dtype=torch.int32).contiguous(),
        values.to(device=index_device, dtype=torch.float32).contiguous(),
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def add_gallery_to_faiss(
    index: Any, method: str, model: Optional[TopKSAE], dataset: Dataset,
    k: int, batch_size: int, model_device: torch.device,
    index_device: torch.device, normalize: bool, print_freq: int = 50,
) -> Tuple[torch.Tensor, int]:
    labels: List[torch.Tensor] = []
    nonzeros = 0
    if model is not None:
        model.eval()
    added_rows = 0
    for batch_number, (features, target) in enumerate(
        cached_feature_batches(dataset, batch_size), start=1
    ):
        z = encode_for_benchmark(method, model, features, k, model_device)
        if normalize:
            z = unit_normalize_embeddings(z, method, k)
        index.add(z.to(index_device).contiguous())
        nonzeros += int(torch.count_nonzero(z).item())
        labels.append(target)
        added_rows += target.numel()
        if print_freq > 0 and (batch_number % print_freq == 0 or added_rows == len(dataset)):
            print(
                f"Retrieval gallery: method={method} k={k} rows={added_rows}/{len(dataset)}",
                flush=True,
            )
    if not labels:
        raise ValueError("cannot retrieve from an empty gallery")
    return torch.cat(labels).to(index_device), nonzeros


@torch.inference_mode()
def search_queries(
    index: Any,
    gallery_labels: torch.Tensor,
    method: str,
    model: Optional[TopKSAE],
    dataset: Dataset,
    k: int,
    batch_size: int,
    model_device: torch.device,
    index_device: torch.device,
    normalize: bool,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    correct, total, distance_sum = 0, 0, 0.0
    query_nnz, preparation_seconds, sparse_products = 0, 0.0, 0
    sparse = isinstance(index, CudaSparseIndex)
    timer = CudaSearchTimer(index_device, args.retrieval_warmup, args.retrieval_repeats)
    if model is not None:
        model.eval()
    for batch_number, (features, target) in enumerate(
        cached_feature_batches(dataset, batch_size), start=1
    ):
        synchronize_device(model_device)
        synchronize_device(index_device)
        started = time.perf_counter()
        if sparse:
            if model is None:
                raise ValueError("sparse queries require a trained model")
            coordinates, values = encode_sparse_for_benchmark(
                method, model, features, k, model_device, index_device
            )
            query_nnz += int(torch.count_nonzero(values).item())
            sparse_products += index.query_product_count(coordinates, values)
        else:
            z = encode_for_benchmark(method, model, features, k, model_device)
            if normalize:
                z = unit_normalize_embeddings(z, method, k)
            query = z.to(index_device).contiguous()
            query_nnz += int(torch.count_nonzero(query).item())
        synchronize_device(index_device)
        preparation_seconds += time.perf_counter() - started
        if sparse:
            scores, neighbors = timer.measure(
                lambda: index.search(coordinates, values), (features.shape[0], k),
                features.shape[0],
            )
            distances = (2.0 - 2.0 * scores).clamp_min(0.0)
        else:
            distances_2d, indices = timer.measure(
                lambda: index.search(query, 1), (features.shape[0], k), features.shape[0]
            )
            neighbors, distances = indices[:, 0], distances_2d[:, 0].clamp_min(0.0)
        # Validation, label lookup, accuracy, and host copies stay outside timing.
        if bool(((neighbors < 0) | (neighbors >= len(gallery_labels))).any().item()):
            raise RuntimeError("retrieval returned an invalid gallery row")
        if not bool(torch.isfinite(distances).all().item()):
            raise RuntimeError("retrieval returned non-finite distances")
        predictions = gallery_labels[neighbors]
        truth = target.to(index_device, non_blocking=True)
        correct += int((predictions == truth).sum().item())
        total += truth.numel()
        distance_sum += float(distances.sum().item())
        if args.print_freq > 0 and (batch_number % args.print_freq == 0 or total == len(dataset)):
            print(
                f"Retrieval queries: method={method} k={k} rows={total}/{len(dataset)} "
                f"running_top1={100.0 * correct / total:.3f}%",
                flush=True,
            )
    if total == 0:
        raise ValueError("cannot benchmark an empty validation query set")
    return {
        "top1": 100.0 * correct / total,
        "mean_neighbor_l2_squared": distance_sum / total,
        "query_samples": total,
        "average_query_nonzero_coordinates": query_nnz / total,
        "query_preparation_seconds": preparation_seconds,
        "sparse_products_per_query": sparse_products / total if sparse else None,
        "sparse_products_per_logical_pass": sparse_products if sparse else None,
        "dense_dictionary_products_per_logical_pass": (
            total * len(gallery_labels) * index.hidden_dim if sparse else None
        ),
        "sparse_product_fraction_of_dense_dictionary": (
            sparse_products / (total * len(gallery_labels) * index.hidden_dim)
            if sparse else None
        ),
        **timer.summary(total),
    }


@torch.inference_mode()
def encode_sparse_gallery(
    method: str,
    model: TopKSAE,
    dataset: Dataset,
    k: int,
    model_device: torch.device,
    index_device: torch.device,
    args: argparse.Namespace,
) -> Tuple[CudaSparseIndex, torch.Tensor]:
    """Build resident GPU postings with bounded temporary encoding/sort storage.

    Encoding batch boundaries are independent of posting-chunk boundaries.
    Each chunk fits the query score-buffer cap and at most 4 Mi Top-K entries,
    corresponding to the index builder's 384 MiB sorting-workspace estimate.
    The returned index records its actual chunk_count and max_chunk_rows;
    these may differ from the requested args.sparse_gallery_chunk at large K.
    Extension compilation and search-workspace preparation happen separately.
    """
    model.eval()
    hidden_dim = int(model.decoder.shape[0])
    if not 0 < k <= hidden_dim:
        raise ValueError(f"invalid sparse K={k} for dictionary width {hidden_dim}")
    score_bytes = args.sparse_score_buffer_mib * 1024**2
    chunk_rows = min(
        args.sparse_gallery_chunk,
        score_bytes // (4 * args.sparse_knn_query_batch),
        max(1, 4 * 1024**2 // k),
        torch.iinfo(torch.int32).max,
    )
    if chunk_rows < 1:
        raise ValueError("Sparse score workspace cannot hold even one gallery row per query")
    index = CudaSparseIndex(
        hidden_dim=hidden_dim,
        device=index_device,
        max_score_buffer_bytes=score_bytes,
        memory_fraction=args.sparse_gpu_memory_fraction,
    )
    label_chunks: List[torch.Tensor] = []
    pending_indices: List[torch.Tensor] = []
    pending_values: List[torch.Tensor] = []
    pending_rows = 0
    last_progress = time.monotonic()
    encoding_batch = min(args.knn_batch_size, chunk_rows)
    for features, target in cached_feature_batches(dataset, encoding_batch):
        indices, values = encode_sparse_for_benchmark(
            method, model, features, k, model_device, index_device
        )
        label_chunks.append(target.detach().to(device="cpu", dtype=torch.long).contiguous())
        start = 0
        while start < indices.shape[0]:
            take = min(chunk_rows - pending_rows, indices.shape[0] - start)
            pending_indices.append(indices[start:start + take])
            pending_values.append(values[start:start + take])
            pending_rows += take
            start += take
            if pending_rows == chunk_rows:
                index.add(torch.cat(pending_indices, dim=0), torch.cat(pending_values, dim=0))
                pending_indices.clear()
                pending_values.clear()
                pending_rows = 0
                if args.print_freq > 0 and (
                    index.chunk_count == 1 or time.monotonic() - last_progress >= 10.0
                ):
                    print(
                        f"Sparse gallery: method={method} k={k} "
                        f"rows={index.row_count}/{len(dataset)} chunks={index.chunk_count} "
                        f"resident_index_mib={index.payload_bytes / 2**20:.1f}",
                        flush=True,
                    )
                    last_progress = time.monotonic()
        del indices, values

    if pending_rows:
        index.add(torch.cat(pending_indices, dim=0), torch.cat(pending_values, dim=0))
        pending_indices.clear()
        pending_values.clear()
    if index.row_count == 0:
        raise ValueError("Cannot retrieve from an empty sparse gallery")
    labels = torch.cat(label_chunks, dim=0).to(device=index_device, dtype=torch.long)
    if labels.numel() != index.row_count:
        raise RuntimeError("Sparse gallery labels do not match posting insertion order")
    if args.print_freq > 0:
        print(
            f"Sparse gallery ready: method={method} k={k} rows={index.row_count} "
            f"chunks={index.chunk_count} resident_index_mib={index.payload_bytes / 2**20:.1f}",
            flush=True,
        )
    return index, labels


def make_faiss_index(
    dimension: int, use_gpu: bool, gpu_device: int, temp_memory_mib: int
) -> Tuple[Any, Optional[Any]]:
    try:
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401 - registers PyTorch tensor interop
    except ImportError as exc:
        raise RuntimeError("CUDA FAISS with PyTorch tensor interop is required") from exc
    if not use_gpu:
        raise ValueError("CPU FAISS is disabled in the GPU retrieval experiment")
    cpu_index = faiss.IndexFlatL2(dimension)
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("GPU FAISS requested, but the installed package is CPU-only")
    if gpu_device >= torch.cuda.device_count():
        raise RuntimeError(f"FAISS GPU {gpu_device} requested, but only "
                           f"{torch.cuda.device_count()} CUDA device(s) are visible")
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(temp_memory_mib * 1024**2)
    # benchmark_method runs on this device's default stream. Events on that
    # stream must enclose FAISS work as well as the custom CUDA sparse search.
    resources.setDefaultNullStreamAllDevices()
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False
    return faiss.index_cpu_to_gpu(resources, gpu_device, cpu_index, options), resources


def retrieval_device(args: argparse.Namespace, model_device: torch.device) -> torch.device:
    if not args.faiss_gpu or not torch.cuda.is_available():
        raise RuntimeError("local dense and sparse retrieval both require CUDA; no CPU fallback")
    ordinal = args.faiss_gpu_device
    if ordinal is None:
        ordinal = model_device.index if model_device.type == "cuda" else 0
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    if not 0 <= ordinal < torch.cuda.device_count():
        raise ValueError(f"retrieval CUDA device {ordinal} is not visible")
    return torch.device("cuda", ordinal)


def preflight_retrieval(args: argparse.Namespace, model_device: torch.device) -> None:
    """Fail on missing GPU backends/toolchains before feature extraction or training."""
    index_device = retrieval_device(args, model_device)
    print(f"Checking CUDA retrieval backends on {index_device}", flush=True)
    with torch.cuda.device(index_device):
        # Constructing a tiny index verifies the installed FAISS GPU bindings;
        # reserve no search scratch until an actual benchmark is prepared.
        index, resources = make_faiss_index(1, True, index_device.index, 0)
        del index, resources
        preload_sparse_retrieval(index_device)
    print("CUDA retrieval backends and sparse extension ready", flush=True)


def benchmark_method(
    method: str,
    model: Optional[TopKSAE],
    train_data: Dataset,
    val_data: Dataset,
    device: torch.device,
    args: argparse.Namespace,
    evaluation_budgets: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    # Resolve an unindexed CUDA device before changing the current device.
    # A caller may use a different retrieval GPU from the SAE encoder GPU.
    if model is not None:
        device = next(model.parameters()).device
    elif device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    index_device = retrieval_device(args, device)
    # Torch and FAISS share the default stream; record events on that same
    # stream. Synchronize the encoder too if it lives on another CUDA device.
    synchronize_device(device)
    with torch.inference_mode(), torch.cuda.device(index_device), torch.cuda.stream(
        torch.cuda.default_stream(index_device)
    ):
        return benchmark_method_on_gpu(
            method, model, train_data, val_data, device, index_device, args,
            evaluation_budgets,
        )


def benchmark_method_on_gpu(
    method: str, model: Optional[TopKSAE], train_data: Dataset, val_data: Dataset,
    device: torch.device, index_device: torch.device, args: argparse.Namespace,
    evaluation_budgets: Optional[Sequence[int]],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    if not len(train_data) or not len(val_data):
        raise ValueError("GPU retrieval requires nonempty gallery and query datasets")
    budgets = (
        sorted(set(evaluation_budgets))
        if evaluation_budgets is not None
        else (
            list(args.topk)
            if method in (MATRYOSHKA, DENSE)
            else sorted(set((*args.sparse_extra_topk, *args.topk)))
        )
    )
    backend = (
        "faiss_gpu_index_flat_l2_fp32"
        if method in (MATRYOSHKA, DENSE)
        else "cuda_sparse_inverted_exact_cosine_fp32"
    )

    for k in budgets:
        synchronize_device(index_device)
        setup_started = time.perf_counter()
        gpu_resources = None
        print(f"GPU exact retrieval: method={method} k={k} device={index_device}", flush=True)
        if method in (MATRYOSHKA, DENSE):
            # FAISS allocates outside PyTorch's caching allocator. Release only
            # unused cached blocks before estimating its full-resident payload.
            # Index growth/search scratch can still require more than this
            # estimate; an allocation failure must never select a CPU backend.
            torch.cuda.empty_cache()
            dense_required_bytes = (
                len(train_data) * (k * 4 + 8)
                + min(args.knn_batch_size, len(train_data)) * k * 12
                + args.faiss_temp_memory_mib * 1024**2
            )
            check_cuda_memory_budget(
                index_device, dense_required_bytes, 1.0,
                "dense FAISS gallery, labels, configured scratch, and encoding buffers",
            )
            index, gpu_resources = make_faiss_index(
                k, True, index_device.index, args.faiss_temp_memory_mib
            )
            gallery_labels, gallery_nnz = add_gallery_to_faiss(
                index, method, model, train_data, k, args.knn_batch_size,
                device, index_device, args.knn_normalize, args.print_freq
            )
            if index.ntotal != len(train_data):
                raise RuntimeError("FAISS gallery row count differs from the dataset")
            index_statistics = {
                "storage": "dense_float32_gpu_flat_index",
                "gallery_nonzeros": gallery_nnz,
                "payload_bytes": int(index.ntotal) * k * 4,
                "score_workspace_bytes": None,
                "faiss_reserved_temp_bytes": args.faiss_temp_memory_mib * 1024**2,
                "chunk_count": None,
                "max_chunk_rows": None,
                "memory_accounting": "payload_only_excludes_FAISS_allocator_and_search_scratch",
                "preflight_estimated_additional_bytes": dense_required_bytes,
            }
        else:
            if model is None:
                raise ValueError(f"{method} sparse retrieval requires a trained model")
            index, gallery_labels = encode_sparse_gallery(
                method, model, train_data, k, device, index_device, args
            )
            if index.row_count != len(train_data):
                raise RuntimeError("sparse gallery row count differs from the dataset")
            gallery_nnz = index.nnz
            index_statistics = {
                "storage": index.storage,
                "gallery_nonzeros": index.nnz,
                "payload_bytes": index.payload_bytes,
                "chunk_count": index.chunk_count,
                "max_chunk_rows": index.max_chunk_rows,
                "index_build_entry_cap": 4 * 1024**2,
                "posting_kernel": "occupancy_partitioned_warp_lists_max8_partitions",
                "memory_accounting": "postings_and_score_buffer_excludes_allocator_and_small_reduction_outputs",
            }
        synchronize_device(index_device)
        gallery_setup_seconds = time.perf_counter() - setup_started
        preparation_started = time.perf_counter()
        if isinstance(index, CudaSparseIndex):
            index.prepare(args.sparse_knn_query_batch)
            index_statistics["score_workspace_bytes"] = index.workspace_bytes
        synchronize_device(index_device)
        preparation_seconds = time.perf_counter() - preparation_started
        index_statistics["label_bytes"] = gallery_labels.numel() * gallery_labels.element_size()
        metrics = search_queries(
            index, gallery_labels, method, model, val_data, k,
            args.knn_query_batch, device, index_device, args.knn_normalize, args,
        )
        metrics.update({
            "gallery_samples": len(gallery_labels),
            "average_gallery_nonzero_coordinates": gallery_nnz / len(gallery_labels),
            "retrieval_backend": backend,
            "index_setup_seconds": gallery_setup_seconds + preparation_seconds,
            "gallery_encoding_transfer_and_build_seconds": gallery_setup_seconds,
            "workspace_and_compilation_seconds": preparation_seconds,
            "index_statistics": index_statistics,
            "retrieval_config": {
                "version": 2,
                "device": str(index_device),
                "device_name": torch.cuda.get_device_name(index_device),
                "device_capability": list(torch.cuda.get_device_capability(index_device)),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": torch.version.cuda,
                "query_batch_size": args.knn_query_batch,
                "configured_gallery_encoding_batch_size": args.knn_batch_size,
                "gallery_encoding_batch_size": min(
                    args.knn_batch_size,
                    index.max_chunk_rows if isinstance(index, CudaSparseIndex) else len(train_data),
                ),
                "encoder_device": str(device),
                "encoder_tf32_enabled": args.tf32,
                "sparse_gallery_chunk_limit": args.sparse_gallery_chunk,
                "sparse_score_buffer_mib": args.sparse_score_buffer_mib,
                "sparse_gpu_memory_fraction": args.sparse_gpu_memory_fraction,
                "faiss_temp_memory_mib": args.faiss_temp_memory_mib,
                "warmup_per_shape": args.retrieval_warmup,
                "repeats_per_batch": args.retrieval_repeats,
                "precision": "float32_values_and_scores_no_float16_index",
                "residency": "entire_gallery_and_current_query_batch_on_same_gpu",
                "timing_scope": TIMING_SCOPE,
                "timing_statistic": "median_of_interleaved_repeat_totals",
                "accuracy": "one_separate_untimed_search_per_query_batch",
                "query_preparation_scope": "encoding_normalization_transfer_validation_and_work_counts_excludes_feature_loading",
                "product_count_scope": "one_logical_query_pass_excludes_repeats_warmup_and_accuracy_calls",
                "tie_policy": (
                    "smallest_gallery_id_for_equal_accumulated_scores"
                    if isinstance(index, CudaSparseIndex) else "FAISS_equal_distance_tie_order"
                ),
                "numerical_note": (
                    "exhaustive_fp32_atomic_sums_may_change_near_ties;"
                    "unit_norm_cosine_equals_L2_up_to_roundoff"
                    if isinstance(index, CudaSparseIndex) else "exhaustive_float32_L2"
                ),
            },
        })
        results[str(k)] = metrics
        log_wandb_metrics(
            f"{method}/knn",
            {"budget": k, **results[str(k)]},
            step_metric="budget",
        )
        del index, gpu_resources, gallery_labels
    return {
        "protocol": "unit_normalized_exact_L2_train_gallery_validation_queries_1NN",
        "retrieval_backend": backend,
        "device": str(index_device),
        "normalized": args.knn_normalize,
        "representation": (
            "full_dense_embedding"
            if method == DENSE
            else (
                "prefix_dimension"
                if method == MATRYOSHKA
                else "topk_sparse_latents"
            )
        ),
        "timing_scope": TIMING_SCOPE,
        "per_topk": results,
    }


def paper_reference_knn(
    values: Mapping[int, float], representation: str
) -> Dict[str, Any]:
    """Represent a published accuracy curve without inventing local timing data."""
    return {
        "protocol": "published_ImageNet1K_exact_L2_1NN_Table_4",
        "retrieval_backend": "reported_by_Wen_et_al_2025",
        "device": None,
        "normalized": True,
        "representation": representation,
        "timing_scope": None,
        "per_topk": {
            str(k): {
                "top1": float(top1),
                "mean_neighbor_l2_squared": None,
                "gallery_samples": PAPER_FF2048_TRAIN_SAMPLES,
                "query_samples": PAPER_FF2048_VAL_SAMPLES,
                "retrieval_search_seconds": None,
                "retrieval_query_batches": None,
                "mean_retrieval_seconds_per_query": None,
                "mean_retrieval_milliseconds_per_query": None,
                "average_gallery_nonzero_coordinates": float(k),
                "average_query_nonzero_coordinates": float(k),
                "retrieval_backend": "reported_by_Wen_et_al_2025",
            }
            for k, top1 in values.items()
        },
    }


def merge_knn_results(parts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not parts:
        raise ValueError("cannot merge an empty list of 1-NN results")
    merged = dict(parts[0])
    merged["per_topk"] = {}
    for part in parts:
        duplicate = set(merged["per_topk"]).intersection(part["per_topk"])
        if duplicate:
            raise ValueError(
                "duplicate K while merging independently trained models: "
                + ", ".join(sorted(duplicate, key=int))
            )
        merged["per_topk"].update(part["per_topk"])
    merged["per_topk"] = dict(
        sorted(merged["per_topk"].items(), key=lambda item: int(item[0]))
    )
    merged["training_model_per_topk"] = True
    return merged


def aggregate_knn_across_seeds(
    per_seed: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Aggregate matched per-K evaluations while retaining every seed result."""
    if not per_seed:
        raise ValueError("cannot aggregate an empty seed mapping")
    seed_names = sorted(per_seed, key=int)
    first = per_seed[seed_names[0]]
    expected_budgets = set(first["per_topk"])
    for seed_name in seed_names[1:]:
        observed = set(per_seed[seed_name]["per_topk"])
        if observed != expected_budgets:
            raise ValueError(
                f"seed {seed_name} has K={sorted(observed, key=int)}, expected "
                f"{sorted(expected_budgets, key=int)}"
            )

    identity_fields = {
        "gallery_samples",
        "query_samples",
        "retrieval_query_batches",
        "retrieval_backend",
        "retrieval_config",
    }
    combined: Dict[str, Dict[str, Any]] = {}
    for budget in sorted(expected_budgets, key=int):
        observations = [
            per_seed[seed_name]["per_topk"][budget]
            for seed_name in seed_names
        ]
        metrics = dict(observations[0])
        for field in ("timing", "index_statistics"):
            if any(field in item for item in observations):
                metrics.pop(field, None)
                metrics[f"{field}_by_seed"] = {
                    seed_name: observation.get(field)
                    for seed_name, observation in zip(seed_names, observations)
                }
        for field in identity_fields:
            values = [item.get(field) for item in observations]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"seed results disagree on {field} at K={budget}: {values}"
                )
        for field in observations[0]:
            if field in identity_fields or field in ("timing", "index_statistics"):
                continue
            values = [item.get(field) for item in observations]
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in values
            ):
                numeric = np.asarray(values, dtype=np.float64)
                metrics[field] = float(numeric.mean())
                metrics[f"{field}_std"] = (
                    float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
                )
        metrics["top1_by_seed"] = {
            seed_name: float(
                per_seed[seed_name]["per_topk"][budget]["top1"]
            )
            for seed_name in seed_names
        }
        combined[budget] = metrics

    aggregate = dict(first)
    aggregate["per_topk"] = combined
    aggregate["training_model_per_topk"] = True
    aggregate["aggregated_across_seeds"] = True
    aggregate["timing_aggregation"] = (
        "arithmetic_mean_of_per_seed_metrics_with_sample_standard_deviation; "
        "repeat_measurements_and_index_statistics_preserved_by_seed"
    )
    aggregate["seeds"] = [int(seed_name) for seed_name in seed_names]
    aggregate["per_seed"] = dict(per_seed)
    return aggregate


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


def paper_reproduction_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    required = (MATRYOSHKA, CSR_V1, MP_SAE_V1)
    if not all(method in results for method in required):
        return []
    mrl = results[MATRYOSHKA]["knn"]["per_topk"]
    csr = results[CSR_V1]["knn"]["per_topk"]
    mpsae = results[MP_SAE_V1]["knn"]["per_topk"]
    rows: List[Dict[str, Any]] = []
    common = sorted(
        set(int(k) for k in mrl)
        & set(int(k) for k in csr)
        & set(int(k) for k in mpsae)
    )
    for k in common:
        mrl_top1 = float(mrl[str(k)]["top1"])
        csr_top1 = float(csr[str(k)]["top1"])
        mpsae_top1 = float(mpsae[str(k)]["top1"])
        rows.append(
            {
                "active_dimension": k,
                "published_mrl_1nn_top1": mrl_top1,
                "published_csr_1nn_top1": PAPER_CSR_1NN_TOP1.get(k),
                "reproduced_csr_1nn_top1": csr_top1,
                "csr_reproduction_gap_pp": (
                    csr_top1 - PAPER_CSR_1NN_TOP1[k]
                    if k in PAPER_CSR_1NN_TOP1 else None
                ),
                "mpsae_1nn_top1": mpsae_top1,
                "mpsae_minus_published_mrl_pp": mpsae_top1 - mrl_top1,
                "mpsae_minus_reproduced_csr_pp": mpsae_top1 - csr_top1,
            }
        )
    return rows


def write_paper_reproduction_table(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    if not rows:
        return
    lines = [
        "| K | Published MRL | Published CSR | Reproduced CSR | CSR gap | MPSAE | MPSAE - MRL | MPSAE - CSR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['active_dimension']} | "
            f"{row['published_mrl_1nn_top1']:.2f} | "
            f"{row['published_csr_1nn_top1']:.2f} | "
            f"{row['reproduced_csr_1nn_top1']:.2f} | "
            f"{row['csr_reproduction_gap_pp']:+.2f} | "
            f"{row['mpsae_1nn_top1']:.2f} | "
            f"{row['mpsae_minus_published_mrl_pp']:+.2f} | "
            f"{row['mpsae_minus_reproduced_csr_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "All values are ImageNet-1K exact L2 1-NN top-1 accuracy (%).",
            "Published MRL/CSR values are from Table 4 of arXiv:2503.01776.",
            "Each reproduced CSR and MPSAE point is trained by an independent fixed-K model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rn18_study_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build the three-seed Appendix E.1 CSR/MPSAE extension table."""
    required = (DENSE, CSR_V1, MP_SAE_V1)
    if not all(method in results for method in required):
        return []
    dense = results[DENSE]["knn"]["per_topk"][str(512)]
    csr = results[CSR_V1]["knn"]["per_topk"]
    mpsae = results[MP_SAE_V1]["knn"]["per_topk"]
    full_top1 = float(dense["top1"])
    common = sorted(set(int(k) for k in csr) & set(int(k) for k in mpsae))
    rows: List[Dict[str, Any]] = []
    for k in common:
        csr_top1 = float(csr[str(k)]["top1"])
        mpsae_top1 = float(mpsae[str(k)]["top1"])
        csr_by_seed = csr[str(k)].get("top1_by_seed", {})
        mpsae_by_seed = mpsae[str(k)].get("top1_by_seed", {})
        paired_seeds = sorted(
            set(csr_by_seed).intersection(mpsae_by_seed), key=int
        )
        paired_deltas = np.asarray(
            [
                float(mpsae_by_seed[seed]) - float(csr_by_seed[seed])
                for seed in paired_seeds
            ],
            dtype=np.float64,
        )
        paired_delta_mean = (
            float(paired_deltas.mean())
            if paired_deltas.size
            else mpsae_top1 - csr_top1
        )
        paired_delta_std = (
            float(paired_deltas.std(ddof=1))
            if paired_deltas.size > 1
            else 0.0
        )
        rows.append(
            {
                "active_dimension": k,
                "dense_ff512_1nn_top1": full_top1,
                "csr_mean_1nn_top1": csr_top1,
                "csr_std_1nn_top1": float(csr[str(k)].get("top1_std", 0.0)),
                "mpsae_mean_1nn_top1": mpsae_top1,
                "mpsae_std_1nn_top1": float(
                    mpsae[str(k)].get("top1_std", 0.0)
                ),
                "mpsae_minus_csr_pp": paired_delta_mean,
                "mpsae_minus_csr_std_pp": paired_delta_std,
                "csr_fidelity_percent": 100.0 * csr_top1 / full_top1,
                "mpsae_fidelity_percent": 100.0 * mpsae_top1 / full_top1,
            }
        )
    return rows


def write_rn18_study_table(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    if not rows:
        return
    lines = [
        "| K | Dense FF512 | CSR mean ± std | MPSAE mean ± std | Paired MPSAE - CSR mean ± std | CSR fidelity | MPSAE fidelity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['active_dimension']} | "
            f"{row['dense_ff512_1nn_top1']:.2f} | "
            f"{row['csr_mean_1nn_top1']:.2f} ± "
            f"{row['csr_std_1nn_top1']:.2f} | "
            f"{row['mpsae_mean_1nn_top1']:.2f} ± "
            f"{row['mpsae_std_1nn_top1']:.2f} | "
            f"{row['mpsae_minus_csr_pp']:+.2f} ± "
            f"{row['mpsae_minus_csr_std_pp']:.2f} | "
            f"{row['csr_fidelity_percent']:.2f}% | "
            f"{row['mpsae_fidelity_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Metric: ImageNet-1K unit-normalized exact L2 1-NN top-1 accuracy (%).",
            "CSR and MPSAE entries are means and sample standard deviations over three matched seeds.",
            "The paper does not publish its seed values; 1227, 1228, and 1229 are this study's predeclared reproducible seed set.",
            "The dense FF512 reference is measured from timm/resnet18.a1_in1k features.",
            "Image preprocessing is a semantic torchvision translation of the released FFCV pipeline, not a bit-exact FFCV execution.",
            "No ResNet-18 MRL value is included because the cited papers do not report one.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    if results.get(MATRYOSHKA, {}).get("paper_reference"):
        plot_paper_ff2048_comparison(results, output_dir)
        return
    if DENSE in results and CSR_V1 in results and MP_SAE_V1 in results:
        plot_paper_rn18_study(results, output_dir)
        return
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


def plot_paper_ff2048_comparison(
    results: Mapping[str, Any], output_dir: Path
) -> None:
    """Plot only the primary fixed-K paper comparison in FF2048 mode."""
    rows = paper_reproduction_rows(results)
    if not rows:
        return
    configure_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    budgets = [int(row["active_dimension"]) for row in rows]
    published_mrl = [float(row["published_mrl_1nn_top1"]) for row in rows]
    published_csr = [float(row["published_csr_1nn_top1"]) for row in rows]
    reproduced_csr = [float(row["reproduced_csr_1nn_top1"]) for row in rows]
    mpsae = [float(row["mpsae_1nn_top1"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].plot(
        budgets, published_mrl, color="#0072B2", marker="o",
        label="Published MRL",
    )
    axes[0].plot(
        budgets, published_csr, color="#009E73", marker="^",
        linestyle=":", label="Published CSR",
    )
    axes[0].plot(
        budgets, reproduced_csr, color="#009E73", marker="^",
        label="Reproduced CSR",
    )
    axes[0].plot(
        budgets, mpsae, color="#D55E00", marker="s", label="MPSAE",
    )
    axes[0].set_title("(a) FF2048 exact 1-NN", loc="left", fontweight="bold")
    axes[0].set_ylabel("ImageNet val. top-1 accuracy (%)")
    axes[0].legend(frameon=False, handlelength=2.2)

    axes[1].plot(
        budgets,
        [float(row["csr_reproduction_gap_pp"]) for row in rows],
        color="#009E73", marker="^", label="Reproduced CSR - published CSR",
    )
    axes[1].plot(
        budgets,
        [float(row["mpsae_minus_reproduced_csr_pp"]) for row in rows],
        color="#D55E00", marker="s", label="MPSAE - reproduced CSR",
    )
    axes[1].plot(
        budgets,
        [float(row["mpsae_minus_published_mrl_pp"]) for row in rows],
        color="#CC79A7", marker="D", linestyle="--",
        label="MPSAE - published MRL",
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_title("(b) Reproduction and method gaps", loc="left", fontweight="bold")
    axes[1].set_ylabel("Delta top-1 accuracy (pp)")
    axes[1].legend(frameon=False, handlelength=2.2)

    for axis in axes:
        axis.set_xlabel("Active dimension K")
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(FixedLocator(budgets))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(
            axis="y", color="#B8B8B8", linestyle=(0, (2, 2)),
            linewidth=0.55, alpha=0.65,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_figure_formats(
        fig, output_dir, "paper_ff2048_reproduction_comparison"
    )
    plt.close(fig)


def plot_paper_rn18_study(
    results: Mapping[str, Any], output_dir: Path
) -> None:
    """Plot the primary three-seed ResNet-18 CSR/MPSAE study."""
    rows = rn18_study_rows(results)
    if not rows:
        return
    configure_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    budgets = [int(row["active_dimension"]) for row in rows]
    dense = float(rows[0]["dense_ff512_1nn_top1"])
    csr = [float(row["csr_mean_1nn_top1"]) for row in rows]
    csr_std = [float(row["csr_std_1nn_top1"]) for row in rows]
    mpsae = [float(row["mpsae_mean_1nn_top1"]) for row in rows]
    mpsae_std = [float(row["mpsae_std_1nn_top1"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].axhline(
        dense, color="#0072B2", linestyle=":", label="Dense FF512"
    )
    axes[0].errorbar(
        budgets, csr, yerr=csr_std, color="#009E73", marker="^",
        capsize=2.5, label="CSR (3 seeds)",
    )
    axes[0].errorbar(
        budgets, mpsae, yerr=mpsae_std, color="#D55E00", marker="s",
        capsize=2.5, label="MPSAE (3 seeds)",
    )
    axes[0].set_title("(a) ResNet-18 exact 1-NN", loc="left", fontweight="bold")
    axes[0].set_ylabel("ImageNet val. top-1 accuracy (%)")
    axes[0].legend(frameon=False, handlelength=2.2)

    axes[1].plot(
        budgets,
        [float(row["mpsae_minus_csr_pp"]) for row in rows],
        color="#D55E00", marker="s", label="MPSAE - CSR",
    )
    axes[1].errorbar(
        budgets,
        [float(row["mpsae_minus_csr_pp"]) for row in rows],
        yerr=[float(row["mpsae_minus_csr_std_pp"]) for row in rows],
        color="#D55E00", marker="s", capsize=2.5, linestyle="none",
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_title("(b) Matched method gap", loc="left", fontweight="bold")
    axes[1].set_ylabel("Delta top-1 accuracy (pp)")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xlabel("Active dimension K")
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(FixedLocator(budgets))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.grid(
            axis="y", color="#B8B8B8", linestyle=(0, (2, 2)),
            linewidth=0.55, alpha=0.65,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_figure_formats(fig, output_dir, "paper_rn18_study_comparison")
    plt.close(fig)


def plot_retrieval_time(results: Mapping[str, Any], output_dir: Path) -> List[str]:
    """Plot measured GPU and synchronized wall times, excluding reference curves."""
    timing_fields = (
        "mean_retrieval_milliseconds_per_query",
        "mean_retrieval_wall_milliseconds_per_query",
    )
    method_order = list(dict.fromkeys((DENSE, *METHODS, *results)))
    measured_methods = [
        method for method in method_order
        if method in results and any(
            isinstance(metric.get(field), (int, float))
            and not isinstance(metric.get(field), bool)
            and math.isfinite(metric[field])
            and metric[field] > 0.0
            for metric in results[method].get("knn", {}).get("per_topk", {}).values()
            for field in timing_fields
        )
    ]
    if not measured_methods:
        return []
    configure_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    colors = {
        DENSE: "#666666", MATRYOSHKA: "#0072B2", CSR_V1: "#56B4E9",
        CSR: "#009E73", MP_SAE_V1: "#E69F00", MP_SAE: "#D55E00",
    }
    markers = {
        DENSE: "X", MATRYOSHKA: "o", CSR_V1: "v", CSR: "^",
        MP_SAE_V1: "D", MP_SAE: "s",
    }
    labels = {
        DENSE: "Dense", MATRYOSHKA: "Matryoshka", CSR_V1: "CSR v1",
        CSR: "CSRv2", MP_SAE_V1: "MPSAE v1", MP_SAE: "MPSAEv2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), constrained_layout=True)
    for axis, field, title in zip(
        axes, timing_fields,
        ("(a) CUDA-event search time", "(b) Synchronized wall search time"),
    ):
        plotted_budgets: set[int] = set()
        for method in measured_methods:
            metrics = results[method]["knn"]["per_topk"]
            budgets = [
                int(value) for value in sorted(metrics, key=int)
                if isinstance(metrics[value].get(field), (int, float))
                and not isinstance(metrics[value].get(field), bool)
                and math.isfinite(metrics[value][field])
                and metrics[value][field] > 0.0
            ]
            if not budgets:
                continue
            batches = {
                metrics[str(k)].get("retrieval_config", {}).get("query_batch_size")
                for k in budgets
            }
            label = labels.get(method, method)
            if len(batches) == 1 and None not in batches:
                label += f" (B={next(iter(batches))})"
            elif len(batches) > 1:
                label += " (B varies)"
            axis.plot(
                budgets, [metrics[str(k)][field] for k in budgets],
                color=colors.get(method), marker=markers.get(method, "o"),
                label=label,
            )
            plotted_budgets.update(budgets)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Representation budget K")
        axis.set_ylabel("Amortized search time (ms/query)")
        if plotted_budgets:
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.xaxis.set_major_locator(FixedLocator(sorted(plotted_budgets)))
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.tick_params(axis="x", labelrotation=45)
            axis.legend(frameon=False)
        else:
            axis.text(
                0.5, 0.5, "No measured timings", ha="center", va="center",
                transform=axis.transAxes,
            )
        axis.grid(axis="y", color="#B8B8B8", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    backbone_name = results[measured_methods[0]].get("backbone", "backbone")
    stem = f"ablation_{backbone_name}_retrieval_time_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure_formats(fig, output_dir, stem)
    plt.close(fig)
    return [f"{stem}.png", f"{stem}.pdf"]


def write_retrieval_timing_table(
    results: Mapping[str, Any], output_dir: Path,
) -> List[str]:
    """Write local measurements while leaving published accuracy-only rows out."""
    rows: List[Dict[str, Any]] = []
    for method in dict.fromkeys((DENSE, *METHODS, *results)):
        if method not in results:
            continue
        result = results[method]
        knn = result.get("knn", {})
        for budget, metrics in sorted(knn.get("per_topk", {}).items(), key=lambda item: int(item[0])):
            gpu_ms = metrics.get("mean_retrieval_milliseconds_per_query")
            wall_ms = metrics.get("mean_retrieval_wall_milliseconds_per_query")
            if not any(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0.0
                for value in (gpu_ms, wall_ms)
            ):
                continue
            config = metrics.get("retrieval_config", {})
            rows.append({
                "method": method,
                "backbone": result.get("backbone"),
                "representation_budget": int(budget),
                "top1": metrics.get("top1"),
                "top1_std": metrics.get("top1_std"),
                "average_gallery_nonzero_coordinates": metrics.get("average_gallery_nonzero_coordinates"),
                "average_query_nonzero_coordinates": metrics.get("average_query_nonzero_coordinates"),
                "mean_retrieval_milliseconds_per_query": gpu_ms,
                "mean_retrieval_milliseconds_per_query_std": metrics.get("mean_retrieval_milliseconds_per_query_std"),
                "mean_retrieval_wall_milliseconds_per_query": wall_ms,
                "mean_retrieval_wall_milliseconds_per_query_std": metrics.get("mean_retrieval_wall_milliseconds_per_query_std"),
                "retrieval_queries_per_second": metrics.get("retrieval_queries_per_second"),
                "sparse_products_per_query": metrics.get("sparse_products_per_query"),
                "sparse_product_fraction_of_dense_dictionary": metrics.get("sparse_product_fraction_of_dense_dictionary"),
                "mean_retrieval_milliseconds_per_batch": metrics.get("mean_retrieval_milliseconds_per_batch"),
                "query_batch_size": config.get("query_batch_size"),
                "gallery_samples": metrics.get("gallery_samples"),
                "query_samples": metrics.get("query_samples"),
                "retrieval_backend": metrics.get("retrieval_backend", knn.get("retrieval_backend")),
                "device": config.get("device", knn.get("device")),
                "device_name": config.get("device_name"),
                "index_setup_seconds": metrics.get("index_setup_seconds"),
                "query_preparation_seconds": metrics.get("query_preparation_seconds"),
                "seed_count": len(knn.get("seeds", [])) if knn.get("aggregated_across_seeds") else 1,
            })
    if not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(rows, output_dir / "retrieval_timing.csv")

    def number(value: Any) -> str:
        return f"{value:.6g}" if isinstance(value, (int, float)) else "n/a"

    lines = [
        "| Method | K | 1-NN top-1 (%) | Gallery nnz/row | Query nnz/row | Batch B | GPU ms/query | Wall ms/query | GPU queries/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['representation_budget']} | "
            f"{number(row['top1'])} | "
            f"{number(row['average_gallery_nonzero_coordinates'])} | "
            f"{number(row['average_query_nonzero_coordinates'])} | "
            f"{number(row['query_batch_size'])} | "
            f"{number(row['mean_retrieval_milliseconds_per_query'])} | "
            f"{number(row['mean_retrieval_wall_milliseconds_per_query'])} | "
            f"{number(row['retrieval_queries_per_second'])} |"
        )
    lines.extend([
        "",
        "Only locally measured retrieval results are included. Published MRL accuracy has no local timing measurement.",
        "GPU time uses CUDA events; wall time includes synchronized search dispatch and completion. Both measure search after query preparation and index construction.",
        "Times per query are amortized over batches of up to B queries, with the final partial batch included. They are not isolated single-query latencies.",
        "Compare timings with the recorded batch size, device, precision, residency, and gallery size. Full configuration and per-repeat measurements are in summary.json; retrieval_timing.csv includes device and setup-time columns.",
        "Sparse product counts report visited query-feature/gallery-posting pairs per logical pass. Their dense-dictionary fraction compares to N times H products per query in the same latent space; it is an operation-count comparison, not a measured speedup or a comparison with smaller dense prefixes.",
        "For multiple seeds, numeric entries are arithmetic means of the corresponding per-seed summaries; the CSV includes available sample standard deviations. Raw timing and index-statistic records remain separate by seed in summary.json.",
    ])
    path = output_dir / "retrieval_timing.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return ["retrieval_timing.csv", "retrieval_timing.md"]


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
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.weights_cache = args.weights_cache.expanduser().resolve()
    if args.ff2048_train_embeddings is not None:
        args.ff2048_train_embeddings = (
            args.ff2048_train_embeddings.expanduser().resolve()
        )
    if args.ff2048_val_embeddings is not None:
        args.ff2048_val_embeddings = (
            args.ff2048_val_embeddings.expanduser().resolve()
        )
    validate_args(args)
    backbone = BACKBONE_SPECS[args.backbone]
    labels = method_labels(backbone)
    enabled_methods = selected_methods(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = choose_device(args.device)
    preflight_retrieval(args, device)  # Check backends/toolchain before expensive work.
    configure_runtime(args, device)
    if args.protocol == PAPER_RN18_STUDY_PROTOCOL:
        wandb_name = f"rn18-appendix-e1-{args.method}"
        wandb_group = "rn18-appendix-e1-csr-mpsae"
        experiment_family = "csr_mpsae_rn18_appendix_e1_study"
        wandb_tags = (
            "csr", "mpsae", "resnet18", "appendix-e1",
            "fixed-k", "three-seed", args.method,
        )
    else:
        wandb_name = f"five-method-{backbone.name}-{args.method}"
        wandb_group = "five-method-architecture-ablation"
        experiment_family = "matryoshka_csr_v1_v2_mpsae_v1_v2"
        wandb_tags = (
            "matryoshka", "csr", "mpsae", "csrv2", "mpsaev2",
            "topk-annealing", args.backbone, args.method,
        )
    init_wandb(
        args,
        default_name=wandb_name,
        default_group=wandb_group,
        extra_config={
            "experiment_family": experiment_family,
            "backbone_display_name": backbone.display_name,
            "backbone_weights": experiment_backbone_weights(args.protocol, backbone),
            "protocol": args.protocol,
            "public_experiment_config": serializable_args(args),
        },
        tags=wandb_tags,
    )
    print(
        f"backbone={backbone.name} feature_dim={backbone.output_dim} "
        f"sae_hidden_dim={args.hidden_dim} protocol={args.protocol} "
        f"device={device} output={args.output_dir}",
        flush=True,
    )

    results: Dict[str, Any] = {}
    dataset_metadata: Dict[str, Any] = {}

    if MATRYOSHKA in enabled_methods:
        if args.protocol == PAPER_FF2048_PROTOCOL:
            nested_dims = list(PAPER_FF2048_BUDGETS)
            results[MATRYOSHKA] = {
                "display_name": "Published MRL ResNet-50",
                "backbone": backbone.name,
                "feature_dim": backbone.output_dim,
                "trainable_parameters": None,
                "training_protocol": (
                    "published_MRL_checkpoint_results_from_CSR_Table_4"
                ),
                "training_epochs": 40,
                "nested_dims": nested_dims,
                "loss_weights": {"classification": 1.0},
                "history": [],
                "paper_reference": {
                    "paper": "Wen et al. 2025, arXiv:2503.01776",
                    "table": 4,
                    "source_model": "RAIVNLab/MRL:r50_mrl1_e0_ff2048.pt",
                    "locally_retrained": False,
                },
                "knn": paper_reference_knn(
                    PAPER_MRL_1NN_TOP1, "matryoshka_prefix_dimension"
                ),
            }
            dataset_metadata[MATRYOSHKA] = {
                "source": "published_Table_4_values",
                "gallery_samples": PAPER_FF2048_TRAIN_SAMPLES,
                "query_samples": PAPER_FF2048_VAL_SAMPLES,
            }
        else:
            seed_all(args.seed)
            matryoshka_model, history = train_matryoshka_backbone(device, args)
            nested_dims = list(matryoshka_model.nested_dims)
            mrl_train_meta = cache_matryoshka_split(
                "train", matryoshka_model, device, args
            )
            mrl_val_meta = cache_matryoshka_split(
                "val", matryoshka_model, device, args
            )
            mrl_cache_dir = args.cache_dir / "matryoshka_finetuned"
            mrl_train_features, mrl_train_labels, _ = cache_paths(
                mrl_cache_dir, "train"
            )
            mrl_val_features, mrl_val_labels, _ = cache_paths(
                mrl_cache_dir, "val"
            )
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
                "training_protocol": (
                    f"non_reproduction_end_to_end_{backbone.name}_"
                    "torchvision_initialized_mrl"
                ),
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
            del matryoshka_model, mrl_train_data, mrl_val_data
            if device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_json(
            results[MATRYOSHKA], args.output_dir / MATRYOSHKA / "results.json"
        )
        if args.protocol == PAPER_FF2048_PROTOCOL:
            atomic_json(
                {
                    "method": MATRYOSHKA,
                    "backbone": args.backbone,
                    "training_protocol": (
                        "published_MRL_checkpoint_results_from_CSR_Table_4"
                    ),
                    "history": [],
                    "paper_reference": results[MATRYOSHKA]["paper_reference"],
                },
                args.output_dir / MATRYOSHKA / "history.json",
            )

    if any(method in enabled_methods for method in SPARSE_METHODS):
        if args.protocol == PAPER_FF2048_PROTOCOL:
            train_data = ReleasedNpzFeatures(args.ff2048_train_embeddings)
            val_data = ReleasedNpzFeatures(args.ff2048_val_embeddings)
            frozen_train_meta = train_data.metadata("train")
            frozen_val_meta = val_data.metadata("validation")
            if args.paper_strict:
                sample_counts = {
                    "training": (len(train_data), PAPER_FF2048_TRAIN_SAMPLES),
                    "validation": (len(val_data), PAPER_FF2048_VAL_SAMPLES),
                }
                bad_counts = [
                    f"{name}={actual} (paper: {expected})"
                    for name, (actual, expected) in sample_counts.items()
                    if actual != expected
                ]
                if bad_counts:
                    raise RuntimeError(
                        "released FF2048 sample counts do not match the paper: "
                        + "; ".join(bad_counts)
                    )
            ff2048_knn = benchmark_method(
                DENSE, None, train_data, val_data, device, args,
                evaluation_budgets=[backbone.output_dim],
            )
            ff2048_observed = ff2048_knn["per_topk"][str(backbone.output_dim)][
                "top1"
            ]
            ff2048_gap = ff2048_observed - PAPER_FF2048_FULL_1NN_TOP1
            dataset_metadata["ff2048_integrity_check"] = {
                "expected_full_1nn_top1": PAPER_FF2048_FULL_1NN_TOP1,
                "observed_full_1nn_top1": ff2048_observed,
                "gap_percentage_points": ff2048_gap,
                "strict_tolerance_percentage_points": (
                    PAPER_FF2048_INTEGRITY_TOLERANCE_PP
                ),
                "passed": (
                    abs(ff2048_gap) <= PAPER_FF2048_INTEGRITY_TOLERANCE_PP
                ),
                "knn": ff2048_knn,
            }
            if (
                args.paper_strict
                and abs(ff2048_gap) > PAPER_FF2048_INTEGRITY_TOLERANCE_PP
            ):
                raise RuntimeError(
                    "released FF2048 integrity check failed before sparse "
                    f"training: observed {ff2048_observed:.4f}% versus "
                    f"paper {PAPER_FF2048_FULL_1NN_TOP1:.2f}% "
                    f"(gap {ff2048_gap:+.4f} pp; allowed "
                    f"{PAPER_FF2048_INTEGRITY_TOLERANCE_PP:.2f} pp)"
                )
        else:
            frozen_backbone = FrozenBackbone(
                args.weights_cache, backbone, args.protocol
            ).to(device)
            if args.channels_last:
                frozen_backbone = frozen_backbone.to(memory_format=torch.channels_last)
            frozen_train_meta = cache_split(
                "train", args.data_root, frozen_backbone, device, args.cache_dir,
                args.feature_batch_size, args.workers, args.prefetch_factor,
                args.channels_last, args.max_train, args.seed,
                args.rebuild_cache, args.data_backend, args.hf_dataset_id,
                args.hf_revision, args.hf_token_env,
            )
            frozen_val_meta = cache_split(
                "val", args.data_root, frozen_backbone, device, args.cache_dir,
                args.feature_batch_size, args.workers, args.prefetch_factor,
                args.channels_last, args.max_val, args.seed,
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
        if args.protocol == PAPER_RN18_STUDY_PROTOCOL:
            sample_counts = {
                "training": (len(train_data), PAPER_FF2048_TRAIN_SAMPLES),
                "validation": (len(val_data), PAPER_FF2048_VAL_SAMPLES),
            }
            bad_counts = [
                f"{name}={actual} (ImageNet-1K: {expected})"
                for name, (actual, expected) in sample_counts.items()
                if actual != expected
            ]
            if args.paper_strict and bad_counts:
                raise RuntimeError(
                    "ResNet-18 study requires the complete ImageNet-1K splits: "
                    + "; ".join(bad_counts)
                )
            dense_knn = benchmark_method(
                DENSE,
                None,
                train_data,
                val_data,
                device,
                args,
                evaluation_budgets=[backbone.output_dim],
            )
            dense_top1 = dense_knn["per_topk"][str(backbone.output_dim)][
                "top1"
            ]
            results[DENSE] = {
                "display_name": "Dense FF512 (timm ResNet-18)",
                "backbone": backbone.name,
                "feature_dim": backbone.output_dim,
                "trainable_parameters": 0,
                "training_protocol": "frozen_pretrained_full_representation",
                "training_epochs": 0,
                "history": [],
                "knn": dense_knn,
            }
            dataset_metadata[DENSE] = {
                "train": frozen_train_meta,
                "validation": frozen_val_meta,
            }
            dataset_metadata["rn18_feature_integrity"] = {
                "backbone_weights": PAPER_RN18_BACKBONE_ID,
                "preprocessing": frozen_train_meta["preprocessing"],
                "preprocessing_equivalence": frozen_train_meta[
                    "preprocessing_equivalence"
                ],
                "official_ffcv_bit_exact": False,
                "paper_configuration_matched": True,
                "source_feature_extractor": (
                    "released_CSR_FFCV_pretrained_embeddings.py"
                ),
                "source_train_order": "random",
                "source_train_drop_last": True,
                "local_train_order": "sequential",
                "local_train_drop_last": False,
                "local_full_training_split_retained": True,
                "full_split_policy_reason": (
                    "the released FFCV extractor randomly drops its final "
                    "incomplete batch, but does not publish the ordering or "
                    "seed needed to identify those examples reproducibly"
                ),
                "feature_extraction_batch_size": frozen_train_meta[
                    "feature_extraction_batch_size"
                ],
                "feature_extraction_workers": frozen_train_meta[
                    "feature_extraction_workers"
                ],
                "feature_extraction_compute": frozen_train_meta[
                    "feature_extraction_compute"
                ],
                "feature_dim": int(train_data.features.shape[1]),
                "expected_feature_dim": 512,
                "observed_full_1nn_top1": dense_top1,
                "published_numeric_target": None,
                "numeric_gate_applied": False,
                "numeric_gate_reason": (
                    "CSR Figure 4 does not tabulate an exact ResNet-18 value"
                ),
                "sample_counts": {
                    name: {"observed": actual, "expected": expected}
                    for name, (actual, expected) in sample_counts.items()
                },
                "provenance_and_shape_passed": (
                    int(train_data.features.shape[1]) == 512
                    and not bad_counts
                ),
                "knn": dense_knn,
            }
            atomic_json(
                results[DENSE], args.output_dir / DENSE / "results.json"
            )
        initial_state = initial_sae_state(
            backbone.output_dim, args.hidden_dim, args.dead_steps, args.seed
        )

        for method, use_annealing in ((CSR_V1, False), (CSR, True)):
            if method not in enabled_methods:
                continue
            if args.protocol in PAPER_FIXED_K_PROTOCOLS and method == CSR_V1:
                csr_histories_by_seed: Dict[
                    str, Dict[str, List[Dict[str, Any]]]
                ] = {}
                csr_schedules_by_seed: Dict[
                    str, Dict[str, Dict[str, Any]]
                ] = {}
                csr_knn_by_seed: Dict[str, Dict[str, Any]] = {}
                csr_parameter_count: Optional[int] = None
                for study_seed in args.study_seeds:
                    seed_key = str(study_seed)
                    seed_histories: Dict[str, List[Dict[str, Any]]] = {}
                    seed_schedules: Dict[str, Dict[str, Any]] = {}
                    seed_knn_parts: List[Dict[str, Any]] = []
                    for fixed_k in args.topk:
                        model_state = initial_sae_state(
                            backbone.output_dim,
                            args.hidden_dim,
                            args.dead_steps,
                            study_seed,
                        )
                        seed_all(study_seed)
                        csr_model, csr_history, csr_schedule = train_csr(
                            model_state,
                            train_data,
                            device,
                            args,
                            method,
                            False,
                            fixed_k_override=fixed_k,
                            artifact_seed=(
                                study_seed
                                if len(args.study_seeds) > 1
                                else None
                            ),
                        )
                        csr_parameter_count = sum(
                            parameter.numel()
                            for parameter in csr_model.parameters()
                            if parameter.requires_grad
                        )
                        csr_part = benchmark_method(
                            method,
                            csr_model,
                            train_data,
                            val_data,
                            device,
                            args,
                            evaluation_budgets=[fixed_k],
                        )
                        if args.protocol == PAPER_FF2048_PROTOCOL:
                            measured = csr_part["per_topk"][str(fixed_k)]
                            measured["paper_reference_top1"] = (
                                PAPER_CSR_1NN_TOP1[fixed_k]
                            )
                            measured["paper_reproduction_gap_pp"] = (
                                measured["top1"]
                                - PAPER_CSR_1NN_TOP1[fixed_k]
                            )
                        seed_histories[str(fixed_k)] = csr_history
                        seed_schedules[str(fixed_k)] = csr_schedule
                        seed_knn_parts.append(csr_part)
                        del csr_model, model_state
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    csr_histories_by_seed[seed_key] = seed_histories
                    csr_schedules_by_seed[seed_key] = seed_schedules
                    csr_knn_by_seed[seed_key] = merge_knn_results(
                        seed_knn_parts
                    )
                csr_knn = (
                    next(iter(csr_knn_by_seed.values()))
                    if len(csr_knn_by_seed) == 1
                    else aggregate_knn_across_seeds(csr_knn_by_seed)
                )
                representative_seed = str(args.study_seeds[0])
                representative_histories = csr_histories_by_seed[
                    representative_seed
                ]
                representative_k = (
                    str(args.v1_train_k)
                    if str(args.v1_train_k) in representative_histories
                    else str(args.topk[0])
                )
                training_protocol = (
                    "paper_Table_3_independent_fixed_K_CSR_models"
                    if args.protocol == PAPER_FF2048_PROTOCOL
                    else (
                        "Appendix_E1_configuration_matched_fixed_K_CSR_models_"
                        "three_seed_mean"
                    )
                )
                results[method] = {
                    "display_name": labels[method],
                    "backbone": backbone.name,
                    "feature_dim": backbone.output_dim,
                    "trainable_parameters": csr_parameter_count,
                    "training_protocol": training_protocol,
                    "training_epochs": args.epochs,
                    "training_k_models": list(args.topk),
                    "training_seeds": list(args.study_seeds),
                    "topk_annealing": {
                        "schedule": "independent_fixed_k_models",
                        "per_seed": csr_schedules_by_seed,
                    },
                    "loss_weights": {
                        "main_reconstruction": args.csr_main_recon_weight,
                        "multi_topk_reconstruction": args.csr_multi_topk_recon_weight,
                        "auxiliary_reconstruction": args.csr_aux_recon_weight,
                        "nonnegative_contrastive": args.csr_contrastive_weight,
                    },
                    "contrastive_temperature": args.contrastive_temperature,
                    "optimizer": {
                        "name": "Adam",
                        "lr": args.csr_lr,
                        "weight_decay": args.weight_decay,
                        "eps": args.adam_eps,
                    },
                    "history": representative_histories[representative_k],
                    "history_representative_seed": int(representative_seed),
                    "history_representative_k": int(representative_k),
                    "history_by_k": representative_histories,
                    "history_by_seed": csr_histories_by_seed,
                    "knn": csr_knn,
                }
                if args.protocol == PAPER_FF2048_PROTOCOL:
                    results[method]["paper_reference_1nn_top1"] = (
                        PAPER_CSR_1NN_TOP1
                    )
                dataset_metadata[method] = {
                    "train": frozen_train_meta, "validation": frozen_val_meta
                }
                atomic_json(
                    results[method], args.output_dir / method / "results.json"
                )
                atomic_json(
                    {
                        "method": method,
                        "backbone": args.backbone,
                        "training_protocol": training_protocol,
                        "training_seeds": list(args.study_seeds),
                        "history_by_seed": csr_histories_by_seed,
                    },
                    args.output_dir / method / "history.json",
                )
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
            if method not in enabled_methods:
                continue
            if args.protocol in PAPER_FIXED_K_PROTOCOLS and method == MP_SAE_V1:
                mp_histories_by_seed: Dict[
                    str, Dict[str, List[Dict[str, Any]]]
                ] = {}
                mp_schedules_by_seed: Dict[
                    str, Dict[str, Dict[str, Any]]
                ] = {}
                mp_knn_by_seed: Dict[str, Dict[str, Any]] = {}
                mp_parameter_count: Optional[int] = None
                for study_seed in args.study_seeds:
                    seed_key = str(study_seed)
                    seed_histories: Dict[str, List[Dict[str, Any]]] = {}
                    seed_schedules: Dict[str, Dict[str, Any]] = {}
                    seed_knn_parts: List[Dict[str, Any]] = []
                    for fixed_k in args.topk:
                        model_state = initial_sae_state(
                            backbone.output_dim,
                            args.hidden_dim,
                            args.dead_steps,
                            study_seed,
                        )
                        seed_all(study_seed)
                        mp_model, mp_history, mp_schedule = train_mp_sae(
                            model_state,
                            train_data,
                            device,
                            args,
                            method,
                            False,
                            fixed_k_override=fixed_k,
                            artifact_seed=(
                                study_seed
                                if len(args.study_seeds) > 1
                                else None
                            ),
                        )
                        mp_parameter_count = sum(
                            parameter.numel()
                            for parameter in mp_model.parameters()
                            if parameter.requires_grad
                        )
                        mp_part = benchmark_method(
                            method,
                            mp_model,
                            train_data,
                            val_data,
                            device,
                            args,
                            evaluation_budgets=[fixed_k],
                        )
                        seed_histories[str(fixed_k)] = mp_history
                        seed_schedules[str(fixed_k)] = mp_schedule
                        seed_knn_parts.append(mp_part)
                        del mp_model, model_state
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    mp_histories_by_seed[seed_key] = seed_histories
                    mp_schedules_by_seed[seed_key] = seed_schedules
                    mp_knn_by_seed[seed_key] = merge_knn_results(
                        seed_knn_parts
                    )
                mp_knn = (
                    next(iter(mp_knn_by_seed.values()))
                    if len(mp_knn_by_seed) == 1
                    else aggregate_knn_across_seeds(mp_knn_by_seed)
                )
                representative_seed = str(args.study_seeds[0])
                representative_histories = mp_histories_by_seed[
                    representative_seed
                ]
                representative_k = (
                    str(args.v1_train_k)
                    if str(args.v1_train_k) in representative_histories
                    else str(args.topk[0])
                )
                training_protocol = (
                    "paper_matched_independent_fixed_K_MPSAE_models"
                    if args.protocol == PAPER_FF2048_PROTOCOL
                    else (
                        "Appendix_E1_configuration_matched_fixed_K_MPSAE_models_"
                        "three_seed_mean"
                    )
                )
                results[method] = {
                    "display_name": labels[method],
                    "backbone": backbone.name,
                    "feature_dim": backbone.output_dim,
                    "trainable_parameters": mp_parameter_count,
                    "training_protocol": training_protocol,
                    "training_epochs": args.epochs,
                    "training_k_models": list(args.topk),
                    "training_seeds": list(args.study_seeds),
                    "topk_annealing": {
                        "schedule": "independent_fixed_k_models",
                        "per_seed": mp_schedules_by_seed,
                    },
                    "loss_weights": {
                        "main_reconstruction": args.mpsae_main_recon_weight,
                        "multi_topk_reconstruction": args.mpsae_nested_recon_weight,
                        "auxiliary_reconstruction": args.mpsae_aux_recon_weight,
                        "mmpot_regularizer": args.mpsae_mmpot_weight,
                    },
                    "optimizer": {
                        "name": "Adam",
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "eps": args.adam_eps,
                    },
                    "history": representative_histories[representative_k],
                    "history_representative_seed": int(representative_seed),
                    "history_representative_k": int(representative_k),
                    "history_by_k": representative_histories,
                    "history_by_seed": mp_histories_by_seed,
                    "knn": mp_knn,
                }
                dataset_metadata[method] = {
                    "train": frozen_train_meta, "validation": frozen_val_meta
                }
                atomic_json(
                    results[method], args.output_dir / method / "results.json"
                )
                atomic_json(
                    {
                        "method": method,
                        "backbone": args.backbone,
                        "training_protocol": training_protocol,
                        "training_seeds": list(args.study_seeds),
                        "history_by_seed": mp_histories_by_seed,
                    },
                    args.output_dir / method / "history.json",
                )
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

    if args.protocol == PAPER_RN18_STUDY_PROTOCOL:
        study_role = "paper_appendix_E1_resnet18_csr_mpsae_extension"
        ablation_variable = None
        feature_source = "locally_encoded_timm_resnet18_a1_in1k_features"
        mrl_source: Optional[str] = None
        retrieval_backends = {
            DENSE: "faiss_gpu_index_flat_l2_fp32",
            CSR_V1: "cuda_sparse_inverted_exact_cosine_fp32",
            MP_SAE_V1: "cuda_sparse_inverted_exact_cosine_fp32",
        }
        evaluation_budgets = {
            DENSE: [backbone.output_dim],
            CSR_V1: list(args.topk),
            MP_SAE_V1: list(args.topk),
        }
        budget_definition = {
            DENSE: "full dense FF512 representation",
            CSR_V1: "number of retained Top-K sparse latents",
            MP_SAE_V1: "number of retained Top-K sparse latents",
        }
        training_epochs = {
            DENSE: 0,
            CSR_V1: args.epochs,
            MP_SAE_V1: args.epochs + args.mpsae_extra_epochs,
        }
    else:
        study_role = (
            "paper_ff2048_primary_comparison"
            if args.protocol == PAPER_FF2048_PROTOCOL
            else "non_reproduction_architecture_ablation_backbone_unit"
        )
        ablation_variable = (
            None if args.protocol == PAPER_FF2048_PROTOCOL else "backbone"
        )
        feature_source = (
            "authors_released_FF2048_embeddings"
            if args.protocol == PAPER_FF2048_PROTOCOL
            else "locally_encoded_torchvision_features"
        )
        mrl_source = (
            "published_CSR_Table_4_values"
            if args.protocol == PAPER_FF2048_PROTOCOL
            else "local_non_reproduction_finetune"
        )
        retrieval_backends = {
            MATRYOSHKA: (
                "reported_by_Wen_et_al_2025"
                if args.protocol == PAPER_FF2048_PROTOCOL
                else "faiss_gpu_index_flat_l2_fp32"
            ),
            **{
                method: "cuda_sparse_inverted_exact_cosine_fp32"
                for method in SPARSE_METHODS
            },
        }
        evaluation_budgets = {
            MATRYOSHKA: list(args.topk),
            **{
                method: sorted(set((*args.sparse_extra_topk, *args.topk)))
                for method in SPARSE_METHODS
            },
        }
        budget_definition = {
            MATRYOSHKA: f"{backbone.display_name} feature-prefix dimension",
            **{
                method: "number of retained Top-K sparse latents at evaluation"
                for method in SPARSE_METHODS
            },
        }
        training_epochs = {
            MATRYOSHKA: (
                40 if args.protocol == PAPER_FF2048_PROTOCOL else args.epochs
            ),
            CSR_V1: args.epochs,
            CSR: args.epochs,
            MP_SAE_V1: args.epochs + args.mpsae_extra_epochs,
            MP_SAE: args.epochs + args.mpsae_extra_epochs,
        }

    summary = {
        "experiment": (
            "CSR_vs_MPSAE_timm_resnet18_Appendix_E1_study"
            if args.protocol == PAPER_RN18_STUDY_PROTOCOL
            else f"Matryoshka_vs_CSR_v1_v2_vs_MPSAE_v1_v2_{backbone.name}"
        ),
        "study_role": study_role,
        "protocol": args.protocol,
        "ablation_variable": ablation_variable,
        "backbone": {
            "name": backbone.name,
            "display_name": backbone.display_name,
            "feature_dim": backbone.output_dim,
            "weights": experiment_backbone_weights(args.protocol, backbone),
        },
        "method_labels": labels,
        "comparison_protocol": {
            "dataset": "ImageNet-1K",
            "feature_source": feature_source,
            "mrl_source": mrl_source,
            "paper_reference": (
                "Wen et al. 2025 Appendix E.1 and Table 9"
                if args.protocol == PAPER_RN18_STUDY_PROTOCOL
                else (
                    "Wen et al. 2025 Tables 3 and 4"
                    if args.protocol == PAPER_FF2048_PROTOCOL
                    else None
                )
            ),
            "metric": "unit_normalized_exact_L2_1NN_top1",
            "gallery": "training_split",
            "queries": "validation_split",
            "unit_normalized": True,
            "retrieval_timing_scope": TIMING_SCOPE,
            "retrieval_backends": retrieval_backends,
            "evaluation_budgets": evaluation_budgets,
            "budget_definition": budget_definition,
            "training_epochs": training_epochs,
            "training_seeds": list(args.study_seeds),
            "training_seed_policy": (
                "Appendix E.1 reports a three-seed mean without publishing "
                "the seed values; this run predeclares its recorded seeds"
                if args.protocol == PAPER_RN18_STUDY_PROTOCOL
                else "single recorded experiment seed"
            ),
            "topk_annealing": {
                method: results.get(method, {}).get("topk_annealing")
                for method in SPARSE_METHODS
            },
            "model_weights_saved": False,
            "sparse_model_training_granularity": (
                "one_independent_model_per_K_for_v1"
                if args.protocol in PAPER_FIXED_K_PROTOCOLS
                else "one_model_evaluated_at_all_K"
            ),
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
    if args.protocol == PAPER_FF2048_PROTOCOL:
        paper_rows = paper_reproduction_rows(results)
        write_comparison_csv(
            paper_rows, args.output_dir / "paper_ff2048_reproduction.csv"
        )
        write_paper_reproduction_table(
            paper_rows, args.output_dir / "paper_ff2048_reproduction.md"
        )
        summary["paper_ff2048_reproduction"] = {
            "reference": "Wen et al. 2025, arXiv:2503.01776, Table 4",
            "rows": paper_rows,
        }
    elif args.protocol == PAPER_RN18_STUDY_PROTOCOL:
        paper_rows = rn18_study_rows(results)
        write_comparison_csv(
            paper_rows, args.output_dir / "paper_rn18_study.csv"
        )
        write_rn18_study_table(
            paper_rows, args.output_dir / "paper_rn18_study.md"
        )
        summary["paper_rn18_study"] = {
            "reference": (
                "Wen et al. 2025, arXiv:2503.01776, Appendix E.1, "
                "Figure 4, and Table 9"
            ),
            "scope": (
                "CSR Appendix E.1 configuration-matched study plus a matched "
                "MPSAE extension; no ResNet-18 MRL baseline exists"
            ),
            "seed_disclosure": (
                "The paper reports a three-seed mean but not the seed values; "
                "the locally predeclared seeds are recorded in config.study_seeds"
            ),
            "preprocessing_disclosure": (
                "semantic torchvision translation of the released FFCV "
                "pipeline; not bit-exact FFCV. The local cache retains the "
                "complete training split because the released extractor's "
                "randomly dropped final batch cannot be reconstructed exactly"
            ),
            "published_curve_is_digitized": False,
            "rows": paper_rows,
        }
    plot_publication_comparison(results, args.output_dir)
    # The FF2048 integrity measurement is a local full dense baseline. It is
    # separate from published MRL and may be used in timing tables as FF2048.
    timing_results = dict(results)
    integrity = dataset_metadata.get("ff2048_integrity_check", {})
    if "knn" in integrity:
        timing_results[DENSE] = {
            "display_name": "Dense FF2048 (local integrity reference)",
            "backbone": backbone.name,
            "knn": integrity["knn"],
        }
    timing_plot_files = plot_retrieval_time(timing_results, args.output_dir)
    timing_table_files = write_retrieval_timing_table(timing_results, args.output_dir)
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
        "loss_curves": (
            []
            if args.protocol in PAPER_FIXED_K_PROTOCOLS
            else [
                f"{artifact_prefix}_training_loss_curves.png",
                f"{artifact_prefix}_training_loss_curves.pdf",
            ]
        ),
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
        "timing_scope": TIMING_SCOPE,
        "timing_version": 2,
        "query_time_semantics": "amortized_batch_time_not_single_query_latency",
        "published_references_have_local_timings": False,
        "ff2048_local_dense_timing_source": (
            "dataset.ff2048_integrity_check.knn" if "knn" in integrity else None
        ),
        "unit_normalized": True,
        "accuracy_plots": (
            [
                "paper_ff2048_reproduction_comparison.png",
                "paper_ff2048_reproduction_comparison.pdf",
            ]
            if args.protocol == PAPER_FF2048_PROTOCOL
            else (
                [
                    "paper_rn18_study_comparison.png",
                    "paper_rn18_study_comparison.pdf",
                ]
                if args.protocol == PAPER_RN18_STUDY_PROTOCOL
                else [
                    f"{artifact_prefix}_representation_accuracy_comparison.png",
                    f"{artifact_prefix}_representation_accuracy_comparison.pdf",
                ]
            )
        ),
        "timing_plots": timing_plot_files,
        "timing_tables": timing_table_files,
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
