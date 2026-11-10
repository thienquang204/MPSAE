"""Backbones, training presets and the experiment grid.

Settings follow the published papers (CSR Table 3/9, CSRv2 Table 11); the
algorithms follow the released code. Nothing here imports torch, so the grid
can be inspected, planned and reported on a machine without CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

# ---------------------------------------------------------------- methods

#: Every trainable configuration. ``csr``/``csrv2`` are baselines reproducing
#: the released algorithms; ``mpsae``/``mpsaev2`` are ours; ``csr_multitopk``
#: is the ablation isolating the OT term from merely having three TopK views.
METHODS = ("csr", "csrv2", "mpsae", "mpsaev2", "csr_multitopk")
BASELINES = ("csr", "csrv2")
OURS = ("mpsae", "mpsaev2")

# A changed objective gets its own storage namespace and campaign identity.
# The unversioned mpsaev2 directory belongs to the former annealed OT-only fit.
METHOD_VARIANTS = {"mpsaev2": "fixed_ot_cl_v1"}
METHOD_EPOCHS = {"mpsaev2": 15}


def method_directory(method: str) -> str:
    variant = METHOD_VARIANTS.get(method)
    return f"{method}__{variant}" if variant else method


def method_from_directory(directory: str) -> str | None:
    """Decode only current method directories, never relabel historical fits."""
    return next((method for method in METHODS if method_directory(method) == directory), None)

#: Seeds for every (backbone, method, K) cell. 1227 is the released CSR seed.
SEEDS = (1227,)


# ------------------------------------------------------------ train preset

@dataclass(frozen=True)
class TrainPreset:
    """Backbone settings; MPSAE-family training uses batches of 1024."""

    name: str
    hidden_dim: int = 8192
    lr: float = 4e-5
    weight_decay: float = 1e-4
    adam_eps: float = 6.25e-10
    adam_betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 10
    batch_size: int = 4096
    k_aux: int = 512
    aux_coef: float = 1 / 32
    nested_coef: float = 0.125
    cl_coef: float = 0.1
    cl_temperature: float = 0.2
    dead_threshold: int = 30
    k_decay_ratio: float = 0.7
    # MPSAE partial-OT matching gap.
    ot_weight: float = 2.0
    ot_mass: float = 0.9
    ot_eta: float = 0.2
    ot_microbatch: int = 32
    ot_max_iters: int = 400
    ot_max_refine_iters: int = 8000
    ot_tol: float = 1e-4

    def __post_init__(self) -> None:
        if self.hidden_dim < 4 or self.batch_size < 2 or self.epochs < 1:
            raise ValueError(f"{self.name}: degenerate preset")
        if self.ot_microbatch < 2:
            raise ValueError(f"{self.name}: OT needs at least two examples per group")


#: CSR Table 3 / CSRv2 Table 11 (ResNet-50 FF2048 ImageNet experiment).
FF2048_PRESET = TrainPreset(name="paper-table3", batch_size=4096, cl_coef=0.1)

#: CSR Table 9 (timm backbones, empirical input-dimension study).
TIMM_PRESET = TrainPreset(name="paper-table9", batch_size=1024, cl_coef=1.0)


def training_preset(method: str, preset: TrainPreset) -> TrainPreset:
    """Resolve method-specific batch size and duration; preserve other settings."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "mpsaev2":
        return replace(preset, batch_size=1024, epochs=METHOD_EPOCHS[method])
    return replace(preset, batch_size=1024) if method in OURS else preset


# ---------------------------------------------------------------- backbone

@dataclass(frozen=True)
class Backbone:
    """One frozen embedding source plus its K grid."""

    name: str
    dim: int
    #: ``mrl`` = MRL pytorch_inference recipe; ``csrv2`` = CSRv2 FFCV recipe.
    recipe: str
    #: Weight location. MRL: "<hf repo>:<file>". timm: model tag.
    source: str
    preset: TrainPreset
    k_list: tuple[int, ...]
    #: Checkpoints reused by experiment E2 (one model, many inference budgets).
    e2_anchors: tuple[int, ...]
    #: Pixel preprocessing. ``None`` takes mean/std from the timm config.
    normalize: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    input_size: int = 224
    #: Published dense 1-NN top-1 to verify the embedding pipeline against.
    reference_dense_top1: float | None = None
    notes: str = ""
    #: Train fewer checkpoints while retaining the full inference grid in k_list.
    train_k_list: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.k_list or sorted(self.k_list) != list(self.k_list):
            raise ValueError(f"{self.name}: K list must be sorted and nonempty")
        if max(self.k_list) > self.dim:
            raise ValueError(f"{self.name}: K must not exceed the embedding dimension {self.dim}")
        if 4 * min(self.k_list) > self.preset.hidden_dim:
            raise ValueError(f"{self.name}: the 4K reconstruction view exceeds hidden_dim")
        if (not self.training_budgets or sorted(set(self.training_budgets)) != list(self.training_budgets)
                or not set(self.training_budgets) <= set(self.k_list)):
            raise ValueError(f"{self.name}: training budgets must be a sorted nonempty subset of K list")
        if not set(self.e2_anchors) <= set(self.training_budgets):
            raise ValueError(f"{self.name}: E2 anchors must be trained K values")
        if self.recipe not in ("mrl", "csrv2"):
            raise ValueError(f"{self.name}: unknown embedding recipe {self.recipe!r}")

    @property
    def training_budgets(self) -> tuple[int, ...]:
        return self.k_list if self.train_k_list is None else self.train_k_list

    @property
    def archive_stem(self) -> str:
        return f"1K_{{split}}_{self.name}"


_POW2 = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
_TIMES3 = (3, 6, 12, 24, 48, 96, 192, 384, 768)
_MODERN_TRAIN_K = (12, 48, 96, 192, 384)
_IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_HALF_STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

BACKBONES: Mapping[str, Backbone] = {
    # ---- branch 1: ResNet, directly comparable with CSR Table 4 / MRL.
    "ff2048": Backbone(
        name="ff2048",
        dim=2048,
        recipe="mrl",
        source="aniketr/mrl-resnet50:fixed-feature/Imagenet1k_R50_ff2048/final_weights.pt",
        preset=FF2048_PRESET,
        k_list=_POW2,
        e2_anchors=(8, 16, 32, 64, 128, 256),
        normalize=_IMAGENET_STATS,
        reference_dense_top1=71.19,
        notes="MRL fixed-feature ResNet-50 (BlurPool). Backbone of CSR Table 4.",
    ),
    "rn18": Backbone(
        name="rn18",
        dim=512,
        recipe="csrv2",
        source="timm/resnet18.a1_in1k",
        preset=TIMM_PRESET,
        k_list=_POW2[:9],
        e2_anchors=(8, 16, 32, 64, 128, 256),
        normalize=_HALF_STATS,
        notes="CSR Table 9 backbone. Mean/std 0.5 exactly as the released CSRv2 script.",
    ),
    # ---- branch 2: modern backbones, our own comparison.
    "convnext_tiny": Backbone(
        name="convnext_tiny",
        dim=768,
        recipe="csrv2",
        source="timm/convnext_tiny.fb_in1k",
        preset=FF2048_PRESET,
        k_list=_TIMES3,
        train_k_list=_MODERN_TRAIN_K,
        e2_anchors=_MODERN_TRAIN_K,
        normalize=None,
        notes="d=768, treated like FF2048.",
    ),
    "vit_b16": Backbone(
        name="vit_b16",
        dim=768,
        recipe="csrv2",
        source="timm/vit_base_patch16_224.augreg_in21k_ft_in1k",
        preset=FF2048_PRESET,
        k_list=tuple(sorted((8, *_TIMES3))),
        train_k_list=(8, *_MODERN_TRAIN_K),
        e2_anchors=(8, *_MODERN_TRAIN_K),
        normalize=None,
        notes="d=768, treated like FF2048. Same augreg family CSR used for ViT-S/L.",
    ),
    "dinov2_vits14": Backbone(
        name="dinov2_vits14",
        dim=384,
        recipe="csrv2",
        source="timm/vit_small_patch14_dinov2.lvd142m",
        preset=TIMM_PRESET,
        k_list=_TIMES3[:8],
        train_k_list=_MODERN_TRAIN_K,
        e2_anchors=_MODERN_TRAIN_K,
        normalize=None,
        notes="d=384, treated like RN18. Built with img_size=224 (pos-embed interpolated).",
    ),
}

BRANCH1 = ("ff2048", "rn18")
BRANCH2 = ("convnext_tiny", "vit_b16", "dinov2_vits14")
#: Default order of work: the paper-comparable branch first.
BACKBONE_ORDER = BRANCH1 + BRANCH2

# Inference selection is independent of the valid training-budget registry.
# Keep this shared by E2 and downstream defaults so legacy entry points do not
# silently evaluate a different grid from the configured campaigns.
INFERENCE_BUDGETS: Mapping[str, tuple[int, ...]] = {
    "ff2048": (16, 32, 256),
    "rn18": (16, 32, 256),
    "vit_b16": (3, 12, 48, 96),
}


def inference_budgets(backbone: str) -> tuple[int, ...]:
    spec = BACKBONES[backbone]
    return INFERENCE_BUDGETS.get(backbone, spec.k_list)


# ------------------------------------------------------------------- cells

@dataclass(frozen=True)
class Cell:
    """One training run: backbone x method x K x seed."""

    backbone: str
    method: str
    k: int
    seed: int

    @property
    def slug(self) -> str:
        return f"{self.backbone}/{method_directory(self.method)}/k{self.k}/seed{self.seed}"

    def spec(self) -> Backbone:
        return BACKBONES[self.backbone]


def grid(backbones=BACKBONE_ORDER, methods=METHODS, seeds=SEEDS,
         k_values=None) -> list[Cell]:
    """Every training cell, in a stable order suitable for resumable runs."""
    cells = []
    for name in backbones:
        spec = BACKBONES[name]
        for k in spec.training_budgets:
            if k_values is not None and k not in set(k_values):
                continue
            for method in methods:
                if method not in METHODS:
                    raise ValueError(f"unknown method {method!r}")
                for seed in seeds:
                    cells.append(Cell(name, method, k, seed))
    return cells


# ------------------------------------------------- CSRv2 annealing schedule

def initial_topk(k: int, hidden_dim: int) -> int:
    """CSRv2's k_init, including the clamps in the released code.

    Paper (App. C.3/D.2): 64 when k < 64, otherwise 4k. Released
    ``main_visual.py`` then clamps to ``hidden_dim // 4`` and up to ``k``.
    """
    if k < 1 or hidden_dim < 4:
        raise ValueError("k and hidden_dim must be positive")
    start = 64 if k < 64 else 4 * k
    if 4 * start > hidden_dim:
        start = hidden_dim // 4
    return max(start, k)


def effective_runs(backbones=BACKBONE_ORDER) -> dict[str, int]:
    """Run counts per backbone, for planning GPU time."""
    counts = {name: len(BACKBONES[name].training_budgets) * len(METHODS) * len(SEEDS)
              for name in backbones}
    counts["total"] = sum(counts.values())
    return counts


def override_preset(spec: Backbone, **changes) -> Backbone:
    """Return a backbone with a modified preset (used by smoke runs)."""
    return replace(spec, preset=replace(spec.preset, **changes))
