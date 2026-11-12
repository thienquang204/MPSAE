"""Torch-free selection of the nine-run pilot; never alters source fit recipes."""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

from ..benchmark.config import Campaign, _section
from ..config import BACKBONES, training_preset
from ..io import canonical_hash

VERSION = "joint-teacher-sparse-ce-v1"
DEFAULTS = dict(epochs="source", batch_size="source", backbone_lr="source", sae_lr="source",
                classifier_lr="source", weight_decay="source", optimizer="adam",
                adam_eps="source", adam_betas="source", drop_last=False, ce_weight=1.0,
                method_weight=1.0, grad_clip=1.0, precision="bf16",
                freeze_bn_stats=True, checkpoint_every=100, log_every=10)


@dataclass(frozen=True)
class FineTuneCampaign:
    campaign: Campaign
    training: dict = field(default_factory=lambda: dict(DEFAULTS))

    @property
    def name(self):
        return self.campaign.name

    @property
    def execution(self):
        return {"storage_mode": "legacy"}  # caller supplies isolated finetune-v1 root

    def cells(self):
        return self.campaign.source_cells()

    def training_options(self, cell):
        preset = training_preset(cell.method, BACKBONES[cell.backbone].preset, k=cell.k)
        result = dict(self.training)
        for key in ("epochs", "batch_size", "weight_decay", "adam_eps", "adam_betas",
                    "backbone_lr", "sae_lr", "classifier_lr"):
            if result[key] == "source":
                result[key] = getattr(preset, "lr" if key.endswith("_lr") else key)
        result["adam_betas"] = list(result["adam_betas"])
        return result

    def resolved(self):
        return {"format": VERSION, "campaign": self.campaign.resolved(),
                "finetune": self.training,
                "resolved_training": {c.slug: self.training_options(c) for c in self.cells()},
                "objective": {"classification": "CE(head(L2_normalize(z_K)), y)",
                    "reconstruction_target": "frozen original backbone, same augmented image",
                    "sparsity_schedule": "fixed target K for all methods, including CSRv2",
                    "regularizer_batch": "physical batch; no gradient accumulation",
                    "optimizer": self.training["optimizer"] + ", constant LR, separate backbone/SAE/head groups",
                    "drop_last": self.training["drop_last"], "teacher_mode": "eval, no gradients"},
                "method_presets": {c.slug: asdict(training_preset(c.method, BACKBONES[c.backbone].preset, k=c.k))
                                   for c in self.cells()}}

    def plan(self):
        base = self.campaign.plan()
        return {**base, "plan_id": canonical_hash(self.resolved()), "config": self.resolved(),
                "mode": "supervised_joint_finetune", "regime": "supervised_joint_finetune",
                "additional_fits": len(self.cells()),
                "frozen_control_1nn_points": base["imagenet_unique_points"],
                "source_checkpoints_required": [c.slug for c in self.cells()]}


def load_config(path):
    import yaml
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fine-tuning config must be a mapping")
    payload = dict(payload)
    training = _section(payload.pop("finetune", {}), DEFAULTS, "finetune")
    campaign = Campaign.from_mapping(payload)
    if (campaign.downstream["enabled"] or not campaign.imagenet["enabled"]
            or not campaign.imagenet["e1"] or not campaign.imagenet["e2"]):
        raise ValueError("fine-tuning requires ImageNet E1/E2 and disabled downstream")
    if set(campaign.backbones) - {"ff2048", "convnext_tiny", "vit_b16"}:
        raise ValueError("fine-tuning pilot supports ff2048, convnext_tiny and vit_b16")
    for key in ("epochs", "batch_size", "checkpoint_every", "log_every"):
        value = training[key]
        if key in ("epochs", "batch_size") and value == "source":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < (2 if key == "batch_size" else 1):
            raise ValueError(f"invalid finetune.{key}")
    for key in ("backbone_lr", "sae_lr", "classifier_lr", "ce_weight", "method_weight", "grad_clip", "weight_decay", "adam_eps"):
        value = training[key]
        if key in ("backbone_lr", "sae_lr", "classifier_lr", "weight_decay", "adam_eps") and value == "source":
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (key != "weight_decay" and value == 0)):
            raise ValueError(f"invalid finetune.{key}")
    if training["optimizer"] not in ("adam", "adamw"):
        raise ValueError("finetune.optimizer must be adam or adamw")
    betas = training["adam_betas"]
    if betas != "source" and (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not 0 <= b < 1 for b in betas)):
        raise ValueError("invalid finetune.adam_betas")
    if training["precision"] not in ("bf16", "float32"):
        raise ValueError("finetune.precision must be bf16 or float32")
    if campaign.execution["max_steps"]:
        raise ValueError("use --smoke for capped synthetic work; scientific fine-tuning uses full epochs")
    return FineTuneCampaign(campaign, training)
