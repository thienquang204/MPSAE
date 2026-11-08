"""Check graph-vs-pretraining metadata without Torch, downloads or CUDA."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from mpsae.config import BACKBONES
from mpsae.embed.csrv2_timm import _timm_model


class FrozenModel:
    def __init__(self, *, native_size, actual_size):
        self.pretrained_cfg = {"input_size": (3, native_size, native_size),
                               "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
        self.patch_embed = SimpleNamespace(img_size=actual_size)
        self.training = True
        self.requires_gradient = True
        self.device = None

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, enabled):
        self.requires_gradient = enabled
        return self

    def to(self, device):
        self.device = device
        return self


def fake_timm(monkeypatch, model, native_size):
    timm = ModuleType("timm")
    calls = []
    timm.get_pretrained_cfg = lambda tag: SimpleNamespace(input_size=(3, native_size, native_size))

    def create_model(tag, **kwargs):
        calls.append((tag, kwargs))
        return model

    timm.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", timm)
    return calls


def test_dinov2_override_checks_graph_and_preserves_pretraining_normalization(monkeypatch):
    model = FrozenModel(native_size=518, actual_size=(224, 224))
    calls = fake_timm(monkeypatch, model, 518)
    release = {"repository": "timm/vit_small_patch14_dinov2.lvd142m", "revision": "a" * 40}
    loaded, config, tag, override = _timm_model(BACKBONES["dinov2_vits14"], release, "cpu")
    assert loaded is model and override == 224
    assert config["input_size"] == (3, 518, 518)  # original checkpoint metadata
    assert config["mean"] == model.pretrained_cfg["mean"]
    assert calls[0][1]["img_size"] == 224
    assert calls[0][1]["pretrained_cfg_overlay"]["hf_hub_id"] == (
        release["repository"] + "@" + release["revision"])
    assert not model.training and not model.requires_gradient and model.device == "cpu"


@pytest.mark.parametrize("actual_size", [None, (518, 518), (224, 518)])
def test_ignored_or_unverifiable_override_fails_before_extraction(monkeypatch, actual_size):
    model = FrozenModel(native_size=518, actual_size=actual_size)
    fake_timm(monkeypatch, model, 518)
    release = {"repository": "timm/vit_small_patch14_dinov2.lvd142m", "revision": "a" * 40}
    with pytest.raises(RuntimeError, match="patch_embed.img_size"):
        _timm_model(BACKBONES["dinov2_vits14"], release, "cpu")


def test_native_resolution_models_do_not_receive_img_size_override(monkeypatch):
    model = FrozenModel(native_size=224, actual_size=None)
    calls = fake_timm(monkeypatch, model, 224)
    release = {"repository": "timm/resnet18.a1_in1k", "revision": "b" * 40}
    _, _, _, override = _timm_model(BACKBONES["rn18"], release, "cpu")
    assert override is None
    assert "img_size" not in calls[0][1]
