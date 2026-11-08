"""mpsae.embeddings_hub: folder names, local/Hub/build decisions, push verification (no torch, no network)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mpsae import embeddings_hub as hub_module
from mpsae.embeddings_hub import (FOLDER_MANIFEST, Hub, folder_name, local_complete, plan, selection, sync,
                                  write_folder_manifest)


class FakeApi:
    """The HfApi calls Hub makes, backed by a dict {repo path: bytes}."""

    def __init__(self):
        self.store: dict[str, bytes] = {}
        self.private = True
        self.created = []

    def list_repo_files(self, repo, repo_type):
        return sorted(self.store)

    def create_repo(self, repo, **options):
        self.created.append((repo, options))

    def repo_info(self, repo, repo_type):
        return SimpleNamespace(private=self.private, sha="a" * 40)

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, commit_message):
        for name in allow_patterns:
            self.store[f"{path_in_repo}/{name}"] = (Path(folder_path) / name).read_bytes()

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.store[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def get_paths_info(self, repo, paths, repo_type, revision):
        entries = []
        for path in paths:
            if path in self.store:
                data = self.store[path]
                entries.append(SimpleNamespace(path=path, size=len(data),
                                               lfs={"sha256": hashlib.sha256(data).hexdigest()}, blob_id=None))
        return entries


def make_folder(root: Path, dataset: str, backbone: str, payload: bytes = b"features") -> str:
    folder = folder_name(dataset, backbone)
    (root / folder).mkdir(parents=True)
    (root / folder / "1K_train_x.npz").write_bytes(payload)
    (root / folder / "1K_train_x.npz.json").write_text("{}")
    write_folder_manifest(root, folder, dataset=dataset, backbone=backbone, recipe="test",
                          weights={"repository": "r", "revision": "b" * 40, "file": "f", "sha256": "c" * 64})
    return folder


def test_folder_names_follow_dataset_backbone_dim():
    assert folder_name("imagenet", "ff2048") == "imagenet_ff2048_2048"
    assert folder_name("sop", "rn18") == "sop_rn18_512"
    assert folder_name("eurosat", "convnext_tiny") == "eurosat_convnext_tiny_768"
    assert folder_name("flowers102", "dinov2_vits14") == "flowers102_dinov2_vits14_384"
    pairs = selection(["sop", "imagenet"], ["rn18"])
    assert pairs == [("imagenet", "rn18"), ("sop", "rn18")]          # ImageNet always first
    assert len(selection()) == 4 * 5
    with pytest.raises(ValueError):
        selection(["dtd"], None)


def test_local_completeness_needs_every_listed_file(tmp_path):
    folder = make_folder(tmp_path, "imagenet", "rn18")
    assert local_complete(tmp_path, folder, verify_hashes=True)
    (tmp_path / folder / "1K_train_x.npz").write_bytes(b"other!!!")   # same size, other bytes
    assert local_complete(tmp_path, folder)
    assert not local_complete(tmp_path, folder, verify_hashes=True)
    (tmp_path / folder / "1K_train_x.npz").unlink()
    assert not local_complete(tmp_path, folder)
    assert not local_complete(tmp_path, "imagenet_ff2048_2048")


def test_push_uploads_manifest_last_and_verifies_remote_hashes(tmp_path):
    api = FakeApi()
    hub = Hub(api, "me/MPSAE-embeddings")
    folder = make_folder(tmp_path, "imagenet", "rn18")
    assert not hub.complete(folder)
    hub.push(folder, tmp_path)
    assert api.created == [("me/MPSAE-embeddings", {"repo_type": "dataset", "private": True, "exist_ok": True})]
    assert set(api.store) == {f"{folder}/1K_train_x.npz", f"{folder}/1K_train_x.npz.json", f"{folder}/{FOLDER_MANIFEST}"}
    assert hub.complete(folder)
    api.store[f"{folder}/1K_train_x.npz"] = b"corrupt!"
    with pytest.raises(RuntimeError, match="hash differs"):
        hub.verify(folder, tmp_path)
    api.private = False
    with pytest.raises(RuntimeError, match="public"):
        hub.push(folder, tmp_path)


def test_sync_covers_the_four_local_hub_cases(tmp_path, monkeypatch):
    api = FakeApi()
    hub = Hub(api, "me/MPSAE-embeddings")
    root, remote_root = tmp_path / "local", tmp_path / "remote"
    both = make_folder(root, "imagenet", "ff2048")                # local and Hub -> use local
    hub.push(both, root)
    local_only = make_folder(root, "imagenet", "rn18")            # local only -> push
    hub_only = make_folder(remote_root, "sop", "ff2048")          # Hub only -> pull
    hub.push(hub_only, remote_root)
    pushes = []
    original_push = hub.push
    monkeypatch.setattr(hub, "push", lambda folder, where: (pushes.append(folder), original_push(folder, where)))

    def pull(folder, local):
        for path in api.list_repo_files(hub.repo, "dataset"):
            if path.startswith(folder + "/"):
                (local / path).parent.mkdir(parents=True, exist_ok=True)
                (local / path).write_bytes(api.store[path])
        assert local_complete(local, folder, verify_hashes=True)
    monkeypatch.setattr(hub, "pull", pull)

    calls = []

    def build_imagenet(backbones, root, workspace, hub, **options):
        calls.append(("imagenet", tuple(backbones)))

    def build_downstream(dataset, backbones, root, hub, **options):
        calls.append((dataset, tuple(backbones)))

    pairs = selection(["imagenet", "sop"], ["ff2048", "rn18"])
    assert plan(pairs, root, hub) == {"imagenet_ff2048_2048": "use", "imagenet_rn18_512": "push",
                                      "sop_ff2048_2048": "pull", "sop_rn18_512": "build"}
    done = sync(pairs, root, tmp_path / "work", hub, builders={"imagenet": build_imagenet,
                                                              "downstream": build_downstream})
    assert done == {"imagenet_ff2048_2048": "used local", "imagenet_rn18_512": "pushed",
                    "sop_ff2048_2048": "pulled", "sop_rn18_512": "built"}
    assert pushes == ["imagenet_rn18_512"]                        # nothing re-uploaded
    assert hub.complete("imagenet_rn18_512")
    assert calls == [("sop", ("rn18",))]                          # only the missing one is built
    assert set(plan(pairs, root, hub).values()) == {"use", "build"}  # the fake builder built nothing


def test_a_local_file_that_changed_is_not_pushed(tmp_path):
    hub = Hub(FakeApi(), "me/MPSAE-embeddings")
    folder = make_folder(tmp_path, "imagenet", "rn18")
    (tmp_path / folder / "1K_train_x.npz").write_bytes(b"changed!")   # same size, other bytes
    with pytest.raises(RuntimeError, match="not pushing"):
        sync(selection(["imagenet"], ["rn18"]), tmp_path, tmp_path / "work", hub)
    assert not hub.complete(folder)


def test_downstream_uses_the_weights_recorded_with_the_imagenet_embeddings(tmp_path):
    api = FakeApi()
    hub = Hub(api, "me/MPSAE-embeddings")
    with pytest.raises(RuntimeError, match="build imagenet_rn18_512 first"):
        hub_module._imagenet_weights("rn18", tmp_path, hub)
    make_folder(tmp_path, "imagenet", "rn18")
    weights = hub_module._imagenet_weights("rn18", tmp_path, hub)
    assert weights["revision"] == "b" * 40
    manifest = json.loads((tmp_path / "imagenet_rn18_512" / FOLDER_MANIFEST).read_text())
    assert manifest["dim"] == 512 and manifest["dataset"] == "imagenet"


WEIGHTS = {"repository": "timm/resnet18.a1_in1k", "revision": "b" * 40, "file": "model.safetensors",
           "sha256": "c" * 64}


def make_downstream_folder(root: Path) -> str:
    """A sop_rn18_512 folder laid out as build_downstream writes it."""
    import numpy as np
    from mpsae.io import atomic_json, canonical_hash, save_npz
    folder = folder_name("sop", "rn18")
    directory = root / folder
    ids = ["sop:a/1.JPG", "sop:a/2.JPG", "sop:b/3.JPG"]
    digest = save_npz(directory / "test.npz", data=np.ones((3, 512)), labels=[0, 0, 1])
    atomic_json(directory / "test.npz.json", {"sha256": digest, "ids_sha256": canonical_hash(ids),
                                               "weights": WEIGHTS})
    atomic_json(directory / "test.ids.json", ids)
    atomic_json(directory / "bundle.json", {"dataset": "sop", "protocol": "sop_test", "backbone": "rn18",
                                            "gallery": "test", "query": "test", "exclude_self": True,
                                            "identity": {"x": 1}, "manifest": {"extraction": {"weights": WEIGHTS}}})
    write_folder_manifest(root, folder, dataset="sop", backbone="rn18", recipe="test", weights=WEIGHTS)
    return folder


def test_campaign_reads_linked_folders_without_extra_copies(tmp_path):
    root, campaign = tmp_path / "embeddings-hub", tmp_path / "direct-v1"
    imagenet = make_folder(root, "imagenet", "rn18")
    (root / imagenet / "1K_val_rn18.npz").write_bytes(b"val")
    write_folder_manifest(root, imagenet, dataset="imagenet", backbone="rn18", recipe="test", weights=WEIGHTS)
    make_downstream_folder(root)
    linked = hub_module.link_campaign(root, campaign, selection(["imagenet", "sop"], ["rn18"]))
    assert linked == ["imagenet_rn18_512", "sop_rn18_512"]
    target = campaign / "embeddings" / "rn18" / "1K_val_rn18.npz"
    assert target.samefile(root / imagenet / "1K_val_rn18.npz")        # hard link, no copy
    assert not (campaign / "embeddings" / "rn18" / FOLDER_MANIFEST).exists()
    assert hub_module.link_campaign(root, campaign, selection(["imagenet"], ["rn18"])) == ["imagenet_rn18_512"]

    cache = campaign / "cache"
    other = {**WEIGHTS, "sha256": "d" * 64}
    assert hub_module.load_campaign_hub_bundle(campaign, "sop", "rn18", other, cache) is None
    assert hub_module.load_campaign_hub_bundle(campaign, "sop", "rn18", None, cache) is None
    bundle = hub_module.load_campaign_hub_bundle(campaign, "sop", "rn18", {**WEIGHTS, "feature": "x"}, cache)
    assert (bundle.name, bundle.backbone, bundle.exclude_self) == ("sop", "rn18", True)
    assert bundle.gallery is bundle.query and bundle.gallery.rows == 3
    assert list(bundle.gallery.labels) == [0, 0, 1]
    assert bundle.gallery.path.resolve().is_relative_to(campaign.resolve())  # receipts stay inside the workspace
