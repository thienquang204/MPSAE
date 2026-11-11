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

    def list_repo_files(self, repo, repo_type, revision=None):
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
    from mpsae.config import BACKBONES
    from mpsae.io import atomic_json, canonical_hash, save_npz
    folder = folder_name("sop", "rn18")
    directory = root / folder
    ids = ["sop:a/1.JPG", "sop:a/2.JPG", "sop:b/3.JPG", "sop:b/4.JPG"]
    digest = save_npz(directory / "test.npz", data=np.ones((4, BACKBONES['rn18'].dim)), labels=[0, 0, 1, 1])
    atomic_json(directory / "test.npz.json", {"sha256": digest, "ids_sha256": canonical_hash(ids),
                                               "weights": WEIGHTS})
    atomic_json(directory / "test.ids.json", ids)
    atomic_json(directory / "bundle.json", {"dataset": "sop", "protocol": "sop_test", "backbone": "rn18",
                                            "gallery": "test", "query": "test", "exclude_self": True,
                                            "identity": {"x": 1}, "manifest": {"extraction": {
                                                "weights": WEIGHTS, "preprocessing": {
                                                    "mean": [0.5] * 3, "std": [0.5] * 3, "center_crop": 224}}}})
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
    assert bundle.gallery is bundle.query and bundle.gallery.rows == 4
    assert list(bundle.gallery.labels) == [0, 0, 1, 1]
    assert bundle.gallery.path.resolve().is_relative_to(campaign.resolve())  # receipts stay inside the workspace
    bundle.gallery.features._mmap.close()


def test_download_only_reports_all_missing_before_any_download(tmp_path, monkeypatch):
    api = FakeApi()
    hub = Hub(api, "me/MPSAE-embeddings")
    api.store["imagenet_rn18_512/folder.json"] = b"{}"
    def forbidden(*args, **kwargs):
        pytest.fail("download/build/upload must not start with an incomplete selection")
    monkeypatch.setattr(hub, "pull", forbidden)
    monkeypatch.setattr(hub, "push", forbidden)
    with pytest.raises(FileNotFoundError) as error:
        sync(selection(["imagenet", "sop"], ["rn18", "ff2048"]), tmp_path, tmp_path, hub,
             download_only=True, builders={"imagenet": forbidden, "downstream": forbidden})
    assert "imagenet_ff2048_2048" in str(error.value)
    assert "sop_rn18_512" in str(error.value)
    assert "sop_ff2048_2048" in str(error.value)
    assert api.created == []


def test_download_only_uses_local_only_and_pulls_public_without_writes(tmp_path, monkeypatch):
    api = FakeApi()
    api.private = False
    hub = Hub(api, "Lukita/MPSAE-embeddings")
    folder = make_folder(tmp_path, "imagenet", "rn18")
    remote = folder_name("sop", "rn18")
    api.store[f"{remote}/folder.json"] = b"{}"
    pulls = []
    monkeypatch.setattr(hub, "pull", lambda name, root: pulls.append(name))
    done = sync(selection(["imagenet", "sop"], ["rn18"]), tmp_path, tmp_path, hub, download_only=True)
    assert done == {folder: "used local", remote: "pulled"}
    assert pulls == [remote]
    assert api.created == []
    (tmp_path / folder / "1K_train_x.npz").write_bytes(b"corrupt!")
    with pytest.raises(RuntimeError, match="not using"):
        sync(selection(["imagenet"], ["rn18"]), tmp_path, tmp_path, hub, download_only=True)


def test_download_and_manifest_use_the_listed_revision(tmp_path, monkeypatch):
    import huggingface_hub
    api = FakeApi()
    folder = make_folder(tmp_path, "imagenet", "rn18")
    api.store[f"{folder}/folder.json"] = b"{}"
    hub = Hub(api, "me/MPSAE-embeddings")
    assert hub.complete(folder)
    api.repo_info = lambda *args, **kwargs: SimpleNamespace(sha="d" * 40)
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: (calls.append(kwargs), tmp_path / folder / FOLDER_MANIFEST)[1])
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: calls.append(kwargs))
    hub.manifest(folder)
    hub.pull(folder, tmp_path)
    assert [call["revision"] for call in calls] == ["a" * 40, "a" * 40]


def test_public_repo_connection_does_not_require_token(monkeypatch):
    import huggingface_hub
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: calls.append(kwargs))
    assert Hub.connect().token is None
    assert calls == [{"token": None}]


def test_cli_loads_custom_env_before_workspace_and_repo_defaults(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "custom.env"
    env_file.write_text(f"MPSAE_WORKSPACE={tmp_path.as_posix()}/features\n"
                        "MPSAE_EMBEDDINGS_REPO=fixture/embeddings\n", encoding="utf-8")
    for name in ("MPSAE_WORKSPACE", "MPSAE_EMBEDDINGS_REPO"):
        # Restore the process environment after the CLI's dotenv loader writes it.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    repos = []
    def connect(repo):
        repos.append(repo)
        return Hub(FakeApi(), repo)
    monkeypatch.setattr(Hub, "connect", connect)
    assert hub_module.main(["status", "--env-file", str(env_file), "--datasets", "eurosat",
                            "--backbones", "rn18"]) == 0
    assert repos == ["fixture/embeddings"]
    assert str(tmp_path / "features" / "embeddings-hub") in capsys.readouterr().out


def test_evaluation_only_selection_does_not_require_imagenet_archives():
    from mpsae.benchmark.config import Campaign
    campaign = Campaign.from_mapping({"backbones": ["rn18"], "imagenet": {"enabled": False}})
    datasets, backbones = hub_module.campaign_selection(campaign)
    assert "imagenet" not in datasets and backbones == ["rn18"]


def test_cli_requests_only_selected_dataset_backbone_pairs(tmp_path, monkeypatch, capsys):
    config = tmp_path / "campaign.json"
    config.write_text(json.dumps({"backbones": ["rn18", "vit_b16"], "downstream": {
        "datasets": ["eurosat", "sop"], "backbones": {"sop": ["rn18"]}},
        "execution": {"storage_mode": "direct_v1", "precomputed_only": True}}))
    monkeypatch.setattr(Hub, "connect", lambda repo: Hub(FakeApi(), repo))
    assert hub_module.main(["status", "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "imagenet_vit_b16_768" in output
    assert "eurosat_vit_b16_768" in output
    assert "sop_rn18_512" in output
    assert "sop_vit_b16_768" not in output


def test_target_runner_keeps_dataset_with_selected_backbone(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from unittest.mock import Mock
    from mpsae import train
    from mpsae.benchmark import data, extract, run, storage
    from mpsae.benchmark.config import Campaign
    campaign = Campaign.from_mapping({"backbones": ["rn18", "vit_b16"],
        "methods": ["csr"], "seeds": [1227], "downstream": {
            "datasets": ["sop"], "backbones": {"sop": ["rn18"]}},
        "execution": {"storage_mode": "direct_v1", "precomputed_only": True}})
    make_downstream_folder(tmp_path / "library")
    hub_module.link_campaign(tmp_path / "library", tmp_path, [("sop", "rn18")])
    def checkpoint(cell, *args, **kwargs):
        assert cell.backbone == "rn18"
        return {"backbone_weights": WEIGHTS}
    def forbidden(*args, **kwargs):
        pytest.fail("selected precomputed bundles must not extract or download images")
    monkeypatch.setattr(run, "checkpoint_record", checkpoint)
    monkeypatch.setattr(run, "verify_feature_provenance", lambda *args: None)
    evaluations = []
    def evaluate(*args, **kwargs):
        evaluations.append(kwargs)
        return {"backbone": args[1].name}
    monkeypatch.setattr(run, "evaluate_point", evaluate)
    monkeypatch.setattr(train, "load_fitted", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(data, "load_dataset_bundle", forbidden)
    monkeypatch.setattr(extract, "extract_bundle", forbidden)
    catalog = {"evaluations": []}
    storage.run_targets(campaign, tmp_path, catalog, device="cpu", workers=0, batch_size=2,
                        observer=None, task=lambda name, op: op())
    assert len(catalog["evaluations"]) == 4  # dense + ResNet inference K=16/32/256
    assert {options["k"] for options in evaluations if "k" in options} == {16, 32, 256}
    assert all(options["task"] == "retrieval" and options["knn"] is None for options in evaluations)
    assert {row["backbone"] for row in catalog["evaluations"]} == {"rn18"}


def test_linked_bundle_rejects_wrong_width_and_pinned_dataset_revision(tmp_path):
    from mpsae.io import atomic_json, read_json
    make_downstream_folder(tmp_path / "library")
    hub_module.link_campaign(tmp_path / "library", tmp_path / "campaign", [("sop", "rn18")])
    assert hub_module.load_campaign_hub_bundle(tmp_path / "campaign", "sop", "rn18", WEIGHTS,
                                               tmp_path / "cache", revision="a" * 40) is None
    directory = tmp_path / "campaign" / "benchmarks" / "sop" / "hub" / "rn18"
    manifest = read_json(directory / FOLDER_MANIFEST)
    manifest["dim"] = 768
    atomic_json(directory / FOLDER_MANIFEST, manifest)
    with pytest.raises(RuntimeError, match="dimension"):
        hub_module.load_campaign_hub_bundle(tmp_path / "campaign", "sop", "rn18", WEIGHTS, tmp_path / "cache")


def test_precomputed_campaign_trains_and_evaluates_without_extractors(tmp_path, monkeypatch, small_preset):
    pytest.importorskip("torch")
    import numpy as np
    from dataclasses import replace
    from mpsae.benchmark import data, extract
    from mpsae.benchmark.config import Campaign
    from mpsae.benchmark.run import run_campaign
    from mpsae.config import BACKBONES
    from mpsae.data import imagenet
    from mpsae.embed import direct
    from mpsae.embed.archive import finalize_archive
    from mpsae.io import atomic_json, canonical_hash, sha256_file

    monkeypatch.setitem(BACKBONES, "rn18", replace(BACKBONES["rn18"], dim=12,
        preset=small_preset, k_list=(2,), e2_anchors=(2,)))
    monkeypatch.setattr(imagenet, "EXPECTED_ROWS", {"train": 16, "validation": 8})
    def forbidden(*args, **kwargs):
        pytest.fail("a precomputed campaign must never download images, load backbones, or extract features")
    monkeypatch.setattr(imagenet, "prepare_imagenet", forbidden)
    monkeypatch.setattr(direct, "extract_source", forbidden)
    monkeypatch.setattr(direct, "preflight_backbones", forbidden)
    monkeypatch.setattr(data, "load_dataset_bundle", forbidden)
    monkeypatch.setattr(extract, "extract_bundle", forbidden)

    campaign = Campaign.from_mapping({"name": "precomputed_test", "backbones": ["rn18"],
        "methods": ["csr"], "seeds": [1227], "imagenet": {"prepare": False, "train_k": [2],
            "e2": False, "linear_probe": False, "timing": False},
        "downstream": {"datasets": ["sop"], "train_k": [2], "eval_k": [2]},
        "execution": {"storage_mode": "direct_v1", "precomputed_only": True,
            "cleanup_raw": False, "max_steps": 1, "continue_on_error": False}})
    library = tmp_path / "embeddings-hub"
    source = library / folder_name("imagenet", "rn18")
    for split, rows in (("train", 16), ("val", 8)):
        path = source / f"1K_{split}_rn18.npz"
        ids = [f"{split}/{index}.jpg" for index in range(rows)]
        ids_path = path.with_suffix(".ids.json")
        atomic_json(ids_path, ids)
        finalize_archive(path, np.random.default_rng(rows).normal(size=(rows, 12)).astype(np.float32),
            np.arange(rows) % 2, manifest={"recipe": direct.RECIPE, "backbone": "rn18", "weights": WEIGHTS,
                "input_size": 224, "normalize": {"mean": [0.5] * 3, "std": [0.5] * 3},
                "ids_sha256": canonical_hash(ids), "sample_ids_sha256": sha256_file(ids_path)})
    write_folder_manifest(library, source.name, dataset="imagenet", backbone="rn18", recipe=direct.RECIPE,
                          weights=WEIGHTS)
    make_downstream_folder(library)
    linked = hub_module.link_campaign(library, tmp_path / "direct-v1", selection(["imagenet", "sop"], ["rn18"]))
    assert len(linked) == 2
    result = run_campaign(campaign, tmp_path, device="cpu", workers=0, embed_batch_size=2)
    assert result["complete"] is True
    assert result["completed"]["source_cells"] == 1
    assert not (tmp_path / "imagenet").exists()
    assert not (tmp_path / "direct-v1" / "cache" / "raw_datasets").exists()


@pytest.mark.parametrize("fault", ["missing", "weights", "revision"])
def test_precomputed_targets_never_fall_back_to_extraction(tmp_path, monkeypatch, fault):
    pytest.importorskip("torch")
    from mpsae.benchmark import data, extract, run, storage
    from mpsae.benchmark.config import Campaign
    campaign = Campaign.from_mapping({"backbones": ["rn18"], "methods": ["csr"], "seeds": [1227],
        "downstream": {"datasets": ["sop"]},
        "execution": {"storage_mode": "direct_v1", "precomputed_only": True}})
    if fault != "missing":
        make_downstream_folder(tmp_path / "library")
        hub_module.link_campaign(tmp_path / "library", tmp_path, [("sop", "rn18")])
    weights = {**WEIGHTS, "sha256": "d" * 64} if fault == "weights" else WEIGHTS
    if fault == "revision":
        campaign.downstream["revisions"]["sop"] = "e" * 40
    monkeypatch.setattr(run, "checkpoint_record", lambda *args, **kwargs: {"backbone_weights": weights})
    def forbidden(*args, **kwargs):
        pytest.fail("missing/incompatible precomputed features must not trigger image downloads or extraction")
    monkeypatch.setattr(data, "load_dataset_bundle", forbidden)
    monkeypatch.setattr(extract, "extract_bundle", forbidden)
    with pytest.raises(FileNotFoundError, match="Extraction is disabled"):
        storage.run_targets(campaign, tmp_path, {}, device="cpu", workers=0, batch_size=2,
                            observer=None, task=lambda name, op: op())


def test_precomputed_sources_never_fall_back_to_image_downloads(tmp_path, monkeypatch):
    from mpsae.benchmark import storage
    from mpsae.benchmark.config import Campaign
    from mpsae.data import imagenet
    campaign = Campaign.from_mapping({"backbones": ["rn18"],
        "execution": {"storage_mode": "direct_v1", "precomputed_only": True}})
    def forbidden(*args, **kwargs):
        pytest.fail("missing precomputed ImageNet embeddings must not trigger downloads")
    monkeypatch.setattr(imagenet, "prepare_imagenet", forbidden)
    with pytest.raises(FileNotFoundError, match="Missing precomputed ImageNet"):
        storage.prepare_sources(campaign, tmp_path, tmp_path / "direct-v1", device="cpu", workers=0,
                                batch_size=2, observer=None, task=lambda name, op: op())
