from io import BytesIO
import pickle

import numpy as np
import pytest
from PIL import Image

from mpsae.finetune import jpeg_store as store


def jpeg(value):
    handle = BytesIO()
    Image.fromarray(np.full((10, 13, 3), value, dtype=np.uint8)).save(handle, format="JPEG")
    return handle.getvalue()


def test_folder_packs_preserve_pixels_order_and_reuse_without_source(tmp_path, monkeypatch):
    from torchvision.datasets import ImageFolder
    root, dest = tmp_path / "raw", tmp_path / "pack"
    for label in (2, 1):
        folder = root / f"n{label:08d}"
        folder.mkdir(parents=True)
        for name in ("z", "a", "d"):
            (folder / f"{name}.JPEG").write_bytes(jpeg(label * 30 + ord(name)))
    expected = ImageFolder(root)
    packed = store.pack_folder(root, dest, expected=6, pack_rows=2)
    assert packed.class_to_idx == expected.class_to_idx
    for i in range(6):
        image, label = packed[i]
        reference, reference_label = expected[i]
        assert label == reference_label
        np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))
    # Spawn workers receive no inherited file handles or giant pickled index.
    state = pickle.dumps(packed)
    assert len(state) < 10000
    restored = pickle.loads(state)
    np.testing.assert_array_equal(np.asarray(restored[2][0]), np.asarray(packed[2][0]))
    restored.close()
    packed.close()
    monkeypatch.setattr("mpsae.data.imagenet.imagenet_folder_stats",
                        lambda *a: pytest.fail("completed immutable cache rescanned originals"))
    reused = store.pack_folder(root / "unavailable", dest, expected=6)
    assert len(reused) == 6
    reused.close()
    path = dest / "pack-00000.jpegpack"
    with path.open("r+b") as handle:
        handle.truncate(3)
    with pytest.raises(RuntimeError, match="incomplete/corrupt"):
        store.PackedImages(dest)


def test_uncommitted_pack_replays_and_checksum_checks(tmp_path):
    rows = [(jpeg(4), 0, "a.JPEG"), (jpeg(8), 0, "b.JPEG")]
    def interrupted():
        yield rows[0]
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        store._write_pack(tmp_path, 0, interrupted(), "source")
    assert not (tmp_path / "pack-00000.json").exists()
    record = store._write_pack(tmp_path, 0, rows, "source")
    store._publish(tmp_path, [record], ["n00000001"], {"revision": "pinned"}, 2)
    packed = store.PackedImages(tmp_path, verify=True)
    packed.close()
    with pytest.raises(RuntimeError, match="source changed"):
        store._write_pack(tmp_path, 0, rows, "different")
    path = tmp_path / "pack-00000.jpegpack"
    with path.open("r+b") as handle:
        handle.write(b"xxxx")
    with pytest.raises(RuntimeError, match="data checksum"):
        store.PackedImages(tmp_path, verify=True)


def test_direct_hf_packing_pins_source_resumes_and_never_materialises_raw_files(tmp_path, monkeypatch):
    from mpsae.data import imagenet as source
    from mpsae.embed import archive
    classes = ["n00000001", "n00000002"]
    shards = {"train-0.parquet": [(jpeg(20), 1, "n00000002_z.JPEG"), (jpeg(30), 0, "n00000001_z.JPEG")],
              "train-1.parquet": [(jpeg(40), 0, "n00000001_a.JPEG"), (jpeg(50), 1, "n00000002_a.JPEG")],
              "validation-0.parquet": [(jpeg(60), 1, "val_z.JPEG"), (jpeg(70), 0, "val_a.JPEG")]}
    monkeypatch.setattr(source, "CLASSES", 2)
    monkeypatch.setattr(source, "EXPECTED_ROWS", {"train": 4, "validation": 2})
    monkeypatch.setattr(source, "_synsets", lambda *a, **kw: classes)
    monkeypatch.setattr(archive, "pinned_release", lambda *a, **kw:
                        {"revision": "a" * 40, "files": {name: {} for name in shards}})
    downloads = []
    def fetch(release, name, **kwargs):
        assert release["revision"] == "a" * 40
        downloads.append(name)
        path = tmp_path / name
        path.write_bytes(b"placeholder")
        return path, "digest"
    monkeypatch.setattr(archive, "hub_file", fetch)
    interrupted = [True]
    def rows(path):
        if path.name == "train-1.parquet" and interrupted[0]:
            interrupted[0] = False
            raise KeyboardInterrupt()
        values = shards[path.name]
        yield ([{"bytes": data, "path": name} for data, _, name in values],
               [label for _, label, _ in values])
    monkeypatch.setattr(source, "_rows", rows)
    with pytest.raises(KeyboardInterrupt):
        store.prepare_packs(tmp_path / "images", download=True)
    datasets, identity = store.prepare_packs(tmp_path / "images", download=True)
    assert downloads.count("train-0.parquet") == 1
    assert not (tmp_path / "images" / "train").exists()
    assert [datasets["train"][i][1] for i in range(4)] == [0, 0, 1, 1]
    for i, value in enumerate((40, 30, 50, 20)):
        np.testing.assert_array_equal(np.asarray(datasets["train"][i][0]),
                                      np.asarray(Image.open(BytesIO(jpeg(value))).convert("RGB")))
    for dataset in datasets.values():
        dataset.close()
    monkeypatch.setattr(archive, "pinned_release", lambda *a, **kw: pytest.fail("offline reuse contacted HF"))
    reused, second_identity = store.prepare_packs(tmp_path / "images")
    assert identity == second_identity
    for dataset in reused.values():
        dataset.close()
