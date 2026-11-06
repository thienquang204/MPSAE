"""Loading the frozen embedding archives that every fit and evaluation reads.

One archive pair per backbone -- ``1K_train_<name>.npz`` and
``1K_val_<name>.npz``, each holding ``data`` (float32) and ``labels`` (int64).
Training uses the whole train split as both the fitting set and the retrieval
gallery, and the 50k validation split as the query set, which is the protocol
of CSR Table 4 and of ``CSRv2/image/faiss_nn.py``.

Nothing is held back. The previous pipeline reserved 1,024 "anchor" rows for a
landmark variant that no longer exists; keeping that reservation would have
shrunk the gallery and made the dense 1-NN gate unable to reach the published
71.19.

Residency matters more than it looks. FF2048's train split is 10.5 GiB in
float32, which fits comfortably in an 80 GiB accelerator, and keeping it
resident turns every step into pure device work -- the difference between a
run that takes a minute and one that is bound by random reads into a
memory-mapped file. ``choose_residency`` picks the best option that actually
fits and says why in the log.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import Backbone
from .io import canonical_hash, human_bytes, open_npz_memmap, read_json
from .logs import get_logger

SPLIT_FILES = {"train": "1K_train_{name}.npz", "validation": "1K_val_{name}.npz"}
#: Leave headroom for the model, Adam state, activations and the OT kernels.
RESIDENT_VRAM_HEADROOM = 12 << 30


@dataclass
class Split:
    """One split: features (array-like), integer labels, provenance."""

    name: str
    features: Any
    labels: Any
    manifest: dict[str, Any]

    @property
    def rows(self) -> int:
        return int(len(self.labels))

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])


@dataclass
class Bundle:
    """Both splits of one backbone, plus the identity they are pinned by."""

    backbone: str
    train: Split
    validation: Split
    identity: dict[str, Any]

    @property
    def dim(self) -> int:
        return self.train.dim

    @property
    def digest(self) -> str:
        return canonical_hash(self.identity)

    def describe(self) -> dict[str, Any]:
        return {"backbone": self.backbone, "dim": self.dim,
                "train_rows": self.train.rows, "validation_rows": self.validation.rows,
                "digest": self.digest}


def archive_path(directory: str | Path, spec: Backbone, split: str) -> Path:
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
    return Path(directory) / spec.name / SPLIT_FILES[split].format(name=spec.name)


def load_bundle(spec: Backbone, embeddings_dir: str | Path, cache_dir: str | Path,
                *, splits=("train", "validation")) -> Bundle:
    """Memory-map both archives and verify them against their manifests."""
    log = get_logger("features")
    loaded: dict[str, Split] = {}
    identity: dict[str, Any] = {"backbone": spec.name, "dim": spec.dim}
    for split in splits:
        path = archive_path(embeddings_dir, spec, split)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {split} embeddings for {spec.name}: {path}. Run the embedding "
                f"stage (pipeline stage S3/S4/S4b) or download the published archive.")
        manifest = read_json(path.with_suffix(".json")) or {}
        expected = manifest.get("sha256")
        features, labels, metadata = open_npz_memmap(path, Path(cache_dir) / "features",
                                                     expected_sha256=expected)
        if features.shape[1] != spec.dim:
            raise ValueError(f"{path}: expected dim {spec.dim}, found {features.shape[1]}")
        if manifest and manifest.get("rows") not in (None, metadata["rows"]):
            raise ValueError(f"{path}: manifest claims {manifest['rows']} rows, found {metadata['rows']}")
        if metadata["label_key"] != "labels":
            # The released CSRv2 loader reads 'labels' while every released
            # producer writes 'label'; an archive with only 'label' silently
            # collapses csrv2's supervised contrastive term to zero.
            log.warning("%s stores labels under %r, not 'labels' -- csrv2 supervision "
                        "would be degenerate on the released archives", path.name,
                        metadata["label_key"])
        loaded[split] = Split(split, features, labels, {**manifest, **metadata})
        identity[split] = {"sha256": metadata["sha256"], "rows": metadata["rows"],
                           "recipe": manifest.get("recipe"), "source": manifest.get("source")}
        log.info("%s/%s: %s rows x %d (%s)", spec.name, split, f"{metadata['rows']:,}",
                 metadata["dim"], human_bytes(metadata["rows"] * metadata["dim"] * 4))
    if "train" not in loaded:
        raise ValueError("a bundle always needs the train split")
    return Bundle(spec.name, loaded["train"], loaded.get("validation", loaded["train"]), identity)


# ---------------------------------------------------------------- residency

def choose_residency(rows: int, dim: int, device, *, requested: str = "auto") -> str:
    """Pick ``"cuda"``, ``"cpu"`` or ``"mmap"`` for the fitting matrix."""
    import torch

    log = get_logger("features")
    needed = rows * dim * 4
    if requested not in ("auto", "cuda", "cpu", "mmap"):
        raise ValueError("residency must be auto, cuda, cpu or mmap")
    if requested != "auto":
        return requested
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(device)
        if needed + RESIDENT_VRAM_HEADROOM < free:
            log.info("fitting matrix resident on %s (%s, %s free)", device,
                     human_bytes(needed), human_bytes(free))
            return "cuda"
        log.info("fitting matrix too large for %s (%s needed, %s free, %s headroom)",
                 device, human_bytes(needed), human_bytes(free),
                 human_bytes(RESIDENT_VRAM_HEADROOM))
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:
        import os

        available = (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                     if hasattr(os, "sysconf") else 0)
    if needed * 1.2 < available:
        log.info("fitting matrix held in host RAM (%s, %s available)",
                 human_bytes(needed), human_bytes(available))
        return "cpu"
    log.info("fitting matrix left memory-mapped (%s needed, %s available)",
             human_bytes(needed), human_bytes(available))
    return "mmap"


class BatchSource:
    """Shuffled fixed-size batches over a feature matrix.

    Mirrors ``DataLoader(..., shuffle=True)`` with the released default
    ``drop_last=False``, so the final short batch of each epoch is kept -- it
    is part of the 313-step epoch that gives FF2048 its 3,130 updates.

    The permutation is a pure function of ``(seed, epoch)``, and resuming
    mid-epoch replays the same permutation and skips forward, so a restart
    cannot reshuffle the remainder of an epoch.
    """

    def __init__(self, features, labels, *, batch_size: int, seed: int,
                 device, residency: str = "auto", with_labels: bool = True) -> None:
        import numpy as np
        import torch

        if batch_size < 2:
            raise ValueError("batch_size must be at least two")
        self.rows, self.dim = int(len(features)), int(features.shape[1])
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.device = torch.device(device)
        self.with_labels = with_labels
        self.residency = choose_residency(self.rows, self.dim, self.device, requested=residency)
        self._labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
        if self.residency == "cuda":
            holder = torch.empty((self.rows, self.dim), dtype=torch.float32, device=self.device)
            step = max(1, (256 << 20) // max(1, self.dim * 4))
            for start in range(0, self.rows, step):
                chunk = np.ascontiguousarray(features[start:start + step], dtype=np.float32)
                holder[start:start + len(chunk)] = torch.from_numpy(chunk).to(self.device)
            self._features = holder
            self._labels = self._labels.to(self.device)
        elif self.residency == "cpu":
            self._features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        else:
            self._features = features

    @property
    def batches_per_epoch(self) -> int:
        return -(-self.rows // self.batch_size)

    def permutation(self, epoch: int):
        import numpy as np

        # SeedSequence keeps epochs independent yet fully determined by the
        # run's seed, so a resumed epoch replays its exact order.
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        return generator.permutation(self.rows)

    def epoch(self, epoch: int, *, skip_batches: int = 0) -> Iterator[tuple]:
        import numpy as np
        import torch

        order = self.permutation(epoch)
        for index in range(skip_batches, self.batches_per_epoch):
            ids = order[index * self.batch_size:(index + 1) * self.batch_size]
            if self.residency == "mmap":
                rows = np.ascontiguousarray(self._features[np.sort(ids)], dtype=np.float32)
                # Restore the shuffled order after the locality-friendly read:
                # OT groups are contiguous slices of the batch, so row order
                # inside a batch is part of the objective.
                inverse = np.argsort(np.argsort(ids))
                batch = torch.from_numpy(rows[inverse]).to(self.device, non_blocking=True)
                labels = self._labels[torch.from_numpy(ids)].to(self.device, non_blocking=True)
            else:
                selector = torch.from_numpy(ids).to(self._features.device)
                batch = self._features[selector]
                if batch.device != self.device:
                    batch = batch.to(self.device, non_blocking=True)
                labels = self._labels[selector.to(self._labels.device)]
                if labels.device != self.device:
                    labels = labels.to(self.device, non_blocking=True)
            yield (index, batch, labels if self.with_labels else None)

    def release(self) -> None:
        self._features = None
        self._labels = None
