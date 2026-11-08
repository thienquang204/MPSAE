"""Pinned datasets and explicit relevance protocols for frozen retrieval.

The HF split called ``train`` is a retrieval gallery, never an instruction to
fit a downstream model.  SOP instead searches its test split against itself,
excluding the query by sample ID and comparing *item IDs*, not categories.
Imports stay light so a campaign can be planned without torch or datasets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import io
import numbers
from pathlib import Path
from typing import Any, Mapping

from ..io import atomic_json, canonical_hash, read_json

DATA_FORMAT = "frozen-retrieval-datasets-v1"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repository: str
    protocol: str
    gallery_split: str
    query_split: str
    relevance_key: str
    id_key: str | None
    expected_rows: Mapping[str, int]
    expected_classes: Mapping[str, int]

    @property
    def exclude_self(self) -> bool:
        return self.gallery_split == self.query_split


DATASETS = {
    "eurosat": DatasetSpec(
        "eurosat", "mteb/eurosat-rgb", "eurosat_hf_train_gallery_test_query_v1",
        "train", "test", "label", "image_id",
        {"train": 16200, "validation": 5400, "test": 5400},
        {"train": 10, "validation": 10, "test": 10}),
    "flowers102": DatasetSpec(
        "flowers102", "pufanyi/flowers102", "flowers102_hf80_10_10_v1",
        "train", "test", "label", None,
        {"train": 6551, "validation": 819, "test": 819},
        {"train": 102}),
    "sop": DatasetSpec(
        "sop", "nyris/stanford-online-products-v1", "sop_hf_test_leave_one_out_v1",
        "test", "test", "item_id", "image_path",
        {"train": 59551, "test": 60502}, {"train": 11318, "test": 11316}),
}


@dataclass
class DatasetSplit:
    name: str
    dataset: Any
    ids: list[str]
    labels: list[int]
    content_hashes: list[str] | None = None

    @property
    def rows(self) -> int:
        return len(self.ids)


@dataclass
class DatasetBundle:
    name: str
    protocol: str
    splits: dict[str, DatasetSplit]
    gallery_split: str
    query_split: str
    exclude_self: bool
    identity: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def gallery(self) -> DatasetSplit:
        return self.splits[self.gallery_split]

    @property
    def query(self) -> DatasetSplit:
        return self.splits[self.query_split]


def dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as error:
        raise ValueError(f"unknown retrieval dataset {name!r}; choose from {tuple(DATASETS)}") from error


def open_rgb(value: Any):
    """Open a HF Image/PIL value without interpreting arbitrary dataset code."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
        value = value.get("path")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    raise TypeError(f"unsupported image value {type(value).__name__}; expected PIL or HF Image")


def image_digest(value: Any) -> str:
    """Hash decoded RGB pixels and shape; metadata/JPEG containers are ignored."""
    image = open_rgb(value)
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _metadata_rows(dataset: Any):
    # Selecting metadata columns before iterating avoids decoding 59k SOP train
    # images just to inspect product IDs and verify the source partition.
    names = getattr(dataset, "column_names", None)
    if names is not None and hasattr(dataset, "select_columns"):
        return iter(dataset.select_columns([key for key in names if key != "image"]))
    return ({key: value for key, value in row.items() if key != "image"} for row in dataset)


def _label_names(dataset: Any, key: str) -> list[str] | None:
    feature = getattr(dataset, "features", {}).get(key)
    names = getattr(feature, "names", None)
    return list(names) if names is not None else None


def _integer_label(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise ValueError(f"{where}: expected a nonnegative integer class label, got {value!r}")
    return int(value)


def _identifier(value: Any, *, where: str) -> str:
    # Nulls must not become the apparently valid sample/product ID "None".
    # Floats (including NaN) and booleans are likewise ambiguous identifiers.
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError(f"{where}: expected a nonempty string or integer ID, got {value!r}")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{where}: expected a nonempty string or integer ID")
    return result


def build_dataset_bundle(name: str, splits: Mapping[str, Any], *, revision: str,
                         strict: bool = True, audit_images: bool = True) -> DatasetBundle:
    """Validate already loaded splits (also useful for small offline fixtures).

    ``strict=False`` permits smaller fixtures but never permits overlapping
    splits, ambiguous product IDs, missing positives or inconsistent labels.
    Revision must still identify the supplied data in callers' manifests.
    """
    spec = dataset_spec(name)
    required = set(spec.expected_rows)
    missing = required - set(splits)
    if missing:
        raise ValueError(f"{name}: missing splits {sorted(missing)}")
    if not revision:
        raise ValueError("a dataset revision is required")
    built: dict[str, DatasetSplit] = {}
    raw_labels: dict[str, list[str | int]] = {}
    categories: dict[str, str] = {}
    common_names: list[str] | None = None
    audited = []
    duplicates: dict[str, int] = {}

    for split_name in spec.expected_rows:
        dataset = splits[split_name]
        rows = len(dataset)
        if not rows or (strict and rows != spec.expected_rows[split_name]):
            raise ValueError(f"{name}/{split_name}: expected {spec.expected_rows[split_name]:,} "
                             f"rows, found {rows:,}")
        names = _label_names(dataset, spec.relevance_key)
        if names is not None:
            if common_names is not None and common_names != names:
                raise ValueError(f"{name}/{split_name}: label-name mapping differs between splits")
            common_names = names
            if strict and name in ("eurosat", "flowers102") and len(names) != (10 if name == "eurosat" else 102):
                raise ValueError(f"{name}/{split_name}: unexpected class-label vocabulary")

        identities, labels = [], []
        # SOP train is a metadata audit only. It is never part of frozen test
        # retrieval or extraction; there is no need to decode its image bytes.
        check_pixels = audit_images and (name != "sop" or split_name == "test")
        hashes: list[str] | None = [] if check_pixels or spec.id_key is None else None
        for index, row in enumerate(_metadata_rows(dataset)):
            if spec.relevance_key not in row:
                raise ValueError(f"{name}/{split_name}: missing relevance key {spec.relevance_key!r}")
            value = row[spec.relevance_key]
            if name == "sop":
                value = _identifier(value, where=f"sop/{split_name} item_id")
                category = _identifier(row.get("category"), where=f"sop/{split_name} category")
                if value in categories and categories[value] != category:
                    raise ValueError(f"SOP item_id {value!r} occurs in multiple categories")
                categories[value] = category
            else:
                value = _integer_label(value, where=f"{name}/{split_name} row {index}")
                if names is not None and value >= len(names):
                    raise ValueError(f"{name}/{split_name}: label {value} exceeds its vocabulary")
                if strict and value >= (10 if name == "eurosat" else 102):
                    raise ValueError(f"{name}/{split_name}: label {value} exceeds the dataset vocabulary")
            labels.append(value)
            digest = None
            if hashes is not None:
                try:
                    digest = image_digest(dataset[index]["image"])
                except Exception as error:
                    raise ValueError(f"{name}/{split_name} row {index}: image decode failed: {error}") from error
                hashes.append(digest)
            if spec.id_key is None:
                source_id = f"rgb-sha256:{digest}"
            else:
                source_id = _identifier(row.get(spec.id_key),
                                        where=f"{name}/{split_name} {spec.id_key}").replace("\\", "/")
            identities.append(f"{name}:{source_id}")
        if len(identities) != rows:
            raise ValueError(f"{name}/{split_name}: metadata iteration changed row count")
        if len(set(identities)) != rows:
            raise ValueError(f"{name}/{split_name}: duplicate sample IDs (or duplicate pixel IDs)")
        expected_classes = spec.expected_classes.get(split_name)
        if strict and expected_classes is not None and len(set(labels)) != expected_classes:
            raise ValueError(f"{name}/{split_name}: expected {expected_classes} relevance classes, "
                             f"found {len(set(labels))}")
        duplicates[split_name] = rows - len(set(hashes)) if hashes is not None else 0
        if check_pixels:
            audited.append(split_name)
        raw_labels[split_name] = labels
        built[split_name] = DatasetSplit(split_name, dataset, identities, [], hashes)

    for offset, lhs in enumerate(built):
        for rhs in list(built)[offset + 1:]:
            if set(built[lhs].ids) & set(built[rhs].ids):
                raise ValueError(f"{name}: sample IDs overlap between {lhs} and {rhs}")
            a, b = built[lhs].content_hashes, built[rhs].content_hashes
            if a is not None and b is not None and set(a) & set(b):
                raise ValueError(f"{name}: identical RGB images overlap between {lhs} and {rhs}")
    if name == "sop" and set(raw_labels["train"]) & set(raw_labels["test"]):
        raise ValueError("SOP train/test item IDs overlap; cannot claim the official product-disjoint split")

    vocabulary = sorted({value for values in raw_labels.values() for value in values})
    mapping = {value: index for index, value in enumerate(vocabulary)}
    for split_name, split in built.items():
        split.labels = [mapping[value] for value in raw_labels[split_name]]
    gallery, query = built[spec.gallery_split], built[spec.query_split]
    counts = Counter(gallery.labels)
    positive_counts = [counts[label] - int(spec.exclude_self) for label in query.labels]
    unsupported = sum(count < 1 for count in positive_counts)
    if unsupported:
        raise ValueError(f"{name}: {unsupported} query images have no relevant gallery image "
                         "after self exclusion; repair the protocol before reporting")
    identity = {
        "format": DATA_FORMAT, "name": name, "repository": spec.repository,
        "revision": revision, "protocol": spec.protocol,
        "gallery_split": spec.gallery_split, "query_split": spec.query_split,
        "relevance_key": spec.relevance_key, "exclude_self": spec.exclude_self,
        "label_vocabulary_sha256": canonical_hash(vocabulary),
        "splits": {key: {"rows": split.rows, "ids_sha256": canonical_hash(split.ids),
                          "labels_sha256": canonical_hash(split.labels),
                          "rgb_sha256": canonical_hash(split.content_hashes) if split.content_hashes is not None else None}
                   for key, split in built.items()},
        "content_audited_splits": audited, "strict_source_counts": strict,
    }
    manifest = {
        **identity, "signature": canonical_hash(identity), "training_on_target": False,
        "classes": {key: len(set(value.labels)) for key, value in built.items()},
        "positive_queries": query.rows, "queries_without_positive": unsupported,
        "positive_count_min": min(positive_counts), "positive_count_max": max(positive_counts),
        "within_split_duplicate_rgb_images": duplicates,
        "source_status": "HF source counts and identity partitions checked; SOP file-list parity "
                         "with Ebay_test.txt is not independently certified" if name == "sop" else
                         "HF split retained; no split resampling",
        "split_note": "custom HF 80/10/10 split, not Oxford official split" if name == "flowers102" else "",
    }
    return DatasetBundle(name, spec.protocol, built, spec.gallery_split, spec.query_split,
                         spec.exclude_self, identity, manifest)


def load_dataset_bundle(name: str, cache_dir: str | Path, *, revision: str = "main",
                        token: str | None = None, strict: bool = True,
                        audit_images: bool = True) -> DatasetBundle:
    """Resolve an immutable HF commit, load Arrow splits and reuse their audit."""
    from datasets import load_dataset

    from ..embed.archive import pinned_release

    spec = dataset_spec(name)
    cache_dir = Path(cache_dir)
    release = pinned_release(spec.repository, repo_type="dataset", revision=revision,
                             cache_dir=cache_dir, token=token)
    loaded = load_dataset(spec.repository, revision=release["revision"], token=token,
                          cache_dir=str(cache_dir / "hf_datasets"))
    request = {"format": DATA_FORMAT, "spec": asdict(spec), "revision": release["revision"],
               "strict": strict, "audit_images": audit_images}
    audit_path = cache_dir / "dataset_audits" / f"{canonical_hash(request)}.json"
    audit = read_json(audit_path)
    if audit and audit.get("request") == canonical_hash(request):
        splits = {}
        for key, metadata in audit["splits"].items():
            if len(loaded[key]) != len(metadata["ids"]) or len(metadata["labels"]) != len(metadata["ids"]):
                raise RuntimeError(f"cached dataset audit {audit_path} disagrees with loaded rows")
            splits[key] = DatasetSplit(key, loaded[key], metadata["ids"], metadata["labels"],
                                       metadata.get("content_hashes"))
        return DatasetBundle(name, spec.protocol, splits, spec.gallery_split, spec.query_split,
                             spec.exclude_self, audit["identity"], audit["manifest"])
    bundle = build_dataset_bundle(name, loaded, revision=release["revision"], strict=strict,
                                   audit_images=audit_images)
    atomic_json(audit_path, {
        "request": canonical_hash(request), "identity": bundle.identity, "manifest": bundle.manifest,
        "splits": {key: {"ids": split.ids, "labels": split.labels,
                          "content_hashes": split.content_hashes}
                   for key, split in bundle.splits.items()},
    })
    return bundle
