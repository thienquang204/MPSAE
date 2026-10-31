"""Audited, disk-backed data preparation for the frozen crop-transfer suite.

Importing this module does no I/O. Downloads, archive extraction and preprocessing
only occur when prepare_datasets/build_task_records are called by the runner.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import struct
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)
VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
POSE_CLASSES = ("aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair",
                "diningtable", "motorbike", "sofa", "train", "tvmonitor")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DATA_VERSION = "crop-transfer-data-v1-nearest-exact-labels"
POSE_SOURCE_REVISION = "1d57050601f4a3f7842b2953be524f47d4f53b36"
POSE_SOURCE = ("https://github.com/shubhtuls/ViewpointsAndKeypoints/blob/"
               + POSE_SOURCE_REVISION + "/preprocess/readVpsDataPascal.m")
EDGE_MODEL_URL = ("https://raw.githubusercontent.com/opencv/opencv_extra/"
                  "4.13.0/testdata/cv/ximgproc/model.yml.gz")
PASCAL3D_URL = "https://cvgl.stanford.edu/data2/PASCAL3D+_release1.1.zip"
VOC_ARCHIVES = {
    "2007_trainval": ("VOCtrainval_06-Nov-2007.tar", "c52e279531787c972589f7e41ab4ae64"),
    "2007_test": ("VOCtest_06-Nov-2007.tar", "b6e924de25625d8de591ea690078ad9f"),
    "2012_trainval": ("VOCtrainval_11-May-2012.tar", "6cd6e144f989b92b3379bac3b3de84fd"),
}


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _lock(path):
    """OS locks release on process exit, so interrupted preparation can resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b" ")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Another data writer holds {path}") from exc
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "time": time.time()}).encode())
        handle.flush()
        yield
    finally:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def download_file(url, destination, *, checksum=None, algorithm="sha256",
                  timeout=60, max_bytes=30 * 1024**3):
    """Stream into .part; safely resume only with an HTTP entity validator."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if checksum and file_hash(destination, algorithm) != checksum:
            raise RuntimeError(f"Checksum mismatch in {destination}; move the corrupt file aside.")
        return destination
    if not url or not str(url).startswith(("https://", "http://")):
        raise ValueError(f"A HTTP(S) dataset URL is required for {destination}")
    partial = destination.with_name(destination.name + ".part")
    progress = destination.with_name(destination.name + ".download.json")
    with _lock(destination.with_name(destination.name + ".lock")):
        for attempt in range(3):
            previous = json.loads(progress.read_text()) if progress.exists() else {}
            offset = partial.stat().st_size if partial.exists() else 0
            if (offset and previous.get("url") == url and previous.get("expected_size") == offset
                    and checksum and file_hash(partial, algorithm) == checksum):
                os.replace(partial, destination)
                atomic_json(progress, {"url": url, "sha256": file_hash(destination),
                                       "bytes": offset, "complete": True})
                return destination
            validator = previous.get("validator") if previous.get("url") == url else None
            headers = {"User-Agent": "frozen-crop-transfer/1.0", "Accept-Encoding": "identity"}
            if offset and validator:
                headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
            else:
                offset = 0
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = response.getcode()
                    if status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not offset or not content_range.startswith(f"bytes {offset}-"):
                            raise RuntimeError("Server returned an invalid resumable byte range")
                    elif status == 200:
                        offset = 0
                    else:
                        raise RuntimeError(f"Unexpected download status {status}")
                    remaining = response.headers.get("Content-Length")
                    expected_size = offset + int(remaining) if remaining else None
                    if expected_size and expected_size > max_bytes:
                        raise RuntimeError(f"Download exceeds byte limit: {expected_size}")
                    validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                    atomic_json(progress, {"url": url, "validator": validator,
                                           "expected_size": expected_size})
                    total, last_log = offset, time.monotonic()
                    with partial.open("ab" if offset else "wb") as output:
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            total += len(block)
                            if total > max_bytes:
                                raise RuntimeError("Download exceeds configured byte limit")
                            output.write(block)
                            if time.monotonic() - last_log > 15:
                                LOG.info("Downloading %s: %.2f GiB", destination.name, total / 1024**3)
                                last_log = time.monotonic()
                        output.flush()
                        os.fsync(output.fileno())
                    if expected_size is not None and total != expected_size:
                        raise IOError(f"Incomplete response: expected {expected_size}, got {total}")
                if checksum and file_hash(partial, algorithm) != checksum:
                    raise RuntimeError(f"Downloaded checksum mismatch for {destination}; "
                                       f"inspect/remove {partial} before retrying")
                os.replace(partial, destination)
                atomic_json(progress, {"url": url, "sha256": file_hash(destination),
                                       "bytes": destination.stat().st_size, "complete": True})
                return destination
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"Cannot download {url}: {exc}. Place the archive at "
                                       f"{destination}, or configure its URL and rerun.") from exc
                LOG.warning("Download interrupted; retry %s/3: %s", attempt + 2, exc)
                time.sleep(attempt + 1)
    raise RuntimeError("Download did not complete")


def _safe_member_path(destination, name):
    name = name.replace("\\", "/")
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or any(part in ("..", "") or ":" in part for part in parts):
        raise ValueError(f"Unsafe archive path: {name!r}")
    destination = Path(destination).resolve()
    target = destination.joinpath(*parts)
    if not target.resolve().is_relative_to(destination):
        raise ValueError(f"Archive path leaves dataset directory: {name!r}")
    if target.is_symlink():
        raise ValueError(f"Archive target is a symlink: {target}")
    return target


def extract_archive(archive, destination, *, max_bytes=80 * 1024**3):
    """Extract regular files only, one at a time; no extractall or archive code."""
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    total = 0

    def copy_member(name, size, stream):
        nonlocal total
        total += size
        if size < 0 or total > max_bytes:
            raise ValueError("Archive exceeds the configured expanded size limit")
        target = _safe_member_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".extracting")
        if temporary.is_symlink():
            raise ValueError(f"Unsafe extraction temporary path: {temporary}")
        with temporary.open("wb") as output:
            shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != size:
            raise IOError(f"Truncated archive member {name}")
        os.replace(temporary, target)

    with _lock(destination / (archive.name + ".extract.lock")):
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise ValueError(f"Archive symlink is forbidden: {member.filename}")
                    target = _safe_member_path(destination, member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        with bundle.open(member) as stream:
                            copy_member(member.filename, member.file_size, stream)
        else:
            with tarfile.open(archive, "r|*") as bundle:
                for member in bundle:
                    target = _safe_member_path(destination, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        with bundle.extractfile(member) as stream:
                            copy_member(member.name, member.size, stream)
                    else:
                        raise ValueError(f"Archive links/devices are forbidden: {member.name}")
    LOG.info("Extracted %s (%.2f GiB)", archive.name, total / 1024**3)


def _read_ids(path):
    ids = [line.split()[0] for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        raise ValueError(f"Invalid/empty/duplicate image IDs in {path}")
    return ids


def _voc_root(root, year):
    candidates = [Path(root) / "VOCdevkit" / f"VOC{year}", Path(root) / f"VOC{year}"]
    return next((x for x in candidates if x.is_dir()), candidates[0])


def _tasks(args):
    tasks = getattr(args, "tasks", ["detection", "mask", "pose"])
    if isinstance(tasks, str):
        tasks = tasks.replace(",", " ").split()
    return [{"A": "detection", "B": "mask", "C": "pose"}.get(t, t) for t in tasks]


def _download_options(args):
    return {"timeout": getattr(args, "download_timeout", 60),
            "max_bytes": getattr(args, "max_download_bytes", 30 * 1024**3)}


def _ensure_voc(args, year, split):
    root = Path(args.data_root)
    voc = _voc_root(root, year)
    manifest = voc / "ImageSets" / "Main" / f"{split}.txt"
    # Check every required source file before accepting an existing extraction.
    ready = manifest.exists()
    if ready:
        ready = all((voc / "JPEGImages" / f"{iid}.jpg").is_file()
                    and (voc / "Annotations" / f"{iid}.xml").is_file()
                    for iid in _read_ids(manifest))
    if year == 2012 and "mask" in _tasks(args) and ready:
        for seg_split in ("train", "val"):
            seg_ids = voc / "ImageSets" / "Segmentation" / f"{seg_split}.txt"
            ready = ready and seg_ids.is_file()
            if seg_ids.is_file():
                ready = ready and all((voc / folder / f"{iid}.png").is_file()
                                      for iid in _read_ids(seg_ids)
                                      for folder in ("SegmentationObject", "SegmentationClass"))
    if ready:
        return voc
    key = f"{year}_{'test' if split == 'test' else 'trainval'}"
    name, md5 = VOC_ARCHIVES[key]
    override = getattr(args, f"voc{year}_{'test' if split == 'test' else 'trainval'}_url", None)
    if year == 2012:
        override = getattr(args, "voc2012_url", None) or override
    url = override or f"https://host.robots.ox.ac.uk/pascal/VOC/voc{year}/{name}"
    archive = root / "archives" / name
    if not archive.exists() and not getattr(args, "download", False):
        raise FileNotFoundError(f"Missing/incomplete VOC{year}: {manifest}. Enable --download, "
                                f"or place {name} at {archive}.")
    download_file(url, archive, checksum=md5, algorithm="md5", **_download_options(args))
    extract_archive(archive, root)
    voc = _voc_root(root, year)
    if not (voc / "ImageSets" / "Main" / f"{split}.txt").exists():
        raise RuntimeError(f"Archive did not provide VOC{year}/{split}")
    return voc


def load_voc_annotation(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError(f"Unexpectedly large VOC XML annotation: {path}")
    tree = ET.parse(path).getroot()
    width, height = int(tree.findtext("size/width")), int(tree.findtext("size/height"))
    if min(width, height) <= 0:
        raise ValueError(f"Invalid image dimensions in {path}")
    objects = []
    for index, element in enumerate(tree.findall("object")):
        name = element.findtext("name")
        if name not in VOC_CLASSES:
            raise ValueError(f"Unknown VOC category {name!r} in {path}")
        coords = [float(element.findtext(f"bndbox/{key}"))
                  for key in ("xmin", "ymin", "xmax", "ymax")]
        box = [coords[0] - 1, coords[1] - 1, coords[2], coords[3]]
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid annotation box in {path}")
        objects.append({"box": box, "class_id": VOC_CLASSES.index(name) + 1,
                        "category": VOC_CLASSES.index(name), "class_name": name,
                        "difficult": bool(int(element.findtext("difficult", "0"))),
                        "occluded": bool(int(element.findtext("occluded", "0"))),
                        "truncated": bool(int(element.findtext("truncated", "0"))),
                        "annotation_index": index})
    return {"image_id": path.stem, "width": width, "height": height, "objects": objects}


def prepare_datasets(args):
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(args)
    result = {"version": DATA_VERSION, "data_root": str(root.resolve()), "datasets": {}}
    for year in ([2007] if "detection" in tasks else []) + ([2012] if set(tasks) & {"mask", "pose"} else []):
        voc = _ensure_voc(args, year, "train")
        if year == 2007:
            voc = _ensure_voc(args, year, "test")
        groups = {"Main": ("train", "val", "trainval", "test") if year == 2007 else ("train", "val", "trainval")}
        if year == 2012 and "mask" in tasks:
            groups["Segmentation"] = ("train", "val")
        expected = {("2007", "Main", "train"): 2501, ("2007", "Main", "val"): 2510,
                    ("2007", "Main", "trainval"): 5011, ("2007", "Main", "test"): 4952,
                    ("2012", "Main", "train"): 5717, ("2012", "Main", "val"): 5823,
                    ("2012", "Main", "trainval"): 11540,
                    ("2012", "Segmentation", "train"): 1464, ("2012", "Segmentation", "val"): 1449}
        info, seen = {}, set()
        for group, splits in groups.items():
            manifests = {}
            for split in splits:
                manifest = voc / "ImageSets" / group / f"{split}.txt"
                ids = _read_ids(manifest)
                count = expected.get((str(year), group, split))
                if count and getattr(args, "strict_dataset_counts", True) and len(ids) != count:
                    raise ValueError(f"Expected {count} images in {manifest}; found {len(ids)}")
                manifests[split] = set(ids)
                info[f"{group}/{split}"] = {"count": len(ids), "sha256": file_hash(manifest)}
                for iid in ids:
                    if iid in seen:
                        continue
                    annotation = load_voc_annotation(voc / "Annotations" / f"{iid}.xml")
                    image_path = voc / "JPEGImages" / f"{iid}.jpg"
                    with Image.open(image_path) as image:
                        if image.size != (annotation["width"], annotation["height"]):
                            raise ValueError(f"Image/XML dimensions disagree: {image_path}")
                        if getattr(args, "data_verify_images", False):
                            image.verify()
                    seen.add(iid)
            if manifests["train"] & manifests["val"]:
                raise ValueError(f"Official {year}/{group} train and val overlap")
            if "trainval" in manifests and manifests["trainval"] != manifests["train"] | manifests["val"]:
                raise ValueError(f"Invalid official {year}/{group}/trainval union")
            if "test" in manifests and manifests["test"] & manifests["trainval"]:
                raise ValueError("VOC2007 trainval/test overlap")
        result["datasets"][f"VOC{year}"] = {"root": str(voc.resolve()), "manifests": info}
    if "detection" in tasks:
        path = Path(getattr(args, "edge_model", None) or root / "models" / "edge_model.yml.gz")
        if not path.is_file():
            if not getattr(args, "download", False):
                raise FileNotFoundError(f"Missing EdgeBoxes structured-edge model {path}; enable --download or --edge-model")
            download_file(getattr(args, "edge_model_url", None) or EDGE_MODEL_URL, path, **_download_options(args))
        result["edge_model"] = {"path": str(path.resolve()), "sha256": file_hash(path),
                                "source": getattr(args, "edge_model_url", None) or EDGE_MODEL_URL}
    if "pose" in tasks:
        pose_root = _pascal3d_root(args)
        if getattr(args, "pascal3d_root", None) and not (pose_root / "Annotations").is_dir():
            raise FileNotFoundError(f"--pascal3d-root must point to an existing extracted PASCAL3D+ release "
                                    f"containing Annotations: {pose_root}. Omit this option for automatic "
                                    "download/extraction under --data-root.")
        voc2012 = _voc_root(root, 2012)
        missing, required = _pose_annotation_files(voc2012, pose_root)
        if missing:
            archive = root / "archives" / "PASCAL3D+_release1.1.zip"
            if getattr(args, "pascal3d_root", None):
                raise FileNotFoundError(f"Incomplete explicit PASCAL3D+ release: missing {missing[0]} "
                                        f"({len(missing)} required annotations missing). Restore the full "
                                        "release, or omit --pascal3d-root to use automatic extraction.")
            if not archive.exists() and not getattr(args, "download", False):
                raise FileNotFoundError(f"Missing PASCAL3D+ at {pose_root}; enable --download, set --pascal3d-root, "
                                        "or provide --pascal3d-url for an accessible release mirror.")
            download_file(getattr(args, "pascal3d_url", None) or PASCAL3D_URL, archive, **_download_options(args))
            extract_archive(archive, root)
            pose_root = _pascal3d_root(args)
            missing, required = _pose_annotation_files(voc2012, pose_root)
            if missing:
                raise FileNotFoundError(f"PASCAL3D+ archive is incomplete: missing {missing[0]} "
                                        f"({len(missing)} missing annotations). Configure a complete release URL.")
        if not all((pose_root / "Annotations" / f"{cls}_pascal").is_dir() for cls in POSE_CLASSES):
            raise ValueError(f"Incomplete PASCAL3D+ release (all 12 Pascal categories required): {pose_root}")
        _verify_rotation_convention()
        release_digest = hashlib.sha256()
        for path in required:
            release_digest.update(str(path.relative_to(pose_root)).encode())
            release_digest.update(file_hash(path).encode())
        result["datasets"]["PASCAL3D+"] = {"root": str(pose_root.resolve()), "source": POSE_SOURCE,
                "conversion": "angle2dcm(theta,elevation-pi/2,-azimuth,'ZXZ'); passive intrinsic radians",
                "annotation_content_sha256": release_digest.hexdigest(), "required_annotation_files": len(required),
                "validity": "finite azimuth/elevation/theta fields; positive viewpoint.distance; non-difficult"}
    atomic_json(root / "preparation_report.json", result)
    return result


def _pascal3d_root(args):
    if getattr(args, "pascal3d_root", None):
        return Path(args.pascal3d_root)
    root = Path(args.data_root)
    return next((root / name for name in ("PASCAL3D+_release1.1", "PASCAL3D+_release1.0", "PASCAL3D+")
                 if (root / name / "Annotations").is_dir()), root / "PASCAL3D+_release1.1")


def _pose_annotation_files(voc, pose_root):
    """Check expected rigid-category files against VOC, not directory existence."""
    required = []
    for iid in _read_ids(voc / "ImageSets" / "Main" / "trainval.txt"):
        annotation = load_voc_annotation(voc / "Annotations" / f"{iid}.xml")
        categories = {obj["class_name"] for obj in annotation["objects"]} & set(POSE_CLASSES)
        required.extend(pose_root / "Annotations" / f"{name}_pascal" / f"{iid}.mat" for name in sorted(categories))
    return [path for path in required if not path.is_file()], required


def viewpoint_to_rotation(azimuth, elevation, theta):
    """Author MATLAB ZXZ DCM convention; arguments are PASCAL3D+ degrees.

    readVpsDataPascal.m at POSE_SOURCE_REVISION uses
    angle2dcm(theta, elevation-pi/2, -azimuth, 'ZXZ'). MATLAB specifies
    right-handed intrinsic passive rotations, composed third @ second @ first.
    https://www.mathworks.com/help/aerotbx/ug/angle2dcm.html
    """
    a, e, t = np.deg2rad(np.asarray([azimuth, elevation, theta], dtype=np.float64))
    def passive_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    c, s = np.cos(e - np.pi / 2), np.sin(e - np.pi / 2)
    passive_x = np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float64)
    return passive_z(-a) @ passive_x @ passive_z(t)


def _verify_rotation_convention():
    # Analytic canonical cases fix units, handedness, azimuth sign and baseline tilt.
    cases = [((0, 90, 0), np.eye(3)),
             ((0, 0, 0), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
             ((90, 90, 0), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
             ((0, 90, 90), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),
             ((90, 0, 0), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))]
    for angles, expected in cases:
        rotation = viewpoint_to_rotation(*angles)
        if not np.allclose(rotation, expected, atol=1e-12) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(rotation), 1):
            raise RuntimeError("PASCAL3D+ author rotation convention verification failed")


def expanded_crop_box(box, variant=0):
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1
    if min(w, h) <= 0 or not np.isfinite(box).all() or variant not in range(5):
        raise ValueError("Invalid crop box/variant")
    dx, dy = [(0, 0), (.1 * w, 0), (-.1 * w, 0), (0, .1 * h), (0, -.1 * h)][variant]
    cx, cy = .5 * (x1 + x2) + dx, .5 * (y1 + y2) + dy
    return [math.floor(cx - .6 * w), math.floor(cy - .6 * h),
            math.ceil(cx + .6 * w), math.ceil(cy + .6 * h)]


def _crop_tensor(array, crop_box, *, fill=0, size=224, mode="nearest-exact"):
    x1, y1, x2, y2 = map(int, crop_box)
    h, w = array.shape[-2:]
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) > 10_000_000:
        raise ValueError("Invalid or excessive crop extent")
    canvas = torch.empty((array.shape[0], y2-y1, x2-x1), dtype=torch.float32)
    canvas[:] = torch.as_tensor(fill, dtype=torch.float32).reshape(-1, 1, 1)
    left, top, right, bottom = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if right > left and bottom > top:
        canvas[:, top-y1:bottom-y1, left-x1:right-x1] = array[:, top:bottom, left:right]
    options = {"align_corners": False, "antialias": True} if mode == "bilinear" else {}
    return F.interpolate(canvas[None], size=(size, size), mode=mode, **options)[0]


class RegionCropDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self._image_path = self._image = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[int(index)]
        if record["image_path"] != self._image_path:
            with Image.open(record["image_path"]) as image:
                self._image = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy()).permute(2, 0, 1) / 255
            self._image_path = record["image_path"]
        crop = _crop_tensor(self._image, record["crop_box"], fill=MEAN, mode="bilinear")
        return (crop - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def load_mask_original(record):
    with Image.open(record["object_mask_path"]) as image:
        objects = np.asarray(image).copy()
    with Image.open(record["class_mask_path"]) as image:
        classes = np.asarray(image).copy()
    valid = (objects != 255) & (classes != 255)
    return (objects == record["instance_id"]) & valid, valid


def mask_target(record, size=224):
    foreground, valid = load_mask_original(record)
    maps = torch.from_numpy(np.stack([foreground, valid]).astype(np.float32))
    crop = _crop_tensor(maps, record["crop_box"], size=size)
    return crop[0], crop[1].bool()


class RecordManifest(Sequence):
    """JSONL records plus a uint64 mmap index; only requested rows enter RAM."""
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.metadata = json.loads(Path(self.path + ".meta.json").read_text())
        self.fingerprint = self.metadata["fingerprint"]
        self._length = int(self.metadata["rows"])
        self._offsets = self._handle = None
        self._owner_pid = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._owner_pid != os.getpid():
            # fork inherits a shared seek offset even if Python objects differ.
            # Open each worker's own description before performing random seeks.
            if self._handle is not None:
                self._handle.close()
            self._handle = self._offsets = None
            self._owner_pid = os.getpid()
        if self._offsets is None:
            self._offsets = np.memmap(self.path + ".idx", dtype="<u8", mode="r", shape=(len(self),))
        if self._handle is None:
            self._handle = open(self.path, "rb")
        self._handle.seek(int(self._offsets[index]))
        return json.loads(self._handle.readline())

    def __iter__(self):
        with open(self.path, "rb") as handle:
            for line in handle:
                yield json.loads(line)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = state["_offsets"] = None
        state["_owner_pid"] = None
        return state


def _iou(boxes, targets):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(boxes[:, None, 2:], targets[None, :, 2:]) - np.maximum(boxes[:, None, :2], targets[None, :, :2])).prod(-1)
    area_a, area_b = (boxes[:, 2:] - boxes[:, :2]).prod(-1), (targets[:, 2:] - targets[:, :2]).prod(-1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


class _EdgeBoxes:
    def __init__(self, args):
        import cv2
        if not cv2.__version__.startswith("4.13."):
            raise RuntimeError(f"Protocol requires OpenCV-contrib 4.13.x; found {cv2.__version__}")
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError("Install opencv-contrib-python-headless 4.13.x for EdgeBoxes")
        path = Path(getattr(args, "edge_model", None) or Path(args.data_root) / "models" / "edge_model.yml.gz")
        self.detector = cv2.ximgproc.createStructuredEdgeDetection(str(path))
        self.edge = cv2.ximgproc.createEdgeBoxes()
        settings = {"Alpha": .65, "Beta": .75, "Eta": 1., "MinScore": .01, "MaxBoxes": 512,
                    "EdgeMinMag": .1, "EdgeMergeThr": .5, "ClusterMinMag": .5,
                    "MaxAspectRatio": 3., "MinBoxArea": 1000., "Gamma": 2., "Kappa": 1.5}
        for setting, value in settings.items():
            getattr(self.edge, "set" + setting)(value)
        self.provenance = {"opencv": cv2.__version__, "model_sha256": file_hash(path),
                           "model_source": getattr(args, "edge_model_url", None) or EDGE_MODEL_URL,
                           "settings": settings, "edge_nms": {"r": 2, "s": 0, "m": 1}}

    def __call__(self, image_path):
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.
        edges = self.detector.detectEdges(rgb)
        orientation = self.detector.computeOrientation(edges)
        suppressed = self.detector.edgesNms(edges, orientation, r=2, s=0, m=1)
        boxes, scores = self.edge.getBoundingBoxes(suppressed, orientation)
        if boxes is None or scores is None:
            return []
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        boxes[:, 2:] += boxes[:, :2]
        order = np.argsort(-scores, kind="stable")
        seen, kept = set(), []
        for original in order:
            box = tuple(boxes[original])
            if not np.isfinite(box).all() or not np.isfinite(scores[original]) or box[2] <= box[0] or box[3] <= box[1] or box in seen:
                continue
            seen.add(box)
            kept.append({"box": list(box), "objectness": float(scores[original]), "original_index": int(original)})
        return kept


def _common(voc, iid, annotation, box, category, object_id, variant=0):
    return {"image_id": iid, "image_path": str((voc / "JPEGImages" / f"{iid}.jpg").resolve()),
            "annotation_path": str((voc / "Annotations" / f"{iid}.xml").resolve()),
            "box": list(box), "crop_box": expanded_crop_box(box, variant),
            "image_size": [annotation["width"], annotation["height"]],
            "width": annotation["width"], "height": annotation["height"],
            "category": category, "class_id": category + 1, "object_id": object_id, "variant": variant}


def _detection_records(voc, iid, annotation, args, split, edge, report):
    image_path = voc / "JPEGImages" / f"{iid}.jpg"
    key = _identity({"image_sha256": file_hash(image_path), "edge": edge.provenance, "version": DATA_VERSION})
    cache = Path(args.data_root) / "cache" / "proposals" / f"{iid}-{key[:16]}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("fingerprint") != key:
            raise RuntimeError(f"Proposal cache identity mismatch: {cache}")
        proposals = cached["proposals"]
    else:
        started = time.monotonic()
        proposals = edge(image_path)
        cached = {"fingerprint": key, "proposals": proposals, "seconds": time.monotonic() - started}
        atomic_json(cache, cached)
    report["proposal_generation_seconds"] += float(cached["seconds"])
    gt = [x for x in annotation["objects"] if not x["difficult"]]
    difficult = [x for x in annotation["objects"] if x["difficult"]]
    report["non_difficult_objects"] += len(gt)
    report["proposal_images"] += 1
    report["proposals_returned"] += len(proposals)
    for budget in (128, 256, 512):
        overlaps = _iou([x["box"] for x in proposals[:budget]], [x["box"] for x in gt])
        best = overlaps.max(0) if len(overlaps) else np.zeros(len(gt))
        for threshold in (.5, .75):
            report[f"recall_hits_P{budget}_IoU{threshold}"] += int((best >= threshold).sum())
    budget = int(getattr(args, "proposal_budget", 256))
    candidates = [(x["box"], i, False) for i, x in enumerate(proposals[:budget])]
    before = _iou([x[0] for x in candidates], [x["box"] for x in gt])
    report["gt_without_positive_proposal"] += int(((before.max(0) if len(before) else np.zeros(len(gt))) < .5).sum())
    if split in ("train", "trainval"):
        seen = {tuple(x[0]) for x in candidates}
        for obj in gt:
            if tuple(obj["box"]) not in seen:
                seen.add(tuple(obj["box"]))
                candidates.append((obj["box"], len(candidates), True))
    for box, proposal_index, is_gt in candidates:
        overlap = _iou([box], [x["box"] for x in gt])[0]
        matched = gt[int(overlap.argmax())] if len(gt) else None
        maximum = float(overlap.max()) if len(gt) else 0.
        if maximum >= .5:
            label = matched["class_id"]
        elif len(difficult) and _iou([box], [x["box"] for x in difficult]).max() >= .5:
            label = -1
        else:
            label = 0 if maximum < .3 else -1
        report["foreground" if label > 0 else "background" if label == 0 else "ignored"] += 1
        report["positive_exact_gt"] += int(label > 0 and any(box == x["box"] for x in gt))
        record = _common(voc, iid, annotation, box, label - 1, f"{iid}/proposal/{proposal_index}")
        record.update(label=label, proposal_index=proposal_index, is_gt=is_gt,
                      matched_box=matched["box"] if maximum >= .5 else None,
                      matched_object_index=matched["annotation_index"] if maximum >= .5 else None,
                      original_iou=maximum)
        yield record


def _mask_records(voc, iid, annotation, skipped):
    object_path, class_path = voc / "SegmentationObject" / f"{iid}.png", voc / "SegmentationClass" / f"{iid}.png"
    with Image.open(object_path) as image:
        objects = np.asarray(image).copy()
    with Image.open(class_path) as image:
        classes = np.asarray(image).copy()
    if objects.shape != classes.shape or objects.shape != (annotation["height"], annotation["width"]):
        raise ValueError(f"Mask/image shape mismatch for {iid}")
    for instance in np.unique(objects):
        if instance in (0, 255):
            continue
        selected = (objects == instance) & (classes != 255)
        labels = np.unique(classes[selected])
        if len(labels) != 1 or not 1 <= labels[0] <= 20:
            skipped.write(json.dumps({"image_id": iid, "instance_id": int(instance), "reason": "empty_or_inconsistent_foreground_class"}) + "\n")
            continue
        yy, xx = np.nonzero(selected)
        box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        for variant in range(5):
            record = _common(voc, iid, annotation, box, int(labels[0]) - 1, f"{iid}/mask/{instance}", variant)
            record.update(instance_id=int(instance), object_mask_path=str(object_path.resolve()), class_mask_path=str(class_path.resolve()))
            yield record


def _pose_records(voc, pose_root, iid, annotation, skipped):
    # scipy is used only for MATLAB annotation I/O; no retrieval or distance path.
    from scipy.io import loadmat
    for category, name in enumerate(POSE_CLASSES):
        path = pose_root / "Annotations" / f"{name}_pascal" / f"{iid}.mat"
        if not path.exists():
            if any(obj["class_name"] == name for obj in annotation["objects"]):
                raise FileNotFoundError(f"Missing PASCAL3D+ annotation for a VOC object: {path}")
            continue  # This image has no annotation in that category's release folder.
        record = loadmat(path, simplify_cells=True).get("record")
        if not isinstance(record, dict):
            raise ValueError(f"Unsupported PASCAL3D+ record structure: {path}")
        objects = record.get("objects", [])
        if isinstance(objects, dict):
            objects = [objects]
        for index, obj in enumerate(objects):
            if not isinstance(obj, dict) or obj.get("class") != name:
                continue
            vp = obj.get("viewpoint")
            reason = None
            if not all(key in obj for key in ("difficult", "occluded", "truncated")):
                reason = "missing_object_validity_fields"
            elif bool(obj["difficult"]):
                reason = "difficult"
            elif not isinstance(vp, dict) or not all(key in vp for key in ("azimuth", "elevation", "theta", "distance")):
                reason = "missing_fine_viewpoint_fields"
            else:
                try:
                    values = np.asarray([vp[key] for key in ("azimuth", "elevation", "theta", "distance")], dtype=float)
                except (ValueError, TypeError):
                    values = np.asarray([])
                if values.shape != (4,) or not np.isfinite(values).all() or values[3] <= 0:
                    reason = "coarse_only_or_invalid_viewpoint"
            if reason:
                skipped.write(json.dumps({"image_id": iid, "category": name, "annotation_index": index, "reason": reason}) + "\n")
                continue
            box = np.asarray(obj.get("bbox", []), dtype=float).reshape(-1)
            if len(box) != 4:
                raise ValueError(f"Missing PASCAL3D+ object box: {path} #{index}")
            box[:2] -= 1
            rotation = viewpoint_to_rotation(vp["azimuth"], vp["elevation"], vp["theta"])
            for variant in range(5):
                item = _common(voc, iid, annotation, box.tolist(), category, f"{iid}/pose/{name}/{index}", variant)
                item.update(pose_class_id=category, rotation=rotation.tolist(),
                            occluded=bool(obj.get("occluded", 0)), truncated=bool(obj.get("truncated", 0)),
                            pose_annotation_path=str(path.resolve()), pose_annotation_index=index)
                yield item


def build_task_records(task, split, args):
    task = {"A": "detection", "B": "mask", "C": "pose"}.get(task, task)
    if task not in ("detection", "mask", "pose"):
        raise ValueError(f"Unknown transfer task {task}")
    allowed = ("train", "val", "trainval", "test") if task == "detection" else ("fit", "dev", "train", "val")
    if split not in allowed:
        raise ValueError(f"Invalid {task} split: {split}")
    voc = _voc_root(args.data_root, 2007 if task == "detection" else 2012)
    official_split = "train" if split in ("fit", "dev") else split
    group = "Segmentation" if task == "mask" else "Main"
    split_path = voc / "ImageSets" / group / f"{official_split}.txt"
    ids = _read_ids(split_path)
    if split in ("fit", "dev"):
        ids = [iid for iid in ids if (int(hashlib.sha256(("mrl-csr-extensions-v1/" + iid).encode()).hexdigest()[:8], 16) % 5 == 0) == (split == "dev")]
    edge = _EdgeBoxes(args) if task == "detection" else None
    pose_root = _pascal3d_root(args) if task == "pose" else None
    inputs = hashlib.sha256()
    images = []
    # Hash content once per image, never once per proposal. This detects source
    # changes and prevents reusing a same-size cache with different annotations.
    for i, iid in enumerate(ids):
        annotation_path = voc / "Annotations" / f"{iid}.xml"
        paths = [voc / "JPEGImages" / f"{iid}.jpg", annotation_path]
        if task == "mask":
            paths += [voc / folder / f"{iid}.png" for folder in ("SegmentationObject", "SegmentationClass")]
        if task == "pose":
            paths += [path for name in POSE_CLASSES if (path := pose_root / "Annotations" / f"{name}_pascal" / f"{iid}.mat").exists()]
        for path in paths:
            inputs.update(str(path.resolve()).encode())
            inputs.update(file_hash(path).encode())
        annotation = load_voc_annotation(annotation_path)
        images.append({"image_id": iid, "annotation_path": str(annotation_path.resolve()),
                       "width": annotation["width"], "height": annotation["height"]})
        if i % 500 == 0:
            LOG.info("Checking %s/%s input hashes: %d/%d images", task, split, i, len(ids))
    identity = {"version": DATA_VERSION, "source_hash": getattr(args, "source_hash", ""),
                "task": task, "split": split, "split_manifest_sha256": file_hash(split_path),
                "image_ids": ids, "input_sha256": inputs.hexdigest(),
                "proposal_budget": getattr(args, "proposal_budget", 256) if edge else None,
                "edge": edge.provenance if edge else None, "pose_source": POSE_SOURCE if pose_root else None,
                "crop_resampling": {"rgb": "bilinear-antialiased-align_corners=False", "labels": "nearest-exact"}}
    key = _identity(identity)
    path = Path(args.data_root) / "cache" / "manifests" / f"{task}-{split}-{key[:20]}.jsonl"
    meta_path, index_path = Path(str(path) + ".meta.json"), Path(str(path) + ".idx")
    if path.exists() and meta_path.exists() and index_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("identity_hash") == key and metadata.get("jsonl_sha256") == file_hash(path) and metadata.get("index_sha256") == file_hash(index_path):
            if not metadata.get("rows"):
                raise RuntimeError(f"Cached manifest has no eligible crops: {path}")
            return RecordManifest(path)
        raise RuntimeError(f"Manifest cache integrity failure: {path}; move it aside before rerunning")
    path.parent.mkdir(parents=True, exist_ok=True)
    report, per_class = Counter(), Counter()
    started = time.monotonic()
    with _lock(str(path) + ".lock"):
        temp, temp_index = Path(str(path) + ".part"), Path(str(path) + ".idx.part")
        skipped_path = Path(str(path) + ".skipped.jsonl")
        count = 0
        with temp.open("wb") as output, temp_index.open("wb") as index, skipped_path.open("w", encoding="utf-8") as skipped:
            for image_index, iid in enumerate(ids):
                annotation = load_voc_annotation(voc / "Annotations" / f"{iid}.xml")
                if task == "detection":
                    records = _detection_records(voc, iid, annotation, args, split, edge, report)
                elif task == "mask":
                    records = _mask_records(voc, iid, annotation, skipped)
                else:
                    records = _pose_records(voc, pose_root, iid, annotation, skipped)
                for record in records:
                    index.write(struct.pack("<Q", output.tell()))
                    output.write(json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n")
                    count += 1
                    if record["variant"] == 0:
                        per_class[str(record["category"])] += 1
                if image_index % 100 == 0:
                    LOG.info("Prepared %s/%s: %d/%d images, %d crops", task, split, image_index, len(ids), count)
            for handle in (output, index):
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp, path)
        os.replace(temp_index, index_path)
        jsonl_hash, index_hash = file_hash(path), file_hash(index_path)
        metadata = {**identity, "identity_hash": key, "rows": count, "images": images,
                    "jsonl_sha256": jsonl_hash, "index_sha256": index_hash,
                    "fingerprint": _identity({"identity": key, "jsonl": jsonl_hash}),
                    "per_class_counts": dict(per_class), "diagnostics": dict(report),
                    "preparation_seconds": time.monotonic() - started,
                    "skipped_records": str(skipped_path.resolve())}
        if edge:
            metadata["proposal_recall"] = {f"P{budget}/IoU{threshold}": report[f"recall_hits_P{budget}_IoU{threshold}"] / max(1, report["non_difficult_objects"])
                                           for budget in (128, 256, 512) for threshold in (.5, .75)}
        atomic_json(meta_path, metadata)
        LOG.info("Prepared %s/%s: %d images, %d crops; per-class counts=%s; diagnostics=%s",
                 task, split, len(ids), count, dict(per_class), dict(report))
        if edge:
            LOG.info("Annotation-independent proposal recall %s: %s", split, metadata["proposal_recall"])
    if count == 0:
        raise RuntimeError(f"No eligible crops in {task}/{split}; inspect {skipped_path}")
    return RecordManifest(path)
