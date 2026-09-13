#!/usr/bin/env python3
"""Download and validate ImageNet-1K from Hugging Face.

The ILSVRC ImageNet repository is gated. Before running this script:

1. Accept the access conditions at
   https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Authenticate with ``hf auth login`` or export an approved ``HF_TOKEN``.

By default, all official ``train``, ``validation``, and ``test`` splits are
downloaded. The resulting cache is directly consumable by
``csr_vs_mmpot_imagenet.py --data-backend hf --data-root CACHE``.
Training uses ``train`` and metrics use the labelled ``validation`` split;
the official ``test`` split is retained for prediction/submission workflows
and is allowed to be unlabelled. ImageNet is large, so make sure CACHE is on a
volume with ample free space.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import DownloadConfig, load_dataset
from tqdm.auto import tqdm


DATASET_PAGE = "https://huggingface.co/datasets/ILSVRC/imagenet-1k"


def parse_splits(value: str) -> List[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"train", "validation", "test"}
    invalid = sorted(set(splits) - allowed)
    if not splits or invalid:
        suffix = f"; invalid values: {', '.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(
            f"expected comma-separated train,validation,test splits{suffix}"
        )
    return list(dict.fromkeys(splits))


def parse_check_samples(value: str) -> Optional[int]:
    if value.strip().lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer or 'all'") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer or 'all'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download gated ILSVRC/imagenet-1k splits into a reusable Hugging Face cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))
    parser.add_argument("--dataset-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--splits",
        type=parse_splits,
        default=["train", "validation", "test"],
        help="official splits to cache; test may not contain ground-truth labels",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="environment variable containing the access token; cached `hf auth login` is also supported",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--check-samples",
        type=parse_check_samples,
        default=None,
        metavar="N|all",
        help="fully decode this many images per split; 'all' checks every image and 0 disables pixel checks",
    )
    return parser


def validate_split(
    split_name: str, dataset: Any, check_samples: Optional[int]
) -> Dict[str, Any]:
    if "image" not in dataset.column_names:
        raise RuntimeError(
            f"split {split_name!r} has columns {dataset.column_names}, expected an image column"
        )
    has_label_column = "label" in dataset.column_names
    labels_available = has_label_column
    target = len(dataset) if check_samples is None else min(check_samples, len(dataset))
    errors: List[Dict[str, Any]] = []
    warning_categories: Counter[str] = Counter()
    warning_examples: List[Dict[str, Any]] = []
    corrupt_exif_warnings = 0

    progress = tqdm(
        range(target),
        total=target,
        desc=f"Validating {split_name}",
        unit="img",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for index in progress:
        image = None
        converted = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sample = dataset[index]
                image = sample["image"]
                if image is None or not hasattr(image, "convert"):
                    raise TypeError("image decoder did not return a PIL-compatible image")
                if image.width < 1 or image.height < 1:
                    raise ValueError(f"invalid image size {image.size}")

                # This follows the same RGB conversion used by the training
                # adapter. load() forces complete pixel decoding instead of
                # accepting an image whose header alone is readable.
                converted = image.convert("RGB")
                converted.load()
                if converted.width < 1 or converted.height < 1:
                    raise ValueError(f"invalid converted image size {converted.size}")

                label = sample.get("label")
                if label is None or int(label) < 0:
                    labels_available = False
                    if split_name != "test":
                        raise ValueError("labelled sample has no valid label")

            for warning in caught:
                category = warning.category.__name__
                message = str(warning.message)
                warning_categories[category] += 1
                if "Corrupt EXIF data" in message:
                    corrupt_exif_warnings += 1
                if len(warning_examples) < 25:
                    warning_examples.append(
                        {"index": index, "category": category, "message": message}
                    )
        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )
            progress.set_postfix(errors=len(errors), refresh=False)
        finally:
            if converted is not None and converted is not image and hasattr(converted, "close"):
                converted.close()
            if image is not None and hasattr(image, "close"):
                image.close()

    fully_checked = target == len(dataset)
    return {
        "rows": len(dataset),
        "columns": list(dataset.column_names),
        "features": str(dataset.features),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "has_label_column": has_label_column,
        "sampled_labels_available": labels_available,
        "decoded_samples_checked": target,
        "fully_checked": fully_checked,
        "validation_status": "passed" if not errors else "failed",
        "decode_error_count": len(errors),
        "decode_errors": errors,
        "warning_count": sum(warning_categories.values()),
        "warning_categories": dict(warning_categories),
        "corrupt_exif_warning_count": corrupt_exif_warnings,
        "warning_examples": warning_examples,
        "cache_files": [entry.get("filename") for entry in dataset.cache_files],
    }


def write_manifest(cache_dir: Path, payload: Dict[str, Any]) -> Path:
    path = cache_dir / "imagenet_download_manifest.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_retries < 0:
        raise ValueError("max-retries must be non-negative")
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    token_value = os.environ.get(args.token_env)
    token: Any = token_value if token_value else True
    download_config = DownloadConfig(max_retries=args.max_retries)
    manifest: Dict[str, Any] = {
        "dataset_id": args.dataset_id,
        "dataset_page": DATASET_PAGE,
        "revision": args.revision,
        "cache_dir": str(cache_dir),
        "requested_splits": args.splits,
        "pixel_check": "all" if args.check_samples is None else args.check_samples,
        "validation_passed": False,
        "completed_at_utc": None,
        "splits": {},
    }

    print(f"Dataset: {args.dataset_id}@{args.revision}")
    print(f"Cache:   {cache_dir}")
    print(f"Splits:  {', '.join(args.splits)}")
    print("Existing cached shards will be reused.", flush=True)
    try:
        for split_name in args.splits:
            print(f"\nDownloading/preparing split: {split_name}", flush=True)
            split = load_dataset(
                path=args.dataset_id,
                split=split_name,
                cache_dir=str(cache_dir),
                revision=args.revision,
                token=token,
                download_config=download_config,
            )
            manifest["splits"][split_name] = validate_split(
                split_name, split, args.check_samples
            )
            split_result = manifest["splits"][split_name]
            print(
                f"Prepared {split_name}: {split_result['rows']:,} rows; "
                f"decoded={split_result['decoded_samples_checked']:,}; "
                f"errors={split_result['decode_error_count']:,}; "
                f"EXIF warnings={split_result['corrupt_exif_warning_count']:,}",
                flush=True,
            )
    except Exception as exc:
        print(
            "\nImageNet download failed. Confirm that you accepted the dataset terms at\n"
            f"{DATASET_PAGE}\nand authenticated with `hf auth login` or {args.token_env}.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1

    manifest["validation_passed"] = all(
        result["validation_status"] == "passed" and result["fully_checked"]
        for result in manifest["splits"].values()
    )
    if manifest["validation_passed"]:
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = write_manifest(cache_dir, manifest)
    if not manifest["validation_passed"]:
        print(
            f"\nImage validation FAILED. Training is blocked. Full report: {manifest_path}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(f"\nImageNet cache is ready. Manifest: {manifest_path}")
    print(
        "The cache is ready for run_all_experiments_docker.sh, or run directly with:\n"
        "  python csr_vs_mmpot_imagenet.py --data-backend hf "
        f"--data-root {cache_dir} --method both"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
