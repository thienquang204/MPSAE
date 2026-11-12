"""Joint training, reports, and evaluation-only transfer of saved checkpoints."""
import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Additional supervised backbone + SAE fine-tuning")
    parser.add_argument("command", choices=("plan", "run", "report", "transfer", "transfer-report", "prepare", "sync", "restore-sources"))
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--source-root", help="Completed frozen fits and provenance; default work/direct-v1")
    parser.add_argument("--repo", help="HF dataset repository for restore-sources")
    parser.add_argument("--revision", default="main", help="Restore source snapshot (resolved to an immutable commit)")
    parser.add_argument("--image-root", help="ImageNet train/val folders; default work/imagenet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extract-batch", type=int, default=64)
    parser.add_argument("--download-images", action="store_true", help="Resume raw ImageNet preparation if needed")
    parser.add_argument("--image-storage", choices=("folder", "packed"), default="packed",
                        help="Reusable original-JPEG packs (default) or existing ImageFolder loader")
    parser.add_argument("--backup-interval", type=float, default=1800,
                        help="Minimum seconds between intermediate checkpoint uploads")
    parser.add_argument("--backup", action="store_true", help="Explicitly enable private HF backup and optional W&B")
    parser.add_argument("--smoke", action="store_true", help="Synthetic capped test; no downloads or remote writes")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    # Stop before dotenv, Torch, CUDA, source restoration or any data downloads.
    if args.command == "plan" or not config.enabled:
        print(json.dumps(config.plan(), indent=2))
        return
    from ..cli import load_dotenv
    from ..io import read_json
    from .run import output_root, run, write_report
    load_dotenv(Path(".env"))
    if args.command == "restore-sources":
        import os
        if not args.repo or args.smoke:
            parser.error("restore-sources needs --repo and cannot use --smoke")
        from .restore import restore_sources
        print(json.dumps(restore_sources(config, args.workspace, args.repo,
            revision=args.revision, token=os.environ.get("HF_TOKEN")), indent=2))
        return
    if args.command == "prepare":
        import os
        if args.smoke:
            parser.error("prepare operates on real images; do not use --smoke")
        image_root = args.image_root or str(Path(args.workspace) / "imagenet")
        if args.image_storage == "packed":
            from .jpeg_store import prepare_packs
            datasets, identity = prepare_packs(image_root, token=os.environ.get("HF_TOKEN"),
                                              download=args.download_images)
            for dataset in datasets.values():
                dataset.close()
        else:
            from ..data.imagenet import prepare_imagenet
            identity = prepare_imagenet(image_root, token=os.environ.get("HF_TOKEN"), download=args.download_images)
        print(json.dumps(identity, indent=2))
        return
    if args.command == "sync":
        if args.smoke:
            parser.error("sync cannot upload smoke results")
        from .backup import sync
        sync(config, output_root(config, args.workspace).resolve())
        return
    if args.command in ("transfer", "transfer-report"):
        from .transfer import run_transfer, report
        if args.command == "transfer":
            result = run_transfer(config, args.workspace, device=args.device, workers=args.workers,
                extract_batch=args.extract_batch, smoke=args.smoke)
        else:
            root = output_root(config, args.workspace, args.smoke)
            path = root / "campaigns" / config.name / config.plan()["plan_id"] / "transfer.json"
            catalog = read_json(path)
            if catalog is None:
                raise FileNotFoundError("no transfer results for this plan; run transfer first")
            result = report(config, root, catalog["records"], smoke=args.smoke)
        print(json.dumps({"complete": result["complete"], "scientific": result["scientific"],
                          "reports": result["out_dir"], "counts": result["counts"]}, indent=2))
        if args.backup and not args.smoke:
            from .backup import sync
            sync(config, output_root(config, args.workspace).resolve())
        if not result["complete"]:
            raise RuntimeError("transfer evaluation/report is incomplete; inspect its manifest")
        return
    if args.command == "report":
        root = output_root(config, args.workspace, args.smoke)
        catalog = read_json(root / "campaigns" / config.name / config.plan()["plan_id"] / "catalog.json")
        if catalog is None:
            raise FileNotFoundError("no results for this fine-tuning campaign; run it first")
        result = write_report(config, root, catalog, [])
        if args.backup and not args.smoke:
            from .backup import sync
            sync(config, root.resolve())
    else:
        result = run(config, args.workspace, source_root=args.source_root, image_root=args.image_root,
            device=args.device, workers=args.workers, extract_batch=args.extract_batch,
            download_images=args.download_images, smoke=args.smoke, backup=args.backup,
            image_storage=args.image_storage, backup_interval=args.backup_interval)
    print(json.dumps({"complete": result["complete"], "scientific": result["scientific"],
                      "comparison": result["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
