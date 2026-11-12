"""python -m mpsae.finetune --config configs/finetune.yaml plan|run|report"""
import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Additional supervised backbone + SAE fine-tuning")
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--source-root", help="Completed frozen fits and provenance; default work/direct-v1")
    parser.add_argument("--image-root", help="ImageNet train/val folders; default work/imagenet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extract-batch", type=int, default=64)
    parser.add_argument("--download-images", action="store_true", help="Resume raw ImageNet preparation if needed")
    parser.add_argument("--backup", action="store_true", help="Explicitly enable private HF backup and optional W&B")
    parser.add_argument("--smoke", action="store_true", help="Synthetic capped test; no downloads or remote writes")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "plan":
        print(json.dumps(config.plan(), indent=2))
        return
    from ..cli import load_dotenv
    from ..io import read_json
    from .run import output_root, run, write_report
    load_dotenv(Path(".env"))
    if args.command == "report":
        root = output_root(config, args.workspace, args.smoke)
        catalog = read_json(root / "campaigns" / config.name / config.plan()["plan_id"] / "catalog.json")
        if catalog is None:
            raise FileNotFoundError("no results for this fine-tuning campaign; run it first")
        result = write_report(config, root, catalog, [])
    else:
        result = run(config, args.workspace, source_root=args.source_root, image_root=args.image_root,
            device=args.device, workers=args.workers, extract_batch=args.extract_batch,
            download_images=args.download_images, smoke=args.smoke, backup=args.backup)
    print(json.dumps({"complete": result["complete"], "scientific": result["scientific"],
                      "comparison": result["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
