"""Command line entry point: ``python -m mpsae <command>``.

Credentials are read from the environment (normally via ``.env``); everything
else has a default that reproduces the paper settings. ``run`` executes the
whole pipeline, which is what ``pipeline.sh`` calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__
from .config import BACKBONE_ORDER, BACKBONES, METHODS, SEEDS
from .logs import get_logger, setup_logging
from .pipeline import STAGES, Paths, PipelineOptions, run_all, status


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Populate ``os.environ`` from a ``KEY=value`` file without overriding it.

    Values already in the environment win, so ``HF_TOKEN=... python -m mpsae``
    behaves the way an operator expects.
    """
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _csv(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpsae", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--env-file", default=".env",
                        help="KEY=value file with HF_TOKEN and friends (default: .env)")
    parser.add_argument("--config", type=Path, default=None,
                        help="one YAML/JSON campaign for ImageNet and frozen downstream benchmarks")
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("MPSAE_WORKSPACE", "work")),
                        help="all inputs, outputs and logs live here")
    parser.add_argument("--device", default=os.environ.get("MPSAE_DEVICE", "cuda:0"))
    parser.add_argument("--backbones", type=_csv,
                        default=_csv(os.environ.get("MPSAE_BACKBONES", ",".join(BACKBONE_ORDER))),
                        help=f"subset of {','.join(BACKBONE_ORDER)}")
    parser.add_argument("--methods", type=_csv,
                        default=_csv(os.environ.get("MPSAE_METHODS", "")) or None,
                        help=f"subset of {','.join(METHODS)} (default: all)")
    parser.add_argument("--seeds", type=_ints,
                        default=_ints(os.environ.get("MPSAE_SEEDS", "")) or None,
                        help=f"default {','.join(map(str, SEEDS))}")
    parser.add_argument("--k", type=_ints, dest="k_values",
                        default=_ints(os.environ.get("MPSAE_K", "")) or None,
                        help="restrict to these budgets (default: each backbone's full grid)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MPSAE_WORKERS", "16")))
    parser.add_argument("--embed-batch-size", type=int,
                        default=int(os.environ.get("MPSAE_EMBED_BATCH", "256")))
    parser.add_argument("--hf-repo", default=os.environ.get("HF_EMBEDDINGS_REPO") or None,
                        help="private dataset repo for the embedding archives; "
                             "omitting it skips the publish stage entirely")
    parser.add_argument("--public", action="store_true",
                        help="publish the repo publicly (default: private -- ImageNet-derived "
                             "features inherit the dataset's non-commercial terms)")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or None)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subsets and capped steps, to prove the plumbing works")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="cap optimiser steps per fit (0 = full schedule)")
    parser.add_argument("--no-e2", action="store_true", help="skip the one-checkpoint experiment")
    parser.add_argument("--no-probe", action="store_true", help="skip linear probes")
    parser.add_argument("--force", type=_csv, default=(),
                        help=f"re-run these stages even if their receipt says done: {','.join(STAGES)},all")
    parser.add_argument("--skip", type=_csv, default=(), help="stages to leave out")
    parser.add_argument("--log-level", default=os.environ.get("MPSAE_LOG_LEVEL", "INFO"))

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run the whole pipeline (default order of stages)")
    one = sub.add_parser("stage", help="run exactly one stage")
    one.add_argument("name", choices=STAGES)
    sub.add_parser("status", help="what is finished and what is outstanding")
    sub.add_parser("plan", help="print the experiment grid and its size")
    report = sub.add_parser("report", help="rebuild tables and figures from existing runs")
    report.add_argument("--out", type=Path, default=None)
    return parser


def to_options(args: argparse.Namespace) -> PipelineOptions:
    for name in args.backbones:
        if name not in BACKBONES:
            raise SystemExit(f"unknown backbone {name!r}; choose from {', '.join(BACKBONE_ORDER)}")
    if args.methods:
        for name in args.methods:
            if name not in METHODS:
                raise SystemExit(f"unknown method {name!r}; choose from {', '.join(METHODS)}")
    return PipelineOptions(
        paths=Paths(workspace=args.workspace / "smoke-legacy" if args.smoke else args.workspace),
        backbones=tuple(args.backbones),
        methods=args.methods, seeds=args.seeds, k_values=args.k_values,
        device=args.device, embed_batch_size=args.embed_batch_size, workers=args.workers,
        hf_repo=args.hf_repo, hf_private=not args.public,
        wandb_project=args.wandb_project, smoke=args.smoke,
        force=tuple(args.force), skip=tuple(args.skip),
        max_steps=args.max_steps or (20 if args.smoke else 0),
        run_e2=not args.no_e2, linear_probe=not args.no_probe)


def command_plan(options: PipelineOptions) -> int:
    # Deliberately avoids mpsae.runs: inspecting the grid must work on a laptop
    # without torch installed.
    from .config import grid

    cells = grid(options.backbones, options.methods or METHODS,
                 options.seeds or SEEDS, options.k_values)
    counts = {name: sum(1 for cell in cells if cell.backbone == name)
              for name in options.backbones}
    print(f"MPSAE {__version__}: {len(cells)} training cells\n")
    header = f"{'backbone':16s} {'d':>5s} {'K budgets':<44s} {'E2 anchors':<22s} {'cells':>6s}"
    print(header)
    print("-" * len(header))
    for name in options.backbones:
        spec = BACKBONES[name]
        print(f"{name:16s} {spec.dim:5d} {','.join(map(str, spec.training_budgets)):<44s} "
              f"{','.join(map(str, spec.e2_anchors)):<22s} {counts[name]:6d}")
    print("-" * len(header))
    print(f"{'total':16s} {'':5s} {'':44s} {'':22s} {len(cells):6d}")
    print(f"\nmethods: {', '.join(options.methods or METHODS)}")
    print(f"seeds:   {', '.join(map(str, options.seeds or SEEDS))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Environment-based defaults are evaluated while constructing the parser.
    # Load the requested file first, rather than reparsing stale defaults.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env-file", default=".env")
    preliminary, _ = bootstrap.parse_known_args(argv)
    loaded = load_dotenv(preliminary.env_file)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 0 or args.embed_batch_size < 1 or args.max_steps < 0:
        parser.error("workers must be nonnegative, embed-batch-size positive, and max-steps nonnegative")
    unknown_stages = (set(args.force) - {*STAGES, "all"}) | (set(args.skip) - set(STAGES))
    if unknown_stages:
        parser.error("unknown stage(s): " + ", ".join(sorted(unknown_stages)))
    if args.config is None and args.command != "stage" and os.environ.get("MPSAE_CONFIG"):
        args.config = Path(os.environ["MPSAE_CONFIG"])
    if args.config is not None:
        from .benchmark.config import load_campaign
        from .benchmark.run import (campaign_status, campaign_workspace, report_campaign,
                                    run_campaign)

        forbidden = {"--backbones", "--methods", "--seeds", "--k", "--max-steps", "--no-e2",
                     "--no-probe", "--force", "--skip", "--hf-repo", "--public"}
        supplied = {part.split("=", 1)[0] for part in (argv if argv is not None else sys.argv[1:])}
        if forbidden & supplied:
            parser.error("with --config, select experiments in YAML rather than "
                         + ", ".join(sorted(forbidden & supplied)))
        if args.command == "stage":
            parser.error("campaigns use run/plan/status/report; omit --config for legacy stage commands")
        try:
            campaign = load_campaign(args.config)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        if args.command == "plan":
            print(json.dumps(campaign.plan(), indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(campaign_status(campaign, args.workspace, smoke=args.smoke), indent=2))
            return 0
        workspace = campaign_workspace(args.workspace, campaign, args.smoke)
        setup_logging(workspace / "logs", name=campaign.name, level=args.log_level)
        if args.command == "report":
            if args.out is not None:
                parser.error("campaign report paths are fixed by campaign name/plan; --out is legacy-only")
            result = report_campaign(campaign, args.workspace, smoke=args.smoke)
        else:
            result = run_campaign(campaign, args.workspace, device=args.device,
                                  workers=args.workers, embed_batch_size=args.embed_batch_size,
                                  smoke=args.smoke)
        print(json.dumps(result, indent=2, default=str))
        return 0
    options = to_options(args)

    if args.command == "plan":
        return command_plan(options)

    options.paths.create()
    setup_logging(options.paths.logs, name=args.command, level=args.log_level)
    log = get_logger("cli")
    if loaded:
        log.info("loaded %d key(s) from %s: %s", len(loaded), args.env_file, ", ".join(loaded))

    if args.command == "status":
        report = status(options)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "report":
        from .report import build_all

        manifest = build_all(options.paths.runs, args.out or options.paths.reports)
        print(json.dumps(manifest, indent=2, default=str))
        return 0
    if args.command == "stage":
        run_all(options, only=args.name)
        return 0
    run_all(options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
