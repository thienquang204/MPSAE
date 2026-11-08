"""End-to-end stages, from a bare GPU box to tables and figures.

Each stage is idempotent and leaves a receipt under ``state/``, so the whole
pipeline can be re-run after an interruption and will only redo what is
missing. ``run_all`` is the single entry point the launcher script uses; the
operator supplies credentials and nothing else.

    env       preflight: CUDA, architecture, memory, disk, credentials, imports
    imagenet  fetch ILSVRC/imagenet-1k and materialise PyTorch-style folders
    ffcv      serialise train/val to FFCV, as the CSRv2 recipe requires
    embed     extract frozen embeddings per backbone and gate on dense 1-NN
    publish   push the embedding archives to a private Hugging Face repo
    train     the experiment grid: fit every cell, evaluate E1 and E2
    report    aggregate, write tables, render figures
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shutil
from typing import Any

from . import __version__
from .config import BACKBONES, BACKBONE_ORDER, Backbone
from .io import atomic_json, human_bytes, read_json
from .logs import banner, get_logger, log_environment, setup_logging, stage

STAGES = ("env", "imagenet", "ffcv", "embed", "publish", "train", "report")
#: Compute capabilities we expect to meet, and the lowest CUDA runtime that has
#: kernels for each. CUDA 13 dropped everything below sm_75, so an older card
#: needs a 12.x build; Blackwell needs 12.8 or any 13.x.
ARCHITECTURE_NOTES = {"8.0": ("A100", (12, 0)),
                      "8.9": ("L40S / RTX 4090", (12, 0)),
                      "9.0": ("H100", (12, 1)),
                      "10.0": ("Blackwell datacentre", (12, 8)),
                      "12.0": ("RTX 5090 / RTX PRO 6000", (12, 8))}
#: CUDA 13 compiles only for sm_75 and above.
CUDA13_MIN_CAPABILITY = 7.5


def _cuda_tuple(version: str | None) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(version).split(".")[:2])
    except (TypeError, ValueError):
        return ()


def _driver_major() -> int | None:
    """NVIDIA driver branch, which bounds the CUDA runtime that can load."""
    import subprocess

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except Exception:
        return None
    first = output.strip().splitlines()[0] if output.strip() else ""
    try:
        return int(first.split(".")[0])
    except ValueError:
        return None


@dataclass
class Paths:
    """Every directory the pipeline writes to, all under one workspace."""

    workspace: Path = Path("work")

    @property
    def imagenet(self) -> Path: return self.workspace / "imagenet"
    @property
    def ffcv(self) -> Path: return self.workspace / "ffcv"
    @property
    def embeddings(self) -> Path: return self.workspace / "embeddings"
    @property
    def cache(self) -> Path: return self.workspace / "cache"
    @property
    def runs(self) -> Path: return self.workspace / "runs"
    @property
    def reports(self) -> Path: return self.workspace / "reports"
    @property
    def logs(self) -> Path: return self.workspace / "logs"
    @property
    def state(self) -> Path: return self.workspace / "state"

    def create(self) -> None:
        for path in (self.workspace, self.imagenet, self.ffcv, self.embeddings,
                     self.cache, self.runs, self.reports, self.logs, self.state):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineOptions:
    """Operator-facing switches. Credentials come from the environment."""

    paths: Paths = field(default_factory=Paths)
    backbones: tuple[str, ...] = BACKBONE_ORDER
    methods: tuple[str, ...] | None = None
    seeds: tuple[int, ...] | None = None
    k_values: tuple[int, ...] | None = None
    device: str = "cuda:0"
    embed_batch_size: int = 256
    workers: int = 16
    embed_seed: int = 1227
    hf_repo: str | None = None
    hf_private: bool = True
    wandb_project: str | None = None
    smoke: bool = False
    force: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    max_steps: int = 0
    run_e2: bool = True
    linear_probe: bool = True

    def forced(self, name: str) -> bool:
        return name in self.force or "all" in self.force


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


# ------------------------------------------------------------------ stage: env

def stage_env(options: PipelineOptions) -> dict[str, Any]:
    """Fail now, with instructions, rather than four hours into a download."""
    log = get_logger("env")
    facts = log_environment("env")
    problems: list[str] = []

    if not facts.get("cuda_available"):
        problems.append("no CUDA device visible to torch; every stage after 'imagenet' needs one")
    for gpu in facts.get("gpus", []):
        name, _ = ARCHITECTURE_NOTES.get(gpu["capability"], ("", None))
        log.info("gpu %s is sm_%s%s", gpu["index"], gpu["capability"].replace(".", ""),
                 f" ({name})" if name else "")
        if gpu["total_memory"] < 20 << 30:
            log.warning("gpu %s has only %s; expect to lower --embed-batch-size and "
                        "the search block sizes", gpu["index"], human_bytes(gpu["total_memory"]))

    if facts.get("torch") and facts.get("gpus"):
        runtime = _cuda_tuple(facts.get("cuda"))
        driver = _driver_major()
        facts["driver_major"] = driver
        log.info("torch is built against CUDA %s; driver branch %s",
                 facts.get("cuda"), driver if driver is not None else "unknown")
        caps = sorted({float(gpu["capability"]) for gpu in facts["gpus"]})
        for capability in caps:
            label = f"sm_{int(capability * 10)}"
            _, needed = ARCHITECTURE_NOTES.get(f"{capability:g}", ("", None))
            if needed and runtime and runtime < needed:
                problems.append(
                    f"{label} needs CUDA {needed[0]}.{needed[1]} or newer, but torch is built "
                    f"against {facts.get('cuda')}; reinstall with scripts/setup_server.sh")
            if runtime and runtime[0] >= 13 and capability < CUDA13_MIN_CAPABILITY:
                problems.append(
                    f"torch is a CUDA {runtime[0]} build, which dropped {label}; that GPU needs "
                    "a CUDA 12.x PyTorch build in a matching environment; "
                    "the native setup script targets A100")
        # The driver bounds the runtime that can load at all: CUDA 13 needs the
        # r580 branch, 12.8 needs r570. This fails at the first kernel launch,
        # often minutes into a stage, so it is worth catching here.
        if driver is not None and runtime:
            required_driver = 580 if runtime[0] >= 13 else (570 if runtime >= (12, 8) else 525)
            if driver < required_driver:
                problems.append(
                    f"driver branch {driver} cannot load a CUDA {facts.get('cuda')} build "
                    f"(needs >= {required_driver}); either update the driver or install a "
                    "matching wheel with scripts/setup_server.sh")

    needs_ffcv = any(BACKBONES[name].recipe == "csrv2" for name in options.backbones)
    for module, why in (("torch", "training"), ("numpy", "everything"),
                        ("timm", "timm backbones"), ("huggingface_hub", "downloads"),
                        ("datasets", "ImageNet"), ("matplotlib", "figures"),
                        *(((("ffcv", "the CSRv2 embedding recipe"),
                            ("cupy", "FFCV GPU normalization")) if needs_ffcv else ()))):
        try:
            __import__(module)
        except Exception as error:
            problems.append(f"missing {module} (needed for {why}): {error}")

    if not _env("HF_TOKEN"):
        problems.append("HF_TOKEN is unset; ILSVRC/imagenet-1k is gated. Accept the terms at "
                        "https://huggingface.co/datasets/ILSVRC/imagenet-1k and export a token")
    if options.hf_repo and not _env("HF_TOKEN"):
        problems.append("publishing needs a write-scoped HF_TOKEN")

    free = shutil.disk_usage(options.paths.workspace).free
    needed = 60 << 30 if options.smoke else 500 << 30
    log.info("workspace %s: %s free (want %s)", options.paths.workspace,
             human_bytes(free), human_bytes(needed))
    if free < needed:
        problems.append(f"workspace has {human_bytes(free)} free, needs about {human_bytes(needed)}")

    for problem in problems:
        log.error("%s", problem)
    if problems:
        raise RuntimeError(f"{len(problems)} preflight problem(s); see the log above")
    log.info("preflight passed")
    return {"environment": facts, "needs_ffcv": needs_ffcv}


# ------------------------------------------------------------- stage: imagenet

def stage_imagenet(options: PipelineOptions) -> dict[str, Any]:
    from .data import prepare_imagenet

    if options.smoke:
        get_logger("imagenet").info(
            "smoke mode: no ImageNet download. The embed stage will synthesise archives "
            "instead, which exercises training, evaluation and reporting end to end.")
        return {"skipped": "smoke"}
    # Its own receipt goes in a nested directory: sharing `state/` would let the
    # outer stage wrapper overwrite the inner one, losing the dataset revision
    # that makes the download reproducible.
    return prepare_imagenet(options.paths.imagenet, token=_env("HF_TOKEN"),
                            state_dir=options.paths.state / "imagenet")


# ----------------------------------------------------------------- stage: ffcv

def stage_ffcv(options: PipelineOptions) -> dict[str, Any]:
    from .data import write_ffcv

    if options.smoke:
        return {"skipped": "smoke"}
    written = {}
    for split, folder in (("train", "train"), ("val", "val")):
        written[split] = str(write_ffcv(options.paths.imagenet / folder,
                                        options.paths.ffcv, split,
                                        num_workers=options.workers))
    return {"files": written}


# ---------------------------------------------------------------- stage: embed

def _embed_one(spec: Backbone, options: PipelineOptions, split: str) -> dict[str, Any]:
    from .embed import extract_ff2048, extract_timm

    if spec.recipe == "mrl":
        return extract_ff2048(spec, options.paths.imagenet, options.paths.embeddings / spec.name,
                              split=split, cache_dir=options.paths.cache,
                              device=options.device, batch_size=options.embed_batch_size,
                              workers=options.workers, hf_token=_env("HF_TOKEN"))
    ffcv_file = next(options.paths.ffcv.glob(
        f"{'train' if split == 'train' else 'val'}_*.ffcv"), None)
    if ffcv_file is None:
        raise FileNotFoundError(f"no FFCV file for {split} in {options.paths.ffcv}; run the ffcv stage")
    return extract_timm(spec, ffcv_file, options.paths.embeddings / spec.name, split=split,
                        cache_dir=options.paths.cache, device=options.device,
                        batch_size=options.embed_batch_size, workers=options.workers,
                        seed=options.embed_seed)


def _synthesise_embeddings(spec: Backbone, options: PipelineOptions, *, split: str,
                           classes: int = 50, per_class: int = 40) -> dict[str, Any]:
    """Fake but structured archives, so a smoke run can reach the report stage.

    Class-conditional Gaussians give a dense 1-NN accuracy well above chance,
    which is enough to prove the retrieval and probe paths are wired correctly.
    These archives are not science and are labelled as such in the manifest.
    """
    import zlib

    import numpy as np

    from .embed import finalize_archive

    rows = classes * (per_class if split == "train" else max(2, per_class // 8))
    # The class centres must be shared by both splits, or a validation query
    # has no counterpart in the gallery and 1-NN sits at chance -- which would
    # make the rehearsal unable to tell a working retrieval path from a broken
    # one. zlib.crc32 rather than hash(): the latter is salted per process.
    base = zlib.crc32(spec.name.encode())
    centres = np.random.default_rng(base).normal(
        scale=1.0, size=(classes, spec.dim)).astype(np.float32)
    noise = np.random.default_rng(base + (0 if split == "train" else 1))
    labels = np.repeat(np.arange(classes), rows // classes).astype(np.int64)
    features = centres[labels] + noise.normal(
        scale=0.35, size=(len(labels), spec.dim)).astype(np.float32)
    name = "1K_train_{}.npz" if split == "train" else "1K_val_{}.npz"
    out_path = options.paths.embeddings / spec.name / name.format(spec.name)
    return finalize_archive(out_path, features, labels, manifest={
        "recipe": "synthetic-smoke", "backbone": spec.name, "split": split,
        "rows": len(labels), "dim": spec.dim, "classes": classes,
        "source": "class-conditional Gaussians", "precision": "float32",
        "deviations": ["SYNTHETIC DATA: smoke-test archive, not an ImageNet embedding"]})


def stage_embed(options: PipelineOptions) -> dict[str, Any]:
    """Extract every selected backbone, then check the dense retrieval gate."""
    from .features import load_bundle
    from .runs import RunSettings, dense_reference
    from .train import FitSettings

    log = get_logger("embed")
    produced: dict[str, Any] = {}
    for name in options.backbones:
        spec = BACKBONES[name]
        with stage(f"embed-{name}", options.paths.state, force=options.forced("embed"),
                   identity={"recipe": spec.recipe, "source": spec.source, "dim": spec.dim,
                             "normalize": spec.normalize, "input_size": spec.input_size,
                             "seed": options.embed_seed, "smoke": options.smoke},
                   description=f"{name}: {spec.recipe} recipe from {spec.source}") as detail:
            if not detail.get("skipped"):
                build = _synthesise_embeddings if options.smoke else None
                for split in ("train", "validation"):
                    detail[split] = (build(spec, options, split=split) if build
                                     else _embed_one(spec, options, split))
        produced[name] = detail

        settings = RunSettings(root=options.paths.runs, embeddings_dir=options.paths.embeddings,
                               cache_dir=options.paths.cache,
                               fit=FitSettings(device=options.device))
        bundle = load_bundle(spec, options.paths.embeddings, options.paths.cache)
        gate = dense_reference(spec, bundle, settings)
        produced[name]["dense_reference"] = gate
        if options.smoke:
            log.info("%s: synthetic dense 1-NN %.2f%% (gate not applicable)", name, gate["top1"])
        elif (spec.reference_dense_top1 is not None
              and abs(gate.get("deviation", 0)) > settings.dense_gate_fail):
            raise RuntimeError(
                f"{name}: dense 1-NN {gate['top1']:.2f}% is {gate['deviation']:+.2f} from the "
                f"published {spec.reference_dense_top1:.2f}%. The embedding recipe does not "
                "match the published one; fix it before training, or the comparison against "
                "the published tables is meaningless.")
        log.info("%s ready: dense 1-NN %.2f%%", name, gate["top1"])
    return {"backbones": produced}


# -------------------------------------------------------------- stage: publish

def stage_publish(options: PipelineOptions) -> dict[str, Any]:
    """Upload the embedding archives to a private Hugging Face dataset repo.

    This sends data off the machine, so it only happens when the operator has
    named a repository; it is never implied by a bare pipeline run.
    """
    log = get_logger("publish")
    if not options.hf_repo:
        log.info("no repository configured (set HF_EMBEDDINGS_REPO); skipping upload")
        return {"uploaded": False, "reason": "no repository configured"}
    from huggingface_hub import HfApi

    token = _env("HF_TOKEN")
    api = HfApi(token=token)
    log.info("uploading %s to %s (private=%s)", options.paths.embeddings,
             options.hf_repo, options.hf_private)
    api.create_repo(options.hf_repo, repo_type="dataset", private=options.hf_private,
                    exist_ok=True)
    uploaded = []
    for name in options.backbones:
        folder = options.paths.embeddings / name
        if not folder.is_dir():
            continue
        api.upload_folder(repo_id=options.hf_repo, repo_type="dataset",
                          folder_path=str(folder), path_in_repo=name,
                          commit_message=f"{name} embeddings ({__version__})")
        uploaded.append(name)
        log.info("uploaded %s", name)
    return {"uploaded": True, "repo": options.hf_repo, "private": options.hf_private,
            "backbones": uploaded}


# ---------------------------------------------------------------- stage: train

def stage_train(options: PipelineOptions) -> dict[str, Any]:
    from .runs import RunSettings, plan, run_grid
    from .train import FitSettings

    cells = plan(options.backbones, options.methods, options.seeds, options.k_values)
    settings = RunSettings(
        root=options.paths.runs, embeddings_dir=options.paths.embeddings,
        cache_dir=options.paths.cache, reports_dir=options.paths.reports,
        linear_probe=options.linear_probe, run_e2=options.run_e2,
        force=options.forced("train"),
        fit=FitSettings(device=options.device, max_steps=options.max_steps,
                        wandb_project=options.wandb_project,
                        force=options.forced("train")))
    return run_grid(cells, settings, code_version=__version__, dense_gate=False)


# --------------------------------------------------------------- stage: report

def stage_report(options: PipelineOptions) -> dict[str, Any]:
    from .report import build_all

    return build_all(options.paths.runs, options.paths.reports)


HANDLERS = {"env": stage_env, "imagenet": stage_imagenet, "ffcv": stage_ffcv,
            "embed": stage_embed, "publish": stage_publish, "train": stage_train,
            "report": stage_report}


def needed_stages(options: PipelineOptions) -> tuple[str, ...]:
    """Drop stages this configuration cannot use."""
    chosen = [name for name in STAGES if name not in options.skip]
    if not any(BACKBONES[name].recipe == "csrv2" for name in options.backbones):
        chosen = [name for name in chosen if name != "ffcv"]
    if not any(BACKBONES[name].recipe == "mrl" for name in options.backbones) \
            and "imagenet" in chosen and "ffcv" not in chosen:
        pass  # the MRL recipe is the only ImageFolder consumer, but FFCV needs the folders too
    return tuple(chosen)


def run_all(options: PipelineOptions, *, only: str | None = None) -> dict[str, Any]:
    """Run the pipeline, or a single stage when ``only`` is given."""
    options.paths.create()
    setup_logging(options.paths.logs, name=only or "pipeline")
    banner(f"MPSAE {__version__} | workspace {options.paths.workspace}"
           + (" | SMOKE" if options.smoke else ""), stage="pipeline")
    log = get_logger("pipeline")
    log.info("backbones: %s", ", ".join(options.backbones))
    if options.smoke:
        log.warning("smoke mode: tiny subsets and step caps. Results are not scientific.")

    chosen = (only,) if only else needed_stages(options)
    for name in chosen:
        if name not in HANDLERS:
            raise ValueError(f"unknown stage {name!r}; choose from {STAGES}")
    atomic_json(options.paths.state / "options.json",
                {"version": __version__, "stages": list(chosen),
                 "options": {key: str(value) for key, value in options.__dict__.items()}})

    outcomes: dict[str, Any] = {}
    for name in chosen:
        with stage(name, options.paths.state, force=options.forced(name),
                   identity={"version": __version__, "options": asdict(options)},
                   description=f"stage {name}") as detail:
            if not detail.get("skipped"):
                detail.update(HANDLERS[name](options) or {})
        outcomes[name] = detail
    log.info("pipeline complete: %s", ", ".join(chosen))
    return outcomes


def status(options: PipelineOptions) -> dict[str, Any]:
    """What is done, what is outstanding.

    Reads the result files directly rather than going through ``mpsae.runs``,
    so it stays usable on a machine without torch.
    """
    from .config import METHODS, SEEDS, grid

    cells = grid(options.backbones, options.methods or METHODS,
                 options.seeds or SEEDS, options.k_values)
    outstanding = []
    for cell in cells:
        path = (options.paths.runs / cell.backbone / cell.method
                / f"k{cell.k}" / f"seed{cell.seed}" / "eval_e1.json")
        record = read_json(path)
        if not record or "top1" not in record:
            outstanding.append(cell)
    stages = {name: bool((read_json(options.paths.state / f"{name}.json") or {}).get("ok"))
              for name in STAGES}
    return {"stages": stages, "cells": len(cells), "outstanding": len(outstanding),
            "done": len(cells) - len(outstanding),
            "by_backbone": {name: sum(1 for cell in outstanding if cell.backbone == name)
                            for name in options.backbones},
            "by_method": {name: sum(1 for cell in outstanding if cell.method == name)
                          for name in (options.methods or METHODS)}}
