"""The experiment grid: train every cell, then evaluate E1 and E2.

E1 (per-K) is the comparison the baselines were designed for: one model
trained at each budget, evaluated at that budget. It is what CSR Table 4
reports and the only fair way to put our methods next to theirs.

E2 (one checkpoint, many budgets) reuses the E1 checkpoints at the anchor
budgets and evaluates them at every budget in the grid. It costs no extra
training and it measures a claim both papers make but neither tests --
CSRv2 Section 3: "Training with the multi-TopK loss L(k)+L(4k)/8 ensures that
CSR could generalize to different k's at test time" -- while the CSR author
writes in issue #17 that changing k at inference "might lead to sub-optimal
performance". Because the measured number of nonzeros can be far below the
nominal budget once ReLU has acted, every E2 point carries its realised nnz.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import contextmanager
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator

from .config import BACKBONES, Backbone, Cell, grid
from .features import Bundle, load_bundle
from .io import atomic_json, canonical_hash, read_json
from .logs import Progress, banner, format_duration, get_logger
from .train import FitSettings, fit, load_fitted
from .timing_config import timing_config, runtime_fingerprint


@dataclass
class RunSettings:
    """Everything the orchestration needs that is not part of the objective."""

    root: Path = Path("runs")
    embeddings_dir: Path = Path("data/embeddings")
    cache_dir: Path = Path("data/cache")
    reports_dir: Path = Path("reports")
    fit: FitSettings = field(default_factory=FitSettings)
    encode_batch_size: int = 1024
    gallery_chunk: int = 32768
    query_block: int = 8192
    linear_probe: bool = True
    probe_epochs: int = 10
    measure_timing: bool = True
    timing_query_batch: int = 512
    timing_warmup: int = 100
    timing_rounds: int = 2000
    timing_gallery_chunk: int = 32768
    timing_build_entry_cap: int = 4 * 1024 * 1024
    timing_retry_skipped: bool = False
    run_e2: bool = True
    e2_k_values: tuple[int, ...] | None = None  # None keeps the registered inference grid
    #: Two tiers on the published dense 1-NN. Within ``warn`` the embedding
    #: recipe is indistinguishable from the published one; beyond ``fail`` it
    #: demonstrably is not, and training would produce numbers that cannot be
    #: compared with the published tables. Between the two, say so and let the
    #: operator decide.
    dense_gate_warn: float = 0.05
    dense_gate_fail: float = 0.3
    force: bool = False
    #: Legacy callers retain code caches. Campaigns can discard them after
    #: each evaluation point to avoid multi-TiB E2 cache accumulation.
    retain_codes: bool = True

    def timing_identity(self) -> dict[str, Any]:
        return {**timing_config(query_batch=self.timing_query_batch, warmup=self.timing_warmup,
                                rounds=self.timing_rounds, gallery_chunk=self.timing_gallery_chunk,
                                build_entry_cap=self.timing_build_entry_cap),
                "runtime": runtime_fingerprint(self.fit.device)}


def cell_dir(settings: RunSettings, cell: Cell) -> Path:
    return Path(settings.root) / cell.slug


def _evaluate_module():
    try:
        from . import evaluate
    except ImportError as error:  # pragma: no cover - surfaced to the operator
        raise RuntimeError(
            "the evaluation package is unavailable; install torch and the project "
            f"requirements before running evaluation ({error})") from error
    return evaluate


def _encode_identity(cell: Cell, bundle: Bundle, split: str, k: int, signature: str) -> dict[str, Any]:
    return {"fit": signature, "split": split, "k": k, "backbone": cell.backbone,
            "method": cell.method, "seed": cell.seed, "data": bundle.identity.get(split)}


# ----------------------------------------------------------- dense reference

def dense_reference(spec: Backbone, bundle: Bundle, settings: RunSettings) -> dict[str, Any]:
    """Measure the frozen backbone itself, and gate the pipeline on it.

    For FF2048 the published "Full Rep." 1-NN is 71.19 (CSR Table 4). Missing
    it means the embedding recipe diverged from MRL's, which would invalidate
    every comparison against the published tables -- so this runs before any
    training and the operator is told rather than discovering it later.
    """
    evaluate = _evaluate_module()
    log = get_logger(f"dense {spec.name}")
    path = Path(settings.root) / spec.name / "dense_reference.json"
    previous = read_json(path)
    identity = canonical_hash(bundle.identity)
    # Older caches treated an all-negative dense vector as empty. Recompute
    # those backbones once; the two released ResNets emit nonnegative ReLU
    # features, so their historical dense references remain compatible.
    zero_policy_ok = (spec.name in ("ff2048", "rn18") or (previous or {}).get("zero_detection")
                      == "all_coordinates_equal_zero")
    if (previous and previous.get("identity") == identity and not settings.force and zero_policy_ok
            and isinstance(previous.get("top1"), (int, float))
            and not isinstance(previous["top1"], bool) and math.isfinite(previous["top1"])):
        log.info("dense 1-NN %.2f%% (cached)", previous["top1"])
        return previous
    result = evaluate.dense_top1(
        bundle.train.features, bundle.train.labels,
        bundle.validation.features, bundle.validation.labels,
        device=settings.fit.device, gallery_chunk=settings.gallery_chunk,
        query_block=settings.query_block, dim=spec.dim)
    record = {"backbone": spec.name, "dim": spec.dim, "identity": identity,
              "zero_detection": "all_coordinates_equal_zero",
              # The dense gallery is the footprint the sparse codes are compared
              # against on the efficiency figure.
              "index_bytes": int(bundle.train.rows) * int(spec.dim) * 4, **result}
    expected = spec.reference_dense_top1
    if expected is not None:
        record["published_top1"] = expected
        record["deviation"] = result["top1"] - expected
        if abs(record["deviation"]) <= settings.dense_gate_warn:
            log.info("dense 1-NN %.2f%% matches the published %.2f%%", result["top1"], expected)
        else:
            log.warning("dense 1-NN %.2f%% deviates %+0.2f from the published %.2f%% "
                        "(warn above %.2f, fail above %.2f)", result["top1"], record["deviation"],
                        expected, settings.dense_gate_warn, settings.dense_gate_fail)
    else:
        log.info("dense 1-NN %.2f%% (no published value for this backbone)", result["top1"])
    atomic_json(path, record)
    return record


# ------------------------------------------------------------- one eval point

@contextmanager
def _code_cache(settings: RunSettings, identity: dict[str, Any]) -> Iterator[Path]:
    """A retained legacy cache, or a uniquely owned transient subdirectory."""
    if settings.retain_codes:
        yield Path(settings.cache_dir) / "codes"
        return
    cache_root = Path(settings.cache_dir).resolve()
    temporary_root = cache_root / "transient-codes"
    if temporary_root.is_symlink():
        raise RuntimeError(f"refusing transient cache through a symlink: {temporary_root}")
    temporary_root.mkdir(parents=True, exist_ok=True)
    if temporary_root.resolve().parent != cache_root:
        raise RuntimeError(f"transient cache escapes configured cache root: {temporary_root}")
    prefix = canonical_hash(identity)[:24] + "-"
    owned = Path(tempfile.mkdtemp(prefix=prefix, dir=temporary_root))
    try:
        yield owned
    finally:
        # Verify the final absolute target immediately before recursion. The
        # invocation owns only this new directory, never the shared codes/.
        target = owned.resolve()
        if (temporary_root.is_symlink() or temporary_root.resolve().parent != cache_root
                or owned.is_symlink() or target.parent != temporary_root.resolve()
                or target == cache_root or not target.name.startswith(prefix)):
            raise RuntimeError(f"refusing unexpected transient cleanup target: {target}")
        if target.exists():
            shutil.rmtree(target)


def _close_code_arrays(codes) -> None:
    """Close mapped files before deletion even when tracebacks retain locals."""
    if not isinstance(codes, dict):
        return
    seen: set[int] = set()
    for key in ("indices", "values"):
        array = codes.get(key)
        while array is not None and id(array) not in seen:
            seen.add(id(array))
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
            array = getattr(array, "base", None)


def _close_failed_cache_maps(error: BaseException, cache: Path) -> None:
    """Close partial encoder writers retained by an exception traceback.

    An encoding failure can happen before encode_split returns its arrays.
    Inspect only small containers/frame locals and close mappings whose files
    resolve inside this invocation's owned cache; source feature maps outside
    it are never touched. Tracebacks and their useful debugging locals remain.
    """
    root = cache.resolve()
    seen: set[int] = set()

    def visit(value, depth: int = 0) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict) and depth < 3:
            for nested in value.values():
                visit(nested, depth + 1)
            return
        if isinstance(value, (tuple, list)) and len(value) <= 32 and depth < 3:
            for nested in value:
                visit(nested, depth + 1)
            return
        array = value
        bases: set[int] = set()
        while array is not None and id(array) not in bases:
            bases.add(id(array))
            mapping = getattr(array, "_mmap", None)
            filename = getattr(array, "filename", None)
            if mapping is not None and filename is not None and not mapping.closed:
                mapped_path = Path(filename).resolve()
                if mapped_path.is_relative_to(root):
                    mapping.close()
            array = getattr(array, "base", None)

    traceback = error.__traceback__
    while traceback is not None:
        for value in list(traceback.tb_frame.f_locals.values()):
            visit(value)
        traceback = traceback.tb_next


def evaluate_at(cell: Cell, bundle: Bundle, model, spec: Backbone, settings: RunSettings,
                *, k: int, signature: str, want_probe: bool, want_timing: bool,
                previous: dict[str, Any] | None = None,
                on_progress: Callable[[dict[str, Any]], None] | None = None
                ) -> dict[str, Any]:
    """Encode gallery and queries at budget ``k``, then score exact 1-NN."""
    identity = {"fit": signature, "data": bundle.identity, "k": k,
                "backbone": cell.backbone, "method": cell.method, "seed": cell.seed}
    with _code_cache(settings, identity) as cache:
        return _evaluate_at_cached(cell, bundle, model, spec, settings, k=k,
                                   signature=signature, want_probe=want_probe,
                                   want_timing=want_timing, cache=cache,
                                   previous=previous, on_progress=on_progress)


def _evaluate_at_cached(cell: Cell, bundle: Bundle, model, spec: Backbone, settings: RunSettings,
                        *, k: int, signature: str, want_probe: bool, want_timing: bool,
                        cache: Path, previous: dict[str, Any] | None = None,
                        on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    evaluate = _evaluate_module()
    gallery = queries = None
    try:
        gallery = evaluate.encode_split(
            model, bundle.train.features, None, k, device=settings.fit.device,
            batch_size=settings.encode_batch_size, cache_dir=cache,
            identity=_encode_identity(cell, bundle, "train", k, signature),
            progress_stage=f"encode {cell.slug} K={k} gallery")
        queries = evaluate.encode_split(
            model, bundle.validation.features, None, k, device=settings.fit.device,
            batch_size=settings.encode_batch_size, cache_dir=cache,
            identity=_encode_identity(cell, bundle, "validation", k, signature),
            progress_stage=f"encode {cell.slug} K={k} queries")
        accuracy_settings = replace(settings, linear_probe=False, measure_timing=False)
        reusable = (previous and previous.get("signature") == signature
                    and previous.get("k") == k and "nnz_mean" in previous
                    and _cached_e1_complete(previous, accuracy_settings))
        if reusable:
            point = dict(previous)
        else:
            scored = evaluate.top1_accuracy(
                gallery, bundle.train.labels, queries, bundle.validation.labels,
                dim=spec.preset.hidden_dim, device=settings.fit.device,
                gallery_chunk=settings.gallery_chunk, query_block=settings.query_block)
            point = {
                "k": k, "top1": scored["top1"], "queries": scored["queries"],
                "empty_queries": scored["empty_queries"], "seconds": scored["seconds"],
                "backend": scored.get("backend"),
                "mean_neighbour_score": scored.get("mean_neighbour_score"),
                "nnz_mean": gallery["stats"]["nnz_mean"],
                "nnz_min": gallery["stats"]["nnz_min"],
                "empty_gallery_codes": gallery["stats"]["empty_codes"],
                "encode_seconds": gallery["stats"]["seconds"] + queries["stats"]["seconds"],
            }
        point["signature"] = signature
        timing_needed = (want_timing and settings.measure_timing
                         and not _cached_e1_complete(point, replace(settings, linear_probe=False)))
        point.setdefault("timing_status", "not_requested")
        if timing_needed:
            point["timing_status"] = "pending"
            point["timing_config"] = settings.timing_identity()
            for stale in ("timing", "timing_error", "retrieval_ms", "index_bytes"):
                point.pop(stale, None)
        if on_progress is not None:
            on_progress(dict(point))
        probe_ready = _cached_e1_complete(point, replace(settings, measure_timing=False))
        if want_probe and settings.linear_probe and not probe_ready:
            probe = evaluate.linear_probe(
                gallery, bundle.train.labels, queries, bundle.validation.labels,
                hidden_dim=spec.preset.hidden_dim, device=settings.fit.device,
                epochs=settings.probe_epochs, seed=cell.seed)
            probe_settings = dict(probe.get("settings") or {})
            probe_settings["epochs"] = int(probe.get("epochs", settings.probe_epochs))
            point.update(probe_top1=probe["top1"], probe_top5=probe.get("top5"),
                         probe_epochs=probe_settings["epochs"], probe_settings=probe_settings)
            if on_progress is not None:
                on_progress(dict(point))
        if timing_needed:
            # Retrieval cost is a secondary metric: it reports the trade-off CSR
            # argues for, but no accuracy depends on it. It refuses to run outside
            # CUDA, and a wall-clock measurement can fail for reasons that have
            # nothing to do with the fit, so it must never take the run down.
            point["timing_status"] = "attempted"
            if on_progress is not None:
                on_progress(dict(point))
            try:
                timing = evaluate.time_sparse_matmul(
                    gallery, queries, dim=spec.preset.hidden_dim, device=settings.fit.device,
                    query_batch=settings.timing_query_batch, warmup=settings.timing_warmup,
                    rounds=settings.timing_rounds, gallery_chunk=settings.timing_gallery_chunk,
                    build_entry_cap=settings.timing_build_entry_cap)
            except Exception as error:
                point.update(timing_status="skipped", timing_error=f"{type(error).__name__}: {error}")
                get_logger(cell.slug).warning("retrieval timing skipped (%s: %s)",
                                              type(error).__name__, error)
            else:
                point.update(retrieval_ms=_milliseconds(timing),
                             index_bytes=_index_bytes(timing), timing=timing, timing_status="success")
            if on_progress is not None:
                on_progress(dict(point))
        return point
    except BaseException as error:
        if not settings.retain_codes:
            _close_failed_cache_maps(error, cache)
        raise
    finally:
        if not settings.retain_codes:
            _close_code_arrays(gallery)
            _close_code_arrays(queries)
            gallery = queries = None


def _milliseconds(timing: dict[str, Any]) -> float | None:
    """Per-batch latency, wherever the timing harness recorded it."""
    for key in ("milliseconds_per_batch", "median_ms"):
        if isinstance(timing.get(key), (int, float)):
            return float(timing[key])
    median = (timing.get("cuda_seconds") or {}).get("median")
    return float(median) * 1000.0 if isinstance(median, (int, float)) else None


def _index_bytes(timing: dict[str, Any]) -> int | None:
    for source in (timing, timing.get("index") or {}):
        value = source.get("index_bytes") if isinstance(source, dict) else None
        if isinstance(value, (int, float)):
            return int(value)
    return None


# ------------------------------------------------------------------ one cell

def _cached_e1_complete(record: dict[str, Any] | None, settings: RunSettings) -> bool:
    """Require requested components, allowing recorded optional timing skips.

    Legacy probe results omitted their epoch count. Refresh those once when a
    probe is requested, rather than guessing which settings produced them.
    """
    if (not record or isinstance(record.get("top1"), bool)
            or not isinstance(record.get("top1"), (int, float))):
        return False
    if not math.isfinite(record["top1"]):
        return False
    if settings.linear_probe:
        score = record.get("probe_top1")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            return False
        probe_settings = record.get("probe_settings") or {}
        if not isinstance(probe_settings, dict):
            return False
        epochs = record.get("probe_epochs", probe_settings.get("epochs"))
        if "epochs" in probe_settings and probe_settings["epochs"] != epochs:
            return False
        if not (isinstance(epochs, int) and not isinstance(epochs, bool) and epochs == settings.probe_epochs):
            return False
    if settings.measure_timing:
        if record.get("timing_config") != settings.timing_identity():
            return False
        if record.get("timing_status") == "success":
            return True
        if record.get("timing_status") == "skipped":
            return not settings.timing_retry_skipped
        return False
    return True


def run_cell(cell: Cell, bundle: Bundle, settings: RunSettings, *, code_version: str
             ) -> dict[str, Any]:
    """Fit one cell, then run E1 and -- at an anchor budget -- E2."""
    spec = BACKBONES[cell.backbone]
    out_dir = cell_dir(settings, cell)
    log = get_logger(cell.slug)
    summary = fit(cell, bundle, out_dir, settings.fit, code_version=code_version, spec=spec)
    signature = summary["signature"]

    e1_path = out_dir / "eval_e1.json"
    e1 = read_json(e1_path)
    if not (_cached_e1_complete(e1, settings) and e1.get("signature") == signature and not settings.force):
        model = load_fitted(cell, out_dir, spec, device=settings.fit.device)
        metadata = {"experiment": "E1", "signature": signature, "backbone": cell.backbone,
                    "method": cell.method, "k": cell.k, "seed": cell.seed,
                    "train_seconds": summary["train_seconds"], "gpu_hours": summary["gpu_hours"]}

        def persist_point(point):
            atomic_json(e1_path, {**point, **metadata})

        point = evaluate_at(cell, bundle, model, spec, settings, k=cell.k,
                            signature=signature, want_probe=True, want_timing=True,
                            previous=e1 if not settings.force else None, on_progress=persist_point)
        e1 = {**point, **metadata}
        atomic_json(e1_path, e1)
        log.info("E1 K=%d: top1 %.2f%% | nnz %.1f | probe %s", cell.k, e1["top1"],
                 e1["nnz_mean"], f"{e1['probe_top1']:.2f}%" if e1.get("probe_top1") else "-")
    else:
        log.info("E1 K=%d: top1 %.2f%% (cached)", cell.k, e1["top1"])

    result = {"cell": cell.__dict__, "fit": summary, "e1": e1}
    if settings.run_e2 and cell.k in spec.e2_anchors:
        result["e2"] = run_e2(cell, bundle, settings, signature=signature, e1=e1)
    return result


def run_e2(cell: Cell, bundle: Bundle, settings: RunSettings, *, signature: str,
           e1: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one anchor at the selected inference budgets, reusing cached points."""
    spec = BACKBONES[cell.backbone]
    budgets = spec.k_list if settings.e2_k_values is None else settings.e2_k_values
    if (not budgets or any(isinstance(k, bool) or not isinstance(k, int)
                           or not 1 <= k <= spec.dim for k in budgets)
            or len(set(budgets)) != len(budgets)):
        raise ValueError("E2 budgets must be distinct positive integers within the backbone width")
    out_dir = cell_dir(settings, cell)
    path = out_dir / "eval_e2.json"
    log = get_logger(f"E2 {cell.slug}")
    previous = read_json(path)
    # Keep unselected historical measurements available for a later wider run,
    # but expose only the requested points to reports and plots.
    done = {point["k"]: point for point in ((previous or {}).get("cached_points", [])
                                          + (previous or {}).get("points", []))} \
        if previous and previous.get("signature") == signature and not settings.force else {}
    # E1 has already refreshed requested metrics at the anchor. Copy all metric
    # provenance too, so a legacy E2 point cannot hide a newly completed probe.
    if cell.k in budgets:
        done[cell.k] = {key: value for key, value in e1.items() if key not in
                       ("experiment", "backbone", "method", "seed", "train_seconds", "gpu_hours")}

    def snapshot():
        record = {"experiment": "E2", "signature": signature, "backbone": cell.backbone,
                  "method": cell.method, "anchor_k": cell.k, "seed": cell.seed,
                  "eval_k": list(budgets),
                  "points": [done[k] for k in sorted(budgets) if k in done]}
        unused = [done[k] for k in sorted(done) if k not in budgets]
        if unused:
            record["cached_points"] = unused
        return record

    missing = [k for k in budgets if not _cached_e1_complete(done.get(k), settings)]
    if missing:
        model = load_fitted(cell, out_dir, spec, device=settings.fit.device)
        progress = Progress(len(missing), stage=f"E2 {cell.slug}", what="budgets",
                            every_seconds=0.0)
        for k in missing:
            def persist_point(point):
                done[k] = point
                atomic_json(path, snapshot())

            # A legacy point inherits the enclosing E2 fit signature. Reuse its
            # accuracy while filling missing probe/timing measurements.
            cached = {**done[k], "signature": signature} if k in done else None
            done[k] = evaluate_at(cell, bundle, model, spec, settings, k=k,
                                  signature=signature, want_probe=True, want_timing=True,
                                  previous=cached, on_progress=persist_point)
            progress.advance(1, extra=f"K={k} top1 {done[k]['top1']:.2f}% nnz {done[k]['nnz_mean']:.1f}")
            atomic_json(path, snapshot())
    record = snapshot()
    atomic_json(path, record)
    log.info("anchor K=%d evaluated at %s", cell.k, ", ".join(str(k) for k in sorted(budgets)))
    return record


# ------------------------------------------------------------------ the grid

def pending(cells: Iterable[Cell], settings: RunSettings) -> list[Cell]:
    """Cells without a completed E1, in grid order."""
    remaining = []
    for cell in cells:
        record = read_json(cell_dir(settings, cell) / "eval_e1.json")
        if settings.force or not _cached_e1_complete(record, settings):
            remaining.append(cell)
    return remaining


def run_grid(cells: Iterable[Cell], settings: RunSettings, *, code_version: str,
             dense_gate: bool = True) -> dict[str, Any]:
    """Work the grid backbone by backbone so each bundle is loaded once."""
    cells = list(cells)
    log = get_logger("grid")
    by_backbone: dict[str, list[Cell]] = {}
    for cell in cells:
        by_backbone.setdefault(cell.backbone, []).append(cell)

    outstanding = pending(cells, settings)
    banner(f"grid: {len(cells)} cells, {len(outstanding)} outstanding", stage="grid")
    overall = Progress(len(cells), stage="grid", what="cells",
                       done=len(cells) - len(outstanding), every_seconds=0.0)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for name, group in by_backbone.items():
        spec = BACKBONES[name]
        bundle = load_bundle(spec, settings.embeddings_dir, settings.cache_dir)
        if dense_gate:
            dense_reference(spec, bundle, settings)
        for cell in group:
            try:
                results.append(run_cell(cell, bundle, settings, code_version=code_version))
            except Exception as error:
                log.error("%s failed: %s: %s", cell.slug, type(error).__name__, error)
                atomic_json(cell_dir(settings, cell) / "failure.json",
                            {"cell": cell.__dict__, "error": f"{type(error).__name__}: {error}",
                             "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
                raise
            overall.advance(1, extra=cell.slug)

    elapsed = time.perf_counter() - started
    manifest = {"cells": len(cells), "completed": len(results),
                "seconds": elapsed, "code_version": code_version}
    atomic_json(Path(settings.root) / "grid.json", manifest)
    log.info("grid finished: %d cells in %s", len(results), format_duration(elapsed))
    return manifest


def plan(backbones=None, methods=None, seeds=None, k_values=None) -> list[Cell]:
    """Build a cell list, defaulting to the full grid."""
    from .config import BACKBONE_ORDER, METHODS, SEEDS

    return grid(backbones or BACKBONE_ORDER, methods or METHODS, seeds or SEEDS, k_values)
