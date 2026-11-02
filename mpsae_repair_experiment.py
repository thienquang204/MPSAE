"""Explicit, resumable stages for the FF2048 repair research experiment.

The default command prints a plan. No stage implicitly advances into a larger
training budget. Strict paper runners and their output schemas are independent.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

from make_plan import DEFAULT_CONFIG, build_plan, load_config
from repair_data import atomic_json, canonical_hash, exclusive_lock, sha256_file


STAGES = ("plan", "preflight", "prepare", "check", "warm", "triage", "attribution", "wide", "arm",
          "lock", "holdout", "confirm", "timing")
SOURCE_FILES = ("mpsae_repair_experiment.py", "mpsae_repair.py", "repair_runtime.py",
                "repair_data.py", "repair_benchmark.py", "prepare_splits.py",
                "compare_predictions.py", "make_plan.py", "check_reference.py", "csr_vs_mmpot_imagenet.py",
                "sparse_gpu_retrieval.py", "retrieval_timing.py",
                "csrc/sparse_retrieval.cpp", "csrc/sparse_retrieval.cu")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_identity():
    root = Path(__file__).resolve().parent
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def json_value(value):
    """Detach logging values without retaining an autograd graph."""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_value(value), allow_nan=False) + "\n")
        stream.flush()


def require_unchanged(path, value):
    """Refuse accidental reuse of artifacts after a recipe/source change."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing identity differs: {path}. Select a new output directory.")
    else:
        atomic_json(path, value)


def validate_implemented_settings(config):
    # These objective constants are deliberate controls, not ignored knobs.
    reconstruction = config["reconstruction"]
    if [reconstruction[key] for key in ("main_weight", "wide_weight", "dead_weight")] != [1., .125, .03125]:
        raise ValueError("This profile implements reconstruction weights 1,1/8,1/32")
    if config["repair"]["alpha_factor"] != .9 or config["repair"]["anneal_fraction"] != .7:
        raise ValueError("This profile fixes alpha factor .9 and auxiliary annealing fraction .7")
    if config["bank"]["cache_teacher_profiles"]:
        raise ValueError("Teacher profiles are calculated on demand; this runner does not implement a profile-file cache")
    if not config["confirmation"]["retain_last_batch"]:
        raise ValueError("Full confirmation must retain its final training batch")
    if config["confirmation"]["methods"] != ["C", "S", "selected_candidate"]:
        raise ValueError("Confirmation uses the unique set C,S,selected_candidate")
    source = config["source"]
    if (source["train_file"] != "FF2048_RN50_Embeds/1K_train_ff2048.npz" or
        source["validation_file"] != "FF2048_RN50_Embeds/1K_val_ff2048.npz" or source["validation_rows"] != 50000):
        raise ValueError("This adapter uses the explicitly named FF2048 training/validation archives and 50,000 validation rows")
    batches = math.ceil(config["pilot"]["fit_per_class"] * config["source"]["classes"] /
                        config["pilot"]["batch_size"])
    if config["pilot"]["warm_updates"] != 2 * batches or config["pilot"]["branch_updates"] != 2 * batches:
        raise ValueError("The matched pilot requires two warm epochs and two continuation epochs")


def screening_decision(results, config):
    """Pure decision logic; missing or invalid evidence never becomes a pass."""
    baseline = results["C"]
    threshold = config["gates"]["minimum_gain_percentage_points"]
    maximum_ratio = config["gates"]["maximum_encoding_ratio"]
    decisions = {}
    for arm in ("S", "N"):
        result = results.get(arm)
        if result is None:
            decisions[arm] = {"promising": False, "status": "failed or unavailable; see arm failure record"}
            continue
        ratio = result.get("encoding_ratio_to_C")
        diagnostics_ok = result.get("diagnostics_acceptable", False)
        cost_ok = ratio is not None and math.isfinite(ratio) and (maximum_ratio is None or ratio <= maximum_ratio)
        delta = result["top1"] - baseline["top1"]
        decisions[arm] = {"delta_percentage_points": delta, "encoding_ratio_to_C": ratio,
                          "diagnostics_acceptable": diagnostics_ok, "encoding_acceptable": cost_ok,
                          "promising": delta >= threshold and diagnostics_ok and cost_ok}
    return {"minimum_gain_percentage_points": threshold,
            "maximum_encoding_ratio": maximum_ratio, "arms": decisions,
            "promising": any(value["promising"] for value in decisions.values()),
            "interpretation": "cheap selection screen; not confirmatory evidence"}


def gate_decision(comparison, diagnostics_ok, encoding_ok, protected_ok, config):
    gates = config["gates"]
    gain_ok = comparison["delta_percentage_points"] >= gates["minimum_gain_percentage_points"]
    interval_ok = (not gates["holdout_requires_positive_lower_ci"] or
                   comparison["ci_lower_percentage_points"] > 0)
    return {"gain_passed": gain_ok, "interval_passed": interval_ok,
            "diagnostics_passed": bool(diagnostics_ok), "encoding_passed": bool(encoding_ok),
            "protected_passed": bool(protected_ok),
            "promoted": bool(gain_ok and interval_ok and diagnostics_ok and encoding_ok and protected_ok)}


class ResearchRun:
    def __init__(self, args, config):
        import numpy as np
        import torch
        from repair_data import load_splits, open_feature_cache

        self.args, self.config = args, config
        validate_implemented_settings(config)
        if args.train_npz is None or args.splits_dir is None:
            raise ValueError("This stage requires --train-npz and --splits-dir; prepare the training-only splits first")
        self.output = args.output_dir
        self.output.mkdir(parents=True, exist_ok=True)
        self.splits, self.manifest = load_splits(args.splits_dir)
        pinned = config["source"].get("train_sha256")
        revision = config["source"].get("revision")
        if not isinstance(pinned, str) or len(pinned) != 64 or any(ch not in "0123456789abcdef" for ch in pinned):
            raise ValueError("Pin source.train_sha256 from trusted released-archive provenance in the configuration before real-data checks or training")
        if not isinstance(revision, str) or len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError("Pin the immutable source.repository revision (40-character commit SHA) alongside the archive digest")
        if self.manifest["source"]["sha256"] != pinned:
            raise ValueError("Split archive differs from the explicitly pinned release digest")
        if args.stage != "confirm" and (args.val_npz is not None or args.val_sha256 is not None):
            raise ValueError("--val-npz/--val-sha256 are reserved for full confirmation; pilot stages never open it")
        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for research training/retrieval; CPU is available only for --stage check")
            torch.cuda.set_device(self.device)
            self.device = torch.device("cuda", torch.cuda.current_device())
        elif args.stage != "check":
            raise ValueError("CPU execution is restricted to mathematical/training-slice checks, never performance measurements")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.stage != "check":
            # Every execution stage can train or evaluate; check the GPU
            # backends before hashing/unpacking the multi-gigabyte archive.
            from repair_benchmark import preflight
            preflight(self.device, config)
        self.features, self.labels, self.source = open_feature_cache(
            args.train_npz, args.cache_dir, expected_sha256=pinned)
        expected = config["source"]
        if (self.source["feature_shape"] != [1281167, 2048] or self.source["classes"] != 1000 or
            expected["expected_rows"] != 1281167 or expected["dimension"] != 2048 or
            expected["classes"] != 1000 or config["model"]["hidden_dim"] != 8192 or
            expected["repository"] != "W1nd-navigator/CSR-precompute-embeds" or
            expected["parent"] != "RAIVNLab/MRL:r50_mrl0_e0_ff2048.pt" or
            args.train_npz.name != "1K_train_ff2048.npz"):
            raise ValueError("Training requires the declared released FF2048 source, 1,281,167 x 2048 / 1000 classes / width8192; synthetic split creation is not a provenance check")
        expected_counts = {"fit_indices": config["pilot"]["fit_per_class"] * 1000,
                           "dev_cheap_indices": config["pilot"]["cheap_per_class"] * 1000,
                           "dev_holdout_indices": config["pilot"]["holdout_per_class"] * 1000,
                           "anchor_indices": config["pilot"]["anchor_count"]}
        for name, count in expected_counts.items():
            if len(self.splits[name]) != count:
                raise ValueError(f"Split/config count mismatch: {name}")
        self.identity = {"protocol": config["protocol"], "configuration_sha256": canonical_hash(config),
                         "source_sha256": self.source["sha256"],
                         "split_sha256": self.manifest["indices_sha256"],
                         "parent": expected["parent"], "code": source_identity()}
        require_unchanged(self.output / "identity.json", self.identity)
        require_unchanged(self.output / "resolved_configuration.json", config)
        atomic_json(self.output / "runtime.json", {"python": sys.version, "torch": str(torch.__version__),
                    "numpy": np.__version__, "device": str(self.device), "platform": platform.platform(),
                    "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU checks only",
                    "cuda": torch.version.cuda, "tf32": False, "amp": False})
        self.decisions_path = self.output / "decisions.json"
        self.decisions = read_json(self.decisions_path) if self.decisions_path.exists() else {
            "protocol": config["protocol"],
            "stages": {stage: {"status": "not_requested"} for stage in STAGES if stage != "plan"},
            "arms": {arm: {"status": "not_requested"} for arm in config["arms"]},
            "local_MRL": {"status": "unavailable", "reason": "No verified aligned local MRL feature cache/checkpoint mapping supplied"},
            "transfer": {"status": "deferred", "reason": "Requires positive confirmation and a separately verified exact FF2048 region parent or budgeted timm refit; no automatic large transfer grid"}}
        self.protection_ids = self.splits["fit_indices"][:32]
        self.protected = np.array(self.features[self.protection_ids], copy=True)
        self.protected_hash = hashlib.sha256(self.protected.tobytes()).hexdigest()

    def save_decisions(self):
        atomic_json(self.decisions_path, self.decisions)

    def tensor(self, ids):
        import numpy as np
        import torch
        values = np.array(self.features[ids], dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite source features")
        return torch.from_numpy(values).to(self.device)

    def protection(self):
        import numpy as np
        after = np.asarray(self.features[self.protection_ids])
        current = hashlib.sha256(after.tobytes()).hexdigest()
        difference = float(np.max(np.abs(after - self.protected)))
        if current != self.protected_hash or difference != 0:
            raise RuntimeError("Protected frozen parent output changed")
        return {"passed": True, "source_sha256": self.source["sha256"],
                "checked_row_ids": self.protection_ids.tolist(), "checksum": current,
                "max_absolute_difference": difference,
                "scope": "read-only released full-feature route; no claim about an unverified image encoder or a trainable largest sparse code"}

    def make_model(self, seed, warm=None, repaired=False, alpha_initialization=None, warm_sha256=None):
        import torch
        from mpsae_repair import ResearchSAE, RepairModule, estimate_alpha, backtrack_alpha
        cfg = self.config
        # Model initialization and repair use isolated streams, independent of
        # data permutations, diagnostic sampling, and masks.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            sae = ResearchSAE(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["reconstruction"]["dead_steps"])
        sae = sae.to(self.device)
        if warm is not None:
            sae.load_state_dict(warm["model"]["sae"], strict=True)
        if not repaired:
            return sae, None, {"alpha": None, "source": "repair disabled; no spectral estimate computed"}
        init_identity = {"experiment": self.identity, "seed": seed,
                         "warm_identity": None if warm is None else warm["identity"],
                         "warm_position": None if warm is None else warm["position"],
                         "warm_checkpoint_sha256": warm_sha256}
        if warm is not None and warm_sha256 is None:
            raise ValueError("Shared repair alpha requires the actual warm checkpoint fingerprint")
        alpha_path = self.output / "initialization" / (canonical_hash(init_identity) + ".json")
        if alpha_initialization is not None:
            alpha_stats = alpha_initialization
        elif alpha_path.exists():
            stored = read_json(alpha_path)
            if stored["identity"] != init_identity:
                raise ValueError("Shared repair initialization identity differs")
            alpha_stats = stored["alpha_initialization"]
        else:
            alpha = estimate_alpha(sae.decoder, iterations=cfg["repair"]["power_iterations"], seed=seed + 17)
            diagnostic = self.tensor(self.splits["fit_indices"][:32])
            alpha, descent = backtrack_alpha(sae, diagnostic, cfg["model"]["k"], alpha)
            alpha_stats = {"alpha": alpha, **descent, "source": "warm dictionary" if warm else "fresh matched dictionary",
                           "power_iterations": cfg["repair"]["power_iterations"],
                           "interpretation": "fixed scale estimate, measured initial-IHT descent only"}
            atomic_json(alpha_path, {"identity": init_identity, "alpha_initialization": alpha_stats})
        repair = RepairModule(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"],
                              cfg["repair"]["rank"], alpha=alpha_stats["alpha"], seed=seed + 29).to(self.device)
        return sae, repair, alpha_stats

    def loss_arguments(self, arm, step, total, bank, masks, diagnostics=False):
        from mpsae_repair import auxiliary_budget
        cfg, spec = self.config, self.config["arms"][arm]
        transport, banks = cfg["transport"], cfg["bank"]
        return dict(mask=spec["mask"], mask_generator=masks,
                    auxiliary_budget=auxiliary_budget(cfg["model"]["k"], cfg["model"]["hidden_dim"], step, total, spec["auxiliary"]),
                    bank=bank, relation_weight=spec["rel_weight"],
                    ot_weight=transport["weight"] if spec["ot_cost"] != "off" else 0.,
                    mixture_weight=0.,
                    teacher_temperature=banks["teacher_temperature"], student_temperature=banks["student_temperature"],
                    anchor_chunk_size=banks["anchor_chunk_size"],
                    mass=transport["mass"], eta=transport["entropy"], max_iters=transport["max_iterations"],
                    tol=transport["tolerance"], microbatch=transport["group_size"], require_feasible=True,
                    max_refine_iters=transport.get("max_refine_iterations", 2000), diagnostics=diagnostics,
                    k_aux=cfg["reconstruction"]["aux_k"], clean_probability=cfg["repair"]["clean_probability"],
                    erase_probability=cfg["repair"]["erase_probability"])

    def bank(self, arm):
        from mpsae_repair import LandmarkBank
        spec = self.config["arms"][arm]
        if not spec["rel_weight"]:
            return None
        ids = self.splits["anchor_indices"]
        return LandmarkBank(ids, self.tensor(ids))

    def diagnostic(self, arm, sae, repair, bank, step, total, directory, seed):
        import torch
        from mpsae_repair import research_loss
        path = directory / f"diagnostics_{step:06d}.json"
        if path.exists():
            return read_json(path)
        ids = self.splits["fit_indices"][:min(64, self.config["pilot"]["batch_size"])]
        generator = torch.Generator(device="cpu").manual_seed(seed + 43)
        counter = sae.inactive_steps.clone()
        _, stats, views = research_loss(sae, self.tensor(ids), self.config["model"]["k"], repair=repair,
            row_ids=ids, update_activity=False,
            **self.loss_arguments(arm, step, total, bank, generator, diagnostics=True))
        if not torch.equal(counter, sae.inactive_steps):
            raise RuntimeError("Diagnostic forward mutated activity counters")
        result = json_value(stats)
        result["branch_step"] = step
        result["row_ids"] = ids.tolist()
        result["protected_output"] = self.protection()
        if "transported_marginals" in views:
            result["transported_marginals"] = json_value(views["transported_marginals"])
        if bank is not None:
            result["keys"] = json_value(bank.last_refresh_diagnostics)
            result["key_refresh_count"] = bank.refresh_count
            result["key_refresh_total_seconds"] = bank.refresh_seconds
        # Thresholds are declared diagnostics, not a theorem or a substitute for accuracy.
        problems = []
        if self.config["arms"][arm]["rel_weight"]:
            if result.get("zero_overlap_fraction", 1.) >= .99:
                problems.append("at least 99% of diagnostic queries have no landmark support overlap")
            if result.get("gradient_relation_selected_near_zero_fraction", 1.) >= .99:
                problems.append("at least 99% of selected relation-gradient entries are near zero")
        result["diagnostics_acceptable"] = not problems
        result["investigate"] = problems
        atomic_json(path, result)
        return result

    def check(self):
        import unittest
        import check_reference
        suite = unittest.defaultTestLoader.loadTestsFromModule(check_reference)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        if not outcome.wasSuccessful():
            raise RuntimeError("Mathematical reference checks failed; no fitting was started")
        directory = self.output / "checks"
        directory.mkdir(exist_ok=True)
        seed = self.config["pilot"]["seed"]
        real_equivalence = check_reference.check_baseline_on_features(
            self.tensor(self.splits["fit_indices"][:8]), k=self.config["model"]["k"],
            hidden_dim=self.config["model"]["hidden_dim"], seed=seed)
        signal_reports = {}
        for arm in ("C", "S", "N"):
            sae, repair, _ = self.make_model(seed, repaired=self.config["arms"][arm]["repair"])
            bank = self.bank(arm)
            if bank is not None:
                bank.refresh(sae, repair, self.config["model"]["k"], step=0)
            target = directory / arm
            target.mkdir(exist_ok=True)
            signal_reports[arm] = self.diagnostic(arm, sae, repair, bank, 0, self.config["pilot"]["branch_updates"], target, seed)
        report = {"identity": self.identity, "completed_at": utc_now(), "math_tests": outcome.testsRun,
                  "protected_output": self.protection(), "status": "passed",
                  "signal_diagnostics": signal_reports,
                  "signal_passed": all(row["diagnostics_acceptable"] for row in signal_reports.values()),
                  "actual_feature_baseline_equivalence": real_equivalence,
                  "signal_policy": "fresh-state signal is recorded; pilot relational continuations are screened again on their actual restored W state",
                  "scope": "CPU/GPU tiny training-slice correctness; no ImageNet accuracy or speed result"}
        atomic_json(self.output / "checks.json", report)
        return report

    def require_checks(self):
        path = self.output / "checks.json"
        if not path.exists() or read_json(path).get("identity") != self.identity or read_json(path).get("status") != "passed":
            raise ValueError("Run the explicit check stage on this source/configuration before spending training compute")

    def require_unlocked(self):
        if "lock" in self.decisions:
            raise ValueError("Candidate/settings are locked. Pilot training and cheap-set selection are closed in this output directory.")

    def require_promising(self, family=None):
        screen = self.decisions.get("triage", {})
        passed = screen.get("promising", False) if family is None else screen.get("arms", {}).get(family, {}).get("promising", False)
        if not passed:
            raise ValueError("The declared cheap-screen gate has not passed; later fits are not launched")

    def stage_record(self, stage, status, **details):
        self.decisions["stages"][stage] = {"status": status, "time": utc_now(), **details}
        self.save_decisions()

    def fit(self, arm, *, confirmation=False, seed=None):
        import numpy as np
        import torch
        from mpsae_repair import research_loss
        from repair_runtime import (epoch_batches, load_checkpoint, named_parameters, pack_adam,
                                    restore_adam, restore_rng, rng_state, save_checkpoint)

        self.require_checks()
        cfg, spec = self.config, self.config["arms"][arm]
        seed = cfg["pilot"]["seed"] if seed is None else int(seed)
        directory = self.output / (f"full_confirmation/seed_{seed}/{arm}" if confirmation else arm)
        directory.mkdir(parents=True, exist_ok=True)
        rows = np.arange(len(self.features), dtype=np.int64) if confirmation else self.splits["fit_indices"]
        batch_size = cfg["confirmation" if confirmation else "pilot"]["batch_size"]
        batches = math.ceil(len(rows) / batch_size)
        total = cfg["confirmation"]["epochs"] * batches if confirmation else cfg["pilot"]["warm_updates" if arm == "W" else "branch_updates"]
        start_epoch = 0 if confirmation or arm == "W" else 2
        warm = None
        warm_hash = None
        if not confirmation and arm != "W":
            warm_path = self.output / "W" / "last.pt"
            if not warm_path.exists():
                raise ValueError("The common completed W checkpoint is required")
            warm = load_checkpoint(warm_path)
            warm_report = read_json(self.output / "W" / "fit.json")
            warm_hash = sha256_file(warm_path)
            if (warm["position"]["next_step"] != cfg["pilot"]["warm_updates"] or
                warm["identity"]["experiment"] != self.identity or warm["identity"]["arm"] != "W" or
                warm["identity"]["seed"] != seed or warm["identity"]["confirmation"] or
                warm["identity"]["start_epoch"] != 0 or warm["model"]["repair"] is not None or
                not warm_report["complete"] or warm_report["checkpoint_sha256"] != warm_hash):
                raise ValueError("Warm checkpoint is incomplete or belongs to another experiment")
        arm_identity = {"experiment": self.identity, "arm": arm, "seed": seed, "confirmation": confirmation,
                        "warm_checkpoint_sha256": warm_hash, "total_updates": total, "batch_size": batch_size,
                        "start_epoch": start_epoch, "fitting_ids_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                        "torch": str(torch.__version__), "device": str(self.device),
                        "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"}
        require_unchanged(directory / "config.json", arm_identity)
        checkpoint_path = directory / "last.pt"
        if (directory / "fit.json").exists():
            completed = read_json(directory / "fit.json")
            if completed.get("complete") and completed.get("identity") == arm_identity:
                if not checkpoint_path.exists() or sha256_file(checkpoint_path) != completed["checkpoint_sha256"]:
                    raise ValueError("Completed endpoint checkpoint is missing or changed")
                key = f"confirmation/seed_{seed}/{arm}" if confirmation else arm
                self.decisions["arms"][key] = {"status": "complete", "updates": total,
                                              "checkpoint_sha256": completed["checkpoint_sha256"]}
                self.save_decisions()
                return None, None, directory, completed
        previous = load_checkpoint(checkpoint_path, expected_identity=arm_identity) if checkpoint_path.exists() else None
        sae, repair, alpha_stats = self.make_model(seed, warm=warm, repaired=spec["repair"],
            alpha_initialization=previous["alpha_initialization"] if previous is not None else None,
            warm_sha256=warm_hash)
        named = named_parameters(sae, repair)
        optim = cfg["optimizer"]
        optimizer = torch.optim.Adam(list(named.values()), lr=optim["lr"], betas=tuple(optim["betas"]),
                                     eps=optim["eps"], weight_decay=optim["weight_decay"])
        masks = torch.Generator(device="cpu").manual_seed(seed + 101)
        bank = self.bank(arm)
        next_step, elapsed = 0, 0.
        if previous is not None:
            sae.load_state_dict(previous["model"]["sae"], strict=True)
            if repair is not None:
                repair.load_state_dict(previous["model"]["repair"], strict=True)
            restore_adam(optimizer, named, previous["optimizer"])
            restore_rng(previous["rng"], mask_generator=masks)
            if bank is not None:
                bank.load_state_dict(previous["bank"])
            next_step = int(previous["position"]["next_step"])
            elapsed = float(previous["elapsed_seconds"])
            alpha_stats = previous["alpha_initialization"]
        elif warm is not None:
            # The SAE keeps its Adam step and both moments; only new factors
            # receive zero moments. Mask streams start identically across arms.
            restore_adam(optimizer, named, warm["optimizer"], allow_new=True)
            restore_rng(warm["rng"], mask_generator=masks)
            masks.manual_seed(seed + 101)
        if next_step > total:
            raise ValueError("Checkpoint extends beyond the declared endpoint")
        if bank is not None and bank.codes is None:
            bank.refresh(sae, repair, cfg["model"]["k"], batch_size=cfg["retrieval"]["encode_batch_size"], step=next_step)
        log_path = directory / "training.jsonl"
        if log_path.exists():
            # A crash can leave durable logs after the last durable optimizer
            # state. Preserve those separately, then replay that same interval.
            lines = log_path.read_text(encoding="utf-8").splitlines()
            committed, abandoned = [], []
            for line in lines:
                try:
                    row = json.loads(line)
                    bucket = committed if row["branch_step"] < next_step else abandoned
                except (ValueError, KeyError):
                    bucket = abandoned
                bucket.append(line)
            if abandoned:
                stamp = time.time_ns()
                (directory / f"uncommitted_training_{stamp}.jsonl").write_text("\n".join(abandoned) + "\n", encoding="utf-8")
                log_path.write_text("\n".join(committed) + ("\n" if committed else ""), encoding="utf-8")
        milestones = {0, total // 2, total} if confirmation else set(cfg["pilot"]["diagnostic_updates"])
        milestones.add(total)
        started = time.perf_counter()

        def save(position):
            duration = elapsed + time.perf_counter() - started
            payload = {"format": "mpsae-repair-checkpoint-v1", "identity": arm_identity,
                       "model": {"sae": sae.state_dict(), "repair": None if repair is None else repair.state_dict()},
                       "optimizer": pack_adam(optimizer, named), "rng": rng_state(mask_generator=masks),
                       "bank": None if bank is None else bank.state_dict(),
                       "position": {"next_step": position, "epoch": start_epoch + position // batches,
                                    "next_batch": position % batches,
                                    "order": "repair_runtime epoch_batches deterministic seed+epoch"},
                       "alpha_initialization": alpha_stats, "elapsed_seconds": duration}
            save_checkpoint(checkpoint_path, payload)

        try:
            if next_step in milestones:
                initial_diagnostic = self.diagnostic(arm, sae, repair, bank, next_step, total, directory, seed)
                if next_step == 0 and not confirmation and arm != "W" and spec["rel_weight"] and not initial_diagnostic["diagnostics_acceptable"]:
                    raise RuntimeError("Restored-W semantic diagnostics need investigation/calibration before this continuation; inspect diagnostics_000000.json")
            epoch = start_epoch + next_step // batches
            while next_step < total:
                offset = next_step % batches
                for batch_number, ids in epoch_batches(rows, seed, epoch, batch_size, offset=offset):
                    if next_step >= total:
                        break
                    x = self.tensor(ids)
                    optimizer.zero_grad(set_to_none=True)
                    loss, stats, views = research_loss(sae, x, cfg["model"]["k"], repair=repair,
                        row_ids=ids, **self.loss_arguments(arm, next_step, total, bank, masks))
                    loss.backward()
                    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in named.values()):
                        raise FloatingPointError("Nonfinite gradient; optimizer update rejected")
                    optimizer.step()
                    record = {"run_id": str(directory.relative_to(self.output)), "seed": seed,
                              "global_step": next_step + (0 if confirmation or arm == "W" else cfg["pilot"]["warm_updates"]),
                              "branch_step": next_step, "epoch": epoch, "batch_number": batch_number,
                              "K": cfg["model"]["k"], "optimizer_lr": optimizer.param_groups[0]["lr"],
                              "wall_seconds": elapsed + time.perf_counter() - started, **json_value(stats)}
                    append_jsonl(log_path, record)
                    del loss, stats, views, x
                    next_step += 1
                    if bank is not None and (next_step % cfg["bank"]["refresh_interval"] == 0 or next_step == total):
                        bank.refresh(sae, repair, cfg["model"]["k"], batch_size=cfg["retrieval"]["encode_batch_size"], step=next_step)
                    if next_step in milestones:
                        self.diagnostic(arm, sae, repair, bank, next_step, total, directory, seed)
                    if next_step % cfg["pilot"]["checkpoint_interval"] == 0 or next_step == total:
                        save(next_step)
                        print(f"{directory.name} seed={seed} updates={next_step}/{total}", flush=True)
                epoch += 1
            if not checkpoint_path.exists():
                save(next_step)
            endpoint = self.diagnostic(arm, sae, repair, bank, total, total, directory, seed)
            report = {"identity": arm_identity, "updates": next_step, "complete": True,
                      "checkpoint_sha256": sha256_file(checkpoint_path), "fit_seconds": elapsed + time.perf_counter() - started,
                      "alpha_initialization": alpha_stats, "diagnostics_acceptable": endpoint["diagnostics_acceptable"],
                      "protected_output": self.protection(),
                      "parameter_count": sum(parameter.numel() for parameter in named.values()),
                      "key_refresh_seconds": bank.refresh_seconds if bank else 0.,
                      "inherited_OT": not confirmation and arm == "S",
                      "artifact_policy": "last.pt atomically replaced every 25 updates; no redundant full-state archive per interval"}
            atomic_json(directory / "fit.json", report)
            key = f"confirmation/seed_{seed}/{arm}" if confirmation else arm
            self.decisions["arms"][key] = {"status": "complete", "updates": total,
                                          "checkpoint_sha256": report["checkpoint_sha256"]}
            self.save_decisions()
            return sae, repair, directory, report
        except BaseException as error:
            failure = {"arm": arm, "seed": seed, "next_step": next_step, "time": utc_now(),
                       "error": repr(error), "traceback": traceback.format_exc(),
                       "resume_from": str(checkpoint_path) if checkpoint_path.exists() else None}
            atomic_json(directory / "failure.json", failure)
            raise

    def load_fit(self, arm, *, confirmation=False, seed=None):
        from mpsae_repair import ResearchSAE, RepairModule
        from repair_runtime import load_checkpoint
        seed = self.config["pilot"]["seed"] if seed is None else seed
        directory = self.output / (f"full_confirmation/seed_{seed}/{arm}" if confirmation else arm)
        report = read_json(directory / "fit.json")
        path = directory / "last.pt"
        if not report["complete"] or report["identity"]["experiment"] != self.identity or sha256_file(path) != report["checkpoint_sha256"]:
            raise ValueError(f"Invalid or changed completed fit: {directory}")
        state = load_checkpoint(path, expected_identity=report["identity"])
        cfg = self.config
        sae = ResearchSAE(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["reconstruction"]["dead_steps"]).to(self.device)
        sae.load_state_dict(state["model"]["sae"])
        repair = None
        if self.config["arms"][arm]["repair"]:
            repair = RepairModule(cfg["model"]["input_dim"], cfg["model"]["hidden_dim"], cfg["repair"]["rank"]).to(self.device)
            repair.load_state_dict(state["model"]["repair"])
        sae.eval()
        if repair is not None:
            repair.eval()
        return sae, repair, directory, report

    def evaluate(self, arm, pool="dev_cheap", *, timed=False, confirmation=False, seed=None,
                 validation=None, dense=False):
        import numpy as np
        from repair_benchmark import benchmark
        seed = self.config["pilot"]["seed"] if seed is None else seed
        if dense:
            sae = repair = None
            directory = self.output / ("full_confirmation/full_parent" if confirmation else "full_parent")
            fit_report = {"checkpoint_sha256": self.source["sha256"], "diagnostics_acceptable": True,
                          "protected_output": self.protection()}
        else:
            sae, repair, directory, fit_report = self.load_fit(arm, confirmation=confirmation, seed=seed)
        if confirmation:
            if validation is None:
                raise ValueError("Final evaluation requires the verified official validation cache")
            val_features, val_labels, val_source = validation
            gallery = np.arange(len(self.features), dtype=np.int64)
            queries = np.arange(len(val_features), dtype=np.int64)
            query_features, query_labels = val_features, val_labels
        else:
            gallery, queries = self.splits["gallery_indices"], self.splits[pool + "_indices"]
            query_features = query_labels = None
            val_source = None
        identity = {"experiment": self.identity, "checkpoint_sha256": fit_report["checkpoint_sha256"],
                    "query_source": val_source["sha256"] if val_source else self.source["sha256"],
                    "arm": "full_parent" if dense else arm, "pool": pool,
                    "confirmation": confirmation, "seed": seed}
        destination = directory / (pool + ("_timing" if timed else ""))
        result = benchmark(self.features, self.labels, gallery, queries, sae, repair, self.config,
                           destination, seed, identity=identity, device=self.device,
                           query_features=query_features, query_labels=query_labels, timed=timed)
        result.update({"diagnostics_acceptable": fit_report["diagnostics_acceptable"],
                       "protected_output": fit_report["protected_output"], "seed": seed,
                       "arm": "full_parent" if dense else arm})
        atomic_json(destination / "result.json", result)
        return result

    def pilot_results(self, arms, *, timed=False):
        results = {arm: self.evaluate(arm, timed=timed) for arm in arms}
        if "C" not in results:
            results["C"] = self.evaluate("C", timed=timed)
        baseline = results["C"].get("encoding_seconds_per_row")
        for arm, result in results.items():
            duration = result.get("encoding_seconds_per_row")
            result["encoding_ratio_to_C"] = duration / baseline if duration is not None and baseline and baseline > 0 else None
        return results

    def comparison_csv(self, name, results):
        path = self.output / name
        fields = ["arm", "top1", "correct", "total", "encoding_ratio_to_C", "diagnostics_acceptable"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for arm, row in results.items():
                writer.writerow({**row, "arm": arm})

    def lock(self, candidate):
        if candidate not in self.config["arms"] or candidate in ("W", "C"):
            raise ValueError("Select a completed candidate arm other than W/C")
        self.require_promising()
        if "lock" in self.decisions:
            self.verify_lock(candidate)
            return self.decisions["lock"]
        arms = list(dict.fromkeys(["C", "S", candidate]))
        # Detailed warm search timing is reserved for this finalist set.
        results = self.pilot_results(arms, timed=True)
        chosen = results[candidate]
        ratio = chosen["encoding_ratio_to_C"]
        maximum = self.config["gates"]["maximum_encoding_ratio"]
        if (chosen["top1"] - results["C"]["top1"] < self.config["gates"]["minimum_gain_percentage_points"] or
            not chosen["diagnostics_acceptable"] or ratio is None or (maximum is not None and ratio > maximum)):
            raise ValueError("Candidate fails the declared cheap accuracy/diagnostic/encoding screen; held-out queries remain unopened")
        checkpoints = {arm: sha256_file(self.output / arm / "last.pt") for arm in arms}
        locked = {"candidate": candidate, "identity": self.identity, "checkpoints": checkpoints,
                  "locked_at": utc_now(), "cheap_results": results,
                  "gate_policy": copy.deepcopy(self.config["gates"]),
                  "calibration": {"student_temperature": self.config["bank"]["student_temperature"],
                                  "scope": "fixed resolved configuration; no automatic temperature sweep"}}
        locked["lock_sha256"] = canonical_hash(locked)
        self.decisions["lock"] = locked
        self.save_decisions()
        atomic_json(self.output / "candidate_lock.json", locked)
        return locked

    def verify_lock(self, candidate=None):
        locked = self.decisions.get("lock")
        if locked is None:
            raise ValueError("Select and lock a completed candidate before opening the development gate")
        if candidate is not None and candidate != locked["candidate"]:
            raise ValueError("The candidate cannot change after locking")
        content = {key: value for key, value in locked.items() if key != "lock_sha256"}
        if canonical_hash(content) != locked["lock_sha256"] or locked["identity"] != self.identity:
            raise ValueError("Locked configuration/code/source differs")
        for arm, fingerprint in locked["checkpoints"].items():
            if sha256_file(self.output / arm / "last.pt") != fingerprint:
                raise ValueError("A locked endpoint checkpoint changed")
        return locked

    def holdout(self):
        from compare_predictions import compare_predictions
        locked = self.verify_lock(self.args.candidate)
        candidate = locked["candidate"]
        previous = self.decisions.get("holdout")
        if previous is not None:
            if previous["lock_sha256"] != locked["lock_sha256"]:
                raise ValueError("Held-out queries were already opened for a different lock")
            if previous["status"] == "complete":
                return previous
        else:
            # Durable marker is committed BEFORE the first query encoding.
            # A crash may resume this exact lock, never a new candidate.
            self.decisions["holdout"] = {"status": "opened", "opened_at": utc_now(),
                                         "lock_sha256": locked["lock_sha256"]}
            self.save_decisions()
        arms = list(dict.fromkeys(["C", "S", candidate]))
        results = {arm: self.evaluate(arm, "dev_holdout") for arm in arms}
        gates = self.config["gates"]
        comparisons = {}
        for baseline in dict.fromkeys(["C", "S"]):
            comparisons[f"{candidate}_minus_{baseline}"] = compare_predictions(
                self.output / baseline / "dev_holdout" / "predictions.csv",
                self.output / candidate / "dev_holdout" / "predictions.csv",
                replicates=gates["bootstrap_replicates"], seed=gates["bootstrap_seed"])
        chosen = results[candidate]
        encoding_ratio = locked["cheap_results"][candidate]["encoding_ratio_to_C"]
        maximum_ratio = gates["maximum_encoding_ratio"]
        verdict = gate_decision(comparisons[f"{candidate}_minus_C"], chosen["diagnostics_acceptable"],
            encoding_ratio is not None and (maximum_ratio is None or encoding_ratio <= maximum_ratio),
            chosen["protected_output"]["passed"], self.config)
        combined = {}
        # Report the 10k union by combining saved predictions, not by searching
        # again, and label its exposure to cheap-set method selection.
        for arm in arms:
            pooled = []
            for pool in ("dev_cheap", "dev_holdout"):
                source = self.output / arm / pool / "predictions.csv"
                with source.open(encoding="utf-8", newline="") as stream:
                    pooled.extend(csv.DictReader(stream))
            seen = [(row["seed"], row["query_id"]) for row in pooled]
            if len(seen) != len(set(seen)):
                raise ValueError("Cheap and held-out prediction IDs overlap")
            correct = sum(row["label"] == row["prediction"] for row in pooled)
            combined[arm] = {"top1": 100. * correct / len(pooled), "correct": correct, "total": len(pooled),
                             "selection_exposure": "includes the cheap method-selection queries"}
            path = self.output / arm / "dev_full_predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(pooled[0]))
                writer.writeheader()
                writer.writerows(pooled)
        report = {**self.decisions["holdout"], "status": "complete", "completed_at": utc_now(),
                  "candidate": candidate, "results": results, "comparisons": comparisons,
                  "verdict": verdict, "dev_full_including_selection": combined,
                  "interpretation": "Failure stops further spending under this policy; it does not prove the method cannot work"}
        self.decisions["holdout"] = report
        self.save_decisions()
        atomic_json(self.output / "holdout_report.json", report)
        self.comparison_csv("comparison_holdout.csv", results)
        return report

    def confirmation(self):
        import numpy as np
        import statistics
        from compare_predictions import compare_predictions
        from repair_data import open_feature_cache
        locked = self.verify_lock(self.args.candidate)
        if not self.decisions.get("holdout", {}).get("verdict", {}).get("promoted", False):
            raise ValueError("Full confirmation requires a passed locked held-out gate")
        if self.args.val_npz is None or not self.args.val_npz.is_file():
            raise ValueError("Full confirmation needs --val-npz pointing to the released official validation archive")
        if self.args.val_npz.name != "1K_val_ff2048.npz":
            raise ValueError("Expected the released 1K_val_ff2048.npz archive")
        arms = list(dict.fromkeys(["C", "S", locked["candidate"]]))
        seeds = self.config["confirmation"]["seeds"]
        configured_pin = self.config["source"].get("validation_sha256")
        if configured_pin is not None and self.args.val_sha256 is not None and configured_pin != self.args.val_sha256:
            raise ValueError("CLI validation pin differs from the predeclared configuration")
        val_pin = configured_pin or self.args.val_sha256
        if not isinstance(val_pin, str) or len(val_pin) != 64 or any(ch not in "0123456789abcdef" for ch in val_pin):
            raise ValueError("Pin source.validation_sha256 before the pilot, or supply --val-sha256 once at confirmation, from the same trusted release")
        require_unchanged(self.output / "full_confirmation" / "validation_pin.json",
                         {"sha256": val_pin, "repository": self.config["source"]["repository"],
                          "revision": self.config["source"]["revision"]})
        validation = open_feature_cache(self.args.val_npz, self.args.cache_dir, expected_sha256=val_pin)
        if validation[2]["feature_shape"] != [50000, 2048] or validation[2]["classes"] != 1000:
            raise ValueError("Official FF2048 validation shape/class count mismatch")
        require_unchanged(self.output / "full_confirmation" / "validation_source.json", validation[2])
        dense = self.evaluate("full_parent", "official_validation", confirmation=True, validation=validation,
                              seed=seeds[0], dense=True)
        integrity = {"measured_top1": dense["top1"], "target_top1": 71.19,
                     "tolerance_percentage_points": .05, "passed": abs(dense["top1"] - 71.19) <= .05}
        atomic_json(self.output / "full_confirmation" / "dense_integrity.json", integrity)
        if not integrity["passed"]:
            raise RuntimeError("Full dense FF2048 integrity gate failed; sparse results must not be reported as the intended comparison")
        # Candidate, recipe, seeds and endpoint are already immutable. This
        # frozen-parent integrity measurement cannot select or tune a method;
        # placing it here avoids spending 4--6 fits on incompatible inputs.
        self.decisions["confirmation_integrity"] = {**integrity,
            "scope": "locked-protocol parent/input check before paid fits; no sparse validation score used for selection"}
        self.save_decisions()
        for seed in seeds:
            for arm in arms:
                fitted = self.fit(arm, confirmation=True, seed=seed)
                del fitted
        results = {arm: [] for arm in arms}
        for seed in seeds:
            for arm in arms:
                results[arm].append(self.evaluate(arm, "official_validation", confirmation=True, seed=seed, validation=validation))
        combined_paths = {}
        for arm in arms:
            path = self.output / "full_confirmation" / f"{arm}_predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = None
                for seed in seeds:
                    source = self.output / "full_confirmation" / f"seed_{seed}" / arm / "official_validation" / "predictions.csv"
                    with source.open(encoding="utf-8", newline="") as read:
                        rows = csv.DictReader(read)
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=rows.fieldnames)
                            writer.writeheader()
                        writer.writerows(rows)
            combined_paths[arm] = path
        summaries = {arm: {"mean_top1": statistics.mean(row["top1"] for row in rows),
                           "sample_std_top1": statistics.stdev(row["top1"] for row in rows) if len(rows) > 1 else None,
                           "per_seed": rows} for arm, rows in results.items()}
        gates = self.config["gates"]
        comparisons = {}
        for arm in arms:
            if arm == "C":
                continue
            for baseline in dict.fromkeys(["C", "S"]):
                if arm == baseline:
                    continue
                report = compare_predictions(combined_paths[baseline], combined_paths[arm],
                    replicates=gates["bootstrap_replicates"], seed=gates["bootstrap_seed"])
                paired = [report_row["delta_percentage_points"] for report_row in report["per_seed"]]
                report["paired_seed_mean"] = statistics.mean(paired)
                report["paired_seed_sample_std"] = statistics.stdev(paired) if len(paired) > 1 else None
                comparisons[f"{arm}_minus_{baseline}"] = report
        result = {"protocol": self.config["protocol"], "candidate": locked["candidate"], "seeds": seeds,
                  "dense_integrity": integrity, "methods": summaries, "comparisons": comparisons,
                  "training": "fresh matched initialization; final fixed epoch; S never receives OT",
                  "limitations": "Two initial paired seeds; query intervals are conditional on these fits. A third seed is advisable before stronger close-result claims.",
                  "transfer": self.decisions["transfer"]}
        atomic_json(self.output / "full_confirmation" / "summary.json", result)
        self.decisions["confirmation"] = {"status": "complete", "methods": arms, "seeds": seeds,
                                         "summary": "full_confirmation/summary.json"}
        self.save_decisions()
        return result

    def dispatch(self):
        stage = self.args.stage
        self.stage_record(stage, "running")
        if stage == "check":
            result = self.check()
        elif stage in ("warm", "triage", "attribution", "wide", "arm"):
            self.require_unlocked()
            self.require_checks()
            if stage in ("attribution", "wide"):
                self.require_promising("N" if stage == "wide" else None)
            if stage == "arm":
                arm = self.args.arm
                if arm is None:
                    raise ValueError("--stage arm requires --arm")
                if arm in ("R", "M", "O"):
                    self.require_promising()
                if arm in ("F", "A"):
                    self.require_promising("N")
                arms = [arm]
            else:
                arms = ["W"] if stage == "warm" else self.config["stages"][stage]
            failed = {}
            for arm in arms:
                try:
                    fitted = self.fit(arm)
                    del fitted
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    if stage != "triage" or arm in ("W", "C"):
                        raise
                    # An invalid experimental branch must remain visible. It
                    # does not silently become a missing successful result.
                    failed[arm] = repr(error)
                    self.decisions["arms"][arm] = {"status": "failed", "error": repr(error)}
                    self.save_decisions()
            if stage == "triage":
                self.evaluate("full_parent", dense=True)
                results = self.pilot_results([arm for arm in ("C", "S", "N") if arm not in failed])
                self.decisions["triage"] = screening_decision(results, self.config)
                self.decisions["triage"]["failed_arms"] = failed
                if not self.decisions["triage"]["promising"]:
                    for later in ("attribution", "wide", "lock", "holdout", "confirm"):
                        self.decisions["stages"][later] = {"status": "skipped", "reason": "cheap-screen gate did not pass"}
                self.save_decisions()
                self.comparison_csv("comparison_cheap.csv", results)
                result = self.decisions["triage"]
            elif stage != "warm":
                result = {arm: self.evaluate(arm) for arm in arms if arm != "W"}
                self.comparison_csv(f"comparison_{stage}.csv", result)
            else:
                result = {"W": "complete; continuations are separate explicit stages"}
        elif stage == "lock":
            result = self.lock(self.args.candidate)
        elif stage == "holdout":
            result = self.holdout()
        elif stage == "confirm":
            result = self.confirmation()
        elif stage == "timing":
            candidate = self.args.candidate or self.decisions.get("lock", {}).get("candidate")
            if candidate is None:
                raise ValueError("Finalist timing requires --candidate or an existing locked candidate")
            if "lock" in self.decisions:
                self.verify_lock(candidate)
            result = self.pilot_results(list(dict.fromkeys(["C", "S", candidate])), timed=True)
            atomic_json(self.output / "finalist_timing.json", result)
        else:
            raise ValueError(f"Unsupported stage {stage}")
        self.stage_record(stage, "complete")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-npz", type=Path)
    parser.add_argument("--val-npz", type=Path)
    parser.add_argument("--val-sha256", help="Trusted validation archive digest, locked on first confirmation if not predeclared in config")
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mpsae_repair"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/mpsae_repair_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate", choices=list("RMOSNFA"))
    parser.add_argument("--arm", choices=list("WCRMOSNFA"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.stage == "plan":
        print(json.dumps(build_plan(config, args.candidate or "N"), indent=2))
        return 0
    if args.stage == "preflight":
        validate_implemented_settings(config)
        if args.val_npz is not None or args.val_sha256 is not None:
            parser.error("Validation options are reserved for confirmation")
        import torch
        from repair_benchmark import preflight
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result = preflight(torch.device(args.device), config)
        print(json.dumps(json_value(result), indent=2))
        return 0
    if args.stage == "prepare":
        if args.train_npz is None or args.splits_dir is None:
            parser.error("--stage prepare requires --train-npz and --splits-dir")
        if args.val_npz is not None or args.val_sha256 is not None:
            parser.error("Preparation uses training data only; validation options are reserved for confirmation")
        validate_implemented_settings(config)
        from prepare_splits import prepare_from_config
        manifest = prepare_from_config(args.train_npz, args.splits_dir, config)
        print(json.dumps({"protocol": manifest["protocol"], "source_sha256": manifest["source"]["sha256"],
                          "counts": manifest["counts"], "warnings": manifest["warnings"]}, indent=2))
        return 0
    if args.stage == "check" and args.train_npz is None:
        import unittest
        import check_reference
        suite = unittest.defaultTestLoader.loadTestsFromModule(check_reference)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        print("Reference checks only. A training-source check is still required before fitting.")
        return 0 if outcome.wasSuccessful() else 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.output_dir / "runner.lock"):
        run = None
        try:
            run = ResearchRun(args, config)
            result = run.dispatch()
            print(json.dumps(json_value(result), indent=2, allow_nan=False))
        except BaseException as error:
            failure = {"stage": args.stage, "time": utc_now(), "error": repr(error), "traceback": traceback.format_exc()}
            atomic_json(args.output_dir / "failure.json", failure)
            if run is not None:
                run.stage_record(args.stage, "failed", error=repr(error))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


