"""Print the staged research budget without importing training libraries or running fits."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

from repair_data import PROTOCOL, atomic_json, canonical_hash


DEFAULT_CONFIG = Path(__file__).with_name("experiments.json")


def _integer(section, key, minimum=0):
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _number(section, key, minimum=0., maximum=None, *, strict_minimum=False):
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if value < minimum or (strict_minimum and value == minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{key} is outside the supported range")
    return value


def validate_config(config):
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected research protocol {PROTOCOL}")
    for name in ("source", "model", "pilot", "optimizer", "reconstruction", "repair", "bank", "transport", "retrieval", "gates", "confirmation", "arms", "stages", "precision"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing configuration object {name}")
    for key in ("expected_rows", "validation_rows", "dimension", "classes"):
        _integer(config["source"], key, 1)
    for key, length in (("revision", 40), ("train_sha256", 64), ("validation_sha256", 64)):
        value = config["source"].get(key)
        if value is not None and (not isinstance(value, str) or len(value) != length or any(ch not in "0123456789abcdef" for ch in value)):
            raise ValueError(f"source.{key} must be null for planning or a lowercase {length}-hex trusted release identifier")
    for key in ("input_dim", "hidden_dim", "k"):
        _integer(config["model"], key, 1)
    _integer(config["pilot"], "seed")
    for key in ("batch_size", "warm_updates", "branch_updates", "fit_per_class", "cheap_per_class", "holdout_per_class", "anchor_count", "checkpoint_interval"):
        _integer(config["pilot"], key, 1)
    diagnostic_updates = config["pilot"].get("diagnostic_updates")
    if (not isinstance(diagnostic_updates, list) or
        any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= config["pilot"]["branch_updates"] for value in diagnostic_updates) or
        len(set(diagnostic_updates)) != len(diagnostic_updates)):
        raise ValueError("pilot.diagnostic_updates must contain unique in-range integer updates")
    for key in ("lr", "eps"):
        _number(config["optimizer"], key, strict_minimum=True)
    _number(config["optimizer"], "weight_decay")
    betas = config["optimizer"].get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("Adam betas must contain two numbers")
    for beta in betas:
        _number({"beta": beta}, "beta", maximum=1.)
        if beta == 1:
            raise ValueError("Adam betas must be less than one")
    for key in ("main_weight", "wide_weight", "dead_weight"):
        _number(config["reconstruction"], key)
    for key in ("aux_k", "dead_steps"):
        _integer(config["reconstruction"], key)
    for key in ("depth", "rank", "power_iterations"):
        _integer(config["repair"], key, 1)
    for key in ("clean_probability", "erase_probability", "anneal_fraction"):
        _number(config["repair"], key, maximum=1.)
    _number(config["repair"], "alpha_factor", strict_minimum=True)
    for key in ("size", "refresh_interval", "anchor_chunk_size"):
        _integer(config["bank"], key, 1)
    for key in ("teacher_temperature", "student_temperature"):
        _number(config["bank"], key, strict_minimum=True)
    for key in ("weight", "mixture_weight"):
        _number(config["transport"], key, maximum=1. if key == "mixture_weight" else None)
    for key in ("mass", "entropy", "tolerance"):
        _number(config["transport"], key, strict_minimum=True)
    _integer(config["transport"], "group_size", 2)
    _integer(config["transport"], "max_iterations", 1)
    if "max_refine_iterations" in config["transport"]:
        _integer(config["transport"], "max_refine_iterations", config["transport"]["max_iterations"])
    for key in ("query_batch_size", "gallery_chunk_size", "score_buffer_mib", "repeats", "warmup", "encode_batch_size"):
        _integer(config["retrieval"], key, 1)
    _integer(config["retrieval"], "faiss_temp_memory_mib")
    if "gpu_memory_fraction" in config["retrieval"]:
        _number(config["retrieval"], "gpu_memory_fraction", maximum=1., strict_minimum=True)
    _number(config["gates"], "minimum_gain_percentage_points", maximum=100.)
    if config["gates"].get("maximum_encoding_ratio") is not None:
        _number(config["gates"], "maximum_encoding_ratio", strict_minimum=True)
    _integer(config["gates"], "bootstrap_replicates", 1)
    _integer(config["gates"], "bootstrap_seed")
    for key in ("batch_size", "epochs"):
        _integer(config["confirmation"], key, 1)
    seeds = config["confirmation"].get("seeds")
    if (not isinstance(seeds, list) or not seeds or
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Confirmation seeds must be distinct nonnegative integers")
    for section, key in (("transport", "mass_normalization"), ("bank", "cache_teacher_profiles"),
                         ("gates", "holdout_requires_positive_lower_ci"), ("confirmation", "retain_last_batch")):
        if not isinstance(config[section].get(key), bool):
            raise ValueError(f"{section}.{key} must be a boolean")
    if config["optimizer"]["name"] != "Adam":
        raise ValueError("The matched protocol requires Adam, not AdamW")
    if config["transport"]["mass_normalization"]:
        raise ValueError("The matching gap must not be divided by transported mass")
    if config["repair"]["depth"] != 1:
        raise ValueError("Only the specified one-step repair is implemented")
    if config["precision"] != {"dtype": "float32", "amp": False, "tf32": False}:
        raise ValueError("The research recipe requires float32 with AMP and TF32 disabled")
    if config["model"]["input_dim"] != config["source"]["dimension"]:
        raise ValueError("Model and source dimensions differ")
    for name in ("k", "hidden_dim", "input_dim"):
        if config["model"][name] < 1:
            raise ValueError(f"model.{name} must be positive")
    if config["model"]["k"] > config["model"]["hidden_dim"]:
        raise ValueError("K exceeds dictionary width")
    for name in ("batch_size", "warm_updates", "branch_updates", "fit_per_class", "cheap_per_class", "holdout_per_class", "anchor_count", "checkpoint_interval"):
        if config["pilot"][name] < 1:
            raise ValueError(f"pilot.{name} must be positive")
    if config["bank"]["size"] != config["pilot"]["anchor_count"]:
        raise ValueError("Bank size must match the locked split landmark count")
    if config["bank"]["size"] < config["source"]["classes"]:
        raise ValueError("The declared landmark pool requires at least one row per class")
    if config["bank"]["confirmation_exclusion"] != "minibatch_union":
        raise ValueError("Full-data profiles require shared minibatch-union exclusions")
    for name in ("teacher_temperature", "student_temperature"):
        if config["bank"][name] <= 0:
            raise ValueError(f"bank.{name} must be positive")
    for name in ("clean_probability", "erase_probability", "anneal_fraction"):
        if not 0 <= config["repair"][name] <= 1:
            raise ValueError(f"repair.{name} must lie in [0,1]")
    if not 0 < config["transport"]["mass"] < 1 or config["transport"]["entropy"] <= 0:
        raise ValueError("This solver profile requires partial mass in (0,1) and positive entropy")
    if not 0 <= config["transport"]["mixture_weight"] <= 1:
        raise ValueError("Transport mixture weight must lie in [0,1]")
    if set(config["arms"]) != set("WCRMOSNFA"):
        raise ValueError("The declared arm matrix must contain W,C,R,M,O,S,N,F,A")
    for arm, spec in config["arms"].items():
        if set(spec) != {"repair", "mask", "rel_weight", "ot_cost", "auxiliary"}:
            raise ValueError(f"Arm {arm} has missing or unsupported fields")
        if spec["ot_cost"] not in {"old", "off", "mixed"} or spec["auxiliary"] not in {"4k", "wide", "anneal"}:
            raise ValueError(f"Unsupported loss or auxiliary policy in arm {arm}")
        if spec["mask"] and not spec["repair"]:
            raise ValueError(f"Arm {arm}: erasure requires repair")
        if not isinstance(spec["repair"], bool) or not isinstance(spec["mask"], bool):
            raise ValueError(f"Arm {arm}: repair and mask must be boolean switches")
        _number(spec, "rel_weight")
    expected_arms = {
        "W": (False, False, 0., "old", "4k"),
        "C": (False, False, 0., "old", "4k"),
        "R": (True, False, 0., "old", "4k"),
        "M": (True, True, 0., "old", "4k"),
        "O": (True, True, 1., "old", "4k"),
        "S": (True, True, 1., "off", "4k"),
        "N": (True, True, 1., "mixed", "4k"),
        "F": (True, True, 1., "mixed", "wide"),
        "A": (True, True, 1., "mixed", "anneal"),
    }
    arm_fields = ("repair", "mask", "rel_weight", "ot_cost", "auxiliary")
    for arm, expected in expected_arms.items():
        if tuple(config["arms"][arm][key] for key in arm_fields) != expected:
            raise ValueError(f"Arm {arm} changes its declared component contrast; use a distinct future protocol/name for an additional ablation")
    expected_stages = {"triage": ["W", "C", "S", "N"], "attribution": ["R", "M", "O"], "wide": ["F", "A"]}
    if config["stages"] != expected_stages:
        raise ValueError("Stage membership/order is fixed to the gated W/C/S/N, R/M/O, F/A design")
    if [config["reconstruction"][key] for key in ("main_weight", "wide_weight", "dead_weight")] != [1., .125, .03125]:
        raise ValueError("This implementation fixes reconstruction weights to 1,1/8,1/32")
    if config["repair"]["alpha_factor"] != .9 or config["repair"]["anneal_fraction"] != .7:
        raise ValueError("This implementation fixes alpha factor .9 and auxiliary annealing fraction .7")
    if config["bank"]["cache_teacher_profiles"]:
        raise ValueError("Frozen profiles are computed on demand; an on-disk teacher-profile cache is not implemented")
    if not config["confirmation"]["retain_last_batch"] or config["confirmation"]["methods"] != ["C", "S", "selected_candidate"]:
        raise ValueError("Confirmation requires the retained final batch and unique C/S/selected_candidate methods")
    # The historical OT objective skips singleton groups, but requires at
    # least one valid group per training minibatch. Reject unsupported batch
    # sizes during planning instead of failing late on a retained final row.
    for section, row_count in (
        ("pilot", config["pilot"]["fit_per_class"] * config["source"]["classes"]),
        ("confirmation", config["source"]["expected_rows"]),
    ):
        batch_size = config[section]["batch_size"]
        if batch_size < 2 or row_count % batch_size == 1:
            raise ValueError(
                f"{section}.batch_size must be at least 2 and must not leave a one-row "
                f"final batch for {row_count:,} rows; OT requires at least two examples "
                "in every retained training minibatch"
            )
    pilot_batches = math.ceil(config["pilot"]["fit_per_class"] * config["source"]["classes"] / config["pilot"]["batch_size"])
    if config["pilot"]["warm_updates"] != 2 * pilot_batches or config["pilot"]["branch_updates"] != 2 * pilot_batches:
        raise ValueError("The paired pilot uses exactly two warm epochs and two continuation epochs")


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def build_plan(config, candidate="N"):
    validate_config(config)
    if candidate not in config["arms"] or candidate in ("W", "C"):
        raise ValueError("Choose a continuation candidate other than the baseline C")
    pilot, source = config["pilot"], config["source"]
    fit_rows = pilot["fit_per_class"] * source["classes"]
    batches = math.ceil(fit_rows / pilot["batch_size"])
    cumulative = 0
    stages = []
    for name in ("triage", "attribution", "wide"):
        arms = config["stages"][name]
        updates = sum(pilot["warm_updates"] if arm == "W" else pilot["branch_updates"] for arm in arms)
        cumulative += updates
        stages.append({"stage": name, "arms": arms, "additional_updates": updates, "cumulative_updates": cumulative,
                       "gate": "explicit initial launch" if name == "triage" else "promising prior results; explicit next-stage launch"})
    full_methods = list(dict.fromkeys(["C", "S", candidate]))
    full_batches = math.ceil(source["expected_rows"] / config["confirmation"]["batch_size"])
    full_fits = len(full_methods) * len(config["confirmation"]["seeds"])
    return {
        "protocol": config["protocol"], "configuration_sha256": canonical_hash(config),
        "status": "plan only; no training, downloads, validation queries, or GPU work executed",
        "release_provenance": {
            "repository": source["repository"],
            "revision": source.get("revision"),
            "train_sha256": source.get("train_sha256"),
            "validation_sha256": source.get("validation_sha256"),
            "training_requires": "A trusted immutable 40-hex release revision and expected training archive SHA-256, obtained independently of the local file being checked.",
            "confirmation_requires": "Expected official validation archive SHA-256 from the same trusted release, in source.validation_sha256 or the confirmation-only --val-sha256 argument; the runner locks that pin separately.",
            "note": "File names, dimensions, and hashing a locally supplied archive do not authenticate its publisher. Null defaults intentionally permit planning only.",
        },
        "pilot": {"fit_rows": fit_rows, "gallery_rows": fit_rows,
                  "cheap_queries": pilot["cheap_per_class"] * source["classes"],
                  "holdout_queries": pilot["holdout_per_class"] * source["classes"],
                  "anchors": pilot["anchor_count"], "batches_per_epoch": batches,
                  "warm_data_epochs": [0, 1], "continuation_data_epochs": [2, 3],
                  "warm_updates": pilot["warm_updates"], "branch_updates": pilot["branch_updates"]},
        "stages": stages,
        "holdout": {"queries": pilot["holdout_per_class"] * source["classes"], "one_time_after_lock": True, "additional_training_updates": 0},
        "confirmation": {"candidate": candidate, "methods": full_methods, "seeds": config["confirmation"]["seeds"],
                         "fits": full_fits, "updates_per_fit": full_batches * config["confirmation"]["epochs"],
                         "total_updates": full_fits * full_batches * config["confirmation"]["epochs"],
                         "training_example_presentations": full_fits * source["expected_rows"] * config["confirmation"]["epochs"],
                         "final_batch_rows": source["expected_rows"] - (full_batches - 1) * config["confirmation"]["batch_size"],
                         "requires_positive_locked_gate": True, "initialization": "matched fresh initialization; no pilot checkpoint"},
        "transfer": {"default": "deferred", "requires": "positive confirmation, compatible exact parent, separately budgeted reduced adapter"},
        "notes": ["Updates are not GPU hours; record actual wall time before expanding.",
                  "S inherits W's old-OT history in the pilot; fresh confirmation S receives no OT.",
                  "Official validation is not opened by pilot selection.",
                  "The full nine-budget curve and existing 570-head transfer grid are not launched."],
    }


def matrix_rows(config):
    k, h = config["model"]["k"], config["model"]["hidden_dim"]
    for arm, spec in config["arms"].items():
        total = config["pilot"]["warm_updates" if arm == "W" else "branch_updates"]
        start = min(h, max(64, 8 * k)) if spec["auxiliary"] != "4k" else min(h, 4 * k)
        end = min(h, 4 * k) if spec["auxiliary"] != "wide" else start
        yield {"arm": arm, **spec, "k": k, "updates": total, "auxiliary_start": start, "auxiliary_end": end,
               "anneal_updates": math.floor(config["repair"]["anneal_fraction"] * total) if spec["auxiliary"] == "anneal" else 0,
               "initialization": "fresh" if arm == "W" else "common_W", "data_epoch_start": 0 if arm == "W" else 2}


def export_matrix(config, path):
    rows = list(matrix_rows(config))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", default="N")
    parser.add_argument("--json", action="store_true", help="Print the resolved plan as JSON")
    parser.add_argument("--export-dir", type=Path, help="Write plan.json and one full configuration per arm")
    parser.add_argument("--matrix-csv", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    plan = build_plan(config, args.candidate)
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.export_dir / "plan.json", plan)
        for row in matrix_rows(config):
            resolved = copy.deepcopy(config)
            resolved["run"] = row
            atomic_json(args.export_dir / f"{row['arm']}.json", resolved)
        export_matrix(config, args.export_dir / "run_matrix.csv")
    if args.matrix_csv:
        export_matrix(config, args.matrix_csv)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"{plan['protocol']} - plan only")
        for stage in plan["stages"]:
            print(f"{stage['stage']}: {','.join(stage['arms'])}; +{stage['additional_updates']:,} updates; cumulative {stage['cumulative_updates']:,}")
        confirmation = plan["confirmation"]
        print(f"Locked holdout: {plan['holdout']['queries']:,} queries; no fitting")
        print(f"Optional fresh confirmation: {confirmation['fits']} fits ({','.join(confirmation['methods'])}); {confirmation['total_updates']:,} updates")
        print("Transfer is deferred. No full training or GPU-hour estimate is implied.")
        if not config["source"].get("train_sha256") or not config["source"].get("revision"):
            print("Before real-data checks/fitting, pin source.revision and source.train_sha256 from trusted release metadata.")


if __name__ == "__main__":
    main()
