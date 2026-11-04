"""Names shared by the selected ImageNet benchmark, CLI and CPU reports."""

BENCHMARK_METHODS = ("csr", "csrv2", "mpsae", "mpsaev2")
METHOD_LABELS = {"csr": "CSR", "csrv2": "CSRv2",
                 "mpsae": "MPSAE", "mpsaev2": "MPSAEv2"}
# Read existing ablation results and accept the former spelling as input.
# New configurations, trial names and result rows always use the v2 name.
METHOD_ALIASES = {"mpsae-annealed": "mpsaev2", "dlmmsae-annealed": "dlmmsaev2"}


# Settings that define a comparable result. Completed fits matching these are
# not trained again, and reports compare every method that shares them, even
# across code versions or methods with extra hyperparameters of their own.
EXPERIMENT_KEYS = ("k", "hidden_dim", "epochs", "max_steps", "batch_size", "train_size", "dev_size",
                   "anchor_count", "evaluation", "max_val", "seed", "data_seed", "eval_topk",
                   "linear_probe", "probe_epochs")


def comparable_settings(settings):
    """The experiment-defining subset of trial settings (dev_size only matters for dev)."""
    selected = {key: settings.get(key) for key in EXPERIMENT_KEYS}
    if selected["evaluation"] != "dev":
        selected.pop("dev_size")
    return selected


def canonical_method(name):
    return METHOD_ALIASES.get(name, name)
