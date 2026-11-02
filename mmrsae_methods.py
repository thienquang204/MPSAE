"""Names shared by the selected ImageNet benchmark, CLI and CPU reports."""

BENCHMARK_METHODS = ("csr", "csrv2", "dlmmsae", "dlmmsaev2")
METHOD_LABELS = {"csr": "CSR", "csrv2": "CSRv2",
                 "dlmmsae": "DLMMSAE", "dlmmsaev2": "DLMMSAEv2"}
# Read existing ablation results and accept the former spelling as input.
# New configurations, trial names and result rows always use the v2 name.
METHOD_ALIASES = {"dlmmsae-annealed": "dlmmsaev2"}


def canonical_method(name):
    return METHOD_ALIASES.get(name, name)
