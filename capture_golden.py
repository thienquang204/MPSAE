#!/usr/bin/env python3
"""Pin the OT numerics of the previous implementation before it is deleted.

Run this on a machine with torch, once, before removing the legacy modules:

    python capture_golden.py

It checks out the pre-rewrite commit into a temporary worktree, imports the old
``mpsae_repair`` from there, evaluates the transport primitives on fixed CPU
inputs, and writes ``tests/golden/ot_reference.json``. ``tests/test_ot.py``
then asserts the rewritten ``mpsae.ot`` reproduces those values.

Only the OT path needs this treatment. It is ours, it was ported verbatim, and
its float32 behaviour depends on association order and clamp placement that no
paper documents. The CSR and CSRv2 objectives are instead tested against the
published source in ``tests/reference/``, which is a stronger oracle than our
own previous attempt at them.

The legacy code stays reachable in git regardless, so this can be run later:
nothing is lost by deleting the working-tree copies first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

#: Merge commit of the pipeline this rewrite replaces.
LEGACY_COMMIT = "89806f5"
LEGACY_MODULE = "mpsae_repair"


def _tensor(payload) -> dict:
    return {"shape": list(payload.shape), "dtype": str(payload.dtype),
            "values": payload.detach().double().cpu().reshape(-1).tolist()}


def fixtures(torch):
    """Deterministic inputs: three nested Top-K codes, as training produces."""
    generator = torch.Generator().manual_seed(20261106)
    cases = []
    for name, group, hidden, k in (("tiny", 4, 32, 2), ("small", 8, 64, 4), ("group", 12, 128, 3)):
        pre = torch.randn(group, hidden, generator=generator)
        views = []
        for budget in (k, 2 * k, 4 * k):
            selected = torch.topk(pre, k=min(budget, hidden), dim=-1)
            dense = torch.zeros_like(pre).scatter_(-1, selected.indices, selected.values)
            views.append(torch.relu(dense))
        cases.append({"name": name, "pre": pre, "views": views, "k": k, "hidden": hidden})
    return cases


def capture(legacy_root: Path) -> dict:
    sys.path.insert(0, str(legacy_root))
    import torch

    legacy = __import__(LEGACY_MODULE)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    captured = {"legacy_commit": LEGACY_COMMIT, "torch": str(torch.__version__), "cases": []}
    for case in fixtures(torch):
        z1, z2, z3 = case["views"]
        cost = legacy.circular_variance_cost(z1, z2, z3)
        plan, slacks, stats = legacy.greenkhorn_mmpot(cost.detach(), 0.9, 0.2, 100, 1e-4)
        for name, grad in (("z1", z1), ("z2", z2), ("z3", z3)):
            grad.requires_grad_(True)
        differentiable = legacy.circular_variance_cost(z1, z2, z3)
        loss, gap_stats, delta = legacy.matching_gap_from_cost(
            differentiable, mass=0.9, eta=0.2, max_iters=100, tol=1e-4,
            require_feasible=True, max_refine_iters=2000)
        gradients = torch.autograd.grad(loss, (z1, z2, z3), allow_unused=True)
        captured["cases"].append({
            "name": case["name"], "k": case["k"], "hidden": case["hidden"],
            "pre": _tensor(case["pre"]),
            "cost": _tensor(cost),
            "cost_checksum": float(cost.double().sum()),
            "plan": _tensor(plan),
            "slacks": [_tensor(value) for value in slacks],
            "solver_stats": {key: float(value) for key, value in stats.items()},
            "gap": float(loss.detach()),
            "gap_stats": {key: float(value) for key, value in gap_stats.items()},
            "delta_checksum": float(delta.double().sum()),
            "gradients": [_tensor(value) if value is not None else None for value in gradients],
            "entropy_plan": float(legacy.entropy_term(plan).double()),
        })
    return captured


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", default=LEGACY_COMMIT)
    parser.add_argument("--out", type=Path, default=Path("tests/golden/ot_reference.json"))
    parser.add_argument("--worktree", type=Path, default=None,
                        help="existing checkout of the legacy commit; skips creating a worktree")
    args = parser.parse_args(argv)

    if args.worktree:
        payload = capture(args.worktree.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="mpsae-legacy-") as scratch:
            target = Path(scratch) / "legacy"
            print(f"checking out {args.commit} into {target}", file=sys.stderr)
            subprocess.run(["git", "worktree", "add", "--detach", str(target), args.commit],
                           check=True)
            try:
                payload = capture(target)
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(target)], check=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(payload['cases'])} cases, "
          f"{args.out.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
