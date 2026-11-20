"""Torch-free transport identities shared by training and ablation planning."""
import math

PARTIAL_SOLVER = "partial-greenkhorn-v1"
BALANCED_SOLVER = "balanced-log-sinkhorn-v1"


def solver_name(mass: float) -> str:
    if (isinstance(mass, bool) or not isinstance(mass, (int, float))
            or not math.isfinite(mass) or not 0 < mass <= 1):
        raise ValueError("transport mass must be finite and satisfy 0 < s <= 1")
    return BALANCED_SOLVER if mass == 1 else PARTIAL_SOLVER
