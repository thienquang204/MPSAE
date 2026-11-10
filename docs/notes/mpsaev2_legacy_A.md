# MPSAEv2 historical annealing variants A and B

Status: **legacy note.** Variant A was specified but not implemented in this
refactor; variant B was implemented and is now retired. Current `mpsaev2`
uses fixed K with both the OT matching gap and instance contrastive loss:
`L(K) + L(4K)/8 + NMSE_aux/32 + 2·OT + γ·C_instance(z_K)`.
Neither historical variant below is the current recipe. This file preserves
their definitions for an explicit future ablation; see [`../methods.md`](../methods.md).

## The two variants

Both start from MPSAE: train at budget K, and regularise with the
multi-marginal partial-OT matching gap over three nested codes of the same
pre-activation. They differ in **what the annealing curriculum shrinks**.

| | **A (specified, not implemented here)** | **B (formerly implemented, retired)** |
|---|---|---|
| main budget — gating, reconstruction, inference | **fixed at K** | **annealed** `k(t)`, CSRv2 rule |
| `L(4k)` term | always `4K` | `4·k(t)` |
| OT views | `K`, `2·a(t)`, `4·a(t)` | `k(t)`, `2·k(t)`, `4·k(t)` |
| what it is | "MPSAE plus a curriculum on the OT comparison" | "CSRv2 with OT instead of contrastive" |
| comparison it supports | tests a curriculum on the OT views | compares OT with label-based contrastive under the same K curriculum, with batch-size differences where applicable |

B used the same main-budget curriculum as CSRv2, with OT replacing the
label-based contrastive term. It also used the MPSAE batch size of 1024, so
comparisons with CSRv2 on 4096-row presets changed batch size and update count.

## Variant A, precisely

Let `T` be the total optimiser steps, `K` the target budget, `h` the dictionary
width, and `f = 0.7` the decay fraction.

```
K0   = min(h, max(64, 4K))
Ta   = max(1, ceil(f * T))
a(t) = round(K0 - (K0 - K) * t / Ta)   for t < Ta
     = K                               for t >= Ta
```

Per step:

* gate and reconstruct at `K` (never `a(t)`), so the served code is identical
  to MPSAE at every point in training;
* the nested reconstruction term stays `L(4K)/8`;
* the OT gap is taken over `(z_K, z_{min(2·a(t), h)}, z_{min(4·a(t), h)})`;
* the activity counter advances three times, on `K`, `4K` and the 2K-equivalent
  OT view, matching every other method.

Worked example, `K=8`, `K0=64`, `f=0.7`:

| point in training | code used and reconstructed | OT views |
|---|---|---|
| step 0 | top-8 | 8 / 128 / 256 |
| 35 % (t = Ta/2) | top-8 | 8 / 72 / 144 |
| ≥ 70 % | top-8 | 8 / 16 / 32 |

Note `ceil` here, against the released CSRv2's `int(T*f)` truncation. That was
the previous implementation's choice; if A is ever rebuilt, decide deliberately
whether to match CSRv2's truncation instead.

## Where the previous implementation lived

Commit `89806f5`, before this rewrite:

* `mmrsae.py:56-72` — `linear_ot_annealed_k`, the `a(t)` above
* `mmrsae.py:377` — passing `ot_budgets=(2*ot_k, 4*ot_k)`
* `mpsae_repair.py:287-294` — building the extra `ot_2k` / `ot_4k` views from
  the shared pre-activation

A caution from that code: the name `mpsaev2` meant **A** in the then-current
runner and **B** in the older runner and the transfer code. The same name
covered two recipes, which is part of why this note exists. The current fixed-K
OT-plus-instance-contrastive recipe is a third objective; it must not reuse
those older checkpoints or measurements as if they described the same method.

## If rebuilding

1. Add a `MPSAEV2_A` entry to `METHODS` under a distinct name — `mpsae_otanneal`,
   not a reuse of `mpsaev2`.
2. In `methods.plan_budget`, keep the main budget at `K` and expose `a(t)`
   separately; `compute_step` then gates `K`, `4K` and `2·a(t)`.
3. Extend `methods.describe` so the run identity records which variant it is;
   otherwise two incompatible recipes produce checkpoints that look alike.
4. Cost: 9–11 budgets × 3 seeds per backbone, so about 33 extra fits on
   `ff2048`.
