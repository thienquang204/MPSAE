# Methods

What each of the five trainable configurations optimises, where the baselines
come from, and which parts are ours. Formulas here match `mpsae/model.py`,
`mpsae/losses.py`, `mpsae/ot.py` and `mpsae/methods.py`; where the published
code and the published paper disagree, both are given and the code wins.

## The problem

A frozen backbone produces a dense embedding `x ∈ R^d` — 2048 for the MRL
fixed-feature ResNet-50, 384 to 768 for the others. Serving that at web scale
is expensive, so we want a code with at most `K` active coordinates that still
supports nearest-neighbour retrieval. The question is what to train that code
against.

MRL answers it by retraining the backbone so that truncating the embedding
degrades gracefully. CSR answers it by freezing the backbone and training a
small sparse autoencoder on top, which is far cheaper and, in CSR's reported
numbers, more accurate at small budgets. We keep CSR's answer and change only
the regulariser.

## The shared autoencoder

Every method in this repository uses the same module, so a difference in
results cannot come from the architecture. It is the released CSR model
(`CSR/Vision/model_zoo.py`, identical in `CSRv2/image/model_zoo.py`):

```
pre = W(x − b_pre) + b_lat          W ∈ R^{h×d},  h = 8192
z   = ReLU(TopK(pre, k))            at most k nonzero
x̂   = Wᵀz + b_pre                   tied decoder, never renormalised
```

`b_pre` and `b_lat` start at zero; `W` gets PyTorch's default `nn.Linear`
initialisation (Kaiming uniform, `a = √5`). There is no LayerNorm — the
released image pipeline always passes `normalize=False`.

### Dead latents

A latent that stops being selected receives no gradient and cannot recover, so
the release tracks how long each has been idle. `stats_last_nonzero` is reset
for latents whose selected pre-activation exceeds `1e-5`, then every counter is
incremented — once per `top_k` call, not once per step. Latents idle for more
than `dead_threshold = 30` gates are *dead*, and the AuxK term asks them to
reconstruct what the main code missed.

Because the release gates three times per optimiser step, `dead_threshold = 30`
is about ten steps, not thirty. That is easy to misread, and it is why every
method here gates exactly three times: the threshold then means the same thing
across the whole comparison.

### The shared objective

```
L = L(k) + 0.125·L(4k) + (1/32)·NMSE_aux + γ·L_reg
```

* `L(k) = MSE(x, x̂_k)` is an elementwise **mean**, not the summed squared norm
  the papers write. The released code uses `nn.MSELoss`, and every loss weight
  was tuned against that, so the mean is what reproduces the numbers.
* `L(4k)` is the nested reconstruction at four times the budget, the
  "multi-TopK" term from Gao et al.'s TopK SAE.
* `NMSE_aux` is the AuxK revival term: the dead latents reconstruct
  `x − x̂.detach() + b_pre.detach()`, normalised by the error of a batch-mean
  predictor, with `k_aux = 512`. The released `nan_to_num(0)` catches only the
  `0/0` case — a constant target reconstructed exactly. A constant target
  reconstructed *inexactly* divides by zero and `nan_to_num`'s default maps
  `+inf` to the float32 maximum rather than to zero. We reproduce that rather
  than improve on it; with real embeddings the denominator is never zero.
* `L_reg` is where the methods differ.

## The five methods

`mpsae` and `mpsaev2` use training batches of **1024** on every backbone and
the original constant learning rate of **4e-5**. `csr`, `csrv2` and
`csr_multitopk` retain their backbone batch-size presets. OT solver limits
remain **400** initial iterations and **8000** refinement iterations;
convergence can stop a solve earlier. Epochs, OT groups of 32, loss weights
and tolerances are unchanged. Update counts and annealing schedules are
computed from the effective batch size. Run identities and campaign metadata
record the effective recipe; checkpoints with different settings cannot resume.

| method | K during training | gates per step | regulariser | extra reconstruction |
|---|---|---|---|---|
| `csr` | fixed K | K, 4K, K | γ·`inbatch_cl` | — |
| `csrv2` | annealed `k(t)` | k, 4k, k | γ·`label_based_cl` | — |
| `mpsae` | fixed K | K, 4K, 2K | 2·OT gap | — |
| `mpsaev2` | annealed `k(t)` | k, 4k, 2k | 2·OT gap | — |
| `csr_multitopk` | fixed K | K, 4K, 2K | γ·`inbatch_cl` | `L(2K)/8` |

γ is 0.1 on the FF2048-style preset and 1.0 on the Table 9 preset.

### csr

`inbatch_cl` stacks two views, L2-normalises them, and maximises agreement at
temperature 0.2 against every other sample in the batch:

```
L = −mean_i log( Σ_{j ∈ pos(i)} exp(sim_ij/τ) / Σ_{j ∈ pos(i)∪neg(i)} exp(sim_ij/τ) )
```

The positive of row `i` is the other copy of the same sample. The two copies
are two identical deterministic forward passes, so the term reduces to a
uniformity penalty on the normalised sparse codes. There is no augmentation
anywhere in the released pipeline.

We evaluate this with log-sum-exp rather than building the `exp` matrix; it is
algebraically the same and `tests/test_against_release.py` proves agreement
against the literal transcription.

### csrv2

Two changes over `csr`, both from the released `image/main_visual.py`:

1. **Linear k-annealing.** `k` falls from `k_init` to the target over the first
   70 % of optimiser steps:

   ```
   decay = int(total_steps × 0.7)
   k(t)  = target                               if t ≥ decay
         = round(k_init·(1−t/decay) + target·t/decay)   otherwise
   ```

   `k_init` is 64 below a target of 64 and `4·target` above, then clamped to
   `h//4` and up to the target. So a target of 768 starts at 2048 rather than
   3072, and a target of 2048 does not anneal at all. The `4k` reconstruction
   follows `4·k(t)`.

2. **Supervised contrastive loss.** `label_based_cl` has the same shape as
   `inbatch_cl` but takes positives to be every same-label sample across both
   copies, excluding only the diagonal self-comparison.

### mpsae — the contribution

CSR's regulariser compares *different examples* at one budget. Ours compares
*the same example* across budgets, and asks that the cheap code already agree
with the expensive one.

Take the three nested codes of one batch group — `z_K`, `z_2K`, `z_4K`, all
read off the same pre-activation — and L2-normalise them to `a`, `b`, `c`. The
cost of matching example `i` at budget K to example `j` at 2K and `l` at 4K is
the mean of the three pairwise cosine distances, scaled into the unit interval:

```
C[i,j,l] = (2/9)·[ (1 − a_i·b_j) + (1 − a_i·c_l) + (1 − b_j·c_l) ]
```

The regulariser is the **matching gap**: how much worse the identity matching
is than the best admissible one.

```
gap = F_η(C, s·J) − min_{P ∈ U_s} F_η(C, P)

F_η(C, P) = ⟨C, P⟩ + η·[ H(P) + Σ_m H(q_m) ]        H(x) = Σ x(log x − 1)
U_s       = { P ≥ 0 : ΣP = s, every marginal ≤ 1/n }
```

`J` is the identity matching, `q_m` the three slack marginals absorbing the
unmatched mass, `s = 0.9`, `η = 0.2`, `n = 32`. Partial transport (`s < 1`)
matters: it lets the solver abandon examples whose budgets genuinely disagree
instead of forcing a match, so a few collapsed codes cannot dominate the term.

If the budgets already agree the identity matching *is* optimal and the gap is
zero; the more a cheap code reorders the neighbourhood relative to an expensive
one, the larger it grows.

The inner minimisation runs under `no_grad` with a greedy Greenkhorn-style
solver on slack variables, updating whichever constraint is worst in KL. The
gradient then comes from the envelope theorem — the optimal plan is held fixed:

```
∂gap/∂C = s·J − P*
```

so one solve yields both the value and the gradient. Groups of 32 are solved
independently and averaged; equal-sized groups go through a batched solver that
keeps per-group scales and convergence decisions separate.

Two guards refine the solve rather than hiding its symptoms: an infeasible plan
(marginal or mass residual above tolerance), and a materially negative gap,
which can only mean the "optimal" plan was not optimal. Neither is clipped.

Prior art, credited: the matching-gap formulation follows Piran et al.
(ICML'24); multi-marginal partial OT is standard. The combination with a nested
Top-K autoencoder, and the use of budget-consistency as the regulariser, is
ours.

### mpsaev2

`csrv2` with the contrastive term replaced by the OT gap: the main budget
anneals, the OT views are `k(t)`, `2k(t)`, `4k(t)`. Against `csrv2`, the
regulariser and, for the 4096-row backbone presets, batch size differ.

This is "variant B". Variant A — fixed main budget, annealing only the OT
comparison views — is specified in [`notes/mpsaev2_legacy_A.md`](notes/mpsaev2_legacy_A.md)
and deliberately not implemented.

### csr_multitopk — the control

MPSAE differs from CSR in two ways at once: it looks at a 2K view, and it
couples the views with OT. Without a control, a gain could be explained by
either. `csr_multitopk` keeps CSR's contrastive term and its gate sequence and
adds only `L(2K)/8`, so:

* **MPSAE > csr_multitopk** compares OT coupling with supervising a 2K view,
  including batch-size and update-count differences where applicable;
* **csr_multitopk > csr** measures what the extra view alone buys.

It follows CSRv2 App. G's "Diverse Multi-TopK" ablation in spirit, which found
extra reconstruction terms help a little but cost more than annealing.

## Findings about the released baselines

Verified by reading CSR @ `b284898` and CSRv2 @ `3b2d4d2`. These change what
"reproduce the baseline" means, so they are recorded rather than quietly
handled.

1. **CSR's contrastive loss compares a code with an identical copy of itself.**
   The second view is the same deterministic forward pass. The CSRv2 paper
   describes CSR's positives as augmentation-based; the released code has no
   augmentation. We reproduce the code.
2. **CSRv2's supervised loss is inert in the release.** `CustomDataset` reads
   `train_data.get('labels')`, every released producer writes `label`. All
   labels become 0, `pos` and `neg` are clamped at `1e-8`, and the loss
   collapses to ≈0 — so the advertised supervision never happens. Our archives
   store `labels`, which makes the term live; the rename is recorded as a
   deviation in every manifest, and `tests/test_against_release.py` pins both
   the working and the degenerate behaviour.
3. **The annealing is linear.** CSRv2 App. G.2 reports cosine as worse in their
   own sensitivity study, so a cosine schedule would be a different algorithm.
4. **`k_init` carries clamps the paper omits** (`h//4`, and up to the target).
5. **The dead-latent counter advances three times per step.**
6. **Released defaults differ from the published tables**: lr 4e-4 vs 4e-5,
   weight decay 0 vs 1e-4, CSRv2 `auxk` 1024 vs 512. These are settings, so we
   take the papers' values and say so.
7. **There is no released image checkpoint for the full CSRv2** and no code for
   FF2048 fine-tuning. The comparable published variant is CSRv2-linear.
8. **The optimiser is `Adam`, not `AdamW`**, so the weight decay is coupled L2
   and applies to the biases too. `StepLR(step_size=10, gamma=1)` never changes
   the rate; there is no warmup and no gradient clipping.

## Precision

Not uniform, and deliberately so — each step matches the release:

| step | precision |
|---|---|
| training | float32, TF32 off |
| embedding extraction | fp16 autocast, stored float32 |
| code encoding for evaluation | fp16 autocast, stored float32 |
| **search** | **float32, TF32 off** |
| linear probe | float32 |

`CSRv2/image/utils.py` wraps the encode in `autocast()`; `faiss_nn.py` casts to
float32 before `IndexFlatL2`. Searching in half precision would reorder
near-ties across a 1.28 M-row gallery and change top-1, so it is never done.
TF32 is disabled explicitly because it would silently alter every matmul in
both the objective and the search.

## Experimental design

**E1, per-K.** One model trained at each budget, evaluated at that budget. This
is what CSR Table 4 reports and the only fair comparison: a CSR model trained
at K=8 has only ever been optimised for eight active dimensions, and the CSR
author writes in issue #17 that changing `k` at inference "might lead to
sub-optimal performance".

**E2, one checkpoint across budgets.** The E1 anchor checkpoints are evaluated
at every budget in the grid, at no extra training cost. This tests a claim both
papers make and neither measures — CSRv2 §3: *"Training with the multi-TopK
loss L(k)+L(4k)/8 ensures that CSR could generalize to different k's at test
time"*. Every E2 point carries its measured `nnz` as well as the nominal
budget, because after ReLU a code can hold far fewer than K nonzeros and a
nominal budget can overstate the real sparsity.

One seed (1227), the released CSR seed. Differences between methods use the
same training seed, which controls initialisation and data order. Results have
n=1; no between-seed standard deviation, uncertainty band or p-value is reported.
Multiple seeds can still be explicitly selected in a separate campaign.

## What is tested

`tests/reference/csr_released.py` transcribes the published source, and
`tests/test_against_release.py` checks our model, both contrastive losses, the
AuxK gradient, the dead-latent counter, the full objective, the gradients and
three Adam steps against it. `tests/test_schedules.py` checks the annealing at
every step of several schedules. `tests/test_methods.py` checks that the five
methods differ only as designed. `tests/test_ot.py` checks feasibility,
nonnegativity, the envelope gradient and group independence, plus the pinned
legacy numerics once `capture_golden.py` has been run on a GPU box.
