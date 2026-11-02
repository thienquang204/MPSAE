# Experiment plan: MPSAE repair, neighborhood cost, and auxiliary annealing

**Historical sparse-start and mixed-cost experiment design.** As of 2 November
2026, active repair/MMRSAE objectives use angular OT only. `experiments.json`
declares `mpsae-repair-angular-v2`; O and N have identical components, so their
difference no longer tests neighborhood OT. This staged study retains its
sparse-start architecture. The current MMRSAE runner instead synthesizes a
landmark-context embedding before tied SAE/Top-K, with nested clean supports
and no post-SAE repair. Fixed input landmarks remain part of the inference model.
MMRSAE-annealed (MMRSAEv2) now anneals actual training K by half-cosine like
CSRv2/MPSAEv2; its default auxiliary budget is 4K(t). Auxiliary-only annealing
below is historical. See [MMRSAE_IMAGENET.md](MMRSAE_IMAGENET.md) for the current
`mmrsae-imagenet-landmark-first-v3` protocol and multi-K comparison.

**Version:** mpsae-repair-neighborhood-v1  
**Date:** 29 October 2026  
**Primary aim:** decide which components justify their accuracy, encoding, and training costs using a small, reproducible sequence of tests.

This is a proposed research extension to the original experiment. This checkout implements its ImageNet stages in `mpsae_repair_experiment.py`, with `run_mpsae_repair.sh` and Docker scope `repair` as entry points. Source inspection supports the implementation mapping below; no experiment results or runtime validation are claimed.

Read `full_theory_detail.md` for the equations and `README.md` for launching. Use a copy of `experiments.json` as the resolved starting configuration and fill its trusted source revision/hash. Keep strict paper-reproduction outputs and research-screen outputs in separate directories and tables. `ENV_FILE=repair.env bash run_all_experiments_docker.sh pilot` runs configured preparation/checks and W/C/S/N, using a copy of `repair.env.example`; later stages remain explicit.

## 1. The cheapest useful order

Run the common warm start W, then **C, S, and N** first.

- **C:** unchanged MPSAE, given the same extra training.
- **S:** learned repair, training masks, and direct teacher-neighborhood KL; no OT during continuation.
- **N:** S plus the proposed mixture of old and neighborhood OT costs.

This answers the important early question: does the complete idea beat continued baseline training, and does OT beat the simpler relational alternative?

If S and N are unpromising with functioning gradients, stop before the rest of the grid. If either is promising, run R, M, and O to understand why. All continuations still start from the same W, not the output of a previously completed arm.

| Stage | Additional work | Cumulative optimizer updates | Purpose |
|---|---:|---:|---|
| Prerequisites and tiny checks | No training | 0 | Catch math, interface, provenance, and split errors |
| W: common warm start | 250 | 250 | Shared original-MPSAE checkpoint |
| Triage: C, S, N | 3 × 250 | **1,000** | Decide whether to investigate further |
| Attribution: R, M, O | 3 × 250 | **1,750** | Separate repair, masks, direct KL, and old OT |
| Optional wide-budget test: F, A | 2 × 250 | **2,250** | Separate wide support from scheduling |
| Locked 8,000-query gate | No training | Unchanged | Avoid spending full-training compute on a weak pilot |
| From-scratch confirmation | 4–6 full fits initially | Separate budget | Remove warm-start bias and measure seed variation |
| Second budget / transfer | Only after confirmation | Separate budget | Test scope beyond the primary setting |

Updates are not equal GPU time across methods. Measure actual wall time before authorizing a larger compute budget. The configuration intentionally has no invented GPU-hour estimate.

## 2. What stays the same as the supplied FF2048 experiment

The primary data source remains:

~~~text
W1nd-navigator/CSR-precompute-embeds
FF2048_RN50_Embeds/1K_train_ff2048.npz
FF2048_RN50_Embeds/1K_val_ff2048.npz
~~~

The documented full-feature parent identity is `RAIVNLab/MRL:r50_mrl0_e0_ff2048.pt`. The archives provide `data` and `label`. Record archive hashes and actual shapes before training.

| Setting | Supplied FF2048 strict recipe | Proposed pilot |
|---|---:|---:|
| Frozen parent dimension | 2048 | 2048 |
| Dictionary width | 8192 | 8192 |
| Main active budget | Separate fit per K | K=8 only |
| Training rows | 1,281,167 | 128,000 |
| Batch size | 4096 | 1024 |
| Training length | 10 epochs / 3,130 updates | W:250; each continuation:250 |
| Seed | 1227 | 1227 |
| Adam learning rate | 4e-5 | Same |
| Adam betas | 0.9, 0.999 | Same |
| Adam epsilon | 6.25e-10 | Same |
| Weight decay | 1e-4 | Same Adam convention; do not switch to AdamW |
| LR scheduler | None | None |
| Dictionary initialization | Kaiming uniform | Same for W |
| Bias initialization | Zero | Same for W |
| Decoder renormalization | None | None |
| Main / wide / dead-feature weights | 1 / 1/8 / 1/32 | Same |
| SAE arithmetic | Float32, AMP/TF32 off | Same |
| OT mass / entropy | 0.9 / 0.2 | Same when OT is enabled |
| OT weight / group size | 2 / 32 | Same when OT is enabled |
| OT solver limit / tolerance | 100 / 1e-4 | Same initial solver settings, actual residuals checked |
| Official validation use | Final benchmark | Not opened for method selection |
| Source status | Existing setup described by attachment | New non-strict research profile |

Do not add mean-centering, different dictionary normalization, different optimizers, or additional training epochs to only the new method. Such changes would need their own controls.

The source document's 71.19% ±0.05 percentage-point dense integrity target belongs to the **full gallery and official validation** protocol. It is not a target for the smaller pilot gallery.

## 3. Deterministic data pools and leakage prevention

### 3.1 Create the split once

~~~bash
python prepare_splits.py --config experiments.json \
  --train-npz /path/to/FF2048_RN50_Embeds/1K_train_ff2048.npz \
  --out-dir runs/mpsae_repair/splits
~~~

With `--config`, the script uses the configured pool sizes and authenticates the declared training archive before creating indices. It hashes the archive, reads its feature-array header, and loads labels without loading the full feature matrix. Without `--config`, the standalone utility still supports synthetic data; those splits alone do not certify release provenance.

It makes a per-class deterministic permutation seeded from the archive fingerprint and a fixed protocol namespace. Labels are used for class-balanced allocation and evaluation, never as SAE or neighborhood targets. This is label-assisted data selection, so do not describe the entire pilot as annotation-free. A strict label-free claim would require a separately declared hash-only split and uniformly sampled landmarks.

| Pool | Count | Use |
|---|---:|---|
| `fit_indices` | 128,000: 128/class | All optimizer minibatches |
| `gallery_indices` | 128,000 | Identical to fitting rows; exact 1-NN gallery |
| `dev_cheap_indices` | 2,000: 2/class | Pilot endpoint comparisons and limited predeclared calibration |
| `dev_holdout_indices` | 8,000: 8/class | Open once after locking the candidate |
| `dev_full_indices` | 10,000 | Union of cheap and held-out development queries |
| `anchor_indices` | 1,024 | One/class plus 24 further training rows; reference bank |

Fitting, full development, and anchor pools are disjoint. The pilot uses 139,024 distinct source rows. Every arm has access to the same declared pools; only the relational arms use the auxiliary landmark information.

The feature indices are row IDs in the hashed archive. They are **not** a recovered mapping to original image filenames. These are held-out rows for compressor fitting, conditional on the frozen pretrained parent; they are not a claim of being unseen during parent pretraining.

The tool records provenance warnings for nonreference row counts or dimensions so it can be tested on synthetic data. The training adapter must enforce the actual FF2048 fingerprint/provenance requirements; a successful split command alone is not a reproduction check.

### 3.2 Evaluation always uses the same gallery

Encode all 128,000 fitting rows as the pilot gallery and the specified development rows as queries. Exclude landmark rows and development rows from the gallery.

Use final clean inference for both sides. A repaired query against an unrepaired gallery would be a different, asymmetric retrieval experiment.

Measure the full frozen parent on this same small gallery. Do not substitute a published full-gallery score for this local reference.

### 3.3 The held-out gate is only opened once

Use the cheap 2,000 queries for triage, attribution, and any scheduled auxiliary comparison. Lock the candidate, all hyperparameters, and evaluation code before using the other 8,000.

The full 10,000-query result is useful for reporting but includes the cheap selection set. Report the 8,000 newly opened queries separately.

If you alter the method after viewing that gate, those queries become development data. Do not continue calling them independent confirmation.

## 4. Exact pilot arm matrix

The table is the complete arm definition. `Old` means the original angular cost. `Mixed` means 0.8 C0 + 0.2 Cnbr inside one partial-OT solve.

| ID | Repair K and 2K | Active-code erasure | Direct KL | OT cost | Raw auxiliary budget | Purpose |
|---|---|---|---|---|---|---|
| W | No | No | 0 | Old | 4K | Shared 250-update warm start |
| C | No | No | 0 | Old | 4K | Equal-extra-training control |
| R | One step | No | 0 | Old | 4K | Repair effect |
| M | One step | Yes | 0 | Old | 4K | Masking effect |
| O | One step | Yes | 1 | Old | 4K | Direct relational-loss effect |
| S | One step | Yes | 1 | Off | 4K | Simpler relational continuation |
| N | One step | Yes | 1 | Mixed | 4K | Proposed new cost |
| F | One step | Yes | 1 | Mixed | Fixed B0 | Fixed-wide control |
| A | One step | Yes | 1 | Mixed | B0→4K | Auxiliary curriculum |

All continuation arms receive **250 additional updates** from W. All learned-repair arms have the same initial repair parameters.

For K=8:

- Main output remains 8.
- Second student view remains 16.
- Third original OT view remains raw 32.
- Normal auxiliary reconstruction uses raw 32.
- F uses raw 64.
- A uses raw 64→32 over the first 175 branch updates, then 32.

The wide reconstruction term has weight 1/8 and replaces the existing raw-4K term. It is never accidentally added twice.

### Interpretation of the contrasts

| Contrast | Question answered |
|---|---|
| R − C | Does learned repair help beyond identical extra training? |
| M − R | Does training-time erasure help the clean deployment path? |
| O − M | Does direct neighborhood distillation help with old OT present? |
| O − S | Does old OT add value after repair and relational distillation? |
| N − O | Does the mixed neighborhood cost improve the old-cost setting? |
| N − S | Does mixed OT add value beyond ordinary relational distillation? |
| F − N | Does a fixed wider reconstruction budget help? |
| A − N and A − F | Is annealing itself useful? |

These are conditional comparisons. For example, M−R measures masking under the old OT objective. It does not prove masks are necessary under the final mixed cost. Before a strong component-contribution claim, use a limited leave-one-component-out test on the final candidate.

### Inherited-OT caveat

W was trained with old OT. Therefore S is **OT-free during continuation**, not a wholly OT-free training procedure. The later from-scratch S arm is necessary to make that stronger comparison.

A branch that drops OT must not reset the SAE optimizer while an OT branch inherits it.

## 5. Checkpointing, initialization, and reproducible branching

The attachment says the current ImageNet SAE workflow saves histories and metrics without resumable model/optimizer checkpoints. Add checkpoint support before launching W; otherwise these branch comparisons cannot be reproduced faithfully.

Save:

- SAE dictionary, both biases, and dead-feature/activity buffers.
- Adam state by parameter name, including steps and moments.
- Branch recipe, source hash, split hash, and parent fingerprint.
- Epoch/minibatch position and exact data-order state.
- Python, NumPy, CPU Torch, and CUDA RNG states where used.
- Independent random streams for data order, masks, repair initialization, and diagnostic sampling.
- Current bank codes and refresh position, or sufficient state to regenerate them exactly.
- Repair state where present and fixed α.

Warm start uses data epochs 0 and 1. Every continuation consumes epochs 2 and 3 with the same permutations. Extra random draws for masks must not perturb the subsequent data order.

Restore Adam moments for inherited SAE parameters. Initialize moments for new U/V parameters to zero, using the same optimizer hyperparameters. Load by parameter identity/name; do not assume a changed parameter-list ordering will produce a valid optimizer restore.

Activity counters must be side-effect controlled. The proposed branches use the same raw K/4K/2K bookkeeping updates once per minibatch. Bank encoding, repair, masks, diagnostic passes, and the scheduled wide branch do not advance them. Log actual final-code activity separately.

Save every 25 branch updates and at branch endpoints. These intermediate checkpoints are for recovery and diagnosis; the pilot evaluates the predeclared final endpoint, not the best checkpoint selected on development accuracy.

## 6. Reference-bank and loss settings

| Field | Initial choice | Why it is fixed in the first test |
|---|---|---|
| Teacher | Frozen FF2048 input feature | No extra backbone extraction; same-parent signal |
| Landmark count | 1024 | Modest memory and useful support coverage at K=8 |
| Landmark membership | Fixed disjoint training pool | Common profile support, no self-matching artifact |
| Key codes | Clean repaired K and 2K | Matches actual student metric |
| Teacher temperature | 0.1 | Explicit starting value |
| Student temperature | 0.1 | Explicit starting value |
| Key refresh | Before update 0 and every 25 updates | Limits drift without a new EMA model |
| Direct KL coefficient | 1 | Same in O/S/N/F/A |
| Mixture weight | 0.2 | One new-cost setting, no broad sweep |
| Mask clean probability | 0.2 per row | Trains some exact deployment-path examples |
| Mask erase probability | 0.25 per active coefficient | Moderate corruption |
| Repair depth / rank | 1 / 32 | Small initial encoding overhead |
| Repair low-rank initialization | U random, V zero | Starts at fixed IHT |
| Transport mass normalization | Off | Preserves existing objective scale |

Precompute or cache frozen teacher logits/profiles if convenient; their cache identity includes teacher, landmark ordering, temperature, and exclusion rule. For 128,000 rows × 1,024 float32 values, a full teacher-profile cache is about 500 MiB. It is optional and must be counted as training storage.

Compute student reference similarities using indexed sparse values in the production adapter. The supplied dense reference is intended for correctness checks.

Do not run banks, teacher profiles, or OT at final embedding inference.

## 7. Stage 0: cheap correctness and signal checks

### 7.1 Run the supplied mathematical checks

~~~bash
python check_reference.py
~~~

They verify support constraints, deterministic nested gates, masking, fixed-IHT behavior, profile detachment, affinity properties, gradient starvation, entropy bookkeeping, and the detached-plan gradient. They do not validate the existing OT solver or train a useful representation.

### 7.2 Check the actual adapter on a tiny real training slice

Use a fixed training-only diagnostic minibatch and a few real feature rows. Verify:

1. No nonfinite values or zero final codes.
2. Served output has at most K positive values with valid dictionary indices.
3. With repair/masks/KL/new cost disabled, the adapter matches the original MPSAE forward loss and gradients within floating-point tolerance.
4. The declared protected long representation is unchanged. With precomputed features this checks the cached full outputs and provenance. If serving the image encoder later, verify its checkpoint, preprocessing, and inference path too. This does not assert preservation of a trainable largest sparse code.
5. All losses use final projected codes where specified.
6. Teacher and student key-bank tensors receive no gradients.
7. Bank IDs and ordering match across teacher/K/2K.
8. Extra forward passes leave legacy activity counters unchanged.
9. OT plan mass, marginal capacities, and unused-capacity entropy use the correct convention.
10. The cost gradient is sJ−P_detached, with no hidden mass or element-count rescaling.

For a **tiny tightly converged** OT problem, also compare finite differences of the full gap to the envelope gradient. This tests the actual solver integration. The supplied fake-plan test cannot establish solver optimality.

### 7.3 Do not misclassify a missing gradient as a tested hypothesis

On the fixed diagnostic batch, log selected-code gradients from reconstruction, direct KL, old-cost contribution, and new-cost contribution separately.

For a mixed-cost arm, solve once for P_mix and hold Δ=sJ−P_mix fixed for both component diagnostics. Differentiate λ_OT(1−ω)〈Δ,C0〉 and λ_OTω〈Δ,Cnbr〉 with respect to the same declared tensors or parameters. Their gradients must sum to the mixed-gap gradient within numerical tolerance. Do not solve two separate OT problems for this diagnostic. If the common comparison space includes the raw 4K view, retain the neighborhood component's zero direct derivative for that view.

If almost every query lacks landmark overlap, or the weighted new-cost gradient is below roughly 1% of the old-cost contribution, first resolve whether the implementation/temperature makes the new term ineffective. This threshold is a diagnostic heuristic, not a theoretical success condition.

If one limited calibration is needed, choose between τ_S=0.1 and 0.05 on training diagnostics and the cheap development set, then lock it across O/S/N/F/A as applicable. Do not perform an unreported large sweep. Keep the original setting and its diagnostic evidence in the record.

K=2 or K=4 are later stress tests. No silent straight-through estimator or dense metric substitution is allowed to rescue the first result.

## 8. Stage 1: triage and attribution

### 8.1 Triage C/S/N

Train W once and branch C/S/N. Evaluate each endpoint on the same 2,000 cheap queries.

Suggested compute-conscious decision policy, configurable **before** evaluation:

- Continue if S or N exceeds C by at least **0.3 percentage points**, with valid outputs and acceptable measured encoding cost.
- If differences are smaller, call the result inconclusive and favor the simpler method unless there is a specific failed diagnostic worth resolving.
- An optional deployment budget is at most 3× the baseline SAE-only encoding time. Record hardware and batch size; this is a practical policy, not a scientific constant.
- Stop a numerically invalid branch and retain its failure record. Do not silently drop failing examples.

At 2,000 queries, 0.3 points is only six extra correct predictions, so it is a **screening** threshold, not convincing evidence by itself.

If S beats N, the simpler relational method remains a legitimate result. Do not retain OT merely because it is the original project idea.

### 8.2 Attribution R/M/O

If triage is promising, run R/M/O from W, keeping every fixed setting identical.

These runs explain whether improvement came from extra sparse inference, training masks, direct teacher supervision, or the transport cost. Report all arms, including negative and null results.

An evaluation-only fixed-IHT check on C is also useful: encode C's initial codes with one nonlearned projected residual step and compare retrieval and reconstruction. This adds no training fit. It does not replace a matched learned-repair control and should be labeled separately.

### 8.3 Limit expensive diagnostics

Record cheap scalar losses every update. Save detailed gradient diagnostics only before training, near the midpoint, and at the endpoint. Full gallery re-encoding and search occur at predeclared endpoints, not every few updates.

Run the precision/timing benchmark only on C, S, and the selected candidate. Screening every method does not require the supplied five-repeat timing workflow.

## 9. Stage 2: wide budget and optional falsification controls

Only run F/A if the N family merits further work. Both still start from W and receive 250 updates.

The final approximately 30% of A uses B=4K, restoring the baseline auxiliary objective. Keep masks, temperatures, key refresh, repair depth, solver settings, and training duration identical.

If A beats N but not F, retain F if its practical costs are acceptable. The result does not support an annealing-specific claim.

Before attributing a gain to the neighborhood structure itself, the most useful optional controls are:

| Control | Purpose | When to spend compute |
|---|---|---|
| Old cost with contrast scale matched to mixed cost | Distinguish semantic structure from changing effective cost/entropy scale | If N clearly beats O |
| Candidate without masking | Test whether masking is necessary in the winning objective | If the final method includes masks |
| Candidate without direct KL | Test whether both relational mechanisms are required | If N wins and simplicity matters |
| Parameter/compute-matched feed-forward re-encoder | Test whether residual structure beats simply adding encoding capacity | Before a strong architectural novelty claim |

For the cost-scale control, estimate a single scalar from fixed training batches before continuation: standard deviation of mixed-cost entries divided by that of old-cost entries, with a documented nonzero floor. Multiply the old cost by that fixed scalar; keep λ_OT and η unchanged. Do not fit the scalar to final validation accuracy.

These controls are not part of the default workload. Add only the ones required by the claim you intend to make. A full factorial grid is unnecessary at this stage.

## 10. Stage 3: one-time development gate

Lock the candidate and evaluate the 8,000 previously unopened queries.

Recommended promotion rule:

- Candidate-minus-C point improvement at least 0.3 percentage points.
- Paired 95% query-bootstrap interval has lower endpoint above zero.
- Protected long-output checks pass.
- Actual support, gradient, solver, encoding-time, and storage diagnostics are acceptable.

For a claim that mixed OT is useful, also report candidate-minus-S under the same protocol. A candidate can beat C yet fail to improve S; that favors the simpler objective.

Failure to satisfy the rule means “not enough evidence for the next compute spend,” not a proof that the idea can never work. Do not repeatedly tune against the 8,000-query gate.

Retain per-query labels and predictions. The supplied helper computes paired reports:

~~~bash
python compare_predictions.py \
  --baseline runs/mpsae_repair/C/dev_holdout/predictions.csv \
  --candidate runs/mpsae_repair/N/dev_holdout/predictions.csv \
  --out runs/mpsae_repair/N_vs_C_holdout.json
~~~

Each CSV has `seed,query_id,label,prediction`. IDs and labels must match exactly between methods. The helper resamples within class, retaining paired method predictions and all supplied seeds together.

These confidence intervals measure query-sampling uncertainty **conditional on the fitted models**. They do not establish variation over training seeds or correct for the exploratory cheap-set search.

## 11. Stage 4: from-scratch full-data confirmation

### 11.1 Minimal confirmation grid

Use K=8 and seeds 1227 and 1228. Compare the unique set:

~~~text
original MPSAE C
relation-only S
selected candidate
~~~

If S is selected, this is two methods × two seeds = **4 fits**. Otherwise it is three methods × two seeds = **6 fits**.

Initialize each from scratch with a matched SAE initialization for a given seed. S receives no OT at any point. The full candidate receives its specified method from the start. Do not import pilot optimizer states, codebooks, repair weights, or learned key banks.

For each confirmation seed, compute α from its freshly initialized dictionary using the same deterministic norm-estimation procedure. Share α and the initial repair parameters across repair-enabled methods for that seed. Keep α fixed during fitting and record it; do not import the pilot's value or tune it on official validation.

Use the attached FF2048 full-training recipe: 1,281,167 rows, batch 4096, retained last batch, ten epochs, 3,130 updates per fit. The final batch has 3,215 examples.

| Unique methods | Full fits | Optimizer updates | Training-example presentations |
|---|---:|---:|---:|
| C and S | 4 | 12,520 | 51,246,680 |
| C, S, and another candidate | 6 | 18,780 | 76,870,020 |

This is an optional larger budget after the cheap gate, not part of the 1,000-update triage.

Use a third paired seed before making a stronger empirical claim, especially for close results. A second budget K=16 is a further gated expansion. Do not fill the full nine-budget curve by default.

### 11.2 Full-data bank exclusion differs from the pilot

Full-data fitting includes all training rows, including landmark rows. Keep the same fixed training-only landmark pool, but remove the union of **current minibatch IDs** from every profile in that minibatch, then renormalize teacher and student probabilities over that shared support.

Use this same shared exclusion for direct KL and all OT groups within the minibatch. It prevents self targets without creating row-dependent profile supports. If no anchors remain, fail explicitly or use a predeclared resampling rule; never silently use inconsistent columns.

This policy allows the stated full-training count while preventing self-matching. It is an explicit protocol difference from the disjoint pilot, not an unnoticed implementation change.

### 11.3 Official validation is final evaluation

After method selection, final fitting can use the entire official training split, including the pilot's development rows. Official validation remains reserved for final reporting.

Evaluate the final fixed-epoch checkpoint, not the best official-validation checkpoint. If the user's existing “carefully tuned MRL” was tuned on official validation, record that exposure; do not describe the resulting system comparison as a fresh unbiased holdout.

## 12. A fair comparison with strong MRL

Use the user's best available local MRL checkpoint as an evaluation-only baseline. Record the checkpoint hash, architecture, preprocessing, training/tuning provenance, and available prefix dimensions. No new backbone training is part of the default plan.

Two issues must be explicit.

**Parent provenance:** the supplied FF2048 full parent differs from the separately trained native MRL model. Sparse-method comparisons can control the parent; absolute MRL-versus-sparse comparisons are system comparisons. Evaluate and report each method's own full parent where possible.

**Row alignment:** the supplied NPZ schema has no canonical image IDs. You cannot pair independently extracted MRL rows with FF2048 rows by class labels or guessed order. Require a verified row mapping or already aligned cache for the small pilot comparison. If unavailable, mark local paired MRL comparison unavailable; do not replace it with the published curve and imply a local measurement.

For complete official partitions, aggregate scores can be compared with validated gallery/query membership even when row order differs. Paired per-image uncertainty still requires verified query identities.

### Equal budget and equal bytes

Report two views:

| Sparse budget | Indexed FP32 payload in supplied transfer format | Same-active-count dense prefix | Near-byte-matched canonical dense prefix | Exact-payload prefix |
|---|---:|---:|---:|---:|
| K=8 | 68 bytes | m=8, 32 bytes | m=16, 64 bytes | m=17, 68 bytes |
| K=16 | 132 bytes | m=16, 64 bytes | m=32, 128 bytes | m=33, 132 bytes |

A noncanonical MRL prefix such as 17 may not have been explicitly included in its training objective; label that fact. Measure actual retrieval-index bytes independently, because posting offsets and other structures change the accounting.

Do not claim a 13-bit coordinate representation merely because h=8192: the existing code stores larger integer types.

## 13. Retrieval and timing

Accuracy remains exact normalized 1-NN top-1, with every gallery row eligible, including sparse rows that share no active coordinate with a query. Reject zero-norm final codes before normalization.

Reuse the documented production backends: CUDA FAISS IndexFlatL2 for dense features and the indexed sparse CUDA search for sparse codes. Tiny CPU checks are not performance measurements.

For finalist timing, retain:

- Dense and sparse query batch 32.
- Sparse gallery chunk limit 65,536, subject to existing entry/workspace caps.
- Sparse score workspace 256 MiB.
- FAISS temporary workspace 512 MiB.
- Three warmups per batch shape and five timed repeats.
- Both CUDA-event and synchronized wall measurements.

The current timing is repeated warm search over resident query batches. It excludes backbone extraction, SAE/repair encoding, normalization, host/device input transfer, and index construction. Report all of those separately where relevant.

For the new method, add:

- Initial SAE encoding time.
- Correction time and total sparse encoding time.
- Gallery encoding/build time.
- Actual index bytes and posting-list workload after repair.
- Optional image-to-retrieval timing only when the full image path is actually measured.

Retain the existing naming distinction: amortized time per query at a given batch size is not isolated single-query latency.

## 14. Transfer only after a positive confirmation

The attached default transfer suite contains 570 final head fits and 96 reusable compressor fits. Do not launch it for this pilot.

Start with one task: the conditional VOC2012 object-mask task, one architecture, K=8, and the primary width-128 MLP readout. Compare the parent, original compressor, and selected compressor with paired seeds. Extend to the relation-only comparator if necessary to support an OT claim.

There is an important data-interface constraint: the existing transfer suite uses timm parents and a separate float32 extraction recipe. A trained FF2048 compressor must not be applied to timm features solely because both have dimension 2048.

Use one of these explicitly budgeted routes:

1. Load and verify the exact matching FF2048 parent, then extract the prescribed region crops. Label the result a reduced research transfer profile.
2. Refit the comparison on the existing transfer parent and its ImageNet cache, accounting for new extraction and compressor training.

If neither is available, defer transfer.

Retain the source experiment's object-box disclosure, crop geometry, valid-pixel treatment, class-balanced sampling, input-scale estimation from fitting data only, 28×28 output and decoding, annotation-only priors, and image-cluster uncertainty. The primary metric remains clean-crop macro object IoU, not semantic-segmentation mIoU or mask AP.

The reduced transfer profile needs an explicit adapter. Existing schedule selection uses all four full parents; do not silently run that entire development grid. Reuse an already locked compatible schedule, or predeclare a reduced parent-only selection procedure using development data and disclose the change.

## 15. Required measurements and output schema

### 15.1 Per-update scalar log

Write `training.jsonl` with:

~~~text
run_id, seed, global_step, branch_step, epoch, wall_seconds
K, auxiliary_budget, mask_erase_fraction, forced_clean_row_fraction, unchanged_start_fraction
loss_total, mse_final_K, mse_raw_aux, loss_dead
loss_rel_unweighted, gap_value, weighted_rel, weighted_ot
cost0_mean, costnbr_mean, costmix_std
solver_mass_error, solver_capacity_violation, solver_iterations
actual_nnz_mean, zero_final_codes, optimizer_lr
~~~

Only compute detailed cost diagnostics for enabled components. “Loss component impact” here means logged magnitudes, not causal effects; the ablations supply causal comparisons.

### 15.2 Endpoint and diagnostic outputs

| Group | Required fields |
|---|---|
| Accuracy | Top-1 %, correct/total, gallery/query IDs, predicted label and neighbor ID |
| Geometry | Teacher KL, teacher-neighbor preservation, profile entropy, diagonal/shuffled affinity |
| Sparsity | Nonzero counts, active-unit coverage, dead-feature rate, support-change rate |
| Semantic gradient | Zero-overlap fraction, near-zero selected-value gradient fraction, component norms |
| Keys | Refresh count/time, old/new key cosine and support Jaccard |
| OT | Mass/capacity residuals, negative estimated gaps, diagonal plan mass, per-example transported marginals |
| Cost scale | Cost standard deviations and ratios to η; weighted old/new gradient contributions |
| Protection | Protected-output identity, checksum/state check, max absolute output difference |
| Cost | Fit time, encoding time, correction time, search time, actual bytes, parameter counts |
| Provenance | Source/parent/split/config/code hashes, package/device versions, RNG seeds |

Save per-query predictions so uncertainty does not require another search run.

### 15.3 Suggested output layout

~~~text
runs/mpsae_repair/
  splits/split_indices.npz
  splits/split_manifest.json
  resolved_configuration.json
  W/checkpoint.pt
  C/ ... R/ ... M/ ... O/ ... S/ ... N/ ... F/ ... A/
  decisions.json
  comparison_cheap.csv
  comparison_holdout.csv
  full_confirmation/
~~~

Each arm contains `config.json`, `training.jsonl`, `last.pt`, endpoint metrics, prediction CSVs, diagnostics, and a failure record if needed.

`decisions.json` records which stages were run, stopped, or skipped; thresholds used; candidate selection; calibration; and when the held-out set was opened. Missing arms are not silently treated as failures or omitted successes.

## 16. Repository integration map

These contracts are implemented by the separate repair adapter where described below. Their runtime behavior remains unvalidated in this review.

| Existing component named in the attachment | Proposed change |
|---|---|
| `mpsae_repair_experiment.py` / `repair_runtime.py` | Separate research adapter loads explicit splits, saves/restores state, and runs one resolved arm |
| SAE definition / representation module | Expose preactivations, pure budgeted encoding, counter control, and one-step repaired encoding |
| Existing MMPOT objective | Accept an externally constructed cost while preserving capacities, entropy, group reduction, and envelope gradient |
| `sparse_gpu_retrieval.py` | Reuse final index/value codes; verify repaired gallery/query compatibility |
| `retrieval_timing.py` | Reuse search timing; add separate repair/encoding measurement outside it |
| Transfer modules | Add a gated reduced profile only after parent compatibility is established |
| Existing aggregators | Keep proposed research outputs separate from strict paper aggregates |

Pseudocode for one branch:

~~~python
restore_common_checkpoint_and_optimizer(W)
initialize_same_repair_if_needed()
for step, batch in shared_continuation_batches:
    # Preserve original raw-view counter/mask timing, including AuxK use.
    raw, dead_feature_term = encode_raw_views_with_legacy_bookkeeping(
        batch.x, K, 2*K, 4*K, auxiliary_budget(step)
    )
    small, medium = maybe_mask_and_repair(raw.K, raw.twoK, batch.x)
    reconstruction = mse(decode(small), batch.x) \
        + 0.125 * mse(decode(raw.auxiliary), batch.x) \
        + dead_feature_term / 32

    # Only relational arms need these profiles or key-bank work.
    profiles = build_profiles_if_enabled(
        teacher=batch.x, small=small, medium=medium,
        detached_clean_bank=bank, common_allowed_ids=allowed_ids
    )
    loss = reconstruction + relation_weight * relational_kl(profiles)
    if ot_enabled:
        for group in ot_groups:
            cost = build_selected_cost(small, medium, raw.fourK, profiles, group)
            plan = existing_partial_solver(cost.detach())
            loss += correctly_reduced_matching_gap(cost, plan)
    backward_and_step(loss)
    refresh_bank_if_due_without_counter_side_effects()
    save_and_log_if_due()
~~~

The real implementation must skip disabled terms rather than evaluate missing profiles and multiply by zero. Group weights must reproduce the existing mean reduction, not sum another full OT loss for every group.

`mpsae_repair.py` implements the research loss and validated detached-plan adapter; `mpsae_repair_experiment.py --config` consumes this schema. The historical strict runner and manual MMRSAE CLI use their own settings interfaces. Unsupported changes to fixed research controls fail configuration validation.

## 17. How to decide what survives

Keep a component when its matched contrast provides a useful accuracy gain for its measured cost, and the effect survives the locked gate or later confirmation.

If masks improve masked reconstruction but not clean retrieval, remove them. If fixed-wide training matches annealing, keep the simpler wide setting. If N does not improve S, describe the result as a relational sparse compressor rather than claiming OT is essential. If strong MRL wins at matched bytes, record that outcome.

The practical deliverable of this experiment is a small number of defensible decisions, including negative ones. It is not a requirement that every ingredient survive.

