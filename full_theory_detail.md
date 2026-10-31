# MPSAE with sparse repair and teacher neighborhoods

**Research specification and implementation guide — 29 October 2026**

**Status:** this document specifies a proposed extension and controlled experiment. This checkout implements it through `mpsae_repair.py`, `mpsae_repair_experiment.py`, and `run_mpsae_repair.sh`, with `EXPERIMENT_SCOPE=repair` for Docker. The current review used source inspection only; it does not establish runtime success or report ImageNet results. The separate default `mmrsae` workflow performs manual fits with these components, rather than the staged control study below.

## 1. The proposal in plain language

A tiny sparse embedding can fail for two different reasons: its first choice of active features may be poor, and the geometry learned by those features may not preserve useful neighbors.

The proposed method addresses both:

1. **Repair the first sparse code once.** Decode it, inspect the remaining reconstruction error, and learn a correction before projecting back to the same K-sparse budget. During training, sometimes erase part of the initial support so the correction learns to revise a poor starting code.
2. **Teach the final code which examples should be nearby.** A frozen full representation provides a probability distribution over a small bank of reference examples. The final sparse code should preserve that distribution. Use it both as a simple distillation control and as information in a new three-way transport cost.
3. **Temporarily reconstruct with more active features.** Keep the actual deployed budget K fixed. Change only the budget of an auxiliary reconstruction branch, and compare that schedule against an equally weighted fixed-wide branch.

The protected longest representation is returned through its original frozen inference path. In the default experiment that path is the original dense parent. A separate small SAE and repair module are trained. If the actual protected endpoint is an existing largest sparse code or an MRL output, retain that exact model, preprocessing, and output route instead. Freezing its dense input alone does not preserve a trainable largest sparse code.

**The first research question is whether the new transport term adds anything beyond repair plus ordinary relational distillation.** The method is allowed to simplify if the answer is no.

### What is fixed, learned, and discarded?

| Object | Training | Deployment |
|---|---|---|
| Original full parent encoder / full-feature cache | Frozen | Original long representation remains available |
| Small SAE dictionary and biases | Trainable, initialized from a common checkpoint in the pilot | Used to form the initial sparse code |
| Low-rank correction module | Trainable | One correction step for each requested small code |
| Teacher reference features | Frozen, training data only | Not needed |
| Student reference codes | Re-encoded without gradients | Not needed |
| Artificial masks | Applied only to student starting codes | Disabled |
| Partial-OT solver and profile losses | Training only | Not needed |
| Auxiliary wide reconstruction | Training only | Not needed |
| Stored compact embedding | Final projected code | At most K positive values, with their coordinate IDs |

### What this proposal does not guarantee

An unchanged frozen full-output path can guarantee preservation of that representation. Hard nonnegative Top-K can guarantee an upper bound on active coordinates. Neither fact guarantees improved small-code retrieval accuracy.

The repaired codes at different budgets need not have nested supports or equal coefficients. Preserving a separate full branch also costs model storage and differs from having one mutable representation whose every prefix is preserved. These distinctions must appear in any eventual paper.

## 2. Package contents and immediate use

| File | Purpose |
|---|---|
| `full_theory_detail.md` | Method, equations, limitations, and component-level hypotheses |
| `full_experiment_detail.md` | Detailed staged experiment, decisions, reporting, and repository integration |
| `experiments.json` | Machine-readable proposed settings and arm definitions |
| `run_matrix.csv` | Human-readable matrix generated from the configuration |
| `make_plan.py` | Standard-library planner; prints costs and can export individual run specifications |
| `prepare_splits.py` | Creates deterministic fitting, development, and landmark indices from the training NPZ |
| `compare_predictions.py` | Computes paired accuracy differences and conditional query-bootstrap intervals |
| `mpsae_repair.py` | Reference kernels and research loss, including the detached-plan gradient contract |
| `check_reference.py` | Tiny CPU mathematical checks; no dataset or GPU required |
| `mpsae_repair_experiment.py` / `run_mpsae_repair.sh` | Configured ImageNet stages and resumable fitting |
| `VALIDATION.md` | What was actually checked while preparing this package |

From the extracted directory:

~~~bash
python make_plan.py
python check_reference.py
python prepare_splits.py --config experiments.json \
  --train-npz /path/to/FF2048_RN50_Embeds/1K_train_ff2048.npz \
  --out-dir runs/mpsae_repair/splits
~~~

The planner does not launch training. The split tool needs NumPy; the reference checks need PyTorch. The research runner accepts `experiments.json`; set its trusted source revision and training hash before real-data preparation. A single `ENV_FILE=repair.env bash run_all_experiments_docker.sh pilot` invocation runs preflight, configured preparation, checks and W/C/S/N after copying/editing `repair.env.example`. Later stages remain explicit. See [the launch instructions](README.md) for paths and selection gates.

The source attachment is “Experiment details and benchmark reference: MRL, CSR, and MPSAE,” uploaded as `Pasted markdown(3).md`. Its SHA-256 is:

~~~text
54736a8b4e51ee86a83c6d5f824a2ccda4f2376708633976803f83b57610a9df
~~~

The original proposal used that attachment. Current integration and launch instructions above reflect a source review of this checkout; runtime validation is recorded separately in `VALIDATION.md`.

## 3. Baseline to preserve as a control

Use row-vector notation. The parent feature is x in R^d, the tied dictionary is D in R^(h×d), and h=8192.

$$
a=(x-b_{\mathrm{pre}})D^\top+b_{\mathrm{enc}},
\qquad
z_k^0=H_k^+(a),
\qquad
\hat x_k=z_k^0D+b_{\mathrm{pre}}.
$$

Here H_k^+ retains the k largest signed preactivations, then sets negative retained values to zero. It is the Euclidean projection onto nonnegative vectors with **at most** k nonzeros. Ranking by absolute magnitude would implement a different operator.

The supplied baseline forms K, 2K, and 4K codes from the same preactivations. Its reconstruction objective is

$$
R_{\mathrm{base}}
=
\operatorname{MSE}(x,z_K^0D+b)
+\frac18\operatorname{MSE}(x,z_{4K}^0D+b)
+\frac1{32}L_{\mathrm{dead}}.
$$

`MSE` means the mean over both samples and parent-feature coordinates. The existing dead-feature AuxK term uses a different normalized residual convention; preserve it and log it separately.

MPSAE adds a partial matching gap with coefficient 2. The supplied controls are mass s=0.9, entropy η=0.2, OT group size n=32, maximum 100 iterations, and constraint tolerance 1e-4. These are the user's research settings, not a claim about the defaults of the M3G paper.

The current experiment uses separate fixed-K models in its strict FF2048 and RN18 studies. A collection of such models must not be described as a single model serving every budget.

## 4. Component A: one learned sparse repair step

### 4.1 Corrupt the starting code, not the target

For a training example and budget k, first form z_k^0. With probability 0.2, leave the whole code clean. Otherwise independently erase each positive active coefficient with probability 0.25:

$$
z_k^- = m\odot z_k^0.
$$

There is no inverse-dropout rescaling. Inactive entries remain zero. Bernoulli erasure means that the realized erased fraction varies; a nominally corrupted row can retain everything by chance. A fully erased starting code is allowed because the residual still supplies the parent feature.

The unconditional expected erased fraction among positive active coefficients is 0.8 × 0.25 = 0.20. For a row with s positive coefficients, the probability that its starting code is actually unchanged is 0.2 + 0.8(0.75)^s, approximately 0.2801 when s=8. Log forced-clean decisions separately from realized unchanged starts and erased coefficients.

The teacher and reference keys stay clean. Use independent mask draws for K and 2K, but the same reproducible mask stream across matched experiment arms.

### 4.2 Observe the residual and correct

$$
r_k=x-b-z_k^-D,
$$

$$
\boxed{
z_k^+
=
H_k^+\!\left(
z_k^-+\alpha r_kD^\top+g_\phi(r_k)
\right).
}
$$

The first correction term is a tied-dictionary projected-gradient step. The learned term is a small low-rank nonlinear correction:

$$
g_\phi(r)=\operatorname{ReLU}(rU)V,
\quad
U\in\mathbb R^{d\times r_0},
\quad
V\in\mathbb R^{r_0\times h}.
$$

Start with rank r_0=32, no biases, random Kaiming initialization for U, and V=0. Initializing both factors to zero would kill learning in the factorization.

This adds r_0(d+h) parameters: 327,680 at d=2048 and 278,528 at d=512. The extra parameters are shared across examples and both repaired training budgets. They are separate from bytes stored per embedding.

The default has one correction step. Increasing depth is a later compute/accuracy experiment.

### 4.3 Choose the step size explicitly

For the fixed update g=0 and a fixed dictionary, a sufficient condition for nonincreasing squared reconstruction error is

$$
0<\alpha\le \|D\|_2^{-2}.
$$

For the tiny reference checks, the code computes this norm exactly. A full SVD on the production dictionary is unnecessary for every training update.

For the pilot, estimate the spectral norm of the warm-start dictionary once using a deterministic power iteration, initialize α=0.9/estimated_norm², and record the value. For from-scratch confirmation, use that seed's newly initialized dictionary instead; share the initial α across its repair-enabled methods. This estimate is a scale-setting heuristic; it is not a certified upper bound. Before treating a fixed-IHT control as a descent method, backtrack α by halves until its measured reconstruction errors on the training diagnostic batch do not increase. A Frobenius-norm bound is certified but can be excessively conservative.

Keep α fixed during each branch comparison. Once the dictionary or learned correction changes, the above fixed-dictionary guarantee no longer establishes descent of the learned model. Evaluate its behavior directly.

### 4.4 Which views are repaired in the cheap pilot?

| Role | Code | Artificial masking? | Reconstruction loss? |
|---|---|---|---|
| Deployed small code | Repaired K | Training only | Main MSE |
| Second student transport view | Repaired 2K | Training only | No new dedicated MSE |
| Third original transport view | Raw 4K | No | Existing auxiliary MSE, except during the explicit budget experiment |
| Landmark keys | Clean repaired K and 2K | No | No gradients through cached keys |

This avoids a third repair projection per training example. Repairing all three views is a separate extension; it is not silently included.

The main reconstruction is evaluated **after the final projection**:

$$
R=
\operatorname{MSE}(x,z_K^+D+b)
+\frac18\operatorname{MSE}(x,z_B^0D+b)
+\frac1{32}L_{\mathrm{dead}}.
$$

In the normal arms B=4K. In the no-repair control, replace z_K^+ with z_K^0.

### 4.5 What the masking idea means

Because

$$
x=b+z_k^-D+r_k,
$$

the repair has access to the full input jointly through the starting code and residual. The intended benefit is better **encoding-time sparse inference**, not recovery of information unavailable to the encoder or reconstruction after only the stored code remains.

At inference, set m=1, calculate the residual from the clean starting code, run one repair step, and store the final sparse code. Both gallery and query embeddings must use this same path.

TDV motivates learning an explicit change conditioned on a current representation [3]. Sparse iterative inference already has substantial prior art [7]. The specific hypothesis here is that erasure-trained, task-supervised repair improves the final fixed-budget embedding.

## 5. Component B: neighborhoods defined by a frozen teacher

### 5.1 A common bank of example identities

Select M=1024 reference examples from ImageNet training data. In the pilot they are disjoint from the fitting and development subsets. Fix their identity and ordering across arms.

For each landmark r, retain:

- Its frozen teacher feature t_r.
- Its clean repaired K student code c_r^K.
- Its clean repaired 2K student code c_r^(2K).

The teacher feature may have a different dimension from the student code. Probability-vector coordinates refer to **example IDs**, not dictionary coordinates, which makes their comparison meaningful.

Default teacher: the same frozen FF2048 feature x used as the reconstruction input. This requires no extra backbone extraction. Using the user's stronger MRL teacher is an explicitly different, aligned-data experiment.

### 5.2 Teacher and student distributions

For teacher feature t_i:

$$
\pi_i(r)
=
\frac{\exp(\cos(t_i,t_r)/\tau_T)}
{\sum_a\exp(\cos(t_i,t_a)/\tau_T)}.
$$

For a final student code at budget k:

$$
p_i^{(k)}(r)
=
\frac{
\exp(\cos(z_{i,k}^{+},\operatorname{sg}(c_r^k))/\tau_S)
}{
\sum_a\exp(\cos(z_{i,k}^{+},\operatorname{sg}(c_a^k))/\tau_S)
}.
$$

Start with τ_T=τ_S=0.1. These are initial engineering settings, not established optimal temperatures. Log distribution entropy and gradients before spending compute on sweeps.

The student cosine is computed in the **actual stored sparse-code space**. A dense projector or decoded-space cosine would be another representation and can improve a loss without improving the intended sparse retrieval metric.

### 5.3 Key refresh and exclusions

Teacher features are fixed. Re-encode the student bank before branch update 0 and after every 25 updates, with masking disabled and no gradients. Refresh again for the final diagnostics.

The small SAE and repair module change, so stale keys are an approximation. At refreshes, record support Jaccard and cosine drift between old and new codes. Momentum keys are an optional future control, not part of the first experiment.

In the full-data confirmation, fitting includes landmark rows. Remove the **union of current minibatch IDs** from the bank for every teacher and student profile, then normalize over the same remaining columns. Use that support for direct KL and every OT group. Per-row or independently chosen per-group exclusions are not used. The disjoint pilot needs no exclusions. If cached full-bank teacher probabilities are used, remove excluded columns and renormalize them too.

### 5.4 Direct relational loss: the essential simple control

$$
L_{\mathrm{rel}}
=
\frac{1}{2n}
\sum_i\left[
\mathrm{KL}(\operatorname{sg}(\pi_i)\Vert p_i^{(K)})
+
\mathrm{KL}(\operatorname{sg}(\pi_i)\Vert p_i^{(2K)})
\right].
$$

Compute this over the whole training minibatch, with the same definition whether OT is on or off. The baseline λ_rel is 1. Use the teacher-to-student direction shown; changing KL direction changes the method.

Similarity-distribution distillation is established in SEED, and relational SSL with cleaner targets is established in ReSSL [4,5]. This loss is a strong control, not a novelty claim.

## 6. Component C: add neighborhood information to the transport cost

### 6.1 Original angular term

Normalize the three code views:

$$
u_i=\operatorname{norm}(z_{i,K}^{+}),\quad
v_j=\operatorname{norm}(z_{j,2K}^{+}),\quad
w_\ell=\operatorname{norm}(z_{\ell,4K}^{0}).
$$

Without repair, these are the original raw views. Define

$$
C^0_{ij\ell}
=
\frac29
\left[
(1-u_i^\top v_j)
+(1-u_i^\top w_\ell)
+(1-v_j^\top w_\ell)
\right].
$$

For nonnegative unit codes, its range is [0,2/3]. The existing [0,1] clamp is therefore inactive except for numerical issues. Keep the original implementation in the control arm.

### 6.2 Proposed additional term

$$
\boxed{
C^{\mathrm{nbr}}_{ij\ell}
=
1-\sum_{r=1}^{M}
\left[
p_i^{(K)}(r)
p_j^{(2K)}(r)
\pi_\ell(r)
\right]^{1/3}.
}
$$

It asks whether two small-code neighborhoods and the full teacher's neighborhood agree.

By Hölder's inequality, the summed affinity is at most 1, so the cost lies in [0,1]. It is zero precisely when the three distributions coincide. The multi-distribution geometric affinity is established mathematics [8]; using it in this particular MPSAE cost is the proposed application.

For example, if the teacher profile is (0.8,0.1,0.1) while both students give (0.1,0.1,0.8), the students agree perfectly with each other but the neighborhood cost is 0.3. Agreement between students alone is insufficient.

### 6.3 One combined cost and one solver call

$$
\boxed{
C^{\mathrm{mix}}
=
(1-\omega)C^0+\omega C^{\mathrm{nbr}},
\qquad \omega=0.2\ \text{by default}.
}
$$

The third axis refers to image ℓ in both terms: its raw 4K code enters C0 and its frozen teacher profile enters Cnbr. No cross-space dot product between the teacher and student is required.

Use one partial-OT solve per group for the combined cost. This is not the sum of two independently solved matching gaps. Do not add a broad final clamp that can suppress combined-cost gradients.

A comparison of ω=0 and ω=0.2 tests the configured mixture. Any different mixture weight is a separately declared experiment.

### 6.4 Partial matching gap and its gradient

For n examples, capacity p_i=1/n and transported mass s, use the same feasible set:

$$
\Pi_s
=
\{P\ge0:
m_r(P)\le p\ (r=1,2,3),\
\sum_{ij\ell}P_{ij\ell}=s\}.
$$

Let q_r(P)=p-m_r(P) denote unused capacity and

$$
H(v)=\sum_a v_a(\log v_a-1),\qquad 0\log0=0.
$$

$$
F_\eta(C,P)
=
\langle C,P\rangle+
\eta\left[H(P)+\sum_{r=1}^{3}H(q_r(P))\right].
$$

Let J_iii=1/n and other entries be zero. Then

$$
G_\eta(C)
=
F_\eta(C,sJ)
-
\min_{P\in\Pi_s}F_\eta(C,P).
$$

For an optimal plan,

$$
\nabla_CG_\eta=sJ-P^\star.
$$

With a detached approximate plan, use this as an approximate envelope gradient. Both cost terms receive gradients through C; the plan, teacher, and cached keys do not.

**Do not detach the entire competing cost.** That would remove the negative P term and leave gradient sJ. The reference helper implements a scalar with the intended gradient and the corresponding entropy-inclusive logged gap value.

Do not silently divide the gap by s, n, or n³. The supplied baseline uses no mass normalization. Group averaging follows the original runner: equally average valid group losses, skipping only a final group of size below two.

A materially negative estimated gap is a solver diagnostic, not something to hide by clamping the training loss to zero. Constraint residuals and optimizer accuracy must be checked. The tiny helper's default plan tolerance is stricter than the existing solver's 1e-4 stopping tolerance; the adapter must validate actual feasibility and tighten or reject inadequate solutions explicitly.

### 6.5 Why not simply add reconstruction to the cost tensor?

A term of the form a_i+b_j+c_l is unary. With balanced fixed marginals, its contribution cancels in a matching gap. With partial marginals it primarily changes which mass is transported; it does not replace an ordinary per-example reconstruction objective.

Keep reconstruction and direct relational calibration outside the matching gap. The neighborhood term depends jointly on the two trainable student distributions and the teacher distribution.

### 6.6 What this addresses, and what it does not

The new term supplies external neighborhood structure. It does not establish that M3G was incapable of using such a cost: that framework already allows different multiway costs [1].

It also does not remove all false-negative pressure. Strict identity matching still distinguishes different images, including images that the teacher regards as semantically similar. Monitor whether improvements in the transport gap correspond to useful neighborhoods and actual retrieval.

A useful algebraic warning concerns the original unrefined nested codes. For nonzero codes from the same preactivation vector,

$$
\cos(z_K^0,z_{2K}^0)=\|z_K^0\|/\|z_{2K}^0\|.
$$

The smaller normalized code is already the best nonnegative K-sparse unit-vector alignment with its own larger code. With full balanced mass and zero entropy, the identity coupling consequently minimizes all three pairwise terms, making that particular matching gap zero. This is a derivation for the nested construction, not a claim that the current partial, entropy-regularized objective is zero. The actual s=0.9, η=0.2 setting can still supply gradients. It explains why architectural agreement alone is not evidence of preserved semantic neighborhoods. Repair can itself break this nesting, which is another reason to keep the R-versus-C control.

The earlier squared compositional-residual tensor is omitted from the default. Its repair motivation survives in the architecture and reconstruction objective. Adding it as a third cost would obscure whether the neighborhood term helps.

## 7. Component D: auxiliary sparsity annealing

### 7.1 Keep the deployed budget fixed

The primary output always has budget K. The K/2K/4K transport roles remain fixed. Only the budget of the one auxiliary reconstruction term changes:

$$
R_{\mathrm{auxwide}}(t)
=
\frac18\operatorname{MSE}(x,z_{B(t)}^0D+b).
$$

It **replaces** the existing 1/8 raw-4K reconstruction term. It is not another term added on top of it.

For the default pilot:

$$
B_0=\min(h,\max(64,8K)),\qquad B_{\mathrm{end}}=4K,
$$

$$
B(t)=\operatorname{round}
\left[
4K+(B_0-4K)\max(0,1-t/T_a)
\right].
$$

Use T_a=floor(0.7T), with zero-based branch update t and T total branch updates. The final approximately 30% of updates uses 4K. At K=8 this is 64→32, while the served code stays at 8.

**This is a deliberate refinement of the earlier B→K sketch.** Ending at 4K returns to the original auxiliary objective and keeps the raw 4K transport view supported by its reconstruction loss. A B→K experiment is still possible, but it changes the final treatment of that view and must be labeled separately.

The 70% idea is inspired by CSRv2's curriculum [2]. Moving an auxiliary branch while preserving deployed K is our adaptation; it is not a reproduction of the complete CSRv2 method.

### 7.2 Three necessary controls

| Arm | Auxiliary budget | What it tests |
|---|---|---|
| N | Fixed 4K | Proposed neighborhood method without curriculum |
| F | Fixed B0 | More reconstruction capacity without a changing budget |
| A | B0→4K, then hold | Curriculum beyond the fixed-wide alternative |

All three use exactly one auxiliary term with coefficient 1/8. A beats N but not F means the evidence favors wider auxiliary training, not annealing.

Do not describe B0=4K as extra support exploration: the baseline already trains those coordinates. Our K=8 setting starts at 64, above the existing 32-coordinate auxiliary budget.

### 7.3 Two implementation traps

First,

$$
H_K^+(H_B^+(a))=H_K^+(a)\quad (B\ge K).
$$

Computing a wider code and immediately truncating it to K before the auxiliary loss gives no wider-support reconstruction signal.

Second, a reconstruction branch cannot train a frozen encoder/dictionary. The proposed small SAE is trainable; the protected full-output branch is separate. If only the repair module is trainable and z_B does not depend on it, this auxiliary loss has no trainable path.

## 8. Complete objective and training step

For each arm, turn components on or off explicitly:

$$
\boxed{
L=
\operatorname{MSE}(x,z_K^+D+b)
+\frac18\operatorname{MSE}(x,z_{B(t)}^0D+b)
+\frac1{32}L_{\mathrm{dead}}
+\lambda_{\mathrm{rel}}L_{\mathrm{rel}}
+\lambda_{\mathrm{OT}}G_\eta(C).
}
$$

Default new-method weights are λ_rel=1 and λ_OT=2. They are hypotheses to test. In the no-OT arm λ_OT=0; in the baseline control λ_rel=0 and repair is disabled.

A minibatch proceeds as follows:

1. Load frozen parent features and row IDs; no class labels enter the loss.
2. Compute the tied SAE preactivations once.
3. Form raw K, 2K, 4K codes and, if different, the scheduled auxiliary B code.
4. Apply the specified masks and repair only K and 2K.
5. Compute main reconstruction from the final repaired K code and auxiliary reconstruction from the raw B code.
6. Form teacher/student profiles against a detached, correctly ordered bank.
7. Compute direct relational KL over both student views.
8. For each OT group, construct C0 and, where enabled, Cnbr; solve once on the chosen combined cost; apply the detached-plan gradient.
9. Add the existing dead-feature term, backpropagate, and update only the small SAE and repair parameters.
10. Refresh keys only at the scheduled updates. Perform legacy activity-counter updates at the same points relative to code computation and dead-feature-mask construction as in the original MPSAE implementation; they are not an additional post-optimizer update.

The baseline calls that update dead-feature counters can have hidden side effects. Exact call order and the point at which the dead-feature mask is read must match the baseline control, not just the number of counter increments. Include counters near the dead-feature threshold in the disabled-components equivalence check. For all proposed branches, preserve the MPSAE control's three raw-view updates—K, 4K, and 2K—once per minibatch. Bank encoding, diagnostic passes, masks, repair, and scheduled B evaluation must not increment those legacy counters. Separately log actual activity of the served code. This prevents extra forward calls from silently changing the dead-feature objective.

The attached CSR and MPSAE recipes have different historical counter behavior. Preserve strict baseline reproductions separately; do not silently claim a counter-harmonized CSR is identical to the supplied CSR implementation.

## 9. Established SSL mechanisms used here

| Mechanism | Source | Application here | Initial status |
|---|---|---|---|
| Teacher similarity distributions | SEED [4] | Direct KL and teacher-side neighborhood information | Included |
| Cleaner target and relational consistency | ReSSL [5] | Clean teacher/key bank, corrupted student start | Included |
| Masked student / unmasked teacher asymmetry | MSN [6] | Inspiration for corruption asymmetry; our masking is on code support | Included as an adaptation |
| Reference queue with momentum keys | MoCo [9] | Possible alternative if key drift is large | Deferred |
| Small/large/intermediate budget sampling | US-Nets [10] | Possible future single-model multi-budget training | Deferred |

There is no reason to add all SSL tricks at once. Raw unit-normalized, hard-sparse 8192-dimensional codes are also not a direct drop-in target for dense-feature variance-floor or covariance regularizers. Their constraints and gradients need separate analysis.

## 10. Sparsity diagnostics that determine whether the idea was tested

### 10.1 Disjoint support can eliminate the semantic gradient

If a query's selected support overlaps none of the selected supports in its landmark bank, every raw sparse cosine is zero. Its derivatives through the selected hard-Top-K values are also zero. A softmax, KL, or geometric-overlap wrapper does not restore missing support gradients.

At K=1, a normalized nonnegative code is a basis vector, so its positive coefficient amplitude cannot alter cosine similarity even when supports overlap.

As a rough heuristic under uniform feature usage, the expected number of overlapping coordinate events is MK²/h. With h=8192 and M=1024, this is 8 at K=8, 2 at K=4, and 0.5 at K=2. Actual usage can be highly nonuniform. This is why the first experiment uses K=8.

Record zero-overlap and near-zero selected-coordinate gradient fractions. Adding a straight-through support estimator or switching to decoded similarity is a separate method change.

### 10.2 Nonzero overlap is not sufficient

Student profiles can remain nearly uniform. Log entropy divided by log M, positive versus shuffled-tuple affinity, and weighted component gradient norms on a fixed training-only diagnostic batch.

A null result with negligible neighborhood gradients is different from a null result after a functioning semantic signal. If calibration is necessary, predeclare a small training/development-only temperature or weight check; report it instead of quietly expanding the search.

### 10.3 Geometry and activity are separate

Track reconstruction, teacher-neighbor preservation, actual 1-NN accuracy, and dictionary usage separately. More active units or lower MSE can coexist with worse retrieval.

Repair can change supports, but hard Top-K still differentiates only through selected values. Auxiliary wide reconstruction provides a separate path to more coordinates; it does not prove that the semantic loss can choose every useful missing atom.

## 11. Cost and deployment accounting

The bank and OT are training-only. The correction adds encoding work before storage.

For each repair step, the main work is sparse decoding zD, a residual-to-dictionary projection rDᵀ, and the low-rank correction. The rDᵀ projection is still dense in the parent dimension and dictionary width. Small K does not make that projection free.

The default h=8192, M=1024, n=32 neighborhood tensor has 32³ entries, about 128 KiB in float32. A useful intermediate has shape n²×M, about 4 MiB. Do not allocate an n³×M tensor.

Production query-to-landmark sparse scoring should gather the bank rows at each query's active coordinates and accumulate K contributions, preferably with anchor chunking. A dense B×h by h×M product in the pedagogical reference is wasteful at the full experiment scale.

Report:

- Backbone extraction and SAE/repair encoding separately.
- Index construction separately from search.
- Search-only timing under the existing CUDA benchmark.
- Training time, key refresh time, and OT time.
- Actual code/index bytes and model parameter bytes.
- Both equal-active-count and equal-storage comparisons with MRL.

In the supplied transfer layout, a padded sparse row costs 8K+4 bytes: FP32 values, uint32 coordinate IDs, and one uint32 nonzero count. A dense FP32 prefix costs 4m bytes. Thus K=8 costs 68 bytes and matches m=17, not m=8. Retrieval posting structures have their own measured overhead; use actual index bytes there.

Do not infer a speedup from an operation-count ratio. Corrected supports may change posting-list lengths even when K stays fixed.

## 12. What would make each component meaningful?

| Component | Required comparison | Evidence to seek | If it fails |
|---|---|---|---|
| Repair | R versus unchanged continuation C | Better clean-code accuracy for measured extra encoding cost | Keep original encoder; do not retain repair solely for lower MSE |
| Training masks | M versus clean-repair R | Better unmasked deployment accuracy | Remove masks |
| Direct relation loss | O versus M | Better teacher geometry and actual accuracy | Check gradient coverage; otherwise remove |
| Existing OT after repair | O versus S | Increment beyond the no-OT continuation | Prefer the simpler relation-only objective if tied |
| Neighborhood OT cost | N versus O and S | Benefit beyond old OT and direct distillation | Do not claim a new transport contribution |
| Wide reconstruction | F versus N | Better small code with fixed wider auxiliary budget | Skip wide branch changes |
| Annealing | A versus both F and N | Benefit specifically from a changing auxiliary budget | Keep fixed-wide or original budget |
| Protected long branch | Output/state comparison | Unchanged declared long representation | Treat any change as an implementation error |
| Generalization | Locked holdout and fresh runs | Gains surviving seed variation and a second budget | Narrow the claim to the tested setting |

The named pilot arms and exact settings are in `full_experiment_detail.md`. The table describes causal contrasts within that controlled experiment, not claimed outcomes.

## 13. Claim boundaries and prior art

Potential contribution: a practically useful combination of residual sparse refinement, teacher-defined multiway neighborhood cost, and fixed-deployment-budget auxiliary training, validated against a strong tuned MRL and simpler relational controls.

Not independently new: masking, residual inference, hard sparsity, teacher distillation, multiway OT, or the multi-distribution affinity. The combination has not been exhaustively checked for all prior art, so this document does not assert a definitive novelty claim.

A useful result could also be negative: if direct relational training matches the new OT method, the evidence favors a simpler compressor. If tuned MRL remains better at matched bytes and encoding time, report that result instead of relying on published curves from a different parent.

## 14. References

1. Piran et al. **Contrasting Multiple Representations with the Multi-Marginal Matching Gap** (ICML 2024). [Source](https://arxiv.org/abs/2405.19532)
2. Guo et al. **CSRv2: Unlocking Ultra-Sparse Embeddings** (2026). [Source](https://arxiv.org/abs/2602.05735)
3. Daithankar et al. **You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences** (2026 preprint). [Source](https://arxiv.org/abs/2606.15956)
4. Fang et al. **SEED: Self-supervised Distillation For Visual Representation** (ICLR 2021). [Source](https://arxiv.org/abs/2101.04731)
5. Zheng et al. **ReSSL: Relational Self-Supervised Learning with Weak Augmentation** (NeurIPS 2021). [Source](https://papers.nips.cc/paper_files/paper/2021/hash/14c4f36143b4b09cbc320d7c95a50ee7-Abstract.html)
6. Assran et al. **Masked Siamese Networks for Label-Efficient Learning** (2022). [Source](https://arxiv.org/abs/2204.07141)
7. Wang, Ling, and Huang. **Learning Deep l0 Encoders** (AAAI 2016). [Source](https://arxiv.org/abs/1509.00153)
8. Jolad et al. **A New Family of Bounded Divergence Measures and Application to Signal Detection**, section 2.5 on multi-distribution affinity. [Source](https://arxiv.org/pdf/1201.0418)
9. He et al. **Momentum Contrast for Unsupervised Visual Representation Learning** (CVPR 2020). [Source](https://arxiv.org/abs/1911.05722)
10. Yu and Huang. **Universally Slimmable Networks and Improved Training Techniques** (ICCV 2019). [Source](https://arxiv.org/abs/1903.05134)
11. Kusupati et al. **Matryoshka Representation Learning**. [Source](https://arxiv.org/abs/2205.13147)
12. Wen et al. **Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation** (ICML 2025). [Source](https://arxiv.org/abs/2503.01776)

The supplied experiment attachment is the authority for current repository settings. The new schedules, defaults, run gates, and implementation contracts in this package are proposed research choices.

