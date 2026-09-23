# ImageNet five-arm architecture ablation

`csr_vs_mmpot_imagenet.py` implements the active thesis experiment:
Matryoshka Representation Learning, CSR v1, MPSAE v1, CSRv2, and MPSAEv2 on
ResNet-18 and ResNet-50. The script filename is retained for compatibility.

## Controlled comparison

For each backbone, the script uses ImageNet-1K and evaluates exact L2 1-NN
accuracy with the training split as gallery and validation split as queries.
Every embedding is unit normalized. ResNet-18 uses matched budgets
`K={8,16,32,64,128,256,512}` and ResNet-50 uses
`K={8,16,32,64,128,256,512,1024,2048}`. All four sparse arms additionally
evaluate `K={1,2,4}`.

### Matryoshka

The pretrained backbone is fine-tuned end to end. Independent classifiers are
attached to power-of-two feature prefixes from 8 through the full backbone
dimension, and all evaluated prefix budgets are included. The objective is:

```text
L_MRL = lambda_MRL * sum_m CrossEntropy(W_m F(x)[:m], y).
```

`lambda_MRL` is configured by `MRL_CLASSIFICATION_WEIGHT` and defaults to
`1.0`. At budget K, exact 1-NN uses the first K feature coordinates.

### CSR v1 fixed-K arm

CSR v1 uses frozen cached backbone features and the CSR reconstruction,
auxiliary, and non-negative contrastive objectives. Its training support is
fixed at `V1_TRAIN_K=32` by default. Evaluation is independent of that training
support and covers every sparse evaluation budget listed above.

### MPSAE v1 fixed-K arm

MPSAE v1 uses the matched frozen features, initial SAE state, and fixed
`V1_TRAIN_K=32` support with the MPSAE reconstruction, nested-view, auxiliary,
and multimarginal partial-transport objectives. It uses the same complete
sparse evaluation-budget grid as CSR v1 and both v2 arms.

### CSRv2 annealing arm

The pretrained backbone is frozen and cached once. A tied Top-K sparse
autoencoder with hidden width `h=4d` retains the existing reconstruction and
non-negative contrastive objective:

```text
L_CSRv2(t) = lambda_main L(K_t) + lambda_4K L(4K_t)
           + lambda_aux L_aux + lambda_NCL L_NCL(K_t).
```

The training support follows the CSRv2-style half-cosine continuation schedule:

```text
K_t = K_target + 0.5 * (K_start - K_target)
                    * (1 + cos(pi * t / T_anneal)).
```

`K_start=64`, `K_target=2`, and `T_anneal` spans 70% of CSRv2's optimization
steps by default. K remains at the target for the rest of training. Integer K
is obtained by rounding and clamping to `[K_target,K_start]`.

The environment controls are `CSRV2_MAIN_RECON_WEIGHT`,
`CSRV2_MULTI_TOPK_RECON_WEIGHT`, `CSRV2_AUX_RECON_WEIGHT`,
`CSRV2_CONTRASTIVE_WEIGHT`, `ANNEAL_START_K`, `TRAIN_K`, and
`ANNEAL_FRACTION`.

### MPSAEv2 annealing arm

MPSAEv2 uses the same frozen features, tied SAE architecture, estimated
pre-bias, random initialization, and training-order seed as CSRv2. Its current
base support `K_t` follows the same half-cosine schedule. The Top-K, Top-2K,
and Top-4K views therefore become Top-`K_t`, Top-`2K_t`, and Top-`4K_t` at
every optimization step:

```text
L_MPSAEv2(t) = lambda_main L(K_t)
             + lambda_nested mean(L(2K_t), L(4K_t))
             + lambda_aux L_aux
             + lambda_MMPOT L_MMPOT-gap(K_t, 2K_t, 4K_t).
```

The MMPOT term uses the three-view circular-variance cost and a three-marginal
entropy-regularized partial-transport optimality gap. Both MPSAE variants are
trained for `base_epochs + 4`; Matryoshka and both CSR variants use
`base_epochs`. Because the schedule
is defined as a fraction, each sparse arm anneals over 70% of its own total
optimization steps and then holds K fixed.

The environment controls are `MPSAEV2_MAIN_RECON_WEIGHT`,
`MPSAEV2_NESTED_RECON_WEIGHT`, `MPSAEV2_AUX_RECON_WEIGHT`,
`MPSAEV2_MMPOT_WEIGHT`, plus the shared annealing controls.

## Scope of the CSRv2 label

This experiment intentionally isolates progressive K annealing inside the
existing matched frozen-backbone protocol. It does not claim to implement the
CSRv2 paper's separate supervised-contrastive or full-backbone fine-tuning
variants. `MPSAEv2` likewise denotes the existing MPSAE objective augmented
with the same annealing curriculum.

## Recorded schedule data

Each sparse method records whether its K is fixed or annealed, its start and
target K, total optimization steps, and applicable annealing metadata. Every epoch also
records its first, last, and mean K. Batch-level W&B logs include the current
K. Evaluation is not annealed: each reported point uses its requested fixed K.

## Retrieval and timing

Matryoshka prefixes use dense exact CPU FAISS `IndexFlatL2`. CSR/MPSAE v1/v2
codes remain in CPU SciPy CSR form and use exact sparse matrix products. Because
all vectors are unit normalized and sparse codes are non-negative, maximizing
the sparse dot product is exactly equivalent to minimizing squared L2 distance.
The benchmark times only the neighbor-search computation; encoding,
normalization, gallery construction, and index construction are excluded. All
five retrieval timings therefore use the CPU while retaining the appropriate
dense or sparse computation. The experiment records total search time and mean
milliseconds per query and creates separate
accuracy and retrieval-time plots for ResNet-18 and ResNet-50.

## Weight-free output policy

No trained model, optimizer, scheduler, scaler, or resume checkpoint is saved.
JSON histories and result summaries are written after every epoch. Frozen
backbone feature caches are reusable. Matryoshka features are regenerated from
the live in-memory model because there is intentionally no trained weight file
to bind a reused cache to.

## One-command execution

```bash
cp .env.example .env
# Replace HF_TOKEN after accepting the ImageNet license.
bash run_all_experiments_docker.sh
```

The container always runs all five arms on both ResNet backbones, then
creates per-budget CSV/JSON tables, pairwise effects, plots, a manifest, a ZIP,
and `RUN_COMPLETE.json`. It does not run the older pairwise-MMPOT or true-MMPOT
experiment families.
