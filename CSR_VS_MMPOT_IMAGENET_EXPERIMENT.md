# ImageNet three-method architecture ablation

`csr_vs_mmpot_imagenet.py` implements the only active thesis experiment:
Matryoshka Representation Learning, Contrastive Sparse Representation, and
MP-SAE on ResNet-18 and ResNet-50.

## Controlled comparison

For each backbone, the script uses ImageNet-1K and evaluates exact L2 1-NN
accuracy with the training split as gallery and validation split as queries.
The matched representation budgets are `K={8,16,32,64,128,256}`.

### Matryoshka

The pretrained backbone is fine-tuned end to end. Independent classifiers are
attached to power-of-two feature prefixes from 8 through the full backbone
dimension, and all evaluated prefix budgets are included. The objective is

```text
L_MRL = lambda_MRL * sum_m CrossEntropy(W_m F(x)[:m], y).
```

`lambda_MRL` is configured by `MRL_CLASSIFICATION_WEIGHT` and defaults to
`1.0`.

At budget `K`, exact 1-NN uses the first `K` feature coordinates.

### CSR

The pretrained backbone is frozen and cached once. A tied Top-K sparse
autoencoder with hidden width `h=4d` is trained from the paper objective

```text
L_CSR = lambda_main L(K) + lambda_4K L(4K)
      + lambda_aux L_aux + lambda_NCL L_NCL.
```

The corresponding environment variables are `CSR_MAIN_RECON_WEIGHT`,
`CSR_MULTI_TOPK_RECON_WEIGHT`, `CSR_AUX_RECON_WEIGHT`, and
`CSR_CONTRASTIVE_WEIGHT`. Their defaults are `1`, `1/8`, `1/32`, and
`1`, respectively.

`L_NCL` treats each non-negative sparse code as its own positive and the other
batch codes as negatives. At evaluation budget `K`, the representation retains
at most `K` positive SAE activations after the Top-K selection and ReLU.

### MP-SAE

MP-SAE uses the same frozen features, tied SAE architecture, estimated
pre-bias, random initialization, training order seed, `K`, and `h` as CSR. Its
Top-K, Top-2K, and Top-4K codes are treated as aligned views. The objective is

```text
L_MP-SAE = lambda_main L(K) + lambda_nested mean(L(2K), L(4K))
         + lambda_aux L_aux + lambda_MMPOT L_MMPOT-gap.
```

The corresponding environment variables are `MPSAE_MAIN_RECON_WEIGHT`,
`MPSAE_NESTED_RECON_WEIGHT`, `MPSAE_AUX_RECON_WEIGHT`, and
`MPSAE_MMPOT_WEIGHT`. Their defaults are `1`, `1/8`, `1/32`, and
`1.3`, respectively.

The MMPOT term uses the three-view circular-variance cost and a
three-marginal entropy-regularized partial-transport optimality gap. MP-SAE is
trained for `base_epochs + 4`; Matryoshka and CSR use `base_epochs`.

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

The container always runs all three methods on both ResNet backbones, then
creates per-budget CSV/JSON tables, pairwise effects, plots, a manifest, a ZIP,
and `RUN_COMPLETE.json`. It does not run the older pairwise-MMPOT or true-MMPOT
experiment families.
