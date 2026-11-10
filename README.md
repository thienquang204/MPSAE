# MPSAE — adaptive sparse embeddings, against CSR and CSRv2

Working notes for this repository. Not a release document.

**What we claim.** CSR sparsifies a frozen embedding with a tied Top-K
autoencoder and regularises the codes with a contrastive term. MPSAE replaces
that term with an entropic **multi-marginal partial-OT matching gap** across
three budgets (K, 2K, 4K). Current MPSAEv2 combines the OT gap with instance
contrastive learning at fixed K. Both use the same Top-K inference code and
the MPSAE-specific optimisation settings below; their effectiveness is tested
against the baselines.

**Ground rules.** Settings follow the published papers (CSR Table 3/9, CSRv2
Table 11). Algorithms follow the *released code*. Where the released code
disagrees with its own paper, the code wins and the disagreement is written
down. Every deviation we make is recorded in an artefact manifest, never
applied silently.

---

## Run it

One configuration selects the ImageNet experiments and downstream frozen
retrieval benchmarks. `.env` holds credentials and environment settings;
`configs/thesis.yaml` selects the experiments. No downstream fine-tuning runs.

```bash
cp .env.example .env                       # if .env does not exist yet
# Fill HF_TOKEN (write access for result backup); WANDB_TOKEN is optional.
bash all_in_one.sh                 # install -> pull embeddings -> run -> back up
```

The all-in-one entry point detects the NVIDIA GPU/CPU architecture, installs
the matching native environment and runs the
campaign. HF_TOKEN identifies the account: it creates/reuses the private dataset
repository `<account>/MPSAE`. W&B uses project `cvpr`, with the account's default
entity; `WANDB_TOKEN` and the standard `WANDB_API_KEY` are both accepted. No
repository, project or team needs to be created manually.

Python runs selected fits and evaluations sequentially, reuses compatible
ImageNet checkpoints, and writes a campaign report from the explicit result
list. Repeating the command resumes unfinished work. See
[`docs/benchmark.md`](docs/benchmark.md) for the dataset protocols, result
identities, figure definitions and server commands. Expensive source fits and
feature archives are backed up as they finish; the final sync includes metrics,
plots and campaign state, also after a failed run. Raw images, caches, `.env`
and W&B internal files are excluded. Upload failures are recorded and can be
retried without retraining:

```bash
bash all_in_one.sh --skip-setup    # resume with installed environment
bash all_in_one.sh --sync-only    # retry backup and W&B report only
bash all_in_one.sh --check        # check/create remote destinations
```

To train from the published embeddings without downloading images or running
backbones, use:

```bash
bash all_in_one.sh
# Add --skip-setup when the server environment is already installed.
```

The default config selects ImageNet/EuroSAT for `ff2048`, `vit_b16`
and SOP for `ff2048`, whose bundles are
available in [Lukita/MPSAE-embeddings](https://huggingface.co/datasets/Lukita/MPSAE-embeddings).
Training still fits the sparse autoencoders. The download step verifies hashes
and links archives into `work/direct-v1/`; the campaign checks dimensions,
sample IDs, backbone weights and preprocessing. `precomputed_only: true` stops
on missing or incompatible features instead of extracting them.

As checked on 2026-11-09, the repository has 13 of the full campaign's 20
bundles: all ImageNet and EuroSAT backbones, plus SOP for `ff2048`, `rn18` and
`convnext_tiny`. Flowers102 and the other two SOP backbones are missing.
`configs/thesis.yaml` and `configs/precomputed.yaml` now select 5 of those bundles.
Flowers102 is excluded from these campaigns because no bundles are published;
this does not establish a dataset defect. SOP remains enabled for the selected
ResNet-50 through `downstream.backbones.sop`. Plans, downloads and
reports use this same selection (120 sparse and 3 dense downstream evaluations).
Unexpected missing or incompatible bundles fail explicitly rather than being
silently dropped. Building new bundles requires a separate config with
`precomputed_only: false`, `--build-missing-embeddings`, and a writable private
embedding repository. See [docs/data.md](docs/data.md)
for downloading separately and using the Python entry point.

The native installer supports Ubuntu 22.04/24.04 on x86_64 and ARM64/SBSA.
It defaults to CUDA 13.0 on H100, A100, RTX A6000 and other supported
Turing-or-newer GPUs; Linux driver >= 580.65.06 is required. It installs the
pinned PyTorch environment and verifies direct JPEG processing and GPU kernels.
FFCV/CuPy are optional legacy dependencies, not part of the default pipeline.
It creates `.env` only when missing; fill in the token after installation.
It does not download datasets or start training. Preview its plan with:

```bash
bash setup_server.sh --dry-run
```

The shell launcher uses the same Python CLI and also saves a console transcript:

```bash
bash pipeline.sh                # configs/thesis.yaml
bash pipeline.sh status
bash pipeline.sh report
```

Use `--config` with direct Python commands to select the campaign explicitly.
Without a config or `MPSAE_CONFIG`, the old ImageNet-only CLI remains available.
The default direct campaign uses `work/direct-v1/` for features, fits and reports;
historical `work/runs/` results remain separate. ImageNet JPEGs at `work/imagenet`
are reused, then retired only after all five feature bundles have verified HF backups. Docker is an optional older
deployment path, described separately in `docs/benchmark.md`.

The smoke configuration synthesises class-structured source and target data
and runs capped fits, retrieval and reporting in a separate workspace. Its
numbers are not science and are labelled synthetic in the manifests.

### Hardware

x86_64 or ARM64/SBSA, ≥ 24 GB VRAM (40–80 GB comfortable), ≥ 64 GB RAM, ≥ 16 CPU cores,
SSD sized for the peak storage budget below.

The default five-backbone campaign needs about 24 GB for one copy of ImageNet
embeddings. Peak disk use also includes original images, one active writer,
download caches and the environment; 500 GB is not an unconditional guarantee.
Dataset preparation and cleanup are sequential. Read the storage and resume
rules in [`docs/server.md`](docs/server.md) before reusing a full server.

There is **no FAISS dependency**. Its GPU wheels lag new architectures. On
L2-normalised vectors, maximum inner product is equivalent to minimum L2
distance, so a blocked FP32 GEMM argmax returns exactly what `IndexFlatL2`
returns, and all search is PyTorch-only.

---

## The experiments

**E1 — matched budget.** Each SAE is trained and evaluated at its training K.
The selected campaign uses K=8 for ResNet-50 and ViT-B/16.

**E2 — fixed checkpoint, inference sweep.** Each of those checkpoints is frozen
and evaluated at **K = 2, 4, 8, 16, 32, 64, 128, 256**. There are still only
10 SAE fits: 2 backbones x 5 methods x seed 1227. This produces 80 ImageNet E2
points (10 reused from E1) and 120 sparse + 3 dense downstream evaluations.
HF backbone embeddings are reused. No SAE is retrained for a different inference K.
ImageNet linear probes are fitted separately on training codes at each K; their
validation accuracy and 1-NN accuracy are both reported. The existing
`imagenet.linear_probe` and `imagenet.timing` flags now apply to E2 as well as E1.
Missing probe/timing components are filled without recomputing completed accuracy.
The extra inference and probe evaluations do take additional time.

Reports use only measurements from the selected experiment. There are no
imported paper curves or published dense numbers. `Dense` means our own measured
full-feature baseline. Short legend names are CSR, CSRv2, CSR-MTK (CSR multi-TopK),
MPSAE, and MPSAEv2. Seed counts, protocols and checkpoint IDs stay in captions/tables.
One seed yields no between-seed uncertainty band.

The main ImageNet figure is `inference_summary.pdf` (also PNG); per-backbone
`e2_generalisation_*` plots show linear probe and 1-NN against inference K.
`e2_nnz_*` uses measured gallery nonzeros instead, so saturation of MPSAE can
compress its horizontal extent there without hiding the larger inference K points
on the main figure. `e2_cost_*` shows search time, sparse matmul time and index
memory where measured. Downstream `quality_*` shows R@1 and mAP@R versus inference
K; `cost_*` shows retrieval cost and stored code size. All curves keep training K
fixed, and all figures export vector PDF, 300-dpi PNG and a separate caption.

After pulling this change, run `bash all_in_one.sh --skip-setup` to fill missing
K evaluations with existing compatible checkpoints. `report` alone only redraws
measurements already present in that campaign's catalog; it cannot create new K
results. Outputs are under `work/direct-v1/reports/thesis/<plan_id>/`.

| backbone | d | recipe | preset | Training K budgets | E2 anchors | cells |
|---|---|---|---|---|---|---|
| `ff2048` | 2048 | MRL | Table 3 | 8 | 8 | 5 |
| `vit_b16` | 768 | CSRv2 | Table 3 | 8 | 8 | 5 |

10 cells = the selected backbone/K pairs x 5 methods x seed 1227.
The previous 20-fit, 35-fit and original 80-fit setups are preserved, including exact YAML and resolved presets,
in [`large_scale_abblation.md`](large_scale_abblation.md) for a later ablation.
The wider reference grids remain registered for explicit custom campaigns.
K never exceeds the backbone's own width, so every point is a compression.
The 768-d backbones are treated like FF2048, the 384-d one like RN18.

### Methods

`mpsae` and `mpsaev2` use training batches of **1024** on every backbone and
the original constant learning rate of **4e-5**. Baseline batch sizes retain
their backbone presets. OT solver limits remain **400** initial iterations
and **8000** refinement iterations; early stopping still applies. **MPSAEv2 trains
for 15 epochs; all other methods retain 10 epochs.** OT group size, loss weights
and the MPSAEv2 objective remain unchanged. Comparisons
with CSR include the batch-size and update-count differences where applicable.

| method | K during training | gates/step | regulariser | extra reconstruction |
|---|---|---|---|---|
| `csr` | fixed | K, 4K, K | γ · `inbatch_cl` | — |
| `csrv2` | annealed | k, 4k, k | γ · `label_based_cl` | — |
| `mpsae` | fixed | K, 4K, 2K | 2 · OT gap | — |
| `mpsaev2` | fixed | K, 4K, 2K | 2 · OT gap + γ · `inbatch_cl` | — |
| `csr_multitopk` | fixed | K, 4K, 2K | γ · `inbatch_cl` | L(2K)/8 |

Shared reconstruction: `B(k) = L(k) + 0.125·L(4k) + (1/32)·NMSE_aux`.
Add the weighted regulariser(s) and extra reconstruction listed above.

Every method gates exactly **three** times per step, so `dead_threshold=30`
means the same thing for all of them. For the baselines that is the released
sequence verbatim: CSR runs three forward passes, the third only to build a
duplicate contrastive view. Ours spend the third gate on the 2K view they
actually consume.

`csr_multitopk` is the control for supervising a 2K view. It shares `csr`'s
K/4K reconstruction and contrastive term, adds `L(2K)/8`, and uses the 2K view
for its third activity-counter update. MPSAE uses the batch size described above. It
follows CSRv2 App. G's "Diverse Multi-TopK" ablation in spirit.

`mpsaev2` now uses **fixed K with both OT and instance contrastive learning**:
`B(K) + 2·gap(z_K,z_2K,z_4K) + γ·C_instance(z_K)`. It needs no labels and adds
no `L(2K)/8` reconstruction. The contrastive coefficient is 0.1 for ResNet-50,
ConvNeXt and ViT, and 1.0 for ResNet-18 and DINO; temperature is 0.2. The
contrastive term operates on the existing K code, so the three activity-counter
updates remain K, 4K, 2K. Against MPSAE, the objective adds contrastive loss;
MPSAEv2 now also trains for 15 epochs versus MPSAE's 10. Batch size and fixed K
match, but this comparison includes training duration. Better accuracy remains
a hypothesis to test.
ImageNet reports include `paired_mpsaev2_vs_mpsae`; downstream paired tables
also include this treatment/control pair, alongside the existing comparisons.
Each difference uses matching seeds and evaluation settings.

MPSAEv2 is displayed as `MPSAEv2`; its internal key and objective variant
remain `mpsaev2` and `fixed_ot_cl_v1`. An otherwise identical 10-epoch
checkpoint continues through epoch 15 with its weights, Adam state, RNG
and activity counters preserved. Only the stopping point changes; the
constant learning rate and fixed-K loss are unchanged. A different loss,
optimizer, seed or dataset still prevents continuation. The completed
15-epoch fit receives a new signature, so E1/E2 and downstream results
are recomputed for the final weights. Interrupted extensions resume.

Only `csrv2` anneals in the current method set. The former MPSAEv2 **variant B**
(annealed main budget with OT only) is retired; its checkpoints are incompatible
with the new objective. Historical variants A and B are described in
[`docs/notes/mpsaev2_legacy_A.md`](docs/notes/mpsaev2_legacy_A.md).
New hybrid fits use `runs/<backbone>/mpsaev2__fixed_ot_cl_v1/k<K>/seed<SEED>/`
and record objective variant `fixed_ot_cl_v1`, while reports retain the logical
method name `mpsaev2`. Existing unversioned MPSAEv2 directories remain intact
and are excluded from current-method reports. The hybrid starts a new fit;
compatible checkpoints for the other methods remain reusable.

---

## Findings about the released baselines

Recorded here because they change what "reproduce the baseline" means. All
verified by reading CSR @ `b284898` and CSRv2 @ `3b2d4d2`.

1. **CSR's contrastive loss compares each code with an identical copy of
   itself.** The "second view" is the same deterministic forward pass. There is
   no augmentation anywhere in the released pipeline, despite how the CSRv2
   paper describes CSR. In effect it is a uniformity term on the normalised
   sparse codes. We reproduce this exactly.
2. **CSRv2's supervised contrastive loss is switched off in the released
   pipeline.** The loader reads `train_data.get('labels')`; every released
   producer writes `label`. All labels become 0 and the loss collapses to ≈0.
   Our archives store `labels`, so the term is live — recorded as a deviation
   in every manifest. `tests/test_against_release.py` pins both behaviours.
3. **k-annealing is linear, not cosine.** `get_current_k` interpolates linearly
   over the first 70 % of steps; CSRv2 App. G.2 reports cosine as worse. A
   cosine schedule would be a different algorithm.
4. **`k_init` has clamps the paper omits**: 64 below K=64 else 4K, then clamped
   to `h//4` and up to K. So K=768 starts at 2048, not 3072, and K=2048 does
   not anneal at all.
5. **The dead-latent counter advances three times per step**, once per `top_k`
   call, so `dead_threshold=30` is about ten optimiser steps, not thirty.
6. **Released defaults differ from the published tables**: lr 4e-4 vs 4e-5,
   weight decay 0 vs 1e-4, CSRv2 `auxk` 1024 vs 512. These are settings, so we
   take the papers' values and say so.
7. **There is no released image checkpoint for the full CSRv2**, and no code
   for FF2048 fine-tuning. The comparable published variant is CSRv2-linear.
8. **No released code produces the FF2048 embeddings.** The extraction script
   only handles timm models, and the published archive's row count is
   incompatible with that script's `drop_last=True`. We therefore extract
   FF2048 with MRL's own retrieval recipe and gate on reproducing the published
   dense 1-NN of 71.19.

---

## Precision

Verified against the release, and it is not uniform:

| step | precision |
|---|---|
| training | float32, TF32 **off** |
| embedding extraction | fp16 autocast, stored float32 |
| code encoding for evaluation | fp16 autocast, stored float32 |
| **search** | **float32, TF32 off** |
| linear probe | float32 |

`CSRv2/image/utils.py` wraps the encode in `autocast()`; `faiss_nn.py` casts to
float32 before `IndexFlatL2`. Searching in fp16 would reorder near-ties across
a 1.28 M-row gallery and change top-1, so it is never done.

---

## Layout

```
mpsae/
  config.py      backbones, paper presets, the grid           (torch-free)
  io.py          atomic JSON, digests, locks, mmapped NPZ
  logs.py        logging, JSONL metrics, stage receipts
  model.py       tied Top-K SAE, released CSR semantics
  losses.py      reconstruction, AuxK, both released CL losses
  schedules.py   CSRv2 linear k-annealing                     (torch-free)
  ot.py          angular cost, Greenkhorn, matching gap       ← our contribution
  methods.py     one objective per method
  features.py    embedding archives, batch source, residency
  train.py       one resumable fit
  runs.py        the grid, E1 and E2
  pipeline.py    stages: env → imagenet → ffcv → embed → publish → train → report
  cli.py         python -m mpsae
  server.py      native NVIDIA GPU/dependency verification
  remote.py      private MPSAE backup, cvpr tracking, all-in-one orchestration
  data/          ImageNet acquisition, FFCV serialisation
  embed/         MRL and CSRv2 extraction recipes, archive writer
  evaluate/      encoding, exact 1-NN, probes, retrieval timing
  report/        aggregation, style, figures (experiment measurements only)
csrc/            CUDA inverted index (optional secondary metric)
*.sh             all_in_one.sh, setup_server.sh, activate_server.sh, pipeline.sh
capture_golden.py one-off OT fixture capture
constraints/     shared native GPU dependency versions
tests/reference/ transcribed CSR/CSRv2 source, used only as a test oracle
docs/            methods, data provenance, pipeline, outputs, server setup
```

Outputs land under `work/`: `runs/<backbone>/<method>/k<K>/seed<S>/` holds
`config.json`, `train.jsonl`, `checkpoint.pt`, `summary.json`, `eval_e1.json`
and, at an anchor, `eval_e2.json`. Tables and figures go to `work/reports/`.

---

## Tests

```bash
pytest tests/ -q                   # torch-dependent tests skip without torch
pytest tests/test_grid.py tests/test_schedules.py -q    # these run anywhere
```

* `test_against_release.py` — our model, losses, counter and one Adam step
  against `tests/reference/csr_released.py`, a transcription of the published
  source. This is what makes the baseline claim defensible.
* `test_methods.py` — the five methods differ *only* as designed; gate counts
  match; the ablation really isolates the OT term.
* `test_schedules.py` — k-annealing against the released function at every step.
* `test_ot.py` — feasibility, nonnegativity, the envelope gradient, group
  independence, and the pinned legacy numerics.

The OT golden fixtures need a one-off capture on a torch machine, because the
float32 behaviour of the solver depends on association order that no paper
documents:

```bash
python capture_golden.py      # reads the pre-rewrite code out of git
```

Until that file exists the OT contract tests still run; only the
bit-for-bit comparison skips.

---

## Documentation

[`docs/`](docs/README.md) has the detail: [`methods.md`](docs/methods.md) for
the objectives and the baseline findings, [`data.md`](docs/data.md) for
provenance, [`pipeline.md`](docs/pipeline.md) for running and debugging,
[`outputs.md`](docs/outputs.md) for the result schema, and
[`server.md`](docs/server.md) for GPU setup. See
[`performance.md`](docs/performance.md) for the training/retrieval optimizations,
timing protocol and standalone GPU profiler.

## Open items

* Evaluation cost is dominated by exact search at the dictionary width
  (h = 8192): roughly 20 s per point, so E1 ≈ 4 h and E2 ≈ 26 h on one H100.
  A sparse-product fast path for small K would cut E2 substantially.
* `csrc/` is built but the inverted-index search driver is not wired in;
  `evaluate/timing.py` currently measures the paper's `torch.sparse` protocol.
* Frozen-transfer retrieval for EuroSAT RGB, the HF custom Flowers102 split,
  and Stanford Online Products is implemented in the campaign path. DTD, VOC,
  SPair, DeepFashion and ROxford/RParis in `data_usage.md` remain future work.
  That file's DTD test split should be 40 images/class, not 10.
* Nothing in this rewrite has executed on a GPU yet. The first real
  `--smoke` run is the actual test.
