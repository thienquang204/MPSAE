# MPSAE — adaptive sparse embeddings, against CSR and CSRv2

Working notes for this repository. Not a release document.

**What we claim.** CSR sparsifies a frozen embedding with a tied Top-K
autoencoder and regularises the codes with a contrastive term. MPSAE replaces
that term with an entropic **multi-marginal partial-OT matching gap** across
three budgets (K, 2K, 4K). We compare CSR, CSRv2 and MPSAE using the same
Top-K inference code and the optimisation settings below. CSR multi-TopK and
MPSAEv2 are excluded from all experiments and reports.

**Ground rules.** Settings follow the published papers (CSR Table 3/9, CSRv2
Table 11). Algorithms follow the *released code*. Where the released code
disagrees with its own paper, the code wins and the disagreement is written
down. Every deviation we make is recorded in an artefact manifest, never
applied silently.

---

## Run it

One configuration selects the ImageNet experiments and frozen product retrieval
on SOP. EuroSAT kNN is deferred from the main campaign; COCO image-text retrieval
and CLIP are future work. `.env` holds credentials and environment settings;
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

The default config selects ImageNet for `ff2048`, `vit_b16`
and SOP for `ff2048`, whose bundles are
available in [Lukita/MPSAE-embeddings](https://huggingface.co/datasets/Lukita/MPSAE-embeddings).
Training still fits the sparse autoencoders. The download step verifies hashes
and links archives into `work/direct-v1/`; the campaign checks dimensions,
sample IDs, backbone weights and preprocessing. `precomputed_only: true` stops
on missing or incompatible features instead of extracting them.

As checked on 2026-11-11, the repository has 13 of the wider registry's 20
bundles: all ImageNet and EuroSAT backbones, plus SOP for `ff2048`, `rn18` and
`convnext_tiny`. Flowers102 and the other two SOP backbones are missing.
`configs/thesis.yaml` and `configs/precomputed.yaml` select 3 of those bundles:
two ImageNet bundles and one SOP bundle.
Flowers102 is excluded from these campaigns because no bundles are published;
this does not establish a dataset defect. SOP remains enabled for the selected
ResNet-50 through `downstream.backbones.sop`. Plans, downloads and
reports use this same selection (48 sparse and 1 dense downstream evaluations).
Unexpected missing or incompatible bundles fail explicitly rather than being
silently dropped. Building new bundles requires a separate config with
`precomputed_only: false`, `--build-missing-embeddings`, and a writable private
embedding repository. See [docs/data.md](docs/data.md)
for downloading separately and using the Python entry point.

To run only the current SOP retrieval benchmark using completed source fits:

```bash
bash all_in_one.sh --config configs/retrieval.yaml --skip-setup
```

This selects ResNet-50, CSR/CSRv2/MPSAE, seed 1227, training checkpoints
K=8/16/32/256 and inference K=2/16/32/256: **12 existing checkpoints, 48 sparse
evaluations and one dense diagnostic**. It downloads only the SOP feature bundle
and performs no source fitting, ImageNet evaluation, linear probing or repeated
ImageNet timing. Use the same workspace as the source fits. Each fit needs its
`config.json`, `summary.json`, `checkpoint.pt` and original backbone
weights/preprocessing provenance, either in the config or the ImageNet archive
sidecars. Missing or incompatible fits stop this campaign. In particular, MPSAE
at training K=256 must use OT weight **0.5**; an old weight-2.0 fit cannot be reused.
Reports are under `work/direct-v1/reports/retrieval/<plan_id>/`.

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
are reused, then retired only after the required feature bundles have verified HF backups. Docker is an optional older
deployment path, described separately in `docs/benchmark.md`.

The smoke configuration synthesises class-structured source and target data
and runs capped fits, classification, retrieval and reporting in a separate workspace. Its
numbers are not science and are labelled synthetic in the manifests.

### Hardware

x86_64 or ARM64/SBSA, ≥ 24 GB VRAM (40–80 GB comfortable), ≥ 64 GB RAM, ≥ 16 CPU cores,
SSD sized for the peak storage budget below.

The wider five-backbone selection needs about 24 GB for one copy of ImageNet
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

**E1 — matched budget.** ResNet-50 is trained separately at **K=8, 16, 32, 256**;
ViT-B/16 is trained separately at **K=12 and K=48**. Each selected training K
has its own independent fit; the old ViT K=8 checkpoint is not a starting point.
Compatible checkpoints at the selected budgets are reused.

**E2 — fixed checkpoint, inference sweep.** Every selected ResNet checkpoint is
evaluated at **K=2, 16, 32, 256**. Each ViT checkpoint is evaluated at **K=12 and K=48**.
These same inference grids apply to every selected downstream checkpoint; the
main downstream selection is SOP with ResNet-50 only. The campaign has **18 SAE
fits**, **60 ImageNet E2 points** (15 shared with E1), and **48 sparse + 1 dense**
downstream evaluations.
HF backbone embeddings are reused. SAE weights remain fixed within each inference
sweep. The shared fallback inference grid also uses 16/32/256 for ResNet-18 when
that backbone is explicitly selected; it is not added to the active campaign.
ImageNet linear probes are fitted separately on training codes at each K; their
validation accuracy and 1-NN accuracy are both reported. The existing
`imagenet.linear_probe` and `imagenet.timing` flags now apply to E2 as well as E1.
Missing probe/timing components are filled without recomputing completed accuracy.
The extra inference and probe evaluations do take additional time.

**Downstream retrieval.** SOP searches its 60,502 test images against the same
gallery, excluding each query's own image. Relevant images share a product
`item_id`, not merely a broad category. The frozen ImageNet-trained SAE is
evaluated at each inference K using Recall@1/10/100, Precision@10 and mAP@R.
There is no target-data training. The HF source is checked for expected counts
and disjoint train/test products; parity with the original `Ebay_test.txt` file
list has not been independently certified. Existing compatible SOP results are
reused even when the campaign selection changes.

**Optional classification.** The DINO weighted kNN implementation remains
available for explicitly selected EuroSAT/Flowers102 campaigns and the synthetic
smoke test. It evaluates 1-NN and 20-NN with temperature 0.07, keeping neighbour
count separate from sparse budget K. EuroSAT results and feature archives are
retained, but do not enter the current main reports. COCO/CLIP is deferred.

Figures show only the measured sparse methods: CSR, CSRv2 and MPSAE. There are
**no published, Dense, or Full-representation reference curves or markers**.
Dense diagnostics remain available as numerical tables.
Seed counts, protocols and checkpoint IDs stay in captions/tables. One seed yields
no between-seed uncertainty band.

The main ImageNet figures are `inference_summary*.pdf` (also PNG); long grids
are split into parts with at most two backbone/training-K rows. Per-backbone
`e2_generalisation_*` plots show linear probe and 1-NN against inference K.
`e2_nnz_*` uses measured gallery nonzeros instead, so saturation of MPSAE can
compress its horizontal extent there without hiding the larger inference K points
on the main figure. `e2_cost_*` shows search time, sparse top-1 search time and index
memory where measured. SOP `quality_*` figures show R@1 and mAP@R;
`nonzeros_*`, `storage_*`, `paired_*` and `cost_*` show sparsity, storage,
paired differences and search cost. Optional classification campaigns produce
`classification_*` figures separately. All curves
keep training K fixed, and all figures export vector PDF, 300-dpi PNG and a
separate caption. The [benchmark protocol](docs/benchmark.md) documents voting,
ties, empty sparse codes and the relationship to the DINO implementation.

Run `bash all_in_one.sh --skip-setup` for the full ImageNet + SOP campaign,
or use `configs/retrieval.yaml` as above to evaluate existing fits only. Compatible
existing checkpoints are reused. `report` alone only redraws measurements already
present in that campaign's catalog; it cannot compute missing measurements. Full
campaign outputs are under `work/direct-v1/reports/thesis/<plan_id>/`.

| backbone | d | recipe | preset | Training K budgets | E2 anchors | cells |
|---|---|---|---|---|---|---|
| `ff2048` | 2048 | MRL | Table 3 | 8, 16, 32, 256 | 8, 16, 32, 256 | 12 |
| `vit_b16` | 768 | CSRv2 | Table 3 | 12, 48 | 12, 48 | 6 |

18 cells = the selected backbone/K pairs x 3 methods x seed 1227.
The previous five-method settings are preserved, including archived YAML and resolved presets,
in [`large_scale_abblation.md`](large_scale_abblation.md) for a later ablation.
The wider reference grids remain registered for explicit custom campaigns.
K never exceeds the backbone's own width, so every point is a compression.
The 768-d backbones are treated like FF2048, the 384-d one like RN18.

### Methods

The selectable methods are **CSR, CSRv2 and MPSAE**. This applies to every
campaign, legacy CLI entry point, ImageNet E1/E2, probes, retrieval, timing and
report. Explicitly selecting `mpsaev2` or `csr_multitopk` is rejected. Old
checkpoints and downloaded embeddings are retained, while excluded methods'
measurements are omitted from newly generated reports.

`mpsae` uses training batches of **1024** on every backbone and the original
constant learning rate of **4e-5**. Baseline batch sizes retain their backbone
presets. All three methods train for **10 epochs**. OT solver limits remain
**400** initial iterations and **8000** refinement iterations; early stopping
still applies. Comparisons with CSR include the batch-size and update-count
differences where applicable.

| method | K during training | gates/step | regulariser | extra reconstruction |
|---|---|---|---|---|
| `csr` | fixed | K, 4K, K | γ · `inbatch_cl` | — |
| `csrv2` | annealed | k, 4k, k | γ · `label_based_cl` | — |
| `mpsae` | fixed | K, 4K, 2K | λ(K) · OT gap | — |

Shared reconstruction: `B(k) = L(k) + 0.125·L(4k) + (1/32)·NMSE_aux`.
Add the weighted regulariser(s) and extra reconstruction listed above.

MPSAE uses **λ(K)=2.0 for training K ≤ 64** and **λ(K)=0.5 for training K > 64**.
The threshold refers to the main training budget, not the 2K/4K views or an
inference sweep. In the current campaign only the ResNet-50 K=256 fit changes.
Existing high-K fits trained with weight 2.0 must be archived outside `runs/`
before fresh training; they cannot be resumed or evaluated as the new recipe.
The selected overrides appear in the campaign plan, and `ot_weight` is logged
alongside the unweighted `loss_ot`.

Every method gates exactly **three** times per step, so `dead_threshold=30`
means the same thing for all of them. For the baselines that is the released
sequence verbatim: CSR runs three forward passes, the third only to build a
duplicate contrastive view. Ours spend the third gate on the 2K view they
actually consume.

Only `csrv2` anneals in the active method set. Historical MPSAEv2 and
CSR multi-TopK definitions are preserved in
[`docs/notes/excluded_methods.md`](docs/notes/excluded_methods.md).
Removing them from experiment selection does not change any active loss or
checkpoint identity. Compatible CSR, CSRv2 and MPSAE checkpoints remain reusable.


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
* `test_methods.py` — active and historical objectives differ as designed; gate counts
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
* The main frozen-transfer benchmark is product retrieval for Stanford Online
  Products. DINO weighted kNN for EuroSAT RGB and the HF custom Flowers102 split
  remains optional. COCO image-text retrieval and CLIP are deferred. DTD, VOC,
  SPair, DeepFashion and ROxford/RParis in `data_usage.md` remain future work.
  That file's DTD test split should be 40 images/class, not 10.
* Nothing in this rewrite has executed on a GPU yet. The first real
  `--smoke` run is the actual test.
