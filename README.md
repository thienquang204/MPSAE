# MPSAE — adaptive sparse embeddings, against CSR and CSRv2

Working notes for this repository. Not a release document.

**What we claim.** CSR sparsifies a frozen embedding with a tied Top-K
autoencoder and regularises the codes with a contrastive term. We replace that
term with an entropic **multi-marginal partial-OT matching gap** that couples
the codes of the *same* example across three budgets (K, 2K, 4K). Inference is
unchanged — the same Top-K code — so the comparison is entirely about the
training objective.

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
cp .env.example .env                         # set HF_TOKEN for ImageNet
docker compose build
docker compose run --rm mpsae plan
docker compose run --rm mpsae                # default: the thesis campaign
docker compose run --rm mpsae status
docker compose run --rm mpsae report
```

The container runs selected fits and evaluations sequentially, reuses compatible
ImageNet checkpoints, and writes a campaign report from the explicit result
list. Repeating the command resumes unfinished work. See
[`docs/benchmark.md`](docs/benchmark.md) for the dataset protocols, result
identities, figure definitions and server commands.

For a host installation instead of Docker:

```bash
cp .env.example .env && $EDITOR .env    # paste HF_TOKEN
./scripts/setup_server.sh               # once: CUDA-matched torch, deps, FFCV
./scripts/pipeline.sh --smoke           # separate synthetic campaign workspace
./scripts/pipeline.sh                   # configs/thesis.yaml
```

Useful afterwards:

```bash
./scripts/pipeline.sh status            # what is done, what is outstanding
./scripts/pipeline.sh plan              # the campaign, no GPU/token needed
./scripts/pipeline.sh stage embed       # one legacy ImageNet stage
./scripts/pipeline.sh report            # rebuild the selected campaign's reports
./scripts/pipeline.sh --config configs/thesis.yaml  # edit scientific selections in YAML
```

The container sets `MPSAE_CONFIG=configs/thesis.yaml`, so `plan`, `status` and
`report` use the same campaign when the Docker command is replaced. On a host
with `MPSAE_CONFIG` unset, `python -m mpsae run` without `--config` retains the
original ImageNet-only interface. Both use the existing `work/runs/` fit layout;
campaigns reference those fits rather than training a second copy per dataset.

The smoke configuration synthesises class-structured source and target data
and runs capped fits, retrieval and reporting in a separate workspace. Its
numbers are not science and are labelled synthetic in the manifests.

### Hardware

x86_64, ≥ 24 GB VRAM (40–80 GB comfortable), ≥ 64 GB RAM, ≥ 16 CPU cores,
≥ 500 GB SSD.

`setup_server.sh` picks the torch wheel from the driver branch and the compute
capability together, because each bounds a different end: the driver caps the
CUDA runtime that can load (13.x needs ≥ 580, 12.8 needs ≥ 570), while CUDA 13
dropped every architecture below sm_75 and so cannot serve a V100. CUDA 13 is
the preferred case — driver ≥ 580 gets `cu132` and current torch. Override with
`MPSAE_TORCH_INDEX`. Details in [`docs/server.md`](docs/server.md).

There is **no FAISS dependency**. Its GPU wheels lag new architectures. On
L2-normalised vectors, maximum inner product is equivalent to minimum L2
distance, so a blocked FP32 GEMM argmax returns exactly what `IndexFlatL2`
returns, and all search is PyTorch-only.

---

## The experiments

**E1 — per-K (the main comparison).** One model trained at each budget,
evaluated at that budget. This is what CSR Table 4 reports and the only fair
way to place our methods beside theirs: a CSR model trained at K=8 has only
ever been optimised for 8 active dimensions.

**E2 — one checkpoint, many budgets.** Reuses the E1 checkpoints at the anchor
budgets and evaluates them at every budget in the grid. No extra training.
This measures a claim both papers make but neither tests — CSRv2 §3: *"Training
with the multi-TopK loss L(k)+L(4k)/8 ensures that CSR could generalize to
different k's at test time"* — while the CSR author writes in issue #17 that
changing k at inference *"might lead to sub-optimal performance"*. Because
post-ReLU a code can have far fewer than K nonzeros, every E2 point carries
its **measured** nnz alongside the nominal budget.

| backbone | d | recipe | preset | K budgets | E2 anchors | cells |
|---|---|---|---|---|---|---|
| `ff2048` | 2048 | MRL | Table 3 | 2,4,…,2048 | 8…256 | 165 |
| `rn18` | 512 | CSRv2 | Table 9 | 2,4,…,512 | 8…256 | 135 |
| `convnext_tiny` | 768 | CSRv2 | Table 3 | 3,6,…,768 | 3…96 | 135 |
| `vit_b16` | 768 | CSRv2 | Table 3 | 3,6,…,768 | 3…96 | 135 |
| `dinov2_vits14` | 384 | CSRv2 | Table 9 | 3,6,…,384 | 3…96 | 120 |

690 cells = backbone × K × 5 methods × 3 seeds (1227, 2026, 2027).
K never exceeds the backbone's own width, so every point is a compression.
The 768-d backbones are treated like FF2048, the 384-d one like RN18.

### Methods

| method | K during training | gates/step | regulariser | extra reconstruction |
|---|---|---|---|---|
| `csr` | fixed | K, 4K, K | γ · `inbatch_cl` | — |
| `csrv2` | annealed | k, 4k, k | γ · `label_based_cl` | — |
| `mpsae` | fixed | K, 4K, 2K | 2 · OT gap | — |
| `mpsaev2` | annealed | k, 4k, 2k | 2 · OT gap | — |
| `csr_multitopk` | fixed | K, 4K, 2K | γ · `inbatch_cl` | L(2K)/8 |

Shared objective: `L(k) + 0.125·L(4k) + (1/32)·NMSE_aux + γ·L_reg`.

Every method gates exactly **three** times per step, so `dead_threshold=30`
means the same thing for all of them. For the baselines that is the released
sequence verbatim: CSR runs three forward passes, the third only to build a
duplicate contrastive view. Ours spend the third gate on the 2K view they
actually consume.

`csr_multitopk` is the control for the OT claim. It shares `csr`'s gates and
contrastive term and adds only `L(2K)/8`, so "MPSAE > csr_multitopk" rules out
the explanation that the gain comes from merely *looking at* a 2K view. It
follows CSRv2 App. G's "Diverse Multi-TopK" ablation in spirit.

`mpsaev2` is **variant B**: the main budget anneals, so it is `csrv2` with the
contrastive term swapped for the OT gap and the comparison isolates exactly
that substitution. Variant A (fixed main budget, annealed OT views only) is
specified in [`docs/notes/mpsaev2_legacy_A.md`](docs/notes/mpsaev2_legacy_A.md)
and is deliberately not implemented.

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
  data/          ImageNet acquisition, FFCV serialisation
  embed/         MRL and CSRv2 extraction recipes, archive writer
  evaluate/      encoding, exact 1-NN, probes, retrieval timing
  report/        aggregation, published references, style, figures
csrc/            CUDA inverted index (optional secondary metric)
scripts/         pipeline.sh, setup_server.sh, capture_golden.py
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
python scripts/capture_golden.py      # reads the pre-rewrite code out of git
```

Until that file exists the OT contract tests still run; only the
bit-for-bit comparison skips.

---

## Documentation

[`docs/`](docs/README.md) has the detail: [`methods.md`](docs/methods.md) for
the objectives and the baseline findings, [`data.md`](docs/data.md) for
provenance, [`pipeline.md`](docs/pipeline.md) for running and debugging,
[`outputs.md`](docs/outputs.md) for the result schema, and
[`server.md`](docs/server.md) for GPU setup.

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
