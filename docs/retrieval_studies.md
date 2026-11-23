# Two MPSAE ImageNet retrieval ablations

The supported configuration is `configs/retrieval_studies.yaml`. It runs only
MPSAE, with frozen ResNet-50 (`ff2048`, MRL fixed-feature BlurPool, 2048-D) and
ViT-B/16 (`vit_b16`, timm augreg, 768-D) cached global image embeddings. No
segmentation, CSR/CSRv2 fits, dense baseline, linear probing or timing loop runs.

| Study | Backbone / train K | Transport mass s | Weight lambda | Batch | Fits |
|---|---|---|---|---|---|
| Mass | ResNet-50: 8,32; ViT-B/16: 12,48 | .2,.4,.6,1 | 2 | 1024 | 16 |
| Weight / baseline optimizer | ResNet-50: 8,32 | 1 | .1,.5,1,2,2.5 | 4096 | 10 |

Total: **26 independent fits and 26 matched-K evaluations**, seed 1227. There
are no extra batch-1024 weight sweeps. Each fit uses 10 epochs, 8192 latents,
Adam LR 4e-5, betas (.9,.999), epsilon 6.25e-10 and coupled weight decay 1e-4.
The larger-batch study matches the ResNet-50 CSR/CSRv2 optimizer recipe. The
actual batch size is explicitly overridden in the trainer and recorded in the
checkpoint identity; the standard MPSAE batch remains 1024 elsewhere.

Equal epochs do not mean equal update counts: with 1,281,167 rows and no dropped
tail, batch 1024 gives 1252 updates/epoch (12520 total), while batch 4096 gives
313 updates/epoch (3130 total). No LR scaling or extra epochs are introduced.
The reported comparison is an epoch-matched optimizer-recipe ablation.
Auxiliary dead-unit counters remain defined per step, as in existing training;
their effective number of seen examples therefore changes with batch size too.

The OT group size stays 32 and entropy coefficient stays .2. At s<1 the existing
partial Greenkhorn solver applies; at exact s=1 the shared balanced log-Sinkhorn
solver applies. No reference-gap normalization by mass is added. Raw losses
across s have different entropy terms; the comparison metric is retrieval
accuracy. All-zero query codes count as errors, and measured gallery nnz is
reported alongside accuracy.

## Completely new RTX 5090 server

Use a Linux image with Python 3, git and an NVIDIA driver supporting CUDA 12.8.
Push the updated code to GitHub's refactor branch, then run:

```bash
curl -fL https://raw.githubusercontent.com/thienquang204/MPSAE/refactor/scripts/retrieval_server_onefile.py -o run_retrieval.py
python3 -u run_retrieval.py --workspace /workspace/retrieval-studies
```

The single launcher clones the repository and creates `.venv-retrieval` using
Python 3.11. It installs torch 2.9.1 from the CUDA 12.8 index plus only the
dependencies in `requirements-retrieval.txt`. It tests real GPU kernels and
backpropagation before downloading embeddings or training. See the official
[PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/) and
[Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).
It installs neither the CUDA compiler toolkit nor torchvision/FFCV/FAISS.
The server provider manages the host driver.

Cached ImageNet embeddings are downloaded from `Lukita/MPSAE-embeddings` by
default, using download-only mode. `--embedding-repo owner/name` selects a
different compatible repository; private bundles require read access. There
are **no uploads**, no raw ImageNet downloads, no backbone extraction and no
backbone fine-tuning. Missing cached bundles fail instead of starting extraction.
Weights/results are saved locally. No tmux is used; keep the terminal connected.

Change a suite setting without editing code, for example:

```bash
# Inspect the plan (environment setup may still run on a fresh machine).
python3 -u run_retrieval.py --workspace /workspace/retrieval-studies --plan
# Reuse locally downloaded embeddings without HF access.
python3 -u run_retrieval.py --workspace /workspace/retrieval-studies --local-embeddings
# Only one study:
python3 -u run_retrieval.py --workspace /workspace/retrieval-studies --studies optimizer
# Existing code, Python environment and precomputed features:
python -m mpsae.benchmark.retrieval_studies --source-root work/direct-v1 --workspace work/retrieval-studies
# Regenerate plots and tables without loading data or training:
python -m mpsae.benchmark.retrieval_studies --workspace work/retrieval-studies --export-only
```

Pass `--source-root /existing/direct-v1 --local-embeddings` to the standalone
launcher to reuse old embeddings and compatible source fits. Exact data,
optimizer, mass, weight, seed and schedule identities must match. The old
1024-batch checkpoint cannot continue as a 4096-batch fit. Repeating the same
command resumes its own interrupted fits/evaluations. Existing repository
checkouts update only when `--update` is explicitly passed.

## Efficiency and scientific consistency

- Load, validate and memory-map each backbone's embedding archives once across
  both studies, rather than hash/reopen them for every weight. Keep at most one
  backbone's mappings in the suite. Compressed NPZ conversion is cached; suitable
  uncompressed NPZ archives are mapped directly by the existing loader.
- `--residency auto` uses GPU-resident training embeddings when enough memory
  remains, otherwise host RAM or memory-mapped batches. On a free RTX 5090 the
  ResNet-50 training matrix is about 9.77 GiB; the selector reserves 12 GiB of
  headroom before choosing GPU residency. Other GPU processes can change this.
- Existing batched OT and shared Top-K calculations are reused. Solver group
  size, FP32 training, disabled TF32 and the existing encoding/search precision
  stay fixed; no ANN approximation is introduced.
- Exact 1-NN evaluates all 50,000 validation queries against the complete
  1,281,167-row train gallery. Default query blocks are 2048 and gallery chunks
  32768; these bound memory without removing examples. Override them if needed.
- Save checkpoints every 512 steps and metrics every 100, plus final checkpoints,
  reducing disk writes and synchronization. A crash may replay up to 511 steps.
- Separate fit/evaluation identities reuse any overlapping conditions. Only
  completed compatible artifacts are reused; no result-based early stopping or
  best-weight selection. Plots are regenerated after each campaign or on failure.

Peak memory and speed on RTX 5090 require a real server run; CPU smoke results
are not timing estimates. The input is cached embeddings; system RAM can fall
back to mapping, but fast local SSD storage is preferable for caches/checkpoints.

## Results

Download `/workspace/retrieval-studies/retrieval-studies-results.zip` directly.
The same archive is retained under `studies/<suite-id>/` so later settings do
not overwrite historical studies. `studies-latest.json` identifies that archive.

- `mass_accuracy.pdf/.png`: 1-NN accuracy vs mass; one panel per backbone,
  one curve per K. No published reference curves.
- `weight_accuracy.pdf/.png`: ResNet-50 accuracy vs lambda at batch 4096,
  one curve per K. No extra batch-1024 weight results are generated.
- `mass_table.tex`, `optimizer_table.tex`: booktabs tables with accuracy,
  actual nnz, mass, weight, batch size and completed/expected seeds.
- `results.csv/.json`: every requested condition, status, metrics and checkpoint
  identity; `summary.csv` and `*_summary.csv`: seed aggregation and update counts.
- `plan.json`, per-campaign provenance and `run-state.json`: resolved settings,
  code/runtime metadata and failure information. Missing values are blank/--.

Figures use vector PDF and 300-dpi PNG; multiple seeds show sample SD, one seed
has no error bar. The default is single-seed sensitivity analysis, not a
significance claim. Report all weights rather than choose the best validation
result and describe it as an independently evaluated improvement. The archive
contains no large checkpoints or embeddings; retain the workspace to resume.
