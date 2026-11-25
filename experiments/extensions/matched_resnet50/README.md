# Matched training settings: CSR, CSRv2 and MPSAE on ResNet-50

This standalone extension compares the three methods under identical optimizer
settings **within each training-batch group**. It does not change the frozen
CVPR 2027 catalog or launch segmentation/downstream experiments.

For inference at multiple K values using the completed checkpoints on the same
server, see [Cross-K inference](CROSS_K.md). This optional extension selects
train K=8 and 32 and inference K=2, 8, 32, 128; it adds no training or probe fits.

For the separate MPSAE learning-rate comparison at batch 4096 (4e-5 versus
8e-5, alongside existing CSR/CSRv2), see [Learning-rate ablation](LEARNING_RATE.md).
One command trains only the new condition and packages the comparison results.

## Experiment plan and counts

| Setting | Value |
|---|---|
| Frozen backbone | ResNet-50 FF2048, the existing MRL fixed-feature embedding source |
| Data | All 1,281,167 ImageNet train embeddings; 50,000 validation embeddings |
| Methods | CSR, CSRv2, MPSAE |
| Seeds | 1227, 2026 |
| Training batch sizes | 1024, 4096 |
| Train K | 2, 8, 32, 128, 256, 512 |
| Inference K | Matched to each checkpoint's train K |
| SAE epochs / learning rate | 10 / constant 4e-5 |
| SAE architecture | Shared tied dictionary, 2048 input, 8192 latent dimensions |
| Optimizer | Adam; betas (0.9, 0.999), epsilon 6.25e-10, coupled weight decay 1e-4 |
| Training precision | Float32, TF32 disabled; no AMP or gradient accumulation |
| Linear probe | Only K=32; same head recipe for all 12 selected checkpoints |
| Probe settings | 10 epochs, Adam lr=1e-3, batch=4096, weight decay=0; zero initialization |
| Timing | Query batches of 512; 5 warm-up and 5 measured rounds |

The shared optimizer applies to **SAE training**. The linear classifier has its
own fixed optimizer, held constant across both SAE batch sizes and all methods.
Validation labels only score the final models; no validation-based checkpoint,
epoch, learning-rate or K selection occurs.

| Work | Calculation | Count |
|---|---|---:|
| SAE fits | 3 methods x 2 batches x 2 seeds x 6 K | **72** |
| Matched 1-NN accuracy evaluations | One per SAE | **72** |
| Linear-head fits and accuracy evaluations | 3 x 2 x 2 at K=32 | **12** |
| Total accuracy evaluations | 72 + 12 | **84** |
| Retrieval / classifier timing sessions | Attached to those evaluations | **72 / 12** |

The original low-sparsity group K={2,8,32} costs 36 SAE fits and 48 accuracy
evaluations, including the probes. Extending to K={128,256,512} adds **36 fits
and 36 evaluations**. Timing repetitions are measurements, not extra models.
There is no 6-by-6 cross-budget inference sweep and no extra dense model fit.

At 10 epochs there are **12,520 updates** for batch 1024 and **3,130 updates**
for batch 4096. Equal epochs mean equal training examples, not equal update
counts across batch sizes. Within a batch group, methods share the same seeded
initialization, epoch permutations, final partial batches, and update count.
Method-specific losses, supervision and CSRv2's K annealing remain intact.
The contrastive terms still see the full selected batch; gradient accumulation
would change their negative sets and is deliberately not substituted.

MPSAE retains the existing objective: mass s=0.9, loss weight **2.0 for K<=64**
and **0.5 for K>64**, OT microbatch=32, eta=0.2 and existing solver tolerances.
These values are explicit in the plan and receipts. This tests matching the
training schedule, not changing every method into the same objective.

## Efficient execution

1. Download or reuse the single ResNet-50 embedding pair, verify/map it once,
   then reuse it throughout the study. No ImageNet images or backbone extraction.
2. Run one SAE at a time. Training uses the existing automatic feature residency
   selection. Checkpoints save every 512 updates; progress logs every 100.
3. Encode each train/validation split once per SAE, as compact coordinate/value
   arrays. Reuse these arrays for accuracy, timing and the optional probe.
4. Build one FP32 sparse CSR gallery in bounded shards. Use that same index and
   exact maximum-inner-product search for all validation queries and timing.
   The full gallery is retained; no approximate search or gallery subsampling.
5. Retain the fitted K=32 probe head for classification profiling, then free
   GPU work buffers and temporary codes before the next checkpoint.
6. Save raw JSON and CSV after every cell. Generate plots and the download ZIP
   at completion or a handled interruption, avoiding repeated figure rendering.
7. Resume compatible completed cells without fitting or evaluating them again.
   Training identities include optimizer/batch/seed/data; performance receipts
   additionally identify the GPU, runtime, search settings and implementation.

This is a fresh standalone study. Historical main-paper checkpoints are not
automatically imported: their training-memory measurements may not exist.
Previously completed cells in this study are reused. Missing performance data
remain missing instead of being inferred from a checkpoint. Use a new workspace
when measuring the entire training cost on different hardware.

## What is measured

- **Training:** trainer-reported fit seconds, number of updates, effective
  residency, stage peak allocated/reserved PyTorch GPU bytes, sampled process
  RSS, and wall time including initialization/checkpoint work.
- **Encoding:** wall time, actual mean nnz, empty-code counts and compact-code
  bytes. ReLU Top-K yields **at most** K active entries, not necessarily K.
- **1-NN:** full-validation top-1, index bytes, index-build time/memory,
  accuracy-pass time/memory, and steady-state search latency/throughput.
  Empty queries count as errors; score ties choose the first gallery row.
- **Linear probing:** final top-1/top-5, head-fit time, fit-plus-validation peak
  memory, head bytes, and classifier inference latency/memory at K=32.
  Classification timing includes compact-code expansion, head forward and
  argmax. It is classifier inference, not nearest-neighbor retrieval.

Retrieval timing excludes SAE encoding, index construction and host-to-device
transfers. Five evenly spaced full query batches are cycled deterministically
when available. `us/query` is **amortized at query batch 512**, not single-query
latency. CUDA events and synchronized wall-time samples are both saved; raw
samples' summaries include median and p90. Five repeats are a short benchmark,
so p90 is a coarse diagnostic. Training seeds and timing repeats are different
sources of variation and are never pooled as independent seeds.

GPU peaks are PyTorch allocator measurements, not whole-device `nvidia-smi`
usage. Process RSS is sampled every 100 ms and may miss brief host peaks. Stage
receipts record baselines and absolute peaks. Reserved memory is recorded but
not confused with the compact index's storage size. The retrieval backend is
explicitly recorded; this extension uses the existing sparse search for both
accuracy and timing, whereas the older main accuracy path uses blocked dense
GEMM. Floating-point near-tie behavior may differ between these kernels.

Do not reduce the training batch or gallery automatically on OOM: that would
change the comparison. `--residency cpu` can move the frozen training embeddings
out of VRAM without changing batches or the loss. Keep that policy the same
for every compared method and record it. The resident sparse index itself is
about **4.89 GiB at K=512** if all K entries are active, plus query/work buffers.
The raw float32 training embeddings occupy about **9.77 GiB**. These are storage
calculations, not a guarantee of total GPU peak usage; measured peaks are saved.

## Fresh server setup and run

From the already-cloned repository:

```bash
bash scripts/setup_matched_resnet50.sh
source .venv-retrieval/bin/activate

# CPU-only plan: verify 72 SAE fits and 84 accuracy evaluations.
python -m mpsae.benchmark.matched_study --plan

# Cached embeddings download once if needed; no result uploads or tmux.
bash scripts/run_matched_resnet50.sh \
  --workspace /workspace/matched-resnet50 \
  --source-root /workspace/MPSAE/work/direct-v1 \
  --download-embeddings
```

Replace the source-root path if your clone is elsewhere. If embeddings already
exist, omit `--download-embeddings`. `--source-root` points at **direct-v1**, not
its parent. A read token is only needed if the selected embedding repository
requires it; results are always saved locally.

Rerun the same command to resume. Configuration lives in
`configs/matched_resnet50.yaml`; its content determines the study directory.
`--device cuda:0`, `--cpu-threads 4` and `--residency auto` are configurable.
The setup installs only cached-embedding dependencies; it does not install
image extraction packages. Run one study process per GPU/workspace to keep
timing and memory measurements comparable.

```bash
# Rebuild tables/plots from saved receipts; no GPU or checkpoint loads.
bash scripts/run_matched_resnet50.sh \
  --workspace /workspace/matched-resnet50 --report-only

# Reprofile existing fits; no SAE optimizer steps allowed.
bash scripts/run_matched_resnet50.sh \
  --workspace /workspace/matched-resnet50 \
  --source-root /workspace/MPSAE/work/direct-v1 --evaluate-only --retime
```

`--evaluate-only` requires completed compatible SAEs. A missing probe head can
still be fitted because it is part of evaluation. `--retime` reruns accuracy
and performance profiling, reusing compatible SAE and probe weights.

## Outputs and interpretation

`work/matched-resnet50/latest.json` identifies the selected study. With the
absolute workspace above, outputs are:

```text
/workspace/matched-resnet50/
  fits/<fit-id>/                 # SAE checkpoint, training logs, memory profiles, probe heads
  studies/<study-id>/
    plan.json, runtime.json
    cells/<batch-method-K-seed>/result.json
    results.json
    reports/
      results.csv               # 72 per-seed rows with accuracy and resource measurements
      accuracy.csv              # 84 metric rows, including 12 probes
      summary.csv               # mean, sample std and n for each metric
      paired_differences.csv    # MPSAE minus each baseline, paired by seed
      retrieval.tex, linear_probe.tex
      *.pdf, *.png              # vector and 300-dpi plots
    matched-results.zip
```

Plots cover accuracy versus K, retrieval latency, index and peak GPU memory,
training time/memory, accuracy-cost tradeoffs, and a K=32 probe comparison.
Error bars are sample standard deviations across the two training seeds.
Missing seeds produce a smaller recorded n and no invented zero standard
deviation. With two seeds, describe variability; do not claim robust statistical
significance. Each table contains per-metric sample counts in its companion CSV.

Download `matched-results.zip` directly from the server. It contains results,
tables, figures and provenance; large checkpoint weights and feature matrices
stay in the workspace. No Hugging Face publication is performed.

## Validation before production

The new tests cover counts, actual batch overrides, matched initialization/data
order, sparse versus dense search including ties/empty queries, probe-head
equivalence, incomplete-checkpoint protection, full synthetic smoke runs,
resumption and exports. CPU smoke outputs live in `SMOKE_ONLY` and are marked
synthetic; they do not claim GPU timings or scientific accuracy.

Checked on 2026-11-25: 14 new tests and 52 existing timing/cache tests passed;
one CUDA-only test was skipped on the CPU verification machine. Both generated
LaTeX tables compiled. Full ImageNet training and GPU profiling are not yet run.
The only shared numerical-module edit adds an optional `return_head` result to
the existing probe API; its default training/scoring behavior is preserved.
The paper source checksum was refreshed for that compatible API addition.

```bash
python -m pytest tests/test_matched_study.py -q
```
