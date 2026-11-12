# Training and retrieval performance

The optimized implementation retains the configured training batches, input row
order, three activity-counter updates, AuxK timing, K schedules, OT groups,
objective coefficients and FP32 training/TF32 policy. It consumes the existing
frozen embeddings downloaded from Hugging Face. It does not add image extraction
to the training loop.

## Training changes

- Gate selections are reused by activity counters instead of recomputing TopK.
  Each distinct requested K still has its own selection, including tie behavior.
- CSR's two identical contrastive views are folded into B x B logits with
  the correct duplicate multiplicities. The generic 2B x 2B reference remains
  available for value/gradient tests. Zero codes use their actual diagonal score.
- Full CUDA OT groups use batched cost, gap, entropy and packed diagnostics.
  The Greenkhorn solver was already batched; its iteration arithmetic and
  feasibility checks remain. Negative-gap refinement is batched as well.
- Optional metrics are collected at log/checkpoint/final steps. Numerical
  finite/feasibility/negative-gap guards still run before an update is accepted.
- GPU-resident features upload the same NumPy permutation once per epoch.
  Labels are not materialized for methods that do not use them. CPU residency
  explicitly copies the feature matrix instead of potentially aliasing a memmap.
- Gradients are cleared before the next forward pass. Checkpoints/summaries
  record `runtime_history`, including the optimizer step where a runtime changes.

Completed fits remain reusable. Resuming a partial fit keeps its optimizer/RNG
state, while runtime history identifies the continuation. Floating-point
reduction ordering can differ between equivalent kernels; these optimizations
do not promise bitwise equality across implementations or GPU architectures.

The default training cadence remains 10 steps per metric log and 100 per
checkpoint. Both are now explicit in the campaign YAML:

```yaml
execution:
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto  # auto, cuda, cpu, mmap
```

Increasing checkpoint intervals can save IO but increases work lost after an
interruption. The implementation does not silently increase the interval, use
mixed precision, change batch size, or weaken OT convergence checks. Whole-step
compilation, custom training kernels and asynchronous checkpointing are not
enabled by this change. `compile_model` remains a reserved legacy field.

## Retrieval K=2048 on a 32 GB GPU

The old builder accumulated GPU chunks and concatenated a full int64 CSR index.
With 1,281,167 rows and 2048 actual nonzeros per row, the index alone was about
29.33 GiB; concatenation temporarily retained both copies. Searching a batch of
512 also created a 2.44 GiB score matrix.

The new `resident-sharded-csr-v1` backend:

1. Counts actual nonzeros with bounded scratch, then checks the full resident
   index plus build/search working set against available VRAM and a reserve.
2. Preallocates individual CSR shards and fills them in bounded entry blocks.
   Local int32 offsets avoid the full-gallery int32 overflow at 2.62 billion
   nonzeros. No global GPU concatenation occurs.
3. Keeps the complete gallery resident and reduces score tiles to global top1,
   preserving first-index ties. The default score tile is 32768 x 512, or 64 MiB.
4. Emits build, warm-up and timed-round progress/ETA outside measured intervals.

At full support the shard payload is approximately 19.55 GiB plus small row
pointers. Actual use depends on nnz and other GPU residents/workspaces. If it
does not fit, timing reports an explicit skip/error; it does not silently
subsample the gallery, lower precision, or stream host data into a resident
timing measurement. Sparse and dense timing backends both tile and merge top1;
their speedup helper checks compatible timing settings and runtime/device
fingerprints (PyTorch/CUDA versions, device name, capability, VRAM and UUID when
available).

```yaml
imagenet:
  timing: true
  timing_query_batch: 512
  timing_warmup: 5
  timing_rounds: 5
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
```

The default schedule is now 5 warmup searches and 5 timed searches. The main
campaign keeps `timing: false`; the example above explicitly enables it. These
are quick estimates with fewer samples than the previous 100/2000 schedule.
The settings are persisted in timing identity, so old timings are not reused
as 5/5 measurements. Standalone sparse and dense timing functions also default
to 5/5; the old 100/2000 protocol remains available by explicit arguments.
Only retrieval timing changes: training epochs, learning rates, checkpoint
frequency and OT solver iterations are unchanged. Smaller gallery chunks bound score memory; a smaller build entry
cap bounds sorting scratch. Neither changes the number of gallery rows.

Accuracy and probe results are persisted before timing starts. On interruption,
retrying the same campaign reuses these results and retries the timing stage
(encoding may repeat when transient code caches were removed). A skipped timing
is reusable under identical timing settings unless `timing_retry_skipped` is
true. Changing timing version/settings or runtime/device invalidates timing only; old trained
models and compatible accuracy remain reusable. Rebenchmark both sparse and
dense backends when comparing speedups; do not combine old monolithic timings
with new sharded timings.

## Measure on the GPU host

### Additional runtime cleanup (2026-11-12)

The main and fine-tuning experiment grids, losses, precision, batch sizes,
epochs, learning rates, OT convergence checks and checkpoint intervals remain
unchanged. No CLIP/VOC campaign is enabled by these optimizations.

- Fine-tuning computes optional loss/sparsity/OT diagnostics only at log steps
  and the final step. The joint loss and gradient finite checks still run every
  step. Progress reports identify the last logged loss rather than presenting
  it as the current step's loss. Joint scalar diagnostics use one packed host
  transfer instead of four separate transfers.
- Completed fine-tuning checkpoints restore only backbone/SAE weights for
  evaluation. They no longer construct a classifier or restore Adam state onto
  CUDA. The existing combined checkpoint is still read on CPU and fully checked;
  there is no new checkpoint format or reduced integrity validation.
- Frozen control SAEs remain on CPU during training and move to CUDA only if
  an uncached evaluation needs them. The teacher returns to CPU after its shared
  feature extraction; the next fit restores it to the training device. The
  fine-tuned backbone is released after extraction, before sparse search/probes.
  Final training gradients are cleared before returning to evaluation.
- Image loaders keep shuffle indices in a compact NumPy array and sequential
  evaluation indices in a `range`. The torch permutation, augmentation seeds,
  batch composition and resume offsets are unchanged. On this CPU environment,
  seven trials with 1,281,167 rows and one torch thread gave median index-build
  times of 36.1 ms (Python list) versus 15.8 ms (NumPy view); index payload falls
  from 44.0 MiB to 9.8 MiB. These are index-only measurements, not training speedups.
- Memory-mapped embedding batches sort row positions once and build the inverse
  permutation in linear time, replacing three sorts while restoring identical
  row order before the loss, including contiguous OT groups.
- Sparse encoding counts active entries from the values already copied to the
  host for storage. This removes four scalar device-to-host synchronizations per
  batch without changing codes, Top-K tie behavior, normalization or statistics.
- Feature extraction holds one inference precision context per split instead
  of entering it for every batch.

Tests compare detailed/quiet fine-tuning losses, gradients and activity counters,
exact interrupted resume, completed-run reuse, nine synthetic training/evaluation
cells, feature row order and sparse encoding/search. CUDA versions of the joint
gradient and nine-cell integration tests skip when CUDA is unavailable. To run
these checks on the server before a long campaign:

```bash
python -m pytest -q tests/test_finetune.py tests/test_data_performance.py tests/test_evaluate.py tests/test_training_performance.py
```

The full-gallery exact FP32 search, linear-probe training and converged OT solver
remain substantial workloads. This change does not replace them with approximate
search, TF32, fewer examples or fewer solver iterations. GPU throughput and peak
VRAM savings must be measured on the experiment host; local CPU tests cannot
establish an RTX 5090 speedup.

The development environment has CPU-only PyTorch. Numerical and orchestration
tests are available, but no RTX 5090 speedup or peak-memory measurement is claimed.
Local validation on 2026-11-10 used Python 3.13 and PyTorch 2.9.1+cpu:
the full suite reported **530 passed, 8 skipped, 1 failed**. The failure is
`tests/test_ot.py::test_matches_the_pinned_legacy_numerics`: its `small` case
returns `0.41512954235076904` versus the stored `0.4151325821876526`
(absolute tolerance `1e-6`). The untouched implementation produced exactly
the same failure in the same environment. The fixture and tolerance were left
unchanged; CUDA equivalence and this historical fixture still need validation
on the experiment host. On Windows, use a short pytest `--basetemp` path because
the integration tests create nested cache hashes that can exceed MAX_PATH.

Run the CUDA parity checks before a long campaign:

```bash
python -m pytest -q tests/test_training_performance.py tests/test_ot_performance.py tests/test_timing_performance.py
```

The standalone profiler does not write to existing fit/checkpoint directories:

```bash
python -m mpsae.profile_train --backbone ff2048 --method mpsae --k 8 --device cuda:0 --steps 50 --warmup 10 --out profile-mpsae-k8.json
```

Without an embeddings path this is labeled synthetic; use `--embeddings-dir`
and `--cache-dir` to exercise local frozen embeddings. Add `--trace` with a
separate output path for a short CPU/CUDA operator trace. The traced work is
separate from the reported benchmark window. See `--help` for checkpoint and
annealing-step options.

Record samples/s, wall/CUDA step p50/p95, allocated/reserved peaks, refinement
counts and data residency. Monitor `nvidia-smi` separately. VRAM occupancy is
not compute utilization; target useful throughput with memory headroom rather
than filling every available byte.
