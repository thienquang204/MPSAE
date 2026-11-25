# Cross-K inference from completed matched-study checkpoints

Run this extension on the same server after the matched ResNet-50 study finishes.
It reads the saved study plan and checkpoint IDs; it never starts SAE training or
fits another linear classifier. Keep the existing workspace, including `fits/`,
`studies/` and `latest.json`, and the existing ImageNet embedding cache.

## Selected experiment

| Setting | Value |
|---|---|
| Backbone | ResNet-50 FF2048 |
| Methods | CSR, CSRv2, MPSAE |
| Original training batches | 1024, 4096 |
| Original seeds | 1227, 2026 |
| Selected checkpoint train K | 8, 32 |
| Inference K for each checkpoint | 2, 8, 32, 128 |
| Metric | ImageNet validation 1-NN against the full training gallery |

This selects **24 existing checkpoints** and produces **96 cross-K 1-NN points**.
Of these, 24 diagonal points already exist: they are reused when their evaluator,
runtime and timing settings match. Thus there are **72 new off-diagonal points**,
**zero new SAE fits**, and **zero new probe fits**. If the evaluator or runtime
differs, diagonal points are re-evaluated as well, without retraining.

The combined accuracy CSV retains the original 72 matched 1-NN and 12 linear-probe
scores, adds the 72 off-diagonal scores, and deduplicates the overlapping diagonal
scores: **156 unique accuracy results** when both studies are complete. The
cross-K CSV itself contains 96 rows, not 156. Missing results remain missing.

## Same-server commands

First commit/push the updated code from your development machine. On the server,
use the existing environment and workspace (adjust paths if you used others):

```bash
cd /workspace/MPSAE
git pull --ff-only origin refactor

# Optional: show the selected counts without loading models or data.
bash scripts/run_matched_crossk.sh \
  --workspace /workspace/matched-resnet50 --plan

bash scripts/run_matched_crossk.sh \
  --workspace /workspace/matched-resnet50 \
  --source-root /workspace/MPSAE/work/direct-v1 \
  --train-k 8 32 --inference-k 2 8 32 128
```

The script uses `.venv-retrieval/bin/python` when available, otherwise `python`.
Override with `MPSAE_PYTHON=/path/to/python` if needed. No installation, token,
checkpoint transfer or Hugging Face upload is required when the previous run's
environment and embedding files are intact. Use `--study-id ID` to select an older
study rather than the one in `latest.json`.

Re-run the same command after an interruption: completed compatible points are
reused. A workspace lock prevents concurrent training/evaluation in the same
matched workspace. Missing, incomplete or checksum-mismatched checkpoints cause
an explicit error; they never trigger fallback training. Changing evaluation code
or runtime after starting a cross-K sweep is rejected to avoid mixing performance
measurements. Preserve that output and use a separate workspace copy in that case.

## Outputs and download

The final output directory is printed and recorded in
`/workspace/matched-resnet50/crossk/latest.json`:

```text
crossk/<sweep-id>/
  plan.json, session.json, results.json, report_status.json
  points/<condition>/result.json
  reports/crossk.csv              # each seed, including actual nnz and timing
  reports/summary.csv             # means, sample std and observation counts
  reports/combined_accuracy.csv   # original matched + cross-K, no duplicate points
  reports/crossk.tex              # LaTeX table; requires booktabs
  reports/{top1,nnz,retrieval_us,index_gib}.{pdf,png}
  crossk-results.zip
```

CSV/JSON tables refresh after every inference point; the ZIP refreshes after each
checkpoint's sweep and on handled exits. Figures refresh at exit or with the same
command plus `--report-only`. `report_status.json` records the figure completeness
so a mid-run archive is not confused with a final report. Partial archives contain
pending rows; CPU synthetic smoke runs do not produce GPU latency figures.

To locate the archive:

```bash
find /workspace/matched-resnet50/crossk -name crossk-results.zip
```

Copy the printed path from your local computer using your current SSH port:

```powershell
scp -P YOUR_PORT root@YOUR_SERVER:/workspace/matched-resnet50/crossk/SWEEP_ID/crossk-results.zip "$HOME/Downloads/"
```

The archive contains tables, figures and evaluation receipts, not model weights.
All original checkpoints, matched-study reports and probe results remain intact.

## Evaluation and resource use

Every budget re-encodes both gallery and queries with that checkpoint's inference
Top-K gate and normalization. Selecting more coordinates may retain zero ReLU
activations, so K is an upper bound on nnz, not a promise of exactly K activations.
Changing inference K does not change learned weights or the training loss recipe.
At K=2 the per-seed CSV also reports percentages of query codes with 0, 1 and 2
nonzeros; these help diagnose instability without asserting a cause in advance.

Embedding arrays are opened once. One checkpoint and one pair of compact code
arrays are processed at a time; temporary codes are removed after each point.
The SAE is moved to CPU before retrieval, leaving the GPU for the sparse index
and query blocks. The CPU retains that model between inference budgets. Retrieval
uses the original evaluator, including timing batch size, warm-up and repetitions;
encoding and index building are excluded from search latency. Full receipts record
encoding, index construction and memory separately. No full cross-K grid is trained.

Figures show mean and sample standard deviation across seeds, separated by
training batch and train K. A single observation has no fabricated standard
deviation. Existing linear-probe scores remain matched-K scores only; this
extension does not evaluate those heads on different inference budgets.

## Verification

`tests/test_matched_crossk.py` exercises all three methods on synthetic data,
checkpoint lookup independent of obsolete absolute paths, diagonal reuse,
cross-K evaluation, resume, combined-table deduplication, archive generation,
checkpoint integrity and absence of training/probe calls. Production GPU speed
and ImageNet accuracy must be measured on the server.
