# MPSAE learning-rate ablation at batch 4096

This standalone extension reads a completed matched ResNet-50 study and compares
four conditions at the same training batch size:

| Condition | SAE learning rate | Training |
|---|---:|---|
| CSR | 4e-5 | Existing checkpoint |
| CSRv2 | 4e-5 | Existing checkpoint |
| MPSAE | 4e-5 | Existing checkpoint |
| MPSAE | 8e-5 | New fit from the same seeded initialization |

The new condition uses **batch 4096, 10 epochs, seeds 1227 and 2026**, and
**train K = 2, 8, 32, 128, 256, 512**. Initialization, example ordering,
constant-LR Adam settings, precision, and losses remain the same. This is not
continued training of the old checkpoints. The baseline remains read-only.

The MPSAE loss retains the K and 4K reconstruction terms, auxiliary term and
MMPOT matching gap on K/2K/4K views. Transport mass is 0.9, entropy coefficient
0.2, group size 32, initial iteration cap 400, refinement cap 8000 and feasibility
tolerance 1e-4. Its weight remains 2 for train K <= 64 and 0.5 above that threshold.
All original baseline recipes are checked against the current recipe before
training; an incompatible recipe fails rather than silently changing the study.

## Run on the existing server

After pushing the updated code, use the environment and downloaded embeddings
from the completed matched study:

```bash
cd /workspace/MPSAE
git pull --ff-only origin refactor

bash scripts/run_matched_lr_ablation.sh \
  --workspace /workspace/matched-resnet50 \
  --source-root /workspace/MPSAE/work/direct-v1
```

That single command trains, evaluates, compares and packages the results. There
is no Hugging Face upload, no new dependency, and no baseline training. The shell
script uses `.venv-retrieval/bin/python`, falling back to `python`; set
`MPSAE_PYTHON=/path/to/python` to override. Replace the two workspace paths if the
original run used different locations. The original workspace must still contain
`fits/`, `studies/` and `latest.json`, not just the downloaded results ZIP.

Optional inspection without training or loading embeddings:

```bash
bash scripts/run_matched_lr_ablation.sh \
  --workspace /workspace/matched-resnet50 --plan
```

Use `--study-id ID` to choose a specific saved original study. The new study
inherits the saved K list, seeds and evaluator settings, including 5 warm-up and
5 measured timing rounds under the standard configuration. It requires the
original three-method, 10-epoch, 4e-5 comparison and its batch-4096 checkpoints.

## Evaluation counts

| New work for MPSAE at 8e-5 | Count |
|---|---:|
| SAE training fits | 12 |
| Matched-K 1-NN scores | 12 |
| Off-diagonal 1-NN scores | 12 |
| Linear-probe fits/scores at K=32 | 2 |
| New accuracy evaluations | **26** |

Cross-K inference selects **train K=8 and 32**, each evaluated at
**inference K=2, 8, 32, 128**. Its four matched points are already counted above.
These cross-K budgets can be changed with `--train-k` and `--inference-k`;
`--train-k` selects cross-K anchors and does not restrict the six new training K.

The four-condition comparison has **96 unique 1-NN scores and 8 probe scores**,
including reused baselines. If the previous cross-K extension already evaluated
the batch-4096 baselines, its compatible results are reused by checkpoint hash.
Otherwise, up to **36 baseline off-diagonal evaluations** run without training.
Thus a source containing only matched results needs up to **62 new accuracy
evaluations**: 26 for the new LR condition plus 36 baseline cross-K points.
No batch-1024 checkpoints are trained or evaluated by this runner.

## Resume, compatibility and resource use

Re-run the same command to resume. New fits use the existing checkpointed trainer;
completed fits, probes and inference points are reused. New checkpoints are
stored in a separate nested workspace, with learning rate in their identity.
Original fit IDs are read from saved records, independent of Git commit changes.

All baseline checkpoint checksums and completed evaluations are checked before
new training. Missing baselines cause an error. Existing cross-K accuracy can be
reused only when its checkpoint, embedding identity and numerical evaluator match.
Historical performance measurements from another hardware/software runtime remain
in the raw receipts but are blank in comparison tables and plots; accuracy is
retained. Numerical evaluator changes require compatible baseline evaluation.
Once a new study starts, changed numerical code/runtime is rejected on resume.

Cached embedding files are reopened via the existing loader; no image extraction
is performed. New training and inference are sequential, with one checkpoint and
one pair of compact gallery/query codes active at a time. Temporary codes are
removed after each budget. The encoder is moved to CPU before sparse retrieval.
Five warm-ups and five timing rounds are unchanged, and timing excludes SAE
encoding, index construction and query transfer, as in the original evaluator.

## Results and direct download

Outputs are separate from the original study:

```text
/workspace/matched-resnet50/lr_ablation/<study-id>/
  plan.json, session.json, state.json, results.json, report_status.json
  training/                     # new MPSAE checkpoints and probe heads
  matched/                      # baseline and new matched evaluation receipts
  cross/                        # off-diagonal receipts
  losses/                       # saved MPSAE training-loss samples
  reports/
    matched_accuracy.csv        # all four conditions, each seed
    crossk_accuracy.csv         # includes matched diagonal without recomputation
    linear_probe.csv            # K=32 only
    accuracy_per_seed.csv       # unique matched and off-diagonal points
    summary.csv                 # per-metric means, sample std and counts
    paired_deltas.csv           # new minus old MPSAE for each matched seed
    paired_summary.csv          # mean/std of paired accuracy changes
    loss_samples.csv
    matched.tex, crossk.tex, linear_probe.tex  # booktabs LaTeX tables
    *.pdf, *.png                # accuracy, speed, memory, nnz and loss figures
  lr-ablation-results.zip
```

Each figure distinguishes MPSAE's two learning rates. Means and sample standard
deviations are computed across seeds; a single seed has no fabricated standard
deviation. Loss plots compare MPSAE to MPSAE only, because the other methods have
different objectives. Curves show the actual logged minibatch samples, with each
seed drawn separately against epoch progress. `loss_ot` is the unweighted gap;
`loss_total` includes the configured loss weights. Missing old logs are reported
in `results.json`, not reconstructed or invented.

CSV tables and the ZIP update after each newly completed checkpoint and cross-K
point, and on handled exits. Figures refresh at exit, including handled failure,
or by rerunning with `--report-only`. `report_status.json` identifies whether
figures lag the current tables. A partial archive is not a complete study.

Locate the archive on the server:

```bash
find /workspace/matched-resnet50/lr_ablation -name lr-ablation-results.zip
```

Then run on your local computer, using the printed path and current SSH port:

```powershell
scp -P YOUR_PORT root@YOUR_SERVER:/workspace/matched-resnet50/lr_ablation/STUDY_ID/lr-ablation-results.zip "$HOME/Downloads/"
```

The archive contains results, figures, loss samples and provenance, without
checkpoint tensors or cached embeddings. New weights remain on the server in
`training/fits/`. The latest output path is also recorded in
`/workspace/matched-resnet50/lr_ablation/latest.json`.

## Verification

`tests/test_lr_ablation.py` checks exact study counts and recipe differences,
paired-seed statistics, all-method baseline reuse, previous cross-K reuse,
fresh training only for the new LR, resume without additional work, unchanged
baseline artifacts, loss figures and checkpoint-free ZIP export. These use
synthetic CPU data; actual ImageNet results and GPU performance require the
server run.
