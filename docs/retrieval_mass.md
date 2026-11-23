# ImageNet transport-mass ablation

For the combined mass study and batch-4096 ResNet-50 weight study with plots,
tables and a one-file RTX 5090 setup, use [retrieval_studies.md](retrieval_studies.md).

This independent runner trains MPSAE with selectable transport masses, then
uses the existing exact ImageNet 1-NN evaluator on the sparse latent codes.
The main retrieval configurations, CSR and CSRv2 are unaffected. All results
stay on local disk; nothing is uploaded to Hugging Face.

## Default experiment

`configs/retrieval_mass.yaml` selects ResNet-50 (`ff2048`) at K=8,32 and
ViT-B/16 (`vit_b16`) at K=12,48, with s=0.2,0.4,0.6,1.0 and seed 1227: **16 fits and
16 matched-K evaluations**. Compatible existing source fits/results are reused.

The SAE recipe remains 10 epochs, batch size 1024, Adam with LR 4e-5, hidden
dimension 8192, OT entropy coefficient 0.2 and OT groups of 32 examples. The
existing loss-weight rule remains: lambda=2 for training K<=64, lambda=0.5 for
K>64. The resolved plan lists the actual preset for each condition. No linear
probe, downstream task or repeated retrieval timing runs in this ablation.

`s` is transported mass, not the loss weight `lambda` or entropy coefficient
`eta`. At s=1 the three marginals equal 1/n, slack entropy is zero, and a
batched cyclic log-domain Sinkhorn solver optimizes the full joint coupling.
The loss is still reference-minus-optimal **matching gap**, with gradient
J-P*. Partial masses retain the historical Greenkhorn solver and numerics.
The balanced solver checks equality residuals and primal/dual objective
consistency, performs bounded float64 refinement if needed, and fails on
unconverged/nonfinite results. Finite-iteration residuals are recorded; a
small primal/dual discrepancy is not claimed as a rigorous rounded-plan
optimality certificate.

See [Piran et al., Definition 3.1](https://arxiv.org/pdf/2405.19532) and
[Lin et al.](https://arxiv.org/pdf/1910.00152). Use the consistent variational
objective with h(P)=sum P(log P-1); raw loss values across masses include
different reference/slack entropy terms and are not comparable accuracies.

## Run with existing embeddings

Run from the repository root in the existing training Python environment:

```bash
bash run_retrieval_mass.sh --plan
bash run_retrieval_mass.sh --source-root work/direct-v1
```

Only s=1, on your choice of backbones and budgets:

```bash
bash run_retrieval_mass.sh --masses 1.0 \
  --backbones ff2048 vit_b16 \
  --train-k ff2048=8,32 vit_b16=12,48 \
  --source-root work/direct-v1 --workspace work/retrieval-mass
```

This selects **4 fits / 4 evaluations**. Changing a training objective starts
a separate fit from scratch, with the same seed initialization; it never
continues a partial-OT checkpoint as balanced training. Repeating the exact
command resumes its own interrupted fits and reuses its completed evaluations.

Backbone choices: `ff2048`, `rn18`, `vit_b16`, `convnext_tiny`,
`dinov2_vits14`. The latter is the existing **DINOv2 Small/14 ImageNet
embedding source**, distinct from the segmentation Base/14 encoder. Give
each selected backbone its K list using `--train-k`. A newly selected backbone
without an explicit K list defaults to 8 for ResNet or 12 for other encoders;
always inspect `--plan` before a large run.

```bash
bash run_retrieval_mass.sh --masses 0.9 1.0 \
  --backbones convnext_tiny dinov2_vits14 \
  --train-k convnext_tiny=12,48 dinov2_vits14=12,48 \
  --seeds 1227 1228 1229 --plan
```

To evaluate every trained checkpoint at additional inference K values:

```bash
bash run_retrieval_mass.sh --masses 1 \
  --backbones ff2048 --train-k ff2048=8,32 \
  --inference-k ff2048=2,8,16,32,64
```

This is **2 fits / 10 evaluations**. Unlike the main E2 anchor selection, this
explicit ablation sweep evaluates every selected checkpoint. Changing only
inference K reuses fits. For a YAML sweep, set `evaluation_mode: sweep` and
provide `inference_k` for every backbone. Matched mode requires an empty map.

Other overrides: `--epochs`, `--lr`, `--batch-size`, `--ot-weight`, `--ot-eta`,
`--ot-microbatch`, `--device`, `--residency`, `--encode-batch-size`,
`--gallery-chunk`, `--query-block`. Training changes create different fit
identities. An explicit weight conflicting with the K>64 recipe is rejected.
For 12GB GPUs, start with `--residency cpu --query-block 1024`; search chunk
sizes control memory, not the number of gallery examples searched.

## New server / cached features

Use the repository's existing server environment setup, or an environment
already containing PyTorch, NumPy, PyYAML and psutil. No FFCV, image downloads
or backbone extraction are needed by this runner. A tiny environment check:

```bash
bash run_retrieval_mass.sh --smoke --device cpu \
  --backbones vit_b16 --train-k vit_b16=3 --masses 0.9 1
```

Synthetic artifacts are clearly marked and isolated under `SMOKE_ONLY`.

Scientific runs expect:

```text
work/direct-v1/embeddings/<backbone>/1K_train_<backbone>.npz
work/direct-v1/embeddings/<backbone>/1K_val_<backbone>.npz
```

Keep the accompanying manifest/identity files. Full split row counts are
required: 1,281,167 train/gallery and 50,000 validation/query images. Existing
archives are memory-mapped once per backbone and shared across masses, K and
seeds. `--embeddings-dir` and `--cache-dir` can select other existing paths.

If you need to download the precomputed archives from the existing embedding
repository, install `huggingface_hub`, then run:

```bash
bash run_retrieval_mass.sh --masses 1 --download-embeddings
```

This is explicitly download-only: a missing remote feature bundle causes an
error instead of extracting raw ImageNet images or uploading anything.
`--embedding-repo owner/name` selects a different embedding repository. Private
repositories require your existing read access/token. `.env` is loaded only
for this opt-in download step.

## Artifacts and reuse

```text
work/retrieval-mass/
  fits/<training-identity>/checkpoint.pt, config.json, summary.json, train.jsonl
  evaluations/<evaluation-identity>/result.json
  sweeps/<sweep-id>/plan.json, runtime.json, attempts.jsonl
  sweeps/<sweep-id>/results.csv, results.json, summary.csv, table.tex
  sweeps/<sweep-id>/retrieval-mass-results.zip
  experiment-index.csv, experiment-index.json, latest.json
```

The fit identity includes data, architecture, seed, complete objective and
schedule; s=1 also records the balanced solver version. Evaluation identity
includes checkpoint SHA-256, train signature, inference K and precision.
Changing the list of selected conditions does not invalidate reusable fits.

Source reuse reads `<source-root>/runs/<backbone>/mpsae/k<K>/seed<seed>`.
It requires a completed, compatible training identity and a verified actual
checkpoint. It does not overwrite source artifacts. A matching E1 result may
also be reused, with this reuse explicitly recorded. Mismatched source recipes
are trained separately; `--no-reuse-source` disables source lookup but still
resumes matching ablation checkpoints. To rerun identical fits from scratch,
use a fresh `--workspace` together with `--no-reuse-source`.

Each completed evaluation is exported immediately. Failed/pending conditions
remain explicit; tables never fill them with zeros. `summary.csv` and
`table.tex` show mean and sample SD over completed seeds, along with the number
completed/expected. `nnz` is measured **gallery mean active latents**.

Rebuild tables without loading embeddings or training:

```bash
bash run_retrieval_mass.sh --export-only
```

Use the same config/CLI selectors as the original sweep. `latest.json` points
to its local ZIP. Download that ZIP normally from your server or notebook;
it includes tables and provenance, not large checkpoints or embeddings. Keep
the workspace/source checkpoints if you want to resume or extend evaluation.
`experiment-index.csv` retains results from all recorded selections in this
workspace. One runner holds a workspace lock; use separate workspaces for
concurrent GPU invocations.
