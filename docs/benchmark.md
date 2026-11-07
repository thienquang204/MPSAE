# Frozen transfer and robustness campaign

The campaign adds downstream retrieval to the existing ImageNet E1/E2 and
linear-probe experiments. Backbones and ImageNet-trained SAEs stay frozen on
the target datasets. It measures transfer without target-dataset adaptation;
no fine-tuning or test-set parameter fitting is performed.

This measures cross-dataset transfer and sensitivity to the inference code
budget K. It does not yet measure robustness to image corruptions or
adversarial perturbations; those need separate datasets and protocols.

## One container, one configuration

```bash
cp .env.example .env
# Set HF_TOKEN after accepting ImageNet access conditions.
docker compose build
docker compose run --rm mpsae plan
docker compose run --rm mpsae
```

The image defaults to `--config configs/thesis.yaml run` and sets
`MPSAE_CONFIG=configs/thesis.yaml`, so replacing the command with `plan`,
`status` or `report` keeps the same campaign. Use `--config` to select another
configuration. Compose mounts
`./work` at `/work` and `./configs` read-only at `/opt/mpsae/configs`, so results
and checkpoints survive the container and configuration changes need no image
rebuild. A token is needed by stages that access gated ImageNet, not to inspect
a plan, status or existing reports.

The image uses Ubuntu 24.04's stable Python in `/opt/venv`, CUDA 13.0 and
the matching [PyTorch cu130 wheels](https://download.pytorch.org/whl/cu130/).
FFCV and [CuPy's matching CUDA wheel](https://docs.cupy.dev/en/stable/install.html)
are mandatory for the default RN18 extraction; dependency installation errors
fail the image build. `ninja` is included for the optional timing extension.

```bash
docker compose run --rm mpsae status
docker compose run --rm mpsae report
docker compose run --rm mpsae --config configs/smoke.yaml --smoke run
```

The smoke campaign uses synthetic source and target data and a separate
`work/smoke-<name>` workspace. It tests plumbing, not scientific quality. A CPU
host installation can use:

```bash
python -m mpsae --config configs/smoke.yaml --smoke --device cpu run
```

The Compose service itself reserves a CUDA GPU. For a host environment,
`scripts/setup_server.sh` installs dependencies and `scripts/pipeline.sh` wraps
the same CLI. The launcher uses an exported `MPSAE_CONFIG` or defaults to
`configs/thesis.yaml`; `--smoke` defaults to `configs/smoke.yaml`. It accepts
global options on either side of the command:

```bash
./scripts/pipeline.sh plan
./scripts/pipeline.sh --config configs/thesis.yaml --device cuda:0
./scripts/pipeline.sh report
./scripts/pipeline.sh stage embed -- --backbones ff2048
```

`stage NAME` remains the original ImageNet-stage interface. On a host with
`MPSAE_CONFIG` unset, `python -m mpsae run` without `--config` remains the
original ImageNet pipeline.
All Python global options go before the command; the shell launcher reorders
them for convenience. With a campaign, scientific selections such as methods,
backbones, seeds and budgets belong in YAML; runtime device/workspace options
remain CLI flags. Campaign reports use their plan-specific output directories.
For an old-style report with a custom output directory, use
`python -m mpsae report --out work/reports/review` with `MPSAE_CONFIG` unset.
In Docker, the equivalent explicit environment override is
`docker compose run --rm -e MPSAE_CONFIG= mpsae report --out /work/reports/review`.

## Configuration and execution

`configs/thesis.yaml` selects backbones, methods, training seeds, ImageNet
experiments and downstream budgets. Training hyperparameters come from the
existing presets, not a second set of benchmark defaults. The resolved campaign
configuration records the settings actually requested.

```yaml
name: thesis
backbones: [ff2048, rn18]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227, 2026, 2027]

imagenet:
  enabled: true
  prepare: true
  train_k: null
  e1: true
  e2: true
  linear_probe: true
  timing: true

downstream:
  enabled: true
  datasets: [eurosat, flowers102, sop]
  train_k: [8]
  eval_k: [2, 4, 8, 16, 32, 64]
  mode: frozen
  revisions: {}

execution:
  train_missing: true
  continue_on_error: true
  retain_codes: false
  max_steps: 0
```

`imagenet.train_k: null` means the selected backbones' full original training
grids. A list restricts the source fits. `downstream.train_k` selects which
ImageNet checkpoints are evaluated; `eval_k` changes their inference budget
without changing any weights. Missing required source checkpoints are trained
only when `train_missing` is enabled. Existing compatible completed fits are
reused. `max_steps: 0` uses the full source training schedule; a positive cap is
part of fit identity and does not turn a capped fit into a full-fit checkpoint.

The runner prepares inputs, then processes the selected work sequentially. Each
evaluation is tied to its actual checkpoint and data/protocol identity. A
restart can reuse compatible features, source fits and evaluation records.
`continue_on_error` records independent failures and allows other independent
points to run; it does not make a failed campaign complete. Missing/failing
points remain visible in the final report. `retain_codes: false` limits disk
growth by discarding regenerable sparse-code caches when their work is done;
metrics and checkpoint provenance remain.

## Where a campaign writes its information

```text
work/
  runs/                                      # original shared ImageNet fits
  embeddings/                                # shared source dense archives
  benchmarks/                                # target data/features/cache
  evaluations/<evaluation_id>/               # individual downstream results
  campaigns/<name>/<plan_id>/
    resolved.json                            # effective scientific settings
    runtime.json                             # device, versions and runtime facts
    status.json                              # completed, failed and missing work
    catalog.json                             # selected artefact references
  reports/<name>/<plan_id>/
    imagenet/                                # selected E1/E2/probe report
    downstream/                              # selected frozen-transfer report
    manifest.json                            # both reports and completeness
  reports/<name>/latest.json                  # pointer to the newest report
```

The plan ID hashes the resolved scientific campaign settings, including the
source presets. Runtime device/precision is recorded separately and included
where it changes evaluation identity. A new dataset or report selection does
not change the identity of an otherwise unchanged source fit. Checkpoints stay
under the existing `runs/<backbone>/<method>/k<K>/seed<S>/` paths; a catalog
references them rather than copying them into each dataset. Report rebuilding
uses only records selected by the current campaign, including for ImageNet.

## The three retrieval protocols

| Dataset source | Gallery | Final queries | Relevance |
|---|---|---|---|
| [`mteb/eurosat-rgb`](https://huggingface.co/datasets/mteb/eurosat-rgb) | 16,200 train images | 5,400 test images | same land-use class |
| [`pufanyi/flowers102`](https://huggingface.co/datasets/pufanyi/flowers102) | 6,551 train images | 819 test images | same flower class |
| [`nyris/stanford-online-products-v1`](https://huggingface.co/datasets/nyris/stanford-online-products-v1) | test images | same test images, excluding the query itself | same `item_id`, not broad `category` |

EuroSAT's 5,400 validation images and Flowers' 819 validation images are kept as
separate splits. The term *train gallery* describes an image collection used
for search; it does not imply backbone or SAE fitting on those target images.

The Flowers mirror explicitly uses a custom 80/10/10 split. Its results must
be labelled as that protocol, not the Oxford official-split benchmark. The
SOP mirror exposes image paths and product IDs but has an empty README; the
adapter audits its actual split counts and product identity separation before
accepting the mirror. This checks the protocol's counts and identity rules;
parity with the official `Ebay_test.txt` file list is not independently
certified. Original SOP has
59,551 training images/11,318 products and 60,502 test images/11,316 products.
Its [reference evaluation](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16/blob/master/code/evaluation/evaluate_recall.m)
searches test against test and excludes the diagonal self-match.

Datasets are pinned to immutable Hub revisions. Manifests retain repository,
revision, split mapping, row counts and label/ID information. `sample_id`
identifies an individual image and drives self-exclusion; it is distinct from
the relevance label shared by related images. No rows are dropped merely
because the last extraction batch is incomplete.

## Shared frozen features and exact ranking

For a given dataset and backbone, all SAE methods read the same saved dense
features. Extraction uses deterministic evaluation preprocessing for gallery
and query alike, with the backbone's configured input size and normalization.
FF2048 uses its BlurPool ResNet-50 weights; timm backbones keep their pinned
model tags. Manifests record the actual transform and precision.

Features enter the SAE at the scaling used by source training. Dense baseline
features and output sparse codes are L2-normalized for cosine retrieval. CUDA
encoding uses float16 autocast with float32 storage; ranking uses float32 and
TF32 disabled. CPU smoke runs are identified as tests, not GPU reproductions.

Ranked retrieval is exhaustive and blocked: it keeps only the needed top-r
candidates rather than materializing the full query-by-gallery score matrix.
Self-matches are excluded by image ID before ranking. Equal scores have a
deterministic gallery-position tie rule. Empty query codes are counted and
scored as misses. The campaign's default no-positive policy is `error`: it
refuses an invalid protocol with queries that have no relevant gallery image,
rather than silently excluding them from the denominator.

## Metrics and plots

Code budget **K** and retrieval rank **r** are different quantities. For example,
`eval_k: 8` with Recall@10 means an at-most-eight-active-coordinate code is used
to return ten nearest images.

* **Recall@1/10/100** uses the metric-learning convention: the fraction of
  eligible queries with at least one relevant image within the first r results.
* **Precision@10** measures the fraction of the first ten results that are
  relevant; small smoke galleries record the actual available depth.
* **mAP@R** averages each query's AP at R, where R is its full number of
  relevant gallery images after self-exclusion. Ranking must reach R; top-100
  alone is insufficient when R is larger. EuroSAT's ten classes make high-rank
  Recall saturate easily, so mAP@R and Precision@10 add information.
* **Macro metrics** average class-level query results so frequent classes do
  not hide weak performance on less frequent ones.

Evaluation records store fractions in `[0,1]`; plot axes show percentages.
Training seeds are repetitions; dense references are measured once and are
not copied three times to manufacture seed statistics.

The downstream report consumes the campaign's explicit records. It never
glob-loads unrelated old files or inserts published ImageNet curves.

| Figure | Exact interpretation |
|---|---|
| `quality_<backbone>_<mode>_train-k<K>` | Rows are dataset/protocols; columns are Recall@1 and mAP@R; x is inference code budget. All points in a curve reuse the stated training-budget checkpoint. |
| `nonzeros_...` | The same accuracies against measured mean gallery nonzeros. Zero-nonzero collapsed codes remain representable on the axis. |
| `storage_...` | Accuracy against measured stored bytes per image. Dense features are a measured point when footprint is supplied; compact code storage is distinct from GPU CSR index bytes. |
| `paired_...` | MPSAE−CSR, MPSAEv2−CSRv2 and multi-TopK−CSR differences formed within seed before averaging, in percentage points. |

Lines are seed means and bands sample standard deviations (`ddof=1`), not
confidence intervals. Legends/tables report n; tables list missing and unpaired
seeds. Datasets, revisions/comparison IDs, protocols, training budgets and modes
remain distinct. Two different checkpoint evaluations for the same condition
and seed are rejected rather than averaged as extra repetitions.

Every figure exports PNG, PDF and a caption. CSV/JSON tables retain full
precision and the evaluation/checkpoint IDs used by each row. The manifest lists
expected/completed points, failures, incomplete records, tables, figures and
skipped plots. Tables still write when matplotlib is unavailable.

ImageNet linear-probe results keep their existing role. Downstream retrieval
does not train a classifier; target-dataset linear probing is a separate
experiment. Search speedup should only be claimed from appropriately paired
timings; quality plots do not imply sparse-serving speedup by themselves.
