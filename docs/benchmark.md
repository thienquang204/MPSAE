# ImageNet benchmarks and optional frozen transfer

The main campaign evaluates ImageNet sparse embeddings from ResNet-50,
ViT-B/16, ConvNeXt-Tiny, ResNet-18 and DINOv2-S/14. Every selected point measures 1-NN;
only the selected ResNet-50 and ViT inference budgets fit linear probes.
All downstream experiments and repeated retrieval timing are disabled.
The target protocols documented below remain available for explicit future
campaigns. COCO image-text retrieval and CLIP remain future work.

## Native Python, one configuration

```bash
cp .env.example .env                         # if creating a new .env
# Fill HF_TOKEN (write access for result backup); WANDB_TOKEN is optional.
bash all_in_one.sh
```

This installs and runs the native campaign, creates/reuses the private HF
dataset `<account>/MPSAE` using HF_TOKEN, and logs to W&B project `cvpr` using
WANDB_TOKEN (WANDB_API_KEY is also accepted). Project and account names do not
need to be supplied. Checkpoints, feature archives and final reports are backed
up, including partial results after failure. Retry only remote synchronization
with `bash all_in_one.sh --sync-only`. A synthetic rehearsal via
`bash all_in_one.sh --smoke` does not write to either service.

The default campaign downloads or reuses five ImageNet embedding bundles:
`ff2048`, `vit_b16`, `convnext_tiny`, `rn18`, `dinov2_vits14`. No downstream bundles are selected.
No raw image download or backbone extraction runs with `precomputed_only: true`.

Direct `python -m mpsae --config ...` commands below run locally. Automatic
cloud operations belong to the all-in-one entry point (`python -m mpsae.remote`).

The native setup targets Ubuntu 22.04/24.04 on x86_64 or ARM64/SBSA and detects
A100, RTX A6000, RTX 4090/5090, H100/H200 and GB300 by compute capability.
It selects the matching CUDA 12.6/12.8/12.9 profile and installs the Python environment,
including mandatory FFCV/CuPy, and verifies real GPU operations before
reporting success. See [server.md](server.md) for installation, logs and
administrator-provided dependencies. The script creates `.env` only if it
does not exist. A token is needed by gated ImageNet access, not to inspect a
plan, status or existing reports.

Use the same config on every command. Selected data and results live in
`work/`, independently of the Python environment. Editing YAML does not require
reinstalling dependencies.

```bash
python -m mpsae --config configs/thesis.yaml status
python -m mpsae --config configs/thesis.yaml report
python -m mpsae --config configs/smoke.yaml --smoke run
```

The smoke campaign uses synthetic source and target data and a separate
`work/smoke-<name>` workspace. It tests plumbing, not scientific quality. A CPU
host installation can use:

```bash
python -m mpsae --config configs/smoke.yaml --smoke --device cpu run
```

`pipeline.sh` wraps the same Python CLI, loads the generated CUDA paths
and saves a console transcript. The launcher uses an exported `MPSAE_CONFIG` or defaults to
`configs/thesis.yaml`; `--smoke` defaults to `configs/smoke.yaml`. It accepts
global options on either side of the command:

```bash
bash pipeline.sh plan
bash pipeline.sh --config configs/thesis.yaml --device cuda:0
bash pipeline.sh report
bash pipeline.sh stage embed -- --backbones ff2048
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
Docker files remain available as an optional separate deployment. They are not
used by the native installer or Python runner. The container defaults to
`configs/thesis.yaml`, mounts `./work` and `./configs`, and uses
`docker compose run --rm mpsae` to run; this path is not required on the native
server. Its CUDA version is configured independently in `compose.yaml`.

## Configuration and execution

`configs/thesis.yaml` selects backbones, methods, training seeds, ImageNet
experiments and downstream budgets. Training hyperparameters come from the
existing presets, not a second set of benchmark defaults. The resolved campaign
configuration records the settings actually requested. The only selectable
methods are `csr`, `csrv2` and `mpsae`; CSR multi-TopK and MPSAEv2 are excluded
from every experiment and newly generated report, even if their old results
remain on disk. Old configurations naming either excluded method must be
updated before running.

```yaml
# Main ImageNet campaign. Previous settings: ../large_scale_abblation.md.
# Precomputed embeddings; source training recipes remain in mpsae/config.py.
# No downstream evaluation or repeated retrieval timing in this campaign.
name: thesis
backbones: [ff2048, vit_b16, convnext_tiny, rn18, dinov2_vits14]
methods: [csr, csrv2, mpsae]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8, 16, 32, 256]
    vit_b16: [12, 48, 192]
    convnext_tiny: [12, 48, 192]
    rn18: [8, 32, 128]
    dinov2_vits14: [12, 48, 192]
  eval_k:                       # fixed checkpoint, inference sweep; E1 also measures train_k
    ff2048: [2, 16, 32, 256]
    vit_b16: [3, 12, 48, 96, 192, 384]
    convnext_tiny: [3, 12, 48, 96, 192, 384]
    rn18: [2, 8, 16, 32, 64, 128, 256]
    dinov2_vits14: [3, 12, 48, 96, 192, 384]
  e1: true
  e2: true
  linear_probe: true
  probe_k:                      # inference K, applied equally to E1 and E2
    ff2048: [2, 16, 32, 256]
    vit_b16: [3, 48, 96, 384]
    convnext_tiny: []            # 1-NN only
    rn18: []                     # 1-NN only
    dinov2_vits14: []            # 1-NN only
  timing: false
  timing_warmup: 5              # short timing protocol, if enabled
  timing_rounds: 5
downstream:
  enabled: false                # no SOP, EuroSAT, Flowers102 or COCO tasks/downloads
  datasets: [sop]                # inactive placeholder for optional future campaigns
execution:
  storage_mode: direct_v1
  precomputed_only: true
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

`imagenet.train_k: null` uses the registered training grid. An explicit list or
per-backbone mapping selects source fits. `imagenet.eval_k` selects E2 budgets;
`null` uses the shared inference defaults shown below. E1 always uses train K.
`max_steps: 0` keeps the full training schedule. A positive cap changes fit
identity and cannot be mistaken for a completed full training schedule.

| Backbone | Training K | Inference K (1-NN) | Linear-probe inference K |
|---|---|---|---|
| ResNet-50 (`ff2048`) | 8, 16, 32, 256 | 2, 16, 32, 256 | 2, 16, 32, 256 |
| ViT-B/16 (`vit_b16`) | 12, 48, 192 | 3, 12, 48, 96, 192, 384 | 3, 48, 96, 384 |
| ConvNeXt-Tiny (`convnext_tiny`) | 12, 48, 192 | 3, 12, 48, 96, 192, 384 | None |
| ResNet-18 (`rn18`) | 8, 32, 128 | 2, 8, 16, 32, 64, 128, 256 | None |
| DINOv2-S/14 (`dinov2_vits14`) | 12, 48, 192 | 3, 12, 48, 96, 192, 384 | None |

With CSR, CSRv2, MPSAE and seed 1227, this selects **48 SAE fits**, **273 E2
points** (45 shared with E1), **276 unique sparse 1-NN conditions**, and **84
unique linear probes**. The probes are part of those evaluated conditions,
not additional SAE fits. Existing compatible fits and metrics are reused;
DINOv2 adds 9 SAE fits to the previous 39-fit selection. These are 3 methods
across 16 backbone/training-K pairs, with one seed. Backbone networks stay
frozen; existing HF embeddings are reused. Linear classifiers add 84 fits
only where requested, independently of the SAE count.

`imagenet.probe_k` selects probes by backbone and **inference K** in both E1
and E2. Thus ResNet-50 E1 at train K=8 and ViT E1 at train K=12/192 use 1-NN
only. The same checkpoints still receive probes at the selected E2 K values.
An empty list or omitted backbone in an explicit probe mapping disables that
backbone's probes. `probe_k: null` keeps legacy all-K probing;
`linear_probe: false` disables all probes regardless of the mapping.

All selected points measure exact 1-NN on sparse codes. Each selected probe
fits a separate linear classifier on ImageNet training codes and evaluates
validation codes. The SAE stays frozen throughout each inference sweep.
Source fits at different training K are independent, not sequential fine-tuning.
Training epochs, learning rates, batch sizes and losses are unchanged. MPSAE's
existing rule uses MMPOT weight 0.5 for **training K > 64**, otherwise 2.0;
inference K never changes the trained objective.

`downstream.enabled: false` omits all target data downloads, extraction,
classification, retrieval and downstream reports. `imagenet.timing: false`
disables repeated retrieval microbenchmarks. If enabled, timing now defaults
to **5 warmup + 5 measured rounds**, including standalone timing calls.
This short schedule gives a quick estimate with fewer samples than the old
100/2000 protocol; timing identities keep the measurements separate.
The training schedule and OT solver limits are unchanged. Ordinary 1-NN search still takes
time, and its elapsed duration can remain in diagnostic tables. No dense or
published reference curves are added; dense ImageNet 1-NN remains a feature
integrity diagnostic.

Selection changes create a new report plan without changing the fit identity.
Completed accuracy is reused when adding missing probes. Historical probe and
timing scores stay in source JSON caches but are filtered out of the current
catalog, tables and plots when not selected. E1/E2 share completed anchor
measurements. Re-running the campaign computes only missing compatible work.

The previous wider training selections are archived in
[`large_scale_abblation.md`](../large_scale_abblation.md). Earlier unselected E2
measurements remain in `eval_e2.json` under `cached_points`; only selected `points`
enter reports. Changing the inference selection creates a new campaign plan,
while compatible source checkpoints and per-point metrics remain reusable.

The runner prepares inputs, then processes the selected work sequentially. Each
evaluation is tied to its actual checkpoint and data/protocol identity. A
restart can reuse compatible features, source fits and evaluation records.
`continue_on_error` records independent failures and allows other independent
points to run; it does not make a failed campaign complete. Missing/failing
points remain visible in the final report. `retain_codes: false` limits disk
growth by discarding regenerable sparse-code caches when their work is done;
metrics and checkpoint provenance remain.

## SOP evaluation only

This historical config is deferred and never launched by the main campaign.
To explicitly opt into SOP evaluation from existing fits in a future run:

```bash
python -m mpsae --config configs/retrieval.yaml plan
bash all_in_one.sh --config configs/retrieval.yaml --skip-setup
```

This selects ResNet-50, CSR/CSRv2/MPSAE and seed 1227. It requires **12 completed
source checkpoints** at training K=8/16/32/256 and evaluates each at inference
K=2/16/32/256, giving **48 sparse SOP records plus one dense diagnostic**.
Backbones and SAEs remain frozen. `imagenet.enabled`, `e1`, `e2`, `linear_probe`
and `timing` are false; `train_missing: false` prevents new fits. Thus the
ImageNet linear probes and retrieval timing benchmarks do not run.

`precomputed_only: true` and `storage_mode: direct_v1` reuse the existing SOP
embedding bundle. The Hub download selection contains SOP/ResNet-50 only;
ImageNet feature arrays, raw images and new backbone extraction are unnecessary.
Use the same workspace as the source fits. Retain each fit's `config.json`,
`summary.json`, `checkpoint.pt` and original pinned backbone
weights/preprocessing provenance. The provenance may already be in the config;
otherwise retain the original ImageNet `.npz.json` sidecars. Do not infer it
from the architecture name.

`continue_on_error: false` stops on a missing or incompatible checkpoint or
feature bundle. The runner validates the saved training recipe even for
evaluation-only imports. In particular, an MPSAE training-K=256 checkpoint must
use MMPOT loss weight **0.5**. An older weight-2.0 checkpoint needs separate
fresh training before it can enter this evaluation campaign; this config will
not train or overwrite it.

Compatible SOP evaluations are shared across campaigns by checkpoint, features,
inference budget and protocol identity. A new report selection does not force
reranking completed compatible points. The report is written under
`work/direct-v1/reports/retrieval/<plan_id>/`, with the newest plan referenced by
`work/direct-v1/reports/retrieval/latest.json`. This config does not select
EuroSAT, COCO or CLIP. The full main campaign remains available separately.

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

## Dataset tasks and splits

| Dataset source | Default task | Support/gallery | Final queries | Labels |
|---|---|---|---|---|
| [`mteb/eurosat-rgb`](https://huggingface.co/datasets/mteb/eurosat-rgb) | DINO weighted kNN classification when explicitly enabled | 16,200 train images | 5,400 test images | 10 land-use classes |
| [`pufanyi/flowers102`](https://huggingface.co/datasets/pufanyi/flowers102) | DINO weighted kNN classification when explicitly enabled | 6,551 train images | 819 test images | flower class |
| [`nyris/stanford-online-products-v1`](https://huggingface.co/datasets/nyris/stanford-online-products-v1) | product retrieval | test images | same test images, excluding the query itself | `item_id`, not broad `category` |

`downstream.tasks` selects a task per dataset. Omitted entries use the defaults
above. The main campaigns disable all downstream work. The historical
`retrieval.yaml` explicitly opts into SOP; EuroSAT and Flowers102 remain optional. Flowers102 is excluded from the main campaigns
because its precomputed bundles are not available in the selected collection.
The Flowers mirror uses a custom HF 80/10/10 split, not the Oxford official split.

EuroSAT's 5,400 validation images and Flowers' 819 validation images remain
separate raw splits. Current feature exporters save only the support/gallery and
query splits, so a validation embedding bundle must not be assumed available.
The fixed neighbour counts and temperature require no validation embeddings.
If tuning is introduced later, choose hyperparameters on validation or a
documented holdout of training support, never on final test labels.

SOP's training and test product IDs are disjoint. A classifier storing training
product labels cannot predict the unseen test product IDs; the campaign therefore
keeps SOP as retrieval. Its [reference evaluation](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16/blob/master/code/evaluation/evaluate_recall.m)
searches test against test and excludes the diagonal self-match. The adapter
checks 59,551 training images/11,318 products and 60,502 test images/11,316 products,
and checks the identity partition. Parity of the HF mirror with the official
`Ebay_test.txt` file list is not independently certified.

Datasets are pinned to immutable Hub revisions. Manifests retain repository,
revision, split mapping, row counts and label/ID information. `sample_id`
identifies an individual image and drives overlap checks or self-exclusion;
it is distinct from the class/product label. No rows are dropped merely
because the final extraction batch is incomplete.

## Optional DINO weighted kNN classification

The classifier follows the weighted vote in the
[official DINO `eval_knn.py`](https://github.com/facebookresearch/dino/blob/main/eval_knn.py).
For L2-normalized query code z and its n nearest support codes, class c receives

```text
score(c) = sum_i exp(cosine(z, z_i) / temperature) * [label_i == c]
prediction = argmax_c score(c)
```

The default `n_neighbors: [1, 20]` evaluates a 1-NN control and a weighted 20-NN
classifier; `temperature: 0.07` is fixed for all active methods, backbones and
sparse budgets. These are classifier parameters, independent of `eval_k`, which
controls the maximum number of nonzero coordinates in each SAE code. The input
to classification is the sparse code itself, never its decoder reconstruction.

Support labels determine the votes. Test labels only score the predictions;
the classifier does not learn weights, update the SAE, or select hyperparameters
from test outcomes. Every query label must occur in the support set, and support
and query sample IDs must be disjoint. A requested neighbour count larger than
the support set is an error rather than an undocumented smaller classifier.

Search retrieves the maximum requested neighbour count once. Voting for smaller
counts reuses the leading neighbours, avoiding repeated encoding and search.
The DINO vote is reproduced on nonzero examples without ties. Two explicit
extensions retain the sparse-evaluation contract: equal neighbour similarities
use gallery-position order and equal class scores use sorted label order; an
all-zero query code abstains and counts as a miss rather than receiving credit
from arbitrary neighbours. DINO's original script uses native `topk`/sorting
ties and does not implement this zero-code abstention. These policies are
recorded in evaluation metadata and apply equally to every method.

With these policies held fixed, 1-NN top-1 equals the previous class-relevance
Recall@1. For n greater than one, classification accuracy is not Recall@n:
the former scores a single voted label; the latter only asks whether any of the
n neighbours has the correct label. Historical retrieval metrics cannot be
renamed to obtain the new classification measurements.

## Shared frozen features and exact ranking

For a given dataset and backbone, all SAE methods read the same saved dense
features. Extraction uses deterministic evaluation preprocessing for gallery
and query alike, with the backbone's configured input size and normalization.
FF2048 uses its BlurPool ResNet-50 weights; timm backbones keep their pinned
model tags. Manifests record the actual transform and precision.

Features enter the SAE at the scaling used by source training. Dense baseline
features and output sparse codes are L2-normalized for cosine search. CUDA
encoding uses float16 autocast with float32 storage; ranking uses float32 and
TF32 disabled. CPU smoke runs are identified as tests, not GPU reproductions.

Neighbour search is exhaustive and blocked: it keeps only the needed top-r
candidates rather than materializing the full query-by-gallery score matrix.
SOP self-matches are excluded by image ID before ranking; classification
rejects overlapping support/query IDs. Equal scores have a
deterministic gallery-position tie rule. Empty query codes are counted and
scored as misses. For retrieval, the default no-positive policy is `error`: it
refuses an invalid protocol with queries that have no relevant gallery image,
rather than silently excluding them from the denominator.

## Metrics and plots

Code budget **K**, classifier neighbour count **n**, and retrieval rank **r**
are different quantities. `eval_k: 8` with `n_neighbors: [20]` means a code with
at most eight active coordinates is used in a classifier with twenty neighbours.

Classification records contain `knn_<n>_top1`, `knn_<n>_top5`,
`knn_<n>_balanced_accuracy` and `knn_<n>_macro_f1` for every requested n.
Balanced accuracy is mean recall over query classes; macro-F1 averages each
class's F1. Empty-query abstentions remain in the denominator and count as
false negatives. As in DINO, the top-5 field uses an effective class rank of
`min(5, n, number_of_classes)`, so for 1-NN it is identical to top-1.

SOP and explicitly selected retrieval tasks retain the following metrics:

* **Recall@1/10/100** uses the metric-learning convention: the fraction of
  eligible queries with at least one relevant image within the first r results.
* **Precision@10** measures the fraction of the first ten results that are
  relevant; small smoke galleries record the actual available depth.
* **mAP@R** averages each query's AP at R, where R is its full number of
  relevant gallery images after self-exclusion. Ranking must reach R; top-100
  alone is insufficient when R is larger. This extra ranking depth is unnecessary
  for kNN classification, which only needs the requested neighbours.
* **Macro metrics** average class-level query results so frequent classes do
  not hide weak performance on less frequent ones.

Evaluation records store fractions in `[0,1]`; plot axes show percentages.
Training seeds are repetitions; dense references are measured once and are
not copied three times to manufacture seed statistics.

The downstream report consumes the campaign's explicit records. It never
glob-loads unrelated old files. No published, Dense or Full-representation
reference curve or storage marker is drawn in any plot.

| Figure | Exact interpretation |
|---|---|
| Optional `classification_<backbone>_<mode>_train-k<K>` | Separate 1-NN/20-NN accuracy panels against inference K, keeping the training-budget checkpoint fixed. |
| `classification_balance_*` | Balanced accuracy and macro-F1, separately for each neighbour count. |
| `classification_nonzeros_*`, `classification_storage_*` | Classifier accuracy against measured nonzeros or stored KiB/image. |
| `classification_paired_*` | Paired top-1 differences for MPSAE versus CSR and CSRv2 within the same classifier setup. |
| Retrieval `quality_*` | SOP R@1 and mAP@R against inference code budget, with datasets/protocols kept separate. |
| `nonzeros_...` | The same accuracies against measured mean gallery nonzeros. Zero-nonzero collapsed codes remain representable on the axis. |
| `storage_...` | Accuracy against stored KiB per image (float32 values plus int32 indices). Only sparse methods are plotted; compact code storage is distinct from GPU CSR index bytes. |
| `classification_cost_*`, retrieval `cost_*` | Shared search wall time per query (ms) and stored KiB/image against inference K. Classification voting time is recorded separately. Search cost is not multiplied by the number of neighbour settings or described as full classifier latency. |
| `paired_...` | MPSAE−CSR and MPSAE−CSRv2 differences formed within seed before averaging, in percentage points. |

Lines are seed means and bands sample standard deviations (`ddof=1`), not
confidence intervals. Captions describe the selected seeds; tables report per-point n; tables list missing and unpaired
seeds. Datasets, revisions/comparison IDs, protocols, training budgets and modes
remain distinct. Classification neighbour counts, temperature and edge-case
policies also identify the protocol and cannot be averaged with another setup.
Two different checkpoint evaluations for the same condition
and seed are rejected rather than averaged as extra repetitions. A fixed-training-K sweep also rejects different checkpoint IDs across inference K for the same seed and protocol.

When their task is selected, classification and retrieval metrics have separate `classification.{json,csv}`
and `retrieval.{json,csv}` tables; `evaluations.{json,csv}` retains the original
records and task metadata. Classification figures with more than four axes
continue in `_partN` files. Top-5 values remain in tables. Every figure exports PNG,
PDF and a caption. CSV/JSON tables retain full
precision and the evaluation/checkpoint IDs used by each row. The manifest lists
expected/completed points, failures, incomplete records, tables, figures and
skipped plots. Tables still write when matplotlib is unavailable.

ImageNet `inference_summary` and `e2_generalisation_*` show 1-NN and only
selected per-K linear-probe accuracy from fixed SAE checkpoints. Downstream DINO kNN stores labelled
support but does not fit a linear head; target-dataset linear probing remains a
separate experiment. All figures omit published and Dense/Full-representation
overlays. Dense measurements remain numerical diagnostics. Search speedup should
only be claimed from appropriately paired timings; quality plots do not imply
sparse-serving speedup by themselves.

Classification has a distinct evaluation identity containing its task,
neighbour counts, temperature and voting/edge-case rules, in addition to feature
provenance, checkpoint and sparse budget. Existing HF features and source
checkpoints are reused. The new vote requires reranking; old retrieval aggregate
files do not contain the neighbour list. `report` alone cannot compute these
missing measurements.

Changing these selections creates a new plan ID. Use the newly generated
`reports/<name>/<plan_id>/` directory (or its updated `latest.json` pointer).
Historical PNG/PDF files are not rewritten merely by pulling source code.
