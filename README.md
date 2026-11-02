# DLMMSAE: ImageNet comparison with CSR

The current MMRSAE architecture synthesizes an embedding from a fixed training
landmark dictionary before a tied SAE applies Top-K. The input dictionary is
saved with the model and also used at inference; there is no post-SAE repair.
The selected benchmark includes exactly four methods:
`csr`, `csrv2`, `dlmmsae`, and `dlmmsaev2`. Each checkpoint is evaluated at inference K=2..512 for RN18 and
2..2048 for RN50 (`MMRSAE_EVAL_TOPK=auto`), producing `top1_by_k.png/.pdf` and
per-K tables. DLMMSAE uses angular OT with separate landmark/context KL. CSRv2
and DLMMSAEv2 call the same CSRv2 half-cosine training-K function;
their auxiliary budget follows 4K(t). Landmark keys refresh whenever K changes.
The current protocol is `mmrsae-imagenet-context-distillation-v4`; checkpoints
from earlier protocol/source versions cannot resume under this recipe. Dictionary
temperature and initial context blend default to 0.1. See
[MMRSAE_IMAGENET.md](MMRSAE_IMAGENET.md) for the full recipe and output schema.

`dlmmsaev2` is the new name for `dlmmsae-annealed`. Existing results with the
old name are displayed as DLMMSAEv2; reports filter out the other ablation
methods. Historical model implementations and saved run data remain available.
The launcher defaults to all four methods; an explicit `--variant dlmmsaev2`
runs that method once. The Python runner alone defaults to `dlmmsae`.

DLMMSAE adds a direct loss on the synthesized context before residual blending
or SAE Top-K. It distills the clean input's cosine relationship distribution
over a fixed training reference set Q, separate from synthesis landmarks A.
The student lookup receives Gaussian feature noise with standard deviation
0.02 times each row's RMS; Q self-matches are excluded consistently from both
distributions. Fixed keys need no EMA encoder or queue. The context weight
defaults to 1 only for DLMMSAE. This supplies a direct query-adapter gradient;
it does not guarantee that the learned context blend stays nonzero or improve
accuracy without measurement.

Endpoint evaluation now also fits a separate supervised linear classifier for
each requested K and for full-parent features, using fitting labels only and
keeping the backbone/SAE frozen. Classification Top-1 is reported separately
from 1-NN retrieval. Set `MMRSAE_LINEAR_PROBE=0` or `--no-linear-probe` to skip
that evaluation. Representation training GPU-hours exclude these classifiers,
diagnostics, checkpoint I/O and retrieval; their synchronized measurement is
stored separately from the broader elapsed `fit_seconds`.

Reports include accuracy versus K, measured active dimensions, retrieval batch
time and training GPU-hours, plus per-method backbone comparisons. PNG/PDFs,
per-K CSVs and `overview.md` are written under `runs/mmrsae/reports/`.
Enable `MMRSAE_DENSE_REFERENCE=1` for full-representation retrieval reference
lines when the dense gallery fits in GPU memory. The full-parent classification
reference is already included with linear probes. Missing measurements remain
missing; these plots do not import numbers or methods from the example papers.

**Docker Compose: configure once, then run.** On a Linux GPU server with Docker
Compose, an NVIDIA driver compatible with CUDA 13.0.2, and NVIDIA Container
Toolkit already installed:

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, WANDB_API_KEY, and your WANDB_ENTITY.
docker compose up
```

Use `HF_TOKEN` with read access to the prepared embedding repository
[quangnguyenopt/NSFW](https://huggingface.co/quangnguyenopt/NSFW/tree/main).
It is a private Hugging Face **model** repository, not a dataset repository.
Local NPZ/JSON pairs also work offline without a Hugging Face token.
W&B is online by default and Compose requires `WANDB_API_KEY` for monitoring.
Credentials stay in the ignored `.env` file. The Compose service uses
`.env`, independently of the shell wrapper's `imagenet.env`.

Both backbones now read prepared NPZ/JSON pairs, preferring files already in
`MMRSAE_EMBEDDINGS`. Development runs download only the two training pairs;
official evaluation additionally downloads validation. Stored RN18 archives
are mapped directly; compressed FF2048 features are streamed once into a
reusable disk map. Raw ImageNet and backbone extraction are unnecessary in
the default mode. See `MMRSAE_IMAGENET.md` for offline copies, cache space,
and the explicit `MMRSAE_EMBEDDING_SOURCE=legacy` mode.

Run these commands from this folder on the server running your experiment.
Docker Engine must be running before Compose can create containers; a working
`docker --version` only confirms that the command-line client is installed.
Run `docker info` on that server and confirm it shows a **Server** section
without an error. Starting Docker Desktop on your PC does not start Docker
on the remote server. GPU access and a compatible host driver are still required.
Running `docker compose` alone shows help; use `docker compose up` to launch.

Compose builds the existing image, checks GPU retrieval, prepares/downloads
data, fits ResNet-50 then ResNet-18, and evaluates the configured experiment.
It runs the current MMRSAE ImageNet workflow. The experiment defaults remain
K=8, seeds 1227/42, 500 updates per backbone, 128,000 fitting/gallery rows and
10,000 training-derived development queries. To explicitly request full-data,
ten-epoch official evaluation, set `MMRSAE_MAX_STEPS=0`, `MMRSAE_TRAIN_SIZE=0`,
`MMRSAE_EPOCHS=10`, and `MMRSAE_EVALUATION=official` before starting.

`GPU_DEVICE_ID=0` selects physical GPU 0, exposed as `cuda:0` inside the service.
Data/cache files persist in `./data`; results and checkpoints persist in
`./runs/mmrsae`. Compose creates those directories. Change `DATA_HOST_DIR` and
`OUTPUT_ROOT` in `.env` to use another disk. Complete training embedding
archives are still needed for subset fitting, but the feature matrices are
not loaded into RAM. Container files are
root-owned by default; optional `HOST_UID`/`HOST_GID` settings hand results back
to those numeric IDs when the existing launcher exits.

For a background run that survives an SSH disconnect:

```bash
docker compose up -d
docker compose ps -a
docker compose logs -f --tail 100 imagenet
```

`ps -a` also shows completed or failed containers, which are absent from the
default running-container list. Read the logs if `imagenet` has exited.
If no container appears, inspect the original `up` command's terminal output:
configuration or image-build errors can prevent container creation, leaving
no container logs to read.
Press **Ctrl+C** to stop following logs; the detached job keeps running.
With foreground `docker compose up`, **Ctrl+C** stops the job instead.

To save the available container output or open a shell inside a running job:

```bash
docker compose logs --no-color imagenet > imagenet-container.log
docker compose exec imagenet bash
```

Use `exit` to leave the shell. `exec` requires a running container; logs remain
available after it exits until the container is removed.

After updating the Python source on the server, rebuild the image to include
the fix; `docker compose restart` reuses the old container code. To check GPU
retrieval without downloading ImageNet or starting training, then launch:

```bash
docker compose run --rm --build imagenet --preflight-only
# Continue after preflight reports "gpu_search_probe": "passed".
docker compose up -d --build imagenet
docker compose logs -f --tail 100 imagenet
```

The console prints the W&B run URLs. `docker compose stop` stops the job;
`docker compose up` resumes compatible saved checkpoints with the same settings
and data. A completed job exits normally. The Compose launch rebuilds changed
source using Docker's layer cache, so keep the code unchanged for a resume.
This setup still requires GPU access from the host; containers cannot install
the host driver or Docker daemon.

**Simple server entry point.** Use [run_imagenet.sh](run_imagenet.sh) for short
commands. It defaults to the controlled ImageNet study in
`full_theory_detail.md` and accepts only the `repair` and `mmrsae` workflows.
On a Linux server with Docker and NVIDIA GPU access, initialize once:

```bash
bash run_imagenet.sh init
```

This creates `imagenet.env` and a private recipe copy, `imagenet.json`, without
overwriting existing files. In `imagenet.json`, fill `source.revision` and
`source.train_sha256` from trusted release metadata. Edit `imagenet.env` for
the server paths and GPU. For example, if the released archives are already
under `/srv/imagenet/FF2048_RN50_Embeds/`, use:

```bash
DATA_HOST_DIR=/srv/imagenet
DOCKER_GPUS=device=0
REPAIR_DEVICE=cuda:0
FF2048_TRAIN_EMBEDDINGS=/data/FF2048_RN50_Embeds/1K_train_ff2048.npz
FF2048_VAL_EMBEDDINGS=/data/FF2048_RN50_Embeds/1K_val_ff2048.npz
```

`DATA_HOST_DIR` must already exist and permit cache writes. Docker mounts it
at `/data`, so the archive paths use `/data/...` inside the container. Leaving
it empty retains the named-volume setup. `DOCKER_GPUS=device=0` selects server
GPU 0; the single selected GPU is `cuda:0` inside the container. The validation
archive is needed only for the later confirmation stage.

After configuring the files, the normal commands are:

```bash
bash run_imagenet.sh plan
bash run_imagenet.sh run
```

`plan` prints the recipe without building Docker or training. `run` performs
preflight, preparation, checks, and W/C/S/N, with 1,000 default fitting updates.
It leaves attribution, holdout and confirmation as explicit later stages.
Results default to `runs/mpsae_repair/`. In another terminal, follow the log:

```bash
bash run_imagenet.sh logs
```

Rerun the same `run` command to reuse compatible checkpoints and caches after
an interruption. Keep the code, recipe, data, output directory and visible GPU
configuration unchanged for resume. To keep the foreground job running across
SSH disconnects, use a persistent terminal session. If `tmux` is installed,
start `tmux new -s imagenet`, run the command there, and detach with **Ctrl-b,
then d**. Reconnect with `tmux attach -t imagenet`.

Later repair commands use the same entry point, for example
`bash run_imagenet.sh attribution`. After reviewing results, select
`REPAIR_CANDIDATE` in the environment file before `lock`, `holdout`, and
`confirm`; the existing gates still apply. `preflight` and `prepare` are also
available separately. Use `--help` for the complete command list.

For manual MMRSAE fits, initialize a separate file with
`bash run_imagenet.sh --env manual.env init mmrsae`, edit it, then use
`bash run_imagenet.sh --env manual.env run`. An already prepared Python/CUDA
environment can use `bash run_imagenet.sh --env local.env --local init`; edit
its local paths, then use `bash run_imagenet.sh --env local.env run`.
The chosen execution mode is saved in the file. The lower-level launchers
remain available below. These commands document server usage; full ImageNet
training and GPU performance measurements still require server execution.

The current comparison is **CSR, CSRv2, DLMMSAE and DLMMSAEv2**. The launcher
runs these methods and the selected budget on ResNet-50 and ResNet-18 sequentially, using
ImageNet only. It includes automatic data preparation, resumable feature and
training state, W&B monitoring, and mandatory GPU retrieval checks before data
work. Default fitting is capped at 500 updates per backbone on 128,000 training
rows, with a separate 10,000-row training-derived development query pool.

Start with [mmrsae.env.example](mmrsae.env.example) as `.env`, configure data
access and optional W&B credentials, then use:

```bash
bash run_all_experiments_docker.sh
```

The lower-level Docker launcher defaults to `EXPERIMENT_SCOPE=mmrsae`. Existing `.env` settings supply
defaults; explicit exported environment settings override them. The new scope
does not invoke transfer or the staged ablation study. For direct execution
in an existing environment use `ENV_FILE=your.local.env bash run_mmrsae_imagenet.sh`; CLI overrides
such as `--backbones resnet18 --variant dlmmsaev2 --k 16` select one manual
experiment. `--plan`, `--preflight-only`, and `--prepare-only` expose the
preparation steps separately. Official validation requires `--evaluation official`.

See [MMRSAE_IMAGENET.md](MMRSAE_IMAGENET.md) for variants, parameter settings,
data provenance, CUDA failure behavior, W&B and checkpoint reuse. CPU checks
are recorded in [VALIDATION.md](VALIDATION.md); they do not establish ImageNet
accuracy or GPU throughput. The legacy workflows
below remain available through explicit scopes and retain their historical
MPSAE method names.

## Budgeted repair and neighborhood experiment

The new research profile implements [full_experiment_detail.md](full_experiment_detail.md)
and [full_theory_detail.md](full_theory_detail.md). Its results live separately
from the strict FF2048/RN18 reproduction tables. Start by inspecting the compute
plan; this command uses only the Python standard library and launches no work:

```bash
python make_plan.py
```

The first training decision uses one 250-update original-MPSAE warm start and
three 250-update continuations: **C**, unchanged MPSAE; **S**, repair plus
teacher-neighborhood KL; and **N**, S plus mixed neighborhood OT. This totals
1,000 updates, with a common 128,000-row gallery and 2,000 cheap development
queries. Attribution and auxiliary-budget arms are separate later stages.
The 8,000-query gate is opened only after locking a candidate; full-data
confirmation starts fresh and costs four or six fits across two seeds. The
570-head transfer grid is not part of this research profile.

Repair adds one residual correction before the final Top-K projection. Both
gallery and query use clean repaired codes at inference. Teacher banks, masking,
KL, OT, and auxiliary reconstruction are training components. The frozen full
parent remains a separate unchanged output. The neighborhood objective is
tested against S so that OT is retained only if its measured benefit justifies
its cost. See [the experiment reference](EXPERIMENT_DETAILS_AND_BENCHMARKS.md)
and [method explanation](MPSAE_NOVELTY_VS_MRL_CSR.md) for the distinction from
the original MPSAE method.

The entry points are `mpsae_repair_experiment.py` and `run_mpsae_repair.sh`.
The shell entry point defaults to planning; it does not silently start training.
`EXPERIMENT_SCOPE=repair` selects this path in the Docker launcher, with
`REPAIR_STAGE=plan` as its default. The historical `all`, `imagenet`, and
`transfer` scopes retain their existing meaning. Select `repair.env` explicitly
with `ENV_FILE=repair.env`. Exported settings override file defaults, and a
positional stage (or `--stage`) overrides `REPAIR_STAGE`. Keep repair outputs
under a separate `SUITE_DIR=mpsae_repair`.
For Docker, set `REPAIR_CONFIG` to an existing **host** JSON file (relative to
this repository, or an absolute path). The host planner reads that file, and
execution mounts the same file read-only at a fixed container path. A custom
configuration does not need to be copied into the image or placed in `/data`.

The separate [repair.env.example](repair.env.example) selects this scope. The
following commands are usage examples; they have not been run against the final
implementation. Before a real-data check, set `source.revision` to an immutable
release commit and `source.train_sha256` to the archive SHA-256 obtained from a
trusted release record in a copy of `experiments.json`. The default null values
allow planning, but intentionally cannot authenticate data or permit training.
Confirmation additionally requires the trusted validation archive hash. Either
set `source.validation_sha256` before the first real-data check, or keep it null
and provide `--val-sha256` (shell/Docker `REPAIR_VAL_SHA256`) when entering
confirmation. The latter is locked in `validation_pin.json` before use; it does
not require editing the immutable pilot configuration. If both values are
present, they must agree. Do not derive trusted provenance solely from a locally
renamed file. No hashes or download revisions are invented here.

For the theory's ImageNet study, copy `repair.env.example` to `repair.env`, set
`REPAIR_CONFIG` to your pinned configuration, and place the training archive at
the configured path in the persistent data volume. Then use:

```bash
ENV_FILE=repair.env bash run_all_experiments_docker.sh plan
ENV_FILE=repair.env bash run_all_experiments_docker.sh pilot
```

`pilot` runs GPU preflight, config-matched split preparation, readiness checks,
and W/C/S/N in one container invocation. Its default budget is 1,000 optimizer
updates. It stops on an error and does not advance to attribution, holdout,
confirmation, or transfer. Rerunning reuses compatible caches/checkpoints.
You can also select `preflight`, `prepare`, `check`, or `triage` individually.
For an already prepared local environment, use
`ENV_FILE=repair.local.env bash run_mpsae_repair.sh pilot`, with local paths in
that file. Direct launchers read an explicit `ENV_FILE`; they do not load `.env`
automatically. Standalone `prepare_splits.py --config your_config.json` also
uses the configured pool sizes and source pins.

The check with training data records readiness for the exact configuration and
source. A mathematical check without a training archive does not satisfy the
training prerequisites. Weak semantic gradients at fresh initialization are
recorded as warnings; they do not prevent the original-MPSAE warm start. Before
a relational continuation spends its 250 updates, diagnostics are repeated on
the restored W state and unacceptable signals stop that branch. Triage handles
the shared warm start and C/S/N continuations. Inspect
`decisions.json` and the endpoint diagnostics before choosing attribution or
wide-budget stages. The CLI's `lock`, `holdout`, and `confirm` stages make the
candidate and subsequent compute decisions explicit. Do not select confirmation
merely to complete the matrix. [VALIDATION.md](VALIDATION.md) records the final
tree's unexecuted status and the earlier checks' limitations.

After the candidate passes the locked development gate, confirmation verifies
the pinned official validation archive and the full frozen parent's 71.19%
integrity reference **before** spending the four-to-six-fit budget. This is an
input/evaluation check after selection has closed. Sparse official-validation
scores are evaluated only after their fixed-epoch fits; they never select the
method, checkpoint, or training duration.

The research implementation streams the feature member of the training NPZ
into a reusable memory-mapped cache, uses indexed student-to-landmark scoring,
and saves resumable branch state. It evaluates endpoints rather than repeatedly
encoding the whole gallery during fitting. Budget reductions preserve the
declared losses, shared branch initialization, and paired data exposure.
The repair step-size estimate is cached once per shared W or fresh confirmation
seed; unchanged completed fits and endpoint evaluations are reused.

No archive-to-image mapping can be inferred from ImageNet class labels. A local
paired MRL comparison needs a verified aligned cache; without it the comparison
is unavailable. Similarly, FF2048 compressors cannot be applied to timm region
features just because both have dimension 2048. Reduced transfer is deferred
until a compatible parent and its separately budgeted extraction are available.

## Run ImageNet plus the new crop-transfer experiments

Set `EXPERIMENT_SCOPE=all` explicitly to run
the historical ImageNet experiment followed by VOC2007 detection, VOC2012
conditional object-mask prediction, and PASCAL3D+ known-box/category viewpoint
estimation. `run_full_pipeline.sh` remains an alias to the configured scope. Set `EXPERIMENT_SCOPE=imagenet`
to run only the earlier experiments, or `EXPERIMENT_SCOPE=transfer` to resume
only the new suite. Set these in the existing `.env` file.

The new suite follows `mrl_csr_rcnn_experiment_setting.txt` and
`mrl_csr_classical_task_extensions.txt`. It evaluates Matryoshka, CSR, MPSAE,
CSRv2, and MPSAEv2 on ResNet18 and ResNet50, with both full-parent references,
seeds 0/1/2, the prescribed affine/MLP readouts, and separate development and
final fitting. Optional regional adaptation, PCA, keypoints, texture, and local
descriptor extensions are outside this primary three-task run.

The transfer suite loads the hashed native MRL checkpoints with their BlurPool
architecture and resolves an immutable revision of each `timm/resnet*.a1_in1k`
parent. The native construction follows the [MRL source](https://github.com/RAIVNLab/MRL)
and [FFCV image model recipe](https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py).
ImageNet-only compressors are fitted once and reused across downstream tasks.
CSR uses the specified h=8192, batch=1024, FP32, 10-epoch Table 9 settings;
MPSAE reuses the existing transport objective. The v2 arms retain the existing
cosine annealing policy, with a separately fitted endpoint for each evaluated k
(start `max(64,4*k)`, endpoint k, annealing fraction 0.7). These additional arms
are method extensions, not fixed-k CSR reproductions. The existing ImageNet
training protocols are unchanged.

Missing VOC, PASCAL3D+, edge-model and parent-weight files are downloaded when
`TRANSFER_DOWNLOAD=1`. Data checks validate splits and annotations before fits;
all-task runs prepare downstream data and raw ImageNet training data before the legacy ImageNet study. The
strict transfer compressors need the **raw ImageNet training images**, even if
the earlier FF2048 study uses only released embeddings. The existing ImageNet
downloader validates that cache and uses `HF_TOKEN`; dataset access must already
be granted to that token. Unavailable downloads stop with a clear error. URL
overrides and local PASCAL3D+/edge-model paths are available in `.env.example`.

Caches use FP32 memory-mapped arrays and bounded image batches. Sparse readouts
retain latent coordinate indices and use an indexed first layer. Sparse files
include uint32 indices/counts and padded slots; reports include their actual
storage, rather than claiming padded files occupy only `8*nnz` bytes. Cached
full vectors are shared across prefix budgets. Keep enough persistent disk for
ImageNet, proposals, all four parent caches, checkpoints, and indexed codes.
Changing source, parent identity, or protocol invalidates reuse and requires a
new output/cache directory; the runner never silently accepts stale results.

Reruns resume transfer extraction and head checkpoints; compressors resume at
completed epochs. OS locks prevent concurrent writers to the same dataset,
cache, or results directory. Completed evaluations are reused. Training stops
on nonfinite losses/gradients or invalid rotations and writes the failure.
The old ImageNet training loop still restarts its fits, so use transfer-only
scope for a transfer restart. This implementation was statically reviewed;
training, dataset/model downloads, compilation, tests, and performance benchmarks were not
executed while adding it.

Monitor the persistent host output under `runs/<SUITE_DIR>/frozen_transfer/`:

- `logs/pipeline.log`, `events.jsonl`, `status.json`: progress and stage status.
- `failure.txt`: an exception trace if the run stops.
- `datasets.json`, `locked_schedules.json`, per-task `controls.json`: data and protocol audits.
- Per-task `data_fit.json`/`data_evaluation.json`: counts, exclusions, proposal
  recall at all three budgets, manifest hashes and annotation provenance.
- `sensitivity.json`: full-parent signal against priors, with predetermined
  affine/width-256 diagnostics when a parent fails to exceed its prior.
- `summary.json`, `summary.csv`, `transfer_results.zip`: per-seed scores,
  sample standard deviations, own-parent differences and paired image-bootstrap
  intervals for mask/pose. AP/IoU/accuracy are proportions; multiply their signed
  differences by 100 for percentage points. Rotation errors use degrees.
- `plots/`: PNG/PDF accuracy-versus-storage curves with own-parent reference
  lines, separate clean/translated conditions, and seed standard deviations.
- `RUN_COMPLETE.json`: written only after the requested transfer grid and
  summary finish. The suite root completion marker covers both stages for `all`.

Per-fit checkpoints, sample plans, and individual prediction artifacts remain
on persistent storage; the portable ZIP contains tables and provenance. Recorded
component timings do not constitute end-to-end deployment latency measurements.

For a prepared local Python/CUDA environment, the same suite can be launched
with `bash run_transfer_experiments.sh`. Inspect its configuration without
training or downloading using `bash run_transfer_experiments.sh --plan`; use
`--prepare-only` for dataset validation/preprocessing, including ImageNet training
data when sparse methods are selected. The Docker path handles
dependency setup. GPU retrieval uses FAISS for dense codes and the CUDA inverted
index for sparse codes; it preflights the compiler/backend and gallery memory.
SciPy remains a dependency for reading PASCAL3D+ MATLAB annotations, not retrieval.

The historical `imagenet` scope reproduces the visual FF2048 protocol of *Beyond
Matryoshka: Revisiting Sparse Coding for Adaptive Representation*
(arXiv:2503.01776) before comparing MPSAE. A second
`paper_rn18_study` protocol follows the paper's Appendix E.1 ResNet-18 setup.

The previous experiment did not use the paper's FF2048 model. It loaded
torchvision ImageNet-1K V1 weights, fine-tuned MRL for 10 epochs, trained one
CSR model at K=32, and reused that model at every evaluation K. Those choices
describe a useful custom ablation, but they cannot reproduce the paper's MRL
or CSR numbers.

The CSR paper also reports a stronger timm ResNet-50d backbone (75.19% full
1-NN and about 74.9% with CSR). That is the best absolute ResNet result in its
Table 4, but it is not the FF2048/MRL-controlled comparison. This repository
therefore uses FF2048 for its strict paper comparison. The current MMRSAE
workflow reports its selected parents and local measurements separately.

## Historical strict ImageNet protocol

Within the explicit `imagenet` scope, the default protocol is paper_ff2048:

- Backbone features are the authors' released ImageNet embeddings from
  W1nd-navigator/CSR-precompute-embeds, generated by the FF2048 ResNet-50
  backbone selected for the paper's MRL comparison.
- MRL values are the published ResNet-50 exact 1-NN values from Table 4.
  They are marked as reference values rather than falsely presented as a
  locally retrained model.
- CSR and MPSAE train an independent fixed-K model for every reported
  K in 8, 16, 32, 64, 128, 256, 512, 1024, and 2048.
- The FF2048 input is checked locally at K=2048 against the reported
  71.19% exact 1-NN result. Strict mode stops before sparse training if the
  absolute gap exceeds 0.05 percentage points.
- A dedicated paper_ff2048_reproduction.csv reports the published MRL and CSR
  curves, the locally reproduced CSR curve, its gap from the paper, and MPSAE.

Paper Table 3 settings are enforced by default:

| Setting | Value |
|---|---:|
| Input dimension d | 2048 |
| SAE width h | 8192 |
| Epochs per K | 10 |
| Batch size | 4096 |
| Adam learning rate | 4e-5 |
| Adam weight decay | 1e-4 |
| Adam epsilon | 6.25e-10 |
| Auxiliary K | 512 |
| Auxiliary coefficient | 1/32 |
| Multi-TopK coefficient | 1/8 |
| CSR contrastive coefficient | 0.1 |
| Contrastive temperature | 0.2 |
| Dead-latent threshold | 30 forward updates |
| Sparse precision | float32 (AMP and TF32 off) |

The local CSR implementation also follows the released code in the details
that materially affect performance: zero pre-bias initialization, tied
encoder/decoder weights, normalized auxiliary MSE, L2-normalized two-view NCL,
temperature scaling, and no post-step decoder renormalization.

MPSAE is matched to CSR on the FF2048 inputs, initialization, optimizer,
epochs, data order, K-specific training, and base reconstruction objective.
Its Top-2K view is used by MMPOT but is no longer given an additional
reconstruction advantage. Extra MPSAE epochs default to zero.

## ResNet-18 Appendix E.1 study

FF2048 is a ResNet-50 representation with d=2048. Neither the MRL paper nor
the CSR paper reports an FF2048/MRL ResNet-18 baseline, so ResNet-18 remains
separate from the headline MRL comparison.

The `paper_rn18_study` protocol matches the configuration that the CSR paper
does report:

| Setting | Value |
|---|---:|
| Frozen backbone | `timm/resnet18.a1_in1k` |
| Released feature-extraction batch | 256 |
| Input dimension d | 512 |
| SAE width h | 8192 |
| Epochs per K | 10 |
| Batch size | 1024 |
| Adam learning rate | 4e-5 |
| Adam weight decay | 1e-4 |
| Adam epsilon | 6.25e-10 |
| Auxiliary K / coefficient | 512 / 1/32 |
| Multi-TopK coefficient | 1/8 |
| CSR contrastive coefficient | 1.0 |
| K | 8, 16, 64, 256 |
| Repetitions | 3 matched, predeclared local seeds |

It measures the full dense FF512 1-NN reference, trains an independent CSR
model and an independently matched MPSAE model for every K and seed, then
reports mean and sample standard deviation. It deliberately has no MRL
column: adding one would require a new, separately specified ResNet-18 MRL
training study rather than a paper reproduction. Figure 4 only plots the CSR
curve, so the harness does not treat digitized plot coordinates as exact
targets. The MPSAE-minus-CSR effect is also reported as a paired-seed mean and
sample standard deviation.

The paper says that it averages three random seeds but does not publish their
values. This harness predeclares 1227, 1228, and 1229 (anchored at the released
code's default 1227) and records them in every result. Feature extraction
follows the released pipeline: train uses random-resized-crop 224 plus
horizontal flip, while validation center-crops at ratio 224/256 and resizes
the crop to 256 square; both use mean/std 0.5. This is a semantic torchvision
translation of FFCV, not a claim of bit-exact FFCV execution.

One source-level difference is recorded explicitly: the released FFCV
extractor uses random ordering and drops its final incomplete training batch.
That order and its seed are not published, so this harness cannot identify the
same omitted 143 images. It instead retains all 1,281,167 training examples
for both methods and for the dense gallery. This is a configuration-matched
study, not a claim of bitwise or exact numerical reproduction of Figure 4.

The older torchvision ResNet-18/50 study remains available under the explicit
torchvision_ablation protocol. Its output is labelled non-reproduction and
must not be pooled with the paper comparison.

## GPU retrieval measurement

Retrieval uses exact squared L2 distance after row normalization. Dense inputs
use CUDA FAISS `IndexFlatL2`. CSR, MPSAE, and their annealed variants use the
CUDA inverted index in `sparse_gpu_retrieval.py` and `csrc/`: each active query
coordinate visits the gallery entries containing that coordinate. Sparse
codes remain indices and float32 values; retrieval does not expand the
gallery to an N-by-8192 dense matrix or use CPU SciPy multiplication.

The inverted index and encoded queries reside on the selected
`FAISS_GPU_DEVICE`. The full gallery must fit; insufficient device memory is
an error, with no CPU fallback or hidden gallery transfers during search.
`SPARSE_GALLERY_CHUNK` bounds the temporary score workspace, not the resident
gallery. Its actual rows are also capped by `4*1024**2` compact entries per
chunk, bounding the builder's sorting-workspace estimate to approximately
384 MiB. Large K can therefore produce smaller chunks than requested;
results record the actual chunk count and largest chunk. Scoring, workspace
initialization, reduction, and the merge of
chunk winners all belong to timed search. This still entails score-buffer
and top-1 work over gallery chunks; a sparse code alone does not guarantee a
speedup, especially at large K or for frequently activated coordinates.

Dense and sparse searches default to the same query batch of 32. Each uses
three untimed warmups for every distinct batch shape and five measured
repetitions per query batch. Repetitions are interleaved by batch; totals
across all batches give each full-query-set measurement. Both CUDA-event
time and synchronized wall time are reported with median and spread. Encoding,
normalization, host-to-device transfers, extension compilation, index
construction, and label lookup are excluded. Time per query is the total
search time divided by query count: it is **amortized time at the reported
batch size**, not isolated single-query latency. Compare measurements with
matching device, query batch, precision, gallery, queries, and timing scope.
Unit normalization rejects zero-norm codes. Gallery rows with no coordinates
in common with a valid query still participate in exact search. Sparse
float32 atomic accumulation can change nearly tied scores across repetitions;
the implementation does not claim bitwise agreement with dense FAISS.
Accuracy comes from one separate untimed search per batch, independent of
which timing repetition is fastest. Results also count the actual visited
posting products and compare that work with N-by-H dense multiplication in
the same latent space. This operation-count fraction is not a speedup estimate.

| Environment setting | Default | Meaning |
|---|---:|---|
| `FAISS_GPU` | `1` | Required by the GPU experiment launchers |
| `FAISS_GPU_DEVICE` | `0` | Shared dense and sparse retrieval GPU |
| `KNN_QUERY_BATCH` | `32` | Dense search batch |
| `SPARSE_KNN_QUERY_BATCH` | `32` | Sparse search batch; must equal `KNN_QUERY_BATCH` |
| `SPARSE_GALLERY_CHUNK` | `65536` | Maximum gallery rows per score chunk, subject to memory limits |
| `RETRIEVAL_WARMUP` | `3` | Untimed repetitions per batch shape; at least 1 |
| `RETRIEVAL_REPEATS` | `5` | Timed repetitions per query batch; at least 1 |
| `SPARSE_SCORE_BUFFER_MIB` | `256` | Score workspace budget |
| `SPARSE_GPU_MEMORY_FRACTION` | `0.8` | Total-device memory ceiling; reserves the remaining fraction as headroom |

The CUDA extension needs a CUDA development toolkit with `nvcc` compatible
with the installed PyTorch, a C++ compiler, and Ninja. The supplied Docker
image includes these and caches extension builds under
`/data/torch_extensions`. First-use compilation happens after gallery
construction and before warmup or timed search. A CUDA-enabled
FAISS installation is required; installing `faiss-cpu` cannot support this
benchmark. Platforms without a matching GPU wheel must supply a compatible
CUDA-enabled FAISS build from source or use the Linux Docker image. Native
Windows also requires `nvcc` and the MSVC C++ toolchain for the local CUDA
extension; `requirements.txt` does not install a CPU FAISS substitute.

Published MRL reference accuracies have no local retrieval time. Speed
comparisons use locally measured dense representations, and the reference
curve must not be given a synthetic latency or speedup. The FF2048 integrity
measurement is reused as a single **Dense FF2048** point at dimension 2048
in the timing table and plots. It is distinct from the published MRL curve
and does not supply measured dense-prefix times at smaller K. The retrieval
implementation was changed by static reasoning only; no execution,
compilation, import check, or benchmark was performed for this change.
No measured correctness or speed improvement is claimed before execution.

## Run

Accept the access conditions for the authors' Hugging Face embedding dataset,
then:

~~~bash
cp .env.example .env
# Put a read-only Hugging Face token in .env.
bash run_all_experiments_docker.sh
~~~

The container downloads:

- FF2048_RN50_Embeds/1K_train_ff2048.npz
- FF2048_RN50_Embeds/1K_val_ff2048.npz

The archives live in the persistent Docker data volume and are reused.

To use already downloaded files outside Docker:

~~~bash
python csr_vs_mmpot_imagenet.py \
  --protocol paper_ff2048 \
  --data-root /path/to/CSR-precompute-embeds \
  --backbone resnet50 \
  --method all
~~~

Strict mode rejects changes to paper-controlled settings. Use
--no-paper-strict only for a deliberately labelled ablation.

For the ResNet-18 study:

~~~bash
cp rn18_study.env.example .env
# Put a read-only Hugging Face token with ImageNet-1K access in .env.
bash run_all_experiments_docker.sh
~~~

The corresponding direct entry point is:

~~~bash
python csr_vs_mmpot_imagenet.py \
  --protocol paper_rn18_study \
  --data-root /path/to/imagenet \
  --data-backend imagefolder \
  --backbone resnet18 \
  --method all
~~~

## Main outputs

The ResNet-50 output directory contains:

- paper_ff2048_reproduction.csv and .md: primary MRL/CSR/MPSAE table.
- paper_ff2048_reproduction_comparison.png and .pdf: primary fixed-K figure.
- summary.json: full protocol, provenance, FF2048 integrity check, histories,
  and all metrics.
- retrieval_timing.csv and .md: locally measured GPU search times and the
  measurement settings used to interpret them.
- csr/k_K/history.json and mpsae/k_K/history.json: independent fixed-K runs.
- csr/results.json and mpsae/results.json: merged per-K evaluation results.
- csrv2 and mpsaev2: clearly separate annealing research extensions.

The suite root additionally contains `ff2048_reproduction_per_budget.csv`,
`ff2048_reproduction_results.md`, and
`imagenet_ff2048_comparison_results.zip`.

The ResNet-18 protocol instead writes `paper_rn18_study.csv/.md` inside the
backbone directory and `rn18_study_per_budget.csv`,
`rn18_study_results.md`, and
`imagenet_rn18_csr_mpsae_study_results.zip` at the suite root.

Trained model and optimizer states are intentionally not persisted. Histories,
metrics, figures, tables, logs, and the released input embeddings are retained.

## Source references

- CSR paper: https://arxiv.org/abs/2503.01776
- Official CSR code: https://github.com/neilwen987/CSR_Adaptive_Rep
- Official FF2048 embeddings:
  https://huggingface.co/datasets/W1nd-navigator/CSR-precompute-embeds
- ResNet-18 model used by CSR Appendix E.1:
  https://huggingface.co/timm/resnet18.a1_in1k
- Original MRL code/checkpoints: https://github.com/RAIVNLab/MRL
