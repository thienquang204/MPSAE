**MMRSAE: manual ImageNet retrieval experiments**

MMRSAE names the current method family: a sparse autoencoder with one learned
residual repair step, trained with reconstruction and configurable neighborhood
objectives. Its public implementation is [mmrsae.py](mmrsae.py). The current
runner is [mmrsae_imagenet.py](mmrsae_imagenet.py); the original MPSAE baseline,
strict paper studies, and staged repair study retain their historical names.

The default workflow trains **one chosen variant at one K and one seed for each
of ResNet-50 and ResNet-18, sequentially**. It runs only ImageNet. It does not
start the transfer suite, a parameter sweep, attribution arms, or promotion
gates. Parameters are set by hand and recorded with each run. This is an
exploratory experiment, not a strict reproduction or a claim of improvement.

**This implementation has received source review only.** No launcher, test,
download, dependency installation, training job, or benchmark was executed for
this change, as requested. The checks described below are implemented checks
that run when the user launches the experiment; they are not completed checks.

The supplied presets are:

| Variant | Reconstruction and repaired sparse code | Neighborhood KL | Transport | Auxiliary raw reconstruction |
|---|---|---|---|---|
| `mmrsae` | Shared tied SAE; masked starting code during training | Enabled | Mixed angular/neighborhood cost | 4K |
| `mmrsae-rel` | Same | Enabled | Disabled | 4K |
| `mmrsae-wide` | Same | Enabled | Mixed cost | min(h, max(64, 8K)) |
| `mmrsae-annealed` | Same | Enabled | Mixed cost | Wide budget decreases to 4K over the first 70% of fitting |
| `mpsae` (baseline) | Same tied SAE, no repair or masking | Disabled | Angular cost only | 4K |
| `csr` (baseline) | Same tied SAE, no repair or masking; adds CSR NCL (gamma 0.1 for ResNet-50, 1.0 for ResNet-18, temperature 0.2) | Disabled | Disabled | 4K |

The two baselines share the SAE initialization, optimizer, data order, budget
and endpoint retrieval with the MMRSAE presets, so their scores are directly
comparable within one protocol. They serve the plain Top-K code.

The served budget remains K for every preset. K and 2K use repair; the third
transport view is raw 4K. The wide budget replaces the existing reconstruction
branch with weight 1/8; it does not add a second wide reconstruction term.
All presets produce clean repaired codes for both gallery and query at
inference. Masking, landmark profiles, KL and transport are training work.
There is no transport solver in retrieval. Manual weight and masking overrides
are recorded separately from the preset name.

**Launching and setting parameters.** For Docker, copy
[mmrsae.env.example](mmrsae.env.example) to `.env`, set the Hugging Face token
and optional W&B credentials, then use the existing entry point:

```bash
bash run_all_experiments_docker.sh
```

`run_full_pipeline.sh` is an alias. Both default to `EXPERIMENT_SCOPE=mmrsae`.
An existing `.env` supplies defaults; exported variables override them. Select
`ENV_FILE=mmrsae.env` to use a separate configuration file. `imagenet` selects the historical
strict runner; **`mmrsae` selects this new ImageNet-only workflow**. The
`all` and `transfer` scopes continue to be opt-in legacy workflows.
Adding `--plan` to the Docker launcher prints the configuration on the host
without building a container or requiring data credentials.

The Docker image installs the declared dependencies and includes the CUDA
compiler needed by the sparse retrieval extension. Data, parent weights and
compiled extensions persist in the named data volume; reports/checkpoints
persist under `runs/mmrsae/` on the host. Paths configured as `/data/...` or
`/output/...` refer to the container. Local files on the host require a mount
or placement in the persistent data volume before they can be used there.

In an already prepared environment, the direct launcher consumes exported
variables or an explicit `ENV_FILE`, and accepts ordinary CLI overrides:

```bash
bash run_mmrsae_imagenet.sh --plan
ENV_FILE=mmrsae.local.env bash run_mmrsae_imagenet.sh --plan
bash run_mmrsae_imagenet.sh --preflight-only
bash run_mmrsae_imagenet.sh --prepare-only
bash run_mmrsae_imagenet.sh --backbones resnet18 --variant mmrsae-rel \
  --k 16 --lr 4e-5 --max-steps 500 --run-name rn18_rel_k16
```

These are usage examples, not commands executed during implementation. The
direct shell launcher reads only an explicitly selected `ENV_FILE`; Docker
loads `.env` by default. Use local paths in a direct-launch configuration.
Explicit CLI arguments override exported values, which override file defaults. `--plan` prints
configuration without GPU initialization or data preparation; the other modes
require the CUDA retrieval check. No CPU execution mode is provided.

| Manual setting | Default | Interpretation |
|---|---|---|
| `MMRSAE_BACKBONES` / `--backbones` | `resnet50,resnet18` | Comma-separated selected architectures |
| `MMRSAE_VARIANT` / `--variant` | `mmrsae` | One preset, no automatic sweep |
| `MMRSAE_K` / `--k` | 8 | Maximum active coordinates per served code |
| `MMRSAE_HIDDEN_DIM` / `--hidden-dim` | 8192 | Dictionary width |
| `MMRSAE_MAX_STEPS` / `--max-steps` | 500 | Per-backbone update cap; zero removes the cap |
| `MMRSAE_EPOCHS` / `--epochs` | 10 | Epoch limit, also enforced when a step cap exists |
| `MMRSAE_BATCH_SIZE` / `--batch-size` | 1024 | Compressor fitting batch |
| `MMRSAE_TRAIN_SIZE` / `--train-size` | 128000 | Fitting/gallery rows; zero selects the available full pool |
| `MMRSAE_DEV_SIZE` / `--dev-size` | 10000 | Training-derived development queries |
| `MMRSAE_ANCHOR_COUNT` / `--anchor-count` | 1024 | Ordered training landmark bank |
| `MMRSAE_SEED` / `--seed` | 1227 | Model initialization, masking and fitting |
| `MMRSAE_DATA_SEED` / `--data-seed` | 42 | Stable allocation/extraction across model-seed changes |
| `MMRSAE_EVAL_EVERY` / `--eval-every` | 0 | Endpoint only; a positive completed-epoch interval adds gallery evaluations |
| `MMRSAE_CHECKPOINT_EVERY` / `--checkpoint-every` | 25 | Updates between resumable checkpoints |

The default ceiling is 1,000 fitting updates across two backbones. This count
excludes data preparation, encoding, search and initialization. Smaller steps
or one backbone reduce fitting cost; `--no-timed-retrieval` avoids repeated
timed searches while retaining endpoint accuracy. Full-parent dense reference
search is an explicit `--dense-reference` option because its gallery can be
large. It is not required for every manual trial.

**Data preparation and split boundaries.** ResNet-50 uses the released
`FF2048_RN50_Embeds/1K_train_ff2048.npz` from
`W1nd-navigator/CSR-precompute-embeds`. ResNet-18 uses the frozen
`timm/resnet18.a1_in1k` parent and raw ImageNet images. The data adapter resolves
remote revisions to immutable commits, records provenance, checks available
archive hashes and schemas, and validates official source sizes and labels.
Dataset access must already be granted to the supplied Hugging Face token;
authorization failures stop preparation.

FF2048 features are streamed to reusable read-only memory maps. RN18 extracts
only the selected fitting, development and landmark rows for the default
subset experiment, and stores float32 features in a resumable cache. It does
not repeat parent extraction for each model seed or loss setting. The parent
and extraction recipe remain part of the cache identity. A prepared
ImageFolder source is available through `--data-backend imagefolder`.

Default `--evaluation dev` uses disjoint training-derived queries and does
not download/open official validation. Labels allocate/evaluate the pools;
reconstruction, repair and neighborhood targets do not use class labels.
Landmarks come from training. Whenever a fitting row is also a landmark,
the minibatch's union of row IDs is removed from all profiles for that update.

Use `--evaluation official` explicitly for an official-validation measurement
after choosing settings. Source/parent/recipe fingerprints remain in the
results. Tuning on those results must be described as validation-informed
selection. A subset-gallery score is not the full-gallery published benchmark;
the 71.19% FF2048 reference applies only to its original full-gallery setting.
No published MRL values are treated as local measurements in this workflow.

`--no-download` requires compatible cached resources or explicitly supplied
local resources with the required provenance. Optional `--ff2048-train` and
`--ff2048-val` paths must refer to the intended released archives; filename
and shape alone do not authenticate a release. See the environment template
for revision/hash overrides. Changing a data or extraction recipe uses a
different cache identity rather than silently reusing incompatible features.

**GPU retrieval and cost reporting.** Before dataset work, the runner checks
the selected CUDA device, GPU FAISS and the custom sparse CUDA extension. It
performs small actual searches to check their operation; compilation or
backend failures raise immediately. CPU-only FAISS and unavailable CUDA are
errors. The full sparse index is also subject to a device-memory budget and
fails if it cannot fit. Runtime CUDA or search failures propagate; there is
no silent CPU fallback. These checks establish backend availability at run
time, not a guaranteed throughput on every GPU.

Retrieval is exact normalized 1-NN. Sparse indices retain coordinates and
FP32 values without expanding the full gallery to width 8192. Chunked scores
limit working memory while all gallery postings remain GPU-resident. Default
query batch is 32, score budget 256 MiB, gallery chunk ceiling 65,536 rows and
index memory fraction 0.8. CUDA-event and synchronized-wall search times are
reported separately from encoding/repair and index construction. Timed search
uses three warmups and five repetitions; reported time per query is amortized
over the configured batch, not isolated single-query latency. A padded K=8
indexed FP32 code uses 68 bytes before file metadata.

**Monitoring, resume and outputs.** Set `WANDB_API_KEY`, optionally
`WANDB_ENTITY`, and `WANDB_PROJECT=MMRSAE` for online monitoring. With no key,
the launcher defaults to offline W&B logging; `WANDB_MODE=disabled` disables
it explicitly. Each backbone has its own run, grouped by the experiment.
Loss components, training progress and endpoint retrieval metrics are logged;
local artifacts remain available independently of the dashboard. Configured
online monitoring failures are surfaced rather than silently disabling it.

Resolved configuration, provenance, split identities, fitting history,
checkpoints, retrieval metrics and per-query predictions are kept per run and
backbone. A compatible restart restores model/optimizer, mask RNG, data order,
landmark refresh state and update position. Reuse the same command/output
directory to resume. A different recipe gets a different derived name unless
one was specified; incompatible state under an explicit name is rejected.
Only one writer should own a run directory.

Source-level savings include bounded preparation/extraction, reusable parent
features, indexed sparse reductions, batched independent OT solves, cached
clean landmark keys and endpoint-only evaluation by default. These are design
choices visible in the code; no speedup or accuracy result is claimed before
execution. [VALIDATION.md](VALIDATION.md) records that limitation.
