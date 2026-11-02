**MMRSAE: manual ImageNet retrieval experiments**

MMRSAE names the current method family: a landmark dictionary synthesizes a
contextual embedding before a tied sparse autoencoder applies Top-K. There is
no post-SAE residual repair. Training uses reconstruction, landmark KL and
angular partial OT. Its public
implementation is [mmrsae.py](mmrsae.py). The current
runner is [mmrsae_imagenet.py](mmrsae_imagenet.py); the original MPSAE baseline,
strict paper studies, and staged repair study retain their historical names.

The Python runner trains **one chosen variant at one target K and one seed for each
of ResNet-50 and ResNet-18, sequentially**, then evaluates each checkpoint at
multiple inference budgets. The supplied environment templates run nine methods:
`mmrsae`, `mmrsae-rel`, `mmrsae-wide`, `mmrsae-annealed`, `mpsae`, `csr`, `csrv2`,
`dlmmsae`, and `dlmmsae-annealed`.
It runs only ImageNet. It does not
start the transfer suite, a parameter sweep, attribution arms, or promotion
gates. Parameters are set by hand and recorded with each run. This is an
exploratory experiment, not a strict reproduction or a claim of improvement.

**Validation scope.** Prepared-embedding I/O has CPU checks for local/offline
loading, targeted Hub downloads, checksum reuse, direct NPZ mapping and split
boundaries. Full ImageNet fitting and Docker/CUDA retrieval still require
server validation. See [VALIDATION.md](VALIDATION.md) for the check record.

**Landmark dictionary before SAE.** Let A contain fixed training-only landmark
features, D have shape h by d, and H_K^+ retain the K largest signed scores
followed by ReLU. The current MMRSAE path is

```text
q = x + ReLU(x U) V
w = softmax(cosine(q, A) / dictionary_temperature)
context = w A
blend = sigmoid(blend_logit)
x_dict = x + blend * (context - x)
a = (x_dict - b_pre) D.T + b_enc
z_K = H_K^+(a)
x_hat = z_K D + b_pre                 # reconstruction target is original x
```

The query adapter U/V uses a rank-32 bottleneck by default; U starts random
and V starts at zero. Temperature defaults to 0.1. The trainable blend starts
at 0.1, so the synthesis initially mixes 90% original embedding with 10%
landmark context. This direct input path avoids replacing x outright with a
landmark barycenter; it is not an information-preservation guarantee. There is
no alpha calibration or IHT step. The final code has at most K positive values.

The fixed landmark features and IDs are initialized from training anchors and
saved in the model checkpoint. Cosine lookup normalizes features, while the
context uses their original values. Main synthesis lookup uses all landmarks
consistently in training and inference. The runner reserves A outside fitting,
gallery and development queries, including when `--train-size 0` selects every
eligible fitting row. Official queries come from a separate validation source.
The public dictionary API can additionally exclude a supplied row ID for the
DLMMSAE auxiliary lookup; the held-out-A experiment policy already prevents
self matches in its main training and inference paths.

All clean budgets use the same SAE preactivation vector a after landmark
synthesis. Therefore clean K/2K/4K/auxiliary supports are nested, including when
each inference budget is encoded separately. During fitting, masking applies
to the full dense positive SAE activations before the final projection:
20% of rows are forced clean; in the remaining rows each positive coefficient
is erased independently with probability 0.25, without rescaling. One masked
activation vector is shared by the K and 2K views; 4K and the auxiliary
reconstruction view use clean activations. The masked views need not nest
inside the clean 4K view. Inference and KL-bank refresh use clean activations.

The input reaches landmark synthesis before any Top-K operation. The fixed
landmark dictionary is required at inference, adding model-buffer memory and
query-to-landmark lookup cost. The separate KL code bank and OT solver are
training-only. Encoding/training cost must be measured; the architecture
change itself establishes no accuracy or speed advantage.
The historical `RepairModule` and staged `EXPERIMENT_SCOPE=repair` study retain
their sparse-start architecture. They are separate from this current runner.

The supplied presets are:

| Variant | Sparse representation and training K | Neighborhood KL | Transport | Auxiliary reconstruction budget B |
|---|---|---|---|---|
| `mmrsae` | Landmark synthesis, then tied SAE Top-K; fixed training K | Enabled | Angular cost only | min(h, 4K), clean SAE scores |
| `mmrsae-rel` | Same | Enabled | Disabled | 4K |
| `mmrsae-wide` | Same | Enabled | Angular cost only | min(h, max(64, 8K)) |
| `mmrsae-annealed` (MMRSAEv2) | Same architecture; half-cosine training-K annealing | Enabled | Angular cost only | min(h, 4K(t)), clean SAE scores |
| `dlmmsae` | Landmark-first SAE plus direct raw-context relational distillation; fixed training K | Enabled, plus context KL over separate Q | Angular cost only | min(h, 4K) |
| `dlmmsae-annealed` | DLMMSAE with half-cosine training-K annealing | Enabled, plus context KL over separate Q | Angular cost only | min(h, 4K(t)) |
| `mpsae` (baseline) | Plain tied SAE Top-K, no landmark synthesis or masking | Disabled | Angular cost only | min(h, 4K), raw scores |
| `csr` (baseline) | Plain tied SAE; CSR NCL (gamma 0.1 for ResNet-50, 1.0 for ResNet-18, temperature 0.2) | Disabled | Disabled | min(h, 4K), raw scores |
| `csrv2` (baseline) | CSR with half-cosine training-K annealing, then fixed target K | Disabled | Disabled | 4 times the current training K |

The three baselines share the SAE initialization, optimizer, data order, target budget
and endpoint retrieval with the MMRSAE presets. They serve the plain Top-K code.
MMRSAE adds computation and parameters, so equal update counts do not mean equal
compute budgets. Report encoding time and fitting cost alongside accuracy.

**DLMMSAE context objective.** [dlmmsae.py](dlmmsae.py) implements a separate
training reference bank Q. By default it contains 1,024 deterministic fitting
rows, sampled with data-seed stream 9131, disjoint from the synthesis bank A.
The runner requires at least three references so a query belonging to Q still
has two alternatives after excluding itself. Both teacher and student use
the same ordered references and the same row-specific exclusion mask:

```text
x_tilde = x + context_noise_std * RMS(x) * normal_noise
c = synthesis(x_tilde; A)              # raw context, before blend and SAE
p_teacher = softmax(cosine(stop_gradient(x), Q) / context_teacher_temperature)
p_context = softmax(cosine(c, Q) / context_student_temperature)
L_context = mean KL(stop_gradient(p_teacher) || p_context)
L_DLMMSAE = L_reconstruction + relation_weight * L_sparse_KL
           + context_weight * L_context + ot_weight * L_angular_OT
```

Noise affects only the auxiliary lookup query; the reconstruction target,
teacher, main SAE path and inference stay clean apart from the existing
pre-Top-K training mask. This is feature perturbation, not image augmentation.
Defaults are context weight 1, both context temperatures 0.1, and noise standard
deviation 0.02 relative to each input row's RMS. Weight 0 disables this component
for an explicit ablation; existing methods default to 0 and reject a nonzero
context weight. Q and A contain fixed original embeddings, so their keys do
not require a momentum encoder or a FIFO queue. Q is training-only, while A
remains part of inference.

The auxiliary gradient reaches the query adapter even when the blend is near
zero or Top-K has no active units. It does not constrain the blend to stay
positive, guarantee useful context, or establish an accuracy improvement.
DLMMSAE logs context KL, teacher entropy, context-to-input cosine, context norm,
self exclusions and, in diagnostics, the context adapter gradient norm.

**All-zero codes.** The final Top-K/ReLU can produce fewer than K positive
coefficients or an empty code. A zero final code receives no gradient through
that gate. Identical nonzero views have zero diagonal angular cost, but this
does not make the full OT matching gap zero; three all-zero views have angular
cost 2/3 under the implemented normalization. As in
the original MPSAE runner, `--allow-empty-codes` (default; `MMRSAE_ALLOW_EMPTY_CODES=1`)
does not stop on such codes and does not change any objective: they keep a zero
direction in the OT cost and zero cosine to landmarks, stay zero rows in the
retrieval index, and an empty query is recorded as a miss. Training logs
`final_nnz_min`, `final_nnz_below_k_fraction`, `final_zero_code_fraction`,
`raw_kth_preactivation_nonpositive_fraction`, `tail_ratio_2k_mean`,
`tail_ratio_4k_mean` and `encoder_bias_mean`; retrieval reports
`empty_gallery_codes`, `empty_query_codes` and `empty_query_fraction`.
`--no-allow-empty-codes` restores the strict abort. The separate repair profile
keeps its strict check.

**Reports and figures.** After every fitting invocation (one method or
`MMRSAE_VARIANTS`, dev or official, complete or failed),
[mmrsae_reporting.py](mmrsae_reporting.py) writes
`<output>/reports/<evaluation>_k<K>_<id>/`: `results.csv`, `results.md`, and PNG
and PDF figures. The overview contains:

| Figure | Measurement |
|---|---|
| `top1_by_k` | 1-NN accuracy vs every measured inference K, one panel per backbone |
| `top1_by_k_backbones` | RN18/RN50 curves and full-representation baselines, one panel per method |
| `classification_top1_by_k` | Independently trained linear-probe Top-1 classification vs K |
| `accuracy_by_active_dim` | Classification and 1-NN side by side, against measured mean positive coordinates (queries/gallery respectively) |
| `top1_vs_retrieval_ms` | 1-NN accuracy vs warm GPU search time per query batch; K labels at markers; excludes encoding |
| `top1_vs_gpu_hours_<backbones>` | 1-NN vs measured representation-training GPU-hours at each K; combined panels use macro accuracy and summed hours over the same backbones |
| `top1_by_method` | Endpoint accuracy at the configured training target K |
| `training_loss`, `dictionary_and_schedule`, `support_health` | Reconstruction/context losses, blend, annealing K and sparse-code health |

Full-representation retrieval references require `MMRSAE_DENSE_REFERENCE=1`
(`--dense-reference`) and enough GPU memory for the dense gallery. Full-parent
classification is always measured when linear probes are enabled and does not
need that index. Missing classification, timing or GPU-hour measurements in
older outputs are omitted; neither paper values nor zeros fill those gaps.
`results.csv` exports per-K numbers, `compute_tradeoff.csv` exports compute
comparisons, and `plot_coverage.json` plus `overview.md` identify available plots
and missing measurements. A synthetic plot-layout test is not an experimental
accuracy result.

Trials are grouped by settings except the method, implementation identity and
GPU hardware/software. Timing curves additionally separate different query
batches and gallery/query identities. This keeps target K, data sizes,
evaluation pools and code versions distinct.
`reports/index.json` lists the groups. `--plan`, `--preflight-only` and
`--prepare-only` produce no report; `MMRSAE_REPORT=0` disables it. Regenerate at
any time without a GPU: `python mmrsae_reporting.py --output-dir runs/mmrsae`.

All enabled MMRSAE/MPSAE transport objectives use angular cost only.
`MMRSAE_MIXTURE_WEIGHT=0` is enforced; nonzero overrides fail before data work.
Landmark teacher-to-student KL stays separate from OT. CSR/CSRv2 and
`mmrsae-rel` do not use transport. The staged repair profile also uses angular
cost under its new `mpsae-repair-angular-v2` protocol; its historical O/N labels
now select identical loss components.

**Accuracy at every inference K.** `MMRSAE_EVAL_TOPK=auto` evaluates
2,4,8,16,32,64,128,256,512 for ResNet-18, and adds 1024,2048 for ResNet-50,
capped by dictionary width. The training target K is always included. A custom
list such as `MMRSAE_EVAL_TOPK=2,4,8,16,32,64,128,256,512` is accepted, with
every value required to be between 1 and h. Each point re-encodes both gallery
and queries at that budget, including landmark synthesis for MMRSAE variants.

These curves evaluate one checkpoint per method/backbone; they do not train
independent models for each K. `top1_by_k.png/.pdf` plots inference K against
top-1 with a distinct curve for every method and a panel per backbone. Tables
include both training K and inference K. Unmeasured budgets are never filled
with copied accuracy. Completed points survive an interrupted evaluation.
Intermediate epoch evaluations still measure only the training target K.

`csrv2`, `mmrsae-annealed`, and `dlmmsae-annealed` use the historical half-cosine training-K schedule
over the first 70% of updates by default:
`MMRSAE_ANNEAL_START_K=64` to the target `MMRSAE_K=8`, followed by a fixed phase.
An unset start uses min(h,max(64,4K)). With A=max(1,ceil(fraction*total_updates)),
zero-based update t uses the rounded value
`target_K + (start_K-target_K)*(1+cos(pi*t/(A-1)))/2` until t=A-1, then target K.
When A=1 the target budget applies immediately. This is actual training-K
annealing, matching the CSRv2/MPSAEv2 strategy; the old linear auxiliary-B
schedule is no longer used by `mmrsae-annealed`.

For example, target K=8 and start K=64 give main K(t)=64->8, intermediate
2K(t)=128->16 and clean 4K(t)=256->32, capped by h. The default auxiliary
reconstruction budget is also B(t)=min(h,4K(t)). Landmark keys at K(t) and
2K(t) refresh whenever K changes, in addition to periodic refreshes. Diagnostics
and resumed training use keys at the same budget as their loss. Console and
scalar logs distinguish current training K from target K. Endpoint inference
continues to use the requested evaluation budgets.

For current MMRSAE variants, every view is projected from SAE activations after
landmark synthesis. Main K and intermediate 2K share the masked dense activations
during fitting; the third transport view and auxiliary reconstruction use clean scores.
The wide budget replaces the existing reconstruction branch with weight 1/8;
it does not add a second wide reconstruction term. There is no separate 2K MSE.
The dead-feature reconstruction term retains weight 1/32; its `k_aux` is a
separate budget from B. MMRSAE variants produce clean landmark-first codes
for both gallery and query at each requested inference budget; baselines
produce plain SAE codes. Masking,
KL profiles and transport are training work. The fixed input landmark
dictionary is also used during inference and is carried in the checkpoint.
There is no transport solver in retrieval. Manual weight and masking overrides
are recorded separately from the preset name.

`cost0_min/max` and the legacy-named `costmix_min/max` now both report the
angular cost range. Solver feasibility/refinement and negative-gap checks stay
enabled. Neighborhood affinity kernels are retained for historical mathematical
reference checks; active training objectives cannot enable them with an override.

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
| `MMRSAE_K` / `--k` | 8 | Training target K; included in the inference evaluation grid |
| `MMRSAE_EVAL_TOPK` / `--eval-topk` | auto | Inference budget grid; each point uses the same trained checkpoint |
| `MMRSAE_ANNEAL_START_K` / `--anneal-start-k` | min(h,max(64,4K)) | CSRv2, MMRSAEv2 and DLMMSAE-annealed initial training budget; templates explicitly select 64 |
| `MMRSAE_ANNEAL_FRACTION` / `--anneal-fraction` | 0.7 | Fraction of CSRv2, MMRSAEv2 and DLMMSAE-annealed fitting used for cosine decay |
| `MMRSAE_HIDDEN_DIM` / `--hidden-dim` | 8192 | Dictionary width |
| `MMRSAE_REPAIR_RANK` / `--dictionary-rank` (alias `--repair-rank`) | 32 | Landmark-query adapter rank; the environment name is retained for compatibility |
| `MMRSAE_DICTIONARY_TEMPERATURE` / `--dictionary-temperature` | 0.1 | Softmax temperature for input landmark lookup |
| `MMRSAE_DICTIONARY_BLEND` / `--dictionary-blend` | 0.1 | Initial trainable mixture of landmark context into x; strictly between 0 and 1 |
| `MMRSAE_CONTEXT_WEIGHT` / `--context-weight` | 1 for DLMMSAE, 0 otherwise | Direct raw-context relational KL weight |
| `MMRSAE_CONTEXT_REFERENCE_COUNT` / `--context-reference-count` | 1024 | Fixed fitting reference rows Q, disjoint from A; at least 3 |
| `MMRSAE_CONTEXT_TEACHER_TEMPERATURE` / `--context-teacher-temperature` | 0.1 | Clean-input cosine profile temperature |
| `MMRSAE_CONTEXT_STUDENT_TEMPERATURE` / `--context-student-temperature` | 0.1 | Raw-context cosine profile temperature |
| `MMRSAE_CONTEXT_NOISE_STD` / `--context-noise-std` | 0.02 | Auxiliary lookup Gaussian noise relative to row RMS; 0 disables |
| `MMRSAE_LINEAR_PROBE` / `--linear-probe`, `--no-linear-probe` | 1 / enabled | Separate supervised classification evaluation at every K and full-parent features |
| `MMRSAE_PROBE_EPOCHS` / `--probe-epochs` | 10 | Epochs for each frozen-representation classifier |
| `MMRSAE_PROBE_BATCH_SIZE` / `--probe-batch-size` | 1024 | Classifier fitting batch |
| `MMRSAE_PROBE_LR` / `--probe-lr` | 0.1 | Classifier learning rate |
| `MMRSAE_PROBE_WEIGHT_DECAY` / `--probe-weight-decay` | 0.0001 | Classifier weight decay |
| `MMRSAE_MAX_STEPS` / `--max-steps` | 500 | Per-backbone update cap; zero removes the cap |
| `MMRSAE_EPOCHS` / `--epochs` | 10 | Epoch limit, also enforced when a step cap exists |
| `MMRSAE_BATCH_SIZE` / `--batch-size` | 1024 | Compressor fitting batch |
| `MMRSAE_TRAIN_SIZE` / `--train-size` | 128000 | Fitting/gallery rows; zero selects every eligible row after reserving anchors and dev queries |
| `MMRSAE_DEV_SIZE` / `--dev-size` | 10000 | Training-derived development queries |
| `MMRSAE_ANCHOR_COUNT` / `--anchor-count` | 1024 | Fixed input dictionary size and ordered training KL landmark bank |
| `MMRSAE_BANK_REFRESH` / `--bank-refresh` | 25 | Periodic key refresh interval; training-K changes also force a refresh |
| `MMRSAE_SEED` / `--seed` | 1227 | Model initialization, masking and fitting |
| `MMRSAE_DATA_SEED` / `--data-seed` | 42 | Stable allocation/extraction across model-seed changes |
| `MMRSAE_EVAL_EVERY` / `--eval-every` | 0 | Endpoint only; a positive completed-epoch interval adds gallery evaluations |
| `MMRSAE_CHECKPOINT_EVERY` / `--checkpoint-every` | 25 | Updates between resumable checkpoints |
| `MMRSAE_EMBEDDING_SOURCE` / `--embedding-source` | `precomputed` | Local/Hub NPZ+JSON pairs; `legacy` enables the older preparation workflow |
| `MMRSAE_EMBEDDINGS_REPO` / `--embeddings-repo` | `quangnguyenopt/NSFW` | Prepared embedding repository |
| `MMRSAE_EMBEDDINGS_REPO_TYPE` / `--embeddings-repo-type` | `model` | Hub repository type, independent of ImageNet's dataset repository |
| `MMRSAE_EMBEDDINGS_REVISION` / `--embeddings-revision` | `main` | Resolved once to an immutable commit in the cache |

One method has a ceiling of 1,000 fitting updates across two backbones; the
nine-method templates therefore allow 18 representation fits and 9,000 fitting updates. This count
excludes linear classifiers, data preparation, encoding, search and initialization. Smaller steps
or one backbone reduce fitting cost; `--no-timed-retrieval` avoids repeated
timed searches while retaining endpoint accuracy. Full-parent dense reference
search is an explicit `--dense-reference` option because its gallery can be
large. It is not required for every manual trial.

**Data preparation and split boundaries.** Both backbones use prepared
archives from [quangnguyenopt/NSFW](https://huggingface.co/quangnguyenopt/NSFW/tree/main).
This is a private Hub **model** repository. Set `HF_TOKEN` to a token with
read access to it; accepting ImageNet terms alone does not grant that access.
Existing local pairs take precedence and require no network or token:

```text
<MMRSAE_EMBEDDINGS>/
  1K_train_ff2048.npz   1K_train_ff2048.json
  1K_train_rn18.npz     1K_train_rn18.json
  1K_val_ff2048.npz     1K_val_ff2048.json
  1K_val_rn18.npz       1K_val_rn18.json
```

The loader downloads only missing files for the requested backbone/split,
directly into this folder. It never downloads `weights.zip`, raw images,
parent checkpoints or an entire repository snapshot. `dev` uses train only;
`official` also opens val. Downloads use the Hub client's resumable cache.
Revisions are pinned once, and the archive LFS hash must agree with its JSON
before an archive download begins. Original receipts retain the FF2048
publisher/parent identity or RN18's parent, transform and image provenance;
the mirror is not relabelled as a new pretrained encoder.

The RN18 files use uncompressed NPZ members, which are mapped directly in
place, including ZIP64 and Fortran/byte-order handling. FF2048 files are
compressed: their feature member is streamed in 8 MiB chunks once into a
read-only NPY disk map. Repeated methods reuse it. Archive checksums are cached
against the absolute path, size and nanosecond `mtime`/`ctime` values;
a changed file or a different expected hash requires checking again or fails.
Feature validation uses batches bounded to about 8 MiB, and training converts
only the consumed batches to FP32. The feature matrix is never loaded whole
into RAM. The legacy FF2048 comparison now uses this bounded-memory reader too.

All four archives total about 8.1 GB. Compressed FF2048 train needs another
10.5 GB of cache disk when first opened (validation adds about 0.41 GB).
RN18 needs no second feature copy. Data/cache paths remain persistent under
Docker's `/data`; keep the cache when resuming or comparing methods.

For an offline server, copy the needed NPZ **and** JSON files to
`MMRSAE_EMBEDDINGS` and set `MMRSAE_DOWNLOAD=0`. To reuse this checkout's
existing `imagenet_embedding_weight` directory with Compose, set in `.env`:

```dotenv
DATA_HOST_DIR=/absolute/server/path/to/imagenet_embedding_weight
MMRSAE_EMBEDDINGS=/data
MMRSAE_DOWNLOAD=0
```

The directory must permit cache writes. For direct local execution, set
`MMRSAE_EMBEDDINGS` to that local directory instead. Explicit
`--ff2048-train/val` and `--rn18-train/val` paths still take precedence.

`MMRSAE_EMBEDDING_SOURCE=legacy` retains the original FF2048 publisher download
and automatic RN18 export from raw images. Missing default RN18 pairs are
exported **after** GPU retrieval preflight, using the resolved CLI/environment
settings. Existing archives still skip extraction. The standalone
`export_rn18_embeddings.py` remains available, including `--data-backend
imagefolder`; raw ImageNet permissions are needed only for those image paths.

Default `--evaluation dev` uses disjoint training-derived queries and does
not download/open official validation. Labels allocate/evaluate the pools;
reconstruction, landmark synthesis and neighborhood targets do not use class
labels. The separate linear-probe evaluation does use fitting labels. Landmarks
come from training and are reserved outside fitting/gallery even with unlimited
training size. The sparse KL implementation still removes the minibatch's union
of row IDs from its profiles if used outside this experiment recipe. DLMMSAE's
Q comes only from fitting rows, with each query's own Q entry removed from both
teacher and student profiles. Input landmark synthesis retains all A anchors
in both training and inference.

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
reported separately from landmark synthesis/SAE encoding and index construction. Timed search
uses three warmups and five repetitions; reported time per query is amortized
over the configured batch, not isolated single-query latency. A padded K=8
indexed FP32 code uses 68 bytes before file metadata.

Representation `training_seconds` measures synchronized wall time around batch
transfer, forward/loss, backward, optimizer updates and training KL-bank refresh.
`training_gpu_hours = training_seconds / 3600` because each fit uses one GPU.
Diagnostics, data preparation, model/reference initialization, logging,
checkpoint I/O, retrieval and linear probes are excluded. The checkpoint retains
this counter across resume. The broader `fit_seconds` includes diagnostics and
other fitting-session work and is never substituted for training GPU-hours.
Hardware, timing scope and evaluation workload must match for speed comparisons.

**Separate classification evaluation.** Linear probes are enabled by default
after endpoint retrieval. Each inference K receives its own supervised linear
classifier trained on frozen sparse codes from fitting rows. Full-parent
features receive a separate classifier under the same declared settings. The
SAE and backbone remain frozen; validation/development labels only score the
finished classifiers, with no accuracy-based checkpoint selection. This is a
locally specified linear-probe protocol, not the backbone's original classifier
or a reproduction of classification numbers in the example figures.
Each classifier uses zero initialization, row-L2-normalized inputs, SGD with
momentum 0.9 and a cosine learning-rate schedule for the declared epoch count.
Sparse codes are cached as coordinates/values and expanded only for the current
minibatch. Classifier checkpoints are saved each epoch. The full-parent probe
streams features without a dense FAISS index and runs even when the optional
full-parent retrieval measurement is disabled.
`MMRSAE_LINEAR_PROBE=0` or `--no-linear-probe` skips it. Classification metrics
live under `summary.json:classification` and remain distinct from retrieval
`top1`, which means 1-NN label accuracy. Classifier fitting and encoding costs
are separate from representation training GPU-hours.

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
landmark refresh state, training-time counter and update position. DLMMSAE also
restores Q and its dedicated perturbation RNG; changed Q IDs/features are rejected.
Its diagnostics use an independent RNG and do not advance training perturbations.
Reuse the same command/output
directory to resume. A different recipe gets a different derived name unless
one was specified; incompatible state under an explicit name is rejected.
Only one writer should own a run directory.

Each evaluated budget writes `retrieval/endpoint/k_<K>/result.json`, metrics
and predictions. `retrieval/endpoint/result.json` and the backbone summary
contain `per_topk`, `evaluation_topk` and `budget_evaluation=single_checkpoint_multi_k`.
Top-level retrieval fields remain the training-target-K point for older readers.
This revision uses `mmrsae-imagenet-context-distillation-v4`; changed architecture,
code and recipe identities prevent resuming a sparse-start or auxiliary-only
annealing checkpoint as this method. Existing runs must keep their original
implementation and identities; new experiments receive new derived run names.

Source-level savings include bounded preparation/extraction, reusable parent
features, indexed sparse reductions, batched independent OT solves, cached
clean landmark keys and endpoint-only evaluation by default. These are design
choices visible in the code; no speedup or accuracy result is claimed before
execution. [VALIDATION.md](VALIDATION.md) records that limitation.
