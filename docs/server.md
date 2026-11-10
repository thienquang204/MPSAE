# Native NVIDIA server: direct extraction and bounded dataset lifecycle

Run Python directly; Docker is not needed. Root-level scripts are
`setup_server.sh`, `activate_server.sh`, and `all_in_one.sh`.

## Install and run

Ubuntu 22.04/24.04, Python 3.10?3.12 (3.12 recommended), x86_64 or ARM64/SBSA.
The provider must install the NVIDIA driver first. CUDA 13.0 is the default;
Linux driver >= 580.65.06 is required. H100, A100 and RTX A6000 use the same
PyTorch 2.9.1 / torchvision 0.24.1 cu130 environment.

```bash
cd /workspace/MPSAE
bash setup_server.sh --dry-run
bash setup_server.sh --cuda 13.0
# Edit .env: HF_TOKEN and optionally WANDB_TOKEN. Preserve an existing .env.
bash all_in_one.sh --skip-setup
```

Or install and run with one command after creating `.env`:

```bash
bash all_in_one.sh --cuda 13.0
```

HF_TOKEN needs gated ImageNet read access and private repository write access.
Accept the dataset conditions for `ILSVRC/imagenet-1k` first. The destination
is private `<account>/MPSAE`; W&B project defaults to `cvpr`. No project name
or HF account name is needed. Tokens are never sourced as shell commands.

```bash
bash all_in_one.sh --skip-setup --device cuda:0 --workers 16 --embed-batch-size 256
bash all_in_one.sh --sync-only
bash all_in_one.sh --smoke       # synthetic science rehearsal, no remote writes
source activate_server.sh
python -m mpsae --config configs/thesis.yaml plan
python -m mpsae --config configs/thesis.yaml status
python -m mpsae --config configs/thesis.yaml report
```

One campaign owns one GPU selected with `--device`; this is not distributed
training across H100/A100/A6000. Use separate workspaces for independent
campaigns. Do not run a historical extractor concurrently against images that
the direct campaign owns and will retire. Inference microbatches shrink on
CUDA OOM; SAE training batch sizes remain the scientific presets.

## Dataset lifecycle

The default `configs/thesis.yaml` selects all five registered backbones and
`execution.storage_mode: direct_v1`, `cleanup_raw: true`:

1. Check all selected pretrained models on the GPU before downloading. Reuse ImageNet JPEGs at `work/imagenet`; complete missing download shards.
2. Extract all selected backbones directly from JPEG batches. No full FFCV copy.
3. Verify archive hash, dimensions, row counts, sample IDs and source dense gates. Upload all
   embeddings, IDs and sidecars, then verify remote hashes at an immutable HF
   commit. Only then remove ImageNet train/val images and owned download cache.
4. Train/evaluate all selected ImageNet SAE runs from saved embeddings.
5. Process EuroSAT, Flowers102, then SOP, one at a time. Extract all selected
   backbones, verify/back up artifacts, reclaim that dataset's raw caches,
   evaluate retrieval from saved features, then proceed to the next dataset.
6. Generate aggregate tables and plots. Partial results are preserved on failure.

For a previously full server, the direct campaign can remove the exact old
`train_500_0.50_90.ffcv` / `val_500_0.50_90.ffcv` and their partials/sidecars
once the corresponding original JPEG split has passed its inventory check.
Those files are redundant derivatives, not inputs to the new protocol.
Other legacy results, checkpoints and cache folders are left alone.

If the disk is too full even to install CUDA 13, stop the old job and run the
stdlib-only migration helper **before setup**:

```bash
bash reclaim_ffcv.sh --workspace work
bash all_in_one.sh --cuda 13.0
```

The helper checks original JPEG inventories, removes only the known redundant
FFCV files, and prints reclaimed/free bytes. It does not delete JPEGs or need
Torch, a working `.venv`, HF access, or CUDA. If inventory is incomplete it
refuses cleanup; it does not guess which other large files can be removed.

If extraction or HF verification fails, raw images are retained. Resume with
the same command. Completed feature bundles are read without loading raw
images again. ImageNet retirement markers also survive interruption during
deletion. If embeddings were subsequently deleted or a new backbone is added
after raw retirement, execution stops with a restore instruction rather than
automatically downloading ImageNet again. HF restore is currently manual.

Local-only Python runs remain available: copy/edit the campaign config to set
`execution.cleanup_raw: false`, then use `python -m mpsae --config <file> run`.
The unmodified default requires the remote entry point because cleanup is
conditional on verified cloud backup. `pipeline.sh` remains a local CLI wrapper.

## Storage and provenance

New results live in `work/direct-v1/{embeddings,runs,benchmarks,reports,campaigns}`.
HF uses the matching `direct-v1/` prefix, protecting historical artifacts.
ImageNet JPEGs are reused from `work/imagenet`, not copied into that workspace.

Five FP32 ImageNet feature pairs need approximately 23.9 GB (22.3 GiB).
With 300 GB of JPEGs, the baseline is ~324 GB, PLUS the active extraction
writer (up to ~10.5 GB), download shard, weights, Python/CUDA environment and
free-space reserve. This is not a guarantee that every 500 GB disk has room:
existing legacy caches and unrelated files count too. Download/extraction
preflight reserves space for remaining embeddings before proceeding. It never
solves a shortage by reducing dataset size or silently switching precision.

NPZ float32 arrays are mapped directly from their uncompressed ZIP members;
there is no second extracted NPY copy. A resumable writer is temporary and is
reclaimed after its archive is published and verified. Sparse evaluation codes
are discarded when their metrics are durable (`retain_codes: false`). Checkpoints
and optimizer states persist for resume; the five-backbone main-result default uses seed 1227 and has 35
source fits and 68 downstream evaluation points (60 sparse + 8 dense).
The model + Adam tensor estimate for those 35 checkpoints is about 3.5 GB,
before serialization overhead and other buffers. The default campaign pulls published embeddings and does not download raw ImageNet.

The direct recipe is explicitly versioned `imagenet-direct-pil-v1`. MRL keeps
its deterministic centre-crop approach. timm train uses per-image seeded PIL
random-resized-crop/hflip. All 1,281,167 train rows and 50,000 validation rows
are kept; images are not resized/recompressed into FFCV first. This changes
the historical FFCV protocol (including its dropped tail), so these embeddings
and fits are isolated and must not be presented as byte-identical reproductions.
Downstream remains frozen evaluation: no fine-tuning on the three target sets.

## Installation checks and legacy runs

Setup installs toolkit only, never replaces the host driver. It checks the
NumPy/PyArrow/datasets binary stack, actual CUDA fp32/fp16 operations, direct
JPEG transforms and the compiled sparse-retrieval extension before success.
NumPy 1.26.4 and PyArrow 25.0.1 remain pinned to avoid the previously reported ABI
failure. Pip downloads use `--no-cache-dir`. Existing `.env` and results survive
setup. Logs are in `.setup-logs`; the verified profile and dependency freeze
are stored under `.venv`.

FFCV, CuPy and Numba are not required/imported by the default direct pipeline.
The historical FFCV implementation remains available in an explicitly selected
legacy campaign. Its optional dependencies are in `requirements-legacy-ffcv.txt`
and require a CUDA 12 environment. CUDA 12.6/12.8/12.9 installer overrides remain
available. Do not install the old CuPy CUDA 12 wheel to repair a CUDA 13 run.

The synthetic smoke command validates downstream orchestration, not real
ImageNet decoding or pretrained numerical parity. Actual CUDA 13 model
extraction and full-scale throughput must be validated on the server; local
CPU tests alone cannot establish H100/A100/A6000 performance.
