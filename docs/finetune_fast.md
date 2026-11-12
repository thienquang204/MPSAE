> **Archived experiment, disabled by default.** All nine additional fits and
> their evaluations were omitted due to memory/compute limits. The commands
> below describe the historical workflow, not the active experiment plan.
> The original recipe is in `configs/archive/finetune_gradient_replay.yaml`.
> Saved artifacts remain available; pass that config explicitly when using
> `report`, `transfer-report` or `sync` to inspect or back up existing output.

# Nine-run fine-tuning on RTX 5090

This launches only ResNet-50, ConvNeXt-Tiny and ViT-B/16, each with CSR, CSRv2
and MPSAE. It does not rerun the 39 frozen-backbone fits or download their large
embedding arrays. The nine completed source SAE fits are still required.

## Start on the server

From the project directory, on a new server only:

```bash
bash setup_server.sh
source activate_server.sh
# Configure HF_TOKEN in .env and accept the ImageNet dataset access conditions.
```

The installer retains pip's package cache and pins the existing CUDA/PyTorch
versions. A normal fine-tuning launch does not reinstall dependencies. Do not
run `all_in_one.sh` for this campaign.

If `work/direct-v1` already contains the nine source fits, skip restoration.
Otherwise restore only checkpoints, configs, summaries and provenance sidecars:

```bash
python -m mpsae.finetune restore-sources --repo quangnguyenopt/MPSAE --workspace work
```

Restoration resolves one immutable HF commit, refuses conflicting existing
files and validates all nine source fits. It downloads no `.npz` embeddings.

Then run:

```bash
bash finetune_only.sh --workspace /workspace/MPSAE/work
```

For another image directory or CPU allocation:

```bash
MPSAE_WORKERS=16 bash finetune_only.sh --workspace /workspace/MPSAE/work \
  --image-root /data/imagenet
```

The launcher defaults to 16 loader workers, extraction batch 64, packed images,
resumable ImageNet download and HF backup. Use fewer workers if the server has
fewer available CPU cores. Worker count is independent of the training batch.
`--extract-batch` changes feature extraction only. The lower-level CLI supports
`--image-storage folder` if retaining the existing ImageFolder path is preferred.

## What remains the same

| Quantity | Setting |
| --- | --- |
| Additional fits | 9, one seed per method/backbone |
| Epochs | 10 for every method |
| Full loss/optimizer batch | CSR/CSRv2 4096; MPSAE 1024 |
| Learning rates | Backbone, SAE, classifier: 4e-5 |
| Optimizer | Adam; original betas, epsilon, weight decay and clipping |
| Training K | ResNet-50: 8; ConvNeXt-Tiny and ViT-B/16: 12 |
| Inference K | ResNet-50: 2, 8, 16, 32, 256; others: 3, 12, 48, 96, 192, 384 |
| Downstream | EuroSAT and SOP weighted kNN plus retrieval |
| Timing benchmark | Disabled here; warm-up/rounds still 5/5 |

All existing linear-probe selections and frozen controls remain enabled.

## How memory is reduced without shrinking the loss batch

`finetune.backbone_microbatch: 32` limits the number of image graphs in VRAM.
For each full logical batch the trainer:

1. Loads deterministic augmentations into CPU chunks. Runs teacher and student
   chunks without retaining backbone graphs; saves student RNG states.
2. Concatenates student and teacher features. Computes the **entire original
   method loss and classification loss once** across all 4096/1024 examples.
3. Backpropagates through SAE/classifier and caches the feature gradients.
4. Replays each student image chunk with its saved RNG, applies its feature
   gradient, clips the combined gradients once and performs one Adam update.

CSR contrastive negatives, CSRv2 label relationships, MPSAE OT groups and SAE
activity updates therefore retain their full-batch semantics. This is not
independent microbatch loss accumulation. Frozen BatchNorm statistics are
required; affine parameters still train. Dropout/drop-path RNG is replayed.
Microbatch execution can differ numerically from one giant GPU batch; it is
not a promise of bitwise equivalence across different CUDA kernels.

Replay adds an extra student forward pass. It enables the requested large loss
batches on a smaller GPU; it is not guaranteed faster than a full-graph batch
on hardware that can already fit that batch. CUDA peak memory and training time
remain in `train.jsonl` and `summary.json`. CPU gradient/resume tests pass, but
the real 5090 peak VRAM and throughput must be measured on the server.

The active campaign is now `finetune_gradient_replay`, so prior optimization
states under `finetune_source_settings` are not silently resumed. Within the
new campaign, rerunning the same command resumes completed optimizer updates.
To change microbatch size after starting, use a new campaign name: it affects
execution and random-mask grouping and is recorded in checkpoint identity.

## ImageNet reuse and the 1 TB RAM server

The new default writes original compressed JPEG bytes into indexed packs at
`<image-root>/jpeg-packs-v1/{train,val}`. On a fresh server it reads pinned HF
parquet directly into packs, avoiding 1.3 million individual output files.
One future shard downloads while the current shard is packed. Completed shards
are durable and resumable; images are not resized or JPEG-reencoded.

If complete `train/val` ImageFolder directories already exist, they are packed
locally once without redownloading. This conversion needs a second compressed
image copy on disk; originals are never deleted. Incomplete raw folders must be
completed through the old preparation pipeline before converting them.

Later runs use the completed pack manifest and compact index offline. They
validate index hashes and pack sizes without rescanning all original files.
Packs are immutable cache artifacts. Optional `PackedImages(path, verify=True)`
also hashes all JPEG data for a full integrity audit; ordinary startup avoids
reading the entire image corpus solely to check hashes.

Persistent loader workers decode images and apply the same seeded augmentations
each epoch. Prefetch queues contain microbatches of 32 images, rather than
4096-image batches. The OS can use spare host RAM to cache compressed packs,
shared by all workers and all fits. It does not eagerly allocate 1 TB, and
1 TB host RAM does not enlarge the GPU's 32 GB VRAM. JPEG decoding and random
augmentations still cost CPU time; cached pooled embeddings cannot train the
backbone or replace the teacher on the current augmented image.

Allow about 71 GiB of disk for the ImageNet feature arrays alone, in addition
to compressed images, source/model checkpoints, downstream caches and results.
Use local SSD/NVMe storage. More host RAM alone does not guarantee faster training.

Preparation can also run separately, once:

```bash
python -m mpsae.finetune prepare --image-root /data/imagenet --download-images
```

## Checkpoints and HF backup

The launcher enables `--backup`. Local checkpoints remain every 100 updates.
Intermediate checkpoints upload at the next checkpoint boundary after 1800
seconds (`--backup-interval`), completed fits upload immediately with their
summaries, evaluation results upload after each completed cell, and final
plots upload at the end. Uploads are synchronous to avoid copying a checkpoint
while it is being replaced. Failures retain local artifacts and final backup
failure is reported rather than silently treated as success.

No raw ImageNet images or feature caches are uploaded. Artifacts go to the
private HF dataset under `finetune-v1/finetune_gradient_replay/`. If a process is
hard-killed, only artifacts uploaded before that point are protected. Retry
backup without training or CUDA:

```bash
python -m mpsae.finetune sync --workspace /workspace/MPSAE/work
```

`report`, `transfer`, and `transfer-report` also accept `--backup` to sync their
finished output. Standalone transfer uploads after the command finishes;
the full nine-run launcher additionally backs up between completed cells.
