# Additional backbone fine-tuning pilot

`configs/finetune.yaml` selects **nine additional fits**, separately from the
39 frozen-backbone fits in `configs/thesis.yaml`. ResNet-50 replaces ResNet-18
in this pilot only. Each fit initializes the original pretrained backbone and
its matching completed SAE checkpoint, then jointly fine-tunes them on raw
ImageNet training images. Existing source checkpoints are read only.

| Backbone | Train K | Methods | Seed | New fits |
| --- | ---: | --- | ---: | ---: |
| ResNet-50 (`ff2048`, original BlurPool architecture) | 8 | CSR, CSRv2, MPSAE | 1227 | 3 |
| ConvNeXt-Tiny | 12 | CSR, CSRv2, MPSAE | 1227 | 3 |
| ViT-B/16 | 12 | CSR, CSRv2, MPSAE | 1227 | 3 |

## Objective and interpretation

Let `f0` be the frozen original backbone, `fθ` the trainable copy, `Eφ,Dφ`
the SAE, and `hψ` a new linear classifier. For the **same augmented image** `a(x)`:

```
t = stop_gradient(f0(a(x)))
u = fθ(a(x))
zq = TopK-ReLU(Eφ(u), q)
Ljoint = ce_weight * CE(hψ(normalize(zK)), y)
       + method_weight * Lmethod(u, reconstruction_target=t)
```

`Lmethod` preserves the existing method coefficients and loss implementations:
main K reconstruction, weighted 4K reconstruction, AuxK, plus instance
contrastive loss for CSR, label-based contrastive loss for CSRv2, or the MMPOT
matching gap between K/2K/4K views for MPSAE. Reconstruction targets come from
the frozen teacher, including the auxiliary residual target. The teacher
prevents the reconstruction target from drifting with the student.

All three methods use **fixed train K** during this continuation. CSRv2's
original annealing has already finished and is not restarted. The MMPOT term
is retained. Its current target-K-dependent coefficient remains unchanged;
none of these three selected training budgets exceeds 64.

This is a **supervised, teacher-anchored joint fine-tuning experiment**.
It changes the training protocol and is not a reproduction of the original
frozen-backbone paper experiment. All methods receive the same added CE loss,
images and number of epochs. Each method inherits its own standalone SAE
batch size and Adam settings; its original regularizer remains different.
No performance improvement is guaranteed.

## Default optimization

| Setting | Value |
| --- | --- |
| Additional epochs | 10 for every method, inherited from its standalone preset |
| Physical image batch | CSR/CSRv2: 4096; MPSAE: 1024; no gradient accumulation |
| Optimizer | Adam, constant learning rate; betas (0.9, 0.999), epsilon `6.25e-10` |
| Backbone / SAE / classifier LR | `4e-5` / `4e-5` / `4e-5`, inherited from the standalone preset |
| Weight decay / gradient clipping | `1e-4` / norm 1 |
| CE / method weights | 1 / 1 |
| Precision | BF16 backbone and teacher; float32 SAE, classifier and losses |
| BatchNorm | Running statistics frozen; affine parameters train |
| Checkpoint / log interval | 100 / 10 steps |
| Training augmentation | Random resized crop 224, scale 0.8–1.0, horizontal flip |
| Evaluation preprocessing | Deterministic resize 256, center crop 224 |

Normalization uses the pinned backbone's preprocessing. The final incomplete
batch is retained, matching standalone training. For 1,281,167 images, CSR/CSRv2
use 313 updates per epoch (3,130 per fit), and MPSAE uses 1,252 (12,520 per fit).
The YAML value `source` resolves through `training_preset(method, backbone, K)`;
the plan's `config.resolved_training` and each fit's checkpoint identity record
the actual numbers. There is no separate fixed batch-16 default.

This replaces the earlier three-epoch, batch-16 AdamW pilot, whose learning
rates were `1e-5` for backbone/SAE and `1e-3` for the head. The new campaign name
`finetune_source_settings` keeps those earlier outputs separate and cannot
resume their optimizer state. Fine-tuning still initializes from the completed
standalone SAE, not from a random SAE. These are 10 additional joint epochs.

## Evaluation and controls

| Backbone | Inference sweep K | Linear probe K |
| --- | --- | --- |
| ResNet-50 | 2, 8, 16, 32, 256 | 2, 16, 32, 256 |
| ConvNeXt-Tiny | 3, 12, 48, 96, 192, 384 | None |
| ViT-B/16 | 3, 12, 48, 96, 192, 384 | 3, 48, 96, 384 |

Each fit also gets E1 at its training K. In total: **51 unique sparse 1-NN
evaluations**, **24 separate linear-probe fits**, and **51 frozen-control 1-NN
evaluations**. The nine joint-training classifiers are discarded for evaluation;
1-NN operates on the final sparse codes, and selected probes are freshly fit.
Each code has at most the requested inference K nonzero entries. The wider
training views are not concatenated into the final code.

Frozen controls use the original SAE and original backbone on exactly the same
images and evaluation preprocessing as the fine-tuned model. Teacher features
are extracted once per backbone and shared across its three methods. This
avoids attributing a preprocessing change in older cached features to training.
`paired_1nn.csv` compares these matched controls against the fine-tuned models.
Frozen-control probes are not added by this pilot; the paired comparison is
1-NN only. Downstream, SOP and repeated retrieval timing remain disabled.
Reports contain experiment results without published reference curves.

## Run on the server

Run from `graduate_thesis_refactor` after its environment has been installed:

```bash
source activate_server.sh
python -m mpsae.finetune plan --config configs/finetune.yaml
python -m mpsae.finetune run --config configs/finetune.yaml --download-images
```

The nine completed source fits must already exist under
`work/direct-v1/runs/<cell>/`, with `checkpoint.pt`, `config.json`, `summary.json`
and pinned backbone provenance in the fit config or embedding archive sidecars.
The runner validates all nine sources and pretrained network graphs first.
It rejects missing, partial, capped, incompatible or synthetic source fits.
It downloads/reuses the exact pretrained weights identified by the source
provenance rather than guessing a model from its name.

`--download-images` reuses/resumes the existing ImageNet preparation pipeline.
Raw images are required: extracted embeddings cannot supply gradients through
the backbone. If images already exist, omit this flag. The default layout is
`work/imagenet/train/<class>/...` and `work/imagenet/val/<class>/...`.
The existing Hugging Face credentials and ImageNet access requirements apply.
Paths can be specified independently:

```bash
python -m mpsae.finetune run --source-root /data/work/direct-v1 \
  --image-root /data/imagenet --workspace /data/work \
  --workers 8 --extract-batch 64
```

This has its own entry point. Do not pass `configs/finetune.yaml` to
`all_in_one.sh`, which runs the frozen campaign schema. Add `--backup` to the
fine-tuning run to opt into the existing private Hugging Face backup machinery
and optional W&B reporting. It uses the separate
`finetune-v1/finetune_source_settings/` remote prefix; local raw images and
feature caches are not uploaded. Without this
flag, results and checkpoints are local only.

## Storage, resume and resource usage

Outputs live in `work/finetune-v1/finetune_source_settings/`. Each checkpoint stores the
trainable backbone, SAE, classifier, optimizer, RNG and next step. Repeating
the command resumes from the last saved completed update. Completed fits and
evaluations are reused. Changing training settings, source weights or data
requires a new campaign `name`; incompatible state is rejected rather than
silently continued. A changed inference grid can reuse the same fit.

Changed backbone weights require new feature extraction. Each fine-tuned
checkpoint gets a separate feature identity, with resumable float32 memory
mapped arrays. Images and decoded batches are not copied into feature caches.
The three frozen-backbone feature sets plus nine fine-tuned feature sets need
approximately **71 GiB** for feature arrays at full ImageNet size, before raw
images, weights, labels, checkpoints and temporary evaluation storage. Existing
frozen-campaign caches remain intact. Ensure adequate disk space before running.

Training executes one fit at a time. BF16 reduces activation memory;
the frozen teacher has no optimizer or gradient graph. Full-image training
still costs substantially more than SAE training on cached embeddings. Training
logs/summary record elapsed time, peak CUDA allocated and reserved bytes.
GPU hours mean elapsed CUDA training wall time, not a utilization measurement.
Physical batches of 1024/4096 raw images have much higher activation costs than
the same batches of cached embeddings. They are likely to exceed a single
RTX 5090's VRAM, especially for ViT. The runner does not silently shrink batches
or substitute gradient accumulation: that would change contrastive/OT groups.
Executing this exact configuration may require additional memory or a separately
implemented gradient-caching/distributed strategy. No large-batch GPU run has
been validated locally. Explicit numeric batch overrides remain available, but
constitute a different setting and require a new campaign name.

Reports are under `reports/<plan_id>/` with `paired_1nn.csv`, joint/frozen
figures and a manifest. Regenerate them with:

```bash
python -m mpsae.finetune report --config configs/finetune.yaml
```

For a synthetic integration check without downloads, GPU or remote writes:

```bash
python -m mpsae.finetune run --smoke --device cpu --workers 0
```

Smoke results use `smoke-finetune-v1/` and are explicitly marked non-scientific.
CPU tests exercise gradients, sparse evaluation, source isolation, exact
interrupted resume and completed-run reuse. Real CUDA training, peak VRAM,
throughput and accuracy must be measured on the server.
