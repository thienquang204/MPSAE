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
images, number of epochs, batch size and optimizer settings. Their original
method regularizers remain different. A random batch of 16 may contain few
same-class positives for CSRv2, so the effect of batch size must be acknowledged
when interpreting the comparison. No performance improvement is guaranteed.

## Default optimization

| Setting | Value |
| --- | --- |
| Additional epochs | 3 for every method |
| Physical image batch | 16, no gradient accumulation |
| Optimizer | AdamW, constant learning rate |
| Backbone / SAE / classifier LR | `1e-5` / `1e-5` / `1e-3` |
| Weight decay / gradient clipping | `1e-4` / norm 1 |
| CE / method weights | 1 / 1 |
| Precision | BF16 backbone and teacher; float32 SAE, classifier and losses |
| BatchNorm | Running statistics frozen; affine parameters train |
| Checkpoint / log interval | 100 / 10 steps |
| Training augmentation | Random resized crop 224, scale 0.8–1.0, horizontal flip |
| Evaluation preprocessing | Deterministic resize 256, center crop 224 |

Normalization uses the pinned backbone's preprocessing. Full training uses
`floor(1,281,167 / 16) = 80,072` updates per epoch, **240,216 updates per fit**.
The last incomplete batch is dropped. These are conservative pilot defaults,
not learning rates selected on validation results. The source method presets
in manifests describe the inherited regularizers; their old batch/epoch/LR
fields do not override the separate `finetune` settings above.

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
and optional W&B reporting. It uses the separate `finetune-v1/finetune/` remote
prefix; local raw images and feature caches are not uploaded. Without this
flag, results and checkpoints are local only.

## Storage, resume and resource usage

Outputs live in `work/finetune-v1/finetune/`. Each checkpoint stores the
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

Training executes one fit at a time. BF16 and batch 16 limit activation memory;
the frozen teacher has no optimizer or gradient graph. Full-image training
still costs substantially more than SAE training on cached embeddings. Training
logs/summary record elapsed time, peak CUDA allocated and reserved bytes.
GPU hours mean elapsed CUDA training wall time, not a utilization measurement.
Batch 16 is a starting point, not a measured VRAM guarantee on RTX 5090.
If it does not fit, lower the common batch for all methods and use a new campaign
name; this changes contrastive/OT batch statistics and must be reported.

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
