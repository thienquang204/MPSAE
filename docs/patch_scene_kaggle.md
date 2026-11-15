# Active segmentation campaign: Cityscapes

The active Kaggle experiment now uses only Cityscapes fine train/validation.
See [the complete copyable Kaggle cells](cityscapes_kaggle.md) and
[the configuration](../configs/patch_scene_cityscapes.yaml).

- SigLIP 2 and DINOv2 frozen, CSR/CSRv2/MPSAE.
- Independent train K=3,12,48, inference K=3,12,48 for every checkpoint.
- All valid patches from every training image at 448x896, streamed from disk.
- 10 epochs, batch 1024, Adam LR 4e-5; no early stopping.
- 18 independent fits and 56 evaluation conditions.
- 2975 fine training images and 500 validation images. Ignore test and coarse.
- 65536 support patches sampled from the entire training patch pool using the
  existing class-frequency scoring rule. This support budget does NOT cap SAE training.
- Original ground-truth masks remain at native resolution for scoring. Predictions
  are resized back using the existing tiled soft-label decoder.
- Per-epoch full-patch order shuffles shards and patches within each shard; the
  final singleton joins its preceding batch. No patches are dropped or repeated.
- CSRv2 alone uses dominant training-mask classes in its loss; other SAE losses
  remain unchanged. 19-class mIoU and per-class IoU are reported, not instance iIoU.

Dataset artifacts and selected model revisions are fingerprinted. Resume checks
fit identity and restores optimizer/RNG state, including mid-epoch progress.

ADE20K/VOC configs and older protocol documentation are historical only. Their
outputs/checkpoints are not deleted. The active default `patch_scene_kaggle.yaml`
also selects only Cityscapes. Historical readers remain for inspecting old results.

The legacy `train_patches` field is 0 (disabled) when `all_train_patches: true`.
Training summaries record the actual full patch count. All memory/runtime estimates
need validation on the target GPU; the full campaign has not been run locally.
