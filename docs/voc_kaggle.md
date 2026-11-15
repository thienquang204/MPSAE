# VOC patch-SAE benchmark

Use the [canonical one-cell Kaggle guide](patch_scene_kaggle.md) with `DATASET="voc"`.
Settings live in `configs/patch_scene_voc.yaml`; the previous SigLIP2/DINOv2 VOC settings are preserved.
This uses fresh patch SAEs, not the retired ResNet/ViT/ConvNeXt checkpoint-adaptation workflow.
