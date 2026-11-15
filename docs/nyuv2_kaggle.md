# NYUv2-40

Use the [canonical one-cell Kaggle guide](patch_scene_kaggle.md) with `DATASET="nyuv2"`.
Settings live only in `configs/patch_scene_nyuv2.yaml`: 20 epochs maximum, training-loss early stopping, matched K.
The labeled MAT now downloads automatically from Hugging Face; no Add Input step is needed.
