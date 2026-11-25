# Experiments outside the current CVPR draft

These supported studies are intentionally outside `../cvpr2027/catalog.yaml`.
Existing paths remain valid so saved run identities and old notebooks work.

| Extension | Configuration | Runner / documentation |
|---|---|---|
| Matched ResNet-50 optimizer comparison | `configs/matched_resnet50.yaml` | [Setup, counts and profiling protocol](matched_resnet50/README.md) |
| Cityscapes patch-SAE nearest-neighbor segmentation | `configs/patch_scene_cityscapes.yaml` | `mpsae.patch_scene`, `docs/segmentation_server.md` |
| VOC linear-probe segmentation | `configs/linear_scene_voc.yaml` | `mpsae.patch_scene.linear_probe`, `docs/ablation79.md` |
| 79-evaluation combined suite | `configs/retrieval_studies.yaml` + `configs/linear_scene_voc.yaml` | `scripts/run_ablation79.sh` |

Disabled backbone fine-tuning remains documented in `configs/archive/` and Git
history. SOP, retired methods and old dataset recipes are not activated by the
paper frontend. No datasets, checkpoints or results are deleted by this cleanup.
