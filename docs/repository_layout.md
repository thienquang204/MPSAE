# Repository map

The maintained repository is `graduate_thesis_refactor` (branch `refactor`).
The neighboring `graduate_thesis` directory contains original scripts and local
development staging folders; it is not the source cloned by Kaggle.

| Area | Responsibility |
|---|---|
| `mpsae/model.py`, `methods.py`, `losses.py`, `ot.py` | Shared SAE models and objectives |
| `mpsae/patch_scene/` | VOC, NYUv2 and Cityscapes extraction, SAE training, evaluation, tables |
| `mpsae/segmentation/` | Patch labels, data transforms, metrics, batching and environment check |
| `configs/patch_scene_voc.yaml` | VOC: 18 fits, 56 evaluations, cross-K sweep |
| `configs/patch_scene_nyuv2.yaml` | NYUv2: 18 fits, 20 matched-K evaluations |
| `configs/patch_scene_cityscapes.yaml` | HF Cityscapes: 18 fits, 20 matched-K evaluations |
| `configs/patch_scene_kaggle.yaml` | Compatibility alias to NYUv2, not a separate campaign |
| `scripts/kaggle_segmentation.py` | Shared Kaggle install/download/run/export launcher |
| `scripts/kaggle_*_cell.py` | Small copy-and-paste frontends |
| `notebooks/patch_scene_kaggle.ipynb` | One-cell notebook frontend |
| `requirements-patch-scene.txt` | Complete segmentation dependency list |
| `mpsae/benchmark/`, `embed/`, `evaluate/`, `report/` | Existing image-level/retrieval pipeline |
| `mpsae/finetune/`, `configs/archive/finetune_gradient_replay.yaml` | Previously disabled backbone experiments |

Frozen encoder -> cached patches -> SAE fit -> weighted neighbor segmentation
-> CSV/LaTeX tables. Encoders: SigLIP2 and DINOv2; methods: CSR, CSRv2, MPSAE.
All three campaigns explicitly use a 65536-reference retrieval bank. VOC samples
SAE training patches; NYUv2 and Cityscapes stream all valid resized-image patches.
See [the Kaggle guide](patch_scene_kaggle.md) for numerical settings and launch cells.

The old `mpsae.scene` pipeline, archived segmentation recipes, old installation
scripts and their exclusive tests were deleted. Git history retains their source.
Small legacy registry/serialization paths remain to preserve cache identities;
the public runner permits only the three supported campaigns.
No saved results, datasets, feature caches or checkpoints were deleted.

## Paper experiment layer

`experiments/cvpr2027/` owns the frozen catalog, paper mapping, receipt import
and CPU reporting. It invokes existing `mpsae` engines and leaves historical
workspace layouts intact. `experiments/extensions/README.md` locates supported
studies outside this draft. Large artifacts remain under configurable `work/`
roots; local machine paths belong in ignored `paths.local.yaml`.
