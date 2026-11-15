# Repository map

The maintained code is `graduate_thesis_refactor` (Git branch `refactor`). The
neighboring `graduate_thesis` directory contains original scripts and local
development staging copies such as `.nyu_work`; it is not the Kaggle source.

| Area | Responsibility / status |
|---|---|
| `mpsae/model.py`, `methods.py`, `losses.py`, `ot.py` | Shared SAE/objectives; unchanged |
| `mpsae/train.py`, `embed/`, `evaluate/`, `benchmark/`, `report/` | Existing image-level/retrieval pipeline; unchanged |
| `mpsae/patch_scene/` | Supported VOC and NYUv2 patch-SAE campaigns |
| `mpsae/segmentation/` | Shared metrics, patch labels, image preparation and batching |
| `configs/patch_scene_voc.yaml` | Canonical preserved VOC patch benchmark |
| `configs/patch_scene_nyuv2.yaml` | Canonical NYUv2 matched-K benchmark |
| `configs/patch_scene_kaggle.yaml` | Compatibility alias to NYUv2; no copied hyperparameters |
| `scripts/kaggle_segmentation.py` | Single Kaggle setup/download/train/export implementation |
| `notebooks/patch_scene_kaggle.ipynb` | One-cell frontend; selects VOC or NYUv2 |
| `configs/archive/`, `docs/archive/` | Historical recipes; not supported launch targets |
| `mpsae/scene/` | Historical adaptation/report compatibility and server checks; new runs retired |
| `mpsae/finetune/` | Previously disabled backbone fine-tuning; untouched |
| `tests/` | Active and historical compatibility regression checks |

Patch pipeline: dataset -> frozen encoder -> cached patch features -> SAE fit ->
neighbor segmentation -> metrics and tables. All methods share the SAE/loss code.
VOC retains its old sampling, 10 epochs and cross-K evaluation; NYUv2 retains full
patch streaming, 20-epoch early stopping and matched-K evaluation.

Legacy dataset registries and low-level readers remain to interpret existing
receipts and test compatibility. Public plan/run/smoke commands reject retired
datasets and encoders before downloading or training. Historical export remains
available with the archived configuration; no numerical IDs or caches are rewritten.
Old `mpsae.scene` plot/publish/doctor remain, but its training/bootstrap commands
are retired. Shared `scene.core`/`scene.datasets` imports forward to the common
package so historical code can still read results.

Do not delete directories based only on age: saved workspaces contain feature
caches, checkpoints and run provenance. This cleanup does not delete results,
datasets, old model checkpoints or the neighboring development staging folders.
