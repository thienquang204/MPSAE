# Kaggle segmentation: VOC or NYUv2

Supported experiments use frozen **SigLIP2 Base and DINOv2 Base** with fresh
CSR/CSRv2/MPSAE patch SAEs. Training algorithms and retrieval calculations are
unchanged by the organizational cleanup. Choose the dataset in the cell below.

| Setting | VOC | NYUv2-40 |
|---|---|---|
| Canonical config | `configs/patch_scene_voc.yaml` | `configs/patch_scene_nyuv2.yaml` |
| Split | 1464 train / 1449 validation | 795 train / 654 official test |
| Input | 448 x 448 | 336 x 448 |
| SAE training patches | Up to 262144; existing per-image sampling | All valid train patches, streamed |
| Train K | 3, 12, 48 | 3, 12, 48 |
| Evaluation | Each fit at K=3,12,48 (preserved VOC sweep) | Each fit only at its training K |
| Epochs | 10, no early stopping | Up to 20; training-loss early stopping |
| Fits / evaluations | 18 / 56 | 18 / 20 |

Both use batch 1024, Adam LR 4e-5, a separate 65536-patch retrieval bank,
30 neighbors and temperature 0.02. CSRv2 uses training-mask labels in its loss.
NYUv2 early stopping begins at epoch 15 after a full epoch at target K, with
patience 3 and relative improvement threshold 0.001; no test-based selection.
The last completed epoch is evaluated. All settings live in the YAML files;
the launcher reads them rather than repeating their values in notebook checks.

Enable **Internet** and **GPU T4 x2** in Kaggle. One GPU also works. Push the
updated code to the GitHub `refactor` branch first. Paste this single code cell:

```python
# Enable Internet and GPU T4 x2 (one GPU also works). No Add Input or HF token needed.
DATASET = "nyuv2"  # Change only this to "voc" for the preserved VOC benchmark.

from pathlib import Path
import subprocess
import runpy

REPO = Path("/kaggle/working/MPSAE")
if not REPO.exists():
    subprocess.run(["git", "clone", "--branch", "refactor", "--single-branch",
                    "https://github.com/thienquang204/MPSAE.git", str(REPO)], check=True)
else:
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    assert branch == "refactor", f"Unexpected branch: {branch}"
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)

launcher = REPO / "scripts/kaggle_segmentation.py"
assert launcher.is_file(), "Push the organizational cleanup to GitHub, then rerun this cell."
runpy.run_path(str(launcher), init_globals={"DATASET": DATASET}, run_name="__main__")
```

The canonical implementation is `scripts/kaggle_segmentation.py`; the notebook
and this snippet only select the dataset and invoke it. No manual dataset upload
or Hugging Face token is required. VOC comes from its pinned HF dataset. NYUv2
downloads the labeled MAT from a pinned public HF mirror, verifies its SHA-256,
then applies the pinned official split and NYU40 mapping. Only RGB is used.

Outputs: `/kaggle/working/<dataset>/`, plus `NYUV2_results.zip` or `VOC_results.zip`.
The ZIP contains CSV/JSON/LaTeX, not weights or feature caches. The notebook displays
the matched-K comparison table; the VOC ZIP also retains its existing inference
sweep. Rerunning in the same workspace resumes compatible work. Save Kaggle outputs
to preserve the workspace across sessions; restore the whole workspace to resume
training, not just the small results ZIP. `/tmp` contains only the environment,
downloaded encoder weights and HF dataset cache and is ephemeral.

Each encoder is extracted once per dataset/config; independent jobs use up to two
GPUs. No backbone fine-tuning, ImageNet training, CLIP, Cityscapes or ADE20K jobs
are launched. NYUv2 GPU runtime/peak-memory estimates still require a real run.
For local MAT input instead of the mirror:

```bash
python -m mpsae.patch_scene.prepare_nyuv2 --source /path/to/input --destination work/nyuv2/dataset
```

Use `python -m mpsae.patch_scene plan --config configs/patch_scene_voc.yaml`
or `configs/patch_scene_nyuv2.yaml` to inspect the exact campaign before running.
