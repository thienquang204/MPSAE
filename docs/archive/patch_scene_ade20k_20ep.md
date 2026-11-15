Historical protocol; superseded by Cityscapes.

# ADE20K SAE rerun: up to 20 epochs

Use `configs/patch_scene_ade20k_20ep.yaml` with `--dataset ade20k`.
This selects SigLIP 2 and DINOv2, CSR/CSRv2/MPSAE, train K=3,12,48,
and inference K=3,12,48 for each fit: 18 fits and 56 evaluation conditions.
VOC is not run by the command below. The original campaign config is unchanged.

Training stops at 20 epochs, or after three consecutive eligible epochs without
more than 0.1% improvement over the last significant best sample-weighted mean
training objective. Monitoring starts at epoch 15, and only after a full epoch
at the target training K (including CSRv2's annealing). Earliest stop is epoch 18.
The last completed epoch is evaluated; no validation-score checkpoint selection
is performed. This is a training-loss plateau heuristic, not proof of convergence
or a held-out generalization criterion. A worsening loss can also trigger it.

`fits/<id>/convergence.json` saves per-epoch losses and patience. `summary.json`
records actual steps, completed epochs and `stop_reason`. Checkpoints persist
the stopping state and partial-epoch loss accumulation for interrupted resumes.
All other optimization settings stay unchanged. CSRv2's existing 70% annealing
rule now spans the first 14 of 20 epochs.

The new training identity prevents reuse of old 10-epoch fits as completed new
fits. New fits start fresh, reusing cached features when compatible. The old
checkpoints/results remain available. Reuse the existing workspace to avoid
extracting features again. The saved source revisions are reused from the
original 10-epoch campaign when available.

After pushing the code and pulling it into Kaggle, run this cell in the existing
environment (no Hugging Face upload/token needed):

```python
import subprocess
from pathlib import Path

repo = Path('/kaggle/working/MPSAE')
python = '/tmp/mpsae-env/bin/python'
subprocess.run([
    python, '-m', 'mpsae.patch_scene', 'run',
    '--config', 'configs/patch_scene_ade20k_20ep.yaml',
    '--workspace', '/kaggle/working/patch-scene-v2',
    '--dataset', 'ade20k', '--devices', 'cuda:0', 'cuda:1',
], cwd=repo, check=True)
```

The parent process updates the campaign tables and results ZIP as evaluations
complete. Locate downloadable archives with:

```python
from IPython.display import FileLink, display
for path in Path('/kaggle/working/patch-scene-v2').rglob('*.zip'):
    display(FileLink(str(path)))
```

Two GPUs run independent jobs; batch size remains 1024 per SAE. This command
does not resume a former 10-epoch fit for another 10 epochs, since that would
change CSRv2's annealing history and make the training recipe inconsistent.
