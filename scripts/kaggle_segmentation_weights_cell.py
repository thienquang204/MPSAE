# One cell in a fresh Kaggle notebook: Internet ON, GPU T4 x2 (one GPU also works).
# Push this update to GitHub's refactor branch before running.
DATASET = 'voc'                         # 'voc', 'nyuv2', or 'cityscapes'
ENCODERS = ['siglip2']                  # add 'dinov2' to compare both encoders
METHODS = ['csr', 'csrv2', 'mpsae']     # ['mpsae'] runs only our method
TRAIN_KS = [3, 12, 48]                 # each SAE evaluates only its own K
MMPOT_MASS = 1.0                       # s=1: balanced MMOT; s<1: partial MMPOT
MMPOT_WEIGHTS = [0.5, 1.0, 2.0]       # lambda in L_other + lambda * L_transport
INCLUDE_DENSE = True

from pathlib import Path
import subprocess
import runpy

REPO = Path('/kaggle/working/MPSAE')
if not REPO.exists():
    subprocess.run(['git', 'clone', '--branch', 'refactor', '--single-branch',
                    'https://github.com/thienquang204/MPSAE.git', str(REPO)], check=True)
else:
    branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()
    assert branch == 'refactor', f'Unexpected branch: {branch}'
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
assert (REPO/'mpsae/patch_scene/weight_sweep.py').is_file(), 'Push the weight-sweep update first.'
runpy.run_path(str(REPO/'scripts/kaggle_segmentation.py'), init_globals={
    'DATASET': DATASET, 'ENCODERS': ENCODERS, 'METHODS': METHODS,
    'TRAIN_KS': TRAIN_KS, 'MMPOT_MASS': MMPOT_MASS,
    'MMPOT_WEIGHTS': MMPOT_WEIGHTS, 'INCLUDE_DENSE': INCLUDE_DENSE,
}, run_name='__main__')
