# One cell in a fresh Kaggle notebook: Internet ON, GPU T4 x2 (or one GPU).
from pathlib import Path
import subprocess
import runpy

MASSES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MMPOT_WEIGHT = None  # None keeps VOC's fixed lambda=2.0; set a float to override.
REPO = Path('/kaggle/working/MPSAE')
if not REPO.exists():
    subprocess.run(['git', 'clone', '--branch', 'refactor', '--single-branch',
                    'https://github.com/thienquang204/MPSAE.git', str(REPO)], check=True)
else:
    branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()
    assert branch == 'refactor', f'Unexpected branch: {branch}'
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
launcher = REPO/'scripts/kaggle_segmentation.py'
assert "globals().get('MMPOT_MASSES')" in launcher.read_text(), 'Push the mass-sweep update first.'
runpy.run_path(str(launcher), init_globals={
    'DATASET': 'voc', 'ENCODERS': ['siglip2'], 'METHODS': ['mpsae'],
    'INCLUDE_DENSE': False, 'MMPOT_WEIGHT': MMPOT_WEIGHT, 'MMPOT_MASSES': MASSES,
}, run_name='__main__')
