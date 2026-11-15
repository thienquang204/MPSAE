# Paste this entire file into ONE cell in a new Kaggle notebook.
# Settings: Internet ON; GPU T4 x2 recommended. No dataset upload or HF token.
# Push the Cityscapes changes to your GitHub refactor branch before running.
from pathlib import Path
import subprocess
import runpy

REPO = Path('/kaggle/working/MPSAE')
if not REPO.exists():
    subprocess.run(['git', 'clone', '--branch', 'refactor', '--single-branch',
                    'https://github.com/thienquang204/MPSAE.git', str(REPO)], check=True)
else:
    branch = subprocess.check_output(
        ['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()
    assert branch == 'refactor', f'Unexpected branch: {branch}'
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)

assert (REPO/'mpsae/patch_scene/cityscapes_hf.py').is_file(), (
    'Cityscapes update missing: push your local refactor branch, then rerun this cell.')
runpy.run_path(str(REPO/'scripts/kaggle_segmentation.py'),
               init_globals={'DATASET': 'cityscapes'}, run_name='__main__')
