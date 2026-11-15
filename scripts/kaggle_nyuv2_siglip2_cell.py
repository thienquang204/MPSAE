# One cell, fresh Kaggle notebook: Internet ON, GPU T4 x2 (or one GPU).
# Push the updated local refactor branch to GitHub before running.
from pathlib import Path
import subprocess
import runpy

repo = Path('/kaggle/working/MPSAE')
if not repo.exists():
    subprocess.run(['git', 'clone', '--branch', 'refactor', '--single-branch',
                    'https://github.com/thienquang204/MPSAE.git', str(repo)], check=True)
else:
    branch = subprocess.check_output(
        ['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    assert branch == 'refactor', f'Unexpected branch: {branch}'
    subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)

assert 'siglip2_mpsae_ot_weight: 0.5' in (
    repo/'configs/patch_scene_nyuv2.yaml').read_text(), (
    'Push the NYUv2 SigLIP2 OT-weight update to GitHub, then rerun.')
runpy.run_path(str(repo/'scripts/kaggle_segmentation.py'),
               init_globals={'DATASET': 'nyuv2', 'ENCODERS': ['siglip2']},
               run_name='__main__')
