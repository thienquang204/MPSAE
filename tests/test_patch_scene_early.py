from dataclasses import replace
import numpy as np
import torch
from mpsae.patch_scene.config import Config
from mpsae.patch_scene.train import fit, update_plateau
from mpsae.patch_scene.artifacts import Hub
from mpsae.config import TrainPreset


def test_ade_only_plan_and_table_scope():
    from mpsae.patch_scene.report import expected_keys
    cfg = Config(datasets=['ade20k'], encoders=['siglip2', 'dinov2'],
                 train_k=[3, 12, 48], inference_k=[3, 12, 48], epochs=20,
                 early_stopping=True)
    assert cfg.plan()['sae_fits'] == 18
    assert cfg.plan()['total_evaluations'] == 56
    assert len(expected_keys(cfg)) == 56
    assert {key[0] for key in expected_keys(cfg)} == {'ade20k'}
    assert cfg.source_identity() == Config().source_identity()


def test_plateau_relative_patience_and_warmup():
    state = dict(best=None, bad_epochs=0, stopped=False)
    update_plateau(state, 10., False, 3, .001)
    assert state['best'] is None
    for loss in [10., 9.999, 9.998]:
        update_plateau(state, loss, True, 3, .001)
    assert not state['stopped']
    update_plateau(state, 9.997, True, 3, .001)
    assert state['stopped']


def test_early_resume_and_completed_reuse(tmp_path):
    torch.set_num_threads(1)
    config = Config(epochs=5, early_stopping=True, early_min_epochs=1,
                    early_patience=2, early_min_delta=.99)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 8)).astype('float32')
    y = np.eye(3, dtype='float32')[np.arange(24)%3]
    preset = TrainPreset(name='early-test', hidden_dim=64, epochs=5, batch_size=8,
                         lr=1e-12, k_aux=32)
    full, partial = Hub(tmp_path/'full'), Hub(tmp_path/'partial')
    args = (config, 'csr', x, y, {'test': True})
    a, sa = fit(*args, full, 'cpu', preset=preset)
    assert sa['stop_reason'] == 'training_loss_plateau'
    assert sa['epochs_completed'] == 3 and sa['steps'] == 9
    assert fit(*args, partial, 'cpu', preset=preset, stop_after=4) is None
    b, sb = fit(*args, partial, 'cpu', preset=preset)
    assert sa['convergence'] == sb['convergence']
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    _, reused = fit(*args, partial, 'cpu', preset=preset)
    assert reused == sb
    assert config.source_identity() == Config().source_identity()
    assert config.identity() != replace(config, early_stopping=False).identity()
