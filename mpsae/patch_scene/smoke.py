"""Synthetic plumbing fixture, always isolated from real campaign artifacts."""
import numpy as np
from .features import save_npz


def fixture(hub, config, encoder='dinov2', dataset='voc'):
    rng = np.random.default_rng(config.seed)
    classes = 19 if dataset == 'cityscapes' else 3
    centers = rng.normal(size=(classes, 768)).astype('float32')
    target = np.arange(64) % classes
    features = centers[target] + 0.05*rng.normal(size=(64, 768)).astype('float32')
    labels = np.eye(classes, dtype='float32')[target]
    qtarget = np.arange(8).reshape(2, 4) % classes
    qfeatures = centers[qtarget] + 0.05*rng.normal(size=(2, 4, 768)).astype('float32')
    truth = np.repeat(np.repeat(qtarget.reshape(2, 2, 2), 2, axis=1), 2, axis=2).astype('uint8')
    relative = f'features/synthetic/{dataset}/{encoder}/validation/000000'
    save_npz(hub.root/relative/'data.npz', features=qfeatures, truth=truth.ravel(),
             offsets=np.array([0, 16, 32]), shapes=np.array([[4, 4], [4, 4]]))
    hub.push(relative)
    manifest = {'identity': {'SMOKE_ONLY': True, 'seed': config.seed, 'encoder': encoder, 'dataset': dataset}, 'classes': classes, 'grid': [2, 2],
                'counts': {'train': 64, 'validation': 2},
                'shards': [{'split': 'validation', 'start': 0, 'stop': 2, 'path': relative}]}
    return manifest, features, labels, np.arange(64)
