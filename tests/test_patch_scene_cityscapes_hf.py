from pathlib import Path
import io
import pickle
import numpy as np
import pytest
from PIL import Image

from mpsae.patch_scene.cityscapes_hf import HFCitySpec, ParquetRows, parquet_files, REVISION
from mpsae.patch_scene.config import load, require_supported


def test_label_ids_not_colors_or_train_ids():
    raw = np.array([[0, 7, 8, 11, 24, 33, 255]], dtype=np.uint8)
    expected = np.array([[255, 0, 1, 2, 11, 18, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(HFCitySpec().remap(raw), expected)
    rgb = np.repeat(raw[..., None], 3, axis=2)
    np.testing.assert_array_equal(HFCitySpec().remap(rgb), expected)
    rgb[0, 0, 1] = 1
    with pytest.raises(ValueError, match='identical'):
        HFCitySpec().remap(rgb)
    with pytest.raises(ValueError):
        HFCitySpec().remap(np.array([[200]], dtype=np.uint8))


def test_native_parquet_rows_no_arrow_copy(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    def png(value):
        stream = io.BytesIO()
        Image.new('RGB', (2048, 1024), (value,)*3).save(stream, format='PNG')
        return {'bytes': stream.getvalue(), 'path': None}
    rows = [{'image': png(i), 'semantic_segmentation': png(7+i)} for i in range(3)]
    path = tmp_path/'sample.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    data = ParquetRows([path])
    assert len(data) == 3 and len(set(data['id'])) == 3
    assert data[2]['id'] == data['id'][2]
    assert np.asarray(data[2]['mask'])[0, 0, 0] == 9
    restored = pickle.loads(pickle.dumps(data))
    assert restored._cache is None and restored[0]['id'] == data[0]['id']
    with pytest.raises(IndexError): data[3]


def test_pinned_train_validation_only_and_plan():
    files = parquet_files()
    assert set(files) == {'train', 'validation'}
    assert [len(v) for v in files.values()] == [14, 3]
    assert all(REVISION in p and '/test-' not in p for v in files.values() for p in v)
    with pytest.raises(ValueError): parquet_files('main')
    root = Path(__file__).resolve().parents[1]
    config = load(root/'configs/patch_scene_cityscapes.yaml')
    require_supported(config)
    assert config.all_train_patches and config.image_hw == (336, 672)
    assert config.epochs == 20 and config.early_stopping
    assert config.evaluation_pairs == ((3,3), (12,12), (48,48))
    plan = config.plan()
    assert (plan['sae_fits'], plan['total_evaluations']) == (18,20)
    assert plan['datasets']['cityscapes'][0] == 'Chris1/cityscapes'
    assert plan['support_candidate_limit'] is None


def test_source_resolution_does_not_require_local_zip(monkeypatch, tmp_path):
    from mpsae.patch_scene.features import resolve_sources
    from mpsae.patch_scene.artifacts import Hub
    from mpsae.patch_scene import cityscapes
    import huggingface_hub
    monkeypatch.setattr(cityscapes, 'load_local', lambda **kw: pytest.fail('local ZIP loader called'))
    class API:
        def model_info(self, repo):
            from types import SimpleNamespace
            return SimpleNamespace(sha='model-sha')
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    config = load(Path(__file__).resolve().parents[1]/'configs/patch_scene_cityscapes.yaml')
    result = resolve_sources(config, Hub(tmp_path))
    assert result['datasets']['cityscapes'] == REVISION
