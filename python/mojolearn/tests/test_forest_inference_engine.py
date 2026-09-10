"""Engine routing and archive contracts; native arithmetic is checked separately."""
from types import SimpleNamespace
import numpy as np
import pytest
from mojolearn import (RandomForestClassifier, RandomForestRegressor,
                       ExtraTreesClassifier, ExtraTreesRegressor)
from mojolearn import _backend, _serialize

CLASSES = [RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor]

@pytest.mark.parametrize('cls', CLASSES)
def test_engine_routes_and_revalidates(cls, monkeypatch):
    old, new = object(), object()
    native = SimpleNamespace(predict=old, predict_gpu_parallel=new)
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    model = cls(inference_engine='parallel_groves', numeric_mode='identical')
    assert model.get_params()['inference_engine'] == 'parallel_groves'
    assert model._prediction_function('predict') is new
    model.inference_engine = 'sequential'
    assert model._prediction_function('predict') is old
    model.inference_engine = 'bad'
    with pytest.raises(ValueError, match='inference_engine'):
        model._prediction_function('predict')
    with pytest.raises(ValueError, match='inference_engine'):
        cls(inference_engine=False)
    model.inference_engine = 'parallel_groves'
    with pytest.raises(RuntimeError, match='rebuild'):
        model._prediction_function('absent')


def fitted(cls, engine):
    model = cls(inference_engine=engine, numeric_mode='identical')
    model._capture_fit_mode()
    model._offsets = np.array([0, 1], dtype=np.int32)
    model._colid = np.array([0], dtype=np.int32)
    model._left_child = np.array([-1], dtype=np.int32)
    model._quesval = np.array([0], dtype=np.float32)
    model._num_outputs = 2 if cls._estimator_type == 'classifier' else 1
    model._leaves = np.array([1] + [0]*(model._num_outputs-1), dtype=np.float32)
    model._n_trees = model.n_features_in_ = model.max_features_ = 1
    model.depth_cap_bound_ = False
    model.max_depth_resolved_ = 1
    if model._num_outputs == 2:
        model.classes_ = np.array(['a', 'b'])
        model.n_classes_ = 2
    return model

@pytest.mark.parametrize('cls', CLASSES)
def test_versioned_gpu_archive_and_legacy_default(cls, tmp_path, monkeypatch):
    path = tmp_path/'model.npz'
    legacy = fitted(cls, 'sequential')
    legacy.save(path)
    with np.load(path, allow_pickle=False) as z:
        old_format = _serialize.scalar_str(z, 'format')
        assert 'numeric_mode' not in z.files
    restored = cls.load(path)
    assert restored.inference_engine == 'sequential'
    gpu = fitted(cls, 'parallel_groves')
    gpu.save(path)
    with pytest.raises(ValueError, match='format'):
        _serialize.read_npz(path, old_format)  # Old readers cannot silently change sums.
    restored = cls.load(path)
    monkeypatch.setattr(_backend, 'default_mode', lambda: 'fast')
    assert restored.inference_engine == 'parallel_groves'
    assert restored._effective_mode() == 'identical'
    np.testing.assert_array_equal(restored._leaves, gpu._leaves)
    gpu.set_params(inference_engine='sequential')
    assert not gpu.__sklearn_is_fitted__()
