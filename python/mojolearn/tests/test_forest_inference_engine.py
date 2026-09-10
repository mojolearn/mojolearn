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


@pytest.mark.parametrize('cls', CLASSES)
def test_resident_model_reuse_invalidation_and_lifetime(cls, monkeypatch):
    import ctypes
    import gc
    import pickle
    prepared, released, predicted = [], [], []

    def prepare(*args):
        prepared.append(args)
        return len(prepared)

    def predict(handle, x, out, dims):
        predicted.append(handle)
        values = np.ctypeslib.as_array((ctypes.c_float * (dims[0]*dims[2])).from_address(out))
        values[:] = 1 / dims[2]
        return dims[0]

    native = SimpleNamespace(forest_prepare_gpu=prepare,
                             forest_predict_resident_gpu=predict,
                             forest_predict_resident_into_gpu=predict,
                             forest_predict_resident_reuse_gpu=predict,
                             forest_release_gpu=released.append)
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    model = fitted(cls, 'parallel_groves')
    X = np.ones((3, 1), dtype=np.float32)
    original = model._leaves
    call = model.predict_proba if model._num_outputs > 1 else model.predict
    first = call(X)
    np.testing.assert_array_equal(call(X), first)
    assert predicted == [1, 1] and len(prepared) == 1
    assert not model._leaves.flags.writeable
    with pytest.raises(ValueError):
        model._leaves.setflags(write=True)
    original[:] = 99
    assert model._leaves[0] == 1

    restored = pickle.loads(pickle.dumps(model))
    assert not hasattr(restored, '_resident_forest')
    np.testing.assert_array_equal(restored.predict_proba(X) if model._num_outputs > 1
                                  else restored.predict(X), first)
    assert predicted[-1] == 2
    del restored
    gc.collect()
    assert released == [2]

    model._leaves = model._leaves.copy()
    call(X)
    gc.collect()
    assert len(prepared) == 3 and predicted[-1] == 3 and 1 in released
    model.set_params(inference_engine='sequential')
    gc.collect()
    assert sorted(released) == [1, 2, 3]


def test_resident_refuses_bad_arrays_before_pointer_handoff(monkeypatch):
    def fail(*args):
        pytest.fail('malformed host arrays reached native pointer ABI')
    native = SimpleNamespace(forest_prepare_gpu=fail, forest_predict_resident_gpu=fail,
                             forest_predict_resident_into_gpu=fail,
                             forest_predict_resident_reuse_gpu=fail,
                             forest_release_gpu=fail)
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    model = fitted(RandomForestRegressor, 'parallel_groves')
    model._leaves = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match='shapes'):
        model.predict(np.ones((2, 1), dtype=np.float32))


def test_resident_default_requires_reuse_binding(monkeypatch):
    native = SimpleNamespace(forest_prepare_gpu=lambda *args: 1,
                             forest_predict_resident_into_gpu=lambda *args: 0,
                             forest_predict_resident_gpu=lambda *args: 0,
                             forest_release_gpu=lambda handle: None)
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    model = fitted(RandomForestRegressor, 'parallel_groves')
    with pytest.raises(RuntimeError, match='rebuild'):
        model.predict(np.ones((2, 1), dtype=np.float32))
