"""Engine routing and archive contracts; native arithmetic is checked separately."""
from types import SimpleNamespace
import numpy as np
import pytest
from mojolearn import (RandomForestClassifier, RandomForestRegressor,
                       ExtraTreesClassifier, ExtraTreesRegressor)
from mojolearn import _backend, _serialize

CLASSES = [RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor]


@pytest.mark.parametrize('cls', CLASSES)
def test_auto_engine_is_fast_only(cls):
    fast = cls(numeric_mode='fast')
    assert fast.inference_engine == 'auto'
    assert fast._prediction_engine() == 'parallel_groves'
    for mode in ('deterministic', 'identical'):
        model = cls(numeric_mode=mode)
        assert model.inference_engine == 'auto'
        assert model._prediction_engine() == 'sequential'
    assert cls(numeric_mode='fast', inference_engine='sequential')._prediction_engine() == 'sequential'


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
    assert not model._leaves.flags["WRITEABLE"]
    with pytest.raises(ValueError):
        np.asarray(model._leaves).setflags(write=True)
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


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('mode', ['fast', 'deterministic', 'identical'])
def test_wp3_public_default_borrows_input_and_output(cls, mode, monkeypatch):
    """Sabotage alternate entries so fallback cannot masquerade as reach."""
    import ctypes
    from mojolearn._arrays import _addr_ro
    from mojolearn._buffer import as_f32_c

    reached = []
    def forbidden(*args):
        pytest.fail('public prediction reached staged or uncached comparison arm')
    def predict(handle, source, destination, dims):
        reached.append((source, destination, tuple(dims)))
        inp = (ctypes.c_float * dims[0]).from_address(source)
        out = (ctypes.c_float * (dims[0] * dims[2])).from_address(destination)
        for row in range(dims[0]):
            for column in range(dims[2]):
                out[row * dims[2] + column] = inp[row] + column
        return dims[0]
    native = SimpleNamespace(forest_prepare_gpu=lambda *args: 1,
                             forest_predict_resident_gpu=forbidden,
                             forest_predict_resident_into_gpu=forbidden,
                             forest_predict_resident_reuse_gpu=predict,
                             forest_release_gpu=lambda handle: None)
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    model = fitted(cls, 'parallel_groves')
    model.numeric_mode = model._fit_numeric_mode = mode
    source, _ = as_f32_c(np.array([[2.], [7.], [-3.]], np.float32), name='X')
    # ET widens its public outputs after this boundary; inspect its native
    # vote buffer before that established public conversion.
    call = model._vote if hasattr(model, "_vote") else (
        model.predict_proba if model._num_outputs > 1 else model.predict)
    result = call(source)
    assert reached[0][0] == _addr_ro(source)
    assert reached[0][1] == _addr_ro(result)
    expected = np.array([[2. + j for j in range(model._num_outputs)],
                         [7. + j for j in range(model._num_outputs)],
                         [-3. + j for j in range(model._num_outputs)]], np.float32)
    np.testing.assert_array_equal(np.asarray(result).reshape(expected.shape).view(np.uint32),
                                  expected.view(np.uint32))


@pytest.mark.parametrize('cls', [RandomForestClassifier, ExtraTreesClassifier])
def test_classifier_combined_prediction_is_exact_and_traverses_once(cls, monkeypatch):
    """The evidence fast path must be the two public answers, not an approximation."""
    calls = []
    model = fitted(cls, 'sequential')
    model.classes_ = [10, 20]

    def vote(X):
        calls.append(len(X))
        from mojolearn._array import Array
        values = Array.from_list([[0.75, 0.25], [0.5, 0.5], [0.125, 0.875]], '<f4')
        return values[:len(X)]

    if cls is RandomForestClassifier:
        monkeypatch.setattr(model, 'predict_proba', vote)
    else:
        monkeypatch.setattr(model, '_vote', vote)
    X = np.ones((3, 1), dtype=np.float32)
    prediction, proba = model._predict_with_proba(X)
    assert calls == [3]
    np.testing.assert_array_equal(prediction, np.array([10, 10, 20]))
    expected = vote(X)
    assert calls == [3, 3]
    expected = expected if cls is RandomForestClassifier else expected.astype('<f8')
    actual_np, expected_np = np.asarray(proba), np.asarray(expected)
    view = np.uint32 if actual_np.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(actual_np.view(view), expected_np.view(view))


@pytest.mark.parametrize('cls', [RandomForestClassifier, ExtraTreesClassifier])
def test_fast_classifier_predict_uses_resident_device_argmax(cls, monkeypatch):
    """FAST labels cross as int32 codes; probability APIs keep their vote path."""
    import ctypes
    calls = []

    def labels(handle, source, destination, dims):
        calls.append(('labels', handle, tuple(dims)))
        out = (ctypes.c_int32 * dims[0]).from_address(destination)
        for row, code in enumerate((1, 0, 1)):
            out[row] = code
        return dims[0]

    def votes(handle, source, destination, dims):
        calls.append(('votes', handle, tuple(dims)))
        out = (ctypes.c_float * (dims[0] * dims[2])).from_address(destination)
        for row in range(dims[0]):
            out[row * 2] = 0.25
            out[row * 2 + 1] = 0.75
        return dims[0]

    native = SimpleNamespace(
        forest_prepare_gpu=lambda *args: 7,
        forest_predict_resident_gpu=votes,
        forest_predict_resident_into_gpu=votes,
        forest_predict_resident_reuse_gpu=votes,
        forest_predict_resident_labels_gpu=labels,
        forest_release_gpu=lambda handle: None,
    )
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    model = fitted(cls, 'parallel_groves')
    model.numeric_mode = model._fit_numeric_mode = 'fast'
    model.classes_ = [10, 20]
    X = np.ones((3, 1), dtype=np.float32)
    np.testing.assert_array_equal(model.predict(X), np.array([20, 10, 20]))
    assert [call[0] for call in calls] == ['labels']
    model.predict_proba(X)
    assert [call[0] for call in calls] == ['labels', 'votes']


@pytest.mark.parametrize('cls', [RandomForestClassifier, ExtraTreesClassifier])
def test_classifier_device_argmax_is_fast_only_and_optional(cls, monkeypatch):
    model = fitted(cls, 'parallel_groves')
    X = np.ones((2, 1), dtype=np.float32)
    def forbidden(*args):
        pytest.fail('non-FAST or sequential dispatch reached device argmax')
    native = SimpleNamespace(forest_predict_resident_labels_gpu=forbidden)
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    # Explicit parallel_groves does not weaken the reproducibility tier.
    assert model._predict_forest_labels(X) is None
    model.numeric_mode = model._fit_numeric_mode = 'fast'
    model.inference_engine = 'sequential'
    assert model._predict_forest_labels(X) is None
    model.inference_engine = 'parallel_groves'
    del native.forest_predict_resident_labels_gpu
    assert model._predict_forest_labels(X) is None


@pytest.mark.parametrize('cls', CLASSES)
def test_identical_auto_ordered_resident_keeps_sequential_bits_and_archive(cls, tmp_path, monkeypatch):
    """2026-09-22: a binary compiled with the strict increasing-tree resident
    route (`forest_ordered_resident() == 1`) serves IDENTICAL `auto` from a
    resident snapshot prepared ORDERED, while the model still predicts and
    saves AS 'sequential'. An explicit IDENTICAL `parallel_groves` prepares
    the 32-grove graph (ordered flag 0), the fold its recorded columns and
    the CPU host groves engine compute. Before the fix `auto` answered
    'parallel_groves', wrote a groves archive and, through the compiled
    default, moved explicit parallel_groves off the grove fold."""
    import ctypes
    prepared = []

    def prepare(*args):
        prepared.append(list(args[-1]))
        return len(prepared)

    def predict(handle, x, out, dims):
        values = np.ctypeslib.as_array((ctypes.c_float * (dims[0]*dims[2])).from_address(out))
        values[:] = 1 / dims[2]
        return dims[0]

    def sequential(*args):
        pytest.fail('IDENTICAL auto on an ordered-resident binary took the List route')

    native = SimpleNamespace(forest_ordered_resident=lambda: 1,
                             forest_prepare_gpu=prepare,
                             forest_predict_resident_reuse_gpu=predict,
                             forest_release_gpu=lambda handle: None,
                             rf_predict_proba=sequential, rf_predict_reg=sequential,
                             et_predict=sequential, et_predict_proba=sequential)
    monkeypatch.setattr(_backend, 'binding', lambda *args: native)
    X = np.ones((3, 1), dtype=np.float32)

    auto = fitted(cls, 'auto')
    assert auto._prediction_engine() == 'sequential'
    assert auto._ordered_resident_auto()
    (auto.predict_proba if auto._num_outputs > 1 else auto.predict)(X)
    assert prepared[-1][-1] == 1
    path = tmp_path/'auto.npz'
    auto.save(path)
    assert cls.load(path).inference_engine == 'sequential'
    with np.load(path, allow_pickle=False) as z:
        assert 'parallel-groves' not in _serialize.scalar_str(z, 'format')

    groves = fitted(cls, 'parallel_groves')
    (groves.predict_proba if groves._num_outputs > 1 else groves.predict)(X)
    assert prepared[-1][-1] == 0

    explicit = fitted(cls, 'sequential')
    assert not explicit._ordered_resident_auto()


def test_identical_auto_on_a_host_binding_is_sequential(monkeypatch):
    """A CPU-only install serves the host binding through a proxy that
    raises ImportError by name for a function it lacks; the ordered-resident
    probe must read that as absent, not refuse every RF prediction."""
    class Host:
        def __getattr__(self, item):
            raise ImportError(f'no CPU implementation of {item}')
    monkeypatch.setattr(_backend, 'binding', lambda *args: Host())
    model = fitted(RandomForestRegressor, 'auto')
    assert model._prediction_engine() == 'sequential'
    assert not model._ordered_resident_auto()
