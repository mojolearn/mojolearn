"""Engine routing and archive contracts, through the real public methods
(the native forest owner, bindings/trees_public_forest.mojo); small inputs."""
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
    model = cls(numeric_mode='identical')
    assert model.inference_engine == 'auto'
    assert model._prediction_engine() == 'sequential'
    assert cls(numeric_mode='fast', inference_engine='sequential')._prediction_engine() == 'sequential'


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


def _data(cls):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    if cls._estimator_type == 'classifier':
        return X, (X[:, 0] > 0.5).astype(np.int32)
    return X, (2.0 * X[:, 0] + X[:, 1]).astype(np.float32)


def real_fit(cls, engine, mode='identical'):
    X, y = _data(cls)
    model = cls(n_estimators=4, max_depth=3, random_state=0, inference_engine=engine, numeric_mode=mode)
    return model.fit(X, y), X


def _outputs(model, X):
    return np.asarray(model.predict_proba(X) if model._estimator_type == 'classifier' else model.predict(X))


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


@pytest.mark.parametrize('cls', CLASSES)
def test_engine_reported_and_revalidated(cls):
    model, X = real_fit(cls, 'parallel_groves')
    assert model.get_params()['inference_engine'] == 'parallel_groves'
    assert model.inference_engine_ == 'parallel_groves'
    sequential, _ = real_fit(cls, 'sequential')
    assert sequential.inference_engine_ == 'sequential'
    model.inference_engine = 'bad'
    with pytest.raises(ValueError, match='inference_engine'):
        model.predict(X)
    # scikit-learn's convention: the constructor stores the value and the
    # fit-time checks refuse it.
    with pytest.raises(ValueError, match='inference_engine'):
        cls(inference_engine=False).fit(*_data(cls))


@pytest.mark.parametrize('cls', CLASSES)
def test_resident_model_reuse_freeze_and_pickle(cls):
    import pickle
    model, X = real_fit(cls, 'parallel_groves')
    first = _outputs(model, X)
    np.testing.assert_array_equal(_bits(_outputs(model, X)), _bits(first))
    # the snapshot's arrays are read-only copies: the caller cannot write
    # through them
    assert model._leaves._readonly
    restored = pickle.loads(pickle.dumps(model))
    assert restored.__dict__.get('_resident_holder') == {}
    np.testing.assert_array_equal(_bits(_outputs(restored, X)), _bits(first))
    model.set_params(inference_engine='sequential')
    assert not model.__sklearn_is_fitted__()


def test_resident_refuses_bad_arrays_before_pointer_handoff():
    from mojolearn._buffer import empty
    model, X = real_fit(RandomForestRegressor, 'parallel_groves')
    model._leaves = empty((0,), '<f4')
    with pytest.raises(ValueError, match='shapes'):
        model.predict(X)


@pytest.mark.parametrize('cls', [RandomForestClassifier, ExtraTreesClassifier])
@pytest.mark.parametrize('engine', ['sequential', 'parallel_groves', 'auto'])
def test_classifier_labels_are_the_argmax_of_the_vote(cls, engine):
    """Whichever route serves the labels (the fused device argmax or the
    vote's argmax), they are the first-max class of `predict_proba`."""
    model, X = real_fit(cls, engine)
    proba = np.asarray(model.predict_proba(X))
    expected = np.asarray(model.classes_)[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(np.asarray(model.predict(X)), expected)


@pytest.mark.parametrize('cls', CLASSES)
def test_identical_auto_keeps_sequential_bits_and_archive(cls, tmp_path):
    """IDENTICAL `auto` predicts and saves AS 'sequential' (the strict
    increasing-tree arithmetic, served resident on a GPU binding); an
    explicit IDENTICAL `parallel_groves` writes the groves archive."""
    auto, X = real_fit(cls, 'auto')
    assert auto.inference_engine_ == 'sequential'
    assert auto._prediction_engine() == 'sequential'
    explicit, _ = real_fit(cls, 'sequential')
    np.testing.assert_array_equal(_bits(_outputs(auto, X)), _bits(_outputs(explicit, X)))
    path = tmp_path / 'auto.npz'
    auto.save(path)
    loaded = cls.load(path)
    assert loaded.inference_engine == loaded.inference_engine_ == 'sequential'
    with np.load(path, allow_pickle=False) as z:
        assert 'parallel-groves' not in _serialize.scalar_str(z, 'format')
    groves, _ = real_fit(cls, 'parallel_groves')
    groves.save(path)
    with np.load(path, allow_pickle=False) as z:
        assert 'parallel-groves' in _serialize.scalar_str(z, 'format')
    assert cls.load(path).inference_engine_ == 'parallel_groves'
