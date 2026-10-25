"""Public forest auxiliary orchestration and existing ordered-kernel parity."""
import gc

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._array import Array
from mojolearn import _verify_native_runtime as audit_module
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._legacy_forest_auxiliary import legacy_auxiliary

OWNERS = ('RandomForestClassifier', 'RandomForestRegressor', 'ExtraTreesClassifier',
          'ExtraTreesRegressor', 'DecisionTreeClassifier', 'DecisionTreeRegressor')
METHODS = ('apply', 'predict_with_leaves', 'staged_predict')
X = np.array([[-1., 0.], [0., .5], [1., 1.], [2., 0.], [-0., 1.]], dtype=np.float32)


def clean(operation, *args, expected_error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if expected_error is not None:
        assert isinstance(audit.exception, expected_error), report
        return audit.exception
    return audit.assert_native()


def fixture(name):
    cls = getattr(ml, name)
    trees = 1 if name.startswith('DecisionTree') else 3
    params = dict(max_depth=1, numeric_mode='identical', random_state=0)
    if not name.startswith('DecisionTree'):
        params.update(n_estimators=trees, bootstrap=False)
    model = cls(**params)
    outputs = 2 if name.endswith('Classifier') else 1
    offsets = np.arange(trees + 1, dtype=np.int32) * 3
    columns = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0][:trees * 3], dtype=np.int32)
    thresholds = np.array([0., 0., 0., .5, 0., 0., 1.5, 0., 0.][:trees * 3], dtype=np.float32)
    left = np.tile(np.array([1, -1, -1], dtype=np.int32), trees)
    values = np.zeros((trees * 3, outputs), dtype=np.float32)
    for tree in range(trees):
        values[tree * 3 + 1] = ([1., 0.] if tree % 2 == 0 else [.5, .5]) if outputs == 2 else [tree + 1.]
        values[tree * 3 + 2] = ([0., 1.] if tree % 2 == 0 else [.25, .75]) if outputs == 2 else [tree + 4.]
    arrays = (offsets, columns, thresholds, left, values.reshape(-1))
    names = ('_offsets', '_colid', '_quesval', '_left_child', '_leaves')
    for key, value in zip(names, arrays):
        model.__dict__[key] = Array.from_list(value.tolist(), '<i4' if value.dtype.kind == 'i' else '<f4')
    model.__dict__.update(_n_trees=trees, _num_outputs=outputs, n_features_in_=2,
                          _fit_numeric_mode='identical', _mojolearn_fitted=True)
    return model, arrays


def reference(x, arrays, outputs):
    offsets, columns, thresholds, children, values = arrays
    trees = len(offsets) - 1
    values = values.reshape(-1, outputs)
    leaf = np.empty((len(x), trees), dtype=np.int32)
    prefixes = np.empty((len(x), trees, outputs), dtype=np.float32)
    for row in range(len(x)):
        total = np.zeros(outputs, dtype=np.float32)
        for tree in range(trees):
            base = int(offsets[tree])
            node = 0
            while children[base + node] != -1:
                node = int(children[base + node]) + int(x[row, columns[base + node]] > thresholds[base + node])
            leaf[row, tree] = node
            total = (total + values[base + node]).astype(np.float32)
            prefixes[row, tree] = total / np.float32(tree + 1)
    return prefixes[:, -1, :].copy(), leaf, prefixes


def assert_bytes(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('name', OWNERS)
@pytest.mark.parametrize('method', METHODS)
def test_all_public_auxiliary_methods_match_retained_path_and_independent_tree_fixture(name, method):
    model, arrays = fixture(name)
    expected = reference(X, arrays, model.__dict__['_num_outputs'])
    retained = legacy_auxiliary(model, X, leaves=True, prefixes=True, prediction=True)
    for old, value in zip(retained, expected):
        assert_bytes(old, value)
    result = clean(getattr(model, method), X)
    if method == 'apply':
        assert type(result) is Array
        assert_bytes(result, expected[1])
    elif method == 'staged_predict':
        assert type(result) is Array
        assert_bytes(result, expected[2])
    else:
        assert type(result) is tuple and all(type(item) is Array for item in result)
        assert_bytes(result[0], expected[0])
        assert_bytes(result[1], expected[1])


@pytest.mark.parametrize('method', METHODS)
def test_both_caller_boundaries_preserve_arrays_and_output_ownership(method, monkeypatch):
    model, _ = fixture('RandomForestRegressor')
    operation = getattr(model, method)
    with monkeypatch.context() as patch:
        patch.setattr(audit_module, '__mojolearn_user_boundary__', False)
        patch.setattr(audit_module, '__mojolearn_internal__', True, raising=False)
        internal = clean(operation, X)
    parts = internal if type(internal) is tuple else (internal,)
    assert all(type(item) is Array for item in parts)
    external = clean(operation, X)
    expected = [np.asarray(item).copy() for item in parts]
    del internal, parts, operation, model
    gc.collect()
    parts = external if type(external) is tuple else (external,)
    assert all(type(item) is Array for item in parts)
    for actual, value in zip(parts, expected):
        assert_bytes(actual, value)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('layout', ['float64_fortran', 'strided', 'list'])
def test_supported_input_storage_preserves_outputs(method, layout):
    model, _ = fixture('ExtraTreesClassifier')
    if layout == 'float64_fortran':
        x = np.asfortranarray(X, dtype=np.float64)
    elif layout == 'strided':
        backing = np.zeros((len(X), 4), dtype=np.float32)
        backing[:, ::2] = X
        x = backing[:, ::2]
    else:
        x = X.tolist()
    expected = clean(getattr(model, method), X)
    actual = clean(getattr(model, method), x)
    if type(actual) is tuple:
        for a, b in zip(actual, expected):
            assert_bytes(a, b)
    else:
        assert_bytes(actual, expected)


@pytest.mark.parametrize('method', METHODS)
def test_unfitted_width_rank_and_empty_refusals_match_the_previous_path(method):
    unfitted = ml.RandomForestRegressor(n_estimators=3, numeric_mode='identical')
    error = clean(getattr(unfitted, method), X, expected_error=RuntimeError)
    assert str(error) == 'this estimator is not fitted yet'
    model, _ = fixture('RandomForestRegressor')
    for bad in (np.zeros((len(X), 3), dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((0, 2), dtype=np.float32)):
        with pytest.raises(ValueError) as old:
            legacy_auxiliary(model, bad, leaves=True)
        error = clean(getattr(model, method), bad, expected_error=ValueError)
        assert str(error) == str(old.value)


@pytest.mark.parametrize('method', METHODS)
def test_python_input_protocol_is_refused_before_execution(method):
    calls = []

    class Input:
        def __array__(self, *args, **kwargs):
            calls.append('array')
            return X

        def __buffer__(self, flags):
            calls.append('buffer')
            return memoryview(X)

    model, _ = fixture('RandomForestRegressor')
    clean(getattr(model, method), Input(), expected_error=TypeError)
    assert not calls


@pytest.mark.parametrize('method', METHODS)
def test_python_model_buffer_protocol_is_refused_before_execution(method):
    calls = []

    class ModelBuffer:
        def __array__(self, *args, **kwargs):
            calls.append('array')
            return np.zeros(4, dtype=np.int32)

        def __buffer__(self, flags):
            calls.append('buffer')
            return memoryview(np.zeros(4, dtype=np.int32))

    model, _ = fixture('RandomForestRegressor')
    model.__dict__['_offsets'] = ModelBuffer()
    clean(getattr(model, method), X, expected_error=TypeError)
    assert not calls


@pytest.mark.parametrize('method', METHODS)
def test_nonfinite_inputs_still_refuse(method):
    model, _ = fixture('ExtraTreesRegressor')
    x = X.copy()
    x[0, 0] = np.nan
    # The retained direct-kernel helper exposes Mojo's generic Exception;
    # the public native owner translates this validation failure to ValueError.
    # Assert both real boundaries, rather than inventing the old helper's type.
    with pytest.raises(Exception, match='valid finite trees and input') as legacy:
        legacy_auxiliary(model, x, leaves=True)
    assert type(legacy.value) is Exception
    error = clean(getattr(model, method), x, expected_error=ValueError)
    assert 'valid finite trees and input' in str(error)


def test_private_combined_request_still_returns_all_three_outputs():
    model, arrays = fixture('RandomForestClassifier')
    expected = reference(X, arrays, 2)
    result = clean(model._forest_auxiliary, X, leaves=True, prefixes=True, prediction=True)
    assert type(result) is tuple and len(result) == 3
    for actual, value in zip(result, expected):
        assert_bytes(actual, value)
