"""Public composition lifecycle regressions; requires the integrated bindings."""
import sys

import numpy as np
import pytest

import mojolearn as ml


@pytest.mark.parametrize('owner', ['rfe', 'imputer'])
def test_unfitted_composition_preserves_not_fitted_contract(owner):
    model = ml.RFE(ml.Ridge()) if owner == 'rfe' else ml.IterativeImputer()
    with pytest.raises(ValueError) as error:
        model.transform(np.ones((4, 3), np.float32))
    assert isinstance(error.value, AttributeError)
    assert 'not fitted' in str(error.value)


@pytest.mark.parametrize('owner', ['rfe', 'imputer'])
def test_composition_refuses_input_hooks_before_execution(owner):
    calls = []

    class Input:
        def __array__(self, *args, **kwargs):
            calls.append('array')
            return np.ones((4, 3), np.float32)

    model = ml.RFE(ml.Ridge()) if owner == 'rfe' else ml.IterativeImputer(max_iter=1)
    with pytest.raises(TypeError):
        model.fit(Input(), np.arange(4, dtype=np.float32))
    assert calls == []


@pytest.mark.parametrize('owner', ['rfe', 'imputer'])
def test_composition_runs_without_inherited_python_lifecycle(owner):
    x = np.arange(32, dtype=np.float32).reshape(8, 4) / 32
    y = x[:, 0].copy()
    model = ml.RFE(ml.Ridge(alpha=1), n_features_to_select=2) if owner == 'rfe' else ml.IterativeImputer(max_iter=1)
    calls = []

    def record(frame, event, arg):
        if event == 'call' and frame.f_code.co_filename.endswith('/mojolearn/_mode.py'):
            if frame.f_code.co_name in ('tracked', 'guarded', 'public'):
                calls.append(frame.f_code.co_qualname)

    previous = sys.getprofile()
    sys.setprofile(record)
    try:
        result = model.fit_transform(x, y)
        again = model.transform(x)
        if owner == 'rfe':
            assert model.get_support(indices=True) == [i for i, keep in enumerate(model.get_support()) if keep]
    finally:
        sys.setprofile(previous)
    assert calls == []
    assert type(result) is np.ndarray and type(again) is np.ndarray
    np.testing.assert_array_equal(result, again)


def test_selector_rejects_python_truth_hooks_inside_object_array_flag():
    calls = []

    class Flag:
        def __bool__(self):
            calls.append('bool')
            return True

    x = np.arange(32, dtype=np.float32).reshape(8, 4) / 32
    model = ml.RFE(ml.Ridge(alpha=1), n_features_to_select=2).fit(x, x[:, 0])
    flag = np.empty(1, dtype=object)
    flag[0] = Flag()
    with pytest.raises(TypeError):
        model.get_support(indices=flag)
    assert calls == []
