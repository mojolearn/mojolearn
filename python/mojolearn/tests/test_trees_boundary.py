# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the trees family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""


# moved from test_x_trees_sanity.py (vdv-pytest-b, 2026-11-24)
import numpy as np
import pytest

import mojolearn as ml


def _xts_clf(n_classes=3, seed=0):
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=1500, n_features=10, n_informative=6, n_classes=n_classes,
                               random_state=seed)
    return train_test_split(X.astype(np.float32), y, test_size=0.3, random_state=seed)


def test_cv_splitter_objects():
    # The meta-estimators admit mojolearn's native splitters and lists of
    # index pairs; a Python splitter's split() would be a callback inside the
    # native fit (bindings/trees_public_meta_door.mojo _cv_value).
    from mojolearn.model_selection import KFold, StratifiedKFold
    Xa, Xb, ya, yb = _xts_clf()
    from sklearn.metrics import accuracy_score
    est = [("dt", ml.DecisionTreeClassifier(max_depth=4))]
    a = ml.StackingClassifier(est, final_estimator=ml.DecisionTreeClassifier(max_depth=3), cv=3).fit(Xa, ya)
    b = ml.StackingClassifier(est, final_estimator=ml.DecisionTreeClassifier(max_depth=3),
                              cv=StratifiedKFold(3)).fit(Xa, ya)
    np.testing.assert_array_equal(np.asarray(a.transform(Xb)), np.asarray(b.transform(Xb)))
    pairs = [(np.asarray(tr).tolist(), np.asarray(te).tolist())
             for tr, te in KFold(4, shuffle=True, random_state=0).split(Xa)]
    c = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), cv=pairs).fit(Xa, ya)
    assert len(c.calibrated_classifiers_) == 4
    d = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), cv=KFold(3, shuffle=True, random_state=1),
                                  method="isotonic", ensemble=False).fit(Xa, ya)
    assert accuracy_score(yb, np.asarray(d.predict(Xb))) > 0.5


# moved from python/mojolearn/tests/test_native_scalar_decode_public.py
# Native buffer scalar conversion of IsolationForest fit options at the public boundary.
import numpy as np
import pytest

from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_iforest_fit as sdp_legacy
from mojolearn.tests.test_host_archive_native import _state as sdp_state
from mojolearn.tests.test_host_iforest_fit_native import _model as sdp_model, _pair as sdp_pair
from mojolearn.tests.test_host_kde_fit_native import _x as sdp_x


@pytest.mark.parametrize('dtype', ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'q', 'Q', 'f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('array', [False, True])
def test_fit_integer_options_decode_native_storage(dtype, array):
    value = np.dtype(dtype).type(2.75)
    if array:
        value = np.asarray(value)
    sdp_pair(sdp_x(), max_depth=value)


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('big_endian', [False, True])
def test_fit_float_options_preserve_native_rounding_and_endianness(dtype, big_endian):
    # The adjacent representable value exercises a consumed saved float,
    # including longdouble->double rounding on Linux extended formats.
    value = np.nextafter(np.dtype(dtype).type(.5), np.dtype(dtype).type(1.))
    value = np.asarray(value, dtype=np.dtype(dtype).newbyteorder('>' if big_endian else '='))
    sdp_pair(sdp_x(), max_features=value)


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
def test_fit_integer_conversion_does_not_round_before_truncation(dtype):
    value = np.nextafter(np.dtype(dtype).type(2), np.dtype(dtype).type(0))
    assert int(value) == 1
    _, model = sdp_pair(sdp_x(), max_depth=np.asarray(value))
    assert model._max_depth == 1


@pytest.mark.parametrize('dtype,value', [('f2', -0.), ('f4', 1e-30), ('f8', -0.),
                                       ('g', '1e-300'), ('F', 1j), ('D', -0j), ('G', 1j)])
def test_fit_truth_reads_native_scalar_components(dtype, value):
    # Complex truth examines both components without calling NumPy bool.
    sdp_pair(sdp_x(), bootstrap=np.asarray(value, dtype=dtype))


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_fit_nonfinite_integer_conversion_keeps_error_class_and_text(dtype, value):
    supplied = np.asarray(value, dtype=dtype)
    old, model = sdp_model(max_depth=supplied), sdp_model(max_depth=supplied)
    with pytest.raises((ValueError, OverflowError)) as expected:
        sdp_legacy.fit(old, sdp_x())
    audit = audit_native_call(model.fit, sdp_x())
    assert type(audit.exception) is type(expected.value), audit.report()
    assert str(audit.exception) == str(expected.value)
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert sdp_state(vars(model)) == sdp_state(vars(old))


@pytest.mark.parametrize('dtype,value', [('g', '-0'), ('g', '.5'), ('G', 1j), ('G', 0j)])
def test_fit_truth_decodes_non_native_extended_components(dtype, value):
    supplied = np.asarray(value, dtype=np.dtype(dtype).newbyteorder('>'))
    before = supplied.tobytes()
    sdp_pair(sdp_x(), bootstrap=supplied)
    assert supplied.tobytes() == before


# moved from python/mojolearn/tests/test_rf_class_weight.py
# RandomForestClassifier class_weight refusals: invalid weights and options, checked at fit.
import numpy as np
import pytest
from mojolearn import RandomForestClassifier


def _rfcw_data(balanced):
    rng = np.random.default_rng(0)
    x = rng.random((40, 4), dtype=np.float32)
    y = np.where(np.arange(40) % 2 == 0, 'a', 'b') if balanced else np.where(x[:, 0] > 0.3, 'a', 'b')
    return x, y


@pytest.mark.parametrize('weights', [{'a': -1}, {'a': np.inf}, {'a': np.nan},
    {'a': True}, {'a': '2'}, {'a': 1e100}, {'a': 1e-100}, {'a': 0, 'b': 0}, {'z': 2}])
def test_invalid_weights(weights):
    x, y = _rfcw_data(balanced=True)
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=2, class_weight=weights).fit(x, y)


@pytest.mark.parametrize('value', ['balanced_subsample', 'unknown', [], 3])
def test_invalid_option(value):
    # checked at fit, before any binding is touched (scikit-learn's convention)
    with pytest.raises(ValueError, match='class_weight'):
        RandomForestClassifier(class_weight=value).fit([[0.0], [1.0]], [0, 1])
