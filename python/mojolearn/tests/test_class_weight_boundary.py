# SPDX-License-Identifier: Apache-2.0
"""class_weight public boundary shared by LinearSVC, RandomForestClassifier and ExtraTreesClassifier: the
admission of the mapping, its refusals in scikit-learn's words, and the refusal of Python hooks on the keys."""
import numpy as np
import pytest
from pathlib import PurePosixPath
from collections import OrderedDict, defaultdict

from mojolearn import ExtraTreesClassifier, LinearSVC, RandomForestClassifier


# ---------------------------------------------------------------- moved from test_class_weight_shared_families.py
# (vdv-pytest-a)


def _cws_model(family, weight):
    if family == "svc":
        return LinearSVC(class_weight=weight, max_iter=100, tol=1e-5)
    cls = RandomForestClassifier if family == "rf" else ExtraTreesClassifier
    return cls(class_weight=weight, n_estimators=3, max_depth=2,
               bootstrap=False, random_state=7)


def _cws_data():
    return (np.array([[-2, 1], [-1, 2], [0, 0], [1, 0], [2, 1], [3, 0], [4, 2], [5, 1]], dtype="f4"),
            np.array([1, 1, 7, 7, 7, 7, 7, 7], dtype="i8"))


@pytest.mark.parametrize("family", ["svc", "rf", "et"])
def test_shared_class_weight_unknown_key_precedes_weight_range(family):
    x, y = _cws_data()
    # scikit-learn 1.9.1's words (sklearn/utils/class_weight.py:114-120), naming y's unweighted classes
    with pytest.raises(ValueError, match=r"^The classes, \[1, 7\], are not in class_weight$"):
        _cws_model(family, {9: -1}).fit(x, y)


@pytest.mark.parametrize("family", ["svc", "rf", "et"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "2", np.str_("2")])
def test_shared_families_keep_numeric_weight_admission(family, value):
    x, y = _cws_data()
    with pytest.raises(ValueError, match="finite nonnegative real"):
        _cws_model(family, {7: value}).fit(x, y)


@pytest.mark.parametrize("family", ["rf", "et"])
@pytest.mark.parametrize("mapping", [OrderedDict, defaultdict])
def test_tree_weight_native_mapping_types_keep_missing_class_default(family, mapping):
    weight = mapping(float, {7: 0.25}) if mapping is defaultdict else mapping([(7, 0.25)])
    x, y = _cws_data()
    actual = _cws_model(family, weight).fit(x, y)
    expected = _cws_model(family, None).fit(x, y, sample_weight=np.where(y == 7, 0.25, 1).astype("f4"))
    np.testing.assert_array_equal(np.asarray(actual.predict_proba(x)), np.asarray(expected.predict_proba(x)))
    assert list(weight.items()) == [(7, 0.25)]


@pytest.mark.parametrize("family", ["rf", "et"])
def test_tree_weight_key_rejects_path_equality_before_set_membership(family, monkeypatch):
    # Exact paths are valid for other estimator parameters, but are not native
    # label keys. Force an integer hash collision to expose unsafe membership.
    key = PurePosixPath("class-weight-key")
    original_hash = PurePosixPath.__hash__
    original_equal = PurePosixPath.__eq__
    calls = []

    def path_hash(self):
        return hash(1) if self is key else original_hash(self)

    def path_equal(self, other):
        if self is key:
            calls.append("eq")
            raise AssertionError("path equality executed inside native fit")
        return original_equal(self, other)

    monkeypatch.setattr(PurePosixPath, "__hash__", path_hash)
    weight = {key: 0.5}
    monkeypatch.setattr(PurePosixPath, "__eq__", path_equal)
    x, y = _cws_data()
    with pytest.raises(TypeError, match="exact native"):
        _cws_model(family, weight).fit(x, y)
    assert calls == []
