# SPDX-License-Identifier: Apache-2.0
"""The linear estimators return scikit-learn's standard nested tag objects,
built by the Python compatibility layer (`_mode.sklearn_tags`); the native
linear owner answers no tag operation."""
import dataclasses

import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.utils import Tags, TargetTags, InputTags, RegressorTags
from mojolearn import LinearRegression, Ridge
from mojolearn._mode import sklearn_tags
from mojolearn._native_classical_bridge import connect

ENTRY, CONTEXT = connect(Ridge, LinearRegression)


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_sklearn_tag_structure_is_standard_and_independent(cls):
    model = cls()
    expected = dataclasses.asdict(sklearn_tags(model))
    first = model.__sklearn_tags__()
    second = model.__sklearn_tags__()
    assert type(first) is Tags
    assert type(first.target_tags) is TargetTags
    assert type(first.input_tags) is InputTags
    assert type(first.regressor_tags) is RegressorTags
    assert first.estimator_type == "regressor" and first.target_tags.required is True
    assert dataclasses.asdict(first) == expected
    assert dataclasses.asdict(second) == expected
    assert first is not second
    assert first.target_tags is not second.target_tags
    first.target_tags.required = False
    assert second.target_tags.required is True


def test_native_owner_has_no_tag_operation():
    with pytest.raises(Exception):
        ENTRY(Ridge(), "sklearn_tags", (), CONTEXT)
