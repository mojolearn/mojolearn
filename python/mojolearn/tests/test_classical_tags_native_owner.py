# SPDX-License-Identifier: Apache-2.0
"""The native linear owner preserves sklearn's standard nested tag objects."""
import dataclasses

import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.utils import Tags, TargetTags, InputTags, RegressorTags
from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge, LinearRegression)


def native_tags(model):
    return ENTRY(model, "sklearn_tags", (), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_native_sklearn_tag_structure_matches_legacy_and_is_independent(cls, monkeypatch):
    model = cls()
    expected = dataclasses.asdict(model.__sklearn_tags__())
    calls = []

    def poison(*args, **kwargs):
        calls.append("constructor")
        raise AssertionError("Python dataclass constructor reached")

    from mojolearn import _mode
    with monkeypatch.context() as patch:
        patch.setattr(_mode, "sklearn_tags", poison)
        for tag_class in (Tags, TargetTags, InputTags, RegressorTags):
            patch.setattr(tag_class, "__init__", poison)
        first = clean(audit_native_call(native_tags, model))
        second = clean(audit_native_call(native_tags, model))
    assert calls == []
    assert type(first) is Tags
    assert type(first.target_tags) is TargetTags
    assert type(first.input_tags) is InputTags
    assert type(first.regressor_tags) is RegressorTags
    assert dataclasses.asdict(first) == expected
    assert dataclasses.asdict(second) == expected
    assert first is not second
    assert first.target_tags is not second.target_tags
    assert first.input_tags is not second.input_tags
    assert first.regressor_tags is not second.regressor_tags
    first.target_tags.required = False
    assert second.target_tags.required is True


def test_native_tags_refuse_python_property_replacement_before_hook(monkeypatch):
    calls = []

    def poison(*args, **kwargs):
        calls.append("property")
        raise AssertionError("Python tag property reached")

    monkeypatch.setattr(InputTags, "allow_nan", property(poison, poison))
    clean(audit_native_call(native_tags, Ridge()), TypeError)
    assert calls == []


def test_native_tags_refuse_python_finalizer_before_allocating_its_record(monkeypatch):
    calls = []

    def poison(*args, **kwargs):
        calls.append("finalizer")

    monkeypatch.setattr(TargetTags, "__del__", poison, raising=False)
    clean(audit_native_call(native_tags, Ridge()), TypeError)
    assert calls == []
