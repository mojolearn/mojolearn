# SPDX-License-Identifier: Apache-2.0
"""Native scorer helpers keep numeric checks, label input and warnings native."""
import math
import warnings

import numpy as np
import pytest

from mojolearn import _expansion_metrics as expansion
from mojolearn import _metrics_impl as metrics
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


def invoke(operation, arguments):
    return expansion._NATIVE(None, operation, arguments, expansion._NATIVE_CONTEXT)


def native(operation, arguments):
    return audit_native_call(invoke, operation, arguments).assert_native()


@pytest.mark.parametrize("scalar", [0.25, np.float16(0.25), np.float32(0.25), np.float64(0.25)])
def test_pinball_real_scalar_uses_native_type_identity_and_storage(scalar):
    y = np.array([1, 2, 4, 8], dtype="f4")
    prediction = np.array([0, 3, 2, 9], dtype="f4")
    delta = y.astype("f8") - prediction
    expected = np.maximum(0.25 * delta, -0.75 * delta).mean()
    got = native("family_metric_pinball", (y, prediction, None, "uniform_average", "identical", scalar))
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("container", ["list", "string-array", "wide-integers", "array"])
def test_classification_label_preparation_has_no_python_protocol_frames(container, monkeypatch):
    actual, predicted = [0, 1, 1, 0], [0, 1, 0, 0]
    if container == "string-array":
        actual = np.array(["cat", "dog", "dog", "cat"])
        predicted = np.array(["cat", "dog", "cat", "cat"])
    elif container == "wide-integers":
        actual = [value + 2**80 for value in actual]
        predicted = [value + 2**80 for value in predicted]
    elif container == "array":
        actual, predicted = Array.from_list(actual, "<i8"), Array.from_list(predicted, "<i8")

    def poison(*args, **kwargs):
        raise AssertionError("Array Python method reached during native label preparation")

    with monkeypatch.context() as patch:
        patch.setattr(Array, "tolist", poison)
        patch.setattr(Array, "__iter__", poison)
        patch.setattr(Array, "__len__", poison)
        got = native("family_mexp_cls_balanced_accuracy", (actual, predicted, None, False, "identical"))
    assert got == pytest.approx(0.75)


@pytest.mark.parametrize("operation", ["classification", "regression"])
@pytest.mark.parametrize("action", ["always", "ignore", "error"])
def test_metric_warning_filters_and_real_exception_category_are_native(operation, action, monkeypatch):
    # Importing sklearn is test setup; the metric must preserve the selected
    # warning category without invoking its module attributes or warn hooks.
    from sklearn.exceptions import UndefinedMetricWarning

    if operation == "classification":
        binding = metrics._get_binding("identical")
        op = "family_mcls_prf"
        arguments = (binding, 0, [0, 0], [0, 0], None, 1, "binary", None, "warn", "identical")
    else:
        op = "family_metric_r2_options"
        y = np.array([1.0], dtype="f4")
        arguments = (y, y, None, "uniform_average", True, "identical")

    def poison(*args, **kwargs):
        raise AssertionError("warnings.warn Python callback reached")

    # Test setup: on a CPU-only install the binding is a _HostBinding whose
    # entries connect on first use (_backend._HostBinding.__getattr__, binding
    # selection glue). Connect them before the audited call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        invoke(op, arguments)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter(action, UndefinedMetricWarning)
        with monkeypatch.context() as patch:
            patch.setattr(warnings, "warn", poison)
            audit = audit_native_call(invoke, op, arguments)
        report = audit.report()
        assert not report["python_frames"]
        assert not report["forbidden_native_calls"]
        assert not report["unresolved_native_calls"]
        assert not report["wrapper_work"]
        assert not report["unobserved_existing_threads"]
        if action == "error":
            assert isinstance(audit.exception, UndefinedMetricWarning)
            assert "ill-defined" in str(audit.exception) or "not well-defined" in str(audit.exception)
        else:
            got = audit.assert_native()
            if operation == "classification":
                assert got == 0
            else:
                assert math.isnan(got)
    assert len(captured) == (1 if action == "always" else 0)
    if captured:
        assert captured[0].category is UndefinedMetricWarning


def test_custom_label_protocol_is_refused_before_hooks_run():
    calls = []

    class Labels(list):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

        def __len__(self):
            calls.append("len")
            return super().__len__()

    audit = audit_native_call(invoke, "family_mexp_cls_balanced_accuracy",
                              (Labels([0, 1]), [0, 1], None, False, "identical"))
    assert isinstance(audit.exception, TypeError)
    assert calls == []
    assert not audit.report()["python_frames"]


def test_native_array_report_materializes_results_without_array_tolist(monkeypatch):
    actual, predicted = Array.from_list([0, 1, 1, 0], "<i8"), Array.from_list([0, 1, 0, 0], "<i8")

    def poison(*args, **kwargs):
        raise AssertionError("Array.tolist Python callback reached")

    with monkeypatch.context() as patch:
        patch.setattr(Array, "tolist", poison)
        report = native("family_mexp_cls_report",
                        (actual, predicted, None, None, None, 2, True, 0, "identical"))
    assert report["accuracy"] == pytest.approx(0.75)
