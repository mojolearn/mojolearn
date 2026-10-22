"""Candidate-only callback and locale checks; the public array door is unchanged."""
import builtins
import locale
import warnings

import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


@pytest.fixture(scope="module")
def resident():
    values = np.array([0., -0., 1.25, 1e20, np.inf], dtype="f4")
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("candidate array qualification requires a device tensor")
    return values, tensor


def audit(tensor, dtype):
    return audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, None), None, _OPT_CONTEXT)


def assert_refused(result, calls):
    assert type(result.exception) is TypeError
    assert not calls
    report = result.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report


@pytest.mark.parametrize("name,mode", [("RuntimeWarning", "warn"), ("FloatingPointError", "raise")])
@pytest.mark.parametrize("replacement", ["function", "class"])
def test_native_array_refuses_replaced_exception_categories(resident, name, mode, replacement):
    _, tensor = resident
    calls = []
    original = getattr(builtins, name)
    def callback(*args):
        calls.append("function")
        return original(*args)
    class Foreign(original):
        def __init__(self, *args):
            calls.append("class")
            super().__init__(*args)
    with warnings.catch_warnings(), np.errstate(over=mode):
        warnings.simplefilter("always")
        setattr(builtins, name, callback if replacement == "function" else Foreign)
        try:
            result = audit(tensor, "f2")
        finally:
            setattr(builtins, name, original)
    assert_refused(result, calls)


def test_native_array_refuses_numpy_error_callback(resident):
    _, tensor = resident
    calls = []
    def callback(*args):
        calls.append("errstate")
    with np.errstate(over="call", call=callback):
        result = audit(tensor, "f2")
    assert_refused(result, calls)


def test_native_array_refuses_python_warning_display(resident):
    _, tensor = resident
    calls = []
    def display(*args, **kwargs):
        calls.append("showwarning")
    with warnings.catch_warnings(), np.errstate(over="warn"):
        warnings.simplefilter("always")
        warnings.showwarning = display
        result = audit(tensor, "f2")
    assert_refused(result, calls)


@pytest.mark.parametrize("dtype", ["S32", "U32"])
def test_native_array_text_preserves_callers_numeric_locale(resident, dtype):
    values, tensor = resident
    previous = locale.setlocale(locale.LC_NUMERIC)
    selected = None
    try:
        for name in ("fr_FR.UTF-8", "de_DE.UTF-8", "fr_FR", "de_DE"):
            try:
                selected = locale.setlocale(locale.LC_NUMERIC, name)
                break
            except locale.Error:
                pass
        if selected is None:
            pytest.skip("no non-default numeric locale is installed")
        result = audit(tensor, dtype).assert_native()
        np.testing.assert_array_equal(result, values.astype(dtype))
        assert locale.setlocale(locale.LC_NUMERIC) == selected
    finally:
        locale.setlocale(locale.LC_NUMERIC, previous)


def test_native_array_refuses_replaced_dtype_version_entry(resident, monkeypatch):
    _, tensor = resident
    calls = []
    def replaced():
        calls.append("version")
        return 0x01000009
    core = np.__dict__.get("_core", np.__dict__.get("core"))
    monkeypatch.setattr(core._multiarray_umath, "_get_ndarray_c_version", replaced)
    assert_refused(audit(tensor, ("f4", 1)), calls)
