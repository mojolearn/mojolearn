# SPDX-License-Identifier: Apache-2.0
"""Legacy column-target warning semantics at the native fit boundary."""
import warnings

import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._sk_validate import _conversion_warning

ENTRY, CONTEXT = connect(Ridge, LinearRegression)
CATEGORY = _conversion_warning()
MESSAGE = ("A column-vector y was passed when a 1d array was expected. Please change "
           "the shape of y to (n_samples, ), for example using ravel().")


def ridge_fit(model, x, y):
    return ENTRY(model, "ridge_fit", (x, y, None), CONTEXT)


def linear_fit(model, x, y):
    return ENTRY(model, "ols_fit", (x, y, None), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)
    return audit.exception


def fixture(layout="ordinary"):
    rng = np.random.default_rng(919)
    x = rng.normal(size=(16, 3)).astype("f4")
    y = (x @ np.array([0.25, -0.5, 0.75], "f4") + 0.5)[:, None]
    if layout == "strided":
        x, y = x[::-1], y[::-1]
    elif layout == "float64":
        y = y.astype("f8")
    return x, y


@pytest.mark.parametrize("cls,operation", [(Ridge, ridge_fit), (LinearRegression, linear_fit)])
@pytest.mark.parametrize("action", ["always", "ignore", "default", "module", "once", "error"])
def test_column_warning_actions_preserve_legacy_fit_and_category(cls, operation, action):
    x, y = fixture()
    old, native = cls(), cls()
    with warnings.catch_warnings(record=True) as reference:
        warnings.simplefilter(action, CATEGORY)
        if action == "error":
            with pytest.raises(CATEGORY, match="column-vector"):
                old.fit(x, y)
        else:
            for _ in range(2):
                old.fit(x, y)
    before = dict(vars(native))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter(action, CATEGORY)
        if action == "error":
            error = clean(audit_native_call(operation, native, x, y), CATEGORY)
            assert str(error) == MESSAGE
            assert vars(native) == before
        else:
            for _ in range(2):
                assert clean(audit_native_call(operation, native, x, y)) is native
    assert [(item.category, str(item.message)) for item in captured] == [
        (item.category, str(item.message)) for item in reference]
    for item in captured:
        assert type(item) is warnings.WarningMessage
        assert item.filename == CONTEXT["warning_filename"]
        assert item.filename == reference[0].filename
        assert item.lineno > 0
        assert item.line is None and item.source is None and item.file is None
    if action != "error":
        np.testing.assert_array_equal(np.asarray(native.coef_).view("u4"), np.asarray(old.coef_).view("u4"))
        assert np.float64(native.intercept_).view("u8") == np.float64(old.intercept_).view("u8")


@pytest.mark.parametrize("cls,operation", [(Ridge, ridge_fit), (LinearRegression, linear_fit)])
@pytest.mark.parametrize("layout", ["ordinary", "strided", "float64"])
def test_column_warning_native_recording_and_independent_flat_target(cls, operation, layout, monkeypatch):
    x, y = fixture(layout)
    wanted = cls().fit(x, np.asarray(y[:, 0], dtype="f4"))
    calls = []

    def poison(*args, **kwargs):
        calls.append("record constructor")
        raise AssertionError("Python warning constructor reached")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", CATEGORY)
        with monkeypatch.context() as patch:
            patch.setattr(warnings.WarningMessage, "__init__", poison)
            native = cls()
            clean(audit_native_call(operation, native, x, y))
    assert calls == []
    assert len(captured) == 1 and str(captured[0].message) == MESSAGE
    np.testing.assert_array_equal(np.asarray(native.coef_).view("u4"), np.asarray(wanted.coef_).view("u4"))


@pytest.mark.parametrize("hook", ["showwarning", "formatwarning", "_showwarnmsg", "_showwarnmsg_impl", "_formatwarnmsg", "_formatwarnmsg_impl"])
def test_column_warning_refuses_custom_hooks_before_calls(hook, monkeypatch):
    x, y = fixture()
    calls = []

    def poison(*args, **kwargs):
        calls.append(hook)
        raise AssertionError("Python warning hook reached")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", CATEGORY)
        monkeypatch.setattr(warnings, hook, poison)
        clean(audit_native_call(ridge_fit, Ridge(), x, y), TypeError)
    assert calls == []


@pytest.mark.parametrize("which", ["message", "module"])
def test_column_warning_refuses_custom_filter_matchers(which):
    x, y = fixture()
    calls = []

    class Matcher:
        def match(self, text):
            calls.append(text)
            raise AssertionError("Python filter matcher reached")

    with warnings.catch_warnings(record=True):
        warnings.filters[:] = [("always", Matcher() if which == "message" else None,
                                CATEGORY, Matcher() if which == "module" else None, 0)]
        clean(audit_native_call(ridge_fit, Ridge(), x, y), TypeError)
    assert calls == []


def test_column_warning_native_regex_filters_and_mutation():
    x, y = fixture()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", CATEGORY)
        warnings.filterwarnings("ignore", message="A column-vector", category=CATEGORY,
                                module=r"mojolearn\._mode$")
        clean(audit_native_call(ridge_fit, Ridge(), x, y))
        assert not captured
        warnings.simplefilter("default", CATEGORY)
        for _ in range(2):
            clean(audit_native_call(ridge_fit, Ridge(), x, y))
        assert len(captured) == 1
        warnings.simplefilter("always", CATEGORY)
        clean(audit_native_call(ridge_fit, Ridge(), x, y))
        assert len(captured) == 2


def test_nested_list_target_retains_legacy_refusal_without_warning():
    x, y = fixture()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        clean(audit_native_call(ridge_fit, Ridge(), x, y.tolist()), ValueError)
    assert not captured


@pytest.mark.parametrize("stream_kind", ["string", "file"])
def test_native_default_warning_output_without_python_formatting(tmp_path, stream_kind):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = tmp_path / "warning_default.py"
    script.write_text('''import io, json, sys, warnings
import numpy as np
from mojolearn import Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._sk_validate import _conversion_warning
ENTRY, CONTEXT = connect(Ridge)
def fit(model, x, y):
    return ENTRY(model, "ridge_fit", (x, y, None), CONTEXT)
x = np.arange(24, dtype="f4").reshape(12, 2)
y = np.arange(12, dtype="f4")[:, None]
warnings.simplefilter("always", _conversion_warning())
stream = io.StringIO() if sys.argv[1] == "string" else open(sys.argv[2], "w", encoding="utf-8")
sys.stderr = stream
report = audit_native_call(fit, Ridge(), x, y)
report.assert_native()
if sys.argv[1] == "string":
    output = stream.getvalue()
else:
    stream.close()
    output = open(sys.argv[2], encoding="utf-8").read()
sys.stderr = sys.__stderr__
print(json.dumps({"output": output, "audit": report.report()}))
''')
    result = subprocess.run([sys.executable, str(script), stream_kind, str(tmp_path / "warning.txt")],
                            env=os.environ.copy(), text=True, capture_output=True)
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert not data["audit"]["python_frames"]
    assert f": DataConversionWarning: {MESSAGE}\n" in data["output"]
    assert "  return method(self, *args, **kwargs)\n" in data["output"]
    assert "mojolearn/_mode.py:" in data["output"]
