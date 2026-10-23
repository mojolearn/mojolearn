# SPDX-License-Identifier: Apache-2.0
"""Audit complete public metadata calls, including normalization and refusals."""
import os
import subprocess
import sys

import pytest

from mojolearn import training as T, _training_impl, _backend
from mojolearn._verify_native_runtime import audit_native_call


OPERATIONS = (T.numeric_mode_used, T.vendor_used)


def _refusal(audit, category, match):
    report = audit.report()
    for key in ("python_frames", "forbidden_native_calls", "unresolved_native_calls",
                "wrapper_work", "unobserved_existing_threads"):
        assert not report[key], report
    assert type(audit.exception) is category, report
    assert match in str(audit.exception), report


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", [None, "identical", " \tIDENTICAL\n", "\u2003identical\u2003", ""])
def test_public_metadata_uses_compiled_values_without_loaders(operation, mode, monkeypatch):
    # Independent test setup reads the retained binary directly before the
    # audited operation. No loader, profiler or oracle runs inside it.
    binding = _backend.binding("_mojolearn_training", "identical")
    expected = (_backend._CODE_MODE[binding.training_numeric_mode()]
                if operation is T.numeric_mode_used else binding.training_vendor())
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "identical")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("loader")
        raise AssertionError("metadata called a Python loader")

    for owner, name in ((_training_impl, "_load"), (_backend, "binding"),
                        (_backend, "default_mode"), (_backend, "load_set")):
        monkeypatch.setattr(owner, name, forbidden)
    value = audit_native_call(operation, mode).assert_native()
    assert type(value) is str and value == expected
    assert not calls


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", ["deterministic", "unknown", " ", b"identical", False, 0])
def test_metadata_rejects_unsupported_modes_natively(operation, mode):
    _refusal(audit_native_call(operation, mode), ValueError, "numeric_mode")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("string_subclass", [False, True])
def test_metadata_rejects_mode_hooks_before_invoking_them(operation, string_subclass):
    calls = []

    class Hooks:
        def __bool__(self):
            calls.append("bool")
            raise AssertionError("mode truth hook ran")

        def __str__(self):
            calls.append("str")
            raise AssertionError("mode string hook ran")

        def strip(self):
            calls.append("strip")
            raise AssertionError("mode strip hook ran")

        def lower(self):
            calls.append("lower")
            raise AssertionError("mode lower hook ran")

    class StringHooks(Hooks, str):
        pass

    mode = StringHooks("identical") if string_subclass else Hooks()
    _refusal(audit_native_call(operation, mode), ValueError, "native str")
    assert not calls


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("category", [ImportError, AttributeError, TypeError, ValueError,
                                      RuntimeError, NotImplementedError])
def test_metadata_preserves_connected_binding_failure_category(operation, category, monkeypatch):
    message = "retained unavailable training binding"
    records = _training_impl._OPT_CONTEXT["training_metadata_bindings"]
    monkeypatch.setitem(records, "identical", category.__name__ + ": " + message)
    _refusal(audit_native_call(operation, "identical"), category, message)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("index", [0, 1])
def test_metadata_refuses_replaced_python_export_before_call(operation, index, monkeypatch):
    records = _training_impl._OPT_CONTEXT["training_metadata_bindings"]
    entries = list(records["identical"])
    calls = []

    def forbidden():
        calls.append("export")
        raise AssertionError("Python metadata export ran")

    entries[index] = forbidden
    monkeypatch.setitem(records, "identical", tuple(entries))
    _refusal(audit_native_call(operation, "identical"), TypeError, "compiled entries")
    assert not calls


@pytest.mark.parametrize("operation", OPERATIONS)
def test_metadata_rechecks_actual_compiled_mode(operation, monkeypatch):
    records = _training_impl._OPT_CONTEXT["training_metadata_bindings"]
    # A fixed native test function returns an unsupported integer code.
    # This exercises the refusal without compiling a deliberately wrong tier.
    assert sys.getrecursionlimit() not in (0, 1)
    monkeypatch.setitem(records, "identical", (sys.getrecursionlimit, records["identical"][1]))
    category = ImportError if _backend._CPU_ONLY is not None else RuntimeError
    _refusal(audit_native_call(operation, "identical"), category, "compiled")


@pytest.mark.parametrize("operation", OPERATIONS)
def test_metadata_reads_current_default_without_python_default_function(operation, monkeypatch):
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "retired")
    _refusal(audit_native_call(operation), ValueError, "numeric_mode")
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "identical")
    expected = _backend.binding("_mojolearn_training", "identical")
    result = audit_native_call(operation).assert_native()
    assert result == ("identical" if operation is T.numeric_mode_used else expected.training_vendor())


def test_metadata_does_not_import_dependencies_or_invoke_import_hooks(tmp_path):
    code = r'''
import builtins, json, sys
class BlockDependencies:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('numpy', 'scipy', 'sklearn', 'pandas'):
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, BlockDependencies())
from mojolearn import training as T, _backend
from mojolearn._verify_native_runtime import audit_native_call
expected = _backend.binding('_mojolearn_training', 'identical').training_vendor()
calls = []
original = builtins.__import__
def forbidden(*args, **kwargs):
    calls.append(args[0])
    raise AssertionError('runtime import hook reached')
audits = []
builtins.__import__ = forbidden
try:
    audits.append(audit_native_call(T.numeric_mode_used, 'identical'))
    audits.append(audit_native_call(T.vendor_used, 'identical'))
finally:
    builtins.__import__ = original
assert not calls, calls
assert audits[0].assert_native() == 'identical'
assert audits[1].assert_native() == expected
assert not any(name in sys.modules for name in ('numpy', 'scipy', 'sklearn', 'pandas'))
print(json.dumps({'native_calls': 2, 'runtime_imports': calls, 'vendor': expected}))
'''
    log = tmp_path / "metadata-no-dependencies.log"
    with log.open("w") as output:
        child = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(),
                               stdout=output, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert child.returncode == 0, log.read_text()[-4000:]
