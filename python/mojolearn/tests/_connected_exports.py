# SPDX-License-Identifier: Apache-2.0
"""Shared check for the import-time module connection contract.

Native helpers read ``builtins`` (and the other fixed modules) through the
records of ``mojolearn._native_loaded_modules``. Since 31fc85981 ("bind native
module lookups to setup identities") a replaced connected export is refused
before any callback runs, with ``TypeError: native module connection export
changed: <module>.<name>`` (bindings/native_loaded_module.mojo; the contract is
pinned by test_native_cv_module_connections.py). Tests that used to expect a
replaced export to be ignored now assert this refusal and then repeat the
operation with the export restored.
"""


def assert_connected_export_refused(audit, module, name):
    report = audit.report()
    assert type(audit.exception) is TypeError, report
    assert str(audit.exception) == "native module connection export changed: " + module + "." + name, report
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[key], report
    assert audit.result is None
