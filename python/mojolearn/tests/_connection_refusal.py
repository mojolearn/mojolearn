# SPDX-License-Identifier: Apache-2.0
"""Shared check: a replaced connected builtin refuses before any hook runs.

Native lookups bind to the module exports retained at import time
(``mojolearn._native_loaded_modules``, commit 31fc85981 "bind native module
lookups to setup identities"): when a process replaces a connected export such
as ``builtins.list`` or ``builtins.float``, every native operation that reads
that module refuses with ``TypeError: native module connection export changed``
instead of calling the replacement. ``test_native_cv_module_connections``
pins the same contract for ``operator``.
"""

_CLEAN_KEYS = ("python_frames", "forbidden_native_calls", "unresolved_native_calls",
               "wrapper_work", "unobserved_existing_threads")


def assert_connection_refusal(audit, name, calls=(), module="builtins"):
    report = audit.report()
    for key in _CLEAN_KEYS:
        assert not report[key], report
    assert type(audit.exception) is TypeError, report
    expected = "native module connection export changed: " + module + "." + name
    assert expected in str(audit.exception), report
    assert not calls, calls
