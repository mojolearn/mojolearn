# SPDX-License-Identifier: Apache-2.0
"""Python boundary of `mojolearn.runtime_report()` (docs/plans/VERIFY_DECOUPLE_2026-11-15.md "API").

One native call: the returned object is a plain dict with the documented
fields, it agrees with the public metadata calls, it loads no binding, it
enters no Python frame, and its binding identities are the files' SHA-256
(checked against hashlib, an independent oracle).
"""
import hashlib
import re
import sys

import mojolearn
from mojolearn._verify_native_runtime import audit_native_call

FIELDS = {
    "version", "vendor", "vendor_how", "numeric_mode", "default_mode", "code_mode", "cpu_only",
    "gpu_arch", "gpu_arch_how", "host_dir", "host_families_built", "bindings", "fallbacks",
    "harness_source", "source_commit", "checkout_root", "package_dir",
}
ROW_FIELDS = {"path", "loaded", "identity", "vendor", "mode", "compiled_mode", "sabotage"}


def _loaded_bindings():
    return {name for name in sys.modules if name.rsplit(".", 1)[-1].startswith("_mojolearn")}


def test_fields_and_public_agreement():
    report = mojolearn.runtime_report()
    assert type(report) is dict and set(report) == FIELDS
    assert report["version"] == mojolearn.__version__
    assert report["vendor"] == mojolearn.vendor()
    assert report["numeric_mode"] == mojolearn.numeric_mode()
    assert report["gpu_arch"] == mojolearn.gpu_arch()
    assert report["gpu_arch_how"] == mojolearn.gpu_arch_how()
    assert report["default_mode"] in ("identical", "fast")
    assert report["code_mode"] in (None, "identical", "fast", "unknown")
    assert type(report["cpu_only"]) is bool and report["cpu_only"] == (report["vendor"] == "cpu")
    assert type(report["host_families_built"]) is tuple
    assert list(report["host_families_built"]) == sorted(report["host_families_built"])
    assert report["fallbacks"] == ()
    assert set(report["harness_source"]) == {"path", "how"}
    assert set(report["source_commit"]) == {"commit", "how"}
    for row in report["bindings"].values():
        assert set(row) == ROW_FIELDS
        assert type(row["loaded"]) is bool


def test_fresh_and_read_only():
    first = mojolearn.runtime_report()
    first["vendor"] = "changed"
    first["bindings"].clear()
    second = mojolearn.runtime_report()
    assert second["vendor"] == mojolearn.vendor()
    assert second is not first


def test_loads_nothing_and_enters_no_python():
    before = _loaded_bindings()
    audit = audit_native_call(mojolearn.runtime_report)
    report = audit.assert_native()
    assert _loaded_bindings() == before
    assert type(report) is dict


def test_identity_is_the_file_sha256():
    report = mojolearn.runtime_report()
    rows = [row for row in report["bindings"].values() if row["loaded"] and row["path"] is not None]
    assert rows, "the package imported no native binding"
    row = rows[0]
    assert re.fullmatch("[0-9a-f]{64}", row["identity"])
    with open(row["path"], "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == row["identity"]
