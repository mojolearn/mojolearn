# SPDX-License-Identifier: Apache-2.0
"""Python boundary of `mojolearn.runtime_report()` (plan VERIFY_DECOUPLE_2026-11-15 (git history) "API").

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
    "version", "vendor", "vendor_how", "numeric_mode", "default_mode", "code_mode", "code_kind", "cpu_only",
    "gpu_arch", "gpu_arch_how", "host_dir", "host_families_built", "host_paths", "bindings", "fallbacks",
    "harness_source", "source_commit", "checkout_root", "package_dir",
}
ROW_FIELDS = {"path", "expected_path", "loaded", "identity", "vendor", "mode", "compiled_mode", "sabotage",
              "column", "detected_column"}


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
    assert report["code_kind"] in ("ptx", "native")
    if report["cpu_only"]:
        assert report["code_kind"] == "native"
    assert type(report["cpu_only"]) is bool and report["cpu_only"] == (report["vendor"] == "cpu")
    assert type(report["host_families_built"]) is tuple
    assert list(report["host_families_built"]) == sorted(report["host_families_built"])
    assert report["fallbacks"] == ()
    assert set(report["harness_source"]) == {"path", "how"}
    assert set(report["source_commit"]) == {"commit", "how"}
    for row in report["bindings"].values():
        assert set(row) == ROW_FIELDS
        assert type(row["loaded"]) is bool
    for name, row in report["bindings"].items():
        if not name.startswith("mojolearn._host."):
            assert row["expected_path"] is None
            assert row["column"] is None and row["detected_column"] is None
    assert set(report["host_paths"]) == set(report["host_families_built"])


def test_host_rows_read_back_their_column():
    # A loaded host binding reads back its kernel-matrix column, as the
    # binding's own `<prefix>_column()` says it.
    report = mojolearn.runtime_report()
    for key, row in report["bindings"].items():
        if not key.startswith("mojolearn._host.") or row["path"] is None:
            continue
        module = sys.modules[key]
        prefix = key.rsplit(".", 1)[-1][len("_mojolearn_"):]
        own = getattr(module, prefix + "_column", None)
        assert row["column"] == (None if own is None else str(own())), key
        basename = key.rsplit(".", 1)[-1]
        if basename in report["host_paths"]:
            assert report["host_paths"][basename] == row["expected_path"], key


def test_host_expected_path_is_the_resolver_answer():
    # The Python resolver `_backend.door_host_binary_path` is the oracle the
    # native field mirrors (override variable, own file, merged file).
    import os
    from mojolearn import _backend
    report = mojolearn.runtime_report()
    rows = {k: r for k, r in report["bindings"].items() if k.startswith("mojolearn._host.")}
    assert rows, "the package imported no host binding"
    for key, row in rows.items():
        want = _backend.door_host_binary_path(key.rsplit(".", 1)[-1])
        assert row["expected_path"] is not None
        assert os.path.realpath(row["expected_path"]) == os.path.realpath(want), key


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
