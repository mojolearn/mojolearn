# SPDX-License-Identifier: Apache-2.0
"""Installed provider admission seams; fresh processes isolate the route cache."""
import subprocess
import sys
import textwrap

import pytest


_SCRIPT = r'''
import ctypes
import sys
import numpy as np
from mojolearn._native_labels_bridge import labels_entry

entry, context = labels_entry()
base = context["labels_base_binding"]
if base is False:
    print("CPU-only install has no GPU label provider to admit")
    raise SystemExit(77)
if isinstance(base, str):
    raise RuntimeError(base)

v1 = base.labels_native_api_v1
v2 = base.labels_native_api_v2
valid = ctypes.pythonapi.PyCapsule_IsValid
valid.argtypes = (ctypes.py_object, ctypes.c_char_p)
valid.restype = ctypes.c_int
assert valid(v1(), b"mojolearn.labels.v1") == 1
assert valid(v2(), b"mojolearn.labels.v2") == 1

variant = sys.argv[1]
if variant == "old_v1":
    # This remains an exact compiled factory of the same extension. Admission
    # must reject its old capsule identity before any GIL-released operation.
    base.labels_native_api_v2 = v1
    expected = "neural native ABI: invalid capsule or version"
else:
    del base.labels_native_api_v2
    expected = "TypeError: native provider requires compiled entry labels_native_api_v2"
try:
    try:
        entry(0, (np.array([3, 1, 3], dtype=np.int64), 0), context)
    except (TypeError, ValueError, RuntimeError) as error:
        assert expected.removeprefix("TypeError: ") in str(error), repr(error)
    else:
        raise AssertionError("old or missing provider was accepted")
finally:
    base.labels_native_api_v2 = v2
print("PASS native label provider admission", variant)
'''


@pytest.mark.parametrize("variant", ["old_v1", "missing_v2"])
def test_label_provider_requires_native_only_version_before_execution(variant):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT), variant],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS native label provider admission {variant}" in result.stdout
