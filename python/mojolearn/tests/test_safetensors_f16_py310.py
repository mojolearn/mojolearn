# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""F16 tensors load on every Python the wheel supports (3.10 up).

memoryview.cast("e") exists only from Python 3.12; the 0.8.15 NVIDIA release
column ran on Python 3.11 and refused hf-checkpoint in
mojolearn/models/safetensors.py with "destination format must be a native
single character format". The reader now copies float16 bytes as they are
and widens them from their bits, on every version."""
import json
import os
import struct
import subprocess
import sys
import tempfile

from mojolearn.models.safetensors import SafetensorsFile

VALUES = [1.0, -2.5, 65504.0, 5.96e-08, 0.0, -0.0]


def _file(d):
    f16 = b"".join(struct.pack("<e", v) for v in VALUES)
    hdr = {"a": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, len(f16)]},
           "s": {"dtype": "F16", "shape": [], "data_offsets": [len(f16), len(f16) + 2]}}
    h = json.dumps(hdr).encode()
    p = os.path.join(d, "m.safetensors")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(h)) + h + f16 + struct.pack("<e", 3.0))
    return p


def test_f16_widens_bit_exactly():
    with tempfile.TemporaryDirectory() as d:
        r = SafetensorsFile(_file(d))
        a, s = r.read("a"), r.read("s")
        r.close()
    got = list(memoryview(a.tobytes()).cast("f"))
    want = [struct.unpack("<f", struct.pack("<f", struct.unpack("<e", struct.pack("<e", v))[0]))[0]
            for v in VALUES]
    assert a.shape == (2, 3) and a.dtype == "<f4"
    assert [struct.pack("<f", x) for x in got] == [struct.pack("<f", x) for x in want]  # -0.0 kept
    assert s.shape == () and s.tobytes() == struct.pack("<f", 3.0)


def test_f16_on_an_older_python_when_one_is_installed():
    """The same read under python3.10 or python3.11, where cast("e") refuses."""
    import shutil
    old = shutil.which("python3.11") or shutil.which("python3.10")
    if not old:
        import pytest
        pytest.skip("no python3.10 or python3.11 on this machine")
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = ("import sys; sys.path.insert(0, %r); from mojolearn.tests.test_safetensors_f16_py310 "
            "import test_f16_widens_bit_exactly as t; t(); print('OK')" % pkg)
    out = subprocess.run([old, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]
