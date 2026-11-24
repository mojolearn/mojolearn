# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`linalg.svdvals` of a WIDE operand (lane vdf-linalg2, 2026-11-24): the linalg entry reads the wide matrix once and
transposes it itself (decomposition/linalg_public_device.mojo `device_svdvals_at` on the device,
decomposition/linalg_host_impl.mojo `svdvals_host_at_run` in the host column). Its bytes must equal the tall entry on
the transpose made outside (the old door's route), on the device and in the host column, and the two columns must
agree with each other. Also the tall operand through the by-address entry (no host List copy) against the List entry.
"""
from core.hostptr import f32_ptr
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from decomposition.linalg_door_impl import svdvals_device_into, svdvals_device_at_into
from decomposition.linalg_host_impl import svdvals_host_run, svdvals_host_at_run
from std.testing import TestSuite


def _wide(rows: Int, cols: Int) -> List[Float32]:
    """A deterministic rows x cols matrix (rows < cols), exact float32 words."""
    var a = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows * cols):  # small-loop: a small fixture
        a[i] = Float32(((i * 37 + 11) % 97) - 48) / Float32(16.0)
    return a^


def _transpose(a: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    var t = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows):  # small-loop: a small fixture
        for j in range(cols):
            t[j * rows + i] = a[i * cols + j]
    return t^


def _same(label: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error(label + ": lengths differ")
    for i in range(len(a)):  # small-loop: a handful of singular values
        if a[i].to_bits() != b[i].to_bits():
            raise Error(label + ": value " + String(i) + " differs: " + String(a[i]) + " vs " + String(b[i]))


def _case(rows: Int, cols: Int) raises:
    var a = _wide(rows, cols)
    var t = _transpose(a, rows, cols)
    var dev_wide = List[Float32](length=rows, fill=0.0)
    var dev_tall = List[Float32](length=rows, fill=0.0)
    var dev_at = List[Float32](length=rows, fill=0.0)
    var host_wide = List[Float32](length=rows, fill=0.0)
    var host_tall = List[Float32](length=rows, fill=0.0)
    svdvals_device_at_into(Int(a.unsafe_ptr()), cols, rows, True, f32_ptr(Int(dev_wide.unsafe_ptr())))
    svdvals_device_into(t, cols, rows, f32_ptr(Int(dev_tall.unsafe_ptr())))
    svdvals_device_at_into(Int(t.unsafe_ptr()), cols, rows, False, f32_ptr(Int(dev_at.unsafe_ptr())))
    svdvals_host_at_run(Int(a.unsafe_ptr()), cols, rows, True, f32_ptr(Int(host_wide.unsafe_ptr())))
    svdvals_host_run(t, cols, rows, f32_ptr(Int(host_tall.unsafe_ptr())))
    _ = a
    _ = t
    _same("device wide vs device tall", dev_wide, dev_tall)
    _same("device by address vs device list", dev_at, dev_tall)
    _same("host wide vs host tall", host_wide, host_tall)
    _same("device vs host column", dev_wide, host_wide)


def test_svdvals_wide() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    _case(5, 48)
    _case(3, 7)
    _case(1, 9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
