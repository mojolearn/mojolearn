# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`linalg.qr`'s resident geqrf / orgqr / triu (lane vdf-linalg2, 2026-11-24; x_decomp/resident_qr.mojo,
XD_DEV_GEQRF / XD_DEV_ORGQR / XD_DEV_TRIU through bindings/linalg_ops_kit_run.mojo). For each shape and mode the
resident kit (h, tau, Q and R kept on the device, downloaded once) must give the same bytes as the host-address route
of the same device image (XD_GEQRF / XD_ORGQR and the host R cut: the route before this lane) and as the host column
(x_decomp_host_v1).
"""
from core.entry_table import EntryRef, entry_ref_word
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.descriptor_schema import XD_OPERATION_COUNT, XD_CAP_RESIDENT
from x_decomp.descriptor_device import x_decomp_device_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from bindings.linalg_ops_kit_run import LinKit
from std.testing import TestSuite


def _kit(device: Bool, resident: Bool) -> LinKit:
    var word = entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0, XD_OPERATION_COUNT)
    var caps = XD_CAP_RESIDENT if resident else 0
    if device:
        return LinKit(EntryRef(x_decomp_device_v1, word), caps)
    return LinKit(EntryRef(x_decomp_host_v1, word), 0)


def _fixture(rows: Int, cols: Int) -> List[Float32]:
    var a = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows * cols):  # small-loop: a small fixture
        a[i] = Float32(((i * 53 + 7) % 89) - 44) / Float32(8.0)
    return a^


def _words(mut kit: LinKit, m: Int) raises -> List[Float32]:
    var count = kit.r(m) * kit.c(m)
    var out = List[Float32](length=max(count, 1), fill=0.0)
    var src = kit.ptr(m)
    for i in range(count):  # small-loop: a small result
        out[i] = src[i]
    return out^


def _qr(device: Bool, resident: Bool, a: List[Float32], rows: Int, cols: Int, mode: Int) raises -> List[Float32]:
    """mode 0 reduced, 1 complete, 2 raw, 3 wide 'r': the door's sequence; Q then R (or h^T then tau) appended."""
    var kit = _kit(device, resident)
    var out = List[Float32]()
    try:
        var tau = 0
        var held = kit.borrow(Int(a.unsafe_ptr()), rows, cols)
        var h = kit.geqrf(held, tau)
        var kk = min(rows, cols)
        if mode == 2:
            var ht = kit.transpose(h)
            out = _words(kit, ht)
            out.extend(_words(kit, tau))
        elif mode == 3:
            var rw = kit.triu(h, rows)
            out = _words(kit, rw)
        else:
            var q = kit.orgqr(h, tau, rows if mode == 1 else kk)
            var r = kit.triu(h, rows if mode == 1 else kk)
            out = _words(kit, q)
            out.extend(_words(kit, r))
        _ = a
    finally:
        kit.close()
    return out^


def _same(label: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error(label + ": lengths differ")
    for i in range(len(a)):  # small-loop: a small result
        if a[i].to_bits() != b[i].to_bits():
            raise Error(label + ": word " + String(i) + " differs")


def _case(rows: Int, cols: Int, mode: Int) raises:
    var a = _fixture(rows, cols)
    var tag = String(rows) + "x" + String(cols) + " mode " + String(mode)
    var resident = _qr(True, True, a, rows, cols, mode)
    var addressed = _qr(True, False, a, rows, cols, mode)
    var host = _qr(False, False, a, rows, cols, mode)
    _same(tag + " resident vs host-address device route", resident, addressed)
    _same(tag + " resident vs host column", resident, host)


def test_resident_qr() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    _case(9, 4, 0)
    _case(9, 4, 1)
    _case(9, 4, 2)
    _case(4, 9, 0)
    _case(4, 9, 3)
    _case(600, 3, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
