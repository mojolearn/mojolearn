# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tests for `checks/scaffold.mojo` and `checks/binding_prelude.mojo`.

Every helper is checked against a value worked out by hand, and the
comparisons that decide a check (bits, same_bits, CaseTally) are shown to
FAIL on a planted wrong input before they are trusted. The upload/download
round trip needs a device and is skipped, by name, without one.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from max.gpu.host import DeviceContext

from checks.fixture_rng import u16_row_f32
from checks.numerics import GLOBAL_NUMERIC_MODE, numeric_mode_name
from checks.scaffold import (
    CaseTally,
    bits,
    card_path,
    download,
    fixture_matrix,
    grid,
    hex32,
    hex64,
    mode_name,
    rows,
    same_bits,
    task_count,
    upload,
)
from checks.vendor import COMPILED_VENDOR
from checks.binding_prelude import (
    compiled_vendor_name,
    f32_ptr,
    host_vendor_binding,
    i32_ptr,
    numeric_mode_binding,
    vendor_binding,
)


def test_grid() raises:
    assert_equal(grid(0, 256), 0)
    assert_equal(grid(1, 256), 1)
    assert_equal(grid(256, 256), 1)
    assert_equal(grid(257, 256), 2)
    assert_equal(grid(70001, 128), 547)


def test_rows() raises:
    var covered = 0
    var n = 1001
    var chunk = 64
    var tasks = task_count(n, chunk)
    assert_equal(tasks, 16)
    var prev_hi = 0
    for t in range(tasks):
        var r = rows(t, chunk, n)
        assert_equal(r.lo, prev_hi)
        assert_true(r.hi > r.lo)
        covered += r.hi - r.lo
        prev_hi = r.hi
    assert_equal(covered, n)
    var past = rows(tasks, chunk, n)
    assert_equal(past.lo, n)
    assert_equal(past.hi, n)
    assert_equal(task_count(0, 8), 1)


def test_bits_and_hex() raises:
    assert_equal(bits(Float32(1.0)), UInt32(0x3F800000))
    assert_equal(bits(Float32(-0.0)), UInt32(0x80000000))
    assert_equal(bits(Float64(1.0)), UInt64(0x3FF0000000000000))
    assert_true(same_bits(Float32(0.5), Float32(0.5)))
    # The planted wrong input: +0 and -0 compare EQUAL as floats and must
    # not as bits, or same_bits proves nothing.
    assert_true(Float32(0.0) == Float32(-0.0))
    assert_false(same_bits(Float32(0.0), Float32(-0.0)))
    assert_equal(hex32(Float32(1.0)), "0x3f800000")
    assert_equal(hex32(Float32(-2.5)), "0xc0200000")
    assert_equal(hex64(UInt64(0x9E3779B97F4A7C15)), "0x9e3779b97f4a7c15")
    assert_equal(hex64(UInt64(0)), "0x0000000000000000")


def test_mode_name() raises:
    assert_equal(mode_name(), numeric_mode_name())


def test_card_path() raises:
    _ = setenv("MOJOLEARN_IDENTITY_TRACE", "", True)
    assert_equal(card_path("/tmp/default.card"), "/tmp/default.card")
    _ = setenv("MOJOLEARN_IDENTITY_TRACE", "/tmp/override.card", True)
    assert_equal(card_path("/tmp/default.card"), "/tmp/override.card")
    _ = setenv("MOJOLEARN_IDENTITY_TRACE", "", True)


def test_fixture_matrix() raises:
    var m = fixture_matrix(7, 3, 11)
    assert_equal(len(m), 21)
    for i in range(7):
        for f in range(3):
            assert_equal(bits(m[i * 3 + f]), bits(u16_row_f32(i, f, 11)))
    # Distinct per cell, so a permutation of cells is visible.
    var distinct = 0
    for a in range(len(m)):
        for b in range(a + 1, len(m)):
            if not same_bits(m[a], m[b]):
                distinct += 1
    assert_equal(distinct, 21 * 20 // 2)


def test_case_tally() raises:
    var ok = CaseTally()
    ok.production("clean", 0)
    ok.sabotage("planted", 3)
    ok.finish("tally-ok")
    var bad = CaseTally()
    bad.production("clean", 2)
    with assert_raises():
        bad.finish("tally-bad-production")
    var inert = CaseTally()
    inert.sabotage("planted", 0)
    with assert_raises():
        inert.finish("tally-inert-sabotage")
    var empty = CaseTally()
    with assert_raises():
        empty.finish("tally-empty")


def test_binding_prelude() raises:
    assert_equal(compiled_vendor_name(), String(COMPILED_VENDOR))
    assert_equal(String(vendor_binding()), String(COMPILED_VENDOR))
    assert_equal(String(host_vendor_binding()), "cpu")
    assert_equal(Int(py=numeric_mode_binding()), GLOBAL_NUMERIC_MODE)
    var xs: List[Float32] = [1.5, -2.0, 3.25]
    var p = f32_ptr(Int(xs.unsafe_ptr()))
    assert_equal(bits(p[unsafe_offset=2]), bits(Float32(3.25)))
    var ys: List[Int32] = [7, -9]
    var q = i32_ptr(Int(ys.unsafe_ptr()))
    assert_equal(q[unsafe_offset=1], Int32(-9))
    with assert_raises():
        _ = f32_ptr(0)
    with assert_raises():
        _ = i32_ptr(0)
    _ = xs^
    _ = ys^


def test_upload_download() raises:
    comptime if not has_accelerator():
        print("SKIP test_upload_download: no accelerator on this box")
        return
    var ctx = DeviceContext()
    var xs: List[Float32] = [1.0, -0.0, 3.5, 1.0e-40, -7.25]
    var buf = upload(ctx, xs)
    var back = download(ctx, buf, len(xs))
    assert_equal(len(back), len(xs))
    for i in range(len(xs)):
        assert_true(same_bits(back[i], xs[i]))
    var head = download(ctx, buf, 2)
    assert_equal(len(head), 2)
    var empty = List[Float32]()
    var ebuf = upload(ctx, empty)
    assert_equal(len(download(ctx, ebuf, 0)), 0)


def main() raises:
    test_grid()
    test_rows()
    test_bits_and_hex()
    test_mode_name()
    test_card_path()
    test_fixture_matrix()
    test_case_tally()
    test_binding_prelude()
    test_upload_download()
    print("scaffold check: PASS")
