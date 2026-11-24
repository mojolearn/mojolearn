# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The metrics' device label and refusal stages over device-resident inputs
(core/metric_labels_device.mojo, lane vdf-xm-labels 2026-11-24) against the
host column, byte for byte.

  0 encode: int32 labels whose range fits the count (the host's dense table)
  1 encode: int64 labels of a range far wider than the count (the host's
    radix route), negative and positive values beyond 32 bits
  2 encode: int8 labels at both ends of the type
  3 encode: 70,000 int16 labels (many blocks, many radix tiles)
  4 gather: 70,000 codes with misses (a code mapped to -1 and a code past
    the table), the first miss in a late block
  5 gather: no miss (the answer is -1), and every row written
  6 scans: 70,000 float32 words; none planted, then a NaN and a later
    infinity (the first index), then a -0.0 and a later -1.0 for the sign
    scan (-0.0 is not negative)

Encode: the labels go up once as packed 4-byte words (as a store slot holds
them); `metric_encode_resident`'s k, its k classes at the labels' width and
its device codes (downloaded here only to compare) must equal
`score_encode_signed`'s on the same bytes. Gather: `metric_gather_resident`'s
rows and first miss against `mexp_gather_i32`. Scans: the device answers
against the expected index computed here and `host_all_finite_f32`.

What a sabotage changes here: a widen that reads the wrong byte offset or
sign-extends wrongly (case 2), a class narrowed at the wrong width (cases
0-3), a code off by one or a class order other than ascending (every encode
case), a gather that stops writing at the first miss or folds the partials
by maximum (cases 4 and 5), a scan that calls -0.0 negative (case 6)."""
from std.testing import TestSuite
from std.memory import bitcast, unsafe_memcpy

from max.gpu.host import DeviceBuffer, DeviceContext

from core.cast_in import CAST_KIND_SIGNED
from core.labels_score_encode import score_encode_signed
from core.metrics_exp_impl import mexp_gather_i32
from core.host_finite_scan import host_all_finite_f32
from core.metric_labels_device import (
    metric_encode_resident,
    metric_gather_resident,
    metric_first_nonfinite_resident,
    metric_first_negative_resident,
)

from checks.native_tests.kit import download, upload
from checks.native_tests.testsuite_cases import require


comptime MLD_CASES = 7
comptime _BP = MutPointer[UInt8, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _encode_case[dt: DType](ctx: DeviceContext, values: List[Scalar[dt]], item: Int, name: String) raises:
    """`values` (item bytes each) encoded on the host and on the device."""
    var n = len(values)
    var src = Int(values.unsafe_ptr())
    # host column
    var h_classes = List[UInt8](length=n * item, fill=UInt8(0))
    var h_codes = List[Int32](length=n, fill=Int32(-7))
    var hk = score_encode_signed(src, item, n, Int(h_classes.unsafe_ptr()), Int(h_codes.unsafe_ptr()))
    # the labels as a store slot holds them: packed bytes in 4-byte words, the tail zero-padded
    var words = (n * item + 3) // 4
    var packed = List[Float32](length=words, fill=Float32(0))
    unsafe_memcpy(dest=_BP(unsafe_from_address=Int(packed.unsafe_ptr())), src=_BP(unsafe_from_address=src),
                  count=n * item)
    var d_raw = upload[DType.float32](ctx, packed)
    var fill_codes = List[Int32](length=n, fill=Int32(-7))
    var d_codes = upload[DType.int32](ctx, fill_codes)
    var d_classes = List[UInt8](length=n * item, fill=UInt8(0))
    var dk = metric_encode_resident(ctx, Int(d_raw.unsafe_ptr()), n, CAST_KIND_SIGNED, item,
                                    Int(d_codes.unsafe_ptr()), Int(d_classes.unsafe_ptr()))
    require(dk == hk, name + ": device k " + String(dk) + ", host k " + String(hk))
    for b in range(hk * item):  # check loop: the k classes' bytes
        require(d_classes[b] == h_classes[b], name + ": class byte " + String(b) + " differs (device "
                + String(Int(d_classes[b])) + ", host " + String(Int(h_classes[b])) + ")")
    var got = download[DType.int32](ctx, d_codes, n)
    for i in range(n):  # check loop: one code per row
        require(got[i] == h_codes[i], name + ": code of row " + String(i) + " device " + String(got[i])
                + ", host " + String(h_codes[i]))
    _ = d_raw^
    _ = d_codes^
    _ = packed^
    _ = values


def _gather_case(ctx: DeviceContext, table: List[Int32], codes: List[Int32], want_first: Int, name: String) raises:
    var n = len(codes)
    var nt = len(table)
    var h_dst = List[Int32](length=n, fill=Int32(-9))
    var hf = mexp_gather_i32(_I32P(unsafe_from_address=Int(table.unsafe_ptr())), nt,
                             _I32P(unsafe_from_address=Int(codes.unsafe_ptr())), n,
                             _I32P(unsafe_from_address=Int(h_dst.unsafe_ptr())))
    require(hf == want_first, name + ": host first miss " + String(hf) + ", expected " + String(want_first))
    var d_codes = upload[DType.int32](ctx, codes)
    var fill_dst = List[Int32](length=n, fill=Int32(-9))
    var d_dst = upload[DType.int32](ctx, fill_dst)
    var df = metric_gather_resident(ctx, Int(table.unsafe_ptr()), nt, Int(d_codes.unsafe_ptr()), n,
                                    Int(d_dst.unsafe_ptr()))
    require(df == hf, name + ": device first miss " + String(df) + ", host " + String(hf))
    var got = download[DType.int32](ctx, d_dst, n)
    for i in range(n):  # check loop: every row written
        require(got[i] == h_dst[i], name + ": row " + String(i) + " device " + String(got[i])
                + ", host " + String(h_dst[i]))
    _ = d_codes^
    _ = d_dst^


def _scan_case(ctx: DeviceContext) raises:
    var n = 70000
    var words = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # fixture
        words[i] = Float32(i % 97) * Float32(0.25) + Float32(1.0)
    var d = upload[DType.float32](ctx, words)
    require(host_all_finite_f32(_F32P(unsafe_from_address=Int(words.unsafe_ptr())), n),
            "scans: host says the clean fixture is not finite")
    require(metric_first_nonfinite_resident(ctx, Int(d.unsafe_ptr()), n) == -1, "scans: clean words reported non-finite")
    require(metric_first_negative_resident(ctx, Int(d.unsafe_ptr()), n) == -1, "scans: clean words reported negative")
    _ = d^
    words[66000] = bitcast[DType.float32](UInt32(0x7FC00000))
    words[69999] = bitcast[DType.float32](UInt32(0x7F800000))
    words[10] = bitcast[DType.float32](UInt32(0x80000000))  # -0.0: finite and not negative
    words[1234] = Float32(-1.0)
    var e = upload[DType.float32](ctx, words)
    require(not host_all_finite_f32(_F32P(unsafe_from_address=Int(words.unsafe_ptr())), n),
            "scans: host missed the planted NaN")
    var nf = metric_first_nonfinite_resident(ctx, Int(e.unsafe_ptr()), n)
    require(nf == 66000, "scans: first non-finite " + String(nf) + ", expected 66000")
    var ng = metric_first_negative_resident(ctx, Int(e.unsafe_ptr()), n)
    require(ng == 1234, "scans: first negative " + String(ng) + ", expected 1234 (-0.0 at 10 is not negative)")
    _ = e^
    _ = words^


def _case(ctx: DeviceContext, which: Int) raises:
    if which == 0:
        var v: List[Int32] = [3, 1, 3, 0, 2, 1, 5, 5]
        _encode_case[DType.int32](ctx, v, 4, "encode int32 dense")
    elif which == 1:
        var v: List[Int64] = [-5000000000000, 7, 7, 123456789012, -5000000000000, 0, -1, 7]
        _encode_case[DType.int64](ctx, v, 8, "encode int64 wide")
    elif which == 2:
        var v: List[Int8] = [-3, 5, -3, 127, -128, 0, 127]
        _encode_case[DType.int8](ctx, v, 1, "encode int8 ends")
    elif which == 3:
        var v = List[Int16](length=70000, fill=Int16(0))
        for i in range(70000):  # fixture
            v[i] = Int16((i * 7919) % 1000 - 500)
        _encode_case[DType.int16](ctx, v, 2, "encode int16 70k")
    elif which == 4:
        var table: List[Int32] = [2, -1, 0, 1]
        var codes = List[Int32](length=70000, fill=Int32(0))
        for i in range(70000):  # fixture: codes 0, 2, 3 only, then a miss late
            codes[i] = Int32(0 if i % 3 == 0 else (2 if i % 3 == 1 else 3))
        codes[66001] = 1  # mapped to -1
        codes[69998] = 9  # past the table
        _gather_case(ctx, table, codes, 66001, "gather with misses")
    elif which == 5:
        var table: List[Int32] = [4, 3, 2, 1, 0]
        var codes = List[Int32](length=70000, fill=Int32(0))
        for i in range(70000):  # fixture
            codes[i] = Int32(i % 5)
        _gather_case(ctx, table, codes, -1, "gather without misses")
    else:
        _scan_case(ctx)


def test_cases() raises:
    var ctx = DeviceContext()
    var failed = String("")
    for which in range(MLD_CASES):  # driver-loop: over the check's cases, one device queue
        try:
            _case(ctx, which)
        except problem:
            failed += "\n      metric_labels_device row " + String(which) + ": " + String(problem)
    if failed.byte_length() > 0:
        raise Error("metric_labels_device: cases failed" + failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
