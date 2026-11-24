# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`mojolearn.linalg` on training device arrays (lane vdf-linalg4, 2026-11-24; gemm/linalg_dev_arrays.mojo): for small
fixtures, the device-array entry (operands uploaded once into pool handles, the result read back from its handle)
gives the same bytes as the host-address entry of the GPU image (gemm/linalg_native.mojo) and as the host column the
CPU image runs (gemm/host/linalg_native.mojo). Identical builds only (the device arrays exist only there).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from core.train_dev_pool import (
    train_dev_alloc, train_dev_free, train_dev_put, train_dev_get, train_dev_put_bytes, train_dev_get_bytes,
    train_dev_words,
)
from gemm.linalg_dev_arrays import (
    gemm_dev_native, to_bf16_dev_native, from_bf16_dev_native, quantize_int8_dev_native, dequantize_int8_dev_native,
    gemm_bf16_dev_native, gemm_int8_dev_native, quantize_int15_dev_native, dequantize_int15_dev_native,
    gemm_int15_dev_native, from_f16_dev_native,
)
from gemm.linalg_native import gemm_native as gemm_native_gpu
from gemm.linalg_native import (
    quantize_int15_native as quantize_int15_gpu, dequantize_int15_native as dequantize_int15_gpu,
    gemm_int15_native as gemm_int15_gpu, from_f16_native as from_f16_gpu,
)
from gemm.host.linalg_native import (
    quantize_int15_native as quantize_int15_host, dequantize_int15_native as dequantize_int15_host,
    gemm_int15_native as gemm_int15_host, from_f16_native as from_f16_host,
)
from gemm.linalg_native import (
    to_bf16_native as to_bf16_gpu, from_bf16_native as from_bf16_gpu, quantize_int8_native as quantize_int8_gpu,
    dequantize_int8_native as dequantize_int8_gpu, gemm_bf16_native as gemm_bf16_gpu, gemm_int8_native as gemm_int8_gpu,
)
from gemm.host.linalg_native import gemm_native as gemm_native_host
from gemm.host.linalg_native import (
    to_bf16_native as to_bf16_host, from_bf16_native as from_bf16_host, quantize_int8_native as quantize_int8_host,
    dequantize_int8_native as dequantize_int8_host, gemm_bf16_native as gemm_bf16_host, gemm_int8_native as gemm_int8_host,
)
from gemm.contract import OP_NN, OP_NT, OP_TN
from std.testing import TestSuite
from std.sys import size_of

comptime _CTX = "MojolearnLinalgDeviceArraysCheckV1"
comptime _FP = MutPointer[Float32, MutUntrackedOrigin]


def _fill(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop: a small fixture
        v[i] = Float32(((i * 37 + salt * 11) % 67) - 33) / Float32(16.0)
    return v^


def _fp(v: List[Float32]) -> _FP:
    return _FP(unsafe_from_address=Int(v.unsafe_ptr()))


def _same(label: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error(label + ": lengths differ")
    for i in range(len(a)):  # small-loop: a small result
        if a[i].to_bits() != b[i].to_bits():
            raise Error(label + ": word " + String(i) + " differs")


def _put(ctx: DeviceContext, v: List[Float32]) raises -> Int:
    var h = train_dev_alloc(ctx, len(v))
    train_dev_put(ctx, h, _fp(v), len(v))
    return h


def _get(ctx: DeviceContext, h: Int, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0.0)
    train_dev_get(ctx, h, _fp(out), n)
    return out^


def _matmul_case(m: Int, n: Int, k: Int, op: Int) raises:
    var a = _fill(m * k, 1 + op)
    var b = _fill(n * k, 5 + op)
    var host_in = List[Float32](length=m * n, fill=0.0)
    var host_col = List[Float32](length=m * n, fill=0.0)
    _ = gemm_native_gpu(Int(host_in.unsafe_ptr()), Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), [m, n, k, op])
    _ = gemm_native_host(Int(host_col.unsafe_ptr()), Int(a.unsafe_ptr()), Int(b.unsafe_ptr()), [m, n, k, op])
    var ctx = process_ctx[_CTX]()
    var ha = _put(ctx, a)
    var hb = _put(ctx, b)
    var hc = train_dev_alloc(ctx, m * n)
    _ = gemm_dev_native(hc, ha, hb, [m, n, k, op])
    var dev = _get(ctx, hc, m * n)
    var label = "matmul m=" + String(m) + " n=" + String(n) + " k=" + String(k) + " op=" + String(op)
    _same(label + " device == host-input", dev, host_in)
    _same(label + " device == host column", dev, host_col)
    train_dev_free(ctx, ha)
    train_dev_free(ctx, hb)
    train_dev_free(ctx, hc)


def test_matmul_device_arrays() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        return
    # NN, NT and TN; a square, a tall and a ragged shape (every op's operand layout is the row-major one of
    # matmul's table: NN a (m, k) b (k, n); NT b (n, k); TN a (k, m)).
    _matmul_case(5, 7, 3, OP_NN)
    _matmul_case(9, 4, 11, OP_NT)
    _matmul_case(6, 6, 6, OP_TN)
    _matmul_case(33, 17, 65, OP_NN)


def _put_typed[T: DType](ctx: DeviceContext, v: List[Scalar[T]]) raises -> Int:
    var nbytes = len(v) * size_of[T]()
    var h = train_dev_alloc(ctx, train_dev_words(len(v), size_of[T]()))
    train_dev_put_bytes(ctx, h, Int(v.unsafe_ptr()), nbytes)
    return h


def _get_typed[T: DType](ctx: DeviceContext, h: Int, n: Int) raises -> List[Scalar[T]]:
    var out = List[Scalar[T]](length=n, fill=Scalar[T](0))
    train_dev_get_bytes(ctx, h, Int(out.unsafe_ptr()), n * size_of[T]())
    return out^


def _same_typed[T: DType](label: String, a: List[Scalar[T]], b: List[Scalar[T]]) raises:
    if len(a) != len(b):
        raise Error(label + ": lengths differ")
    for i in range(len(a)):  # small-loop: a small result
        if a[i] != b[i]:
            raise Error(label + ": element " + String(i) + " differs")


def test_bf16_conversions_device_arrays() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        return
    var n = 77
    var x = _fill(n, 3)
    var bits_gpu = List[UInt16](length=n, fill=0)
    var bits_host = List[UInt16](length=n, fill=0)
    _ = to_bf16_gpu(Int(bits_gpu.unsafe_ptr()), Int(x.unsafe_ptr()), [n])
    _ = to_bf16_host(Int(bits_host.unsafe_ptr()), Int(x.unsafe_ptr()), [n])
    var ctx = process_ctx[_CTX]()
    var hx = _put(ctx, x)
    var hbits = train_dev_alloc(ctx, train_dev_words(n, 2))
    _ = to_bf16_dev_native(hbits, hx, [n])
    var bits_dev = _get_typed[DType.uint16](ctx, hbits, n)
    _same_typed[DType.uint16]("to_bf16 device == host-input", bits_dev, bits_gpu)
    _same_typed[DType.uint16]("to_bf16 device == host column", bits_dev, bits_host)
    var wide_gpu = List[Float32](length=n, fill=0.0)
    var wide_host = List[Float32](length=n, fill=0.0)
    _ = from_bf16_gpu(Int(wide_gpu.unsafe_ptr()), Int(bits_gpu.unsafe_ptr()), [n])
    _ = from_bf16_host(Int(wide_host.unsafe_ptr()), Int(bits_gpu.unsafe_ptr()), [n])
    var hw = train_dev_alloc(ctx, n)
    _ = from_bf16_dev_native(hw, hbits, [n])
    var wide_dev = _get(ctx, hw, n)
    _same("from_bf16 device == host-input", wide_dev, wide_gpu)
    _same("from_bf16 device == host column", wide_dev, wide_host)
    # matmul_bf16: a float32 (and a as bits), b bits, OP_NT
    var m = 6
    var k = n // 7
    var nn = 7
    var a = _fill(m * k, 9)
    var c_gpu = List[Float32](length=m * nn, fill=0.0)
    var c_host = List[Float32](length=m * nn, fill=0.0)
    _ = gemm_bf16_gpu(Int(c_gpu.unsafe_ptr()), Int(a.unsafe_ptr()), Int(bits_gpu.unsafe_ptr()), [m, nn, k, OP_NT, 0])
    _ = gemm_bf16_host(Int(c_host.unsafe_ptr()), Int(a.unsafe_ptr()), Int(bits_gpu.unsafe_ptr()), [m, nn, k, OP_NT, 0])
    var ha = _put(ctx, a)
    var hc = train_dev_alloc(ctx, m * nn)
    _ = gemm_bf16_dev_native(hc, ha, hbits, [m, nn, k, OP_NT, 0])
    var c_dev = _get(ctx, hc, m * nn)
    _same("matmul_bf16 device == host-input", c_dev, c_gpu)
    _same("matmul_bf16 device == host column", c_dev, c_host)
    for h in [hx, hbits, hw, ha, hc]:  # small-loop: the fixture's handles
        train_dev_free(ctx, h)


def test_int8_device_arrays() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        return
    var m = 5
    var n = 4
    var k = 19
    var a = _fill(m * k, 21)
    var b = _fill(n * k, 22)
    var qa_g = List[Int8](length=m * k, fill=0)
    var ea_g = List[Int32](length=m, fill=0)
    var qa_h = List[Int8](length=m * k, fill=0)
    var ea_h = List[Int32](length=m, fill=0)
    var qb_g = List[Int8](length=n * k, fill=0)
    var eb_g = List[Int32](length=n, fill=0)
    _ = quantize_int8_gpu(Int(qa_g.unsafe_ptr()), Int(ea_g.unsafe_ptr()), Int(a.unsafe_ptr()), [m, k])
    _ = quantize_int8_host(Int(qa_h.unsafe_ptr()), Int(ea_h.unsafe_ptr()), Int(a.unsafe_ptr()), [m, k])
    _ = quantize_int8_gpu(Int(qb_g.unsafe_ptr()), Int(eb_g.unsafe_ptr()), Int(b.unsafe_ptr()), [n, k])
    var ctx = process_ctx[_CTX]()
    var ha = _put(ctx, a)
    var hb = _put(ctx, b)
    var hqa = train_dev_alloc(ctx, train_dev_words(m * k, 1))
    var hea = train_dev_alloc(ctx, m)
    var hqb = train_dev_alloc(ctx, train_dev_words(n * k, 1))
    var heb = train_dev_alloc(ctx, n)
    _ = quantize_int8_dev_native(hqa, hea, ha, [m, k])
    _ = quantize_int8_dev_native(hqb, heb, hb, [n, k])
    var qa_d = _get_typed[DType.int8](ctx, hqa, m * k)
    var ea_d = _get_typed[DType.int32](ctx, hea, m)
    _same_typed[DType.int8]("quantize_int8 codes device == host-input", qa_d, qa_g)
    _same_typed[DType.int8]("quantize_int8 codes device == host column", qa_d, qa_h)
    _same_typed[DType.int32]("quantize_int8 exponents device == host-input", ea_d, ea_g)
    _same_typed[DType.int32]("quantize_int8 exponents device == host column", ea_d, ea_h)
    var y_g = List[Float32](length=m * k, fill=0.0)
    var y_h = List[Float32](length=m * k, fill=0.0)
    _ = dequantize_int8_gpu(Int(y_g.unsafe_ptr()), Int(qa_g.unsafe_ptr()), Int(ea_g.unsafe_ptr()), [m, k])
    _ = dequantize_int8_host(Int(y_h.unsafe_ptr()), Int(qa_g.unsafe_ptr()), Int(ea_g.unsafe_ptr()), [m, k])
    var hy = train_dev_alloc(ctx, m * k)
    _ = dequantize_int8_dev_native(hy, hqa, hea, [m, k])
    var y_d = _get(ctx, hy, m * k)
    _same("dequantize_int8 device == host-input", y_d, y_g)
    _same("dequantize_int8 device == host column", y_d, y_h)
    var c_g = List[Float32](length=m * n, fill=0.0)
    var c_h = List[Float32](length=m * n, fill=0.0)
    _ = gemm_int8_gpu(Int(c_g.unsafe_ptr()), Int(qa_g.unsafe_ptr()), Int(ea_g.unsafe_ptr()), Int(qb_g.unsafe_ptr()),
                      Int(eb_g.unsafe_ptr()), [m, n, k])
    _ = gemm_int8_host(Int(c_h.unsafe_ptr()), Int(qa_g.unsafe_ptr()), Int(ea_g.unsafe_ptr()), Int(qb_g.unsafe_ptr()),
                       Int(eb_g.unsafe_ptr()), [m, n, k])
    var hc = train_dev_alloc(ctx, m * n)
    _ = gemm_int8_dev_native(hc, hqa, hea, hqb, heb, [m, n, k])
    var c_d = _get(ctx, hc, m * n)
    _same("matmul_int8 device == host-input", c_d, c_g)
    _same("matmul_int8 device == host column", c_d, c_h)
    # a typed upload and download move exactly the elements' bytes (an odd int8 count: the word padding never moves)
    var hq = _put_typed[DType.int8](ctx, qa_g)
    _same_typed[DType.int8]("int8 bytes round trip", _get_typed[DType.int8](ctx, hq, m * k), qa_g)
    for h in [ha, hb, hqa, hea, hqb, heb, hy, hc, hq]:  # small-loop: the fixture's handles
        train_dev_free(ctx, h)


def _a(v: List[Int8]) -> Int:
    return Int(v.unsafe_ptr())


def _ai(v: List[Int32]) -> Int:
    return Int(v.unsafe_ptr())


def test_int15_device_arrays() raises:
    """lane vdf-linalg5: quantize_int15, dequantize_int15 and matmul_int15 on pool handles == the host-address GPU
    entry == the host column."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        return
    var m = 6
    var n = 5
    var k = 23
    var a = _fill(m * k, 31)
    var b = _fill(n * k, 32)
    var ah_g = List[Int8](length=m * k, fill=0)
    var al_g = List[Int8](length=m * k, fill=0)
    var ea_g = List[Int32](length=m, fill=0)
    var ah_h = List[Int8](length=m * k, fill=0)
    var al_h = List[Int8](length=m * k, fill=0)
    var ea_h = List[Int32](length=m, fill=0)
    var bh_g = List[Int8](length=n * k, fill=0)
    var bl_g = List[Int8](length=n * k, fill=0)
    var eb_g = List[Int32](length=n, fill=0)
    _ = quantize_int15_gpu([_a(ah_g), _a(al_g), _ai(ea_g), Int(a.unsafe_ptr())], [m, k])
    _ = quantize_int15_host([_a(ah_h), _a(al_h), _ai(ea_h), Int(a.unsafe_ptr())], [m, k])
    _ = quantize_int15_gpu([_a(bh_g), _a(bl_g), _ai(eb_g), Int(b.unsafe_ptr())], [n, k])
    var ctx = process_ctx[_CTX]()
    var ha = _put(ctx, a)
    var hb = _put(ctx, b)
    var hah = train_dev_alloc(ctx, train_dev_words(m * k, 1))
    var hal = train_dev_alloc(ctx, train_dev_words(m * k, 1))
    var hea = train_dev_alloc(ctx, m)
    var hbh = train_dev_alloc(ctx, train_dev_words(n * k, 1))
    var hbl = train_dev_alloc(ctx, train_dev_words(n * k, 1))
    var heb = train_dev_alloc(ctx, n)
    _ = quantize_int15_dev_native([hah, hal, hea, ha], [m, k])
    _ = quantize_int15_dev_native([hbh, hbl, heb, hb], [n, k])
    var ah_d = _get_typed[DType.int8](ctx, hah, m * k)
    var al_d = _get_typed[DType.int8](ctx, hal, m * k)
    var ea_d = _get_typed[DType.int32](ctx, hea, m)
    _same_typed[DType.int8]("quantize_int15 hi device == host-input", ah_d, ah_g)
    _same_typed[DType.int8]("quantize_int15 hi device == host column", ah_d, ah_h)
    _same_typed[DType.int8]("quantize_int15 lo device == host-input", al_d, al_g)
    _same_typed[DType.int8]("quantize_int15 lo device == host column", al_d, al_h)
    _same_typed[DType.int32]("quantize_int15 exponents device == host-input", ea_d, ea_g)
    _same_typed[DType.int32]("quantize_int15 exponents device == host column", ea_d, ea_h)
    var y_g = List[Float32](length=m * k, fill=0.0)
    var y_h = List[Float32](length=m * k, fill=0.0)
    _ = dequantize_int15_gpu([Int(y_g.unsafe_ptr()), _a(ah_g), _a(al_g), _ai(ea_g)], [m, k])
    _ = dequantize_int15_host([Int(y_h.unsafe_ptr()), _a(ah_g), _a(al_g), _ai(ea_g)], [m, k])
    var hy = train_dev_alloc(ctx, m * k)
    _ = dequantize_int15_dev_native([hy, hah, hal, hea], [m, k])
    var y_d = _get(ctx, hy, m * k)
    _same("dequantize_int15 device == host-input", y_d, y_g)
    _same("dequantize_int15 device == host column", y_d, y_h)
    var c_g = List[Float32](length=m * n, fill=0.0)
    var c_h = List[Float32](length=m * n, fill=0.0)
    _ = gemm_int15_gpu([Int(c_g.unsafe_ptr()), _a(ah_g), _a(al_g), _ai(ea_g), _a(bh_g), _a(bl_g), _ai(eb_g)], [m, n, k])
    _ = gemm_int15_host([Int(c_h.unsafe_ptr()), _a(ah_g), _a(al_g), _ai(ea_g), _a(bh_g), _a(bl_g), _ai(eb_g)], [m, n, k])
    var hc = train_dev_alloc(ctx, m * n)
    _ = gemm_int15_dev_native([hc, hah, hal, hea, hbh, hbl, heb], [m, n, k])
    var c_d = _get(ctx, hc, m * n)
    _same("matmul_int15 device == host-input", c_d, c_g)
    _same("matmul_int15 device == host column", c_d, c_h)
    for h in [ha, hb, hah, hal, hea, hbh, hbl, heb, hy, hc]:  # small-loop: the fixture's handles
        train_dev_free(ctx, h)


def test_from_f16_device_arrays() raises:
    """lane vdf-linalg5: from_f16 on pool handles == the host-address GPU entry == the host column (normals,
    subnormals, zeros, infinities and a NaN among the bit patterns)."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        return
    var n = 41
    var bits = List[UInt16](length=n, fill=0)
    for i in range(n):  # small-loop: a small fixture
        bits[i] = UInt16((i * 1597 + 7) % 65536)
    bits[0] = 0x0000
    bits[1] = 0x8000
    bits[2] = 0x7C00
    bits[3] = 0xFC00
    bits[4] = 0x7E00
    bits[5] = 0x0001
    var w_g = List[Float32](length=n, fill=0.0)
    var w_h = List[Float32](length=n, fill=0.0)
    _ = from_f16_gpu(Int(w_g.unsafe_ptr()), Int(bits.unsafe_ptr()), [n])
    _ = from_f16_host(Int(w_h.unsafe_ptr()), Int(bits.unsafe_ptr()), [n])
    var ctx = process_ctx[_CTX]()
    var hbits = _put_typed[DType.uint16](ctx, bits)
    var hw = train_dev_alloc(ctx, n)
    _ = from_f16_dev_native(hw, hbits, [n])
    var w_d = _get(ctx, hw, n)
    _same("from_f16 device == host-input", w_d, w_g)
    _same("from_f16 device == host column", w_d, w_h)
    train_dev_free(ctx, hbits)
    train_dev_free(ctx, hw)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
