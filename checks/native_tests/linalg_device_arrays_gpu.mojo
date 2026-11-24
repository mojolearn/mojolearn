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
from core.train_dev_pool import train_dev_alloc, train_dev_free, train_dev_put, train_dev_get
from gemm.linalg_dev_arrays import gemm_dev_native
from gemm.linalg_native import gemm_native as gemm_native_gpu
from gemm.host.linalg_native import gemm_native as gemm_native_host
from gemm.contract import OP_NN, OP_NT, OP_TN
from std.testing import TestSuite

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
