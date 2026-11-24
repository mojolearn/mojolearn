# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`training.rms_norm_forward/backward` and `embedding_forward/backward` on training device arrays (lane vdf-linalg2,
2026-11-24; training/dev_tensors_layers.mojo): for small fixtures, the device-array form (inputs uploaded once into
handles, results read back from handles) gives the same bytes as the host-address entry of the same image
(training/samba_ops.mojo `samba_*_host`) and as the host column (training/host/samba_ops_oracle.mojo).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from max.gpu.host import DeviceContext
from core.neural_context import neural_ctx
from training.dev_tensors import train_dev_alloc, train_dev_free, train_dev_put, train_dev_get
from training.dev_tensors_layers import (
    train_rms_norm_forward_dev, train_rms_norm_backward_dev, train_embedding_forward_dev, train_embedding_backward_dev,
)
from training.samba_ops import (
    samba_rms_norm_forward_host, samba_rms_norm_backward_host, samba_embedding_forward_host,
    samba_embedding_backward_host,
)
from training.host.samba_ops_oracle import (
    host_samba_rms_norm_forward, host_samba_rms_norm_backward, host_samba_embedding_forward,
    host_samba_embedding_backward,
)
from std.testing import TestSuite

comptime _CTX = "MojolearnTrainingLayersDeviceArraysCheckV1"
comptime _FP = MutPointer[Float32, MutUntrackedOrigin]
comptime _IP = MutPointer[Int32, MutUntrackedOrigin]


def _fill(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop: a small fixture
        v[i] = Float32(((i * 29 + salt * 13) % 61) - 30) / Float32(32.0)
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



def test_rms_norm_device_arrays() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var ctx = neural_ctx[_CTX]()
    var m = 7
    var dm = 12
    var eps = Float32(1.0e-5)
    var x = _fill(m * dm, 1)
    var w = _fill(dm, 2)
    var dy = _fill(m * dm, 3)
    # forward
    var y_host = List[Float32](length=m * dm, fill=0.0)
    _ = samba_rms_norm_forward_host(ctx, _fp(y_host), _fp(x), _fp(w), m, dm, eps)
    var hx = _put(ctx, x)
    var hw = _put(ctx, w)
    var hy = train_dev_alloc(ctx, m * dm)
    _ = train_rms_norm_forward_dev(ctx, hy, hx, hw, m, dm, eps)
    _same("rms forward device array vs host input", _get(ctx, hy, m * dm), y_host)
    _same("rms forward device vs host column", y_host, host_samba_rms_norm_forward(x, w, m, dm, eps))
    # backward
    var dx_host = List[Float32](length=m * dm, fill=0.0)
    var dw_host = List[Float32](length=dm, fill=0.0)
    _ = samba_rms_norm_backward_host(ctx, _fp(dx_host), _fp(dw_host), _fp(dy), _fp(x), _fp(w), m, dm, eps)
    var hdy = _put(ctx, dy)
    var hdx = train_dev_alloc(ctx, m * dm)
    var hdw = train_dev_alloc(ctx, dm)
    _ = train_rms_norm_backward_dev(ctx, hdx, hdw, hdy, hx, hw, m, dm, eps)
    _same("rms backward dx device array vs host input", _get(ctx, hdx, m * dm), dx_host)
    _same("rms backward dw device array vs host input", _get(ctx, hdw, dm), dw_host)
    var oracle = host_samba_rms_norm_backward(dy, x, w, m, dm, eps)
    _same("rms backward dx device vs host column", dx_host, oracle[0])
    _same("rms backward dw device vs host column", dw_host, oracle[1])
    for h in [hx, hw, hy, hdy, hdx, hdw]:  # small-loop: the handles above
        train_dev_free(ctx, h)


def test_embedding_device_arrays() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var ctx = neural_ctx[_CTX]()
    var n = 9
    var vocab = 5
    var width = 6
    var w = _fill(vocab * width, 4)
    var dy = _fill(n * width, 5)
    var ids: List[Int32] = [3, 0, 4, 4, 1, 2, 0, 3, 4]
    var ip = _IP(unsafe_from_address=Int(ids.unsafe_ptr()))
    var y_host = List[Float32](length=n * width, fill=0.0)
    _ = samba_embedding_forward_host(ctx, _fp(y_host), _fp(w), ip, n, vocab, width)
    var hw = _put(ctx, w)
    var hy = train_dev_alloc(ctx, n * width)
    _ = train_embedding_forward_dev(ctx, hy, hw, ip, n, vocab, width)
    _same("embedding forward device array vs host input", _get(ctx, hy, n * width), y_host)
    _same("embedding forward device vs host column", y_host, host_samba_embedding_forward(w, ids, n, vocab, width))
    var dw_host = List[Float32](length=vocab * width, fill=0.0)
    _ = samba_embedding_backward_host(ctx, _fp(dw_host), _fp(dy), ip, n, vocab, width)
    var hdy = _put(ctx, dy)
    var hdw = train_dev_alloc(ctx, vocab * width)
    _ = train_embedding_backward_dev(ctx, hdw, hdy, ip, n, vocab, width)
    _same("embedding backward device array vs host input", _get(ctx, hdw, vocab * width), dw_host)
    _same("embedding backward device vs host column", dw_host,
          host_samba_embedding_backward(dy, ids, n, vocab, width))
    _ = ids
    for h in [hw, hy, hdy, hdw]:  # small-loop: the handles above
        train_dev_free(ctx, h)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
