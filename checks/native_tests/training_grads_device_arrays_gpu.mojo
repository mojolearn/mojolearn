# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`training.chunked_lm_head_loss`, `clip_grad_norm_`, `accumulate_grads` and `mlp_inference_sessions` on training
device arrays (lane vdf-linalg3, 2026-11-24; training/dev_tensors_layers.mojo): for small fixtures, the device-array
form (inputs uploaded once into handles, results read back from handles) gives the same bytes as the host-address
entry of the same image and, for the first three, as the host column (core/mojolearn_training_host_impl.mojo).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from max.gpu.host import DeviceContext
from core.neural_context import neural_ctx
from training.dev_tensors import train_dev_alloc, train_dev_free, train_dev_put, train_dev_get
from training.dev_tensors_layers import (
    train_chunked_lm_head_dev, train_clip_dev, train_accumulate_dev, train_mlp_sessions_dev,
)
from training.chunked_lm_head_v2 import chunked_lm_head_v2_train_host
from training.estimator import identical_clip_grad_norm_addrs_host
from training.samba_ops import samba_accumulate_host
from training.neural_session_mlp import nn_mlp_sessions_device
from core.mojolearn_training_host_impl import (
    chunked_lm_head_v2_train_host_into, clip_grad_norm_host_into, accumulate_host_into,
)
from std.testing import TestSuite

comptime _CTX = "MojolearnTrainingGradsDeviceArraysCheckV1"
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


def test_chunked_lm_head_device_arrays() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var ctx = neural_ctx[_CTX]()
    var rows = 6
    var vocab = 37
    var width = 8
    var hidden = _fill(rows * width, 1)
    var weight = _fill(vocab * width, 2)
    var targets: List[Int32] = [3, 0, 36, 17, 5, 29]
    var tp = _IP(unsafe_from_address=Int(targets.unsafe_ptr()))
    var loss_h = List[Float32](length=1, fill=0.0)
    var mx = List[Float32](length=rows, fill=0.0)
    var dn = List[Float32](length=rows, fill=0.0)
    var dh_h = List[Float32](length=rows * width, fill=0.0)
    var dw_h = List[Float32](length=vocab * width, fill=0.0)
    _ = chunked_lm_head_v2_train_host(ctx, _fp(loss_h), _fp(mx), _fp(dn), _fp(dh_h), _fp(dw_h), _fp(hidden),
                                      _fp(weight), tp, rows, vocab, width)
    var hh = _put(ctx, hidden)
    var hw = _put(ctx, weight)
    var hdh = train_dev_alloc(ctx, rows * width)
    var hdw = train_dev_alloc(ctx, vocab * width)
    var loss_d = List[Float32](length=1, fill=0.0)
    _ = train_chunked_lm_head_dev(ctx, _fp(loss_d), hdh, hdw, hh, hw, tp, rows, vocab, width, True)
    _same("lm head loss device array vs host input", loss_d, loss_h)
    _same("lm head d_hidden device array vs host input", _get(ctx, hdh, rows * width), dh_h)
    _same("lm head d_weight device array vs host input", _get(ctx, hdw, vocab * width), dw_h)
    var loss_only = List[Float32](length=1, fill=0.0)
    _ = train_chunked_lm_head_dev(ctx, _fp(loss_only), -1, -1, hh, hw, tp, rows, vocab, width, False)
    _same("lm head loss without gradient, device array vs host input", loss_only, loss_h)
    # the host column
    var loss_c = List[Float32](length=1, fill=0.0)
    var mx_c = List[Float32](length=rows, fill=0.0)
    var dn_c = List[Float32](length=rows, fill=0.0)
    var dh_c = List[Float32](length=rows * width, fill=0.0)
    var dw_c = List[Float32](length=vocab * width, fill=0.0)
    var a: List[Int] = [Int(loss_c.unsafe_ptr()), Int(mx_c.unsafe_ptr()), Int(dn_c.unsafe_ptr()),
                        Int(dh_c.unsafe_ptr()), Int(dw_c.unsafe_ptr()), Int(hidden.unsafe_ptr()),
                        Int(weight.unsafe_ptr()), Int(targets.unsafe_ptr())]
    _ = chunked_lm_head_v2_train_host_into(a, rows, vocab, width)
    _same("lm head loss device vs host column", loss_h, loss_c)
    _same("lm head d_hidden device vs host column", dh_h, dh_c)
    _same("lm head d_weight device vs host column", dw_h, dw_c)
    _ = targets
    for h in [hh, hw, hdh, hdw]:  # small-loop: the handles above
        train_dev_free(ctx, h)


def test_clip_device_arrays() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var ctx = neural_ctx[_CTX]()
    var g1 = _fill(40, 3)
    var g2 = _fill(9, 4)
    var max_norm = Float32(0.5)
    # the host-address entry, on copies
    var c1 = g1.copy()
    var c2 = g2.copy()
    var offsets: List[Int32] = [0, 40, 49]
    var op = offsets.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var info_h = List[Float32](length=2, fill=0.0)
    var addrs: List[Int] = [Int(c1.unsafe_ptr()), Int(c2.unsafe_ptr())]
    _ = identical_clip_grad_norm_addrs_host(ctx, addrs, op, _fp(info_h), 2, max_norm)
    # the device arrays
    var h1 = _put(ctx, g1)
    var h2 = _put(ctx, g2)
    var info_d = List[Float32](length=2, fill=0.0)
    var handles: List[Int] = [h1, h2]
    var sizes: List[Int] = [40, 9]
    _ = train_clip_dev(ctx, handles, sizes, _fp(info_d), max_norm)
    _same("clip info device array vs host input", info_d, info_h)
    _same("clip tensor 0 device array vs host input", _get(ctx, h1, 40), c1)
    _same("clip tensor 1 device array vs host input", _get(ctx, h2, 9), c2)
    # the host column, on the packed copy
    var packed = g1.copy()
    for i in range(len(g2)):  # small-loop: a small fixture
        packed.append(g2[i])
    var info_c = List[Float32](length=2, fill=0.0)
    _ = clip_grad_norm_host_into(_fp(packed), op, _fp(info_c), 2, max_norm)
    _same("clip info device vs host column", info_h, info_c)
    var flat = c1.copy()
    for i in range(len(c2)):  # small-loop: a small fixture
        flat.append(c2[i])
    _same("clip tensors device vs host column", flat, packed)
    _ = offsets
    for h in [h1, h2]:  # small-loop: the handles above
        train_dev_free(ctx, h)


def test_accumulate_device_arrays() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var ctx = neural_ctx[_CTX]()
    var a = 4
    var sizes: List[Int] = [10, 3]
    var n = 13
    var parts = List[Float32]()
    var ins = List[Int]()
    for i in range(a):  # small-loop: the microbatches of a small fixture
        var t0 = _fill(10, 10 + i)
        var t1 = _fill(3, 20 + i)
        for v in t0:  # small-loop: a small fixture
            parts.append(v)
        for v in t1:  # small-loop: a small fixture
            parts.append(v)
        ins.append(_put(ctx, t0))
        ins.append(_put(ctx, t1))
    var out_h = List[Float32](length=n, fill=0.0)
    _ = samba_accumulate_host(ctx, _fp(out_h), _fp(parts), n, a, -1)
    var outs: List[Int] = [train_dev_alloc(ctx, 10), train_dev_alloc(ctx, 3)]
    _ = train_accumulate_dev(ctx, outs, ins, sizes, a, -1)
    var got = _get(ctx, outs[0], 10)
    for v in _get(ctx, outs[1], 3):  # small-loop: a small result
        got.append(v)
    _same("accumulate device array vs host input", got, out_h)
    var out_c = List[Float32](length=n, fill=0.0)
    var words: List[Int] = [Int(out_c.unsafe_ptr()), Int(parts.unsafe_ptr())]
    _ = accumulate_host_into(words, n, a, -1)
    _same("accumulate device vs host column", out_h, out_c)
    for h in ins:  # small-loop: the handles above
        train_dev_free(ctx, h)
    for h in outs:  # small-loop: the handles above
        train_dev_free(ctx, h)


def test_mlp_sessions_device_arrays() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var ctx = neural_ctx[_CTX]()
    var width = 5
    var hidden = 7
    var out_width = 3
    var rows: List[Int] = [4, 1, 6]
    var w1 = _fill(hidden * width, 5)
    var b1 = _fill(hidden, 6)
    var w2 = _fill(out_width * hidden, 7)
    var b2 = _fill(out_width, 8)
    var s0 = _fill(4 * width, 9)
    var s1 = _fill(1 * width, 10)
    var s2 = _fill(6 * width, 11)
    var total = 11
    var y_h = List[Float32](length=total * out_width, fill=0.0)
    var addrs: List[Int] = [Int(s0.unsafe_ptr()), Int(s1.unsafe_ptr()), Int(s2.unsafe_ptr())]
    _ = nn_mlp_sessions_device(ctx, addrs, rows, _fp(w1), _fp(b1), _fp(w2), _fp(b2), _fp(y_h), width, hidden,
                               out_width)
    var hs: List[Int] = [_put(ctx, s0), _put(ctx, s1), _put(ctx, s2)]
    var hw1 = _put(ctx, w1)
    var hb1 = _put(ctx, b1)
    var hw2 = _put(ctx, w2)
    var hb2 = _put(ctx, b2)
    var hy = train_dev_alloc(ctx, total * out_width)
    _ = train_mlp_sessions_dev(ctx, hy, hs, rows, hw1, hb1, hw2, hb2, width, hidden, out_width)
    _same("mlp sessions device array vs host input", _get(ctx, hy, total * out_width), y_h)
    for h in hs:  # small-loop: the handles above
        train_dev_free(ctx, h)
    for h in [hw1, hb1, hw2, hb2, hy]:  # small-loop: the handles above
        train_dev_free(ctx, h)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
