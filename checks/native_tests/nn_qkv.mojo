# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Attention over explicit q, k and v: `Session.attention` (lane nn-blocks-qkv, 2026-11-25; session code 122,
autograd/nn_qkv.mojo, the row body autograd/nn_qkv_ops.mojo), through the call the Python door makes
(`OpsSession.op(OPS_ATTENTION_QKV, [q, k, v, (mask)], [causal, scale given, mask kind], [scale])`).

test_nn_qkv_cases, per case:
  - the output against a float64 reference written here (PyTorch's scaled_dot_product_attention with
    enable_gqa=True: per (b, h, i), softmax over the keys left in of scale * q.k + additive mask, times v; a
    row with no key left is zeros), |a - e| <= 2e-5 + 2e-5 |e|;
  - the gradients of q, k and v of the loss sum(seed * out) (seed a fixed pattern) from the recording session's
    backward against central finite differences of the float64 reference's same loss (h = 1e-3, float64, so
    the differences are the true gradient to about 1e-6), |g - fd| <= 2e-3 (1 + |fd|), on EVERY word;
  - the inference session's output equals the recording session's word for word (recording changes no value);
  - a case with a fully masked query row: that output row is exactly +0.0 and q's gradient there exactly 0,
    every word finite (PyTorch gives NaN there; Mojolearn's documented result is zeros);
  - the device column runs every case on DeviceExec AND HostExec and requires the same words for the output
    and the three gradients.
  Cases: 0 self-attention; 1 causal; 2 cross-attention (Lq 3, Lk 5, dv 2) with an additive float mask
  broadcast over heads and queries, one -inf word; 3 grouped heads (4 query heads over 2 key/value heads) with a
  boolean (ids) mask (Lq, Lk) broadcast over batch and heads, query row 1 fully masked; 4 inputs without a heads
  axis, scale 0.5, a boolean mask with one fully masked row; 5 causal cross-attention, Lq 3 < Lk 5 (the lower
  triangle aligned at the top left: query 0 sees key 0 only); 6 case 2 with the float mask trainable: its
  gradient (summed over the axes it broadcasts along) against central differences of the float64 loss along
  every mask word (the -inf word's is 0).
  Every case also runs the seeded backward (`vjp` with the seed as v: ImportRules, the path of backward_seed,
  vjp and the custom-backward wrapper) and requires the scalar backward's words for every gradient.
test_nn_qkv_refusals: causal with a mask, a feature size mismatch, query heads not a multiple of the key/value
  heads, a mask that does not broadcast, q of rank 2; each refused with its words.
Fault injection (tools/identity_lanes/coverage_sabotage.tsv, lane nn-blocks-qkv's two arms): the causal mask one
off on the device only makes cases 1 and 5 FAIL on the device column; a fully masked row of NaN on the host
only makes cases 3 and 4 FAIL on the CPU column.
Serial by nature: each case opens sessions through the process-wide session table (autograd/ops_api.mojo
`_OPS_SESSIONS`, unlocked lists), as core_ops_session_blocks says.
"""
from std.math import exp, sqrt, isfinite
from std.memory import bitcast
from std.testing import TestSuite
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE
from autograd.nn_qkv import OPS_ATTENTION_QKV
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_SIGNED
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_higher_common import tensor, total, values, gradient, same_words, op2
from checks.native_tests.core_ops_common import normal
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, check_cases_serial, require

comptime NN_QKV_CASES = 7
comptime _MUL = 4
comptime _FD_H = Float64(1.0e-3)


struct _Case(Movable):
    var name: String
    var B: Int
    var H: Int
    var Hkv: Int
    var Lq: Int
    var Lk: Int
    var d: Int
    var dv: Int
    var rank3: Bool
    var causal: Bool
    var kind: Int              # 0 no mask, 1 boolean (ids words), 2 additive (float values)
    var mdims: List[Int]
    var mvals: List[Float32]
    var scale_given: Bool
    var scale: Float64
    var dead_row: Int          # a query row every head and batch row leaves fully masked, -1 none
    var mask_grad: Bool        # the float mask is a trainable input (its gradient checked too)

    def __init__(out self, name: String, B: Int, H: Int, Hkv: Int, Lq: Int, Lk: Int, d: Int, dv: Int):
        self.name = name
        self.B = B
        self.H = H
        self.Hkv = Hkv
        self.Lq = Lq
        self.Lk = Lk
        self.d = d
        self.dv = dv
        self.rank3 = False
        self.causal = False
        self.kind = 0
        self.mdims = List[Int]()
        self.mvals = List[Float32]()
        self.scale_given = False
        self.scale = 0.0
        self.dead_row = -1
        self.mask_grad = False

    def dims(self, heads: Int, length: Int, width: Int) -> List[Int]:
        var out = List[Int]()
        out.append(self.B)
        if not self.rank3:
            out.append(heads)
        out.append(length)
        out.append(width)
        return out^


def _neg_inf32() -> Float32:
    return bitcast[DType.float32](UInt32(0xFF800000))


def _case(which: Int) -> _Case:
    if which == 0:
        return _Case("self", 2, 2, 2, 4, 4, 3, 3)
    if which == 1:
        var c = _Case("causal", 2, 2, 2, 4, 4, 3, 3)
        c.causal = True
        return c^
    if which == 2 or which == 6:
        var c = _Case("cross-additive" if which == 2 else "cross-additive-mask-grad", 2, 2, 2, 3, 5, 4, 2)
        c.mask_grad = which == 6
        c.kind = 2
        var md: List[Int] = [2, 1, 1, 5]
        c.mdims = md^
        var m = normal(10, 71)
        for i in range(10):
            m[i] = m[i] * Float32(0.5)
        m[9] = _neg_inf32()          # batch row 1 never sees key 4
        c.mvals = m^
        return c^
    if which == 3:
        var c = _Case("gqa-bool", 2, 4, 2, 3, 4, 3, 2)
        c.kind = 1
        var md: List[Int] = [3, 4]
        var mv: List[Float32] = [1.0, 1.0, 0.0, 1.0,
                                 0.0, 0.0, 0.0, 0.0,       # query 1: nothing left
                                 0.0, 1.0, 1.0, 1.0]
        c.mdims = md^
        c.mvals = mv^
        c.dead_row = 1
        return c^
    if which == 4:
        var c = _Case("rank3-scale", 2, 1, 1, 3, 2, 2, 3)
        c.rank3 = True
        c.kind = 1
        var md: List[Int] = [2, 3, 2]
        var mv: List[Float32] = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                                 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]   # batch row 1, query 2: nothing left
        c.mdims = md^
        c.mvals = mv^
        c.scale_given = True
        c.scale = 0.5
        return c^
    var c = _Case("causal-cross", 1, 1, 1, 3, 5, 3, 2)
    c.causal = True
    return c^


def _count(dims: List[Int]) -> Int:
    var n = 1
    for k in range(len(dims)):
        n *= dims[k]
    return n


def _mask_at(c: _Case, b: Int, h: Int, i: Int, j: Int) -> Float64:
    """The mask word broadcast to (b, h, i, j) (right-aligned; an axis of length 1 broadcasts)."""
    var coords = List[Int]()
    coords.append(b)
    if not c.rank3:
        coords.append(h)
    coords.append(i)
    coords.append(j)
    var nd = len(coords)
    var mr = len(c.mdims)
    var idx = 0
    for u in range(nd):
        var mu = u - (nd - mr)
        if mu < 0:
            continue
        var e = c.mdims[mu]
        idx = idx * e + (coords[u] if e > 1 else 0)
    return Float64(c.mvals[idx])


def _reference(c: _Case, q: List[Float64], k: List[Float64], v: List[Float64]) -> List[Float64]:
    var G = c.H // c.Hkv
    var scale = c.scale if c.scale_given else 1.0 / sqrt(Float64(c.d))
    var out = List[Float64](length=c.B * c.H * c.Lq * c.dv, fill=0.0)
    var s = List[Float64](length=c.Lk, fill=0.0)
    var live = List[Bool](length=c.Lk, fill=False)
    for b in range(c.B):
        for h in range(c.H):
            var kh = h // G
            for i in range(c.Lq):
                var m = Float64(0)
                var found = False
                for j in range(c.Lk):
                    live[j] = not (c.causal and j > i)
                    var mv = Float64(0)
                    if c.kind != 0:
                        mv = _mask_at(c, b, h, i, j)
                    if c.kind == 1 and mv == 0.0:
                        live[j] = False
                    if live[j]:
                        var dot = Float64(0)
                        for t in range(c.d):
                            dot += (q[((b * c.H + h) * c.Lq + i) * c.d + t]
                                    * k[((b * c.Hkv + kh) * c.Lk + j) * c.d + t])
                        var x = scale * dot
                        if c.kind == 2:
                            x += mv
                        if x == Float64(_neg_inf32()):
                            live[j] = False
                        s[j] = x
                        if live[j] and (not found or x > m):
                            m = x
                            found = True
                if not found:
                    continue
                var denom = Float64(0)
                for j in range(c.Lk):
                    if live[j]:
                        s[j] = exp(s[j] - m)
                        denom += s[j]
                for j in range(c.Lk):
                    if not live[j]:
                        continue
                    var p = s[j] / denom
                    for u in range(c.dv):
                        out[((b * c.H + h) * c.Lq + i) * c.dv + u] += p * v[((b * c.Hkv + kh) * c.Lk + j) * c.dv + u]
    return out^


def _loss(c: _Case, q: List[Float64], k: List[Float64], v: List[Float64], seed: List[Float32]) -> Float64:
    var y = _reference(c, q, k, v)
    var acc = Float64(0)
    for i in range(len(y)):
        acc += Float64(seed[i]) * y[i]
    return acc


def _wide(x: List[Float32]) -> List[Float64]:
    var out = List[Float64](length=len(x), fill=0.0)
    for i in range(len(x)):
        out[i] = Float64(x[i])
    return out^


def _fd_check(c: _Case, which: Int, got: List[Float32], q: List[Float64], k: List[Float64], v: List[Float64],
              seed: List[Float32]) raises:
    """Central differences of the float64 loss along every word of q (0), k (1) or v (2)."""
    var names: List[String] = ["q", "k", "v"]
    var qq = q.copy()
    var kk = k.copy()
    var vv = v.copy()
    var n = len(got)
    for w in range(n):
        var plus: Float64
        var minus: Float64
        if which == 0:
            var x0 = qq[w]
            qq[w] = x0 + _FD_H
            plus = _loss(c, qq, kk, vv, seed)
            qq[w] = x0 - _FD_H
            minus = _loss(c, qq, kk, vv, seed)
            qq[w] = x0
        elif which == 1:
            var x0 = kk[w]
            kk[w] = x0 + _FD_H
            plus = _loss(c, qq, kk, vv, seed)
            kk[w] = x0 - _FD_H
            minus = _loss(c, qq, kk, vv, seed)
            kk[w] = x0
        else:
            var x0 = vv[w]
            vv[w] = x0 + _FD_H
            plus = _loss(c, qq, kk, vv, seed)
            vv[w] = x0 - _FD_H
            minus = _loss(c, qq, kk, vv, seed)
            vv[w] = x0
        var fd = (plus - minus) / (2.0 * _FD_H)
        var a = Float64(got[w])
        require(isfinite(a), c.name + ": d" + names[which] + " non-finite at " + String(w))
        require(abs(a - fd) <= 2e-3 * (1.0 + abs(fd)), c.name + ": d" + names[which] + " at " + String(w) + " is "
                + String(a) + ", finite differences give " + String(fd))


def _fd_mask(which: Int, got: List[Float32], q: List[Float64], k: List[Float64], v: List[Float64],
             seed: List[Float32]) raises:
    """Central differences of the float64 loss along every word of the (broadcast) float mask: the mask's
    gradient sums d loss / d (scaled score + mask) over every score the word is added to."""
    var c = _case(which)
    require(len(got) == len(c.mvals), c.name + ": d mask extent")
    for w in range(len(got)):
        var x0 = c.mvals[w]
        c.mvals[w] = Float32(Float64(x0) + _FD_H)
        var plus = _loss(c, q, k, v, seed)
        c.mvals[w] = Float32(Float64(x0) - _FD_H)
        var minus = _loss(c, q, k, v, seed)
        var step = (Float64(Float32(Float64(x0) + _FD_H)) - Float64(Float32(Float64(x0) - _FD_H)))
        c.mvals[w] = x0
        var fd = Float64(0) if step != step or step == 0.0 else (plus - minus) / step
        var a = Float64(got[w])
        require(isfinite(a), c.name + ": d mask non-finite at " + String(w))
        require(abs(a - fd) <= 2e-3 * (1.0 + abs(fd)), c.name + ": d mask at " + String(w) + " is " + String(a)
                + ", finite differences give " + String(fd))


struct _Run(Movable):
    var out: List[Float32]
    var dq: List[Float32]
    var dk: List[Float32]
    var dv: List[Float32]
    var dm: List[Float32]
    var plain: List[Float32]

    def __init__(out self):
        self.out = List[Float32]()
        self.dq = List[Float32]()
        self.dk = List[Float32]()
        self.dv = List[Float32]()
        self.dm = List[Float32]()
        self.plain = List[Float32]()


def _mask[E: Exec & Movable & Deinitable](mut s: OpsSession[E], c: _Case, grad: Bool) raises -> Int:
    """The case's mask as a session tensor (0 for none): ids words for a boolean mask, float values otherwise
    (trainable with `grad`)."""
    if c.kind == 1:
        var words = List[Int32](length=len(c.mvals), fill=0)
        for i in range(len(c.mvals)):
            words[i] = Int32(c.mvals[i])
        var cols = c.mdims[len(c.mdims) - 1]
        var src = CastIn(Int(words.unsafe_ptr()), CAST_KIND_SIGNED, 4, len(words) // cols, cols, cols * 4, 4)
        var h = s.upload_ids(src, ShapeN.of(c.mdims))
        _ = len(words)  # keep-alive past the upload
        return h
    if c.kind == 2:
        return tensor(s, c.mvals, c.mdims, grad)
    return 0


def _call[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], c: _Case, hq: Int, hk: Int, hv: Int, hm: Int,
) raises -> Int:
    var hs: List[Int] = [hq, hk, hv]
    if hm != 0:
        hs.append(hm)
    var ints: List[Int] = [1 if c.causal else 0, 1 if c.scale_given else 0, c.kind]
    var fs: List[Float32] = [Float32(c.scale)]
    return s.op(OPS_ATTENTION_QKV, hs, ints, fs)


def _run[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, rid: Int, c: _Case, qv: List[Float32], kv: List[Float32], vv: List[Float32], seed: List[Float32],
) raises -> _Run:
    var r = _Run()
    var qd = c.dims(c.H, c.Lq, c.d)
    var kd = c.dims(c.Hkv, c.Lk, c.d)
    var vd = c.dims(c.Hkv, c.Lk, c.dv)
    var od = c.dims(c.H, c.Lq, c.dv)
    var s = ops_session[E, KIND](rid)
    var hq = tensor(s[], qv, qd, True)
    var hk = tensor(s[], kv, kd, True)
    var hv = tensor(s[], vv, vd, True)
    var hm = _mask(s[], c, c.mask_grad)
    var out = _call(s[], c, hq, hk, hv, hm)
    r.out = values(s[], out, od)
    var hseed = tensor(s[], seed, od)
    var weighted = op2(s[], _MUL, out, hseed)
    var loss = total(s[], weighted)
    s[].backward(loss, False)
    r.dq = gradient(s[], hq, qd)
    r.dk = gradient(s[], hk, kd)
    r.dv = gradient(s[], hv, vd)
    if c.mask_grad:
        r.dm = gradient(s[], hm, c.mdims)
    s[].reset()
    # The seeded backward (Session.vjp, backward_seed: ImportRules) of the same call with the seed itself must
    # give the scalar backward's words (d sum(out * seed) / d out is the seed, exactly).
    var out2 = _call(s[], c, hq, hk, hv, hm)
    var wrt: List[Int] = [hq, hk, hv]
    if c.mask_grad:
        wrt.append(hm)
    var g = s[].vjp(out2, wrt, hseed, False)
    same_words(values(s[], g[0], qd), r.dq, c.name + ": vjp dq differs from backward's")
    same_words(values(s[], g[1], kd), r.dk, c.name + ": vjp dk differs from backward's")
    same_words(values(s[], g[2], vd), r.dv, c.name + ": vjp dv differs from backward's")
    if c.mask_grad:
        same_words(values(s[], g[3], c.mdims), r.dm, c.name + ": vjp d mask differs from backward's")
    s[].reset()
    var p = ops_session[E, KIND](sid)
    var pq = tensor(p[], qv, qd)
    var pk = tensor(p[], kv, kd)
    var pv = tensor(p[], vv, vd)
    var pm = _mask(p[], c, False)
    var plain_out = _call(p[], c, pq, pk, pv, pm)
    r.plain = values(p[], plain_out, od)
    p[].reset()
    return r^


def _check_case[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, rid: Int, which: Int) raises -> _Run:
    var c = _case(which)
    var qv = normal(c.B * c.H * c.Lq * c.d, 100 + which)
    var kv = normal(c.B * c.Hkv * c.Lk * c.d, 200 + which)
    var vv = normal(c.B * c.Hkv * c.Lk * c.dv, 300 + which)
    var seed = normal(c.B * c.H * c.Lq * c.dv, 400 + which)
    var r = _run[E, KIND](sid, rid, c, qv, kv, vv, seed)
    var q64 = _wide(qv)
    var k64 = _wide(kv)
    var v64 = _wide(vv)
    var want = _reference(c, q64, k64, v64)
    require(len(r.out) == len(want), c.name + ": output extent")
    for i in range(len(want)):
        var a = Float64(r.out[i])
        require(isfinite(a), c.name + ": output non-finite at " + String(i))
        require(abs(a - want[i]) <= 2e-5 + 2e-5 * abs(want[i]), c.name + ": output at " + String(i) + " is "
                + String(a) + ", the float64 reference " + String(want[i]))
    same_words(r.plain, r.out, c.name + ": the recorded forward differs from the unrecorded one")
    _fd_check(c, 0, r.dq, q64, k64, v64, seed)
    _fd_check(c, 1, r.dk, q64, k64, v64, seed)
    _fd_check(c, 2, r.dv, q64, k64, v64, seed)
    if c.mask_grad:
        _fd_mask(which, r.dm, q64, k64, v64, seed)
    if c.dead_row >= 0:
        for b in range(c.B):
            for h in range(c.H):
                var row = (b * c.H + h) * c.Lq + c.dead_row
                for u in range(c.dv):
                    require(bitcast[DType.uint32](r.out[row * c.dv + u]) == UInt32(0),
                            c.name + ": a fully masked row's output is not +0.0")
                for t in range(c.d):
                    require(r.dq[row * c.d + t] == Float32(0), c.name + ": a fully masked row passes a gradient to q")
    return r^


def _open_and_run[E: Exec & Movable & Deinitable, KIND: Int](var plain: E, var recording: E, which: Int) raises -> _Run:
    """`which` on a fresh inference session over `plain` and a fresh recording one over `recording`."""
    var sid = ops_session_open[E, KIND](plain^, False)
    var rid = ops_session_open[E, KIND](recording^, True)
    try:
        var r = _check_case[E, KIND](sid, rid, which)
        ops_session_close[E, KIND](rid)
        ops_session_close[E, KIND](sid)
        return r^
    except error:
        ops_session_close[E, KIND](rid)
        ops_session_close[E, KIND](sid)
        raise error


def test_nn_qkv_cases() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(which: Int) raises:
            _ = _open_and_run[HostExec, OPS_HOST](HostExec(), HostExec(), which)
        check_cases_serial("nn_qkv", NN_QKV_CASES, host_case)
    else:
        def device_case(which: Int) raises:
            var dev = _open_and_run[DeviceExec, OPS_DEVICE](DeviceExec(), DeviceExec(), which)
            var host = _open_and_run[HostExec, OPS_HOST](HostExec(), HostExec(), which)
            same_words(dev.out, host.out, "nn_qkv case " + String(which) + ": device and host outputs")
            same_words(dev.dq, host.dq, "nn_qkv case " + String(which) + ": device and host dq")
            same_words(dev.dk, host.dk, "nn_qkv case " + String(which) + ": device and host dk")
            same_words(dev.dv, host.dv, "nn_qkv case " + String(which) + ": device and host dv")
            same_words(dev.dm, host.dm, "nn_qkv case " + String(which) + ": device and host d mask")
        check_cases_serial("nn_qkv", NN_QKV_CASES, device_case)


def _refused[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], hs: List[Int], ints: List[Int], words: String,
) raises:
    var fs: List[Float32] = [Float32(0)]
    var raised = False
    try:
        _ = s.op(OPS_ATTENTION_QKV, hs, ints, fs)
    except problem:
        raised = True
        require(String(problem).find(words) >= 0, "attention refused with '" + String(problem) + "', not '" + words
                + "'")
    require(raised, "attention did not refuse: " + words)


def _refusals[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    var s = ops_session[E, KIND](sid)
    var d1243: List[Int] = [1, 2, 4, 3]
    var d1244: List[Int] = [1, 2, 4, 4]
    var d1343: List[Int] = [1, 3, 4, 3]
    var d1134: List[Int] = [1, 1, 3, 4]
    var d44: List[Int] = [4, 4]
    var d43: List[Int] = [4, 3]
    var q = tensor(s[], normal(24, 1), d1243)
    var k = tensor(s[], normal(24, 2), d1243)
    var v = tensor(s[], normal(24, 3), d1243)
    var k4 = tensor(s[], normal(32, 4), d1244)
    var k3h = tensor(s[], normal(36, 5), d1343)
    var m = tensor(s[], normal(12, 6), d1134)
    var m44 = tensor(s[], normal(16, 7), d44)
    var q2 = tensor(s[], normal(12, 8), d43)
    var causal_add: List[Int] = [1, 0, 2]
    var none: List[Int] = [0, 0, 0]
    var additive: List[Int] = [0, 0, 2]
    var with_mask: List[Int] = [q, k, v, m44]
    _refused(s[], with_mask, causal_add, "causal=True cannot be combined with an explicit mask")
    var plain: List[Int] = [q, k4, v]
    _refused(s[], plain, none, "q and k must have the same feature size")
    var heads: List[Int] = [q, k3h, k3h]
    _refused(s[], heads, none, "must be a multiple of the key/value heads")
    var bad_mask: List[Int] = [q, k, v, m]
    _refused(s[], bad_mask, additive, "does not broadcast to the scores")
    var flat: List[Int] = [q2, q2, q2]
    _refused(s[], flat, none, "must all be (batch, heads, length, features)")
    s[].reset()


def test_nn_qkv_refusals() raises:
    comptime if CPU_COLUMN_BUILD:
        var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
        try:
            _refusals[HostExec, OPS_HOST](sid)
        except error:
            ops_session_close[HostExec, OPS_HOST](sid)
            raise error
        ops_session_close[HostExec, OPS_HOST](sid)
    else:
        var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
        try:
            _refusals[DeviceExec, OPS_DEVICE](sid)
        except error:
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
            raise error
        ops_session_close[DeviceExec, OPS_DEVICE](sid)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
