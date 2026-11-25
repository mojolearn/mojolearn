# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The ops.Session forms of Sequential's norms, prelu and mish, multi-head attention and positional encoding, losses
and raw state-space scans (lane session-ops-rest, 2026-11-25; session codes 160..174, autograd/session_ops_rest.mojo),
through the call the Python door makes (`OpsSession.op(code, handles, ints, floats)`).

test_session_ops_rest_layers, 17 cases, each run twice on the same inputs:
  (a) the Session op in a recording session: the output, then backward of sum(seed * output) (or of the loss
      itself), the gradient of every trainable input, and the running statistics after the call;
  (b) the Sequential layer's own forward function on a fresh tape (rankn_batch_norm, graph_group_norm,
      graph_instance_norm, nn_act_forward, nn_mha_forward with a packed parameter vector laid out as Sequential
      lays it, nn_pos_enc_forward, the losses' tape functions, nn_ctc.ctc_loss, the graph SSM scans) with the same
      seed, differentiated with ModelRules, Sequential's backward rules.
  The output, every gradient and the running statistics must be the SAME WORDS. Cases: 0 batch_norm training with
  running statistics; 1 batch_norm eval; 2 group_norm (2 groups of 2); 3 instance_norm training with running
  statistics; 4 prelu per channel; 5 mish; 6 multi_head_attention self, causal, biases; 7 cross-attention with
  key_lengths (kdim 6, one row's last two keys masked); 8 positional_encoding; 9 smooth_l1_loss (beta 0.5); 10
  nll_loss (ids, a repeated class); 11 kl_div (batchmean, a zero probability); 12 binary_cross_entropy_with_logits
  (pos_weight 2.5); 13 ctc_loss ((T, N, C) against Sequential's (N, T, C); a repeated label); 14 selective_scan;
  15 ssd_scan; 16 trapezoid_scan.
  The device column runs every case on DeviceExec AND HostExec and requires the same words.
test_session_ops_rest_reference (CPU column): float64 references of cases 0 (forward and running statistics), 2,
  4, 5, 8, 9, 10, 11 and 12, and the gradients of cases 2, 4, 5, 9, 10, 11 and 12 against central differences of
  the float64 objective (h = 1e-3), |g - fd| <= 2e-3 (1 + |fd|). Multi-head attention, the CTC loss and the scans
  have their float64 references in their layers' own suites (nn_attention.mojo, nn_losses_2.mojo, the SSM suites);
  here they are proved to give those layers' words.
Serial by nature: each case opens sessions through the process-wide session table (autograd/ops_api.mojo
`_OPS_SESSIONS`), as core_ops_session_blocks says.
"""
from std.math import exp, sqrt, log, log1p, tanh, sin, cos, isfinite
from std.memory import bitcast
from std.testing import TestSuite
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE
from autograd.tape import Tape, Tensor
from autograd.ops import AXIS_ALL
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_as_shape
from autograd.model_blocks import ModelRules
from autograd.rankn_conv import rankn_batch_norm
from autograd.nn_norms import graph_group_norm, graph_instance_norm
from autograd.nn_activations import nn_act_forward, LAYER_PRELU, LAYER_MISH
from autograd.nn_attention import nn_mha_forward, nn_pos_enc_forward
from autograd.losses import smooth_l1, nll, kl_div, binary_cross_entropy_with_logits
from autograd.nn_ctc import ctc_loss, ctc_refuse_rows
from autograd.embedding_ops import embedding_check_words
from autograd.ssm_primitives import graph_selective_scan
from autograd.ssm_m2_primitives import graph_ssd_scan
from autograd.ssm_m3_primitives import graph_trapezoid_scan
from autograd.session_ops_rest import (
    SOR_BATCH_NORM, SOR_GROUP_NORM, SOR_INSTANCE_NORM, SOR_PRELU, SOR_MISH, SOR_MHA, SOR_POS_ENC, SOR_SMOOTH_L1,
    SOR_NLL, SOR_KL_DIV, SOR_BCE_POS_WEIGHT, SOR_CTC, SOR_SELECTIVE_SCAN, SOR_SSD_SCAN, SOR_TRAPEZOID_SCAN,
)
from core.cast_in import CastIn, CAST_KIND_SIGNED
from sequence.ops import FP
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_higher_common import tensor, total, gradient, op2
from checks.native_tests.core_ops_common import normal
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, check_cases_serial, require

comptime SOR_CASES = 17
comptime _MUL = 4
comptime _EPS = Float32(1.0e-5)
comptime _MOM = Float32(0.25)
#: input kinds of a case
comptime _K_FLOAT = 0
comptime _K_TRAIN = 1
comptime _K_IDS = 2
comptime _K_STATE = 3
comptime _K_SAME0 = 4   # the handle of input 0 again (self-attention's key and value)


@fieldwise_init
struct _In(Copyable, Movable):
    var v: List[Float32]
    var dims: List[Int]
    var kind: Int


struct _Res(Movable):
    var y: List[Float32]
    var grads: List[List[Float32]]
    var state: List[List[Float32]]

    def __init__(out self):
        self.y = List[Float32]()
        self.grads = List[List[Float32]]()
        self.state = List[List[Float32]]()


struct _Case(Movable):
    var name: String
    var code: Int
    var ins: List[_In]
    var ints: List[Int]
    var fs: List[Float32]
    var scalar: Bool

    def __init__(out self, name: String, code: Int, scalar: Bool):
        self.name = name
        self.code = code
        self.ins = List[_In]()
        self.ints = List[Int]()
        self.fs = List[Float32]()
        self.scalar = scalar

    def add(mut self, var v: List[Float32], var dims: List[Int], kind: Int):
        self.ins.append(_In(v^, dims^, kind))


def _count(dims: List[Int]) -> Int:
    var n = 1
    for i in range(len(dims)):  # small-loop(fixture)
        n *= dims[i]
    return n


def _away_from_zero(var v: List[Float32], gap: Float32) -> List[Float32]:
    """Values moved off [-gap, gap] (a kink of prelu or smooth L1 stays out of the finite differences)."""
    for i in range(len(v)):  # small-loop(fixture)
        if v[i] >= Float32(0) and v[i] < gap:
            v[i] += gap
        elif v[i] < Float32(0) and v[i] > -gap:
            v[i] -= gap
    return v^


def _positive(var v: List[Float32], lo: Float32, hi: Float32) -> List[Float32]:
    """|v| mapped into [lo, hi) (step sizes, scales)."""
    for i in range(len(v)):  # small-loop(fixture)
        var a = v[i] if v[i] >= Float32(0) else -v[i]
        v[i] = lo + (hi - lo) * (a / (a + Float32(1)))
    return v^


def _log_softmax_rows(v: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32](length=rows * cols, fill=Float32(0))
    for r in range(rows):  # small-loop(fixture)
        var m = Float64(v[r * cols])
        for c in range(cols):  # small-loop(fixture)
            m = max(m, Float64(v[r * cols + c]))
        var s = 0.0
        for c in range(cols):  # small-loop(fixture)
            s += exp(Float64(v[r * cols + c]) - m)
        for c in range(cols):  # small-loop(fixture)
            out[r * cols + c] = Float32(Float64(v[r * cols + c]) - m - log(s))
    return out^


def _ctc_lp_tnc() -> List[Float32]:
    """(T 4, N 2, C 3) log-probabilities: a log_softmax over C of each (t, n) row."""
    return _log_softmax_rows(normal(24, 1301), 8, 3)


def _case(which: Int) raises -> _Case:
    if which == 0 or which == 1:
        var train = which == 0
        var label = String("batch_norm training")
        var xd: List[Int] = [3, 2, 2, 2]
        if not train:
            label = String("batch_norm eval")
            xd = [2, 2, 3]
        var c = _Case(label, SOR_BATCH_NORM, False)
        c.add(normal(_count(xd), 100 + which), xd^, _K_TRAIN)
        var rd: List[Int] = [2, 1]
        var rm: List[Float32] = [Float32(0.125), Float32(-0.25)]
        var rv: List[Float32] = [Float32(1.5), Float32(0.75)]
        c.add(rm^, rd.copy(), _K_STATE if train else _K_FLOAT)
        c.add(rv^, rd.copy(), _K_STATE if train else _K_FLOAT)
        var w: List[Float32] = [Float32(1.25), Float32(0.5)]
        var b: List[Float32] = [Float32(0.1), Float32(-0.2)]
        c.add(w^, rd.copy(), _K_TRAIN)
        c.add(b^, rd^, _K_TRAIN)
        c.ints = [1 if train else 0, 1, 1]
        c.fs = [_MOM, _EPS]
        return c^
    if which == 2:
        var c = _Case("group_norm", SOR_GROUP_NORM, False)
        var xd: List[Int] = [2, 4, 3]
        c.add(normal(24, 102), xd^, _K_TRAIN)
        var wd: List[Int] = [4, 1]
        c.add(_positive(normal(4, 202), Float32(0.5), Float32(1.5)), wd.copy(), _K_TRAIN)
        c.add(normal(4, 302), wd^, _K_TRAIN)
        c.ints = [2, 1]
        c.fs = [_EPS]
        return c^
    if which == 3:
        var c = _Case("instance_norm training", SOR_INSTANCE_NORM, False)
        var xd: List[Int] = [2, 2, 3]
        c.add(normal(12, 103), xd^, _K_TRAIN)
        var rd: List[Int] = [2, 1]
        var rm: List[Float32] = [Float32(0.0), Float32(0.5)]
        var rv: List[Float32] = [Float32(1.0), Float32(2.0)]
        c.add(rm^, rd.copy(), _K_STATE)
        c.add(rv^, rd.copy(), _K_STATE)
        c.add(_positive(normal(2, 203), Float32(0.5), Float32(1.5)), rd.copy(), _K_TRAIN)
        c.add(normal(2, 303), rd^, _K_TRAIN)
        c.ints = [1, 1, 1]
        c.fs = [_MOM, _EPS]
        return c^
    if which == 4:
        var c = _Case("prelu", SOR_PRELU, False)
        var xd: List[Int] = [2, 3, 2]
        c.add(_away_from_zero(normal(12, 104), Float32(0.1)), xd^, _K_TRAIN)
        var wd: List[Int] = [3, 1]
        c.add(_positive(normal(3, 204), Float32(0.05), Float32(0.4)), wd^, _K_TRAIN)
        return c^
    if which == 5:
        var c = _Case("mish", SOR_MISH, False)
        var xd: List[Int] = [2, 5]
        c.add(normal(10, 105), xd^, _K_TRAIN)
        return c^
    if which == 6:
        # self-attention, E 4, 2 heads, causal, biases: [x, x, in_proj (4, 12), out_proj (4, 4), b_in 12, b_out 4]
        var c = _Case("multi_head_attention self causal", SOR_MHA, False)
        var xd: List[Int] = [2, 3, 4]
        c.add(normal(24, 106), xd^, _K_TRAIN)
        c.add(List[Float32](), List[Int](), _K_SAME0)
        var wi: List[Int] = [4, 12]
        var wo: List[Int] = [4, 4]
        var bi: List[Int] = [12, 1]
        var bo: List[Int] = [4, 1]
        c.add(_scaled(normal(48, 206), Float32(0.4)), wi^, _K_TRAIN)
        c.add(_scaled(normal(16, 306), Float32(0.4)), wo^, _K_TRAIN)
        c.add(_scaled(normal(12, 406), Float32(0.1)), bi^, _K_TRAIN)
        c.add(_scaled(normal(4, 506), Float32(0.1)), bo^, _K_TRAIN)
        c.ints = [2, 1, 1, 1, 0, 0, 0, 0, 0]
        c.fs = [Float32(0)]
        return c^
    if which == 7:
        # cross-attention, kdim 6, key_lengths [5, 3]: [x, m, Wq, [Wk | Wv], out_proj, bq, [bk | bv], b_out, lengths]
        var c = _Case("multi_head_attention cross key_lengths", SOR_MHA, False)
        var xd: List[Int] = [2, 3, 4]
        var md: List[Int] = [2, 5, 6]
        c.add(normal(24, 107), xd^, _K_TRAIN)
        c.add(normal(60, 207), md^, _K_TRAIN)
        var wq: List[Int] = [4, 4]
        var wkv: List[Int] = [6, 8]
        var wo: List[Int] = [4, 4]
        var bq: List[Int] = [4, 1]
        var bkv: List[Int] = [8, 1]
        var bo: List[Int] = [4, 1]
        var ld: List[Int] = [2, 1]
        c.add(_scaled(normal(16, 307), Float32(0.4)), wq^, _K_TRAIN)
        c.add(_scaled(normal(48, 407), Float32(0.4)), wkv^, _K_TRAIN)
        c.add(_scaled(normal(16, 507), Float32(0.4)), wo^, _K_TRAIN)
        c.add(_scaled(normal(4, 607), Float32(0.1)), bq^, _K_TRAIN)
        c.add(_scaled(normal(8, 707), Float32(0.1)), bkv^, _K_TRAIN)
        c.add(_scaled(normal(4, 807), Float32(0.1)), bo^, _K_TRAIN)
        var lens: List[Float32] = [Float32(5), Float32(3)]
        c.add(lens^, ld^, _K_FLOAT)
        c.ints = [2, 0, 1, 0, 1, 0, 0, 0, 0]
        c.fs = [Float32(0)]
        return c^
    if which == 8:
        var c = _Case("positional_encoding", SOR_POS_ENC, False)
        var xd: List[Int] = [2, 3, 4]
        c.add(normal(24, 108), xd^, _K_TRAIN)
        c.ints = [0, 0, 0, 0]
        c.fs = [Float32(0)]
        return c^
    if which >= 9 and which <= 12:
        var code = SOR_SMOOTH_L1 if which == 9 else (SOR_NLL if which == 10 else (
            SOR_KL_DIV if which == 11 else SOR_BCE_POS_WEIGHT))
        var c = _Case(_loss_name(which), code, True)
        var d: List[Int] = [3, 4]
        var p = normal(12, 109 + which)
        if which == 10 or which == 11:
            p = _log_softmax_rows(p, 3, 4)
        c.add(p^, d.copy(), _K_TRAIN)
        if which == 9:
            # differences kept off smooth L1's threshold 0.5 (its kink)
            var t = normal(12, 209)
            var pp = c.ins[0].v.copy()
            for i in range(12):  # small-loop(fixture)
                var dd = pp[i] - t[i]
                var a = dd if dd >= Float32(0) else -dd
                if a > Float32(0.4) and a < Float32(0.6):
                    t[i] += Float32(0.3)
            c.add(t^, d^, _K_FLOAT)
            c.ints = [0]
            c.fs = [Float32(0.5)]
        elif which == 10:
            var ids: List[Float32] = [Float32(2), Float32(0), Float32(2)]
            var idd: List[Int] = [3, 1]
            c.add(ids^, idd^, _K_IDS)
            c.ints = [0]
        elif which == 11:
            var q: List[Float32] = [0.5, 0.25, 0.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25]
            c.add(q^, d^, _K_FLOAT)
            c.ints = [2]
        else:
            var y: List[Float32] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            c.add(y^, d^, _K_FLOAT)
            c.ints = [0]
            c.fs = [Float32(2.5)]
        return c^
    if which == 13:
        # ctc_loss: (T 4, N 2, C 3); targets [[1, 1], [2, 0]], input lengths [4, 3], target lengths [2, 1]
        var c = _Case("ctc_loss", SOR_CTC, True)
        var ld: List[Int] = [4, 2, 3]
        c.add(_ctc_lp_tnc(), ld^, _K_TRAIN)
        var tg: List[Float32] = [1.0, 1.0, 2.0, 0.0]
        var tgd: List[Int] = [2, 2]
        c.add(tg^, tgd^, _K_IDS)
        var il: List[Float32] = [4.0, 3.0]
        var tl: List[Float32] = [2.0, 1.0]
        var nd: List[Int] = [2, 1]
        c.add(il^, nd.copy(), _K_IDS)
        c.add(tl^, nd^, _K_IDS)
        c.ints = [0, 0, 0]
        return c^
    if which == 14:
        var c = _Case("selective_scan", SOR_SELECTIVE_SCAN, False)
        var ud: List[Int] = [2, 3, 2]
        var ad: List[Int] = [2, 4]
        var bd: List[Int] = [2, 3, 4]
        var dd: List[Int] = [2, 1]
        c.add(normal(12, 114), ud.copy(), _K_TRAIN)
        c.add(_positive(normal(12, 214), Float32(0.05), Float32(0.5)), ud^, _K_TRAIN)
        c.add(_scaled(normal(8, 314), Float32(0.5)), ad^, _K_TRAIN)
        c.add(normal(24, 414), bd.copy(), _K_TRAIN)
        c.add(normal(24, 514), bd^, _K_TRAIN)
        c.add(normal(2, 614), dd^, _K_TRAIN)
        return c^
    if which == 15:
        var c = _Case("ssd_scan", SOR_SSD_SCAN, False)
        var xd: List[Int] = [1, 3, 320]
        var td: List[Int] = [1, 3, 1]
        var hd: List[Int] = [1, 1]
        c.add(_scaled(normal(960, 115), Float32(0.5)), xd^, _K_TRAIN)
        c.add(_positive(normal(3, 215), Float32(0.05), Float32(0.5)), td^, _K_TRAIN)
        c.add(_scaled(normal(1, 315), Float32(0.5)), hd.copy(), _K_TRAIN)
        c.add(normal(1, 415), hd^, _K_TRAIN)
        return c^
    if which == 16:
        var c = _Case("trapezoid_scan", SOR_TRAPEZOID_SCAN, False)
        var vd: List[Int] = [2, 3, 4]
        var hd: List[Int] = [2, 3, 2]
        var dd: List[Int] = [2, 1]
        c.add(normal(24, 116), vd.copy(), _K_TRAIN)
        c.add(_scaled(normal(24, 216), Float32(0.5)), vd.copy(), _K_TRAIN)
        c.add(_scaled(normal(24, 316), Float32(0.5)), vd^, _K_TRAIN)
        c.add(_positive(normal(12, 416), Float32(0.05), Float32(0.5)), hd.copy(), _K_TRAIN)
        var a = _positive(normal(12, 516), Float32(0.1), Float32(1.0))
        for i in range(12):  # small-loop(fixture)
            a[i] = -a[i]
        c.add(a^, hd.copy(), _K_TRAIN)
        c.add(_positive(normal(12, 616), Float32(0.0), Float32(1.0)), hd^, _K_TRAIN)
        c.add(normal(2, 716), dd^, _K_TRAIN)
        c.ints = [2]
        return c^
    raise Error("session_ops_rest: no case " + String(which))


def _loss_name(which: Int) -> String:
    if which == 9:
        return "smooth_l1_loss"
    if which == 10:
        return "nll_loss"
    if which == 11:
        return "kl_div"
    return "binary_cross_entropy_with_logits"


def _scaled(var v: List[Float32], f: Float32) -> List[Float32]:
    for i in range(len(v)):  # small-loop(fixture)
        v[i] = v[i] * f
    return v^


def _seed(n: Int, which: Int) -> List[Float32]:
    return normal(n, 900 + which)


# ---------------------------------------------------------------- (a) the Session op
def _upload_ids[E: Exec & Movable & Deinitable](mut s: OpsSession[E], v: List[Float32], dims: List[Int]) raises -> Int:
    var words = List[Int32](length=len(v), fill=0)
    for i in range(len(v)):  # small-loop(fixture)
        words[i] = Int32(Int(v[i]))
    var cols = dims[len(dims) - 1]
    var src = CastIn(Int(words.unsafe_ptr()), CAST_KIND_SIGNED, 4, len(words) // cols, cols, cols * 4, 4)
    var h = s.upload_ids(src, ShapeN.of(dims))
    _ = len(words)  # keep-alive past the upload
    return h


def _read[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> List[Float32]:
    var n = s.shape_nd(h).count()
    var out = List[Float32](length=n, fill=Float32(0))
    s.download(h, FP(unsafe_from_address=Int(out.unsafe_ptr())))
    return out^


def _session_run[E: Exec & Movable & Deinitable, KIND: Int](rid: Int, c: _Case, seed: List[Float32]) raises -> _Res:
    var r = _Res()
    var s = ops_session[E, KIND](rid)
    var hs = List[Int]()
    for i in range(len(c.ins)):  # small-loop(case inputs)
        var k = c.ins[i].kind
        if k == _K_SAME0:
            hs.append(hs[0])
        elif k == _K_IDS:
            hs.append(_upload_ids(s[], c.ins[i].v, c.ins[i].dims))
        else:
            var h = tensor(s[], c.ins[i].v, c.ins[i].dims, k == _K_TRAIN)
            if k == _K_STATE:
                s[].declare_state(h)
            hs.append(h)
    var y = s[].op(c.code, hs, c.ints, c.fs)
    r.y = _read(s[], y)
    var loss = y
    if not c.scalar:
        var od = List[Int]()
        var ys = s[].shape_nd(y)
        for k in range(ys.ndim):  # small-loop(rank)
            od.append(ys.d[k])
        var hseed = tensor(s[], seed, od)
        loss = total(s[], op2(s[], _MUL, y, hseed))
    s[].backward(loss, False)
    for i in range(len(c.ins)):  # small-loop(case inputs)
        if c.ins[i].kind == _K_TRAIN:
            r.grads.append(gradient(s[], hs[i], c.ins[i].dims))
    for i in range(len(c.ins)):  # small-loop(case inputs)
        if c.ins[i].kind == _K_STATE:
            r.state.append(_read(s[], hs[i]))
    s[].reset()
    return r^


# ---------------------------------------------------------------- (b) the Sequential layer on a tape
def _leaf[E: Exec](mut tape: Tape, mut ex: E, v: List[Float32], dims: List[Int], grad: Bool) raises -> Tensor:
    var cols = dims[len(dims) - 1]
    return tape.tensor(ex, v, len(v) // cols, cols, grad)


def _nd[E: Exec](mut tape: Tape, mut ex: E, t: Tensor, dims: List[Int]) raises -> Tensor:
    return tape.view_nd(ex, t, ShapeN.of(dims))


def _upload[E: Exec](mut ex: E, v: List[Float32]) raises -> FP:
    var p = ex.alloc(len(v))
    ex.upload(p, FP(unsafe_from_address=Int(v.unsafe_ptr())), len(v))
    return p


def _download[E: Exec](mut ex: E, p: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), p, n)
    return out^


def _concat(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = a.copy()
    for i in range(len(b)):  # small-loop(fixture)
        out.append(b[i])
    return out^


def _slice(a: List[Float32], start: Int, n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):  # small-loop(fixture)
        out.append(a[start + i])
    return out^


def _seeded_backward[E: Exec](mut tape: Tape, mut ex: E, y: Tensor, seed: List[Float32]) raises:
    var r = tape.rows(y)
    var cols = tape.cols(y)
    var ym = rn_as_shape(tape, ex, y, ShapeN.matrix(r, cols))
    var st = tape.tensor(ex, seed, r, cols, False)
    var loss = tape.sum(ex, tape.multiply(ex, ym, st), AXIS_ALL)
    tape.backward_scalar_with_rules(ex, loss, ModelRules())


def _tape_run[E: Exec](mut ex: E, which: Int, c: _Case, seed: List[Float32]) raises -> _Res:
    var r = _Res()
    var tape = Tape()
    var ins = c.ins.copy()
    if which == 0 or which == 1:
        var train = which == 0
        var x = _leaf(tape, ex, ins[0].v, ins[0].dims, True)
        var d3: List[Int] = [ins[0].dims[0], 2, len(ins[0].v) // (ins[0].dims[0] * 2)]
        var x3 = _nd(tape, ex, x, d3)
        var g = tape.tensor(ex, ins[3].v, 1, 2, True)
        var b = tape.tensor(ex, ins[4].v, 1, 2, True)
        var run = _upload(ex, _concat(ins[1].v, ins[2].v))
        var y = rankn_batch_norm(tape, ex, x3, g, b, run, _EPS, _MOM, train)
        r.y = tape.values(ex, y)
        _seeded_backward(tape, ex, y, seed)
        r.grads.append(tape.gradients(ex, x))
        r.grads.append(tape.gradients(ex, g))
        r.grads.append(tape.gradients(ex, b))
        if train:
            var words = _download(ex, run, 4)
            r.state.append(_slice(words, 0, 2))
            r.state.append(_slice(words, 2, 2))
        ex.free(run)
    elif which == 2:
        var x = _leaf(tape, ex, ins[0].v, ins[0].dims, True)
        var w = tape.tensor(ex, ins[1].v, 1, 4, True)
        var b = tape.tensor(ex, ins[2].v, 1, 4, True)
        var y = graph_group_norm(tape, ex, _nd(tape, ex, x, ins[0].dims), 2, 4, 3, 2, Optional[Tensor](w),
                                 Optional[Tensor](b), _EPS)
        r.y = tape.values(ex, y)
        _seeded_backward(tape, ex, y, seed)
        r.grads.append(tape.gradients(ex, x))
        r.grads.append(tape.gradients(ex, w))
        r.grads.append(tape.gradients(ex, b))
    elif which == 3:
        var x = _leaf(tape, ex, ins[0].v, ins[0].dims, True)
        var w = tape.tensor(ex, ins[3].v, 1, 2, True)
        var b = tape.tensor(ex, ins[4].v, 1, 2, True)
        var run = _upload(ex, _concat(ins[1].v, ins[2].v))
        var y = graph_instance_norm(tape, ex, _nd(tape, ex, x, ins[0].dims), 2, 2, 3, Optional[Tensor](w),
                                    Optional[Tensor](b), Optional[FP](run), _MOM, _EPS, True)
        r.y = tape.values(ex, y)
        _seeded_backward(tape, ex, y, seed)
        r.grads.append(tape.gradients(ex, x))
        r.grads.append(tape.gradients(ex, w))
        r.grads.append(tape.gradients(ex, b))
        var words = _download(ex, run, 4)
        r.state.append(_slice(words, 0, 2))
        r.state.append(_slice(words, 2, 2))
        ex.free(run)
    elif which == 4 or which == 5:
        # Sequential's activation layer: the slope in params_ P, its gradient into Gr
        var x = _leaf(tape, ex, ins[0].v, ins[0].dims, True)
        var dims = ins[0].dims.copy()
        var sample = List[Int]()
        for k in range(1, len(dims)):  # small-loop(rank)
            sample.append(dims[k])
        var nP = 3 if which == 4 else 1
        var pv = List[Float32](length=1, fill=Float32(0))
        if which == 4:
            pv = ins[1].v.copy()
        var P = _upload(ex, pv)
        var Gr = _upload(ex, List[Float32](length=nP, fill=Float32(0)))
        var y = nn_act_forward(tape, ex, LAYER_PRELU if which == 4 else LAYER_MISH, nP, Float32(0),
                               ShapeN.of(sample), P, Gr, 0, _nd(tape, ex, x, dims), True)
        r.y = tape.values(ex, y)
        _seeded_backward(tape, ex, y, seed)
        r.grads.append(tape.gradients(ex, x))
        if which == 4:
            r.grads.append(_download(ex, Gr, 3))
        ex.free(P)
        ex.free(Gr)
    elif which == 6 or which == 7:
        # Sequential's multihead_attention: params_ [in_proj | out_proj.weight | in_proj_bias | out_proj.bias]
        var self_attn = which == 6
        var x = _leaf(tape, ex, ins[0].v, ins[0].dims, True)
        var xs = _nd(tape, ex, x, ins[0].dims)
        var m = x
        var ms = xs
        var w_in: List[Float32]
        var wo: List[Float32]
        var b_in: List[Float32]
        var bo: List[Float32]
        if self_attn:
            w_in = ins[2].v.copy()
            wo = ins[3].v.copy()
            b_in = ins[4].v.copy()
            bo = ins[5].v.copy()
        else:
            m = _leaf(tape, ex, ins[1].v, ins[1].dims, True)
            ms = _nd(tape, ex, m, ins[1].dims)
            w_in = _concat(ins[2].v, ins[3].v)
            wo = ins[4].v.copy()
            b_in = _concat(ins[5].v, ins[6].v)
            bo = ins[7].v.copy()
        var pv = _concat(_concat(_concat(w_in, wo), b_in), bo)
        var nP = len(pv)
        var P = _upload(ex, pv)
        var Gr = _upload(ex, List[Float32](length=nP, fill=Float32(0)))
        var xargs: List[Int] = [2, 3, -2, 4, 0, 0]
        if not self_attn:
            xargs = [2, 5, -1, 4, 6, 0]
        var lens: Optional[FP] = None
        var lbuf = P
        if not self_attn:
            lbuf = _upload(ex, ins[8].v)
            lens = Optional[FP](lbuf)
        var y = nn_mha_forward(tape, ex, xargs, 0, Float32(0), ShapeN.matrix(3, 4), P, Gr, 0, xs, ms, True,
                               UInt64(0), 0, lens)
        r.y = tape.values(ex, y)
        _seeded_backward(tape, ex, y, seed)
        r.grads.append(tape.gradients(ex, x))
        var gw = _download(ex, Gr, nP)
        var at = 0
        if self_attn:
            var sizes: List[Int] = [48, 16, 12, 4]
            for k in range(4):  # small-loop(4 parameters)
                r.grads.append(_slice(gw, at, sizes[k]))
                at += sizes[k]
        else:
            r.grads.append(tape.gradients(ex, m))
            var sizes: List[Int] = [16, 48, 16, 4, 8, 4]
            for k in range(6):  # small-loop(6 parameters)
                r.grads.append(_slice(gw, at, sizes[k]))
                at += sizes[k]
            ex.free(lbuf)
        ex.free(P)
        ex.free(Gr)
    elif which == 8:
        var x = _leaf(tape, ex, ins[0].v, ins[0].dims, True)
        var y = nn_pos_enc_forward(tape, ex, Float32(0), ShapeN.matrix(3, 4), _nd(tape, ex, x, ins[0].dims), False,
                                   UInt64(0), 0)
        r.y = tape.values(ex, y)
        _seeded_backward(tape, ex, y, seed)
        r.grads.append(tape.gradients(ex, x))
    elif which >= 9 and which <= 12:
        var p = tape.tensor(ex, ins[0].v, 3, 4, True)
        var tv = ins[1].v.copy()
        if which == 10:
            # the fit's one-hot rows of the class ids (OP_ONE_HOT writes exactly these words)
            tv = List[Float32](length=12, fill=Float32(0))
            for n in range(3):  # small-loop(fixture)
                tv[n * 4 + Int(ins[1].v[n])] = Float32(1)
        var t = tape.tensor(ex, tv, 3, 4, False)
        var loss: Tensor
        if which == 9:
            loss = smooth_l1(tape, ex, p, t, Float32(0.5), 0)
        elif which == 10:
            loss = nll(tape, ex, p, t, 0)
        elif which == 11:
            loss = kl_div(tape, ex, p, t, 2)
        else:
            loss = binary_cross_entropy_with_logits(tape, ex, p, t, 0, Float32(2.5))
        r.y = tape.values(ex, loss)
        tape.backward_scalar_with_rules(ex, loss, ModelRules())
        r.grads.append(tape.gradients(ex, p))
    elif which == 13:
        # Sequential's (N, T, C) log-probabilities and its packed target rows [input length, target length, labels]
        var tnc = ins[0].v.copy()
        var ntc = List[Float32](length=24, fill=Float32(0))
        for t in range(4):  # small-loop(fixture)
            for n in range(2):  # small-loop(fixture)
                for k in range(3):  # small-loop(fixture)
                    ntc[(n * 4 + t) * 3 + k] = tnc[(t * 2 + n) * 3 + k]
        var lp = tape.tensor(ex, ntc, 8, 3, True)
        var rows: List[Float32] = [4.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 0.0]
        var rt = tape.tensor(ex, rows, 2, 4, False)
        var loss = ctc_loss(tape, ex, lp, rt, 4, 3, 0, 0, False)
        r.y = tape.values(ex, loss)
        tape.backward_scalar_with_rules(ex, loss, ModelRules())
        var g = tape.gradients(ex, lp)
        # back to the Session's (T, N, C) order
        var gt = List[Float32](length=24, fill=Float32(0))
        for t in range(4):  # small-loop(fixture)
            for n in range(2):  # small-loop(fixture)
                for k in range(3):  # small-loop(fixture)
                    gt[(t * 2 + n) * 3 + k] = g[(n * 4 + t) * 3 + k]
        r.grads.append(gt^)
    else:
        var leaves = List[Tensor]()
        var views = List[Tensor]()
        for i in range(len(ins)):  # small-loop(case inputs)
            var l = _leaf(tape, ex, ins[i].v, ins[i].dims, True)
            leaves.append(l)
            if len(ins[i].dims) > 2:
                views.append(_nd(tape, ex, l, ins[i].dims))
            else:
                views.append(l)
        var y: Tensor
        if which == 14:
            y = graph_selective_scan(tape, ex, views[0], views[1], views[3], views[4], views[2], views[5])
        elif which == 15:
            y = graph_ssd_scan(tape, ex, views[0], views[1], views[2], views[3])
        else:
            y = graph_trapezoid_scan(tape, ex, views[0], views[1], views[2], views[3], views[4], views[5], views[6], 2)
        r.y = tape.values(ex, y)
        _seeded_backward(tape, ex, y, seed)
        for i in range(len(leaves)):  # small-loop(case inputs)
            r.grads.append(tape.gradients(ex, leaves[i]))
    tape.release(ex)
    return r^


# ---------------------------------------------------------------- comparisons
def _same(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": " + String(len(a)) + " words against " + String(len(b)))
    for i in range(len(a)):  # small-loop(check comparison)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]),
                what + ": the words differ at " + String(i) + " (" + String(a[i]) + " against " + String(b[i]) + ")")


def _same_res(a: _Res, b: _Res, what: String) raises:
    _same(a.y, b.y, what + ": output")
    require(len(a.grads) == len(b.grads), what + ": gradient count")
    for k in range(len(a.grads)):  # small-loop(gradients)
        _same(a.grads[k], b.grads[k], what + ": gradient " + String(k))
    require(len(a.state) == len(b.state), what + ": running statistics count")
    for k in range(len(a.state)):  # small-loop(statistics)
        _same(a.state[k], b.state[k], what + ": running statistics " + String(k))


def _seed_for(which: Int, c: _Case) raises -> List[Float32]:
    """The output's seed: as many values as the output has (a scalar loss takes none)."""
    if c.scalar:
        return List[Float32]()
    var n = 0
    if which <= 5 or which == 8:
        n = len(c.ins[0].v)
    elif which == 6 or which == 7:
        n = 24
    elif which == 14:
        n = 12
    elif which == 15:
        n = 3 * 64
    else:
        n = 24
    return _seed(n, which)


def _case_on[E: Exec & Movable & Deinitable, KIND: Int](var session_ex: E, var tape_ex: E, which: Int) raises -> _Res:
    """Case `which`: the Session op on a recording session over `session_ex`, and the Sequential layer on a tape
    over `tape_ex`; they must give the same words. Returns the Session's."""
    var c = _case(which)
    var seed = _seed_for(which, c)
    var rid = ops_session_open[E, KIND](session_ex^, True)
    var got = _Res()
    try:
        got = _session_run[E, KIND](rid, c, seed)
    except error:
        ops_session_close[E, KIND](rid)
        raise error
    ops_session_close[E, KIND](rid)
    var want = _tape_run(tape_ex, which, c, seed)
    _same_res(got, want, c.name + " (the Session op against the Sequential layer)")
    return got^


def test_session_ops_rest_layers() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(which: Int) raises:
            _ = _case_on[HostExec, OPS_HOST](HostExec(), HostExec(), which)
        check_cases_serial("session_ops_rest", SOR_CASES, host_case)
    else:
        def device_case(which: Int) raises:
            var dev = _case_on[DeviceExec, OPS_DEVICE](DeviceExec(), DeviceExec(), which)
            var host = _case_on[HostExec, OPS_HOST](HostExec(), HostExec(), which)
            _same_res(dev, host, "session_ops_rest case " + String(which) + ": device and host")
        check_cases_serial("session_ops_rest", SOR_CASES, device_case)


# ---------------------------------------------------------------- float64 references (CPU column)
def _wide(v: List[Float32]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(v)):  # small-loop(fixture)
        out.append(Float64(v[i]))
    return out^


def _close(got: List[Float32], want: List[Float64], tol: Float64, what: String) raises:
    require(len(got) == len(want), what + ": extent")
    for i in range(len(want)):  # small-loop(check comparison)
        var a = Float64(got[i])
        require(isfinite(a), what + ": non-finite at " + String(i))
        require(abs(a - want[i]) <= tol * (1.0 + abs(want[i])),
                what + ": at " + String(i) + " is " + String(a) + ", the float64 reference " + String(want[i]))


def _ref_group_norm(x: List[Float64], w: List[Float64], b: List[Float64]) -> List[Float64]:
    """(2, 4, 3), 2 groups of 2 channels, eps 1e-5 (biased variance)."""
    var out = List[Float64](length=24, fill=0.0)
    for n in range(2):  # small-loop(fixture)
        for g in range(2):  # small-loop(fixture)
            var base = n * 12 + g * 6
            var mean = 0.0
            for k in range(6):  # small-loop(fixture)
                mean += x[base + k]
            mean /= 6.0
            var v = 0.0
            for k in range(6):  # small-loop(fixture)
                v += (x[base + k] - mean) * (x[base + k] - mean)
            v /= 6.0
            for k in range(6):  # small-loop(fixture)
                var ch = g * 2 + k // 3
                out[base + k] = (x[base + k] - mean) / sqrt(v + 1.0e-5) * w[ch] + b[ch]
    return out^


def _ref_prelu(x: List[Float64], w: List[Float64]) -> List[Float64]:
    """(2, 3, 2), one slope per channel (axis 1)."""
    var out = List[Float64]()
    for i in range(len(x)):  # small-loop(fixture)
        var ch = (i // 2) % 3
        out.append(x[i] if x[i] > 0.0 else w[ch] * x[i])
    return out^


def _ref_mish(x: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(x)):  # small-loop(fixture)
        out.append(x[i] * tanh(log1p(exp(x[i]))))
    return out^


def _ref_pos_enc(x: List[Float64]) -> List[Float64]:
    """(2, 3, 4): x + sin / cos of p / 10000^(2j / 4)."""
    var out = x.copy()
    for n in range(2):  # small-loop(fixture)
        for p in range(3):  # small-loop(fixture)
            for j in range(2):  # small-loop(fixture)
                var ang = Float64(p) * exp(-log(10000.0) * Float64(2 * j) / 4.0)
                out[(n * 3 + p) * 4 + 2 * j] += sin(ang)
                out[(n * 3 + p) * 4 + 2 * j + 1] += cos(ang)
    return out^


def _ref_loss(which: Int, p: List[Float64], t: List[Float32]) -> Float64:
    """The (3, 4) losses of cases 9..12, float64."""
    var acc = 0.0
    for i in range(12):  # small-loop(fixture)
        var y = Float64(t[i])
        if which == 9:
            var d = abs(p[i] - y)
            acc += 0.5 * d * d / 0.5 if d < 0.5 else d - 0.25
        elif which == 11:
            if y > 0.0:
                acc += y * (log(y) - p[i])
        elif which == 12:
            var s = 1.0 / (1.0 + exp(-p[i]))
            acc += -(2.5 * y * log(s) + (1.0 - y) * log(1.0 - s))
    if which == 10:
        for n in range(3):  # small-loop(fixture)
            acc += -p[n * 4 + Int(t[n])]
        return acc / 3.0
    if which == 11:
        return acc / 3.0
    return acc / 12.0


def _objective(which: Int, xs: List[List[Float64]], c: _Case, seed: List[Float32]) -> Float64:
    """sum(seed * f(inputs)) (a loss itself for cases 9..12) in float64."""
    var y: List[Float64]
    if which == 2:
        y = _ref_group_norm(xs[0], xs[1], xs[2])
    elif which == 4:
        y = _ref_prelu(xs[0], xs[1])
    elif which == 5:
        y = _ref_mish(xs[0])
    elif which == 8:
        y = _ref_pos_enc(xs[0])
    else:
        return _ref_loss(which, xs[0], c.ins[1].v)
    var acc = 0.0
    for i in range(len(y)):  # small-loop(fixture)
        acc += Float64(seed[i]) * y[i]
    return acc


def _fd_check(which: Int, c: _Case, seed: List[Float32], got: _Res) raises:
    """Every trainable input's gradient against central differences of the float64 objective."""
    var xs = List[List[Float64]]()
    var trainable = List[Int]()
    for i in range(len(c.ins)):  # small-loop(case inputs)
        xs.append(_wide(c.ins[i].v))
        if c.ins[i].kind == _K_TRAIN:
            trainable.append(i)
    for g in range(len(trainable)):  # small-loop(trainable inputs)
        var at = trainable[g]
        for j in range(len(xs[at])):  # small-loop(fixture)
            var keep = xs[at][j]
            xs[at][j] = keep + 1.0e-3
            var up = _objective(which, xs, c, seed)
            xs[at][j] = keep - 1.0e-3
            var down = _objective(which, xs, c, seed)
            xs[at][j] = keep
            var fd = (up - down) / 2.0e-3
            var a = Float64(got.grads[g][j])
            require(abs(a - fd) <= 2.0e-3 * (1.0 + abs(fd)), c.name + ": gradient " + String(g) + " at " + String(j)
                    + " is " + String(a) + ", central differences " + String(fd))


def _reference_case(which: Int) raises:
    var c = _case(which)
    var seed = _seed_for(which, c)
    var got = _case_on[HostExec, OPS_HOST](HostExec(), HostExec(), which)
    if which == 0:
        # batch_norm training, (3, 2, 2, 2): per channel over 12 values; running = 0.75 r + 0.25 stat (unbiased var)
        var x = _wide(c.ins[0].v)
        var want = List[Float64](length=24, fill=0.0)
        var rm = List[Float64]()
        var rv = List[Float64]()
        for ch in range(2):  # small-loop(fixture)
            var mean = 0.0
            for n in range(3):  # small-loop(fixture)
                for k in range(4):  # small-loop(fixture)
                    mean += x[n * 8 + ch * 4 + k]
            mean /= 12.0
            var v = 0.0
            for n in range(3):  # small-loop(fixture)
                for k in range(4):  # small-loop(fixture)
                    var d = x[n * 8 + ch * 4 + k] - mean
                    v += d * d
            for n in range(3):  # small-loop(fixture)
                for k in range(4):  # small-loop(fixture)
                    var at = n * 8 + ch * 4 + k
                    want[at] = (x[at] - mean) / sqrt(v / 12.0 + 1.0e-5) * Float64(c.ins[3].v[ch]) + Float64(
                        c.ins[4].v[ch])
            rm.append(0.75 * Float64(c.ins[1].v[ch]) + 0.25 * mean)
            rv.append(0.75 * Float64(c.ins[2].v[ch]) + 0.25 * v / 11.0)
        _close(got.y, want, 2.0e-5, c.name + ": output")
        _close(got.state[0], rm, 2.0e-5, c.name + ": running_mean")
        _close(got.state[1], rv, 2.0e-5, c.name + ": running_var")
        return
    var xs = List[List[Float64]]()
    for i in range(len(c.ins)):  # small-loop(case inputs)
        xs.append(_wide(c.ins[i].v))
    if c.scalar:
        var want: List[Float64] = [_ref_loss(which, xs[0], c.ins[1].v)]
        _close(got.y, want, 2.0e-5, c.name + ": loss")
    else:
        var want: List[Float64]
        if which == 2:
            want = _ref_group_norm(xs[0], xs[1], xs[2])
        elif which == 4:
            want = _ref_prelu(xs[0], xs[1])
        elif which == 5:
            want = _ref_mish(xs[0])
        else:
            want = _ref_pos_enc(xs[0])
        _close(got.y, want, 2.0e-5, c.name + ": output")
    if which != 8:
        _fd_check(which, c, seed, got)


def _reference_cases() -> List[Int]:
    var cases: List[Int] = [0, 2, 4, 5, 8, 9, 10, 11, 12]
    return cases^


def test_session_ops_rest_reference() raises:
    comptime if CPU_COLUMN_BUILD:
        def ref_case(k: Int) raises:
            _reference_case(_reference_cases()[k])
        check_cases_serial("session_ops_rest reference", len(_reference_cases()), ref_case)


comptime SOR_CTC_ADMISSION_CASES = 7


def _ctc_admission_on[E: Exec & Movable & Deinitable](var ex: E, which: Int) raises:
    """lane fable-review-2: Session.ctc_loss's admission on the executor (nn_ctc.ctc_refuse_rows: OP_CTC_ADMIT, the
    first-bad search, two words back). Packed rows (N 2, 2 + S 2) with T 4, C 3, blank 0. Case 0 is admitted
    (sample 1's second label is the blank, past its target length 1, so it is never read); each other case
    spoils one word and names it; case 6 spoils two and must name the earlier one (the host loop's order)."""
    var rows: List[Float32] = [4.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 0.0]
    var want = List[String]()
    if which == 1:
        rows[4] = 5.0
        want.append(String("input_lengths must be whole numbers from 0 to the 4 steps"))
        want.append(String("sample 1 has"))
    elif which == 2:
        rows[1] = 2.5
        want.append(String("target_lengths must be whole numbers from 0 to the targets' width (2)"))
        want.append(String("sample 0 has"))
    elif which == 3:
        rows[3] = 3.0
        want.append(String("targets must be class indices in [0, 3)"))
        want.append(String("sample 0 label 1 is"))
    elif which == 4:
        rows[6] = 0.0
        want.append(String("targets must not hold the blank index (0)"))
        want.append(String("sample 1 label 0"))
    elif which == 5:
        rows[2] = 1.5
        want.append(String("targets must be class indices in [0, 3)"))
        want.append(String("sample 0 label 0 is"))
    elif which == 6:
        rows[0] = -1.0
        rows[6] = 0.0
        want.append(String("input_lengths must be whole numbers"))
        want.append(String("sample 0 has"))
    var n = len(rows)
    var cw = embedding_check_words(n)
    var packed = ex.alloc(n)
    var flags = ex.alloc(n)
    var scratch_a = ex.alloc(cw)
    var scratch_b = ex.alloc(cw)
    ex.upload(packed, FP(unsafe_from_address=Int(rows.unsafe_ptr())), n)
    var message = String("")
    try:
        ctc_refuse_rows(ex, packed, 2, 4, 4, 3, 0, flags, scratch_a, scratch_b, String("ctc_loss"))
    except error:
        message = String(error)
    ex.free(packed)
    ex.free(flags)
    ex.free(scratch_a)
    ex.free(scratch_b)
    _ = len(rows)  # the upload's source stays alive past the launches that read it
    if len(want) == 0:
        require(message == "", "ctc admission case 0: admitted rows were refused: " + message)
        return
    for k in range(len(want)):  # small-loop(the message's parts)
        require(want[k] in message, "ctc admission case " + String(which) + ": the refusal must say '" + want[k]
                + "', got '" + message + "'")


def test_session_ops_rest_ctc_admission() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(which: Int) raises:
            _ctc_admission_on[HostExec](HostExec(), which)
        check_cases_serial("session_ops_rest ctc admission", SOR_CTC_ADMISSION_CASES, host_case)
    else:
        def device_case(which: Int) raises:
            _ctc_admission_on[DeviceExec](DeviceExec(), which)
            _ctc_admission_on[HostExec](HostExec(), which)
        check_cases_serial("session_ops_rest ctc admission", SOR_CTC_ADMISSION_CASES, device_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
