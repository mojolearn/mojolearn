# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The block operations of `mojolearn.ops.Session` (lane fx-ops-session-blocks,
2026-11-18): `mamba_block` / `mamba_step`, `attention_block` /
`attention_step`, `small_mlp`, `lm_head_loss` and the Lion, Adamax and NAdam
optimizers, through the typed calls the Python door makes
(autograd/ops_block_frontend.mojo `session_block_create`,
`session_block_step`; `OpsSession.op` codes 87 and 88;
`ops_optimizer_codes`).

"The model's own call" below is the form every model program runs a block
in: `OpsSession.upload_block` for the weights, the zero state and the state
words, then `OpsSession.op` of the step code on them
(mamba/impl/mamba_block_program.mojo `MambaBlockProgram`,
transformer/causal_lm_program.mojo's layer loop, training/samba_program.mojo).

  0  Mamba-1  (b 2, d_model 8)   the public block and step against the
  1  Mamba-2  (b 2, d_model 32)  model's own call on the same weights: a
  2  Mamba-3  (b 2, d_model 32)  three-token prefill, then a one-token
  3  attention (b 2, d_model 8,  step; the outputs, the carried state and
     2 heads, s_max 8)           the state words agree word for word. For
                                 attention also: three one-token steps on a
                                 fresh block give the prefill's words.
  4  A two-layer attention model composed through the public calls (prefill
     of three tokens, then a decode step) against the model's own two-layer
     loop on the same weights: the same words.
  5  Training through each of the four step ops in a recording session: the
     recorded step, backward, the packed weights' gradient and x's gradient
     are finite and not all zero, a second recording gives the same words,
     the recorded forward equals the unrecorded step of a fresh block (the
     Session contract: recording does not change values), and one Adam step
     moves the packed weights and leaves them finite.
  6  small_mlp against a float64 formula (and db2 = the row count exactly
     for a summed loss); lm_head_loss against the float64 mean
     cross-entropy; both recorded and not, with finite gradients.
  7  Lion, Adamax and NAdam: the name codes, one update each on a summed
     loss (Lion p - lr sign(g) and Adamax p - lr / (1 + eps) within 1e-6;
     NAdam finite, moved, mu product advanced) and the state buffers each
     keeps.

  8  RMSprop and Adagrad (one update each against the closed form), and the
     gradient clips: clip_grad_norm over TWO tensors (4 and 5 elements, all
     gradients 1, so the norm is 3: with max_norm 1.5 and SGD at lr 1 every
     element moves by 1.5 / (3 + 1e-6), which a per-tensor clip would not
     give), and clip_grad_value 0.25.
  9  attention_block with options (q/k/v, output and MLP biases): the
     public option words and present pieces against the model's own call
     with the block options tail, word for word.
 10  lstm and gru: step 1 of each sequence against the float64 PyTorch
     formulas from a zero state, recorded == unrecorded, and finite
     non-zero gradients of weight_ih and x.
 11  conv_block and conv_linear on a session that is not on the x_cnn
     queue (the public Session's case) against float64 formulas, and their
     gradients.
 12  lm_rows, lm_pick, lm_alive and lm_embedding: greedy picks, the eos
     stop and the padded finished row; the embedding rows of the picks.

Weights are a fixed finite pattern (norm weights 1): the properties hold for
any finite weights. Cross-vendor words of the same kernels are the release
verification's (the mamba, transformer, chunked LM head and MLP lanes);
this suite checks that the public route IS the model's route.

Serial by nature: each fixture opens sessions through the process-wide
session table (autograd/ops_api.mojo `_OPS_SESSIONS`), whose open and close
append to and pop from unlocked lists (core_program_loop_append says the
same), and the fixtures share one device context. Thirteen small fixtures, one
after another. No Python participates.
"""
from std.math import exp, log, tanh
from std.memory import bitcast
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE
from autograd.block_step_ops import (
    OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP, OPS_ATTN_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS,
    block_weight_sizes, block_state_sizes, block_part_floats, block_int_words,
)
from autograd.ops_block_frontend import (
    session_block_create, session_block_step, session_attention_option_words, session_lm_rows, session_lm_pick,
    session_lm_alive, session_rnn,
)
from sequence.ops import CELL_LSTM, CELL_GRU
from autograd.ops_frontend_impl import clip_high_unbounded
from autograd.optimizer_requests import OpsOptimizerRequest, OpsOptimizerExtras, ops_optimizer_codes
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_higher_common import tensor, total, values, gradient, same_words
from checks.native_tests.core_ops_common import dims2, normal, all_finite
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, check_cases_serial, require


comptime CORE_OPS_SESSION_BLOCKS_FIXTURES = 13

comptime _B = 2
comptime _L = 3
comptime _SMALL_MLP = 88
comptime _LM_HEAD = 87


def _block_ints(code: Int) -> List[Int]:
    """The step op's integers: [b, d_model] for the Mamba blocks (Mamba-2
    and Mamba-3 need d_model a multiple of 32: 64-wide heads over
    d_inner = 2 d_model); attention's eight (b, d_model, n_heads, n_kv,
    head_dim, intermediate, s_max, window 0 = full causal)."""
    if code == OPS_M1_STEP:
        var m1: List[Int] = [_B, 8]
        return m1^
    if code == OPS_ATTN_STEP:
        var at: List[Int] = [_B, 8, 2, 2, 4, 16, 8, 0]
        return at^
    var m23: List[Int] = [_B, 32]
    return m23^


def _block_fs(code: Int) -> List[Float32]:
    """Mamba-2's dt clamp, the public default (0, inf); nothing otherwise."""
    var fs = List[Float32]()
    if code == OPS_M2_STEP:
        fs.append(Float32(0.0))
        fs.append(clip_high_unbounded())
    return fs^


struct _Weights(Movable):
    """One block's host weights: every piece in one flat list, with each
    piece's start and size (`block_weight_sizes` order, absent pieces 0)."""
    var flat: List[Float32]
    var at: List[Int]
    var sizes: List[Int]

    def __init__(out self, code: Int, ints: List[Int], seed: Int) raises:
        var sizes = block_weight_sizes(code, ints)
        var at = List[Int](capacity=len(sizes))
        var count = 0
        for k in range(len(sizes)):
            at.append(count)
            count += sizes[k]
        var flat = List[Float32](length=count + 1, fill=Float32(0))
        for k in range(len(sizes)):
            # The norm weights (the first piece; attention's first two) are
            # 1; every other piece a small fixed pattern, finite and not
            # symmetric across rows or heads.
            var is_norm = k == 0 or (code == OPS_ATTN_STEP and k == 1)
            for i in range(sizes[k]):
                if is_norm:
                    flat[at[k] + i] = Float32(1.0)
                else:
                    flat[at[k] + i] = Float32((i * 7 + k * 5 + seed * 3 + 3) % 11 - 5) * Float32(0.03)
        self.flat = flat^
        self.at = at^
        self.sizes = sizes^

    def all_pieces(self) -> List[CastIn]:
        """One source per piece, an empty one for an absent piece: what a
        model program hands `upload_block`."""
        var out = List[CastIn](capacity=len(self.sizes))
        for k in range(len(self.sizes)):
            if self.sizes[k] > 0:
                out.append(CastIn.f32_c(Int(self.flat.unsafe_ptr()) + 4 * self.at[k], self.sizes[k], 1))
            else:
                out.append(CastIn.f32_c(0, 0, 1))
        return out^

    def present_pieces(self) -> List[CastIn]:
        """One source per piece that exists: what the public call hands over."""
        var out = List[CastIn]()
        for k in range(len(self.sizes)):
            if self.sizes[k] > 0:
                out.append(CastIn.f32_c(Int(self.flat.unsafe_ptr()) + 4 * self.at[k], self.sizes[k], 1))
        return out^


def _model_block[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, ints: List[Int], w: _Weights,
) raises -> List[Int]:
    """The model's own block: [weights, state, words (0: none)] handles made
    as `MambaBlockProgram.__init__` and the causal LM program make them."""
    var wh = s.upload_block(code, ints, BLOCK_PART_WEIGHTS, w.all_pieces(), List[Int]())
    _ = len(w.flat)  # keep-alive past the staging copies
    var ssizes = block_state_sizes(code, ints)
    var most = 1
    for k in range(len(ssizes)):
        most = max(most, ssizes[k])
    var zero = List[Float32](length=most, fill=Float32(0))
    var zsrc = List[CastIn](capacity=len(ssizes))
    for k in range(len(ssizes)):
        zsrc.append(CastIn.f32_c(Int(zero.unsafe_ptr()), ssizes[k], 1))
    var sh = s.upload_block(code, ints, BLOCK_PART_STATE, zsrc, List[Int]())
    _ = len(zero)  # keep-alive past the staging copies
    var qh = 0
    var nk = block_int_words(code)
    if nk > 0:
        qh = s.upload_block(code, ints, BLOCK_PART_WORDS, List[CastIn](), List[Int](length=nk, fill=0))
    if code == OPS_ATTN_STEP and ints[7] == 0:
        s.declare_append(sh, qh)
    var out: List[Int] = [wh, sh, qh]
    return out^


def _model_step[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, ints: List[Int], block: List[Int], x: Int,
) raises -> Int:
    var hs: List[Int] = [x, block[0], block[1]]
    if block[2] != 0:
        hs.append(block[2])
    return s.op(code, hs, ints.copy(), _block_fs(code))


def _public_step[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, code: Int, ints: List[Int], block: List[Int], x: Int,
) raises -> Int:
    """`Session.mamba_step` / `Session.attention_step` as the door calls them."""
    var config = List[Int]()
    config.append(code)
    for k in range(len(ints)):
        config.append(ints[k])
    var fs = _block_fs(code)
    var lo = fs[0] if len(fs) == 2 else Float32(0)
    var hi = fs[1] if len(fs) == 2 else Float32(0)
    return session_block_step[E, KIND](sid, x, block[0], block[1], block[2], config, lo, hi, code == OPS_ATTN_STEP)


def _model_out[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, ints: List[Int], block: List[Int], x: List[Float32], tokens: Int,
) raises -> List[Float32]:
    """Upload x (b, tokens, d_model), run the model's own step, read y back."""
    var d = _dims3(code, tokens)
    var xh = tensor(s, x, d)
    var yh = _model_step(s, code, ints, block, xh)
    return values(s, yh, d)


def _public_out[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, code: Int, ints: List[Int], block: List[Int], x: List[Float32], tokens: Int,
) raises -> List[Float32]:
    """Upload x (b, tokens, d_model), run the public step, read y back."""
    var s = ops_session[E, KIND](sid)
    var d = _dims3(code, tokens)
    var xh = tensor(s[], x, d)
    var yh = _public_step[E, KIND](sid, code, ints, block, xh)
    return values(s[], yh, d)


def _x(code: Int, tokens: Int, seed: Int) -> List[Float32]:
    """b * tokens * d_model input values in about [-0.7, 0.7]."""
    var dm = _block_ints(code)[1]
    var out = List[Float32](length=_B * tokens * dm, fill=Float32(0))
    for i in range(len(out)):
        out[i] = Float32((i * 5 + seed * 7 + 2) % 13 - 6) * Float32(0.11)
    return out^


def _dims3(code: Int, tokens: Int) -> List[Int]:
    var d: List[Int] = [_B, tokens, _block_ints(code)[1]]
    return d^


def _token(x: List[Float32], dm: Int, tokens: Int, t: Int) -> List[Float32]:
    """Token t of every row of a (b, tokens, d_model) list, as (b, 1, d_model)."""
    var out = List[Float32](length=_B * dm, fill=Float32(0))
    for r in range(_B):
        for j in range(dm):
            out[r * dm + j] = x[(r * tokens + t) * dm + j]
    return out^


def _block_tail[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, ints: List[Int], block: List[Int],
) raises -> List[List[Float32]]:
    """The carried state tensor and the state words of a block."""
    var out = List[List[Float32]]()
    out.append(values(s, block[1], dims2(1, block_part_floats(code, ints, BLOCK_PART_STATE))))
    if block[2] != 0:
        out.append(values(s, block[2], dims2(1, block_int_words(code))))
    return out^


def _same_route[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, code: Int) raises:
    """Fixtures 0-3: the public block and step against the model's own call."""
    _same_route_ints[E, KIND](sid, code, _block_ints(code))


def _same_route_ints[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, code: Int, ints: List[Int]) raises:
    """`_same_route` for the step op's integers `ints` (fixture 9 appends
    the attention block's option words)."""
    var s = ops_session[E, KIND](sid)
    var dm = ints[1]
    var w = _Weights(code, ints, 0)
    var model_block = _model_block(s[], code, ints, w)
    var public_block = session_block_create[E, KIND](sid, code, ints, w.present_pieces(), False)
    _ = len(w.flat)  # keep-alive past the staging copies
    var x3 = _x(code, _L, 1)
    var x1 = _x(code, 1, 2)
    var tag = "step op " + String(code)
    # a three-token prefill, then a one-token step that continues it
    var ym = _model_out(s[], code, ints, model_block, x3, _L)
    var yp = _public_out[E, KIND](sid, code, ints, public_block, x3, _L)
    all_finite(ym, tag + ": the model's prefill")
    same_words(yp, ym, tag + ": the public prefill differs from the model's own call")
    var zm = _model_out(s[], code, ints, model_block, x1, 1)
    var zp = _public_out[E, KIND](sid, code, ints, public_block, x1, 1)
    same_words(zp, zm, tag + ": the public step after the prefill differs from the model's own call")
    var tm = _block_tail(s[], code, ints, model_block)
    var tp = _block_tail(s[], code, ints, public_block)
    for k in range(len(tm)):
        same_words(tp[k], tm[k], tag + ": the public block's carried state or state words differ (part "
                   + String(k) + ")")
    if code == OPS_ATTN_STEP:
        # decoding token by token gives the prefill's words
        var fresh = session_block_create[E, KIND](sid, code, ints, w.present_pieces(), False)
        _ = len(w.flat)  # keep-alive past the staging copies
        for t in range(_L):
            var xt = _token(x3, dm, _L, t)
            var yt = _public_out[E, KIND](sid, code, ints, fresh, xt, 1)
            same_words(yt, _token(yp, dm, _L, t), tag + ": token " + String(t)
                       + " decoded alone differs from the prefill")


def _two_layers[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    """Fixture 4: a two-layer attention model through the public calls
    against the model's own two-layer loop on the same weights."""
    var s = ops_session[E, KIND](sid)
    var code = OPS_ATTN_STEP
    var ints = _block_ints(code)
    var w0 = _Weights(code, ints, 0)
    var w1 = _Weights(code, ints, 1)
    var m0 = _model_block(s[], code, ints, w0)
    var m1 = _model_block(s[], code, ints, w1)
    var p0 = session_block_create[E, KIND](sid, code, ints, w0.present_pieces(), False)
    var p1 = session_block_create[E, KIND](sid, code, ints, w1.present_pieces(), False)
    _ = len(w0.flat)  # keep-alive past the staging copies
    _ = len(w1.flat)
    var tokens: List[Int] = [_L, 1]
    for r in range(len(tokens)):
        var x = _x(code, tokens[r], 3 + r)
        var d = _dims3(code, tokens[r])
        var xm = tensor(s[], x, d)
        var hm = _model_step(s[], code, ints, m0, xm)
        var om = _model_step(s[], code, ints, m1, hm)
        var ym = values(s[], om, d)
        var xp = tensor(s[], x, d)
        var hp = _public_step[E, KIND](sid, code, ints, p0, xp)
        var op_ = _public_step[E, KIND](sid, code, ints, p1, hp)
        var yp = values(s[], op_, d)
        all_finite(ym, "two attention layers: the model's own forward")
        same_words(yp, ym, "two attention layers, run " + String(r)
                   + ": the composed model differs from the model's own forward")
    var tm = _block_tail(s[], code, ints, m1)
    var tp = _block_tail(s[], code, ints, p1)
    for k in range(len(tm)):
        same_words(tp[k], tm[k], "two attention layers: the second layer's cache or position differs")


def _any_nonzero(v: List[Float32]) -> Bool:
    for i in range(len(v)):
        if v[i] != Float32(0):
            return True
    return False


def _differs(a: List[Float32], b: List[Float32]) -> Bool:
    for i in range(len(a)):
        if a[i] != b[i]:
            return True
    return False


def _train_step_op[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, rid: Int, code: Int) raises:
    """Fixture 5, one step op: train through it in the recording session
    `rid`; `sid` is the inference session the recorded forward is compared
    with (`Session(record=...)`: "values are the same bits either way")."""
    var s = ops_session[E, KIND](rid)
    var ints = _block_ints(code)
    var w = _Weights(code, ints, 2)
    var block = session_block_create[E, KIND](rid, code, ints, w.present_pieces(), True)
    var plain = session_block_create[E, KIND](sid, code, ints, w.present_pieces(), False)
    _ = len(w.flat)  # keep-alive past the staging copies
    var tag = "training through step op " + String(code)
    var wdims = dims2(1, block_part_floats(code, ints, BLOCK_PART_WEIGHTS))
    var xv = _x(code, _L, 4)
    var xdims = _dims3(code, _L)
    var x = tensor(s[], xv, xdims, True)
    var runs = List[List[Float32]]()
    for r in range(2):
        var y = _public_step[E, KIND](rid, code, ints, block, x)
        var loss = total(s[], y)
        s[].backward(loss, False)
        var yv = values(s[], y, xdims)
        var gw = gradient(s[], block[0], wdims)
        var gx = gradient(s[], x, xdims)
        all_finite(yv, tag + ": the recorded output")
        all_finite(gw, tag + ": the packed weights' gradient")
        all_finite(gx, tag + ": x's gradient")
        require(_any_nonzero(gw), tag + ": the packed weights' gradient is all zero")
        require(_any_nonzero(gx), tag + ": x's gradient is all zero")
        runs.append(yv^)
        runs.append(gw^)
        runs.append(gx^)
        if r == 0:
            s[].reset()
    same_words(runs[3], runs[0], tag + ": a second recording gives another output")
    same_words(runs[4], runs[1], tag + ": a second recording gives another weight gradient")
    same_words(runs[5], runs[2], tag + ": a second recording gives another input gradient")
    # a recorded step starts from the zero state: the unrecorded step of a
    # fresh block on the same weights and tokens gives the same words
    var yi = _public_out[E, KIND](sid, code, ints, plain, xv, _L)
    same_words(runs[0], yi, tag + ": the recorded forward differs from the unrecorded one")
    var before = values(s[], block[0], wdims)
    var opt = s[].optimizer(OpsOptimizerRequest(1, False, 0, 1, Float32(0.01), Float32(0.0), Float32(0.9),
                                                Float32(0.999), Float32(1e-8), Float32(0.0), Float32(1.0),
                                                Float32(0.0)))
    var params: List[Int] = [block[0]]
    s[].step(opt, params)
    s[].reset()
    var after = values(s[], block[0], wdims)
    all_finite(after, tag + ": the packed weights after one Adam step")
    require(_differs(after, before), tag + ": one Adam step left the packed weights unchanged")
    # the updated block still runs
    var y2h = _public_step[E, KIND](rid, code, ints, block, x)
    var y2 = values(s[], y2h, xdims)
    all_finite(y2, tag + ": the output after the update")
    s[].reset()


def _close(a: Float32, e: Float64, what: String) raises:
    require(abs(Float64(a) - e) <= 2e-4 + 2e-4 * abs(e), what + ": got " + String(Float64(a)) + ", expected "
            + String(e))


def _small_mlp_and_head[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, rid: Int) raises:
    """Fixture 6: small_mlp and lm_head_loss, recorded and not."""
    var rows = 3
    var n_in = 4
    var hid = 5
    var n_out = 2
    var xv = normal(rows * n_in, 11)
    var w1 = normal(hid * n_in, 12)
    var b1 = normal(hid, 13)
    var w2 = normal(n_out * hid, 14)
    var b2 = normal(n_out, 15)
    var want = List[Float64](length=rows * n_out, fill=0)
    for r in range(rows):
        for o in range(n_out):
            var acc = Float64(b2[o])
            for h in range(hid):
                var pre = Float64(b1[h])
                for j in range(n_in):
                    pre += Float64(xv[r * n_in + j]) * Float64(w1[h * n_in + j])
                if pre > 0.0:
                    acc += pre * Float64(w2[o * hid + h])
            want[r * n_out + o] = acc
    var vocab = 6
    var width = 3
    var n_rows = 4
    var hv = normal(n_rows * width, 21)
    var wv = normal(vocab * width, 22)
    var codes: List[Float32] = [0.0, 5.0, 2.0, 3.0]
    var want_loss = Float64(0)
    for r in range(n_rows):
        var denom = Float64(0)
        var picked = Float64(0)
        for v in range(vocab):
            var logit = Float64(0)
            for j in range(width):
                logit += Float64(hv[r * width + j]) * Float64(wv[v * width + j])
            denom += exp(logit)
            if v == Int(codes[r]):
                picked = logit
        want_loss += log(denom) - picked
    want_loss = want_loss / Float64(n_rows)
    var sids: List[Int] = [sid, rid]
    var logits_runs = List[List[Float32]]()
    for m in range(2):
        var recording = m == 1
        var s = ops_session[E, KIND](sids[m])
        var hx = tensor(s[], xv, dims2(rows, n_in))
        var h1 = tensor(s[], w1, dims2(hid, n_in), recording)
        var hb1 = tensor(s[], b1, dims2(1, hid), recording)
        var h2 = tensor(s[], w2, dims2(n_out, hid), recording)
        var hb2 = tensor(s[], b2, dims2(1, n_out), recording)
        var hh = tensor(s[], hv, dims2(n_rows, width), recording)
        var hw = tensor(s[], wv, dims2(vocab, width), recording)
        var hc = tensor(s[], codes, dims2(n_rows, 1))
        # the small MLP: one recording, one backward
        var hs: List[Int] = [hx, h1, hb1, h2, hb2]
        var logits = s[].op(_SMALL_MLP, hs, List[Int](), List[Float32]())
        var got = values(s[], logits, dims2(rows, n_out))
        for i in range(len(got)):
            _close(got[i], want[i], "small_mlp logits at " + String(i))
        logits_runs.append(got^)
        if recording:
            var summed = total(s[], logits)
            s[].backward(summed, False)
            var db2 = gradient(s[], hb2, dims2(1, n_out))
            for o in range(n_out):
                require(db2[o] == Float32(rows), "small_mlp: db2 of a summed loss is the row count")
            all_finite(gradient(s[], h1, dims2(hid, n_in)), "small_mlp: dw1")
            require(_any_nonzero(gradient(s[], h2, dims2(n_out, hid))), "small_mlp: dw2 is all zero")
        s[].reset()
        # the LM head loss: its own recording and backward
        var lhs: List[Int] = [hh, hw, hc]
        var loss = s[].op(_LM_HEAD, lhs, List[Int](), List[Float32]())
        var lv = values(s[], loss, dims2(1, 1))
        _close(lv[0], want_loss, "lm_head_loss")
        if recording:
            s[].backward(loss, False)
            var dh = gradient(s[], hh, dims2(n_rows, width))
            var dw = gradient(s[], hw, dims2(vocab, width))
            all_finite(dh, "lm_head_loss: d hidden")
            all_finite(dw, "lm_head_loss: d weight")
            require(_any_nonzero(dh) and _any_nonzero(dw), "lm_head_loss: a gradient is all zero")
        s[].reset()
    same_words(logits_runs[1], logits_runs[0], "small_mlp: the recorded forward differs from the unrecorded one")


def _new_optimizers[E: Exec & Movable & Deinitable, KIND: Int](rid: Int) raises:
    """Fixture 7: Lion, Adamax and NAdam through the session optimizer."""
    var names: List[String] = ["lion", "adamax", "nadam"]
    var kinds: List[Int] = [7, 8, 9]
    var s = ops_session[E, KIND](rid)
    var start: List[Float32] = [1.0, -2.0, 0.5, -0.25]
    var lr = Float32(0.125)
    for k in range(3):
        var codes = ops_optimizer_codes(names[k], "constant")
        require(codes[0] == kinds[k] and codes[1] == 0, "optimizer name code of " + names[k])
        var p = tensor(s[], start, dims2(4, 1), True)
        var opt = s[].optimizer(OpsOptimizerRequest(kinds[k], False, 0, 1, lr, Float32(0.0), Float32(0.9),
                                                    Float32(0.99), Float32(1e-8), Float32(0.0), Float32(1.0),
                                                    Float32(0.0)))
        var summed = total(s[], p)
        s[].backward(summed, False)  # the gradient of a sum: 1 for every element
        var params: List[Int] = [p]
        s[].step(opt, params)
        s[].reset()
        var got = values(s[], p, dims2(4, 1))
        all_finite(got, names[k] + ": the parameters after one update")
        for i in range(4):
            if kinds[k] == 7:
                # Lion: p - lr sign((1 - beta1) g) with g = 1
                _close(got[i], Float64(start[i]) - Float64(lr), "lion update at " + String(i))
            elif kinds[k] == 8:
                # Adamax: m / (1 - beta1) = g, u = |g| + eps
                _close(got[i], Float64(start[i]) - Float64(lr) / (1.0 + 1e-8), "adamax update at " + String(i))
            else:
                require(got[i] != start[i], "nadam left a parameter unchanged")
        var state = s[].optimizer_state_value(opt, p)
        require(state.step == 1 and state.kind == kinds[k], names[k] + ": the state's step count or kind")
        require(state.first != 0, names[k] + ": the first moment buffer is missing")
        if kinds[k] == 7:
            require(state.second == 0, "lion keeps one state buffer")
        else:
            require(state.second != 0, names[k] + ": the second state buffer is missing")
        if kinds[k] == 9:
            require(state.mu_product != Float32(1.0), "nadam: the mu product did not advance")
        s[].reset()
        s[].free(p)
    var refused = False
    try:
        _ = ops_optimizer_codes("rprop", "constant")
    except:
        refused = True
    require(refused, "an unknown optimizer kind was accepted")


def _sgd_request(kind: Int, lr: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(kind, False, 0, 1, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               Float32(0.0), Float32(1.0), Float32(0.0))


def _extras_and_clips[E: Exec & Movable & Deinitable, KIND: Int](rid: Int) raises:
    """Fixture 8: RMSprop, Adagrad, clip_grad_norm over two tensors, clip_grad_value."""
    var s = ops_session[E, KIND](rid)
    var start: List[Float32] = [1.0, -2.0, 0.5, -0.25]
    var lr = Float32(0.125)
    var names: List[String] = ["rmsprop", "adagrad"]
    for k in range(2):
        var codes = ops_optimizer_codes(names[k], "constant")
        require(codes[0] == 3 + k, "optimizer name code of " + names[k])
        var p = tensor(s[], start, dims2(4, 1), True)
        var opt = s[].optimizer_with(_sgd_request(3 + k, lr),
                                     OpsOptimizerExtras(Float32(0.99), False, Float32(0.0), Float32(0.0),
                                                        Float32(0.0), Float32(0.0)))
        var summed = total(s[], p)
        s[].backward(summed, False)
        var params: List[Int] = [p]
        s[].step(opt, params)
        s[].reset()
        var got = values(s[], p, dims2(4, 1))
        # g = 1. RMSprop: v = (1 - alpha) g^2 = 0.01, p -= lr g / (sqrt(v) + eps);
        # Adagrad: sum = g^2 = 1, p -= lr g / (sqrt(sum) + eps)
        var move = Float64(lr) / (1.0 + 1e-8)
        if k == 0:
            move = Float64(lr) / (0.1 + 1e-8)
        for i in range(4):
            _close(got[i], Float64(start[i]) - move, names[k] + " update at " + String(i))
        var state = s[].optimizer_state_value(opt, p)
        require(state.step == 1 and state.second != 0, names[k] + ": the state's step count or squared-gradient buffer")
        s[].reset()
        s[].free(p)
    # clip_grad_norm over two tensors, then clip_grad_value
    var a0 = List[Float32](length=4, fill=Float32(2.0))
    var b0 = List[Float32](length=5, fill=Float32(-1.0))
    for m in range(2):
        var pa = tensor(s[], a0, dims2(4, 1), True)
        var pb = tensor(s[], b0, dims2(5, 1), True)
        var clip_norm = Float32(1.5) if m == 0 else Float32(0.0)
        var clip_value = Float32(0.0) if m == 0 else Float32(0.25)
        var extras = OpsOptimizerExtras(Float32(0.99), False, Float32(0.0), Float32(0.0), clip_norm, clip_value)
        var opt = s[].optimizer_with(_sgd_request(0, Float32(1.0)), extras)
        var ta = total(s[], pa)
        var tb = total(s[], pb)
        var hs: List[Int] = [ta, tb]
        var both = s[].op(2, hs, List[Int](), List[Float32]())  # add
        s[].backward(both, False)
        var params: List[Int] = [pa, pb]
        s[].step(opt, params)
        s[].reset()
        var ga = values(s[], pa, dims2(4, 1))
        var gb = values(s[], pb, dims2(5, 1))
        var move = 0.25
        var what = String("clip_grad_value")
        if m == 0:
            move = 1.5 / (3.0 + 1e-6)
            what = String("clip_grad_norm over two tensors")
        for i in range(4):
            _close(ga[i], 2.0 - move, what + ", first tensor at " + String(i))
        for i in range(5):
            _close(gb[i], -1.0 - move, what + ", second tensor at " + String(i))
        s[].reset()
        s[].free(pa)
        s[].free(pb)


def _option_words(ints: List[Int], qkv_bias: Bool, o_bias: Bool, norm_bias: Bool, mlp_bias: Bool) raises -> List[Int]:
    """The public option words of the fixture's block (head_dim ints[4],
    s_max ints[6]) with the four bias switches; everything else default."""
    return session_attention_option_words(
        ints[4], ints[6], "rmsnorm", "swiglu", Float32(10000.0), "", Float32(1.0), Float32(1.0), Float32(4.0),
        8192, 0, qkv_bias, o_bias, Float32(1e-6), norm_bias, mlp_bias, False, Float32(0.0))


def _attention_options[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    """Fixture 9: the attention block with q/k/v, output and MLP biases."""
    var ints = _block_ints(OPS_ATTN_STEP)
    var words = _option_words(ints, True, True, False, True)
    require(len(words) == 17, "attention options: a non-default record sends seventeen words")
    var plain = _option_words(ints, False, False, False, False)
    require(len(plain) == 0, "attention options: the default record sends no words")
    var refused = False
    try:
        _ = _option_words(ints, False, False, True, False)
    except:
        refused = True
    require(refused, "attention options: norm_bias without layernorm was accepted")
    for k in range(len(words)):
        ints.append(words[k])
    _same_route_ints[E, KIND](sid, OPS_ATTN_STEP, ints)


def _sigmoid(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-v))


def _recurrent[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, rid: Int) raises:
    """Fixture 10: lstm and gru."""
    var steps = 2
    var batch = 2
    var d_in = 3
    var hid = 4
    var xv = normal(steps * batch * d_in, 31)
    var cells: List[Int] = [CELL_LSTM, CELL_GRU]
    for c in range(2):
        var gates = 4 if c == 0 else 3
        var gh = gates * hid
        var wih = normal(gh * d_in, 32 + c)
        var whh = normal(gh * hid, 34 + c)
        var bih = normal(gh, 36 + c)
        var bhh = normal(gh, 38 + c)
        var name = String("lstm")
        if c == 1:
            name = String("gru")
        # step 1 from the zero state, PyTorch's formulas in float64
        var want = List[Float64](length=batch * hid, fill=0)
        for r in range(batch):
            for j in range(hid):
                var pre = List[Float64](length=gates, fill=0)
                for g in range(gates):
                    var acc = Float64(bih[g * hid + j])
                    for i in range(d_in):
                        acc += Float64(xv[r * d_in + i]) * Float64(wih[(g * hid + j) * d_in + i])
                    pre[g] = acc
                if c == 0:
                    # gates i, f, g, o; h_0 = c_0 = 0: every gate adds b_hh
                    var gi = _sigmoid(pre[0] + Float64(bhh[j]))
                    var gg = tanh(pre[2] + Float64(bhh[2 * hid + j]))
                    var go = _sigmoid(pre[3] + Float64(bhh[3 * hid + j]))
                    want[r * hid + j] = go * tanh(gi * gg)
                else:
                    # gates r, z, n: n = tanh(W_in x + b_in + r b_hn), h = (1 - z) n
                    var gr = _sigmoid(pre[0] + Float64(bhh[j]))
                    var gz = _sigmoid(pre[1] + Float64(bhh[hid + j]))
                    var gn = tanh(pre[2] + gr * Float64(bhh[2 * hid + j]))
                    want[r * hid + j] = (1.0 - gz) * gn
        var sids: List[Int] = [sid, rid]
        var runs = List[List[Float32]]()
        var odims: List[Int] = [steps, batch, hid]
        var xdims: List[Int] = [steps, batch, d_in]
        for m in range(2):
            var recording = m == 1
            var s = ops_session[E, KIND](sids[m])
            var hx = tensor(s[], xv, xdims, recording)
            var h1 = tensor(s[], wih, dims2(gh, d_in), recording)
            var h2 = tensor(s[], whh, dims2(gh, hid), recording)
            var h3 = tensor(s[], bih, dims2(1, gh), recording)
            var h4 = tensor(s[], bhh, dims2(1, gh), recording)
            var y = session_rnn[E, KIND](sids[m], cells[c], hx, h1, h2, h3, h4)
            var got = values(s[], y, odims)
            all_finite(got, name + ": the outputs")
            for i in range(batch * hid):
                _close(got[i], want[i], name + " step 1 at " + String(i))
            if recording:
                var summed = total(s[], y)
                s[].backward(summed, False)
                var gw = gradient(s[], h1, dims2(gh, d_in))
                var gx = gradient(s[], hx, xdims)
                all_finite(gw, name + ": d weight_ih")
                all_finite(gx, name + ": d x")
                all_finite(gradient(s[], h2, dims2(gh, hid)), name + ": d weight_hh")
                require(_any_nonzero(gw) and _any_nonzero(gx), name + ": a gradient is all zero")
            runs.append(got^)
            s[].reset()
        same_words(runs[1], runs[0], name + ": the recorded forward differs from the unrecorded one")


def _conv_ops[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, rid: Int) raises:
    """Fixture 11: conv_block (3 x 3 over one 4 x 4 image, 2 x 2 pool) and conv_linear."""
    var img = normal(16, 41)
    var ker = normal(9, 42)
    var cb: List[Float32] = [0.25]
    var best = Float64(0)
    for i in range(2):
        for j in range(2):
            var acc = Float64(cb[0])
            for a in range(3):
                for b in range(3):
                    acc += Float64(img[(i + a) * 4 + (j + b)]) * Float64(ker[a * 3 + b])
            if acc < 0.0:
                acc = 0.0
            if (i == 0 and j == 0) or acc > best:
                best = acc
    var lw: List[Float32] = [0.5, -1.5]
    var lb: List[Float32] = [0.125, 0.75]
    var sids: List[Int] = [sid, rid]
    for m in range(2):
        var recording = m == 1
        var s = ops_session[E, KIND](sids[m])
        s[].steps.cnn_other_queue = True  # as a public Session: not on the x_cnn queue
        var xdims: List[Int] = [1, 1, 4, 4]
        var kdims: List[Int] = [1, 1, 3, 3]
        var odims: List[Int] = [1, 1, 1, 1]
        var hx = tensor(s[], img, xdims)
        var hk = tensor(s[], ker, kdims, recording)
        var hb = tensor(s[], cb, dims2(1, 1), recording)
        var hw = tensor(s[], lw, dims2(2, 1), recording)
        var hlb = tensor(s[], lb, dims2(1, 2), recording)
        var bh: List[Int] = [hx, hk, hb]
        var bi: List[Int] = [0, 1, 2]
        var pooled = s[].op(90, bh, bi, List[Float32]())
        var pv = values(s[], pooled, odims)
        _close(pv[0], best, "conv_block")
        var lh: List[Int] = [pooled, hw, hlb]
        var logits = s[].op(91, lh, List[Int](), List[Float32]())
        var lv = values(s[], logits, dims2(1, 2))
        for o in range(2):
            _close(lv[o], best * Float64(lw[o]) + Float64(lb[o]), "conv_linear at " + String(o))
        if recording:
            var summed = total(s[], logits)
            s[].backward(summed, False)
            var dlb = gradient(s[], hlb, dims2(1, 2))
            require(dlb[0] == Float32(1.0) and dlb[1] == Float32(1.0), "conv_linear: d bias of a summed output is 1")
            all_finite(gradient(s[], hk, kdims), "conv_block: d weight")
            all_finite(gradient(s[], hb, dims2(1, 1)), "conv_block: d bias")
        s[].reset()


def _generation[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    """Fixture 12: lm_rows, lm_pick, lm_alive, lm_embedding."""
    var s = ops_session[E, KIND](sid)
    var eos: List[Int] = [1]
    var rows = session_lm_rows[E, KIND](sid, 2, eos)
    var table = normal(5 * 3, 51)
    var th = tensor(s[], table, dims2(5, 3))
    # run 0: row 0 picks 3, row 1 picks the eos id 1
    var l0: List[Float32] = [0.0, 0.5, -1.0, 2.0, 1.0, 0.25, 3.0, -2.0, 0.0, 1.5]
    # run 1: row 0 picks the eos id; row 1 is finished and gets the pad id 0
    var l1: List[Float32] = [0.0, 4.0, -1.0, 2.0, 1.0, 9.0, 3.0, -2.0, 7.0, 1.5]
    var first: List[Int] = [3, 1]
    var second: List[Int] = [1, 0]
    for r in range(2):
        var lh = 0
        if r == 0:
            lh = tensor(s[], l0, dims2(2, 5))
        else:
            lh = tensor(s[], l1, dims2(2, 5))
        var ids = session_lm_pick[E, KIND](sid, lh, rows, 0, False, 0, 0, 0, Float32(1.0), Float32(1.0))
        var iv = values(s[], ids, dims2(2, 1))
        var alive = session_lm_alive[E, KIND](sid, ids, rows)
        var av = values(s[], alive, dims2(1, 1))
        var eh: List[Int] = [th, ids]
        var pad: List[Int] = [-1]
        var emb = s[].op(93, eh, pad, List[Float32]())
        var ev = values(s[], emb, dims2(2, 3))
        for b in range(2):
            var want = first[b]
            if r == 1:
                want = second[b]
            require(Int(bitcast[DType.int32](iv[b])) == want, "lm_pick: run " + String(r) + " row " + String(b))
            for j in range(3):
                require(ev[b * 3 + j] == table[want * 3 + j], "lm_embedding: the picked id's row")
        var want_alive = Float32(1.0)
        if r == 1:
            want_alive = Float32(0.0)
        require(av[0] == want_alive, "lm_alive after run " + String(r))



def run_core_ops_session_blocks_fixture[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, rid: Int, fixture_id: Int,
) raises:
    """`sid` an inference session, `rid` a recording one, both fresh."""
    require(fixture_id < CORE_OPS_SESSION_BLOCKS_FIXTURES, "unknown session block fixture")
    var step_codes: List[Int] = [OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP, OPS_ATTN_STEP]
    if fixture_id < 4:
        _same_route[E, KIND](sid, step_codes[fixture_id])
    elif fixture_id == 4:
        _two_layers[E, KIND](sid)
    elif fixture_id == 5:
        for k in range(len(step_codes)):  # driver-loop: over the four step ops, each on its own block of the one recording session
            _train_step_op[E, KIND](sid, rid, step_codes[k])
    elif fixture_id == 6:
        _small_mlp_and_head[E, KIND](sid, rid)
    elif fixture_id == 7:
        _new_optimizers[E, KIND](rid)
    elif fixture_id == 8:
        _extras_and_clips[E, KIND](rid)
    elif fixture_id == 9:
        _attention_options[E, KIND](sid)
    elif fixture_id == 10:
        _recurrent[E, KIND](sid, rid)
    elif fixture_id == 11:
        _conv_ops[E, KIND](sid, rid)
    else:
        _generation[E, KIND](sid)


# ---- The host and device columns ----
# The fixtures run one after another on both columns: each opens its sessions through the process-wide session table
# (as the old host and device loops did).
def test_ops_session_blocks() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_fixture(fixture_id: Int) raises:
            var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
            var rid = ops_session_open[HostExec, OPS_HOST](HostExec(), True)
            try:
                run_core_ops_session_blocks_fixture[HostExec, OPS_HOST](sid, rid, fixture_id)
            except error:
                ops_session_close[HostExec, OPS_HOST](rid)
                ops_session_close[HostExec, OPS_HOST](sid)
                raise error
            ops_session_close[HostExec, OPS_HOST](rid)
            ops_session_close[HostExec, OPS_HOST](sid)
        check_cases_serial("ops_session_blocks", CORE_OPS_SESSION_BLOCKS_FIXTURES, host_fixture)
    else:
        def device_fixture(fixture_id: Int) raises:
            var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
            var rid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), True)
            try:
                run_core_ops_session_blocks_fixture[DeviceExec, OPS_DEVICE](sid, rid, fixture_id)
            except error:
                ops_session_close[DeviceExec, OPS_DEVICE](rid)
                ops_session_close[DeviceExec, OPS_DEVICE](sid)
                raise error
            ops_session_close[DeviceExec, OPS_DEVICE](rid)
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
        check_cases_serial("ops_session_blocks", CORE_OPS_SESSION_BLOCKS_FIXTURES, device_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
