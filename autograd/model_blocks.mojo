# SPDX-License-Identifier: Apache-2.0
"""The neural building blocks of a Sequential description (lane neural-blocks, 2026-11-22).

Andrew, 2026-11-22: the Transformer, Mamba, Samba, CausalLM and byte-LM blocks are monolithic; break them
into pieces people compose (Sequential layers and its residual / parallel / gated containers), on autograd,
on every vendor and the CPU column. Each layer here is a thin declaration over an EXISTING tape op and its
EXISTING backward rule; no kernel is written or edited here:

  code  layer                    tape op (file)                                        rules
  32    silu                     imp_unary IMP_SILU (autograd/import_api.mojo)          ImportRules 192
  33    gelu (a0 1 = tanh form)  imp_unary IMP_GELU_ERF / IMP_GELU_TANH                ImportRules 192
  34    softplus                 imp_unary IMP_SOFTPLUS                                 ImportRules 192
  35    rms_norm (f eps)         graph_rms_norm (autograd/transformer_primitives.mojo)  RankNRules 208
  36    rotary (a0 head_dim,     graph_rotary, positions 0..T-1 of the (T, width)       RankNRules 209
        f theta)                 sample
  37    multiply (a0 source)     tape.multiply with an earlier output (the gate of a    tape (MULTIPLY)
                                 gated MLP / SwiGLU / Mamba's y * silu(z))
  38    narrow (axis, start,     rankn_narrow (autograd/rankn_layout.mojo): Mamba's     RankNRules 160..
        length)                  causal crop after a padded depthwise conv1d
  39    embedding (num, dim)     imp_gather_nd over the (num, dim) table by the         tape EMBEDDING
                                 sample's float ids (OP_IMP_FLAT_IDS, then
                                 tape.embedding)
  40    attention (heads,        Q/K/V/O tape.linear, view + permute (B, Hk, G, T, hd), tape / RankNRules /
        kv_heads, rotary 0/1,    graph_rotary, rankn_matmul, tape.scale, the causal     ImportRules
        window, causal 0/1;      (or sliding-window) mask by imp_binary / imp_where on
        f theta)                 device iota positions, tape.softmax, rankn_matmul
  41    recurrent (hidden, cell) rnn_scan_op (autograd/rules_recurrent.mojo), the       RecurrentRules
                                 recurrent estimators' own layer, on the time-major      16..18
                                 view (permute (T, B, F))
  42    selective_scan (state)   graph_selective_scan (autograd/ssm_primitives.mojo),   SsmRules 240
                                 the Mamba-1 core as a generic op (lane neural-blocks-2,
                                 2026-11-22): the (length, 2 dim + 2 state) input is
                                 [u | delta | B | C] (narrow views), parameters A_log
                                 (dim, state) and D (dim); its arithmetic is the Mamba-1
                                 block kernels' own thread bodies
  43    ssd_scan (heads)         graph_ssd_scan (autograd/ssm_m2_primitives.mojo),      SsmRules 241
                                 the Mamba-2 SSD core (lane neural-blocks-3): the
                                 (length, 65 heads + 256) input is [x | B | C | dt],
                                 parameters A_log and D (heads); the block's bodies
  44    trapezoid_scan (heads,   graph_trapezoid_scan (autograd/ssm_m3_primitives.mojo), SsmRules 242
        state)                   the Mamba-3 trapezoid core as a generic op (lane
                                 neural-blocks-3-m3, 2026-11-22): the (length, heads (P +
                                 2 state + 3)) input is [v | k | q | dt | A | trap],
                                 parameter D (heads); its arithmetic is the Mamba-3
                                 decode step's bodies (autograd/kernels/
                                 mamba3_trapezoid_body.mojo)
  45    lm_head (vocab)          graph_transformer_linear (the logits, h @ weight^T);   TransformerPrimitive
                                 under the token_cross_entropy loss the fit stops      Rules; StepOpRules
                                 before the layer and runs graph_lm_head_v2 (the fused  115 (lm_head_v2)
                                 head + loss, no logits); weight (vocab, features)
                                 (lane neural-blocks-3-lm, 2026-11-22)

`ModelRules` (below) is the backward of a version-2 description: ImportRules (the import family plus
RankNRules, which holds NeuralRules and TransformerPrimitiveRules) plus RecurrentRules. Every tag the
description could record before this lane is answered exactly as RankNRules answered it, so existing
models keep their bits; the only added cost is the tag-range branch per custom node in backward.

The monolithic blocks (TransformerBlock, Mamba*Block, SambaStack, CausalLM, the byte LM trainer) are NOT
routed through these layers: they keep their fused, hand-tuned kernels as the fast presets (AGENTS.md "No
refactor adds cost on a hot path", "A measured winner is never deleted on reasoning"). A composed model and
its monolith agree to the fold order of their kernels, not bit for bit
(checks/native_tests/neural_blocks.mojo states the measured differences).
"""
from std.collections import Array
from std.math import isfinite
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_SEQ_IOTA, OP_FILL, CELL_RNN_TANH, CELL_LSTM, CELL_GRU, gates_of
from checks.numerics import ftz, identical_div, identical_sqrt
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import rn_shape, rn_dims, rn_with_batch
from autograd.rankn_bmm import rankn_matmul
from autograd.rankn_layout import rankn_narrow
from autograd.rankn_fold import rankn_binary
from autograd.ops import SUBTRACT
from autograd.transformer_primitives import graph_rms_norm, graph_rotary
# lm_head (lane neural-blocks-3-lm): the logits are the transformer projection op (the neural GEMM cell, the
# cell the fused head's panels run); the token loss is the fused op's one recording
from autograd.transformer_primitives import graph_transformer_linear
from autograd.lm_head_v2_graph import graph_lm_head_v2
from autograd.lm_head_v2_op import LM_HEAD_V2_TAG
from autograd.rules_step_ops import StepOpRules
from autograd.import_api import ImportRules, imp_unary, imp_binary, imp_where, imp_gather_nd
from autograd.import_ops import IMP_SILU, IMP_GELU_ERF, IMP_GELU_TANH, IMP_SOFTPLUS, IMP_GT, IMP_GE, IMP_OR
from autograd.rules_recurrent import RecurrentRules, RNN_LINEAR, RNN_LAYER, RNN_LAST_STEP, rnn_scan_op
from autograd.ssm_primitives import SsmRules, SSM_TAG_FIRST, SSM_TAG_LAST, graph_selective_scan
from autograd.ssm_ops import ssm_m1_states_supported
# Mamba-2 (lane neural-blocks-3)
from autograd.ssm_m2_primitives import graph_ssd_scan, SSD_HEADDIM, SSD_STATES
# Mamba-3 (lane neural-blocks-3-m3)
from autograd.ssm_m3_primitives import graph_trapezoid_scan

comptime LAYER_SILU = 32
comptime LAYER_GELU = 33
comptime LAYER_SOFTPLUS = 34
comptime LAYER_RMS_NORM = 35
comptime LAYER_ROTARY = 36
comptime LAYER_MULTIPLY = 37
comptime LAYER_NARROW = 38
comptime LAYER_EMBEDDING = 39
comptime LAYER_ATTENTION = 40
comptime LAYER_RECURRENT = 41
comptime LAYER_SELECTIVE_SCAN = 42
# lane neural-blocks-3 (2026-11-22): the Mamba-2 SSD core, the Mamba-3 trapezoid core, the LM head
comptime LAYER_SSD_SCAN = 43
comptime LAYER_TRAPEZOID_SCAN = 44
comptime LAYER_LM_HEAD = 45
comptime BLOCK_FIRST = 32
comptime BLOCK_LAST = 45

#: block_init_bounds' marker for Mamba's S4D-real init of A_log: row d holds log(1), .., log(state)
comptime BLOCK_INIT_S4D = Float32(-1.0e9)

#: recurrent cells (a1 of LAYER_RECURRENT): the public names rnn, lstm, gru
comptime BLOCK_CELL_RNN = 0
comptime BLOCK_CELL_LSTM = 1
comptime BLOCK_CELL_GRU = 2

#: the score written at a masked attention position before the row softmax: exp(MASKED - max) is +0.0
#: for every finite row max, as a -inf fill would give, without an infinity in the saved scores
comptime ATTENTION_MASKED = Float32(-1.0e30)

comptime _ARGS = 6


def is_block(code: Int) -> Bool:
    return code >= BLOCK_FIRST and code <= BLOCK_LAST


def _x(xargs: List[Int], i: Int, k: Int) -> Int:
    return xargs[_ARGS * i + k]


def _cell_of(which: Int) -> Int:
    """The sequence/ops.mojo cell code of a public cell (BLOCK_CELL_*); -1 when unknown."""
    if which == BLOCK_CELL_RNN:
        return CELL_RNN_TANH
    if which == BLOCK_CELL_LSTM:
        return CELL_LSTM
    if which == BLOCK_CELL_GRU:
        return CELL_GRU
    return -1


def _cell(which: Int) raises -> Int:
    var c = _cell_of(which)
    if c < 0:
        raise Error("recurrent: the cell must be rnn, lstm or gru")
    return c


def _heads(xargs: List[Int], i: Int) -> Tuple[Int, Int]:
    var h = _x(xargs, i, 0)
    var hk = _x(xargs, i, 1)
    return (h, h if hk == 0 else hk)


# ---------------------------------------------------------------- description
def block_shape(code: Int, xargs: List[Int], i: Int, f: Float32, cur: ShapeN, name: String) raises -> ShapeN:
    """Validate layer i (a block code) on its per-sample input shape `cur`; its output shape."""
    var a0 = _x(xargs, i, 0)
    var a1 = _x(xargs, i, 1)
    if code == LAYER_SILU or code == LAYER_SOFTPLUS or code == LAYER_MULTIPLY:
        # (multiply's source shape is checked by the interpreter, autograd/model.mojo)
        return cur
    if code == LAYER_GELU:
        if a0 != 0 and a0 != 1:
            raise Error(name + " (GELU): approximate must be 0 (erf) or 1 (tanh)")
        return cur
    if code == LAYER_RMS_NORM:
        if not (f > Float32(0)) or not isfinite(f):
            raise Error(name + " (RMSNorm): epsilon must be positive")
        return cur
    if code == LAYER_ROTARY:
        if cur.ndim < 2:
            raise Error(name + " (Rotary): the input needs (length, features) axes")
        if a0 < 2 or a0 % 2 != 0 or cur.cols() % a0 != 0:
            raise Error(name + " (Rotary): head_dim must be even and divide the features")
        if not (f > Float32(0)) or not isfinite(f):
            raise Error(name + " (Rotary): theta must be positive")
        return cur
    if code == LAYER_NARROW:
        var a2 = _x(xargs, i, 2)
        if a0 < 0 or a0 >= cur.ndim:
            raise Error(name + " (Narrow): axis must be a sample axis")
        if a2 < 1 or a1 < 0 or a1 + a2 > cur.d[a0]:
            raise Error(name + " (Narrow): [start, start + length) must lie inside the axis")
        var nd = rn_dims(cur)
        nd[a0] = a2
        return rn_shape(nd, name + " (Narrow)")
    if code == LAYER_EMBEDDING:
        if a0 < 1 or a1 < 1:
            raise Error(name + " (Embedding): num_embeddings and embedding_dim must be positive")
        if cur.ndim > RN_MAX_RANK - 2:
            raise Error(name + " (Embedding): the ids have too many axes")
        var ed = rn_dims(cur)
        ed.append(a1)
        return rn_shape(ed, name + " (Embedding)")
    if code == LAYER_ATTENTION:
        if cur.ndim != 2:
            raise Error(name + " (Attention): the input must be (length, features)")
        var hh = _heads(xargs, i)
        var h = hh[0]
        var hk = hh[1]
        var d = cur.d[1]
        if h < 1 or hk < 1 or h % hk != 0 or d % h != 0:
            raise Error(name + " (Attention): heads must divide the features and kv_heads must divide heads")
        var hd = d // h
        var rot = _x(xargs, i, 2)
        var window = _x(xargs, i, 3)
        var causal = _x(xargs, i, 4)
        if (rot != 0 and rot != 1) or (causal != 0 and causal != 1) or window < 0:
            raise Error(name + " (Attention): rotary and causal must be 0 or 1, window >= 0")
        if window > 0 and causal == 0:
            raise Error(name + " (Attention): a sliding window needs causal attention")
        if rot == 1 and (hd % 2 != 0 or not (f > Float32(0)) or not isfinite(f)):
            raise Error(name + " (Attention): rotary needs an even head size and a positive theta")
        return cur
    if code == LAYER_RECURRENT:
        if cur.ndim != 2:
            raise Error(name + " (Recurrent): the input must be (length, features)")
        _ = _cell(a1)
        if a0 < 1:
            raise Error(name + " (Recurrent): hidden_size must be positive")
        var rd = List[Int]()
        rd.append(cur.d[0])
        rd.append(a0)
        return rn_shape(rd, name + " (Recurrent)")
    if code == LAYER_SELECTIVE_SCAN:
        if cur.ndim != 2:
            raise Error(name + " (SelectiveScan): the input must be (length, features)")
        if not ssm_m1_states_supported(a0):
            raise Error(name + " (SelectiveScan): d_state must be 4, 8 or 16")
        var wide = cur.d[1] - 2 * a0
        if wide < 2 or wide % 2 != 0:
            raise Error(name + " (SelectiveScan): the features must be 2 * dim + 2 * d_state, [u | delta | B | C]")
        var sd = List[Int]()
        sd.append(cur.d[0])
        sd.append(wide // 2)
        return rn_shape(sd, name + " (SelectiveScan)")
    # Mamba-2 (lane neural-blocks-3): the SSD core over [x | B | C | dt] features
    if code == LAYER_SSD_SCAN:
        if cur.ndim != 2:
            raise Error(name + " (SsdScan): the input must be (length, features)")
        if a0 < 1:
            raise Error(name + " (SsdScan): heads must be at least 1")
        if cur.d[1] != a0 * (SSD_HEADDIM + 1) + 2 * SSD_STATES:
            raise Error(name + " (SsdScan): the features must be heads * 64 + 256 + heads, [x | B | C | dt]")
        var od = List[Int]()
        od.append(cur.d[0])
        od.append(a0 * SSD_HEADDIM)
        return rn_shape(od, name + " (SsdScan)")
    # Mamba-3 (lane neural-blocks-3-m3)
    if code == LAYER_TRAPEZOID_SCAN:
        if cur.ndim != 2:
            raise Error(name + " (TrapezoidScan): the input must be (length, features)")
        var th = a0
        var tn = _x(xargs, i, 1)
        if th < 1 or tn < 1:
            raise Error(name + " (TrapezoidScan): heads and d_state must be positive")
        var vw = cur.d[1] - th * (2 * tn + 3)
        if vw < th or vw % th != 0:
            raise Error(name + " (TrapezoidScan): the features must be heads * (headdim + 2 * d_state + 3), "
                        + "[v | k | q | dt | A | trap]")
        var td = List[Int]()
        td.append(cur.d[0])
        td.append(vw)
        return rn_shape(td, name + " (TrapezoidScan)")
    # lm_head (lane neural-blocks-3-lm)
    if code == LAYER_LM_HEAD:
        if cur.ndim != 1 and cur.ndim != 2:
            raise Error(name + " (LMHead): the input must be (features,) or (length, features)")
        if a0 < 2:
            raise Error(name + " (LMHead): vocab must be at least 2")
        var hd = rn_dims(cur)
        hd[cur.ndim - 1] = a0
        return rn_shape(hd, name + " (LMHead)")
    # end lm_head
    raise Error(name + ": unknown layer code " + String(code))


def _scan_dim(xargs: List[Int], i: Int, cur: ShapeN) -> Int:
    return (cur.d[1] - 2 * _x(xargs, i, 0)) // 2


def _trap_v(xargs: List[Int], i: Int, cur: ShapeN) -> Int:
    """The trapezoid scan's v width, heads * headdim."""
    return cur.d[1] - _x(xargs, i, 0) * (2 * _x(xargs, i, 1) + 3)


def block_param_sizes(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Int]:
    """The element counts of layer i's parameter tensors, in the flat vector's order (empty: none)."""
    var out = List[Int]()
    if code == LAYER_RMS_NORM:
        out.append(cur.cols())
    elif code == LAYER_EMBEDDING:
        out.append(_x(xargs, i, 0) * _x(xargs, i, 1))
    elif code == LAYER_ATTENTION:
        var hh = _heads(xargs, i)
        var d = cur.d[1]
        var hd = d // hh[0]
        out.append(d * hh[0] * hd)   # q: (features, heads * head_dim)
        out.append(d * hh[1] * hd)   # k: (features, kv_heads * head_dim)
        out.append(d * hh[1] * hd)   # v
        out.append(hh[0] * hd * d)   # o: (heads * head_dim, features)
    elif code == LAYER_RECURRENT:
        var hsz = _x(xargs, i, 0)
        var gh = gates_of(_cell_of(_x(xargs, i, 1))) * hsz
        out.append(gh * cur.d[1])  # weight_ih (gates * hidden, features)
        out.append(gh * hsz)       # weight_hh (gates * hidden, hidden)
        out.append(gh)             # bias_ih
        out.append(gh)             # bias_hh
    elif code == LAYER_SELECTIVE_SCAN:
        var dim = _scan_dim(xargs, i, cur)
        out.append(dim * _x(xargs, i, 0))  # A_log (dim, state)
        out.append(dim)                    # D
    elif code == LAYER_SSD_SCAN:  # Mamba-2 (lane neural-blocks-3)
        out.append(_x(xargs, i, 0))  # A_log (heads)
        out.append(_x(xargs, i, 0))  # D (heads)
    elif code == LAYER_TRAPEZOID_SCAN:  # Mamba-3 (lane neural-blocks-3-m3)
        out.append(_x(xargs, i, 0))        # D (heads)
    # lm_head (lane neural-blocks-3-lm)
    elif code == LAYER_LM_HEAD:
        out.append(_x(xargs, i, 0) * cur.cols())  # weight (vocab, features)
    # end lm_head
    return out^


def block_param_names(code: Int) -> List[String]:
    """The checkpoint names (after the layer tag) of the tensors block_param_sizes lists."""
    var out = List[String]()
    if code == LAYER_RMS_NORM:
        out.append(".weight")
    elif code == LAYER_EMBEDDING:
        out.append(".weight")
    elif code == LAYER_ATTENTION:
        out.append(".q_weight")
        out.append(".k_weight")
        out.append(".v_weight")
        out.append(".o_weight")
    elif code == LAYER_RECURRENT:
        out.append(".weight_ih")
        out.append(".weight_hh")
        out.append(".bias_ih")
        out.append(".bias_hh")
    elif code == LAYER_SELECTIVE_SCAN:
        out.append(".A_log")
        out.append(".D")
    elif code == LAYER_SSD_SCAN:  # Mamba-2 (lane neural-blocks-3)
        out.append(".A_log")
        out.append(".D")  # restored: lost in the neural-blocks-3-m3 merge, 9e745eb1a8
    elif code == LAYER_TRAPEZOID_SCAN:  # Mamba-3 (lane neural-blocks-3-m3)
        out.append(".D")
    # lm_head (lane neural-blocks-3-lm)
    elif code == LAYER_LM_HEAD:
        out.append(".weight")
    # end lm_head
    return out^


def block_reads(code: Int, xargs: List[Int], i: Int, j: Int) -> Bool:
    """True when block layer i reads earlier layer j's output (not through its input)."""
    return code == LAYER_MULTIPLY and _x(xargs, i, 0) == j


# ---------------------------------------------------------------- init
def _fill[E: Exec](mut ex: E, dst: FP, n: Int, value: Float32) raises:
    var a = Args()
    a.p0 = dst
    a.f0 = value
    ex.launch[OP_FILL](a, n)


def _glorot(fan: Int) -> Float32:
    return ftz(identical_sqrt(ftz(identical_div(Float32(6), Float32(fan)))))


def block_init_bounds(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Float32]:
    """Per parameter tensor: the uniform bound of its seeded init, or a negative value -v for a constant
    fill of v - 1 (RMSNorm's weight: -2 fills 1.0)."""
    var out = List[Float32]()
    if code == LAYER_RMS_NORM:
        out.append(Float32(-2))
    elif code == LAYER_EMBEDDING:
        out.append(_glorot(_x(xargs, i, 0) + _x(xargs, i, 1)))
    elif code == LAYER_ATTENTION:
        var hh = _heads(xargs, i)
        var d = cur.d[1]
        var hd = d // hh[0]
        out.append(_glorot(d + hh[0] * hd))
        out.append(_glorot(d + hh[1] * hd))
        out.append(_glorot(d + hh[1] * hd))
        out.append(_glorot(hh[0] * hd + d))
    elif code == LAYER_RECURRENT:
        # PyTorch's recurrent init: every tensor uniform in [-1/sqrt(hidden), 1/sqrt(hidden)]
        var b = ftz(identical_div(Float32(1), identical_sqrt(Float32(_x(xargs, i, 0)))))
        for _ in range(4):  # small-loop(4: the layer's parameter tensors)
            out.append(b)
    elif code == LAYER_SELECTIVE_SCAN:
        # Mamba's init: A_log[d, n] = log(n + 1) (S4D-real), D = 1
        out.append(BLOCK_INIT_S4D)
        out.append(Float32(-2))
    elif code == LAYER_SSD_SCAN:
        # Mamba-2 (lane neural-blocks-3): A_log[h] = log(h + 1), so A spans -1 .. -heads; D = 1
        out.append(BLOCK_INIT_S4D)
        out.append(Float32(-2))  # D = 1 (restored: lost in the neural-blocks-3-m3 merge, 9e745eb1a8)
    elif code == LAYER_TRAPEZOID_SCAN:  # Mamba-3 (lane neural-blocks-3-m3): D = 1
        out.append(Float32(-2))
    # lm_head (lane neural-blocks-3-lm): PyTorch's Linear init, uniform in [-1/sqrt(features), 1/sqrt(features)]
    elif code == LAYER_LM_HEAD:
        out.append(ftz(identical_div(Float32(1), identical_sqrt(Float32(cur.cols())))))
    # end lm_head
    return out^


# ---------------------------------------------------------------- forward
def _p[E: Exec](mut tape: Tape, mut ex: E, P: FP, Gr: FP, off: Int, rows: Int, cols: Int, train: Bool) raises -> Tensor:
    if train:
        return tape.variable_into(ex, P + off, Gr + off, rows, cols)
    return tape.variable(ex, P + off, rows, cols, False)


def _sh(dims: List[Int]) raises -> ShapeN:
    return rn_shape(dims, "model")


def _attention_mask[E: Exec](mut tape: Tape, mut ex: E, T: Int, window: Int) raises -> Tensor:
    """The (T, T) condition, nonzero where query i may NOT read key j: j > i, and with a window also
    i - j >= window. Built on the executor from iota positions (no host loop over positions)."""
    var pos = tape.reserve(ex, T)
    var a = Args()
    a.p0 = pos
    ex.launch[OP_SEQ_IOTA](a, T)
    var col = tape.variable(ex, pos, T, 1, False)
    var row = tape.view_nd(ex, col, ShapeN.matrix(1, T))
    var cond = imp_binary(tape, ex, row, col, IMP_GT)
    if window > 0:
        var diff = rankn_binary(tape, ex, col, row, SUBTRACT)
        var wp = tape.reserve(ex, 1)
        _fill(ex, wp, 1, Float32(window))
        var w = tape.variable(ex, wp, 1, 1, False)
        cond = imp_binary(tape, ex, cond, imp_binary(tape, ex, diff, w, IMP_GE), IMP_OR)
    return cond


def block_attention[E: Exec](
    mut tape: Tape, mut ex: E, h: Tensor, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
    B: Int, T: Int, D: Int, H: Int, Hk: Int, rotary: Bool, theta: Float32, window: Int, causal: Bool,
) raises -> Tensor:
    """Multi-head (grouped-query when Hk < H) self-attention of h (B, T, D) with weights (in, out):
    softmax(q k^T / sqrt(hd) + mask) v, then the output projection. Query head g of kv group k is head
    k * (H / Hk) + g (Llama's repeat_kv order)."""
    var hd = D // H
    var G = H // Hk
    var q = tape.linear(ex, h, wq, None, 0)
    var k = tape.linear(ex, h, wk, None, 0)
    var v = tape.linear(ex, h, wv, None, 0)
    var q3: List[Int] = [B, T, H * hd]
    var k3: List[Int] = [B, T, Hk * hd]
    q = tape.view_nd(ex, q, _sh(q3))
    k = tape.view_nd(ex, k, _sh(k3))
    v = tape.view_nd(ex, v, _sh(k3))
    if rotary:
        q = graph_rotary(tape, ex, q, hd, 0, theta)
        k = graph_rotary(tape, ex, k, hd, 0, theta)
    var q5: List[Int] = [B, T, Hk, G, hd]
    var k5: List[Int] = [B, T, Hk, 1, hd]
    var qp: List[Int] = [0, 2, 3, 1, 4]   # (B, Hk, G, T, hd)
    var kp: List[Int] = [0, 2, 3, 4, 1]   # (B, Hk, 1, hd, T)
    var qh = tape.permute(ex, tape.view_nd(ex, q, _sh(q5)), qp)
    var kt = tape.permute(ex, tape.view_nd(ex, k, _sh(k5)), kp)
    var vh = tape.permute(ex, tape.view_nd(ex, v, _sh(k5)), qp)
    var scores = rankn_matmul(tape, ex, qh, kt)
    scores = tape.scale(ex, scores, ftz(identical_div(Float32(1), identical_sqrt(Float32(hd)))))
    if causal:
        var cond = _attention_mask(tape, ex, T, window)
        var mp = tape.reserve(ex, 1)
        _fill(ex, mp, 1, ATTENTION_MASKED)
        var masked = tape.variable(ex, mp, 1, 1, False)
        scores = imp_where(tape, ex, cond, masked, scores)
    var probs = tape.softmax(ex, scores)
    var s5: List[Int] = [B, Hk, G, T, T]
    probs = tape.view_nd(ex, probs, _sh(s5))
    var ctx = rankn_matmul(tape, ex, probs, vh)
    var cp: List[Int] = [0, 3, 1, 2, 4]   # (B, T, Hk, G, hd)
    ctx = tape.permute(ex, ctx, cp)
    var c3: List[Int] = [B, T, H * hd]
    ctx = tape.view_nd(ex, ctx, _sh(c3))
    var y = tape.linear(ex, ctx, wo, None, 0)
    var y3: List[Int] = [B, T, D]
    return tape.view_nd(ex, y, _sh(y3))


def block_recurrent[E: Exec](
    mut tape: Tape, mut ex: E, h: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor,
    B: Int, T: Int, F: Int, Hs: Int, cell: Int,
) raises -> Tensor:
    """One recurrent layer over h (B, T, F) from a zero state: (B, T, hidden), every step's state."""
    var tp: List[Int] = [1, 0, 2]
    var xt = tape.permute(ex, h, tp)
    var tm = tape.view_nd(ex, xt, ShapeN.matrix(T * B, F))
    var y = rnn_scan_op(tape, ex, tm, w_ih, w_hh, b_ih, b_hh, cell, T, B)
    var y3: List[Int] = [T, B, Hs]
    var yt = tape.view_nd(ex, y, _sh(y3))
    return tape.permute(ex, yt, tp)


def block_forward[E: Exec](
    mut tape: Tape, mut ex: E, code: Int, xargs: List[Int], i: Int, f: Float32, cur: ShapeN,
    P: FP, Gr: FP, off: Int, h: Tensor, other: Optional[Tensor], train: Bool,
) raises -> Tensor:
    """Record block layer i on the tape from h (B,) + cur; `other` is the earlier output a multiply
    layer reads."""
    var B = tape.rows(h) * tape.cols(h) // cur.count()
    var a0 = _x(xargs, i, 0)
    var a1 = _x(xargs, i, 1)
    if code == LAYER_SILU:
        return imp_unary(tape, ex, h, IMP_SILU)
    if code == LAYER_GELU:
        return imp_unary(tape, ex, h, IMP_GELU_TANH if a0 == 1 else IMP_GELU_ERF)
    if code == LAYER_SOFTPLUS:
        return imp_unary(tape, ex, h, IMP_SOFTPLUS)
    if code == LAYER_RMS_NORM:
        var w = _p(tape, ex, P, Gr, off, 1, cur.cols(), train)
        return graph_rms_norm(tape, ex, h, w, f)
    if code == LAYER_ROTARY:
        return graph_rotary(tape, ex, h, a0, 0, f)
    if code == LAYER_MULTIPLY:
        if not other:
            raise Error("model: multiply reads an earlier output")
        return tape.multiply(ex, h, other.value())
    if code == LAYER_NARROW:
        return rankn_narrow(tape, ex, h, a0 + 1, a1, _x(xargs, i, 2))
    if code == LAYER_EMBEDDING:
        var table = _p(tape, ex, P, Gr, off, a0, a1, train)
        var ids = List[Tensor]()
        ids.append(h)
        var axes: List[Int] = [0]
        return imp_gather_nd(tape, ex, table, ids, axes)
    if code == LAYER_ATTENTION:
        var hh = _heads(xargs, i)
        var T = cur.d[0]
        var D = cur.d[1]
        var hd = D // hh[0]
        var nq = D * hh[0] * hd
        var nk = D * hh[1] * hd
        var wq = _p(tape, ex, P, Gr, off, D, hh[0] * hd, train)
        var wk = _p(tape, ex, P, Gr, off + nq, D, hh[1] * hd, train)
        var wv = _p(tape, ex, P, Gr, off + nq + nk, D, hh[1] * hd, train)
        var wo = _p(tape, ex, P, Gr, off + nq + 2 * nk, hh[0] * hd, D, train)
        return block_attention(tape, ex, h, wq, wk, wv, wo, B, T, D, hh[0], hh[1], _x(xargs, i, 2) == 1, f,
                               _x(xargs, i, 3), _x(xargs, i, 4) == 1)
    if code == LAYER_RECURRENT:
        var T = cur.d[0]
        var F = cur.d[1]
        var cell = _cell(a1)
        var gh = gates_of(cell) * a0
        var w_ih = _p(tape, ex, P, Gr, off, gh, F, train)
        var w_hh = _p(tape, ex, P, Gr, off + gh * F, gh, a0, train)
        var b_ih = _p(tape, ex, P, Gr, off + gh * F + gh * a0, 1, gh, train)
        var b_hh = _p(tape, ex, P, Gr, off + gh * F + gh * a0 + gh, 1, gh, train)
        return block_recurrent(tape, ex, h, w_ih, w_hh, b_ih, b_hh, B, T, F, a0, cell)
    if code == LAYER_SELECTIVE_SCAN:
        var dim = _scan_dim(xargs, i, cur)
        var u = rankn_narrow(tape, ex, h, 2, 0, dim)
        var dl = rankn_narrow(tape, ex, h, 2, dim, dim)
        var bm = rankn_narrow(tape, ex, h, 2, 2 * dim, a0)
        var cm = rankn_narrow(tape, ex, h, 2, 2 * dim + a0, a0)
        var a_log = _p(tape, ex, P, Gr, off, dim, a0, train)
        var d_skip = _p(tape, ex, P, Gr, off + dim * a0, 1, dim, train)
        return graph_selective_scan(tape, ex, u, dl, bm, cm, a_log, d_skip)
    if code == LAYER_SSD_SCAN:  # Mamba-2 (lane neural-blocks-3)
        var wide = a0 * SSD_HEADDIM + 2 * SSD_STATES
        var xbc = rankn_narrow(tape, ex, h, 2, 0, wide)
        var dts = rankn_narrow(tape, ex, h, 2, wide, a0)
        var a_log2 = _p(tape, ex, P, Gr, off, 1, a0, train)
        var d_skip2 = _p(tape, ex, P, Gr, off + a0, 1, a0, train)
        return graph_ssd_scan(tape, ex, xbc, dts, a_log2, d_skip2)
    # Mamba-3 (lane neural-blocks-3-m3): [v | k | q | dt | A | trap] as narrow views
    if code == LAYER_TRAPEZOID_SCAN:
        var tv = _trap_v(xargs, i, cur)
        var tk = a0 * _x(xargs, i, 1)
        var v = rankn_narrow(tape, ex, h, 2, 0, tv)
        var k = rankn_narrow(tape, ex, h, 2, tv, tk)
        var q = rankn_narrow(tape, ex, h, 2, tv + tk, tk)
        var dt = rankn_narrow(tape, ex, h, 2, tv + 2 * tk, a0)
        var av = rankn_narrow(tape, ex, h, 2, tv + 2 * tk + a0, a0)
        var tr = rankn_narrow(tape, ex, h, 2, tv + 2 * tk + 2 * a0, a0)
        var t_skip = _p(tape, ex, P, Gr, off, 1, a0, train)
        return graph_trapezoid_scan(tape, ex, v, k, q, dt, av, tr, t_skip, a0)
    # lm_head (lane neural-blocks-3-lm): the logits (B,) + cur[:-1] + (vocab,), h @ weight^T
    if code == LAYER_LM_HEAD:
        var w = _p(tape, ex, P, Gr, off, a0, cur.cols(), train)
        return graph_transformer_linear(tape, ex, h, w)
    # end lm_head
    raise Error("model: unknown block code " + String(code))


# lm_head (lane neural-blocks-3-lm)
def block_lm_head_loss[E: Exec](
    mut tape: Tape, mut ex: E, xargs: List[Int], i: Int, cur: ShapeN, P: FP, Gr: FP, off: Int,
    hidden: Tensor, codes: Tensor, train: Bool,
) raises -> Tensor:
    """The token cross-entropy of lm_head layer i from its input `hidden` ((B,) + cur) against `codes` (one
    class value per token, B * cur.rows() of them): the fused head-and-loss op (autograd/lm_head_v2_graph.mojo),
    which never forms the logits; the mean over every token."""
    var width = cur.cols()
    var rows = tape.rows(hidden) * tape.cols(hidden) // width
    var w = _p(tape, ex, P, Gr, off, _x(xargs, i, 0), width, train)
    return graph_lm_head_v2(tape, ex, hidden, w, codes, rows, _x(xargs, i, 0), width)
# end lm_head


# ---------------------------------------------------------------- backward
@fieldwise_init
struct ModelRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The backward of a version-2 Sequential description: ImportRules (import family + RankNRules,
    which holds NeuralRules and TransformerPrimitiveRules) plus RecurrentRules (16..18)."""

    @staticmethod
    def _recurrent(tag: Int) -> Bool:
        return tag == RNN_LINEAR or tag == RNN_LAYER or tag == RNN_LAST_STEP

    @staticmethod
    def _ssm(tag: Int) -> Bool:
        return tag >= SSM_TAG_FIRST and tag <= SSM_TAG_LAST

    def supports(self, tag: Int) -> Bool:
        if Self._recurrent(tag):
            return True
        # lm_head (lane neural-blocks-3-lm): the fused head's tag, StepOpRules' own answers
        if tag == LM_HEAD_V2_TAG:
            return True
        if Self._ssm(tag):
            return SsmRules().supports(tag)
        return ImportRules().supports(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if Self._recurrent(tag):
            return RecurrentRules().zeroed_destination(tag, saved, edge)
        if tag == LM_HEAD_V2_TAG:
            return StepOpRules(0).zeroed_destination(tag, saved, edge)
        if Self._ssm(tag):
            return SsmRules().zeroed_destination(tag, saved, edge)
        return ImportRules().zeroed_destination(tag, saved, edge)

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if Self._recurrent(tag):
            return RecurrentRules().writes_total(tag, saved, edge)
        if tag == LM_HEAD_V2_TAG:
            return StepOpRules(0).writes_total(tag, saved, edge)
        if Self._ssm(tag):
            return SsmRules().writes_total(tag, saved, edge)
        return ImportRules().writes_total(tag, saved, edge)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if Self._recurrent(tag):
            RecurrentRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif tag == LM_HEAD_V2_TAG:
            StepOpRules(0).apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif Self._ssm(tag):
            SsmRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        else:
            ImportRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)


# ---------------------------------------------------------------- weight import / export (load_state_dict, state_dict)
def block_tensor_dims(code: Int, xargs: List[Int], i: Int, cur: ShapeN, slot: Int, torch_layout: Bool) -> List[Int]:
    """The shape of block layer i's parameter tensor `slot` as a source gives it: PyTorch's layout
    (torch_layout; a projection weight (out, in)) or params_' own (a projection weight (in, out))."""
    var out = List[Int]()
    if code == LAYER_RMS_NORM:
        out.append(cur.cols())
    elif code == LAYER_EMBEDDING:
        out.append(_x(xargs, i, 0))
        out.append(_x(xargs, i, 1))
    elif code == LAYER_ATTENTION:
        var hh = _heads(xargs, i)
        var d = cur.d[1]
        var hd = d // hh[0]
        var fi = d if slot < 3 else hh[0] * hd
        var fo = d if slot == 3 else (hh[0] if slot == 0 else hh[1]) * hd
        if torch_layout:
            out.append(fo)
            out.append(fi)
        else:
            out.append(fi)
            out.append(fo)
    elif code == LAYER_RECURRENT:
        # PyTorch's layout is params_' own
        var hsz = _x(xargs, i, 0)
        var gh = gates_of(_cell_of(_x(xargs, i, 1))) * hsz
        out.append(gh)
        if slot == 0:
            out.append(cur.d[1])
        elif slot == 1:
            out.append(hsz)
    elif code == LAYER_SELECTIVE_SCAN:
        var dim = _scan_dim(xargs, i, cur)
        out.append(dim)
        if slot == 0:
            out.append(_x(xargs, i, 0))
    elif code == LAYER_SSD_SCAN:  # Mamba-2 (lane neural-blocks-3): A_log and D, one value per head each
        out.append(_x(xargs, i, 0))
    elif code == LAYER_TRAPEZOID_SCAN:  # Mamba-3 (lane neural-blocks-3-m3)
        out.append(_x(xargs, i, 0))
    # lm_head (lane neural-blocks-3-lm): (vocab, features) in both layouts (PyTorch's Linear weight)
    elif code == LAYER_LM_HEAD:
        out.append(_x(xargs, i, 0))
        out.append(cur.cols())
    # end lm_head
    return out^


def block_tensor_transposed(code: Int, slot: Int) -> Bool:
    """True when PyTorch's layout of the tensor is the transpose of params_' (the attention projections)."""
    return code == LAYER_ATTENTION
