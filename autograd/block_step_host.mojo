# SPDX-License-Identifier: Apache-2.0
"""The HOST ARM of the block step ops (lane x7b-step-ops-host-arm, 2026-11-09;
docs/plans/EXECUTOR_X1_API.md section 2).

A host session (`OpsSession[HostExec]`) runs OPS_M1_STEP, OPS_M2_STEP,
OPS_M3_STEP and OPS_ATTN_STEP through the host block bodies over the SAME
packed tensors as the device table (autograd/block_step_ops.mojo): the
weights are read from the weights tensor at its `block_offsets` slots at
every run, the carried float state is read from the state tensor and written
back after a successful run (a failed run leaves it as it was), the integers
are the words tensor's. Session storage is host memory here: a tensor's FP is
its floats.

The bodies are the oracles the CPU identity column has always run, each equal
to its device block's bits: `transformer_block_oracle` and
`transformer_block_backward_oracle` (attention: prefill, chunked
continuation and the L = 1 step are one function), `mamba_block_oracle`,
`mamba2_block_oracle`, `mamba3_block_oracle`. Every cross-row fold inside them
is a gemm v1 call and every per-row stage runs over host tasks, so a run uses
every core; consecutive layers are dependent and run in order.

A recorded OPS_ATTN_STEP (Program.fit's replay on a host session) is the same
zero-state prefill at position 0 with its stages and weights kept in the
entry; `host_attn_step_backward` runs the backward oracle over them
(StepOpRules reaches it through `Exec.step_backward`).

This is a HOST module: no GPU binding imports it. The entries live in the host
executor (`HostExec.steps`, sequence/exec.mojo), and generic session code
reaches the arm through the executor trait (`Exec.step_run`,
`Exec.step_backward`, `Exec.step_forget`, `Exec.step_weights_changed`,
`Exec.step_clear`), so the device build never instantiates or imports it.
"""
from std.memory import bitcast, memcpy
from sequence.ops import FP
from core.host_lanes import host_f32_uninit
from autograd.block_step_ops import (
    OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP, OPS_ATTN_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE,
    block_weight_sizes, block_state_sizes, block_offsets, block_guard, block_part_floats, step_op_info,
    _check_ints, _attn_dims, _attn_opts, _tokens, _expect, _m2_check_words, _m3_check_words, _same_ints,
)
from mamba.checks.mamba_fixture import MambaDims, MambaWeights
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba2_fixture import Mamba2Dims, Mamba2Weights
from mamba.checks.mamba2_oracle import Mamba2State, mamba2_block_oracle
from mamba.checks.mamba3_fixture import Mamba3Dims, Mamba3Weights
from mamba.checks.mamba3_oracle import Mamba3State, mamba3_block_oracle
from transformer.checks.transformer_fixture import TransformerDims, TransformerWeights, ScorePlant
from transformer.checks.transformer_oracle import (
    RopeTable, TransformerKVCache, TransformerStages, build_rope_table_opts, transformer_block_oracle,
)
from transformer.checks.transformer_backward_oracle import transformer_block_backward_oracle


struct HostStepScratch(Movable):
    """The host arm's entries of one host executor: attention entries (the
    rotary table, a training run's kept stages) and Mamba entries (the last
    run's reports). Nothing else is resident: weights, state and words are
    the session tensors'."""
    var hattn: List[_HostAttnBlock]
    var hmamba: List[_HostMambaBlock]

    def __init__(out self):
        self.hattn = List[_HostAttnBlock]()
        self.hmamba = List[_HostMambaBlock]()

    def count(self) -> Int:
        return len(self.hattn) + len(self.hmamba)

    def forget(mut self, addr: Int):
        """Drop every entry over the session buffer at `addr` (written from
        outside a step, or freed)."""
        if self.count() == 0:
            return
        var i = len(self.hattn) - 1
        while i >= 0:  # driver-loop: over host attention entries
            if self.hattn[i].w_addr == addr or self.hattn[i].s_addr == addr or self.hattn[i].q_addr == addr:
                _ = self.hattn.pop(i)
            i -= 1
        i = len(self.hmamba) - 1
        while i >= 0:  # driver-loop: over host Mamba entries
            if self.hmamba[i].w_addr == addr or self.hmamba[i].s_addr == addr or self.hmamba[i].q_addr == addr:
                _ = self.hmamba.pop(i)
            i -= 1

    def weights_changed(mut self, addr: Int):
        """The weights at `addr` were rewritten in place: an attention
        entry's kept training stages (computed over the old weights) go, as
        the device entry does; the weights themselves are read at every run."""
        for i in range(len(self.hattn) - 1, -1, -1):  # driver-loop: over host attention entries
            if self.hattn[i].w_addr == addr:
                _ = self.hattn.pop(i)

    def clear(mut self):
        self.hattn.clear()
        self.hmamba.clear()


def _host_take(p: FP, at: Int, n: Int) raises -> List[Float32]:
    """The n floats at p + at, as an owned list (the oracles take lists)."""
    var out = host_f32_uninit(n)
    if n > 0:
        memcpy(dest=FP(unsafe_from_address=Int(out.unsafe_ptr())), src=p + at, count=n)
    return out^


def _host_put(p: FP, at: Int, values: List[Float32], n: Int, what: String) raises:
    """p[at : at + n] = values (refused unless it holds n floats)."""
    if len(values) != n:
        raise Error("ops: host step op: " + what + " holds " + String(len(values)) + " values, the layout "
                    + String(n))
    if n > 0:
        memcpy(dest=p + at, src=FP(unsafe_from_address=Int(values.unsafe_ptr())), count=n)


def _host_words(p: FP, k: Int) -> List[Int]:
    """The k int32 words of a host words tensor."""
    var out = List[Int](capacity=k)
    for i in range(k):  # small-loop(k: at most two state words): word decode, no data
        out.append(Int(bitcast[DType.int32](p[i])))
    return out^


def _host_set_words(p: FP, words: List[Int]):
    for i in range(len(words)):  # small-loop(words: at most two state words): word encode, no data
        p[i] = bitcast[DType.float32](Int32(words[i]))


def _host_pieces(p: FP, code: Int, ints: List[Int], part: Int) raises -> List[List[Float32]]:
    """Every piece of `part` (weights or float state) of the packed tensor
    at p, in `block_*_sizes` order (an absent optional piece empty)."""
    var sizes = block_weight_sizes(code, ints) if part == BLOCK_PART_WEIGHTS else block_state_sizes(code, ints)
    var at = block_offsets(sizes, block_guard(code))
    var out = List[List[Float32]](capacity=len(sizes))
    for i in range(len(sizes)):  # driver-loop: over the block's weight or state pieces
        out.append(_host_take(p, at[i], sizes[i]))
    return out^


def _host_store_state(p: FP, code: Int, ints: List[Int], pieces: List[List[Float32]]) raises:
    """The carried float state back into the state tensor at p, piece by
    piece at its slot (the gaps keep their zeros)."""
    var sizes = block_state_sizes(code, ints)
    var at = block_offsets(sizes, block_guard(code))
    if len(pieces) != len(sizes):
        raise Error("ops: host step op: the state has " + String(len(sizes)) + " pieces")
    for i in range(len(sizes)):  # driver-loop: over the block's state pieces
        _host_put(p, at[i], pieces[i], sizes[i], "state piece " + String(i))


struct _HostAttnBlock(Movable):
    """A host attention entry: the rotary table of its integer arguments and
    the last RECORDED run's stages and weights (for the backward)."""
    var w_addr: Int
    var s_addr: Int
    var q_addr: Int
    var ints: List[Int]
    var dims: TransformerDims
    var rope: RopeTable
    #: the last recorded run's token count (0: none) and its kept stages
    var train_l: Int
    var kept: Optional[TransformerStages]
    var kept_w: Optional[TransformerWeights]

    def __init__(out self, w: FP, s: FP, q: FP, ints: List[Int]) raises:
        var d = _attn_dims(ints)
        var opts = _attn_opts(ints)
        # RoPE positions = s_max, as the device table `LlamaRopeTable(ctx,
        # dims, opts, s_max)` of the device entry.
        self.dims = TransformerDims(d.d_model, d.n_heads, d.n_kv, d.head_dim, d.intermediate, ints[6])
        self.rope = build_rope_table_opts(self.dims, opts)
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.q_addr = Int(q)
        self.ints = ints.copy()
        self.train_l = 0
        self.kept = Optional[TransformerStages]()
        self.kept_w = Optional[TransformerWeights]()


def _host_attn_weights(w: FP, ints: List[Int], dims: TransformerDims) raises -> TransformerWeights:
    """The block's weights from the packed tensor (`block_weight_sizes`
    order: the options constructor's twenty, `LlamaDeviceWeights`)."""
    var p = _host_pieces(w, OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS)
    var tw = TransformerWeights(dims)
    tw.opts = _attn_opts(ints)
    tw.norm1_w = p[0].copy()
    tw.norm2_w = p[1].copy()
    tw.w_q = p[2].copy()
    tw.w_k = p[3].copy()
    tw.w_v = p[4].copy()
    tw.w_o = p[5].copy()
    tw.w_gate = p[6].copy()
    tw.w_up = p[7].copy()
    tw.w_down = p[8].copy()
    tw.b_q = p[9].copy()
    tw.b_k = p[10].copy()
    tw.b_v = p[11].copy()
    tw.b_o = p[12].copy()
    tw.norm1_b = p[13].copy()
    tw.norm2_b = p[14].copy()
    tw.b_up = p[15].copy()
    tw.b_down = p[16].copy()
    tw.b_gate = p[17].copy()
    tw.qn_w = p[18].copy()
    tw.kn_w = p[19].copy()
    return tw^


def _host_attn_entry(mut sc: HostStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int]) raises -> Int:
    """The host entry over (weights, kv, words), built at the triple's
    first run; the operand counts are checked at every run."""
    _expect(OPS_ATTN_STEP, "weights", counts[1], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_ATTN_STEP, "kv", counts[2], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_ATTN_STEP, "state words", counts[3], 1)
    for i in range(len(sc.hattn)):  # driver-loop: over host attention entries
        if (sc.hattn[i].w_addr == Int(bufs[1]) and sc.hattn[i].s_addr == Int(bufs[2])
                and sc.hattn[i].q_addr == Int(bufs[3]) and _same_ints(sc.hattn[i].ints, ints)):
            return i
    sc.hattn.append(_HostAttnBlock(bufs[1], bufs[2], bufs[3], ints))
    return len(sc.hattn) - 1


def _host_attn_on(mut blk: _HostAttnBlock, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP,
                  pos0: Int, keep: Bool) raises -> Int:
    """One block call at carried position pos0 (`transformer_block_oracle`:
    prefill, chunked continuation and the L = 1 step are one function), the
    cache read from and written back to the state tensor, the words to the
    new position. With `keep` the stages and weights stay for the backward.
    Returns L."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_ATTN_STEP, counts[0], b, dm)
    var smax = ints[6]
    if pos0 < 0 or pos0 > smax:
        raise Error("ValueError: ops: attention_step: cached tokens must be in [0, " + String(smax)
                    + "], got " + String(pos0))
    var n = b * l * dm
    var w = _host_attn_weights(bufs[1], ints, blk.dims)
    var ssz = block_state_sizes(OPS_ATTN_STEP, ints)
    var sat = block_offsets(ssz, block_guard(OPS_ATTN_STEP))
    var cache = TransformerKVCache(b, blk.dims, smax, ints[7])
    if len(cache.k) != ssz[0] or len(cache.v) != ssz[1]:
        raise Error("ops: attention_step: the host cache layout differs from the state tensor's")
    cache.k = _host_take(bufs[2], sat[0], ssz[0])
    cache.v = _host_take(bufs[2], sat[1], ssz[1])
    cache.used = pos0
    var x = _host_take(bufs[0], 0, n)
    var st = transformer_block_oracle(w, x, b, l, cache, blk.rope, ScorePlant.none())
    _host_put(y_out, 0, st.residual2_out, n, "the block output")
    _host_put(bufs[2], sat[0], cache.k, ssz[0], "the key cache")
    _host_put(bufs[2], sat[1], cache.v, ssz[1], "the value cache")
    var words: List[Int] = [cache.used]
    _host_set_words(bufs[3], words)
    if keep:
        blk.kept = st^
        blk.kept_w = w^
        blk.train_l = l
    return l


def _host_run_attn(mut sc: HostStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP) raises:
    var e = _host_attn_entry(sc, bufs, counts, ints)
    var pos0 = _host_words(bufs[3], 1)[0]
    _ = _host_attn_on(sc.hattn[e], bufs, counts, ints, y_out, pos0, False)


def host_attn_step_train_run(
    mut sc: HostStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP,
) raises -> Int:
    """`attn_step_train_run` on the host: the zero-state prefill at
    position 0 with its stages kept in the entry for the backward. Returns
    L. A failed run drops the triple's entry."""
    _check_ints(OPS_ATTN_STEP, ints)
    if len(bufs) != 4 or len(counts) != 4:
        raise Error("ops: attention_step takes four tensors (x, weights, kv, words)")
    var l = _tokens(OPS_ATTN_STEP, counts[0], ints[0], ints[1])
    if l > ints[6]:
        raise Error("ValueError: ops: attention_step: a training run of " + String(l)
                    + " tokens needs s_max >= " + String(l) + ", got " + String(ints[6]))
    try:
        var e = _host_attn_entry(sc, bufs, counts, ints)
        _ = _host_attn_on(sc.hattn[e], bufs, counts, ints, y_out, 0, True)
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err
    return l


def host_attn_step_backward(
    mut sc: HostStepScratch, w: FP, s: FP, q: FP, d_out: FP, d_x: FP, d_w: FP,
) raises:
    """`attn_step_backward` on the host: `transformer_block_backward_oracle`
    over the entry's kept stages and weights, d(residual2) `d_out`; writes
    d_x and the nine weight gradients at their `block_offsets` slots of
    `d_w` (the optional pieces' gradients are the device rule's: none)."""
    var e = -1
    for i in range(len(sc.hattn)):  # driver-loop: over host attention entries
        if sc.hattn[i].w_addr == Int(w) and sc.hattn[i].s_addr == Int(s) and sc.hattn[i].q_addr == Int(q):
            e = i
    if e < 0 or sc.hattn[e].train_l < 1 or not sc.hattn[e].kept or not sc.hattn[e].kept_w:
        raise Error("ops: attention_step backward: the recorded forward's stages are gone"
                    + " (its weights or state were written outside the step after the run)")
    ref blk = sc.hattn[e]
    var ints = blk.ints.copy()
    var b = ints[0]
    var l = blk.train_l
    var n = b * l * ints[1]
    var dout = _host_take(d_out, 0, n)
    var g = transformer_block_backward_oracle(blk.kept_w.value(), blk.kept.value(), dout, b, l, 0, blk.rope, ints[7])
    _host_put(d_x, 0, g.d_x, n, "d_x")
    var sizes = block_weight_sizes(OPS_ATTN_STEP, ints)
    var at = block_offsets(sizes, block_guard(OPS_ATTN_STEP))
    # block_weight_sizes order: norm1, norm2, q, k, v, o, gate, up, down
    _host_put(d_w, at[0], g.dw_norm1, sizes[0], "dw_norm1")
    _host_put(d_w, at[1], g.dw_norm2, sizes[1], "dw_norm2")
    _host_put(d_w, at[2], g.dw_q, sizes[2], "dw_q")
    _host_put(d_w, at[3], g.dw_k, sizes[3], "dw_k")
    _host_put(d_w, at[4], g.dw_v, sizes[4], "dw_v")
    _host_put(d_w, at[5], g.dw_o, sizes[5], "dw_o")
    _host_put(d_w, at[6], g.dw_gate, sizes[6], "dw_gate")
    _host_put(d_w, at[7], g.dw_up, sizes[7], "dw_up")
    _host_put(d_w, at[8], g.dw_down, sizes[8], "dw_down")


struct _HostMambaBlock(Movable):
    """A host Mamba entry: the last run's reports (Mamba-2 [h_last],
    Mamba-3 [h_last, k_last, v_last, theta_last]; Mamba-1 none)."""
    var code: Int
    var w_addr: Int
    var s_addr: Int
    var q_addr: Int
    var b: Int
    var dm: Int
    var reports: List[List[Float32]]

    def __init__(out self, code: Int, w: Int, s: Int, q: Int, b: Int, dm: Int):
        self.code = code
        self.w_addr = w
        self.s_addr = s
        self.q_addr = q
        self.b = b
        self.dm = dm
        self.reports = List[List[Float32]]()


def _host_mamba_reports(mut sc: HostStepScratch, code: Int, bufs: List[FP], b: Int, dm: Int,
                        var reports: List[List[Float32]]):
    var q = Int(bufs[3]) if len(bufs) > 3 else 0
    for i in range(len(sc.hmamba)):  # driver-loop: over host Mamba entries
        if (sc.hmamba[i].code == code and sc.hmamba[i].w_addr == Int(bufs[1]) and sc.hmamba[i].s_addr == Int(bufs[2])
                and sc.hmamba[i].q_addr == q):
            sc.hmamba[i].b = b
            sc.hmamba[i].dm = dm
            sc.hmamba[i].reports = reports^
            return
    var blk = _HostMambaBlock(code, Int(bufs[1]), Int(bufs[2]), q, b, dm)
    blk.reports = reports^
    sc.hmamba.append(blk^)


def _host_run_m1(mut sc: HostStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP) raises:
    """`mamba_block_oracle` over the packed weights and state (`_m1_weight_sizes`
    / `_m1_state_sizes` order: MambaWeights' and MambaState's fields)."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M1_STEP, counts[0], b, dm)
    _expect(OPS_M1_STEP, "weights", counts[1], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M1_STEP, "state", counts[2], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_STATE))
    var dims = MambaDims.of(dm)
    var p = _host_pieces(bufs[1], OPS_M1_STEP, ints, BLOCK_PART_WEIGHTS)
    var w = MambaWeights(dims)
    w.norm_w = p[0].copy()
    w.w_in = p[1].copy()
    w.conv_w = p[2].copy()
    w.conv_b = p[3].copy()
    w.w_x = p[4].copy()
    w.w_dt = p[5].copy()
    w.b_dt = p[6].copy()
    w.a_log = p[7].copy()
    w.d_skip = p[8].copy()
    w.w_out = p[9].copy()
    var q = _host_pieces(bufs[2], OPS_M1_STEP, ints, BLOCK_PART_STATE)
    var st = MambaState(b, dims)
    st.conv_win = q[0].copy()
    st.h = q[1].copy()
    var n = b * l * dm
    var x = _host_take(bufs[0], 0, n)
    var stages = mamba_block_oracle(w, x, b, l, st)
    _host_put(y_out, 0, stages.residual_out, n, "the block output")
    var out = List[List[Float32]]()
    out.append(st.conv_win.copy())
    out.append(st.h.copy())
    _host_store_state(bufs[2], OPS_M1_STEP, ints, out)


def _host_run_m2(mut sc: HostStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], fs: List[Float32],
                 y_out: FP) raises:
    """`mamba2_block_oracle` over the packed weights and state (Mamba2Weights'
    and Mamba2State's fields), buf_len the state word; the h_last report kept."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M2_STEP, counts[0], b, dm)
    _expect(OPS_M2_STEP, "weights", counts[1], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M2_STEP, "state", counts[2], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M2_STEP, "state words", counts[3], 1)
    var words = _host_words(bufs[3], 1)
    _m2_check_words(words)
    var dims = Mamba2Dims.of(dm)
    var p = _host_pieces(bufs[1], OPS_M2_STEP, ints, BLOCK_PART_WEIGHTS)
    var w = Mamba2Weights(dims)
    w.norm_w = p[0].copy()
    w.w_in = p[1].copy()
    w.conv_w = p[2].copy()
    w.conv_b = p[3].copy()
    w.dt_bias = p[4].copy()
    w.a_log = p[5].copy()
    w.d_skip = p[6].copy()
    w.gnorm_w = p[7].copy()
    w.w_out = p[8].copy()
    var q = _host_pieces(bufs[2], OPS_M2_STEP, ints, BLOCK_PART_STATE)
    var st = Mamba2State(b, dims)
    st.conv_win = q[0].copy()
    st.h = q[1].copy()
    st.buf_xbc = q[2].copy()
    st.buf_dtraw = q[3].copy()
    st.buf_len = words[0]
    var n = b * l * dm
    var x = _host_take(bufs[0], 0, n)
    var stages = mamba2_block_oracle(w, x, b, l, fs[0], fs[1], st)
    _host_put(y_out, 0, stages.residual_out, n, "the block output")
    var out = List[List[Float32]]()
    out.append(st.conv_win.copy())
    out.append(st.h.copy())
    out.append(st.buf_xbc.copy())
    out.append(st.buf_dtraw.copy())
    _host_store_state(bufs[2], OPS_M2_STEP, ints, out)
    var after: List[Int] = [st.buf_len]
    _host_set_words(bufs[3], after)
    var reports = List[List[Float32]]()
    reports.append(stages.h_last.copy())
    _host_mamba_reports(sc, OPS_M2_STEP, bufs, b, dm, reports^)


def _host_run_m3(mut sc: HostStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP) raises:
    """`mamba3_block_oracle` over the packed weights and state (Mamba3Weights'
    and Mamba3State's fields), [buf_len, pending] the state words; the four
    reports kept."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M3_STEP, counts[0], b, dm)
    _expect(OPS_M3_STEP, "weights", counts[1], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M3_STEP, "state", counts[2], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M3_STEP, "state words", counts[3], 2)
    var words = _host_words(bufs[3], 2)
    _m3_check_words(words)
    var dims = Mamba3Dims.of(dm)
    var p = _host_pieces(bufs[1], OPS_M3_STEP, ints, BLOCK_PART_WEIGHTS)
    var w = Mamba3Weights(dims)
    w.norm_w = p[0].copy()
    w.w_in = p[1].copy()
    w.dt_bias = p[2].copy()
    w.bnorm_w = p[3].copy()
    w.cnorm_w = p[4].copy()
    w.b_bias = p[5].copy()
    w.c_bias = p[6].copy()
    w.d_skip = p[7].copy()
    w.w_out = p[8].copy()
    var q = _host_pieces(bufs[2], OPS_M3_STEP, ints, BLOCK_PART_STATE)
    var st = Mamba3State(b, dims)
    st.theta = q[0].copy()
    st.h = q[1].copy()
    st.buf_qrot = q[2].copy()
    st.buf_krot = q[3].copy()
    st.buf_v = q[4].copy()
    st.buf_dt = q[5].copy()
    st.buf_sig = q[6].copy()
    st.buf_adt = q[7].copy()
    st.pend_k = q[8].copy()
    st.pend_v = q[9].copy()
    st.buf_len = words[0]
    st.pending = words[1] == 1
    var n = b * l * dm
    var x = _host_take(bufs[0], 0, n)
    var stages = mamba3_block_oracle(w, x, b, l, st)
    _host_put(y_out, 0, stages.residual_out, n, "the block output")
    var out = List[List[Float32]]()
    out.append(st.theta.copy())
    out.append(st.h.copy())
    out.append(st.buf_qrot.copy())
    out.append(st.buf_krot.copy())
    out.append(st.buf_v.copy())
    out.append(st.buf_dt.copy())
    out.append(st.buf_sig.copy())
    out.append(st.buf_adt.copy())
    out.append(st.pend_k.copy())
    out.append(st.pend_v.copy())
    _host_store_state(bufs[2], OPS_M3_STEP, ints, out)
    var after: List[Int] = [st.buf_len, 1 if st.pending else 0]
    _host_set_words(bufs[3], after)
    var reports = List[List[Float32]]()
    reports.append(stages.h_last.copy())
    reports.append(stages.k_last.copy())
    reports.append(stages.v_last.copy())
    reports.append(stages.theta_last.copy())
    _host_mamba_reports(sc, OPS_M3_STEP, bufs, b, dm, reports^)


def host_block_step_run(
    mut sc: HostStepScratch, code: Int, bufs: List[FP], counts: List[Int], ints: List[Int], fs: List[Float32],
    y_out: FP,
) raises:
    """`block_step_run` on a host session: the same operands (host storage),
    the host block bodies. A failed run writes no state (the state goes back
    only after the body returns) and drops the triple's entries."""
    try:
        if code == OPS_M1_STEP:
            _host_run_m1(sc, bufs, counts, ints, y_out)
        elif code == OPS_M2_STEP:
            _host_run_m2(sc, bufs, counts, ints, fs, y_out)
        elif code == OPS_M3_STEP:
            _host_run_m3(sc, bufs, counts, ints, y_out)
        elif code == OPS_ATTN_STEP:
            _host_run_attn(sc, bufs, counts, ints, y_out)
        else:
            raise Error("ops: " + String(code) + " is not a block step op")
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's three or four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


def host_block_step_reports(sc: HostStepScratch, code: Int, w_addr: Int, s_addr: Int, dsts: List[Int]) raises:
    """`block_step_reports_to_host` of a host session: the last run's reports
    of the host entry over (weights, state) into the caller's arrays (a 0
    address skips that report)."""
    var want = 1 if code == OPS_M2_STEP else 4
    if code != OPS_M2_STEP and code != OPS_M3_STEP:
        raise Error("ops: " + String(code) + " keeps no reports")
    if len(dsts) != want:
        raise Error("ops: " + String(step_op_info(code).name) + " has " + String(want) + " reports")
    for i in range(len(sc.hmamba)):  # driver-loop: over host Mamba entries
        if sc.hmamba[i].code == code and sc.hmamba[i].w_addr == w_addr and sc.hmamba[i].s_addr == s_addr:
            for k in range(want):  # small-loop(the one or four report buffers)
                if dsts[k] != 0:
                    ref r = sc.hmamba[i].reports[k]
                    _host_put(FP(unsafe_from_address=dsts[k]), 0, r, len(r), "report " + String(k))
            return
    raise Error("ops: " + String(code) + ": no resident block over these weights and state (run the step first)")




def host_step_run(
    mut sc: HostStepScratch, code: Int, bufs: List[FP], counts: List[Int], ints: List[Int], fs: List[Float32],
    y_out: FP, record: Bool,
) raises:
    """`HostExec.step_run`: a recorded run (the session's tape is recording;
    OPS_ATTN_STEP only, the session refuses the others) is the training
    prefill with its stages kept, any other run the op's host body."""
    if record:
        if code != OPS_ATTN_STEP:
            raise Error("ValueError: ops: " + String(step_op_info(code).name)
                        + " runs in an inference session (Session(record=False))")
        _ = host_attn_step_train_run(sc, bufs, counts, ints, y_out)
        return
    host_block_step_run(sc, code, bufs, counts, ints, fs, y_out)
