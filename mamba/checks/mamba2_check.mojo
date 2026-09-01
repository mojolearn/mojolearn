# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate file of profile `mojolearn.identical.mamba2.fp32.v1`
(`mamba/IDENTICAL_MAMBA2_CONTRACT.md` section 8, commit e3b46e95). NOT A
PORT: it runs the device block (`mamba/impl/mamba_ssm/modules/mamba2.mojo`
around `ssd_minimal.mojo`) against the host oracle
(`mamba/checks/mamba2_oracle.mojo`) and compares every recorded stage BY
BITS.

**NOTHING IN THIS FILE HAS RUN.** Every command below is RUN OWED; no
kernel here has ever compiled. The pixi task registrations
(`check-mamba2-block` etc.) are the ORCHESTRATOR's to add -- pixi.toml is
not this lane's editable set -- so the raw commands are spelled:

    # gates (a) card==oracle + card tags, (b) 8 repeated launches,
    # (c) batch composition with the negative control; one shape per
    # build (MOJOLEARN_MAMBA2_CHECK_B/_L/_DM, defaults 1/4/32 -- the
    # 2026-08-23 one-compile-at-a-time rule):
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba2_check.mojo

    # gate (d): decode == prefill per token (case m2_base_b2_l4_d32),
    # with the misalignment control:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba2_check.mojo decode

    # gate (d)'s chunk-crossing arm: prefill L1 = 250, decode through
    # token 262 (crosses the Q = 256 boundary) -- the handoff arm and the
    # boundary crossing in one run:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba2_check.mojo decode-cross

    # gate (d2): the boundary handoff, prefill 512 then a fresh call with
    # initial_states = h_last == prefill 520:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba2_check.mojo handoff

    # gate (e): the planted nonfinite refusal audit + clean control:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba2_check.mojo refusal

    # gate (g)'s byte gate (corpus generated at f45fa796): every input
    # tensor of all 17 cases byte-compared against this check's own
    # generator, binary-safe reads; missing / unreadable / wrong-size
    # each refuse under their OWN name, never a skip:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba2_check.mojo corpus

    # gate (f): ONE sabotage arm per build, verdict INVERTED; each arm
    # names its witnessing fixture and the check ASSERTS the witnessing
    # property before crediting the arm (a fixture that cannot witness
    # its arm is the recurring defect):
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_MAMBA2_SABOTAGE_SEGSUM_DESCENDING=1 -I . \\
        mamba/checks/mamba2_check.mojo
    #   ... likewise CHUNK_SIZE_128, STATEPASS_MATRIX, STATEPASS_UNFUSED,
    #   PAIR_DT_B, FOLD_SERIAL_ZERO_SEED, S6_BIAS_LAST, S6_TAPS_REVERSED,
    #   CLAMP_BEFORE_SOFTPLUS, GATE_NORM_BEFORE; and the decode arm:
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_MAMBA2_SABOTAGE_STEP_UPSTREAM_RECURRENCE=1 -I . \\
        mamba/checks/mamba2_check.mojo decode

WITNESSING, BY CONSTRUCTION AND ASSERTED (contract 8f's table): the armed
run picks the fixture the contract names and FIRST asserts the property
that makes the arm falsifiable there -- STATEPASS arms: >= 3 working
chunks (m2_statepass_b1_l513_d32) AND the nonzero-initial_states two-chunk
case (m2_init_states_b1_l257_d32), both asserted (chunk count computed,
initial h downloaded and counted nonzero); CLAMP_BEFORE_SOFTPLUS: an
IN-CHECK PLANTED fixture (`clamp_witness_weights`: two heads, dt_bias
planted one head per limit, run under (0.001, 0.1)) with the check
COUNTING lo-bound and hi-bound cells on the oracle's clean dt and
refusing a zero count on EITHER limit as VACUOUS -- the corpus dt_limit
case's own single-head hashed bias landed inside the band and bound zero
cells, which that refusal caught at first arming, exactly as designed;
SEGSUM_DESCENDING: two differing dt values inside one chunk,
counted; CHUNK_SIZE_128 and FOLD_SERIAL_ZERO_SEED: L > 128 asserted;
PAIR_DT_B / S6_* / GATE_NORM_BEFORE: hashed values, any shape;
STEP_UPSTREAM_RECURRENCE: an L >= 2 decode. An armed run whose fixture
fails its witnessing assertion raises VACUOUS, never passes.

GATE (d)'s COMPARABILITY (DEVIATION 790, the oracle's docstring is the
authority): token-shaped stages compare bitwise PER TOKEN; chunk-shaped
stages (`dacs.out`, `seg.L`, `cb.G`, `decay.states`, `cstate.out`,
`pass.states`) compare at the FINAL decoded token only, where the decode
call's open working chunk has the same fill level as the prefill's chunk
`(L-1) // Q` -- gathered slice against slice; the carried state
(conv.window, boundary h, the buffer rows) and `ssd.h_last` compare at
the final token too. The misalignment control (decode t against prefill
t+1, must differ) guards the gate against comparing a buffer with itself.

Under CHUNK_SIZE_128 the chunk-shaped stage BUFFERS change shape, so the
armed compare covers the token-shaped stages only and says so in its
banner -- the arm must bite in `ydiag.out` onward, which is token-shaped.
"""

from std.memory import bitcast
from std.os import getenv
from std.sys import argv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from checks.numerics import ftz
from mamba.checks.mamba2_fixture import (
    BITS_POS_INF,
    BITS_QNAN,
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    M2_TID_X,
    Mamba2CorpusCase,
    Mamba2Dims,
    Mamba2Weights,
    bits32_hex,
    corpus_tensor,
    f32_from_bits,
    m2_case_init_states,
    m2_case_seed,
    m2_case_weights,
    m2_case_x,
    m2_corpus_case,
    m2_corpus_dir,
    mode_name,
)
from mamba.checks.mamba2_oracle import (
    Mamba2Stages,
    Mamba2State,
    mamba2_block_oracle,
)
from mamba.impl.mamba_ssm.modules.mamba2 import (
    BLOCK2_ANY_SABOTAGE,
    Mamba2DeviceStages,
    Mamba2DeviceState,
    Mamba2DeviceWeights,
    allocate_inference_cache,
    mamba2_block_forward,
    mamba2_sabotage_name,
)
from mamba.impl.mamba_ssm.modules.ssd_minimal import m2_n_chunks, m2_q_eff
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    mamba_download,
    mamba_upload,
)

comptime TRACE_PATH = "/tmp/mojolearn_mamba2_block_tiny.trace"


def card_path() -> String:
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(TRACE_PATH)


def env_int(name: String, dflt: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return dflt
    var v = 0
    for i in range(s.byte_length()):
        var c = ord(String(s[byte=i]))
        if c < 48 or c > 57:
            raise Error(
                String("mamba2_check: ") + name + " is not a number: '" + s + "'"
            )
        v = v * 10 + (c - 48)
    return v


# ===========================================================================
# THE STAGE TABLE: card order (contract section 7 minus input.x), plus the
# RESUMPTION h as compare entry 25 (not a card tag; the boundary state the
# next call reads, which is exactly where a resumption bug lives).
# ===========================================================================

comptime KIND_TOKEN = 0
comptime KIND_GLOBAL = 1
comptime KIND_BATCH = 2
comptime KIND_CHUNK = 3

comptime N_STAGES = 26
comptime STAGE_DACS = 9
comptime STAGE_CONV = 4
comptime STAGE_DT = 7
comptime STAGE_XD = 8
comptime STAGE_YDIAG = 12
comptime STAGE_PASS = 15
comptime STAGE_GGATE = 19


def stage_name(i: Int) raises -> String:
    if i == 0:
        return String("norm.sumsq")
    if i == 1:
        return String("norm.out")
    if i == 2:
        return String("in_proj.out")
    if i == 3:
        return String("A.out")
    if i == 4:
        return String("conv.out")
    if i == 5:
        return String("silu.out")
    if i == 6:
        return String("conv.window")
    if i == 7:
        return String("dt.out")
    if i == 8:
        return String("xd.out")
    if i == 9:
        return String("dacs.out")
    if i == 10:
        return String("seg.L")
    if i == 11:
        return String("cb.G")
    if i == 12:
        return String("ydiag.out")
    if i == 13:
        return String("decay.states")
    if i == 14:
        return String("cstate.out")
    if i == 15:
        return String("pass.states")
    if i == 16:
        return String("yoff.out")
    if i == 17:
        return String("scan.y")
    if i == 18:
        return String("skip.out")
    if i == 19:
        return String("gnorm.gate")
    if i == 20:
        return String("gnorm.sumsq")
    if i == 21:
        return String("gnorm.out")
    if i == 22:
        return String("out_proj.out")
    if i == 23:
        return String("residual.out")
    if i == 24:
        return String("ssd.h_last")
    if i == 25:
        return String("state.h")
    raise Error("mamba2_check: no stage " + String(i))


def stage_kind(i: Int) -> Int:
    if i == 3:
        return KIND_GLOBAL
    if i == 6 or i == 24 or i == 25:
        return KIND_BATCH
    if i == 9 or i == 10 or i == 11 or i == 13 or i == 14 or i == 15:
        return KIND_CHUNK
    return KIND_TOKEN


def stage_token_width(i: Int, dims: Mamba2Dims) raises -> Int:
    """Elements per TOKEN row for KIND_TOKEN stages."""
    var dm = dims.d_model
    var di = dims.d_inner
    var nh = dims.nheads
    if i == 0 or i == 20:
        return 1
    if i == 1 or i == 22 or i == 23:
        return dm
    if i == 2:
        return dims.d_in_proj()
    if i == 4 or i == 5:
        return dims.conv_dim()
    if i == 7:
        return nh
    if i == 8 or i == 12 or i == 16 or i == 17 or i == 18:
        return nh * M2_HEADDIM
    if i == 19 or i == 21:
        return di
    raise Error("stage_token_width: stage " + String(i) + " is not TOKEN")


def stage_per_batch(i: Int, dims: Mamba2Dims, nc: Int) raises -> Int:
    """Elements per BATCH ROW for BATCH and CHUNK stages (every chunk
    stage's layout leads with B)."""
    var nh = dims.nheads
    var qv = m2_q_eff()
    if i == 6:
        return dims.conv_dim() * M2_D_CONV
    if i == 24 or i == 25:
        return nh * M2_HEADDIM * M2_D_STATE
    if i == 9 or i == 13:
        return nh * nc * qv
    if i == 10:
        return nc * nh * qv * qv
    if i == 11:
        return nc * qv * qv
    if i == 14 or i == 15:
        return nc * nh * M2_HEADDIM * M2_D_STATE
    raise Error("stage_per_batch: stage " + String(i) + " is TOKEN/GLOBAL")


# ===========================================================================
# Dumps
# ===========================================================================


def oracle_dump(st: Mamba2Stages, state: Mamba2State) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    out.append(st.norm_sumsq.copy())
    out.append(st.norm_out.copy())
    out.append(st.in_proj.copy())
    out.append(st.a_out.copy())
    out.append(st.conv_out.copy())
    out.append(st.silu_out.copy())
    out.append(st.conv_win.copy())
    out.append(st.dt_out.copy())
    out.append(st.xd_out.copy())
    out.append(st.dacs_out.copy())
    out.append(st.seg_l.copy())
    out.append(st.cb_g.copy())
    out.append(st.ydiag_out.copy())
    out.append(st.decay_states.copy())
    out.append(st.cstate_out.copy())
    out.append(st.pass_states.copy())
    out.append(st.yoff_out.copy())
    out.append(st.scan_y.copy())
    out.append(st.skip_out.copy())
    out.append(st.gnorm_gate.copy())
    out.append(st.gnorm_sumsq.copy())
    out.append(st.gnorm_out.copy())
    out.append(st.out_proj.copy())
    out.append(st.residual_out.copy())
    out.append(st.h_last.copy())
    out.append(state.h.copy())
    return out^


def work_slice(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    q0: Int,
    width: Int,
) raises -> List[Float32]:
    """Rows [q0, q0+l) of a [B, T, width] working buffer, as [M, width]
    (DEVIATION 790's card slice, done on the host)."""
    var t_work = q0 + l
    var whole = mamba_download(ctx, buf, b * t_work * width)
    var out = List[Float32]()
    for bb in range(b):
        for li in range(l):
            for j in range(width):
                out.append(whole[((bb * t_work) + (q0 + li)) * width + j])
    return out^


def device_dump(
    ctx: DeviceContext,
    mut d: Mamba2DeviceStages,
    mut state: Mamba2DeviceState,
    b: Int,
    l: Int,
    q0: Int,
    dims: Mamba2Dims,
) raises -> List[List[Float32]]:
    var m = b * l
    var dm = dims.d_model
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var nc = d.nc
    var qv = m2_q_eff()
    comptime p_dim = M2_HEADDIM
    comptime n_state = M2_D_STATE
    var out = List[List[Float32]]()
    out.append(mamba_download(ctx, d.norm_sumsq, m))
    out.append(mamba_download(ctx, d.norm_out, m * dm))
    out.append(mamba_download(ctx, d.in_proj, m * dip))
    out.append(mamba_download(ctx, d.a_out, nh))
    out.append(mamba_download(ctx, d.conv_out, m * cd))
    out.append(mamba_download(ctx, d.silu_out, m * cd))
    out.append(mamba_download(ctx, d.conv_win, b * cd * M2_D_CONV))
    out.append(work_slice(ctx, d.dt_work, b, l, q0, nh))
    out.append(work_slice(ctx, d.xd_work, b, l, q0, nh * p_dim))
    out.append(mamba_download(ctx, d.dacs, b * nh * nc * qv))
    out.append(mamba_download(ctx, d.seg_l, b * nc * nh * qv * qv))
    out.append(mamba_download(ctx, d.cb_g, b * nc * qv * qv))
    out.append(work_slice(ctx, d.ydiag_work, b, l, q0, nh * p_dim))
    out.append(mamba_download(ctx, d.decay, b * nh * nc * qv))
    out.append(mamba_download(ctx, d.cstate, b * nc * nh * p_dim * n_state))
    out.append(
        mamba_download(ctx, d.pass_states, b * nc * nh * p_dim * n_state)
    )
    out.append(work_slice(ctx, d.yoff_work, b, l, q0, nh * p_dim))
    out.append(work_slice(ctx, d.y_work, b, l, q0, nh * p_dim))
    out.append(mamba_download(ctx, d.skip_out, m * nh * p_dim))
    out.append(mamba_download(ctx, d.gnorm_gate, m * di))
    out.append(mamba_download(ctx, d.gnorm_sumsq, m))
    out.append(mamba_download(ctx, d.gnorm_out, m * di))
    out.append(mamba_download(ctx, d.out_proj, m * dm))
    out.append(mamba_download(ctx, d.residual_out, m * dm))
    out.append(mamba_download(ctx, d.h_last, b * nh * p_dim * n_state))
    out.append(mamba_download(ctx, state.h, b * nh * p_dim * n_state))
    return out^


# ===========================================================================
# Comparing (bitwise, by BITS, never by float compare -- row 49)
# ===========================================================================


@fieldwise_init
struct StageDiff(Copyable, Movable):
    var name: String
    var n_cells: Int
    var n_diff: Int
    var first: Int


def compare_stage(
    name: String, a: List[Float32], b: List[Float32], loud: Bool
) raises -> StageDiff:
    if len(a) != len(b):
        raise Error(
            String("mamba2_check: stage ")
            + name
            + " has "
            + String(len(a))
            + " cells on one side and "
            + String(len(b))
            + " on the other"
        )
    var n_diff = 0
    var first = -1
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            n_diff += 1
            if first < 0:
                first = i
    if loud:
        if n_diff == 0:
            print("  OK    " + name + "  (" + String(len(a)) + " cells)")
        else:
            print(
                "  MOVED "
                + name
                + "  "
                + String(n_diff)
                + " of "
                + String(len(a))
                + " cells, first cell "
                + String(first)
                + "  a "
                + bits32_hex(a[first])
                + "  b "
                + bits32_hex(b[first])
            )
    return StageDiff(name, len(a), n_diff, first)


def compare_dumps(
    a: List[List[Float32]], b: List[List[Float32]], loud: Bool
) raises -> List[StageDiff]:
    var out = List[StageDiff]()
    for i in range(N_STAGES):
        out.append(compare_stage(stage_name(i), a[i], b[i], loud))
    return out^


def first_moved(diffs: List[StageDiff]) -> String:
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            return diffs[i].name.copy()
    return String("")


def total_moved(diffs: List[StageDiff]) -> Int:
    var n = 0
    for i in range(len(diffs)):
        n += diffs[i].n_diff
    return n


# ===========================================================================
# One paired run: device + oracle on the same case, fresh zero state (or
# the case's initial_states on BOTH sides).
# ===========================================================================


def run_pair(
    ctx: DeviceContext,
    case_k: Int,
    b: Int,
    l: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises -> List[List[List[Float32]]]:
    """Returns [device_dump, oracle_dump] for case `case_k`'s weights at
    shape (b, l): x is the case's hashed x when the shape matches the
    case, else the case-seed x regenerated at this shape (same element
    rule, so nothing is hand-picked)."""
    var c = m2_corpus_case(case_k)
    var w = m2_case_weights(case_k)
    var dims = w.dims.copy()
    var x: List[Float32]
    if b == c.b and l == c.l:
        x = m2_case_x(case_k)
    else:
        x = corpus_tensor(
            m2_case_seed(c.seed_index), M2_TID_X, b * l * dims.d_model, -2.0, 2.0
        )
    var init = m2_case_init_states(case_k)

    # ---- device
    var dw = Mamba2DeviceWeights(ctx, w)
    var dstate = allocate_inference_cache(ctx, b, dims)
    if c.has_init_states:
        var hb = mamba_upload(ctx, init)
        ctx.enqueue_copy(dst_buf=dstate.h, src_buf=hb)
        ctx.synchronize()
        _ = hb^
    var dstages = Mamba2DeviceStages(ctx, b, l, 0, dims)
    var dx = mamba_upload(ctx, x)
    mamba2_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, c.dt_lo, c.dt_hi, trace, prefix
    )
    var ddump = device_dump(ctx, dstages, dstate, b, l, 0, dims)

    # ---- oracle
    var ostate = Mamba2State(b, dims)
    if c.has_init_states:
        ostate.set_initial_states(init)
    var ost = mamba2_block_oracle(w, x, b, l, c.dt_lo, c.dt_hi, ostate)
    var odump = oracle_dump(ost, ostate)

    var out = List[List[List[Float32]]]()
    out.append(ddump^)
    out.append(odump^)
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    return out^


# ===========================================================================
# GATE (a): card == oracle, bitwise, every stage; and the card's 26 tags
# in section 7's order, once each.
# ===========================================================================


def contract_section_7_tags() -> List[String]:
    var want: List[String] = [
        String("input.x"),
        String("norm.sumsq"),
        String("norm.out"),
        String("in_proj.out"),
        String("A.out"),
        String("conv.out"),
        String("silu.out"),
        String("conv.window"),
        String("dt.out"),
        String("xd.out"),
        String("dacs.out"),
        String("seg.L"),
        String("cb.G"),
        String("ydiag.out"),
        String("decay.states"),
        String("cstate.out"),
        String("pass.states"),
        String("yoff.out"),
        String("scan.y"),
        String("skip.out"),
        String("gnorm.gate"),
        String("gnorm.sumsq"),
        String("gnorm.out"),
        String("out_proj.out"),
        String("residual.out"),
        String("ssd.h_last"),
    ]
    return want^


def check_card_tags(path: String, prefix: String) raises:
    var want = contract_section_7_tags()
    var lines = read_trace_lines(path)
    print(
        "card: "
        + String(len(lines))
        + " records, contract section 7 wants "
        + String(len(want))
    )
    if len(lines) != len(want):
        raise Error(
            String("mamba2_check: the card has ")
            + String(len(lines))
            + " records and contract section 7 lists "
            + String(len(want))
        )
    for i in range(len(lines)):
        var fields = lines[i].split("\t")
        if len(fields) < 2:
            raise Error(
                String("mamba2_check: malformed trace record: ") + lines[i]
            )
        var got = String(fields[1])
        var expect = prefix + "." + want[i]
        if got != expect:
            raise Error(
                String("mamba2_check: card record ")
                + String(i)
                + " is '"
                + got
                + "', contract section 7 wants '"
                + expect
                + "'"
            )
    print("card: all 26 tags present, once each, in section 7's order")


def gate_a(
    ctx: DeviceContext, case_k: Int, b: Int, l: Int, loud: Bool
) raises -> List[StageDiff]:
    var trace = IdentityTrace.to_path(card_path())
    var pair = run_pair(ctx, case_k, b, l, trace, String("m2"))
    var diffs = compare_dumps(pair[0], pair[1], loud)
    return diffs^


# ===========================================================================
# GATE (b): 8 repeated launches, same bits per stage.
# ===========================================================================


def gate_b(ctx: DeviceContext, case_k: Int, b: Int, l: Int) raises:
    var base = List[List[Float32]]()
    for launch in range(8):
        var trace = IdentityTrace.disabled()
        var pair = run_pair(
            ctx, case_k, b, l, trace, String("m2rep") + String(launch)
        )
        if launch == 0:
            base = pair[0].copy()
        else:
            var diffs = compare_dumps(base, pair[0], False)
            var n = total_moved(diffs)
            if n != 0:
                raise Error(
                    String("GATE B FAILED: launch ")
                    + String(launch)
                    + " differs from launch 0 on "
                    + String(n)
                    + " cells (first stage: "
                    + first_moved(diffs)
                    + ")"
                )
    print("GATE B PASS: 8 repeated launches, identical bits, every stage")


# ===========================================================================
# GATE (c): batch-composition invariance, sliced from ONE x, with the
# negative control (rows 0 and 1 of the B = 2 run must differ somewhere).
# ===========================================================================


def row_slice_of(
    dump: List[List[Float32]],
    i: Int,
    bb: Int,
    l: Int,
    dims: Mamba2Dims,
    nc: Int,
) raises -> List[Float32]:
    var kind = stage_kind(i)
    var out = List[Float32]()
    if kind == KIND_GLOBAL:
        for j in range(len(dump[i])):
            out.append(dump[i][j])
        return out^
    if kind == KIND_TOKEN:
        var w = stage_token_width(i, dims)
        for j in range(l * w):
            out.append(dump[i][bb * l * w + j])
        return out^
    var per_b = stage_per_batch(i, dims, nc)
    for j in range(per_b):
        out.append(dump[i][bb * per_b + j])
    return out^


def gate_c(ctx: DeviceContext, case_k: Int, l: Int) raises:
    var c = m2_corpus_case(case_k)
    var w = m2_case_weights(case_k)
    var dims = w.dims.copy()
    var seed = m2_case_seed(c.seed_index)
    var x3 = corpus_tensor(seed, M2_TID_X, 3 * l * dims.d_model, -2.0, 2.0)
    var nc = m2_n_chunks(l)

    var dumps = List[List[List[Float32]]]()
    for bsz in range(1, 4):
        var x = List[Float32]()
        for i in range(bsz * l * dims.d_model):
            x.append(x3[i])
        var dw = Mamba2DeviceWeights(ctx, w)
        var dstate = allocate_inference_cache(ctx, bsz, dims)
        var dstages = Mamba2DeviceStages(ctx, bsz, l, 0, dims)
        var dx = mamba_upload(ctx, x)
        var trace = IdentityTrace.disabled()
        mamba2_block_forward(
            ctx,
            dstages,
            dstate,
            dw,
            dx,
            bsz,
            l,
            c.dt_lo,
            c.dt_hi,
            trace,
            String("m2c") + String(bsz),
        )
        dumps.append(device_dump(ctx, dstages, dstate, bsz, l, 0, dims))
        _ = dw^
        _ = dstate^
        _ = dstages^
        _ = dx^

    # NEGATIVE CONTROL first: rows 0 and 1 of the B = 2 run must differ.
    var control_diff = 0
    for i in range(N_STAGES):
        if stage_kind(i) == KIND_GLOBAL:
            continue
        var r0 = row_slice_of(dumps[1], i, 0, l, dims, nc)
        var r1 = row_slice_of(dumps[1], i, 1, l, dims, nc)
        var d = compare_stage(stage_name(i), r0, r1, False)
        control_diff += d.n_diff
    if control_diff == 0:
        raise Error(
            "GATE C VACUOUS: rows 0 and 1 of the B = 2 run are identical"
            " on every stage -- the slicer cannot tell two rows apart, so"
            " every comparison below would compare a row with itself."
        )

    var fails = 0
    for i in range(N_STAGES):
        if stage_kind(i) == KIND_GLOBAL:
            continue
        # row 0 across B = 1, 2, 3
        var a = row_slice_of(dumps[0], i, 0, l, dims, nc)
        for bi in range(1, 3):
            var bslice = row_slice_of(dumps[bi], i, 0, l, dims, nc)
            fails += compare_stage(stage_name(i), a, bslice, False).n_diff
        # row 1 between B = 2 and B = 3
        var r1a = row_slice_of(dumps[1], i, 1, l, dims, nc)
        var r1b = row_slice_of(dumps[2], i, 1, l, dims, nc)
        fails += compare_stage(stage_name(i), r1a, r1b, False).n_diff
    if fails != 0:
        raise Error(
            String("GATE C FAILED: ")
            + String(fails)
            + " cells depend on batch composition"
        )
    print(
        "GATE C PASS: row bits independent of launch companions (B in"
        " {1,2,3}), negative control differed on",
        control_diff,
        "cells",
    )


# ===========================================================================
# GATE (d): decode == prefill, per token; chunk/state stages at the final
# token (DEVIATION 790's comparability clause); the misalignment control.
# `l1 > 0` runs a prefill of l1 FIRST and decodes tokens l1..l_total-1
# (the handoff + chunk-crossing arm).
# ===========================================================================


def gate_d(
    ctx: DeviceContext, case_k: Int, l1: Int, l_total: Int
) raises -> Int:
    var c = m2_corpus_case(case_k)
    var w = m2_case_weights(case_k)
    var dims = w.dims.copy()
    var b = 1  # decode chains are per-sequence; composition is gate (c)'s
    var seed = m2_case_seed(c.seed_index)
    var x = corpus_tensor(seed, M2_TID_X, b * l_total * dims.d_model, -2.0, 2.0)
    var qv = m2_q_eff()

    # ---- the reference prefill: one call, L tokens, fresh state.
    var dw = Mamba2DeviceWeights(ctx, w)
    var pre_state = allocate_inference_cache(ctx, b, dims)
    var pre_stages = Mamba2DeviceStages(ctx, b, l_total, 0, dims)
    var dx = mamba_upload(ctx, x)
    var trace = IdentityTrace.disabled()
    mamba2_block_forward(
        ctx,
        pre_stages,
        pre_state,
        dw,
        dx,
        b,
        l_total,
        c.dt_lo,
        c.dt_hi,
        trace,
        String("m2d.pre"),
    )
    var pre = device_dump(ctx, pre_stages, pre_state, b, l_total, 0, dims)
    var nc_pre = pre_stages.nc

    # ---- the decode chain: optional prefill of l1, then one token at a
    #      time through the SAME entry point.
    var state = allocate_inference_cache(ctx, b, dims)
    if l1 > 0:
        var x1 = List[Float32]()
        for i in range(b * l1 * dims.d_model):
            x1.append(x[i])
        var st1 = Mamba2DeviceStages(ctx, b, l1, 0, dims)
        var dx1 = mamba_upload(ctx, x1)
        mamba2_block_forward(
            ctx, st1, state, dw, dx1, b, l1, c.dt_lo, c.dt_hi, trace,
            String("m2d.p1"),
        )
        _ = st1^
        _ = dx1^

    comptime p_dim = M2_HEADDIM
    comptime n_state = M2_D_STATE
    var fails = 0
    var misalign_diff = 0
    for li in range(l1, l_total):
        var xt = List[Float32]()
        for j in range(dims.d_model):
            xt.append(x[li * dims.d_model + j])
        var q0 = state.buf_len
        var dstages = Mamba2DeviceStages(ctx, b, 1, q0, dims)
        var dxt = mamba_upload(ctx, xt)
        mamba2_block_forward(
            ctx, dstages, state, dw, dxt, b, 1, c.dt_lo, c.dt_hi, trace,
            String("m2d.t") + String(li),
        )
        var dec = device_dump(ctx, dstages, state, b, 1, q0, dims)

        # token-shaped stages: this token's row vs the prefill's row li.
        for i in range(N_STAGES):
            if stage_kind(i) != KIND_TOKEN:
                continue
            var wdt = stage_token_width(i, dims)
            var prow = List[Float32]()
            for j in range(wdt):
                prow.append(pre[i][li * wdt + j])
            var d = compare_stage(
                stage_name(i) + " [tok " + String(li) + "]",
                dec[i],
                prow,
                False,
            )
            if d.n_diff > 0 and fails == 0:
                print(
                    "  GATE D first mismatch:",
                    stage_name(i),
                    "at token",
                    li,
                    "on",
                    d.n_diff,
                    "cells",
                )
            fails += d.n_diff
            # THE MISALIGNMENT CONTROL, once, on the first decoded token:
            # against prefill row li + 1, which MUST differ -- a gate that
            # cannot fail is not a gate.
            if li == l1 and li + 1 < l_total and stage_kind(i) == KIND_TOKEN:
                var prow2 = List[Float32]()
                for j in range(wdt):
                    prow2.append(pre[i][(li + 1) * wdt + j])
                misalign_diff += compare_stage(
                    stage_name(i), dec[i], prow2, False
                ).n_diff

        # A.out: global, once per step.
        fails += compare_stage(String("A.out"), dec[3], pre[3], False).n_diff

        # at the FINAL token: the carried state, h_last, and the
        # chunk-shaped stages of the open chunk against the prefill's
        # chunk (l_total - 1) // Q (same fill level -- DEVIATION 790).
        if li == l_total - 1:
            fails += compare_stage(
                String("conv.window"), dec[6], pre[6], False
            ).n_diff
            fails += compare_stage(
                String("ssd.h_last"), dec[24], pre[24], False
            ).n_diff
            var cstar = (l_total - 1) // qv
            var nh = dims.nheads
            # dacs.out / decay.states [B, H, C, Q]: chunk cstar. `dec[idx]`
            # is passed as a BORROW, never bound to a var -- List[Float32]
            # is not implicitly copyable (the house trap).
            for si in range(2):
                var idx = 9 if si == 0 else 13
                var wantv = List[Float32]()
                for hh in range(nh):
                    for j in range(qv):
                        wantv.append(
                            pre[idx][(hh * nc_pre + cstar) * qv + j]
                        )
                fails += compare_stage(
                    stage_name(idx) + " [final chunk]", dec[idx], wantv, False
                ).n_diff
            # cstate.out / pass.states [B, C, H, P, N]: chunk cstar.
            for si in range(2):
                var idx = 14 if si == 0 else 15
                var wantv = List[Float32]()
                for j in range(nh * p_dim * n_state):
                    wantv.append(
                        pre[idx][cstar * nh * p_dim * n_state + j]
                    )
                fails += compare_stage(
                    stage_name(idx) + " [final chunk]", dec[idx], wantv, False
                ).n_diff
            # seg.L [B, C, H, Q, Q] and cb.G [B, C, Q, Q]: chunk cstar.
            var wantL = List[Float32]()
            for j in range(nh * qv * qv):
                wantL.append(pre[10][cstar * nh * qv * qv + j])
            fails += compare_stage(
                String("seg.L [final chunk]"), dec[10], wantL, False
            ).n_diff
            var wantG = List[Float32]()
            for j in range(qv * qv):
                wantG.append(pre[11][cstar * qv * qv + j])
            fails += compare_stage(
                String("cb.G [final chunk]"), dec[11], wantG, False
            ).n_diff
            # the resumption h.
            fails += compare_stage(
                String("state.h"), dec[25], pre[25], False
            ).n_diff
        _ = dstages^
        _ = dxt^

    if misalign_diff == 0:
        raise Error(
            "GATE D VACUOUS: the misalignment control (decode token"
            " compared against prefill token + 1) differed on ZERO cells."
            " The gate is comparing a buffer with itself."
        )
    _ = dw^
    _ = pre_state^
    _ = pre_stages^
    _ = dx^
    _ = state^
    return fails


# ===========================================================================
# GATE (d2): the boundary handoff -- prefill 512, then a FRESH call with
# initial_states = h_last, against prefill 520. Section 5: bit-equal ONLY
# because 512 is a multiple of Q.
# ===========================================================================


def gate_d2(ctx: DeviceContext, case_k: Int) raises:
    var c = m2_corpus_case(case_k)
    var w = m2_case_weights(case_k)
    var dims = w.dims.copy()
    var b = 1
    comptime l_a = 512
    comptime l_extra = 8
    var l_all = l_a + l_extra
    var seed = m2_case_seed(c.seed_index)
    var x = corpus_tensor(seed, M2_TID_X, b * l_all * dims.d_model, -2.0, 2.0)
    var trace = IdentityTrace.disabled()

    var dw = Mamba2DeviceWeights(ctx, w)

    # whole prefill
    var s_all = allocate_inference_cache(ctx, b, dims)
    var st_all = Mamba2DeviceStages(ctx, b, l_all, 0, dims)
    var dx_all = mamba_upload(ctx, x)
    mamba2_block_forward(
        ctx, st_all, s_all, dw, dx_all, b, l_all, c.dt_lo, c.dt_hi, trace,
        String("m2d2.all"),
    )
    var all_dump = device_dump(ctx, st_all, s_all, b, l_all, 0, dims)

    # prefill 512, hand h_last to a FRESH call as initial_states
    var s_a = allocate_inference_cache(ctx, b, dims)
    var st_a = Mamba2DeviceStages(ctx, b, l_a, 0, dims)
    var x_a = List[Float32]()
    for i in range(b * l_a * dims.d_model):
        x_a.append(x[i])
    var dx_a = mamba_upload(ctx, x_a)
    mamba2_block_forward(
        ctx, st_a, s_a, dw, dx_a, b, l_a, c.dt_lo, c.dt_hi, trace,
        String("m2d2.a"),
    )
    var s_b = allocate_inference_cache(ctx, b, dims)
    ctx.enqueue_copy(dst_buf=s_b.h, src_buf=st_a.h_last)
    # the conv WINDOW must carry too: h_last alone is NOT the resumption
    # state (section 5); this arm hands over exactly (h_last, window) at a
    # chunk boundary, which is the one case they suffice.
    ctx.enqueue_copy(dst_buf=s_b.conv_win, src_buf=st_a.conv_win)
    ctx.synchronize()
    var st_b = Mamba2DeviceStages(ctx, b, l_extra, 0, dims)
    var x_b = List[Float32]()
    for i in range(b * l_extra * dims.d_model):
        x_b.append(x[b * l_a * dims.d_model + i])
    var dx_b = mamba_upload(ctx, x_b)
    mamba2_block_forward(
        ctx, st_b, s_b, dw, dx_b, b, l_extra, c.dt_lo, c.dt_hi, trace,
        String("m2d2.b"),
    )
    var b_dump = device_dump(ctx, st_b, s_b, b, l_extra, 0, dims)

    # compare the continuation's token stages against rows 512..519.
    var fails = 0
    for i in range(N_STAGES):
        if stage_kind(i) != KIND_TOKEN:
            continue
        var wdt = stage_token_width(i, dims)
        var wantv = List[Float32]()
        for li in range(l_extra):
            for j in range(wdt):
                wantv.append(all_dump[i][(l_a + li) * wdt + j])
        fails += compare_stage(stage_name(i), b_dump[i], wantv, False).n_diff
    fails += compare_stage(
        String("ssd.h_last"), b_dump[24], all_dump[24], False
    ).n_diff
    if fails != 0:
        raise Error(
            String("GATE D2 FAILED: ")
            + String(fails)
            + " cells differ between the h_last handoff continuation and"
            + " the whole prefill (the handoff landed ON a chunk boundary,"
            + " so this is a defect, not the off-boundary caveat)"
        )
    print(
        "GATE D2 PASS: prefill 512 + initial_states = h_last continuation"
        " == prefill 520, bitwise, on the continuation's tokens and h_last"
    )
    _ = dw^
    _ = s_all^
    _ = st_all^
    _ = dx_all^
    _ = s_a^
    _ = st_a^
    _ = dx_a^
    _ = s_b^
    _ = st_b^
    _ = dx_b^


# ===========================================================================
# GATE (e): the planted nonfinite audit. Reach measured (the plant read
# back off the device), refusal BY NAME, ZERO stages recorded; plus the
# clean-call control.
# ===========================================================================


def plant_in_list(mut xs: List[Float32], bits: UInt32):
    xs[len(xs) // 2] = f32_from_bits(bits)


def gate_e_one_plant(
    ctx: DeviceContext,
    case_k: Int,
    b: Int,
    l: Int,
    name_idx: Int,
    bits: UInt32,
) raises -> String:
    """Plant `bits` at cell len/2 of input `name_idx` (the refusal walk's
    order), run, and require refusal naming that input with 0 records.
    Returns the refused name."""
    var c = m2_corpus_case(case_k)
    var w = m2_case_weights(case_k)
    var dims = w.dims.copy()
    var x = corpus_tensor(
        m2_case_seed(c.seed_index), M2_TID_X, b * l * dims.d_model, -2.0, 2.0
    )
    var names: List[String] = [
        String("x"),
        String("norm.weight"),
        String("in_proj.weight"),
        String("conv1d.weight"),
        String("conv1d.bias"),
        String("dt_bias"),
        String("A_log"),
        String("D"),
        String("norm_gated.weight"),
        String("out_proj.weight"),
        String("state.conv_win"),
        String("state.h"),
        String("state.buf_xbc"),
        String("state.buf_dtraw"),
    ]
    if name_idx == 0:
        plant_in_list(x, bits)
    if name_idx == 1:
        plant_in_list(w.norm_w, bits)
    if name_idx == 2:
        plant_in_list(w.w_in, bits)
    if name_idx == 3:
        plant_in_list(w.conv_w, bits)
    if name_idx == 4:
        plant_in_list(w.conv_b, bits)
    if name_idx == 5:
        plant_in_list(w.dt_bias, bits)
    if name_idx == 6:
        plant_in_list(w.a_log, bits)
    if name_idx == 7:
        plant_in_list(w.d_skip, bits)
    if name_idx == 8:
        plant_in_list(w.gnorm_w, bits)
    if name_idx == 9:
        plant_in_list(w.w_out, bits)

    var dw = Mamba2DeviceWeights(ctx, w)
    var dstate = allocate_inference_cache(ctx, b, dims)
    var dstages = Mamba2DeviceStages(ctx, b, l, 0, dims)
    var dx = mamba_upload(ctx, x)

    # State plants (the three-piece state is a named input too, contract
    # section 6): download the zero buffer, plant at len/2, copy back.
    if name_idx >= 10:
        var nsz: Int
        if name_idx == 10:
            nsz = b * dims.conv_dim() * M2_D_CONV
        elif name_idx == 11:
            nsz = b * dims.nheads * M2_HEADDIM * M2_D_STATE
        elif name_idx == 12:
            nsz = b * M2_CHUNK_SIZE * dims.conv_dim()
        else:
            nsz = b * M2_CHUNK_SIZE * dims.nheads
        var host = List[Float32]()
        for _ in range(nsz):
            host.append(0.0)
        plant_in_list(host, bits)
        var planted = mamba_upload(ctx, host)
        if name_idx == 10:
            ctx.enqueue_copy(dst_buf=dstate.conv_win, src_buf=planted)
        elif name_idx == 11:
            ctx.enqueue_copy(dst_buf=dstate.h, src_buf=planted)
        elif name_idx == 12:
            ctx.enqueue_copy(dst_buf=dstate.buf_xbc, src_buf=planted)
        else:
            ctx.enqueue_copy(dst_buf=dstate.buf_dtraw, src_buf=planted)
        ctx.synchronize()
        _ = planted^

    # REACH: the planted buffer read back off the device, nonfinite cells
    # counted. A refusal that fired for another reason must not pass.
    var reached = 0
    var readback: List[Float32]
    if name_idx == 0:
        readback = mamba_download(ctx, dx, b * l * dims.d_model)
    elif name_idx == 1:
        readback = mamba_download(ctx, dw.norm_w, dims.d_model)
    elif name_idx == 2:
        readback = mamba_download(
            ctx, dw.w_in, dims.d_in_proj() * dims.d_model
        )
    elif name_idx == 3:
        readback = mamba_download(ctx, dw.conv_w, dims.conv_dim() * M2_D_CONV)
    elif name_idx == 4:
        readback = mamba_download(ctx, dw.conv_b, dims.conv_dim())
    elif name_idx == 5:
        readback = mamba_download(ctx, dw.dt_bias, dims.nheads)
    elif name_idx == 6:
        readback = mamba_download(ctx, dw.a_log, dims.nheads)
    elif name_idx == 7:
        readback = mamba_download(ctx, dw.d_skip, dims.nheads)
    elif name_idx == 8:
        readback = mamba_download(ctx, dw.gnorm_w, dims.d_inner)
    elif name_idx == 9:
        readback = mamba_download(ctx, dw.w_out, dims.d_model * dims.d_inner)
    elif name_idx == 10:
        readback = mamba_download(
            ctx, dstate.conv_win, b * dims.conv_dim() * M2_D_CONV
        )
    elif name_idx == 11:
        readback = mamba_download(
            ctx, dstate.h, b * dims.nheads * M2_HEADDIM * M2_D_STATE
        )
    elif name_idx == 12:
        readback = mamba_download(
            ctx, dstate.buf_xbc, b * M2_CHUNK_SIZE * dims.conv_dim()
        )
    else:
        readback = mamba_download(
            ctx, dstate.buf_dtraw, b * M2_CHUNK_SIZE * dims.nheads
        )
    for i in range(len(readback)):
        var au = bitcast[DType.uint32](readback[i]) & UInt32(0x7FFFFFFF)
        if au >= UInt32(0x7F800000):
            reached += 1
    if reached != 1:
        raise Error(
            String("GATE E VACUOUS: the plant in '")
            + names[name_idx]
            + "' was not found on the device (reached = "
            + String(reached)
            + "); the refusal below would prove nothing"
        )

    var trace = IdentityTrace.to_path(
        card_path() + ".refusal", String(""), True
    )
    var refused = String("")
    try:
        mamba2_block_forward(
            ctx, dstages, dstate, dw, dx, b, l, c.dt_lo, c.dt_hi, trace,
            String("m2e") + String(name_idx),
        )
    except e:
        refused = String(e)
    if refused == "":
        raise Error(
            String("GATE E FAILED: the plant in '")
            + names[name_idx]
            + "' was NOT refused"
        )
    if refused.find(names[name_idx]) < 0:
        raise Error(
            String("GATE E FAILED: the plant in '")
            + names[name_idx]
            + "' was refused under the wrong name: "
            + refused
        )
    if trace.seq != 0:
        raise Error(
            String("GATE E FAILED: ")
            + String(trace.seq)
            + " stages were recorded before the refusal of '"
            + names[name_idx]
            + "' (contract section 6: zero)"
        )
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    return names[name_idx].copy()


def gate_e(ctx: DeviceContext, case_k: Int, b: Int, l: Int) raises:
    # THE CLEAN CONTROL FIRST: without it a walk that raises
    # unconditionally passes every plant while gating nothing.
    var c = m2_corpus_case(case_k)
    var trace = IdentityTrace.to_path(card_path() + ".clean", String(""), True)
    var pair = run_pair(ctx, case_k, b, l, trace, String("m2e.clean"))
    if trace.seq != 26:
        raise Error(
            String("GATE E VACUOUS: the clean control recorded ")
            + String(trace.seq)
            + " stages, not 26 -- the refusal walk (or the recorder) is"
            + " broken and every plant below would pass for the wrong"
            + " reason"
        )
    _ = pair^
    var n = 0
    for idx in range(14):
        var nm = gate_e_one_plant(ctx, case_k, b, l, idx, BITS_QNAN)
        var nm2 = gate_e_one_plant(ctx, case_k, b, l, idx, BITS_POS_INF)
        if nm != nm2:
            raise Error("GATE E: plant naming disagreed between NaN and inf")
        n += 2
    print(
        "GATE E PASS:",
        n,
        "plants (quiet NaN + inf per each of the 14 names, cell len/2),",
        "each read back off the device first, each refused BY NAME with 0"
        " stages recorded; clean control recorded all 26",
    )


# ===========================================================================
# GATE (g)'s byte gate: MISSING-FIXTURE REFUSAL. The corpus lane has not
# generated mamba2 cases; until it does, this arm refuses BY NAME.
# ===========================================================================


def _read_fixture_bytes(path: String, case_name: String) raises -> List[UInt8]:
    """BINARY-SAFE fixture read: the bench lanes' `read_bytes()` pattern
    (rf_bench.mojo::_read_all). The first spelling here used text-mode
    `fh.read()`, which raises on raw float32 bytes WITH THE FILE PRESENT,
    and a catch-all then renamed that failure "MISSING FIXTURE" -- the
    smoke-arm defect class (every failure wearing one name). Each failure
    kind now refuses under its OWN name; refuse-not-skip stands."""
    var opened = False
    var bytes = List[UInt8]()
    try:
        var f = open(path, "r")
        opened = True
        bytes = f.read_bytes()
        f.close()
    except e:
        if not opened:
            raise Error(
                String("MISSING FIXTURE: cannot open ")
                + path
                + " (case '"
                + case_name
                + "'). Either the corpus generator has not emitted it, or"
                + " the working directory is not the repo root -- the path"
                + " is repo-root-relative. This arm REFUSES rather than"
                + " skips (contract 8g)."
            )
        raise Error(
            String("UNREADABLE FIXTURE: ")
            + path
            + " opened but read_bytes failed: "
            + String(e)
        )
    return bytes^


def _compare_fixture_file(
    dir_path: String, fname: String, expected: List[Float32], case_name: String
) raises:
    """THE FIRST ASSERTION of gate (g): file bytes == this check's own
    generator's bytes, element by element, little-endian float32. Two
    independently written implementations of `mojolearn.mamba.corpus.
    hash.v1` agreeing bit for bit is itself a gate (the Mamba-1 fixture's
    rule); a disagreement means one of the two tables is wrong ON THE
    RECORD, never silently."""
    var path = dir_path + "/" + fname
    var bytes = _read_fixture_bytes(path, case_name)
    if len(bytes) != 4 * len(expected):
        raise Error(
            String("WRONG-SIZE FIXTURE: ")
            + path
            + " holds "
            + String(len(bytes))
            + " bytes; the generator's tensor is "
            + String(len(expected))
            + " float32 values ("
            + String(4 * len(expected))
            + " bytes)"
        )
    var n_diff = 0
    var first = -1
    var first_file = UInt32(0)
    for i in range(len(expected)):
        var u = (
            UInt32(bytes[4 * i])
            | (UInt32(bytes[4 * i + 1]) << 8)
            | (UInt32(bytes[4 * i + 2]) << 16)
            | (UInt32(bytes[4 * i + 3]) << 24)
        )
        if u != bitcast[DType.uint32](expected[i]):
            n_diff += 1
            if first < 0:
                first = i
                first_file = u
    if n_diff != 0:
        raise Error(
            String("FIXTURE BYTE MISMATCH: ")
            + path
            + ": "
            + String(n_diff)
            + " of "
            + String(len(expected))
            + " cells differ from the in-check generator, first cell "
            + String(first)
            + " file "
            + bits32_hex(f32_from_bits(first_file))
            + " generator "
            + bits32_hex(expected[first])
            + ". Two implementations of mojolearn.mamba.corpus.hash.v1"
            + " disagree; one of the two tables is wrong ON THE RECORD."
        )


def gate_corpus(case_k: Int) raises:
    """Gate (g)'s byte gate: EVERY input tensor of the case (x, the ten
    parameters, initial_states where the case carries one) byte-compared
    against this check's reimplementation of the generator."""
    var c = m2_corpus_case(case_k)
    var nm = String(c.name)
    var dir_path = m2_corpus_dir(case_k)
    var w = m2_case_weights(case_k)
    _compare_fixture_file(dir_path, String("x.f32"), m2_case_x(case_k), nm)
    _compare_fixture_file(
        dir_path, String("block_norm.weight.f32"), w.norm_w, nm
    )
    _compare_fixture_file(dir_path, String("in_proj.weight.f32"), w.w_in, nm)
    _compare_fixture_file(dir_path, String("conv1d.weight.f32"), w.conv_w, nm)
    _compare_fixture_file(dir_path, String("conv1d.bias.f32"), w.conv_b, nm)
    _compare_fixture_file(dir_path, String("dt_bias.f32"), w.dt_bias, nm)
    _compare_fixture_file(dir_path, String("A_log.f32"), w.a_log, nm)
    _compare_fixture_file(dir_path, String("D.f32"), w.d_skip, nm)
    _compare_fixture_file(dir_path, String("norm.weight.f32"), w.gnorm_w, nm)
    _compare_fixture_file(dir_path, String("out_proj.weight.f32"), w.w_out, nm)
    if c.has_init_states:
        _compare_fixture_file(
            dir_path,
            String("initial_states.f32"),
            m2_case_init_states(case_k),
            nm,
        )
    print(
        "corpus case",
        c.name,
        ": every input tensor byte-identical to the in-check generator",
    )


# ===========================================================================
# GATE (f): the armed run. One arm per build; verdict INVERTED; the
# fixture's witnessing property asserted before the arm is credited.
# ===========================================================================


def count_dt_nonuniform(case_k: Int, b: Int, l: Int) raises -> Int:
    """Host-side witness counter: pairs of differing dt cells within the
    first chunk (SEGSUM needs a nonuniform dt to bite)."""
    var c = m2_corpus_case(case_k)
    var w = m2_case_weights(case_k)
    var state = Mamba2State(b, w.dims)
    var x = m2_case_x(case_k) if (b == c.b and l == c.l) else corpus_tensor(
        m2_case_seed(c.seed_index), M2_TID_X, b * l * w.dims.d_model, -2.0, 2.0
    )
    var st = mamba2_block_oracle(w, x, b, l, c.dt_lo, c.dt_hi, state)
    var n = 0
    for i in range(1, len(st.dt_out)):
        if bitcast[DType.uint32](st.dt_out[i]) != bitcast[DType.uint32](
            st.dt_out[0]
        ):
            n += 1
    return n


def clamp_witness_weights() raises -> Mamba2Weights:
    """The armed CLAMP_BEFORE_SOFTPLUS run's PLANTED witnessing fixture,
    deterministic BY CONSTRUCTION -- not by a lucky hashed draw. Found
    necessary at first arming: the corpus dt_limit case has ONE head at
    d_model 32, its single hashed dt_bias landed in the (0.001, 0.1)
    INTERIOR, the clamp bound ZERO cells, and the vacuity refusal fired
    exactly as designed (that refusal stays, below, guarding this plant
    against regression too).

    Construction: case 2's (m2_base_b3_l4_d64, TWO heads, benign hashed
    weights) with dt_bias planted per head, one head per limit:
    dt_bias[0] = -8.0 -> biased dt sits near -8 (dt_raw's fan-in-scaled
    spread is ~0.3 sigma), softplus < 0.001 on every head-0 cell, the LO
    limit binds; dt_bias[1] = -1.0 -> softplus ~0.31 > 0.1 on every
    head-1 cell, the HI limit binds. Both margins are ~4 sigma, so both
    limits bind on EVERY cell of their head under the (0.001, 0.1)
    limits, and the order swap (clamping the biased value near -8/-1
    FIRST, then softplus) cannot fail to move dt.out."""
    var w = m2_case_weights(2)
    if w.dims.nheads < 2:
        raise Error(
            "clamp_witness_weights: the witness needs two heads (one per"
            " limit); case 2 is no longer d_model 64"
        )
    w.dt_bias[0] = -8.0
    w.dt_bias[1] = -1.0
    return w^


def run_pair_custom(
    ctx: DeviceContext,
    w: Mamba2Weights,
    x: List[Float32],
    b: Int,
    l: Int,
    dt_lo: Float32,
    dt_hi: Float32,
) raises -> List[List[List[Float32]]]:
    """`run_pair` on an EXPLICIT (weights, x, dt_limit) instead of a
    corpus case -- the planted sabotage fixtures go through here. Fresh
    zero state on both sides."""
    var dims = w.dims.copy()
    var dw = Mamba2DeviceWeights(ctx, w)
    var dstate = allocate_inference_cache(ctx, b, dims)
    var dstages = Mamba2DeviceStages(ctx, b, l, 0, dims)
    var dx = mamba_upload(ctx, x)
    var trace = IdentityTrace.disabled()
    mamba2_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, dt_lo, dt_hi, trace,
        String("m2cw"),
    )
    var ddump = device_dump(ctx, dstages, dstate, b, l, 0, dims)
    var ostate = Mamba2State(b, dims)
    var ost = mamba2_block_oracle(w, x, b, l, dt_lo, dt_hi, ostate)
    var odump = oracle_dump(ost, ostate)
    var out = List[List[List[Float32]]]()
    out.append(ddump^)
    out.append(odump^)
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    return out^


def sabotage_main(ctx: DeviceContext) raises:
    var arm = mamba2_sabotage_name()
    print(
        "SABOTAGE ARM ARMED:",
        arm,
        "-- the verdict is INVERTED: a clean compare is the FAILURE"
        " ([[reached-but-inert]]).",
    )
    var decode_arm = False
    for a in argv():
        if a == "decode":
            decode_arm = True

    if arm == "STEP_UPSTREAM_RECURRENCE":
        if not decode_arm:
            raise Error(
                "STEP_UPSTREAM_RECURRENCE gates gate (d): run this build"
                " with the `decode` argument"
            )
        var fails = gate_d(ctx, 1, 0, 4)
        if fails == 0:
            raise Error(
                "SABOTAGE ARM REPORTED ZERO MISMATCHES on gate (d): the"
                " upstream step recurrence agreed with the chunked prefill"
                " bitwise, which DEVIATION 786 says is impossible -- the"
                " gate is blind or the arm is not reached."
            )
        print(
            "SABOTAGE ARM FAILS AS REQUIRED:",
            fails,
            "cells moved on gate (d). DEVIATION 786's clause is load"
            " bearing.",
        )
        return

    # every other arm gates clause (a) on its witnessing fixture.
    var case_k = 1  # m2_base_b2_l4_d32: hashed values, any-shape arms
    var b = 2
    var l = 4
    var expect_first = String("")
    var token_only = False
    if arm == "SEGSUM_DESCENDING":
        expect_first = String("dacs.out")
        var nonuni = count_dt_nonuniform(case_k, b, l)
        if nonuni == 0:
            raise Error(
                "SABOTAGE VACUOUS: every dt cell is identical; a reversed"
                " fold over a uniform dt is bitwise inert"
            )
        print("  witness: nonuniform dt cells =", nonuni)
    elif arm == "CHUNK_SIZE_128":
        case_k = 5
        b = 2
        l = 257
        expect_first = String("ydiag.out")
        token_only = True
        if l <= 128:
            raise Error("SABOTAGE VACUOUS: CHUNK_SIZE_128 needs L > 128")
        print(
            "  witness: L = 257 > 128; chunk-shaped stages change SHAPE"
            " under this arm, so the compare covers token-shaped stages"
            " (stated, not silent)"
        )
    elif arm == "STATEPASS_MATRIX" or arm == "STATEPASS_UNFUSED":
        # BOTH witnessing fixtures, contract 8f: the three-chunk case AND
        # the nonzero-initial_states two-chunk case; each must move.
        var cases: List[Int] = [8, 15]
        for i in range(2):
            var ck = cases[i]
            var cc = m2_corpus_case(ck)
            if ck == 8 and m2_n_chunks(cc.l) < 3:
                raise Error(
                    "SABOTAGE VACUOUS: the L = 513 case no longer has >= 3"
                    " working chunks"
                )
            if ck == 15:
                if not cc.has_init_states:
                    raise Error(
                        "SABOTAGE VACUOUS: the initstate case carries no"
                        " initial_states"
                    )
                var init = m2_case_init_states(ck)
                var nz = 0
                for j in range(len(init)):
                    if bitcast[DType.uint32](init[j]) != UInt32(0):
                        nz += 1
                if nz == 0:
                    raise Error(
                        "SABOTAGE VACUOUS: the planted initial_states are"
                        " all zeros"
                    )
                print("  witness: nonzero initial_states cells =", nz)
            var diffs = gate_a(ctx, ck, cc.b, cc.l, False)
            var n = total_moved(diffs)
            if n == 0:
                raise Error(
                    String("SABOTAGE ARM INERT on its witnessing fixture ")
                    + String(cc.name)
                    + " -- a zero-init two-chunk case is bitwise inert"
                    + " there by DEVIATION 785's own math, so a fixture"
                    + " that cannot witness has been substituted, or the"
                    + " arm is not reached"
                )
            var fm = first_moved(diffs)
            if fm != "pass.states":
                raise Error(
                    String("SABOTAGE ARM moved '")
                    + fm
                    + "' first; DEVIATION 785's seam writes pass.states"
                    + " and nothing earlier may move"
                )
            print("  ", cc.name, "-> moved, first stage pass.states")
        print(
            "SABOTAGE ARM FAILS AS REQUIRED on both witnessing fixtures."
        )
        return
    elif arm == "PAIR_DT_B":
        expect_first = String("xd.out")
    elif arm == "FOLD_SERIAL_ZERO_SEED":
        case_k = 5
        b = 2
        l = 257
        expect_first = String("ydiag.out")
        print(
            "  witness: L = 257 puts real values in BOTH k = 256 leaves;"
            " at L <= 128 the second leaf folds only zeros and the arm is"
            " bitwise inert"
        )
    elif arm == "S6_BIAS_LAST" or arm == "S6_TAPS_REVERSED":
        expect_first = String("conv.out")
    elif arm == "CLAMP_BEFORE_SOFTPLUS":
        # The PLANTED fixture (clamp_witness_weights' docstring is the
        # record of why): case 2's shape, dt_bias planted one head per
        # limit, run under the (0.001, 0.1) limits. Verdict handled
        # inline, like the STATEPASS branch.
        var cw = m2_corpus_case(2)
        var ww = clamp_witness_weights()
        var xw = m2_case_x(2)
        var lo = Float32(0.001)
        var hi = Float32(0.1)
        var pairw = run_pair_custom(ctx, ww, xw, cw.b, cw.l, lo, hi)
        # Witnessing asserted, BOTH limits, counted on the ORACLE's clean
        # dt.out (the vacuity refusal stays: a future fixture regression
        # must fire it, exactly as the first hashed fixture did).
        var lo_bind = 0
        var hi_bind = 0
        for i in range(len(pairw[1][STAGE_DT])):
            var u = bitcast[DType.uint32](pairw[1][STAGE_DT][i])
            if u == bitcast[DType.uint32](lo):
                lo_bind += 1
            if u == bitcast[DType.uint32](hi):
                hi_bind += 1
        if lo_bind == 0 or hi_bind == 0:
            raise Error(
                String("SABOTAGE VACUOUS: the planted dt_limit fixture")
                + " binds lo on "
                + String(lo_bind)
                + " and hi on "
                + String(hi_bind)
                + " cells; BOTH limits must bind or the order swap is not"
                + " fully witnessed"
            )
        print(
            "  witness: clamp binds lo on", lo_bind, "and hi on", hi_bind,
            "cells (planted, by construction)",
        )
        var diffs_c = compare_dumps(pairw[0], pairw[1], False)
        var n_c = total_moved(diffs_c)
        if n_c == 0:
            raise Error(
                "SABOTAGE ARM REPORTED ZERO MISMATCHES on the planted"
                " witness: the gate is blind, or the arm is not reached"
                " ([[reached-but-inert]])."
            )
        var fm_c = first_moved(diffs_c)
        if fm_c != "dt.out":
            raise Error(
                String("SABOTAGE ARM moved '")
                + fm_c
                + "' first; the S9 order swap writes dt.out and nothing"
                + " earlier may move"
            )
        print(
            "SABOTAGE ARM FAILS AS REQUIRED:",
            n_c,
            "cells moved, first stage dt.out (the stage the sabotaged"
            " seam writes). The undone clause is load bearing.",
        )
        return
    elif arm == "GATE_NORM_BEFORE":
        expect_first = String("gnorm.gate")
    else:
        raise Error(String("mamba2_check: unknown armed name ") + arm)

    var diffs: List[StageDiff]
    if token_only:
        # chunk-shaped buffers change shape under this arm: compare the
        # token-shaped and batch/global stages only.
        var trace = IdentityTrace.disabled()
        var pair = run_pair(ctx, case_k, b, l, trace, String("m2sab"))
        diffs = List[StageDiff]()
        for i in range(N_STAGES):
            if stage_kind(i) == KIND_CHUNK:
                continue
            diffs.append(
                compare_stage(stage_name(i), pair[0][i], pair[1][i], False)
            )
    else:
        diffs = gate_a(ctx, case_k, b, l, False)
    var n = total_moved(diffs)
    if n == 0:
        raise Error(
            "SABOTAGE ARM REPORTED ZERO MISMATCHES: the gate is blind, or"
            " the arm is not reached on its witnessing fixture"
            " ([[reached-but-inert]])."
        )
    var fm = first_moved(diffs)
    if fm != expect_first:
        raise Error(
            String("SABOTAGE ARM moved '")
            + fm
            + "' first; its own seam writes '"
            + expect_first
            + "' and nothing earlier may move"
        )
    print(
        "SABOTAGE ARM FAILS AS REQUIRED:",
        n,
        "cells moved, first stage",
        fm,
        "(the stage the sabotaged seam writes). The undone clause is load"
        " bearing.",
    )


# ===========================================================================
# main
# ===========================================================================


def main() raises:
    var mode = String("gates")
    for a in argv():
        if a == "decode":
            mode = String("decode")
        if a == "decode-cross":
            mode = String("decode-cross")
        if a == "handoff":
            mode = String("handoff")
        if a == "refusal":
            mode = String("refusal")
        if a == "corpus":
            mode = String("corpus")

    print(
        "mamba2 block gate (IDENTICAL_MAMBA2_CONTRACT.md section 8);"
        " profile mojolearn.identical.mamba2.fp32.v1; build mode",
        mode_name(),
    )

    comptime if BLOCK2_ANY_SABOTAGE:
        var ctx_s = DeviceContext()
        sabotage_main(ctx_s)
        return

    var ctx = DeviceContext()

    if mode == "corpus":
        # walk every case, every input tensor, byte-compared; the FIRST
        # missing/unreadable/wrong-size/mismatching file refuses under its
        # own name.
        for k in range(17):
            gate_corpus(k)
        print(
            "GATE (g) BYTE GATE PASS: all 17 cases' input tensors are"
            " byte-identical between the corpus files and this check's"
            " generator (two implementations of the hash spec agree)."
        )
        return

    if mode == "refusal":
        gate_e(ctx, 1, 2, 4)
        return

    if mode == "decode":
        var fails = gate_d(ctx, 1, 0, 4)
        if fails != 0:
            raise Error(
                String("GATE D FAILED: ")
                + String(fails)
                + " cells differ between the decode chain and the prefill"
            )
        print(
            "GATE D PASS: decode == prefill bitwise, per token, all token"
            " stages + final-token chunk/state stages (misalignment"
            " control differed)"
        )
        return

    if mode == "decode-cross":
        # prefill 250, decode through 262: crosses the Q = 256 boundary
        # at token 256 -- a boundary never crossed is a clause never
        # gated. Also the handoff arm (prefill L1 + decode L2). Case 3's
        # weights (m2_base_b1_l256_d32); x regenerated at L = 262 under
        # the same element rule.
        var fails = gate_d(ctx, 3, 250, 262)
        if fails != 0:
            raise Error(
                String("GATE D (cross) FAILED: ")
                + String(fails)
                + " cells differ; the run crossed the chunk boundary at"
                + " token 256"
            )
        print(
            "GATE D (cross) PASS: prefill 250 + decode through 262 =="
            " prefill 262, bitwise; the chain crossed the Q = 256"
            " boundary and the buffer emptied and refilled"
        )
        return

    if mode == "handoff":
        gate_d2(ctx, 3)
        return

    # default: gates (a) + card, (b), (c) at the env shape (ONE shape per
    # build -- the 2026-08-23 crash rule).
    var b = env_int(String("MOJOLEARN_MAMBA2_CHECK_B"), 1)
    var l = env_int(String("MOJOLEARN_MAMBA2_CHECK_L"), 4)
    var dm = env_int(String("MOJOLEARN_MAMBA2_CHECK_DM"), 32)
    var case_k = 0
    if dm == 64:
        case_k = 2  # m2_base_b3_l4_d64's weights; x regenerated at (B, L)
    elif dm != 32:
        raise Error("mamba2_check: d_model must be 32 or 64 (contract s3)")
    print("shape: B =", b, "L =", l, "d_model =", dm)

    var diffs = gate_a(ctx, case_k, b, l, True)
    var n = total_moved(diffs)
    if n != 0:
        raise Error(
            String("GATE A FAILED: ")
            + String(n)
            + " cells differ between the device card and the host oracle"
            + " (first stage: "
            + first_moved(diffs)
            + ")"
        )
    check_card_tags(card_path(), String("m2"))
    print(
        "GATE A PASS: every stage bit-identical to the oracle at this"
        " shape."
    )
    gate_b(ctx, case_k, b, l)
    gate_c(ctx, case_k, l)
    print(
        "One shape, one launch history, one box. Everything else in"
        " contract section 8 is OWED: the sabotage arms (one build each),"
        " decode / decode-cross / handoff / refusal / corpus arms, the"
        " other shapes, FAST recording, and every column that is not this"
        " machine."
    )
