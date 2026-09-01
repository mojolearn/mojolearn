# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate file of profile `mojolearn.identical.mamba3.siso.fp32.v1`
(`mamba/IDENTICAL_MAMBA3_CONTRACT.md` section 8). NOT A PORT: it runs the
device block (`mamba/impl/mamba_ssm/modules/mamba3.mojo` around
`ops/mamba3_siso.mojo`) against the host oracle
(`mamba/checks/mamba3_oracle.mojo`) and compares every recorded stage BY
BITS.

**NOTHING IN THIS FILE HAS RUN.** Every command below is RUN OWED; no
kernel here has ever compiled. PRECONDITION (contract phase M3-0, also
RUN OWED): the portable trig pair's device certification,
`tools/with_identical_mode.sh pixi run check-portable-trig` (or
`pixi run mojo run -I . checks/portable_trig_check.mojo`), and mamba2's
gates re-green if any shared file moved. The pixi task registration
(`check-mamba3-block`) is the ORCHESTRATOR's to confirm; the raw commands
are spelled:

    # gates (a) card==oracle + the 28 card tags, (b) 8 repeated launches,
    # (c) batch composition with the negative control; one shape per
    # build (MOJOLEARN_MAMBA3_CHECK_B/_L/_DM, defaults 1/4/32 -- the
    # 2026-08-23 one-compile-at-a-time rule):
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba3_check.mojo

    # gate (d): decode == prefill per token (case m3_base_b2_l4_d32),
    # with the misalignment control:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba3_check.mojo decode

    # gate (d)'s chunk-crossing arm (contract 8d: prefill 60, decode
    # through token 70; crosses the Q = 64 boundary):
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba3_check.mojo decode-cross

    # the Input_States continuation (contract section 5 claim 2): the
    # SAME continuation call on both sides, device vs oracle, bitwise --
    # this is NOT a bitwise claim against an unbroken prefill; no d2
    # bitwise arm exists BY THEOREM (DEVIATION 831; anyone who "fixes"
    # this to a prefill handoff gate has misread the trapezoid). The
    # prefill-equivalence row is the CORPUS's, under tolerance:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba3_check.mojo continuation

    # gate (e): the planted nonfinite refusal audit + clean control
    # (18 names x quiet NaN + inf, reach read back off the device):
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba3_check.mojo refusal

    # gate (g)'s byte gate: REFUSES BY NAME until the corpus lane
    # generates mamba/corpus/mamba3/ (a missing fixture is a refusal,
    # never a skip); afterwards every input tensor of all 18 cases is
    # byte-compared against this check's own generator:
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        mamba/checks/mamba3_check.mojo corpus

    # gate (f): ONE sabotage arm per build, verdict INVERTED; each arm
    # names its witnessing fixture and the check ASSERTS the witnessing
    # property before crediting the arm:
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_MAMBA3_SABOTAGE_SEGSUM_DESCENDING=1 -I . \\
        mamba/checks/mamba3_check.mojo
    #   ... likewise CHUNK_SIZE_32, TRAP_LEFT_ONLY, ANGLE_MOD_PER_CHUNK,
    #   ANGLE_MOD_AT_END, ROTATE_HALF_SPLIT, DIAG_INCLUDE_SUBTRACT,
    #   STATE_TERM_SCALE_FIRST, A_FLOOR_UNCLAMPED, RESUME_KERNEL_ASSOC;
    #   FOLD_SERIAL_ZERO_SEED refuses VACUOUS by name (DEVIATION 834);
    #   and the decode arm:
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_MAMBA3_SABOTAGE_STEP_UPSTREAM_RECURRENCE=1 -I . \\
        mamba/checks/mamba3_check.mojo decode

WITNESSING, BY CONSTRUCTION AND ASSERTED (contract 8f's table):
SEGSUM_DESCENDING: nonuniform ADT counted on the oracle's clean card;
CHUNK_SIZE_32: L = 65 > 32 asserted, chunk-shaped stages change SHAPE so
the armed compare covers non-chunk stages and says so; TRAP_LEFT_ONLY:
cells where scale != gamma (recomputed through the same seam functions)
counted -- a uniform-trap fixture binds zero and refuses VACUOUS;
ANGLE_MOD_PER_CHUNK / ANGLE_MOD_AT_END: 2pi-seam mod engagements at
non-final, non-chunk-boundary tokens counted by re-running the serial
recurrence on the host from the oracle's own card (the planted
m3_adv_angle_crossing case); ROTATE_HALF_SPLIT / DIAG_INCLUDE_SUBTRACT:
nonzero angle.theta cells counted (rotation rounding is the separator);
A_FLOOR_UNCLAMPED: cells clamped EXACTLY to -A_floor counted BY BITS on
the clean A.out, plus at least one unclamped cell (the planted
m3_adv_a_floor case); STATE_TERM_SCALE_FIRST: >= 2 working chunks AND
nonzero state entering chunk 1+ counted; RESUME_KERNEL_ASSOC: nonzero
Input_States pieces counted; STEP_UPSTREAM_RECURRENCE: an L >= 2 decode.
An armed run whose fixture fails its witnessing assertion raises
VACUOUS, never passes.

GATE (d)'s COMPARABILITY (DEVIATION 832(iii), the oracle's docstring is
the authority): token-shaped stages compare bitwise PER TOKEN except
`trap.scale` and `kscale.out`, which are NOT prefix-stable (a token's
beta' leg becomes real only when its successor arrives) and compare at
the FINAL decoded token ONLY; chunk-shaped stages (`dacs.out`, `seg.L`,
`pass.states`) compare at the final token against the prefill's chunks
from (L-2)//Q onward (same fill by the sealed-buffer construction); the
carried state (state.h, state.theta) and the four reports compare at the
final token. The misalignment control (decode t against prefill t+1,
must differ) guards the gate against comparing a buffer with itself.
"""

from std.memory import bitcast
from std.os import getenv
from std.sys import argv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from checks.numerics import ftz, identical_mul_add, identical_tanh
from mamba.checks.mamba3_fixture import (
    BITS_POS_INF,
    BITS_QNAN,
    M3_A_FLOOR,
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    M3_PI,
    M3_TWO_PI,
    M3_TID_X,
    Mamba3CorpusCase,
    Mamba3Dims,
    Mamba3Weights,
    bits32_hex,
    corpus_tensor,
    f32_from_bits,
    m3_case_init_h,
    m3_case_init_k,
    m3_case_init_theta,
    m3_case_init_v,
    m3_case_seed,
    m3_case_weights,
    m3_case_x,
    m3_corpus_case,
    m3_corpus_dir,
    mode_name,
)
from mamba.checks.mamba3_oracle import (
    Mamba3Stages,
    Mamba3State,
    m3_mod_2pi,
    mamba3_block_oracle,
    pinned_mul,
)
from mamba.impl.mamba_ssm.modules.mamba3 import (
    BLOCK3_ANY_SABOTAGE,
    Mamba3DeviceStages,
    Mamba3DeviceState,
    Mamba3DeviceWeights,
    allocate_inference_cache,
    mamba3_block_forward,
    mamba3_sabotage_name,
)
from mamba.impl.mamba_ssm.ops.mamba3_siso import m3_n_chunks, m3_q_eff
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    mamba_download,
    mamba_upload,
)

comptime TRACE_PATH = "/tmp/mojolearn_mamba3_block_tiny.trace"


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
                String("mamba3_check: ") + name + " is not a number: '" + s + "'"
            )
        v = v * 10 + (c - 48)
    return v


# ===========================================================================
# THE STAGE TABLE: card order (contract section 7 minus input.x), with
# `pass.states` inserted as a COMPARE-ONLY entry after seg.L (it is not a
# card tag), and the carried state (state.h, state.theta) as the last two
# compare entries -- the boundary a resumption bug lives in.
# ===========================================================================

comptime KIND_TOKEN = 0
comptime KIND_BATCH = 2
comptime KIND_CHUNK = 3

comptime N_STAGES = 30
comptime STAGE_A = 3
comptime STAGE_ADT = 5
comptime STAGE_SIGMA = 6
comptime STAGE_SCALE = 7
comptime STAGE_THETA = 10
comptime STAGE_KSCALE = 14
comptime STAGE_DACS = 15
comptime STAGE_SEG = 16
comptime STAGE_PASS = 17


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
        return String("dt.out")
    if i == 5:
        return String("adt.out")
    if i == 6:
        return String("trap.sigma")
    if i == 7:
        return String("trap.scale")
    if i == 8:
        return String("bcnorm.B")
    if i == 9:
        return String("bcnorm.C")
    if i == 10:
        return String("angle.theta")
    if i == 11:
        return String("rot.q")
    if i == 12:
        return String("rot.k")
    if i == 13:
        return String("qkdot.out")
    if i == 14:
        return String("kscale.out")
    if i == 15:
        return String("dacs.out")
    if i == 16:
        return String("seg.L")
    if i == 17:
        return String("pass.states")
    if i == 18:
        return String("yintra.out")
    if i == 19:
        return String("ystate.out")
    if i == 20:
        return String("skip.out")
    if i == 21:
        return String("gate.out")
    if i == 22:
        return String("out_proj.out")
    if i == 23:
        return String("residual.out")
    if i == 24:
        return String("ssd.h_last")
    if i == 25:
        return String("ssd.k_last")
    if i == 26:
        return String("ssd.v_last")
    if i == 27:
        return String("ssd.theta_last")
    if i == 28:
        return String("state.h")
    if i == 29:
        return String("state.theta")
    raise Error("mamba3_check: no stage " + String(i))


def stage_kind(i: Int) -> Int:
    if i == 15 or i == 16 or i == 17:
        return KIND_CHUNK
    if i >= 24:
        return KIND_BATCH
    return KIND_TOKEN


def stage_final_only(i: Int) -> Bool:
    """DEVIATION 832(iii): trap.scale and kscale.out are NOT
    prefix-stable; gate (d) compares them at the final decoded token
    only."""
    return i == STAGE_SCALE or i == STAGE_KSCALE


def stage_token_width(i: Int, dims: Mamba3Dims) raises -> Int:
    var dm = dims.d_model
    var nh = dims.nheads
    if i == 0:
        return 1
    if i == 1 or i == 22 or i == 23:
        return dm
    if i == 2:
        return dims.d_in_proj()
    if i == 3 or i == 4 or i == 5 or i == 6 or i == 7 or i == 13:
        return nh
    if i == 8 or i == 9:
        return M3_D_STATE
    if i == 10:
        return nh * M3_NUM_ROPE_ANGLES
    if i == 11 or i == 12 or i == 14:
        return nh * M3_D_STATE
    if i == 18 or i == 19 or i == 20 or i == 21:
        return nh * M3_HEADDIM
    raise Error("stage_token_width: stage " + String(i) + " is not TOKEN")


def stage_per_batch(i: Int, dims: Mamba3Dims, nc: Int) raises -> Int:
    var nh = dims.nheads
    var qv = m3_q_eff()
    if i == 15:
        return nh * nc * qv
    if i == 16:
        return nc * nh * qv * qv
    if i == 17:
        return nc * nh * M3_HEADDIM * M3_D_STATE
    if i == 24 or i == 28:
        return nh * M3_HEADDIM * M3_D_STATE
    if i == 25:
        return nh * M3_D_STATE
    if i == 26:
        return nh * M3_HEADDIM
    if i == 27 or i == 29:
        return nh * M3_NUM_ROPE_ANGLES
    raise Error("stage_per_batch: stage " + String(i) + " is TOKEN")


# ===========================================================================
# Dumps
# ===========================================================================


def oracle_dump(st: Mamba3Stages, state: Mamba3State) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    out.append(st.norm_sumsq.copy())
    out.append(st.norm_out.copy())
    out.append(st.in_proj.copy())
    out.append(st.a_out.copy())
    out.append(st.dt_out.copy())
    out.append(st.adt_out.copy())
    out.append(st.trap_sigma.copy())
    out.append(st.trap_scale.copy())
    out.append(st.bcnorm_b.copy())
    out.append(st.bcnorm_c.copy())
    out.append(st.angle_theta.copy())
    out.append(st.rot_q.copy())
    out.append(st.rot_k.copy())
    out.append(st.qkdot_out.copy())
    out.append(st.kscale_out.copy())
    out.append(st.dacs_out.copy())
    out.append(st.seg_l.copy())
    out.append(st.pass_states.copy())
    out.append(st.yintra_out.copy())
    out.append(st.ystate_out.copy())
    out.append(st.skip_out.copy())
    out.append(st.gate_out.copy())
    out.append(st.out_proj.copy())
    out.append(st.residual_out.copy())
    out.append(st.h_last.copy())
    out.append(st.k_last.copy())
    out.append(st.v_last.copy())
    out.append(st.theta_last.copy())
    out.append(state.h.copy())
    out.append(state.theta.copy())
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
    (DEVIATION 832(ii)'s card slice, done on the host)."""
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
    mut d: Mamba3DeviceStages,
    mut state: Mamba3DeviceState,
    b: Int,
    l: Int,
    q0: Int,
    dims: Mamba3Dims,
) raises -> List[List[Float32]]:
    var m = b * l
    var dm = dims.d_model
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var nc = d.nc
    var qv = m3_q_eff()
    comptime p_dim = M3_HEADDIM
    comptime n_state = M3_D_STATE
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var out = List[List[Float32]]()
    out.append(mamba_download(ctx, d.norm_sumsq, m))
    out.append(mamba_download(ctx, d.norm_out, m * dm))
    out.append(mamba_download(ctx, d.in_proj, m * dip))
    out.append(mamba_download(ctx, d.a_out, m * nh))
    out.append(mamba_download(ctx, d.dt_out, m * nh))
    out.append(work_slice(ctx, d.adt_work, b, l, q0, nh))
    out.append(work_slice(ctx, d.sig_work, b, l, q0, nh))
    out.append(work_slice(ctx, d.scale_work, b, l, q0, nh))
    out.append(mamba_download(ctx, d.bcnorm_b, m * n_state))
    out.append(mamba_download(ctx, d.bcnorm_c, m * n_state))
    out.append(mamba_download(ctx, d.theta_out, m * nh * r_ang))
    out.append(work_slice(ctx, d.rotq_work, b, l, q0, nh * n_state))
    out.append(work_slice(ctx, d.rotk_work, b, l, q0, nh * n_state))
    out.append(mamba_download(ctx, d.qkdot, m * nh))
    out.append(work_slice(ctx, d.kscale_work, b, l, q0, nh * n_state))
    out.append(mamba_download(ctx, d.dacs, b * nh * nc * qv))
    out.append(mamba_download(ctx, d.seg_l, b * nc * nh * qv * qv))
    out.append(
        mamba_download(ctx, d.pass_states, b * nc * nh * p_dim * n_state)
    )
    out.append(mamba_download(ctx, d.yintra, m * nh * p_dim))
    out.append(mamba_download(ctx, d.ystate, m * nh * p_dim))
    out.append(mamba_download(ctx, d.skip_out, m * nh * p_dim))
    out.append(mamba_download(ctx, d.gate_out, m * nh * p_dim))
    out.append(mamba_download(ctx, d.out_proj, m * dm))
    out.append(mamba_download(ctx, d.residual_out, m * dm))
    out.append(mamba_download(ctx, d.h_last, b * nh * p_dim * n_state))
    out.append(mamba_download(ctx, d.k_last, b * nh * n_state))
    out.append(mamba_download(ctx, d.v_last, b * nh * p_dim))
    out.append(mamba_download(ctx, d.theta_last, b * nh * r_ang))
    out.append(mamba_download(ctx, state.h, b * nh * p_dim * n_state))
    out.append(mamba_download(ctx, state.theta, b * nh * r_ang))
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
            String("mamba3_check: stage ")
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
# the case's four-piece Input_States on BOTH sides).
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
    shape (b, l): x is the case's hashed x when the shape matches, else
    the case-seed x regenerated at this shape (same element rule, so
    nothing is hand-picked)."""
    var c = m3_corpus_case(case_k)
    var w = m3_case_weights(case_k)
    var dims = w.dims.copy()
    var x: List[Float32]
    if b == c.b and l == c.l:
        x = m3_case_x(case_k)
    else:
        x = corpus_tensor(
            m3_case_seed(c.seed_index), M3_TID_X, b * l * dims.d_model,
            -2.0, 2.0,
        )

    # ---- device
    var dw = Mamba3DeviceWeights(ctx, w)
    var dstate = allocate_inference_cache(ctx, b, dims)
    if c.has_init_states:
        dstate.set_input_states(
            ctx,
            m3_case_init_theta(case_k),
            m3_case_init_h(case_k),
            m3_case_init_k(case_k),
            m3_case_init_v(case_k),
        )
    var dstages = Mamba3DeviceStages(ctx, b, l, 0, dims)
    var dx = mamba_upload(ctx, x)
    mamba3_block_forward(ctx, dstages, dstate, dw, dx, b, l, trace, prefix)
    var ddump = device_dump(ctx, dstages, dstate, b, l, 0, dims)

    # ---- oracle
    var ostate = Mamba3State(b, dims)
    if c.has_init_states:
        ostate.set_input_states(
            m3_case_init_theta(case_k),
            m3_case_init_h(case_k),
            m3_case_init_k(case_k),
            m3_case_init_v(case_k),
        )
    var ost = mamba3_block_oracle(w, x, b, l, ostate)
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
# GATE (a): card == oracle, bitwise, every stage; and the card's 28 tags
# in section 7's order, once each.
# ===========================================================================


def contract_section_7_tags() -> List[String]:
    var want: List[String] = [
        String("input.x"),
        String("norm.sumsq"),
        String("norm.out"),
        String("in_proj.out"),
        String("A.out"),
        String("dt.out"),
        String("adt.out"),
        String("trap.sigma"),
        String("trap.scale"),
        String("bcnorm.B"),
        String("bcnorm.C"),
        String("angle.theta"),
        String("rot.q"),
        String("rot.k"),
        String("qkdot.out"),
        String("kscale.out"),
        String("dacs.out"),
        String("seg.L"),
        String("yintra.out"),
        String("ystate.out"),
        String("skip.out"),
        String("gate.out"),
        String("out_proj.out"),
        String("residual.out"),
        String("ssd.h_last"),
        String("ssd.k_last"),
        String("ssd.v_last"),
        String("ssd.theta_last"),
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
            String("mamba3_check: the card has ")
            + String(len(lines))
            + " records and contract section 7 lists "
            + String(len(want))
        )
    for i in range(len(lines)):
        var fields = lines[i].split("\t")
        if len(fields) < 2:
            raise Error(
                String("mamba3_check: malformed trace record: ") + lines[i]
            )
        var got = String(fields[1])
        var expect = prefix + "." + want[i]
        if got != expect:
            raise Error(
                String("mamba3_check: card record ")
                + String(i)
                + " is '"
                + got
                + "', contract section 7 wants '"
                + expect
                + "'"
            )
    print("card: all 28 tags present, once each, in section 7's order")


def gate_a(
    ctx: DeviceContext, case_k: Int, b: Int, l: Int, loud: Bool
) raises -> List[StageDiff]:
    var trace = IdentityTrace.to_path(card_path())
    var pair = run_pair(ctx, case_k, b, l, trace, String("m3"))
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
            ctx, case_k, b, l, trace, String("m3rep") + String(launch)
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
    dims: Mamba3Dims,
    nc: Int,
) raises -> List[Float32]:
    var kind = stage_kind(i)
    var out = List[Float32]()
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
    var c = m3_corpus_case(case_k)
    var w = m3_case_weights(case_k)
    var dims = w.dims.copy()
    var seed = m3_case_seed(c.seed_index)
    var x3 = corpus_tensor(seed, M3_TID_X, 3 * l * dims.d_model, -2.0, 2.0)
    var nc = m3_n_chunks(l)

    var dumps = List[List[List[Float32]]]()
    for bsz in range(1, 4):
        var x = List[Float32]()
        for i in range(bsz * l * dims.d_model):
            x.append(x3[i])
        var dw = Mamba3DeviceWeights(ctx, w)
        var dstate = allocate_inference_cache(ctx, bsz, dims)
        var dstages = Mamba3DeviceStages(ctx, bsz, l, 0, dims)
        var dx = mamba_upload(ctx, x)
        var trace = IdentityTrace.disabled()
        mamba3_block_forward(
            ctx, dstages, dstate, dw, dx, bsz, l, trace,
            String("m3c") + String(bsz),
        )
        dumps.append(device_dump(ctx, dstages, dstate, bsz, l, 0, dims))
        _ = dw^
        _ = dstate^
        _ = dstages^
        _ = dx^

    # NEGATIVE CONTROL first: rows 0 and 1 of the B = 2 run must differ.
    var control_diff = 0
    for i in range(N_STAGES):
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
        var a = row_slice_of(dumps[0], i, 0, l, dims, nc)
        for bi in range(1, 3):
            var bslice = row_slice_of(dumps[bi], i, 0, l, dims, nc)
            fails += compare_stage(stage_name(i), a, bslice, False).n_diff
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
# GATE (d): decode == prefill, per token, under DEVIATION 832(iii)'s
# comparability; the misalignment control. `l1 > 0` runs a prefill of l1
# FIRST and decodes tokens l1..l_total-1 (the chunk-crossing arm).
# ===========================================================================


def gate_d(
    ctx: DeviceContext, case_k: Int, l1: Int, l_total: Int
) raises -> Int:
    var c = m3_corpus_case(case_k)
    var w = m3_case_weights(case_k)
    var dims = w.dims.copy()
    var b = 1  # decode chains are per-sequence; composition is gate (c)'s
    var seed = m3_case_seed(c.seed_index)
    var x = corpus_tensor(seed, M3_TID_X, b * l_total * dims.d_model, -2.0, 2.0)
    var qv = m3_q_eff()
    var nh = dims.nheads
    comptime p_dim = M3_HEADDIM
    comptime n_state = M3_D_STATE
    comptime r_ang = M3_NUM_ROPE_ANGLES

    # ---- the reference prefill: one call, L tokens, fresh state.
    var dw = Mamba3DeviceWeights(ctx, w)
    var pre_state = allocate_inference_cache(ctx, b, dims)
    var pre_stages = Mamba3DeviceStages(ctx, b, l_total, 0, dims)
    var dx = mamba_upload(ctx, x)
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, pre_stages, pre_state, dw, dx, b, l_total, trace,
        String("m3d.pre"),
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
        var st1 = Mamba3DeviceStages(ctx, b, l1, 0, dims)
        var dx1 = mamba_upload(ctx, x1)
        mamba3_block_forward(
            ctx, st1, state, dw, dx1, b, l1, trace, String("m3d.p1")
        )
        _ = st1^
        _ = dx1^

    var fails = 0
    var misalign_diff = 0
    for li in range(l1, l_total):
        var xt = List[Float32]()
        for j in range(dims.d_model):
            xt.append(x[li * dims.d_model + j])
        var q0 = state.buf_len
        var dstages = Mamba3DeviceStages(ctx, b, 1, q0, dims)
        var dxt = mamba_upload(ctx, xt)
        mamba3_block_forward(
            ctx, dstages, state, dw, dxt, b, 1, trace,
            String("m3d.t") + String(li),
        )
        var dec = device_dump(ctx, dstages, state, b, 1, q0, dims)
        var nc_dec = dstages.nc

        # token-shaped, prefix-stable stages: this token's row vs the
        # prefill's row li (trap.scale / kscale.out excluded until the
        # final token -- DEVIATION 832(iii)).
        for i in range(N_STAGES):
            if stage_kind(i) != KIND_TOKEN:
                continue
            if stage_final_only(i) and li != l_total - 1:
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
            # against prefill row li + 1, which MUST differ.
            if li == l1 and li + 1 < l_total and not stage_final_only(i):
                var prow2 = List[Float32]()
                for j in range(wdt):
                    prow2.append(pre[i][(li + 1) * wdt + j])
                misalign_diff += compare_stage(
                    stage_name(i), dec[i], prow2, False
                ).n_diff

        # at the FINAL token: the chunk-shaped stages against the
        # prefill's chunks s.. (same fill, the sealed construction), the
        # reports and the carried state.
        if li == l_total - 1:
            var s = 0
            if l_total >= 2:
                s = (l_total - 2) // qv
            # dacs.out [B, H, C, Q]
            var wantd = List[Float32]()
            for hh in range(nh):
                for cc in range(nc_dec):
                    for j in range(qv):
                        wantd.append(
                            pre[STAGE_DACS][(hh * nc_pre + (s + cc)) * qv + j]
                        )
            fails += compare_stage(
                String("dacs.out [final]"), dec[STAGE_DACS], wantd, False
            ).n_diff
            # seg.L [B, C, H, Q, Q]
            var wantl = List[Float32]()
            for cc in range(nc_dec):
                for j in range(nh * qv * qv):
                    wantl.append(
                        pre[STAGE_SEG][(s + cc) * nh * qv * qv + j]
                    )
            fails += compare_stage(
                String("seg.L [final]"), dec[STAGE_SEG], wantl, False
            ).n_diff
            # pass.states [B, C, H, P, N]
            var wantp = List[Float32]()
            for cc in range(nc_dec):
                for j in range(nh * p_dim * n_state):
                    wantp.append(
                        pre[STAGE_PASS][(s + cc) * nh * p_dim * n_state + j]
                    )
            fails += compare_stage(
                String("pass.states [final]"), dec[STAGE_PASS], wantp, False
            ).n_diff
            # the reports and the carried state, whole.
            for i in range(24, N_STAGES):
                fails += compare_stage(
                    stage_name(i), dec[i], pre[i], False
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
# GATE (e): the planted nonfinite audit. Reach measured (the plant read
# back off the device), refusal BY NAME, ZERO stages recorded; plus the
# clean-call control. 18 names (contract section 6's inventory for this
# block: x, the nine weights, the eight state pieces).
# ===========================================================================

comptime N_REFUSAL_NAMES = 18


def refusal_names() -> List[String]:
    var names: List[String] = [
        String("x"),
        String("norm.weight"),
        String("in_proj.weight"),
        String("dt_bias"),
        String("B_norm.weight"),
        String("C_norm.weight"),
        String("B_bias"),
        String("C_bias"),
        String("D"),
        String("out_proj.weight"),
        String("state.theta"),
        String("state.h"),
        String("state.buf_qrot"),
        String("state.buf_krot"),
        String("state.buf_v"),
        String("state.buf_dt"),
        String("state.buf_sig"),
        String("state.buf_adt"),
    ]
    return names^


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
    order), run, and require refusal naming that input with 0 records."""
    var c = m3_corpus_case(case_k)
    var w = m3_case_weights(case_k)
    var dims = w.dims.copy()
    var nh = dims.nheads
    var x = corpus_tensor(
        m3_case_seed(c.seed_index), M3_TID_X, b * l * dims.d_model, -2.0, 2.0
    )
    var names = refusal_names()
    if name_idx == 0:
        plant_in_list(x, bits)
    if name_idx == 1:
        plant_in_list(w.norm_w, bits)
    if name_idx == 2:
        plant_in_list(w.w_in, bits)
    if name_idx == 3:
        plant_in_list(w.dt_bias, bits)
    if name_idx == 4:
        plant_in_list(w.bnorm_w, bits)
    if name_idx == 5:
        plant_in_list(w.cnorm_w, bits)
    if name_idx == 6:
        plant_in_list(w.b_bias, bits)
    if name_idx == 7:
        plant_in_list(w.c_bias, bits)
    if name_idx == 8:
        plant_in_list(w.d_skip, bits)
    if name_idx == 9:
        plant_in_list(w.w_out, bits)

    var dw = Mamba3DeviceWeights(ctx, w)
    var dstate = allocate_inference_cache(ctx, b, dims)
    var dstages = Mamba3DeviceStages(ctx, b, l, 0, dims)
    var dx = mamba_upload(ctx, x)

    # State plants: download the zero buffer's size, plant at len/2, copy
    # back.
    if name_idx >= 10:
        var nsz: Int
        if name_idx == 10:
            nsz = b * nh * M3_NUM_ROPE_ANGLES
        elif name_idx == 11:
            nsz = b * nh * M3_HEADDIM * M3_D_STATE
        elif name_idx == 12 or name_idx == 13:
            nsz = b * M3_CHUNK_SIZE * nh * M3_D_STATE
        elif name_idx == 14:
            nsz = b * M3_CHUNK_SIZE * nh * M3_HEADDIM
        else:
            nsz = b * M3_CHUNK_SIZE * nh
        var host = List[Float32]()
        for _ in range(nsz):
            host.append(0.0)
        plant_in_list(host, bits)
        var planted = mamba_upload(ctx, host)
        if name_idx == 10:
            ctx.enqueue_copy(dst_buf=dstate.theta, src_buf=planted)
        elif name_idx == 11:
            ctx.enqueue_copy(dst_buf=dstate.h, src_buf=planted)
        elif name_idx == 12:
            ctx.enqueue_copy(dst_buf=dstate.buf_qrot, src_buf=planted)
        elif name_idx == 13:
            ctx.enqueue_copy(dst_buf=dstate.buf_krot, src_buf=planted)
        elif name_idx == 14:
            ctx.enqueue_copy(dst_buf=dstate.buf_v, src_buf=planted)
        elif name_idx == 15:
            ctx.enqueue_copy(dst_buf=dstate.buf_dt, src_buf=planted)
        elif name_idx == 16:
            ctx.enqueue_copy(dst_buf=dstate.buf_sig, src_buf=planted)
        else:
            ctx.enqueue_copy(dst_buf=dstate.buf_adt, src_buf=planted)
        ctx.synchronize()
        _ = planted^

    # REACH: the planted buffer read back off the device, nonfinite
    # cells counted. A refusal that fired for another reason must not
    # pass.
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
        readback = mamba_download(ctx, dw.dt_bias, nh)
    elif name_idx == 4:
        readback = mamba_download(ctx, dw.bnorm_w, M3_D_STATE)
    elif name_idx == 5:
        readback = mamba_download(ctx, dw.cnorm_w, M3_D_STATE)
    elif name_idx == 6:
        readback = mamba_download(ctx, dw.b_bias, nh * M3_D_STATE)
    elif name_idx == 7:
        readback = mamba_download(ctx, dw.c_bias, nh * M3_D_STATE)
    elif name_idx == 8:
        readback = mamba_download(ctx, dw.d_skip, nh)
    elif name_idx == 9:
        readback = mamba_download(ctx, dw.w_out, dims.d_model * dims.d_inner)
    elif name_idx == 10:
        readback = mamba_download(
            ctx, dstate.theta, b * nh * M3_NUM_ROPE_ANGLES
        )
    elif name_idx == 11:
        readback = mamba_download(
            ctx, dstate.h, b * nh * M3_HEADDIM * M3_D_STATE
        )
    elif name_idx == 12:
        readback = mamba_download(
            ctx, dstate.buf_qrot, b * M3_CHUNK_SIZE * nh * M3_D_STATE
        )
    elif name_idx == 13:
        readback = mamba_download(
            ctx, dstate.buf_krot, b * M3_CHUNK_SIZE * nh * M3_D_STATE
        )
    elif name_idx == 14:
        readback = mamba_download(
            ctx, dstate.buf_v, b * M3_CHUNK_SIZE * nh * M3_HEADDIM
        )
    elif name_idx == 15:
        readback = mamba_download(ctx, dstate.buf_dt, b * M3_CHUNK_SIZE * nh)
    elif name_idx == 16:
        readback = mamba_download(ctx, dstate.buf_sig, b * M3_CHUNK_SIZE * nh)
    else:
        readback = mamba_download(ctx, dstate.buf_adt, b * M3_CHUNK_SIZE * nh)
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
        mamba3_block_forward(
            ctx, dstages, dstate, dw, dx, b, l, trace,
            String("m3e") + String(name_idx),
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
    var trace = IdentityTrace.to_path(card_path() + ".clean", String(""), True)
    var pair = run_pair(ctx, case_k, b, l, trace, String("m3e.clean"))
    if trace.seq != 28:
        raise Error(
            String("GATE E VACUOUS: the clean control recorded ")
            + String(trace.seq)
            + " stages, not 28 -- the refusal walk (or the recorder) is"
            + " broken and every plant below would pass for the wrong"
            + " reason"
        )
    _ = pair^
    var n = 0
    for idx in range(N_REFUSAL_NAMES):
        var nm = gate_e_one_plant(ctx, case_k, b, l, idx, BITS_QNAN)
        var nm2 = gate_e_one_plant(ctx, case_k, b, l, idx, BITS_POS_INF)
        if nm != nm2:
            raise Error("GATE E: plant naming disagreed between NaN and inf")
        n += 2
    print(
        "GATE E PASS:",
        n,
        "plants (quiet NaN + inf per each of the 18 names, cell len/2),",
        "each read back off the device first, each refused BY NAME with 0"
        " stages recorded; clean control recorded all 28",
    )


# ===========================================================================
# GATE (g)'s byte gate: MISSING-FIXTURE REFUSAL until the corpus lane
# generates mamba/corpus/mamba3/. Binary-safe reads (the mamba2 lesson:
# text-mode read raised on raw float32 bytes WITH THE FILE PRESENT and a
# catch-all renamed it MISSING FIXTURE; each failure kind refuses under
# its OWN name here).
# ===========================================================================


def _read_fixture_bytes(path: String, case_name: String) raises -> List[UInt8]:
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
                + "'). Either the corpus generator has not emitted the"
                + " mamba3 family yet (generation is RUN OWED on the"
                + " corpus side), or the working directory is not the"
                + " repo root. This arm REFUSES rather than skips"
                + " (contract 8g)."
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
    var c = m3_corpus_case(case_k)
    var nm = String(c.name)
    var dir_path = m3_corpus_dir(case_k)
    var w = m3_case_weights(case_k)
    _compare_fixture_file(dir_path, String("x.f32"), m3_case_x(case_k), nm)
    _compare_fixture_file(
        dir_path, String("block_norm.weight.f32"), w.norm_w, nm
    )
    _compare_fixture_file(dir_path, String("in_proj.weight.f32"), w.w_in, nm)
    _compare_fixture_file(dir_path, String("dt_bias.f32"), w.dt_bias, nm)
    _compare_fixture_file(
        dir_path, String("B_norm.weight.f32"), w.bnorm_w, nm
    )
    _compare_fixture_file(
        dir_path, String("C_norm.weight.f32"), w.cnorm_w, nm
    )
    _compare_fixture_file(dir_path, String("B_bias.f32"), w.b_bias, nm)
    _compare_fixture_file(dir_path, String("C_bias.f32"), w.c_bias, nm)
    _compare_fixture_file(dir_path, String("D.f32"), w.d_skip, nm)
    _compare_fixture_file(dir_path, String("out_proj.weight.f32"), w.w_out, nm)
    if c.has_init_states:
        _compare_fixture_file(
            dir_path, String("init_theta.f32"), m3_case_init_theta(case_k), nm
        )
        _compare_fixture_file(
            dir_path, String("init_h.f32"), m3_case_init_h(case_k), nm
        )
        _compare_fixture_file(
            dir_path, String("init_k.f32"), m3_case_init_k(case_k), nm
        )
        _compare_fixture_file(
            dir_path, String("init_v.f32"), m3_case_init_v(case_k), nm
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


def oracle_card_of(case_k: Int) raises -> Mamba3Stages:
    """The oracle's clean card for a case at its own shape (host-side
    witness counting)."""
    var c = m3_corpus_case(case_k)
    var w = m3_case_weights(case_k)
    var state = Mamba3State(c.b, w.dims)
    if c.has_init_states:
        state.set_input_states(
            m3_case_init_theta(case_k),
            m3_case_init_h(case_k),
            m3_case_init_k(case_k),
            m3_case_init_v(case_k),
        )
    var x = m3_case_x(case_k)
    return mamba3_block_oracle(w, x, c.b, c.l, state)


def count_nonuniform(vals: List[Float32]) -> Int:
    var n = 0
    for i in range(1, len(vals)):
        if bitcast[DType.uint32](vals[i]) != bitcast[DType.uint32](vals[0]):
            n += 1
    return n


def count_nonzero(vals: List[Float32]) -> Int:
    var n = 0
    for i in range(len(vals)):
        var u = bitcast[DType.uint32](vals[i]) & UInt32(0x7FFFFFFF)
        if u != UInt32(0):
            n += 1
    return n


def count_angle_crossings(case_k: Int) raises -> Int:
    """Re-run the S10 recurrence on the host from the oracle's own card
    (in_proj angle columns, dt.out) and count mod engagements at tokens
    that are neither the sequence's last nor a chunk boundary -- the
    property that makes BOTH angle arms falsifiable (a fixture that never
    crosses is vacuous, contract 8f by name)."""
    var c = m3_corpus_case(case_k)
    var w = m3_case_weights(case_k)
    var dims = w.dims.copy()
    var st = oracle_card_of(case_k)
    var nh = dims.nheads
    var dip = dims.d_in_proj()
    var c_ang = dims.col_angle()
    var qv = m3_q_eff()
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var crossings = 0
    for bb in range(c.b):
        for hh in range(nh):
            for r in range(r_ang):
                var run = Float32(0.0)
                for li in range(c.l):
                    var mm = bb * c.l + li
                    var a = ftz(
                        pinned_mul(
                            identical_tanh(
                                ftz(st.in_proj[mm * dip + c_ang + r])
                            ),
                            M3_PI,
                        )
                    )
                    var inc = ftz(
                        pinned_mul(a, ftz(st.dt_out[mm * nh + hh]))
                    )
                    var x = ftz(run + inc)
                    var modded = m3_mod_2pi(x)
                    if bitcast[DType.uint32](modded) != bitcast[
                        DType.uint32
                    ](x):
                        # the mod ENGAGED here
                        if li + 1 < c.l and ((li + 1) % qv) != 0:
                            crossings += 1
                    run = modded
    return crossings


def sabotage_main(ctx: DeviceContext) raises:
    var arm = mamba3_sabotage_name()
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

    if arm == "FOLD_SERIAL_ZERO_SEED":
        raise Error(
            "SABOTAGE VACUOUS BY CONSTRUCTION (DEVIATION 834):"
            " FOLD_SERIAL_ZERO_SEED has no witnessable in-core site at"
            " Q = 64 -- every mamba3 core contraction is ONE gemm-v1"
            " serial leaf (k <= 128), so the serial-chain respelling is"
            " bit-identical to the profile everywhere it could engage."
            " The fold clause's real falsifiers are the gemm lane's own"
            " certified arms at k > 128. This armed build refuses rather"
            " than passing or skipping."
        )

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
                " upstream step recurrence agreed with the chunked"
                " prefill bitwise, which DEVIATION 831 says is impossible"
                " -- the gate is blind or the arm is not reached."
            )
        print(
            "SABOTAGE ARM FAILS AS REQUIRED:",
            fails,
            "cells moved on gate (d). DEVIATION 831's clause is load"
            " bearing.",
        )
        return

    # every other arm gates clause (a) on its witnessing fixture.
    var case_k = 1  # m3_base_b2_l4_d32: hashed values, any-shape arms
    var expect_first = String("")
    var token_only = False
    if arm == "SEGSUM_DESCENDING":
        expect_first = String("dacs.out")
        var st = oracle_card_of(case_k)
        var nonuni = count_nonuniform(st.adt_out)
        if nonuni == 0:
            raise Error(
                "SABOTAGE VACUOUS: every ADT cell is identical; a"
                " reversed fold over a uniform ADT is bitwise inert"
            )
        print("  witness: nonuniform ADT cells =", nonuni)
    elif arm == "CHUNK_SIZE_32":
        case_k = 5  # m3_base_b1_l65_d64
        expect_first = String("yintra.out")
        token_only = True
        var cc = m3_corpus_case(case_k)
        if cc.l <= 32:
            raise Error("SABOTAGE VACUOUS: CHUNK_SIZE_32 needs L > 32")
        print(
            "  witness: L =",
            cc.l,
            "> 32; chunk-shaped stage buffers change SHAPE under this"
            " arm, so the compare covers the non-chunk stages (stated,"
            " not silent)",
        )
    elif arm == "TRAP_LEFT_ONLY":
        expect_first = String("kscale.out")
        # Witness: cells where scale != gamma, gamma recomputed through
        # the SAME seam functions from the clean dt.out / trap.sigma.
        var st = oracle_card_of(case_k)
        var moved = 0
        for i in range(len(st.trap_scale)):
            var g = ftz(
                pinned_mul(ftz(st.dt_out[i]), ftz(st.trap_sigma[i]))
            )
            if bitcast[DType.uint32](g) != bitcast[DType.uint32](
                st.trap_scale[i]
            ):
                moved += 1
        if moved == 0:
            raise Error(
                "SABOTAGE VACUOUS: scale == gamma on every recorded cell"
                " (no beta' leg is live); the Euler respelling is inert"
                " on this fixture"
            )
        print("  witness: cells with a live beta' leg =", moved)
    elif arm == "ANGLE_MOD_PER_CHUNK" or arm == "ANGLE_MOD_AT_END":
        case_k = 13  # m3_adv_angle_crossing_b1_l48_d32
        expect_first = String("angle.theta")
        var crossings = count_angle_crossings(case_k)
        if crossings == 0:
            raise Error(
                "SABOTAGE VACUOUS: the angle recurrence never crossed the"
                " 2pi seam at a non-final, non-boundary token; a fixture"
                " that never crosses cannot witness the mod placement"
                " (contract 8f names this exact vacuity)"
            )
        print("  witness: in-chunk 2pi-seam mod engagements =", crossings)
    elif arm == "ROTATE_HALF_SPLIT":
        expect_first = String("rot.q")
        var st = oracle_card_of(case_k)
        var nz = count_nonzero(st.angle_theta)
        if nz == 0:
            raise Error(
                "SABOTAGE VACUOUS: every angle is zero; both pairings"
                " rotate by identity and agree bitwise"
            )
        print("  witness: nonzero angle.theta cells =", nz)
    elif arm == "DIAG_INCLUDE_SUBTRACT":
        expect_first = String("skip.out")
        var st = oracle_card_of(case_k)
        var nz = count_nonzero(st.angle_theta)
        if nz == 0:
            raise Error(
                "SABOTAGE VACUOUS: every angle is zero; with no rotation"
                " rounding the two diagonal spellings can coincide -- the"
                " arm needs a nonzero-angle fixture (contract 8f)"
            )
        print("  witness: nonzero angle.theta cells =", nz)
    elif arm == "STATE_TERM_SCALE_FIRST":
        case_k = 9  # m3_state_b1_l129_d32, three working chunks
        expect_first = String("ystate.out")
        var cc = m3_corpus_case(case_k)
        var nc = m3_n_chunks(cc.l)
        if nc < 2:
            raise Error(
                "SABOTAGE VACUOUS: STATE_TERM_SCALE_FIRST needs L > Q"
                " (a chunk with nonzero incoming state)"
            )
        var st = oracle_card_of(case_k)
        var nh = m3_case_weights(case_k).dims.nheads
        comptime pn = M3_HEADDIM * M3_D_STATE
        var nz = 0
        for c2 in range(1, nc):
            for j in range(nh * pn):
                var u = bitcast[DType.uint32](
                    st.pass_states[c2 * nh * pn + j]
                ) & UInt32(0x7FFFFFFF)
                if u != UInt32(0):
                    nz += 1
        if nz == 0:
            raise Error(
                "SABOTAGE VACUOUS: the state entering every chunk past"
                " the first is exactly zero; scale-first vs"
                " contract-then-scale is bitwise inert on zeros"
            )
        print("  witness: nonzero incoming-state cells (chunks 1+):", nz)
    elif arm == "RESUME_KERNEL_ASSOC":
        case_k = 16  # m3_init_states_b1_l65_d32
        expect_first = String("pass.states")
        var cc = m3_corpus_case(case_k)
        if not cc.has_init_states:
            raise Error(
                "SABOTAGE VACUOUS: the continuation case carries no"
                " Input_States"
            )
        var nzh = count_nonzero(m3_case_init_h(case_k))
        var nzk = count_nonzero(m3_case_init_k(case_k))
        var nzv = count_nonzero(m3_case_init_v(case_k))
        if nzh == 0 or nzk == 0 or nzv == 0:
            raise Error(
                "SABOTAGE VACUOUS: a zero Input_States piece makes both"
                " S22 associations produce identical bits"
            )
        print(
            "  witness: nonzero init h/k/v cells =", nzh, nzk, nzv
        )
    elif arm == "A_FLOOR_UNCLAMPED":
        case_k = 12  # m3_adv_a_floor_b2_l8_d32, planted dd_A rows
        expect_first = String("A.out")
        var st = oracle_card_of(case_k)
        var bound = 0
        var unbound = 0
        var floor_bits = bitcast[DType.uint32](-M3_A_FLOOR)
        for i in range(len(st.a_out)):
            if bitcast[DType.uint32](st.a_out[i]) == floor_bits:
                bound += 1
            else:
                unbound += 1
        if bound == 0:
            raise Error(
                "SABOTAGE VACUOUS: the A_floor clamp bound ZERO cells on"
                " the clean card -- the planted dd_A rows are not doing"
                " their job (the clamp binds only below dd_A < -9999,"
                " contract section 3) and dropping the clamp is inert"
            )
        if unbound == 0:
            raise Error(
                "SABOTAGE VACUOUS: EVERY cell clamped -- the fixture"
                " cannot show the clamp's pass-through side"
            )
        print(
            "  witness: clamp binds on", bound, "cells, passes", unbound,
            "(planted, by construction)",
        )
    else:
        raise Error(String("mamba3_check: unknown armed name ") + arm)

    var cc2 = m3_corpus_case(case_k)
    var diffs: List[StageDiff]
    if token_only:
        var trace = IdentityTrace.disabled()
        var pair = run_pair(ctx, case_k, cc2.b, cc2.l, trace, String("m3sab"))
        diffs = List[StageDiff]()
        for i in range(N_STAGES):
            if stage_kind(i) == KIND_CHUNK:
                continue
            diffs.append(
                compare_stage(stage_name(i), pair[0][i], pair[1][i], False)
            )
    else:
        diffs = gate_a(ctx, case_k, cc2.b, cc2.l, False)
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
    # The pi/2pi bit pins (contract section 2a): a literal drift would
    # silently move every rotation, so it is asserted before anything.
    if bitcast[DType.uint32](M3_PI) != UInt32(0x40490FDB):
        raise Error("mamba3_check: Float32(pi) is not 0x40490FDB")
    if bitcast[DType.uint32](M3_TWO_PI) != UInt32(0x40C90FDB):
        raise Error("mamba3_check: Float32(2pi) is not 0x40C90FDB")

    var mode = String("gates")
    for a in argv():
        if a == "decode":
            mode = String("decode")
        if a == "decode-cross":
            mode = String("decode-cross")
        if a == "continuation":
            mode = String("continuation")
        if a == "refusal":
            mode = String("refusal")
        if a == "corpus":
            mode = String("corpus")

    print(
        "mamba3 block gate (IDENTICAL_MAMBA3_CONTRACT.md section 8);"
        " profile mojolearn.identical.mamba3.siso.fp32.v1; build mode",
        mode_name(),
    )

    comptime if BLOCK3_ANY_SABOTAGE:
        var ctx_s = DeviceContext()
        sabotage_main(ctx_s)
        return

    var ctx = DeviceContext()

    if mode == "corpus":
        for k in range(18):
            gate_corpus(k)
        print(
            "GATE (g) BYTE GATE PASS: all 18 cases' input tensors are"
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
            "GATE D PASS: decode == prefill bitwise, per token"
            " (prefix-stable stages) + final-token trap.scale/kscale/"
            "chunk/state/report stages (misalignment control differed)"
        )
        return

    if mode == "decode-cross":
        # contract 8d: prefill 60, decode through token 70 -- crosses the
        # Q = 64 boundary, so the sealed buffer fills, seals and refills.
        # Case 17's weights (m3_decode_b1_l60p10_d32).
        var fails = gate_d(ctx, 17, 60, 70)
        if fails != 0:
            raise Error(
                String("GATE D (cross) FAILED: ")
                + String(fails)
                + " cells differ; the run crossed the chunk boundary at"
                + " token 64"
            )
        print(
            "GATE D (cross) PASS: prefill 60 + decode through 70 =="
            " prefill 70, bitwise; the chain crossed the Q = 64 boundary"
            " and the sealed buffer folded and refilled"
        )
        return

    if mode == "continuation":
        # The Input_States continuation, device vs oracle on the SAME
        # call (bitwise). NOT a prefill-equivalence claim -- DEVIATION
        # 831 proves that claim false; the corpus row checks it under
        # tolerance instead.
        var cc = m3_corpus_case(16)
        var nzh = count_nonzero(m3_case_init_h(16))
        if not cc.has_init_states or nzh == 0:
            raise Error(
                "CONTINUATION VACUOUS: the case carries no nonzero"
                " Input_States"
            )
        var diffs = gate_a(ctx, 16, cc.b, cc.l, True)
        var n = total_moved(diffs)
        if n != 0:
            raise Error(
                String("CONTINUATION FAILED: ")
                + String(n)
                + " cells differ between the device continuation call and"
                + " the oracle's (first stage: "
                + first_moved(diffs)
                + ")"
            )
        print(
            "CONTINUATION PASS: the four-piece Input_States call is"
            " bit-identical to the oracle, S22 correction included, on",
            nzh,
            "nonzero incoming h cells. (No bitwise claim against an"
            " unbroken prefill exists -- DEVIATION 831.)"
        )
        return

    # default: gates (a) + card, (b), (c) at the env shape (ONE shape per
    # build -- the 2026-08-23 crash rule).
    var b = env_int(String("MOJOLEARN_MAMBA3_CHECK_B"), 1)
    var l = env_int(String("MOJOLEARN_MAMBA3_CHECK_L"), 4)
    var dm = env_int(String("MOJOLEARN_MAMBA3_CHECK_DM"), 32)
    var case_k = 1  # m3_base_b2_l4_d32's weights; x regenerated at (B, L)
    if dm == 64:
        case_k = 2  # m3_base_b3_l4_d64's weights
    elif dm != 32:
        raise Error("mamba3_check: d_model must be 32 or 64 (fixture set)")
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
    check_card_tags(card_path(), String("m3"))
    print(
        "GATE A PASS: every stage bit-identical to the oracle at this"
        " shape."
    )
    gate_b(ctx, case_k, b, l)
    gate_c(ctx, case_k, l)
    print(
        "One shape, one launch history, one box. Everything else in"
        " contract section 8 is OWED: the sabotage arms (one build each),"
        " decode / decode-cross / continuation / refusal / corpus arms,"
        " the other shapes and L set, FAST recording, the portable-trig"
        " device certification (phase M3-0), and every column that is not"
        " this machine (phase M3-6)."
    )
