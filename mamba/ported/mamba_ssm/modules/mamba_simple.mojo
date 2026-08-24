"""`mamba_ssm/modules/mamba_simple.py::Mamba.step` (:208-253) and
`::allocate_inference_cache` (:255-266), state-spaces/mamba `e9594ce`.

PORTED. The DECODE half of profile `mojolearn.identical.mamba1.fp32.v1`
(`mamba/IDENTICAL_MAMBA_CONTRACT.md`, section 5). One token at a time,
carrying the two pieces of recurrent state their `step` carries: the conv
WINDOW (`conv_state`, the last `d_conv = 4` conv inputs, oldest first) and
the SSM state h (`ssm_state`), both zeros before the first token
(`allocate_inference_cache`).

## What this file is NOT

It is not a second copy of the block's arithmetic. Contract section 5 says
prefill and decode are bit-identical BY CONSTRUCTION and not by luck: ONE
spelling serves both paths, because two spellings that agree today are two
spellings that can drift tomorrow. So `mamba_step` reads as their `step`
reads -- the same order of the same operations, cited line by line below --
but every seam it reaches is the block spelling with `l = 1`, and the only
arithmetic written out longhand in this file is the SABOTAGE arm, which
exists to be falsified.

That is why upstream's own `step` has two arms at :215 and :238 (the torch
fallback and the fused CUDA kernel) and this file has one. Those two arms
do NOT agree bitwise -- the CUDA `selective_state_update` rounds
`B * (delta * u)` where the torch reference rounds `(delta * B) * u`
(contract seam S8, `selective_scan_fwd_kernel.cuh:162,222`) -- so a port
that mirrored the branch would mirror a bitwise fork. DEVIATION 732.

## The four departures from their spelling, numbered

DEVIATION 721 -- THE BIAS SEED. Their step (:218-220) sums the conv taps
first and adds `conv1d.bias` AFTER; the prefill kernels seed the
accumulator WITH the bias (MAX `causal_conv1d.mojo:190-205`, and the CUDA
`causal_conv1d` kernel likewise). Those are two different roundings of the
same conv, and the reference's own two paths therefore disagree with each
other. The profile takes the prefill kernels' bias SEED on BOTH paths,
because keeping both spellings would make contract gate D (decode ==
prefill, bitwise, per token) false by construction -- a gate that can only
be met by an accident of rounding is not a gate. This is the one place in
this lane where "do what they do" is not available, because there is no
single thing they do. Adopted at the narrowest possible width: only where
the bias enters the accumulator moves; the tap order (k ascending, oldest
first), the fusion (one fma per tap, contract S13), and the flush at every
seam are all unchanged. `conv_step_upstream_bias_last` below is their
:218-220 order, kept as the sabotage arm, and `check_decode_equals_prefill`
runs it and must FAIL: that is the proof that 721 is load bearing and not
cosmetic.

DEVIATION 732 -- ONE ARM, NOT TWO. Their :215 and :238 select between a
torch fallback and a fused CUDA kernel at import time. The two arms are not
bitwise equal (S8 above; also the CUDA scan's `D * u` seeding, S11). The
profile is the reference's arm, and this file has no kernel-present branch
to take. The name `causal_conv1d_update` appears nowhere here for the same
reason.

DEVIATION 733 -- THE ROLL, OUT OF PLACE. Their :216-217 update the window
in place: `roll(conv_state, -1)` then `conv_state[:, :, -1] = x`. The block
spelling rebuilds the window from the sequence and the incoming window
(`mamba_oracle.mojo`'s `new_win` loop: position `l - d_conv + j`, read from
the sequence when it is nonnegative and from the incoming window
otherwise). At `l = 1` the two are the same four values in the same order
-- `[w1, w2, w3, x]` -- because the window carries PRE-conv values, which
is what makes one spelling able to serve both paths at all. A copy is not
an arithmetic seam (contract section 4), so this moves no bits; it is
recorded because it is a visible difference in the port and because the
identity between them is a claim the gate checks (`conv.window` after every
step, compared against the prefill card's).

DEVIATION 734 -- THE CACHE'S SIGNATURE. `allocate_inference_cache(self,
batch_size, max_seqlen, dtype=None, **kwargs)` becomes
`allocate_inference_cache(batch_size, dims)`. `max_seqlen` is dropped
because Mamba's cache does not depend on it (their own body ignores it too:
the shapes at :258-265 are `(B, d_inner, d_conv)` and `(B, d_inner,
d_state)`, no sequence length in either); `dtype` and `device` are dropped
because the profile is Float32 everywhere (contract section 3) and this
lane ships one dtype, so a dtype argument could only ever carry a value the
profile forbids. The ZEROS -- the whole content of their function -- are
unchanged, and are what makes the first token's conv read zero padding and
its scan start from h = 0.

DEVIATION 735 -- A BLOCK STEP, NOT A MIXER STEP. Their `Mamba.step` is the
MIXER only; the norm and the residual live in `mamba_ssm/modules/block.py`,
whose order is `Add -> LN -> Mixer` with the residual threaded between
blocks. The profile's block order is HuggingFace's instead (contract
section 2 and section 1's block-order pin: `MambaBlock.forward` MM:505-530,
`residual = hidden; hidden = norm(hidden); hidden = mixer(hidden); hidden =
residual + hidden`), so the decode step here covers the whole block, norm
and residual included. The two upstreams genuinely disagree about where
those two operations sit, the contract already chose, and a decode step
that stopped at `out_proj` could not be compared against a prefill card
that does not (gate D is per STAGE, and `norm.sumsq` and `residual.out` are
stages).

## Where the arithmetic comes from, today and tomorrow

`mamba/mojo_only/mamba_oracle.mojo` is the landed interface authority for
this lane (buffer conventions, stage list, seam order), and its
`mamba_block_oracle` is the block spelling this step calls. The DEVICE
spelling of the same block --
`mamba/ported/transformers/models/mamba/modeling_mamba.mojo`, and the scan
in `mamba/ported/mamba_ssm/ops/selective_scan_interface.mojo` -- is being
written beside this file and is not on disk yet. When it lands,
`_one_block_call` below is the single line that changes, and NO arithmetic
in this file changes with it, because there is no arithmetic in this file
to change. That is the point of the delegation: the decode path has no
private conv and no private scan to keep in sync.

Run the gate:

    pixi run check-mamba-decode                  gate D, several corpus cases
    pixi run check-mamba-decode probe            + the per-stage reach probe
    pixi run check-mamba-decode sabotage         DEVIATION 721 undone, must FAIL
    pixi run check-mamba-decode sabotage-window  the state carry cut, must FAIL
"""

from std.memory import bitcast
from std.sys import argv

from core.identity_trace import IdentityTrace
from mojo_only.numerics import ftz, identical_mul_add
from mamba.mojo_only.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
    bits32_hex,
    corpus_case,
    corpus_case_weights,
    corpus_case_x,
    mode_name,
)
from mamba.mojo_only.mamba_oracle import (
    MambaStages,
    MambaState,
    mamba_block_oracle,
)


comptime DECODE_TOKENS = 1
"""Their :210 assert, as a constant: "Only support decoding with 1 token at
a time for now"."""


# ===========================================================================
# allocate_inference_cache (mamba_simple.py:255-266)
# ===========================================================================


def allocate_inference_cache(batch_size: Int, dims: MambaDims) -> MambaState:
    """Their :255-266: two ZERO tensors, `(B, d_inner, d_conv)` and
    `(B, d_inner, d_state)`. DEVIATION 734 for the dropped
    `max_seqlen`/`dtype`/`device` arguments.

    The zeros are load bearing twice over. The zero WINDOW is what makes the
    first decode token's conv read the same taps prefill's zero padding
    reads (contract section 5), and the zero h is `selective_scan_ref:160`'s
    initial state. `MambaState.__init__` is that allocation; this function
    is their NAME for it, so a reader coming from `mamba_simple.py` finds
    the call site they expect."""
    return MambaState(batch_size, dims)


# ===========================================================================
# Mamba.step (mamba_simple.py:208-253)
# ===========================================================================


def _one_block_call(
    w: MambaWeights,
    x: List[Float32],
    b: Int,
    l: Int,
    mut state: MambaState,
) raises -> MambaStages:
    """THE ONE SPELLING, the single call site both paths go through.

    Today it is `mamba_block_oracle` (the landed host authority). When
    `mamba/ported/transformers/models/mamba/modeling_mamba.mojo` lands, this
    body becomes its device forward and nothing else in this file moves.
    Prefill passes `l = L`; the decode step passes `l = DECODE_TOKENS`. The
    conv reads position `l - 3 + k` from the sequence when that is
    nonnegative and from `state.conv_win` otherwise, so at `l = 1` this IS
    their :216-221 chain, and the scan takes h in a buffer and runs any
    `l`."""
    return mamba_block_oracle(w, x, b, l, state)


def mamba_step(
    w: MambaWeights,
    hidden_states: List[Float32],
    b: Int,
    mut state: MambaState,
) raises -> MambaStages:
    """One decode token, `Mamba.step`'s semantics, line for line.

    `hidden_states` is `(B, 1, d_model)` token-major -- their :211
    `hidden_states.squeeze(1)`, which in a row-major list is the same bytes
    read as `(B, d_model)`; the squeeze moves nothing.

    `state` is their `(conv_state, ssm_state)` pair, updated IN PLACE, as
    their :216-217 and :243 `copy_` update theirs. They return the two
    tensors as well (:253); here the mutation is the return, because a
    `MambaState` is one object and returning a copy of what the caller
    already holds is a second thing to keep in sync.

    The returned card is the whole stage list of contract section 7 for
    this token (DEVIATION 735: a block step, so `norm.*` and `residual.out`
    are in it). Their :253 returns `out.unsqueeze(1)` alone; the block
    output is `stages.residual_out` and the mixer output is
    `stages.out_proj`, both bit-for-bit what a caller reading their return
    value would get.

    Their step, and where each line goes:

      :210  assert seqlen == 1                 -> the length check below
      :211  xz = in_proj(hidden.squeeze(1))    -> `in_proj.out`   (S17)
      :212  x, z = xz.chunk(2, -1)             -> a copy, not a stage
      :216  conv_state = roll(conv_state, -1)  -> DEVIATION 733, the window
      :217  conv_state[:, :, -1] = x              rebuilt out of place
      :218  x = sum(conv_state * weight, -1)   -> `conv.out`, seam S13, with
      :220  x = x + conv1d.bias                   the bias SEEDED, not added
                                                  last: DEVIATION 721
      :221  x = act(x)                         -> `silu.out` (S12's function)
      :231  x_db = x_proj(x)                   -> `x_proj.out`  (S17)
      :232  dt, B, C = split(...)              -> copies
      :234  dt = F.linear(dt, dt_proj.weight)  -> `dt_proj.out`, bias NOT
                                                  added here (:233's comment)
      :235  A = -exp(A_log)                    -> `A.out`        (S15)
      :240  dt = softplus(dt + dt_proj.bias)   -> `softplus.out` (S14)
      :241  dA = exp(einsum(dt, A))            -> S5, S6
      :242  dB = einsum(dt, B)                 -> S7
      :243  ssm_state = ssm_state*dA + x*dB    -> `scan.h`, S8 and S9 (fused:
                                                  contract section 4's S9)
      :244  y = einsum(ssm_state, C)           -> `scan.y`       (S10)
      :245  y = y + D * x                      -> `skip.out`     (S11)
      :246  y = y * act(z)                     -> `gate.out`     (S12)
      :252  out = out_proj(y)                  -> `out_proj.out` (S17)
    """
    var dm = w.dims.d_model
    if len(hidden_states) != b * DECODE_TOKENS * dm:
        raise Error(
            String("mamba_step: ")
            + String(len(hidden_states))
            + " values for B = "
            + String(b)
            + " and d_model = "
            + String(dm)
            + "; the step decodes exactly one token per batch row"
            + " (mamba_simple.py:210)"
        )
    return _one_block_call(w, hidden_states, b, DECODE_TOKENS, state)


def step_token_view(
    x: List[Float32], bb_count: Int, li: Int, l: Int, d_model: Int
) -> List[Float32]:
    """Token `li` of every batch row of a `(B, L, d_model)` sequence, as the
    `(B, 1, d_model)` argument `mamba_step` takes. A gather of raw values;
    no arithmetic, so no seam."""
    var out = List[Float32]()
    for bb in range(bb_count):
        for j in range(d_model):
            out.append(x[(bb * l + li) * d_model + j])
    return out^


# ===========================================================================
# The stage card (contract section 7), tagged
# ===========================================================================


def record_step_card(
    mut trace: IdentityTrace,
    prefix: String,
    x: List[Float32],
    st: MambaStages,
) raises:
    """Every stage of one call, in the card's order, with the driver's
    prefix (`core/identity_trace.mojo` rule 2: a tag names a position in the
    ALGORITHM, never a property of the machine -- `mamba.decode.tok03.` is a
    position, `block64.` would not be).

    `conv.window` and `scan.h` are the two this lane owes: the state AFTER
    the call, which is the whole content of the recurrence. Both are copies
    of values computed at other seams, which is exactly why they are cheap
    to record and worth recording -- a decode/prefill divergence in the
    carried state shows up here one token before it shows up in an output.
    """
    trace.record_list_f32(prefix + "input.x", x)
    trace.record_list_f32(prefix + "norm.sumsq", st.norm_sumsq)
    trace.record_list_f32(prefix + "norm.out", st.norm_out)
    trace.record_list_f32(prefix + "in_proj.out", st.in_proj)
    trace.record_list_f32(prefix + "A.out", st.a_out)
    trace.record_list_f32(prefix + "conv.out", st.conv_out)
    trace.record_list_f32(prefix + "silu.out", st.silu_out)
    trace.record_list_f32(prefix + "conv.window", st.conv_win)
    trace.record_list_f32(prefix + "x_proj.out", st.x_proj)
    trace.record_list_f32(prefix + "dt_proj.out", st.dt_proj)
    trace.record_list_f32(prefix + "softplus.out", st.softplus_out)
    trace.record_list_f32(prefix + "scan.y", st.scan_y)
    trace.record_list_f32(prefix + "scan.h", st.scan_h)
    trace.record_list_f32(prefix + "skip.out", st.skip_out)
    trace.record_list_f32(prefix + "gate.out", st.gate_out)
    trace.record_list_f32(prefix + "out_proj.out", st.out_proj)
    trace.record_list_f32(prefix + "residual.out", st.residual_out)


# ===========================================================================
# THE SABOTAGE ARM -- their :218-220 order, restored on purpose
# ===========================================================================


def conv_step_upstream_bias_last(
    w: MambaWeights,
    in_proj: List[Float32],
    win_before: List[Float32],
    b: Int,
    di: Int,
) -> List[Float32]:
    """`mamba_simple.py:218-220` VERBATIM in its bias placement: the taps
    are summed from `+0.0` and `conv1d.bias` is added to the finished sum.
    DEVIATION 721 is the decision NOT to spell the decode conv this way;
    this function is that decision's falsifier and is called only by the
    `sabotage` arm of the gate below.

    ONE VARIABLE MOVES. The tap order (k ascending, oldest first), the
    per-tap fma (contract S13), the flush at every seam and the window are
    all the pinned spelling; only where the bias enters changes. Their torch
    line also rounds each `conv_state * weight` product on its own before
    summing, which is a SECOND difference -- but it is a difference the
    contract already decided (S13 is FUSED, and MAX's and the CUDA kernel's
    prefill conv are both fused), so folding it in here would confound the
    sabotage. A sabotage with two variables cannot tell you which one the
    gate caught.

    Exact at every position, not just the first token, for a reason worth
    stating: the conv window carries PRE-conv values (:217 stores `x`, the
    in_proj hidden, not the conv output), so a sabotaged `conv.out` never
    feeds back into a later `conv.out`. Recomputing this stage from the
    unsabotaged inputs is therefore exactly what a fully sabotaged decode
    would produce at this stage, at every token.
    """
    var out = List[Float32]()
    for bb in range(b):
        for d in range(di):
            var acc = Float32(0.0)  # +0.0 seed: the bias is NOT here (:218)
            for k in range(D_CONV):
                var p = (DECODE_TOKENS - 1) - (D_CONV - 1) + k
                var xv: Float32
                if p >= 0:
                    xv = in_proj[bb * 2 * di + d]
                else:
                    xv = win_before[(bb * di + d) * D_CONV + (D_CONV + p)]
                acc = ftz(
                    identical_mul_add(
                        ftz(w.conv_w[d * D_CONV + k]), ftz(xv), acc
                    )
                )
            out.append(ftz(acc + ftz(w.conv_b[d])))  # :220, the bias LAST
    return out^


# =========================================================================
# THE GATE: stage compares, the reach probe, and gate D itself
# =========================================================================


def _cmp_stage(
    name: String,
    where: String,
    dec: List[Float32],
    d_off: Int,
    pre: List[Float32],
    p_off: Int,
    n: Int,
    mut fails: Int,
    mut reported: Int,
    mut first_name: String,
) raises:
    """Bitwise, by BITS and never by compare (contract section 6 / row 49:
    Metal flushes compare operands, and IEEE says `-0.0 == +0.0`, so a float
    compare would call a moved sign bit agreement)."""
    for j in range(n):
        var a = bitcast[DType.uint32](dec[d_off + j])
        var c = bitcast[DType.uint32](pre[p_off + j])
        if a != c:
            fails += 1
            if first_name == "":
                first_name = name
            if reported < 4:
                reported += 1
                print(
                    "    MISMATCH",
                    name,
                    where,
                    "cell",
                    j,
                    "decode",
                    bits32_hex(dec[d_off + j]),
                    "prefill",
                    bits32_hex(pre[p_off + j]),
                )


# ---------------------------------------------------------------------------
# The card as an indexed list, so the compare is ONE loop over the stage list
# and every stage's comparator can be probed by index (`verify reach, not
# output`, and reach is PER BRANCH: fourteen stage compares are fourteen
# branches, and a sabotage that moves the conv says nothing about whether the
# `norm.sumsq` compare is even looking at the right two cells).
# ---------------------------------------------------------------------------

comptime STAGE_A_OUT = 3
"""`A.out` is `[d_inner, 16]`, a function of `A_log` alone: the same buffer
for every token and every batch row, so its compare takes offset 0 on both
sides while every other token stage is offset by its row."""
comptime STAGE_CONV_OUT = 4
comptime TOKEN_STAGES = 14
comptime STATE_STAGES = 4
comptime ALL_STAGES = TOKEN_STAGES + STATE_STAGES


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
        return String("x_proj.out")
    if i == 7:
        return String("dt_proj.out")
    if i == 8:
        return String("softplus.out")
    if i == 9:
        return String("scan.y")
    if i == 10:
        return String("skip.out")
    if i == 11:
        return String("gate.out")
    if i == 12:
        return String("out_proj.out")
    if i == 13:
        return String("residual.out")
    if i == 14:
        return String("conv.window")
    if i == 15:
        return String("scan.h")
    if i == 16:
        return String("state.conv_win")
    if i == 17:
        return String("state.h")
    raise Error("stage_name: no stage " + String(i))


def stage_width(i: Int, dm: Int, di: Int, xr: Int) raises -> Int:
    """Elements per token row (per BATCH row, at `l = 1`)."""
    if i == 0:
        return 1
    if i == 1 or i == 12 or i == 13:
        return dm
    if i == 2:
        return 2 * di
    if i == 3:
        return di * D_STATE
    if i == 6:
        return xr
    if i >= 4 and i <= 11:
        return di
    raise Error("stage_width: stage " + String(i) + " is not a token stage")


def token_stage_lists(
    st: MambaStages, conv_out: List[Float32]
) -> List[List[Float32]]:
    """The fourteen token stages of one card, in contract section 7's order.

    `conv_out` is passed separately because the sabotage arm substitutes it;
    in every other run it is `st.conv_out` unchanged. Copies -- a copy moves
    bits untouched (contract section 4), which is the whole reason a stage
    list can be assembled this way without becoming a second computation."""
    var out = List[List[Float32]]()
    out.append(st.norm_sumsq.copy())
    out.append(st.norm_out.copy())
    out.append(st.in_proj.copy())
    out.append(st.a_out.copy())
    out.append(conv_out.copy())
    out.append(st.silu_out.copy())
    out.append(st.x_proj.copy())
    out.append(st.dt_proj.copy())
    out.append(st.softplus_out.copy())
    out.append(st.scan_y.copy())
    out.append(st.skip_out.copy())
    out.append(st.gate_out.copy())
    out.append(st.out_proj.copy())
    out.append(st.residual_out.copy())
    return out^


def flip_low_bit(v: Float32) -> Float32:
    """One bit of one cell, by bits: the smallest possible perturbation, and
    one that exists for every value including a zero of either sign."""
    return bitcast[DType.float32](bitcast[DType.uint32](v) ^ UInt32(1))


# ===========================================================================
# GATE D -- decode == prefill, bitwise, at every position
# ===========================================================================


comptime SABOTAGE_NONE = 0
comptime SABOTAGE_BIAS_LAST = 1
comptime SABOTAGE_NO_CARRY = 2


def check_decode_equals_prefill(
    case_k: Int,
    sabotage: Int,
    mut trace: IdentityTrace,
    probe_stage: Int,
    mut first_name: String,
    quiet: Bool,
) raises -> Int:
    """Contract gate (d): "decode == prefill bitwise at every position".

    One corpus case, run twice: once as a single prefill of L tokens, once
    as L decode steps of one token each carrying `MambaState`. Every stage
    of contract section 7 is compared for every (batch row, position), and
    the carried state (`conv.window`, `scan.h`) is compared after the last
    token against the prefill card's -- both as the CARD records it and as
    the `MambaState` the caller carries holds it, because a card that agrees
    while the carried buffer does not would decode the next token from the
    wrong window.

    The claim this gate checks is STRUCTURAL -- one spelling serves both
    paths, so the two runs execute the same seams in the same order on the
    same bits -- which is why the honest use of the gate is its sabotage and
    probe arms. A gate that can only pass is an instrument that has never
    moved. Returns the number of mismatching cells.

    `probe_stage >= 0` flips ONE BIT of ONE CELL of that stage in the decode
    card before the compare: the per-branch reach test. It is not a claim
    about the block; it is a claim about this function.
    """
    var c = corpus_case(case_k)
    var w = corpus_case_weights(case_k)
    var x = corpus_case_x(case_k)
    var dims = w.dims.copy()
    var b = c.b
    var l = c.l
    var dm = dims.d_model
    var di = dims.d_inner
    var xr = dims.x_proj_rows()
    var tag = String("mamba.") + String(c.name) + "."

    # ---- the prefill card: one call, L tokens, a zero cache -------------
    var pre_state = allocate_inference_cache(b, dims)
    var pre = _one_block_call(w, x, b, l, pre_state)
    record_step_card(trace, tag + "prefill.", x, pre)
    var pre_stages = token_stage_lists(pre, pre.conv_out)

    # ---- the decode chain: L calls of one token, the cache carried ------
    var state = allocate_inference_cache(b, dims)
    var fails = 0
    var reported = 0
    for li in range(l):
        var xt = step_token_view(x, b, li, l, dm)
        var win_before = state.conv_win.copy()
        if sabotage == SABOTAGE_NO_CARRY:
            # The state carry CUT: every token starts from the zero cache
            # `allocate_inference_cache` hands out. This is the sabotage for
            # the OTHER half of contract section 5 -- that the window and h
            # are what make one spelling serve both paths -- and it must
            # fail from token 1 on (token 0 legitimately agrees: at token 0
            # the carried state IS the zero cache).
            state = allocate_inference_cache(b, dims)
            win_before = state.conv_win.copy()
        var st = mamba_step(w, xt, b, state)
        var pos = String("[tok ") + String(li) + "]"

        var conv_out = st.conv_out.copy()
        if sabotage == SABOTAGE_BIAS_LAST:
            conv_out = conv_step_upstream_bias_last(
                w, st.in_proj, win_before, b, di
            )

        var dec_stages = token_stage_lists(st, conv_out)
        if probe_stage >= 0 and probe_stage < TOKEN_STAGES:
            dec_stages[probe_stage][0] = flip_low_bit(
                dec_stages[probe_stage][0]
            )

        for bb in range(b):
            var pt = bb * l + li  # the prefill card's row for this token
            var where = pos + String(" b") + String(bb)
            for si in range(TOKEN_STAGES):
                var wdt = stage_width(si, dm, di, xr)
                var d_off = bb * wdt
                var p_off = pt * wdt
                if si == STAGE_A_OUT:
                    d_off = 0
                    p_off = 0
                _cmp_stage(
                    stage_name(si),
                    where,
                    dec_stages[si],
                    d_off,
                    pre_stages[si],
                    p_off,
                    wdt,
                    fails,
                    reported,
                    first_name,
                )

        var tok = String("decode.tok")
        if li < 10:
            tok += "0"
        tok += String(li) + "."
        record_step_card(trace, tag + tok, xt, st)

        # The carried state at the LAST token is the prefill card's state:
        # every earlier token's window and h belong to a prefix run the
        # prefill card does not record (contract section 7 records the state
        # after the call, and the prefill call is one call).
        if li == l - 1:
            var nw = b * di * D_CONV
            var nh = b * di * D_STATE
            var end_dec = List[List[Float32]]()
            end_dec.append(st.conv_win.copy())
            end_dec.append(st.scan_h.copy())
            end_dec.append(state.conv_win.copy())
            end_dec.append(state.h.copy())
            var end_pre = List[List[Float32]]()
            end_pre.append(pre.conv_win.copy())
            end_pre.append(pre.scan_h.copy())
            end_pre.append(pre.conv_win.copy())
            end_pre.append(pre.scan_h.copy())
            if probe_stage >= TOKEN_STAGES:
                var ei = probe_stage - TOKEN_STAGES
                end_dec[ei][0] = flip_low_bit(end_dec[ei][0])
            for ei in range(STATE_STAGES):
                var n = nw if (ei == 0 or ei == 2) else nh
                _cmp_stage(
                    stage_name(TOKEN_STAGES + ei),
                    pos,
                    end_dec[ei],
                    0,
                    end_pre[ei],
                    0,
                    n,
                    fails,
                    reported,
                    first_name,
                )

    if not quiet:
        # The seven `d_inner`-wide stages are conv.out, silu.out,
        # dt_proj.out, softplus.out, scan.y, skip.out and gate.out.
        var per_token = 1 + dm + 2 * di + di * D_STATE + xr + 7 * di + 2 * dm
        print(
            "  ",
            c.name,
            "B =",
            b,
            "L =",
            l,
            "d_model =",
            dm,
            "->",
            fails,
            "mismatching cells of",
            l * b * per_token + 2 * b * di * (D_CONV + D_STATE),
            "compared",
        )
    return fails


def probe_every_stage_compare(mut trace: IdentityTrace) raises:
    """REACH, per branch. Each of the eighteen compares in the gate above is
    its own branch, and a sabotage that moves the conv proves nothing about
    whether the `norm.sumsq` compare is reading the right two cells -- the
    classic `reached-but-inert` shape, where an offset bug makes a stage
    compare a buffer with ITSELF and report agreement forever.

    So: flip one bit of one cell of one stage of the decode card, and
    require that THAT stage is the one the gate reports first. Eighteen
    runs, one stage each, on the smallest case with more than one token and
    more than one batch row (`base_b2_l4_d8`)."""
    print("  reach probe: one bit flipped per stage, on base_b2_l4_d8")
    for si in range(ALL_STAGES):
        var first = String("")
        var n = check_decode_equals_prefill(1, SABOTAGE_NONE, trace, si, first, True)
        var want = stage_name(si)
        if n == 0:
            raise Error(
                String("REACH FAILURE: a flipped bit in stage '")
                + want
                + "' was not seen by the gate. That compare is inert:"
                + " it is looking at the wrong cells, or at none."
            )
        if first != want:
            raise Error(
                String("REACH FAILURE: a flipped bit in stage '")
                + want
                + "' was first reported against stage '"
                + first
                + "'. The stage list and the compare loop disagree."
            )
        print("    ", want, "-> caught,", n, "cells")


def main() raises:
    var sabotage = SABOTAGE_NONE
    var probe = False
    for a in argv():
        if a == "sabotage":
            sabotage = SABOTAGE_BIAS_LAST
        if a == "sabotage-window":
            sabotage = SABOTAGE_NO_CARRY
        if a == "probe":
            probe = True

    print(
        "mamba decode step (mamba_simple.py:208-266, e9594ce); gate D:",
        "decode == prefill, bitwise, per token; build mode",
        mode_name(),
    )
    if sabotage == SABOTAGE_BIAS_LAST:
        print(
            "SABOTAGE: DEVIATION 721 undone -- the decode conv sums its taps",
            "from +0.0 and adds conv1d.bias last (:218-220). Gate D MUST"
            " fail.",
        )
    if sabotage == SABOTAGE_NO_CARRY:
        print(
            "SABOTAGE: the state carry cut -- every token starts from the",
            "zero cache. Gate D MUST fail (from token 1 on).",
        )

    var trace = IdentityTrace()

    # Deliberately small (Andrew's order 3: no performance work, small
    # data). The cases are chosen for COVERAGE of the recurrence, not size:
    # L = 1 (the window is all zeros and one decode step is the whole
    # sequence), L = 4 (the first token whose conv reads no padding at all
    # is token 3), L > d_conv with B > 1 (a window that has rolled, several
    # batch rows), the signed-zero case (a -0.0 must survive the carry with
    # its sign, contract section 6), the softplus guard case and the
    # A-near-zero case (where the scan's own arithmetic is most fragile),
    # and the gate-saturation case (a huge z through S12).
    var cases = List[Int]()
    cases.append(0)   # base_b1_l1_d8
    cases.append(1)   # base_b2_l4_d8
    cases.append(2)   # base_b3_l16_d8
    cases.append(5)   # base_b3_l4_d16
    cases.append(8)   # adv_softplus_guard_b2_l8_d8
    cases.append(10)  # adv_a_near_zero_b3_l8_d8
    cases.append(11)  # adv_signed_zeros_b2_l8_d8
    cases.append(12)  # adv_gate_saturation_b1_l8_d16

    var total = 0
    var missed = 0
    var spurious = 0
    for i in range(len(cases)):
        var first = String("")
        var n = check_decode_equals_prefill(
            cases[i], sabotage, trace, -1, first, False
        )
        total += n
        # A sabotage arm is judged PER CASE, not in total: an arm that moves
        # one shape and leaves seven alone would total a large number and
        # still mean the clause is load bearing on one shape only.
        var one_token = corpus_case(cases[i]).l == 1
        var must_move = sabotage == SABOTAGE_BIAS_LAST or not one_token
        # The state-carry sabotage CANNOT move an L = 1 case and must not
        # be credited for one: at token 0 the carried state IS the zero
        # cache `allocate_inference_cache` hands out, so cutting the carry
        # cuts nothing. That case is the arm's own control -- if it moved,
        # the arm would be perturbing something other than the carry.
        if must_move and n == 0:
            missed += 1
        if (not must_move) and n != 0:
            spurious += 1

    if sabotage == SABOTAGE_NONE:
        if total != 0:
            raise Error(
                String("GATE D FAILED: ")
                + String(total)
                + " cells differ between the decode chain and the prefill"
                + " card. Contract section 5 says this is structural, so a"
                + " failure here is a SECOND spelling that has crept in."
            )
        print(
            "GATE D PASS: every stage of contract section 7 is bit-identical",
            "between",
            len(cases),
            "prefill cards and their decode chains, at every position,",
            "including the carried conv.window and scan.h.",
        )
        if probe:
            probe_every_stage_compare(trace)
        print(
            "This gate is a theorem, not a coincidence: both runs reach the",
            "same seams through _one_block_call. Run `sabotage`,",
            "`sabotage-window` and `probe` -- they must fail or catch, or",
            "this line means nothing.",
        )
    else:
        if total == 0:
            raise Error(
                "SABOTAGE ARM REPORTED ZERO MISMATCHES: the gate is blind."
                " A sabotage that passes is a gate that is not reaching the"
                " path it claims to check (`reached-but-inert`)."
            )
        if missed != 0:
            raise Error(
                String("SABOTAGE ARM MISSED ")
                + String(missed)
                + " of the cases it must move. Every case must move (bar the"
                + " state-carry arm's L = 1 control), or the sabotage is"
                + " shape-dependent and the clause it undoes is only load"
                + " bearing on some shapes."
            )
        if spurious != 0:
            raise Error(
                String("SABOTAGE ARM MOVED ")
                + String(spurious)
                + " case(s) it cannot legitimately move (an L = 1 case under"
                + " the state-carry arm). The arm is perturbing something"
                + " other than the clause it names."
            )
        print(
            "SABOTAGE ARM FAILS AS REQUIRED ON EVERY CASE IT CAN MOVE:",
            total,
            "mismatching cells. The undone clause is load bearing.",
        )
