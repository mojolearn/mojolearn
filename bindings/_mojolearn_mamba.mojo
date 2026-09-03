# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the Mamba-1, Mamba-2 and Mamba-3 block lanes.

A FOURTEENTH extension module, and a separate one on purpose, for the
reason `bindings/_mojolearn_estimators.mojo`'s header gives: an
independently changing binding must not become a merge point. This file
closes the "PyPI surface: NONE EXISTS" row of `mamba/FEATURE_PARITY.md`'s
consumer table -- the certified Mamba-1 block (profile
`mojolearn.identical.mamba1.fp32.v1`, three-vendor card) and the Mamba-2
block (profile `mojolearn.identical.mamba2.fp32.v1`, Apple-gated,
cross-vendor legs owed) exported no Python symbol at all before it. The
Mamba-3 entries (profile `mojolearn.identical.mamba3.siso.fp32.v1`,
Apple-gated at the kernel level 2026-09-01, cross-vendor legs owed)
joined later the same day, under the same ABI.

THE BINDING CALLS THE CERTIFIED ENTRY POINTS AND NOTHING ELSE. Every
forward below goes through `mamba_block_forward`
(`mamba/impl/transformers/models/mamba/modeling_mamba.mojo`),
`mamba2_block_forward` (`mamba/impl/mamba_ssm/modules/mamba2.mojo`) or
`mamba3_block_forward` (`mamba/impl/mamba_ssm/modules/mamba3.mojo`) --
the SAME functions `mamba/checks/mamba_check.mojo`, `mamba2_check.mojo`
and `mamba3_check.mojo` gate. NO arithmetic is respelled in this file: a
binding-side copy of any seam would be a second spelling of a pinned
rounding, which is the drift the whole lane exists to forbid. The decode
entry points are the contract's own construction -- the block forward at
`l = 1` with the state carried (mamba1 contract section 5; mamba2
DEVIATION 786 and mamba3 DEVIATION 831, decode is PREFILL RESUMPTION) --
exactly as `mamba2_step` and `mamba3_step` spell it ("no arithmetic of
its own").

DEVIATION 791 -- THE TWO-LIST ABI, ADOPTED UP FRONT. Each entry point
takes exactly TWO Python lists, `addrs` (every NumPy buffer address, in
an exact order written in each docstring and mirrored at the
`_mamba_impl.py` call site) and `params` (the scalars). This is not a
preference: `PythonModuleBuilder.def_function` stops elaborating above
roughly nine arguments, MEASURED on 2026-09-01 when `gpr_fit`'s
ten-positional first spelling failed at def_function elaboration
(`bindings/_mojolearn_gp.mojo`'s header carries the log path), and
`mamba1_forward` needs FOURTEEN addresses. The fold is the tree's own
precedent (knn_search's SCALARS-ARRIVE-AS-ONE-LIST banner; the gp
binding's addrs list). Every array whose address goes into `addrs` MUST
be bound to a Python local for the duration of the call: an address
inside a list keeps nothing alive (`python/mojolearn/_arrays.py`).

DEVIATION 792 -- THE STATE CROSSES AS CALLER-OWNED BUFFERS, IN PLACE.
The recurrent state is EXPLICIT ARGUMENTS, never hidden in an object on
this side of the boundary (`mamba/FEATURE_PARITY.md` section 3's SHIP
NOW row says why: explicit buffers are strictly more testable, and the
exact-state-handoff requirement is that a consumer can round-trip the
bytes). Mamba-1 carries TWO pieces (contract section 5): the conv
WINDOW `[B, d_inner, 4]` (last d_conv PRE-conv inputs, oldest first)
and the SSM state h `[B, d_inner, 16]`. Mamba-2 carries THREE (its
contract section 5): the conv window `[B, conv_dim, 4]` over the full
xBC width, the chunk-BOUNDARY h `[B, H, 64, 128]`, and the open chunk's
buffer -- post-conv/post-SiLU xBC rows `[B, 256, conv_dim]` plus raw dt
rows `[B, 256, H]` -- with `buf_len` (how many buffer rows are valid)
crossing as a params scalar IN and as the return value OUT, because it
is the one piece of state that is an integer and not a float buffer.
Every state buffer is READ at entry and WRITTEN BACK before return, so
the caller's arrays always hold the post-call state; zeros in, for a
fresh sequence, are `allocate_inference_cache`'s exact semantics.
Nonzero h in IS the mamba2 `initial_states` path (ssd_minimal.py:64-66,
prepended as chunk -1's state) -- no separate argument exists because
no separate buffer exists. `h_last` `[B, H, 64, 128]` is additionally
written out on every Mamba-2 call: it is the contract's REPORT stage
(the S17 value after the final PADDED chunk), deliberately distinct
from the resumption h, and the distinction is the contract's, not this
file's.

DEVIATION 793 -- WHAT THIS FILE REFUSES, AND WHAT IT LEAVES TO THE
LANE. Refused HERE: a null address, an `addrs`/`params` list of the
wrong length, and a `buf_len` outside `[0, 256)` -- each of which means
the two sides of THIS boundary disagree and no lane refusal exists for
it. Everything else goes down UNJUDGED so the lane's own refusals stay
reachable from Python: non-finite inputs and weights are refused BY
NAME on the device path (`mamba_refuse_bad_inputs` /
`mamba2_refuse_bad_inputs`, contract section 6), B/L positivity and the
shape cross-checks are `mamba_block_forward`'s and
`mamba2_block_forward`'s, and a d_model that is not a multiple of 32 is
`Mamba2Dims.of`'s refusal, raised in Mojo with the profile rule in the
message. The dtype refusals (float32 ONLY; bf16/fp16/float64 refused BY
NAME) live in `_mamba_impl.py`, because dtype is a NumPy property this
side cannot see -- by the time bits arrive here they are float32 by
construction or the wrapper has already refused. A SABOTAGED BUILD
REFUSES TO INITIALIZE: every sabotage arm in the lane exists to be run
by a gate and to FAIL, and a wheel that quietly served one would be the
accepted-and-ignored failure at library scale, so `PyInit` aborts by
name if any arm is compiled in.

DEVIATION 794 -- THE MAMBA-3 STATE IS TEN BUFFERS PLUS TWO SCALARS, AND
`pending` IS CONSUMED. DEVIATION 792's caller-owned-buffers rule extends
to Mamba-3's DEVIATION-832 state, which has TEN float pieces: theta
`[B, H, 32]`, the SEALED chunk-boundary h `[B, H, 64, 128]`, the six
last-working-chunk buffers (rotated-unscaled q/k `[B, 64, H, 128]`, raw
v `[B, 64, H, 64]`, dt / sigma(trap) / ADT `[B, 64, H]`), and the
pending Input_States pair k `[B, H, 128]` / v `[B, H, 64]`. Two scalars
cross in `params`: `buf_len`, whose valid range is [0, 64] INCLUSIVE --
DELIBERATELY unlike mamba2's [0, 256) exclusive, because the mamba3
buffer NEVER EMPTIES (DEVIATION 832(i): r in [1, Q] after every call; 0
only before the first token) -- and `pending`, a 0/1 flag saying the
k/v pair (with theta and h) is an upstream Input_States continuation.
`pending` crosses IN only: in a shipped binding it is ALWAYS False on
return, because the unarmed core consumes it (`mamba3_block_forward`
clears it after S22) and an armed build aborts at PyInit below -- so
the return value stays `buf_len` alone, mamba2's shape, and the wrapper
reassigns `state.pending = False` on that guarantee. The pending path
goes through the lane's OWN `Mamba3DeviceState.set_input_states`, so
its fresh-state refusal (Input_States only has upstream meaning at
buf_len 0) stays reachable from Python instead of being respelled here.
FOUR report buffers are written on every Mamba-3 call -- h_last,
k_last, v_last, theta_last, the contract's REPORT stages, distinct from
the resumption state exactly as mamba2's h_last is.

THE THREE-TIER SEMANTICS ARE THE BUILD'S, NOT THIS FILE'S. The numeric
mode (fast / deterministic / identical) is a compile-time define
(`checks/numerics.mojo`), one binary per tier under
`python/mojolearn/[<tier>/]_mojolearn_mamba.so`; `mamba_numeric_mode`
below is the read-back `_mamba_impl.py` cross-checks so a wrong-arm
measurement cannot be correctly labelled by accident.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.

RUN LEDGER. The Mamba-1/Mamba-2 entries BUILT AND GATED 2026-09-01 (all
three tiers green through `tests/test_mamba_surface.py`, the
FEATURE_PARITY consumer table's PyPI row; one box, one vendor). The
Mamba-3 entries, added later the same day, are BUILT AND GATED: green
at `08a38a13` in all three tiers, corpus arm added 2026-09-03. ONE APPLE
M4 -- this binding FAULTS on AMD in every tier (GPU memory access fault,
rc 134), an open defect. Build:
`bash bindings/build_mamba.sh` (per tier via MOJOLEARN_NUMERIC_MODE).
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR

from mamba.checks.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
)
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    BLOCK_ANY_SABOTAGE,
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
    mamba_block_forward,
    mamba_download,
    mamba_upload,
)
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
    Mamba2Weights,
)
from mamba.impl.mamba_ssm.modules.mamba2 import (
    BLOCK2_ANY_SABOTAGE,
    Mamba2DeviceStages,
    Mamba2DeviceState,
    Mamba2DeviceWeights,
    mamba2_block_forward,
)
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
    Mamba3Weights,
)
from mamba.impl.mamba_ssm.modules.mamba3 import (
    BLOCK3_ANY_SABOTAGE,
    Mamba3DeviceStages,
    Mamba3DeviceState,
    Mamba3DeviceWeights,
    mamba3_block_forward,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _read_f32(addr: Int, n: Int) raises -> List[Float32]:
    """The first `n` float32 of a borrowed NumPy buffer, as a host list.
    A COPY, deliberately: the device helpers below take host lists, and
    the borrow ends when this returns, so no pointer is retained."""
    var p = _f32_ptr(addr)
    var out = List[Float32]()
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


def _write_f32(addr: Int, values: List[Float32]) raises:
    """A host list into a borrowed NumPy buffer, element for element."""
    var p = _f32_ptr(addr)
    for i in range(len(values)):
        p.unsafe_store(i, values[i])


def mamba_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC. The same shape as `gp_numeric_mode` and
    for the same reason: `_mamba_impl.py` reads it once and refuses to
    run if the binary it loaded disagrees with the mode the package
    asked for."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mamba_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none' -- the accelerator API this
    binary was compiled for, folded in from `checks/vendor.mojo`. The
    answer comes from the binary that actually loaded, never from the
    directory it sat in."""
    return PythonObject(String(COMPILED_VENDOR))


# ===========================================================================
# Mamba-1: profile mojolearn.identical.mamba1.fp32.v1
# ===========================================================================


def _mamba1_run(a: List[Int], b: Int, l: Int, dm: Int) raises:
    """The GIL-free half of `mamba1_forward_binding`: everything after
    the `PythonObject`s have been read. Builds the host weights, uploads
    the caller's state, runs THE certified entry point once, and writes
    the output and the post-call state back into the caller's buffers.

    `[[mojo-buffer-freed-at-last-use]]`: every device buffer below is
    still alive when `mamba_block_forward` returns because that function
    synchronizes before it does, and the explicit transfers at the end
    hold them past the last download anyway."""
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()

    # Host weights, THROUGH the lane's own struct (its field table is the
    # upstream shape authority); values arrive as given bits, unjudged
    # (the initializer rows of FEATURE_PARITY.md are REFUSED as bitwise
    # claims; weights-in is what ships).
    var w = MambaWeights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], 2 * di * dm)
    w.conv_w = _read_f32(a[3], di * D_CONV)
    w.conv_b = _read_f32(a[4], di)
    w.w_x = _read_f32(a[5], xr * di)
    w.w_dt = _read_f32(a[6], di * r)
    w.b_dt = _read_f32(a[7], di)
    w.a_log = _read_f32(a[8], di * D_STATE)
    w.d_skip = _read_f32(a[9], di)
    w.w_out = _read_f32(a[10], dm * di)

    var ctx = DeviceContext()
    var dw = MambaDeviceWeights(ctx, w)
    # The caller's state, uploaded over the fresh zeros. Zeros in IS
    # allocate_inference_cache; anything else is a carried sequence.
    var dstate = MambaDeviceState(ctx, b, dims)
    dstate.conv_win = mamba_upload(ctx, _read_f32(a[11], b * di * D_CONV))
    dstate.h = mamba_upload(ctx, _read_f32(a[12], b * di * D_STATE))
    var dstages = MambaDeviceStages(ctx, b, l, dims)
    var dx = mamba_upload(ctx, _read_f32(a[0], b * l * dm))

    var trace = IdentityTrace.disabled()
    mamba_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, trace, String("py")
    )

    # The block output is the residual stage (contract section 2's
    # `hidden = residual + mixer(norm(residual))`), and the state buffers
    # go back to their owners.
    _write_f32(a[13], mamba_download(ctx, dstages.residual_out, b * l * dm))
    _write_f32(a[11], mamba_download(ctx, dstate.conv_win, b * di * D_CONV))
    _write_f32(a[12], mamba_download(ctx, dstate.h, b * di * D_STATE))
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^


def _mamba1_addrs(addrs: PythonObject, what: String) raises -> List[Int]:
    if len(addrs) != 14:
        raise Error(
            what
            + ": addrs must contain 14 addresses (x, norm.weight,"
            " in_proj.weight, conv1d.weight, conv1d.bias, x_proj.weight,"
            " dt_proj.weight, dt_proj.bias, A_log, D, out_proj.weight,"
            " conv_window, h, y_out), got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(14):
        a.append(Int(py=addrs[i]))
    return a^


def mamba1_forward_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One Mamba-1 block call: prefill from the state the caller hands
    in (zeros = a fresh sequence), any B and L. The decode step is this
    SAME function at L = 1 (contract section 5: one spelling for both
    paths is what makes gate D a theorem); `mamba1_decode_step` below is
    that call spelled as their `step`'s name.

    `addrs` is the FOURTEEN buffer addresses, in this exact order
    (mirrored in `python/mojolearn/_mamba_impl.py`; the module header
    says why they are a list and not fourteen arguments):

        0  x               B * L * d_model float32, read
        1  norm.weight     d_model, read
        2  in_proj.weight  2*d_inner * d_model, read
        3  conv1d.weight   d_inner * 4, read (their [d_inner,1,4], the 1
                            dropped -- same bytes)
        4  conv1d.bias     d_inner, read
        5  x_proj.weight   (dt_rank + 32) * d_inner, read
        6  dt_proj.weight  d_inner * dt_rank, read
        7  dt_proj.bias    d_inner, read
        8  A_log           d_inner * 16, read
        9  D               d_inner, read
       10  out_proj.weight d_model * d_inner, read
       11  conv_window     B * d_inner * 4, READ AND WRITTEN (state)
       12  h               B * d_inner * 16, READ AND WRITTEN (state)
       13  y_out           B * L * d_model, WRITTEN (the block output,
                            residual add included -- mamba1 DEVIATION 735)

    `params` is, in this exact order: 0 B, 1 L, 2 d_model. Profile
    constants (d_state 16, d_conv 4, expand 2, dt_rank ceil(dm/16)) are
    NOT parameters: changing one is a v2 profile, not a knob
    (FEATURE_PARITY.md section 1's SHIP LATER rows).

    Returns 0. Non-finite inputs/weights, B/L <= 0 and shape
    disagreements are refused BY NAME in Mojo, downstream (DEVIATION
    793's split)."""
    var a = _mamba1_addrs(addrs, String("mamba1_forward"))
    if len(params) != 3:
        raise Error(
            "mamba1_forward: params must contain 3 values (B, L, d_model),"
            " got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    with GILReleased(Python()):
        _mamba1_run(a, b, l, dm)
    return PythonObject(0)


def mamba1_decode_step_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One decode token: `Mamba.step`'s semantics, the profile's
    spelling -- the block forward at L = 1 with the state carried, NO
    arithmetic of its own (`mamba_step`'s rule, mamba1 DEVIATIONS
    721/732/733: upstream's step arm rounds differently and is the
    lane's required-RED sabotage, so there is deliberately no second
    entry path to drift).

    `addrs`: the same FOURTEEN as `mamba1_forward` with L = 1 shapes
    (x and y_out are B * d_model). `params`: 0 B, 1 d_model."""
    var a = _mamba1_addrs(addrs, String("mamba1_decode_step"))
    if len(params) != 2:
        raise Error(
            "mamba1_decode_step: params must contain 2 values (B, d_model),"
            " got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    with GILReleased(Python()):
        _mamba1_run(a, b, 1, dm)
    return PythonObject(0)


# ===========================================================================
# Mamba-2: profile mojolearn.identical.mamba2.fp32.v1
# ===========================================================================


def _mamba2_run(
    a: List[Int],
    b: Int,
    l: Int,
    dm: Int,
    q0: Int,
    dt_lo: Float32,
    dt_hi: Float32,
) raises -> Int:
    """The GIL-free half of the two Mamba-2 entry points. Returns the
    post-call `buf_len` (DEVIATION 792: the one integer piece of the
    three-piece state)."""
    # Mamba2Dims.of REFUSES a d_model that is not a multiple of 32, by
    # name, with the profile rule -- reached BEFORE any buffer size below
    # is computed from it, which is why nothing here pre-judges dm.
    var dims = Mamba2Dims.of(dm)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    if q0 < 0 or q0 >= M2_CHUNK_SIZE:
        raise Error(
            String("mamba2: buf_len must be in [0, ")
            + String(M2_CHUNK_SIZE)
            + "), got "
            + String(q0)
            + "; the open-chunk buffer holds at most CHUNK_SIZE - 1 rows"
            " between calls (contract section 5), so the two sides of this"
            " boundary disagree about the state"
        )

    var w = Mamba2Weights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], dip * dm)
    w.conv_w = _read_f32(a[3], cd * M2_D_CONV)
    w.conv_b = _read_f32(a[4], cd)
    w.dt_bias = _read_f32(a[5], nh)
    w.a_log = _read_f32(a[6], nh)
    w.d_skip = _read_f32(a[7], nh)
    w.gnorm_w = _read_f32(a[8], di)
    w.w_out = _read_f32(a[9], dm * di)

    var h_n = b * nh * M2_HEADDIM * M2_D_STATE

    var ctx = DeviceContext()
    var dw = Mamba2DeviceWeights(ctx, w)
    # The caller's three-piece state over the fresh zeros. A nonzero h
    # with q0 == 0 IS the initial_states path (module header, DEVIATION
    # 792).
    var dstate = Mamba2DeviceState(ctx, b, dims)
    dstate.conv_win = mamba_upload(ctx, _read_f32(a[10], b * cd * M2_D_CONV))
    dstate.h = mamba_upload(ctx, _read_f32(a[11], h_n))
    dstate.buf_xbc = mamba_upload(ctx, _read_f32(a[12], b * M2_CHUNK_SIZE * cd))
    dstate.buf_dtraw = mamba_upload(
        ctx, _read_f32(a[13], b * M2_CHUNK_SIZE * nh)
    )
    dstate.buf_len = q0
    var dstages = Mamba2DeviceStages(ctx, b, l, q0, dims)
    var dx = mamba_upload(ctx, _read_f32(a[0], b * l * dm))

    var trace = IdentityTrace.disabled()
    mamba2_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, dt_lo, dt_hi, trace, String("py")
    )

    _write_f32(a[14], mamba_download(ctx, dstages.residual_out, b * l * dm))
    # h_last: the REPORT stage (S17 after the final PADDED chunk), NOT
    # the resumption state -- contract section 5's distinction, kept.
    _write_f32(a[15], mamba_download(ctx, dstages.h_last, h_n))
    _write_f32(a[10], mamba_download(ctx, dstate.conv_win, b * cd * M2_D_CONV))
    _write_f32(a[11], mamba_download(ctx, dstate.h, h_n))
    _write_f32(a[12], mamba_download(ctx, dstate.buf_xbc, b * M2_CHUNK_SIZE * cd))
    _write_f32(
        a[13], mamba_download(ctx, dstate.buf_dtraw, b * M2_CHUNK_SIZE * nh)
    )
    var out_len = dstate.buf_len
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    return out_len


def _mamba2_addrs(addrs: PythonObject, what: String) raises -> List[Int]:
    if len(addrs) != 16:
        raise Error(
            what
            + ": addrs must contain 16 addresses (x, norm.weight,"
            " in_proj.weight, conv1d.weight, conv1d.bias, dt_bias, A_log,"
            " D, gated norm.weight, out_proj.weight, conv_window, h,"
            " buffer_xbc, buffer_dtraw, y_out, h_last_out), got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(16):
        a.append(Int(py=addrs[i]))
    return a^


def mamba2_forward_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One Mamba-2 block call: prefill from the state the caller hands
    in -- fresh (all zeros, buf_len 0), resumed (a carried three-piece
    state: chunked-prefill continuation), or with `initial_states` (a
    nonzero h at buf_len 0). Returns the post-call `buf_len`.

    `addrs` is the SIXTEEN buffer addresses, in this exact order
    (mirrored in `python/mojolearn/_mamba_impl.py`; H = d_inner/64 =
    d_model/32, CD = d_inner + 256):

        0  x                 B * L * d_model float32, read
        1  norm.weight       d_model, read (the BLOCK norm; the corpus
                              calls it block_norm.weight)
        2  in_proj.weight    (2*d_inner + 256 + H) * d_model, read
                              (column order z | xBC | dt, mamba2.py:211-215)
        3  conv1d.weight     CD * 4, read (their [CD,1,4], the 1 dropped)
        4  conv1d.bias       CD, read
        5  dt_bias           H, read
        6  A_log             H, read (PER HEAD -- not per state)
        7  D                 H, read (D_has_hdim False)
        8  norm.weight       d_inner, read (the GATED norm, RMSNormGated)
        9  out_proj.weight   d_model * d_inner, read
       10  conv_window       B * CD * 4, READ AND WRITTEN (state piece 1)
       11  h                 B * H * 64 * 128, READ AND WRITTEN (state
                              piece 2, the chunk-BOUNDARY state)
       12  buffer_xbc        B * 256 * CD, READ AND WRITTEN (state piece
                              3a, the open chunk's post-conv/post-SiLU rows)
       13  buffer_dtraw      B * 256 * H, READ AND WRITTEN (state piece 3b,
                              raw in_proj dt rows)
       14  y_out             B * L * d_model, WRITTEN (the block output,
                              residual add included)
       15  h_last_out        B * H * 64 * 128, WRITTEN (the REPORT stage;
                              NOT the resumption state -- contract section 5)

    `params` is, in this exact order:

        0  B
        1  L
        2  d_model     must be a multiple of 32; refused otherwise BY
                        NAME in Mojo (Mamba2Dims.of)
        3  buf_len     valid rows of the open-chunk buffer, 0 for a
                        fresh or chunk-aligned sequence
        4  dt_lo       the dt_limit clamp's lower bound (runtime INPUT,
        5  dt_hi        default (0, +inf); seam S9, DEVIATION 788 --
                        crosses UNJUDGED, the clamp defines its own
                        NaN/zero-sign behavior by construction)

    Profile constants (d_state 128, d_conv 4, expand 2, headdim 64,
    ngroups 1, CHUNK_SIZE 256 -- PART OF THE ARITHMETIC, DEVIATION 783)
    are NOT parameters; changing one is a v2."""
    var a = _mamba2_addrs(addrs, String("mamba2_forward"))
    if len(params) != 6:
        raise Error(
            "mamba2_forward: params must contain 6 values (B, L, d_model,"
            " buf_len, dt_lo, dt_hi), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var q0 = Int(py=params[3])
    var dt_lo = Float32(Float64(py=params[4]))
    var dt_hi = Float32(Float64(py=params[5]))
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba2_run(a, b, l, dm, q0, dt_lo, dt_hi)
    return PythonObject(out_len)


def mamba2_decode_step_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One decode token: `Mamba2.step`'s semantics, the profile's
    spelling -- PREFILL RESUMPTION at L = 1 (DEVIATION 786), through the
    same `mamba2_block_forward` the forward entry calls, with NO
    arithmetic of its own (`mamba2_step`'s rule; the upstream per-token
    recurrence rounds differently and is the required-RED
    STEP_UPSTREAM_RECURRENCE arm, never a mode).

    `addrs`: the same SIXTEEN as `mamba2_forward` with L = 1 shapes
    (x and y_out are B * d_model). `params`: 0 B, 1 d_model, 2 buf_len,
    3 dt_lo, 4 dt_hi. Returns the post-call buf_len."""
    var a = _mamba2_addrs(addrs, String("mamba2_decode_step"))
    if len(params) != 5:
        raise Error(
            "mamba2_decode_step: params must contain 5 values (B, d_model,"
            " buf_len, dt_lo, dt_hi), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var q0 = Int(py=params[2])
    var dt_lo = Float32(Float64(py=params[3]))
    var dt_hi = Float32(Float64(py=params[4]))
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba2_run(a, b, 1, dm, q0, dt_lo, dt_hi)
    return PythonObject(out_len)


# ===========================================================================
# Mamba-3: profile mojolearn.identical.mamba3.siso.fp32.v1
# ===========================================================================


def _mamba3_run(
    a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int
) raises -> Int:
    """The GIL-free half of the two Mamba-3 entry points. Returns the
    post-call `buf_len` (DEVIATION 794: `pending` needs no return slot
    because a shipped binding always consumes it)."""
    # Mamba3Dims.of REFUSES a d_model that is not a multiple of 32, by
    # name -- reached BEFORE any buffer size below is computed from it.
    var dims = Mamba3Dims.of(dm)
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    if q0 < 0 or q0 > M3_CHUNK_SIZE:
        raise Error(
            String("mamba3: buf_len must be in [0, ")
            + String(M3_CHUNK_SIZE)
            + "] (INCLUSIVE -- the buffer never empties, DEVIATION"
            " 832(i): r in [1, Q] after every call, 0 only before the"
            " first token), got "
            + String(q0)
            + "; the two sides of this boundary disagree about the state"
        )
    if pend != 0 and pend != 1:
        raise Error(
            "mamba3: pending must be 0 or 1, got "
            + String(pend)
            + "; the two sides of this boundary disagree about the state"
        )

    var w = Mamba3Weights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], dip * dm)
    w.dt_bias = _read_f32(a[3], nh)
    w.bnorm_w = _read_f32(a[4], M3_D_STATE)
    w.cnorm_w = _read_f32(a[5], M3_D_STATE)
    w.b_bias = _read_f32(a[6], nh * M3_D_STATE)
    w.c_bias = _read_f32(a[7], nh * M3_D_STATE)
    w.d_skip = _read_f32(a[8], nh)
    w.w_out = _read_f32(a[9], dm * di)

    var theta_n = b * nh * M3_NUM_ROPE_ANGLES
    var h_n = b * nh * M3_HEADDIM * M3_D_STATE
    var qrow_n = b * M3_CHUNK_SIZE * nh
    var k_n = b * nh * M3_D_STATE
    var v_n = b * nh * M3_HEADDIM

    var ctx = DeviceContext()
    var dw = Mamba3DeviceWeights(ctx, w)
    # The caller's ten-piece state over the fresh zeros (DEVIATION 794).
    var dstate = Mamba3DeviceState(ctx, b, dims)
    dstate.buf_qrot = mamba_upload(
        ctx, _read_f32(a[12], qrow_n * M3_D_STATE)
    )
    dstate.buf_krot = mamba_upload(
        ctx, _read_f32(a[13], qrow_n * M3_D_STATE)
    )
    dstate.buf_v = mamba_upload(ctx, _read_f32(a[14], qrow_n * M3_HEADDIM))
    dstate.buf_dt = mamba_upload(ctx, _read_f32(a[15], qrow_n))
    dstate.buf_sig = mamba_upload(ctx, _read_f32(a[16], qrow_n))
    dstate.buf_adt = mamba_upload(ctx, _read_f32(a[17], qrow_n))
    dstate.buf_len = q0
    if pend == 1:
        # THROUGH the lane's own helper, AFTER buf_len is set, so its
        # fresh-state refusal (buf_len must be 0) fires from Python by
        # the lane's own words (DEVIATION 794).
        dstate.set_input_states(
            ctx,
            _read_f32(a[10], theta_n),
            _read_f32(a[11], h_n),
            _read_f32(a[18], k_n),
            _read_f32(a[19], v_n),
        )
    else:
        dstate.theta = mamba_upload(ctx, _read_f32(a[10], theta_n))
        dstate.h = mamba_upload(ctx, _read_f32(a[11], h_n))
        # Idle outside a pending continuation, uploaded anyway so the
        # caller's bytes round-trip unchanged (DEVIATION 792's rule).
        dstate.pend_k = mamba_upload(ctx, _read_f32(a[18], k_n))
        dstate.pend_v = mamba_upload(ctx, _read_f32(a[19], v_n))
    var dstages = Mamba3DeviceStages(ctx, b, l, q0, dims)
    var dx = mamba_upload(ctx, _read_f32(a[0], b * l * dm))

    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, trace, String("py")
    )

    _write_f32(a[20], mamba_download(ctx, dstages.residual_out, b * l * dm))
    # The FOUR reports (contract section 7's ssd.* stages), NOT the
    # resumption state -- DEVIATION 794's last clause.
    _write_f32(a[21], mamba_download(ctx, dstages.h_last, h_n))
    _write_f32(a[22], mamba_download(ctx, dstages.k_last, k_n))
    _write_f32(a[23], mamba_download(ctx, dstages.v_last, v_n))
    _write_f32(a[24], mamba_download(ctx, dstages.theta_last, theta_n))
    _write_f32(a[10], mamba_download(ctx, dstate.theta, theta_n))
    _write_f32(a[11], mamba_download(ctx, dstate.h, h_n))
    _write_f32(
        a[12], mamba_download(ctx, dstate.buf_qrot, qrow_n * M3_D_STATE)
    )
    _write_f32(
        a[13], mamba_download(ctx, dstate.buf_krot, qrow_n * M3_D_STATE)
    )
    _write_f32(a[14], mamba_download(ctx, dstate.buf_v, qrow_n * M3_HEADDIM))
    _write_f32(a[15], mamba_download(ctx, dstate.buf_dt, qrow_n))
    _write_f32(a[16], mamba_download(ctx, dstate.buf_sig, qrow_n))
    _write_f32(a[17], mamba_download(ctx, dstate.buf_adt, qrow_n))
    _write_f32(a[18], mamba_download(ctx, dstate.pend_k, k_n))
    _write_f32(a[19], mamba_download(ctx, dstate.pend_v, v_n))
    var out_len = dstate.buf_len
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    return out_len


def _mamba3_addrs(addrs: PythonObject, what: String) raises -> List[Int]:
    if len(addrs) != 25:
        raise Error(
            what
            + ": addrs must contain 25 addresses (x, block norm.weight,"
            " in_proj.weight, dt_bias, B_norm.weight, C_norm.weight,"
            " B_bias, C_bias, D, out_proj.weight, theta, h, buffer_qrot,"
            " buffer_krot, buffer_v, buffer_dt, buffer_sig, buffer_adt,"
            " pending_k, pending_v, y_out, h_last_out, k_last_out,"
            " v_last_out, theta_last_out), got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(25):
        a.append(Int(py=addrs[i]))
    return a^


def mamba3_forward_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One Mamba-3 (SISO) block call: prefill from the state the caller
    hands in -- fresh (all zeros, buf_len 0, pending 0), resumed (a
    carried ten-piece state: chunked-prefill continuation), or an
    upstream Input_States continuation (pending 1 on a FRESH state; the
    lane's own set_input_states refuses anything else by name). Returns
    the post-call `buf_len`.

    `addrs` is the TWENTY-FIVE buffer addresses, in this exact order
    (mirrored in `python/mojolearn/_mamba_impl.py`; H = d_inner/64 =
    d_model/32 heads, N = 128, P = 64, R = 32, Q = 64, d_in_proj =
    2*d_inner + 256 + 3*H + 32):

        0  x                 B * L * d_model float32, read
        1  norm.weight       d_model, read (the BLOCK norm; the fixture
                              calls it block_norm.weight)
        2  in_proj.weight    d_in_proj * d_model, read (column order
                              z | x | B | C | dd_dt | dd_A | trap |
                              angle, mamba3.py:106-107)
        3  dt_bias           H, read
        4  B_norm.weight     N, read (the S21 B RMSNorm, eps 1e-5)
        5  C_norm.weight     N, read (the S21 C RMSNorm)
        6  B_bias            H * N, read (per-head, AFTER the norm)
        7  C_bias            H * N, read
        8  D                 H, read
        9  out_proj.weight   d_model * d_inner, read
       10  theta             B * H * R, READ AND WRITTEN (state: the
                              serial rotary angles, always in [0, 2pi))
       11  h                 B * H * P * N, READ AND WRITTEN (state: the
                              SEALED chunk-boundary h, DEVIATION 832)
       12  buffer_qrot       B * Q * H * N, READ AND WRITTEN (state:
                              rotated-UNSCALED q rows of the last
                              working chunk)
       13  buffer_krot       B * Q * H * N, READ AND WRITTEN (state)
       14  buffer_v          B * Q * H * P, READ AND WRITTEN (state: raw
                              v = the x split, there is NO conv)
       15  buffer_dt         B * Q * H, READ AND WRITTEN (state)
       16  buffer_sig        B * Q * H, READ AND WRITTEN (state:
                              sigma(trap))
       17  buffer_adt        B * Q * H, READ AND WRITTEN (state)
       18  pending_k         B * H * N, READ AND WRITTEN (the
                              Input_States pair, live only under
                              pending = 1; round-trips otherwise)
       19  pending_v         B * H * P, READ AND WRITTEN
       20  y_out             B * L * d_model, WRITTEN (the block output,
                              residual add included)
       21  h_last_out        B * H * P * N, WRITTEN (REPORT stage)
       22  k_last_out        B * H * N, WRITTEN (REPORT stage)
       23  v_last_out        B * H * P, WRITTEN (REPORT stage)
       24  theta_last_out    B * H * R, WRITTEN (REPORT stage)

    `params` is, in this exact order:

        0  B
        1  L
        2  d_model     must be a multiple of 32; refused otherwise BY
                        NAME in Mojo (Mamba3Dims.of)
        3  buf_len     valid rows of the last-working-chunk buffer, in
                        [0, 64] INCLUSIVE (DEVIATION 794); 0 for a
                        fresh sequence ONLY
        4  pending     0 or 1; 1 marks theta/h/pending_k/pending_v as
                        an upstream Input_States continuation, consumed
                        by this call (DEVIATION 794)

    There is NO dt clamp parameter: mamba3 S6 is bias -> softplus and
    nothing else (contract section 0; do not import mamba2's dt_lo /
    dt_hi here). Profile constants (d_state 128, headdim 64, ngroups 1,
    rope_fraction 0.5 = 32 angles, A_floor 1e-4, CHUNK_SIZE 64 -- PART
    OF THE ARITHMETIC, DEVIATIONS 827/783) are NOT parameters; changing
    one is a v2."""
    var a = _mamba3_addrs(addrs, String("mamba3_forward"))
    if len(params) != 5:
        raise Error(
            "mamba3_forward: params must contain 5 values (B, L, d_model,"
            " buf_len, pending), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var q0 = Int(py=params[3])
    var pend = Int(py=params[4])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, l, dm, q0, pend)
    return PythonObject(out_len)


def mamba3_decode_step_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One decode token: `Mamba3.step`'s semantics, the profile's
    spelling -- PREFILL RESUMPTION at L = 1 (DEVIATION 831), through the
    same `mamba3_block_forward` the forward entry calls, with NO
    arithmetic of its own (`mamba3_step`'s rule; the upstream per-token
    recurrence rounds differently and is the required-RED
    STEP_UPSTREAM_RECURRENCE arm, never a mode).

    `addrs`: the same TWENTY-FIVE as `mamba3_forward` with L = 1 shapes
    (x and y_out are B * d_model). `params`: 0 B, 1 d_model, 2 buf_len,
    3 pending. Returns the post-call buf_len."""
    var a = _mamba3_addrs(addrs, String("mamba3_decode_step"))
    if len(params) != 4:
        raise Error(
            "mamba3_decode_step: params must contain 4 values (B,"
            " d_model, buf_len, pending), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var q0 = Int(py=params[2])
    var pend = Int(py=params[3])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, 1, dm, q0, pend)
    return PythonObject(out_len)


@export
def PyInit__mojolearn_mamba() abi("C") -> PythonObject:
    # DEVIATION 793's last clause: a sabotage arm exists to be run by a
    # gate and to FAIL; a Python surface that quietly served one would
    # be a wrong answer wearing a green label. Refuse to exist instead.
    comptime if BLOCK_ANY_SABOTAGE or BLOCK2_ANY_SABOTAGE or BLOCK3_ANY_SABOTAGE:
        abort(
            String(
                "_mojolearn_mamba: refusing to initialize -- a sabotage"
                " arm is compiled into this binary. Sabotage defines are"
                " for the lane gates (mamba/checks/), never for a shipped"
                " binding; rebuild with bash bindings/build_mamba.sh and"
                " no MOJOLEARN_MAMBA_SABOTAGE_*,"
                " MOJOLEARN_MAMBA2_SABOTAGE_* or"
                " MOJOLEARN_MAMBA3_SABOTAGE_* define."
            )
        )
    try:
        var m = PythonModuleBuilder("_mojolearn_mamba")
        m.def_function[mamba_vendor_binding]("mamba_vendor")
        m.def_function[mamba_numeric_mode_binding]("mamba_numeric_mode")
        m.def_function[mamba1_forward_binding]("mamba1_forward")
        m.def_function[mamba1_decode_step_binding]("mamba1_decode_step")
        m.def_function[mamba2_forward_binding]("mamba2_forward")
        m.def_function[mamba2_decode_step_binding]("mamba2_decode_step")
        m.def_function[mamba3_forward_binding]("mamba3_forward")
        m.def_function[mamba3_decode_step_binding]("mamba3_decode_step")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_mamba: ", e))
