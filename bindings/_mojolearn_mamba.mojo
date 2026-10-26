# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the Mamba-1, Mamba-2 and Mamba-3 block lanes.

A FOURTEENTH extension module, and a separate one on purpose, for the
reason `bindings/_mojolearn_estimators.mojo`'s header gives: an
independently changing binding must not become a merge point. This file
closes the "PyPI surface: NONE EXISTS" row of `archive/evidence/mamba/FEATURE_PARITY.md`'s
consumer table -- the certified Mamba-1 block (profile
`mojolearn.identical.mamba1.fp32.v1`, three-vendor card) and the Mamba-2
block (profile `mojolearn.identical.mamba2.fp32.v1`, Apple-gated,
cross-vendor legs owed) exported no Python symbol at all before it. The
Mamba-3 entries (profile `mojolearn.identical.mamba3.siso.fp32.v1`,
Apple-gated at the kernel level 2026-09-01, cross-vendor legs owed)
joined later the same day, under the same ABI.

THE BINDING CALLS THE CERTIFIED ENTRY POINTS AND NOTHING ELSE. Every
forward below goes through `mamba_block_forward`
(`mamba/impl/modeling/modeling_mamba.mojo`),
`mamba2_block_forward` (`mamba/impl/modules/mamba2.mojo`) or
`mamba3_block_forward` (`mamba/impl/modules/mamba3.mojo`) --
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
this side of the boundary (`archive/evidence/mamba/FEATURE_PARITY.md` section 3's SHIP
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
k/v pair (with theta and h) is a reference Input_States continuation.
`pending` crosses IN only: in a shipped binding it is ALWAYS False on
return, because the unarmed core consumes it (`mamba3_block_forward`
clears it after S22) and an armed build aborts at PyInit below -- so
the return value stays `buf_len` alone, mamba2's shape, and the wrapper
reassigns `state.pending = False` on that guarantee. The pending path
goes through the lane's OWN `Mamba3DeviceState.set_input_states`, so
its fresh-state refusal (Input_States only has reference meaning at
buf_len 0) stays reachable from Python instead of being respelled here.
FOUR report buffers are written on every Mamba-3 call -- h_last,
k_last, v_last, theta_last, the contract's REPORT stages, distinct from
the resumption state exactly as mamba2's h_last is.

THE TIER SEMANTICS ARE THE BUILD'S, NOT THIS FILE'S. The numeric
mode (fast / identical) is a compile-time define
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

# DEVIATION 2486: shared byte-preserving host copies.
from core.hostptr import f32_ptr, read_f32, copy_f32
from std.memory import bitcast, memcpy
from std.time import perf_counter_ns
from std.sys.compile import is_defined, get_defined_int
# TOMBSTONE: MOJOLEARN_IDN_M1_SCAN=1 (NN34 affine_prefix: profile/state/branches in this file; LOSER avg 2.30x mamba1-forward) deleted 2026-10-11 by 291629888; code recoverable at 291629888^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_SCAN_1.patch
from mamba.impl.ops.mamba3_siso import m3_phase_tick
from mamba.impl.modules.mamba3_transfer import m3_upload, m3_download
from std.os import abort, getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from core.device_scan import device_first_nonfinite
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
from checks.vendor import COMPILED_VENDOR

from mamba.checks.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
)
from mamba.impl.modeling.modeling_mamba import (
    BLOCK_ANY_SABOTAGE,
    MAMBA_GUARD,
    mamba_block_forward,
    mamba_copy_in,
    mamba_device_alloc,
    mamba_download,
    mamba_upload,
    mamba_zeros,
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
)
from mamba.impl.modules.mamba_simple import mamba_step
#: lane afn-mamba (2026-10-03): under FAST + Apple + `-D MOJOLEARN_AFN_MAMBA_ARENA`
#: every block call's device buffers are views of ONE arena and the caller's
#: arrays are copied in with no per-buffer wait (`_mamba{1,2,3}_run_arena`).
from mamba.impl.modules.afn_defines import AFN_MAMBA_ARENA, IDN_MAMBA_ALLOC_NOWAIT
from mamba.impl.modules.afn_defines import IDN_MAMBA3_REPORTS_ON_REQUEST
from mamba.impl.modules.afn_defines import IDN_M3_SESSION_STAGE_REUSE
from mamba.impl.modules.afn_arena import MambaArena
from mamba.impl.modeling.modeling_mamba import mamba1_arena_floats
from mamba.impl.modules.mamba2 import mamba2_arena_floats
from mamba.impl.modules.mamba3 import mamba3_arena_floats
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
    Mamba2Weights,
)
from mamba.impl.modules.mamba2 import (
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
from mamba.impl.modules.mamba3 import (
    BLOCK3_ANY_SABOTAGE,
    Mamba3DeviceStages,
    Mamba3DeviceState,
    Mamba3DeviceWeights,
    mamba3_block_forward,
)


from autograd.kernels.mamba2_proj import ANY_BWD2_SABOTAGE
# TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA3_OFF (the hand Mamba-3 prefill backward walk; superseded by the tape) deleted 2026-10-19 by bc0a9a24e; code recoverable at ba37327dc^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA3_OFF.patch
from autograd.kernels.mamba3_proj import ANY_BWD3_SABOTAGE
# The Mamba-1 / Mamba-2 backward entries assemble the VJP on the tape
# (autograd/rules_mamba.mojo), which launches the hand kernels in the hand order.
# TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA_OFF (hand Mamba-1/2 prefill backward glue, superseded by the tape) deleted 2026-10-19 by 47582a7fb; code recoverable at 47582a7fb^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_OFF.patch
from autograd.rules_mamba import mamba1_tape_backward, mamba2_tape_backward
from autograd.rules_mamba3 import mamba3_tape_backward, mamba3_tape_backward_on
from autograd.kernels.mamba1_prefill import Mamba1PrefillGradients
from autograd.kernels.mamba2_prefill import Mamba2PrefillGradients
from autograd.kernels.mamba1_proj import ANY_BWD_SABOTAGE as M1_ANY_BWD_SABOTAGE
from autograd.kernels.mamba1 import ANY_BWD_BLOCK_SABOTAGE
from autograd.kernels.mamba1_scan import ANY_BWD_SCAN_SABOTAGE
from gemm.checks.gemm_backward import ANY_BWD_SABOTAGE as GEMM_ANY_BWD_SABOTAGE
from gemm.checks.gemm_identical import ANY_SABOTAGE as GEMM_ANY_SABOTAGE

from mamba.impl.mamba_shell_impl import (
    _f32_ptr,
    _read_f32,
    _write_f32,
    M3_DIRECT_TRANSFER,
    _m3_read_f32,
    _m3_write_f32,
    _m3_upload_addr,
    _m3_download_addr,
    _mamba1_run,
    _mamba1_run_arena,
    _mamba1_backward_run,
    Mamba1DecodeSession,
    _m1_require_session_open,
    _m1_session_open_run,
    _m1_session_step_run,
    _m1_session_export_run,
    _m1_session_load_run,
    _mamba2_run,
    _mamba2_run_arena,
    Mamba2DecodeSession,
    _m2_require_session_open,
    _m2_session_open_run,
    _m2_session_step_run,
    _m2_session_export_run,
    _m2_session_load_run,
    _m3_load_weights,
    _mamba3_run,
    _mamba3_run_owned,
    _mamba3_run_arena,
    Mamba3DecodeSession,
    _m3_require_session_open,
    _m3_session_copy_in,
    _m3_session_open_run,
    IDN_M3_DECODE_WINDOW_REUSE,
    _m3_session_step_run,
    _m3_session_export_run,
    NN40_OWNED_WEIGHTS,
    Mamba3PrefillSession,
    _m3_retain_stages,
    _m3_weight_lens,
    _m3_weights_same_bits,
    _m3_weights_host_copy,
    _m3_weights_recopy,
    _m3_prefill_weights,
    _m3_prefill_run,
    _m3_x_same_bits,
    _m3_prefill_backward_forward,
    _M3_BWD_NONFINITE_DY,
    _m3_prefill_backward_run,
    _m3_install_weights_run,
    _m3_export_weights_run,
    IDN_SAMBA_M3_FORWARD_TAPE,
    Mamba3ForwardTape,
    _m3_backward_tape_run,
    _mamba2_backward_run,
    _mamba3_backward_run,
)



def mamba_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL. The same shape as `gp_numeric_mode`,
    with the same reason: `_mamba_impl.py` reads it once and refuses to
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



def mamba1_profile_binding() -> PythonObject:
    return PythonObject(1)  # NN34 profile 2 (M1_SCAN=1) deleted 2026-10-11


def mamba1_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Zero-state synchronous IDENTICAL VJP. Addresses: x, ten weights in
    forward order, grad_output, grad_x, ten weight gradients in forward
    order (23 total). Scalars: B, L, d_model. No state/cache is accepted."""
    if len(addrs) != 23 or len(params) != 3:
        raise Error("mamba1 backward: expected 23 addresses and B, L, d_model")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba1 backward: B, L and d_model must be positive")
    var a = List[Int]()
    for i in range(23):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("mamba1 backward: null buffer address at slot " + String(i))
        a.append(address)
    with GILReleased(Python()):
        _mamba1_backward_run(a, b, l, dm)
    return PythonObject(0)


def _mamba1_addrs(addrs: PythonObject, what: String) raises -> List[Int]:
    comptime count = 14
    if len(addrs) != count:
        raise Error(
            what
            + ": addrs must contain 14 addresses (x, norm.weight,"
            " in_proj.weight, conv1d.weight, conv1d.bias, x_proj.weight,"
            " dt_proj.weight, dt_proj.bias, A_log, D, out_proj.weight,"
            " conv_window, h, y_out), got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(count):  # small-loop(count: buffer addresses of one block call): reads the pointer list, not data
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
    var position = 0
    with GILReleased(Python()):
        _mamba1_run(a, b, l, dm, False, position)
    return PythonObject(0)


def mamba1_decode_step_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One decode token: `Mamba.step`'s semantics, the profile's
    spelling -- the block forward at L = 1 with the state carried, NO
    arithmetic of its own (`mamba_step`'s rule, mamba1 DEVIATIONS
    721/732/733: the reference's step arm rounds differently and is the
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
    var position = 0
    with GILReleased(Python()):
        _mamba1_run(a, b, 1, dm, True, position)
    return PythonObject(0)


def mamba1_session_create_binding() raises -> PythonObject:
    """A closed session. Creation performs no GPU operation."""
    return PythonObject(alloc=Mamba1DecodeSession())


def mamba1_session_open_binding(
    session: PythonObject, addrs: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Open a closed session: `addrs` is TWELVE addresses, the ten weights
    in `mamba1_forward`'s order (slots 1 to 10 there), then conv_window and
    h; `params` is [B, d_model]. Everything is COPIED to the device."""
    var owner = session.downcast_value_ptr[Mamba1DecodeSession]()
    var count = 12
    if len(addrs) != count or len(params) != 2:
        raise Error("mamba1_session_open: expected 12 addresses and 2 scalars")
    if owner[].busy:
        raise Error("mamba1 session: busy")
    if owner[].is_open():
        raise Error("mamba1 session: already open; close it first")
    var a = List[Int]()
    a.append(0)
    for i in range(count):  # small-loop(count: session buffer addresses): reads the pointer list, not data
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("mamba1_session_open: null buffer address at slot " + String(i))
        a.append(address)
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    if b <= 0:
        raise Error("mamba1_session_open: B must be positive")
    var position = 0
    owner[].busy = True
    owner[].usable = False
    try:
        with GILReleased(Python()):
            _m1_session_open_run(owner[], a, b, dm, position)
    except error:
        owner[].busy = False
        owner[].release()
        raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(0)


def mamba1_session_step_binding(
    session: PythonObject, addrs: PythonObject,
) raises -> PythonObject:
    """One decode token on an open session: `addrs` = [x, y_out], both
    B * d_model float32. A failed call marks the session lost."""
    var owner = session.downcast_value_ptr[Mamba1DecodeSession]()
    if len(addrs) != 2:
        raise Error("mamba1_session_step: expected 2 addresses")
    var px = Int(py=addrs[0])
    var py = Int(py=addrs[1])
    if px == 0 or py == 0:
        raise Error("mamba1_session_step: null buffer address")
    _m1_require_session_open(owner[])
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m1_session_step_run(owner[], px, py)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(0)


def mamba1_session_export_state_binding(
    session: PythonObject, addrs: PythonObject,
) raises -> PythonObject:
    """Copy the resident state into the caller's buffers: `addrs` =
    [conv_window, h]. The session stays open."""
    var owner = session.downcast_value_ptr[Mamba1DecodeSession]()
    if len(addrs) != 2:
        raise Error("mamba1_session_export_state: expected 2 addresses")
    var pw = Int(py=addrs[0])
    var ph = Int(py=addrs[1])
    if pw == 0 or ph == 0:
        raise Error("mamba1_session_export_state: null buffer address")
    _m1_require_session_open(owner[])
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m1_session_export_run(owner[], pw, ph)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(0)


def mamba1_session_load_state_binding(
    session: PythonObject, addrs: PythonObject,
) raises -> PythonObject:
    """Replace the resident state with the caller's bytes: `addrs` =
    [conv_window, h]. The explicit refresh for the state half."""
    var owner = session.downcast_value_ptr[Mamba1DecodeSession]()
    if len(addrs) != 2:
        raise Error("mamba1_session_load_state: expected 2 addresses")
    var pw = Int(py=addrs[0])
    var ph = Int(py=addrs[1])
    if pw == 0 or ph == 0:
        raise Error("mamba1_session_load_state: null buffer address")
    _m1_require_session_open(owner[])
    var position = 0
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m1_session_load_run(owner[], pw, ph)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(0)


# TOMBSTONE: mamba1_session_info, mamba1_forward_fresh, mamba2_forward_fresh (dead native entries: no live caller) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_mamba.patch


def mamba1_session_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba1DecodeSession]()
    owner[].close()
    return PythonObject(0)


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
    for i in range(16):  # small-loop(addrs: fixed-length buffer address list)
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
    arithmetic of its own (`mamba2_step`'s rule; the reference per-token
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


def mamba2_session_create_binding() raises -> PythonObject:
    return PythonObject(alloc=Mamba2DecodeSession())


def mamba2_session_open_binding(session: PythonObject, addrs: PythonObject,
                                params: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba2DecodeSession]()
    if len(addrs) != 13 or len(params) != 5:
        raise Error("mamba2_session_open: expected 13 addresses and 5 scalars")
    if Int(py=params[0]) <= 0 or Int(py=params[1]) <= 0:
        raise Error("mamba2_session_open: B and d_model must be positive")
    if owner[].busy or owner[].is_open():
        raise Error("mamba2_session_open: session is busy or already open")
    var a = List[Int]()
    for i in range(13):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0:
            raise Error("mamba2_session_open: null buffer address")
        a.append(p)
    owner[].busy = True
    owner[].usable = False
    try:
        with GILReleased(Python()):
            _m2_session_open_run(owner[], a, Int(py=params[0]), Int(py=params[1]),
                Int(py=params[2]), Float32(Float64(py=params[3])), Float32(Float64(py=params[4])))
    except error:
        owner[].busy = False
        owner[].release()
        raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(0)


def mamba2_session_step_binding(session: PythonObject, addrs: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba2DecodeSession]()
    if len(addrs) != 3:
        raise Error("mamba2_session_step: expected x, y, h_last addresses")
    var px = Int(py=addrs[0]); var py = Int(py=addrs[1]); var ph = Int(py=addrs[2])
    if px == 0 or py == 0 or ph == 0:
        raise Error("mamba2_session_step: null buffer address")
    _m2_require_session_open(owner[])
    owner[].busy = True
    var q = 0
    try:
        with GILReleased(Python()):
            q = _m2_session_step_run(owner[], px, py, ph)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    return PythonObject(q)


def mamba2_session_export_state_binding(session: PythonObject, addrs: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba2DecodeSession]()
    if len(addrs) != 4:
        raise Error("mamba2_session_export_state: expected 4 addresses")
    var a = List[Int]()
    for i in range(4):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0: raise Error("mamba2_session_export_state: null buffer address")
        a.append(p)
    _m2_require_session_open(owner[])
    owner[].busy = True
    var q = 0
    try:
        with GILReleased(Python()): q = _m2_session_export_run(owner[], a)
    except error:
        owner[].busy = False; owner[].usable = False; raise error
    owner[].busy = False
    return PythonObject(q)


def mamba2_session_load_state_binding(session: PythonObject, addrs: PythonObject,
                                      params: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba2DecodeSession]()
    if len(addrs) != 4 or len(params) != 1:
        raise Error("mamba2_session_load_state: expected 4 addresses and buf_len")
    var a = List[Int]()
    for i in range(4):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0: raise Error("mamba2_session_load_state: null buffer address")
        a.append(p)
    _m2_require_session_open(owner[])
    owner[].busy = True
    try:
        with GILReleased(Python()): _m2_session_load_run(owner[], a, Int(py=params[0]))
    except error:
        owner[].busy = False; owner[].usable = False; raise error
    owner[].busy = False
    return PythonObject(0)


def mamba2_session_close_binding(session: PythonObject) raises -> PythonObject:
    session.downcast_value_ptr[Mamba2DecodeSession]()[].close()
    return PythonObject(0)


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
    for i in range(25):  # small-loop(addrs: fixed-length buffer address list)
        a.append(Int(py=addrs[i]))
    return a^


def mamba3_forward_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One Mamba-3 (SISO) block call: prefill from the state the caller
    hands in -- fresh (all zeros, buf_len 0, pending 0), resumed (a
    carried ten-piece state: chunked-prefill continuation), or a
    reference Input_States continuation (pending 1 on a FRESH state; the
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
                        a reference Input_States continuation, consumed
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


def mamba3_forward_fresh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Fresh zero-state prefill whose resumption state has no caller owner.

    The 15 pointers are x, nine weights, y, and four public reports. The
    ordinary device state constructor supplies exact zeros; the same block
    and ordered refusals run. No discarded state crosses the host boundary.
    """
    if len(addrs) != 15 or len(params) != 3:
        raise Error("mamba3_forward_fresh: expected 15 addresses and 3 parameters (B, L, d_model)")
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        a.append(Int(py=addrs[i]))
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        a.append(0)
    for i in range(10, 15):  # small-loop(addrs: fixed-length buffer address list)
        a.append(Int(py=addrs[i]))
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b < 1 or l < 1:
        raise Error("mamba3_forward_fresh: B and L must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run[True](a, b, l, dm, 0, 0)
    return PythonObject(out_len)


def mamba3_decode_step_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One decode token: `Mamba3.step`'s semantics, the profile's
    spelling -- PREFILL RESUMPTION at L = 1 (DEVIATION 831), through the
    same `mamba3_block_forward` the forward entry calls, with NO
    arithmetic of its own (`mamba3_step`'s rule; the reference per-token
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


def mamba3_state_owned_binding(session: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`mamba3_forward` (step 0) or `mamba3_decode_step` (step 1) on the
    installed owned snapshot of prefill `session` (lane migrate-neural-2):
    the weights are read on the device, never from the caller's arrays.
    `addrs` is the same twenty-five as `mamba3_forward` (the nine weight
    slots are not read; 0 is accepted); `params` = [B, L, d_model, buf_len,
    pending, generation, step] (L = 1 for a step). Returns the post-call
    buf_len."""
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    var a = _mamba3_addrs(addrs, String("mamba3_state_owned"))
    if len(params) != 7:
        raise Error("mamba3_state_owned: params must contain 7 values (B, L, d_model, buf_len, pending, generation, step)")
    if owner[].busy or not owner[].usable or not owner[].owned_weights:
        raise Error("mamba3 owned forward: session is not ready")
    if Int(py=params[5]) != owner[].weight_generation:
        raise Error("mamba3 owned forward: stale weight generation")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var q0 = Int(py=params[3])
    var pend = Int(py=params[4])
    var step = Int(py=params[6])
    if step != 0 and step != 1:
        raise Error("mamba3_state_owned: step must be 0 or 1")
    if step == 1 and l != 1:
        raise Error("mamba3_state_owned: a decode step is one token (L = 1)")
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _mamba3_run_owned(owner[], a, b, l, dm, q0, pend)
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(out_len)


def mamba3_owned_weights_enabled_binding() raises -> PythonObject:
    return PythonObject(NN40_OWNED_WEIGHTS)


def mamba3_prefill_session_install_weights_binding(session: PythonObject, addrs: PythonObject, d_model: PythonObject) raises -> PythonObject:
    """Copy nine tensors into an immutable session-owned generation.
    Complete all uploads before replacing the previous snapshot. External
    arrays may mutate afterwards: owned calls no longer consult them."""
    comptime if not NN40_OWNED_WEIGHTS:
        raise Error("mamba3 owned weights: NN40 arm is disabled")
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    if owner[].busy or not owner[].usable:
        raise Error("mamba3 owned weights: session is busy or unusable")
    if len(addrs) != 9:
        raise Error("mamba3 owned weights: expected nine weight addresses")
    var native_d_model = Int(py=d_model)
    var dims = Mamba3Dims.of(native_d_model)
    var a = List[Int]()
    a.append(0)
    for i in range(9):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("mamba3 owned weights: null weight address")
        a.append(address)
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_install_weights_run(owner[], a, dims, native_d_model)
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(owner[].weight_generation)


def mamba3_prefill_session_forward_owned_binding(session: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """[x,y,h_last,k_last,v_last,theta_last], [B,L,d_model,generation].
    A null report follows the existing reports-optional contract."""
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    if len(addrs) != 6 or len(params) != 4:
        raise Error("mamba3 owned forward: expected six addresses and four scalars")
    if owner[].busy or not owner[].usable or not owner[].owned_weights:
        raise Error("mamba3 owned forward: session is not ready")
    var generation = Int(py=params[3])
    if generation != owner[].weight_generation:
        raise Error("mamba3 owned forward: stale weight generation")
    var b = Int(py=params[0]); var l = Int(py=params[1]); var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm != owner[].dm:
        raise Error("mamba3 owned forward: invalid shape or configuration")
    var a = List[Int](length=25, fill=0)
    a[0] = Int(py=addrs[0])
    a[20] = Int(py=addrs[1])
    if a[0] == 0 or a[20] == 0:
        raise Error("mamba3 owned forward: null input/output")
    for i in range(4):  # small-loop(addrs: fixed-length buffer address list)
        a[21 + i] = Int(py=addrs[2 + i])
        comptime if not IDN_MAMBA3_REPORTS_ON_REQUEST:
            if a[21 + i] == 0:
                raise Error("mamba3 owned forward: report address required")
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _m3_prefill_run(owner[], a, b, l, dm, True)
    except error:
        owner[].busy = False
        owner[].usable = False
        owner[].release()
        raise error
    owner[].busy = False
    return PythonObject(out_len)


def mamba3_prefill_session_backward_owned_binding(session: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """[x,dy,dx,nine dWeights], [B,L,d_model,generation]. Explicit stale
    generation refusal prevents differentiating a different snapshot."""
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    if len(addrs) != 12 or len(params) != 4:
        raise Error("mamba3 owned backward: expected twelve addresses and four scalars")
    if owner[].busy or not owner[].usable or not owner[].owned_weights:
        raise Error("mamba3 owned backward: session is not ready")
    if Int(py=params[3]) != owner[].weight_generation:
        raise Error("mamba3 owned backward: stale weight generation")
    var b = Int(py=params[0]); var l = Int(py=params[1]); var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm != owner[].dm:
        raise Error("mamba3 owned backward: invalid shape or configuration")
    var a = List[Int](length=21, fill=0)
    a[0] = Int(py=addrs[0])
    for i in range(1, 12):  # small-loop(addrs: fixed-length buffer address list)
        a[9 + i] = Int(py=addrs[i])
        if a[9 + i] == 0:
            raise Error("mamba3 owned backward: null buffer address")
    if a[0] == 0:
        raise Error("mamba3 owned backward: null input")
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_prefill_backward_run(owner[], a, b, l, dm, True)
    except error:
        owner[].busy = False
        owner[].usable = False
        owner[].release()
        raise error
    owner[].busy = False
    return PythonObject(None)


def mamba3_forward_tape_enabled_binding() raises -> PythonObject:
    return PythonObject(IDN_SAMBA_M3_FORWARD_TAPE)


def mamba3_forward_tape_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """11 addresses: x, nine immutable-at-call weights, y. Params B,L,DM.
    Returns an opaque single-use native snapshot; reports are not requested."""
    comptime if not IDN_SAMBA_M3_FORWARD_TAPE:
        raise Error("mamba3 forward tape: experiment disabled")
    if len(addrs) != 11 or len(params) != 3:
        raise Error("mamba3 forward tape: expected 11 addresses and 3 shape scalars")
    var b = Int(py=params[0]); var l = Int(py=params[1]); var dm = Int(py=params[2])
    if b < 1 or l < 1 or dm < 1:
        raise Error("mamba3 forward tape: positive B, L and d_model required")
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0: raise Error("mamba3 forward tape: null input or weight")
        a.append(address)
    for i in range(10): a.append(0)  # small-loop(unused address slots: fixed count)
    var output = Int(py=addrs[10])
    if output == 0: raise Error("mamba3 forward tape: null output")
    a.append(output)
    for i in range(4): a.append(0)  # small-loop(unused address slots: fixed count)
    var owner = Mamba3PrefillSession()
    owner.busy = True
    try:
        with GILReleased(Python()):
            _ = _m3_prefill_run(owner, a, b, l, dm, force_retain=True)
    except error:
        owner.busy = False
        owner.usable = False
        owner.release()
        raise error
    owner.busy = False
    return PythonObject(alloc=Mamba3ForwardTape(owner^))


def mamba3_backward_tape_binding(tape: PythonObject, addrs: PythonObject) raises -> PythonObject:
    """11 addresses: dy, dx and nine forward-order parameter gradients."""
    comptime if not IDN_SAMBA_M3_FORWARD_TAPE:
        raise Error("mamba3 backward tape: experiment disabled")
    var handle = tape.downcast_value_ptr[Mamba3ForwardTape]()
    if handle[].consumed: raise Error("mamba3 backward tape: consumed or closed")
    if handle[].owner.busy: raise Error("mamba3 backward tape: busy")
    if not handle[].owner.usable or not handle[].owner.stages_valid:
        raise Error("mamba3 backward tape: invalid forward snapshot")
    if len(addrs) != 11:
        raise Error("mamba3 backward tape: expected 11 addresses")
    var a = List[Int]()
    for i in range(11):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0: raise Error("mamba3 backward tape: null gradient buffer")
        a.append(address)
    handle[].owner.busy = True
    # A failed backward also consumes the tape: no retry can accidentally use
    # a partially mutated stage. The caller may repeat the original forward.
    handle[].consumed = True
    try:
        with GILReleased(Python()):
            _m3_backward_tape_run(handle[].owner, a)
    except error:
        handle[].owner.busy = False
        handle[].owner.usable = False
        handle[].owner.release()
        raise error
    handle[].owner.busy = False
    handle[].owner.usable = False
    handle[].owner.release()
    return PythonObject(0)


def mamba3_close_tape_binding(tape: PythonObject) raises -> PythonObject:
    var handle = tape.downcast_value_ptr[Mamba3ForwardTape]()
    if handle[].owner.busy: raise Error("mamba3 close tape: busy")
    handle[].consumed = True
    handle[].owner.usable = False
    handle[].owner.release()
    return PythonObject(0)


def mamba3_prefill_session_create_binding() raises -> PythonObject:
    return PythonObject(alloc=Mamba3PrefillSession())


def mamba3_prefill_session_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    if owner[].busy: raise Error("mamba3 prefill session: busy")
    owner[].usable = False
    owner[].release()
    return PythonObject(0)


def mamba3_prefill_session_forward_binding(session: PythonObject, addrs: PythonObject,
                                           params: PythonObject) raises -> PythonObject:
    """`mamba3_forward_fresh` on a session: the same 15 pointers (x, nine
    weights, y, and the four public reports) and 3 parameters (B, L,
    d_model); the weights retained across calls."""
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    if len(addrs) != 15 or len(params) != 3:
        raise Error("mamba3_prefill_session_forward: expected 15 addresses and 3 parameters (B, L, d_model)")
    if owner[].busy: raise Error("mamba3 prefill session: busy")
    if not owner[].usable: raise Error("mamba3 prefill session: lost after a failed call")
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0: raise Error("mamba3_prefill_session_forward: null buffer address")
        a.append(p)
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        a.append(0)
    for i in range(10, 15):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0:
            # lane/fam2-lm: a null REPORT address (11..14) means "do not
            # download this report"; y (10) is always required.
            var report_optional = False
            comptime if IDN_MAMBA3_REPORTS_ON_REQUEST:
                report_optional = i >= 11
            if not report_optional:
                raise Error("mamba3_prefill_session_forward: null buffer address")
        a.append(p)
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b < 1 or l < 1:
        raise Error("mamba3_prefill_session_forward: B and L must be positive")
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _m3_prefill_run(owner[], a, b, l, dm)
    except error:
        owner[].busy = False
        owner[].usable = False
        owner[].release()
        raise error
    owner[].busy = False
    return PythonObject(out_len)


def mamba3_prefill_session_backward_binding(session: PythonObject, addrs: PythonObject,
                                            params: PythonObject) raises -> PythonObject:
    """`mamba3_backward` on a session: the same 21 pointers and 3
    parameters (B, L, d_model); the weights and the last forward's stages
    retained across calls (see `_m3_prefill_backward_run`)."""
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    if len(addrs) != 21 or len(params) != 3:
        raise Error("mamba3_prefill_session_backward: expected 21 addresses and 3 scalars")
    if owner[].busy: raise Error("mamba3 prefill session: busy")
    if not owner[].usable: raise Error("mamba3 prefill session: lost after a failed call")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba3_prefill_session_backward: B, L and d_model must be positive")
    var a = List[Int]()
    for i in range(21):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("mamba3_prefill_session_backward: null buffer address at slot " + String(i))
        a.append(address)
    # The entry's own refusal of a non-finite grad_output now runs on the
    # device, first thing in `_m3_prefill_backward_run` (cpu2-l11-neural);
    # that refusal touched nothing, so the session stays usable.
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_prefill_backward_run(owner[], a, b, l, dm)
    except error:
        owner[].busy = False
        if String(error).startswith(_M3_BWD_NONFINITE_DY):
            raise error
        owner[].usable = False
        owner[].release()
        raise error
    owner[].busy = False
    return PythonObject(0)


def mamba3_prefill_session_reports_optional_binding() raises -> PythonObject:
    """True where `mamba3_prefill_session_forward` admits a null report
    address (IDN_MAMBA3_REPORTS_ON_REQUEST); the Python block asks before
    it passes one."""
    comptime if IDN_MAMBA3_REPORTS_ON_REQUEST:
        return PythonObject(True)
    else:
        return PythonObject(False)


def mamba3_prefill_session_info_binding(session: PythonObject) raises -> PythonObject:
    """[weight_uploads, weight_recopies, weight_reuses, backward_reuses,
    backward_recomputes, stages_held] as ints: the A/B evidence for the
    byte compares (how often the weights and the forward were reused)."""
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    var out = Python.list()
    _ = py_call(out.append, PythonObject(owner[].weight_uploads))
    _ = py_call(out.append, PythonObject(owner[].weight_recopies))
    _ = py_call(out.append, PythonObject(owner[].weight_reuses))
    _ = py_call(out.append, PythonObject(owner[].backward_reuses))
    _ = py_call(out.append, PythonObject(owner[].backward_recomputes))
    _ = py_call(out.append, PythonObject(1 if owner[].stages_valid else 0))
    return out


def mamba3_session_create_binding() raises -> PythonObject:
    return PythonObject(alloc=Mamba3DecodeSession())


def mamba3_session_open_binding(session: PythonObject, addrs: PythonObject,
                                params: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba3DecodeSession]()
    if len(addrs) != 19 or len(params) != 4:
        raise Error("mamba3_session_open: expected 19 addresses and 4 scalars")
    if Int(py=params[0]) <= 0 or Int(py=params[1]) <= 0:
        raise Error("mamba3_session_open: B and d_model must be positive")
    if owner[].busy or owner[].is_open(): raise Error("mamba3 session: busy or open")
    var a = List[Int]()
    for i in range(19):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0: raise Error("mamba3_session_open: null buffer address")
        a.append(p)
    owner[].busy = True
    owner[].usable = False
    try:
        with GILReleased(Python()):
            _m3_session_open_run(owner[], a, Int(py=params[0]), Int(py=params[1]),
                                 Int(py=params[2]), Int(py=params[3]))
    except error:
        owner[].busy = False; owner[].release(); raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(0)


def mamba3_session_step_binding(session: PythonObject, addrs: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba3DecodeSession]()
    if len(addrs) != 6: raise Error("mamba3_session_step: expected 6 addresses")
    var a = List[Int]()
    for i in range(6):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0: raise Error("mamba3_session_step: null buffer address")
        a.append(p)
    _m3_require_session_open(owner[])
    owner[].busy = True
    var q = 0
    try:
        with GILReleased(Python()): q = _m3_session_step_run(owner[], a)
    except error:
        owner[].busy = False; owner[].usable = False; raise error
    owner[].busy = False
    return PythonObject(q)


def mamba3_session_export_state_binding(session: PythonObject, addrs: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba3DecodeSession]()
    if len(addrs) != 10: raise Error("mamba3_session_export_state: expected 10 addresses")
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0: raise Error("mamba3_session_export_state: null buffer address")
        a.append(p)
    _m3_require_session_open(owner[])
    owner[].busy = True
    var q = 0
    try:
        with GILReleased(Python()): q = _m3_session_export_run(owner[], a)
    except error:
        owner[].busy = False; owner[].usable = False; raise error
    owner[].busy = False
    var result = Python.list()
    _ = py_call(result.append, PythonObject(q))
    _ = py_call(result.append, PythonObject(1 if owner[].state.value().pending else 0))
    return result


def mamba3_session_load_state_binding(session: PythonObject, addrs: PythonObject,
                                      params: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[Mamba3DecodeSession]()
    if len(addrs) != 10 or len(params) != 2:
        raise Error("mamba3_session_load_state: expected 10 addresses and 2 scalars")
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        var p = Int(py=addrs[i])
        if p == 0: raise Error("mamba3_session_load_state: null buffer address")
        a.append(p)
    _m3_require_session_open(owner[])
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_session_copy_in(owner[], a, Int(py=params[0]), Int(py=params[1]))
    except error:
        owner[].busy = False; owner[].usable = False; raise error
    owner[].busy = False
    return PythonObject(0)


def mamba3_session_close_binding(session: PythonObject) raises -> PythonObject:
    session.downcast_value_ptr[Mamba3DecodeSession]()[].close()
    return PythonObject(0)


def mamba2_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Zero-state IDENTICAL VJP: x, nine forward-order weights, grad_output,
    grad_x, nine weight gradients (21 addresses); B, L, d_model, dt_lo, dt_hi."""
    if len(addrs) != 21 or len(params) != 5:
        raise Error("mamba2 backward: expected 21 addresses and 5 scalars")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba2 backward: B, L and d_model must be positive")
    var a = List[Int]()
    for i in range(21):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("mamba2 backward: null buffer address at slot " + String(i))
        a.append(address)
    var dt_lo = Float32(Float64(py=params[3]))
    var dt_hi = Float32(Float64(py=params[4]))
    with GILReleased(Python()):
        _mamba2_backward_run(a, b, l, dm, dt_lo, dt_hi)
    return PythonObject(0)


def mamba3_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Zero-state IDENTICAL VJP: x, nine forward-order weights, grad_output,
    grad_x, nine weight gradients (21 addresses); B, L, d_model."""
    if len(addrs) != 21 or len(params) != 3:
        raise Error("mamba3 backward: expected 21 addresses and 3 scalars")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba3 backward: B, L and d_model must be positive")
    var a = List[Int]()
    for i in range(21):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("mamba3 backward: null buffer address at slot " + String(i))
        a.append(address)
    with GILReleased(Python()):
        _mamba3_backward_run(a, b, l, dm)
    return PythonObject(0)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_mamba.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_mamba() -> PythonObject:
    # FAST AND IDENTICAL (lane neural, 2026-09-27). This lane was
    # IDENTICAL-only from 2026-09-10 because its fused kernels were once gated
    # on IDENTICAL and the lower tiers fell back to slower unfused arms. Those
    # fallbacks are gone: FAST runs the same kernels and the same launches with
    # the pins in checks/numerics.mojo compiled to the free schedule. FAST
    # promises quality, never bits.
    # DEVIATION 793's last clause: a sabotage arm exists to be run by a
    # gate and to FAIL; a Python surface that quietly served one would
    # be a wrong answer wearing a green label. Refuse to exist instead.
    comptime if (BLOCK_ANY_SABOTAGE or BLOCK2_ANY_SABOTAGE or BLOCK3_ANY_SABOTAGE
                 or M1_ANY_BWD_SABOTAGE or ANY_BWD2_SABOTAGE or ANY_BWD3_SABOTAGE
                 or ANY_BWD_BLOCK_SABOTAGE
                 or ANY_BWD_SCAN_SABOTAGE or GEMM_ANY_BWD_SABOTAGE
                 or GEMM_ANY_SABOTAGE):
        abort(
            String(
                "_mojolearn_mamba: refusing to initialize -- a sabotage"
                " arm is compiled into this binary. Sabotage defines are"
                " for the lane gates (mamba/checks/), never for a shipped"
                " binding; rebuild with bash bindings/build_mamba.sh and"
                " no MOJOLEARN_MAMBA_SABOTAGE_*,"
                " MOJOLEARN_MAMBA2_SABOTAGE_* or"
                " MOJOLEARN_MAMBA3_SABOTAGE_* or GEMM sabotage define."
            )
        )
    try:
        var m = PythonModuleBuilder("_mojolearn_mamba")
        m.def_function[mamba_vendor_binding]("mamba_vendor")
        m.def_function[mamba_numeric_mode_binding]("mamba_numeric_mode")
        m.def_function[mamba1_profile_binding]("mamba1_profile")
        m.def_function[mamba1_forward_binding]("mamba1_forward")
        m.def_function[mamba1_backward_binding]("mamba1_backward")
        m.def_function[mamba1_decode_step_binding]("mamba1_decode_step")
        # DEVIATION 2941: the Mamba-1 resident decode session.
        _ = m.add_type[Mamba1DecodeSession]("_Mamba1DecodeSession")
        m.def_function[mamba1_session_create_binding]("mamba1_session_create")
        m.def_function[mamba1_session_open_binding]("mamba1_session_open")
        m.def_function[mamba1_session_step_binding]("mamba1_session_step")
        m.def_function[mamba1_session_export_state_binding]("mamba1_session_export_state")
        m.def_function[mamba1_session_load_state_binding]("mamba1_session_load_state")
        # TOMBSTONE: mamba1_session_info, mamba1_forward_fresh, mamba2_forward_fresh (dead native entries: no live caller) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_mamba.patch
        m.def_function[mamba1_session_close_binding]("mamba1_session_close")
        m.def_function[mamba2_backward_binding]("mamba2_backward")
        m.def_function[mamba2_forward_binding]("mamba2_forward")
        m.def_function[mamba2_decode_step_binding]("mamba2_decode_step")
        _ = m.add_type[Mamba2DecodeSession]("_Mamba2DecodeSession")
        m.def_function[mamba2_session_create_binding]("mamba2_session_create")
        m.def_function[mamba2_session_open_binding]("mamba2_session_open")
        m.def_function[mamba2_session_step_binding]("mamba2_session_step")
        m.def_function[mamba2_session_export_state_binding]("mamba2_session_export_state")
        m.def_function[mamba2_session_load_state_binding]("mamba2_session_load_state")
        m.def_function[mamba2_session_close_binding]("mamba2_session_close")
        m.def_function[mamba3_backward_binding]("mamba3_backward")
        _ = m.add_type[Mamba3ForwardTape]("_Mamba3ForwardTape")
        m.def_function[mamba3_forward_tape_enabled_binding]("mamba3_forward_tape_enabled")
        m.def_function[mamba3_forward_tape_binding]("mamba3_forward_tape")
        m.def_function[mamba3_backward_tape_binding]("mamba3_backward_tape")
        m.def_function[mamba3_close_tape_binding]("mamba3_close_tape")
        m.def_function[mamba3_forward_binding]("mamba3_forward")
        comptime if GLOBAL_NUMERIC_MODE <= NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL"]():
            m.def_function[mamba3_forward_fresh_binding]("mamba3_forward_fresh")
            # lane/neural-net-experiment: the fresh prefill on a session.
            _ = m.add_type[Mamba3PrefillSession]("_Mamba3PrefillSession")
            m.def_function[mamba3_prefill_session_create_binding]("mamba3_prefill_session_create")
            m.def_function[mamba3_prefill_session_export_weights_binding]("mamba3_prefill_session_export_weights")
            m.def_function[mamba3_owned_weights_enabled_binding]("mamba3_owned_weights_enabled")
            m.def_function[mamba3_prefill_session_install_weights_binding]("mamba3_prefill_session_install_weights")
            m.def_function[mamba3_prefill_session_forward_owned_binding]("mamba3_prefill_session_forward_owned")
            m.def_function[mamba3_state_owned_binding]("mamba3_state_owned")
            m.def_function[mamba3_prefill_session_backward_owned_binding]("mamba3_prefill_session_backward_owned")
            m.def_function[mamba3_prefill_session_close_binding]("mamba3_prefill_session_close")
            m.def_function[mamba3_prefill_session_forward_binding]("mamba3_prefill_session_forward")
            m.def_function[mamba3_prefill_session_backward_binding]("mamba3_prefill_session_backward")
            m.def_function[mamba3_prefill_session_info_binding]("mamba3_prefill_session_info")
            m.def_function[mamba3_prefill_session_reports_optional_binding](
                "mamba3_prefill_session_reports_optional"
            )
        m.def_function[mamba3_decode_step_binding]("mamba3_decode_step")
        _ = m.add_type[Mamba3DecodeSession]("_Mamba3DecodeSession")
        m.def_function[mamba3_session_create_binding]("mamba3_session_create")
        m.def_function[mamba3_session_open_binding]("mamba3_session_open")
        m.def_function[mamba3_session_step_binding]("mamba3_session_step")
        m.def_function[mamba3_session_export_state_binding]("mamba3_session_export_state")
        m.def_function[mamba3_session_load_state_binding]("mamba3_session_load_state")
        m.def_function[mamba3_session_close_binding]("mamba3_session_close")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_mamba: ", e))


def mamba3_prefill_session_export_weights_binding(session: PythonObject, addrs: PythonObject, generation: PythonObject) raises -> PythonObject:
    """Download the installed immutable generation, never borrowed arrays."""
    var owner = session.downcast_value_ptr[Mamba3PrefillSession]()
    if owner[].busy or not owner[].usable or not owner[].owned_weights or not owner[].w or Int(py=generation) != owner[].weight_generation:
        raise Error("mamba3 owned export: missing, busy, unusable or stale generation")
    if len(addrs) != 9:
        raise Error("mamba3 owned export: expected nine writable buffers")
    var a = List[Int]()
    for i in range(9):  # small-loop(addrs: fixed-length buffer address list)
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error("mamba3 owned export: null output address")
        a.append(address)
    var dm = owner[].dm
    var dims = Mamba3Dims.of(dm)
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_export_weights_run(owner[], a, dm, dims)
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return generation
