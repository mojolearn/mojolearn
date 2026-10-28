# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_mamba` family: the Mamba-1, Mamba-2 and
Mamba-3 blocks' forward, decode step and zero-state IDENTICAL prefill
backward on the host (lane/cpu-training-mamba, 2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. Every entry below
carries the GPU binding's name, address list and params list
(`bindings/_mojolearn_mamba.mojo`), so `Mamba1Block`, `Mamba2Block` and
`Mamba3Block` (`python/mojolearn/_mamba_impl.py`) run unchanged on a
CPU-only install through `_backend._HOST_MODULES`:

  `mamba1_forward`, `mamba1_decode_step`
        `mamba/checks/mamba_oracle.mojo::mamba_block_oracle`, the host
        Float32 oracle the device block is gated against bitwise.
  `mamba2_forward`, `mamba2_decode_step`
        `mamba/checks/mamba2_oracle.mojo::mamba2_block_oracle` (the three
        piece state, the dt clamp, the open-chunk buffer).
  `mamba3_forward`, `mamba3_forward_fresh`, `mamba3_decode_step`
        `mamba/checks/mamba3_oracle.mojo::mamba3_block_oracle`.
  `mamba1_backward`, `mamba2_backward`, `mamba3_backward`
        the zero-state prefill VJPs over the device's own kernels
        (`mamba/host/mamba1_prefill_chain.mojo`,
        `mamba/host/mamba2_prefill_chain.mojo`,
        `mamba/host/mamba3_prefill_chain.mojo`: the launch walks the
        device's tape rules run), forward stages
        included, as `tools/mamba_host_gen.py` writes them out for the host
        under `mamba/host/gen/`: every kernel a serial loop over its launch
        grid, every buffer a host allocation and the device GEMM
        `gemm_oracle` (`mamba/host/device_shim.mojo`). The Mamba-1 host
        backward oracle is NOT used: on the M4 it differs from the device
        VJP in the low bits of eight of eleven gradients.

The sabotage arm is `gemm_oracle`'s (`-D MOJOLEARN_HOST_SABOTAGE=1` walks
every GEMM leaf descending), which every block's projections reach.
"""
from std.memory import memcpy
from std.sys.compile import is_defined
from std.os import abort
from mamba.host.native_block_dispatch import NATIVE_MAMBA_HOST_BLOCK_API, NATIVE_MAMBA_HOST_STATE_API
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import f32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE
from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims, MambaWeights
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
    Mamba2Weights,
)
from mamba.checks.mamba2_oracle import Mamba2State, mamba2_block_oracle
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
    Mamba3Weights,
)
from mamba.checks.mamba3_oracle import Mamba3State, mamba3_block_oracle
from mamba.host.gen.mamba2_prefill_chain import mamba2_prefill_backward
from mamba.host.gen.mamba3_prefill_chain import mamba3_prefill_backward
from mamba.host.gen.mamba1_prefill_chain import mamba1_prefill_backward
# TOMBSTONE: MOJOLEARN_IDN_M1_SCAN=1 (NN34 affine_prefix: profile/state/branches in this file; LOSER avg 2.30x mamba1-forward) deleted 2026-10-11 by 291629888; code recoverable at 291629888^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_SCAN_1.patch
from mamba.host.device_shim import DeviceContext as M1HostContext
from mamba.host.gen.identity_trace import IdentityTrace as M1HostTrace
from mamba.host.gen.modeling_mamba import MambaDeviceState as M1HostState, MambaDeviceWeights as M1HostWeights, MambaDeviceStages as M1HostStages, mamba_upload as m1_host_upload, mamba_download as m1_host_download, mamba_block_forward as m1_host_forward


from mamba.host.mamba_host_shell_impl import (
    _write,
    _m1_weights,
    _mamba1_run,
    _mamba1_backward_run,
    _m2_weights,
    _mamba2_run,
    _mamba2_backward_run,
    _m3_weights,
    _mamba3_run,
    _mamba3_backward_run,
    NN40_OWNED_WEIGHTS,
    M3OwnedHostSession,
    _m3_owned_install_run,
    _m3_owned_forward_run,
    _m3_owned_state_run,
    _m3_owned_backward_run,
    _m3_owned_export_run,
    _nn40_host_require,
)



def _addrs(addrs: PythonObject, n: Int, what: String) raises -> List[Int]:
    if len(addrs) != n:
        raise Error(
            what + ": addrs must contain " + String(n) + " addresses, got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(n):  # small-loop(addrs: fixed address count per entry, set by the caller)
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error(what + ": null buffer address at slot " + String(i))
        a.append(address)
    return a^


# ===========================================================================
# THE READ-BACK
# ===========================================================================


def mamba_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "mamba host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mamba_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mamba_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "mamba host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_mamba_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def mamba_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


def mamba_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mamba_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mamba1_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`bindings/_mojolearn_mamba.mojo::mamba1_forward_binding`'s contract:
    14 addresses, params B, L, d_model."""
    var a = _addrs(addrs, 14, String("mamba1_forward"))
    if len(params) != 3:
        raise Error("mamba1_forward: params must contain 3 values (B, L, d_model), got " + String(len(params)))
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var position = 0
    with GILReleased(Python()):
        _mamba1_run(a, b, l, dm, position)
    return PythonObject(0)


def mamba1_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1 with the state carried; params B, d_model."""
    var a = _addrs(addrs, 14, String("mamba1_decode_step"))
    if len(params) != 2:
        raise Error("mamba1_decode_step: params must contain 2 values (B, d_model), got " + String(len(params)))
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var position = 0
    with GILReleased(Python()):
        _mamba1_run(a, b, 1, dm, position)
    return PythonObject(0)


def mamba1_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Zero-state IDENTICAL VJP: 23 addresses, params B, L, d_model."""
    if len(addrs) != 23 or len(params) != 3:
        raise Error("mamba1 backward: expected 23 addresses and B, L, d_model")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba1 backward: B, L and d_model must be positive")
    var a = _addrs(addrs, 23, String("mamba1 backward"))
    with GILReleased(Python()):
        _mamba1_backward_run(a, b, l, dm)
    return PythonObject(0)


def mamba2_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """16 addresses; params B, L, d_model, buf_len, dt_lo, dt_hi. Returns
    the post-call buf_len."""
    var a = _addrs(addrs, 16, String("mamba2_forward"))
    if len(params) != 6:
        raise Error(
            "mamba2_forward: params must contain 6 values (B, L, d_model,"
            " buf_len, dt_lo, dt_hi), got " + String(len(params))
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


def mamba2_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1; params B, d_model, buf_len, dt_lo, dt_hi."""
    var a = _addrs(addrs, 16, String("mamba2_decode_step"))
    if len(params) != 5:
        raise Error(
            "mamba2_decode_step: params must contain 5 values (B, d_model,"
            " buf_len, dt_lo, dt_hi), got " + String(len(params))
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


def mamba2_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """21 addresses; params B, L, d_model, dt_lo, dt_hi."""
    if len(addrs) != 21 or len(params) != 5:
        raise Error("mamba2 backward: expected 21 addresses and 5 scalars")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba2 backward: B, L and d_model must be positive")
    var a = _addrs(addrs, 21, String("mamba2 backward"))
    var dt_lo = Float32(Float64(py=params[3]))
    var dt_hi = Float32(Float64(py=params[4]))
    with GILReleased(Python()):
        _mamba2_backward_run(a, b, l, dm, dt_lo, dt_hi)
    return PythonObject(0)


def mamba3_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """25 addresses; params B, L, d_model, buf_len, pending."""
    var a = _addrs(addrs, 25, String("mamba3_forward"))
    if len(params) != 5:
        raise Error(
            "mamba3_forward: params must contain 5 values (B, L, d_model,"
            " buf_len, pending), got " + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var q0 = Int(py=params[3])
    var pend = Int(py=params[4])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, l, dm, q0, pend, False)
    return PythonObject(out_len)


def mamba3_forward_fresh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """15 addresses (x, nine weights, y, four reports); params B, L,
    d_model. A zero state whose resumption pieces are discarded."""
    if len(addrs) != 15 or len(params) != 3:
        raise Error("mamba3_forward_fresh: expected 15 addresses and 3 parameters (B, L, d_model)")
    var a = List[Int]()
    for i in range(10):  # small-loop(addrs: fixed-length buffer address list)
        a.append(Int(py=addrs[i]))
    for _ in range(10):  # small-loop(addrs: fixed-length buffer address list)
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
        out_len = _mamba3_run(a, b, l, dm, 0, 0, True)
    return PythonObject(out_len)


def mamba3_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1; params B, d_model, buf_len, pending."""
    var a = _addrs(addrs, 25, String("mamba3_decode_step"))
    if len(params) != 4:
        raise Error(
            "mamba3_decode_step: params must contain 4 values (B,"
            " d_model, buf_len, pending), got " + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var q0 = Int(py=params[2])
    var pend = Int(py=params[3])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, 1, dm, q0, pend, False)
    return PythonObject(out_len)


def mamba3_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """21 addresses; params B, L, d_model."""
    if len(addrs) != 21 or len(params) != 3:
        raise Error("mamba3 backward: expected 21 addresses and 3 scalars")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba3 backward: B, L and d_model must be positive")
    var a = _addrs(addrs, 21, String("mamba3 backward"))
    with GILReleased(Python()):
        _mamba3_backward_run(a, b, l, dm)
    return PythonObject(0)


def mamba3_forward_tape_enabled_binding() raises -> PythonObject:
    # NI48 is scheduling-only. The host uses its unchanged forward/recompute
    # arithmetic; it must not advertise a native retained-stage API it lacks.
    return PythonObject(False)


# The module's one C export. (The tape flag above was once inserted between
# this decorator and PyInit, which left PyInit unexported: an import read
# "dynamic module does not define module export function".)
# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_mamba_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).

def mamba_state_api_binding() raises -> PythonObject:
    var api = NATIVE_MAMBA_HOST_STATE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def mamba_block_api_binding() raises -> PythonObject:
    var api = NATIVE_MAMBA_HOST_BLOCK_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def mamba_native_session_handle_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    return PythonObject(Int(owner))


def build__mojolearn_mamba_host() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_mamba_host")
        m.def_function[mamba_block_api_binding]("mamba_block_api_v1")
        m.def_function[mamba_state_api_binding]("mamba_state_api_v1")
        m.def_function[mamba_native_session_handle_binding]("mamba_prefill_native_handle_v1")
        m.def_function[mamba_host_numeric_mode_binding]("mamba_host_numeric_mode")
        m.def_function[mamba_host_vendor_binding]("mamba_host_vendor")
        m.def_function[mamba_host_column_binding]("mamba_host_column")
        m.def_function[mamba_host_sabotage_binding]("mamba_host_sabotage")
        m.def_function[mamba_vendor_binding]("mamba_vendor")
        m.def_function[mamba_numeric_mode_binding]("mamba_numeric_mode")
        m.def_function[mamba1_profile_binding]("mamba1_profile")
        m.def_function[mamba1_forward_binding]("mamba1_forward")
        m.def_function[mamba1_backward_binding]("mamba1_backward")
        m.def_function[mamba1_decode_step_binding]("mamba1_decode_step")
        m.def_function[mamba2_forward_binding]("mamba2_forward")
        m.def_function[mamba2_decode_step_binding]("mamba2_decode_step")
        m.def_function[mamba2_backward_binding]("mamba2_backward")
        m.def_function[mamba3_owned_weights_enabled_binding]("mamba3_owned_weights_enabled")
        m.def_function[mamba3_owned_create_binding]("mamba3_prefill_session_create")
        m.def_function[mamba3_owned_reports_optional_binding]("mamba3_prefill_session_reports_optional")
        m.def_function[mamba3_owned_info_binding]("mamba3_prefill_session_info")
        m.def_function[mamba3_owned_close_binding]("mamba3_prefill_session_close")
        m.def_function[mamba3_owned_install_binding]("mamba3_prefill_session_install_weights")
        m.def_function[mamba3_owned_forward_binding]("mamba3_prefill_session_forward_owned")
        m.def_function[mamba3_state_owned_binding]("mamba3_state_owned")
        m.def_function[mamba3_owned_backward_binding]("mamba3_prefill_session_backward_owned")
        m.def_function[mamba3_owned_export_binding]("mamba3_prefill_session_export_weights")
        m.def_function[mamba3_forward_binding]("mamba3_forward")
        m.def_function[mamba3_forward_fresh_binding]("mamba3_forward_fresh")
        m.def_function[mamba3_decode_step_binding]("mamba3_decode_step")
        m.def_function[mamba3_backward_binding]("mamba3_backward")
        m.def_function[mamba3_forward_tape_enabled_binding]("mamba3_forward_tape_enabled")
        return m.finalize()
    except error:
        abort(String("failed to create _mojolearn_mamba_host: ", error))


def mamba1_profile_binding() -> PythonObject:
    return PythonObject(1)  # NN34 profile 2 (M1_SCAN=1) deleted 2026-10-11


def mamba3_owned_weights_enabled_binding() -> PythonObject:
    return PythonObject(NN40_OWNED_WEIGHTS)


def mamba3_owned_create_binding() raises -> PythonObject:
    return PythonObject(alloc=M3OwnedHostSession())


def mamba3_owned_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    if owner[].busy:
        raise Error("mamba3 owned host: session busy")
    owner[].weights = None
    owner[].usable = False
    return PythonObject(0)


def mamba3_owned_install_binding(session: PythonObject, addrs: PythonObject, d_model_obj: PythonObject) raises -> PythonObject:
    comptime if not NN40_OWNED_WEIGHTS:
        raise Error("mamba3 owned host: NN40 arm disabled")
    var d_model = Int(py=d_model_obj)
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    if owner[].busy or not owner[].usable:
        raise Error("mamba3 owned host: session busy or unusable")
    var a = _addrs(addrs, 9, String("mamba3 owned install"))
    var wa: List[Int] = [0]
    wa.extend(a.copy())
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_owned_install_run(owner[], wa, d_model)
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(owner[].generation)


def mamba3_owned_forward_binding(session: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    var a = _addrs(addrs, 6, String("mamba3 owned forward"))
    if len(params) != 4:
        raise Error("mamba3 owned forward: expected B, L, d_model, generation")
    var b = Int(py=params[0]); var l = Int(py=params[1]); var dm = Int(py=params[2]); var generation = Int(py=params[3])
    _nn40_host_require(owner[], dm, generation)
    if b <= 0 or l <= 0:
        raise Error("mamba3 owned forward: positive B and L required")
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_owned_forward_run(owner[], a, b, l, dm)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    owner[].weight_reuses += 1
    return PythonObject(0)


def mamba3_state_owned_binding(session: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`mamba3_state_owned` (host column): `mamba3_forward` (step 0) or
    `mamba3_decode_step` (step 1) on the installed owned snapshot; `addrs`
    the same twenty-five as `mamba3_forward` (the nine weight slots are not
    read); `params` = [B, L, d_model, buf_len, pending, generation, step].
    Returns the post-call buf_len."""
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    var a = _addrs(addrs, 25, String("mamba3_state_owned"))
    if len(params) != 7:
        raise Error("mamba3_state_owned: params must contain 7 values (B, L, d_model, buf_len, pending, generation, step)")
    var b = Int(py=params[0]); var l = Int(py=params[1]); var dm = Int(py=params[2])
    var q0 = Int(py=params[3]); var pend = Int(py=params[4]); var generation = Int(py=params[5])
    var step = Int(py=params[6])
    _nn40_host_require(owner[], dm, generation)
    if step != 0 and step != 1:
        raise Error("mamba3_state_owned: step must be 0 or 1")
    if step == 1 and l != 1:
        raise Error("mamba3_state_owned: a decode step is one token (L = 1)")
    owner[].busy = True
    var out_len = 0
    try:
        with GILReleased(Python()):
            out_len = _m3_owned_state_run(owner[], a, b, l, dm, q0, pend)
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    owner[].weight_reuses += 1
    return PythonObject(out_len)


def mamba3_owned_backward_binding(session: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    var a = _addrs(addrs, 12, String("mamba3 owned backward"))
    if len(params) != 4:
        raise Error("mamba3 owned backward: expected B, L, d_model, generation")
    var b = Int(py=params[0]); var l = Int(py=params[1]); var dm = Int(py=params[2]); var generation = Int(py=params[3])
    _nn40_host_require(owner[], dm, generation)
    if b <= 0 or l <= 0:
        raise Error("mamba3 owned backward: positive B and L required")
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_owned_backward_run(owner[], a, b, l, dm)
    except error:
        owner[].busy = False
        owner[].usable = False
        raise error
    owner[].busy = False
    owner[].weight_reuses += 1
    owner[].backward_recomputes += 1
    return PythonObject(0)


def mamba3_owned_export_binding(session: PythonObject, addrs: PythonObject, generation_obj: PythonObject) raises -> PythonObject:
    var generation = Int(py=generation_obj)
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    if not owner[].weights:
        raise Error("mamba3 owned export: no snapshot")
    var dm = owner[].weights.value().dims.d_model
    _nn40_host_require(owner[], dm, generation)
    var a = _addrs(addrs, 9, String("mamba3 owned export"))
    var dims = owner[].weights.value().dims.copy()
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _m3_owned_export_run(owner[], a, dm, dims)
    except error:
        owner[].busy = False
        raise error
    owner[].busy = False
    return PythonObject(generation)


def mamba3_owned_reports_optional_binding() -> PythonObject:
    """The host owner currently materializes every report. Advertising the
    capability explicitly keeps Samba's discard-report probe callable."""
    return PythonObject(False)


def mamba3_owned_info_binding(session: PythonObject) raises -> PythonObject:
    """Same six-field public metadata schema; host backward recomputes and
    holds no saved stages. Counters report snapshot use, not device traffic."""
    var owner = session.downcast_value_ptr[M3OwnedHostSession]()
    if owner[].busy:
        raise Error("mamba3 owned host info: session busy")
    var result = Python.list()
    result.append(PythonObject(owner[].generation))
    result.append(PythonObject(0))
    result.append(PythonObject(owner[].weight_reuses))
    result.append(PythonObject(0))
    result.append(PythonObject(owner[].backward_recomputes))
    result.append(PythonObject(0))
    return result
