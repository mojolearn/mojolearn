# SPDX-License-Identifier: Apache-2.0
"""Synchronous owned-state boundary for a runtime-shaped decoder language model.

Runtime shapes are compile/host checked; device qualification is separate.
No GPU/context is created during import.
The Python caller owns correctly sized, contiguous, aligned live arrays. Native
span checks cannot prove that an arbitrary integer address names allocated RAM.
No borrowed pointer survives a call. Optional owned sessions retain device state.
Outputs are published only after successful computation, validation and
synchronization; the stateless ABI also tears down its context before return.

DEVIATION 2514 (device-owned step): the `byte_lm_session_*`
entries below make the session's device buffers the
ONLY copy of the model, the optimizer state, the last gradient and the flags
as long as it is open. State crosses the boundary at `open` (validated on the
host Lists, uploaded once), at `export_state` / `export_gradients` (pinned
staging, then `copy_f32` into the caller's arrays) and never per step; a
step moves the ids in and the loss and the flags out. A failed step is
rolled back on the device from the shadow the trainer takes before its
update; `rollback` exposes the same restore for a failure the Python layer
detects after native success.
"""
# DEVIATION 2486: shared byte-preserving host copies.
from core.hostptr import f32_ptr, i32_ptr, read_f32, copy_f32, list_i32
from training.byte_lm_native_status import byte_lm_status
from training.byte_lm_native_dispatch import BYTE_LM_NATIVE_TABLE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from training.byte_lm_stateless_logits import byte_lm_stateless_logits_native
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from std.gpu import block_dim, block_idx, thread_idx
from core.device_zero import enqueue_fill
from std.ffi import _Global
from std.memory import bitcast, memcmp
from std.os import abort, getenv
from std.python import Python, PythonObject
from std.time import perf_counter_ns
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from training.byte_lm_binding_impl import (
    ByteLMSession, BYTE_LM_CONTEXT_KEEPER,
    _span_cells, _validate_addresses, _read_f32, _require_inputs_unchanged, _write_f32,
    _btick, _bbytes, _require_same_bits,
    _require_binding_profile, _validate_slot_table,
    _read_flags, _read_ids, _write_flags, _write_flags_n, _write_loss, _nonfinite,
    _same_optimizer_bits, _require_open, _mark_if_lost, _admit_session_scalars, _stage_download,
)
# DEVIATION 2630: the step phase timers and counters (core/step_phase.mojo;
# compiled only under -D MOJOLEARN_STEP_PHASE_TIMERS=1).
from core.step_phase import (
    step_count_sync,
    step_counts_now,
    step_counts_report,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.byte_lm_afn import BYTE_LM_FAST_APPLE
from core.neural_context import neural_ctx
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
from training.byte_lm_session_ops import (
    BYTE_LM_NEURAL_CTX,
    byte_session_open, byte_session_step, byte_session_eval,
    byte_session_export_state, byte_session_export_gradients, byte_session_rollback,
    byte_session_logits, byte_owner_step, byte_owner_export_state,
    byte_owner_export_gradients, byte_owner_logits, byte_scatter_named, byte_file_sha256,
)
from bindings.sha256_door import sha256_hexdigest
comptime _NEURAL_CTX = BYTE_LM_NEURAL_CTX
from checks.vendor import COMPILED_VENDOR
from gemm.checks.gemm_identical import TUNED_STAGE_FTZ, GEMM_REUSE_GROUP_WS
from training.checks.optimizer_contract import OptimizerConfig
from training.checks.train_loop import download_f32, download_f32_into
from training.byte_lm_config import ByteConfig, byte_lm_arithmetic_suffix
from training.byte_lm_stateless_run import byte_lm_run_native, byte_lm_run_finalize
from training.byte_lm import (
    BYTE_PROFILE, ByteTrainer, byte_train_step, byte_train_step_resident,
    byte_eval_loss, byte_eval_loss_resident, byte_rollback,
    byte_validate_state, byte_validate_device_state, byte_validate_optimizer,
    byte_validate_tokens, byte_lm_fault_inject_available,
    byte_attention_eager_cells, byte_lm_attn_bwd_corner_refuses,
    byte_lm_attn_kv_corner_guard,
    byte_lm_attn_sticky_fallback, byte_lm_ce_aliased,
)
from training.byte_lm_optimizer_pool import pool_fault_available
from core.device_arena import arena_stats, device_arena_on
from training.byte_lm_model_pool import ByteModelPool
from training.byte_lm_offload import ByteOffloadedReplay
from training.byte_lm_parallel import ByteParallelTrainer
from training.byte_lm_logits import (
    ByteLogitsScratch,
    BYTE_LOGITS_MAX_BATCH,
    BYTE_LOGITS_MAX_CELLS,
    byte_logits_from_params,
    byte_logits_from_params_into,
    byte_logits_resident,
    byte_logits_resident_into,
    byte_next_bytes_resident_into,
    byte_logits_validate,
    byte_logits_validate_params,
)
from transformer.impl.llama.fused_attention import (
    ATTN_ARM_DEFAULT,
    ATTN_ARM_TRIAL,
    fused_attention_arm_backward_resolved,
    fused_attention_arm_forward_resolved,
    fused_attention_arm_from_env,
    fused_attention_arm_name,
    attention_v1_backward_memory_profile,
    attention_v1_retained_exp_bytes,
)
# DEVIATION 2648: the step glue arm read-back (core/step_glue.mojo).
from core.step_glue import (
    step_glue_arm_from_env,
    step_glue_arm_name,
    step_glue_trial_build,
)


def byte_lm_step_glue_arm_binding() raises -> PythonObject:
    """DEVIATION 2648 read-back, so a result names the glue arm the step ran
    instead of an environment variable that may be unset or misspelled:
    [arm, trial_build]. A trial build reads MOJOLEARN_STEP_GLUE_ARM and
    raises on an invalid name, exactly as the launchers do; every other
    build returns `shipped` without reading the environment. Reads
    constants and the environment only; no GPU operation."""
    var arm = step_glue_arm_from_env()
    var out = Python.list()
    out.append(PythonObject(step_glue_arm_name(arm)))
    out.append(PythonObject(1 if step_glue_trial_build() else 0))
    return out


def byte_lm_attention_arm_binding() raises -> PythonObject:
    """DEVIATION 2534 read-back, so a result can never confuse `baseline`,
    `stash_tiled` and the column's default: [arm, default, trial_build,
    resolved_hd64]. `arm` is the name of the arm this process's launchers
    run (a trial build reads MOJOLEARN_ATTN_ARM and raises on an unknown
    name, exactly as the launchers do; every other build returns the
    default without reading the environment); `default` is the column's
    kernel-matrix row (`attn_default_arm_for`); `trial_build` is 1 under
    `-D MOJOLEARN_ATTN_ARM_TRIAL=1`; `resolved_hd64` is the arm with its
    geometry resolved as the launchers resolve it at head_dim 64 on this
    build. Reads constants and the environment only; no GPU operation.
    On a shipped Apple build (lane/neural-apple, 2026-09-28) the default is
    an estash word gated by memory: `arm` reads the round 3 word until a
    trainer in this process has been granted the estash word
    (`attention_estash_memory_grant`), so a read-back taken before the first
    trainer exists names the fallback."""
    var arm = fused_attention_arm_from_env()
    var resolved = fused_attention_arm_forward_resolved(arm) | fused_attention_arm_backward_resolved(arm)
    var out = Python.list()
    out.append(PythonObject(fused_attention_arm_name(arm)))
    out.append(PythonObject(fused_attention_arm_name(ATTN_ARM_DEFAULT)))
    out.append(PythonObject(1 if ATTN_ARM_TRIAL else 0))
    out.append(PythonObject(fused_attention_arm_name(resolved)))
    return out

# TOMBSTONE: byte_lm_context_keeper_active, byte_lm_attention_estash_gate (dead native entries: diagnostic read-backs, no package or gate reader) deleted 2026-10-26 by 9d5242f68; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_byte_lm.patch


def byte_lm_attention_memory_profile_binding(shape: PythonObject) raises -> PythonObject:
    """Build-profile readback plus retained-exp allocation for one layer."""
    var cfg = _byte_config(shape)
    var out = Python.list()
    out.append(PythonObject(attention_v1_backward_memory_profile()))
    out.append(PythonObject(attention_v1_retained_exp_bytes(cfg.batch,cfg.length,cfg.n_heads,cfg.length)))
    return out


def byte_lm_native_api_binding() raises -> PythonObject:
    var api = BYTE_LM_NATIVE_TABLE.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def byte_lm_native_session_handle_binding(session: PythonObject) raises -> PythonObject:
    # Only the extension that minted this exact native value may unwrap it.
    # The Python provider retains session and module for every use of this handle.
    return PythonObject(Int(session.downcast_value_ptr[ByteLMSession]()))


def byte_lm_session_create_binding() raises -> PythonObject:
    # Lazy: creation and import perform no GPU operation.
    return PythonObject(alloc=ByteLMSession())


def byte_lm_session_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteLMSession]()
    owner[].close()
    return PythonObject(0)


def byte_lm_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def byte_lm_vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


def byte_lm_profile_binding() raises -> PythonObject:
    return PythonObject(ByteConfig().profile())


def _byte_lm_run(addresses: PythonObject, params: PythonObject, shape: ByteConfig,
                 mut session: ByteLMSession, retain: Bool = False) raises -> PythonObject:
    """Shared implementation. All sizes are ELEMENTS, not bytes.

    addresses[11] = [in_param, in_m, in_v, in_flags_i32, in_ids_i32,
                     out_param, out_m, out_v, out_grad, out_flags_i32, out_loss_f32]
    params[12] = [action, completed, kind, lr, beta1, beta2, eps, weight_decay,
                  momentum, dampening, nesterov, max_norm]
    action=0: eval; out_grad MUST be 0; returns unchanged completed step.
    action=1: training; writes pre-update gradients; returns completed+1.
    Param/m/v/grad spans use shape.n_total(); flags[shape.n_tensors()] int32(0/1);
    IDs[B,L+1] int32[0,shape.vocab_size). kind=2(AdamW). All config fields explicit. Positive learning rate, no clipping or SGD options.
    """
    if len(addresses) != 11 or len(params) != 12:
        raise Error("byte LM: expected 11 addresses and 12 scalar parameters")
    var action = _params_action(params)
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 11)
    var locked_teardown = String(getenv("MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL")) == "1"
    var result = 0
    with GILReleased(Python()):
        result = byte_lm_run_native(addr, action, completed, cfg, shape, session, retain, locked_teardown)
    if not retain and locked_teardown:
        byte_lm_run_finalize(session)
    return PythonObject(result)


# ---------------------------------------------------------------------------
# DEVIATION 2514 step 5: the device-owned session entries.
# ---------------------------------------------------------------------------


def _read_addresses(addresses: PythonObject, count: Int) raises -> List[Int]:
    if len(addresses) != count:
        raise Error("byte LM: expected " + String(count) + " addresses")
    var addr = List[Int]()
    for i in range(count):  # small-loop(count: the call's address slots or shards): reads the addresses, no data
        addr.append(Int(py=addresses[i]))
    return addr^


def _params_completed(params: PythonObject) raises -> Int:
    if len(params) != 12:
        raise Error("byte LM: expected 12 scalar parameters")
    var completed = Int(py=params[1])
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step outside admitted bound")
    return completed


def _params_action(params: PythonObject) raises -> Int:
    if len(params) != 12:
        raise Error("byte LM: expected 12 scalar parameters")
    var action = Int(py=params[0])
    if action != 0 and action != 1:
        raise Error("byte LM: action/nesterov must be 0 or 1")
    return action


def _params_optimizer(params: PythonObject) raises -> OptimizerConfig:
    """The same twelve scalars `_byte_lm_run` reads, admitted the same way."""
    if len(params) != 12:
        raise Error("byte LM: expected 12 scalar parameters")
    var kind = Int(py=params[2])
    var nesterov = Int(py=params[10])
    if nesterov != 0 and nesterov != 1:
        raise Error("byte LM: action/nesterov must be 0 or 1")
    var cfg = OptimizerConfig(kind,
        Float32(Float64(py=params[3])), Float32(Float64(py=params[4])),
        Float32(Float64(py=params[5])), Float32(Float64(py=params[6])),
        Float32(Float64(py=params[7])), Float32(Float64(py=params[8])),
        Float32(Float64(py=params[9])), nesterov == 1,
        Float32(Float64(py=params[11])))
    byte_validate_optimizer(cfg)
    return cfg^


def byte_lm_fault_inject_available_binding() raises -> PythonObject:
    """True only in a gate build compiled with -D MOJOLEARN_BYTE_LM_FAULT_INJECT=1."""
    return PythonObject(byte_lm_fault_inject_available())


def byte_lm_gemm_reuse_group_ws_binding() raises -> PythonObject:
    """Read grouped-scratch reuse from the loaded training binary."""
    return PythonObject(GEMM_REUSE_GROUP_WS)


def byte_lm_gemm_stage_ftz_binding() raises -> PythonObject:
    """Read operand-flush placement from the loaded training binary."""
    return PythonObject(TUNED_STAGE_FTZ)


def byte_lm_ce_aliased_binding() raises -> PythonObject:
    """DEVIATION 3011: False in a build carrying
    -D MOJOLEARN_BYTE_LM_CE_UNALIASED=1, which allocates the five [M, V]
    cross-entropy buffers separately instead of overlaying three of them.
    The A/B that claims aliasing moves no bit reads this to prove its two
    arms are two arms."""
    return PythonObject(byte_lm_ce_aliased())


def byte_lm_attn_sticky_fallback_binding() raises -> PythonObject:
    """DEVIATION 3110: False in a build carrying
    -D MOJOLEARN_ATTN_NO_STICKY=1, which relaunches the fused attention
    kernels for a layer that has already refused and then discards the
    launch. The A/B that claims the latch moves no bit reads this to prove
    its two arms are two arms."""
    return PythonObject(byte_lm_attn_sticky_fallback())


def byte_lm_attn_bwd_corner_refuses_binding() raises -> PythonObject:
    """DEVIATION 3112: False in the measurement build carrying
    -D MOJOLEARN_ATTN_NO_BWD_CORNER=1."""
    return PythonObject(byte_lm_attn_bwd_corner_refuses())


def byte_lm_attn_kv_corner_guard_binding() raises -> PythonObject:
    """DEVIATION 3111: False in a build carrying
    the qualified replay column or explicit guard defines."""
    return PythonObject(byte_lm_attn_kv_corner_guard())


def byte_lm_session_open_binding(session: PythonObject, addresses: PythonObject,
                                 params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """Admit host state ONCE and upload it ONCE (design 1.1 item 1).

    addresses[4] = [in_param, in_m, in_v, in_flags_i32]; params[12] as
    `_byte_lm_run` (action ignored; completed is params[1]). Returns the
    admitted completed step. Refused on an open session: close it first.
    The work is `byte_session_open` (training/byte_lm_session_ops.mojo).
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 4)
    var result = completed
    with GILReleased(Python()):
        result = byte_session_open(owner[], addr, completed, cfg, cfg_shape)
    return PythonObject(result)


def byte_lm_session_step_binding(session: PythonObject, addresses: PythonObject,
                                 params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """One device-owned step. addresses[4] = [in_ids_i32, in_flags_i32,
    out_loss_f32, out_flags_i32]; params[12] as `_byte_lm_run` with
    action=1. Returns completed+1. The work is `byte_session_step`.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    if _params_action(params) != 1:
        raise Error("byte LM: session step requires action 1")
    var completed = _params_completed(params)
    if completed >= 999999:
        raise Error("byte LM: completed step outside admitted bound")
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 4)
    var result = completed
    with GILReleased(Python()):
        result = byte_session_step(owner[], addr, completed, cfg, cfg_shape)
    return PythonObject(result)


def byte_lm_session_eval_binding(session: PythonObject, addresses: PythonObject,
                                 params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """Forward-only loss on the resident state. addresses[3] = [in_ids_i32,
    in_flags_i32, out_loss_f32]; params[12] with action=0. Returns the
    unchanged completed step. The work is `byte_session_eval`.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    if _params_action(params) != 0:
        raise Error("byte LM: session eval requires action 0")
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 3)
    var result = completed
    with GILReleased(Python()):
        result = byte_session_eval(owner[], addr, completed, cfg, cfg_shape)
    return PythonObject(result)


def byte_lm_session_export_state_binding(session: PythonObject, addresses: PythonObject,
                                         shape: PythonObject) raises -> PythonObject:
    """Copy the resident state out. addresses[4] = [out_param, out_m,
    out_v, out_flags_i32]. Returns the completed step. The work is
    `byte_session_export_state`.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var addr = _read_addresses(addresses, 4)
    var result = 0
    with GILReleased(Python()):
        result = byte_session_export_state(owner[], addr, cfg_shape)
    return PythonObject(result)


def byte_lm_session_export_gradients_binding(session: PythonObject, addresses: PythonObject,
                                             shape: PythonObject) raises -> PythonObject:
    """Copy the gradient of the LAST completed step out. addresses[1] =
    [out_grad]. Returns the step the gradient belongs to. The work is
    `byte_session_export_gradients`.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var addr = _read_addresses(addresses, 1)
    var result = 0
    with GILReleased(Python()):
        result = byte_session_export_gradients(owner[], addr[0], cfg_shape)
    return PythonObject(result)


def byte_lm_session_rollback_binding(session: PythonObject) raises -> PythonObject:
    """Restore the device state from the shadow of the last step that
    reached its shadow point (design 4.2 item 6). Returns the completed step
    after the call. The work is `byte_session_rollback`.
    """
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var result = 0
    with GILReleased(Python()):
        result = byte_session_rollback(owner[])
    return PythonObject(result)


def byte_lm_session_info_binding(session: PythonObject) raises -> PythonObject:
    """[completed_steps, grad_step, usable, open] as integers; -1, -1 for
    the two steps when no trainer is open. Reads fields only.

    DEVIATION 3010 appends six more, all `len()` of buffers the trainer
    already owns (`byte_attention_eager_cells`, training/byte_lm.mojo):
    forward eager cells, forward `aexp` cells, backward eager cells,
    layers grown forward, layers grown backward, layers with a full
    `aexp`. They are ZERO when no trainer is open. Existing callers index
    positions 0 to 3 and are unaffected; nothing here launches, downloads
    or synchronizes. Open trainers append the forward/backward list lengths,
    then one triple per paired layer:
    forward launch status, backward launch status, current materialization.
    Status -1 means no fused attempt, otherwise the actual FUSED_* code.
    Policy flags, released cells and repair-site masks follow these triples;
    byte_attention_eager_cells documents their append-only ordering."""
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var values = byte_lm_status(owner[])
    var out = Python.list()
    for value in values:  # driver-loop: over completed native session metadata
        out.append(PythonObject(value))
    return out


def _byte_config(shape: PythonObject) raises -> ByteConfig:
    if len(shape) != 7 and len(shape) != 9 and len(shape) != 10:
        raise Error("byte LM: expected 7, 9 or 10 shape integers (B,L,DM,H,KV,HD,FF[,layers,vocab[,chunked_head_v2]])")
    var operator_module = Python.import_module("operator")
    var values = List[Int]()
    for i in range(len(shape)):  # small-loop(shape: 7 to 10 configuration words)
        var type_name = String(py=shape[i].__class__.__name__)
        if type_name == "bool" or type_name == "bool_":
            raise Error("byte LM: shape dimensions must be integers, not booleans")
        values.append(Int(py=py_call(operator_module.index, shape[i])))
    if len(values) == 7:
        values.append(2)
        values.append(256)
    var chunked = False
    if len(values) == 10:
        if values[9] != 0 and values[9] != 1:
            raise Error("byte LM: chunked head selector must be 0 or 1")
        chunked = values[9] == 1
    var cfg = ByteConfig(values[0], values[1], values[2], values[3],
                         values[4], values[5], values[6], values[7], values[8], chunked)
    cfg.validate()
    return cfg^


def byte_lm_config_profile_binding(shape: PythonObject) raises -> PythonObject:
    """Host-only shape admission and profile negotiation; creates no context."""
    var cfg = _byte_config(shape)
    return PythonObject(cfg.profile())


def byte_lm_run_binding(addresses: PythonObject, params: PythonObject) raises -> PythonObject:
    """Preserved v1 ABI: the original B2/L32/DM32 profile."""
    var session = ByteLMSession()
    return _byte_lm_run(addresses, params, ByteConfig(), session)


def byte_lm_run_configured_binding(addresses: PythonObject, params: PythonObject,
                                  shape: PythonObject) raises -> PythonObject:
    """Runtime shape ABI; validate lengths before dereferencing any address."""
    var cfg = _byte_config(shape)
    var session = ByteLMSession()
    return _byte_lm_run(addresses, params, cfg, session)


# ---------------------------------------------------------------------------
# DEVIATION 2658: forward-only logits (training/byte_lm_logits.mojo).
# ---------------------------------------------------------------------------


def _logits_dims(dims: PythonObject, shape: ByteConfig) raises -> List[Int]:
    """[batch, length] as integers (never booleans), in bounds, before any
    address is read."""
    if len(dims) != 2:
        raise Error("byte LM logits: expected dims [batch, length]")
    var operator_module = Python.import_module("operator")
    var out = List[Int]()
    for i in range(2):  # small-loop(dims: batch and length words)
        var type_name = String(py=dims[i].__class__.__name__)
        if type_name == "bool" or type_name == "bool_":
            raise Error("byte LM logits: dims must be integers, not booleans")
        out.append(Int(py=py_call(operator_module.index, dims[i])))
    if out[0] < 1 or out[0] > BYTE_LOGITS_MAX_BATCH or out[1] < 1 or out[1] > shape.length:
        raise Error("byte LM logits: batch in [1, " + String(BYTE_LOGITS_MAX_BATCH)
                    + "] and length in [1, " + String(shape.length) + "]")
    if out[0] * out[1] > BYTE_LOGITS_MAX_CELLS // shape.vocab_size:
        raise Error("byte LM logits: batch * length * vocab exceeds the admitted span")
    return out^


def byte_lm_logits_binding(addresses: PythonObject, dims: PythonObject,
                           shape: PythonObject) raises -> PythonObject:
    """Stateless forward-only logits. addresses[3] = [in_param_f32
    (n_total), in_ids_i32 (batch * length), out_logits_f32 (batch * length *
    vocab)]; dims = [batch, length]. Every input is read and admitted before
    any device work; the context is created, used, drained of its pending
    frees (DEVIATION 2520) and destroyed before anything is published.
    Returns the number of logits written."""
    _require_binding_profile()
    var cfg = _byte_config(shape)
    var bl = _logits_dims(dims, cfg)
    var addr = _read_addresses(addresses, 3)
    var written = 0
    with GILReleased(Python()):
        written = byte_lm_stateless_logits_native(addr, bl[0], bl[1], cfg)
    return PythonObject(written)


def _claimed_step(completed: PythonObject) raises -> Int:
    var claimed = Int(py=Python.import_module("operator").index(completed))
    if claimed < 0 or claimed >= 1000000:
        raise Error("byte LM: completed step outside admitted bound")
    return claimed


def byte_lm_session_logits_binding(session: PythonObject, addresses: PythonObject,
                                   dims: PythonObject, shape: PythonObject,
                                   completed: PythonObject) raises -> PythonObject:
    """Forward-only logits on an open resident session. addresses[2] =
    [in_ids_i32 (batch * length), out_logits_f32 (batch * length * vocab)];
    dims = [batch, length]; `completed` is the caller's committed step, which
    must equal the session's. Writes no parameter, moment, flag or step.
    Returns the number of logits written. The work is `byte_session_logits`."""
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var bl = _logits_dims(dims, cfg_shape)
    var claimed = _claimed_step(completed)
    var addr = _read_addresses(addresses, 2)
    var result = 0
    with GILReleased(Python()):
        result = byte_session_logits(owner[], addr, bl[0], bl[1], claimed, cfg_shape, False)
    return PythonObject(result)


def byte_lm_session_next_bytes_binding(session: PythonObject, addresses: PythonObject,
                                       dims: PythonObject, shape: PythonObject,
                                       completed: PythonObject) raises -> PythonObject:
    """cpu3-seq (2026-10-04): the greedy next byte of every row on an open
    resident session. addresses[2] = [in_ids_i32 (batch * length),
    out_next_i32 (batch)]; dims = [batch, length]; `completed` as in
    `byte_lm_session_logits`. The logits stay on the device. Returns
    `batch`. The work is `byte_session_logits`."""
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var bl = _logits_dims(dims, cfg_shape)
    var claimed = _claimed_step(completed)
    var addr = _read_addresses(addresses, 2)
    var result = 0
    with GILReleased(Python()):
        result = byte_session_logits(owner[], addr, bl[0], bl[1], claimed, cfg_shape, True)
    return PythonObject(result)


# ---------------------------------------------------------------------------
# The single-device trainer owner (mojolearn.SmallByteLanguageModelTrainer):
# one entry per public method; each unpacks its arguments, calls ONE typed
# owner function (training/byte_lm_session_ops.mojo) and returns its typed
# result. Nothing is assigned to a Python object here.
# ---------------------------------------------------------------------------


def _read_address_list(addresses: PythonObject) raises -> List[Int]:
    var addr = List[Int]()
    for i in range(len(addresses)):  # small-loop(addresses: the call's address slots, one per buffer): reads the addresses, no data
        addr.append(Int(py=addresses[i]))
    return addr^


def byte_lm_trainer_step_binding(session: PythonObject, addresses: PythonObject,
                                 params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """`train_step` (params[0] = 1) or `evaluate` (params[0] = 0) of a
    resident trainer in one call: admission of the host state when the
    session is not open, the step or evaluation, and the gradient export
    split by name when out_grad is given. addresses as `byte_owner_step`;
    params[12] as `_byte_lm_run`. Returns the completed step after the
    call; the loss, flags and gradients are written to the outputs."""
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var action = _params_action(params)
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_address_list(addresses)
    var result = completed
    with GILReleased(Python()):
        result = byte_owner_step(owner[], addr, completed, cfg, cfg_shape, action == 1)
    return PythonObject(result)


def byte_lm_trainer_export_state_binding(session: PythonObject, addresses: PythonObject,
                                         shape: PythonObject, completed: PythonObject,
                                         release: PythonObject) raises -> PythonObject:
    """The trainer's state export in one call. addresses[5] = [in_flags_i32,
    out_param, out_m, out_v, out_flags_i32]; `completed` the committed step;
    a true `release` closes the session after the export. Returns the
    exported step. The work is `byte_owner_export_state`."""
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var claimed = _claimed_step(completed)
    var close = Bool(py=release)
    var addr = _read_addresses(addresses, 5)
    var result = 0
    with GILReleased(Python()):
        result = byte_owner_export_state(owner[], addr, claimed, cfg_shape, close)
    return PythonObject(result)


def byte_lm_trainer_export_gradients_binding(session: PythonObject, addresses: PythonObject,
                                             shape: PythonObject, completed: PythonObject) raises -> PythonObject:
    """The trainer's gradient export in one call. addresses = [out_grad]
    and, for the gradient by name, one output per parameter tensor. Returns
    the gradient's step, which must be `completed`. The work is
    `byte_owner_export_gradients`."""
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var claimed = _claimed_step(completed)
    var addr = _read_address_list(addresses)
    var result = 0
    with GILReleased(Python()):
        result = byte_owner_export_gradients(owner[], addr, claimed, cfg_shape)
    return PythonObject(result)


def _byte_lm_trainer_logits(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                            params: PythonObject, shape: PythonObject, next_bytes: Bool) raises -> PythonObject:
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var bl = _logits_dims(dims, cfg_shape)
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 6)
    var result = 0
    with GILReleased(Python()):
        result = byte_owner_logits(owner[], addr, bl[0], bl[1], completed, cfg, cfg_shape, next_bytes)
    return PythonObject(result)


def byte_lm_trainer_logits_binding(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                                   params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """The trainer's `logits` in one call. addresses[6] = [in_ids_i32,
    out_logits_f32, in_flags_i32, in_param, in_m, in_v] (the last three 0
    on an open session); dims = [batch, length]; params[12] as
    `_byte_lm_run` (the committed step and optimizer the admission uses).
    Returns the number of logits written."""
    return _byte_lm_trainer_logits(session, addresses, dims, params, shape, False)


def byte_lm_trainer_next_bytes_binding(session: PythonObject, addresses: PythonObject, dims: PythonObject,
                                       params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """The trainer's `next_bytes` in one call: as `byte_lm_trainer_logits`
    with an int32 output of `batch` greedy next bytes. Returns `batch`."""
    return _byte_lm_trainer_logits(session, addresses, dims, params, shape, True)


def byte_lm_trainer_run_binding(addresses: PythonObject, named: PythonObject,
                                params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """`train_step` / `evaluate` of a stateless trainer in one call:
    `byte_lm_run_configured` (addresses[11], params[12]), then, for a
    training step with `named` holding one output per parameter tensor, the
    returned gradient split by name. Returns the completed step."""
    var cfg = _byte_config(shape)
    var named_addr = _read_address_list(named)
    if len(named_addr) > 0:
        if Int(py=params[0]) != 1:
            raise Error("byte LM trainer: an evaluation returns no gradient")
        var slots: List[Int] = [Int(py=addresses[8])]
        var cells: List[Int] = [cfg.n_total()]
        for j in range(len(named_addr)):  # driver-loop: over model parameter tensors
            slots.append(named_addr[j])
            cells.append(cfg.param_count(j))
        if len(named_addr) != cfg.n_tensors():
            raise Error("byte LM: expected one named output per parameter tensor")
        _validate_slot_table(slots, cells, 0)
    var session = ByteLMSession()
    var result = _byte_lm_run(addresses, params, cfg, session)
    if len(named_addr) > 0:
        byte_scatter_named(Int(py=addresses[8]), named_addr, cfg)
    return result


def sha256_hex_binding(value: PythonObject) raises -> PythonObject:
    """hashlib.sha256(value).hexdigest() for a bytes-like value, by the Mojo SHA-256."""
    return sha256_hexdigest(value)


def byte_lm_file_sha256_binding(path: PythonObject) raises -> PythonObject:
    """The SHA-256 of the file at `path`, streamed by `byte_file_sha256`."""
    var name = String(py=path)
    var hex = String()
    with GILReleased(Python()):
        hex = byte_file_sha256(name)
    return PythonObject(hex)


def byte_lm_parallel_create_binding() raises -> PythonObject:
    return PythonObject(alloc=ByteParallelTrainer())


def byte_lm_parallel_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].close()
    return PythonObject(0)


def byte_lm_parallel_open_binding[pooled: Bool = False](session: PythonObject, addresses: PythonObject,
    params: PythonObject, shape: PythonObject, devices: PythonObject, shard_count: PythonObject) raises -> PythonObject:
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 4)
    var cells: List[Int] = [cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_tensors()]
    _validate_slot_table(addr, cells, 4)
    var p = _read_f32(addr[0], cells[0])
    var m = _read_f32(addr[1], cells[1])
    var v = _read_f32(addr[2], cells[2])
    var flags = _read_flags(addr[3], cells[3])
    byte_validate_state(p, m, v, flags, completed, cfg_shape)
    var device_ids = List[Int]()
    for i in range(Int(py=devices.__len__())):  # small-loop(devices: GPU device ids of the replicas): one id per device, not data
        device_ids.append(Int(py=devices[i]))
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    # Retain the GIL: it is also the native object's exclusion lock.
    owner[].open(device_ids, Int(py=shard_count), p, m, v, flags, completed, cfg, cfg_shape, pooled)
    return PythonObject(completed)


def byte_lm_parallel_step_binding(session: PythonObject, addresses: PythonObject,
    completed_arg: PythonObject) raises -> PythonObject:
    var completed = Int(py=completed_arg)
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    if completed != owner[].trainers[0].completed_steps:
        raise Error("byte LM parallel: completed-step mismatch")
    var count = owner[].logical_shards
    var addr = _read_addresses(addresses, count)
    var shape = owner[].trainers[0].config.copy()
    var n_ids = shape.batch * (shape.length + 1)
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: logical shards of the step): per shard slot sizes, no data
        cells.append(n_ids)
    _validate_slot_table(addr, cells, count)
    var shards = List[List[Int32]]()
    for i in range(count):  # small-loop(count: logical shards of the step): one memcpy of each shard's ids, the device checks them
        shards.append(_read_ids(addr[i], n_ids))
    var losses = owner[].step(shards)
    var out = Python.list()
    try:
        for i in range(len(losses)):  # small-loop(losses: one loss per shard): returns the shard losses
            out.append(PythonObject(losses[i]))
    except error:
        owner[].rollback()
        raise error
    return out


def byte_lm_parallel_shard_gradient_binding(session: PythonObject, addresses: PythonObject,
    completed_arg: PythonObject) raises -> PythonObject:
    """addresses = [ids int32[B*(L+1)], out float32[n]]; returns the loss. No update."""
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    if Int(py=completed_arg) != owner[].trainers[0].completed_steps:
        raise Error("byte LM parallel: completed-step mismatch")
    var shape = owner[].trainers[0].config.copy()
    var n = shape.n_total()
    var n_ids = shape.batch * (shape.length + 1)
    var addr = _read_addresses(addresses, 2)
    var cells: List[Int] = [n_ids, n]
    _validate_slot_table(addr, cells, 1)
    var loss = owner[].shard_gradient(_read_ids(addr[0], n_ids))
    download_f32_into(owner[].contexts[0], owner[].trainers[0].buffers.grad, n, f32_ptr(addr[1]))  # DEVIATION 3120
    return PythonObject(loss)


def byte_lm_parallel_apply_gradient_binding(session: PythonObject, addresses: PythonObject,
    completed_arg: PythonObject) raises -> PythonObject:
    """addresses = [summed float32[n]]; commits one step; returns completed steps."""
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    if Int(py=completed_arg) != owner[].trainers[0].completed_steps:
        raise Error("byte LM parallel: completed-step mismatch")
    var n = owner[].trainers[0].config.n_total()
    var addr = _read_addresses(addresses, 1)
    var cells: List[Int] = [n]
    _validate_slot_table(addr, cells, 1)
    owner[].apply_gradient(f32_ptr(addr[0]))  # lane cpu4-python: straight to the device
    return PythonObject(owner[].trainers[0].completed_steps)


def byte_lm_parallel_export_binding(session: PythonObject, addresses: PythonObject,
    rank_arg: PythonObject, gradients_arg: PythonObject) raises -> PythonObject:
    var rank = Int(py=rank_arg)
    var gradients = Bool(gradients_arg)
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    if rank < 0 or rank >= len(owner[].trainers):
        raise Error("byte LM parallel: invalid replica index")
    ref tr = owner[].trainers[rank]
    ref ctx = owner[].contexts[rank]
    var n = tr.config.n_total()
    var count = 1 if gradients else 4
    var addr = _read_addresses(addresses, count)
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: one or four exported arrays): per array cell counts, not data
        cells.append(tr.config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr, cells, 0)
    tr.validate_device_state(ctx, tr.completed_steps)
    if owner[].pool_optimizer:
        for i in range(len(owner[].trainers)):  # driver-loop: over replica trainers
            owner[].trainers[i].validate_device_state(owner[].contexts[i], tr.completed_steps)
    # DEVIATION 3120: every array goes from the device straight into the
    # caller's memory (`download_f32_into`, one transfer each): no pinned
    # host buffer, no element loop into a List, no second host copy. The
    # pooled m/v land at each owner's `optimizer_first`, which is where the
    # old List concatenation put them (owners are contiguous, in order,
    # and cover [0, n); checked below before any copy). On a raise the
    # caller's arrays hold an unspecified partial export.
    if gradients:
        if tr.grad_step != tr.completed_steps:
            raise Error("byte LM parallel: no committed gradient")
        download_f32_into(ctx, tr.buffers.grad, n, f32_ptr(addr[0]))
    else:
        if owner[].pool_optimizer:
            var at = 0
            for i in range(len(owner[].trainers)):  # small-loop(trainers: one trainer per replica device): checks range offsets per replica, not data
                if owner[].trainers[i].buffers.optimizer_first != at:
                    raise Error("byte LM parallel: pooled optimizer ranges are not contiguous")
                at += owner[].trainers[i].buffers.optimizer_count
            if at != n:
                raise Error("byte LM parallel: pooled optimizer ranges do not cover the model")
        download_f32_into(ctx, tr.buffers.param, n, f32_ptr(addr[0]))
        if owner[].pool_optimizer:
            for i in range(len(owner[].trainers)):  # driver-loop: over replica trainers, one download each
                var first = owner[].trainers[i].buffers.optimizer_first
                var owned = owner[].trainers[i].buffers.optimizer_count
                download_f32_into(owner[].contexts[i], owner[].trainers[i].buffers.m_state, owned,
                                  f32_ptr(addr[1] + 4 * first))
                download_f32_into(owner[].contexts[i], owner[].trainers[i].buffers.v_state, owned,
                                  f32_ptr(addr[2] + 4 * first))
        else:
            download_f32_into(ctx, tr.buffers.m_state, n, f32_ptr(addr[1]))
            download_f32_into(ctx, tr.buffers.v_state, n, f32_ptr(addr[2]))
        _write_flags(addr[3], tr.buffers.buf_initialized)
    return PythonObject(tr.completed_steps)


def byte_lm_pool_fault_available_binding() raises -> PythonObject:
    return PythonObject(pool_fault_available())


def byte_lm_parallel_reduction_pool_available_binding() raises -> PythonObject:
    return PythonObject(1)


def byte_lm_parallel_ownership_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    var out = Python.list()
    for i in range(len(owner[].trainers)):  # small-loop(trainers: one trainer per replica device): reports per replica ranges, no data
        ref b = owner[].trainers[i].buffers
        var row = Python.list()
        row.append(PythonObject(b.optimizer_first))
        row.append(PythonObject(b.optimizer_count))
        row.append(PythonObject(4*(len(b.m_state)+len(b.v_state))))
        row.append(PythonObject(4*(len(b.shadow_p)+len(b.shadow_m)+len(b.shadow_v))))
        var reduction_bytes = 0
        if owner[].pool_optimizer:
            reduction_bytes = 4*(len(owner[].pool_totals[i])+len(owner[].pool_incoming[i]))
        elif i == 0:
            reduction_bytes = 4*len(owner[].total.value())
        row.append(PythonObject(reduction_bytes))
        out.append(row)
    return out


def byte_lm_parallel_set_lr_binding(session: PythonObject, lr_arg: PythonObject) raises -> PythonObject:
    """Set the learning rate of every replica for the NEXT update (a
    per-step schedule); returns the float32 bits that were stored, so the
    caller can hold the recorded schedule to what the device will use."""
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    var lr = Float32(Float64(py=lr_arg))
    owner[].set_learning_rate(lr)
    return PythonObject(Int(bitcast[DType.uint32](owner[].trainers[0].optimizer.lr)))


def byte_lm_parallel_fold_reset_binding(session: PythonObject, addresses: PythonObject) raises -> PythonObject:
    """addresses = [] (an empty fold) or [prefix float32[n]]: start this worker's
    device fold empty or from a received prefix."""
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    var n = owner[].trainers[0].config.n_total()
    if Int(py=addresses.__len__()) == 0:
        owner[].fold_clear()
        return PythonObject(0)
    var addr = _read_addresses(addresses, 1)
    var cells: List[Int] = [n]
    _validate_slot_table(addr, cells, 1)
    owner[].fold_reset(f32_ptr(addr[0]))  # lane cpu4-python: straight to the device
    return PythonObject(1)


def byte_lm_parallel_shard_gradient_fold_binding(session: PythonObject, addresses: PythonObject,
    completed_arg: PythonObject) raises -> PythonObject:
    """addresses = [ids int32[B*(L+1)]]: one shard's gradient, folded into the
    device total with the step's own ordered add; returns the loss."""
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    if Int(py=completed_arg) != owner[].trainers[0].completed_steps:
        raise Error("byte LM parallel: completed-step mismatch")
    var shape = owner[].trainers[0].config.copy()
    var n_ids = shape.batch * (shape.length + 1)
    var addr = _read_addresses(addresses, 1)
    var cells: List[Int] = [n_ids]
    _validate_slot_table(addr, cells, 1)
    return PythonObject(owner[].shard_gradient_fold(_read_ids(addr[0], n_ids)))


def byte_lm_parallel_fold_add_binding(session: PythonObject, addresses: PythonObject) raises -> PythonObject:
    """addresses = [gradient float32[n]]: fold a host-held gradient into the device total."""
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    var n = owner[].trainers[0].config.n_total()
    var addr = _read_addresses(addresses, 1)
    var cells: List[Int] = [n]
    _validate_slot_table(addr, cells, 1)
    owner[].fold_add(f32_ptr(addr[0]))  # lane cpu4-python: straight to the device
    return PythonObject(0)


def byte_lm_parallel_fold_export_binding(session: PythonObject, addresses: PythonObject) raises -> PythonObject:
    """addresses = [out float32[n]]: the device fold's total."""
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    var n = owner[].trainers[0].config.n_total()
    var addr = _read_addresses(addresses, 1)
    var cells: List[Int] = [n]
    _validate_slot_table(addr, cells, 1)
    owner[].fold_export_into(f32_ptr(addr[0]))  # DEVIATION 3120
    return PythonObject(n)


def byte_lm_parallel_rollback_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    owner[].rollback()
    return PythonObject(owner[].trainers[0].completed_steps)


def byte_lm_model_pool_create_binding() raises -> PythonObject:
    return PythonObject(alloc=ByteModelPool())


def byte_lm_model_pool_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteModelPool]()
    owner[].close()
    return PythonObject(0)


def byte_lm_model_pool_open_binding(session: PythonObject, addresses: PythonObject,
    params: PythonObject, shape: PythonObject, devices: PythonObject, shard_count: PythonObject) raises -> PythonObject:
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 4)
    var cells: List[Int] = [cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_tensors()]
    _validate_slot_table(addr, cells, 4)
    var p = _read_f32(addr[0], cells[0])
    var m = _read_f32(addr[1], cells[1])
    var v = _read_f32(addr[2], cells[2])
    var flags = _read_flags(addr[3], cells[3])
    byte_validate_state(p, m, v, flags, completed, cfg_shape)
    var device_ids = List[Int]()
    for i in range(Int(py=devices.__len__())):  # small-loop(devices: the caller's device list): reads device ordinals, no data
        device_ids.append(Int(py=devices[i]))
    var owner = session.downcast_value_ptr[ByteModelPool]()
    # Retain the GIL: it is also the native object's exclusion lock.
    owner[].open(device_ids, Int(py=shard_count), p, m, v, flags, completed, cfg, cfg_shape)
    return PythonObject(completed)


def byte_lm_model_pool_step_binding(session: PythonObject, addresses: PythonObject,
    completed_arg: PythonObject) raises -> PythonObject:
    var completed = Int(py=completed_arg)
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteModelPool]()
    owner[].require_open()
    if completed != owner[].completed:
        raise Error("byte LM parallel: completed-step mismatch")
    var count = owner[].logical_shards
    var addr = _read_addresses(addresses, count)
    var shape = owner[].config.copy()
    var n_ids = shape.batch * (shape.length + 1)
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: logical shards of the step): per shard slot sizes, no data
        cells.append(n_ids)
    _validate_slot_table(addr, cells, count)
    var shards = List[List[Int32]]()
    for i in range(count):  # small-loop(count: logical shards of the step): one memcpy of each shard's ids, the device checks them
        shards.append(_read_ids(addr[i], n_ids))
    var losses = owner[].step(shards)
    try:
        var out = Python.list()
        for i in range(len(losses)):  # small-loop(losses: one loss per shard): returns the shard losses
            out.append(PythonObject(losses[i]))
        return out
    except error:
        owner[].rollback()
        raise error


def byte_lm_model_pool_export_binding(session: PythonObject, addresses: PythonObject,
    rank_arg: PythonObject, gradients_arg: PythonObject) raises -> PythonObject:
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteModelPool]()
    owner[].require_open()
    if Int(py=rank_arg) != 0:
        raise Error("byte model pool: canonical export has no replica rank; use rank=0")
    var gradients = Bool(gradients_arg)
    var n = owner[].config.n_total()
    var count = 1 if gradients else 4
    var addr = _read_addresses(addresses,count)
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: one or four exported arrays): per array slot sizes, no data
        cells.append(owner[].config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr,cells,0)
    if gradients:
        var g = owner[].export_values(3)
        copy_f32(g.unsafe_ptr(),f32_ptr(addr[0]),n)
    else:
        # Complete and validate all readbacks before publishing any output.
        var p = owner[].export_values(0)
        var m = owner[].export_values(1)
        var v = owner[].export_values(2)
        copy_f32(p.unsafe_ptr(),f32_ptr(addr[0]),n)
        copy_f32(m.unsafe_ptr(),f32_ptr(addr[1]),n)
        copy_f32(v.unsafe_ptr(),f32_ptr(addr[2]),n)
        _write_flags(addr[3],owner[].flags)
    return PythonObject(owner[].completed)


def byte_lm_model_pool_rollback_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteModelPool]()
    owner[].require_open()
    owner[].rollback()
    return PythonObject(owner[].completed)


def byte_lm_model_pool_ownership_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteModelPool]()
    owner[].require_open()
    var out = Python.list()
    for i in range(len(owner[].chunks)):  # small-loop(chunks: one model chunk per device): reports per chunk ranges, no data
        ref chunk = owner[].chunks[i]
        var row = Python.list()
        row.append(PythonObject(chunk.owner))
        row.append(PythonObject(chunk.first))
        row.append(PythonObject(len(chunk.p)))
        row.append(PythonObject(4*len(chunk.p)))
        row.append(PythonObject(4*(len(chunk.m)+len(chunk.v))))
        row.append(PythonObject(4*(len(chunk.shadow_p)+len(chunk.shadow_m)+len(chunk.shadow_v))))
        row.append(PythonObject(4*len(chunk.g)))
        out.append(row)
    return out


def byte_lm_offload_create_binding() raises -> PythonObject:
    return PythonObject(alloc=ByteOffloadedReplay())


def byte_lm_offload_close_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteOffloadedReplay]()
    owner[].close()
    return PythonObject(0)


def byte_lm_offload_open_binding(session: PythonObject, addresses: PythonObject,
    params: PythonObject, shape: PythonObject, devices: PythonObject, shard_count: PythonObject) raises -> PythonObject:
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 4)
    var cells: List[Int] = [cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_total(), cfg_shape.n_tensors()]
    _validate_slot_table(addr, cells, 4)
    var p = _read_f32(addr[0], cells[0])
    var m = _read_f32(addr[1], cells[1])
    var v = _read_f32(addr[2], cells[2])
    var flags = _read_flags(addr[3], cells[3])
    byte_validate_state(p, m, v, flags, completed, cfg_shape)
    var device_ids = List[Int]()
    for i in range(Int(py=devices.__len__())):  # small-loop(devices: the caller's device list): reads device ordinals, no data
        device_ids.append(Int(py=devices[i]))
    var owner = session.downcast_value_ptr[ByteOffloadedReplay]()
    # Retain the GIL: it is also the native object's exclusion lock.
    owner[].open(device_ids, Int(py=shard_count), p, m, v, flags, completed, cfg, cfg_shape)
    return PythonObject(completed)


def byte_lm_offload_step_binding(session: PythonObject, addresses: PythonObject,
    completed_arg: PythonObject) raises -> PythonObject:
    var completed = Int(py=completed_arg)
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteOffloadedReplay]()
    owner[].require_open()
    if completed != owner[].completed:
        raise Error("byte LM parallel: completed-step mismatch")
    var count = owner[].logical_shards
    var addr = _read_addresses(addresses, count)
    var shape = owner[].config.copy()
    var n_ids = shape.batch * (shape.length + 1)
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: logical shards of the step): per shard slot sizes, no data
        cells.append(n_ids)
    _validate_slot_table(addr, cells, count)
    var shards = List[List[Int32]]()
    for i in range(count):  # small-loop(count: logical shards of the step): one memcpy of each shard's ids, the device checks them
        shards.append(_read_ids(addr[i], n_ids))
    var losses = owner[].step(shards)
    try:
        var out = Python.list()
        for i in range(len(losses)):  # small-loop(losses: one loss per shard): returns the shard losses
            out.append(PythonObject(losses[i]))
        return out
    except error:
        owner[].rollback()
        raise error


def byte_lm_offload_export_binding(session: PythonObject, addresses: PythonObject,
    rank_arg: PythonObject, gradients_arg: PythonObject) raises -> PythonObject:
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteOffloadedReplay]()
    owner[].require_open()
    if Int(py=rank_arg) != 0:
        raise Error("byte model pool: canonical export has no replica rank; use rank=0")
    var gradients = Bool(gradients_arg)
    var n = owner[].config.n_total()
    var count = 1 if gradients else 4
    var addr = _read_addresses(addresses,count)
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: one or four exported arrays): per array slot sizes, no data
        cells.append(owner[].config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr,cells,0)
    if gradients:
        var g = owner[].export_values(3)
        copy_f32(g.unsafe_ptr(),f32_ptr(addr[0]),n)
    else:
        # Complete and validate all readbacks before publishing any output.
        var p = owner[].export_values(0)
        var m = owner[].export_values(1)
        var v = owner[].export_values(2)
        copy_f32(p.unsafe_ptr(),f32_ptr(addr[0]),n)
        copy_f32(m.unsafe_ptr(),f32_ptr(addr[1]),n)
        copy_f32(v.unsafe_ptr(),f32_ptr(addr[2]),n)
        _write_flags(addr[3],owner[].flags)
    return PythonObject(owner[].completed)


def byte_lm_offload_rollback_binding(session: PythonObject) raises -> PythonObject:
    var owner = session.downcast_value_ptr[ByteOffloadedReplay]()
    owner[].require_open()
    owner[].rollback()
    return PythonObject(owner[].completed)


# TOMBSTONE: byte_lm_launch_probe (dead native entry: tool-only launch-cost probe; byte_lm_session_launch_probe with it) deleted 2026-10-26 by be2f9a1be; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/byte_lm_launch_probe.patch


# ---------------------------------------------------------------------------
# Private byte-LM schedule descriptor and factory (Program pooled/offload lane,
# 2026-10-28). One compiled table drives the replica, layer-pool and offload
# sessions above with integer descriptors only: the tokenizer-host owner
# (bindings/parallel_byte_native.mojo) retains this image's provider, the
# session value minted by byte_lm_schedule_create_v1 and its handle, and never
# selects per-instance Python callables. The numerical sessions are unchanged.
# ---------------------------------------------------------------------------
comptime BYTE_LM_SCHEDULE_ABI_NAME = "mojolearn.training.byte_lm_schedule.v1"
comptime BYTE_LM_SCHEDULE_OPERATION_COUNT = 15
comptime BYTE_SCHEDULE_REPLICA = 0
comptime BYTE_SCHEDULE_MODEL_POOL = 1
comptime BYTE_SCHEDULE_OFFLOAD = 2
comptime BYTE_SCHEDULE_MAGIC = 0x42594C4D53434844
comptime BYTE_SCHEDULE_CLOSE = 0
comptime BYTE_SCHEDULE_OPEN = 1
comptime BYTE_SCHEDULE_STEP = 2
comptime BYTE_SCHEDULE_EXPORT = 3
comptime BYTE_SCHEDULE_ROLLBACK = 4
comptime BYTE_SCHEDULE_OWNERSHIP = 5
comptime BYTE_SCHEDULE_SET_LR = 6
comptime BYTE_SCHEDULE_SHARD_GRADIENT = 7
comptime BYTE_SCHEDULE_APPLY_GRADIENT = 8
comptime BYTE_SCHEDULE_FOLD_RESET = 9
comptime BYTE_SCHEDULE_SHARD_GRADIENT_FOLD = 10
comptime BYTE_SCHEDULE_FOLD_ADD = 11
comptime BYTE_SCHEDULE_FOLD_EXPORT = 12
comptime BYTE_SCHEDULE_KIND = 13
comptime BYTE_SCHEDULE_CAPABILITY = 14


struct ByteScheduleSession(Movable, Writable):
    """The one session value of a byte-LM schedule. Construction is lazy:
    no device work until OPEN. Only `kind`'s member is ever opened."""
    var magic: Int
    var kind: Int
    var replica: ByteParallelTrainer
    var pool: ByteModelPool
    var offload: ByteOffloadedReplay

    def __init__(out self, kind: Int):
        self.magic = BYTE_SCHEDULE_MAGIC
        self.kind = kind
        self.replica = ByteParallelTrainer()
        self.pool = ByteModelPool()
        self.offload = ByteOffloadedReplay()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteScheduleSession(kind=", self.kind, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _byte_schedule_kind(value: Int) raises -> Int:
    if value != BYTE_SCHEDULE_REPLICA and value != BYTE_SCHEDULE_MODEL_POOL and value != BYTE_SCHEDULE_OFFLOAD:
        raise Error("byte LM schedule: unknown schedule kind")
    return value


def _byte_schedule_capability(kind: Int) -> Int:
    # Bit i set: operation i is implemented for this schedule. The replica
    # session has every operation; the layer pool has no split-step/fold
    # door; the offloaded replay also has no resident optimizer ownership.
    if kind == BYTE_SCHEDULE_REPLICA:
        return (1 << BYTE_LM_SCHEDULE_OPERATION_COUNT) - 1
    var common = (1 << BYTE_SCHEDULE_CLOSE) | (1 << BYTE_SCHEDULE_OPEN) | (1 << BYTE_SCHEDULE_STEP) | (
        1 << BYTE_SCHEDULE_EXPORT) | (1 << BYTE_SCHEDULE_ROLLBACK) | (1 << BYTE_SCHEDULE_KIND) | (1 << BYTE_SCHEDULE_CAPABILITY)
    if kind == BYTE_SCHEDULE_MODEL_POOL:
        return common | (1 << BYTE_SCHEDULE_OWNERSHIP)
    return common


def _byte_schedule_session(address: Int) raises -> MutPointer[ByteScheduleSession, MutUntrackedOrigin]:
    if address == 0:
        raise Error("byte LM schedule: missing admitted session")
    var session = MutPointer[ByteScheduleSession, MutUntrackedOrigin](unsafe_from_address=address)
    if session[].magic != BYTE_SCHEDULE_MAGIC:
        raise Error("byte LM schedule: handle is not a schedule session")
    _ = _byte_schedule_kind(session[].kind)
    return session


def _byte_schedule_shards(addresses: List[Int], first: Int, count: Int, shape: ByteConfig) raises -> List[List[Int32]]:
    if count < 1 or len(addresses) != first + count:
        raise Error("byte LM parallel: logical shard count mismatch")
    var n_ids = shape.batch * (shape.length + 1)
    var slots = List[Int]()
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: logical shards of the step): per shard slot sizes, no data
        slots.append(addresses[first + i])
        cells.append(n_ids)
    _validate_slot_table(slots, cells, count)
    var shards = List[List[Int32]]()
    for i in range(count):  # small-loop(count: logical shards of the step): one memcpy of each shard's ids, the device checks them
        shards.append(_read_ids(slots[i], n_ids))
    return shards^


def _byte_schedule_losses(address: Int, losses: List[Float32], count: Int) raises -> Int:
    if len(losses) != count:
        raise Error("RuntimeError: parallel step returned wrong loss count")
    var out = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # small-loop(count: one loss per logical shard): copies shard losses, no data
        out.store(i, Float64(losses[i]))
    return count


def _byte_schedule_open(mut session: ByteScheduleSession, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    # a = [handle, parameters, m, v, flags]; p = [completed, kind, nesterov,
    # pooled, n_dims, dims..., shards, n_devices, devices...]; f = the eight
    # optimizer scalars (lr, beta1, beta2, eps, weight_decay, momentum,
    # dampening, max_norm).
    _require_binding_profile()
    if len(a) != 5 or len(f) != 8 or len(p) < 6:
        raise Error("byte LM schedule: invalid open descriptor")
    var completed = p[0]
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step outside admitted bound")
    if p[2] != 0 and p[2] != 1 or p[3] != 0 and p[3] != 1:
        raise Error("byte LM: action/nesterov must be 0 or 1")
    var n_dims = p[4]
    if n_dims != 9 and n_dims != 10 or len(p) < 5 + n_dims + 2:
        raise Error("byte LM schedule: invalid shape descriptor")
    var chunked = False
    if n_dims == 10:
        if p[14] != 0 and p[14] != 1:
            raise Error("byte LM: chunked head selector must be 0 or 1")
        chunked = p[14] == 1
    var shape = ByteConfig(p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], chunked)
    shape.validate()
    var at = 5 + n_dims
    var shards = p[at]
    var n_devices = p[at + 1]
    if n_devices < 1 or len(p) != at + 2 + n_devices:
        raise Error("byte LM schedule: invalid device descriptor")
    var device_ids = List[Int]()
    for i in range(n_devices):  # small-loop(n_devices: GPU device ordinals of the session): ids, not data
        device_ids.append(p[at + 2 + i])
    var cfg = OptimizerConfig(p[1], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]),
        Float32(f[5]), Float32(f[6]), p[2] == 1, Float32(f[7]))
    byte_validate_optimizer(cfg)
    var addr: List[Int] = [a[1], a[2], a[3], a[4]]
    var cells: List[Int] = [shape.n_total(), shape.n_total(), shape.n_total(), shape.n_tensors()]
    _validate_slot_table(addr, cells, 4)
    var params = _read_f32(addr[0], cells[0])
    var m = _read_f32(addr[1], cells[1])
    var v = _read_f32(addr[2], cells[2])
    var flags = _read_flags(addr[3], cells[3])
    byte_validate_state(params, m, v, flags, completed, shape)
    # Retain the GIL: it is also the native object's exclusion lock.
    if session.kind == BYTE_SCHEDULE_REPLICA:
        session.replica.open(device_ids, shards, params, m, v, flags, completed, cfg, shape, p[3] == 1)
    elif session.kind == BYTE_SCHEDULE_MODEL_POOL:
        if p[3] != 0:
            raise Error("byte LM schedule: optimizer pooling belongs to the replica schedule")
        session.pool.open(device_ids, shards, params, m, v, flags, completed, cfg, shape)
    else:
        if p[3] != 0:
            raise Error("byte LM schedule: optimizer pooling belongs to the replica schedule")
        session.offload.open(device_ids, shards, params, m, v, flags, completed, cfg, shape)
    return completed


def _byte_schedule_export_canonical(mut session: ByteScheduleSession, a: List[Int], gradients: Bool) raises -> Int:
    # Layer-pool and offload export: the canonical state, no replica rank.
    var config = session.pool.config.copy() if session.kind == BYTE_SCHEDULE_MODEL_POOL else session.offload.config.copy()
    var n = config.n_total()
    var count = 1 if gradients else 4
    if len(a) != 1 + count:
        raise Error("byte LM schedule: invalid export descriptor")
    var addr = List[Int]()
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: one or four exported arrays): per array slot sizes, no data
        addr.append(a[1 + i])
        cells.append(config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr, cells, 0)
    if session.kind == BYTE_SCHEDULE_MODEL_POOL:
        if gradients:
            var g = session.pool.export_values(3)
            copy_f32(g.unsafe_ptr(), f32_ptr(addr[0]), n)
        else:
            # Complete and validate all readbacks before publishing any output.
            var params = session.pool.export_values(0)
            var m = session.pool.export_values(1)
            var v = session.pool.export_values(2)
            copy_f32(params.unsafe_ptr(), f32_ptr(addr[0]), n)
            copy_f32(m.unsafe_ptr(), f32_ptr(addr[1]), n)
            copy_f32(v.unsafe_ptr(), f32_ptr(addr[2]), n)
            _write_flags(addr[3], session.pool.flags)
        return session.pool.completed
    if gradients:
        var g = session.offload.export_values(3)
        copy_f32(g.unsafe_ptr(), f32_ptr(addr[0]), n)
    else:
        var params = session.offload.export_values(0)
        var m = session.offload.export_values(1)
        var v = session.offload.export_values(2)
        copy_f32(params.unsafe_ptr(), f32_ptr(addr[0]), n)
        copy_f32(m.unsafe_ptr(), f32_ptr(addr[1]), n)
        copy_f32(v.unsafe_ptr(), f32_ptr(addr[2]), n)
        _write_flags(addr[3], session.offload.flags)
    return session.offload.completed


def _byte_schedule_export_replica(mut owner: ByteParallelTrainer, a: List[Int], rank: Int, gradients: Bool) raises -> Int:
    # The replica export of byte_lm_parallel_export_binding, over descriptors.
    if rank < 0 or rank >= len(owner.trainers):
        raise Error("byte LM parallel: invalid replica index")
    ref tr = owner.trainers[rank]
    ref ctx = owner.contexts[rank]
    var n = tr.config.n_total()
    var count = 1 if gradients else 4
    if len(a) != 1 + count:
        raise Error("byte LM schedule: invalid export descriptor")
    var addr = List[Int]()
    var cells = List[Int]()
    for i in range(count):  # small-loop(count: one or four exported arrays): per array cell counts, not data
        addr.append(a[1 + i])
        cells.append(tr.config.n_tensors() if i == 3 else n)
    _validate_slot_table(addr, cells, 0)
    tr.validate_device_state(ctx, tr.completed_steps)
    if owner.pool_optimizer:
        for i in range(len(owner.trainers)):  # driver-loop: over replica trainers
            owner.trainers[i].validate_device_state(owner.contexts[i], tr.completed_steps)
    if gradients:
        if tr.grad_step != tr.completed_steps:
            raise Error("byte LM parallel: no committed gradient")
        download_f32_into(ctx, tr.buffers.grad, n, f32_ptr(addr[0]))
        return tr.completed_steps
    if owner.pool_optimizer:
        var at = 0
        for i in range(len(owner.trainers)):  # small-loop(trainers: one trainer per replica device): checks range offsets per replica, not data
            if owner.trainers[i].buffers.optimizer_first != at:
                raise Error("byte LM parallel: pooled optimizer ranges are not contiguous")
            at += owner.trainers[i].buffers.optimizer_count
        if at != n:
            raise Error("byte LM parallel: pooled optimizer ranges do not cover the model")
    download_f32_into(ctx, tr.buffers.param, n, f32_ptr(addr[0]))
    if owner.pool_optimizer:
        for i in range(len(owner.trainers)):  # driver-loop: over replica trainers, one download each
            var first = owner.trainers[i].buffers.optimizer_first
            var owned = owner.trainers[i].buffers.optimizer_count
            download_f32_into(owner.contexts[i], owner.trainers[i].buffers.m_state, owned, f32_ptr(addr[1] + 4 * first))
            download_f32_into(owner.contexts[i], owner.trainers[i].buffers.v_state, owned, f32_ptr(addr[2] + 4 * first))
    else:
        download_f32_into(ctx, tr.buffers.m_state, n, f32_ptr(addr[1]))
        download_f32_into(ctx, tr.buffers.v_state, n, f32_ptr(addr[2]))
    _write_flags(addr[3], tr.buffers.buf_initialized)
    return tr.completed_steps


def _byte_schedule_ownership(mut session: ByteScheduleSession, a: List[Int], p: List[Int]) raises -> Int:
    # a = [handle, out Int words]; p = [capacity]. Replica rows are five words
    # (first, count, moment, rollback, reduction bytes); layer-pool rows are
    # seven (owner index, first, count, parameter, moment, rollback, gradient bytes).
    if len(a) != 2 or a[1] == 0 or len(p) != 1:
        raise Error("byte LM schedule: invalid ownership descriptor")
    var words = List[Int]()
    if session.kind == BYTE_SCHEDULE_REPLICA:
        ref owner = session.replica
        owner.require_open()
        for i in range(len(owner.trainers)):  # small-loop(trainers: one trainer per replica device): per replica ranges, no data
            ref b = owner.trainers[i].buffers
            words.append(b.optimizer_first)
            words.append(b.optimizer_count)
            words.append(4 * (len(b.m_state) + len(b.v_state)))
            words.append(4 * (len(b.shadow_p) + len(b.shadow_m) + len(b.shadow_v)))
            var reduction_bytes = 0
            if owner.pool_optimizer:
                reduction_bytes = 4 * (len(owner.pool_totals[i]) + len(owner.pool_incoming[i]))
            elif i == 0:
                reduction_bytes = 4 * len(owner.total.value())
            words.append(reduction_bytes)
    elif session.kind == BYTE_SCHEDULE_MODEL_POOL:
        session.pool.require_open()
        for i in range(len(session.pool.chunks)):  # small-loop(chunks: one model chunk per owner): per chunk ranges, no data
            ref chunk = session.pool.chunks[i]
            words.append(chunk.owner)
            words.append(chunk.first)
            words.append(len(chunk.p))
            words.append(4 * len(chunk.p))
            words.append(4 * (len(chunk.m) + len(chunk.v)))
            words.append(4 * (len(chunk.shadow_p) + len(chunk.shadow_m) + len(chunk.shadow_v)))
            words.append(4 * len(chunk.g))
    else:
        raise Error("NotImplementedError: offloaded state lives on the host; no resident GPU optimizer owners")
    if len(words) > p[0]:
        raise Error("byte LM schedule: ownership output is too short")
    var out = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[1])
    for i in range(len(words)):  # small-loop(words: captured ownership record words): copies record words, no data
        out[i] = words[i]
    return len(words)


def _byte_schedule_require(session: ByteScheduleSession, op: Int) raises:
    if (_byte_schedule_capability(session.kind) >> op) & 1 == 0:
        raise Error("NotImplementedError: byte-LM schedule has no operation " + String(op))


def byte_lm_schedule_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == BYTE_SCHEDULE_CAPABILITY:
        if len(a) != 0 or len(p) != 1 or len(f) != 0:
            raise Error("byte LM schedule: capability takes one schedule kind")
        return _byte_schedule_capability(_byte_schedule_kind(p[0]))
    if len(a) < 1:
        raise Error("byte LM schedule: missing admitted session")
    var session = _byte_schedule_session(a[0])
    _byte_schedule_require(session[], op)
    var kind = session[].kind
    if op == BYTE_SCHEDULE_KIND:
        return kind
    if op == BYTE_SCHEDULE_CLOSE:
        if kind == BYTE_SCHEDULE_REPLICA:
            session[].replica.close()
        elif kind == BYTE_SCHEDULE_MODEL_POOL:
            session[].pool.close()
        else:
            session[].offload.close()
        return 0
    if op == BYTE_SCHEDULE_OPEN:
        return _byte_schedule_open(session[], a, p, f)
    if op == BYTE_SCHEDULE_ROLLBACK:
        if kind == BYTE_SCHEDULE_REPLICA:
            session[].replica.require_open()
            session[].replica.rollback()
            return session[].replica.trainers[0].completed_steps
        if kind == BYTE_SCHEDULE_MODEL_POOL:
            session[].pool.require_open()
            session[].pool.rollback()
            return session[].pool.completed
        session[].offload.require_open()
        session[].offload.rollback()
        return session[].offload.completed
    if op == BYTE_SCHEDULE_OWNERSHIP:
        return _byte_schedule_ownership(session[], a, p)
    _require_binding_profile()
    if op == BYTE_SCHEDULE_STEP:
        # a = [handle, losses float64[count], shard ids...]; p = [completed, count].
        if len(p) != 2 or len(a) < 3 or a[1] == 0:
            raise Error("byte LM schedule: invalid step descriptor")
        var completed = p[0]
        var count = p[1]
        if kind == BYTE_SCHEDULE_REPLICA:
            ref owner = session[].replica
            owner.require_open()
            if completed != owner.trainers[0].completed_steps:
                raise Error("byte LM parallel: completed-step mismatch")
            if count != owner.logical_shards:
                raise Error("logical shard count mismatch")
            var losses = owner.step(_byte_schedule_shards(a, 2, count, owner.trainers[0].config))
            try:
                return _byte_schedule_losses(a[1], losses, count)
            except error:
                owner.rollback()
                raise error
        if kind == BYTE_SCHEDULE_MODEL_POOL:
            ref owner = session[].pool
            owner.require_open()
            if completed != owner.completed:
                raise Error("byte LM parallel: completed-step mismatch")
            if count != owner.logical_shards:
                raise Error("logical shard count mismatch")
            var losses = owner.step(_byte_schedule_shards(a, 2, count, owner.config))
            try:
                return _byte_schedule_losses(a[1], losses, count)
            except error:
                owner.rollback()
                raise error
        ref owner = session[].offload
        owner.require_open()
        if completed != owner.completed:
            raise Error("byte LM parallel: completed-step mismatch")
        if count != owner.logical_shards:
            raise Error("logical shard count mismatch")
        var losses = owner.step(_byte_schedule_shards(a, 2, count, owner.config))
        try:
            return _byte_schedule_losses(a[1], losses, count)
        except error:
            owner.rollback()
            raise error
    if op == BYTE_SCHEDULE_EXPORT:
        # p = [rank, gradients]
        if len(p) != 2 or p[1] != 0 and p[1] != 1:
            raise Error("byte LM schedule: invalid export descriptor")
        if kind == BYTE_SCHEDULE_REPLICA:
            session[].replica.require_open()
            return _byte_schedule_export_replica(session[].replica, a, p[0], p[1] == 1)
        if kind == BYTE_SCHEDULE_MODEL_POOL:
            session[].pool.require_open()
        else:
            session[].offload.require_open()
        if p[0] != 0:
            raise Error("byte model pool: canonical export has no replica rank; use rank=0")
        return _byte_schedule_export_canonical(session[], a, p[1] == 1)
    # The remaining operations are the replica's split-step and fold doors;
    # the capability check above already refused them on other schedules.
    ref owner = session[].replica
    owner.require_open()
    var n = owner.trainers[0].config.n_total()
    var n_ids = owner.trainers[0].config.batch * (owner.trainers[0].config.length + 1)
    if op == BYTE_SCHEDULE_SET_LR:
        if len(a) != 1 or len(f) != 1:
            raise Error("byte LM schedule: invalid learning-rate descriptor")
        owner.set_learning_rate(Float32(f[0]))
        return Int(bitcast[DType.uint32](owner.trainers[0].optimizer.lr))
    if op == BYTE_SCHEDULE_SHARD_GRADIENT or op == BYTE_SCHEDULE_SHARD_GRADIENT_FOLD:
        # a = [handle, ids, (gradient out,) loss float64 out]; p = [completed].
        var fold = op == BYTE_SCHEDULE_SHARD_GRADIENT_FOLD
        if len(p) != 1 or len(a) != (3 if fold else 4) or a[len(a) - 1] == 0:
            raise Error("byte LM schedule: invalid shard-gradient descriptor")
        if p[0] != owner.trainers[0].completed_steps:
            raise Error("byte LM parallel: completed-step mismatch")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n_ids]
        if not fold:
            slots.append(a[2])
            cells.append(n)
        _validate_slot_table(slots, cells, 1)
        var loss = Float32(0)
        if fold:
            loss = owner.shard_gradient_fold(_read_ids(a[1], n_ids))
        else:
            loss = owner.shard_gradient(_read_ids(a[1], n_ids))
            download_f32_into(owner.contexts[0], owner.trainers[0].buffers.grad, n, f32_ptr(a[2]))  # DEVIATION 3120
        MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[len(a) - 1])[0] = Float64(loss)
        return 0
    if op == BYTE_SCHEDULE_APPLY_GRADIENT:
        if len(a) != 2 or len(p) != 1:
            raise Error("byte LM schedule: invalid apply-gradient descriptor")
        if p[0] != owner.trainers[0].completed_steps:
            raise Error("byte LM parallel: completed-step mismatch")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n]
        _validate_slot_table(slots, cells, 1)
        owner.apply_gradient(f32_ptr(a[1]))
        return owner.trainers[0].completed_steps
    if op == BYTE_SCHEDULE_FOLD_RESET:
        if len(a) == 1:
            owner.fold_clear()
            return 0
        if len(a) != 2:
            raise Error("byte LM schedule: invalid fold descriptor")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n]
        _validate_slot_table(slots, cells, 1)
        owner.fold_reset(f32_ptr(a[1]))
        return 1
    if op == BYTE_SCHEDULE_FOLD_ADD or op == BYTE_SCHEDULE_FOLD_EXPORT:
        if len(a) != 2:
            raise Error("byte LM schedule: invalid fold descriptor")
        var slots: List[Int] = [a[1]]
        var cells: List[Int] = [n]
        _validate_slot_table(slots, cells, 1 if op == BYTE_SCHEDULE_FOLD_ADD else 0)
        if op == BYTE_SCHEDULE_FOLD_ADD:
            owner.fold_add(f32_ptr(a[1]))
            return 0
        owner.fold_export_into(f32_ptr(a[1]))  # DEVIATION 3120
        return n
    raise Error("byte LM schedule: unknown operation")


def byte_lm_schedule_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                        result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= BYTE_LM_SCHEDULE_OPERATION_COUNT:
            raise Error("byte LM schedule: unknown operation")
        if result == 0 or (nf > 0 and ff == 0):
            raise Error("byte LM schedule: null scalar or result span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(nf: at most the eight optimizer scalars): reads scalars, no data
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = byte_lm_schedule_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ByteLmScheduleTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(BYTE_LM_SCHEDULE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(byte_lm_schedule_v1)


comptime BYTE_LM_SCHEDULE_TABLE = _Global[StorageType=ByteLmScheduleTable,
    name="MojoByteLmScheduleTableV1", init_fn=ByteLmScheduleTable.__init__]


def byte_lm_schedule_api_binding() raises -> PythonObject:
    var api = BYTE_LM_SCHEDULE_TABLE.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def byte_lm_schedule_create_binding(kind: PythonObject) raises -> PythonObject:
    # Lazy: creation performs no GPU operation. The caller retains this value
    # for every use of its handle; its deallocation closes the session.
    var type_name = String(py=kind.__class__.__name__)
    if type_name != "int":
        raise Error("byte LM schedule: kind must be an int")
    return PythonObject(alloc=ByteScheduleSession(_byte_schedule_kind(Int(py=kind))))


def byte_lm_schedule_handle_binding(session: PythonObject) raises -> PythonObject:
    # Only the extension that minted this exact native value may unwrap it.
    return PythonObject(Int(session.downcast_value_ptr[ByteScheduleSession]()))


def byte_lm_arithmetic_suffix_binding() raises -> PythonObject:
    # The whole byte-LM arithmetic suffix that ByteConfig.profile() appends
    # (neural + CE token-tree + GEMM leaf + attention tags), not only the
    # neural part: the loader checks profile == PROFILE + this, so an A/B arm
    # build (e.g. MOJOLEARN_IDN_GEMM_LEAF) loads (bytelm-profile-harness).
    return PythonObject(byte_lm_arithmetic_suffix())


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_byte_lm.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_byte_lm() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_byte_lm")
        module.def_function[byte_lm_arithmetic_suffix_binding]("byte_lm_arithmetic_suffix")
        module.def_function[byte_lm_numeric_mode_binding]("byte_lm_numeric_mode")
        module.def_function[byte_lm_vendor_binding]("byte_lm_vendor")
        module.def_function[byte_lm_profile_binding]("byte_lm_profile")
        module.def_function[byte_lm_run_binding]("byte_lm_run")
        module.def_function[byte_lm_config_profile_binding]("byte_lm_config_profile")
        module.def_function[byte_lm_run_configured_binding]("byte_lm_run_configured")
        _ = module.add_type[ByteLMSession]("_ByteLMSession")
        module.def_function[byte_lm_session_create_binding]("byte_lm_session_create")
        module.def_function[byte_lm_native_api_binding]("byte_lm_trainer_api_v1")
        module.def_function[byte_lm_native_session_handle_binding]("byte_lm_native_session_handle_v1")
        module.def_function[byte_lm_session_close_binding]("byte_lm_session_close")
        # TOMBSTONE: byte_lm_launch_probe (dead native entry: tool-only launch-cost probe, deleted with tools/byte_lm_launch_probe.py and its byte_lm_session_launch_probe partner) deleted 2026-10-26 by be2f9a1be; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/byte_lm_launch_probe.patch
        # TOMBSTONE: byte_lm_context_keeper_active, byte_lm_attention_estash_gate (dead native entries: diagnostic read-backs, no package or gate reader) deleted 2026-10-26 by 9d5242f68; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_byte_lm.patch
        # DEVIATION 2514: the device-owned session entries.
        module.def_function[byte_lm_session_open_binding]("byte_lm_session_open")
        module.def_function[byte_lm_session_step_binding]("byte_lm_session_step")
        module.def_function[byte_lm_session_eval_binding]("byte_lm_session_eval")
        module.def_function[byte_lm_session_export_state_binding]("byte_lm_session_export_state")
        module.def_function[byte_lm_session_export_gradients_binding]("byte_lm_session_export_gradients")
        # DEVIATION 2658: forward-only logits.
        module.def_function[byte_lm_logits_binding]("byte_lm_logits")
        module.def_function[byte_lm_session_logits_binding]("byte_lm_session_logits")
        module.def_function[byte_lm_session_next_bytes_binding]("byte_lm_session_next_bytes")
        module.def_function[byte_lm_trainer_step_binding]("byte_lm_trainer_step")
        module.def_function[byte_lm_trainer_export_state_binding]("byte_lm_trainer_export_state")
        module.def_function[byte_lm_trainer_export_gradients_binding]("byte_lm_trainer_export_gradients")
        module.def_function[byte_lm_trainer_logits_binding]("byte_lm_trainer_logits")
        module.def_function[byte_lm_trainer_next_bytes_binding]("byte_lm_trainer_next_bytes")
        module.def_function[byte_lm_trainer_run_binding]("byte_lm_trainer_run")
        module.def_function[sha256_hex_binding]("sha256_hex")
        module.def_function[byte_lm_file_sha256_binding]("byte_lm_file_sha256")
        module.def_function[byte_lm_session_rollback_binding]("byte_lm_session_rollback")
        module.def_function[byte_lm_session_info_binding]("byte_lm_session_info")
        module.def_function[byte_lm_fault_inject_available_binding]("byte_lm_fault_inject_available")
        module.def_function[byte_lm_gemm_reuse_group_ws_binding]("byte_lm_gemm_reuse_group_ws")
        module.def_function[byte_lm_gemm_stage_ftz_binding]("byte_lm_gemm_stage_ftz")
        module.def_function[byte_lm_ce_aliased_binding]("byte_lm_ce_aliased")
        module.def_function[byte_lm_attn_sticky_fallback_binding]("byte_lm_attn_sticky_fallback")
        module.def_function[byte_lm_attn_kv_corner_guard_binding]("byte_lm_attn_kv_corner_guard")
        module.def_function[byte_lm_attn_bwd_corner_refuses_binding]("byte_lm_attn_bwd_corner_refuses")
        # DEVIATION 2534: the attention arm read-back (arm, default, trial, resolved).
        module.def_function[byte_lm_attention_arm_binding]("byte_lm_attention_arm")
        # TOMBSTONE: byte_lm_context_keeper_active, byte_lm_attention_estash_gate (dead native entries: diagnostic read-backs, no package or gate reader) deleted 2026-10-26 by 9d5242f68; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_byte_lm.patch
        module.def_function[byte_lm_attention_memory_profile_binding]("byte_lm_attention_memory_profile")
        # DEVIATION 2648: the step glue arm read-back (arm, trial).
        module.def_function[byte_lm_step_glue_arm_binding]("byte_lm_step_glue_arm")
        _ = module.add_type[ByteParallelTrainer]("_ByteParallelTrainer")
        module.def_function[byte_lm_parallel_create_binding]("byte_lm_parallel_create")
        module.def_function[byte_lm_parallel_close_binding]("byte_lm_parallel_close")
        module.def_function[byte_lm_parallel_open_binding[False]]("byte_lm_parallel_open")
        module.def_function[byte_lm_parallel_open_binding[True]]("byte_lm_parallel_open_pooled")
        module.def_function[byte_lm_pool_fault_available_binding]("byte_lm_pool_fault_available")
        module.def_function[byte_lm_parallel_reduction_pool_available_binding]("byte_lm_parallel_reduction_pool_available")
        module.def_function[byte_lm_parallel_ownership_binding]("byte_lm_parallel_ownership")
        module.def_function[byte_lm_parallel_step_binding]("byte_lm_parallel_step")
        module.def_function[byte_lm_parallel_export_binding]("byte_lm_parallel_export")
        module.def_function[byte_lm_parallel_shard_gradient_binding]("byte_lm_parallel_shard_gradient")
        module.def_function[byte_lm_parallel_apply_gradient_binding]("byte_lm_parallel_apply_gradient")
        module.def_function[byte_lm_parallel_rollback_binding]("byte_lm_parallel_rollback")
        module.def_function[byte_lm_parallel_set_lr_binding]("byte_lm_parallel_set_lr")
        module.def_function[byte_lm_parallel_fold_reset_binding]("byte_lm_parallel_fold_reset")
        module.def_function[byte_lm_parallel_shard_gradient_fold_binding]("byte_lm_parallel_shard_gradient_fold")
        module.def_function[byte_lm_parallel_fold_add_binding]("byte_lm_parallel_fold_add")
        module.def_function[byte_lm_parallel_fold_export_binding]("byte_lm_parallel_fold_export")
        _ = module.add_type[ByteOffloadedReplay]("_ByteOffloadedReplay")
        module.def_function[byte_lm_offload_create_binding]("byte_lm_offload_create")
        module.def_function[byte_lm_offload_open_binding]("byte_lm_offload_open")
        module.def_function[byte_lm_offload_close_binding]("byte_lm_offload_close")
        module.def_function[byte_lm_offload_step_binding]("byte_lm_offload_step")
        module.def_function[byte_lm_offload_export_binding]("byte_lm_offload_export")
        module.def_function[byte_lm_offload_rollback_binding]("byte_lm_offload_rollback")
        _ = module.add_type[ByteModelPool]("_ByteModelPool")
        module.def_function[byte_lm_model_pool_create_binding]("byte_lm_model_pool_create")
        module.def_function[byte_lm_model_pool_open_binding]("byte_lm_model_pool_open")
        module.def_function[byte_lm_model_pool_close_binding]("byte_lm_model_pool_close")
        module.def_function[byte_lm_model_pool_step_binding]("byte_lm_model_pool_step")
        module.def_function[byte_lm_model_pool_export_binding]("byte_lm_model_pool_export")
        module.def_function[byte_lm_model_pool_rollback_binding]("byte_lm_model_pool_rollback")
        module.def_function[byte_lm_model_pool_ownership_binding]("byte_lm_model_pool_ownership")
        # Private schedule descriptor/factory (bindings/parallel_byte_native.mojo).
        _ = module.add_type[ByteScheduleSession]("_ByteScheduleSession")
        module.def_function[byte_lm_schedule_api_binding]("byte_lm_schedule_api_v1")
        module.def_function[byte_lm_schedule_create_binding]("byte_lm_schedule_create_v1")
        module.def_function[byte_lm_schedule_handle_binding]("byte_lm_schedule_handle_v1")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_byte_lm: ", error))
