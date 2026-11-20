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
from core.experiment_guards_neural import NEURAL_GUARDS_OK
from core.hostptr import f32_ptr, i32_ptr, read_f32, copy_f32, list_i32
from training.byte_lm_native_status import byte_lm_status
from training.byte_lm_native_dispatch import BYTE_LM_NATIVE_TABLE
from training.byte_lm_stateless_logits import byte_lm_stateless_logits_native
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from training.byte_lm_glue_entry import publish_byte_lm_entries
from max.gpu import block_dim, block_idx, thread_idx
from core.device_zero import enqueue_fill
from std.memory import bitcast, unsafe_memcmp
from std.os import abort, getenv
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.p3_lm_tokenizer_cpython_unpack import lm_try_index
from std.time import perf_counter_ns
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.byte_lm_teardown_switch import byte_lm_teardown_with_gil
from training.byte_lm_binding_impl import (
    ByteLMSession,
    _span_cells, _validate_addresses, _require_inputs_unchanged, _write_f32,
    _btick, _bbytes, _require_same_bits,
    _require_binding_profile, _validate_slot_table,
    _write_flags_n, _write_loss, _nonfinite,
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
from training.checks.train_loop import download_f32
from training.byte_lm_config import ByteConfig
from training.byte_lm_stateless_run import byte_lm_run_native, byte_lm_run_finalize
from training.byte_lm import (
    BYTE_PROFILE, ByteTrainer, byte_train_step, byte_train_step_resident,
    byte_eval_loss, byte_eval_loss_resident, byte_rollback,
    byte_validate_device_state, byte_validate_optimizer,
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
from core.shared_slot import SharedSlot
# RUN (lane rs-embedding-bytelm): typed Mojo only, no Python import.
from training.byte_lm_config_run import byte_config_words
from training.byte_lm_door_run import (
    BYTE_LM_SCHEDULE_TABLE,
    ByteScheduleSession,
    _byte_schedule_kind,
    byte_model_pool_export_run,
    byte_offload_export_run,
    byte_open_state,
    byte_parallel_export_run,
    byte_step_shards,
)


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
    var locked_teardown = byte_lm_teardown_with_gil()
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
    for i in range(count):  # driver-loop: over count: the call's address slots or shards: reads the addresses, no data
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
    var values = List[Int]()
    for i in range(len(shape)):  # driver-loop: over shape: 7 to 10 configuration words
        var item = shape[i]
        # Fast path: PyNumber_Index through entries resolved once (no class
        # name String, no operator call); any refusal keeps the original path.
        var word = 0
        if lm_try_index(item, word):
            values.append(word)
            continue
        var type_name = String(py=item.__class__.__name__)
        if type_name == "bool" or type_name == "bool_":
            raise Error("byte LM: shape dimensions must be integers, not booleans")
        values.append(Int(py=py_call(native_loaded_module("operator").index, item)))
    return byte_config_words(values^, "byte LM")


def byte_lm_config_profile_binding(shape: PythonObject) raises -> PythonObject:
    """Host-only shape admission and profile negotiation; creates no context."""
    var cfg = _byte_config(shape)
    return PythonObject(cfg.profile())


# TOMBSTONE: byte_lm_run (dead native entry: v1 default-shape ABI, only caller tools/byte_lm_config_binding_check.py, deleted with it) deleted 2026-11-18 by lane/fx-dead-small; code recoverable at 7c1aad6561.
# Restore: git apply experiments/removed/byte_lm_run.patch


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
    var out = List[Int]()
    for i in range(2):  # driver-loop: over dims: batch and length words
        var item = dims[i]
        var word = 0
        if lm_try_index(item, word):
            out.append(word)
            continue
        var type_name = String(py=item.__class__.__name__)
        if type_name == "bool" or type_name == "bool_":
            raise Error("byte LM logits: dims must be integers, not booleans")
        out.append(Int(py=py_call(native_loaded_module("operator").index, item)))
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
    var claimed = 0
    if not lm_try_index(completed, claimed):
        claimed = Int(py=native_loaded_module("operator").index(completed))
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


# ---------------------------------------------------------------------------
# The single-device trainer owner (mojolearn.SmallByteLanguageModelTrainer):
# one entry per public method; each unpacks its arguments, calls ONE typed
# owner function (training/byte_lm_session_ops.mojo) and returns its typed
# result. Nothing is assigned to a Python object here.
# ---------------------------------------------------------------------------


def _read_address_list(addresses: PythonObject) raises -> List[Int]:
    var addr = List[Int]()
    for i in range(len(addresses)):  # driver-loop: over addresses: the call's address slots, one per buffer: reads the addresses, no data
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


def byte_lm_parallel_create_binding() raises -> PythonObject:
    return PythonObject(alloc=ByteParallelTrainer())


# TOMBSTONE: byte_lm_parallel_open, byte_lm_parallel_open_pooled, byte_lm_parallel_close, byte_lm_parallel_rollback, byte_lm_parallel_shard_gradient, byte_lm_parallel_apply_gradient, byte_lm_parallel_reduction_pool_available, byte_lm_parallel_fold_reset, byte_lm_parallel_shard_gradient_fold, byte_lm_parallel_fold_add (dead native entries: no caller; the parallel trainer runs through parallel_shard_call) deleted 2026-11-08 by f41ec913f2; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/dead_entries_byte_lm_parallel.patch


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
    var shards = byte_step_shards(addr, count, owner[].trainers[0].config)
    var losses = owner[].step(shards)
    var out = Python.list()
    try:
        for i in range(len(losses)):  # driver-loop: over losses: one loss per shard: returns the shard losses
            out.append(PythonObject(losses[i]))
    except error:
        owner[].rollback()
        raise error
    return out


def byte_lm_parallel_export_binding(session: PythonObject, addresses: PythonObject,
    rank_arg: PythonObject, gradients_arg: PythonObject) raises -> PythonObject:
    var rank = Int(py=rank_arg)
    var gradients = Bool(gradients_arg)
    _require_binding_profile()
    var owner = session.downcast_value_ptr[ByteParallelTrainer]()
    owner[].require_open()
    if rank < 0 or rank >= len(owner[].trainers):
        raise Error("byte LM parallel: invalid replica index")
    var count = 1 if gradients else 4
    var addr = _read_addresses(addresses, count)
    return PythonObject(byte_parallel_export_run(owner[], addr, rank, gradients))


def byte_lm_pool_fault_available_binding() raises -> PythonObject:
    return PythonObject(pool_fault_available())

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
    var shards = byte_step_shards(addr, count, owner[].config)
    var losses = owner[].step(shards)
    try:
        var out = Python.list()
        for i in range(len(losses)):  # driver-loop: over losses: one loss per shard: returns the shard losses
            out.append(PythonObject(losses[i]))
        return out
    except error:
        owner[].rollback()
        raise error


# TOMBSTONE: byte_lm_launch_probe (dead native entry: tool-only launch-cost probe; byte_lm_session_launch_probe with it) deleted 2026-10-26 by be2f9a1be; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/byte_lm_launch_probe.patch
# The private byte-LM schedule descriptor table (session value, operations,
# C entry) is RUN code in training/byte_lm_door_run.mojo; the door keeps
# its capsule, factory and handle exports below.


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


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_byte_lm.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_byte_lm() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_byte_lm")
        # The native trainer door (bindings/byte_lm_trainer_native.mojo) reads
        # the metadata, arm read-backs, stateless step and stateless logits
        # through the trainer table published below (lane calltable-byte-lm).
        # These Python entries remain only for unconverted callers: the Python
        # session methods of python/mojolearn/_byte_lm_impl.py, tools and tests.
        module.def_function[byte_lm_numeric_mode_binding]("byte_lm_numeric_mode")
        module.def_function[byte_lm_vendor_binding]("byte_lm_vendor")
        module.def_function[byte_lm_profile_binding]("byte_lm_profile")
        # TOMBSTONE: byte_lm_run (dead native entry: v1 default-shape ABI, only caller tools/byte_lm_config_binding_check.py, deleted with it) deleted 2026-11-18 by lane/fx-dead-small; code recoverable at 7c1aad6561.
        module.def_function[byte_lm_config_profile_binding]("byte_lm_config_profile")
        module.def_function[byte_lm_run_configured_binding]("byte_lm_run_configured")
        _ = module.add_type[ByteLMSession]("_ByteLMSession")
        module.def_function[byte_lm_session_create_binding]("byte_lm_session_create")
        module.def_function[byte_lm_session_close_binding]("byte_lm_session_close")
        # TOMBSTONE: byte_lm_launch_probe (dead native entry: tool-only launch-cost probe, deleted with tools/byte_lm_launch_probe.py and its byte_lm_session_launch_probe partner) deleted 2026-10-26 by be2f9a1be; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/byte_lm_launch_probe.patch
        # TOMBSTONE: byte_lm_context_keeper_active, byte_lm_attention_estash_gate (dead native entries: diagnostic read-backs, no package or gate reader) deleted 2026-10-26 by 9d5242f68; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_byte_lm.patch
        # DEVIATION 2514: the device-owned session entries.
        module.def_function[byte_lm_session_open_binding]("byte_lm_session_open")
        module.def_function[byte_lm_session_step_binding]("byte_lm_session_step")
        module.def_function[byte_lm_session_export_state_binding]("byte_lm_session_export_state")
        # DEVIATION 2658: forward-only logits.
        module.def_function[byte_lm_logits_binding]("byte_lm_logits")
        module.def_function[byte_lm_session_logits_binding]("byte_lm_session_logits")
        module.def_function[byte_lm_trainer_step_binding]("byte_lm_trainer_step")
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
        # DEVIATION 2648: the step glue arm read-back (arm, trial).
        _ = module.add_type[ByteParallelTrainer]("_ByteParallelTrainer")
        module.def_function[byte_lm_parallel_create_binding]("byte_lm_parallel_create")
        # TOMBSTONE: byte_lm_parallel_open, _open_pooled, _close, _rollback, _shard_gradient, _apply_gradient, _reduction_pool_available, _fold_reset, _shard_gradient_fold, _fold_add (dead native entries: no caller) deleted 2026-11-08 by f41ec913f2; code recoverable at 41ffcec665.
        # Restore: git apply experiments/removed/dead_entries_byte_lm_parallel.patch
        module.def_function[byte_lm_pool_fault_available_binding]("byte_lm_pool_fault_available")
        module.def_function[byte_lm_parallel_step_binding]("byte_lm_parallel_step")
        module.def_function[byte_lm_parallel_export_binding]("byte_lm_parallel_export")
        module.def_function[byte_lm_parallel_set_lr_binding]("byte_lm_parallel_set_lr")
        module.def_function[byte_lm_parallel_fold_export_binding]("byte_lm_parallel_fold_export")
        _ = module.add_type[ByteOffloadedReplay]("_ByteOffloadedReplay")
        module.def_function[byte_lm_offload_step_binding]("byte_lm_offload_step")
        _ = module.add_type[ByteModelPool]("_ByteModelPool")
        # Private schedule descriptor/factory (bindings/parallel_byte_native.mojo).
        _ = module.add_type[ByteScheduleSession]("_ByteScheduleSession")
        module.def_function[byte_lm_schedule_api_binding]("byte_lm_schedule_api_v1")
        module.def_function[byte_lm_schedule_create_binding]("byte_lm_schedule_create_v1")
        module.def_function[byte_lm_schedule_handle_binding]("byte_lm_schedule_handle_v1")
        # The native trainer door's table (core/byte_lm_entry_table.mojo).
        publish_byte_lm_entries()
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_byte_lm: ", error))
