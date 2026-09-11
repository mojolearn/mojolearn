# SPDX-License-Identifier: Apache-2.0
"""Synchronous owned-state boundary for a runtime-shaped decoder language model.

Runtime shapes are compile/host checked; device qualification is separate.
No GPU/context is created during import.
The Python caller owns correctly sized, contiguous, aligned live arrays. Native
span checks cannot prove that an arbitrary integer address names allocated RAM.
No borrowed pointer survives a call. Optional owned sessions retain device state.
Outputs are published only after successful computation, validation and
synchronization; the stateless ABI also tears down its context before return.

DEVIATION 2514 (device-owned step, design
docs/lanes/DESIGN_lm_device_owned_step_2026-09-11.md): the `byte_lm_session_*`
entries below `byte_lm_session_run` make the session's device buffers the
ONLY copy of the model, the optimizer state, the last gradient and the flags
while it is open. State crosses the boundary at `open` (validated on the
host Lists, uploaded once), at `export_state` / `export_gradients` (pinned
staging, then `copy_f32` into the caller's arrays) and never per step; a
step moves the ids in and the loss and the flags out. A failed step is
rolled back on the device from the shadow the trainer takes before its
update; `rollback` exposes the same restore for a failure the Python layer
detects after native success. `byte_lm_session_run` (mirror in, mirror
out) is kept for the transition and is unchanged.
"""
# DEVIATION 2486: shared byte-preserving host copies.
from bindings.hostptr import f32_ptr, read_f32, copy_f32
from std.ffi import _Global
from std.memory import bitcast
from std.os import abort, getenv
from std.python import Python, PythonObject
from std.time import perf_counter_ns
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
# DEVIATION 2630: the step phase timers and counters (core/step_phase.mojo;
# compiled only under -D MOJOLEARN_STEP_PHASE_TIMERS=1).
from core.step_phase import (
    step_count_sync,
    step_counts_now,
    step_counts_report,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from training.checks.optimizer_oracle import OptimizerConfig
from training.checks.train_loop import download_f32
from training.byte_lm_config import ByteConfig
from training.byte_lm import (
    BYTE_PROFILE, ByteTrainer, byte_train_step, byte_train_step_resident,
    byte_eval_loss, byte_eval_loss_resident, byte_rollback,
    byte_validate_state, byte_validate_optimizer,
    byte_validate_tokens, byte_lm_fault_inject_available,
)
from training.byte_lm_logits import (
    BYTE_LOGITS_MAX_BATCH,
    BYTE_LOGITS_MAX_CELLS,
    byte_logits_from_params,
    byte_logits_resident,
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
    build. Reads constants and the environment only; no GPU operation."""
    var arm = fused_attention_arm_from_env()
    var resolved = fused_attention_arm_forward_resolved(arm) | fused_attention_arm_backward_resolved(arm)
    var out = Python.list()
    out.append(PythonObject(fused_attention_arm_name(arm)))
    out.append(PythonObject(fused_attention_arm_name(ATTN_ARM_DEFAULT)))
    out.append(PythonObject(1 if ATTN_ARM_TRIAL else 0))
    out.append(PythonObject(fused_attention_arm_name(resolved)))
    return out


struct ByteLMSession(Movable, Writable):
    """Python-owned device lifetime; no global handles or retained host pointers.

    Upstream LlamaModel owns its layers across calls:
    transformers/models/llama/modeling_llama.py:334-359,402-412.
    This binding retains our existing ByteTrainer without changing its kernels.
    """
    var ctx: Optional[DeviceContext]
    var trainer: Optional[ByteTrainer]
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.trainer = Optional[ByteTrainer]()
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteLMSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("ByteLMSession")

    def __deinit__(deinit self):
        # Buffers must die before their context, and the frees they enqueue
        # must drain before the context goes (DEVIATION 2520, below).
        _ = self.trainer^
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        _ = self.ctx^

    def close(mut self) raises:
        if self.busy:
            raise Error("byte LM: session is busy")
        self.usable = False
        if self.ctx:
            self.ctx.value().synchronize()
        self.trainer = None
        # DEVIATION 2520: releasing the trainer enqueues its buffer frees on
        # the context's stream; destroying the context with those frees in
        # flight left the MAX runtime's allocator lock held on an RTX 4090
        # pod, and the next context's first enqueueCreateBuffer blocked in
        # pthread_mutex_lock forever (native backtrace, run 6, 2026-09-11).
        # Draining here is what the passing variant did.
        if self.ctx:
            self.ctx.value().synchronize()
        self.ctx = None


struct _ContextKeeper(Defaultable, Movable):
    """DEVIATION 2513: an OPT-IN process-lifetime DeviceContext.

    On one RTX 4090 pod every DeviceContext created after another was
    destroyed in the same process never returned from its first use
    (docs/lanes/BRIEF_byte_lm_lifetime_2026-09-10.md, run 3), while a
    second context created while the first was still alive always did.
    When MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1 is set at a call that creates a
    context, `ensure()` creates ONE separate context here first and never
    releases it, so a live context exists whenever any later per-call or
    resident context is created. The per-call path, the resident session
    and the teardown order are unchanged; this is a diagnostic mitigation,
    OFF by default, and not a claim about the cause.

    Storage: `std.ffi._Global`, the pattern the trees and RF bindings use
    for their export registries (ET_EXPORTS, RF_EXPORTS). There is no
    module-level `var` in this tree on purpose; the _Global slot is
    runtime-owned, created once by name, and lives until process teardown.
    """
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()

    def ensure(mut self) raises:
        if not self.ctx:
            self.ctx = DeviceContext()

    def active(self) -> Bool:
        return Bool(self.ctx)


comptime BYTE_LM_CONTEXT_KEEPER = _Global[StorageType=_ContextKeeper,
    name="MojoByteLMContextKeeperIdentical", init_fn=_ContextKeeper.__init__]


def byte_lm_context_keeper_active_binding() raises -> PythonObject:
    """DEVIATION 2513 read-back: True once the keeper holds a context.
    Creates the empty slot if absent; never creates a DeviceContext."""
    return PythonObject(BYTE_LM_CONTEXT_KEEPER.get_or_create_ptr()[].active())


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
    return PythonObject(String(BYTE_PROFILE))


def _span_cells(index: Int, shape: ByteConfig) raises -> Int:
    if index == 3 or index == 9:
        return shape.n_tensors()
    if index == 4:
        return shape.batch * (shape.length + 1)
    if index == 10:
        return 1
    return shape.n_total()


def _validate_addresses(addresses: List[Int], action: Int, shape: ByteConfig) raises:
    # All spans use validated runtime-profile lengths. Validate addition before any
    # pointer construction/dereference; reject null, misalignment and wraparound.
    for i in range(11):
        if i == 8 and action == 0:
            if addresses[i] != 0:
                raise Error("byte LM eval requires null gradient output; no gradient is computed")
            continue
        var size_bytes = _span_cells(i, shape) * 4
        if addresses[i] <= 0 or addresses[i] % 4 != 0 or addresses[i] > Int(0x7FFFFFFFFFFFFFFF) - size_bytes:
            raise Error("byte LM: null/misaligned/overflowing span at address slot " + String(i))
    # Inputs may share storage because all are copied before numerical work.
    # Every output must be disjoint from all inputs and other outputs.
    for i in range(5, 11):
        if i == 8 and action == 0:
            continue
        for j in range(i):
            if j == 8 and action == 0:
                continue
            if (addresses[i] < addresses[j] + _span_cells(j, shape) * 4
                and addresses[j] < addresses[i] + _span_cells(i, shape) * 4):
                raise Error("byte LM: output overlaps another live span")


def _read_f32(address: Int, n: Int) raises -> List[Float32]:
    return read_f32(address, n)


def _write_f32(address: Int, values: List[Float32]) raises:
    copy_f32(values.unsafe_ptr(), f32_ptr(address), len(values))


def _btick(on: Bool, mut t: Int, name: String):
    """DEVIATION 2499: host-side phase timer for the binding, the same line
    shape as the transformer binding's `_btick` and the block timers
    (`timing <name> <ms> ms`), behind `MOJOLEARN_TRANSFORMER_TIMING=1`.
    No device wait here: every phase this brackets is host work, or ends
    with a `download_f32` (which waits inside) or the trainer's own waits."""
    if not on:
        return
    var now = Int(perf_counter_ns())
    print(
        "timing " + name + " " + String(Float64(now - t) / 1000000.0) + " ms"
    )
    t = now


def _bbytes(on: Bool, name: String, n_bytes: Int):
    if not on:
        return
    print("timing " + name + " " + String(n_bytes) + " bytes")


def _require_same_bits(before: List[Float32], after: List[Float32]) raises:
    if len(before) != len(after):
        raise Error("byte LM: authoritative state length mismatch")
    for i in range(len(before)):
        if bitcast[DType.uint32](before[i]) != bitcast[DType.uint32](after[i]):
            raise Error("byte LM: authoritative state bits mismatch")


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
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("byte LM binding requires IDENTICAL")
    if String(COMPILED_VENDOR) != "cuda" and String(COMPILED_VENDOR) != "hip" and String(COMPILED_VENDOR) != "metal":
        raise Error("byte LM binding requires CUDA, HIP or Metal")
    if len(addresses) != 11 or len(params) != 12:
        raise Error("byte LM: expected 11 addresses and 12 scalar parameters")
    shape.validate()
    var n = shape.n_total()
    var action = Int(py=params[0])
    var completed = Int(py=params[1])
    var kind = Int(py=params[2])
    var nesterov = Int(py=params[10])
    if (action != 0 and action != 1) or (nesterov != 0 and nesterov != 1):
        raise Error("byte LM: action/nesterov must be 0 or 1")
    if completed < 0 or completed >= 1000000 or (action == 1 and completed >= 999999):
        raise Error("byte LM: completed step outside admitted bound")
    var cfg = OptimizerConfig(kind,
        Float32(Float64(py=params[3])), Float32(Float64(py=params[4])),
        Float32(Float64(py=params[5])), Float32(Float64(py=params[6])),
        Float32(Float64(py=params[7])), Float32(Float64(py=params[8])),
        Float32(Float64(py=params[9])), nesterov == 1,
        Float32(Float64(py=params[11])))
    byte_validate_optimizer(cfg)
    var addr = List[Int]()
    for i in range(11):
        addr.append(Int(py=addresses[i]))
    _validate_addresses(addr, action, shape)
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    # DEVIATION 2513: one getenv per call, same cost class as `ton`; the
    # keeper itself is touched only on a call that creates a context.
    var keep_context = String(getenv("MOJOLEARN_BYTE_LM_KEEP_CONTEXT")) == "1"
    # DEVIATION 2518: two OPT-IN teardown variants for the stateless path
    # (retain=False) only, one getenv each per call, OFF by default, no
    # effect on the resident session or on any arithmetic. Each prints one
    # witness line when it is on, so a run can prove the branch was reached.
    #   sync_before_teardown: synchronize() immediately before the trainer
    #     (its ~250 buffers) is released, and again after that release and
    #     before the context is released, so every stream-ordered buffer
    #     free has drained while the context is still alive.
    #   teardown_with_gil: release the trainer and the context AFTER the
    #     GILReleased block, i.e. with the GIL held, the way close() does,
    #     instead of inside it.
    var sync_before_teardown = String(getenv("MOJOLEARN_BYTE_LM_SYNC_BEFORE_TEARDOWN")) == "1"
    var teardown_with_gil = String(getenv("MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL")) == "1"
    var tk = Int(perf_counter_ns())
    # No GPU work before all borrowed inputs become validated owned host lists.
    var initial_p = _read_f32(addr[0], n)
    var initial_m = _read_f32(addr[1], n)
    var initial_v = _read_f32(addr[2], n)
    var flags_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[3])
    var ids_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[4])
    var flags = List[Bool]()
    var ids = List[Int32]()
    for i in range(shape.n_tensors()):
        var flag = flags_ptr.unsafe_load(i)
        if flag != 0 and flag != 1:
            raise Error("byte LM: momentum flags must be exactly 0 or 1")
        flags.append(flag == 1)
    for i in range(shape.batch * (shape.length + 1)):
        ids.append(ids_ptr.unsafe_load(i))
    # Host-to-host: three n-float Lists plus flags and ids from Python memory.
    _btick(ton, tk, "step.bind_read_inputs")
    _bbytes(ton, "step.bind_read_inputs_bytes", 3 * n * 4 + shape.n_tensors() * 4 + shape.batch * (shape.length + 1) * 4)
    byte_validate_state(initial_p, initial_m, initial_v, flags, completed, shape)
    byte_validate_tokens(ids, shape)
    _btick(ton, tk, "step.bind_validate_inputs")
    var out_p = List[Float32]()
    var out_m = List[Float32]()
    var out_v = List[Float32]()
    var out_g = List[Float32]()
    var out_flags = List[Bool]()
    var loss = Float32(0)
    var result_step = completed
    if session.busy or not session.usable:
        raise Error("byte LM: session is busy, closed or failed; recreate it")
    session.busy = True
    session.usable = False
    try:
        with GILReleased(Python()):
            var reused = Bool(session.ctx)
            if not session.ctx:
                if keep_context:
                    # DEVIATION 2513: a SEPARATE keeper context, created
                    # before the per-call one and never released. Idempotent
                    # after the first creating call. Off: this branch is
                    # not entered and nothing below changes.
                    BYTE_LM_CONTEXT_KEEPER.get_or_create_ptr()[].ensure()
                session.ctx = DeviceContext()
                session.trainer = ByteTrainer(session.ctx.value(), initial_p, initial_m,
                    initial_v, flags, completed, cfg, shape)
            ref ctx = session.ctx.value()
            # First call only: context and ByteTrainer construction (uploads).
            _btick(ton, tk, "step.bind_context_or_trainer_setup")
            if reused:
                # Reuse must never silently ignore a supplied checkpoint/config.
                if session.trainer.value().config.profile() != shape.profile():
                    raise Error("byte LM: resident model shape mismatch")
                if session.trainer.value().completed_steps != completed:
                    raise Error("byte LM: resident completed-step mismatch")
                var prior_cfg = session.trainer.value().optimizer.copy()
                if (prior_cfg.kind != cfg.kind or prior_cfg.nesterov != cfg.nesterov
                    or bitcast[DType.uint32](prior_cfg.lr) != bitcast[DType.uint32](cfg.lr) or bitcast[DType.uint32](prior_cfg.beta1) != bitcast[DType.uint32](cfg.beta1)
                    or bitcast[DType.uint32](prior_cfg.beta2) != bitcast[DType.uint32](cfg.beta2) or bitcast[DType.uint32](prior_cfg.eps) != bitcast[DType.uint32](cfg.eps)
                    or bitcast[DType.uint32](prior_cfg.weight_decay) != bitcast[DType.uint32](cfg.weight_decay)
                    or bitcast[DType.uint32](prior_cfg.momentum) != bitcast[DType.uint32](cfg.momentum) or bitcast[DType.uint32](prior_cfg.dampening) != bitcast[DType.uint32](cfg.dampening)
                    or bitcast[DType.uint32](prior_cfg.max_norm) != bitcast[DType.uint32](cfg.max_norm)):
                    raise Error("byte LM: resident optimizer mismatch")
                _require_same_bits(initial_p, download_f32(ctx, session.trainer.value().buffers.param, n))
                _require_same_bits(initial_m, download_f32(ctx, session.trainer.value().buffers.m_state, n))
                _require_same_bits(initial_v, download_f32(ctx, session.trainer.value().buffers.v_state, n))
                for i in range(shape.n_tensors()):
                    if flags[i] != session.trainer.value().buffers.buf_initialized[i]:
                        raise Error("byte LM: resident flags mismatch")
                # Three download_f32 (each waits inside) plus three host
                # bit compares; the device queue is empty at this tick.
                _btick(ton, tk, "step.bind_resident_admission")
                _bbytes(ton, "step.bind_resident_admission_bytes", 3 * n * 4)
            if action == 1:
                var capture = byte_train_step(ctx, session.trainer.value(), ids)
                # The trainer printed its own `step.*` lines; its last tick
                # (`step.capture_copy`) ended with nothing queued.
                if ton:
                    tk = Int(perf_counter_ns())
                loss = capture.loss
                result_step = capture.completed_steps
                out_p = capture.after_params^
                capture.after_params = List[Float32]()
                out_m = capture.after_m^
                capture.after_m = List[Float32]()
                out_v = capture.after_v^
                capture.after_v = List[Float32]()
                out_g = capture.gradients^
                capture.gradients = List[Float32]()
                out_flags = capture.after_flags^
                capture.after_flags = List[Bool]()
                _btick(ton, tk, "step.bind_move_outputs")
            else:
                loss = byte_eval_loss(ctx, session.trainer.value(), ids)
                if ton:
                    tk = Int(perf_counter_ns())
                # Read actual post-evaluation state rather than simply echoing the
                # inputs, which would hide an accidental mutation in evaluation.
                out_p = download_f32(ctx, session.trainer.value().buffers.param, n)
                out_m = download_f32(ctx, session.trainer.value().buffers.m_state, n)
                out_v = download_f32(ctx, session.trainer.value().buffers.v_state, n)
                out_flags = session.trainer.value().buffers.buf_initialized.copy()
                result_step = session.trainer.value().completed_steps
                _require_same_bits(initial_p, out_p)
                _require_same_bits(initial_m, out_m)
                _require_same_bits(initial_v, out_v)
                for i in range(shape.n_tensors()):
                    if flags[i] != out_flags[i]:
                        raise Error("byte LM eval changed momentum flags")
                _btick(ton, tk, "step.bind_eval_readback")
                _bbytes(ton, "step.bind_eval_readback_bytes", 3 * n * 4)
            byte_validate_state(out_p, out_m, out_v, out_flags, result_step, shape)
            if result_step != completed + action:
                raise Error("byte LM: successful result has wrong completed step")
            if (bitcast[DType.uint32](loss) & UInt32(0x7F800000)) == UInt32(0x7F800000):
                raise Error("byte LM: nonfinite returned loss")
            if action == 1:
                if len(out_g) != n:
                    raise Error("byte LM: wrong gradient length")
                for i in range(n):
                    if (bitcast[DType.uint32](out_g[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
                        raise Error("byte LM: nonfinite returned gradient")
            # Host scans of out_p/m/v (byte_validate_state) and out_g.
            _btick(ton, tk, "step.bind_validate_outputs")
            ctx.synchronize()
            _btick(ton, tk, "step.bind_final_sync")
            if not retain and not teardown_with_gil:
                # DEVIATION 2520 (was 2518 variant (a), measured on the RTX
                # 4090 run 6: this order PASSES where the bare release hung):
                # release the trainer, then DRAIN the frees it enqueued before
                # the context is destroyed. The native backtrace of the hang
                # was the next context's first enqueueCreateBuffer blocked in
                # pthread_mutex_lock inside libKGENCompilerRTShared, the
                # runtime allocator's lock the dying context never returned.
                if sync_before_teardown:
                    print("byte LM teardown variant: sync_before_teardown")
                session.trainer = None
                session.ctx.value().synchronize()
                session.ctx = None
    except error:
        session.busy = False
        raise error
    session.busy = False
    session.usable = retain
    if not retain and teardown_with_gil:
        # DEVIATION 2518 variant (b): the same two releases, after the
        # GILReleased block has re-acquired the GIL (the resident close()
        # shape: synchronized above, released with the GIL held). Off, the
        # teardown above ran and both Optionals are already empty.
        print("byte LM teardown variant: teardown_with_gil")
        session.trainer = None
        # DEVIATION 2520: drain the enqueued frees before the context goes.
        if session.ctx:
            session.ctx.value().synchronize()
        session.ctx = None
    # Publication starts after the synchronized GPU scope succeeds. The
    # Python wrapper commits these fresh arrays atomically to its state object.
    _write_f32(addr[5], out_p)
    _write_f32(addr[6], out_m)
    _write_f32(addr[7], out_v)
    if action == 1:
        _write_f32(addr[8], out_g)
    var flags_out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[9])
    for i in range(shape.n_tensors()):
        flags_out.unsafe_store(i, Int32(out_flags[i]))
    var loss_out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr[10])
    loss_out.unsafe_store(0, loss)
    # Host-to-host: the four n-float outputs copied into Python memory.
    _btick(ton, tk, "step.bind_publish")
    _bbytes(ton, "step.bind_publish_bytes", (3 + action) * n * 4)
    return PythonObject(result_step)


# ---------------------------------------------------------------------------
# DEVIATION 2514 step 5: the device-owned session entries.
# ---------------------------------------------------------------------------


def _require_binding_profile() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("byte LM binding requires IDENTICAL")
    if String(COMPILED_VENDOR) != "cuda" and String(COMPILED_VENDOR) != "hip" and String(COMPILED_VENDOR) != "metal":
        raise Error("byte LM binding requires CUDA, HIP or Metal")


def _validate_slot_table(addresses: List[Int], cells: List[Int], n_inputs: Int) raises:
    """The slot-table form of `_validate_addresses`: slot `i` names
    `cells[i]` four-byte cells; slots `[0, n_inputs)` are inputs, the rest
    outputs. Every slot is non-null, four-aligned and cannot wrap. Inputs
    may share storage (each is copied before any device work); every output
    must be disjoint from every other slot. Validated before any pointer is
    built, as the eleven-slot form does."""
    if len(addresses) != len(cells):
        raise Error("byte LM: expected " + String(len(cells)) + " addresses")
    for i in range(len(cells)):
        var size_bytes = cells[i] * 4
        if addresses[i] <= 0 or addresses[i] % 4 != 0 or addresses[i] > Int(0x7FFFFFFFFFFFFFFF) - size_bytes:
            raise Error("byte LM: null/misaligned/overflowing span at address slot " + String(i))
    for i in range(n_inputs, len(cells)):
        for j in range(len(cells)):
            if j == i:
                continue
            if (addresses[i] < addresses[j] + cells[j] * 4
                and addresses[j] < addresses[i] + cells[i] * 4):
                raise Error("byte LM: output overlaps another live span")


def _read_addresses(addresses: PythonObject, count: Int) raises -> List[Int]:
    if len(addresses) != count:
        raise Error("byte LM: expected " + String(count) + " addresses")
    var addr = List[Int]()
    for i in range(count):
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


def _read_flags(address: Int, n_tensors: Int) raises -> List[Bool]:
    var flags_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    var flags = List[Bool]()
    for i in range(n_tensors):
        var flag = flags_ptr.unsafe_load(i)
        if flag != 0 and flag != 1:
            raise Error("byte LM: momentum flags must be exactly 0 or 1")
        flags.append(flag == 1)
    return flags^


def _read_ids(address: Int, count: Int) -> List[Int32]:
    var ids_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    var ids = List[Int32]()
    for i in range(count):
        ids.append(ids_ptr.unsafe_load(i))
    return ids^


def _write_flags(address: Int, flags: List[Bool]):
    var flags_out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(flags)):
        flags_out.unsafe_store(i, Int32(flags[i]))


def _write_loss(address: Int, loss: Float32):
    var loss_out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    loss_out.unsafe_store(0, loss)


def _nonfinite(value: Float32) -> Bool:
    return (bitcast[DType.uint32](value) & UInt32(0x7F800000)) == UInt32(0x7F800000)


def _same_optimizer_bits(a: OptimizerConfig, b: OptimizerConfig) -> Bool:
    """Bitwise equality of every optimizer scalar, signed zero included:
    the spelling `_byte_lm_run`'s resident admission uses."""
    return (a.kind == b.kind and a.nesterov == b.nesterov
        and bitcast[DType.uint32](a.lr) == bitcast[DType.uint32](b.lr)
        and bitcast[DType.uint32](a.beta1) == bitcast[DType.uint32](b.beta1)
        and bitcast[DType.uint32](a.beta2) == bitcast[DType.uint32](b.beta2)
        and bitcast[DType.uint32](a.eps) == bitcast[DType.uint32](b.eps)
        and bitcast[DType.uint32](a.weight_decay) == bitcast[DType.uint32](b.weight_decay)
        and bitcast[DType.uint32](a.momentum) == bitcast[DType.uint32](b.momentum)
        and bitcast[DType.uint32](a.dampening) == bitcast[DType.uint32](b.dampening)
        and bitcast[DType.uint32](a.max_norm) == bitcast[DType.uint32](b.max_norm))


def _require_open(session: ByteLMSession) raises:
    if session.busy:
        raise Error("byte LM: session is busy")
    if not session.trainer or not session.ctx:
        raise Error("byte LM: session is not open")
    if not session.usable:
        raise Error("byte LM: session lost; restore a retained export")


def _mark_if_lost(mut session: ByteLMSession):
    """After a failed call: a trainer whose `healthy` stayed False is a lost
    context (its rollback re-scan did not answer); the session is then
    unusable until it is closed, and `byte_lm_session_info` says so."""
    if session.trainer:
        if not session.trainer.value().healthy:
            session.usable = False


def _admit_session_scalars(session: ByteLMSession, completed: Int, cfg: OptimizerConfig,
                           flags: List[Bool], shape: ByteConfig) raises:
    """The per-step scalar admission kept from `_byte_lm_run` (design 3,
    item 3): profile, completed step, optimizer bits and the flags against
    `buf_initialized`. Scalars and `n_tensors` bools; no array."""
    if session.trainer.value().config.profile() != shape.profile():
        raise Error("byte LM: resident model shape mismatch")
    if session.trainer.value().completed_steps != completed:
        raise Error("byte LM: resident completed-step mismatch")
    if not _same_optimizer_bits(session.trainer.value().optimizer, cfg):
        raise Error("byte LM: resident optimizer mismatch")
    if len(flags) != len(session.trainer.value().buffers.buf_initialized):
        raise Error("byte LM: resident flags mismatch")
    for i in range(len(flags)):
        if flags[i] != session.trainer.value().buffers.buf_initialized[i]:
            raise Error("byte LM: resident flags mismatch")


def _stage_download(ctx: DeviceContext, mut host: HostBuffer[DType.float32],
                    mut buf: DeviceBuffer[DType.float32], n: Int) raises:
    """`n` floats of a device buffer into a pinned host buffer the caller
    owns (the export staging of design section 3); waits inside."""
    if n < 1:
        return
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    ctx.synchronize()


def byte_lm_fault_inject_available_binding() raises -> PythonObject:
    """True only in a gate build compiled with -D MOJOLEARN_BYTE_LM_FAULT_INJECT=1."""
    return PythonObject(byte_lm_fault_inject_available())


def byte_lm_session_open_binding(session: PythonObject, addresses: PythonObject,
                                 params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """Admit host state ONCE and upload it ONCE (design 1.1 item 1).

    addresses[4] = [in_param, in_m, in_v, in_flags_i32]; params[12] as
    `_byte_lm_run` (action ignored; completed is params[1]). The Lists are
    validated by `byte_validate_state` before any device operation and the
    trainer re-validates them at construction. Returns the admitted
    completed step. Refused on an open session: close it first.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var n = cfg_shape.n_total()
    var n_tensors = cfg_shape.n_tensors()
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 4)
    var cells: List[Int] = [n, n, n, n_tensors]
    _validate_slot_table(addr, cells, 4)
    if owner[].busy:
        raise Error("byte LM: session is busy")
    if Bool(owner[].trainer) or Bool(owner[].ctx):
        raise Error("byte LM: session is already open; close it first")
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var keep_context = String(getenv("MOJOLEARN_BYTE_LM_KEEP_CONTEXT")) == "1"
    var tk = Int(perf_counter_ns())
    # No GPU work before all borrowed inputs become validated owned host lists.
    var initial_p = _read_f32(addr[0], n)
    var initial_m = _read_f32(addr[1], n)
    var initial_v = _read_f32(addr[2], n)
    var flags = _read_flags(addr[3], n_tensors)
    _btick(ton, tk, "open.bind_read_inputs")
    _bbytes(ton, "open.bind_read_inputs_bytes", 3 * n * 4 + n_tensors * 4)
    byte_validate_state(initial_p, initial_m, initial_v, flags, completed, cfg_shape)
    _btick(ton, tk, "open.bind_validate_inputs")
    owner[].busy = True
    owner[].usable = False
    try:
        with GILReleased(Python()):
            if keep_context:
                # DEVIATION 2513: the same keeper the per-call path uses.
                BYTE_LM_CONTEXT_KEEPER.get_or_create_ptr()[].ensure()
            owner[].ctx = DeviceContext()
            owner[].trainer = ByteTrainer(owner[].ctx.value(), initial_p, initial_m,
                initial_v, flags, completed, cfg, cfg_shape)
            owner[].ctx.value().synchronize()
            # Context creation, allocation and the one upload of 3n floats.
            _btick(ton, tk, "open.bind_context_and_upload")
            _bbytes(ton, "open.bind_context_and_upload_bytes", 3 * n * 4)
    except error:
        owner[].busy = False
        owner[].trainer = None
        owner[].ctx = None
        raise error
    owner[].busy = False
    owner[].usable = True
    return PythonObject(completed)


def byte_lm_session_step_binding(session: PythonObject, addresses: PythonObject,
                                 params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """One device-owned step. addresses[4] = [in_ids_i32, in_flags_i32,
    out_loss_f32, out_flags_i32]; params[12] as `_byte_lm_run` with
    action=1. In: the ids (8 KB at the target) and the flags for the scalar
    admission. Out: the 4 B loss and the flags. Returns completed+1. A
    failure after the update is rolled back on the device by the trainer
    before it propagates; a lost context marks the session unusable.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var n_tensors = cfg_shape.n_tensors()
    var n_ids = cfg_shape.batch * (cfg_shape.length + 1)
    if _params_action(params) != 1:
        raise Error("byte LM: session step requires action 1")
    var completed = _params_completed(params)
    if completed >= 999999:
        raise Error("byte LM: completed step outside admitted bound")
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 4)
    var cells: List[Int] = [n_ids, n_tensors, 1, n_tensors]
    _validate_slot_table(addr, cells, 2)
    var ids = _read_ids(addr[0], n_ids)
    var flags = _read_flags(addr[1], n_tensors)
    byte_validate_tokens(ids, cfg_shape)
    _require_open(owner[])
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    var loss = Float32(0)
    var result_step = completed
    var out_flags = List[Bool]()
    # DEVIATION 2630: the step's launch, synchronize, copy and allocation
    # counts (core/step_phase.mojo): zeros and no print on a build without
    # -D MOJOLEARN_STEP_PHASE_TIMERS=1.
    var counts0 = step_counts_now()
    owner[].busy = True
    try:
        with GILReleased(Python()):
            ref ctx = owner[].ctx.value()
            _admit_session_scalars(owner[], completed, cfg, flags, cfg_shape)
            _btick(ton, tk, "step.bind_scalar_admission")
            var result = byte_train_step_resident(ctx, owner[].trainer.value(), ids)
            if ton:
                tk = Int(perf_counter_ns())
            loss = result.loss
            result_step = result.completed_steps
            out_flags = result.flags.copy()
            if result_step != completed + 1:
                raise Error("byte LM: successful result has wrong completed step")
            if _nonfinite(loss):
                raise Error("byte LM: nonfinite returned loss")
            if len(out_flags) != n_tensors:
                raise Error("byte LM: wrong flags length")
            step_count_sync()
            ctx.synchronize()
            _btick(ton, tk, "step.bind_final_sync")
            step_counts_report(counts0)
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    _write_flags(addr[3], out_flags)
    _write_loss(addr[2], loss)
    _btick(ton, tk, "step.bind_publish")
    _bbytes(ton, "step.bind_publish_bytes", 4 + n_tensors * 4)
    return PythonObject(result_step)


def byte_lm_session_eval_binding(session: PythonObject, addresses: PythonObject,
                                 params: PythonObject, shape: PythonObject) raises -> PythonObject:
    """Forward-only loss on the resident state. addresses[3] = [in_ids_i32,
    in_flags_i32, out_loss_f32]; params[12] with action=0. Returns the
    unchanged completed step. No state download (design 2.3); gate G1
    exports before and after an evaluation to show it writes nothing.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var n_tensors = cfg_shape.n_tensors()
    var n_ids = cfg_shape.batch * (cfg_shape.length + 1)
    if _params_action(params) != 0:
        raise Error("byte LM: session eval requires action 0")
    var completed = _params_completed(params)
    var cfg = _params_optimizer(params)
    var addr = _read_addresses(addresses, 3)
    var cells: List[Int] = [n_ids, n_tensors, 1]
    _validate_slot_table(addr, cells, 2)
    var ids = _read_ids(addr[0], n_ids)
    var flags = _read_flags(addr[1], n_tensors)
    byte_validate_tokens(ids, cfg_shape)
    _require_open(owner[])
    var loss = Float32(0)
    var result_step = completed
    owner[].busy = True
    try:
        with GILReleased(Python()):
            ref ctx = owner[].ctx.value()
            _admit_session_scalars(owner[], completed, cfg, flags, cfg_shape)
            loss = byte_eval_loss_resident(ctx, owner[].trainer.value(), ids)
            result_step = owner[].trainer.value().completed_steps
            if result_step != completed:
                raise Error("byte LM: successful result has wrong completed step")
            if _nonfinite(loss):
                raise Error("byte LM: nonfinite returned loss")
            for i in range(n_tensors):
                if flags[i] != owner[].trainer.value().buffers.buf_initialized[i]:
                    raise Error("byte LM eval changed momentum flags")
            ctx.synchronize()
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    _write_loss(addr[2], loss)
    return PythonObject(result_step)


def byte_lm_session_export_state_binding(session: PythonObject, addresses: PythonObject,
                                         shape: PythonObject) raises -> PythonObject:
    """Copy the resident state out. addresses[4] = [out_param, out_m,
    out_v, out_flags_i32], all outputs, all disjoint. The device state is
    re-scanned (`byte_validate_device_state`) before the download, staged
    in pinned host buffers and published into the caller's arrays with
    `copy_f32` only after the synchronized scope succeeds; no List is
    built and the session retains no pointer. Returns the completed step.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var n = cfg_shape.n_total()
    var n_tensors = cfg_shape.n_tensors()
    var addr = _read_addresses(addresses, 4)
    var cells: List[Int] = [n, n, n, n_tensors]
    _validate_slot_table(addr, cells, 0)
    _require_open(owner[])
    if owner[].trainer.value().config.profile() != cfg_shape.profile():
        raise Error("byte LM: resident model shape mismatch")
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    ref ctx = owner[].ctx.value()
    var hp = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var flags = List[Bool]()
    var step = 0
    owner[].busy = True
    try:
        with GILReleased(Python()):
            owner[].trainer.value().validate_device_state(ctx, owner[].trainer.value().completed_steps)
            _btick(ton, tk, "export.validate_device_state")
            _stage_download(ctx, hp, owner[].trainer.value().buffers.param, n)
            _stage_download(ctx, hm, owner[].trainer.value().buffers.m_state, n)
            _stage_download(ctx, hv, owner[].trainer.value().buffers.v_state, n)
            flags = owner[].trainer.value().buffers.buf_initialized.copy()
            step = owner[].trainer.value().completed_steps
            ctx.synchronize()
            _btick(ton, tk, "export.download_state")
            _bbytes(ton, "export.download_state_bytes", 3 * n * 4)
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    copy_f32(hp.unsafe_ptr(), f32_ptr(addr[0]), n)
    copy_f32(hm.unsafe_ptr(), f32_ptr(addr[1]), n)
    copy_f32(hv.unsafe_ptr(), f32_ptr(addr[2]), n)
    _write_flags(addr[3], flags)
    _btick(ton, tk, "export.publish")
    _bbytes(ton, "export.publish_bytes", 3 * n * 4 + n_tensors * 4)
    _ = hp^
    _ = hm^
    _ = hv^
    return PythonObject(step)


def byte_lm_session_export_gradients_binding(session: PythonObject, addresses: PythonObject,
                                             shape: PythonObject) raises -> PythonObject:
    """Copy the gradient of the LAST completed step out. addresses[1] =
    [out_grad]. Refused unless a step completed since open, rollback or
    any failure (`grad_step == completed_steps`); the device `grad` is
    scratch for the next backward. Returns the step the gradient belongs to.
    """
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var n = cfg_shape.n_total()
    var addr = _read_addresses(addresses, 1)
    var cells: List[Int] = [n]
    _validate_slot_table(addr, cells, 0)
    _require_open(owner[])
    if owner[].trainer.value().config.profile() != cfg_shape.profile():
        raise Error("byte LM: resident model shape mismatch")
    var grad_step = owner[].trainer.value().grad_step
    if grad_step < 0 or grad_step != owner[].trainer.value().completed_steps:
        raise Error("byte LM: no gradient to export; complete a step first")
    var ton = String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    ref ctx = owner[].ctx.value()
    var hg = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _stage_download(ctx, hg, owner[].trainer.value().buffers.grad, n)
            ctx.synchronize()
            _btick(ton, tk, "export.download_gradients")
            _bbytes(ton, "export.download_gradients_bytes", n * 4)
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    for i in range(n):
        if _nonfinite(hg.unsafe_ptr().unsafe_load(i)):
            raise Error("byte LM: nonfinite returned gradient")
    copy_f32(hg.unsafe_ptr(), f32_ptr(addr[0]), n)
    _btick(ton, tk, "export.publish_gradients")
    _ = hg^
    return PythonObject(grad_step)


def byte_lm_session_rollback_binding(session: PythonObject) raises -> PythonObject:
    """Restore the device state from the shadow of the last step that
    reached its shadow point (design 4.2 item 6): for a failure the Python
    layer detects after native success. Nothing to restore (no such step,
    or the trainer already rolled back inside the step) is not an error.
    Returns the completed step after the call. A rollback whose re-scan
    does not answer marks the session lost.
    """
    var owner = session.downcast_value_ptr[ByteLMSession]()
    _require_open(owner[])
    owner[].busy = True
    try:
        with GILReleased(Python()):
            _ = byte_rollback(owner[].ctx.value(), owner[].trainer.value())
            owner[].ctx.value().synchronize()
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    return PythonObject(owner[].trainer.value().completed_steps)


def byte_lm_session_info_binding(session: PythonObject) raises -> PythonObject:
    """[completed_steps, grad_step, usable, open] as integers; -1, -1 for
    the two steps when no trainer is open. Reads fields only."""
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var completed = -1
    var grad_step = -1
    var is_open = 0
    if owner[].trainer:
        completed = owner[].trainer.value().completed_steps
        grad_step = owner[].trainer.value().grad_step
        is_open = 1
    var out = Python.list()
    out.append(PythonObject(completed))
    out.append(PythonObject(grad_step))
    out.append(PythonObject(1 if owner[].usable else 0))
    out.append(PythonObject(is_open))
    return out


def _byte_config(shape: PythonObject) raises -> ByteConfig:
    if len(shape) != 7 and len(shape) != 9:
        raise Error("byte LM: expected 7 or 9 shape integers (B,L,DM,H,KV,HD,FF[,layers,vocab])")
    var operator_module = Python.import_module("operator")
    var values = List[Int]()
    for i in range(len(shape)):
        var type_name = String(py=shape[i].__class__.__name__)
        if type_name == "bool" or type_name == "bool_":
            raise Error("byte LM: shape dimensions must be integers, not booleans")
        values.append(Int(py=operator_module.index(shape[i])))
    if len(values) == 7:
        values.append(2)
        values.append(256)
    var cfg = ByteConfig(values[0], values[1], values[2], values[3],
                         values[4], values[5], values[6], values[7], values[8])
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


def byte_lm_session_run_binding(session: PythonObject, addresses: PythonObject,
                                params: PythonObject, shape: PythonObject) raises -> PythonObject:
    var cfg = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    return _byte_lm_run(addresses, params, cfg, owner[], retain=True)


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
    for i in range(2):
        var type_name = String(py=dims[i].__class__.__name__)
        if type_name == "bool" or type_name == "bool_":
            raise Error("byte LM logits: dims must be integers, not booleans")
        out.append(Int(py=operator_module.index(dims[i])))
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
    var m = bl[0] * bl[1]
    var cells_out = m * cfg.vocab_size
    var n = cfg.n_total()
    var addr = _read_addresses(addresses, 3)
    var cells: List[Int] = [n, m, cells_out]
    _validate_slot_table(addr, cells, 2)
    var params = _read_f32(addr[0], n)
    var ids = _read_ids(addr[1], m)
    byte_logits_validate(ids, bl[0], bl[1], cfg)
    byte_logits_validate_params(params, cfg)
    var keep_context = String(getenv("MOJOLEARN_BYTE_LM_KEEP_CONTEXT")) == "1"
    var logits = List[Float32]()
    var session = ByteLMSession()
    session.busy = True
    try:
        with GILReleased(Python()):
            if keep_context:
                # DEVIATION 2513: the same keeper the other per-call path uses.
                BYTE_LM_CONTEXT_KEEPER.get_or_create_ptr()[].ensure()
            session.ctx = DeviceContext()
            logits = byte_logits_from_params(session.ctx.value(), params, ids, bl[0], bl[1], cfg)
            session.ctx.value().synchronize()
            session.ctx = None
    except error:
        session.busy = False
        raise error
    session.busy = False
    session.usable = False
    if len(logits) != cells_out:
        raise Error("byte LM logits: wrong logits length")
    copy_f32(logits.unsafe_ptr(), f32_ptr(addr[2]), cells_out)
    return PythonObject(cells_out)


def byte_lm_session_logits_binding(session: PythonObject, addresses: PythonObject,
                                   dims: PythonObject, shape: PythonObject,
                                   completed: PythonObject) raises -> PythonObject:
    """Forward-only logits on an open resident session. addresses[2] =
    [in_ids_i32 (batch * length), out_logits_f32 (batch * length * vocab)];
    dims = [batch, length]; `completed` is the caller's committed step, which
    must equal the session's, the scalar half of the admission
    `byte_lm_session_eval` performs (a read-only call that silently used a
    different state than the caller believes is a wrong provenance, not a
    convenience). Writes no parameter, moment, flag or step. Returns the
    number of logits written."""
    _require_binding_profile()
    var cfg_shape = _byte_config(shape)
    var owner = session.downcast_value_ptr[ByteLMSession]()
    var bl = _logits_dims(dims, cfg_shape)
    var claimed = Int(py=Python.import_module("operator").index(completed))
    if claimed < 0 or claimed >= 1000000:
        raise Error("byte LM: completed step outside admitted bound")
    var m = bl[0] * bl[1]
    var cells_out = m * cfg_shape.vocab_size
    var addr = _read_addresses(addresses, 2)
    var cells: List[Int] = [m, cells_out]
    _validate_slot_table(addr, cells, 1)
    var ids = _read_ids(addr[0], m)
    byte_logits_validate(ids, bl[0], bl[1], cfg_shape)
    _require_open(owner[])
    if owner[].trainer.value().config.profile() != cfg_shape.profile():
        raise Error("byte LM: resident model shape mismatch")
    if owner[].trainer.value().completed_steps != claimed:
        raise Error("byte LM: resident completed-step mismatch")
    var step_before = owner[].trainer.value().completed_steps
    var logits = List[Float32]()
    owner[].busy = True
    try:
        with GILReleased(Python()):
            ref ctx = owner[].ctx.value()
            logits = byte_logits_resident(ctx, owner[].trainer.value(), ids, bl[0], bl[1])
            if owner[].trainer.value().completed_steps != step_before:
                raise Error("byte LM logits changed the completed step")
            ctx.synchronize()
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    if len(logits) != cells_out:
        raise Error("byte LM logits: wrong logits length")
    copy_f32(logits.unsafe_ptr(), f32_ptr(addr[1]), cells_out)
    return PythonObject(cells_out)


@export
def PyInit__mojolearn_byte_lm() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_byte_lm")
        module.def_function[byte_lm_numeric_mode_binding]("byte_lm_numeric_mode")
        module.def_function[byte_lm_vendor_binding]("byte_lm_vendor")
        module.def_function[byte_lm_profile_binding]("byte_lm_profile")
        module.def_function[byte_lm_run_binding]("byte_lm_run")
        module.def_function[byte_lm_config_profile_binding]("byte_lm_config_profile")
        module.def_function[byte_lm_run_configured_binding]("byte_lm_run_configured")
        _ = module.add_type[ByteLMSession]("_ByteLMSession")
        module.def_function[byte_lm_session_create_binding]("byte_lm_session_create")
        module.def_function[byte_lm_session_close_binding]("byte_lm_session_close")
        module.def_function[byte_lm_session_run_binding]("byte_lm_session_run")
        module.def_function[byte_lm_context_keeper_active_binding]("byte_lm_context_keeper_active")
        # DEVIATION 2514: the device-owned session entries.
        module.def_function[byte_lm_session_open_binding]("byte_lm_session_open")
        module.def_function[byte_lm_session_step_binding]("byte_lm_session_step")
        module.def_function[byte_lm_session_eval_binding]("byte_lm_session_eval")
        module.def_function[byte_lm_session_export_state_binding]("byte_lm_session_export_state")
        module.def_function[byte_lm_session_export_gradients_binding]("byte_lm_session_export_gradients")
        # DEVIATION 2658: forward-only logits.
        module.def_function[byte_lm_logits_binding]("byte_lm_logits")
        module.def_function[byte_lm_session_logits_binding]("byte_lm_session_logits")
        module.def_function[byte_lm_session_rollback_binding]("byte_lm_session_rollback")
        module.def_function[byte_lm_session_info_binding]("byte_lm_session_info")
        module.def_function[byte_lm_fault_inject_available_binding]("byte_lm_fault_inject_available")
        # DEVIATION 2534: the attention arm read-back (arm, default, trial, resolved).
        module.def_function[byte_lm_attention_arm_binding]("byte_lm_attention_arm")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_byte_lm: ", error))
