# SPDX-License-Identifier: Apache-2.0
# Stateless whole training/evaluation owner extracted from its original binding.
# GIL policy is provided by the shared admission layer. The optional locked
# teardown remains a separate finalization phase before publication to users.
from std.os import getenv
from std.time import perf_counter_ns
from std.memory import bitcast
from core.hostptr import list_i32
from core.neural_context import neural_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from training.byte_lm_afn import BYTE_LM_FAST_APPLE
from training.byte_lm_binding_impl import (ByteLMSession, BYTE_LM_CONTEXT_KEEPER,
    _validate_addresses, _read_f32, _require_inputs_unchanged, _write_f32,
    _btick, _bbytes, _require_same_bits, _write_flags_n, _write_loss)
from training.byte_lm_session_ops import BYTE_LM_NEURAL_CTX
from training.byte_lm_config import ByteConfig
from training.checks.optimizer_contract import OptimizerConfig
from training.checks.train_loop import download_f32
from training.byte_lm import (ByteTrainer, byte_train_step, byte_eval_loss,
    byte_validate_state, byte_validate_device_state, byte_validate_optimizer, byte_validate_tokens)
comptime _NEURAL_CTX = BYTE_LM_NEURAL_CTX


def byte_lm_run_native(addr: List[Int], action: Int, completed: Int, cfg: OptimizerConfig,
                       shape: ByteConfig, mut session: ByteLMSession, retain: Bool,
                       teardown_with_gil: Bool) raises -> Int:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and not BYTE_LM_FAST_APPLE:
        raise Error("byte LM binding requires IDENTICAL (or FAST on Apple)")
    if String(COMPILED_VENDOR) != "cuda" and String(COMPILED_VENDOR) != "hip" and String(COMPILED_VENDOR) != "metal":
        raise Error("byte LM binding requires CUDA, HIP or Metal")
    if len(addr) != 11:
        raise Error("byte LM: expected 11 addresses")
    shape.validate()
    var n = shape.n_total()
    if action != 0 and action != 1:
        raise Error("byte LM: action/nesterov must be 0 or 1")
    if completed < 0 or completed >= 1000000 or (action == 1 and completed >= 999999):
        raise Error("byte LM: completed step outside admitted bound")
    byte_validate_optimizer(cfg)
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
    var tk = Int(perf_counter_ns())
    # No GPU work before all borrowed inputs become validated owned host lists.
    var initial_p = _read_f32(addr[0], n)
    var initial_m = _read_f32(addr[1], n)
    var initial_v = _read_f32(addr[2], n)
    var flags_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[3])
    var ids_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[4])
    var flags = List[Bool]()
    var ids = List[Int32]()
    for i in range(shape.n_tensors()):  # small-loop(n_tensors: momentum flags, one per parameter tensor): validates per tensor flags, not data
        var flag = flags_ptr.unsafe_load(i)
        if flag != 0 and flag != 1:
            raise Error("byte LM: momentum flags must be exactly 0 or 1")
        flags.append(flag == 1)
    # one memcpy of the borrowed token ids (no per-element host loop)
    ids = list_i32(ids_ptr, shape.batch * (shape.length + 1))
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
        var reused = Bool(session.ctx)
        if not session.ctx:
            if keep_context:
                # DEVIATION 2513: a SEPARATE keeper context, created
                # before the per-call one and never released. Idempotent
                # after the first creating call. Off: this branch is
                # not entered and nothing below changes.
                BYTE_LM_CONTEXT_KEEPER.get_or_create_ptr()[].ensure()
            session.ctx = neural_ctx[_NEURAL_CTX]()
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
            _require_same_bits(ctx, initial_p, session.trainer.value().buffers.param, n)
            _require_same_bits(ctx, initial_m, session.trainer.value().buffers.m_state, n)
            _require_same_bits(ctx, initial_v, session.trainer.value().buffers.v_state, n)
            for i in range(shape.n_tensors()):  # small-loop(n_tensors: momentum flags, one per parameter tensor): compares per tensor flags, not data
                if flags[i] != session.trainer.value().buffers.buf_initialized[i]:
                    raise Error("byte LM: resident flags mismatch")
            # Three device bit compares (each waits inside on one int
            # read back); the device queue is empty at this tick.
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
            _require_same_bits(ctx, initial_p, session.trainer.value().buffers.param, n)
            _require_same_bits(ctx, initial_m, session.trainer.value().buffers.m_state, n)
            _require_same_bits(ctx, initial_v, session.trainer.value().buffers.v_state, n)
            for i in range(shape.n_tensors()):  # small-loop(n_tensors: momentum flags, one per parameter tensor): compares per tensor flags, not data
                if flags[i] != out_flags[i]:
                    raise Error("byte LM eval changed momentum flags")
            _btick(ton, tk, "step.bind_eval_readback")
            _bbytes(ton, "step.bind_eval_readback_bytes", 3 * n * 4)
        # lane cgr4-download-loop: the returned state is validated where
        # it lives (the trainer's device buffers it was downloaded from):
        # device scans, the same predicates and messages, no host walk
        ref tr = session.trainer.value()
        byte_validate_device_state(ctx, tr.scan, tr.buffers.param, tr.buffers.m_state,
                                   tr.buffers.v_state, out_flags, result_step, shape)
        if result_step != completed + action:
            raise Error("byte LM: successful result has wrong completed step")
        if (bitcast[DType.uint32](loss) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("byte LM: nonfinite returned loss")
        if action == 1:
            if len(out_g) != n:
                raise Error("byte LM: wrong gradient length")
            if tr.scan.first_nonfinite(ctx, tr.buffers.grad, n) >= 0:
                raise Error("byte LM: nonfinite returned gradient")
        # Device scans of param/m/v (byte_validate_device_state) and grad.
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
            session.logits_scratch = None
            session.trainer = None
            session.ctx.value().synchronize()
            session.ctx = None
    except error:
        session.busy = False
        raise error
    session.busy = False
    session.usable = retain
    # lane py-runtime round 2: the input-unchanged guard runs here, in the
    # binding, before anything is published
    _require_inputs_unchanged(addr, initial_p, initial_m, initial_v, flags, ids, shape)
    # Publication starts after the synchronized GPU scope succeeds. The
    # Python wrapper commits these fresh arrays atomically to its state object.
    _write_f32(addr[5], out_p)
    _write_f32(addr[6], out_m)
    _write_f32(addr[7], out_v)
    if action == 1:
        _write_f32(addr[8], out_g)
    _write_flags_n(addr[9], out_flags, shape.n_tensors())
    _write_loss(addr[10], loss)
    # Host-to-host: the four n-float outputs copied into Python memory.
    _btick(ton, tk, "step.bind_publish")
    _bbytes(ton, "step.bind_publish_bytes", (3 + action) * n * 4)
    return result_step


def byte_lm_run_finalize(mut session: ByteLMSession) raises:
    # DEVIATION 2518 variant (b): the same two releases, after the
    # GILReleased block has re-acquired the GIL (the resident close()
    # shape: synchronized above, released with the GIL held). Off, the
    # teardown above ran and both Optionals are already empty.
    print("byte LM teardown variant: teardown_with_gil")
    session.logits_scratch = None
    session.trainer = None
    # DEVIATION 2520: drain the enqueued frees before the context goes.
    if session.ctx:
        session.ctx.value().synchronize()
    session.ctx = None
