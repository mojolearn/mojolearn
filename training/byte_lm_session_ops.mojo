# SPDX-License-Identifier: Apache-2.0
"""The byte LM resident session's operations and the single-device trainer
owner behind `mojolearn.SmallByteLanguageModelTrainer`.

The session operations (open, step, eval, state and gradient export,
rollback, logits, next bytes) are the bodies of the `byte_lm_session_*`
entries of `bindings/_mojolearn_byte_lm.mojo`, moved here word for word with
their order kept; the binding unpacks its Python arguments, calls ONE
function here with the GIL released and converts the typed result back.

The trainer owner (`byte_owner_*`) composes those same operations into the
one native call each public trainer method makes: the admission of the host
state when no session is open, the step or evaluation, the gradient export
and its split into named tensors, and the rollback after a failure. No
arithmetic is added: every number comes from the session's `ByteTrainer`
(training/byte_lm.mojo), so the bits are the session entries' bits.

No Python here.
"""
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, HostBuffer
from core.hostptr import f32_ptr, i32_ptr, copy_f32
from core.neural_context import neural_ctx
from core.sha256 import Sha256
from core.step_phase import step_count_sync, step_counts_now, step_counts_report
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.checks.optimizer_contract import OptimizerConfig
from training.byte_lm_config import ByteConfig
from training.byte_lm import (
    ByteTrainer, byte_train_step_resident, byte_eval_loss_resident, byte_rollback,
    byte_validate_state, byte_validate_tokens,
)
from training.byte_lm_logits import (
    byte_logits_validate, byte_logits_resident_into, byte_next_bytes_resident_into,
)
from training.byte_lm_binding_impl import (
    ByteLMSession, BYTE_LM_CONTEXT_KEEPER,
    _read_f32, _btick, _bbytes, _validate_slot_table,
    _read_flags, _read_ids, _write_flags, _write_loss, _nonfinite,
    _require_open, _mark_if_lost, _admit_session_scalars, _stage_download,
)

# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime BYTE_LM_NEURAL_CTX = "MojoNeuralByteLMContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralByteLMContextFast"

comptime _FILE_CHUNK = 1048576


# ---------------------------------------------------------------------------
# The session operations (moved from the `byte_lm_session_*` entries).
# ---------------------------------------------------------------------------


def byte_session_open(mut session: ByteLMSession, addr: List[Int], completed: Int,
                      cfg: OptimizerConfig, cfg_shape: ByteConfig) raises -> Int:
    """Admit host state ONCE and upload it ONCE (design 1.1 item 1).

    addr[4] = [in_param, in_m, in_v, in_flags_i32]; `completed` and `cfg`
    are the caller's admitted step and optimizer. The Lists are validated by
    `byte_validate_state` before any device operation. Returns the admitted
    completed step. Refused on an open session: close it first.
    """
    var owner = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n = cfg_shape.n_total()
    var n_tensors = cfg_shape.n_tensors()
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
        if keep_context:
            # DEVIATION 2513: the same keeper the per-call path uses.
            BYTE_LM_CONTEXT_KEEPER.get_or_create_ptr()[].ensure()
        owner[].ctx = neural_ctx[BYTE_LM_NEURAL_CTX]()
        owner[].trainer = ByteTrainer(owner[].ctx.value(), initial_p, initial_m,
            initial_v, flags, completed, cfg, cfg_shape)
        owner[].ctx.value().synchronize()
        # Context creation, allocation and the one upload of 3n floats.
        _btick(ton, tk, "open.bind_context_and_upload")
        _bbytes(ton, "open.bind_context_and_upload_bytes", 3 * n * 4)
    except error:
        owner[].busy = False
        owner[].logits_scratch = None
        owner[].trainer = None
        owner[].ctx = None
        raise error
    owner[].busy = False
    owner[].usable = True
    return completed


def byte_session_step_run(mut session: ByteLMSession, ids: List[Int32], flags: List[Bool],
                          loss_addr: Int, flags_addr: Int, completed: Int,
                          cfg: OptimizerConfig, cfg_shape: ByteConfig) raises -> Int:
    """One device-owned step on an open session, after its inputs were read
    and admitted. Out: the 4 B loss at `loss_addr` and the flags at
    `flags_addr`. Returns completed+1. A failure after the update is rolled
    back on the device by the trainer before it propagates; a lost context
    marks the session unusable."""
    var owner = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n_tensors = cfg_shape.n_tensors()
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
    _write_flags(flags_addr, out_flags)
    _write_loss(loss_addr, loss)
    _btick(ton, tk, "step.bind_publish")
    _bbytes(ton, "step.bind_publish_bytes", 4 + n_tensors * 4)
    return result_step


def byte_session_step(mut session: ByteLMSession, addr: List[Int], completed: Int,
                      cfg: OptimizerConfig, cfg_shape: ByteConfig) raises -> Int:
    """One device-owned step. addr[4] = [in_ids_i32, in_flags_i32,
    out_loss_f32, out_flags_i32]. In: the ids and the flags for the scalar
    admission. Out: the 4 B loss and the flags. Returns completed+1."""
    var n_tensors = cfg_shape.n_tensors()
    var n_ids = cfg_shape.batch * (cfg_shape.length + 1)
    if completed >= 999999:
        raise Error("byte LM: completed step outside admitted bound")
    var cells: List[Int] = [n_ids, n_tensors, 1, n_tensors]
    _validate_slot_table(addr, cells, 2)
    var ids = _read_ids(addr[0], n_ids)
    var flags = _read_flags(addr[1], n_tensors)
    byte_validate_tokens(ids, cfg_shape)
    _require_open(session)
    return byte_session_step_run(session, ids, flags, addr[2], addr[3], completed, cfg, cfg_shape)


def byte_session_eval_run(mut session: ByteLMSession, ids: List[Int32], flags: List[Bool],
                          loss_addr: Int, completed: Int, cfg: OptimizerConfig,
                          cfg_shape: ByteConfig) raises -> Int:
    """Forward-only loss on the resident state of an open session, after its
    inputs were read and admitted. Writes the loss at `loss_addr`. Returns
    the unchanged completed step."""
    var owner = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n_tensors = cfg_shape.n_tensors()
    var loss = Float32(0)
    var result_step = completed
    owner[].busy = True
    try:
        ref ctx = owner[].ctx.value()
        _admit_session_scalars(owner[], completed, cfg, flags, cfg_shape)
        loss = byte_eval_loss_resident(ctx, owner[].trainer.value(), ids)
        result_step = owner[].trainer.value().completed_steps
        if result_step != completed:
            raise Error("byte LM: successful result has wrong completed step")
        if _nonfinite(loss):
            raise Error("byte LM: nonfinite returned loss")
        for i in range(n_tensors):  # small-loop(n_tensors: momentum flags, one per parameter tensor): compares per tensor flags, not data
            if flags[i] != owner[].trainer.value().buffers.buf_initialized[i]:
                raise Error("byte LM eval changed momentum flags")
        ctx.synchronize()
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    _write_loss(loss_addr, loss)
    return result_step


def byte_session_eval(mut session: ByteLMSession, addr: List[Int], completed: Int,
                      cfg: OptimizerConfig, cfg_shape: ByteConfig) raises -> Int:
    """Forward-only loss on the resident state. addr[3] = [in_ids_i32,
    in_flags_i32, out_loss_f32]. Returns the unchanged completed step. No
    state download (design 2.3)."""
    var n_tensors = cfg_shape.n_tensors()
    var n_ids = cfg_shape.batch * (cfg_shape.length + 1)
    var cells: List[Int] = [n_ids, n_tensors, 1]
    _validate_slot_table(addr, cells, 2)
    var ids = _read_ids(addr[0], n_ids)
    var flags = _read_flags(addr[1], n_tensors)
    byte_validate_tokens(ids, cfg_shape)
    _require_open(session)
    return byte_session_eval_run(session, ids, flags, addr[2], completed, cfg, cfg_shape)


def byte_session_export_state(mut session: ByteLMSession, addr: List[Int],
                              cfg_shape: ByteConfig) raises -> Int:
    """Copy the resident state out. addr[4] = [out_param, out_m, out_v,
    out_flags_i32], all outputs, all disjoint. The device state is re-scanned
    (`byte_validate_device_state`) before the download, staged in pinned host
    buffers and published into the caller's arrays with `copy_f32` only after
    the synchronized scope succeeds; no List is built and the session retains
    no pointer. Returns the completed step."""
    var owner = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n = cfg_shape.n_total()
    var n_tensors = cfg_shape.n_tensors()
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
    return step


def byte_session_export_gradients(mut session: ByteLMSession, out_addr: Int,
                                  cfg_shape: ByteConfig) raises -> Int:
    """Copy the gradient of the LAST completed step to `out_addr` (n_total
    floats). Refused unless a step completed since open, rollback or any
    failure (`grad_step == completed_steps`); the device `grad` is scratch for
    the next backward. Returns the step the gradient belongs to."""
    var owner = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n = cfg_shape.n_total()
    var addr: List[Int] = [out_addr]
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
    var bad_g = -1
    owner[].busy = True
    try:
        # lane cgr4-download-loop: the finite scan runs on the device
        bad_g = owner[].trainer.value().scan.first_nonfinite(
            ctx, owner[].trainer.value().buffers.grad, n)
        _stage_download(ctx, hg, owner[].trainer.value().buffers.grad, n)
        ctx.synchronize()
        _btick(ton, tk, "export.download_gradients")
        _bbytes(ton, "export.download_gradients_bytes", n * 4)
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    if bad_g >= 0:
        raise Error("byte LM: nonfinite returned gradient")
    copy_f32(hg.unsafe_ptr(), f32_ptr(addr[0]), n)
    _btick(ton, tk, "export.publish_gradients")
    _ = hg^
    return grad_step


def byte_session_rollback(mut session: ByteLMSession) raises -> Int:
    """Restore the device state from the shadow of the last step that
    reached its shadow point (design 4.2 item 6). Nothing to restore is not
    an error. Returns the completed step after the call. A rollback whose
    re-scan does not answer marks the session lost."""
    var owner = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    _require_open(owner[])
    owner[].busy = True
    try:
        _ = byte_rollback(owner[].ctx.value(), owner[].trainer.value())
        owner[].ctx.value().synchronize()
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    return owner[].trainer.value().completed_steps


def byte_session_logits_run(mut session: ByteLMSession, ids: List[Int32], batch: Int,
                            length: Int, out_addr: Int, claimed: Int,
                            cfg_shape: ByteConfig, next_bytes: Bool) raises -> Int:
    """Forward-only logits (or, with `next_bytes`, the greedy next byte of
    every row, argmax on the device) on an open session, after the ids were
    read and admitted. `claimed` is the caller's committed step, which must
    equal the session's. Writes no parameter, moment, flag or step. Returns
    the number of values written."""
    var owner = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    _require_open(owner[])
    if owner[].trainer.value().config.profile() != cfg_shape.profile():
        raise Error("byte LM: resident model shape mismatch")
    if owner[].trainer.value().completed_steps != claimed:
        raise Error("byte LM: resident completed-step mismatch")
    var step_before = owner[].trainer.value().completed_steps
    owner[].busy = True
    try:
        ref ctx = owner[].ctx.value()
        if next_bytes:
            byte_next_bytes_resident_into(ctx, owner[].trainer.value(), ids, batch, length,
                                          owner[].logits_scratch, i32_ptr(out_addr))
            if owner[].trainer.value().completed_steps != step_before:
                raise Error("byte LM next bytes changed the completed step")
        else:
            # lane/neural-net-experiment: one device-to-host copy into the
            # caller's array (`byte_logits_resident_into`).
            byte_logits_resident_into(ctx, owner[].trainer.value(), ids, batch, length,
                                      owner[].logits_scratch, f32_ptr(out_addr))
            if owner[].trainer.value().completed_steps != step_before:
                raise Error("byte LM logits changed the completed step")
        ctx.synchronize()
    except error:
        owner[].busy = False
        _mark_if_lost(owner[])
        raise error
    owner[].busy = False
    return batch if next_bytes else batch * length * cfg_shape.vocab_size


def byte_session_logits(mut session: ByteLMSession, addr: List[Int], batch: Int, length: Int,
                        claimed: Int, cfg_shape: ByteConfig, next_bytes: Bool) raises -> Int:
    """addr[2] = [in_ids_i32 (batch * length), out (batch * length * vocab
    float32 logits, or batch int32 next bytes)]; batch and length already
    admitted by the binding."""
    var m = batch * length
    var cells: List[Int] = [m, batch if next_bytes else m * cfg_shape.vocab_size]
    _validate_slot_table(addr, cells, 1)
    var ids = _read_ids(addr[0], m)
    byte_logits_validate(ids, batch, length, cfg_shape)
    return byte_session_logits_run(session, ids, batch, length, addr[1], claimed, cfg_shape, next_bytes)


# ---------------------------------------------------------------------------
# The trainer owner: one native call per public trainer method.
# ---------------------------------------------------------------------------


def byte_scatter_named(flat_addr: Int, named: List[Int], cfg_shape: ByteConfig) raises:
    """Copy the flat n_total float vector at `flat_addr` into one caller
    buffer per parameter tensor, in registry order (`ByteConfig.offsets`):
    the named split of an exported gradient. A copy, no arithmetic."""
    var offsets = cfg_shape.offsets()
    if len(named) != cfg_shape.n_tensors():
        raise Error("byte LM: expected one named output per parameter tensor")
    if flat_addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    for j in range(len(named)):  # driver-loop: over model parameter tensors
        var count = offsets[j + 1] - offsets[j]
        if count > 0:
            copy_f32(f32_ptr(flat_addr + 4 * offsets[j]), f32_ptr(named[j]), count)


def _named_cells(cfg_shape: ByteConfig, mut slots: List[Int], mut cells: List[Int],
                 named: List[Int]) raises:
    """Append the named outputs' slots and sizes to a slot table."""
    for j in range(len(named)):  # driver-loop: over model parameter tensors
        slots.append(named[j])
        cells.append(cfg_shape.param_count(j))


def byte_owner_mark(mut session: ByteLMSession, committed: Int):
    """After a failed owner call: the session is lost when its trainer is
    no longer healthy, or when it no longer stands at the committed step.
    Reads fields only."""
    _mark_if_lost(session)
    if session.trainer:
        if session.trainer.value().completed_steps != committed:
            session.usable = False


def byte_owner_recover(mut session: ByteLMSession, committed: Int):
    """The failure path of a step or evaluation (design 4.2 item 6): roll
    the device back to the committed step (a no-op when the trainer already
    rolled back inside the step, or the failure preceded its update), then
    `byte_owner_mark`. A rollback that raises loses the session."""
    if not session.trainer or not session.ctx:
        return
    try:
        _ = byte_session_rollback(session)
    except:
        session.usable = False
        return
    byte_owner_mark(session, committed)


def byte_owner_step(mut session: ByteLMSession, addr: List[Int], completed: Int,
                    cfg: OptimizerConfig, cfg_shape: ByteConfig, train: Bool) raises -> Int:
    """One training step (`train`) or evaluation of the trainer, the whole
    of `train_step` / `evaluate` on a resident session.

    addr = [in_ids_i32 (batch * (length + 1)), in_flags_i32 (the committed
    flags), in_param, in_m, in_v (the host state, admitted and uploaded when
    the session is not open; all three 0 when it is), out_loss_f32,
    out_flags_i32 (training; 0 for an evaluation), out_grad_f32 (n_total; 0
    unless the step returns its gradient), then, after a nonzero out_grad,
    one output per parameter tensor for the gradient by name].

    The ids are admitted first, before any device work; then the session is
    opened from the host state when it is not open; then the step or the
    evaluation runs; then a returned gradient is exported and split by
    name. Any failure after the session opened rolls the device back to
    `completed` (`byte_owner_recover`) before it propagates. Returns the
    completed step after the call.
    """
    var s = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n = cfg_shape.n_total()
    var n_tensors = cfg_shape.n_tensors()
    var n_ids = cfg_shape.batch * (cfg_shape.length + 1)
    if len(addr) < 8:
        raise Error("byte LM trainer: expected at least 8 addresses")
    var full = addr[7] != 0
    if full and not train:
        raise Error("byte LM trainer: an evaluation returns no gradient")
    if len(addr) != (8 + n_tensors if full else 8):
        raise Error("byte LM trainer: expected one named gradient output per parameter tensor")
    if train and completed >= 999999:
        raise Error("byte LM: completed step outside admitted bound")
    if not train and addr[6] != 0:
        raise Error("byte LM trainer: an evaluation requires a null flags output")
    var admit = addr[2] != 0 or addr[3] != 0 or addr[4] != 0
    var slots: List[Int] = [addr[0], addr[1]]
    var cells: List[Int] = [n_ids, n_tensors]
    if admit:
        for i in range(2, 5):  # small-loop(state: three host state address slots)
            slots.append(addr[i])
            cells.append(n)
    var n_inputs = len(slots)
    slots.append(addr[5])
    cells.append(1)
    if train:
        slots.append(addr[6])
        cells.append(n_tensors)
    var named = List[Int]()
    if full:
        slots.append(addr[7])
        cells.append(n)
        for j in range(n_tensors):  # driver-loop: over model parameter tensors
            named.append(addr[8 + j])
        _named_cells(cfg_shape, slots, cells, named)
    _validate_slot_table(slots, cells, n_inputs)
    var ids = _read_ids(addr[0], n_ids)
    var flags = _read_flags(addr[1], n_tensors)
    byte_validate_tokens(ids, cfg_shape)
    if not s[].trainer:
        if not admit:
            raise Error("byte LM: session is not open")
        var state: List[Int] = [addr[2], addr[3], addr[4], addr[1]]
        _ = byte_session_open(s[], state, completed, cfg, cfg_shape)
    _require_open(s[])
    var step = completed
    try:
        if train:
            step = byte_session_step_run(s[], ids, flags, addr[5], addr[6], completed, cfg, cfg_shape)
            if full:
                _ = byte_session_export_gradients(s[], addr[7], cfg_shape)
                byte_scatter_named(addr[7], named, cfg_shape)
        else:
            step = byte_session_eval_run(s[], ids, flags, addr[5], completed, cfg, cfg_shape)
    except error:
        byte_owner_recover(s[], completed)
        raise error
    return step


def byte_owner_export_state(mut session: ByteLMSession, addr: List[Int], completed: Int,
                            cfg_shape: ByteConfig, release: Bool) raises -> Int:
    """The trainer's state export: addr[5] = [in_flags_i32 (the committed
    flags), out_param, out_m, out_v, out_flags_i32]. The resident state is
    re-scanned and downloaded (`byte_session_export_state`); the exported
    step must be `completed` and the exported flags the committed ones.
    With `release`, the session is closed after the export (and also when
    the export fails, so a failed `close()` leaves no device state behind).
    Returns the exported step."""
    var s = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n_tensors = cfg_shape.n_tensors()
    if len(addr) != 5:
        raise Error("byte LM trainer: expected 5 addresses")
    var flag_slot: List[Int] = [addr[0]]
    var flag_cells: List[Int] = [n_tensors]
    _validate_slot_table(flag_slot, flag_cells, 1)
    var expected = _read_flags(addr[0], n_tensors)
    var outputs: List[Int] = [addr[1], addr[2], addr[3], addr[4]]
    var step = 0
    try:
        step = byte_session_export_state(s[], outputs, cfg_shape)
        if step != completed:
            raise Error("Byte-LM state export returned a different completed step")
        var exported = _read_flags(addr[4], n_tensors)
        for i in range(n_tensors):  # small-loop(n_tensors: momentum flags, one per parameter tensor): compares per tensor flags, not data
            if exported[i] != expected[i]:
                raise Error("Byte-LM exported flags differ from the admitted flags")
    except error:
        if release:
            try:
                s[].close()
            except:
                pass
        raise error
    if release:
        s[].close()
    return step


def byte_owner_export_gradients(mut session: ByteLMSession, addr: List[Int], completed: Int,
                                cfg_shape: ByteConfig) raises -> Int:
    """The trainer's gradient export: addr = [out_grad_f32 (n_total)] and,
    for the gradient by name, one output per parameter tensor after it. The
    exported gradient must belong to `completed`. Returns its step."""
    var n_tensors = cfg_shape.n_tensors()
    if len(addr) != 1 and len(addr) != 1 + n_tensors:
        raise Error("byte LM trainer: expected the flat gradient and, optionally, one output per parameter tensor")
    var slots: List[Int] = [addr[0]]
    var cells: List[Int] = [cfg_shape.n_total()]
    var named = List[Int]()
    for j in range(len(addr) - 1):  # driver-loop: over model parameter tensors
        named.append(addr[1 + j])
    _named_cells(cfg_shape, slots, cells, named)
    _validate_slot_table(slots, cells, 0)
    var step = byte_session_export_gradients(session, addr[0], cfg_shape)
    if step != completed:
        raise Error("Byte-LM gradient export returned a different step")
    if len(named) > 0:
        byte_scatter_named(addr[0], named, cfg_shape)
    return step


def byte_owner_logits(mut session: ByteLMSession, addr: List[Int], batch: Int, length: Int,
                      completed: Int, cfg: OptimizerConfig, cfg_shape: ByteConfig,
                      next_bytes: Bool) raises -> Int:
    """The trainer's `logits` / `next_bytes` on a resident session.

    addr = [in_ids_i32 (batch * length), out (float32 logits batch * length
    * vocab, or int32 next bytes batch), in_flags_i32, in_param, in_m, in_v
    (the host state, admitted and uploaded when the session is not open;
    the last three 0 when it is)]. The ids are admitted before any device
    work. Nothing is rolled back on a failure (no state is written); the
    session is lost when it no longer reports itself usable at `completed`.
    Returns the number of values written."""
    var s = MutPointer(to=session).unsafe_origin_cast[MutAnyOrigin]()
    var n = cfg_shape.n_total()
    var n_tensors = cfg_shape.n_tensors()
    if len(addr) != 6:
        raise Error("byte LM trainer: expected 6 addresses")
    var m = batch * length
    var admit = addr[3] != 0 or addr[4] != 0 or addr[5] != 0
    var slots: List[Int] = [addr[0], addr[2]]
    var cells: List[Int] = [m, n_tensors]
    if admit:
        for i in range(3, 6):  # small-loop(state: three host state address slots)
            slots.append(addr[i])
            cells.append(n)
    var n_inputs = len(slots)
    slots.append(addr[1])
    cells.append(batch if next_bytes else m * cfg_shape.vocab_size)
    _validate_slot_table(slots, cells, n_inputs)
    var ids = _read_ids(addr[0], m)
    byte_logits_validate(ids, batch, length, cfg_shape)
    if not s[].trainer:
        if not admit:
            raise Error("byte LM: session is not open")
        var state: List[Int] = [addr[3], addr[4], addr[5], addr[2]]
        _ = byte_session_open(s[], state, completed, cfg, cfg_shape)
    var written = 0
    try:
        written = byte_session_logits_run(s[], ids, batch, length, addr[1], completed,
                                          cfg_shape, next_bytes)
    except error:
        byte_owner_mark(s[], completed)
        raise error
    return written


def byte_file_sha256(path: String) raises -> String:
    """The SHA-256 of the file at `path` (64 lowercase hex digits), read in
    1 MiB chunks into the streamed Mojo SHA-256 (core/sha256.mojo)."""
    var hasher = Sha256()
    with open(path, "r") as fh:
        while True:  # cpu-route: the file read on the host, one chunk per pass
            var chunk = fh.read_bytes(_FILE_CHUNK)
            if len(chunk) == 0:
                break
            hasher.update_list(chunk)
    return hasher.hexdigest()
