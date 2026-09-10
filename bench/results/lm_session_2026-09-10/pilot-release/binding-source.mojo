# SPDX-License-Identifier: Apache-2.0
"""Synchronous owned-state boundary for a runtime-shaped decoder language model.

Runtime shapes are compile/host checked; device qualification is separate.
No GPU/context is created during import.
The Python caller owns correctly sized, contiguous, aligned live arrays. Native
span checks cannot prove that an arbitrary integer address names allocated RAM.
No borrowed pointer survives a call. Optional owned sessions retain device state.
Outputs are published only after successful computation, validation and
synchronization; the stateless ABI also tears down its context before return.
"""
from std.memory import bitcast
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from training.checks.optimizer_oracle import OptimizerConfig
from training.checks.train_loop import download_f32
from training.byte_lm_config import ByteConfig
from training.byte_lm import (
    BYTE_PROFILE, ByteTrainer, byte_train_step,
    byte_eval_loss, byte_validate_state, byte_validate_optimizer,
    byte_validate_tokens,
)


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
        # Every operation synchronizes. Buffers must die before their context.
        _ = self.trainer^
        _ = self.ctx^

    def close(mut self) raises:
        if self.busy:
            raise Error("byte LM: session is busy")
        self.usable = False
        if self.ctx:
            self.ctx.value().synchronize()
        self.trainer = None
        self.ctx = None


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


def _read_f32(address: Int, n: Int) -> List[Float32]:
    var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Float32]()
    for i in range(n):
        out.append(ptr.unsafe_load(i))
    return out^


def _write_f32(address: Int, values: List[Float32]):
    var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(values)):
        ptr.unsafe_store(i, values[i])


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
    byte_validate_state(initial_p, initial_m, initial_v, flags, completed, shape)
    byte_validate_tokens(ids, shape)
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
                session.ctx = DeviceContext()
                session.trainer = ByteTrainer(session.ctx.value(), initial_p, initial_m,
                    initial_v, flags, completed, cfg, shape)
            ref ctx = session.ctx.value()
            if reused:
                # Reuse must never silently ignore a supplied checkpoint/config.
                if session.trainer.value().config.profile() != shape.profile():
                    raise Error("byte LM: resident model shape mismatch")
                if session.trainer.value().completed_steps != completed:
                    raise Error("byte LM: resident completed-step mismatch")
                var prior_cfg = session.trainer.value().optimizer.copy()
                if (prior_cfg.kind != cfg.kind or prior_cfg.nesterov != cfg.nesterov
                    or prior_cfg.lr != cfg.lr or prior_cfg.beta1 != cfg.beta1
                    or prior_cfg.beta2 != cfg.beta2 or prior_cfg.eps != cfg.eps
                    or prior_cfg.weight_decay != cfg.weight_decay
                    or prior_cfg.momentum != cfg.momentum or prior_cfg.dampening != cfg.dampening
                    or prior_cfg.max_norm != cfg.max_norm):
                    raise Error("byte LM: resident optimizer mismatch")
                _require_same_bits(initial_p, download_f32(ctx, session.trainer.value().buffers.param, n))
                _require_same_bits(initial_m, download_f32(ctx, session.trainer.value().buffers.m_state, n))
                _require_same_bits(initial_v, download_f32(ctx, session.trainer.value().buffers.v_state, n))
                for i in range(shape.n_tensors()):
                    if flags[i] != session.trainer.value().buffers.buf_initialized[i]:
                        raise Error("byte LM: resident flags mismatch")
            if action == 1:
                var capture = byte_train_step(ctx, session.trainer.value(), ids)
                out_p = capture.after_params.copy()
                out_m = capture.after_m.copy()
                out_v = capture.after_v.copy()
                out_g = capture.gradients.copy()
                out_flags = capture.after_flags.copy()
                loss = capture.loss
                result_step = capture.completed_steps
            else:
                loss = byte_eval_loss(ctx, session.trainer.value(), ids)
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
            ctx.synchronize()
            if not retain:
                # Already synchronized: preserve the stateless teardown without
                # adding close()'s second drain to the comparison baseline.
                session.trainer = None
                session.ctx = None
    except error:
        session.busy = False
        raise error
    session.busy = False
    session.usable = retain
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
    return PythonObject(result_step)


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
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_byte_lm: ", error))
