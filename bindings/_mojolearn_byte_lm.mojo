# SPDX-License-Identifier: Apache-2.0
"""Synchronous owned-state boundary for the fixed two-block byte LM.

AUTHORED, UNCOMPILED, UNQUALIFIED. No GPU/context is created during import.
The Python caller owns correctly sized, contiguous, aligned live arrays. Native
span checks cannot prove that an arbitrary integer address names allocated RAM.
No pointer or DeviceContext survives a call. Outputs are published only after
successful computation, validation, synchronization and context-scope teardown.
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
from training.byte_lm import (
    BYTE_PROFILE, BYTE_N_TOTAL, BYTE_J, ByteTrainer, byte_train_step,
    byte_eval_loss, byte_validate_state, byte_validate_optimizer,
    byte_validate_tokens,
)


def byte_lm_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def byte_lm_vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


def byte_lm_profile_binding() raises -> PythonObject:
    return PythonObject(String(BYTE_PROFILE))


def _span_cells(index: Int) -> Int:
    if index == 3 or index == 9:
        return BYTE_J
    if index == 4:
        return 66
    if index == 10:
        return 1
    return BYTE_N_TOTAL


def _validate_addresses(addresses: List[Int], action: Int) raises:
    # All spans are bounded fixed-profile lengths. Validate addition before any
    # pointer construction/dereference; reject null, misalignment and wraparound.
    for i in range(11):
        if i == 8 and action == 0:
            if addresses[i] != 0:
                raise Error("byte LM eval requires null gradient output; no gradient is computed")
            continue
        var size_bytes = _span_cells(i) * 4
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
            if (addresses[i] < addresses[j] + _span_cells(j) * 4
                and addresses[j] < addresses[i] + _span_cells(i) * 4):
                raise Error("byte LM: output overlaps another live span")


def _read_f32(address: Int) -> List[Float32]:
    var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Float32]()
    for i in range(BYTE_N_TOTAL):
        out.append(ptr.unsafe_load(i))
    return out^


def _write_f32(address: Int, values: List[Float32]):
    var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(values)):
        ptr.unsafe_store(i, values[i])


def _require_same_bits(before: List[Float32], after: List[Float32]) raises:
    if len(before) != len(after):
        raise Error("byte LM eval changed state length")
    for i in range(len(before)):
        if bitcast[DType.uint32](before[i]) != bitcast[DType.uint32](after[i]):
            raise Error("byte LM eval changed authoritative state")


def byte_lm_run_binding(addresses: PythonObject, params: PythonObject) raises -> PythonObject:
    """ABI v1. All fixed sizes are ELEMENTS, not bytes.

    addresses[11] = [in_param, in_m, in_v, in_flags_i32, in_ids_i32,
                     out_param, out_m, out_v, out_grad, out_flags_i32, out_loss_f32]
    params[12] = [action, completed, kind, lr, beta1, beta2, eps, weight_decay,
                  momentum, dampening, nesterov, max_norm]
    action=0: eval; out_grad MUST be 0; returns unchanged completed step.
    action=1: training; writes pre-update gradients; returns completed+1.
    Param/m/v/grad spans34944 FP32; flags20 int32(0/1); IDs66 int32[0,256).
    kind=2(AdamW). All config fields explicit; only the fixed legal profile is
    admitted. Positive learning rate, no clipping or SGD options.
    """
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("byte LM binding requires IDENTICAL")
    if String(COMPILED_VENDOR) != "cuda" and String(COMPILED_VENDOR) != "hip" and String(COMPILED_VENDOR) != "metal":
        raise Error("byte LM binding requires CUDA, HIP or Metal")
    if len(addresses) != 11 or len(params) != 12:
        raise Error("byte LM: expected 11 addresses and 12 scalar parameters")
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
    _validate_addresses(addr, action)
    # No GPU work before all borrowed inputs become validated owned host lists.
    var initial_p = _read_f32(addr[0])
    var initial_m = _read_f32(addr[1])
    var initial_v = _read_f32(addr[2])
    var flags_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[3])
    var ids_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[4])
    var flags = List[Bool]()
    var ids = List[Int32]()
    for i in range(BYTE_J):
        var flag = flags_ptr.unsafe_load(i)
        if flag != 0 and flag != 1:
            raise Error("byte LM: momentum flags must be exactly 0 or 1")
        flags.append(flag == 1)
    for i in range(66):
        ids.append(ids_ptr.unsafe_load(i))
    byte_validate_state(initial_p, initial_m, initial_v, flags, completed)
    byte_validate_tokens(ids)
    var out_p = List[Float32]()
    var out_m = List[Float32]()
    var out_v = List[Float32]()
    var out_g = List[Float32]()
    var out_flags = List[Bool]()
    var loss = Float32(0)
    var result_step = completed
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var trainer = ByteTrainer(ctx, initial_p, initial_m, initial_v, flags, completed, cfg)
        if action == 1:
            var capture = byte_train_step(ctx, trainer, ids)
            out_p = capture.after_params.copy()
            out_m = capture.after_m.copy()
            out_v = capture.after_v.copy()
            out_g = capture.gradients.copy()
            out_flags = capture.after_flags.copy()
            loss = capture.loss
            result_step = capture.completed_steps
        else:
            loss = byte_eval_loss(ctx, trainer, ids)
            # Read actual post-evaluation state rather than simply echoing the
            # inputs, which would hide an accidental mutation in evaluation.
            out_p = download_f32(ctx, trainer.buffers.param, BYTE_N_TOTAL)
            out_m = download_f32(ctx, trainer.buffers.m_state, BYTE_N_TOTAL)
            out_v = download_f32(ctx, trainer.buffers.v_state, BYTE_N_TOTAL)
            out_flags = trainer.buffers.buf_initialized.copy()
            result_step = trainer.completed_steps
            _require_same_bits(initial_p, out_p)
            _require_same_bits(initial_m, out_m)
            _require_same_bits(initial_v, out_v)
            for i in range(BYTE_J):
                if flags[i] != out_flags[i]:
                    raise Error("byte LM eval changed momentum flags")
        byte_validate_state(out_p, out_m, out_v, out_flags, result_step)
        if result_step != completed + action:
            raise Error("byte LM: successful result has wrong completed step")
        if (bitcast[DType.uint32](loss) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("byte LM: nonfinite returned loss")
        if action == 1:
            if len(out_g) != BYTE_N_TOTAL:
                raise Error("byte LM: wrong gradient length")
            for i in range(BYTE_N_TOTAL):
                if (bitcast[DType.uint32](out_g[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
                    raise Error("byte LM: nonfinite returned gradient")
        ctx.synchronize()
        _ = trainer^
        # Keep the context alive until the trainer's device allocations die.
        _ = ctx^
    # Publication starts after the entire GPU/context scope succeeds. The
    # Python wrapper commits these fresh arrays atomically to its state object.
    _write_f32(addr[5], out_p)
    _write_f32(addr[6], out_m)
    _write_f32(addr[7], out_v)
    if action == 1:
        _write_f32(addr[8], out_g)
    var flags_out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr[9])
    for i in range(BYTE_J):
        flags_out.unsafe_store(i, Int32(out_flags[i]))
    var loss_out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr[10])
    loss_out.unsafe_store(0, loss)
    return PythonObject(result_step)


@export
def PyInit__mojolearn_byte_lm() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_byte_lm")
        module.def_function[byte_lm_numeric_mode_binding]("byte_lm_numeric_mode")
        module.def_function[byte_lm_vendor_binding]("byte_lm_vendor")
        module.def_function[byte_lm_profile_binding]("byte_lm_profile")
        module.def_function[byte_lm_run_binding]("byte_lm_run")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_byte_lm: ", error))
