# SPDX-License-Identifier: Apache-2.0
"""One compiled Samba boundary for public and distributed model operations."""
from std.python import Python, PythonObject
from bindings.samba_keys_unpack import samba_keys
from bindings.samba_keys_unpack import (
    SAMBA_OPN_ALLOCATE_STATE,
    SAMBA_OPN_APPLY,
    SAMBA_OPN_CHECKPOINT_LOAD,
    SAMBA_OPN_CHECKPOINT_READ,
    SAMBA_OPN_CHECKPOINT_SAVE,
    SAMBA_OPN_CONSTRUCT,
    SAMBA_OPN_DECODE_SESSION_CLOSE,
    SAMBA_OPN_DECODE_SESSION_DECODE,
    SAMBA_OPN_DECODE_SESSION_DEVICE,
    SAMBA_OPN_DECODE_SESSION_DROP,
    SAMBA_OPN_DECODE_SESSION_ENTER,
    SAMBA_OPN_DECODE_SESSION_EXIT,
    SAMBA_OPN_DECODE_SESSION_INIT,
    SAMBA_OPN_DECODE_SESSION_IS_OPEN,
    SAMBA_OPN_DECODE_SESSION_LOAD,
    SAMBA_OPN_DECODE_SESSION_OPEN,
    SAMBA_OPN_DECODE_SESSION_REPR,
    SAMBA_OPN_DECODE_SESSION_STATE,
    SAMBA_OPN_DECODE_SESSION_STEP,
    SAMBA_OPN_DECODE_SESSION_SYNC,
    SAMBA_OPN_FORWARD,
    SAMBA_OPN_FROM_STATE,
    SAMBA_OPN_GENERATOR_INIT,
    SAMBA_OPN_GENERATOR_NEXT,
    SAMBA_OPN_GENERATOR_RESTORE,
    SAMBA_OPN_GENERATOR_SNAPSHOT,
    SAMBA_OPN_GRADIENTS,
    SAMBA_OPN_INFERENCE_CHECKPOINT,
    SAMBA_OPN_INFERENCE_CONSTRUCT,
    SAMBA_OPN_INFERENCE_FORWARD,
    SAMBA_OPN_INFERENCE_PARAMETERS,
    SAMBA_OPN_INFERENCE_REFUSE,
    SAMBA_OPN_INFERENCE_REPR,
    SAMBA_OPN_INFERENCE_STEP,
    SAMBA_OPN_LOAD_WEIGHTS,
    SAMBA_OPN_LOSS,
    SAMBA_OPN_MODEL_GRADIENTS,
    SAMBA_OPN_MODEL_TRAIN_STEP,
    SAMBA_OPN_PARAMETERS,
    SAMBA_OPN_PUBLISH,
    SAMBA_OPN_SNAPSHOT,
    SAMBA_OPN_STATE_CONSTRUCT,
    SAMBA_OPN_STEP,
    SAMBA_OPN_VALIDATE,
)
from bindings.py_keys import py_op_code
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get
from bindings.samba_state_native_door import samba_fields, samba_snapshot, samba_validate_state, samba_from_state, samba_publish, samba_apply_gradients_from_state
from bindings.samba_training_native_door import samba_gradients, samba_gradients_from_state
from bindings.generator_state_native_pack import generator_init, generator_snapshot, generator_restore, generator_next
from bindings.samba_construct_native_pack import samba_construct, samba_load_weights, samba_parameters
from bindings.samba_step_native_door import samba_train_step
from bindings.samba_inference_native_door import samba_forward, samba_allocate, samba_loss, samba_state_construct
from bindings.samba_checkpoint_native_pack import samba_checkpoint_save, samba_checkpoint_load, samba_checkpoint_read
from bindings.samba_cpu_inference_pack import samba_inference_construct, samba_inference_parameters, samba_inference_checkpoint, samba_inference_repr
from bindings.samba_admission_door import samba_admit
from core.weight_epoch import weight_epoch_bump, registry_epoch_bump
from bindings.samba_decode_native_unpack import (
    samba_decode_open,
    samba_decode_init,
    samba_decode_step,
    samba_decode_tokens,
    samba_decode_load,
    samba_decode_state,
    samba_decode_device,
)
from bindings.samba_decode_native_pack import (
    samba_decode_sync,
    samba_decode_close,
    samba_decode_drop,
    samba_decode_is_open,
    samba_decode_repr,
)


def _samba_public_items(value: PythonObject) raises -> PythonObject:
    # The public loss_and_grads divisor: None (this call's own target count)
    # or a positive integer. A str, a bool or zero is refused by name; the
    # train step's internal microbatch calls do not pass through here.
    if value is None:
        return value
    var bi = native_loaded_module("builtins")
    if py_type(value) is bi.bool:
        raise Error("TypeError: Samba: num_items must be a positive integer")
    var items = 0
    try:
        items = Int(py=py_call(native_loaded_module("operator").index, value))
    except error:
        raise Error("TypeError: Samba: num_items must be a positive integer")
    if items < 1:
        raise Error("Samba: num_items must be a positive integer")
    return PythonObject(items)


def _samba_writes_registry(operation: Int) -> Bool:
    """The operations that write the stack's flat registry in place (integ-w4,
    lane x45c's weight key in SambaStackProgram): each one is native code
    writing by address and moves the weight epoch, so a resident program
    reloads at its next call. Lane x56d (the contract of every neural
    family, decided 2026-11-09): the registry and the public parameter views
    are read-only to callers (bindings/samba_state_native.mojo `samba_view`),
    so no write outside these operations can leave a resident copy stale.
    Lane fix-samba-weight-epoch-per-model (2026-11-13): the door stamps the
    WRITTEN model registry's own epoch (core/weight_epoch.mojo
    `registry_epoch_bump`; a Samba resident copy is keyed by it) and moves the
    process epoch. Lane bw3-samba (2026-11-20): moved here from
    training/samba_public_run.mojo, an integer test of the operation code."""
    return (operation == SAMBA_OPN_MODEL_TRAIN_STEP or operation == SAMBA_OPN_APPLY or operation == SAMBA_OPN_PUBLISH
            or operation == SAMBA_OPN_CHECKPOINT_LOAD or operation == SAMBA_OPN_LOAD_WEIGHTS)


def _samba_written_registry(operation: Int, values: PythonObject) raises -> Int:
    """UNPACK (lane fix-samba-weight-epoch-per-model, 2026-11-13): the host
    address of the model registry (`flat`) a writing operation writes in
    place, 0 when it writes none. `model_train_step`, `publish` and
    `load_weights` write their model's (values[0]) registry. `apply` writes
    the validated state's private parameter copy and `checkpoint_load` builds
    a new model (samba_validate_state copies): no existing registry, so no
    other copy is stamped. Read only on a FAILED write (cold): a completed
    write stamps its registry itself, from the address it holds
    (samba_step_native_door `_samba_train_step_admitted`,
    samba_state_native_door `_samba_publish_checked`,
    samba_construct_native_pack `samba_load_weights`)."""
    if operation != SAMBA_OPN_MODEL_TRAIN_STEP and operation != SAMBA_OPN_PUBLISH and operation != SAMBA_OPN_LOAD_WEIGHTS:
        return 0
    var nk = samba_keys()
    var flat = py_get(samba_fields(values[0]), nk[].k_flat, PythonObject(None))
    if flat is None:
        return 0
    return Int(py=flat._addr)


def _samba_stamp_failed(operation: Int, values: PythonObject) raises:
    """A failed write may have changed part of the registry: move its own
    epoch (its resident copies reload) and the process epoch."""
    registry_epoch_bump(_samba_written_registry(operation, values))
    weight_epoch_bump()


def samba_public_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var writes = False
    var operation = -1
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var nk = samba_keys()
        operation = py_op_code(nk[].ops, args[0])
        var values = args[1]
        var context = args[2]
        if operation == SAMBA_OPN_DECODE_SESSION_DROP:
            # Finalizer: best effort, never raised out of __del__.
            try:
                return samba_decode_drop(values[0]).steal_data()
            except:
                return PythonObject(None).steal_data()
        samba_admit(operation, values, context)
        writes = _samba_writes_registry(operation)
        # Hot-path audit 2026-11-13 (BLOCK pack:69/72): the door moves only
        # the process epoch, one atomic add before and after a write, as
        # before the per-registry lane (the per-block programs are keyed by
        # views into the registry and read it). The written registry's own
        # epoch is stamped by the write itself, after it, from the registry
        # object the write already holds: no fields / `flat` / `_addr` read
        # here per call. (A stamp before the write is not needed: the bytes
        # change only in the write, and a copy keyed before it or during it
        # is not current after the stamp that follows it.)
        if writes:
            weight_epoch_bump()
        var result = PythonObject(None)
        if operation == SAMBA_OPN_MODEL_TRAIN_STEP:
            result = samba_train_step(values[0], values[1], values[2], context)
        elif operation == SAMBA_OPN_FORWARD:
            result = samba_forward(values[0], values[1], values[2], values[3], False, context)
        elif operation == SAMBA_OPN_STEP:
            result = samba_forward(values[0], values[1], values[2], PythonObject(None), True, context)
        elif operation == SAMBA_OPN_ALLOCATE_STATE:
            result = samba_allocate(values[0], values[1], values[2], context)
        elif operation == SAMBA_OPN_STATE_CONSTRUCT:
            result = samba_state_construct(values[0], values[1], values[2], values[3])
        elif operation == SAMBA_OPN_LOSS:
            result = samba_loss(values[0], values[1], values[2], context)
        elif operation == SAMBA_OPN_CHECKPOINT_SAVE:
            result = samba_checkpoint_save(values[0], values[1], context)
        elif operation == SAMBA_OPN_CHECKPOINT_LOAD:
            result = samba_checkpoint_load(values[0], values[1], values[2], context)
        elif operation == SAMBA_OPN_CHECKPOINT_READ:
            result = samba_checkpoint_read(values[0], context)
        elif operation == SAMBA_OPN_INFERENCE_CONSTRUCT:
            result = samba_inference_construct(values[0], values[1], values[2], context)
        elif operation == SAMBA_OPN_INFERENCE_PARAMETERS:
            result = samba_inference_parameters(values[0])
        elif operation == SAMBA_OPN_INFERENCE_CHECKPOINT:
            result = samba_inference_checkpoint(values[0], values[1], context)
        elif operation == SAMBA_OPN_INFERENCE_FORWARD:
            result = samba_forward(values[0], values[1], values[2], values[3], False, context)
        elif operation == SAMBA_OPN_INFERENCE_STEP:
            result = samba_forward(values[0], values[1], values[2], PythonObject(None), True, context)
        elif operation == SAMBA_OPN_DECODE_SESSION_OPEN:
            result = samba_decode_open(values[0], values[1], context)
        elif operation == SAMBA_OPN_DECODE_SESSION_INIT:
            result = samba_decode_init(values[0], values[1], values[2], context)
        elif operation == SAMBA_OPN_DECODE_SESSION_STEP:
            result = samba_decode_step(values[0], values[1], context)
        elif operation == SAMBA_OPN_DECODE_SESSION_DECODE:
            result = samba_decode_tokens(values[0], values[1], context)
        elif operation == SAMBA_OPN_DECODE_SESSION_SYNC:
            result = samba_decode_sync(values[0], context)
        elif operation == SAMBA_OPN_DECODE_SESSION_LOAD:
            result = samba_decode_load(values[0], context)
        elif operation == SAMBA_OPN_DECODE_SESSION_CLOSE or operation == SAMBA_OPN_DECODE_SESSION_EXIT:
            result = samba_decode_close(values[0], context)
            if operation == SAMBA_OPN_DECODE_SESSION_EXIT:
                result = PythonObject(False)
        elif operation == SAMBA_OPN_DECODE_SESSION_ENTER:
            result = values[0]
        elif operation == SAMBA_OPN_DECODE_SESSION_STATE:
            result = samba_decode_state(values[0])
        elif operation == SAMBA_OPN_DECODE_SESSION_IS_OPEN:
            result = samba_decode_is_open(values[0])
        elif operation == SAMBA_OPN_DECODE_SESSION_DEVICE:
            result = samba_decode_device(values[0])
        elif operation == SAMBA_OPN_DECODE_SESSION_REPR:
            result = samba_decode_repr(values[0])
        elif operation == SAMBA_OPN_INFERENCE_REPR:
            result = samba_inference_repr(values[0])
        elif operation == SAMBA_OPN_INFERENCE_REFUSE:
            raise Error("NotImplementedError: SambaInference." + String(values[0]) + ": inference is forward only; use SambaStack for training")
        elif operation == SAMBA_OPN_CONSTRUCT:
            result = samba_construct(values[0], values[1], context)
        elif operation == SAMBA_OPN_LOAD_WEIGHTS:
            result = samba_load_weights(values[0], values[1], context)
        elif operation == SAMBA_OPN_PARAMETERS:
            result = samba_parameters(values[0])
        elif operation == SAMBA_OPN_GENERATOR_INIT:
            result = generator_init(values[0], values[1], values[2])
        elif operation == SAMBA_OPN_GENERATOR_SNAPSHOT:
            result = generator_snapshot(values[0])
        elif operation == SAMBA_OPN_GENERATOR_RESTORE:
            result = generator_restore(values[0], values[1])
        elif operation == SAMBA_OPN_GENERATOR_NEXT:
            result = generator_next(values[0])
        elif operation == SAMBA_OPN_SNAPSHOT:
            result = samba_snapshot(values[0], Bool(py=values[1]), context)
        elif operation == SAMBA_OPN_VALIDATE:
            result = samba_validate_state(values[0], context)
        elif operation == SAMBA_OPN_FROM_STATE:
            result = samba_from_state(values[0], context)
        elif operation == SAMBA_OPN_PUBLISH:
            result = samba_publish(values[0], values[1], context)
        elif operation == SAMBA_OPN_APPLY:
            result = samba_apply_gradients_from_state(values[0], values[1], context)
        elif operation == SAMBA_OPN_GRADIENTS:
            result = samba_gradients_from_state(values[0], values[1], values[2], values[3], Int(py=values[4]), context)
        elif operation == SAMBA_OPN_MODEL_GRADIENTS:
            result = samba_gradients(values[0], values[1], values[2], _samba_public_items(values[3]), values[4], Int(py=values[5]), context)
        else:
            raise Error("Samba: unknown native state operation")
        if writes:
            weight_epoch_bump()
        return result.steal_data()
    except error:
        if writes:
            # a failed write may have changed part of the registry
            try:
                var failed_args = PythonObject(from_borrowed=args_ptr)
                _samba_stamp_failed(operation, failed_args[1])
            except:
                try:
                    weight_epoch_bump()
                except:
                    pass
        var message = String(error)
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("OSError:"):
            return raise_python_exception(error_without_class(error, "OSError"), ExceptionType("PyExc_OSError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
