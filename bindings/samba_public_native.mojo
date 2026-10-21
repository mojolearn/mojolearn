# SPDX-License-Identifier: Apache-2.0
"""One compiled Samba boundary for public and distributed model operations."""
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.samba_state_native import samba_snapshot, samba_validate_state, samba_from_state, samba_publish, samba_apply_gradients_from_state
from bindings.samba_training_native import samba_gradients, samba_gradients_from_state
from bindings.generator_state_native import generator_init, generator_snapshot, generator_restore, generator_next
from bindings.samba_construct_native import samba_construct, samba_load_weights, samba_parameters
from bindings.samba_step_native import samba_train_step
from bindings.samba_inference_native import samba_forward, samba_allocate, samba_loss, samba_state_construct
from bindings.samba_checkpoint_native import samba_checkpoint_save, samba_checkpoint_load, samba_checkpoint_read
from bindings.samba_cpu_inference import samba_inference_construct, samba_inference_parameters, samba_inference_checkpoint, samba_inference_repr


def samba_public_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var operation = String(args[0])
        var values = args[1]
        var context = args[2]
        var result = PythonObject(None)
        if operation == "model_train_step":
            result = samba_train_step(values[0], values[1], values[2], context)
        elif operation == "forward":
            result = samba_forward(values[0], values[1], values[2], values[3], False, True, context)
        elif operation == "step":
            result = samba_forward(values[0], values[1], values[2], PythonObject(None), True, True, context)
        elif operation == "allocate_state":
            result = samba_allocate(values[0], values[1], values[2], context)
        elif operation == "state_construct":
            result = samba_state_construct(values[0], values[1], values[2], values[3])
        elif operation == "loss":
            result = samba_loss(values[0], values[1], values[2], context)
        elif operation == "checkpoint_save":
            result = samba_checkpoint_save(values[0], values[1], context)
        elif operation == "checkpoint_load":
            result = samba_checkpoint_load(values[0], values[1], values[2], context)
        elif operation == "checkpoint_read":
            result = samba_checkpoint_read(values[0], context)
        elif operation == "inference_construct":
            result = samba_inference_construct(values[0], values[1], values[2], context)
        elif operation == "inference_parameters":
            result = samba_inference_parameters(values[0])
        elif operation == "inference_checkpoint":
            result = samba_inference_checkpoint(values[0], values[1], context)
        elif operation == "inference_forward":
            result = samba_forward(values[0], values[1], values[2], values[3], False, False, context)
        elif operation == "inference_step":
            result = samba_forward(values[0], values[1], values[2], PythonObject(None), True, False, context)
        elif operation == "inference_repr":
            result = samba_inference_repr(values[0])
        elif operation == "inference_refuse":
            raise Error("NotImplementedError: SambaInference." + String(values[0]) + ": inference is forward only; use SambaStack for training")
        elif operation == "construct":
            result = samba_construct(values[0], values[1], context)
        elif operation == "load_weights":
            result = samba_load_weights(values[0], values[1], context)
        elif operation == "parameters":
            result = samba_parameters(values[0])
        elif operation == "generator_init":
            result = generator_init(values[0], values[1], values[2])
        elif operation == "generator_snapshot":
            result = generator_snapshot(values[0])
        elif operation == "generator_restore":
            result = generator_restore(values[0], values[1])
        elif operation == "generator_next":
            result = generator_next(values[0])
        elif operation == "snapshot":
            result = samba_snapshot(values[0], Bool(py=values[1]), context)
        elif operation == "validate":
            result = samba_validate_state(values[0], context)
        elif operation == "from_state":
            result = samba_from_state(values[0], context)
        elif operation == "publish":
            result = samba_publish(values[0], values[1], context)
        elif operation == "apply":
            result = samba_apply_gradients_from_state(values[0], values[1], context)
        elif operation == "gradients":
            result = samba_gradients_from_state(values[0], values[1], values[2], values[3], Int(py=values[4]), context)
        elif operation == "model_gradients":
            result = samba_gradients(values[0], values[1], values[2], values[3], values[4], Int(py=values[5]), context)
        else:
            raise Error("Samba: unknown native state operation")
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error, ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("OSError:"):
            return raise_python_exception(error, ExceptionType("PyExc_OSError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        return raise_python_exception(error, ExceptionType.ValueError)
