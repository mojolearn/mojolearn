# SPDX-License-Identifier: Apache-2.0
"""Native zero-state training prefill and immutable tape ownership."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.neural_block_public import neural_training_entries, neural_mamba3_restore, neural_block_forward_binding, _neural_input
from bindings.causal_lm_construct import _model_f32, _model_shape
from bindings.causal_lm_public import _native_function


def _prefill_mamba3(block: PythonObject, x: PythonObject, entries: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = bi.object.__getattribute__(block, "__dict__")
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=fields["d_model"])
    var nh = Int(py=fields["nheads"])
    var weights = fields["_w"]
    neural_mamba3_restore(fields, entries, weights)
    var owned = fields.get("_owned_generation")
    var session = (Bool(py=bi.hasattr(entries, "mamba3_prefill_session_forward")) or owned is not None) and getenv("MOJOLEARN_MAMBA3_LEGACY_SETUP", "0") != "1"
    var skip = False
    if session and Bool(py=fields.get("_discard_reports", PythonObject(False))) and Bool(py=bi.hasattr(entries, "mamba3_prefill_session_reports_optional")):
        skip = Bool(py=_native_function(entries, "mamba3_prefill_session_reports_optional")())
    var reports = bi.list()
    var report_addresses = bi.list()
    var report_shapes = Python.list(py_ints([b, nh, 64, 128]), py_ints([b, nh, 128]), py_ints([b, nh, 64]), py_ints([b, nh, 32]))
    for i in range(4):  # driver-loop: over the four native Mamba3 report buffers
        var report = PythonObject(None)
        if not skip:
            var dims = List[Int]()
            for axis in range(len(report_shapes[i])):  # driver-loop: over report shape metadata
                dims.append(Int(py=report_shapes[i][axis]))
            report = generation_array(dims)
        reports.append(report)
        report_addresses.append(0 if skip else Int(py=report._addr))
    var out = generation_array([b, l, d])
    var addresses = Python.list(x._addr)
    for i in range(len(weights)):  # driver-loop: over fixed Mamba3 weight addresses
        addresses.append(weights[i]._addr)
    addresses.append(out._addr)
    addresses.extend(report_addresses)
    var params = py_ints([b, l, d])
    if session:
        if fields.get("_prefill_session") is None or fields.get("_prefill_binding") is not entries.canonical:
            fields["_prefill_session"] = _native_function(entries, "mamba3_prefill_session_create")()
            fields["_prefill_binding"] = entries.canonical
        if owned is not None:
            var owned_addresses = Python.list(x._addr, out._addr)
            owned_addresses.extend(report_addresses)
            params.append(owned)
            _native_function(entries, "mamba3_prefill_session_forward_owned")(fields["_prefill_session"], owned_addresses, params)
        else:
            _native_function(entries, "mamba3_prefill_session_forward")(fields["_prefill_session"], addresses, params)
    else:
        _native_function(entries, "mamba3_forward_fresh")(addresses, params)
    var names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
    for i in range(4):  # driver-loop: over report result metadata slots
        fields[names[i]] = reports[i]
    return out


def neural_prefill_forward(block: PythonObject, input: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = Python.import_module("builtins")
    var fields = bi.object.__getattribute__(block, "__dict__")
    var kind = String(fields["_native_kind"])
    if kind != "transformer" and kind != "mamba3":
        raise Error("neural prefill tape requires TransformerBlock or Mamba3Block")
    var entries = neural_training_entries(fields, context)
    var x = _neural_input(input, Int(py=fields["d_model"]), False, context)
    var enabled = Bool(py=bi.hasattr(entries, kind + "_forward_tape_enabled"))
    if enabled:
        enabled = Bool(py=_native_function(entries, kind + "_forward_tape_enabled")())
    if kind == "transformer":
        enabled = enabled and fields["_int15"] is None and not Bool(py=fields["_extended"])
    if not enabled:
        var out = PythonObject(None)
        if kind == "transformer":
            out = neural_block_forward_binding(block, x, PythonObject(None), False, PythonObject(None), context)
        else:
            out = _prefill_mamba3(block, x, entries)
        return Python.tuple(out, PythonObject(None))
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=x.shape[2])
    var out = generation_array([b, l, d])
    var addresses = Python.list(x._addr)
    var weights = fields["_w"]
    var shapes = bi.list()
    for i in range(len(weights)):  # driver-loop: over immutable tape input weights
        addresses.append(weights[i]._addr)
        shapes.append(weights[i].shape)
    addresses.append(out._addr)
    var params = py_ints([b, l, d])
    var lock = PythonObject(None)
    if kind == "transformer":
        for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate"), String("window")]:  # driver-loop: over transformer launch dimensions
            params.append(fields[name])
        lock = fields["_runtime_lock"]
    if lock is not None:
        lock.acquire()
    var handle = PythonObject(None)
    try:
        handle = _native_function(entries, kind + "_forward_tape")(addresses, params)
        if lock is not None:
            lock.release()
    except error:
        if lock is not None:
            lock.release()
        raise error
    return Python.tuple(out, Python.tuple(entries, handle, x.shape, bi.tuple(shapes)))


def neural_prefill_backward(block: PythonObject, tape: PythonObject, grad_output: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = Python.import_module("builtins")
    var fields = bi.object.__getattribute__(block, "__dict__")
    var kind = String(fields["_native_kind"])
    var dims = List[Int]()
    for i in range(len(tape[2])):  # driver-loop: over immutable input shape metadata
        dims.append(Int(py=tape[2][i]))
    var dy = _model_f32(grad_output, "grad_output", context)
    _model_shape(dy, dims, "grad_output")
    var gradients = Python.list(generation_array(dims))
    var shapes = tape[3]
    for i in range(len(shapes)):  # driver-loop: over immutable parameter gradient shapes
        var shape = List[Int]()
        for axis in range(len(shapes[i])):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=shapes[i][axis]))
        gradients.append(generation_array(shape))
    var addresses = Python.list(dy._addr)
    for i in range(len(gradients)):  # driver-loop: over gradient output addresses
        addresses.append(gradients[i]._addr)
    var lock = fields["_runtime_lock"] if kind == "transformer" else PythonObject(None)
    if lock is not None:
        lock.acquire()
    try:
        _native_function(tape[0], kind + "_backward_tape")(tape[1], addresses)
        if lock is not None:
            lock.release()
    except error:
        if lock is not None:
            lock.release()
        raise error
    var result = bi.dict()
    result["x"] = gradients[0]
    var names = bi.object.__getattribute__(block, "_W_NAMES")
    for i in range(len(names)):  # driver-loop: over named gradient result metadata
        result[names[i]] = gradients[i + 1]
    return result


def neural_prefill_close(tape: PythonObject) raises:
    if tape is None:
        return
    var bi = Python.import_module("builtins")
    var kind = "transformer" if Bool(py=bi.hasattr(tape[0], "transformer_close_tape")) else "mamba3"
    _native_function(tape[0], kind + "_close_tape")(tape[1])


def neural_prefill_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var op = String(args[0])
        var values = args[1]
        var context = args[2]
        var result = PythonObject(None)
        if op == "forward":
            result = neural_prefill_forward(values[0], values[1], context)
        elif op == "backward":
            result = neural_prefill_backward(values[0], values[1], values[2], context)
        elif op == "close":
            neural_prefill_close(values[0])
        else:
            raise Error("neural prefill: unknown operation")
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error, ExceptionType.ValueError)
