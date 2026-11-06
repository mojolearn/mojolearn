# SPDX-License-Identifier: Apache-2.0
# Whole public backend metadata operations. Providers/loader outcomes are
# connected once at package import; product calls never enter Python loaders.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error


def _text(value: PythonObject, name: String, optional: Bool = False) raises -> PythonObject:
    if optional and value is None:
        return value
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: backend " + name + " requires an exact native string")
    return value


def _mode(fields: PythonObject) raises -> PythonObject:
    var mode = _text(fields["_DEFAULT_MODE"], "default mode")
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("ValueError: backend default numeric mode is not supported")
    return mode


def _refuse(error: PythonObject) raises:
    if error is None:
        return
    # The retained import-time exception is already constructed. Restoring it
    # invokes neither an exception constructor nor a Python loader/callback.
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(py_type(error)._obj_ptr, error._obj_ptr)
    var failure = stash_python_error()
    raise failure^


def _read(entry: PythonObject, context: PythonObject, vendor: Bool = False) raises -> PythonObject:
    ref cpy = Python().cpython()
    if py_type(entry) is not context["native_function_type"]:
        raise Error("TypeError: backend read-back requires a compiled native entry")
    var args = Python.tuple()
    var raw = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not raw:
        if vendor and (cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(context["readback_skip_errors"][0]._obj_ptr) != 0
                or cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(context["readback_skip_errors"][1]._obj_ptr) != 0):
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
            return PythonObject(None)
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=raw)


def backend_public_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = context["backend_fields"]
    if operation == "gpu_arch" or operation == "gpu_arch_how":
        if fields["_CPU_ONLY"] is not None:
            return PythonObject(None) if operation == "gpu_arch" else PythonObject("no GPU binary set loaded (CPU-only install, DEVIATION 2615)")
        return _text(fields["_ARCH_SELECTED"] if operation == "gpu_arch" else fields["_ARCH_HOW"], "architecture metadata", True)
    var mode = _mode(fields)
    if operation == "set_default_mode":
        if len(args) != 1:
            raise Error("TypeError: set_numeric_mode requires one mode argument")
        var requested = mode if args[0] is None else _text(args[0], "numeric mode")
        var text_type = py_type(PythonObject(""))
        requested = py_call(text_type.lower, py_call(text_type.strip, requested))
        if String(requested) != "identical" and String(requested) != "fast":
            raise Error("ValueError: mojolearn: numeric_mode=" + String(py_call(text_type.__repr__, requested)) + "; it must be 'fast', or 'identical' (the default)")
        _refuse(context["plans"][requested]["error"])
        py_setitem(fields, "_DEFAULT_MODE", requested)
        return mode
    if operation == "numeric_mode":
        var selected = py_get(fields, "_SELECTED")
        if selected is None:
            selected = PythonObject("fast")
        _ = _text(selected, "selected mode")
        var entry = context["selected_mode"] if mode == selected else context["plans"][mode]["mode"]
        if entry is not None:
            var code = _read(entry, context)
            if py_type(code) is not py_type(PythonObject(0)):
                raise Error("TypeError: compiled numeric mode requires a native integer code")
            var compiled = py_get(context["mode_codes"], code, PythonObject("unknown"))
            if compiled != mode:
                raise Error("RuntimeError: mojolearn: the loaded gbdt binary was compiled " + String(compiled) + " but the selector loaded the " + String(mode) + " set -- a binary is in the wrong directory; rebuild both sets")
        return mode
    if operation == "vendor":
        if fields["_CPU_ONLY"] is not None:
            return PythonObject("cpu")
        var said = PythonObject(None)
        var entries = context["plans"][mode]["vendors"]
        for i in range(len(entries)):  # driver-loop: native provider metadata readbacks in selector order
            said = _read(entries[i], context, True)
            if said is not None:
                said = _text(said, "compiled vendor")
                break
        var expected = _text(fields["_VENDOR_SELECTED"], "selected vendor", True)
        var layout = fields["_LAYOUT"]
        if py_type(layout) is not py_type(Python.tuple()) or len(layout) != 2:
            raise Error("TypeError: backend layout requires native import-connected metadata")
        var kind = _text(layout[0], "layout kind")
        if String(kind) == "vendor":
            if said is not None and said != expected:
                raise Error("RuntimeError: mojolearn: the loaded " + String(mode) + " binaries were compiled for " + String(said) + " but the selector opened the " + String(expected) + " set -- a binary is in the wrong vendor directory; rebuild both sets")
            return expected
        if said is not None:
            py_setitem(fields, "_VENDOR_SELECTED", said)
            return said
        return expected
    raise Error("ValueError: unknown backend metadata operation")


def backend_public_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3:
            raise Error("TypeError: native backend operation requires operation, arguments and context")
        var operation = _text(args[0], "operation")
        if py_type(args[1]) is not py_type(Python.tuple()):
            raise Error("TypeError: native backend arguments require a tuple")
        return backend_public_operation(String(operation), args[1], args[2]).steal_data()
    except error:
        return optimizer_python_error(error)
