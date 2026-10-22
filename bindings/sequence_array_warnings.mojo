# SPDX-License-Identifier: Apache-2.0
# Native warning delivery for sequence array-protocol casts.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_warnings import _warning_fields, _warning_class_fields, _warning_match, _warning_write
from bindings.optimizer_public_words import stash_python_error
from core.sequence_array_cast import SEQ_CAST_OVERFLOW, SEQ_CAST_UNDERFLOW, SEQ_CAST_INVALID


def sequence_array_errstate(context: PythonObject) raises -> PythonObject:
    var module = py_get(context["modules"], "numpy._core._multiarray_umath")
    if module is None:
        module = py_get(context["modules"], "numpy.core._multiarray_umath")
    if module is None or py_type(module) is not context["module_type"]:
        raise Error("TypeError: array output requires NumPy's native error-state metadata")
    var entry = py_get(py_fields(module), "_get_extobj_dict")
    var warning = context["sequence_array_warnings"]
    if entry is None or py_type(entry) is not warning["warning_native_function"] or String(entry.__name__) != "_get_extobj_dict" or entry.__self__ is not module:
        raise Error("TypeError: NumPy error-state metadata must use its native entry")
    return _warning_fields(py_call(entry))


def sequence_array_report(status: Int, settings: PythonObject, context: PythonObject) raises:
    var warning = context["sequence_array_warnings"]
    var b = Python.import_module("builtins")
    # NumPy reports flags in this order and only once per completed cast,
    # regardless of how many individual elements or fields set the flag.
    var flags: List[Int] = [SEQ_CAST_OVERFLOW, SEQ_CAST_UNDERFLOW, SEQ_CAST_INVALID]
    var names: List[String] = ["over", "under", "invalid"]
    var descriptions: List[String] = ["overflow", "underflow", "invalid value"]
    for i in range(3):  # driver-loop: over the three fixed cast-status categories
        if (status & flags[i]) == 0:
            continue
        var action = settings[PythonObject(names[i])]
        if py_type(action) is not b.str:
            raise Error("TypeError: NumPy error-state actions require native strings")
        var mode = String(action)
        var message = descriptions[i] + " encountered in cast"
        if mode == "ignore":
            continue
        if mode == "warn":
            sequence_array_warn(message, warning)
        elif mode == "raise":
            sequence_array_exception("FloatingPointError", message)
        elif mode == "print":
            _warning_write("Warning: " + message + "\n", warning)
        elif mode == "log":
            # Native text streams remain supported. The shared writer checks
            # exact native IO/codec identities before reaching their methods.
            var local = py_call(b.dict, warning)
            var fields = py_call(b.dict, warning["warning_sys_fields"])
            py_setitem(fields, "stderr", settings["call"])
            py_setitem(local, "warning_sys_fields", fields)
            _warning_write("Warning: " + message + "\n", local)
        else:
            raise Error("TypeError: Python NumPy error callbacks cannot execute in array output; use native errstate ignore, warn or raise")


def sequence_array_exception(kind: String, var message: String) raises:
    ref cpy = Python().cpython()
    var category = py_fields(Python.import_module("builtins"))[PythonObject(kind)]
    cpy.PyErr_SetString(category._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
    var failure = stash_python_error()
    raise failure^


def sequence_array_warn(message: String, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var fields = _warning_fields(context["warning_fields"])
    var category = b.RuntimeWarning
    # The existing warning originated at the __array__ public method. Its
    # native metadata retains that location for filtering and repetition.
    var filename = context["warning_filename"]
    var lineno = Int(py=context["warning_lineno"])
    var module = context["warning_module"]
    var filters = py_get(fields, "filters")
    var log = PythonObject(None)
    if Bool(py=context["warning_context_enabled"]):
        var active = py_get(context["warning_context_var"], context["warning_global_context"])
        if active is not context["warning_global_context"]:
            if py_type(active) is not context["warning_context_class"]:
                raise Error("TypeError: custom warning contexts are not native")
            var stored = _warning_fields(py_call(context["warning_context_fields"].__get__, active))
            filters = py_get(stored, "_filters")
            log = py_get(stored, "log")
    if py_type(filters) is not b.list or (log is not None and py_type(log) is not b.list):
        raise Error("TypeError: warning filters and recording require native lists")
    var old = context["warning_filter_rows"]
    var changed = filters is not context["warning_filter_owner"] or len(old) != len(filters)
    if not changed:
        for k in range(len(filters)):  # driver-loop: over filter row identities for normal filter mutation
            changed = changed or old[k] is not filters[k]
    if changed:
        py_setitem(context, "warning_seen", b.dict())
        py_setitem(context, "warning_filter_owner", filters)
        py_setitem(context, "warning_filter_rows", py_call(b.tuple, filters))
    var action = py_get(fields, "defaultaction", "default")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, category)
    for k in range(len(filters)):  # driver-loop: over configured warning filters in their declared order
        var row = filters[k]
        if py_type(row) is not b.tuple or len(row) != 5 or py_type(row[0]) is not b.str or py_type(row[4]) is not b.int:
            raise Error("TypeError: warning filter rows must use native scalar metadata")
        _ = _warning_class_fields(row[2])
        var category_match = False
        for j in range(len(mro)):  # driver-loop: over native warning category ancestry
            category_match = category_match or row[2] is mro[j]
        if category_match and (Int(py=row[4]) == 0 or Int(py=row[4]) == lineno) and _warning_match(row[1], PythonObject(message), context) and _warning_match(row[3], module, context):
            action = row[0]
            break
    if py_type(action) is not b.str:
        raise Error("TypeError: warning actions must be native strings")
    var mode = String(action)
    if mode == "ignore":
        return
    if mode == "error":
        sequence_array_exception("RuntimeWarning", message)
    if mode != "always" and mode != "all" and mode != "default" and mode != "module" and mode != "once":
        raise Error("RuntimeError: unrecognized warning filter action")
    var key = Python.tuple(message, category, PythonObject(None) if mode == "once" else module, PythonObject(lineno if mode == "default" else 0))
    if mode != "always" and mode != "all":
        if Bool(py=py_get(context["warning_seen"], key, False)):
            return
    if not Bool(py=context["warning_standard_hooks"]):
        raise Error("TypeError: custom Python warning hooks are not native")
    var hooks = context["warning_functions"]
    var names = Python.tuple("showwarning", "formatwarning", "_showwarnmsg", "_formatwarnmsg", "_formatwarnmsg_impl")
    for k in range(len(names)):  # driver-loop: over standard warning display hooks before any output
        if py_get(fields, names[k]) is not hooks[names[k]]:
            raise Error("TypeError: custom Python warning hooks are not native")
    var display = py_get(fields, "_showwarnmsg_impl")
    if py_type(display) is context["warning_native_function"] and String(display.__name__) == "append" and py_type(display.__self__) is b.list:
        log = display.__self__
    elif py_type(display) is context["warning_python_function"] and String(display.__name__) == "_showwarnmsg_impl" and display.__code__.co_filename == context["warning_stdlib_file"]:
        pass
    else:
        raise Error("TypeError: custom warning recording callbacks are not native")
    if log is not None:
        var record_class = context["warning_record"]
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__, record_class)
        if len(ancestry) != 2 or ancestry[1] is not b.object or py_get(_warning_class_fields(record_class), "__del__") is not None:
            raise Error("TypeError: custom warning record finalizers are not native")
        var record = py_call(b.object.__new__, record_class)
        var stored = py_call(context["warning_record_fields"].__get__, record)
        py_setitem(stored, "message", py_call(category, message))
        py_setitem(stored, "category", category)
        py_setitem(stored, "filename", filename)
        py_setitem(stored, "lineno", lineno)
        py_setitem(stored, "file", PythonObject(None))
        py_setitem(stored, "line", PythonObject(None))
        py_setitem(stored, "source", PythonObject(None))
        py_setitem(stored, "_category_name", "RuntimeWarning")
        py_call(log.append, record)
        if mode != "always" and mode != "all":
            py_setitem(context["warning_seen"], key, True)
        return
    var text = String(filename) + ":" + String(lineno) + ": RuntimeWarning: " + message + "\n"
    var line = String(context["warning_source_line"])
    if line.byte_length() > 0:
        text += "  " + line + "\n"
    _warning_write(text, context)
    if mode != "always" and mode != "all":
        py_setitem(context["warning_seen"], key, True)
