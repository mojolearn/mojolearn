# SPDX-License-Identifier: Apache-2.0
# Native warning delivery for sequence array-protocol casts.
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_warnings_unpack import _warning_fields, _warning_class_fields, _warning_match, _warning_write
from bindings.optimizer_public_words_unpack import stash_python_error
from core.sequence_array_cast import SEQ_CAST_OVERFLOW, SEQ_CAST_UNDERFLOW, SEQ_CAST_INVALID


from bindings.sequence_array_warnings_pack import (
    sequence_array_category,
)
from bindings.sequence_keys_unpack import sequence_keys
from bindings.py_balanced import py_item


def sequence_array_report(status: Int, settings: PythonObject, context: PythonObject,
                          operation: String = "cast", suppress_error: Bool = False) raises:
    var nk = sequence_keys()
    var warning = py_item(context, nk[].k_sequence_array_warnings)
    var b = native_loaded_module("builtins")
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
        var message = descriptions[i] + " encountered in " + operation
        if mode == "ignore":
            continue
        if mode == "warn":
            sequence_array_warn(message, warning, "RuntimeWarning", suppress_error)
        elif mode == "raise":
            if not suppress_error:
                sequence_array_exception("FloatingPointError", message)
        elif mode == "print":
            _warning_write("Warning: " + message + "\n", warning)
        elif mode == "log":
            # Native text streams remain supported. The shared writer checks
            # exact native IO/codec identities before reaching their methods.
            var local = py_call(b.dict, warning)
            var fields = py_call(b.dict, py_item(warning, nk[].k_warning_sys_fields))
            py_setitem(fields, nk[].k_stderr, py_item(settings, nk[].k_call))
            py_setitem(local, nk[].k_warning_sys_fields, fields)
            _warning_write("Warning: " + message + "\n", local)
        else:
            raise Error("TypeError: Python NumPy error callbacks cannot execute in array output; use native errstate ignore, warn or raise")


def sequence_array_exception(kind: String, var message: String) raises:
    ref cpy = Python().cpython()
    var category = sequence_array_category(kind)
    cpy.PyErr_SetString(category._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
    var failure = stash_python_error()
    raise failure^


def sequence_array_warn(message: String, context: PythonObject, kind: String = "RuntimeWarning",
                        suppress_error: Bool = False) raises:
    var b = native_loaded_module("builtins")
    var nk = sequence_keys()
    var fields = _warning_fields(py_item(context, nk[].k_warning_fields))
    var category = sequence_array_category(kind)
    # The existing warning originated at the __array__ public method. Its
    # native metadata retains that location for filtering and repetition.
    var filename = py_item(context, nk[].k_warning_filename)
    var lineno = Int(py=py_item(context, nk[].k_warning_lineno))
    var module = py_item(context, nk[].k_warning_module)
    var filters = py_get(fields, nk[].k_filters)
    # Optional, not a PythonObject(None) sentinel: a context's `log` of None is absent (modular/modular#7546).
    var log = Optional[PythonObject]()
    if Bool(py=py_item(context, nk[].k_warning_context_enabled)):
        var active = py_get(py_item(context, nk[].k_warning_context_var), py_item(context, nk[].k_warning_global_context))
        if active is not py_item(context, nk[].k_warning_global_context):
            if py_type(active) is not py_item(context, nk[].k_warning_context_class):
                raise Error("TypeError: custom warning contexts are not native")
            var stored = _warning_fields(py_call(py_item(context, nk[].k_warning_context_fields).__get__, active))
            filters = py_get(stored, nk[].k__filters)
            var recorded = py_get(stored, nk[].k_log)
            if recorded is not None:
                log = recorded
    if py_type(filters) is not b.list or (Bool(log) and py_type(log.value()) is not b.list):
        raise Error("TypeError: warning filters and recording require native lists")
    var old = py_item(context, nk[].k_warning_filter_rows)
    var changed = filters is not py_item(context, nk[].k_warning_filter_owner) or len(old) != len(filters)
    if not changed:
        for k in range(len(filters)):  # driver-loop: over filter row identities for normal filter mutation
            changed = changed or old[k] is not filters[k]
    if changed:
        py_setitem(context, nk[].k_warning_seen, b.dict())
        py_setitem(context, nk[].k_warning_filter_owner, filters)
        py_setitem(context, nk[].k_warning_filter_rows, py_call(b.tuple, filters))
    var action = py_get(fields, nk[].k_defaultaction, "default")
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, category)
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
        if suppress_error:
            return
        sequence_array_exception(kind, message)
    if mode != "always" and mode != "all" and mode != "default" and mode != "module" and mode != "once":
        raise Error("RuntimeError: unrecognized warning filter action")
    var key = Python.tuple(message, category, PythonObject(None) if mode == "once" else module, PythonObject(lineno if mode == "default" else 0))
    if mode != "always" and mode != "all":
        if Bool(py=py_get(py_item(context, nk[].k_warning_seen), key, False)):
            return
    if not Bool(py=py_item(context, nk[].k_warning_standard_hooks)):
        raise Error("TypeError: custom Python warning hooks are not native")
    var hooks = py_item(context, nk[].k_warning_functions)
    var names = Python.tuple("showwarning", "formatwarning", "_showwarnmsg", "_formatwarnmsg", "_formatwarnmsg_impl")
    for k in range(len(names)):  # driver-loop: over standard warning display hooks before any output
        if py_get(fields, names[k]) is not hooks[names[k]]:
            raise Error("TypeError: custom Python warning hooks are not native")
    var display = py_get(fields, nk[].k__showwarnmsg_impl)
    if py_type(display) is py_item(context, nk[].k_warning_native_function) and String(display.__name__) == "append" and py_type(display.__self__) is b.list:
        log = display.__self__
    elif py_type(display) is py_item(context, nk[].k_warning_python_function) and String(display.__name__) == "_showwarnmsg_impl" and display.__code__.co_filename == py_item(context, nk[].k_warning_stdlib_file):
        pass
    else:
        raise Error("TypeError: custom warning recording callbacks are not native")
    if log:
        var record_class = py_item(context, nk[].k_warning_record)
        var ancestry = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, record_class)
        if len(ancestry) != 2 or ancestry[1] is not b.object or py_get(_warning_class_fields(record_class), nk[].k___del__) is not None:
            raise Error("TypeError: custom warning record finalizers are not native")
        var record = py_call(b.object.__new__, record_class)
        var stored = py_call(py_item(context, nk[].k_warning_record_fields).__get__, record)
        py_setitem(stored, nk[].k_message, py_call(category, message))
        py_setitem(stored, nk[].k_category, category)
        py_setitem(stored, nk[].k_filename, filename)
        py_setitem(stored, nk[].k_lineno, lineno)
        py_setitem(stored, nk[].k_file, PythonObject(None))
        py_setitem(stored, nk[].k_line, PythonObject(None))
        py_setitem(stored, nk[].k_source, PythonObject(None))
        py_setitem(stored, nk[].k__category_name, PythonObject(kind))
        py_call(log.value().append, record)
        if mode != "always" and mode != "all":
            py_setitem(py_item(context, nk[].k_warning_seen), key, True)
        return
    var text = String(filename) + ":" + String(lineno) + ": " + kind + ": " + message + "\n"
    var line = String(py_item(context, nk[].k_warning_source_line))
    if line.byte_length() > 0:
        text += "  " + line + "\n"
    _warning_write(text, context)
    if mode != "always" and mode != "all":
        py_setitem(py_item(context, nk[].k_warning_seen), key, True)
