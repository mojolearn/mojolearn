# SPDX-License-Identifier: Apache-2.0
# Native handling for established classical validation warnings. The
# warnings module's Python display and formatting hooks are never invoked.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem

comptime COLUMN_WARNING = "A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel()."


def _warning_fields(value: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: warning metadata requires native dictionaries")
    var keys = py_call(b.list, value.keys())
    for k in range(len(keys)):  # driver-loop: over native warning metadata keys before indexed access
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: warning metadata keys must be native strings")
    return value


def _warning_class_fields(cls: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: warning classes require the native metaclass")
    return py_call(b.type.__dict__["__dict__"].__get__, cls)


def _warning_category(context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    # scikit-learn's DataConversionWarning once sklearn is loaded, else
    # mojolearn's own (_sk_validate._conversion_warning); the error path reads
    # the same class back from the context.
    var cls = context["warning_category_default"]
    var module = py_get(Python.import_module("sys").modules, "sklearn.exceptions")
    if module is not None and py_type(module) is py_type(Python.import_module("sys")):
        var loaded = py_get(py_call(b.object.__getattribute__, module, "__dict__"), "DataConversionWarning")
        if loaded is not None:
            cls = loaded
    return cls


def _warning_admit_category(cls: PythonObject, context: PythonObject) raises:
    # Every category, including a supplied UserWarning, must be admitted
    # before reading its metadata or constructing a warning instance.
    # A C exception entry can otherwise invoke Python subclass hooks.
    var b = Python.import_module("builtins")
    _ = _warning_class_fields(cls)
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var names = Python.tuple("__new__", "__init__", "__str__", "__getattribute__")
    for j in range(len(mro)):  # driver-loop: over the warning exception ancestry
        if py_get(_warning_class_fields(mro[j]), "__del__") is not None:
            raise Error("TypeError: custom warning finalizers are not native")
    for k in range(len(names)):  # driver-loop: over standard native exception slots
        var found = PythonObject(None)
        for j in range(len(mro)):  # driver-loop: over native exception slot declarations
            found = py_get(_warning_class_fields(mro[j]), names[k])
            if found is not None:
                break
        if found is not context["warning_category_slots"][names[k]]:
            raise Error("TypeError: custom warning exception hooks are not native")


def _warning_match(pattern: PythonObject, text: PythonObject, context: PythonObject) raises -> Bool:
    if pattern is None:
        return True
    if py_type(pattern) is not context["warning_pattern_type"]:
        raise Error("TypeError: warning filters require native regular expressions")
    return py_call(context["warning_pattern_match"], pattern, text) is not None


def _warning_write(text: String, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var stream = py_get(_warning_fields(context["warning_sys_fields"]), "stderr")
    if stream is None:
        return
    if py_type(stream) is context["warning_stringio"]:
        py_call(context["warning_stringio_write"], stream, text)
        return
    if py_type(stream) is not context["warning_textio"]:
        raise Error("TypeError: custom warning output callbacks are not native")
    var buffer = py_call(context["warning_textio_buffer"].__get__, stream)
    if py_type(buffer) is context["warning_buffered"]:
        buffer = py_call(context["warning_buffered_raw"].__get__, buffer)
    if py_type(buffer) is not context["warning_fileio"]:
        raise Error("TypeError: warning output requires native file storage")
    var encoding = py_call(context["warning_textio_encoding"].__get__, stream)
    var errors = py_call(context["warning_textio_errors"].__get__, stream)
    if py_type(encoding) is not b.str or py_type(errors) is not b.str:
        raise Error("TypeError: warning output encoding must be native metadata")
    var name = String(py_call(encoding.lower().replace, "_", "-"))
    if name != "utf-8" and name != "utf8" and name != "ascii" and name != "latin-1" and name != "iso8859-1":
        raise Error("TypeError: warning output requires a native UTF-8, ASCII or Latin-1 encoder")
    if py_type(py_call(context["warning_lookup_error"], errors)) is not context["warning_native_function"]:
        raise Error("TypeError: custom warning codec callbacks are not native")
    py_call(context["warning_textio_write"], stream, text)


def classical_warn_column(context: PythonObject) raises:
    classical_warning(COLUMN_WARNING, _warning_category(context), context)


def classical_warning(message: String, category: PythonObject, context: PythonObject, location: PythonObject = PythonObject(None), error_prefix: String = "DataConversionWarning") raises:
    _warning_admit_category(category, context)
    py_setitem(context, "warning_category", category)
    var b = Python.import_module("builtins")
    var fields = _warning_fields(context["warning_fields"])
    # Preserve the old stacklevel=4 location in _guard_cpu_training; warning
    # filters and default/module repetition therefore keep their old scope.
    var filename = context["warning_filename"]
    var lineno = Int(py=context["warning_lineno"])
    var module = context["warning_module"]
    var source_line = context["warning_source_line"]
    if location is not None:
        filename = location[0]
        lineno = Int(py=location[1])
        module = location[2]
        source_line = PythonObject("")
    var category_name = String(py_call(b.type.__getattribute__, category, "__name__"))
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
        raise Error(error_prefix + ": " + message)
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
        py_setitem(stored, "_category_name", category_name)
        py_call(log.append, record)
        if mode != "always" and mode != "all":
            py_setitem(context["warning_seen"], key, True)
        return
    var text = String(filename) + ":" + String(lineno) + ": " + category_name + ": " + message + "\n"
    var line = String(source_line)
    if line.byte_length() > 0:
        text += "  " + line + "\n"
    _warning_write(text, context)
    if mode != "always" and mode != "all":
        py_setitem(context["warning_seen"], key, True)
