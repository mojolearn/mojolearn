# SPDX-License-Identifier: Apache-2.0
# Native handling for the established column-vector target warning. The
# warnings module's Python display and formatting hooks are never invoked.
from std.python import Python, PythonObject

comptime COLUMN_WARNING = "A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel()."


def _warning_fields(value: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if b.type(value) is not b.dict:
        raise Error("TypeError: warning metadata requires native dictionaries")
    var keys = b.list(value.keys())
    for k in range(len(keys)):  # driver-loop: over native warning metadata keys before indexed access
        if b.type(keys[k]) is not b.str:
            raise Error("TypeError: warning metadata keys must be native strings")
    return value


def _warning_class_fields(cls: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if b.type(cls) is not b.type:
        raise Error("TypeError: warning classes require the native metaclass")
    return b.type.__dict__["__dict__"].__get__(cls)


def _warning_category(context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = context["warning_category"]
    _ = _warning_class_fields(cls)
    var mro = b.type.__dict__["__mro__"].__get__(cls)
    var names = Python.tuple("__new__", "__init__", "__str__", "__getattribute__")
    for j in range(len(mro)):  # driver-loop: over the warning exception ancestry
        if _warning_class_fields(mro[j]).get("__del__") is not None:
            raise Error("TypeError: custom warning finalizers are not native")
    for k in range(len(names)):  # driver-loop: over standard native exception slots
        var found = PythonObject(None)
        for j in range(len(mro)):  # driver-loop: over native exception slot declarations
            found = _warning_class_fields(mro[j]).get(names[k])
            if found is not None:
                break
        if found is not context["warning_category_slots"][names[k]]:
            raise Error("TypeError: custom warning exception hooks are not native")
    return cls


def _warning_match(pattern: PythonObject, text: PythonObject, context: PythonObject) raises -> Bool:
    if pattern is None:
        return True
    if Python.import_module("builtins").type(pattern) is not context["warning_pattern_type"]:
        raise Error("TypeError: warning filters require native regular expressions")
    return context["warning_pattern_match"](pattern, text) is not None


def _warning_write(text: String, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var stream = _warning_fields(context["warning_sys_fields"]).get("stderr")
    if stream is None:
        return
    if b.type(stream) is context["warning_stringio"]:
        context["warning_stringio_write"](stream, text)
        return
    if b.type(stream) is not context["warning_textio"]:
        raise Error("TypeError: custom warning output callbacks are not native")
    var buffer = context["warning_textio_buffer"].__get__(stream)
    if b.type(buffer) is context["warning_buffered"]:
        buffer = context["warning_buffered_raw"].__get__(buffer)
    if b.type(buffer) is not context["warning_fileio"]:
        raise Error("TypeError: warning output requires native file storage")
    var encoding = context["warning_textio_encoding"].__get__(stream)
    var errors = context["warning_textio_errors"].__get__(stream)
    if b.type(encoding) is not b.str or b.type(errors) is not b.str:
        raise Error("TypeError: warning output encoding must be native metadata")
    var name = String(encoding.lower().replace("_", "-"))
    if name != "utf-8" and name != "utf8" and name != "ascii" and name != "latin-1" and name != "iso8859-1":
        raise Error("TypeError: warning output requires a native UTF-8, ASCII or Latin-1 encoder")
    if b.type(context["warning_lookup_error"](errors)) is not context["warning_native_function"]:
        raise Error("TypeError: custom warning codec callbacks are not native")
    context["warning_textio_write"](stream, text)


def classical_warn_column(context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var fields = _warning_fields(context["warning_fields"])
    var category = _warning_category(context)
    # Preserve the old stacklevel=4 location in _guard_cpu_training; warning
    # filters and default/module repetition therefore keep their old scope.
    var filename = context["warning_filename"]
    var lineno = Int(py=context["warning_lineno"])
    var module = context["warning_module"]
    var filters = fields.get("filters")
    var log = PythonObject(None)
    if Bool(py=context["warning_context_enabled"]):
        var active = context["warning_context_var"].get(context["warning_global_context"])
        if active is not context["warning_global_context"]:
            if b.type(active) is not context["warning_context_class"]:
                raise Error("TypeError: custom warning contexts are not native")
            var stored = _warning_fields(context["warning_context_fields"].__get__(active))
            filters = stored.get("_filters")
            log = stored.get("log")
    if b.type(filters) is not b.list or (log is not None and b.type(log) is not b.list):
        raise Error("TypeError: warning filters and recording require native lists")
    var old = context["warning_filter_rows"]
    var changed = filters is not context["warning_filter_owner"] or len(old) != len(filters)
    if not changed:
        for k in range(len(filters)):  # driver-loop: over filter row identities for normal filter mutation
            changed = changed or old[k] is not filters[k]
    if changed:
        context["warning_seen"] = b.dict()
        context["warning_filter_owner"] = filters
        context["warning_filter_rows"] = b.tuple(filters)
    var action = fields.get("defaultaction", "default")
    var mro = b.type.__dict__["__mro__"].__get__(category)
    for k in range(len(filters)):  # driver-loop: over configured warning filters in their declared order
        var row = filters[k]
        if b.type(row) is not b.tuple or len(row) != 5 or b.type(row[0]) is not b.str or b.type(row[4]) is not b.int:
            raise Error("TypeError: warning filter rows must use native scalar metadata")
        _ = _warning_class_fields(row[2])
        var category_match = False
        for j in range(len(mro)):  # driver-loop: over native warning category ancestry
            category_match = category_match or row[2] is mro[j]
        if category_match and (Int(py=row[4]) == 0 or Int(py=row[4]) == lineno) and _warning_match(row[1], PythonObject(COLUMN_WARNING), context) and _warning_match(row[3], module, context):
            action = row[0]
            break
    if b.type(action) is not b.str:
        raise Error("TypeError: warning actions must be native strings")
    var mode = String(action)
    if mode == "ignore":
        return
    if mode == "error":
        raise Error("DataConversionWarning: " + COLUMN_WARNING)
    if mode != "always" and mode != "all" and mode != "default" and mode != "module" and mode != "once":
        raise Error("RuntimeError: unrecognized warning filter action")
    var key = Python.tuple(COLUMN_WARNING, category, PythonObject(None) if mode == "once" else module, PythonObject(lineno if mode == "default" else 0))
    if mode != "always" and mode != "all":
        if Bool(py=context["warning_seen"].get(key, False)):
            return
    if not Bool(py=context["warning_standard_hooks"]):
        raise Error("TypeError: custom Python warning hooks are not native")
    var hooks = context["warning_functions"]
    var names = Python.tuple("showwarning", "formatwarning", "_showwarnmsg", "_formatwarnmsg", "_formatwarnmsg_impl")
    for k in range(len(names)):  # driver-loop: over standard warning display hooks before any output
        if fields.get(names[k]) is not hooks[names[k]]:
            raise Error("TypeError: custom Python warning hooks are not native")
    var display = fields.get("_showwarnmsg_impl")
    if b.type(display) is context["warning_native_function"] and String(display.__name__) == "append" and b.type(display.__self__) is b.list:
        log = display.__self__
    elif b.type(display) is context["warning_python_function"] and String(display.__name__) == "_showwarnmsg_impl" and display.__code__.co_filename == context["warning_stdlib_file"]:
        pass
    else:
        raise Error("TypeError: custom warning recording callbacks are not native")
    if log is not None:
        var record_class = context["warning_record"]
        var ancestry = b.type.__dict__["__mro__"].__get__(record_class)
        if len(ancestry) != 2 or ancestry[1] is not b.object or _warning_class_fields(record_class).get("__del__") is not None:
            raise Error("TypeError: custom warning record finalizers are not native")
        var record = b.object.__new__(record_class)
        var stored = context["warning_record_fields"].__get__(record)
        stored["message"] = category(COLUMN_WARNING)
        stored["category"] = category
        stored["filename"] = filename
        stored["lineno"] = lineno
        stored["file"] = PythonObject(None)
        stored["line"] = PythonObject(None)
        stored["source"] = PythonObject(None)
        stored["_category_name"] = "DataConversionWarning"
        log.append(record)
        if mode != "always" and mode != "all":
            context["warning_seen"][key] = True
        return
    var text = String(filename) + ":" + String(lineno) + ": DataConversionWarning: " + COLUMN_WARNING + "\n"
    var line = String(context["warning_source_line"])
    if line.byte_length() > 0:
        text += "  " + line + "\n"
    _warning_write(text, context)
    if mode != "always" and mode != "all":
        context["warning_seen"][key] = True
