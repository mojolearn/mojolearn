# SPDX-License-Identifier: Apache-2.0
# Native handling for established classical validation warnings. The
# warnings module's Python display and formatting hooks are never invoked.
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_keys_unpack import classical_keys
from bindings.py_balanced import py_item

comptime COLUMN_WARNING = "A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel()."


def _warning_fields(value: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: warning metadata requires native dictionaries")
    var keys = py_call(b.list, value.keys())
    for k in range(len(keys)):  # driver-loop: over native warning metadata keys before indexed access
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: warning metadata keys must be native strings")
    return value


def _warning_class_fields(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: warning classes require the native metaclass")
    var nk = classical_keys()
    var fields = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, cls)
    var keys = py_call(b.list,fields)
    for k in range(len(keys)):  # driver-loop: class metadata keys precede lookup that could invoke Python equality
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: warning class metadata keys must be native strings")
    return fields


def _warning_category(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    # scikit-learn's DataConversionWarning once sklearn is loaded, else
    # mojolearn's own (_sk_validate._conversion_warning); the error path reads
    # the same class back from the context.
    var nk = classical_keys()
    var cls = py_item(context, nk[].k_warning_category_default)
    var module = py_get(native_loaded_module("sys").modules, "sklearn.exceptions")
    if module is not None and py_type(module) is py_type(native_loaded_module("sys")):
        var loaded = py_get(py_call(b.object.__getattribute__, module, "__dict__"), nk[].k_DataConversionWarning)
        if loaded is not None:
            cls = loaded
    return cls


def _warning_admit_category(cls: PythonObject, context: PythonObject) raises:
    # Every category, including a supplied UserWarning, must be admitted
    # before reading its metadata or constructing a warning instance.
    # A C exception entry can otherwise invoke Python subclass hooks.
    var b = native_loaded_module("builtins")
    _ = _warning_class_fields(cls)
    var nk = classical_keys()
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    var warning = False
    var names = Python.tuple("__new__", "__init__", "__str__", "__getattribute__")
    for j in range(len(mro)):  # driver-loop: over the warning exception ancestry
        warning = warning or mro[j] is py_item(context, nk[].k_warning_base)
        if py_get(_warning_class_fields(mro[j]), nk[].k___del__) is not None:
            raise Error("TypeError: custom warning finalizers are not native")
    if not warning or py_type(py_call(b.type.__getattribute__,cls,"__name__")) is not b.str:
        raise Error("TypeError: warning category names and ancestry must be native metadata")
    for k in range(len(names)):  # driver-loop: over standard native exception slots
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var declared = Optional[PythonObject]()
        for j in range(len(mro)):  # driver-loop: over native exception slot declarations
            var slot = py_get(_warning_class_fields(mro[j]), names[k])
            if slot is not None:
                declared = slot
                break
        if not declared:
            # every warning class inherits BaseException's slots; none declared is not a native warning
            raise Error("TypeError: custom warning exception hooks are not native")
        var found = declared.value()
        var admitted = found is py_item(context, nk[].k_warning_category_slots)[names[k]]
        if k == 0:
            var constructors = py_item(context, nk[].k_warning_native_new)
            for j in range(len(constructors)):  # driver-loop: exact retained native constructors of builtin warning bases
                admitted = admitted or found is constructors[j]
        elif k == 1:
            var initializers = py_item(context, nk[].k_warning_native_init)
            for j in range(len(initializers)):  # driver-loop: exact retained builtin warning initializers, including CPython 3.11's per-category slots
                admitted = admitted or found is initializers[j]
        if not admitted:
            raise Error("TypeError: custom warning exception hooks are not native")


def _warning_match(pattern: PythonObject, text: PythonObject, context: PythonObject) raises -> Bool:
    if pattern is None:
        return True
    var nk = classical_keys()
    if py_type(pattern) is not py_item(context, nk[].k_warning_pattern_type):
        raise Error("TypeError: warning filters require native regular expressions")
    return py_call(py_item(context, nk[].k_warning_pattern_match), pattern, text) is not None


def _warning_write(text: String, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var nk = classical_keys()
    var stream = py_get(_warning_fields(py_item(context, nk[].k_warning_sys_fields)), nk[].k_stderr)
    if stream is None:
        return
    if py_type(stream) is py_item(context, nk[].k_warning_stringio):
        py_call(py_item(context, nk[].k_warning_stringio_write), stream, text)
        return
    if py_type(stream) is not py_item(context, nk[].k_warning_textio):
        raise Error("TypeError: custom warning output callbacks are not native")
    var buffer = py_call(py_item(context, nk[].k_warning_textio_buffer).__get__, stream)
    if py_type(buffer) is py_item(context, nk[].k_warning_buffered):
        buffer = py_call(py_item(context, nk[].k_warning_buffered_raw).__get__, buffer)
    if py_type(buffer) is not py_item(context, nk[].k_warning_fileio):
        raise Error("TypeError: warning output requires native file storage")
    var encoding = py_call(py_item(context, nk[].k_warning_textio_encoding).__get__, stream)
    var errors = py_call(py_item(context, nk[].k_warning_textio_errors).__get__, stream)
    if py_type(encoding) is not b.str or py_type(errors) is not b.str:
        raise Error("TypeError: warning output encoding must be native metadata")
    var name = String(py_call(encoding.lower().replace, "_", "-"))
    if name != "utf-8" and name != "utf8" and name != "ascii" and name != "latin-1" and name != "iso8859-1":
        raise Error("TypeError: warning output requires a native UTF-8, ASCII or Latin-1 encoder")
    if py_type(py_call(py_item(context, nk[].k_warning_lookup_error), errors)) is not py_item(context, nk[].k_warning_native_function):
        raise Error("TypeError: custom warning codec callbacks are not native")
    py_call(py_item(context, nk[].k_warning_textio_write), stream, text)


def classical_warn_column(context: PythonObject) raises:
    var location = PythonObject(None)
    classical_warning(COLUMN_WARNING, _warning_category(context), context, location)


def classical_warning(message: String, category: PythonObject, context: PythonObject, location: PythonObject, error_prefix: String = "DataConversionWarning") raises:
    _warning_admit_category(category, context)
    var nk = classical_keys()
    py_setitem(context, nk[].k_warning_category, category)
    var b = native_loaded_module("builtins")
    var fields = _warning_fields(py_item(context, nk[].k_warning_fields))
    # Preserve the old stacklevel=4 location in _guard_cpu_training; warning
    # filters and default/module repetition therefore keep their old scope.
    var filename = py_item(context, nk[].k_warning_filename)
    var lineno = Int(py=py_item(context, nk[].k_warning_lineno))
    var module = py_item(context, nk[].k_warning_module)
    var source_line = py_item(context, nk[].k_warning_source_line)
    if location is not None:
        filename = location[0]
        lineno = Int(py=location[1])
        module = location[2]
        source_line = PythonObject("")
    var category_name = String(py_call(b.type.__getattribute__, category, "__name__"))
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
        raise Error(error_prefix + ": " + message)
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
        py_setitem(stored, nk[].k__category_name, category_name)
        py_call(log.value().append, record)
        if mode != "always" and mode != "all":
            py_setitem(py_item(context, nk[].k_warning_seen), key, True)
        return
    var text = String(filename) + ":" + String(lineno) + ": " + category_name + ": " + message + "\n"
    var line = String(source_line)
    if line.byte_length() > 0:
        text += "  " + line + "\n"
    _warning_write(text, context)
    if mode != "always" and mode != "all":
        py_setitem(py_item(context, nk[].k_warning_seen), key, True)
