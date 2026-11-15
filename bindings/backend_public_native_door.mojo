# SPDX-License-Identifier: Apache-2.0
# Whole public backend metadata operations. Providers/loader outcomes are
# connected once at package import; product calls never enter Python loaders.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.os import getenv
from std.os.path import isfile
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_item, py_contains, py_fields
from core.sha256_file import sha256_file
from bindings.optimizer_public_words_unpack import stash_python_error, optimizer_python_error
from bindings.py_cpython_table import cpython_table
from bindings.py_optional import py_or_none, py_present


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
            cpython_table()[].PyErr_Clear()
            return PythonObject(None)
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=raw)


def _gpu_arch(fields: PythonObject, how: Bool) raises -> PythonObject:
    if fields["_CPU_ONLY"] is not None:
        return PythonObject("no GPU binary set loaded (CPU-only install, DEVIATION 2615)") if how else PythonObject(None)
    return _text(fields["_ARCH_HOW"] if how else fields["_ARCH_SELECTED"], "architecture metadata", True)


def _gbdt_mode_entry(fields: PythonObject, context: PythonObject, mode: PythonObject) raises -> PythonObject:
    """The loaded gbdt `gbdt_numeric_mode` entry for `mode`, or None."""
    var selected = py_get(fields, "_SELECTED")
    if selected is None:
        selected = PythonObject("fast")
    _ = _text(selected, "selected mode")
    var entry = context["selected_mode"] if mode == selected else context["plans"][mode]["mode"]
    return entry


def _numeric_mode(fields: PythonObject, context: PythonObject, mode: PythonObject) raises -> PythonObject:
    var entry = _gbdt_mode_entry(fields, context, mode)
    if entry is not None:
        var code = _read(entry, context)
        if py_type(code) is not py_type(PythonObject(0)):
            raise Error("TypeError: compiled numeric mode requires a native integer code")
        var compiled = py_get(context["mode_codes"], code, PythonObject("unknown"))
        if compiled != mode:
            raise Error("RuntimeError: mojolearn: the loaded gbdt binary was compiled " + String(compiled) + " but the selector loaded the " + String(mode) + " set -- a binary is in the wrong directory; rebuild both sets")
    return mode


def _vendor(fields: PythonObject, context: PythonObject, mode: PythonObject) raises -> PythonObject:
    if fields["_CPU_ONLY"] is not None:
        return PythonObject("cpu")
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var found_said = Optional[PythonObject]()
    var entries = context["plans"][mode]["vendors"]
    for i in range(len(entries)):  # driver-loop: native provider metadata readbacks in selector order
        var read = _read(entries[i], context, True)
        if read is not None:
            found_said = _text(read, "compiled vendor")
            break
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var said = Optional[PythonObject]()
    if found_said:
        said = py_present(found_said.value())
    var expected = _text(fields["_VENDOR_SELECTED"], "selected vendor", True)
    var layout = fields["_LAYOUT"]
    if py_type(layout) is not py_type(Python.tuple()) or len(layout) != 2:
        raise Error("TypeError: backend layout requires native import-connected metadata")
    var kind = _text(layout[0], "layout kind")
    if String(kind) == "vendor":
        if Bool(said) and said.value() != expected:
            raise Error("RuntimeError: mojolearn: the loaded " + String(mode) + " binaries were compiled for " + String(said.value()) + " but the selector opened the " + String(expected) + " set -- a binary is in the wrong vendor directory; rebuild both sets")
        return expected
    if Bool(said):
        py_setitem(fields, "_VENDOR_SELECTED", said.value())
        return py_or_none(said)
    return expected


# ---- runtime_report (lane verify-runtime-report, 2026-11-15) ----------------
# mojolearn.runtime_report(): the one read-only "what am I running" report the
# shipped verify modules read instead of product internals. The field contract
# is docs/plans/VERIFY_DECOUPLE_2026-11-15.md "API". Its own call, never on a
# product call path: it loads no binding, enters no Python loader and no
# Python-level function (posix.listdir/getcwd and C methods of str, list and
# tuple are the only callees), and reads the selection made at import.


def _env(name: String) -> String:
    return String(getenv(name, "").strip())


def _is_hex_commit(value: String) -> Bool:
    var n = value.byte_length()
    if n < 7 or n > 40:
        return False
    for b in value.as_bytes():  # small-loop(at most 40 bytes of one commit id)
        var c = Int(b)
        if not ((c >= ord("0") and c <= ord("9")) or (c >= ord("a") and c <= ord("f"))):
            return False
    return True


def _entry(key: String, value: PythonObject, how: String) raises -> PythonObject:
    var row = Python.dict()
    py_setitem(row, PythonObject(key), value)
    py_setitem(row, PythonObject("how"), PythonObject(how))
    return row


def _checkout_root(context: PythonObject) raises -> PythonObject:
    # The nearest package parent (six at most, listed once at connect) that
    # holds tools/identity_break.py.
    var parents = context["package_parents"]
    for i in range(len(parents)):  # small-loop(six parent directories at most)
        var base = String(parents[i])
        if isfile(base + "/tools/identity_break.py"):
            return PythonObject(base)
    return PythonObject(None)


def _harness_source(context: PythonObject, root: PythonObject) raises -> PythonObject:
    var explicit = _env("MOJOLEARN_IDENTITY_BREAK")
    if explicit.byte_length() > 0 and isfile(explicit):
        return _entry("path", PythonObject(explicit), "MOJOLEARN_IDENTITY_BREAK")
    var copy = String(context["package_dir"]) + "/_identity_break.py"
    if isfile(copy):
        return _entry("path", PythonObject(copy), "wheel copy")
    if root is not None:
        return _entry("path", PythonObject(String(root) + "/tools/identity_break.py"), "checkout")
    return _entry("path", PythonObject(None), "missing")


def _source_commit(context: PythonObject) raises -> PythonObject:
    var explicit = _env("MOJOLEARN_COMMIT")
    if explicit.byte_length() > 0:
        return _entry("commit", PythonObject(explicit), "MOJOLEARN_COMMIT")
    var witness = String(context["package_dir"]) + "/identity_columns/COMMIT"
    if isfile(witness):
        var text: String
        with open(witness, "r") as fh:
            text = fh.read()
        var words = py_call(py_type(PythonObject("")).split, PythonObject(text))
        if len(words) > 0:
            var value = String(words[0])
            if _is_hex_commit(value):
                return _entry("commit", PythonObject(value), "wheel COMMIT witness")
    return _entry("commit", PythonObject(None), "git rev-parse HEAD of the checkout (the harness reads it)")


def _host_dir(context: PythonObject) raises -> String:
    var override = _env("MOJOLEARN_HOST_DIR")
    if override.byte_length() == 0:
        return String(context["package_dir"]) + "/host"
    if override.startswith("/"):
        return override
    return String(py_call(context["getcwd"])) + "/" + override


def _host_families(context: PythonObject, host_dir: String) raises -> PythonObject:
    var found = Python.dict()
    var names = Python.list()
    try:
        names = py_call(context["listdir"], PythonObject(host_dir))
    except:
        names = Python.list()  # no host directory: no host binding is built
    for i in range(len(names)):  # small-loop(file names of the one host binding directory)
        var name = String(names[i])
        if name.startswith("_mojolearn_") and name.endswith("_host.so"):
            py_setitem(found, py_call(py_type(PythonObject("")).removesuffix, names[i], PythonObject(".so")), PythonObject(True))
    # Merged host files: the index the package read when it resolved a host
    # binding (`_merged._INDEX_CACHE`: path -> (stamp, {member: file})).
    var index_path = host_dir + "/" + String(context["merged_index_name"])
    var cached = py_get(context["merged_cache"], PythonObject(index_path))
    if cached is not None:
        var owners = py_item(cached, PythonObject(1))
        var members = py_call(context["list_type"], owners)
        for i in range(len(members)):  # small-loop(member names of the one merged index)
            var member = String(members[i])
            if member.endswith("_host") and isfile(String(py_item(owners, members[i]))):
                py_setitem(found, members[i], PythonObject(True))
    var built = py_call(context["list_type"], found)
    _ = py_call(built.sort)
    return py_call(py_type(Python.tuple()), built)


def _identity(cache: PythonObject, path: String) raises -> PythonObject:
    # One SHA-256 per distinct file: a merged file serves many members.
    var known = py_get(cache, PythonObject(path))
    if known is not None:
        return known
    var digest = PythonObject(sha256_file(path))
    py_setitem(cache, PythonObject(path), digest)
    return digest


def _binding_row(path: String, loaded: Bool, identity: PythonObject, vendor: PythonObject, mode: PythonObject) raises -> PythonObject:
    var row = Python.dict()
    py_setitem(row, PythonObject("path"), PythonObject(path))
    py_setitem(row, PythonObject("loaded"), PythonObject(True) if loaded else PythonObject(False))
    py_setitem(row, PythonObject("identity"), identity)
    py_setitem(row, PythonObject("vendor"), vendor)
    py_setitem(row, PythonObject("mode"), mode)
    return row


def _bindings(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var rows = Python.dict()
    var hashes = Python.dict()
    var modules = context["modules"]
    var keys = py_call(context["list_type"], modules)
    var text_type = py_type(PythonObject(""))
    for i in range(len(keys)):  # driver-loop: over the loaded module table, once per report
        if py_type(keys[i]) is not text_type:
            continue
        var key = String(keys[i])
        var parts = key.split(".")
        var last = String(parts[len(parts) - 1])
        if not last.startswith("_mojolearn"):
            continue
        if not (key.startswith("mojolearn.") or key == last):
            continue
        var module = py_get(modules, keys[i])
        # Exact module type only: a CPU-only `_HostBinding`, a tier set or a
        # refusing stub would load or refuse on attribute access. The
        # instance dictionary is read without class or instance hooks.
        if module is None or py_type(module) is not context["module_type"]:
            continue
        var namespace = py_fields(module)
        var file = py_get(namespace, PythonObject("__file__"))
        if file is None or py_type(file) is not text_type:
            continue
        var file_path = String(file)
        if not isfile(file_path):
            continue
        var fn_name: String
        if last == "_mojolearn":
            fn_name = "mojolearn_vendor"
        else:
            fn_name = String(py_call(text_type.removeprefix, PythonObject(last), PythonObject("_mojolearn_"))) + "_vendor"
        var vendor = PythonObject(None)
        var fn = py_get(namespace, PythonObject(fn_name))
        if fn is not None and py_type(fn) is context["native_function_type"]:
            var read = _read(fn, context, True)
            if read is not None:
                vendor = _text(read, "compiled vendor")
        var tier = PythonObject(None)
        if len(parts) == 4 and String(parts[1]) == "_sets":
            tier = PythonObject(String(parts[2]))
        py_setitem(rows, keys[i], _binding_row(file_path, True, _identity(hashes, file_path), vendor, tier))
    # Members of a loaded tier that have not loaded yet (their file is known).
    var sets = fields["_SETS"]
    var tiers = py_call(context["list_type"], sets)
    for i in range(len(tiers)):  # driver-loop: over the numeric-mode tiers (two at most)
        var deferred = py_get(py_fields(py_item(sets, tiers[i])), PythonObject("_deferred"))
        if deferred is None:
            continue
        var names = py_call(context["list_type"], deferred)
        for j in range(len(names)):  # driver-loop: over one tier's not-yet-loaded members
            var key = PythonObject("mojolearn._sets." + String(tiers[i]) + "." + String(names[j]))
            if py_contains(rows, key):
                continue
            var path = String(py_item(deferred, names[j]))
            var identity = PythonObject(None)
            if isfile(path):
                identity = _identity(hashes, path)
            py_setitem(rows, key, _binding_row(path, False, identity, PythonObject(None), tiers[i]))
    return rows


def _runtime_report(fields: PythonObject, context: PythonObject, mode: PythonObject) raises -> PythonObject:
    var report = Python.dict()
    py_setitem(report, PythonObject("version"), context["version"])
    py_setitem(report, PythonObject("vendor"), _vendor(fields, context, mode))
    py_setitem(report, PythonObject("numeric_mode"), _numeric_mode(fields, context, mode))
    py_setitem(report, PythonObject("default_mode"), mode)
    var code_mode = PythonObject(None)
    var entry = _gbdt_mode_entry(fields, context, mode)
    if entry is not None:
        code_mode = py_get(context["mode_codes"], _read(entry, context), PythonObject("unknown"))
    py_setitem(report, PythonObject("code_mode"), code_mode)
    py_setitem(report, PythonObject("cpu_only"), PythonObject(False) if fields["_CPU_ONLY"] is None else PythonObject(True))
    py_setitem(report, PythonObject("gpu_arch"), _gpu_arch(fields, False))
    py_setitem(report, PythonObject("gpu_arch_how"), _gpu_arch(fields, True))
    var host_dir = _host_dir(context)
    py_setitem(report, PythonObject("host_dir"), PythonObject(host_dir))
    py_setitem(report, PythonObject("host_families_built"), _host_families(context, host_dir))
    py_setitem(report, PythonObject("bindings"), _bindings(fields, context))
    py_setitem(report, PythonObject("fallbacks"), context["fallbacks"])
    var root = _checkout_root(context)
    py_setitem(report, PythonObject("harness_source"), _harness_source(context, root))
    py_setitem(report, PythonObject("source_commit"), _source_commit(context))
    py_setitem(report, PythonObject("checkout_root"), root)
    py_setitem(report, PythonObject("package_dir"), context["package_dir"])
    return report


def backend_public_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = context["backend_fields"]
    if operation == "gpu_arch" or operation == "gpu_arch_how":
        return _gpu_arch(fields, operation == "gpu_arch_how")
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
        return _numeric_mode(fields, context, mode)
    if operation == "vendor":
        return _vendor(fields, context, mode)
    if operation == "runtime_report":
        if len(args) != 0:
            raise Error("TypeError: runtime_report takes no arguments")
        return _runtime_report(fields, context, mode)
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
