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
from bindings.mode_answer_door import connect_mode_answer_words


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


def _mode_name(fields: PythonObject, mode: PythonObject) raises -> PythonObject:
    """The name of the process default as the user chose it: `_DEFAULT_NAME`
    ('fast' while the identical set answers that name, `_backend._FAST_ANSWER`),
    else the set word `mode` itself."""
    var name = py_get(fields, "_DEFAULT_NAME")
    if name is None:
        return mode
    return _text(name, "default mode name")


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
    # One SHA-256 per distinct file and file state: a merged file serves many
    # members, and a report read repeatedly (verify reads it per lane) hashes
    # a binding again only when its size or modification time changed. The
    # cache lives in the import-connected context; key (path, mtime_ns, size).
    var st = py_call(cache["stat"], PythonObject(path))
    var key = Python.tuple(PythonObject(path), st.st_mtime_ns, st.st_size)
    var digests = cache["digests"]
    var known = py_get(digests, key)
    if known is not None:
        return known
    var digest = PythonObject(sha256_file(path))
    py_setitem(digests, key, digest)
    return digest


def _host_expected_path(context: PythonObject, host_dir: String, basename: String) raises -> PythonObject:
    """The file the host binding `basename` (`_mojolearn_<x>_host`) loads or
    would load, as `_backend.door_host_binary_path(basename)` answers it: the
    `MOJOLEARN_<X>_HOST_BINARY` override (absolute, else joined to the working
    directory), else `_merged.binding_path(host_dir, basename)`: its own
    `<basename>.so` when present, else the merged file its index names (from
    the index the package read), else the own path (absent)."""
    var prefix = String(basename.removeprefix("_mojolearn_"))
    var override = _env("MOJOLEARN_" + prefix.upper() + "_BINARY")
    if override.byte_length() > 0:
        if override.startswith("/"):
            return PythonObject(override)
        return PythonObject(String(py_call(context["getcwd"])) + "/" + override)
    var own = host_dir + "/" + basename + ".so"
    if isfile(own):
        return PythonObject(own)
    var cached = py_get(context["merged_cache"], PythonObject(host_dir + "/" + String(context["merged_index_name"])))
    if cached is not None:
        var merged = py_get(py_item(cached, PythonObject(1)), PythonObject(basename))
        if merged is not None and isfile(String(merged)):
            return merged
    return PythonObject(own)


def _binding_row(path: String, loaded: Bool, identity: PythonObject, vendor: PythonObject, mode: PythonObject) raises -> PythonObject:
    var row = Python.dict()
    py_setitem(row, PythonObject("path"), PythonObject(path))
    py_setitem(row, PythonObject("expected_path"), PythonObject(None))
    py_setitem(row, PythonObject("loaded"), PythonObject(True) if loaded else PythonObject(False))
    py_setitem(row, PythonObject("identity"), identity)
    py_setitem(row, PythonObject("vendor"), vendor)
    py_setitem(row, PythonObject("mode"), mode)
    py_setitem(row, PythonObject("compiled_mode"), PythonObject(None))
    py_setitem(row, PythonObject("sabotage"), PythonObject(None))
    py_setitem(row, PythonObject("column"), PythonObject(None))
    py_setitem(row, PythonObject("detected_column"), PythonObject(None))
    return row


def _binding_row_unresolved() raises -> PythonObject:
    var row = Python.dict()
    py_setitem(row, PythonObject("path"), PythonObject(None))
    py_setitem(row, PythonObject("expected_path"), PythonObject(None))
    py_setitem(row, PythonObject("loaded"), PythonObject(True))
    py_setitem(row, PythonObject("identity"), PythonObject(None))
    py_setitem(row, PythonObject("vendor"), PythonObject(None))
    py_setitem(row, PythonObject("mode"), PythonObject(None))
    py_setitem(row, PythonObject("compiled_mode"), PythonObject(None))
    py_setitem(row, PythonObject("sabotage"), PythonObject(None))
    py_setitem(row, PythonObject("column"), PythonObject(None))
    py_setitem(row, PythonObject("detected_column"), PythonObject(None))
    return row


def _host_column(namespace: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    # One host binding's column read-back as text, None when the binding
    # exports no such entry. Read through `_read` like the vendor read-back.
    var entry = py_get(namespace, PythonObject(name))
    if entry is None or py_type(entry) is not context["native_function_type"]:
        return PythonObject(None)
    var said = _read(entry, context, True)
    if said is None:
        return said
    return py_call(py_type(PythonObject("")), said)


def _bindings(fields: PythonObject, context: PythonObject, host_dir: String) raises -> PythonObject:
    var rows = Python.dict()
    var hashes = context["identity_cache"]
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
        # A host binding (`mojolearn._host._mojolearn_<x>_host`) also says the
        # file it should have loaded (`expected_path`), so verify compares the
        # loaded `path` with it and reads no private resolver.
        var expected = PythonObject(None)
        if key.startswith("mojolearn._host."):
            expected = _host_expected_path(context, host_dir, last)
        var module = py_get(modules, keys[i])
        # Exact module type only: a CPU-only `_HostBinding`, a tier set or a
        # refusing stub would load or refuse on attribute access. The
        # instance dictionary is read without class or instance hooks.
        if module is None or py_type(module) is not context["module_type"]:
            continue
        var namespace = py_fields(module)
        var file = py_get(namespace, PythonObject("__file__"))
        if file is None:
            # An ExtensionFileLoader module may omit __file__: its spec origin.
            var spec = py_get(namespace, PythonObject("__spec__"))
            if spec is not None:
                file = py_get(py_fields(spec), PythonObject("origin"))
        if file is None or py_type(file) is not text_type or not isfile(String(file)):
            # A loaded binding module with no resolvable file is reported
            # with path None; verify refuses it (never skipped silently).
            var unresolved = _binding_row_unresolved()
            py_setitem(unresolved, PythonObject("expected_path"), expected)
            py_setitem(rows, keys[i], unresolved)
            continue
        var file_path = String(file)
        # The binding's own read-backs: `<suffix>_vendor()` and
        # `<suffix>_numeric_mode()` (`mojolearn_vendor` on `_mojolearn`).
        var suffix: String
        if last == "_mojolearn":
            suffix = "mojolearn"
        else:
            suffix = String(py_call(text_type.removeprefix, PythonObject(last), PythonObject("_mojolearn_")))
        var vendor = PythonObject(None)
        var vendor_entry = py_get(namespace, PythonObject(suffix + "_vendor"))
        if vendor_entry is not None and py_type(vendor_entry) is context["native_function_type"]:
            var read = _read(vendor_entry, context, True)
            if read is not None:
                vendor = _text(read, "compiled vendor")
        var compiled = PythonObject(None)
        var mode_fn = py_get(namespace, PythonObject(suffix + "_numeric_mode"))
        if mode_fn is not None and py_type(mode_fn) is context["native_function_type"]:
            var code = _read(mode_fn, context, True)
            if code is not None:
                compiled = py_get(context["mode_codes"], code, PythonObject("unknown"))
        var tier = PythonObject(None)
        if len(parts) == 4 and String(parts[1]) == "_sets":
            tier = PythonObject(String(parts[2]))
        # A sabotage build says so through `<suffix>_sabotage()` (host bindings).
        var sabotage = PythonObject(None)
        var sabotage_fn = py_get(namespace, PythonObject(suffix + "_sabotage"))
        if sabotage_fn is not None and py_type(sabotage_fn) is context["native_function_type"]:
            var said = _read(sabotage_fn, context, True)
            if said is not None:
                sabotage = PythonObject(True) if Bool(py=said) else PythonObject(False)
        var row = _binding_row(file_path, True, _identity(hashes, file_path), vendor, tier)
        py_setitem(row, PythonObject("compiled_mode"), compiled)
        py_setitem(row, PythonObject("sabotage"), sabotage)
        py_setitem(row, PythonObject("expected_path"), expected)
        # A host binding's kernel-matrix column read-backs: `<suffix>_column()`
        # and, where it exports one, `<suffix>_detected_column()` (the column
        # the accelerator predicates detected). The CPU column's host record
        # reads them here (lane verify-product-entries, 2026-11-16).
        if key.startswith("mojolearn._host."):
            py_setitem(row, PythonObject("column"), _host_column(namespace, suffix + "_column", context))
            py_setitem(row, PythonObject("detected_column"), _host_column(namespace, suffix + "_detected_column", context))
        py_setitem(rows, keys[i], row)
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
    # `code_kind`: which code the GPU set carries, "ptx" exactly when the PTX
    # set was selected at import (`_backend._BASELINE_SELECTION`, the receipt
    # `_ptx_base` writes), else "native" (native GPU code, or the CPU
    # bindings). One meaning per field: `code_mode` is the compiled numeric mode.
    var ptx = py_get(fields, "_BASELINE_SELECTION")
    py_setitem(report, PythonObject("code_kind"), PythonObject("native") if ptx is None else PythonObject("ptx"))
    py_setitem(report, PythonObject("cpu_only"), PythonObject(False) if fields["_CPU_ONLY"] is None else PythonObject(True))
    py_setitem(report, PythonObject("gpu_arch"), _gpu_arch(fields, False))
    py_setitem(report, PythonObject("gpu_arch_how"), _gpu_arch(fields, True))
    var host_dir = _host_dir(context)
    var families = _host_families(context, host_dir)
    py_setitem(report, PythonObject("host_dir"), PythonObject(host_dir))
    py_setitem(report, PythonObject("host_families_built"), families)
    # `host_paths`: every built host family -> the file it loads or would
    # load (the `expected_path` its binding row carries once loaded), so the
    # CPU column records each family's file bytes without loading it.
    var host_paths = Python.dict()
    for i in range(len(families)):  # driver-loop: over the built host families, once per report
        py_setitem(host_paths, families[i], _host_expected_path(context, host_dir, String(families[i])))
    py_setitem(report, PythonObject("host_paths"), host_paths)
    # How the vendor was decided (the words `_backend.vendor_how()` uses).
    if fields["_CPU_ONLY"] is not None:
        var built = String("none")
        if len(families) > 0:
            built = String(py_call(PythonObject(", ").join, families))
        py_setitem(report, PythonObject("vendor_how"), PythonObject("no GPU binary set loaded; the CPU bindings under mojolearn/host/ only (DEVIATION 2615; built: " + built + ")"))
    else:
        py_setitem(report, PythonObject("vendor_how"), _text(fields["_VENDOR_HOW"], "vendor decision", True))
    py_setitem(report, PythonObject("bindings"), _bindings(fields, context, host_dir))
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
    if operation == "connect_mode_answer":
        # Import only: hand the selector's one decision to every door's mode
        # admission (core/mode_answer.mojo).
        # The families answered are the compiled constant
        # MODE_FAMILIES_ANSWERED; a single int argument is the built-install
        # test's trial mask (never passed by the package).
        var trial_families = -1
        if len(args) == 1:
            trial_families = Int(py=args[0])
        connect_mode_answer_words(py_get(fields, "_FAST_ANSWER") is not None, PythonObject("identical"), trial_families)
        return PythonObject(None)
    var mode = _mode(fields)
    if operation == "set_default_mode":
        if len(args) != 1:
            raise Error("TypeError: set_numeric_mode requires one mode argument")
        var previous = _mode_name(fields, mode)
        var requested = previous if args[0] is None else _text(args[0], "numeric mode")
        var text_type = py_type(PythonObject(""))
        requested = py_call(text_type.lower, py_call(text_type.strip, requested))
        if String(requested) != "identical" and String(requested) != "fast":
            raise Error("ValueError: mojolearn: numeric_mode=" + String(py_call(text_type.__repr__, requested)) + "; it must be 'fast', or 'identical' (the default)")
        # THE NAME `fast` WHERE NO FAST SET IS BUILT (lane fx-fast-selector):
        # off Apple the identical set answers it. The selector decided that
        # once at import (`_backend._FAST_ANSWER`: 'identical', or None where
        # `fast` names its own set); here the set becomes the default every
        # door reads and the name is kept for numeric_mode() only.
        var answered = requested
        var name = PythonObject(None)
        if String(requested) == "fast":
            var answer = py_get(fields, "_FAST_ANSWER")
            if answer is not None:
                answered = _text(answer, "fast answer")
                name = requested
        # A CPU-ONLY INSTALL RUNS `identical` (the host column of the
        # verification): the selector installed the host bindings under the
        # canonical names and there is no tier directory, so the import-time
        # plan for either tier holds the no-GPU refusal. Setting the mode the
        # install already runs is accepted; 'fast' is Apple only and refused
        # by name (cpu-surface-defects, 2026-11-16).
        if fields["_CPU_ONLY"] is not None:
            if String(answered) != "identical":
                raise Error("ValueError: mojolearn: numeric_mode='fast' is available on Apple GPUs only; this CPU-only install runs 'identical'. Pass 'identical' or drop the call.")
            py_setitem(fields, "_DEFAULT_MODE", answered)
            py_setitem(fields, "_DEFAULT_NAME", name)
            return previous
        _refuse(context["plans"][answered]["error"])
        py_setitem(fields, "_DEFAULT_MODE", answered)
        py_setitem(fields, "_DEFAULT_NAME", name)
        return previous
    if operation == "numeric_mode":
        return _mode_name(fields, _numeric_mode(fields, context, mode))
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
