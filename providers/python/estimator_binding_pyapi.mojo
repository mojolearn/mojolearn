# SPDX-License-Identifier: Apache-2.0
# Shared import-connected binding admission for estimator frontends. The caller
# enters compiled code once; runtime selection reads current mode/registry data
# and never invokes Python _bind/load_set/_ModeSet.__getattr__ callbacks.
from std.ffi import _Global
from std.python import Python, PythonObject
from std.os import getenv
from bindings.native_loaded_module import native_loaded_module
from bindings.admitted_call import AdmittedCall
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.mode_metadata_native import _fields
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from providers.python.x_neighbors_admission_pyapi import xn_admit_checked


def estimator_binding(model: PythonObject, name: String, context: PythonObject, check_mode: Bool = True) raises -> PythonObject:
    # Read ordinary instance storage with exact dictionary/key admission.
    # A Python __getattribute__ or __dict__ property is never a mode reader.
    var requested = py_get(_fields(model), "numeric_mode") if model is not None else PythonObject(None)
    return estimator_binding_mode(requested, name, context, check_mode)


def estimator_binding(mut call: AdmittedCall, model: PythonObject, name: String, context: PythonObject,
                      check_mode: Bool = True) raises -> PythonObject:
    # The door's admission record: its admitted fields are the mode reader
    # and the resolved binding is kept on the record for the rest of the call.
    var requested = PythonObject(None)
    if model is not None:
        if not call.fields_admitted:
            call.admit_fields(_fields(model))
        requested = py_get(call.fields, "numeric_mode")
    return estimator_binding_mode(call, requested, name, context, check_mode)


def _selected_mode(bi: PythonObject, requested: PythonObject, context: PythonObject) raises -> String:
    # Callers with retained numeric_mode_ pass that already captured scalar.
    # Do not reread the current model setting: transforms must use the fitted
    # mode when their public contract stores one. Validate exact str before
    # any normalization protocol; None alone selects the current default.
    var mode = requested
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not bi.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var selected_mode = String(py=mode)
    if selected_mode != "identical" and selected_mode != "fast":
        # Only a non-canonical spelling pays the two str method calls.
        selected_mode = String(py=mode.strip().lower())
    if selected_mode != "identical" and selected_mode != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    return selected_mode


def estimator_binding_mode(mut call: AdmittedCall, requested: PythonObject, name: String, context: PythonObject,
                           check_mode: Bool = True) raises -> PythonObject:
    # One resolution per (binding name, mode, checked) per public call (rows
    # N219, N256, N267: a GP classifier resolved its binding four times). The
    # mode word is still validated on every call; the registry reads, the
    # live sys.modules slot and the compiled-mode check run once per call and
    # again after a GIL release window (AdmittedCall.released()).
    var bi = call.builtins()
    var selected_mode = _selected_mode(bi, requested, context)
    var found = call.find_binding(name, selected_mode, check_mode)
    if found >= 0:
        return call.binding_at(found)
    var module = _estimator_binding_selected(bi, call.module("sys"), selected_mode, name, context, check_mode)
    call.remember_binding(name, selected_mode, check_mode, module)
    return module


def estimator_binding_mode(requested: PythonObject, name: String, context: PythonObject,
                           check_mode: Bool = True) raises -> PythonObject:
    # Callers without an admission record.
    var bi = native_loaded_module("builtins")
    var selected_mode = _selected_mode(bi, requested, context)
    return _estimator_binding_selected(bi, native_loaded_module("sys"), selected_mode, name, context, check_mode)


def _estimator_binding_selected(bi: PythonObject, sys_module: PythonObject, selected_mode: String, name: String,
                                context: PythonObject, check_mode: Bool) raises -> PythonObject:
    var backend = context["backend_fields"]
    var record = context["family_binding_connections"][name][selected_mode]
    if not Bool(py=context["family_binding_offerings"][name][selected_mode]):
        raise Error(String(record[0]))
    if backend["_CPU_ONLY"] is not None:
        if py_type(record[0]) is bi.str:
            raise Error(String(record[0]))
        # Import setup resolves the lazy HostBinding facade to its exact native
        # module. Read the live module slot so replacement/removal is observed.
        var module = py_get(sys_module.modules, record[1])
        if module is None:
            raise Error("ImportError: native host binding was removed after import-time connection: " + String(record[1]))
        if py_type(module) is not py_type(bi):
            raise Error("TypeError: native host binding must be an exact extension module")
        return estimator_checked_mode(module, String(record[2]), name, selected_mode) if check_mode else module
    var sets = backend["_SETS"]
    if py_type(sets) is not bi.dict:
        raise Error("TypeError: native binding registry must be an exact dictionary")
    var selected = py_get(sets, selected_mode)
    if selected is None:
        if py_type(record[0]) is bi.str:
            raise Error(String(record[0]))
        raise Error("TypeError: native binding requires its import-connected mode set")
    if py_type(selected) is not context["family_mode_set_type"]:
        raise Error("TypeError: native binding requires its import-connected mode set")
    # Direct native dictionary descriptor, pinned at import. Replacing a
    # Python __getattribute__/__dict__ hook cannot run through this lookup.
    var fields = py_call(context["family_mode_set_fields"], selected, context["family_mode_set_type"])
    var modules = fields["_modules"]
    if py_type(modules) is not bi.dict:
        raise Error("TypeError: native binding modules must be an exact dictionary")
    var module = py_get(modules, name)
    if module is None:
        if py_type(record[0]) is bi.str:
            raise Error(String(record[0]))
        raise Error("ImportError: native binding was removed after import-time connection: " + name)
    if py_type(module) is not py_type(bi):
        raise Error("TypeError: native binding must be an exact extension module")
    # Getter names are immutable import metadata, but the function is read
    # from the live replacement module at every public operation.
    var reader = String(record[2]) if len(record) > 2 else String(context["family_binding_readers"][name])
    return estimator_checked_mode(module, reader, name, selected_mode) if check_mode else module


struct EstimatorModeWords(Defaultable, Movable):
    # Compiled numeric-mode words read once per (module, getter entry) for the
    # process (lane perf-provider-cache, 2026-11-06). A mode getter returns a
    # compile-time constant of its loaded image; calling it on every public
    # operation repeated the same native round trip. The getter is still read
    # from the module's live namespace every call (neural_descriptor_entry),
    # so a replaced entry or module is a different object, misses here and is
    # called again. Module and getter are held strongly (a loaded extension
    # is never unloaded), which keeps the identity compare sound.
    var modules: List[PythonObject]
    var getters: List[PythonObject]
    var words: List[Int]

    def __init__(out self):
        self.modules = List[PythonObject]()
        self.getters = List[PythonObject]()
        self.words = List[Int]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever EstimatorModeWords' layout changes.
comptime ESTIMATOR_MODE_WORDS = _Global[StorageType=EstimatorModeWords,
    name="MojolearnEstimatorModeWordsV1", init_fn=EstimatorModeWords.__init__]


def _estimator_mode_word(module: PythonObject, getter: PythonObject) raises -> Int:
    var g = ESTIMATOR_MODE_WORDS.get_or_create_ptr()
    for i in range(len(g[].modules)):  # small-loop(admitted mode getters): a few compiled extension modules per process, never data
        if g[].modules[i] is module and g[].getters[i] is getter:
            return g[].words[i]
    var got = Int(py=py_call(getter))
    if len(g[].modules) >= 128:
        g[].modules.clear()
        g[].getters.clear()
        g[].words.clear()
    g[].modules.append(module)
    g[].getters.append(getter)
    g[].words.append(got)
    return got


def estimator_checked_mode(module: PythonObject, reader: String, name: String,
                           requested: String) raises -> PythonObject:
    # XN identical estimators read mode/capabilities from their native table
    # once. Downstream xn_admit reuses this operation-local retained snapshot.
    # Existing fast admission and explicit legacy/module callers stay separate.
    if name == "_mojolearn_x_neighbors" and requested == "identical":
        return xn_admit_checked(module)
    # These families read mode from their admitted native table once per
    # operation. Preserve that single read instead of calling the legacy door
    # first; plain frontdoors below still perform the original checked readback.
    # Kernel identical mode always uses the versioned native descriptor. Its
    # owner validates descriptor mode once; fast retains its legacy readback.
    if name == "_mojolearn_kernel_methods" and requested == "identical":
        return module
    # Name tests first: the environment is read only for the two families
    # the switch can affect, not on every public operation.
    if ((name == "_mojolearn_svm" or name == "_mojolearn_x_neighbors")
            and getenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "0") == "1"):
        if requested != "identical":
            raise Error("native dispatch candidate requires identical mode")
        return module
    var getter = neural_descriptor_entry(module, reader, False)
    if getter is not None:
        var got = _estimator_mode_word(module, getter)
        var expected = 1 if requested == "identical" else 0
        if got != expected:
            var compiled = String("identical") if got == 1 else (String("fast") if got == 0 else String("unknown"))
            raise Error("RuntimeError: mojolearn: " + name + " was compiled for " + compiled
                        + ", but this call requested " + requested + "; rebuild the " + requested + " binding")
    return module


def estimator_arguments(mut call: AdmittedCall, model: PythonObject, name: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # As below, resolving through the door's admission record.
    if len(args) == 0 or args[0] is not None:
        return args
    var selected = estimator_binding(call, model, name, context)
    var values = Python.list(selected)
    for i in range(1, len(args)):  # driver-loop: over fixed public method arguments
        _ = py_call(values.append, args[i])
    return py_call(call.builtins().tuple, values)


def estimator_arguments(model: PythonObject, name: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # Preserve explicit native/internal module arguments. Thin public calls
    # reserve None for binding admission here, once after entering native code.
    if len(args) == 0 or args[0] is not None:
        return args
    var selected = estimator_binding(model, name, context)
    var values = Python.list(selected)
    for i in range(1, len(args)):  # driver-loop: over fixed public method arguments
        _ = py_call(values.append, args[i])
    return py_call(native_loaded_module("builtins").tuple, values)
