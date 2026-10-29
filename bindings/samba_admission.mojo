# SPDX-License-Identifier: Apache-2.0
"""Native admission before Samba can encounter a Python-defined protocol."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_contains, py_get, py_type


def _samba_exact(cls: PythonObject, types: PythonObject) raises -> Bool:
    for i in range(len(types)):  # driver-loop: over connected native ABI type identities
        if cls is types[i]:
            return True
    return False


def _samba_numpy(cls: PythonObject, admission: PythonObject, bi: PythonObject) raises -> Bool:
    var module = py_get(admission["modules"], "numpy")
    if module is None or py_type(module) is not admission["module_type"]:
        return False
    # NumPy's admitted ndarray/scalar types are static extension types.
    # A Python subclass substituted into module metadata cannot pass by name.
    var flags = Int(py=py_call(admission["type_flags"], cls))
    if (flags & 512) != 0:
        return False
    var fields = py_call(bi.object.__getattribute__, module, "__dict__")
    var names = admission["numpy_names"]
    for i in range(len(names)):  # driver-loop: over supported native NumPy type identities
        if cls is py_get(fields, names[i]):
            return True
    return False


def samba_admit_schedule(value: PythonObject, admission: PythonObject) raises:
    if value is not None and not _samba_exact(py_type(value), admission["schedule_types"]):
        raise Error("NotImplementedError: Samba requires ConstantLR, WarmupLinearLR or WarmupCosineLR; Python schedule callbacks cannot execute in the native runtime")


def _samba_mapping_key(value: PythonObject, admission: PythonObject, bi: PythonObject, depth: Int) raises:
    if depth > 64:
        raise Error("TypeError: Samba mapping keys exceed the native metadata limit")
    var cls = py_type(value)
    if _samba_exact(cls, admission["scalar_types"]) or _samba_numpy(cls, admission, bi):
        return
    if cls is bi.tuple:
        for i in range(len(value)):  # driver-loop: over native tuple key elements before hashing
            _samba_mapping_key(value[i], admission, bi, depth + 1)
        return
    raise Error("TypeError: Samba mapping keys require native scalar types without Python hash or equality hooks")


def _samba_metadata_with_types(value: PythonObject, admission: PythonObject, bi: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Samba scalar metadata exceeds the native nesting limit")
    var cls = py_type(value)
    if _samba_exact(cls, admission["scalar_types"]):
        return
    if _samba_numpy(cls, admission, bi):
        var module = admission["modules"]["numpy"]
        if cls is not py_call(bi.object.__getattribute__, module, "__dict__")["ndarray"]:
            return
    if cls is bi.list or cls is bi.tuple:
        for i in range(len(value)):  # driver-loop: over scalar configuration and shape metadata
            _samba_metadata_with_types(value[i], admission, bi, depth + 1)
        return
    if _samba_exact(cls, admission["mapping_types"]):
        var pairs = py_call(bi.list, value.items())
        for i in range(len(pairs)):  # driver-loop: over already admitted metadata mapping entries
            _samba_mapping_key(pairs[i][0], admission, bi, depth + 1)
            _samba_metadata_with_types(pairs[i][1], admission, bi, depth + 1)
        return
    raise Error("TypeError: Samba scalar and shape metadata cannot use array or object conversion protocols")


def _samba_admit_value_with_types(value: PythonObject, admission: PythonObject, bi: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Samba input nesting exceeds the native metadata limit")
    var cls = py_type(value)
    if _samba_exact(cls, admission["scalar_types"]) or _samba_exact(cls, admission["opaque_types"]):
        return
    if cls is bi.list or cls is bi.tuple:
        for i in range(len(value)):  # driver-loop: over admitted native container elements
            _samba_admit_value_with_types(value[i], admission, bi, depth + 1)
        return
    if _samba_exact(cls, admission["mapping_types"]):
        if cls is admission["defaultdict_type"]:
            var factory = value.default_factory
            var numpy_factory = py_type(factory) is bi.type and _samba_numpy(factory, admission, bi)
            if factory is not None and not _samba_exact(factory, admission["native_factories"]) and not numpy_factory:
                raise Error("TypeError: Samba mapping defaults require native builtin constructors; Python callbacks cannot execute")
        var pairs = py_call(bi.list, value.items())
        for i in range(len(pairs)):  # driver-loop: over native mapping entries before lookup
            _samba_mapping_key(pairs[i][0], admission, bi, depth + 1)
            _samba_admit_value_with_types(pairs[i][1], admission, bi, depth + 1)
        return
    if cls is admission["array_class"]:
        var getters = admission["array_getters"]
        var names = admission["array_metadata"]
        for i in range(len(names)):  # driver-loop: over native Array slot metadata
            _samba_metadata_with_types(py_call(getters[names[i]], value), admission, bi, depth + 1)
        var view = py_call(getters["_mv"], value)
        if py_type(view) is not bi.memoryview:
            raise Error("TypeError: Samba Array storage requires a native memoryview")
        _samba_admit_value_with_types(view, admission, bi, depth + 1)
        return
    if cls is bi.memoryview:
        var owner = value.obj
        if owner is not None and not _samba_exact(py_type(owner), admission["buffer_types"]) and not _samba_numpy(py_type(owner), admission, bi):
            raise Error("TypeError: Samba requires a memoryview over a native buffer exporter")
        return
    if _samba_exact(cls, admission["buffer_types"]) or _samba_exact(cls, admission["path_types"]) or _samba_numpy(cls, admission, bi):
        return
    var records = admission["records"]
    for i in range(len(records)):  # driver-loop: over connected model/config/state class identities
        var record = records[i]
        if value is record[0]:
            return
        if cls is record[0]:
            var fields = py_call(record[1], value)
            if py_type(fields) is not bi.dict:
                raise Error("TypeError: Samba owner metadata requires its native dictionary")
            var names = record[2]
            var keys = py_call(bi.list, fields)
            for j in range(len(keys)):  # driver-loop: over native record keys before hash lookup
                if py_type(keys[j]) is not bi.str:
                    raise Error("TypeError: Samba owner metadata requires native string keys")
            for j in range(len(names)):  # driver-loop: over stored owner metadata and buffer references
                var name = names[j]
                if py_contains(fields, name):
                    var item = fields[name]
                    var key = String(name)
                    if key == "lr_schedule":
                        samba_admit_schedule(item, admission)
                    elif py_contains(admission["metadata_fields"], name):
                        _samba_metadata_with_types(item, admission, bi, depth + 1)
                    elif key == "_runtime_lock":
                        if item is not None and not _samba_exact(py_type(item), admission["opaque_types"]):
                            raise Error("TypeError: Samba runtime locks require native lock owners")
                    elif key == "layers" or key == "params" or key == "_blocks" or key == "_wopt" or (key == "offsets" and cls is not admission["optimizer_class"]):
                        if py_type(item) is not bi.list and py_type(item) is not bi.tuple:
                            raise Error("TypeError: Samba owner sequences require native lists or tuples")
                        if key == "offsets" or (key == "layers" and cls is admission["config_class"]):
                            _samba_metadata_with_types(item, admission, bi, depth + 1)
                    elif key == "_w" or key == "arrays":
                        if py_type(item) is not bi.list and py_type(item) is not bi.tuple and not _samba_exact(py_type(item), admission["mapping_types"]):
                            raise Error("TypeError: Samba parameter registries require native containers")
                    _samba_admit_value_with_types(item, admission, bi, depth + 1)
            return
    raise Error("TypeError: Samba requires native input types; Python-defined protocols and subclasses are not supported")


def samba_admit(operation: String, values: PythonObject, context: PythonObject) raises:
    # One checked builtin connection for this GIL-held recursive admission.
    var bi = native_loaded_module("builtins")
    var admission = context["admission"]
    var guards = admission["guards"]
    for i in range(len(guards)):  # driver-loop: over registered runtime record/Array layouts
        if py_call(admission["type_mro"], guards[i][2]) is not guards[i][3]:
            raise Error("TypeError: Samba native owner inheritance has changed")
        var namespace = guards[i][0]
        var pairs = guards[i][1]
        for j in range(len(pairs)):  # driver-loop: over descriptors that native execution reads or writes
            if py_get(namespace, pairs[j][0]) is not pairs[j][1]:
                raise Error("TypeError: Samba native owner layout has a replaced Python descriptor or attribute hook")
    if operation == "construct":
        # The public wrapper creates exact tuples; check this before indexing
        # so a direct C-boundary call cannot execute an arbitrary __getitem__.
        if py_type(values) is not bi.tuple or len(values) != 2 or py_type(values[1]) is not bi.tuple or len(values[1]) != 11:
            raise Error("TypeError: Samba constructor requires its native argument tuple")
        samba_admit_schedule(values[1][7], admission)
    _samba_admit_value_with_types(values, admission, bi)
    # Type identity alone is insufficient: Array is an admitted buffer, but
    # bool(Array), list(Array), or scalar conversion is a Python operation.
    if operation == "initialize" or operation == "from_dict":
        _samba_metadata_with_types(values[1], admission, bi)
    elif operation == "construct":
        for i in [3, 4, 5, 6, 8, 9, 10]:  # driver-loop: over scalar constructor metadata slots
            _samba_metadata_with_types(values[1][i], admission, bi)
    elif operation == "allocate_state" or operation == "state_construct":
        _samba_metadata_with_types(values[1], admission, bi)
        _samba_metadata_with_types(values[2], admission, bi)
        if operation == "state_construct":
            if py_type(values[3]) is not bi.list and py_type(values[3]) is not bi.tuple:
                raise Error("TypeError: Samba state layers require a native list or tuple")
    elif operation == "generator_init" or operation == "generator_restore":
        _samba_metadata_with_types(values[1], admission, bi)
        if operation == "generator_init":
            _samba_metadata_with_types(values[2], admission, bi)
    elif operation == "snapshot":
        _samba_metadata_with_types(values[1], admission, bi)
    elif operation == "checkpoint_load":
        _samba_metadata_with_types(values[2], admission, bi)
    elif operation == "gradients":
        _samba_metadata_with_types(values[3], admission, bi)
        _samba_metadata_with_types(values[4], admission, bi)
    elif operation == "model_gradients":
        for i in [3, 4, 5]:  # driver-loop: over count, RNG stream and offset metadata
            _samba_metadata_with_types(values[i], admission, bi)


# Existing helper entry points also serve parallel-neural admission. Keep their
# signatures and checks; each independent walk acquires one native connection.
def _samba_metadata(value: PythonObject, admission: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Samba scalar metadata exceeds the native nesting limit")
    var bi = native_loaded_module("builtins")
    _samba_metadata_with_types(value, admission, bi, depth)


def samba_admit_value(value: PythonObject, admission: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Samba input nesting exceeds the native metadata limit")
    var bi = native_loaded_module("builtins")
    _samba_admit_value_with_types(value, admission, bi, depth)
