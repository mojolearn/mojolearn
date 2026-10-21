# SPDX-License-Identifier: Apache-2.0
"""Native admission before Samba can encounter a Python-defined protocol."""
from std.python import Python, PythonObject


def _samba_exact(cls: PythonObject, types: PythonObject) raises -> Bool:
    for i in range(len(types)):  # driver-loop: over connected native ABI type identities
        if cls is types[i]:
            return True
    return False


def _samba_numpy(cls: PythonObject, admission: PythonObject) raises -> Bool:
    var bi = Python.import_module("builtins")
    var module = admission["modules"].get("numpy")
    if module is None or bi.type(module) is not admission["module_type"]:
        return False
    # NumPy's admitted ndarray/scalar types are static extension types.
    # A Python subclass substituted into module metadata cannot pass by name.
    var flags = Int(py=admission["type_flags"](cls))
    if (flags & 512) != 0:
        return False
    var fields = bi.object.__getattribute__(module, "__dict__")
    var names = admission["numpy_names"]
    for i in range(len(names)):  # driver-loop: over supported native NumPy type identities
        if cls is fields.get(names[i]):
            return True
    return False


def samba_admit_schedule(value: PythonObject, admission: PythonObject) raises:
    if value is not None and not _samba_exact(Python.import_module("builtins").type(value), admission["schedule_types"]):
        raise Error("NotImplementedError: Samba requires ConstantLR, WarmupLinearLR or WarmupCosineLR; Python schedule callbacks cannot execute in the native runtime")


def samba_admit_value(value: PythonObject, admission: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Samba input nesting exceeds the native metadata limit")
    var bi = Python.import_module("builtins")
    var cls = bi.type(value)
    if _samba_exact(cls, admission["scalar_types"]) or _samba_exact(cls, admission["opaque_types"]):
        return
    if cls is bi.list or cls is bi.tuple:
        for i in range(len(value)):  # driver-loop: over admitted native container elements
            samba_admit_value(value[i], admission, depth + 1)
        return
    if _samba_exact(cls, admission["mapping_types"]):
        var pairs = bi.list(value.items())
        for i in range(len(pairs)):  # driver-loop: over native mapping entries before lookup
            samba_admit_value(pairs[i][0], admission, depth + 1)
            samba_admit_value(pairs[i][1], admission, depth + 1)
        return
    if cls is admission["array_class"]:
        var getters = admission["array_getters"]
        var names = admission["array_metadata"]
        for i in range(len(names)):  # driver-loop: over native Array slot metadata
            samba_admit_value(getters[names[i]](value), admission, depth + 1)
        var view = getters["_mv"](value)
        if bi.type(view) is not bi.memoryview:
            raise Error("TypeError: Samba Array storage requires a native memoryview")
        samba_admit_value(view, admission, depth + 1)
        return
    if cls is bi.memoryview:
        var owner = value.obj
        if owner is not None and not _samba_exact(bi.type(owner), admission["buffer_types"]) and not _samba_numpy(bi.type(owner), admission):
            raise Error("TypeError: Samba requires a memoryview over a native buffer exporter")
        return
    if _samba_exact(cls, admission["buffer_types"]) or _samba_exact(cls, admission["path_types"]) or _samba_numpy(cls, admission):
        return
    var records = admission["records"]
    for i in range(len(records)):  # driver-loop: over connected model/config/state class identities
        var record = records[i]
        if value is record[0]:
            return
        if cls is record[0]:
            var fields = record[1](value)
            if bi.type(fields) is not bi.dict:
                raise Error("TypeError: Samba owner metadata requires its native dictionary")
            var names = record[2]
            var keys = bi.list(fields)
            for j in range(len(keys)):  # driver-loop: over native record keys before hash lookup
                if bi.type(keys[j]) is not bi.str:
                    raise Error("TypeError: Samba owner metadata requires native string keys")
            for j in range(len(names)):  # driver-loop: over stored owner metadata and buffer references
                var name = names[j]
                if Bool(py=fields.__contains__(name)):
                    var item = fields[name]
                    if String(name) == "lr_schedule":
                        samba_admit_schedule(item, admission)
                    samba_admit_value(item, admission, depth + 1)
            return
    raise Error("TypeError: Samba requires native input types; Python-defined protocols and subclasses are not supported")


def samba_admit(operation: String, values: PythonObject, context: PythonObject) raises:
    var admission = context["admission"]
    var guards = admission["guards"]
    for i in range(len(guards)):  # driver-loop: over registered runtime record/Array layouts
        if admission["type_mro"](guards[i][2]) is not guards[i][3]:
            raise Error("TypeError: Samba native owner inheritance has changed")
        var namespace = guards[i][0]
        var pairs = guards[i][1]
        for j in range(len(pairs)):  # driver-loop: over descriptors that native execution reads or writes
            if namespace.get(pairs[j][0]) is not pairs[j][1]:
                raise Error("TypeError: Samba native owner layout has a replaced Python descriptor or attribute hook")
    if operation == "construct":
        # The public wrapper creates exact tuples; check this before indexing
        # so a direct C-boundary call cannot execute an arbitrary __getitem__.
        var bi = Python.import_module("builtins")
        if bi.type(values) is not bi.tuple or len(values) != 2 or bi.type(values[1]) is not bi.tuple or len(values[1]) != 11:
            raise Error("TypeError: Samba constructor requires its native argument tuple")
        samba_admit_schedule(values[1][7], admission)
    samba_admit_value(values, admission)
