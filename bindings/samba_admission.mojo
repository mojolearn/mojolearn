# SPDX-License-Identifier: Apache-2.0
"""Native admission before Samba can encounter a Python-defined protocol."""
from std.python import Python, PythonObject
from std.ffi import _Global
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.p3_guard_records import guard_row_admitted, guard_row_admit
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type


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
            # A model record passed to an operation is admitted in full once;
            # later operations compare its type tree with the stored shadow
            # and admit in full again only when it changed (see
            # SambaAdmissionShadows below).
            var remembered = depth <= 1
            if remembered and _samba_admitted_before(value, admission, bi):
                return
            _samba_admit_record(value, cls, record, admission, bi, depth)
            if remembered:
                _samba_remember(value, admission, bi)
            return
    raise Error("TypeError: Samba requires native input types; Python-defined protocols and subclasses are not supported")


def _samba_admit_record(value: PythonObject, cls: PythonObject, record: PythonObject, admission: PythonObject, bi: PythonObject, depth: Int) raises:
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


# ---------------------------------------------------------------- admitted once

# Admission of a model record walks every object the record reaches, so doing
# it on every operation cost O(record size) per step. A record is now admitted
# in full once; the walk's result is kept as a SHADOW: the record's type tree
# (each object's exact type, each container's size, each mapping's and record
# dictionary's key objects in order, the default factory of a defaultdict and
# the owner type of a memoryview). It holds no values and no buffers, so it
# keeps nothing of the model alive. Admission depends only on that tree, so a
# record whose live tree equals an admitted shadow is admitted; any write that
# changes a type, a size or a key (a replaced field, an added key, a resized
# registry, a different memoryview owner) makes the comparison fail and the
# full admission runs again, with its own errors, then the shadow is renewed.
# Writes that keep the tree (the step counter, the learning rate, a new Array
# of the same layout) need no re-admission. The table lives here, keyed by
# object id, never in the model's __dict__; a recycled id is still compared
# against the live tree, so it can only fall back to the full admission.
comptime _SHADOW_LEAF = 0
comptime _SHADOW_CLASS = 1
comptime _SHADOW_SEQUENCE = 2
comptime _SHADOW_MAPPING = 3
comptime _SHADOW_ARRAY = 4
comptime _SHADOW_VIEW = 5
comptime _SHADOW_RECORD = 6
# Bound on remembered records; models beyond it are admitted in full again.
comptime _SHADOW_LIMIT = 64


struct SambaAdmissionShadows(Defaultable, Movable):
    var table: PythonObject

    def __init__(out self):
        self.table = PythonObject(None)


# Named globals may be shared by every Mojo extension in the process: bump the
# name's version whenever the shadow layout above changes.
comptime SAMBA_ADMISSION_SHADOWS = _Global[StorageType=SambaAdmissionShadows,
    name="MojolearnSambaAdmissionShadowsV1", init_fn=SambaAdmissionShadows.__init__]


def _samba_shadow_table() raises -> PythonObject:
    var g = SAMBA_ADMISSION_SHADOWS.get_or_create_ptr()
    if g[].table is None:
        g[].table = Python.dict()
    return g[].table


def _samba_shadow(value: PythonObject, admission: PythonObject, bi: PythonObject) raises -> PythonObject:
    # Built only after `value` passed the full admission, in that walk's order.
    var cls = py_type(value)
    if _samba_exact(cls, admission["scalar_types"]) or _samba_exact(cls, admission["opaque_types"]):
        return Python.list(PythonObject(_SHADOW_LEAF), cls)
    if cls is bi.list or cls is bi.tuple:
        var node = Python.list(PythonObject(_SHADOW_SEQUENCE), cls, PythonObject(len(value)))
        for i in range(len(value)):  # driver-loop: over admitted container elements
            _ = py_call(node.append, _samba_shadow(value[i], admission, bi))
        return node
    if _samba_exact(cls, admission["mapping_types"]):
        var pairs = py_call(bi.list, value.items())
        var keys = Python.list()
        var children = Python.list()
        for i in range(len(pairs)):  # driver-loop: over admitted mapping entries
            _ = py_call(keys.append, pairs[i][0])
            _ = py_call(children.append, _samba_shadow(pairs[i][1], admission, bi))
        var factory = value.default_factory if cls is admission["defaultdict_type"] else PythonObject(None)
        return Python.list(PythonObject(_SHADOW_MAPPING), cls, keys, children, factory)
    if cls is admission["array_class"]:
        var getters = admission["array_getters"]
        var names = admission["array_metadata"]
        var children = Python.list()
        for i in range(len(names)):  # driver-loop: over native Array slot metadata
            _ = py_call(children.append, _samba_shadow(py_call(getters[names[i]], value), admission, bi))
        var view = py_call(getters["_mv"], value)
        return Python.list(PythonObject(_SHADOW_ARRAY), children, _samba_shadow(view, admission, bi))
    if cls is bi.memoryview:
        var owner = value.obj
        if owner is None:
            return Python.list(PythonObject(_SHADOW_VIEW), PythonObject(False), PythonObject(None))
        return Python.list(PythonObject(_SHADOW_VIEW), PythonObject(True), py_type(owner))
    if _samba_exact(cls, admission["buffer_types"]) or _samba_exact(cls, admission["path_types"]) or _samba_numpy(cls, admission, bi):
        return Python.list(PythonObject(_SHADOW_LEAF), cls)
    var records = admission["records"]
    for i in range(len(records)):  # driver-loop: over connected model/config/state class identities
        var record = records[i]
        if value is record[0]:
            return Python.list(PythonObject(_SHADOW_CLASS), value)
        if cls is record[0]:
            var fields = py_call(record[1], value)
            var keys = py_call(bi.list, fields)
            var present = Python.list()
            var names = record[2]
            for j in range(len(names)):  # driver-loop: over stored owner metadata and buffer references
                var name = names[j]
                if py_contains(fields, name):
                    _ = py_call(present.append, Python.tuple(name, _samba_shadow(fields[name], admission, bi)))
            return Python.list(PythonObject(_SHADOW_RECORD), cls, record[1], keys, present)
    raise Error("TypeError: Samba requires native input types; Python-defined protocols and subclasses are not supported")


def _samba_shadow_same(value: PythonObject, shadow: PythonObject, admission: PythonObject, bi: PythonObject) raises -> Bool:
    # True only when `value`'s live type tree equals the admitted shadow.
    # Reads exactly what the admission read, without its type-table scans.
    var kind = Int(py=shadow[0])
    if kind == _SHADOW_LEAF:
        return py_type(value) is shadow[1]
    if kind == _SHADOW_CLASS:
        return value is shadow[1]
    if kind == _SHADOW_SEQUENCE:
        if py_type(value) is not shadow[1]:
            return False
        var n = Int(py=shadow[2])
        if len(value) != n:
            return False
        for i in range(n):  # driver-loop: over admitted container elements
            if not _samba_shadow_same(value[i], shadow[3 + i], admission, bi):
                return False
        return True
    if kind == _SHADOW_MAPPING:
        if py_type(value) is not shadow[1]:
            return False
        var keys = shadow[2]
        var children = shadow[3]
        var pairs = py_call(bi.list, value.items())
        if len(pairs) != len(keys):
            return False
        for i in range(len(pairs)):  # driver-loop: over admitted mapping entries
            if pairs[i][0] is not keys[i]:
                return False
            if not _samba_shadow_same(pairs[i][1], children[i], admission, bi):
                return False
        if shadow[1] is admission["defaultdict_type"] and value.default_factory is not shadow[4]:
            return False
        return True
    if kind == _SHADOW_ARRAY:
        if py_type(value) is not admission["array_class"]:
            return False
        var getters = admission["array_getters"]
        var names = admission["array_metadata"]
        var children = shadow[1]
        for i in range(len(names)):  # driver-loop: over native Array slot metadata
            if not _samba_shadow_same(py_call(getters[names[i]], value), children[i], admission, bi):
                return False
        return _samba_shadow_same(py_call(getters["_mv"], value), shadow[2], admission, bi)
    if kind == _SHADOW_VIEW:
        if py_type(value) is not bi.memoryview:
            return False
        var owner = value.obj
        if not Bool(py=shadow[1]):
            return owner is None
        return owner is not None and py_type(owner) is shadow[2]
    if kind == _SHADOW_RECORD:
        if py_type(value) is not shadow[1]:
            return False
        var fields = py_call(shadow[2], value)
        if py_type(fields) is not bi.dict:
            return False
        var keys = shadow[3]
        var live = py_call(bi.list, fields)
        if len(live) != len(keys):
            return False
        for j in range(len(live)):  # driver-loop: over native record keys, by identity
            if live[j] is not keys[j]:
                return False
        var present = shadow[4]
        for j in range(len(present)):  # driver-loop: over the record's admitted fields
            if not _samba_shadow_same(fields[present[j][0]], present[j][1], admission, bi):
                return False
        return True
    return False


def _samba_admitted_before(value: PythonObject, admission: PythonObject, bi: PythonObject) raises -> Bool:
    var entry = py_get(_samba_shadow_table(), py_call(bi.id, value))
    if entry is None or entry[0] is not admission:
        return False
    return _samba_shadow_same(value, entry[1], admission, bi)


def _samba_remember(value: PythonObject, admission: PythonObject, bi: PythonObject) raises:
    var table = _samba_shadow_table()
    if len(table) >= _SHADOW_LIMIT:
        _ = py_call(table.clear)
    py_setitem(table, py_call(bi.id, value), Python.tuple(admission, _samba_shadow(value, admission, bi)))


# Class-watcher slot of a passed Samba layout guard (one record per class).
comptime _SAMBA_GUARD_SLOT = "p3_samba_layout_guard_v1_"


def samba_admit(operation: String, values: PythonObject, context: PythonObject) raises:
    # One checked builtin connection for this GIL-held recursive admission.
    var bi = native_loaded_module("builtins")
    var admission = context["admission"]
    var guards = admission["guards"]
    var mods = NativeModules()
    for i in range(len(guards)):  # driver-loop: over registered runtime record/Array layouts
        # A guard depends only on its class rows (own namespace and MRO). Once
        # it passed, the class-watcher record (bindings/native_class_admission.mojo)
        # stays valid until the class or an ancestor is modified, which bumps
        # its epoch; Python 3.10/3.11 have no watcher and keep the full check.
        var guarded = guards[i][2]
        if guard_row_admitted(mods, _SAMBA_GUARD_SLOT, i, guarded, guards[i]):
            continue
        if py_call(admission["type_mro"], guards[i][2]) is not guards[i][3]:
            raise Error("TypeError: Samba native owner inheritance has changed")
        var namespace = guards[i][0]
        var pairs = guards[i][1]
        for j in range(len(pairs)):  # driver-loop: over descriptors that native execution reads or writes
            if py_get(namespace, pairs[j][0]) is not pairs[j][1]:
                raise Error("TypeError: Samba native owner layout has a replaced Python descriptor or attribute hook")
        guard_row_admit(mods, _SAMBA_GUARD_SLOT, i, guarded, guards[i])
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
