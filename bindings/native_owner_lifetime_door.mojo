# SPDX-License-Identifier: Apache-2.0
# Admit retained object ownership before a native operation releases it.
# Import-time metadata comes from _native_owner_lifetime.connect_owner_context;
# callers connect already-loaded NumPy before admission and preserve all old
# fields until every replaced owner passes. Identical retained owners need no
# release admission. This helper never mutates or frees the admitted value.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_loaded_module_door import NativeModules, native_namespace_string_keys, native_mapping_string_keys, native_loaded_module
from bindings.native_class_admission_door import class_record, class_admit, class_watch_ready, dict_keys_admitted, dict_keys_admit
from bindings.py_cpython_table import cpython_table
from bindings.native_resident_capsule_unpack import resident_capsule_owned
from bindings.native_owner_registry_unpack import NativeOwnerPins, registered_native_owner_pins
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.native_keys_unpack import native_keys
from bindings.py_balanced import py_item


comptime _ARRAY_CLASS_SLOT = "owner_release_array_class_v1"


def _records_death_callback(callback: PythonObject, context: PythonObject) raises -> Bool:
    """True for the death callback of bindings/py_model_records.mojo:
    an exact bound method (PyMethod_New) of the compiled `release` entry of
    its `_mojolearn_model_records_lifetime` module, bound to an int token.
    Each binding holds its own records table, so the entry is recognized by
    its exact types and its module name, not by identity. Releasing its
    referent runs only that native release; no Python callback runs. The
    reads below are C getset slots of exact built-in types; the classes come
    from the import-time owner context (_native_owner_lifetime)."""
    var nk = native_keys()
    var method_class = py_get(context, nk[].k_owner_method_class)
    var builtin_class = py_get(context, nk[].k_owner_builtin_function_class)
    if method_class is None or builtin_class is None or py_type(callback) is not method_class:
        return False
    var b = py_item(context, nk[].k_owner_native_types)
    var function = callback.__func__
    if py_type(function) is not builtin_class or py_type(callback.__self__) is not py_item(b, nk[].k_int):
        return False
    var module = function.__module__
    var name = function.__name__
    if py_type(module) is not py_item(b, nk[].k_str) or py_type(name) is not py_item(b, nk[].k_str):
        return False
    return String(py=module) == "_mojolearn_model_records_lifetime" and String(py=name) == "release"


def native_release_weakrefs(value: PythonObject, context: PythonObject) raises:
    var nk = native_keys()
    var refs = py_call(py_item(context, nk[].k_owner_getweakrefs),value)
    for i in range(len(refs)):  # driver-loop: over retained weakrefs before releasing any application callback
        if py_type(refs[i]) is not py_item(context, nk[].k_owner_weakref_class):
            raise Error("TypeError: native owner reinitialization cannot release weak-reference callbacks")
        var callback = py_call(py_item(context, nk[].k_owner_weakref_callback).__get__,refs[i])
        # Since lane ne1-e-records every stamped model carries the records'
        # native death callback; releasing such a model (a fitted member, a
        # kernel node, a learner) runs only that native release.
        if callback is not None and not _records_death_callback(callback, context):
            raise Error("TypeError: native owner reinitialization cannot release weak-reference callbacks")


def _released_dtype(value: PythonObject, context: PythonObject, depth: Int) raises -> Bool:
    var nk = native_keys()
    if py_get(context,nk[].k_ndarray) is None:
        return False
    var classes = py_get(context,nk[].k_owner_dtype_classes)
    if classes is None:
        classes = Python.list()
        var scalars = Python.list()
        var factory = native_numpy_dtype(context)
        for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: fixed immutable NumPy dtype classes, including distinct native long-long aliases
            var dtype = py_call(factory,code)
            _ = py_call(classes.append,py_type(dtype))
            if code != "O" and code != "V":
                _ = py_call(scalars.append,dtype.type)
        py_setitem(context,nk[].k_owner_dtype_classes,classes)
        py_setitem(context,nk[].k_owner_dtype_scalars,scalars)
    for i in range(len(classes)):  # driver-loop: over exact native dtype identities before inspecting their C descriptors
        if py_type(value) is classes[i]:
            # A structured/subarray dtype and its native metadata do not
            # imply a Python callback. Admit their actual retained owners,
            # including field titles and metadata values, without hashing
            # application-controlled dictionary keys.
            for mapping in [value.metadata, value.fields]:  # driver-loop: NumPy-owned dictionary proxies
                if mapping is not None:
                    var pairs = py_call(py_item(py_item(context, nk[].k_owner_native_types), nk[].k_list),mapping.items())
                    for j in range(len(pairs)):  # driver-loop: native dtype key/value ownership, no key lookups
                        native_release_admit(pairs[j][0],context,depth+1)
                        native_release_admit(pairs[j][1],context,depth+1)
            native_release_admit(value.names,context,depth+1)
            native_release_admit(value.subdtype,context,depth+1)
            return True
    return False


def _native_item(value: PythonObject, index: PythonObject) raises -> PythonObject:
    # Callers admit the exact ndarray (or its native flat iterator), dtype
    # and native integer/field-name index before this public CPython API.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr,index._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _released_object_elements(value: PythonObject, context: PythonObject, depth: Int) raises:
    # Follow field views without following their .base back to the array:
    # dtype ownership and the original buffer owner are checked separately.
    var dtype = value.dtype
    if dtype.fields is not None:
        var names = dtype.names
        for i in range(len(names)):  # driver-loop: structured fields retain object owners independently of native numeric bytes
            var field = _native_item(value,names[i])
            _released_object_elements(field,context,depth+1)
    elif Bool(py=dtype.hasobject):
        var flattened = value.flat
        for i in range(Int(py=value.size)):  # driver-loop: object-array indexing returns the retained object reference, without conversion
            native_release_admit(_native_item(flattened,PythonObject(i)),context,depth+1)


def _is_class_labels(cls: PythonObject, b: PythonObject, context: PythonObject) raises -> Bool:
    """`cls` is exactly the loaded mojolearn._labels.ClassLabels, still a
    direct list subclass whose class dictionary defines no finalizer, no
    slots and no __getattribute__ (read through the type's own descriptors,
    as the Array admission below does)."""
    var nk = native_keys()
    var modules = py_get(context, nk[].k_owner_sys_modules)
    if modules is None:
        return False
    var module = py_get(modules, "mojolearn._labels")
    if module is None:
        return False
    if cls is not py_get(module.__dict__, nk[].k_ClassLabels):
        return False
    var fields = py_call(py_item(py_item(b, nk[].k_type).__dict__, nk[].k___dict__).__get__, cls)
    var ancestry = py_call(py_item(py_item(b, nk[].k_type).__dict__, nk[].k___mro__).__get__, cls)
    if len(ancestry) != 3 or ancestry[1] is not py_item(b, nk[].k_list) or ancestry[2] is not py_item(b, nk[].k_object):
        return False
    if py_get(fields, nk[].k___del__) is not None or py_get(fields, nk[].k___getattribute__) is not None:
        return False
    var slots = py_get(fields, nk[].k___slots__)
    return slots is not None and py_type(slots) is py_item(b, nk[].k_tuple) and len(slots) == 0


comptime _PY_TPFLAGS_HAVE_GC = 1 << 14

# Only audited native registrars add field owners to the exact-identity
# registry. Their native inspectors enumerate retained Python pins, which
# follow the same recursive admission as ordinary fields below. Prepared-only
# caches are not estimator fields and are not admitted through this API.


def _native_owner_pins(cls: PythonObject, b: PythonObject) raises -> Optional[NativeOwnerPins]:
    # Identity is necessary, but a registered native type may not acquire
    # Python lifecycle/attribute hooks after registration. Preserve the
    # existing direct-object, no-dictionary/weakref/GC layout policy as well.
    var nk = native_keys()
    if py_type(cls) is not py_item(b, nk[].k_type):
        return None
    var pins = registered_native_owner_pins(cls)
    if not pins:
        return None
    var type_fields = py_item(b, nk[].k_type).__dict__
    var ancestry = py_call(py_item(type_fields, nk[].k___mro__).__get__, cls)
    if len(ancestry) != 2 or ancestry[1] is not py_item(b, nk[].k_object):
        return None
    var mapping = py_call(py_item(type_fields, nk[].k___dict__).__get__, cls)
    if not native_mapping_string_keys(mapping, py_item(b, nk[].k_str)):
        return None
    if (py_get(mapping, nk[].k___module__) is not None or py_get(mapping, nk[].k___del__) is not None
            or py_get(mapping, nk[].k___getattribute__) is not None):
        return None
    if Int(py=py_call(py_item(type_fields, nk[].k___dictoffset__).__get__, cls)) != 0:
        return None
    if Int(py=py_call(py_item(type_fields, nk[].k___weakrefoffset__).__get__, cls)) != 0:
        return None
    if (Int(py=py_call(py_item(type_fields, nk[].k___flags__).__get__, cls)) & _PY_TPFLAGS_HAVE_GC) != 0:
        return None
    return pins


def _owner_string_dictionary(fields: PythonObject, b: PythonObject) raises -> Bool:
    # Provider pins reach module/entry admission too. A foreign key in a
    # module namespace or sys.modules must not run equality during lookup.
    var nk = native_keys()
    if py_type(fields) is not py_item(b, nk[].k_dict):
        return False
    if not dict_keys_admitted(fields):
        if not native_namespace_string_keys(fields._obj_ptr, py_item(b, nk[].k_str)):
            return False
        dict_keys_admit(fields)
    return True


def _retained_module(value: PythonObject, cls: PythonObject, b: PythonObject, context: PythonObject) raises -> Bool:
    """`value` is an exact module that `sys.modules` still holds under its own
    `__name__` (the compiled binding a resident entry or provider records).
    Dropping the model's reference cannot finalize it. The name is read
    from the module's own dictionary (the ModuleType `__dict__` slot), never
    through a module `__getattr__`."""
    var nk = native_keys()
    var module_class = py_get(context, nk[].k_owner_module_class)
    var modules = py_get(context, nk[].k_owner_sys_modules)
    if module_class is None or modules is None or cls is not module_class:
        return False
    var namespace = py_call(py_item(context, nk[].k_owner_module_dict).__get__, value)
    if not _owner_string_dictionary(namespace, b) or not _owner_string_dictionary(modules, b):
        return False
    var name = py_get(namespace, nk[].k___name__)
    if name is None or py_type(name) is not py_item(b, nk[].k_str):
        return False
    return py_get(modules, name) is value


def _retained_entry(value: PythonObject, cls: PythonObject, b: PythonObject, context: PythonObject) raises -> Bool:
    """`value` is an exact compiled function bound to a module `sys.modules`
    still holds (a binding's entry recorded in a resident holder or entry
    table: a release, prepare or factory entry). Dropping a reference to it
    runs nothing. The bound module is read with PyCFunction_GetSelf."""
    var nk = native_keys()
    var builtin_class = py_get(context, nk[].k_owner_builtin_function_class)
    if builtin_class is None or cls is not builtin_class:
        return False
    var bound = cpython_table()[].PyCFunction_GetSelf(value._obj_ptr)
    if not bound:
        return False
    var module = PythonObject(from_borrowed=bound)
    return _retained_module(module, py_type(module), b, context)


def native_release_admit(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    # Reinitialization decrements the previous fields' references. Admit the
    # complete owners that could be destroyed before replacing any field.
    # Keeping an identical value is handled by the caller and drops no owner.
    var nk = native_keys()
    var b = py_item(context, nk[].k_owner_native_types)
    var cls = py_type(value)
    if value is None or cls is py_item(b, nk[].k_str) or cls is py_item(b, nk[].k_bytes) or cls is py_item(b, nk[].k_bytearray) or cls is py_item(b, nk[].k_bool) or cls is py_item(b, nk[].k_int) or cls is py_item(b, nk[].k_float) or cls is py_item(b, nk[].k_complex):
        return
    if depth > 64:
        raise Error("TypeError: native owner reinitialization owner nesting exceeds the native bound")
    native_release_weakrefs(value,context)
    if cls is py_item(b, nk[].k_tuple) or cls is py_item(b, nk[].k_list):
        for i in range(len(value)):  # driver-loop: old metadata containers may own application finalizers transitively
            native_release_admit(value[i],context,depth+1)
        return
    if _is_class_labels(cls, b, context):
        # A fitted classifier's `classes_` (mojolearn._labels.ClassLabels, an
        # exact list subclass with empty __slots__ and no finalizer): its
        # labels are read through the C list accessors, never its Python
        # __getitem__, and admitted as a list's elements are.
        var table = cpython_table()
        var n_classes = table[].PyList_Size(value._obj_ptr)
        for i in range(n_classes):  # driver-loop: over the fitted classes of the model, no element protocols
            native_release_admit(PythonObject(from_borrowed=table[].PyList_GetItem(value._obj_ptr, i)),context,depth+1)
        return
    if cls is py_item(b, nk[].k_dict):
        var pairs = py_call(py_item(b, nk[].k_list),value.items())
        for i in range(len(pairs)):  # driver-loop: inspect dictionary owners without key equality lookups
            native_release_admit(pairs[i][0],context,depth+1)
            native_release_admit(pairs[i][1],context,depth+1)
        return
    if cls is py_item(context, nk[].k_array_class):
        var slots = py_item(context, nk[].k_owner_array_slots)
        var owners = py_call(py_item(b, nk[].k_list),slots)
        if not class_watch_ready():
            # No class record can be kept (Python 3.10/3.11): the single
            # admit-and-release loop, without building a record payload.
            var plain_ancestry = py_call(py_item(py_item(b, nk[].k_type).__dict__, nk[].k___mro__).__get__,cls)
            if len(plain_ancestry) != 2 or plain_ancestry[1] is not py_item(b, nk[].k_object):
                raise Error("TypeError: native owner reinitialization cannot inherit Array lifecycle hooks")
            var plain_mapping = py_call(py_item(py_item(b, nk[].k_type).__dict__, nk[].k___dict__).__get__,cls)
            var plain_keys = py_call(py_item(b, nk[].k_list),plain_mapping)
            for i in range(len(plain_keys)):  # driver-loop: over native Array class keys before class-owner lookup
                if py_type(plain_keys[i]) is not py_item(b, nk[].k_str):
                    raise Error("TypeError: native owner reinitialization requires native Array class keys")
            if py_get(plain_mapping,nk[].k___del__) is not None or py_get(plain_mapping,nk[].k___getattribute__) is not None:
                raise Error("TypeError: native owner reinitialization cannot invoke Array lifecycle hooks")
            for i in range(len(owners)):  # driver-loop: over all retained Array slot owners, including otherwise unused metadata
                var plain_descriptor = slots[owners[i]]
                if py_get(plain_mapping,owners[i]) is not plain_descriptor:
                    raise Error("TypeError: native owner reinitialization requires native Array owner slots")
                native_release_admit(py_call(plain_descriptor.__get__,value,cls),context,depth+1)
            return
        # The Array class admission (ancestry, exact keys, no lifecycle hooks,
        # connected slot descriptors) depends only on the class rows and the
        # import-time slot table: once per class state, not once per Array.
        var mods = NativeModules()
        var record = class_record(mods, _ARRAY_CLASS_SLOT, cls)
        var admitted = record is not None and record[0] is slots and Int(py=record[1]) == len(owners)
        if admitted:
            for i in range(len(owners)):  # driver-loop: over the admitted slot descriptors
                if record[2][i] is not slots[owners[i]]:
                    admitted = False
                    break
        if not admitted:
            var ancestry = py_call(py_item(py_item(b, nk[].k_type).__dict__, nk[].k___mro__).__get__,cls)
            if len(ancestry) != 2 or ancestry[1] is not py_item(b, nk[].k_object):
                raise Error("TypeError: native owner reinitialization cannot inherit Array lifecycle hooks")
            var mapping = py_call(py_item(py_item(b, nk[].k_type).__dict__, nk[].k___dict__).__get__,cls)
            var keys = py_call(py_item(b, nk[].k_list),mapping)
            for i in range(len(keys)):  # driver-loop: over native Array class keys before class-owner lookup
                if py_type(keys[i]) is not py_item(b, nk[].k_str):
                    raise Error("TypeError: native owner reinitialization requires native Array class keys")
            if py_get(mapping,nk[].k___del__) is not None or py_get(mapping,nk[].k___getattribute__) is not None:
                raise Error("TypeError: native owner reinitialization cannot invoke Array lifecycle hooks")
            var descriptors = Python.list()
            for i in range(len(owners)):  # driver-loop: over all retained Array slot owners, including otherwise unused metadata
                var descriptor = slots[owners[i]]
                if py_get(mapping,owners[i]) is not descriptor:
                    raise Error("TypeError: native owner reinitialization requires native Array owner slots")
                _ = py_call(descriptors.append,descriptor)
            class_admit(mods, _ARRAY_CLASS_SLOT, cls, Python.tuple(slots, PythonObject(len(owners)), py_call(py_item(b, nk[].k_tuple),descriptors)))
        for i in range(len(owners)):  # driver-loop: over all retained Array slot owners, including otherwise unused metadata
            native_release_admit(py_call(slots[owners[i]].__get__,value,cls),context,depth+1)
        return
    if cls is py_item(b, nk[].k_memoryview):
        native_release_admit(value.obj,context,depth+1)
        return
    if cls is py_item(context, nk[].k_array_type) or cls is py_item(context, nk[].k_owner_mmap_class):
        return
    if cls is py_get(context,nk[].k_ndarray):
        var dtype = value.dtype
        if not _released_dtype(dtype,context,depth+1):
            raise Error("TypeError: native owner reinitialization requires an admitted native dtype")
        if Bool(py=dtype.hasobject):
            _released_object_elements(value,context,depth+1)
        native_release_admit(value.base,context,depth+1)
        return
    if _released_dtype(value,context,depth):
        return
    var scalars = py_get(context,nk[].k_owner_dtype_scalars)
    if scalars is not None:
        for i in range(len(scalars)):  # driver-loop: over exact primitive NumPy scalar identities owning no application objects
            if cls is scalars[i]:
                return
    # A fitted model's own compiled owners (its kernel-methods provider, the
    # resident entry's resource) and the binding module they record: set_params
    # and every other reset drop them without running Python code.
    var pins = _native_owner_pins(cls, b)
    if pins:
        var inspect = pins.value()
        var count = inspect(Int(value._obj_ptr), -1)
        if count < 0:
            raise Error("TypeError: native owner reinitialization cannot inspect retained native owners")
        for index in range(count):  # driver-loop: native owner's retained metadata in declaration order
            var address = inspect(Int(value._obj_ptr), index)
            if address <= 0:
                raise Error("TypeError: native owner reinitialization cannot inspect retained native owners")
            var held = borrowed_native_owner(address)
            native_release_admit(held, context, depth + 1)
        _ = value  # the inspector returns borrowed pins from this retained owner
        return
    if _retained_module(value, cls, b, context):
        return
    if _retained_entry(value, cls, b, context):
        return
    var capsule_class = py_get(context, nk[].k_owner_capsule_class)
    if capsule_class is not None and cls is capsule_class and resident_capsule_owned(value):
        return
    # The refused value's class leads the words (lane gs-fix-2: GaussianProcessClassifier.set_params on the GPU
    # install was refused with no hint of which fitted field held it). Refusal path only.
    raise Error("TypeError: native owner reinitialization cannot release Python-owned metadata ("
                + String(py=py_call(native_loaded_module("builtins").repr, cls)) + "); retain the same owner or release it in application code first")
