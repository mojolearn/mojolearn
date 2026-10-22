# SPDX-License-Identifier: Apache-2.0
# Retained pandas owners are storage, not callbacks. Inspect the exact admitted
# ownership graph before a resident operation drops its references. Native
# traversal uses CPython's documented stable slot API, never gc.get_referents
# (which raises Python audit events), Python properties, or private struct offsets.
from std.memory import bitcast, MutPointer
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.native_numpy_metadata import native_numpy_module_fields, native_numpy_dtype
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.optimizer_public_words import stash_python_error


comptime _Words = MutPointer[Int, MutUntrackedOrigin]
comptime _Visit = def(PyObjectPtr, _Words) thin abi("C") -> Int32
comptime _Traverse = def(PyObjectPtr, _Visit, _Words) thin abi("C") -> Int32


def _visit_owner(value: PyObjectPtr, state: _Words) abi("C") -> Int32:
    # tp_traverse visitors must not allocate or adjust any Python references.
    # The source owner remains live across both passes. Only scalar addresses
    # enter the preallocated native buffer, and the GIL remains held.
    var index = state[0]
    if index < state[1]:
        state[2 + index] = Int(bitcast[UInt](value))
    state[0] = index + 1
    return 0


def _native_owned_references(value: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    # Py_tp_traverse=71 is part of CPython's stable ABI (Include/typeslots.h).
    var address = cpy.lib.get_function[Int]("PyType_GetSlot")(py_type(value)._obj_ptr, Int32(71))
    var result = Python.list()
    if address == 0:
        return result
    var traverse = bitcast[_Traverse](address)
    var count = List[Int](length=2, fill=0)
    var pointer = count.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if traverse(value._obj_ptr, _visit_owner, pointer) != 0:
        raise Error("TypeError: native pandas owner traversal failed")
    var capacity = count[0]
    var storage = List[Int](length=capacity + 2, fill=0)
    storage[1] = capacity
    var output = storage.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if traverse(value._obj_ptr, _visit_owner, output) != 0 or storage[0] != capacity:
        raise Error("TypeError: native pandas ownership changed during traversal")
    for i in range(capacity):  # driver-loop: retain borrowed native references only after the traversal has returned
        var owner = PythonObject(from_borrowed=bitcast[PyObjectPtr](storage[2 + i]))
        _ = py_call(result.append, owner)
    # The owning value is borrowed by this function and remains live through
    # the final reference promotion; native buffers survive both FFI calls.
    _ = count^
    _ = storage^
    return result


def _pandas_classes(context: PythonObject) raises -> PythonObject:
    var result = Python.list()
    var specifications = context["arena_pandas_owner_specs"]
    var b = context["arena_native_types"]
    var modules = context["modules"]
    var module_keys = py_call(b["list"], modules)
    for i in range(len(module_keys)):  # driver-loop: even late optional-module lookup must not invoke key equality hooks
        if py_type(module_keys[i]) is not b["str"]:
            raise Error("TypeError: resident optional-module keys must be native strings")
    for i in range(len(specifications)):  # driver-loop: fixed supported pandas metadata classes; importing pandas is caller work
        var module = py_get(modules, specifications[i][0])
        if module is None:
            continue
        if py_type(module) is not context["native_module_type"]:
            raise Error("TypeError: resident pandas owners require native module metadata")
        var fields = native_numpy_module_fields(module)
        var names = specifications[i][1]
        for j in range(len(names)):  # driver-loop: absent version-specific native classes are not admission failures
            var cls = py_get(fields, names[j])
            if cls is not None:
                var ancestry = py_call(b["type"].__dict__["__mro__"].__get__, py_type(cls))
                var type_object = False
                for k in range(len(ancestry)):  # driver-loop: dateutil timezone classes have ordinary factory metaclasses; no factory or metadata hook is called
                    type_object = type_object or ancestry[k] is b["type"]
                if not type_object:
                    raise Error("TypeError: resident pandas owner classes require native type metadata")
                _ = py_call(result.append, Python.tuple(cls, 1))
    var interval = py_get(modules, "pandas._libs.interval")
    if interval is not None:
        if py_type(interval) is not context["native_module_type"]:
            raise Error("TypeError: resident interval owners require native module metadata")
        var nodes = py_get(native_numpy_module_fields(interval), "NODE_CLASSES")
        if nodes is not None:
            if py_type(nodes) is not b["dict"]:
                raise Error("TypeError: resident interval node registry must be a native dictionary")
            var pairs = py_call(b["list"], nodes.items())
            for i in range(len(pairs)):  # driver-loop: the native interval registry exposes all typed nodes without constructing a tree
                var cls = pairs[i][1]
                if py_type(cls) is not b["type"]:
                    raise Error("TypeError: resident interval node must be a native type")
                var flags = py_call(b["type"].__dict__["__flags__"].__get__, cls)
                var module = py_call(b["type"].__getattribute__, cls, "__module__")
                if Int(py=flags) & 512 != 0 or py_type(module) is not b["str"] or String(module) != "pandas._libs.interval":
                    raise Error("TypeError: resident interval nodes must be static native extension types")
                _ = py_call(result.append, Python.tuple(cls, 2))
    var pytz = py_get(modules, "pytz")
    var tzinfo = py_get(modules, "pytz.tzinfo")
    if pytz is not None and tzinfo is not None:
        if py_type(pytz) is not context["native_module_type"] or py_type(tzinfo) is not context["native_module_type"]:
            raise Error("TypeError: resident timezone owners require native module metadata")
        var fields = native_numpy_module_fields(pytz)
        var base = py_get(native_numpy_module_fields(tzinfo), "BaseTzInfo")
        var cache = py_get(fields, "_tzinfo_cache")
        var values = Python.list()
        var utc = py_get(fields, "utc")
        if utc is not None:
            _ = py_call(values.append, utc)
        if cache is not None:
            if py_type(cache) is not b["dict"]:
                raise Error("TypeError: resident timezone cache must be a native dictionary")
            var pairs = py_call(b["list"], cache.items())
            for i in range(len(pairs)):  # driver-loop: dynamic timezone classes are registered by existing cache identity, never constructed or imported during cleanup
                _ = py_call(values.append, pairs[i][1])
        for i in range(len(values)):  # driver-loop: pytz timezone instances have ordinary native Python-object storage, including cyclic offset dictionaries
            var cls = py_type(values[i])
            if py_type(cls) is not b["type"]:
                raise Error("TypeError: resident timezone owners require ordinary type metadata")
            var ancestry = py_call(b["type"].__dict__["__mro__"].__get__, cls)
            var known = False
            for j in range(len(ancestry)):  # driver-loop: inspect the connected timezone class ancestry without invoking metadata hooks
                known = known or ancestry[j] is base
            if not known:
                raise Error("TypeError: resident timezone cache contains an unsupported owner")
            _ = py_call(result.append, Python.tuple(cls, 1))
    if py_get(context, "ndarray") is not None:
        var factory = native_numpy_dtype(context)
        for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: pandas index caches may retain native scalar/dtype classes themselves
            var dtype = py_call(factory, code)
            # Class references are safe to retain, but their instances still
            # require the explicit NumPy dtype/object-buffer owner walker.
            _ = py_call(result.append, Python.tuple(dtype.type, 0))
            _ = py_call(result.append, Python.tuple(py_type(dtype), 0))
    return result


def _admit_pandas_class(cls: PythonObject, context: PythonObject) raises:
    var b = context["arena_native_types"]
    var ancestry = py_call(b["type"].__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: reject actual finalizers before any source-owner reference is replaced
        var fields = py_call(b["type"].__dict__["__dict__"].__get__, ancestry[i])
        var keys = py_call(b["list"], fields)
        for j in range(len(keys)):  # driver-loop: class metadata cannot trigger application key equality
            if py_type(keys[j]) is not b["str"]:
                raise Error("TypeError: resident pandas class keys must be native strings")
        if py_contains(fields, "__del__"):
            raise Error("TypeError: resident cleanup cannot release a pandas application finalizer")


def _native_field(descriptor: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(descriptor) is not context["arena_getset_descriptor"]:
        raise Error("TypeError: resident buffer owners require native readonly field descriptors")
    ref cpy = Python().cpython()
    var call = cpy.lib.get_function[PyObjectPtr]("PyObject_CallObject")
    var matches = cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")
    var clear = cpy.lib.get_function[NoneType]("PyErr_Clear")
    var getter = descriptor.__get__
    var args = Python.tuple(value, py_type(value))
    var result = call(getter._obj_ptr, args._obj_ptr)
    _ = getter^
    _ = args^
    if not result:
        if matches(context["arena_attribute_error"]._obj_ptr) != 0:
            # A Cython typed memoryview field can be uninitialized. Its
            # native readonly getter raises AttributeError and owns no buffer.
            clear()
            return PythonObject(None)
        var error = stash_python_error()
        raise error^
    return PythonObject(from_owned=result)


def _interval_buffers(value: PythonObject, context: PythonObject, classes: PythonObject,
                     seen: PythonObject) raises:
    var b = context["arena_native_types"]
    var cls = py_type(value)
    var mapping = py_call(b["type"].__dict__["__dict__"].__get__, cls)
    # These are the seven readonly typed-memoryview owners declared by
    # pandas/_libs/intervaltree.pxi.in. Cython omits them from tp_traverse;
    # allocating .as_array or inspecting only GC referents would be unsound.
    for name in ["center_left_values", "center_right_values", "left", "right", "center_left_indices", "center_right_indices", "indices"]:  # driver-loop: native interval buffer fields, including inactive cached storage
        var view = _native_field(py_get(mapping, name), value, context)
        if view is None:
            continue
        var view_class = py_type(view)
        if py_type(view_class) is not b["type"]:
            raise Error("TypeError: resident interval storage requires a native memoryview class")
        var flags = py_call(b["type"].__dict__["__flags__"].__get__, view_class)
        var module = py_call(b["type"].__getattribute__, view_class, "__module__")
        var label = py_call(b["type"].__getattribute__, view_class, "__name__")
        if Int(py=flags) & 512 != 0 or py_type(module) is not b["str"] or py_type(label) is not b["str"]:
            raise Error("TypeError: resident interval storage requires static native Cython memoryviews")
        if String(label) != "_memoryviewslice" or (String(module) != "_cyutility" and String(module) != "pandas._libs.interval"):
            raise Error("TypeError: resident interval storage has an unsupported native owner class")
        _admit_pandas_class(view_class, context)
        native_release_weakrefs(view, context)
        var ancestry = py_call(b["type"].__dict__["__mro__"].__get__, view_class)
        var base_descriptor = PythonObject(None)
        for i in range(len(ancestry)):  # driver-loop: the shared Cython base owns the actual buffer exporter
            var fields = py_call(b["type"].__dict__["__dict__"].__get__, ancestry[i])
            var descriptor = py_get(fields, "base")
            if descriptor is not None:
                base_descriptor = descriptor
                break
        _resident_owner(_native_field(base_descriptor, view, context), context, classes, seen)


def _resident_owner(value: PythonObject, context: PythonObject, classes: PythonObject,
                    seen: PythonObject) raises:
    var b = context["owner_native_types"]
    var cls = py_type(value)
    if value is None or cls is b["str"] or cls is b["bytes"] or cls is b["bytearray"] or cls is b["bool"] or cls is b["int"] or cls is b["float"] or cls is b["complex"]:
        return
    for i in range(len(seen)):  # driver-loop: ordinary metadata cycles are admitted once by identity, without equality or hashing
        if seen[i] is value:
            return
    _ = py_call(seen.append, value)
    if cls is b["tuple"] or cls is b["list"] or cls is context["arena_set"] or cls is context["arena_frozenset"]:
        native_release_weakrefs(value, context)
        var items = value if cls is b["tuple"] or cls is b["list"] else py_call(b["list"], value)
        for i in range(len(items)):  # driver-loop: exact native containers expose their owners without Python iteration hooks
            _resident_owner(items[i], context, classes, seen)
        return
    if cls is b["dict"]:
        var pairs = py_call(b["list"], value.items())
        for i in range(len(pairs)):  # driver-loop: both metadata keys and values can own finalizers; no keyed lookup occurs
            _resident_owner(pairs[i][0], context, classes, seen)
            _resident_owner(pairs[i][1], context, classes, seen)
        return
    if cls is context["arena_slice"]:
        _resident_owner(value.start, context, classes, seen)
        _resident_owner(value.stop, context, classes, seen)
        _resident_owner(value.step, context, classes, seen)
        return
    if cls is context["arena_range"] or cls is b["object"]:
        return
    var calendar_types = context["arena_datetime_types"]
    if cls is calendar_types[0] or cls is calendar_types[3]:
        return
    if cls is calendar_types[1] or cls is calendar_types[2]:
        _resident_owner(value.tzinfo, context, classes, seen)
        return
    if cls is calendar_types[4]:
        _resident_owner(py_call(context["arena_timezone_name"], value, PythonObject(None)), context, classes, seen)
        _resident_owner(py_call(context["arena_timezone_offset"], value, PythonObject(None)), context, classes, seen)
        return
    if cls is context["owner_weakref_class"]:
        native_release_weakrefs(value, context)
        _resident_owner(py_call(context["owner_weakref_callback"].__get__, value), context, classes, seen)
        # A weak reference does not own its referent. Releasing this reference
        # does not invoke its callback; the callback itself may own finalizers.
        return
    for i in range(len(classes)):  # driver-loop: exact, connected owner classes have known native storage/destruction
        if value is classes[i][0]:
            # The connected registry holds every class through the complete
            # cleanup. Releasing an instance cannot destroy its class metadata.
            return
        if Int(py=classes[i][1]) != 0 and cls is classes[i][0]:
            native_release_weakrefs(value, context)
            _admit_pandas_class(cls, context)
            if Int(py=classes[i][1]) == 2:
                _interval_buffers(value, context, classes, seen)
            var owners = _native_owned_references(value)
            for j in range(len(owners)):  # driver-loop: native pandas fields include hidden cached arrays, manager blocks and index metadata
                _resident_owner(owners[j], context, classes, seen)
            return
    native_release_admit(value, context)


def resident_release_admit(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var classes = _pandas_classes(context)
    var seen = Python.list()
    _resident_owner(value, context, classes, seen)
    # Keep admitted classes and all actual owners alive through the caller's
    # cleanup. The caller consumes this guard after its final dictionary edit.
    return Python.tuple(classes, seen)
