# SPDX-License-Identifier: Apache-2.0
# Read-only cache operations. No upload, release or hidden-owner traversal.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.optimizer_public_words import stash_python_error


def _string_keys(mapping: PythonObject, context: PythonObject) raises:
    var keys = py_call(context["arena_native_types"]["list"],mapping)
    for i in range(len(keys)):  # driver-loop: admit metadata keys before lookup can invoke equality
        if py_type(keys[i]) is not context["arena_native_types"]["str"]:
            raise Error("TypeError: cache metadata keys must be native strings")


def _mapping(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = py_call(context["cache_type_dict"],cls)
    _string_keys(result,context)
    return result


def _attribute(value: PythonObject, name: String, default: PythonObject,
               required: Bool, context: PythonObject) raises -> PythonObject:
    return _attribute(value,PythonObject(name),default,required,context)


def _attribute(value: PythonObject, name: PythonObject, default: PythonObject,
               required: Bool, context: PythonObject) raises -> PythonObject:
    # GenericGetDict returns actual storage, including ordinary Python-created
    # instance dictionaries, without consulting a replaced __dict__ property.
    # Data descriptors are admitted independently before generic slot lookup.
    var b = context["arena_native_types"]
    var cls = py_type(value)
    if py_type(cls) is not b["type"]:
        raise Error("TypeError: cache metadata cannot invoke a custom metaclass")
    var ancestry = py_call(context["cache_type_mro"],cls)
    var namespace = False
    for i in range(len(ancestry)):  # driver-loop: a retained native namespace slot applies only to its own owner ancestry
        namespace = namespace or ancestry[i] is context["cache_namespace_class"]
    var selected = PythonObject(None)
    var found = False
    for i in range(len(ancestry)):  # driver-loop: ordinary attribute lookup without calling Python hooks
        var mapping = _mapping(ancestry[i],context)
        var hook = py_get(mapping,"__getattribute__")
        # CPython 3.11 exposes a distinct native SimpleNamespace wrapper;
        # newer CPython inherits object's wrapper. Both perform ordinary
        # native lookup. Never admit arbitrary C descriptors by type/name,
        # or a namespace wrapper transplanted onto an unrelated class.
        var native_hook = hook is None or hook is context["cache_object_getattribute"] or (namespace and hook is context["cache_namespace_getattribute"])
        if not native_hook or py_get(mapping,"__getattr__") is not None:
            raise Error("TypeError: cache metadata requires native attribute access; Python hooks cannot execute")
        if not found and py_contains(mapping,PythonObject(name)):
            selected = mapping[PythonObject(name)]
            found = True
    var member = found and py_type(selected) is context["cache_member_type"]
    var descriptor = False
    var data_descriptor = False
    if found and not member:
        var descriptor_mro = py_call(context["cache_type_mro"],py_type(selected))
        for i in range(len(descriptor_mro)):  # driver-loop: inspect descriptor slots without invoking them
            var mapping = _mapping(descriptor_mro[i],context)
            descriptor = descriptor or py_contains(mapping,"__get__")
            data_descriptor = data_descriptor or py_contains(mapping,"__set__") or py_contains(mapping,"__delete__")
    if member:
        # Exact native member descriptors never dispatch to Python code. A
        # foreign member raises its normal native owner/type error.
        ref cpy = Python().cpython()
        var key = PythonObject(name)
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(value._obj_ptr,key._obj_ptr)
        _ = key^
        if not pointer:
            if not required and cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(context["cache_attribute_error"]._obj_ptr) != 0:
                cpy.lib.get_function[NoneType]("PyErr_Clear")()
                return default
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=pointer)
    if descriptor and data_descriptor:
        raise Error("TypeError: cache metadata cannot invoke Python or unregistered native descriptors")
    ref cpy = Python().cpython()
    var fields_pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(value._obj_ptr,PyObjectPtr())
    if fields_pointer:
        var fields = PythonObject(from_owned=fields_pointer)
        if py_type(fields) is not b["dict"]:
            raise Error("TypeError: cache metadata requires an ordinary instance dictionary")
        _string_keys(fields,context)
        if py_contains(fields,PythonObject(name)):
            return fields[PythonObject(name)]
    elif cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(context["cache_attribute_error"]._obj_ptr) != 0:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
    else:
        var failure = stash_python_error()
        raise failure^
    if found:
        if descriptor:
            raise Error("TypeError: cache metadata cannot invoke Python or unregistered native descriptors")
        return selected
    if required:
        raise Error("AttributeError: cache metadata has no '" + String(name) + "' attribute")
    return default


def _hash_key(value: PythonObject, context: PythonObject) raises:
    # Exact tuples cannot form a cycle without a non-tuple container. Walk
    # them iteratively so admission does not impose an arbitrary depth limit
    # before CPython performs the actual native hashing/equality operation.
    var pending = List[PythonObject]()
    pending.append(value)
    var types = context["cache_hash_types"]
    while len(pending):  # driver-loop: explicit native work stack over immutable key metadata
        var item = pending.pop()
        var cls = py_type(item)
        var native = False
        for i in range(len(types)):  # driver-loop: exact immutable builtin hashing and equality slots
            native = native or cls is types[i]
        if native:
            continue
        if cls is not context["arena_native_types"]["tuple"]:
            raise Error("TypeError: cache keys require native immutable values without Python equality or hash hooks")
        for i in range(len(item)):  # driver-loop: tuple hashing must not dispatch to an application key
            pending.append(item[i])


def arena_cache_stats(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    if operation == "resident_stats":
        if py_type(model) is not context["cache_local_type"]:
            raise Error("TypeError: resident statistics require the native thread-local owner")
        var mapping = _mapping(context["cache_local_type"],context)
        if py_get(mapping,"__getattribute__") is not context["cache_local_getattribute"]:
            raise Error("TypeError: resident thread-local attribute entry has changed")
        var fields = py_call(context["cache_local_getattribute"],model,"__dict__")
        if py_type(fields) is not b["dict"]:
            raise Error("TypeError: resident thread-local fields require a native dictionary")
        _string_keys(fields,context)
        var scope = py_get(fields,"scope")
        if scope is None:
            return PythonObject(None)
        var caches = _attribute(scope,"caches",PythonObject(None),True,context)
        if py_type(caches) is not b["dict"]:
            raise Error("TypeError: resident cache registry requires a native dictionary")
        var pairs = py_call(b["list"],py_call(caches.items))
        var result = Python.dict()
        for i in range(len(pairs)):  # driver-loop: materialize current-thread counters in registry order
            var pair = pairs[i]
            var key = pair[0]
            var entry = pair[1]
            if (py_type(key) is not b["tuple"] and py_type(key) is not b["list"]) or (py_type(entry) is not b["tuple"] and py_type(entry) is not b["list"]):
                raise Error("TypeError: resident cache registry requires native key/entry pairs")
            for pair_value in [key,entry]:  # driver-loop: preserve the ordinary two-field unpacking failure contract
                var count = len(pair_value)
                if count < 2:
                    raise Error("ValueError: not enough values to unpack (expected 2, got " + String(count) + ")")
                if count > 2:
                    raise Error("ValueError: too many values to unpack (expected 2)")
            var prefix = key[1]
            _hash_key(prefix,context)
            var uploads = _attribute(entry[0],"uploads",PythonObject(None),True,context)
            var hits = _attribute(entry[0],"hits",PythonObject(None),True,context)
            py_setitem(result,prefix,py_call(b["tuple"],Python.list(uploads,hits)))
        return result
    var entries = _attribute(model,"_entries",PythonObject(None),True,context)
    if py_type(entries) is not b["dict"]:
        raise Error("TypeError: DeviceCache entries require a native dictionary")
    if operation == "cache_len":
        return PythonObject(len(entries))
    raise Error("ValueError: unknown native cache-statistics operation")
