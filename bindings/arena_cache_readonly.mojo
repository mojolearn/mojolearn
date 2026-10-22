# SPDX-License-Identifier: Apache-2.0
# Read-only cache operations. No upload, release or hidden-owner traversal.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.optimizer_public_words import stash_python_error
from bindings.training_scalar_native import training_scalar_number, training_scalar_warning, training_void_bytes
from bindings.native_numpy_metadata import native_numpy_type
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from bindings.sequence_array_native import sequence_array_release


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
    # GenericGetDict returns actual storage, including ordinary Python-created
    # instance dictionaries, without consulting a replaced __dict__ property.
    # Data descriptors are admitted independently before generic slot lookup.
    var b = context["arena_native_types"]
    var cls = py_type(value)
    if py_type(cls) is not b["type"]:
        raise Error("TypeError: cache metadata cannot invoke a custom metaclass")
    var ancestry = py_call(context["cache_type_mro"],cls)
    var selected = PythonObject(None)
    var found = False
    for i in range(len(ancestry)):  # driver-loop: ordinary attribute lookup without calling Python hooks
        var mapping = _mapping(ancestry[i],context)
        var hook = py_get(mapping,"__getattribute__")
        if (hook is not None and hook is not context["cache_object_getattribute"]) or py_get(mapping,"__getattr__") is not None:
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
        raise Error("AttributeError: cache metadata has no '" + name + "' attribute")
    return default


def _hash_key(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: cache keys exceed native nesting bounds")
    var cls = py_type(value)
    var types = context["cache_hash_types"]
    for i in range(len(types)):  # driver-loop: exact immutable builtin hashing and equality slots
        if cls is types[i]:
            return
    if cls is context["arena_native_types"]["tuple"]:
        for i in range(len(value)):  # driver-loop: tuple hashing must not dispatch to an application key
            _hash_key(value[i],context,depth+1)
        return
    var scalar = context["cache_scalar_context"]
    if py_get(scalar["modules"],"numpy") is not None:
        # A NumPy string prefix remains a native string under the existing
        # `prefix + suffix` store connection. Its immutable equality/hash
        # slots do not dispatch through numeric ufunc or warning machinery.
        for code in ["S","U"]:  # driver-loop: native string prefixes supported by the existing store connector
            if cls is sequence_dtype_create(PythonObject(code),scalar).type:
                return
    raise Error("TypeError: cache keys require native immutable values without Python equality or hash hooks")


def _itemsize_four(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> Bool:
    if depth > 64:
        raise Error("TypeError: cache itemsize exceeds native nesting bounds")
    var b = context["arena_native_types"]
    var cls = py_type(value)
    var scalar = context["cache_scalar_context"]
    var admitted = False
    var reference = PythonObject(4)
    var types = context["cache_itemsize_types"]
    for i in range(len(types)):  # driver-loop: primitive comparison never invokes application protocols
        admitted = admitted or cls is types[i]
    if not admitted and py_get(scalar["modules"],"numpy") is not None:
        var classes = sequence_dtype_classes(scalar)
        if sequence_dtype_instance(value,classes):
            # A dtype descriptor compares unequal to the integer four; do
            # not consult any metadata object retained by that descriptor.
            return False
        if cls is native_numpy_type("ndarray",scalar):
            var dtype = value.dtype
            var native = sequence_dtype_instance(dtype,classes)
            for code in ["q","Q"]:  # driver-loop: native integer aliases retain their scalar layout
                native = native or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code),scalar))
            if not native:
                raise Error("TypeError: cache itemsize requires a native ndarray dtype")
            var count = Int(py=value.size)
            if count != 1:
                # NumPy 2.2 finalized the empty-array truth deprecation:
                # https://github.com/numpy/numpy/pull/27160
                if count == 0 and sequence_array_release(scalar) < 202:
                    training_scalar_warning("The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.",
                                            context["cache_deprecation_warning"],"cache_get",scalar)
                    # Legacy `if empty_array != 4` is false, hence admitted.
                    return True
                if count == 0:
                    raise Error("ValueError: The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.")
                raise Error("ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
            var indexes = Python.list()
            for i in range(Int(py=value.ndim)):  # driver-loop: unbox one native scalar without ndarray comparison/computation
                _ = py_call(indexes.append,0)
            var index = py_call(b["tuple"],indexes)
            ref cpy = Python().cpython()
            var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr,index._obj_ptr)
            _ = index^
            if not pointer:
                var failure = stash_python_error()
                raise failure^
            var item = PythonObject(from_owned=pointer)
            var result = _itemsize_four(item,context,depth+1)
            _ = item^
            return result
        for i in range(len(classes)):  # driver-loop: NumPy's exact built-in scalar identities
            admitted = admitted or cls is classes[i][1]
        for code in ["q","Q"]:  # driver-loop: admit distinct long-long scalar aliases
            admitted = admitted or cls is sequence_dtype_create(PythonObject(code),scalar).type
        if admitted:
            var kind = String(value.dtype.kind)
            if kind == "V":
                raise Error("TypeError: Cannot compare structured or void to non-void arrays.")
            if kind == "M":
                return False
            if kind == "m":
                # Bare integer comparisons use generic units. Emit NumPy
                # 2.5's warning natively before comparing the public buffer
                # word, rather than entering its warning callback path.
                # https://numpy.org/doc/2.5/release/2.5.0-notes.html
                if sequence_array_release(scalar) >= 205:
                    training_scalar_warning("The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`). Please use a specific unit instead.",
                                            context["cache_deprecation_warning"],"cache_get",scalar)
                var raw = training_void_bytes(value)
                var word = UInt64(0)
                for i in range(8):  # driver-loop: read the exact native timedelta buffer word
                    var byte = i if Bool(py=context["cache_little_endian"]) else 7-i
                    word |= UInt64(Int(py=raw[byte])) << (8*i)
                return word == 4
            if kind == "b" or kind == "S" or kind == "U":
                return False
            # NumPy 1.x compares some narrow scalars with a Python int by
            # dispatching through its mutable numeric-ops table. Matching
            # exact scalar types use their native scalar comparison slot.
            # The only constructed value is the fixed exact builtin int 4;
            # no input conversion, ndarray computation or dtype hook runs.
            reference = py_call(cls,4)
    if not admitted:
        raise Error("TypeError: cache itemsize requires native scalar metadata; Python equality hooks cannot execute")
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(value._obj_ptr,reference._obj_ptr,Int32(2))
    _ = reference^
    if result < 0:
        var failure = stash_python_error()
        raise failure^
    return result != 0


def arena_cache_readonly(model: PythonObject, operation: String, args: PythonObject,
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
            if (py_type(key) is not b["tuple"] and py_type(key) is not b["list"]) or len(key) != 2 or (py_type(entry) is not b["tuple"] and py_type(entry) is not b["list"]) or len(entry) != 2:
                raise Error("TypeError: resident cache registry requires native key/entry pairs")
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
    if operation != "cache_get":
        raise Error("ValueError: unknown read-only cache operation")
    # The old method resolved self._key. Its ordinary fixed implementation is
    # now native; a subclass callback must be refused rather than ignored.
    var ancestry = py_call(context["cache_type_mro"],py_type(model))
    for i in range(len(ancestry)):  # driver-loop: admit the existing cache-key contract before reading caller metadata
        var mapping = _mapping(ancestry[i],context)
        if py_contains(mapping,"_key"):
            if mapping["_key"] is not context["cache_key_descriptor"]:
                raise Error("TypeError: DeviceCache.get cannot invoke a custom key callback")
            break
    ref cpy = Python().cpython()
    var fields_pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(model._obj_ptr,PyObjectPtr())
    if not fields_pointer:
        var failure = stash_python_error()
        raise failure^
    var fields = PythonObject(from_owned=fields_pointer)
    _string_keys(fields,context)
    if py_contains(fields,"_key") and fields["_key"] is not context["cache_key_function"]:
        raise Error("TypeError: DeviceCache.get cannot invoke an instance key callback")
    var source = args[0]
    if not _itemsize_four(_attribute(source,"itemsize",PythonObject(4),False,context),context):
        raise Error("ValueError: DeviceCache: a store slot holds 4-byte words")
    var scalar = context["cache_scalar_context"]
    var address = training_scalar_number(_attribute(source,"_addr",PythonObject(None),True,context),"cache_get",scalar,True)
    var size = training_scalar_number(_attribute(source,"size",PythonObject(None),True,context),"cache_get",scalar,True)
    var keys = py_call(b["list"],entries)
    for i in range(len(keys)):  # driver-loop: prevent resident dictionary equality from dispatching to custom stored keys
        _hash_key(keys[i],context)
    var key = py_call(b["tuple"],Python.list(address,size))
    var entry = py_get(entries,key)
    if entry is None:
        return PythonObject(None)
    if py_type(entry) is not b["tuple"] and py_type(entry) is not b["list"]:
        raise Error("TypeError: DeviceCache entries require native slot/owner sequences")
    var index = PythonObject(0)
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(entry._obj_ptr,index._obj_ptr)
    _ = index^
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)
