# SPDX-License-Identifier: Apache-2.0
# Read-only public DeviceCache lookup. Store mutation/lifecycle remain separate.
#
# Lane arena (2026-11-21): the cache is one native `DeviceCacheState`
# (`core/arena_state.mojo`, `ARENA_CACHES`) behind the owner object in the
# instance's `_owner` slot. UNPACK here is the slot read (`arena_cache_view`)
# and the admission of the caller's buffer metadata (`arena_cache_key`); the
# lookup is a typed read of the state. The instance dictionary, its entries
# dictionary and the `_key` callback admission are gone.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.arena_cache_stats_unpack import _mapping
from bindings.arena_cache_stats_pack import _attribute
from bindings.arena_owner_door import ArenaOwnerView, ARENA_OWNER_CACHE, arena_owner_held
from bindings.optimizer_public_words_unpack import stash_python_error
from core.arena_state import ARENA_CACHES, arena_handle
from bindings.training_scalar_native_door import training_scalar_warning, training_void_bytes
from bindings.native_numpy_metadata_pack import native_numpy_type, native_numpy_module_fields
from bindings.native_scalar_decode_unpack import native_scalar_buffer
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.sequence_dtype_native_door import sequence_dtype_instance, sequence_dtype_create
from bindings.arena_cache_get_run import _temporal_integer_name


def _one_of(cls: PythonObject, types: PythonObject) raises -> Bool:
    for i in range(len(types)):  # driver-loop: exact retained native identities only
        if cls is types[i]:
            return True
    return False


def _inherited_primitive(value: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_type(value)
    if py_type(cls) is not context["arena_native_types"]["type"]:
        return False
    # Exact NumPy static scalar classes follow the storage decoder instead.
    # Python-created subclasses may retain builtin number/string slots when
    # every intervening class inherits those exact native implementations.
    if (Int(py=py_call(context["cache_type_flags"],cls)) & 512) == 0:
        return False
    var ancestry = py_call(context["cache_type_mro"],cls)
    var index = -1
    for i in range(len(ancestry)):  # driver-loop: locate an inherited native primitive base without protocols
        if _one_of(ancestry[i],context["cache_primitive_bases"]):
            index = i
            break
    if index < 0:
        return False
    var base = _mapping(ancestry[index],context)
    for i in range(index):  # driver-loop: admit only unchanged native conversion/comparison/text slots
        var mapping = _mapping(ancestry[i],context)
        for name in ["__int__","__index__","__float__","__trunc__","__eq__","__ne__","__repr__","__str__"]:  # driver-loop: fixed native primitive slot surface
            var key = PythonObject(name)
            if py_contains(mapping,key) and mapping[key] is not py_get(base,key):
                raise Error("TypeError: cache scalar metadata cannot invoke overridden Python conversion, equality or text hooks")
    return True


def _classes(context: PythonObject) raises -> PythonObject:
    var scalar = context["cache_scalar_context"]
    # Connect exact descriptor identities through fixed native dtype metadata.
    # Do not traverse the mutable Python dtype-alias registry or call NumPy
    # scalar constructors/conversion methods.
    var stored = py_get(scalar,"cache_dtype_classes")
    if stored is not None:
        return stored
    var result = Python.list()
    for code in ["?","i1","u1","i2","u2","i4","u4","i8","u8","q","Q","f2","f4","f8","g","c8","c16","G","O","S","U","V","M8","m8"]:  # driver-loop: fixed native descriptor/scalar identity connection
        var item = sequence_dtype_create(PythonObject(code),scalar)
        _ = py_call(result.append,Python.tuple(py_type(item),item.type))
    var module = py_get(scalar["modules"],"numpy.dtypes")
    if module is not None:
        if py_type(module) is not scalar["module_type"]:
            raise Error("TypeError: cache dtype metadata requires NumPy's native module")
        var strings = py_get(native_numpy_module_fields(module),"StringDType")
        if strings is not None:
            var item = sequence_dtype_create(PythonObject("T"),scalar)
            if py_type(item) is not strings:
                raise Error("TypeError: cache StringDType native identity has changed")
            _ = py_call(result.append,Python.tuple(strings,item.type))
    py_setitem(scalar,"cache_dtype_classes",result)
    return result


def _release(context: PythonObject) raises -> Int:
    var scalar = context["cache_scalar_context"]
    var module = py_get(scalar["modules"],"numpy")
    if module is None or py_type(module) is not scalar["module_type"]:
        raise Error("TypeError: cache compatibility requires NumPy's native module")
    var version = py_get(native_numpy_module_fields(module),"__version__")
    if py_type(version) is not context["arena_native_types"]["str"]:
        raise Error("TypeError: cache compatibility requires native version text")
    var parts = String(version).split(".")
    if len(parts) < 2:
        raise Error("TypeError: cache compatibility requires NumPy major.minor metadata")
    return Int(String(parts[0]))*100 + Int(String(parts[1]))


def _dtype(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var scalar = context["cache_scalar_context"]
    if py_get(scalar["modules"],"numpy") is None:
        raise Error("TypeError: cache metadata requires native scalar values; Python protocols cannot execute")
    var classes = _classes(context)
    var cls = py_type(value)
    var array = cls is native_numpy_type("ndarray",scalar)
    var admitted = array
    for i in range(len(classes)):  # driver-loop: exact immutable native scalar class identities
        admitted = admitted or cls is classes[i][1]
    for code in ["q","Q"]:  # driver-loop: distinct long-long aliases retain their native layout
        admitted = admitted or cls is sequence_dtype_create(PythonObject(code),scalar).type
    if not admitted:
        raise Error("TypeError: cache metadata requires native scalar values; Python protocols cannot execute")
    var dtype = value.dtype
    var native = sequence_dtype_instance(dtype,classes)
    for code in ["q","Q"]:  # driver-loop: distinct long-long descriptor identities
        native = native or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code),scalar))
    if not native:
        raise Error("TypeError: cache metadata requires a built-in native dtype")
    return dtype


def _array_item(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var indexes = Python.list()
    for i in range(Int(py=value.ndim)):  # driver-loop: scalar unboxing at an all-zero native index
        _ = py_call(indexes.append,0)
    var index = py_call(context["arena_native_types"]["tuple"],indexes)
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr,index._obj_ptr)
    _ = index^
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def _temporal_word(value: PythonObject, context: PythonObject) raises -> Int64:
    var raw = training_void_bytes(value)
    var word = UInt64(0)
    for i in range(8):  # driver-loop: datetime/timedelta expose one native signed storage word
        var byte = i if Bool(py=context["cache_little_endian"]) else 7-i
        word |= UInt64(Int(py=raw[byte])) << UInt64(8*i)
    return Int64(word)


def _itemsize_four(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> Bool:
    if depth > 64:
        raise Error("TypeError: cache itemsize exceeds native nesting bounds")
    var cls = py_type(value)
    if _one_of(cls,context["cache_itemsize_types"]) or _inherited_primitive(value,context):
        # Exact builtin slots compare only with the fixed builtin int four.
        # Unlike NumPy scalar comparison this cannot dispatch a numeric ufunc.
        ref cpy = Python().cpython()
        var four = PythonObject(4)
        var equal = cpy.lib.get_function[Int32]("PyObject_RichCompareBool")(value._obj_ptr,four._obj_ptr,Int32(2))
        _ = four^
        if equal < 0:
            var failure = stash_python_error()
            raise failure^
        return equal != 0
    var scalar = context["cache_scalar_context"]
    if py_get(scalar["modules"],"numpy") is not None:
        var classes = _classes(context)
        if sequence_dtype_instance(value,classes):
            return False  # dtype descriptor is never equal to integer four
    var dtype = _dtype(value,context)
    var kind = String(dtype.kind)
    if kind == "V":
        raise Error("TypeError: Cannot compare structured or void to non-void arrays.")
    var array = cls is native_numpy_type("ndarray",scalar)
    if array:
        var count = Int(py=value.size)
        if count != 1:
            if count == 0 and _release(context) < 202:
                training_scalar_warning("The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.",
                                        context["cache_deprecation_warning"],"cache_get",scalar)
                return True  # legacy `if empty_array != 4` is false
            if count == 0:
                raise Error("ValueError: The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.")
            raise Error("ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
        if kind == "O" or kind == "T" or kind == "M" or kind == "m":
            var item = _array_item(value,context)
            var result = _itemsize_four(item,context,depth+1)
            _ = item^
            return result
    if kind == "b" or kind == "S" or kind == "U" or kind == "M":
        return False
    if kind == "m":
        if _release(context) >= 205:
            training_scalar_warning("The 'generic' unit for NumPy timedelta is deprecated, and will raise an error in the future. This includes implicit conversion of bare integers (e.g. `+ 1`). Please use a specific unit instead.",
                                    context["cache_deprecation_warning"],"cache_get",scalar)
        return _temporal_word(value,context) == 4
    if kind == "i" or kind == "u" or kind == "f" or kind == "c":
        return native_scalar_buffer(value).equals_integer(4)
    raise Error("TypeError: cache itemsize requires native scalar metadata without Python equality hooks")


def _integer(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: cache integer metadata exceeds native nesting bounds")
    var cls = py_type(value)
    var scalar = context["cache_scalar_context"]
    if cls is context["arena_memoryview"]:
        native_release_admit(value,context)
    if _one_of(cls,context["cache_integer_types"]) or cls is context["arena_memoryview"] or _inherited_primitive(value,context):
        # CPython owns parsing of exact builtin strings and arbitrary-precision
        # integers. No supplied protocol or NumPy number slot reaches it.
        ref cpy = Python().cpython()
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_Long")(value._obj_ptr)
        if not pointer:
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=pointer)
    var dtype = _dtype(value,context)
    var kind = String(dtype.kind)
    var array = cls is native_numpy_type("ndarray",scalar)
    if array:
        var ndim = Int(py=value.ndim)
        if ndim > 0 and _release(context) >= 204:
            raise Error("TypeError: only 0-dimensional arrays can be converted to Python scalars")
        if Int(py=value.size) != 1:
            raise Error("TypeError: only size-1 arrays can be converted to Python scalars")
        if ndim > 0:
            training_scalar_warning("Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)",
                                    context["cache_deprecation_warning"],"cache_get",scalar)
        if kind == "c":
            raise Error("TypeError: int() argument must be a string, a bytes-like object or a real number, not 'complex'")
        if kind == "O" or kind == "S" or kind == "U" or kind == "T" or kind == "M" or kind == "m":
            var item = _array_item(value,context)
            var result = _integer(item,context,depth+1)
            _ = item^
            return result
    if kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c":
        if kind == "c":
            var exceptions = py_get(scalar["modules"],"numpy.exceptions")
            if exceptions is None or py_type(exceptions) is not scalar["module_type"]:
                raise Error("TypeError: complex metadata requires NumPy's native exception metadata")
            var category = py_get(native_numpy_module_fields(exceptions),"ComplexWarning")
            training_scalar_warning("Casting complex values to real discards the imaginary part",category,"cache_get",scalar)
        return native_scalar_buffer(value).to_python_number(True)
    if kind == "V":
        if dtype.fields is not None:
            raise Error("TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'")
        return _integer(training_void_bytes(value),context,depth+1)
    if kind == "S":
        return _integer(training_void_bytes(value),context,depth+1)
    if kind == "U":
        # Public Unicode API copies a native str subclass into exact builtin
        # storage without invoking its __str__ or NumPy numeric conversion.
        ref cpy = Python().cpython()
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyUnicode_FromObject")(value._obj_ptr)
        if not pointer:
            var failure = stash_python_error()
            raise failure^
        return _integer(PythonObject(from_owned=pointer),context,depth+1)
    if kind == "M" or kind == "m":
        var word = _temporal_word(value,context)
        # The layout string is read only for a value that is not NaT, as before.
        var layout = String("")
        if UInt64(word) != UInt64(1)<<63:
            layout = String(dtype.str)
        var name = _temporal_integer_name(word, kind, layout)
        if name.byte_length() == 0:
            return PythonObject(word)
        raise Error("TypeError: int() argument must be a string, a bytes-like object or a real number, not '"+name+"'")
    raise Error("TypeError: cache integer metadata requires native scalar values without Python conversion hooks")


@fieldwise_init
struct ArenaCacheKey(ImplicitlyCopyable, Movable):
    """A buffer's identity in a cache: its host address and its 4-byte word
    count."""
    var address: Int
    var words: Int


def arena_cache_view(model: PythonObject, context: PythonObject) raises -> ArenaOwnerView:
    """UNPACK of every DeviceCache operation: the native owner in the
    instance's `_owner` slot (one C slot read, one downcast)."""
    return arena_owner_held(context["cache_owner_get"], model, context["cache_class"], ARENA_OWNER_CACHE, "DeviceCache")


def arena_cache_key(source: PythonObject, context: PythonObject) raises -> ArenaCacheKey:
    # UNPACK: the caller's buffer metadata (`itemsize`, `_addr`, `size`) is
    # admitted natively; no key callback of a subclass exists or can run.
    if not _itemsize_four(_attribute(source,"itemsize",PythonObject(4),False,context),context):
        raise Error("ValueError: DeviceCache: a store slot holds 4-byte words")
    var address = _integer(_attribute(source,"_addr",PythonObject(None),True,context),context)
    var size = _integer(_attribute(source,"size",PythonObject(None),True,context),context)
    return ArenaCacheKey(_key_word(address, -1, -1), _key_word(size, 0, Int.MAX))


def _key_word(value: PythonObject, below: Int, above: Int) raises -> Int:
    """An admitted exact int as a key word. One outside the Int range names
    no buffer: `below` for a negative one, `above` for a positive one (lane
    gs-fix: an address past 2^63 or a size of -(2^100) raised "Python int too
    large to convert to C ssize_t" where the lookup answers no entry and
    id_of refuses an empty buffer, test_cache_get_native /
    test_cache_upload_native on the L40S install). An in-range value pays
    the one conversion it paid before."""
    try:
        return Int(py=value)
    except:
        return below if Bool(py=value < PythonObject(0)) else above


def arena_cache_get(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var view = arena_cache_view(model, context)
    var key = arena_cache_key(args[0], context)
    # RUN: one typed lookup; `get` counts no hit.
    var lease = ARENA_CACHES.acquire(arena_handle(view.word))
    var found = lease[].find(key.address, key.words, False)
    if found < 0:
        return PythonObject(None)
    return PythonObject(found)
