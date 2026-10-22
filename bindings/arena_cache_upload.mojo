# SPDX-License-Identifier: Apache-2.0
# Whole public DeviceCache.id_of. Constructor/release/close remain separate.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.arena_cache_stats import _attribute, _mapping, _hash_key
from bindings.arena_cache_get import arena_cache_key
from bindings.optimizer_public_words import stash_python_error, PENDING_PYTHON_ERROR, PYTHON_ERROR_PENDING


def _entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var entries = _attribute(model,"_entries",PythonObject(None),True,context)
    if py_type(entries) is not context["arena_native_types"]["dict"]:
        raise Error("TypeError: DeviceCache entries require a native dictionary")
    var keys = py_call(context["arena_native_types"]["list"],entries)
    for i in range(len(keys)):  # driver-loop: refuse application hashing/equality before a native dictionary lookup
        _hash_key(keys[i],context)
    return entries


def _increment(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Cache counters are ordinary integers, with no machine-width bound.
    # Read the exact builtin's binary representation, perform the addition
    # in Mojo, and parse the already-computed result with the fixed native
    # integer constructor. No NumPy arithmetic/conversion or Python __add__.
    var b = context["arena_native_types"]
    if py_type(value) is not b["int"] and py_type(value) is not b["bool"]:
        # Preserve the incumbent native float/complex counter forms and
        # ordinary builtin error text. Exact retained builtin slots cannot
        # invoke application numeric hooks; NumPy arithmetic is not admitted.
        var types = context["cache_counter_types"]
        var admitted = False
        for i in range(len(types)):  # driver-loop: fixed native counter/value-error identities
            admitted = admitted or py_type(value) is types[i]
        if not admitted:
            raise Error("TypeError: DeviceCache counters require native builtin values without Python arithmetic hooks")
        ref cpy = Python().cpython()
        var one = PythonObject(1)
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_InPlaceAdd")(value._obj_ptr,one._obj_ptr)
        _ = one^
        if not pointer:
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=pointer)
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyNumber_ToBase")(value._obj_ptr,Int32(2))
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var encoded = PythonObject(from_owned=pointer)
    var text = String(encoded)
    var negative = text.startswith("-")
    var offset = 3 if negative else 2
    var digits = List[UInt8]()
    for i in range(offset,text.byte_length()):  # driver-loop: exact arbitrary-precision binary digits
        digits.append(text.as_bytes()[i])
    var carry = True
    for i in range(len(digits)-1,-1,-1):  # driver-loop: increment a positive magnitude or decrement a negative one
        if not carry:
            break
        if negative:
            if digits[i] == 49:
                digits[i] = 48
                carry = False
            else:
                digits[i] = 49
        elif digits[i] == 48:
            digits[i] = 49
            carry = False
        else:
            digits[i] = 48
    var result = String("-") if negative else String("")
    if carry and not negative:
        result += "1"
    for i in range(len(digits)):  # driver-loop: publish the native integer's computed binary spelling
        result += "1" if digits[i] == 49 else "0"
    return py_call(b["int"],PythonObject(result),2)


def _counter_next(model: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    # Validate setters as well as readers before any cache mutation. Native
    # member descriptors and ordinary instance dictionaries retain their
    # normal behavior; application descriptors/setattr hooks cannot execute.
    var ancestry = py_call(context["cache_type_mro"],py_type(model))
    for i in range(len(ancestry)):  # driver-loop: fixed native attribute-write surface
        var mapping = _mapping(ancestry[i],context)
        var hook = py_get(mapping,"__setattr__")
        if hook is not None and hook is not context["cache_object_setattr"]:
            raise Error("TypeError: DeviceCache counters cannot invoke a custom attribute setter")
    return _increment(_attribute(model,name,PythonObject(None),True,context),context)


def _counter_set(model: PythonObject, name: String, value: PythonObject) raises:
    ref cpy = Python().cpython()
    var key = PythonObject(name)
    var status = cpy.lib.get_function[Int32]("PyObject_GenericSetAttr")(model._obj_ptr,key._obj_ptr,value._obj_ptr)
    _ = key^
    if status != 0:
        var failure = stash_python_error()
        raise failure^


def _hit(model: PythonObject, key: PythonObject, context: PythonObject) raises -> PythonObject:
    var entries = _entries(model,context)
    var hit = py_get(entries,key)
    if hit is None:
        return PythonObject(None)
    # Match the established order: increment before indexing the hit. A
    # malformed empty entry still increments the hit counter then raises.
    var next_hits = _counter_next(model,"hits",context)
    var types = context["cache_entry_types"]
    var admitted = False
    for i in range(len(types)):  # driver-loop: native indexing and ordinary non-subscriptable error slots
        admitted = admitted or py_type(hit) is types[i]
    if not admitted:
        raise Error("TypeError: DeviceCache entries require native indexing without Python hooks")
    if py_type(hit) is context["arena_native_types"]["dict"]:
        var keys = py_call(context["arena_native_types"]["list"],hit)
        for i in range(len(keys)):  # driver-loop: native dictionary indexing cannot compare foreign keys
            _hash_key(keys[i],context)
    _counter_set(model,"hits",next_hits)
    ref cpy = Python().cpython()
    var zero = PythonObject(0)
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(hit._obj_ptr,zero._obj_ptr)
    _ = zero^
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    # A separate hit flag preserves a legitimate None-valued stored slot.
    return Python.tuple(PythonObject(from_owned=pointer))


def _route(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var put = _attribute(model,"_put",PythonObject(None),True,context)
    var free = _attribute(model,"_free",PythonObject(None),True,context)
    var routes = context["cache_store_routes"]
    for i in range(len(routes)):  # driver-loop: exact import-connected native put/free pairs only
        var pair = routes[i]
        if put is pair[0] and free is pair[1]:
            return pair
    raise Error("TypeError: DeviceCache upload requires a connected native x_prep or x_metrics store; Python callbacks cannot execute")


def _prepare(model: PythonObject, context: PythonObject) raises -> PythonObject:
    # All app-owned instance dictionaries/entries/counter temporaries die in
    # this scope before the put entry releases the GIL. Only fixed native
    # functions and caller-owned key/source/model arguments cross that call.
    _ = _counter_next(model,"uploads",context)
    return _route(model,context)


def _native_call(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(entry._obj_ptr,args._obj_ptr,PyObjectPtr())
    _ = args
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


struct CachePendingError(Movable):
    # Own the fetched triple in fixed fields while cleanup can release the
    # GIL. Avoid allocating/copying a List after a resident slot needs cleanup.
    var kind: Int
    var value: Int
    var traceback: Int

    def __init__(out self):
        self.kind = 0
        self.value = 0
        self.traceback = 0


def _take_pending(error: Error) raises -> CachePendingError:
    var saved = CachePendingError()
    if String(error) == PYTHON_ERROR_PENDING:
        var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
        saved.kind = pending[].words[0]
        saved.value = pending[].words[1]
        saved.traceback = pending[].words[2]
        pending[].words[0] = 0
        pending[].words[1] = 0
        pending[].words[2] = 0
    return saved^


def _drop_pending(mut saved: CachePendingError) raises:
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[NoneType]("Py_DecRef")
    if saved.kind:
        release(saved.kind)
        saved.kind = 0
    if saved.value:
        release(saved.value)
        saved.value = 0
    if saved.traceback:
        release(saved.traceback)
        saved.traceback = 0


def _restore_pending(mut saved: CachePendingError) raises:
    if saved.kind:
        var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
        pending[].words[0] = saved.kind
        pending[].words[1] = saved.value
        pending[].words[2] = saved.traceback
        saved.kind = 0
        saved.value = 0
        saved.traceback = 0


def _publish(model: PythonObject, source: PythonObject, key: PythonObject,
             slot: PythonObject, route: PythonObject, context: PythonObject,
             mut published: Bool) raises:
    # A put may release the GIL. Re-read the actual current cache instead of
    # publishing into a stale dictionary retained across another thread's
    # close/reinitialization. Never overwrite a concurrently published owner.
    var current = _route(model,context)
    if current[0] is not route[0] or current[1] is not route[1]:
        raise Error("RuntimeError: DeviceCache native store changed during upload")
    var entries = _entries(model,context)
    if py_get(entries,key) is not None:
        raise Error("RuntimeError: DeviceCache entry was published concurrently during upload")
    var next_uploads = _counter_next(model,"uploads",context)
    var entry = Python.tuple(slot,source)
    py_setitem(entries,key,entry)
    published = True
    # The published entry now owns the resource. If a native setter fails,
    # the entry remains visible and must not point at a freed slot.
    _counter_set(model,"uploads",next_uploads)


def arena_cache_id_of(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var source = args[0]
    var key = arena_cache_key(model,source,context)
    var hit = _hit(model,key,context)
    if hit is not None:
        return hit[0]
    _ = hit^
    # Native exact integer ordering avoids imposing a machine-word limit
    # before the already-established ABI performs its own range admission.
    if Bool(py=key[1] < PythonObject(1)):
        raise Error("ValueError: DeviceCache: an empty buffer has no device copy")
    var route = _prepare(model,context)
    var slot = _native_call(route[0],key)
    var published = False
    try:
        _publish(model,source,key,slot,route,context,published)
    except error:
        if not published:
            # Cleanup releases the GIL. The primary exception must be owned
            # locally, not left in the shared native boundary's pending slot
            # where an unrelated thread's operation could overwrite it.
            var primary = _take_pending(error)
            try:
                # Use the originally admitted free entry, even if the
                # cache was reinitialized while upload released the GIL.
                _ = _native_call(route[1],Python.tuple(slot))
            except cleanup_error:
                var cleanup = _take_pending(cleanup_error)
                _drop_pending(primary)
                _drop_pending(cleanup)
                raise Error("RuntimeError: DeviceCache publication failed: "+String(error)+"; unpublished slot "+String(slot)+" cleanup also failed: "+String(cleanup_error))
            _restore_pending(primary)
        raise error^
    return slot
