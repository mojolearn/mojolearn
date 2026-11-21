# SPDX-License-Identifier: Apache-2.0
# Read-only cache operations. No upload, release or hidden-owner traversal.
# The PACK half of the former `bindings/arena_cache_stats.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/arena_cache_stats_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_item
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.sequence_dtype_native_door import sequence_dtype_create
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import native_namespace_string_keys, native_mapping_string_keys
from bindings.p3_array_arena_records_unpack import P3_STATS_LAYOUT, p3_layout_admitted, p3_layout_admit
from bindings.arena_owner_door import ARENA_OWNER_CACHE, arena_owner_held
from bindings.arena_scope_door import arena_scope_object, arena_scope_word
from bindings.arena_cache_run import arena_scope_stats
from core.arena_state import ARENA_CACHES, arena_handle


from bindings.arena_cache_stats_unpack import (
    _mapping,
    _string_keys,
)


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
    # p3-array-arena: the class rows' key and hook admission (independent of
    # `name`) is kept per class state (type watcher; any modification of the
    # class or an ancestor re-runs it). The name lookup itself stays per call.
    var admitted = p3_layout_admitted(P3_STATS_LAYOUT, cls, context["cache_type_mro"])
    var namespace = False
    if not admitted:
        for i in range(len(ancestry)):  # driver-loop: a retained native namespace slot applies only to its own owner ancestry
            namespace = namespace or ancestry[i] is context["cache_namespace_class"]
    # Optional, not a PythonObject(None) sentinel: its presence is the old `found` flag (modular/modular#7546).
    var selected = Optional[PythonObject]()
    for i in range(len(ancestry)):  # driver-loop: ordinary attribute lookup without calling Python hooks
        var mapping = py_call(context["cache_type_dict"],ancestry[i]) if admitted else _mapping(ancestry[i],context)
        if not admitted:
            var hook = py_get(mapping,"__getattribute__")
            # CPython 3.11 exposes a distinct native SimpleNamespace wrapper;
            # newer CPython inherits object's wrapper. Both perform ordinary
            # native lookup. Never admit arbitrary C descriptors by type/name,
            # or a namespace wrapper transplanted onto an unrelated class.
            var native_hook = hook is None or hook is context["cache_object_getattribute"] or (namespace and hook is context["cache_namespace_getattribute"])
            if not native_hook or py_get(mapping,"__getattr__") is not None:
                raise Error("TypeError: cache metadata requires native attribute access; Python hooks cannot execute")
        if not selected and py_contains(mapping,name):
            selected = mapping[name]
            if admitted:
                break
    if not admitted:
        p3_layout_admit(P3_STATS_LAYOUT, cls, context["cache_type_mro"])
    var found = Bool(selected)
    var member = found and py_type(selected.value()) is context["cache_member_type"]
    var descriptor = False
    var data_descriptor = False
    if found and not member:
        var descriptor_mro = py_call(context["cache_type_mro"],py_type(selected.value()))
        for i in range(len(descriptor_mro)):  # driver-loop: inspect descriptor slots without invoking them
            var mapping = _mapping(descriptor_mro[i],context)
            descriptor = descriptor or py_contains(mapping,"__get__")
            data_descriptor = data_descriptor or py_contains(mapping,"__set__") or py_contains(mapping,"__delete__")
    if member:
        # Exact native member descriptors never dispatch to Python code. A
        # foreign member raises its normal native owner/type error.
        ref cpy = Python().cpython()
        var key = name
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(value._obj_ptr,key._obj_ptr)
        _ = key^
        if not pointer:
            if not required and cpython_table()[].PyErr_ExceptionMatches(context["cache_attribute_error"]._obj_ptr) != 0:
                cpython_table()[].PyErr_Clear()
                return default
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=pointer)
    if descriptor and data_descriptor:
        raise Error("TypeError: cache metadata cannot invoke Python or unregistered native descriptors")
    ref cpy = Python().cpython()
    var fields_pointer = cpython_table()[].PyObject_GenericGetDict(value._obj_ptr,PyObjectPtr())
    if fields_pointer:
        var fields = PythonObject(from_owned=fields_pointer)
        if py_type(fields) is not b["dict"]:
            raise Error("TypeError: cache metadata requires an ordinary instance dictionary")
        _string_keys(fields,context)
        if py_contains(fields,name):
            return py_item(fields, name)  # tail-call fix (2026-11-12): a std subscript returned directly is emitted as an LLVM tail call; LLVM may drop the borrowed key's stack store (Mojo 1.2.0). Read through the C API.
    elif cpython_table()[].PyErr_ExceptionMatches(context["cache_attribute_error"]._obj_ptr) != 0:
        cpython_table()[].PyErr_Clear()
    else:
        var failure = stash_python_error()
        raise failure^
    if found:
        if descriptor:
            raise Error("TypeError: cache metadata cannot invoke Python or unregistered native descriptors")
        return selected.value()
    if required:
        raise Error("AttributeError: cache metadata has no '" + String(name) + "' attribute")
    return default


def arena_cache_stats(model: PythonObject, operation: String, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # Lane arena (2026-11-21): reads of the native cache and scope states.
    var b = context["arena_native_types"]
    if operation == "resident_stats":
        # {prefix: (uploads, hits)} of the calling thread's scope, in the
        # order its caches were created, or None outside a scope.
        var scope = arena_scope_object(model, context["cache_local_type"], b["str"])
        if scope is None:
            return PythonObject(None)
        var counts = arena_scope_stats(arena_scope_word(scope, context["cache_scope_owner_get"],
                                                        context["cache_scope_class"]))
        var result = Python.dict()
        for i in range(len(counts)):  # driver-loop: over the scope's caches (one per binding family and mode)
            py_setitem(result, PythonObject(counts[i].prefix),
                       Python.tuple(PythonObject(counts[i].uploads), PythonObject(counts[i].hits)))
        return result
    var view = arena_owner_held(context["cache_owner_get"], model, context["cache_class"], ARENA_OWNER_CACHE,
                                "DeviceCache")
    var lease = ARENA_CACHES.acquire(arena_handle(view.word))
    if operation == "cache_len":
        return PythonObject(lease[].live)
    if operation == "cache_attribute":
        # The read-only public counters: native integers.
        var name = args[0]
        if py_type(name) is not b["str"]:
            raise Error("TypeError: DeviceCache attribute names are native strings")
        var which = String(py=name)
        if which == "hits":
            return PythonObject(lease[].hits)
        if which == "uploads":
            return PythonObject(lease[].uploads)
        raise Error("AttributeError: 'DeviceCache' object has no attribute '" + which + "'")
    raise Error("ValueError: unknown native cache-statistics operation")
