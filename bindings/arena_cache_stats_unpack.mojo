# SPDX-License-Identifier: Apache-2.0
# Read-only cache operations. No upload, release or hidden-owner traversal.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_item
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.sequence_dtype_native_door import sequence_dtype_create
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import native_namespace_string_keys, native_mapping_string_keys
from bindings.p3_array_arena_records_unpack import P3_STATS_LAYOUT, p3_layout_admitted, p3_layout_admit, P3_DICT_CACHE_KEYS, p3_dict_admitted, p3_dict_admit


def _string_keys(mapping: PythonObject, context: PythonObject) raises:
    # p3-array-arena: exact dicts and class mappingproxies are walked in place
    # (no key list); any other mapping keeps the list walk. Same refusal.
    var b = context["arena_native_types"]
    var cls = py_type(mapping)
    if cls is b["dict"]:
        if not native_namespace_string_keys(mapping._obj_ptr, b["str"]):
            raise Error("TypeError: cache metadata keys must be native strings")
        return
    if cls is py_type(py_call(context["cache_type_dict"], b["object"])):
        if not native_mapping_string_keys(mapping, b["str"]):
            raise Error("TypeError: cache metadata keys must be native strings")
        return
    var keys = py_call(b["list"],mapping)
    for i in range(len(keys)):  # driver-loop: admit metadata keys before lookup can invoke equality
        if py_type(keys[i]) is not b["str"]:
            raise Error("TypeError: cache metadata keys must be native strings")


def _mapping(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = py_call(context["cache_type_dict"],cls)
    _string_keys(result,context)
    return result


struct HashKeyTypes(Movable):
    """The admitted key types of one cache operation, read once for all the
    keys it walks: the builtin hashing types, the tuple type and (on first
    need) the two NumPy string scalar types."""
    var types: List[PythonObject]
    var tuple_type: PythonObject
    var strings_ready: Bool
    var strings: List[PythonObject]

    def __init__(out self, context: PythonObject) raises:
        self.types = List[PythonObject]()
        var types = context["cache_hash_types"]
        for i in range(len(types)):  # driver-loop: exact immutable builtin hashing and equality slots, read once
            self.types.append(types[i])
        self.tuple_type = context["arena_native_types"]["tuple"]
        self.strings_ready = False
        self.strings = List[PythonObject]()


def _hash_key(value: PythonObject, context: PythonObject) raises:
    var types = HashKeyTypes(context)
    _hash_key(value, context, types)


def _hash_key(value: PythonObject, context: PythonObject, mut known: HashKeyTypes) raises:
    # Exact tuples cannot form a cycle without a non-tuple container. Walk
    # them iteratively so admission does not impose an arbitrary depth limit
    # before CPython performs the actual native hashing/equality operation.
    var pending = List[PythonObject]()
    pending.append(value)
    while len(pending):  # driver-loop: explicit native work stack over immutable key metadata
        var item = pending.pop()
        var cls = py_type(item)
        var native = False
        for i in range(len(known.types)):  # driver-loop: exact immutable builtin hashing and equality slots
            native = native or cls is known.types[i]
        if native:
            continue
        if cls is known.tuple_type:
            # A tuple is never one of the two NumPy string scalar types below;
            # walk it without building their descriptors for every node.
            for i in range(len(item)):  # driver-loop: tuple hashing must not dispatch to an application key
                pending.append(item[i])
            continue
        # The earlier read-only owner admitted exact NumPy byte/unicode
        # scalar prefixes. Preserve that surface when get/stats share this
        # iterative key walk; these immutable scalar slots cannot call an
        # application's __hash__ or __eq__. Creating the native descriptor
        # validates NumPy's connected dtype entry without importing NumPy or
        # constructing/converting the caller's scalar. Subclasses stay out.
        # The two descriptors are created once per operation, on first need.
        if not known.strings_ready:
            var scalar = context["cache_scalar_context"]
            if py_get(scalar["modules"], "numpy") is not None:
                for code in ["S", "U"]:  # driver-loop: the two supported immutable NumPy string scalar identities
                    known.strings.append(sequence_dtype_create(PythonObject(code), scalar).type)
            known.strings_ready = True
        for i in range(len(known.strings)):  # small-loop(two NumPy string scalar identities)
            native = native or cls is known.strings[i]
        if native:
            continue
        raise Error("TypeError: cache keys require native immutable values without Python equality or hash hooks")


def _admit_entry_keys(entries: PythonObject, context: PythonObject) raises:
    """Refuse application hashing/equality among the keys of one exact dict.

    Walks the dictionary in place with PyDict_Next (no key list, no hashing
    or equality hooks). The keys DeviceCache itself publishes are exact
    (int, int) tuples: those are admitted by three native type identity
    checks. Any other key takes the full `_hash_key` admission, so the
    accepted and refused key sets are unchanged.

    p3-array-arena: the walk runs only when the dictionary changed since its
    last full admission (dictionary watcher, Python >= 3.12)."""
    if p3_dict_admitted(P3_DICT_CACHE_KEYS, entries):
        return
    ref cpy = Python().cpython()
    var tuple_type = context["arena_native_types"]["tuple"]
    var int_type = context["arena_native_types"]["int"]
    var tuple_size = cpy.lib.get_function[Int]("PyTuple_Size")
    var tuple_item = cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")
    var next_item = cpy.lib.get_function[Int32]("PyDict_Next")
    var position = Int(0)
    var key = PyObjectPtr()
    var value = PyObjectPtr()
    var others = List[PythonObject]()
    while next_item(entries._obj_ptr, MutPointer(to=position), MutPointer(to=key), MutPointer(to=value)) != 0:  # driver-loop: over current exact cache dictionary keys
        var stored = PythonObject(from_borrowed=key)
        if py_type(stored) is tuple_type and tuple_size(key) == 2:
            var first = PythonObject(from_borrowed=tuple_item(key, 0))
            var second = PythonObject(from_borrowed=tuple_item(key, 1))
            if py_type(first) is int_type and py_type(second) is int_type:
                continue
        others.append(stored^)
    if len(others) == 0:
        p3_dict_admit(P3_DICT_CACHE_KEYS, entries)
        return
    # Full admission runs after the walk, so nothing it reads can disturb
    # the in-place dictionary iteration.
    var key_types = HashKeyTypes(context)
    for i in range(len(others)):  # driver-loop: keys not of the published (int, int) shape
        _hash_key(others[i], context, key_types)
    p3_dict_admit(P3_DICT_CACHE_KEYS, entries)
