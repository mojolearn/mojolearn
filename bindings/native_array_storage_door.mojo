# SPDX-License-Identifier: Apache-2.0
# Allocate the existing array.array stores without calling its audited Python
# constructor. PySequence_Repeat is public CPython stable ABI; on an exact
# array.array its native array_repeat allocates a fresh ordinary array owner.
# Fixed one-word seeds are import-time metadata, never caller input or output.
from std.ffi import c_int, c_size_t, external_call
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.sys.info import CompilationTarget
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_cpython_table import cpython_table
from bindings.cpython_layout import (
    ARRAY_OB_SIZE, ARRAY_OB_ITEM, ARRAY_ALLOCATED, LAYOUT_ARRAY_BLOCK, CPYTHON_LAYOUT_MODE,
)
from bindings.py_version_tags import cpython_layout_groups


def _storage_metadata() raises -> PythonObject:
    var module = native_loaded_module("mojolearn._native_array_storage")
    ref cpy = Python().cpython()
    var fields = cpython_table()[].PyModule_GetDict(module._obj_ptr)
    if not fields:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=cpy.Py_NewRef(fields))
    _ = module  # retain the module until its dictionary reference is owned
    return result


def _storage_type(metadata: PythonObject) raises -> PythonObject:
    var cls = py_get(metadata, "ARRAY_TYPE")
    # Both metadata fields are Python-visible. A replacement class must be
    # refused before len/getitem/typecode or the sequence-repeat protocol.
    if py_type(cls) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: native array storage requires its extension type")
    var module = native_loaded_module("array")
    ref cpy = Python().cpython()
    var owner = cpython_table()[].PyType_GetModule(cls._obj_ptr)
    if not owner:
        cpython_table()[].PyErr_Clear()
        raise Error("TypeError: native array storage requires its extension type")
    if owner != module._obj_ptr:
        raise Error("TypeError: native array storage type has a different extension owner")
    return cls


#: Items in the zero block a large pre-3.12 store is repeated from: 4096
#: items is 4-32 KB, one memcpy each at streaming speed (cost reasoning:
#: the per-call overhead of a memcpy is amortized over a block, whatever
#: the array's shape).
comptime _ZERO_BLOCK_ITEMS = 4096


def _repeat_copies_per_item() raises -> Bool:
    # CPython 3.10 and 3.11 array_repeat: one memcpy per repeated copy;
    # 3.12 moved it to a bulk repeat. Read from the running interpreter.
    var info = native_loaded_module("sys").version_info
    return Int(py=info[0]) == 3 and Int(py=info[1]) < 12


def _zero_seed(code: String) raises -> PythonObject:
    """The fixed one-item zero store of `code` (import-time metadata),
    refused unless it is still an exact one-item array of that code whose
    raw bytes are all zero."""
    var metadata = _storage_metadata()
    var cls = _storage_type(metadata)
    var seeds = py_get(metadata, "ZERO_STORES")
    if py_type(seeds) is not py_type(Python.dict()):
        raise Error("TypeError: native array storage metadata is unavailable")
    var seed = py_get(seeds, PythonObject(code))
    if seed is None or py_type(seed) is not cls:
        raise Error("TypeError: unsupported native array storage typecode " + code)
    if len(seed) != 1 or String(seed.typecode) != code:
        raise Error("TypeError: native array storage seed changed")
    # Numerical equality would admit -0.0. Zero-filled buffers require every
    # raw byte to be zero, including the IEEE sign bit in float prototypes.
    var address = native_buffer_address(seed)
    var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(Int(py=seed.itemsize)):  # small-loop(seed.itemsize: scalar storage byte width): at most eight native storage bytes
        if bytes[i] != 0:
            raise Error("TypeError: native array storage seed changed")
    return seed


def native_array_zeros(code: String, count: Int) raises -> PythonObject:
    """A new ZERO-FILLED array.array of `count` items of `code`: for a
    destination that is written in part, accumulated into, or whose zeros are
    read. A destination that one native writer fills in full before anything
    reads it takes `native_array_unfilled` (no fill pass)."""
    if count < 0:
        raise Error("ValueError: native array element count must be nonnegative")
    var seed = _zero_seed(code)
    ref cpy = Python().cpython()
    if count >= 2 * _ZERO_BLOCK_ITEMS and _repeat_copies_per_item():
        # 0.8.66 regression C3 (knn fit 8.5x, iforest fit 39.6x, gbdt-rank
        # 1.24x): before CPython 3.12, array_repeat runs one memcpy per
        # repeated copy, so a one-item seed costs one call per element. A
        # zero block of _ZERO_BLOCK_ITEMS items is built first and repeated
        # whole (one memcpy per block), then the remainder (fewer than
        # _ZERO_BLOCK_ITEMS items) is appended. Every byte is still zero;
        # 3.12 and later already repeat in bulk and keep the one call below.
        var block_raw = cpython_table()[].PySequence_Repeat(seed._obj_ptr, _ZERO_BLOCK_ITEMS)
        if not block_raw:
            raise cpy.unsafe_get_error()
        var block = PythonObject(from_owned=block_raw)
        var whole_raw = cpython_table()[].PySequence_Repeat(block._obj_ptr, count // _ZERO_BLOCK_ITEMS)
        _ = block  # retain the block owner through the native repeat operation
        if not whole_raw:
            raise cpy.unsafe_get_error()
        var whole = PythonObject(from_owned=whole_raw)
        var rest = count % _ZERO_BLOCK_ITEMS
        if rest > 0:
            var tail_raw = cpython_table()[].PySequence_Repeat(seed._obj_ptr, rest)
            if not tail_raw:
                raise cpy.unsafe_get_error()
            var tail = PythonObject(from_owned=tail_raw)
            _ = py_call(whole.extend, tail)
        _ = seed  # retain the original owner through the native repeat operations
        return whole
    var result = cpython_table()[].PySequence_Repeat(seed._obj_ptr, count)
    _ = seed  # retain the original owner through the native repeat operation
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


# The array.array object words `native_array_unfilled` writes (CPython
# Modules/arraymodule.c `arrayobject`): the offsets and the versions they are
# written on are bindings/cpython_layout.mojo (THE table, group
# LAYOUT_ARRAY_BLOCK, checked by ctypes on 3.10-3.14 in
# tools/tests/test_cpython_layout.py). The layout is proven ONCE per process
# on a live seed (lane fb5-cpython-layout, 2026-11-17; it was probed on every
# call, one more buffer export per output array): a version without a row
# (free-threaded, 3.15+) or a build that fails the proof (Py_TRACE_REFS)
# keeps the zero-filled store.
comptime _ARRAY_WORD_SIZE = ARRAY_OB_SIZE // 8
comptime _ARRAY_WORD_ITEM = ARRAY_OB_ITEM // 8
comptime _ARRAY_WORD_ALLOCATED = ARRAY_ALLOCATED // 8


def _array_words(value: PythonObject) -> MutPointer[Int, MutUntrackedOrigin]:
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(value._obj_ptr))


def _array_layout_matches(seed: PythonObject) raises -> Bool:
    """True when the one-item seed's object words are where
    `native_array_unfilled` writes them: ob_size is 1, ob_item is the address
    its buffer export reports (a heap address no header word can equal), and
    allocated holds the item."""
    var words = _array_words(seed)
    var item = native_buffer_address(seed)
    var matched = (words[_ARRAY_WORD_SIZE] == 1 and words[_ARRAY_WORD_ITEM] == item
                   and words[_ARRAY_WORD_ALLOCATED] >= 1)
    _ = seed  # retain the seed while its words are read
    return matched


def _array_block_writable(seed: PythonObject) raises -> Bool:
    """True when this interpreter's table row lists the array block words and
    the one-time proof on a live seed passed (resolved once per process)."""
    var g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
    if g[].arrays != -2:
        return g[].arrays == 1
    var proven = (cpython_layout_groups() & LAYOUT_ARRAY_BLOCK) != 0 and _array_layout_matches(seed)
    g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
    g[].arrays = 1 if proven else 0
    return proven


def native_array_unfilled(code: String, count: Int) raises -> PythonObject:
    """A new array.array of `count` items of `code` whose contents are NOT
    initialised (lane fix-output-zero-fill, 2026-11-14): the destination of
    one native writer (a kernel download, a memcpy, a cast) that writes every
    item before anything reads it or the array escapes. Never for a
    destination written in part or accumulated into: those keep
    `native_array_zeros`.

    The array is a plain empty array of `code` (the public sequence-repeat of
    its seed, zero times) whose item block is then ONE PyMem_Malloc of
    count * itemsize bytes, installed as the array's own block (ob_item,
    allocated, ob_size). The array frees and resizes that block with
    PyMem_Free / PyMem_Realloc as it does every block it allocates itself, so
    the result is the same object `native_array_zeros` returns minus the fill
    pass. array.array has no public uninitialised constructor; 0.8.55 and the
    zero allocator filled every output (a memset or per-block memcpy over the
    whole destination: 80.8% of the gemm row and 86.7% of lm-forward's
    snapshot in the 2026-11-14 L40S profile) before the writer overwrote it.
    The object layout is proven once per process (`_array_block_writable`);
    when it is not, the zero-filled store is returned (the same bytes the
    writer then overwrites)."""
    if count < 0:
        raise Error("ValueError: native array element count must be nonnegative")
    if count == 0:
        return native_array_zeros(code, 0)
    var seed = _zero_seed(code)
    if not _array_block_writable(seed):
        return native_array_zeros(code, count)
    var itemsize = Int(py=seed.itemsize)
    if count > (Int.MAX // 2) // itemsize:
        raise Error("MemoryError: native array of " + String(count) + " items is too large")
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PySequence_Repeat(seed._obj_ptr, 0)
    _ = seed  # retain the original owner through the native repeat operation
    if not raw:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=raw)
    var words = _array_words(result)
    # A fresh empty array owns no block (CPython newarrayobject, size 0).
    if words[_ARRAY_WORD_SIZE] != 0 or words[_ARRAY_WORD_ITEM] != 0 or words[_ARRAY_WORD_ALLOCATED] != 0:
        return native_array_zeros(code, count)
    var block = cpython_table()[].PyMem_Malloc(count * itemsize)
    if block == 0:
        raise Error("MemoryError: native array of " + String(count) + " items could not be allocated")
    words[_ARRAY_WORD_ITEM] = block
    words[_ARRAY_WORD_ALLOCATED] = count
    words[_ARRAY_WORD_SIZE] = count
    return result


def native_scratch_bytes(nbytes: Int) raises -> PythonObject:
    # 0.8.66 regression C3 (gbdt-rank-pairlogit fit 1.24x, +405 ms on
    # istella): an UNINITIALIZED private byte store (a bytearray made by
    # PyByteArray_FromStringAndSize with a NULL source; no fill pass), for a
    # destination that one native call writes in full before anything reads
    # it and that never escapes the operation (never a fitted field, never a
    # returned Array). 0.8.55 used an uninitialized PyMem_RawMalloc store for
    # the same destinations. The data is CPython's allocation (16-byte
    # aligned), never resized, and alive while the returned object is.
    if nbytes < 0:
        raise Error("ValueError: native scratch byte count must be nonnegative")
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyByteArray_FromStringAndSize(0, nbytes if nbytes > 0 else 1)
    if not raw:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def native_scratch_address(scratch: PythonObject) raises -> Int:
    """The data address of a `native_scratch_bytes` store."""
    var address = cpython_table()[].PyByteArray_AsString(scratch._obj_ptr)
    if address == 0:
        ref cpy = Python().cpython()
        raise cpy.unsafe_get_error()
    return address


# TOMBSTONE: native_scratch_advise_huge (the madvise(MADV_HUGEPAGE) helper; its two callers were the arms MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE, LOSS in A/B round 2, and MOJOLEARN_XPREP_FAST_ARENA_HUGEPAGE, NOISE in A/B round 3) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
# Restore: git apply experiments/removed/MOJOLEARN_XPREP_FAST_ARENA_HUGEPAGE.patch; record in docs/TOMBSTONES.md.


def native_array_empty(code: String) raises -> PythonObject:
    return native_array_zeros(code, 0)


def native_array_from_bytes(code: String, source: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(source)
    if cls is not b.bytes and cls is not b.bytearray and cls is not b.memoryview:
        raise Error("TypeError: native array bytes require an exact native byte buffer")
    var result = native_array_empty(code)
    _ = py_call(result.frombytes, source)
    return result


def native_array_from_values(code: String, source: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(source)
    var metadata = _storage_metadata()
    var native_array = _storage_type(metadata)
    if cls is not b.list and cls is not b.tuple and cls is not native_array:
        raise Error("TypeError: native array values require a native array or exact numeric sequence")
    # Callers normalize NumPy/buffer leaves through their existing native input
    # paths. Reject Python scalar conversion protocols before native append can
    # invoke them. array.array iteration itself yields exact Python primitives.
    if cls is not native_array:
        # Borrowed slot reads and type-pointer compares: no boxed index or
        # PythonObject item per element (an array.array source needs no
        # check: its iteration yields exact primitives).
        ref cpy = Python().cpython()
        var t = cpython_table()
        var is_list = cls is b.list
        var int_type = b.int._obj_ptr
        var float_type = b.float._obj_ptr
        var bool_type = b.bool._obj_ptr
        for i in range(len(source)):  # driver-loop: over caller-provided scalar ABI types before native conversion
            var raw = t[].PyList_GetItem(source._obj_ptr, i) if is_list else t[].PyTuple_GetItem(source._obj_ptr, i)
            if not raw:
                raise cpy.unsafe_get_error()
            var item_type = cpy.PyObject_Type(raw)
            var exact = item_type == int_type or item_type == float_type or item_type == bool_type
            cpy.Py_DecRef(item_type)
            if not exact:
                raise Error("TypeError: native array values cannot invoke scalar conversion hooks")
    var result = native_array_empty(code)
    # array(code, another_array) converts each scalar when their typecodes
    # differ. array.extend(another_array) instead refuses that input. Preserve
    # constructor conversion through an exact native list, whose leaves were
    # admitted above; no Python iterator or scalar-conversion hook is called.
    var values = source
    if cls is native_array and String(source.typecode) != code:
        values = py_call(b.list, source)
    _ = py_call(result.extend, values)
    return result
