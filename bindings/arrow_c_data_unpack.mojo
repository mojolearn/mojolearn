# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UNPACK of an Arrow-backed pandas column through the Arrow C data
interface (lane arrow-strings, 2026-11-21; AGENTS.md "Data-interchange
protocols at the boundary", Andrew 2026-11-25).

pandas 3 stores text as Arrow by default when pyarrow is installed
(`ArrowStringArray`; pandas 2 `string[pyarrow]` and `ArrowDtype` columns
too). Such a column is admitted here, at the boundary, and its buffers are
read natively afterwards (`gbdt/data/category_map.mojo`):

  1. admission by exact class before any call: the column's array is the
     loaded `ArrowStringArray`, `ArrowStringArrayNumpySemantics` or
     `ArrowExtensionArray` class (pandas is never imported); its Arrow
     storage, the instance field `_pa_array` (`_data` before pandas 2.1),
     is an exact `pyarrow.lib.ChunkedArray`; each chunk an exact
     `pyarrow.lib.StringArray`, `LargeStringArray` or `DictionaryArray`;
  2. one protocol call per chunk: `chunk.__arrow_c_array__()` (the Arrow
     PyCapsule interface, pyarrow 14 and later), which returns the two
     capsules named `arrow_schema` and `arrow_array`; an error it raises is
     propagated as the Python error it is. The chunk list is read once per
     column (`ChunkedArray.chunks`);
  3. the ArrowSchema and ArrowArray structs are read as words (the C data
     interface ABI, 64-bit pointers): the format string, the offset, the
     length, the buffers (validity, offsets, data; for a dictionary array the
     indices and the dictionary's own three buffers);
  4. the capsules are kept alive by the caller until every native read is
     done, then dropped: each capsule's destructor calls its struct's
     release callback (the PyCapsule interface's lifetime rule for a
     consumer that does not move the struct out), which frees what the
     export retained. Nothing here takes the structs over.

The structs (Arrow C data interface specification, stable ABI):

    struct ArrowSchema {                 struct ArrowArray {
      const char* format;      // 0        int64_t length;          // 0
      const char* name;        // 1        int64_t null_count;      // 1
      const char* metadata;    // 2        int64_t offset;          // 2
      int64_t flags;           // 3        int64_t n_buffers;       // 3
      int64_t n_children;      // 4        int64_t n_children;      // 4
      ArrowSchema** children;  // 5        const void** buffers;    // 5
      ArrowSchema* dictionary; // 6        ArrowArray** children;   // 6
      void (*release)(...);    // 7        ArrowArray* dictionary;  // 7
      void* private_data;      // 8        void (*release)(...);    // 8
    };                                     void* private_data;      // 9
                                         };

Formats read: "u" (utf8, int32 offsets), "U" (large utf8, int64 offsets),
and a dictionary whose index format is one of "c" "C" "s" "S" "i" "I" "l"
"L" (int8 .. uint64) over "u" / "U" values. Anything else is refused by
name with its format string.
"""

from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_item
from bindings.py_cpython_table import cpython_table
from bindings.classical_pandas_door import _pd_class, _pd_fields
from bindings.native_loaded_module_door import native_loaded_module

comptime _Words = MutPointer[Int, MutUntrackedOrigin]
comptime _Bytes = MutPointer[UInt8, MutUntrackedOrigin]

# ArrowSchema words
comptime _SCHEMA_FORMAT = 0
comptime _SCHEMA_DICTIONARY = 6
comptime _SCHEMA_RELEASE = 7
# ArrowArray words
comptime _ARRAY_LENGTH = 0
comptime _ARRAY_OFFSET = 2
comptime _ARRAY_N_BUFFERS = 3
comptime _ARRAY_BUFFERS = 5
comptime _ARRAY_DICTIONARY = 7
comptime _ARRAY_RELEASE = 8


@fieldwise_init
struct ArrowTextPiece(Copyable, Movable):
    """One chunk of an Arrow text column as plain addresses. `index_size`
    0: a string (`wide` False) or large_string array, `offsets` / `data` its
    buffers. `index_size` 1, 2, 4 or 8: a dictionary array, `offsets` its
    index buffer, and the `dict_*` fields its dictionary of `dict_rows`
    text values. `validity` 0 = no nulls."""
    var rows: Int
    var offset: Int
    var validity: Int
    var offsets: Int
    var data: Int
    var wide: Bool
    var index_size: Int
    var index_signed: Bool
    var dict_rows: Int
    var dict_offset: Int
    var dict_validity: Int
    var dict_offsets: Int
    var dict_data: Int
    var dict_wide: Bool


def arrow_backed_pandas(values: PythonObject) raises -> Bool:
    """Whether a pandas column's array is one of pandas' Arrow-backed
    classes, by exact loaded identity (an absent class matches nothing)."""
    var cls = py_type(values)
    return cls is _pd_class("pandas.core.arrays.string_arrow", "ArrowStringArray") or cls is _pd_class(
        "pandas.core.arrays.string_arrow", "ArrowStringArrayNumpySemantics") or cls is _pd_class(
        "pandas.core.arrays.arrow.array", "ArrowExtensionArray")


def pandas_class_name(cls: PythonObject) raises -> String:
    """`module.qualname` of a class for a refusal, read by `type`'s own
    descriptors (no metaclass hook). Built only when a refusal is raised."""
    var b = native_loaded_module("builtins")
    var module = py_call(py_item(b.type.__dict__, PythonObject("__module__")).__get__, cls)
    var qual = py_call(py_item(b.type.__dict__, PythonObject("__qualname__")).__get__, cls)
    return String(py=module) + "." + String(py=qual)


def _format_text(address: Int) -> String:
    """The Arrow format string at `address` for a refusal (at most 16
    bytes, printable ASCII only)."""
    var out = String()
    if address == 0:
        return out^
    var p = _Bytes(unsafe_from_address=address)
    for i in range(16):  # small-loop(format: an Arrow format string): bounded by 16 bytes
        var c = Int(p[i])
        if c == 0:
            break
        if c >= 32 and c < 127:
            out += chr(c)
        else:
            out += "?"
    return out^


def _format_is_text(address: Int, mut wide: Bool) -> Bool:
    """Whether an Arrow format string is "u" (utf8) or "U" (large utf8);
    `wide` is set for "U"."""
    if address == 0:
        return False
    var p = _Bytes(unsafe_from_address=address)
    var c = Int(p[0])
    if c == 0 or Int(p[1]) != 0:
        return False
    if c == 117:  # 'u'
        wide = False
        return True
    if c == 85:  # 'U'
        wide = True
        return True
    return False


def _format_index(address: Int, mut signed: Bool) -> Int:
    """The byte width of an Arrow integer format of one character ("c" "s"
    "i" "l" signed, "C" "S" "I" "L" unsigned), or 0."""
    if address == 0:
        return 0
    var p = _Bytes(unsafe_from_address=address)
    var c = Int(p[0])
    if c == 0 or Int(p[1]) != 0:
        return 0
    signed = c >= 97
    if c == 99 or c == 67:  # 'c' 'C'
        return 1
    if c == 115 or c == 83:  # 's' 'S'
        return 2
    if c == 105 or c == 73:  # 'i' 'I'
        return 4
    if c == 108 or c == 76:  # 'l' 'L'
        return 8
    return 0


def _capsule(capsule: PythonObject, var name: String, what: String) raises -> Int:
    """The struct address a PyCapsule of the Arrow PyCapsule interface
    holds, after checking its name."""
    var t = cpython_table()
    var at = Int(name.as_c_string_slice().unsafe_ptr())
    if Int(t[].PyCapsule_IsValid(capsule._obj_ptr, at)) == 0:
        raise Error("TypeError: mojolearn: " + what + " returned no '" + name + "' capsule from "
                    + "__arrow_c_array__")
    var address = Int(t[].PyCapsule_GetPointer(capsule._obj_ptr, at))
    if address == 0:
        raise Python().cpython().unsafe_get_error()
    _ = name^
    return address


def _text_buffers(array: Int, what: String, mut validity: Int, mut offsets: Int, mut data: Int) raises:
    """The three buffers of a string / large_string ArrowArray."""
    var a = _Words(unsafe_from_address=array)
    if a[_ARRAY_N_BUFFERS] != 3:
        raise Error("TypeError: mojolearn: " + what + " is an Arrow string array with "
                    + String(a[_ARRAY_N_BUFFERS]) + " buffers (3 expected)")
    var buffers = _Words(unsafe_from_address=a[_ARRAY_BUFFERS])
    validity = buffers[0]
    offsets = buffers[1]
    data = buffers[2]
    if offsets == 0:
        raise Error("TypeError: mojolearn: " + what + " is an Arrow string array without offsets")


def arrow_text_pieces(values: PythonObject, what: String, mut keep: List[PythonObject],
                      mut total: Int) raises -> List[ArrowTextPiece]:
    """The chunks of an Arrow-backed pandas text column (`arrow_backed_pandas`
    is True) as plain addresses; `total` = their rows. The capsules are
    appended to `keep`: the caller holds them until its native reads end,
    then drops them (their destructors call the structs' release
    callbacks)."""
    var b = native_loaded_module("builtins")
    var fields = _pd_fields(values)
    var chunked = py_get(fields, "_pa_array")
    if chunked is None:
        chunked = py_get(fields, "_data")
    var chunked_class = _pd_class("pyarrow.lib", "ChunkedArray")
    if chunked_class is None or py_type(chunked) is not chunked_class:
        raise Error("TypeError: mojolearn: " + what + " is an Arrow-backed pandas array whose storage is not "
                    + "an exact pyarrow ChunkedArray")
    var string_class = _pd_class("pyarrow.lib", "StringArray")
    var large_class = _pd_class("pyarrow.lib", "LargeStringArray")
    var dictionary_class = _pd_class("pyarrow.lib", "DictionaryArray")
    var chunks = chunked.chunks
    if py_type(chunks) is not b.list:
        raise Error("TypeError: mojolearn: " + what + " is a pyarrow ChunkedArray whose chunks are not a list")
    keep.append(chunked)
    keep.append(chunks)
    var pieces = List[ArrowTextPiece]()
    total = 0
    for c in range(len(chunks)):  # driver-loop: over the Arrow chunks of one column (each chunk's rows run on every worker)
        var chunk = chunks[c]
        var ccls = py_type(chunk)
        if ccls is not string_class and ccls is not large_class and ccls is not dictionary_class:
            raise Error("TypeError: mojolearn: " + what + " is an Arrow column of " + pandas_class_name(ccls)
                        + "; an Arrow column is admitted as string, large_string or a dictionary of those")
        # the protocol call: the chunk exports its schema and array
        var pair = py_call(chunk.__arrow_c_array__)
        keep.append(pair)
        if py_type(pair) is not b.tuple or len(pair) != 2:
            raise Error("TypeError: mojolearn: " + what + "'s __arrow_c_array__ did not return two capsules")
        var schema = _capsule(pair[0], "arrow_schema", what)
        var array = _capsule(pair[1], "arrow_array", what)
        var s = _Words(unsafe_from_address=schema)
        var a = _Words(unsafe_from_address=array)
        if s[_SCHEMA_RELEASE] == 0 or a[_ARRAY_RELEASE] == 0:
            raise Error("TypeError: mojolearn: " + what + "'s Arrow export is already released")
        var rows = a[_ARRAY_LENGTH]
        var offset = a[_ARRAY_OFFSET]
        if rows < 0 or offset < 0:
            raise Error("TypeError: mojolearn: " + what + " is an Arrow array of negative length or offset")
        var wide = False
        var validity = 0
        var offsets = 0
        var data = 0
        if _format_is_text(s[_SCHEMA_FORMAT], wide):
            _text_buffers(array, what, validity, offsets, data)
            pieces.append(ArrowTextPiece(rows, offset, validity, offsets, data, wide, 0, False,
                                         0, 0, 0, 0, 0, False))
            total += rows
            continue
        var signed = False
        var index_size = _format_index(s[_SCHEMA_FORMAT], signed)
        var dict_schema = s[_SCHEMA_DICTIONARY]
        var dict_array = a[_ARRAY_DICTIONARY]
        var dict_wide = False
        if index_size == 0 or dict_schema == 0 or dict_array == 0:
            raise Error("TypeError: mojolearn: " + what + " is an Arrow array of format '"
                        + _format_text(s[_SCHEMA_FORMAT]) + "'; an Arrow column is admitted as string ('u'), "
                        + "large_string ('U') or a dictionary of those")
        var ds = _Words(unsafe_from_address=dict_schema)
        if not _format_is_text(ds[_SCHEMA_FORMAT], dict_wide):
            raise Error("TypeError: mojolearn: " + what + " is an Arrow dictionary of format '"
                        + _format_text(ds[_SCHEMA_FORMAT]) + "' values; its values are admitted as string "
                        + "('u') or large_string ('U')")
        if a[_ARRAY_N_BUFFERS] != 2:
            raise Error("TypeError: mojolearn: " + what + " is an Arrow dictionary array with "
                        + String(a[_ARRAY_N_BUFFERS]) + " buffers (2 expected)")
        var ab = _Words(unsafe_from_address=a[_ARRAY_BUFFERS])
        validity = ab[0]
        var indices = ab[1]
        if indices == 0 and rows > 0:
            raise Error("TypeError: mojolearn: " + what + " is an Arrow dictionary array without indices")
        var da = _Words(unsafe_from_address=dict_array)
        var dict_rows = da[_ARRAY_LENGTH]
        var dict_offset = da[_ARRAY_OFFSET]
        if dict_rows < 0 or dict_offset < 0:
            raise Error("TypeError: mojolearn: " + what + " is an Arrow dictionary of negative length or offset")
        var dict_validity = 0
        var dict_offsets = 0
        var dict_data = 0
        _text_buffers(dict_array, what + " dictionary", dict_validity, dict_offsets, dict_data)
        pieces.append(ArrowTextPiece(rows, offset, validity, indices, 0, False, index_size, signed,
                                     dict_rows, dict_offset, dict_validity, dict_offsets, dict_data, dict_wide))
        total += rows
    _ = values  # keep the pandas array alive through the reads above
    return pieces^
