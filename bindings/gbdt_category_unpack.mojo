# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UNPACK of the categorical columns of `GradientBoosting` (lane cat-ctr,
2026-11-21; docs/plans/FUTURE_WORK_2026-11-21.md section 1, step 1).

A categorical column may hold any values: numbers that are not dense codes,
text, None. This file turns them into the dense codes the fit and the apply
read (`gbdt/data/category_map.mojo` has the key, the code and the stored
map), at the boundary, before RUN:

  * a float-admittable X (an array of numbers) is admitted exactly as it was
    (`admit_f32`, column-major) and the categorical columns' Float32 words
    are keyed natively;
  * an X that holds text or None in a categorical column (an exact list or
    tuple of exact list or tuple rows, an exact object ndarray through its
    own object words) is read cell by cell (`_read_cell`, the one cell reader) with exact-type
    C-API reads into one column-major float32 block (the numeric columns)
    and one key block (the categorical columns). That read is one pass
    under the GIL over Python objects, which is what the input is; it is
    serial because CPython objects are read under the GIL. No Python method
    of a cell runs: a cell is an exact int, float, bool, str, bytes or None,
    or an exact NumPy scalar of the loaded numpy module (its integer, bool,
    float32 / float64, str_ and bytes_ classes), whose value is read from
    the buffer it exports; anything else is refused by name. Text and bytes
    with the same UTF-8 bytes are one category;
  * an exact pandas DataFrame or Series is read from its blocks
    (`_pandas_frame`): numeric blocks cast in parallel, object and
    Python-backed string columns through the same cell reader, a
    Categorical through its codes array and its k categories (keyed once,
    then every code mapped in parallel; code -1 is the missing category).
    An Arrow-backed column (pandas 3's default text storage with pyarrow,
    `string[pyarrow]`, an `ArrowDtype` string or dictionary-of-string
    column, and Arrow-backed categories of a Categorical) is read through
    the Arrow C data interface (`bindings/arrow_c_data_unpack.mojo`: one
    `__arrow_c_array__` per chunk at UNPACK) and its validity, offsets and
    UTF-8 buffers keyed natively, rows across the workers with the GIL
    released (a null is the missing category);
  * an exact NumPy unicode (U) or bytes (S) array is read natively from
    its fixed-width cells, rows across the workers with the GIL released
    (`category_fixed_text`): the categorical columns keyed as text, the
    other columns parsed in Mojo as numbers (a cell that is not one is
    refused by name).

The codes are written over the categorical columns of the float block the
fit reads. A block the door made (a cast, a transpose, the object path) is
written in place; the caller's own array is never written, so a borrowed X
is copied once first (only then).

Everything after the cell reads is typed Mojo on plain words. The map is
kept as the fitted attribute `category_map_` (one int64 Array, the words of
the stored map), saved and loaded with the model, and searched in place at
predict.
"""

from std.memory import unsafe_memcpy, bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_typed_call import py_call_typed
from bindings.py_cpython_table import cpython_table, CPythonTable, cpython_utf8
from bindings.sequential_arrays_door import sequential_address
from bindings.cpython_borrowed_unpack import borrowed_type_word
from bindings.classical_pandas_door import (
    _pd_class_word,
    _pd_class,
    _pd_fields,
    _pd_get,
    _pd_locations_native,
    _pd_values_native,
    native_pandas_exact,
)
from bindings.classical_keys_unpack import classical_keys
from bindings.native_loaded_module_door import native_loaded_module
from core.pandas_input_plan import pandas_block_count
from core.sequential_input_plan import sequence_promote
from core.cast_in import cast_in_kind
from core.classical_pandas_impl import pd_scatter_numeric
from bindings.trees_public_inputs_door import trees_builtins, trees_address
from bindings.trees_keys_unpack import trees_keys
from bindings.admitted_input_unpack import AdmittedX, admit_f32, admitted_vector, admitted_words
from bindings.arrow_c_data_unpack import ArrowTextPiece, arrow_backed_pandas, arrow_text_pieces, pandas_class_name
from gbdt.data.category_map import (
    CategoryMaps,
    CategoryKeyPtr,
    CategoryF32Ptr,
    CATEGORY_KEY_MISSING,
    category_integer_in_range,
    category_key_integer,
    category_key_f64,
    category_key_text,
    category_keys_from_f32,
    category_block_columns,
    category_block_feature,
    category_apply_column,
    category_keys_from_codes,
    category_values_from_codes,
    category_fixed_text,
    category_keys_from_arrow_text,
    category_keys_from_arrow_indices,
    CATEGORY_TEXT_INVALID,
    CATEGORY_TEXT_NOT_A_NUMBER,
)

comptime _CatWordPtr = MutPointer[Int, MutUntrackedOrigin]

# What a cell's exact class says of the value its buffer holds
# (`_CellWords.buffer_kind`); 0 is "not a buffer cell this file admits".
comptime _CELL_SIGNED = 1
comptime _CELL_UNSIGNED = 2
comptime _CELL_BOOL = 3
comptime _CELL_FLOAT = 4
comptime _CELL_STR = 5
comptime _CELL_BYTES = 6
#: pandas' `NA` (`pandas._libs.missing.NAType`): the missing category, NaN in
#: a numeric column
comptime _CELL_MISSING = 7

comptime _Table = UnsafePointer[CPythonTable, MutAnyOrigin]
comptime _ViewPtr = UnsafePointer[Int, MutUntrackedOrigin]
comptime _ObjectWordPtr = MutPointer[PyObjectPtr, MutUntrackedOrigin]

comptime _CellI8 = MutPointer[Int8, MutUntrackedOrigin]
comptime _CellI16 = MutPointer[Int16, MutUntrackedOrigin]
comptime _CellI32 = MutPointer[Int32, MutUntrackedOrigin]
comptime _CellI64 = MutPointer[Int64, MutUntrackedOrigin]
comptime _CellU8 = MutPointer[UInt8, MutUntrackedOrigin]
comptime _CellU16 = MutPointer[UInt16, MutUntrackedOrigin]
comptime _CellU32 = MutPointer[UInt32, MutUntrackedOrigin]
comptime _CellU64 = MutPointer[UInt64, MutUntrackedOrigin]
comptime _CellF32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _CellF64 = MutPointer[Float64, MutUntrackedOrigin]


def gbdt_category_columns(cat: PythonObject) raises -> List[Int]:
    """The `cat_features` indices as an ascending list without repeats.
    The range of each index is checked by the caller against the width it
    admits (`_flags` for a float X, the row width for an object X)."""
    var b = trees_builtins()
    var out = List[Int]()
    var listed = py_call(b.list, cat)
    for k in range(len(listed)):  # driver-loop: over the feature-list argument indices
        var at = Int(py=py_call_typed(b.int, listed[k]))
        var pos = 0
        while pos < len(out) and out[pos] < at:  # small-loop(out: the categorical column indices): sorted insert of one index
            pos += 1
        if pos < len(out) and out[pos] == at:
            continue
        out.append(at)
        var i = len(out) - 1
        while i > pos:  # small-loop(out: the categorical column indices): shifts the tail for the insert
            out[i] = out[i - 1]
            i -= 1
        out[pos] = at
    return out^


def gbdt_category_block_list(category_map: PythonObject, n_features: Int) raises -> List[Int]:
    """The categorical columns a stored map names (checked against its word
    count and the fitted width first)."""
    var words = Int(py=category_map.size)
    var block = CategoryKeyPtr(unsafe_from_address=trees_address(category_map))
    var c = category_block_columns(block, words, n_features)
    var out = List[Int](capacity=c)
    for i in range(c):  # driver-loop: over the categorical columns of the model
        out.append(category_block_feature(block, i))
    _ = category_map  # keep the Python object alive through the native reads above
    return out^


struct _CellWords(Movable):
    """The exact builtin classes a cell is admitted by, as class words, read
    once per call."""
    var int_word: Int
    var float_word: Int
    var bool_word: Int
    var str_word: Int
    var none_word: Int
    var list_word: Int
    var tuple_word: Int
    #: exact `bytes`, and the exact scalar classes of the loaded numpy module
    #: (none when numpy is not loaded) with each one's `_CELL_*` kind
    var bytes_word: Int
    var np_words: List[Int]
    var np_kinds: List[Int]

    def __init__(out self) raises:
        var one = PythonObject(1)
        var real = PythonObject(1.5)
        var truth = PythonObject(True)
        var text = PythonObject("")
        var nothing = PythonObject(None)
        var listed = Python.list()
        var tupled = Python.tuple()
        self.int_word = borrowed_type_word(one._obj_ptr)
        self.float_word = borrowed_type_word(real._obj_ptr)
        self.bool_word = borrowed_type_word(truth._obj_ptr)
        self.str_word = borrowed_type_word(text._obj_ptr)
        self.none_word = borrowed_type_word(nothing._obj_ptr)
        self.list_word = borrowed_type_word(listed._obj_ptr)
        self.tuple_word = borrowed_type_word(tupled._obj_ptr)
        # class words by C dict lookups in the loaded modules: nothing is
        # imported, no attribute hook runs; an absent name is word 0, which
        # no live object's class is
        var words = List[Int]()
        var kinds = List[Int]()
        var bytes_word = 0
        var modules = cpython_table()[].PyImport_GetModuleDict()
        if modules:
            bytes_word = _pd_class_word(modules, "builtins", "bytes")
            var signed: List[String] = ["int8", "int16", "int32", "int64", "intc", "longlong"]
            var unsigned: List[String] = ["uint8", "uint16", "uint32", "uint64", "uintc", "ulonglong"]
            var floats: List[String] = ["float32", "float64"]
            for i in range(len(signed)):  # small-loop(signed: NumPy signed integer scalar names): one class word each
                var word = _pd_class_word(modules, "numpy", signed[i])
                if word != 0:
                    words.append(word)
                    kinds.append(_CELL_SIGNED)
            for i in range(len(unsigned)):  # small-loop(unsigned: NumPy unsigned integer scalar names): one class word each
                var word = _pd_class_word(modules, "numpy", unsigned[i])
                if word != 0:
                    words.append(word)
                    kinds.append(_CELL_UNSIGNED)
            for i in range(len(floats)):  # small-loop(floats: NumPy float scalar names): one class word each
                var word = _pd_class_word(modules, "numpy", floats[i])
                if word != 0:
                    words.append(word)
                    kinds.append(_CELL_FLOAT)
            var bool_word = _pd_class_word(modules, "numpy", "bool_")
            if bool_word != 0:
                words.append(bool_word)
                kinds.append(_CELL_BOOL)
            var str_word = _pd_class_word(modules, "numpy", "str_")
            if str_word != 0:
                words.append(str_word)
                kinds.append(_CELL_STR)
            var np_bytes_word = _pd_class_word(modules, "numpy", "bytes_")
            if np_bytes_word != 0:
                words.append(np_bytes_word)
                kinds.append(_CELL_BYTES)
            # pandas.NA, the missing cell of pandas' object and string
            # storage (pandas not loaded = no word)
            var na_word = _pd_class_word(modules, "pandas._libs.missing", "NAType")
            if na_word != 0:
                words.append(na_word)
                kinds.append(_CELL_MISSING)
        self.bytes_word = bytes_word
        self.np_words = words^
        self.np_kinds = kinds^
        _ = one
        _ = real
        _ = truth
        _ = text
        _ = nothing
        _ = listed
        _ = tupled

    def buffer_kind(self, word: Int) -> Int:
        """The `_CELL_*` kind of a cell whose exact class is `word` and is
        not one of the builtin number, str or None classes: `bytes`, or a
        NumPy scalar class. 0 when it is neither."""
        if word == 0:
            return 0
        if word == self.bytes_word:
            return _CELL_BYTES
        for i in range(len(self.np_words)):  # small-loop(np_words: NumPy scalar classes, under twenty): one class-word compare each
            if self.np_words[i] == word:
                return self.np_kinds[i]
        return 0


def _rows_hold_text(rows: PythonObject, columns: List[Int], w: _CellWords, name: String) raises -> Bool:
    """Whether a list or tuple X holds a str, a bytes, None or a NumPy
    scalar in a categorical column (then the object read runs; otherwise X
    is admitted as numbers, as it always was). Borrowed class-word reads
    only; a builtin int, float or bool cell costs three compares."""
    var t = cpython_table()
    var outer_list = borrowed_type_word(rows._obj_ptr) == w.list_word
    var n_rows = len(rows)
    var found = False
    for r in range(n_rows):  # boundary-read: one class word per categorical cell of a user list (no buffer to view)
        var row = t[].PyList_GetItem(rows._obj_ptr, r) if outer_list else t[].PyTuple_GetItem(rows._obj_ptr, r)
        var rw = borrowed_type_word(row)
        if rw != w.list_word and rw != w.tuple_word:
            # not rows of cells: the numeric admission names the refusal
            return False
        var row_list = rw == w.list_word
        var width = Int(t[].PyList_Size(row)) if row_list else Int(t[].PyTuple_Size(row))
        for c in range(len(columns)):  # driver-loop: over the categorical columns
            var j = columns[c]
            if j < 0 or j >= width:
                continue
            var cell = t[].PyList_GetItem(row, j) if row_list else t[].PyTuple_GetItem(row, j)
            var cw = borrowed_type_word(cell)
            if cw == w.int_word or cw == w.float_word or cw == w.bool_word:
                continue
            if cw == w.str_word or cw == w.none_word or w.buffer_kind(cw) != 0:
                found = True
                break
        if found:
            break
    _ = rows  # keep the Python object alive through the borrowed reads above
    _ = name
    return found


def _at(r: Int, j: Int) -> String:
    """The `[r][j]` of a refusal (built only when one is raised)."""
    return "[" + String(r) + "][" + String(j) + "]"


def _cell_text(t: _Table, cell: PyObjectPtr, name: String, r: Int, j: Int) raises -> UInt64:
    """The key of an exact str cell (or `numpy.str_`, which is a str)."""
    # cpython_utf8: the size through a volatile slot (the former
    # `Int(MutPointer(to=size))` form can forward the stored 0; lane sr-gbdt)
    var size = 0
    var utf8 = cpython_utf8(t, cell, size)
    if utf8 == 0:
        t[].PyErr_Clear()
        raise Error("mojolearn: " + name + _at(r, j) + " is not valid text (it cannot be encoded as UTF-8)")
    # CPython's UTF-8 cache lives as long as the cell, which its container holds
    return category_key_text(utf8, size)


def _read_cell(t: _Table, w: _CellWords, cell: PyObjectPtr, categorical: Bool, view_ptr: _ViewPtr,
               name: String, r: Int, j: Int) raises -> Tuple[UInt64, Float32]:
    """One cell of an X given as Python cells, read by its exact class with
    C reads only (no method of the cell runs): its key when `categorical`
    (the value slot is 0), else its Float32 value (the key slot is 0). The
    one reader of every container of cells: list or tuple rows, an object
    ndarray's words, a pandas object or string column, a Categorical's
    categories."""
    if not cell:
        raise Error("TypeError: mojolearn: " + name + _at(r, j) + " is an empty object slot")
    var cw = borrowed_type_word(cell)
    if cw == w.float_word:
        var f = t[].PyFloat_AsDouble(cell)
        if categorical:
            return (category_key_f64(f), Float32(0.0))
        return (UInt64(0), Float32(f))
    if cw == w.int_word or cw == w.bool_word:
        var v = Int(t[].PyLong_AsLongLong(cell))
        if v == -1 and t[].PyErr_Occurred():
            t[].PyErr_Clear()
            raise Error("mojolearn: " + name + _at(r, j) + " is an integer outside the 64-bit range")
        if categorical:
            if not category_integer_in_range(v):
                raise Error("mojolearn: " + name + _at(r, j)
                            + " is an integer category outside -2**61 .. 2**61; pass it as text")
            return (category_key_integer(v), Float32(0.0))
        return (UInt64(0), Float32(v))
    if cw == w.str_word and categorical:
        return (_cell_text(t, cell, name, r, j), Float32(0.0))
    if cw == w.none_word and categorical:
        return (CATEGORY_KEY_MISSING, Float32(0.0))
    var kind = w.buffer_kind(cw)
    if kind == 0 and categorical:
        raise Error("TypeError: mojolearn: " + name + _at(r, j)
                    + " is not an int, float, bool, str, bytes, None, pandas.NA or a NumPy scalar of "
                    + "those; a categorical cell is admitted by its exact type")
    if kind == 0 or ((kind == _CELL_STR or kind == _CELL_BYTES) and not categorical):
        raise Error("TypeError: mojolearn: " + name + _at(r, j)
                    + " is not a number; only the cat_features columns may hold text or None")
    if kind == _CELL_MISSING:
        if categorical:
            return (CATEGORY_KEY_MISSING, Float32(0.0))
        return (UInt64(0), bitcast[DType.float32](UInt32(0x7FC00000)))
    if kind == _CELL_STR:
        return (_cell_text(t, cell, name, r, j), Float32(0.0))
    # the value the cell exports (bytes, or a NumPy scalar's own storage,
    # native byte order): read, then released at once
    if t[].PyObject_GetBuffer(cell, view_ptr, Int32(0)) != 0:
        t[].PyErr_Clear()
        raise Error("TypeError: mojolearn: " + name + _at(r, j) + " does not export its value as a buffer")
    var at = view_ptr[0]
    var nbytes = view_ptr[2]
    var key = UInt64(0)
    var whole = 0
    var real = Float64(0.0)
    var is_real = False
    var too_wide = False
    var known = True
    if kind == _CELL_BYTES:
        key = category_key_text(at, nbytes)
    elif kind == _CELL_BOOL and nbytes == 1:
        whole = 1 if _CellU8(unsafe_from_address=at)[0] != UInt8(0) else 0
    elif kind == _CELL_SIGNED and nbytes == 1:
        whole = Int(_CellI8(unsafe_from_address=at)[0])
    elif kind == _CELL_SIGNED and nbytes == 2:
        whole = Int(_CellI16(unsafe_from_address=at)[0])
    elif kind == _CELL_SIGNED and nbytes == 4:
        whole = Int(_CellI32(unsafe_from_address=at)[0])
    elif kind == _CELL_SIGNED and nbytes == 8:
        whole = Int(_CellI64(unsafe_from_address=at)[0])
    elif kind == _CELL_UNSIGNED and nbytes == 1:
        whole = Int(_CellU8(unsafe_from_address=at)[0])
    elif kind == _CELL_UNSIGNED and nbytes == 2:
        whole = Int(_CellU16(unsafe_from_address=at)[0])
    elif kind == _CELL_UNSIGNED and nbytes == 4:
        whole = Int(_CellU32(unsafe_from_address=at)[0])
    elif kind == _CELL_UNSIGNED and nbytes == 8:
        var wide = _CellU64(unsafe_from_address=at)[0]
        if wide >= (UInt64(1) << UInt64(63)):
            too_wide = True
            real = Float64(wide)
        else:
            whole = Int(wide)
    elif kind == _CELL_FLOAT and nbytes == 4:
        is_real = True
        real = Float64(_CellF32(unsafe_from_address=at)[0])
    elif kind == _CELL_FLOAT and nbytes == 8:
        is_real = True
        real = _CellF64(unsafe_from_address=at)[0]
    else:
        known = False
    t[].PyBuffer_Release(view_ptr)
    if not known:
        raise Error("TypeError: mojolearn: " + name + _at(r, j) + " is a NumPy scalar of "
                    + String(nbytes) + " bytes this reader does not hold")
    if kind == _CELL_BYTES:
        return (key, Float32(0.0))
    if is_real:
        if categorical:
            return (category_key_f64(real), Float32(0.0))
        return (UInt64(0), Float32(real))
    if categorical:
        if too_wide or not category_integer_in_range(whole):
            raise Error("mojolearn: " + name + _at(r, j)
                        + " is an integer category outside -2**61 .. 2**61; pass it as text")
        return (category_key_integer(whole), Float32(0.0))
    if too_wide:
        return (UInt64(0), Float32(real))
    return (UInt64(0), Float32(whole))


def _cat_slots(columns: List[Int], n_features: Int) raises -> List[Int]:
    """`slot[j]` = the index of column j in `columns`, or -1 (a numeric
    column); every index checked against the width."""
    var slot_of = List[Int](length=n_features, fill=-1)
    for c in range(len(columns)):  # driver-loop: over the categorical columns
        if columns[c] < 0 or columns[c] >= n_features:
            raise Error("mojolearn: cat_features index " + String(columns[c]) + " out of range for "
                        + String(n_features) + " features")
        slot_of[columns[c]] = c
    return slot_of^


def _object_column(t: _Table, w: _CellWords, base: Int, stride: Int, n_rows: Int, j: Int, slot: Int,
                   kp: CategoryKeyPtr, mp: CategoryF32Ptr, view_ptr: _ViewPtr, name: String) raises:
    """Column j of X given as borrowed object words (row r's cell at
    `base + r * stride`): its keys to `kp[slot * n_rows + r]` when `slot >= 0`,
    else its values to `mp[j * n_rows + r]`. Serial: every cell is a CPython
    object read under the GIL."""
    var categorical = slot >= 0
    for r in range(n_rows):  # boundary-read: one exact-type read per object word of a user column (no numbers to view)
        var cell = _ObjectWordPtr(unsafe_from_address=base + r * stride)[0]
        var got = _read_cell(t, w, cell, categorical, view_ptr, name, r, j)
        if categorical:
            kp[slot * n_rows + r] = got[0]
        else:
            mp[j * n_rows + r] = got[1]


def _object_rows(rows: PythonObject, name: String, columns: List[Int], mut keys: List[UInt64],
                 context: PythonObject) raises -> AdmittedX:
    """An X given as rows of Python cells: the numeric columns as one
    column-major float32 block and the categorical columns as keys
    (`keys[c * n_rows + r]` for the c-th categorical column). Serial: every
    cell is a CPython object read under the GIL."""
    var t = cpython_table()
    var w = _CellWords()
    var outer_list = borrowed_type_word(rows._obj_ptr) == w.list_word
    var n_rows = len(rows)
    if n_rows == 0:
        raise Error("mojolearn: " + name + " has no rows")
    var first = t[].PyList_GetItem(rows._obj_ptr, 0) if outer_list else t[].PyTuple_GetItem(rows._obj_ptr, 0)
    var fw = borrowed_type_word(first)
    if fw != w.list_word and fw != w.tuple_word:
        raise Error("TypeError: mojolearn: " + name + " with text categories must be rows of cells "
                    + "(a list or tuple of lists or tuples, a NumPy array or a pandas DataFrame)")
    var n_features = Int(t[].PyList_Size(first)) if fw == w.list_word else Int(t[].PyTuple_Size(first))
    if n_features == 0:
        raise Error("mojolearn: " + name + " has no features")
    var slot_of = _cat_slots(columns, n_features)
    var matrix = admitted_vector(n_rows * n_features, "<f4", context)
    var mp = CategoryF32Ptr(unsafe_from_address=trees_address(matrix))
    keys.resize(len(columns) * n_rows, UInt64(0))
    var kp = CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr()))
    var sp = _CatWordPtr(unsafe_from_address=Int(slot_of.unsafe_ptr()))
    # the Py_buffer words of a bytes or NumPy scalar cell, one for the call
    var view = List[Int](length=10, fill=0)
    var view_ptr = view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for r in range(n_rows):  # boundary-read: one exact-type slot read per cell of a user list (no buffer to view)
        var row = t[].PyList_GetItem(rows._obj_ptr, r) if outer_list else t[].PyTuple_GetItem(rows._obj_ptr, r)
        var rw = borrowed_type_word(row)
        var row_list = rw == w.list_word
        if not row_list and rw != w.tuple_word:
            raise Error("TypeError: mojolearn: " + name + "[" + String(r) + "] must be a list or tuple of cells")
        var width = Int(t[].PyList_Size(row)) if row_list else Int(t[].PyTuple_Size(row))
        if width != n_features:
            raise Error("mojolearn: " + name + "[" + String(r) + "] has " + String(width) + " cells for "
                        + String(n_features) + " features")
        for j in range(n_features):  # boundary-read: the cells of one row
            var cell = t[].PyList_GetItem(row, j) if row_list else t[].PyTuple_GetItem(row, j)
            var slot = sp[j]
            var got = _read_cell(t, w, cell, slot >= 0, view_ptr, name, r, j)
            if slot >= 0:
                kp[slot * n_rows + r] = got[0]
            else:
                mp[j * n_rows + r] = got[1]
    _ = rows  # keep the Python object alive through the borrowed reads above
    _ = view^
    _ = slot_of^
    var block_address = trees_address(matrix)
    return admitted_words(matrix, block_address, n_rows, n_features, False)


def _object_array(X: PythonObject, name: String, columns: List[Int], mut keys: List[UInt64],
                  context: PythonObject) raises -> AdmittedX:
    """An exact 2-D object ndarray: its cells read from the array's own
    object words (any strides), column by column, by `_object_column`. The
    array holds every cell, so the borrowed words stay live through the
    read; no list of the cells is built."""
    var t = cpython_table()
    var w = _CellWords()
    var n_rows = Int(py=X.shape[0])
    var n_features = Int(py=X.shape[1])
    if n_rows == 0:
        raise Error("mojolearn: " + name + " has no rows")
    if n_features == 0:
        raise Error("mojolearn: " + name + " has no features")
    var slot_of = _cat_slots(columns, n_features)
    var address = Int(py=sequential_address(X))
    var row_stride = Int(py=X.strides[0])
    var col_stride = Int(py=X.strides[1])
    var matrix = admitted_vector(n_rows * n_features, "<f4", context)
    var mp = CategoryF32Ptr(unsafe_from_address=trees_address(matrix))
    keys.resize(len(columns) * n_rows, UInt64(0))
    var kp = CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr()))
    var view = List[Int](length=10, fill=0)
    var view_ptr = view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for j in range(n_features):  # driver-loop: over the columns of X (each column's cells read by _object_column)
        _object_column(t, w, address + j * col_stride, row_stride, n_rows, j, slot_of[j], kp, mp, view_ptr, name)
    _ = X  # keep the array (and so every cell) alive through the borrowed reads above
    _ = view^
    _ = slot_of^
    var block_address = trees_address(matrix)
    return admitted_words(matrix, block_address, n_rows, n_features, False)


def _fixed_text_array(X: PythonObject, wide: Bool, name: String, columns: List[Int], mut keys: List[UInt64],
                      context: PythonObject) raises -> AdmittedX:
    """An exact 2-D NumPy unicode (`wide`) or bytes array, read natively
    from its fixed-width cells with the GIL released, rows across the
    workers (`category_fixed_text`): the categorical columns keyed as text
    (the keys a str or bytes cell with the same UTF-8 bytes gets), the other
    columns parsed as numbers, a cell that is not one refused by name. No
    Python str or bytes object is made."""
    var n_rows = Int(py=X.shape[0])
    var n_features = Int(py=X.shape[1])
    if n_rows == 0:
        raise Error("mojolearn: " + name + " has no rows")
    if n_features == 0:
        raise Error("mojolearn: " + name + " has no features")
    if wide and not Bool(py=X.dtype.isnative):
        raise Error("TypeError: mojolearn: " + name + " is a unicode array in non-native byte order; "
                    + "pass it in native order (`X.astype(X.dtype.newbyteorder('='))`)")
    var slot_of = _cat_slots(columns, n_features)
    var address = Int(py=sequential_address(X))
    var row_stride = Int(py=X.strides[0])
    var col_stride = Int(py=X.strides[1])
    var itemsize = Int(py=X.dtype.itemsize)
    var matrix = admitted_vector(n_rows * n_features, "<f4", context)
    var mp = CategoryF32Ptr(unsafe_from_address=trees_address(matrix))
    keys.resize(len(columns) * n_rows, UInt64(0))
    var kp = CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr()))
    var slots = Int(slot_of.unsafe_ptr())
    var failed: Tuple[Int, Int, Int] = (Int(-1), Int(-1), Int(0))
    with GILReleased(Python()):
        failed = category_fixed_text(address, itemsize, wide, row_stride, col_stride, n_rows, n_features,
                                     slots, kp, mp)
    _ = X  # keep the array alive through the native reads above
    _ = slot_of^
    if failed[2] == CATEGORY_TEXT_INVALID:
        raise Error("mojolearn: " + name + _at(failed[0], failed[1])
                    + " is not valid text (it cannot be encoded as UTF-8)")
    if failed[2] == CATEGORY_TEXT_NOT_A_NUMBER:
        raise Error("ValueError: mojolearn: " + name + _at(failed[0], failed[1])
                    + " is not a number; only the cat_features columns may hold text")
    var block_address = trees_address(matrix)
    return admitted_words(matrix, block_address, n_rows, n_features, False)


# The storage forms of a pandas column this file reads (`_PdPart.form`).
comptime _PD_NUMERIC = 1
comptime _PD_OBJECT = 2
comptime _PD_CATEGORICAL = 3
#: an Arrow-backed text column (its chunks in `_PdPart.arrow`); as a
#: Categorical's `cat_form`, Arrow-backed categories (`_PdPart.cat_arrow`)
comptime _PD_ARROW = 4
#: the `_Nd.kind` of storage read through Arrow (not a NumPy kind)
comptime _ARROW_KIND = "A"


@fieldwise_init
struct _Nd(Copyable, Movable):
    """The metadata of one exact ndarray of pandas storage (rank 1: one
    column; rank 2: pandas' (columns, rows) block layout)."""
    var address: Int
    var rows: Int
    var cols: Int
    var row_stride: Int
    var col_stride: Int
    var itemsize: Int
    var typecode: Int
    var kind: String
    var native: Bool


def _nd_meta(a: PythonObject, ndarray: PythonObject, what: String) raises -> _Nd:
    if py_type(a) is not ndarray:
        raise Error("TypeError: mojolearn: " + what + " is not stored in an exact NumPy array")
    var rank = Int(py=a.ndim)
    if rank != 1 and rank != 2:
        raise Error("TypeError: mojolearn: " + what + " is stored in a " + String(rank) + "-D array")
    var rows = Int(py=a.shape[rank - 1])
    var cols = 1 if rank == 1 else Int(py=a.shape[0])
    var row_stride = Int(py=a.strides[rank - 1])
    var col_stride = 0 if rank == 1 else Int(py=a.strides[0])
    return _Nd(Int(py=sequential_address(a)), rows, cols, row_stride, col_stride,
               Int(py=a.dtype.itemsize), Int(py=a.dtype.num), String(py=a.dtype.kind),
               Bool(py=a.dtype.isnative))


@fieldwise_init
struct _PdPart(Copyable, Movable):
    """One pandas block as this file reads it: `store` is the block's
    storage (numbers or object words) or, for a Categorical, its codes;
    `cats` is a Categorical's categories (numbers or object words)."""
    var form: Int
    var store: _Nd
    var positions: List[Int]
    var cat_form: Int
    var cats: _Nd
    var arrow: List[ArrowTextPiece]
    var cat_arrow: List[ArrowTextPiece]


def _is_category_column(columns: List[Int], at: Int) -> Bool:
    """Whether column `at` is one of the `cat_features` columns."""
    for c in range(len(columns)):  # small-loop(columns: the categorical column indices): one compare each
        if columns[c] == at:
            return True
    return False


def _arrow_piece_keys(piece: ArrowTextPiece, dst: CategoryKeyPtr) raises:
    """The keys of one Arrow chunk's rows to `dst[0 .. piece.rows)`: a
    string / large_string chunk keyed row by row in parallel; a dictionary
    chunk's values keyed once, then every index mapped in parallel. Plain
    buffers only: the caller releases the GIL."""
    if piece.index_size == 0:
        category_keys_from_arrow_text(piece.validity, piece.offsets, piece.data, piece.wide, piece.offset,
                                      piece.rows, dst)
        return
    var k = piece.dict_rows
    var table = List[UInt64](length=max(k, 1), fill=UInt64(0))
    var tp = CategoryKeyPtr(unsafe_from_address=Int(table.unsafe_ptr()))
    category_keys_from_arrow_text(piece.dict_validity, piece.dict_offsets, piece.dict_data, piece.dict_wide,
                                  piece.dict_offset, k, tp)
    category_keys_from_arrow_indices(piece.validity, piece.offsets, piece.index_size, piece.index_signed,
                                     piece.offset, piece.rows, tp, k, dst)
    _ = table^


def _arrow_column_keys(pieces: List[ArrowTextPiece], dst: CategoryKeyPtr) raises:
    """The keys of an Arrow column's chunks, one after another from `dst`,
    with the GIL released."""
    var row = 0
    for q in range(len(pieces)):  # driver-loop: over one column's Arrow chunks (each chunk's rows run on every worker)
        var piece = pieces[q].copy()
        with GILReleased(Python()):
            _arrow_piece_keys(piece, dst + row)
        row += piece.rows


def _pd_text_storage(values: PythonObject, ndarray: PythonObject, what: String) raises -> _Nd:
    """The object words of pandas' Python-backed string array (`StringArray`,
    `StringArrayNumpySemantics`): its `NDArrayBacked._ndarray`, an object
    array of str and pandas.NA. Any other class is refused by name (the
    Arrow-backed ones are read by the caller before this)."""
    var cls = py_type(values)
    if cls is _pd_class("pandas.core.arrays.string_", "StringArray") or cls is _pd_class(
            "pandas.core.arrays.string_", "StringArrayNumpySemantics"):
        var nd = _nd_meta(_pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", values), ndarray, what)
        if nd.kind != "O" or nd.cols != 1:
            raise Error("TypeError: mojolearn: " + what + " is a pandas string array whose storage is not "
                        + "one object column")
        return nd^
    raise Error("TypeError: mojolearn: " + what + " is stored as " + pandas_class_name(cls)
                + "; a pandas column is admitted as NumPy numbers, NumPy objects, a pandas string "
                + "array (Python or Arrow storage), an Arrow string column or a Categorical")


def _pandas_frame(X: PythonObject, name: String, columns: List[Int], mut keys: List[UInt64],
                  context: PythonObject) raises -> AdmittedX:
    """An exact pandas DataFrame (or Series: one column) with categorical
    columns, read from its blocks: numeric blocks cast in parallel
    (`pd_scatter_numeric`, column-major, pandas' promotion), object and
    Python-backed string columns cell by cell (`_object_column`), a
    Categorical through its codes array and its categories (the k
    categories are keyed once, then every row's code is mapped in parallel:
    no loop over the rows' values in Python). pandas is never imported: the
    classes are its loaded ones by exact identity and every field is read
    through its native descriptor or instance dictionary."""
    var b = native_loaded_module("builtins")
    var nk = classical_keys()
    var ndarray = py_get(context, trees_keys()[].k_ndarray)
    var series = py_type(X) is _pd_class("pandas.core.series", "Series")
    var manager = py_get(_pd_fields(X), nk[].k__mgr)
    var blocks = PythonObject(None)
    if series:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: pandas Series requires native single-block storage")
        blocks = py_get(_pd_fields(manager), nk[].k_blocks)
    else:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
            raise Error("TypeError: pandas DataFrame requires native block storage")
        blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not b.tuple:
        raise Error("pandas input is empty or has invalid block metadata")
    pandas_block_count(len(blocks), series)
    # the admitted classes, read once per call
    var numpy_block = _pd_class("pandas.core.internals.blocks", "NumpyBlock")
    var numeric_block = _pd_class("pandas.core.internals.blocks", "NumericBlock")
    var object_block = _pd_class("pandas.core.internals.blocks", "ObjectBlock")
    var extension_block = _pd_class("pandas.core.internals.blocks", "ExtensionBlock")
    var categorical = _pd_class("pandas.core.arrays.categorical", "Categorical")
    var categorical_dtype = _pd_class("pandas.core.dtypes.dtypes", "CategoricalDtype")
    var index_class = _pd_class("pandas.core.indexes.base", "Index")
    var integer_array = _pd_class("pandas.core.arrays.integer", "IntegerArray")
    var floating_array = _pd_class("pandas.core.arrays.floating", "FloatingArray")
    var boolean_array = _pd_class("pandas.core.arrays.boolean", "BooleanArray")
    var parts = List[_PdPart]()
    var owned = List[PythonObject]()
    var n_rows = -1
    var n_features = 0
    # pandas' promotion of the numeric blocks (core/pandas_input_plan.mojo's
    # rule): through float64 first when the frame's common type is object,
    # bool with numbers, or float64
    var common_kind = String()
    var common_size = 0
    var has_bool = False
    var has_number = False
    var objects = False
    var none = _Nd(0, 0, 0, 0, 0, 0, 0, String(), True)
    for k in range(len(blocks)):  # driver-loop: over native pandas storage blocks
        var block = blocks[k]
        var bcls = py_type(block)
        if bcls is not numpy_block and bcls is not numeric_block and bcls is not object_block and bcls is not extension_block:
            raise Error("TypeError: pandas requires ordinary native blocks without subclass hooks")
        var values = _pd_get("pandas._libs.internals", "Block", "values", block)
        var positions = List[Int](length=1, fill=0) if series else _pd_locations_native(block, context)
        var what = name + " block " + String(k)
        var vcls = py_type(values)
        var form = _PD_NUMERIC
        var cat_form = 0
        var cats = none.copy()
        var store: _Nd
        var pieces = List[ArrowTextPiece]()
        var cat_pieces = List[ArrowTextPiece]()
        if vcls is ndarray or vcls is integer_array or vcls is floating_array or vcls is boolean_array:
            var data = values
            if vcls is not ndarray:
                # the nullable numbers' own admission (missing values refused there)
                var pair = _pd_values_native(values, series, context)
                data = pair[0]
                objects = objects or pair[1]
            store = _nd_meta(data, ndarray, what)
            if store.kind == "O":
                form = _PD_OBJECT
                objects = True
            elif store.kind == "b" or store.kind == "i" or store.kind == "u" or store.kind == "f":
                sequence_promote(common_kind, common_size, store.kind, store.itemsize)
                has_bool = has_bool or store.kind == "b"
                has_number = has_number or store.kind != "b"
            else:
                raise Error("TypeError: mojolearn: " + what + " holds NumPy dtype kind '" + store.kind
                            + "'; a pandas column is admitted as numbers, objects, strings or a Categorical")
            owned.append(data)
        elif vcls is categorical:
            form = _PD_CATEGORICAL
            objects = True
            var codes = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_ndarray", values)
            var dtype = _pd_get("pandas._libs.arrays", "NDArrayBacked", "_dtype", values)
            if py_type(dtype) is not categorical_dtype:
                raise Error("TypeError: mojolearn: " + what + " is a Categorical whose dtype is not an exact "
                            + "pandas CategoricalDtype")
            var categories = py_get(_pd_fields(dtype), "_categories")
            if py_type(categories) is not index_class:
                raise Error("TypeError: mojolearn: " + what + " is a Categorical whose categories are "
                            + "not an exact pandas Index")
            var cat_data = py_get(_pd_fields(categories), "_data")
            store = _nd_meta(codes, ndarray, what + " codes")
            if store.kind != "i" or store.cols != 1 or not store.native:
                raise Error("TypeError: mojolearn: " + what + " is a Categorical whose codes are not "
                            + "native signed integers")
            if py_type(cat_data) is ndarray:
                cats = _nd_meta(cat_data, ndarray, what + " categories")
            elif arrow_backed_pandas(cat_data):
                # pandas 3's str categories: an Arrow string Index
                var cat_rows = 0
                cat_pieces = arrow_text_pieces(cat_data, what + " categories", owned, cat_rows)
                cats = _Nd(0, cat_rows, 1, 0, 0, 0, 0, String(_ARROW_KIND), True)
            else:
                cats = _pd_text_storage(cat_data, ndarray, what + " categories")
            if cats.cols != 1:
                raise Error("TypeError: mojolearn: " + what + " is a Categorical whose categories are not 1-D")
            if cats.kind == _ARROW_KIND:
                cat_form = _PD_ARROW
            elif cats.kind == "O":
                cat_form = _PD_OBJECT
            elif cats.kind == "b" or cats.kind == "i" or cats.kind == "u" or cats.kind == "f":
                cat_form = _PD_NUMERIC
            else:
                raise Error("TypeError: mojolearn: " + what + " is a Categorical of NumPy dtype kind '"
                            + cats.kind + "'; categories are admitted as numbers or text")
            owned.append(values)
            owned.append(codes)
            owned.append(dtype)
            owned.append(categories)
            owned.append(cat_data)
        elif arrow_backed_pandas(values):
            # Arrow storage: its chunks' buffers through the C data interface
            form = _PD_ARROW
            objects = True
            # admitted at a categorical column only: refused before any call
            if len(positions) != 1 or not _is_category_column(columns, positions[0]):
                raise Error("TypeError: mojolearn: " + what + " is an Arrow-backed pandas column ("
                            + pandas_class_name(vcls) + ") outside cat_features; an Arrow column is "
                            + "admitted as text of a cat_features column")
            owned.append(values)
            var arrow_rows = 0
            pieces = arrow_text_pieces(values, what, owned, arrow_rows)
            store = _Nd(0, arrow_rows, 1, 0, 0, 0, 0, String(_ARROW_KIND), True)
        else:
            form = _PD_OBJECT
            objects = True
            store = _pd_text_storage(values, ndarray, what)
            owned.append(values)
        if store.rows < 1 or (n_rows >= 0 and store.rows != n_rows):
            raise Error("pandas numeric block dimensions differ or are empty")
        if len(positions) != store.cols:
            raise Error("pandas block column positions differ from numeric storage")
        n_rows = store.rows
        n_features += store.cols
        parts.append(_PdPart(form, store^, positions^, cat_form, cats^, pieces^, cat_pieces^))
    var seen = List[Bool](length=n_features, fill=False)
    for p in range(len(parts)):  # driver-loop: over the blocks' column placements
        for q in range(len(parts[p].positions)):  # driver-loop: over one block's columns
            var at = parts[p].positions[q]
            if at < 0 or at >= n_features or seen[at]:
                raise Error("pandas block column positions overlap or exceed the matrix")
            seen[at] = True
    var double_round = objects or (has_bool and has_number) or (common_kind == "f" and common_size == 8)
    var slot_of = _cat_slots(columns, n_features)
    var matrix = admitted_vector(n_rows * n_features, "<f4", context)
    var mp = CategoryF32Ptr(unsafe_from_address=trees_address(matrix))
    keys.resize(len(columns) * n_rows, UInt64(0))
    var kp = CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr()))
    var t = cpython_table()
    var w = _CellWords()
    var view = List[Int](length=10, fill=0)
    var view_ptr = view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    try:
        for p in range(len(parts)):  # driver-loop: over the admitted blocks (rows of numbers and codes run on every worker)
            var part = parts[p].copy()
            var s = part.store.copy()
            if part.form == _PD_NUMERIC:
                var kind = cast_in_kind(s.typecode, s.itemsize, "pandas input")
                for col in range(s.cols):  # driver-loop: over one block's columns; each cast spans parallel row tasks
                    var dest = part.positions[col]
                    var column = mp + dest * n_rows
                    with GILReleased(Python()):
                        pd_scatter_numeric(s.address, s.itemsize, s.native, kind, s.row_stride, s.col_stride,
                                           col, 0, n_rows, 1, double_round, column, True)
                        if slot_of[dest] >= 0:
                            # the same keys a numeric X's categorical column gets
                            category_keys_from_f32(column, n_rows, kp + slot_of[dest] * n_rows)
            elif part.form == _PD_ARROW:
                # admitted only at a categorical column (checked at the walk)
                var dest = part.positions[0]
                _arrow_column_keys(part.arrow, kp + slot_of[dest] * n_rows)
            elif part.form == _PD_OBJECT:
                for col in range(s.cols):  # driver-loop: over one block's columns (cells read under the GIL)
                    var dest = part.positions[col]
                    _object_column(t, w, s.address + col * s.col_stride, s.row_stride, n_rows, dest,
                                   slot_of[dest], kp, mp, view_ptr, name)
            else:
                var dest = part.positions[0]
                var slot = slot_of[dest]
                var c = part.cats.copy()
                var k = c.rows
                # the k categories, read once: keys for a categorical column,
                # numbers for a numeric one (text there is refused by the reader)
                var cat_keys = List[UInt64](length=max(k, 1), fill=UInt64(0))
                var cat_values = List[Float32](length=max(k, 1), fill=Float32(0.0))
                var ck = CategoryKeyPtr(unsafe_from_address=Int(cat_keys.unsafe_ptr()))
                var cv = CategoryF32Ptr(unsafe_from_address=Int(cat_values.unsafe_ptr()))
                if part.cat_form == _PD_ARROW:
                    if slot < 0:
                        raise Error("TypeError: mojolearn: " + name + " column " + String(dest)
                                    + " is a Categorical of Arrow text; only the cat_features columns may "
                                    + "hold text")
                    _arrow_column_keys(part.cat_arrow, ck)
                elif part.cat_form == _PD_OBJECT:
                    # written as column 0 of a k-row block (`ck[r]` or `cv[r]`)
                    _object_column(t, w, c.address, c.row_stride, k, 0, 0 if slot >= 0 else -1,
                                   ck, cv, view_ptr, name + " column " + String(dest) + " categories")
                else:
                    var cat_kind = cast_in_kind(c.typecode, c.itemsize, "pandas categories")
                    pd_scatter_numeric(c.address, c.itemsize, c.native, cat_kind, c.row_stride, 0,
                                       0, 0, k, 1, True, cv, True)
                    if slot >= 0:
                        category_keys_from_f32(cv, k, ck)
                with GILReleased(Python()):
                    if slot >= 0:
                        category_keys_from_codes(s.address, s.itemsize, s.row_stride, n_rows, ck, k,
                                                 kp + slot * n_rows)
                    else:
                        category_values_from_codes(s.address, s.itemsize, s.row_stride, n_rows, cv, k,
                                                   mp + dest * n_rows)
                _ = cat_keys^
                _ = cat_values^
    finally:
        _ = owned
        _ = manager
        _ = blocks
        _ = X
    _ = view^
    _ = slot_of^
    var block_address = trees_address(matrix)
    return admitted_words(matrix, block_address, n_rows, n_features, False)


def gbdt_category_admit(X: PythonObject, name: String, columns: List[Int], mut keys: List[UInt64],
                        context: PythonObject) raises -> AdmittedX:
    """X of a fit or a predict with categorical columns, column-major.
    `keys` is left empty when X is numbers (the categorical columns are then
    keyed from their Float32 words by `gbdt_category_fit` /
    `gbdt_category_apply`); it holds the categorical cells' keys when X
    holds text or None there, and for every pandas DataFrame or Series."""
    var b = trees_builtins()
    var nk = trees_keys()
    var cls = py_type(X)
    var ndarray = py_get(context, nk[].k_ndarray)
    if ndarray is not None and cls is ndarray:
        # an exact ndarray: its dtype kind is a C slot read
        var kind = String(py=X.dtype.kind)
        if kind == "O" or kind == "U" or kind == "S":
            if Int(py=X.ndim) != 2:
                raise Error("mojolearn: " + name + " must be 2-D, got " + String(Int(py=X.ndim)) + "-D")
        # an object array: its cells read from its own object words
        if kind == "O":
            return _object_array(X, name, columns, keys, context)
        # unicode and bytes arrays: read natively from their fixed-width cells
        if kind == "U" or kind == "S":
            return _fixed_text_array(X, kind == "U", name, columns, keys, context)
    elif cls is b.list or cls is b.tuple:
        var w = _CellWords()
        if len(X) != 0 and _rows_hold_text(X, columns, w, name):
            return _object_rows(X, name, columns, keys, context)
    elif native_pandas_exact(X):
        # a DataFrame or Series: read from its blocks here, whatever they
        # hold (numbers too: one walk of the frame, written column-major)
        return _pandas_frame(X, name, columns, keys, context)
    return admit_f32(X, 2, name, context, order="F", finite=False)


def _owned_block(mut x: AdmittedX, context: PythonObject) raises:
    """Make `x` a block the door may write: the caller's own array is copied
    once (it is never written); a block the door made is left as it is."""
    if not x.borrowed:
        return
    var count = x.rows * x.cols
    var block = admitted_vector(count, "<f4", context)
    var address = trees_address(block)
    unsafe_memcpy(
        dest=CategoryF32Ptr(unsafe_from_address=address),
        src=CategoryF32Ptr(unsafe_from_address=x.address),
        count=count,
    )
    var rows = x.rows
    var cols = x.cols
    x = admitted_words(block, address, rows, cols, False)


def gbdt_category_fit(mut x: AdmittedX, columns: List[Int], keys: List[UInt64],
                      context: PythonObject) raises -> PythonObject:
    """Fit the category map of every column in `columns` from the fit's X
    and write the dense codes over those columns of the block the fit reads.
    Returns the stored map (one int64 Array, `category_map_`)."""
    _owned_block(x, context)
    var n_rows = x.rows
    var maps = CategoryMaps()
    var from_cells = len(keys) != 0
    var scratch = List[UInt64]()
    if not from_cells:
        scratch = List[UInt64](unsafe_uninit_length=n_rows)
    var sp = CategoryKeyPtr(unsafe_from_address=Int(scratch.unsafe_ptr()))
    var kp = CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr()))
    for c in range(len(columns)):  # driver-loop: over the categorical columns (each column's rows run on every worker)
        var column = CategoryF32Ptr(unsafe_from_address=x.address + 4 * columns[c] * n_rows)
        if from_cells:
            maps.fit_column(columns[c], kp + c * n_rows, n_rows, column)
        else:
            category_keys_from_f32(column, n_rows, sp)
            maps.fit_column(columns[c], sp, n_rows, column)
    var words = maps.block_words()
    var stored = admitted_vector(words, "<i8", context)
    maps.write_block(CategoryKeyPtr(unsafe_from_address=trees_address(stored)))
    _ = scratch^
    _ = keys
    _ = x  # keep the Python object alive through the native writes above
    return stored


def gbdt_category_apply(mut x: AdmittedX, keys: List[UInt64], category_map: PythonObject,
                        n_features: Int, context: PythonObject) raises:
    """Write the dense codes of a fitted map over the categorical columns of
    a predict's (or a held-out set's) X; a value the fit did not see takes
    the unseen code. The map is searched in place."""
    var words = Int(py=category_map.size)
    var block = CategoryKeyPtr(unsafe_from_address=trees_address(category_map))
    var n_cat = category_block_columns(block, words, n_features)
    _owned_block(x, context)
    var n_rows = x.rows
    var from_cells = len(keys) != 0
    var scratch = List[UInt64]()
    if not from_cells:
        scratch = List[UInt64](unsafe_uninit_length=n_rows)
    var sp = CategoryKeyPtr(unsafe_from_address=Int(scratch.unsafe_ptr()))
    var kp = CategoryKeyPtr(unsafe_from_address=Int(keys.unsafe_ptr()))
    for c in range(n_cat):  # driver-loop: over the categorical columns (each column's rows run on every worker)
        var feature = category_block_feature(block, c)
        var column = CategoryF32Ptr(unsafe_from_address=x.address + 4 * feature * n_rows)
        if from_cells:
            category_apply_column(block, c, kp + c * n_rows, n_rows, column)
        else:
            category_keys_from_f32(column, n_rows, sp)
            category_apply_column(block, c, sp, n_rows, column)
    _ = scratch^
    _ = keys
    _ = category_map  # keep the Python object alive through the native reads above
    _ = x  # keep the Python object alive through the native writes above
