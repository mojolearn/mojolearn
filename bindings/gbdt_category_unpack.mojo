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
    tuple of exact list or tuple rows, or an exact object / unicode / bytes
    ndarray through its own `tolist`) is read cell by cell with exact-type
    C-API reads into one column-major float32 block (the numeric columns)
    and one key block (the categorical columns). That read is one pass
    under the GIL over Python objects, which is what the input is; it is
    serial because CPython objects are read under the GIL. No Python method
    of a cell runs: a cell is an exact int, float, bool, str, bytes or None,
    or an exact NumPy scalar of the loaded numpy module (its integer, bool,
    float32 / float64, str_ and bytes_ classes), whose value is read from
    the buffer it exports; anything else is refused by name. Text and bytes
    with the same UTF-8 bytes are one category.

The codes are written over the categorical columns of the float block the
fit reads. A block the door made (a cast, a transpose, the object path) is
written in place; the caller's own array is never written, so a borrowed X
is copied once first (only then).

Everything after the cell reads is typed Mojo on plain words. The map is
kept as the fitted attribute `category_map_` (one int64 Array, the words of
the stored map), saved and loaded with the model, and searched in place at
predict.
"""

from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_typed_call import py_call_typed
from bindings.py_cpython_table import cpython_table
from bindings.cpython_borrowed_unpack import borrowed_type_word
from bindings.classical_pandas_door import _pd_class_word
from bindings.trees_public_inputs_door import trees_builtins, trees_address
from bindings.trees_keys_unpack import trees_keys
from bindings.admitted_input_unpack import AdmittedX, admit_f32, admitted_vector, admitted_words
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
                    + "(a list or tuple of lists or tuples, or a NumPy object array)")
    var n_features = Int(t[].PyList_Size(first)) if fw == w.list_word else Int(t[].PyTuple_Size(first))
    if n_features == 0:
        raise Error("mojolearn: " + name + " has no features")
    var slot_of = List[Int](length=n_features, fill=-1)
    for c in range(len(columns)):  # driver-loop: over the categorical columns
        if columns[c] < 0 or columns[c] >= n_features:
            raise Error("mojolearn: cat_features index " + String(columns[c]) + " out of range for "
                        + String(n_features) + " features")
        slot_of[columns[c]] = c
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
            var cw = borrowed_type_word(cell)
            var slot = sp[j]
            if cw == w.float_word:
                var f = t[].PyFloat_AsDouble(cell)
                if slot >= 0:
                    kp[slot * n_rows + r] = category_key_f64(f)
                else:
                    mp[j * n_rows + r] = Float32(f)
            elif cw == w.int_word or cw == w.bool_word:
                var v = Int(t[].PyLong_AsLongLong(cell))
                if v == -1 and t[].PyErr_Occurred():
                    t[].PyErr_Clear()
                    raise Error("mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                + "] is an integer outside the 64-bit range")
                if slot >= 0:
                    if not category_integer_in_range(v):
                        raise Error("mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                    + "] is an integer category outside -2**61 .. 2**61; pass it as text")
                    kp[slot * n_rows + r] = category_key_integer(v)
                else:
                    mp[j * n_rows + r] = Float32(v)
            elif cw == w.str_word and slot >= 0:
                var size = 0
                var utf8 = t[].PyUnicode_AsUTF8AndSize(cell, Int(MutPointer(to=size)))
                if utf8 == 0:
                    t[].PyErr_Clear()
                    raise Error("mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                + "] is not valid text (it cannot be encoded as UTF-8)")
                # CPython's UTF-8 cache lives as long as the cell, which its row holds
                kp[slot * n_rows + r] = category_key_text(utf8, size)
            elif cw == w.none_word and slot >= 0:
                kp[slot * n_rows + r] = CATEGORY_KEY_MISSING
            else:
                var kind = w.buffer_kind(cw)
                if kind == 0 and slot >= 0:
                    raise Error("TypeError: mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                + "] is not an int, float, bool, str, bytes, None or a NumPy scalar of "
                                + "those; a categorical cell is admitted by its exact type")
                if kind == 0 or ((kind == _CELL_STR or kind == _CELL_BYTES) and slot < 0):
                    raise Error("TypeError: mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                + "] is not a number; only the cat_features columns may hold text or None")
                if kind == _CELL_STR:
                    # numpy.str_ is a str: the same C read, no method of the cell
                    var size = 0
                    var utf8 = t[].PyUnicode_AsUTF8AndSize(cell, Int(MutPointer(to=size)))
                    if utf8 == 0:
                        t[].PyErr_Clear()
                        raise Error("mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                    + "] is not valid text (it cannot be encoded as UTF-8)")
                    kp[slot * n_rows + r] = category_key_text(utf8, size)
                    continue
                # the value the cell exports (bytes, or a NumPy scalar's own
                # storage, native byte order): read, then released at once
                if t[].PyObject_GetBuffer(cell, view_ptr, Int32(0)) != 0:
                    t[].PyErr_Clear()
                    raise Error("TypeError: mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                + "] does not export its value as a buffer")
                var at = view[0]
                var nbytes = view[2]
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
                    raise Error("TypeError: mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                + "] is a NumPy scalar of " + String(nbytes) + " bytes this reader does not hold")
                if kind == _CELL_BYTES:
                    kp[slot * n_rows + r] = key
                elif is_real:
                    if slot >= 0:
                        kp[slot * n_rows + r] = category_key_f64(real)
                    else:
                        mp[j * n_rows + r] = Float32(real)
                elif slot >= 0:
                    if too_wide or not category_integer_in_range(whole):
                        raise Error("mojolearn: " + name + "[" + String(r) + "][" + String(j)
                                    + "] is an integer category outside -2**61 .. 2**61; pass it as text")
                    kp[slot * n_rows + r] = category_key_integer(whole)
                elif too_wide:
                    mp[j * n_rows + r] = Float32(real)
                else:
                    mp[j * n_rows + r] = Float32(whole)
    _ = rows  # keep the Python object alive through the borrowed reads above
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
    holds text or None there."""
    var b = trees_builtins()
    var nk = trees_keys()
    var cls = py_type(X)
    var ndarray = py_get(context, nk[].k_ndarray)
    if ndarray is not None and cls is ndarray:
        # an exact ndarray: its dtype kind is a C slot read, `tolist` its own C method
        var kind = String(py=X.dtype.kind)
        # object, unicode and bytes arrays: their cells through the array's
        # own C `tolist` (str for U, bytes for S), then the exact-type read
        if kind == "O" or kind == "U" or kind == "S":
            if Int(py=X.ndim) != 2:
                raise Error("mojolearn: " + name + " must be 2-D, got " + String(Int(py=X.ndim)) + "-D")
            var listed = py_call(ndarray.tolist, X)
            return _object_rows(listed, name, columns, keys, context)
    elif cls is b.list or cls is b.tuple:
        var w = _CellWords()
        if len(X) != 0 and _rows_hold_text(X, columns, w, name):
            return _object_rows(X, name, columns, keys, context)
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
