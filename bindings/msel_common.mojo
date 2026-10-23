# SPDX-License-Identifier: Apache-2.0
# Shared pieces of the native model_selection owner (lane mcb2-msel): the
# base binding's fold and index helpers resolved by name (the same entries
# python/mojolearn/model_selection.py called through `_buffer._native`, so a
# test that plants a callable there still sees the call), the input step
# (`_buffer._materialize`, `_labels.flatten_labels` / `encode_labels`, the
# input admission the Python called and that this owner hands the caller's
# objects to unchanged), int64 row Arrays, the counter random generator of
# the splitters (`_expansion_metrics.CounterRng`, its x_metrics programs
# built with the shared native `_Prog`), the group codes (`_GroupCodes`) and
# the mask and index splits. Every helper is called with the arguments the
# Python passed, in the same order, so the folds and draws are the same.
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.optimizer_public_words import stash_python_error, PYTHON_ERROR_PENDING, _restore_python_error
from bindings.classical_arrays import classical_address, classical_wrap_array
from bindings.family_common import (
    FAMILY_I64P, FAMILY_I32P, FAMILY_F64P, family_array, family_call, family_call_kw, family_isinstance,
    family_is_str, family_repr, family_str, family_type_name,
)
from bindings.metrics_reg_native import _Prog, _reg_run, _arena_base, _arena_int
from bindings.native_buffer_owner import native_storage_allocate

comptime MSEL_NONE = -1
comptime MSEL_U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime MSEL_U32P = MutPointer[UInt32, MutUntrackedOrigin]
#: `_expansion_metrics._FOLD_ROWS_BOUND`: the largest K-fold row table.
comptime MSEL_FOLD_ROWS_BOUND = 1 << 27
comptime MSEL_GOLDEN: UInt64 = 0x9E3779B97F4A7C15


# ---------------------------------------------------------------- calls


def msel_b() raises -> PythonObject:
    return Python.import_module("builtins")


def msel_native(name: String, context: PythonObject) raises -> PythonObject:
    """The base binding's helper `name` (`_buffer._native(name)`, binding
    acquisition, cached there)."""
    return family_call(context["family_msel_native"], PythonObject(name))


def msel_entry(name: String, context: PythonObject) raises -> PythonObject:
    """A base binding helper, resolved at each use as `_native(name)` was
    (its own cache keeps the lookup one dict read; a planted callable is
    seen)."""
    return msel_native(name, context)


def msel_try(callable: PythonObject, args: PythonObject, classes: PythonObject,
             context: PythonObject) raises -> PythonObject:
    """`callable(*args)`; an exception of `classes` returns the miss
    sentinel (`context["family_msel_miss"]`), any other is restored at the
    door unchanged."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, PyObjectPtr())
    if result:
        return PythonObject(from_owned=result)
    if cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(classes._obj_ptr) != 0:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return context["family_msel_miss"]
    var failure = stash_python_error()
    raise failure^


def msel_apply(callable: PythonObject, args: PythonObject) raises -> PythonObject:
    """`callable(*args)` for an argument tuple; its exception is restored at
    the door unchanged."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def msel_missed(value: PythonObject, context: PythonObject) -> Bool:
    return value is context["family_msel_miss"]


def _error_class(message: String) raises -> PythonObject:
    """The Python class the door (bindings/classical_native.mojo) raises
    for an owner Error: its mapped prefixes, else ValueError."""
    var b = msel_b()
    var names: List[String] = ["TypeError", "OverflowError", "NotImplementedError", "ImportError", "IndexError",
                               "AttributeError", "RuntimeError"]
    for i in range(len(names)):  # small-loop(error classes: the door's prefixes)
        if message.startswith(names[i] + ": "):
            return py_call(b.getattr, b, PythonObject(names[i]))
    return b.ValueError


def msel_catch(error: Error, classes: PythonObject) raises -> Bool:
    """Whether `error` (an owner Error or a stashed Python exception) is an
    instance of `classes`, as the Python `except classes:` tested. A caught
    stashed exception is cleared; an uncaught one stays stashed, so the
    caller re-raises `error` unchanged."""
    var message = String(error)
    if message == PYTHON_ERROR_PENDING:
        _ = _restore_python_error()
        ref cpy = Python().cpython()
        if cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(classes._obj_ptr) != 0:
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
            return True
        _ = stash_python_error()
        return False
    return Bool(py=py_call(msel_b().issubclass, _error_class(message), classes))


def msel_warn(message: String, category: PythonObject) raises:
    """`warnings.warn(message, category)`, attributed to the public call's
    caller."""
    var keywords = Python.dict()
    py_setitem(keywords, "stacklevel", PythonObject(2))
    _ = family_call_kw(Python.import_module("warnings").warn, keywords, PythonObject(message), category)


# ---------------------------------------------------------------- values


def msel_is_bool(value: PythonObject) raises -> Bool:
    """`_labels.is_bool`: Python's bool or NumPy's bool_."""
    return family_isinstance(value, msel_b().bool) or family_type_name(value) == "bool_"


def msel_is_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    return family_isinstance(value, context["family_msel_integral"])


def msel_is_int_arg(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(v, numbers.Integral) and not is_bool(v)`."""
    return msel_is_integral(value, context) and not msel_is_bool(value)


def msel_is_real(value: PythonObject, context: PythonObject) raises -> Bool:
    return family_isinstance(value, context["family_msel_real"])


def msel_int(value: PythonObject) raises -> Int:
    return Int(py=family_call(msel_b().int, value))


def msel_op(name: String) raises -> PythonObject:
    return py_call(msel_b().getattr, Python.import_module("operator"), PythonObject(name))


def msel_cmp(name: String, a: PythonObject, b: PythonObject) raises -> Bool:
    """`operator.<name>(a, b)` as a truth value (the Python comparison)."""
    return Bool(py=family_call(msel_b().bool, family_call(msel_op(name), a, b)))


def msel_materialize(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer._materialize(value, name)[0]` (the input step)."""
    return family_call(context["family_msel_materialize"], value, PythonObject(name))[0]


def msel_try_materialize(value: PythonObject, name: String, classes: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """`_materialize(value, name)[0]`, or the miss sentinel on `classes`."""
    var got = msel_try(context["family_msel_materialize"], Python.tuple(value, PythonObject(name)), classes, context)
    if msel_missed(got, context):
        return got
    return got[0]


def msel_flatten(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels(value)` (the label input step)."""
    return family_call(context["family_msel_flatten"], value)


def msel_encode(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.encode_labels(value)`: (classes, int32 codes)."""
    return family_call(context["family_msel_encode"], value)


def msel_labels_list(y: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if y is None:
        raise Error("The '" + name + "' parameter should not be None.")
    return msel_flatten(y, context)


def msel_n_samples(x: PythonObject) raises -> Int:
    """`_n_samples(X)`: shape[0] when X has a nonempty shape, else len(X)."""
    var b = msel_b()
    if Bool(py=family_call(b.hasattr, x, PythonObject("shape"))):
        var shape = family_call(b.getattr, x, PythonObject("shape"))
        if Int(py=family_call(b.len, shape)) > 0:
            return msel_int(shape[0])
    return Int(py=family_call(b.len, x))


def msel_is_array(value: PythonObject, context: PythonObject) raises -> Bool:
    return py_type(value) is context["array_class"]


# ---------------------------------------------------------------- arrays


def msel_addr(array: PythonObject) raises -> Int:
    return Int(py=classical_address(array))


def msel_p64(array: PythonObject) raises -> FAMILY_I64P:
    return FAMILY_I64P(unsafe_from_address=msel_addr(array))


def msel_p32(array: PythonObject) raises -> FAMILY_I32P:
    return FAMILY_I32P(unsafe_from_address=msel_addr(array))


def msel_pf64(array: PythonObject) raises -> FAMILY_F64P:
    return FAMILY_F64P(unsafe_from_address=msel_addr(array))


def msel_new(n: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    """`empty((n,), typestr)` (zero filled)."""
    return family_array(Python.tuple(n), typestr, context)


def msel_new2(rows: Int, cols: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    return family_array(Python.tuple(rows, cols), typestr, context)


def msel_i64_list(values: List[Int], context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, '<i8')`."""
    var out = msel_new(len(values), "<i8", context)
    var p = msel_p64(out)
    for i in range(len(values)):  # driver-loop: over a short result list (sizes, folds)
        p[i] = Int64(values[i])
    return out


def msel_i32_list(values: List[Int], context: PythonObject) raises -> PythonObject:
    var out = msel_new(len(values), "<i4", context)
    var p = msel_p32(out)
    for i in range(len(values)):  # driver-loop: over a short per-group or per-class table
        p[i] = Int32(values[i])
    return out


def msel_f64_list(values: List[Float64], context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, '<f8')`."""
    var out = msel_new(len(values), "<f8", context)
    var p = msel_pf64(out)
    for i in range(len(values)):  # driver-loop: over one score per fold or candidate
        p[i] = values[i]
    return out


def msel_f64_block(values: List[Float64], rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(flat, '<f8').reshape((rows, cols))`."""
    var out = msel_new2(rows, cols, "<f8", context)
    var p = msel_pf64(out)
    for i in range(len(values)):  # driver-loop: over the score block (sizes or candidates x folds)
        p[i] = values[i]
    return out


def msel_read_i64(array: PythonObject) raises -> List[Int]:
    """The values of a short int64 Array (fold sizes, class counts)."""
    var out = List[Int]()
    var p = msel_p64(array)
    for i in range(Int(py=array.size)):  # driver-loop: over a short table (classes, folds)
        out.append(Int(p[i]))
    return out^


def msel_slice_i64(array: PythonObject, start: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """`array[start:start + count]` of a C int64 Array, copied."""
    var out = msel_new(count, "<i8", context)
    if count > 0:
        memcpy(dest=msel_p64(out), src=msel_p64(array) + start, count=count)
    return out


def msel_index_i64(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A 1-D integer index Array as C int64 (`value` itself when it is one,
    else the base binding's `cast_elements` widening, the `astype('<i8')`)."""
    var dtype = String(py=value.dtype)
    if dtype == "<i8":
        return value
    var codes = context["family_msel_codes"]
    if not py_contains(codes, PythonObject(dtype)):
        # a narrow integer dtype the cast does not name: `array('q', view)`
        # widens it in C
        var store = family_call(Python.import_module("array").array, PythonObject("q"), value._mv)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)
    var out = msel_new(Int(py=value.size), "<i8", context)
    if Int(py=value.size) > 0:
        _ = family_call(msel_entry("cast_elements", context), PythonObject(msel_addr(value)), codes[PythonObject(dtype)],
                        PythonObject(msel_addr(out)), PythonObject(3), value.size)
    return out


def msel_arange(start: Int, n: Int, context: PythonObject) raises -> PythonObject:
    """`_arange_rows(start, n)`: int64 rows start .. start + n - 1."""
    var out = msel_new(n, "<i8", context)
    if n > 0:
        _ = family_call(msel_entry("arange_i64", context), PythonObject(msel_addr(out)), PythonObject(start), PythonObject(n))
    return out


def msel_range_index(lo: Int, hi: Int, context: PythonObject) raises -> PythonObject:
    """`_as_index(range(lo, hi))`."""
    return msel_arange(lo, max(hi - lo, 0), context)


def msel_split_mask(mask: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_split_mask`: (train, test), the ascending rows whose mask byte is 0
    / nonzero, by `count_mask_u8` and `select_mask_u8_i64`."""
    var maddr = msel_addr(mask)
    var n_test = msel_int(family_call(msel_entry("count_mask_u8", context), PythonObject(maddr), PythonObject(n)))
    var test = msel_new(n_test, "<i8", context)
    var train = msel_new(n - n_test, "<i8", context)
    _ = family_call(msel_entry("select_mask_u8_i64", context), PythonObject(maddr), PythonObject(n),
                    PythonObject(msel_addr(test) if n_test else 0), PythonObject(msel_addr(train) if n - n_test else 0))
    return Python.tuple(train, test)


def msel_split_indices(iaddr: Int, k: Int, n: Int, context: PythonObject) raises -> PythonObject:
    """`_split_indices`: (train, test) from k int64 test rows at `iaddr`."""
    var mask = msel_new(n, "<u1", context)
    _ = family_call(msel_entry("mask_from_indices_u8", context), PythonObject(iaddr if k else 0), PythonObject(k),
                    PythonObject(n), PythonObject(msel_addr(mask)))
    return msel_split_mask(mask, n, context)


def msel_index_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_index_copy(value)`: an int64 / int32 1-D buffer widened to an
    owned int64 Array; anything else `_as_index(flatten_labels(value))`."""
    var b = msel_b()
    if not family_isinstance(value, Python.tuple(b.list, b.tuple, b.range)):
        var a = msel_try_materialize(value, "cv indices", Python.tuple(b.TypeError, b.ValueError), context)
        if not msel_missed(a, context) and Int(py=a.ndim) == 1:
            var dtype = String(py=a.dtype)
            if dtype == "<i8":
                return msel_slice_i64(a, 0, Int(py=a.size), context)
            if dtype == "<i4":
                return msel_index_i64(a, context)
    return msel_as_index(msel_flatten(value, context), context)


def msel_as_index(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_as_index(values)` of a Python sequence of ints: `array('q', values)`
    (ints and bools as they are, anything else through int()) as an owned
    int64 Array."""
    var b = msel_b()
    var array_mod = Python.import_module("array")
    var store = msel_try(array_mod.array, Python.tuple(PythonObject("q"), values), b.TypeError, context)
    if msel_missed(store, context):
        store = family_call(array_mod.array, PythonObject("q"), py_call(b.map, b.int, values))
    return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)


# ---------------------------------------------------------------- groups


struct MselGroups(Movable):
    """`_GroupCodes`: groups (or labels) under the order rule as int32 codes
    with their per-code row counts."""
    var codes: PythonObject
    var classes: PythonObject
    var m: Int
    var n: Int
    var counts: List[Int]
    var scratch: PythonObject

    def __init__(out self, values: PythonObject, n: Int, name: String, context: PythonObject) raises:
        if values is None:
            _ = msel_labels_list(values, name, context)
        _ = msel_entry("fold_ids", context)
        _ = msel_entry("select_fold_i64", context)
        _ = msel_entry("gather_i32", context)
        var got = msel_encode(values, context)
        var classes = got[0]
        var codes = got[1]
        if Int(py=codes.ndim) != 1 or Int(py=codes.size) != n:
            raise Error(name + " has " + String(Int(py=codes.size)) + " labels for " + String(n) + " samples")
        if String(py=codes.dtype) != "<i4":
            var narrow = msel_new(n, "<i4", context)
            _ = family_call(msel_entry("cast_elements", context), PythonObject(msel_addr(codes)),
                            context["family_msel_codes"][codes.dtype], PythonObject(msel_addr(narrow)),
                            PythonObject(2), PythonObject(n))
            codes = narrow
        var m = len(classes)
        self.counts = List[Int](length=m, fill=0)
        if m > 0:
            var class_counts = msel_new(m, "<i8", context)
            var fold_store = msel_new(n, "<i4", context)
            var fold_counts = msel_new(2, "<i8", context)
            _ = family_call(msel_entry("fold_ids", context), PythonObject(msel_addr(codes)), PythonObject(n), PythonObject(m),
                            PythonObject(2), PythonObject(msel_addr(class_counts)), PythonObject(msel_addr(fold_store)),
                            PythonObject(msel_addr(fold_counts)))
            self.counts = msel_read_i64(class_counts)
        self.codes = codes
        self.classes = classes
        self.m = m
        self.n = n
        self.scratch = PythonObject(None)

    def mapped(self, table: List[Int], context: PythonObject) raises -> PythonObject:
        """Each row's `table[code]` as int32 words."""
        var tbl = msel_i32_list(table, context)
        return self.mapped_words(tbl, context)

    def mapped_words(self, tbl: PythonObject, context: PythonObject) raises -> PythonObject:
        var dst = msel_new(self.n, "<i4", context)
        _ = family_call(msel_entry("gather_i32", context), PythonObject(msel_addr(tbl)), tbl.size,
                        PythonObject(msel_addr(self.codes)), PythonObject(self.n), PythonObject(msel_addr(dst)))
        return dst

    def rows(self, words: PythonObject, want: Int, size: Int, context: PythonObject) raises -> PythonObject:
        """(train, test): the ascending rows whose word is `want` (test) and
        every other row (train)."""
        var test = msel_new(size, "<i8", context)
        var train = msel_new(self.n - size, "<i8", context)
        var got = msel_int(family_call(msel_entry("select_fold_i64", context), PythonObject(msel_addr(words)),
                                       PythonObject(self.n), PythonObject(want), PythonObject(msel_addr(test)),
                                       PythonObject(msel_addr(train))))
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
        return Python.tuple(train, test)

    def only(mut self, words: PythonObject, want: Int, size: Int, context: PythonObject) raises -> PythonObject:
        """The ascending rows whose word is `want` (the rest discarded)."""
        if self.scratch is None:
            self.scratch = msel_new(self.n, "<i8", context)
        var out = msel_new(size, "<i8", context)
        var got = msel_int(family_call(msel_entry("select_fold_i64", context), PythonObject(msel_addr(words)),
                                       PythonObject(self.n), PythonObject(want), PythonObject(msel_addr(out)),
                                       PythonObject(msel_addr(self.scratch))))
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
        return out

    def split_by_fold(self, to_fold: PythonObject, k: Int, sizes: PythonObject,
                      context: PythonObject) raises -> PythonObject:
        """[(train, test)] per fold from each group's fold (int32 words) and
        each fold's row count (an int64 Array)."""
        var words = self.mapped_words(to_fold, context)
        var out = Python.list()
        var sz = msel_p64(sizes)
        for f in range(k):  # driver-loop: over folds
            _ = py_call(out.append, self.rows(words, f, Int(sz[f]), context))
        return out


# ---------------------------------------------------------------- random draws


def msel_mix64(var x: UInt64) -> UInt64:
    """splitmix64's finalizer (`_expansion_metrics._mix64`), in wrapping
    64-bit arithmetic: the exact Python integers modulo 2**64."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


def msel_u64(value: PythonObject) raises -> UInt64:
    """`int(value) & (2**64 - 1)` of a Python int."""
    var b = msel_b()
    var whole = family_call(b.int, value)
    var lo = Int(py=py_call(msel_op("and_"), whole, PythonObject(0xFFFFFFFF)))
    var hi = Int(py=py_call(msel_op("and_"), py_call(msel_op("rshift"), whole, PythonObject(32)), PythonObject(0xFFFFFFFF)))
    return (UInt64(hi) << 32) | UInt64(lo)


def msel_signed32(v: UInt64) -> Int:
    """A 32-bit word as the signed parameter `_Prog.stage` records."""
    var w = Int(v & 0xFFFFFFFF)
    return w - (1 << 32) if w >= (1 << 31) else w


struct MselRng(Movable):
    """`CounterRng`: draw k of seed s is a device permutation keyed by
    `_mix64(s * GOLDEN + k + 1)`."""
    var seed: UInt64
    var draws: UInt64

    def __init__(out self, random_state: PythonObject, context: PythonObject) raises:
        var state = random_state
        if state is None:
            state = family_call(msel_b().int.from_bytes, py_call(Python.import_module("os").urandom, PythonObject(8)),
                                PythonObject("little"))
        if (msel_is_bool(state) or not msel_is_integral(state, context)
                or msel_cmp("lt", state, PythonObject(0))):
            raise Error("mojolearn model_selection: random_state must be None or a non-negative int; "
                        + "numpy RandomState / Generator objects are NOT IMPLEMENTED (their stream "
                        + "is numpy's, x_metrics/split.mojo DEVIATION 6108)")
        self.seed = msel_u64(state)
        self.draws = 0

    def salt(mut self) -> UInt64:
        var s = msel_mix64(self.seed * MSEL_GOLDEN + self.draws + 1)
        self.draws += 1
        return s

    def permute_stage(mut self, mut prog: _Prog, n: Int) raises -> Int:
        """The next draw as a `permute` stage of `prog`; returns its slot."""
        var s = self.salt()
        var out = prog.alloc(max(n, 1))
        prog.stage("permute", 1, [n, out, msel_signed32(s), msel_signed32(s >> 32)])
        return out

    def permutation_rows(mut self, sizes: List[Int], context: PythonObject) raises -> PythonObject:
        """One draw per size, in order, as int64 row Arrays (`permutation_rows`)."""
        var prog = _Prog(context["family_metrics_ops"])
        var outs = List[Int]()
        for i in range(len(sizes)):  # driver-loop: over the requested draws
            var o = self.permute_stage(prog, sizes[i])
            var slot = prog.alloc(2 * sizes[i])
            var w = prog.want(slot, 2 * sizes[i])
            if sizes[i] > 0:
                prog.stage("rows64", sizes[i], [o, w])
            outs.append(w)
        var result = Python.list()
        if len(outs) == 0:
            return result
        var arena = _reg_run(prog, PythonObject(None), context)
        var base = _arena_base(arena)
        for i in range(len(sizes)):  # driver-loop: over the requested draws
            var rows = msel_new(sizes[i], "<i8", context)
            if sizes[i] > 0:
                memcpy(dest=msel_p64(rows), src=FAMILY_I64P(unsafe_from_address=base + 4 * outs[i]), count=sizes[i])
            _ = py_call(result.append, rows)
        _ = prog^
        return result

    def permutation(mut self, size: Int, context: PythonObject) raises -> PythonObject:
        """`permutation(size)`: one draw read back as a list of ints (the
        arena words as `array('i')`, as `_Prog.ints` read them)."""
        var prog = _Prog(context["family_metrics_ops"])
        var slot = self.permute_stage(prog, size)
        var o = prog.want(slot, size)
        var arena = _reg_run(prog, PythonObject(None), context)
        _ = prog^
        var store = py_call(Python.import_module("array").array, PythonObject("i"))
        _ = py_call(store.frombytes, py_call(py_call(arena.__getitem__, py_call(msel_b().slice, PythonObject(o), PythonObject(o + size))).tobytes))
        return py_call(msel_b().list, store)

    def approximate_modes(mut self, counts: List[Int], n_train: Int, n_test: Int,
                          context: PythonObject) raises -> List[Int]:
        """StratifiedShuffleSplit's per-class train then test counts (two
        `approx_mode` stages of one program; the draws they report are
        consumed). Returns n_i followed by t_i."""
        var k = len(counts)
        var prog = _Prog(context["family_metrics_ops"])
        var c = prog.put_i32(msel_i32_list(counts, context))
        var out_slot = prog.alloc(2 * k)
        var out = prog.want(out_slot, 2 * k)
        var used_slot = prog.alloc(2)
        var used = prog.want(used_slot, 2)
        var w0 = msel_signed32(self.seed)
        var w1 = msel_signed32(self.seed >> 32)
        var w2 = msel_signed32(self.draws)
        var w3 = msel_signed32(self.draws >> 32)
        prog.stage("approx_mode", k, [k, c, MSEL_NONE, n_train, out, MSEL_NONE, used, w0, w1, w2, w3])
        prog.stage("approx_mode", k, [k, c, out, n_test, out + k, used, used + 1, w0, w1, w2, w3])
        var arena = _reg_run(prog, PythonObject(None), context)
        self.draws += UInt64(_arena_int(arena, used) + _arena_int(arena, used + 1))
        var both = List[Int]()
        for i in range(2 * k):  # driver-loop: over classes (train and test counts)
            both.append(_arena_int(arena, out + i))
        _ = prog^
        return both^


def msel_mix_seed(seed: UInt64, r: Int) -> Int:
    """`_mix_seed(seed, r)`: repeat r's seed."""
    return Int(msel_mix64(seed * MSEL_GOLDEN + UInt64(0x632BE59BD9B4E019) * UInt64(r + 1)) >> 1)


# ---------------------------------------------------------------- shuffle sizes


def _size_kind(v: PythonObject, context: PythonObject) raises -> String:
    if v is None:
        return "n"
    if msel_is_bool(v):
        return "x"
    if msel_is_integral(v, context):
        return "i"
    if msel_is_real(v, context):
        return "f"
    return "x"


def msel_shuffle_sizes(n: Int, test_size_in: PythonObject, train_size: PythonObject,
                       default_test: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
    """`_validate_shuffle_split`: (n_train, n_test) with its refusals."""
    var test_size = test_size_in
    if test_size is None and train_size is None:
        test_size = default_test
    var tk = _size_kind(test_size, context)
    var rk = _size_kind(train_size, context)
    var pn = PythonObject(n)
    var zero = PythonObject(0)
    var one = PythonObject(1)
    if ((tk == "i" and (msel_cmp("ge", test_size, pn) or msel_cmp("le", test_size, zero)))
            or (tk == "f" and not (msel_cmp("lt", zero, test_size) and msel_cmp("lt", test_size, one)))):
        raise Error("test_size=" + family_str(test_size) + " should be either positive and smaller than the number of "
                    + "samples " + String(n) + " or a float in the (0, 1) range")
    if ((rk == "i" and (msel_cmp("ge", train_size, pn) or msel_cmp("le", train_size, zero)))
            or (rk == "f" and not (msel_cmp("lt", zero, train_size) and msel_cmp("lt", train_size, one)))):
        raise Error("train_size=" + family_str(train_size) + " should be either positive and smaller than the number "
                    + "of samples " + String(n) + " or a float in the (0, 1) range")
    if rk == "x":
        raise Error("Invalid value for train_size: " + family_str(train_size))
    if tk == "x":
        raise Error("Invalid value for test_size: " + family_str(test_size))
    var add = msel_op("add")
    var mul = msel_op("mul")
    if rk == "f" and tk == "f" and msel_cmp("gt", family_call(add, train_size, test_size), one):
        raise Error("The sum of test_size and train_size = " + family_str(family_call(add, train_size, test_size))
                    + ", should be in the (0, 1) range. Reduce test_size and/or train_size.")
    var math = Python.import_module("math")
    var n_test = PythonObject(None)
    var n_train = PythonObject(None)
    if tk == "f":
        n_test = family_call(math.ceil, family_call(mul, test_size, pn))
    else:
        n_test = test_size
    if rk == "f":
        n_train = family_call(math.floor, family_call(mul, train_size, pn))
    else:
        n_train = train_size
    if train_size is None:
        n_train = family_call(msel_op("sub"), pn, n_test)
    elif test_size is None:
        n_test = family_call(msel_op("sub"), pn, n_train)
    var total = family_call(add, n_train, n_test)
    if msel_cmp("gt", total, pn):
        raise Error("The sum of train_size and test_size = " + family_str(total) + ", should be smaller than "
                    + "the number of samples " + String(n) + ". Reduce test_size and/or train_size.")
    var tr = msel_int(n_train)
    var te = msel_int(n_test)
    if tr == 0:
        raise Error("With n_samples=" + String(n) + ", test_size=" + family_str(test_size) + " and train_size="
                    + family_str(train_size) + ", the resulting train set will be empty. Adjust any of the "
                    + "aforementioned parameters.")
    return (tr, te)


# ---------------------------------------------------------------- resident rows


def msel_resident_op(res: PythonObject, op: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """One `_Resident` lifecycle operation through its compiled door
    (bindings/arena_resident_native.mojo), as the Python methods call it."""
    return family_call(context["family_msel_resident_call"], res, PythonObject(op), args,
                       context["family_msel_resident_context"])


def msel_resident_new(context: PythonObject) raises -> PythonObject:
    """`_Resident()`: the instance allocated and initialized by its door."""
    var res = native_storage_allocate(context["family_msel_resident_class"])
    _ = msel_resident_op(res, "msel_resident_init", Python.tuple(), context)
    return res


def msel_resident_take(res: PythonObject, values: PythonObject, indices: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    return msel_resident_op(res, "msel_resident_take", Python.tuple(values, indices), context)


def msel_resident_release(res: PythonObject, values: PythonObject, context: PythonObject) raises:
    _ = msel_resident_op(res, "msel_resident_release", Python.tuple(values), context)


def msel_resident_close(res: PythonObject, context: PythonObject) raises:
    _ = msel_resident_op(res, "msel_resident_close", Python.tuple(), context)


def msel_take_any(values: PythonObject, indices: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_take_any` / `_take_rows`: a list or tuple taken by index (a list),
    else one device gather out of a `_Resident` used once."""
    var b = msel_b()
    if family_isinstance(values, Python.tuple(b.list, b.tuple)):
        return py_call(b.list, py_call(b.map, py_call(b.getattr, values, PythonObject("__getitem__")), indices._mv))
    var res = msel_resident_new(context)
    var out = PythonObject(None)
    try:
        out = msel_resident_take(res, values, indices, context)
    finally:
        msel_resident_close(res, context)
    return out


def msel_lazy(model: PythonObject, op: String, items: PythonObject, context: PythonObject) raises -> PythonObject:
    """A lazy iterator of `door(model, op, item, context)` over `items` (each
    an argument tuple): `map` over the compiled door, so a splitter with
    one split per row or per combination makes each split on demand, as the
    Python generator did, with no Python frame."""
    var itertools = Python.import_module("itertools")
    return py_call(msel_b().map, context["family_msel_door"], py_call(itertools.repeat, model),
                   py_call(itertools.repeat, PythonObject(op)), items, py_call(itertools.repeat, context))


# ---------------------------------------------------------------- scorers


def _msel_attr(model: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` with its own AttributeError."""
    return family_call(Python.import_module("builtins").getattr, model, PythonObject(name))

def msel_get_scorer(scoring: PythonObject, context: PythonObject) raises -> PythonObject:
    """`get_scorer(scoring)`."""
    var b = Python.import_module("builtins")
    if scoring is None or Bool(py=py_call(b.callable, scoring)):
        return scoring
    if family_is_str(scoring):
        var table = context["family_msel_scorers"]
        if not py_contains(table, scoring):
            raise Error(family_repr(scoring) + " is not a valid scoring value. Use "
                        + "sklearn.metrics.get_scorer_names() to get valid options.")
        return table[scoring]
    raise Error("scoring must be a str, a callable or None, got " + family_repr(scoring))


def msel_score(estimator: PythonObject, x: PythonObject, y: PythonObject, scorer: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`model_selection._score(estimator, X, y, scorer)`: the estimator's own
    `score` without a scorer; a real scalar, as a float."""
    var b = Python.import_module("builtins")
    var value: PythonObject
    if scorer is None:
        value = family_call(_msel_attr(estimator, "score"), x, y)
    else:
        value = family_call(scorer, estimator, x, y)
    if not family_isinstance(value, context["family_msel_real"]):
        raise Error("TypeError: scoring must return a real scalar")
    return family_call(b.float, value)
