# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The user-object reader: a Python object that cannot cross into Mojo as a
# buffer (a list of label objects, a ragged nested list, a user's index
# pairs or group ids) is handed to ONE native call, which reads it once
# through the C API and returns typed stores (array.array words) or the
# objects placed where they go. No Python bytecode runs per element: the
# per-element loops below are the read itself, each tagged with the object
# it reads. Lane purge-user-objects (2026-10-22) moved these loops here from
# model_selection.py, ensemble.py, _umap_impl.py, _metrics_impl.py,
# linear_model.py, resample.py, _array.py, _expansion_trees.py and
# _expansion_prep.py (python/mojolearn/_user_objects.py is the Python door).
#
#   ("take", values, address, count, width)  -> [values[i] for i in idx]
#   ("first_seen", values)                    -> (classes, int32 codes) by dict equality
#   ("lookup", classes, values)               -> (int32 codes, -1 unseen; distinct unseen values)
#   ("kinds", values)                         -> counts per kind (KIND_* below)
#   ("ints", values) / ("floats", values)     -> int64 / float64 words of int(v) / float(v)
#   ("strs", values)                          -> [str(v) for v in values]
#   ("column", rows, j) / ("column_f32", rows, j) -> [r[j] for r in rows] / its float32 words
#   ("transpose", cols)                       -> [list(r) for r in zip(*cols)]
#   ("scatter", rows, address, count, vals)   -> rows[idx[k]] = vals[k]
#   ("flatten", obj)                          -> every leaf (tolist() honoured at each level)
#   ("nested", obj, array_type)               -> (shape, leaves) of a rectangular nested list
#   ("group_runs", ids)                       -> int64 run lengths of a ranking pool's group ids
#   ("pairs", rows, weights, n_rows)          -> (int64 winner/loser words, float64 weights)
#   ("sorted_codes", values, unknown)         -> float32 codes over sorted((str(type(v)), v))
#   ("f32_exact", values)                     -> [float(v)] when float32 holds every real label, else None
#   ("indicator", rows)                       -> (status, float32 words, width) of a 0/1 label matrix
#   ("group_rows", codes, k)                  -> k int64 stores: the rows of each code
#   ("group_permute", values, rows, perms)    -> values permuted within each group's rows
#
# A refusal comes back as ("__refuse__", exception name, message); the door
# raises it, so the exception type and wording are the ones the Python had.
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type
from core.user_objects import group_run_sizes

comptime KIND_STR = 0         # isinstance(v, str)
comptime KIND_BOOL = 1        # bool or NumPy bool_ (_labels.is_bool)
comptime KIND_EXACT_INT = 2   # type(v) is int
comptime KIND_INTEGRAL = 3    # isinstance(v, numbers.Integral)
comptime KIND_REAL = 4        # isinstance(v, numbers.Real)
comptime KIND_INTVAL = 5      # Integral, or a finite integer-valued Real
comptime KIND_SEQ = 6         # isinstance(v, (list, tuple))
comptime KIND_INT_OR_BOOL = 7 # Integral or bool / bool_
comptime KIND_PLAIN_INT = 8   # Integral and not bool / bool_
comptime KIND_REAL_NONBOOL = 9  # isinstance(v, numbers.Real) and not isinstance(v, bool)
comptime N_KINDS = 10


struct _Env(Movable):
    var b: PythonObject
    var op: PythonObject
    var integral: PythonObject
    var real: PythonObject
    var seq: PythonObject
    var array: PythonObject

    def __init__(out self) raises:
        var b = Python.import_module("builtins")
        var numbers = Python.import_module("numbers")
        self.seq = Python.tuple(b.list, b.tuple)
        self.integral = numbers.Integral
        self.real = numbers.Real
        self.op = Python.import_module("operator")
        self.array = Python.import_module("array").array
        self.b = b

    def isinstance(self, v: PythonObject, cls: PythonObject) raises -> Bool:
        return Bool(py=py_call(self.b.isinstance, v, cls))

    def eq(self, a: PythonObject, b: PythonObject) raises -> Bool:
        return Bool(py=py_call(self.op.eq, a, b))

    def lt(self, a: PythonObject, b: PythonObject) raises -> Bool:
        return Bool(py=py_call(self.op.lt, a, b))

    def is_bool(self, v: PythonObject) raises -> Bool:
        if self.isinstance(v, self.b.bool):
            return True
        return String(py=py_type(v).__name__) == "bool_"

    def text(self, v: PythonObject) raises -> String:
        return String(py=py_call(self.b.str, v))

    def py_repr(self, v: PythonObject) raises -> String:
        return String(py=py_call(self.b.repr, v))

    def store(self, code: String, n: Int, itemsize: Int) raises -> PythonObject:
        return py_call(self.array, code, py_call(self.b.bytes, itemsize * max(n, 0)))


def _address(store: PythonObject) raises -> Int:
    return Int(py=store.buffer_info()[0])


def _refusal(kind: String, message: String) raises -> Error:
    # the door raises it as `kind` (TypeError, ValueError, IndexError, OverflowError)
    return Error("__refuse__" + kind + ":" + message)


def _index_at(address: Int, k: Int, width: Int) -> Int:
    if width == 4:
        return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[k])
    return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)[k])


def _take(env: _Env, values: PythonObject, address: Int, count: Int, width: Int) raises -> PythonObject:
    var n = len(values)
    var out = Python.list()
    for k in range(count):  # cpu-route: Python list of label objects read once by user_objects_c (take)
        var i = _index_at(address, k, width)
        var j = i + n if i < 0 else i
        if j < 0 or j >= n:
            raise _refusal("IndexError", "list index out of range")
        _ = py_call(out.append, values[j])
    return out


def _first_seen(env: _Env, values: PythonObject) raises -> PythonObject:
    # {v: i for i, v in enumerate(dict.fromkeys(values))} and each value's code:
    # dict equality and hashing, the object's own protocol, read from Mojo
    var n = len(values)
    var index = Python.dict()
    var classes = Python.list()
    var codes = env.store("i", n, 4)
    var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(codes))
    for i in range(n):  # cpu-route: Python list of label objects read once by user_objects_c (first_seen)
        var v = values[i]
        var k = len(classes)
        var got = py_call(index.setdefault, v, PythonObject(k))
        var code = Int(py=got)
        if code == k:
            _ = py_call(classes.append, v)
        out[i] = Int32(code)
    return Python.tuple(classes, codes)


def _lookup(env: _Env, classes: PythonObject, values: PythonObject) raises -> PythonObject:
    # {c: i for i, c in enumerate(classes)}.get(v, -1) per value (a repeated
    # class keeps its last index, as the comprehension does), and the distinct
    # unseen values in first-seen order
    var index = Python.dict()
    for i in range(len(classes)):  # cpu-route: Python list of class objects read once by user_objects_c (lookup)
        _ = py_call(index.__setitem__, classes[i], PythonObject(i))
    var n = len(values)
    var codes = env.store("i", n, 4)
    var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(codes))
    var unseen = Python.dict()
    var missing = Python.list()
    var none = PythonObject(None)
    for i in range(n):  # cpu-route: Python list of label objects read once by user_objects_c (lookup)
        var v = values[i]
        var got = py_call(index.get, v, none)
        if got is none:
            out[i] = Int32(-1)
            var before = len(unseen)
            _ = py_call(unseen.setdefault, v, none)
            if len(unseen) != before:
                _ = py_call(missing.append, v)
        else:
            out[i] = Int32(Int(py=got))
    return Python.tuple(codes, missing)


def _float_is_integer(x: Float64) -> Bool:
    # math.isfinite(x) and x.is_integer()
    if not (x - x == 0.0):
        return False
    if x >= 9007199254740992.0 or x <= -9007199254740992.0:
        return True  # every float this large is an integer
    return Float64(Int(x)) == x


def _is_intval(env: _Env, v: PythonObject) raises -> Bool:
    # math.isfinite(v) and float(v).is_integer(), for a Real
    return _float_is_integer(Float64(py=py_call(env.b.float, v)))


def _kinds(env: _Env, values: PythonObject) raises -> PythonObject:
    var counts = List[Int](length=N_KINDS, fill=0)
    var n = len(values)
    for i in range(n):  # cpu-route: Python list of label objects read once by user_objects_c (kinds)
        var v = values[i]
        var is_b = env.is_bool(v)
        var integral = env.isinstance(v, env.integral)
        var real = integral or env.isinstance(v, env.real)
        if env.isinstance(v, env.b.str):
            counts[KIND_STR] += 1
        if is_b:
            counts[KIND_BOOL] += 1
        if py_type(v) is env.b.int:
            counts[KIND_EXACT_INT] += 1
        if integral:
            counts[KIND_INTEGRAL] += 1
            counts[KIND_INTVAL] += 1
        elif real and _is_intval(env, v):
            counts[KIND_INTVAL] += 1
        if real:
            counts[KIND_REAL] += 1
        if env.isinstance(v, env.seq):
            counts[KIND_SEQ] += 1
        if integral or is_b:
            counts[KIND_INT_OR_BOOL] += 1
        if integral and not is_b:
            counts[KIND_PLAIN_INT] += 1
        if real and not env.isinstance(v, env.b.bool):
            counts[KIND_REAL_NONBOOL] += 1
    var out = Python.list()
    for k in range(N_KINDS):  # driver-loop: over the kind counters
        _ = py_call(out.append, PythonObject(counts[k]))
    return out


def _as_list(env: _Env, values: PythonObject) raises -> PythonObject:
    if py_type(values) is env.b.list:
        return values
    return py_call(env.b.list, values)


def _ints(env: _Env, values: PythonObject) raises -> PythonObject:
    var vals = _as_list(env, values)
    var n = len(vals)
    var words = env.store("q", n, 8)
    var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(words))
    var lo = PythonObject(-9223372036854775807 - 1)
    var hi = PythonObject(9223372036854775807)
    for i in range(n):  # cpu-route: Python int objects read once by user_objects_c (ints)
        var x = py_call(env.b.int, vals[i])
        if env.lt(x, lo) or env.lt(hi, x):
            raise _refusal("OverflowError", "Python int too large to convert to C long")
        out[i] = Int64(Int(py=x))
    return words


def _floats(env: _Env, values: PythonObject) raises -> PythonObject:
    var vals = _as_list(env, values)
    var n = len(vals)
    var words = env.store("d", n, 8)
    var out = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=_address(words))
    for i in range(n):  # cpu-route: Python number objects read once by user_objects_c (floats)
        out[i] = Float64(py=py_call(env.b.float, vals[i]))
    return words


def _strs(env: _Env, values: PythonObject) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # cpu-route: Python str objects read once by user_objects_c (strs)
        _ = py_call(out.append, py_call(env.b.str, values[i]))
    return out


def _column(env: _Env, rows: PythonObject, j: Int) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(rows)):  # cpu-route: Python label rows read once by user_objects_c (column)
        _ = py_call(out.append, rows[i][j])
    return out


def _column_f32(env: _Env, rows: PythonObject, j: Int) raises -> PythonObject:
    var n = len(rows)
    var words = env.store("f", n, 4)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_address(words))
    for i in range(n):  # cpu-route: Python probability rows read once by user_objects_c (column_f32)
        out[i] = Float32(Float64(py=py_call(env.b.float, rows[i][j])))
    return words


def _transpose(env: _Env, cols: PythonObject) raises -> PythonObject:
    var m = len(cols)
    var out = Python.list()
    if m == 0:
        return out
    var n = len(cols[0])
    for c in range(1, m):  # driver-loop: over the output columns
        n = min(n, len(cols[c]))
    for i in range(n):  # cpu-route: Python label columns read once by user_objects_c (transpose)
        var row = Python.list()
        for c in range(m):  # driver-loop: over the output columns
            _ = py_call(row.append, cols[c][i])
        _ = py_call(out.append, row)
    return out


def _scatter(env: _Env, rows: PythonObject, address: Int, count: Int, vals: PythonObject) raises -> PythonObject:
    var n = min(count, len(vals))
    for k in range(n):  # cpu-route: Python prediction objects read once by user_objects_c (scatter)
        _ = py_call(rows.__setitem__, PythonObject(_index_at(address, k, 8)), vals[k])
    return PythonObject(None)


def _flatten_into(env: _Env, obj: PythonObject, out: PythonObject) raises:
    var o = obj
    if Bool(py=py_call(env.b.hasattr, o, "tolist")):
        o = py_call(o.tolist)
    if env.isinstance(o, env.seq):
        for i in range(len(o)):  # cpu-route: nested Python list read once by user_objects_c (flatten)
            _flatten_into(env, o[i], out)
        return
    _ = py_call(out.append, o)


def _nested_into(env: _Env, obj: PythonObject, array_type: PythonObject, out: PythonObject) raises -> PythonObject:
    # _array._flatten: an Array contributes its shape and values; a str,
    # bytes or object without __len__ is a leaf; a list is rectangular
    if env.isinstance(obj, array_type):
        _ = py_call(out.extend, py_call(py_call(obj._as_c)._values))
        return obj.shape
    if env.isinstance(obj, Python.tuple(env.b.str, env.b.bytes)) or not Bool(py=py_call(env.b.hasattr, obj, "__len__")):
        _ = py_call(out.append, obj)
        return Python.tuple()
    var seq = py_call(env.b.list, obj)
    var n = len(seq)
    if n == 0:
        return Python.tuple(PythonObject(0))
    var first = _nested_into(env, seq[0], array_type, out)
    for i in range(1, n):  # cpu-route: ragged user Python list read once by user_objects_c (nested)
        var shape = _nested_into(env, seq[i], array_type, out)
        if not env.eq(shape, first):
            raise _refusal("ValueError", "mojolearn: nested sequence is not rectangular (" + env.text(shape)
                           + " vs " + env.text(first) + ")")
    return py_call(env.op.add, Python.tuple(PythonObject(n)), first)


def _key_bytes(env: _Env, raw: PythonObject, mut keys: List[UInt8]) raises -> Int:
    # the bytes of a bytes object, appended to `keys` (one memcpy through array('B'))
    var arr = py_call(env.array, "B", raw)
    var n = len(arr)
    var at = len(keys)
    for _ in range(n):  # driver-loop: over one key's bytes (room for the memcpy)
        keys.append(UInt8(0))
    if n > 0:
        memcpy(dest=keys.unsafe_ptr() + at,
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=_address(arr)), count=n)
    _ = arr^
    return n


def _group_runs(env: _Env, ids: PythonObject) raises -> PythonObject:
    # ensemble._group_id_key per id (bytes as is, str as UTF-8, an integer as
    # its decimal spelling, anything else refused), then the runs natively
    var n = len(ids)
    var keys = List[UInt8]()
    var starts = List[Int](capacity=n)
    var lengths = List[Int](capacity=n)
    for i in range(n):  # cpu-route: group_id objects (int/str/bytes) read once by user_objects_c (group_runs)
        var v = ids[i]
        var raw: PythonObject
        if env.isinstance(v, env.b.bytes):
            raw = v
        elif env.isinstance(v, env.b.str):
            raw = py_call(v.encode, "utf-8")
        elif env.isinstance(v, env.integral) and not env.isinstance(v, env.b.bool):
            raw = py_call(py_call(env.b.str, py_call(env.b.int, v)).encode, "ascii")
        else:
            raise _refusal("ValueError", "mojolearn: group_id[" + String(i) + "] object (" + env.py_repr(v)
                           + ") is unsuitable (should be string or integral type)")
        starts.append(len(keys))
        lengths.append(_key_bytes(env, raw, keys))
    var sizes = List[Int]()
    var bad = group_run_sizes(keys, starts, lengths, sizes)
    if bad >= 0:
        raise _refusal("ValueError", "mojolearn: group Ids are not consecutive: the rows of group "
                       + env.py_repr(ids[bad]) + " are split into more than one run (row " + String(bad)
                       + "); CatBoost requires every group's rows to be contiguous")
    var words = env.store("q", len(sizes), 8)
    var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(words))
    for k in range(len(sizes)):  # driver-loop: over the groups
        out[k] = Int64(sizes[k])
    return words


def _pairs(env: _Env, rows: PythonObject, weights: PythonObject, n_rows: Int) raises -> PythonObject:
    # ensemble._pairs_arrays: each pair checked in their Pool's order, then
    # flattened winner, loser; one finite nonnegative weight per pair
    var n = len(rows)
    var flat = env.store("q", 2 * n, 8)
    var fw = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(flat))
    for i in range(n):  # cpu-route: user pairs (Python index pairs) read once by user_objects_c (pairs)
        var pair = rows[i]
        if len(pair) != 2:
            raise _refusal("ValueError", "mojolearn: Length of pairs[" + String(i) + "] isn't equal to 2.")
        for j in range(2):  # driver-loop: over the two rows of a pair
            var index = pair[j]
            if not env.isinstance(index, env.integral) or env.isinstance(index, env.b.bool):
                raise _refusal("ValueError", "mojolearn: Invalid pairs[" + String(i) + "][" + String(j) + "] = '"
                               + env.text(index) + "' value type=" + env.text(py_type(index))
                               + ": must be an integer.")
            var x = py_call(env.b.int, index)
            if env.lt(x, PythonObject(0)) or not env.lt(x, PythonObject(n_rows)):
                raise _refusal("ValueError", "mojolearn: pairs[" + String(i) + "][" + String(j) + "] = "
                               + env.text(x) + " is outside the " + String(n_rows) + " rows")
            fw[2 * i + j] = Int64(Int(py=x))
        if fw[2 * i] == fw[2 * i + 1]:
            raise _refusal("ValueError", "mojolearn: pairs[" + String(i) + "] pairs row " + String(fw[2 * i])
                           + " with itself")
    var wout = env.store("d", n, 8)
    var ww = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=_address(wout))
    if weights is PythonObject(None):
        for i in range(n):  # driver-loop: over the pairs (unit weights)
            ww[i] = 1.0
        return Python.tuple(flat, wout)
    if len(weights) != n:
        raise _refusal("ValueError", "mojolearn: len(pairs_weight) = " + String(len(weights))
                       + " is not equal to len(pairs) = " + String(n) + " ")
    for i in range(n):  # cpu-route: user pairs_weight objects read once by user_objects_c (pairs)
        var w = weights[i]
        var ok = env.isinstance(w, env.real)
        var x = 0.0
        if ok:
            x = Float64(py=py_call(env.b.float, w))
            ok = x - x == 0.0 and x >= 0.0
        if not ok:
            raise _refusal("ValueError", "mojolearn: pairs_weight[" + String(i)
                           + "] must be a finite nonnegative number")
        ww[i] = x
    return Python.tuple(flat, wout)


def _sorted_codes(env: _Env, values: PythonObject, unknown: PythonObject) raises -> PythonObject:
    # UMAP's categorical target: sorted({v for v in vals if v != unknown},
    # key=lambda v: (str(type(v)), v)) as float codes, `unknown` kept as -1
    var n = len(values)
    var seen = Python.dict()
    var keys = Python.list()
    var none = PythonObject(None)
    for i in range(n):  # cpu-route: categorical y label objects read once by user_objects_c (sorted_codes)
        var v = values[i]
        if not env.eq(v, unknown):
            var before = len(seen)
            _ = py_call(seen.setdefault, v, none)
            if len(seen) != before:
                _ = py_call(keys.append, Python.tuple(py_call(env.b.str, py_type(v)), v))
    # a stable merge sort of the distinct labels by their keys' `<`
    var m = len(keys)
    var order = List[Int](capacity=m)
    for i in range(m):  # driver-loop: over the distinct classes
        order.append(i)
    var buf = List[Int](length=m, fill=0)
    var width = 1
    while width < m:  # driver-loop: over merge widths
        var lo = 0
        while lo < m:  # driver-loop: over merged runs
            var mid = min(lo + width, m)
            var hi = min(lo + 2 * width, m)
            var a = lo
            var b = mid
            var k = lo
            while a < mid and b < hi:  # driver-loop: over one merge of classes
                if env.lt(keys[order[b]], keys[order[a]]):
                    buf[k] = order[b]
                    b += 1
                else:
                    buf[k] = order[a]
                    a += 1
                k += 1
            while a < mid:  # driver-loop: over one merge's tail
                buf[k] = order[a]
                a += 1
                k += 1
            while b < hi:  # driver-loop: over one merge's tail
                buf[k] = order[b]
                b += 1
                k += 1
            lo = hi
        for t in range(m):  # driver-loop: over the distinct classes
            order[t] = buf[t]
        width *= 2
    var code = Python.dict()
    for r in range(m):  # driver-loop: over the distinct classes
        _ = py_call(code.__setitem__, keys[order[r]][1], PythonObject(r))
    var words = env.store("f", n, 4)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_address(words))
    for i in range(n):  # cpu-route: categorical y label objects read once by user_objects_c (sorted_codes)
        var v = values[i]
        if env.eq(v, unknown):
            out[i] = Float32(-1.0)
        else:
            out[i] = Float32(Int(py=code[v]))
    return words


def _group_rows(env: _Env, codes: PythonObject, k: Int) raises -> PythonObject:
    # the rows of each code 0..k-1, increasing, one int64 store per code
    var n = len(codes)
    var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_address(codes))
    var counts = List[Int](length=k, fill=0)
    for i in range(n):  # cpu-route: int32 group codes read once by user_objects_c (group_rows)
        counts[Int(cp[i])] += 1
    var stores = Python.list()
    var at = List[Int](length=k, fill=0)
    var ptrs = List[Int](capacity=k)
    for c in range(k):  # driver-loop: over the groups
        var st = env.store("q", counts[c], 8)
        ptrs.append(_address(st) if counts[c] > 0 else 0)
        _ = py_call(stores.append, st)
    for i in range(n):  # cpu-route: int32 group codes read once by user_objects_c (group_rows)
        var c = Int(cp[i])
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ptrs[c])[at[c]] = Int64(i)
        at[c] += 1
    return stores


def _group_permute(env: _Env, values: PythonObject, rows: PythonObject, perms: PythonObject) raises -> PythonObject:
    # permutation_test_score's groups route: yp = list(values), then per
    # group, yp[rows[a]] = values[rows[perm[a]]]
    var out = py_call(env.b.list, values)
    for g in range(len(rows)):  # driver-loop: over the groups
        var r = rows[g]
        var p = perms[g]
        var m = min(len(r), len(p))
        if m == 0:
            continue
        var rp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(r))
        var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_address(p))
        for a in range(m):  # cpu-route: Python label objects read once by user_objects_c (group_permute)
            _ = py_call(out.__setitem__, PythonObject(Int(rp[a])), values[Int(rp[Int(pp[a])])])
    return out


def _f32_exact(env: _Env, values: PythonObject) raises -> PythonObject:
    # the labels as Python floats when every one is a real number (not a
    # bool) that float32 holds exactly and none is NaN, else None
    var n = len(values)
    var out = Python.list()
    for i in range(n):  # cpu-route: Python label objects read once by user_objects_c (f32_exact)
        var v = values[i]
        if env.isinstance(v, env.b.bool) or not env.isinstance(v, env.real):
            return PythonObject(None)
        var x = Float64(py=py_call(env.b.float, v))
        if x != x or Float64(Float32(x)) != x:
            return PythonObject(None)
        _ = py_call(out.append, PythonObject(x))
    return out


def _indicator(env: _Env, rows: PythonObject) raises -> PythonObject:
    # _expansion_prep._multilabel_indicator of list rows: (0, float32 words,
    # width), (1, None, 0) when the rows' lengths differ, (2, None, 0) when a
    # value is not an integral real or more than two distinct values occur
    var n = len(rows)
    var w = len(rows[0])
    for i in range(n):  # cpu-route: Python list-of-lists y read once by user_objects_c (indicator)
        if len(rows[i]) != w:
            return Python.tuple(PythonObject(1), PythonObject(None), PythonObject(0))
    var words = env.store("f", n * w, 4)
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_address(words))
    var d0 = 0.0
    var d1 = 0.0
    var nd = 0
    for i in range(n):  # cpu-route: Python list-of-lists y read once by user_objects_c (indicator)
        var r = rows[i]
        for j in range(w):  # driver-loop: over one row's columns
            var v = r[j]
            if env.isinstance(v, env.b.bool) or not env.isinstance(v, env.real):
                return Python.tuple(PythonObject(2), PythonObject(None), PythonObject(0))
            var x = Float64(py=py_call(env.b.float, v))
            if not _float_is_integer(x):
                return Python.tuple(PythonObject(2), PythonObject(None), PythonObject(0))
            if nd == 0 or (x != d0 and (nd == 1 or x != d1)):
                if nd == 0:
                    d0 = x
                elif nd == 1:
                    d1 = x
                nd += 1
                if nd > 2:
                    return Python.tuple(PythonObject(2), PythonObject(None), PythonObject(0))
            out[i * w + j] = Float32(x)
    return Python.tuple(PythonObject(0), words, PythonObject(w))


def user_objects_call(call: PythonObject) raises -> PythonObject:
    var env = _Env()
    var operation = String(py=call[0])
    if operation == "take":
        return _take(env, call[1], Int(py=call[2]), Int(py=call[3]), Int(py=call[4]))
    if operation == "first_seen":
        return _first_seen(env, call[1])
    if operation == "lookup":
        return _lookup(env, call[1], call[2])
    if operation == "kinds":
        return _kinds(env, call[1])
    if operation == "ints":
        return _ints(env, call[1])
    if operation == "floats":
        return _floats(env, call[1])
    if operation == "strs":
        return _strs(env, call[1])
    if operation == "column":
        return _column(env, call[1], Int(py=call[2]))
    if operation == "column_f32":
        return _column_f32(env, call[1], Int(py=call[2]))
    if operation == "transpose":
        return _transpose(env, call[1])
    if operation == "scatter":
        return _scatter(env, call[1], Int(py=call[2]), Int(py=call[3]), call[4])
    if operation == "flatten":
        var out = Python.list()
        _flatten_into(env, call[1], out)
        return out
    if operation == "nested":
        var flat = Python.list()
        var shape = _nested_into(env, call[1], call[2], flat)
        return Python.tuple(shape, flat)
    if operation == "group_runs":
        return _group_runs(env, call[1])
    if operation == "pairs":
        return _pairs(env, call[1], call[2], Int(py=call[3]))
    if operation == "sorted_codes":
        return _sorted_codes(env, call[1], call[2])
    if operation == "f32_exact":
        return _f32_exact(env, call[1])
    if operation == "indicator":
        return _indicator(env, call[1])
    if operation == "group_rows":
        return _group_rows(env, call[1], Int(py=call[2]))
    if operation == "group_permute":
        return _group_permute(env, call[1], call[2], call[3])
    raise Error("TypeError: unknown user-object operation " + operation)


def user_objects_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        return user_objects_call(call).steal_data()
    except error:
        var message = String(error)
        if message.startswith("__refuse__"):
            # ("__refuse__", kind, message): the door raises it as that kind
            var rest = String(message[byte=10:])
            var colon = rest.find(":")
            var refusal = Python.tuple(PythonObject("__refuse__"), PythonObject(String(rest[byte=:colon])),
                                       PythonObject(String(rest[byte=colon + 1:])))
            return refusal.steal_data()
        if message.startswith("TypeError: "):
            return raise_python_exception(Error(String(message[byte=11:])), ExceptionType.TypeError)
        return raise_python_exception(error^, ExceptionType.ValueError)
