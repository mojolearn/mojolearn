# SPDX-License-Identifier: Apache-2.0
# The meta-estimators' public operations in one native call each:
# VotingClassifier/VotingRegressor, StackingClassifier/StackingRegressor,
# MultiOutputRegressor/MultiOutputClassifier, OneVsRestClassifier and
# CalibratedClassifierCV (python/mojolearn/_expansion_trees.py). This is the
# Python orchestration those methods ran, step for step: the parameter checks
# with their wording, the member clones, the cv folds, the x_trees glue
# kernels in the same order with the same arguments, and the fitted state.
#
# Members are mojolearn estimators. Each member is fitted and asked for its
# outputs through its own public methods, which are its native entries (one
# native call each); a member of any other class is a Python object whose
# methods would run Python inside this call, and is refused by name with
# NotImplementedError (the decision taken for Bagging* and AdaBoost*).
#
# Round trips removed against the Python path: X is converted to float32
# once per operation and handed to every member; the fold rows are copied
# out of the one device fold buffer as int32 Arrays (no memoryview list).
#
# CalibratedClassifierCV(GaussianNB) no longer takes the x_prep one-program
# fit (MOJOLEARN_CALIB_GNB_FOLDS, FAST + Apple only): that program was
# assembled by the prep family's Python program builder, so the native
# driver takes the reference route (one GaussianNB fit per fold through its
# native entry, the same calibrators). IDENTICAL never took it; the x_prep
# kernels stay in place for the prep family.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_wrap_array, _store_code
from bindings.trees_public_inputs import (
    TreesMatrix, TreesLabels, TreesView, trees_builtins, trees_table, trees_entry, trees_helper, trees_empty,
    trees_vector, trees_reshape, trees_address, trees_f32, trees_view, trees_check_rank, trees_itemsize,
    trees_check_sample_weight, trees_encode_labels, trees_decode_labels, trees_cast,
)
from bindings.trees_public_forest import _check_mode, _repr, _str, _truthy, _is_bool, _is_integral, _refuse
from bindings.trees_public_boost import (
    _mode, _xt, _x, _P, _F, _A, _x_f32, _as_array, _fsum, _arange, _class_major, _gather, _gather_codes,
    _gather_vec, _argmax, _scale,
)
from bindings.trees_public_score import trees_accuracy, trees_r2


comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


# ------------------------------------------------------------- utilities


def meta_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The connected meta schema of the model's class (or an ancestor)."""
    var b = trees_builtins()
    var table = context["trees_meta_classes"]
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var kind = py_get(table, mro[i])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: the native meta-estimator owner requires a connected meta-estimator class")


def _attr(owner: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """`getattr(owner, name, default)`."""
    return py_call(trees_builtins().getattr, owner, PythonObject(name), default)


def _has(owner: PythonObject, name: String) raises -> Bool:
    return Bool(py=py_call(trees_builtins().hasattr, owner, PythonObject(name)))


def _call_kw(callable: PythonObject, args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
    """`callable(*args, **kwargs)` of a Python tuple and dict, borrowing both."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, kwargs._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _owner(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def _copy_block(array: PythonObject, start: Int, stop: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    """`array[start:stop]` of a flat Array as a new Array: a block copy in C."""
    var b = trees_builtins()
    var item = trees_itemsize(typestr)
    var view = py_call(b.memoryview, array._store)
    var raw = py_call(py_call(view.cast, "B").__getitem__, py_call(b.slice, PythonObject(start * item), PythonObject(stop * item)))
    var store = py_call(context["array_type"], PythonObject(_store_code(typestr)))
    _ = py_call(store.frombytes, py_call(raw.tobytes))
    return classical_wrap_array(store, Python.tuple(stop - start), typestr, "C", context)


def _f64_word(array: PythonObject, index: Int) raises -> Float64:
    return _F64P(unsafe_from_address=trees_address(array))[index]


def _x_array(X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_f32_c(X, ndim=2, name="X")` as the Array the members read."""
    var m = _x_f32(X, context)
    return _as_array(m, Python.tuple(m.rows, m.cols), context)


def _rows(array: PythonObject) raises -> Int:
    return Int(py=array.shape[0])


def _cols(array: PythonObject) raises -> Int:
    return Int(py=array.shape[1])


def _i32_vec(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`as_i32_c(value, ndim=1, name=name)` of a member's predicted codes."""
    var view = trees_view(value, name, context)
    trees_check_rank(view, 1, name)
    var n = view.shape[0]
    var out = trees_vector(n, "<i4", context)
    trees_cast(view.addr, view.typestr, trees_address(out), "<i4", n, context)
    _ = view^
    return out


def _f32_array(value: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    """`as_f32_c(value, ndim=ndim, name=name)` as an Array."""
    var m = trees_f32(value, ndim, "C", name, context)
    var shape = Python.tuple(m.rows, m.cols) if ndim == 2 else Python.tuple(m.rows)
    return _as_array(m, shape, context)


def _out2d(x: PythonObject, n: Int, context: PythonObject) raises -> TreesMatrix:
    """`_trees_output_2d(x, n)`: a member's output as a float32 (n, c) block
    (a 1-D output is c = 1)."""
    var shape = _attr(x, "shape", Python.tuple())
    if len(shape) == 2:
        return trees_f32(x, 2, "C", "output", context)
    var v = trees_f32(x, 1, "C", "output", context)
    if v.rows != n:
        raise Error("cannot reshape Array of size " + String(v.rows) + " into shape (" + String(n) + ", 1)")
    return TreesMatrix(v.keep, v.addr, n, 1, True)


def _int_list(values: PythonObject) raises -> PythonObject:
    return py_call(trees_builtins().list, py_call(trees_builtins().map, trees_builtins().int, values))


def _codes_array(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list([int(c) for c in values], "<i4")`."""
    var store = py_call(context["array_type"], PythonObject("i"), _int_list(values))
    return classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)


def _sub_cols(member: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_sub_cols`: a member's classes (the ensemble's codes) as int32."""
    return _codes_array(_attr(member, "classes_", PythonObject(None)), context)


def _index_of_one(member: PythonObject) raises -> Int:
    """`[int(c) for c in member.classes_].index(1)`, or -1 when 1 is absent."""
    var subs = _int_list(_attr(member, "classes_", PythonObject(None)))
    for i in range(len(subs)):  # driver-loop: over the member's fitted classes
        if Int(py=subs[i]) == 1:
            return i
    return -1


def _one_col(xt: PythonObject, p: TreesMatrix, j: Int, context: PythonObject) raises -> PythonObject:
    """Column j of a float32 (n, c) block as an (n, 1) Array (`_gather`)."""
    var rows = _arange(xt, p.rows, context)
    var col = _codes_array(Python.list(PythonObject(j)), context)
    return _gather(xt, p.addr, p.rows, p.cols, rows, col, context)


# ------------------------------------------------------------- x_trees glue


def _acc_cols_w(xt: PythonObject, acc: PythonObject, x: PythonObject, cols: PythonObject, n: Int, k: Int,
                weight: Float64, context: PythonObject) raises:
    var xa = trees_f32(x, 2, "C", "sub-estimator output", context)
    _ = py_call_typed(_x(xt, "x_trees_accumulate_cols"), _A(acc), _P(xa.addr), _A(cols),
                      Python.list(_P(n), _P(k), _P(len(cols)), _F(weight)))
    _ = xa^


def _acc_w(xt: PythonObject, acc: PythonObject, x: PythonObject, n: Int, weight: Float64,
           context: PythonObject) raises:
    var xa = trees_f32(x, 1, "C", "sub-estimator output", context)
    _ = py_call_typed(_x(xt, "x_trees_accumulate"), _A(acc), _P(xa.addr), Python.list(_P(n), _F(weight)))
    _ = xa^


def _acc_votes(xt: PythonObject, acc: PythonObject, codes: PythonObject, n: Int, k: Int, on: Float64) raises:
    _ = py_call_typed(_x(xt, "x_trees_accumulate_onehot"), _A(acc), _A(codes),
                      Python.list(_P(n), _P(k), _F(on), _F(0.0)))


def _place(xt: PythonObject, dst: PythonObject, n_rows: Int, n_cols: Int, blk: TreesMatrix, rows: PythonObject,
           col0: Int) raises -> Int:
    """`_TreesWrapperBase._place`: the block's columns into dst at col0."""
    _ = py_call_typed(_x(xt, "x_trees_scatter"), _A(dst), _P(blk.addr), _A(rows),
                      Python.list(_P(n_rows), _P(n_cols), _P(len(rows)), _P(blk.cols), _P(col0)))
    return blk.cols


def _zeros(count: Int, context: PythonObject) raises -> PythonObject:
    return trees_vector(count, "<f8", context)


def _class_zero(xt: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_class_major([0.0], n)`."""
    return _class_major(xt, Python.list(PythonObject(0.0)), n, context)


# ---------------------------------------------------------------- members


def _is_mojolearn(est: PythonObject) raises -> Bool:
    var module = _attr(py_type(est), "__module__", PythonObject(""))
    if py_type(module) is not trees_builtins().str:
        return False
    var name = String(module)
    return name == "mojolearn" or name.startswith("mojolearn.")


def _require_member(est: PythonObject, owner: String) raises:
    if not _is_mojolearn(est):
        raise Error("NotImplementedError: " + owner + ": " + _repr(est) + " is not a mojolearn estimator; "
                    + "the native ensemble fits and predicts only mojolearn members. "
                    + "Refused by name rather than run in Python.")


def _clone(est: PythonObject) raises -> PythonObject:
    """`_trees_clone(est)`: `type(est)(**est.get_params(deep=False))`."""
    var b = trees_builtins()
    var params = Python.dict()
    if _has(est, "get_params"):
        params = py_call(b.dict, py_call(est.get_params, deep=PythonObject(False)))
    return _call_kw(py_type(est), Python.tuple(), params)


def _fit_member(member: PythonObject, X: PythonObject, y: PythonObject, weights: PythonObject) raises:
    if weights is None:
        _ = py_call(member.fit, X, y)
    else:
        _ = py_call(member.fit, X, y, sample_weight=weights)


def _takes_weights(est: PythonObject) raises -> Bool:
    """`"sample_weight" in inspect.signature(est.fit).parameters`: the
    named parameters of the fit function, past its `functools.wraps`
    wrappers."""
    var method = _attr(est, "fit", PythonObject(None))
    for _ in range(16):  # small-loop(wrapper chain): functools.wraps layers of one method
        if not _has(method, "__wrapped__"):
            break
        method = _attr(method, "__wrapped__", PythonObject(None))
    var code = _attr(method, "__code__", PythonObject(None))
    if code is None:
        return False
    var count = Int(py=code.co_argcount) + Int(py=code.co_kwonlyargcount)
    var names = code.co_varnames
    for i in range(count):  # small-loop(parameters): the fit function's named parameters
        if String(names[i]) == "sample_weight":
            return True
    return False


def _member_weights(names: PythonObject, ests: PythonObject, sample_weight: PythonObject, n: Int,
                    context: PythonObject) raises -> PythonObject:
    """`_trees_member_weights`: None stays None; otherwise every member must
    take `sample_weight`, and the weights are checked once against X."""
    if sample_weight is None:
        return PythonObject(None)
    for i in range(len(ests)):  # driver-loop: over members
        if not _takes_weights(ests[i]):
            var who = String("") if names[i] is None else " " + _str(names[i])
            raise Error("Underlying estimator" + who + " does not support sample weights.")
    var w = trees_check_sample_weight(sample_weight, n, context)
    return _as_array(w, Python.tuple(n), context)


struct Active(Copyable, Movable):
    """`_trees_estimators(estimators)`: the active names and estimators."""
    var names: PythonObject
    var ests: PythonObject

    def __init__(out self, names: PythonObject, ests: PythonObject):
        self.names = names
        self.ests = ests


def _active(estimators: PythonObject, owner: String) raises -> Active:
    var b = trees_builtins()
    if not _truthy(estimators):
        raise Error("estimators must be a non-empty list of (name, estimator)")
    var items = py_call(b.list, estimators)
    var all_names = Python.list()
    var names = Python.list()
    var ests = Python.list()
    for i in range(len(items)):  # driver-loop: over the (name, estimator) entries
        var pair = py_call(b.tuple, items[i])
        if len(pair) != 2:
            raise Error("estimators must be a non-empty list of (name, estimator)")
        _ = py_call(all_names.append, pair[0])
    if len(py_call(b.set, all_names)) != len(all_names):
        raise Error("estimator names must be unique")
    for i in range(len(items)):  # driver-loop: over the (name, estimator) entries
        var pair = py_call(b.tuple, items[i])
        var est = pair[1]
        if py_type(est) is b.str and String(est) == "drop":
            continue
        _ = py_call(names.append, pair[0])
        _ = py_call(ests.append, est)
    if len(ests) == 0:
        raise Error("all estimators are 'drop'")
    for i in range(len(ests)):  # driver-loop: over members
        _require_member(ests[i], owner)
    return Active(names, ests)


def _named(names: PythonObject, fitted: PythonObject) raises -> PythonObject:
    var out = Python.dict()
    for i in range(len(fitted)):  # driver-loop: over members
        py_setitem(out, names[i], fitted[i])
    return out


def _set(model: PythonObject, name: String, value: PythonObject) raises:
    py_setattr(model, name, value)


def _check_x(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_TreesWrapperBase._check_X`."""
    if not _truthy(_attr(model, "_fitted", PythonObject(False))):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var Xa = _x_array(X, context)
    var fitted = Int(py=_attr(model, "n_features_in_", PythonObject(0)))
    if _cols(Xa) != fitted:
        raise Error("X has " + String(_cols(Xa)) + " features, fit saw " + String(fitted))
    return Xa


def _refuse_jobs(model: PythonObject, verbose: Bool, why: String) raises:
    if verbose:
        if _attr(model, "n_jobs", PythonObject(None)) is not None or _truthy(_attr(model, "verbose", PythonObject(0))):
            _refuse("n_jobs/verbose", why)
    elif _attr(model, "n_jobs", PythonObject(None)) is not None:
        _refuse("n_jobs", why)


def _import(module: String, name: String) raises -> PythonObject:
    """A mojolearn public class, from its loaded module."""
    return py_call(trees_builtins().getattr, Python.import_module(module), PythonObject(name))


# ------------------------------------------------------------------ cv folds


def _cv_value(cv: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_cv(cv)`: an int number of folds (None: 5), or `cv` itself when
    it is a splitter or an iterable of (train, test) index pairs."""
    var b = trees_builtins()
    if cv is None:
        return PythonObject(5)
    var number = py_call(b.getattr, Python.import_module("numbers"), PythonObject("Number"))
    var plain = Bool(py=py_call(b.isinstance, cv, Python.tuple(b.str, b.bytes, number)))
    if _has(cv, "split") or (not plain and _has(cv, "__iter__")):
        return cv
    if _is_bool(cv) or not _is_integral(cv, context) or Bool(py=cv < PythonObject(2)):
        raise Error("cv=" + _repr(cv) + ": an int >= 2, a splitter with split(X, y), or an iterable of"
                    + " (train, test) index pairs")
    return py_call(b.int, cv)


def _index_i64(v: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_index_i64`: a cv index list as an int64 Array."""
    try:
        var view = trees_view(v, "cv rows", context)
        var n = view.size()
        var out = trees_vector(n, "<i8", context)
        trees_cast(view.addr, view.typestr, trees_address(out), "<i8", n, context)
        _ = view^
        return out
    except:
        var store = py_call(context["array_type"], PythonObject("q"), v)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)


def _native_folds(xt: PythonObject, n_splits: Int, n: Int, codes: PythonObject, n_classes: Int,
                  context: PythonObject) raises -> PythonObject:
    """`_trees_native_folds`: the fold rows the device made, each fold's
    (train rows, test rows) as int32 Arrays."""
    var counts = trees_vector(n_splits + 1, "<i4", context)
    var rows = trees_vector(n_splits * n, "<i4", context)
    var c32 = counts
    var k = 0
    if codes is not None:
        c32 = codes
        k = n_classes
    var status = Int(py=py_call_typed(_x(xt, "x_trees_device_folds"), _A(c32), _A(rows), _A(counts),
                                      Python.list(_P(n), _P(n_splits), _P(k))))
    if status == 1:
        raise Error("n_splits=" + String(n_splits) + " cannot be greater than the number of members in each class")
    if status != 0:
        raise Error("y codes outside [0, n_classes)")
    var cnt = _I32P(unsafe_from_address=trees_address(counts))
    var used = 0
    for i in range(n_splits):  # driver-loop: over the cv folds
        if Int(cnt[i]) > 0:
            used = i
    used += 1
    var out = Python.list()
    for i in range(used):  # driver-loop: over the cv folds
        var c = Int(cnt[i])
        var base = i * n
        var tr = _copy_block(rows, base, base + n - c, "<i4", context)
        var te = _copy_block(rows, base + n - c, base + n, "<i4", context)
        _ = py_call(out.append, Python.tuple(tr, te))
    return out


def _splits(xt: PythonObject, cv: PythonObject, X: PythonObject, y: PythonObject, n: Int, codes: PythonObject,
            partition: Bool, n_classes: Int, context: PythonObject) raises -> PythonObject:
    """`_trees_splits`: [(train rows, test rows)] as int32 Arrays."""
    var b = trees_builtins()
    var c = _cv_value(cv, context)
    var shape = _attr(X, "shape", PythonObject(None))
    var n_x = n
    if shape is not None and len(shape) > 0:
        n_x = Int(py=py_call(b.int, shape[0]))
    var n_y = len(codes) if codes is not None else n
    if n != n_x or n != n_y or n_x != n_y:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(n_x) + ", "
                    + String(n_y) + "]")
    if py_type(c) is b.int:
        return _native_folds(xt, Int(py=c), n, codes, n_classes, context)
    var pairs = c
    if _has(c, "split"):
        pairs = py_call(c.split, X, y)
    var out = Python.list()
    var seen = trees_vector(n if n > 1 else 1, "<i8", context)
    var check = trees_helper(context, "check_indices_i64")
    var bincount = trees_helper(context, "bincount_i64")
    for pair in pairs:  # cpu-route: user cv splitter or iterable of index pairs, validated natively per pair
        var both = py_call(b.tuple, pair)
        if len(both) != 2:
            raise Error("each cv split must be a (train, test) pair")
        var tr = _index_i64(both[0], context)
        var te = _index_i64(both[1], context)
        var ntr = Int(py=tr.size)
        var nte = Int(py=te.size)
        if ntr == 0 or nte == 0:
            raise Error("every cv split needs train and test rows")
        if Int(py=py_call_typed(check, _A(tr), _P(ntr), _P(n))) == 1:
            raise Error("a cv index is outside [0, " + String(n) + ")")
        if Int(py=py_call_typed(check, _A(te), _P(nte), _P(n))) == 1:
            raise Error("a cv index is outside [0, " + String(n) + ")")
        _ = py_call_typed(bincount, _A(te), _P(3), _P(nte), _P(n), _A(seen), _P(1))
        var tr32 = trees_vector(ntr, "<i4", context)
        var te32 = trees_vector(nte, "<i4", context)
        trees_cast(trees_address(tr), "<i8", trees_address(tr32), "<i4", ntr, context)
        trees_cast(trees_address(te), "<i8", trees_address(te32), "<i4", nte, context)
        _ = py_call(out.append, Python.tuple(tr32, te32))
    if len(out) == 0:
        raise Error("cv produced no splits")
    if partition:
        var m = n if n > 1 else 1
        var stat = trees_helper(context, "reduce_stat")
        var low = Int(py=py_call_typed(stat, _A(seen), _P(3), _P(m), _P(0)))
        var high = Int(py=py_call_typed(stat, _A(seen), _P(3), _P(m), _P(1)))
        if low != 1 or high != 1:
            raise Error("cross_val_predict only works for partitions: every row must be in exactly one test set")
    return out
