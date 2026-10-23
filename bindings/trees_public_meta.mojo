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
from bindings.py_attr import py_attr
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
    var raw = py_call(py_attr(py_call(view.cast, "B"), "__getitem__"), py_call(b.slice, PythonObject(start * item), PythonObject(stop * item)))
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
    _ = out  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
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
    var _keep_ret_v = TreesMatrix(v.keep, v.addr, n, 1, True)
    _ = v  # keep the Python object alive through the native call above
    return _keep_ret_v^


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
    _ = xa  # keep the Python object alive through the native call above
    _ = xa^


def _acc_w(xt: PythonObject, acc: PythonObject, x: PythonObject, n: Int, weight: Float64,
           context: PythonObject) raises:
    var xa = trees_f32(x, 1, "C", "sub-estimator output", context)
    _ = py_call_typed(_x(xt, "x_trees_accumulate"), _A(acc), _P(xa.addr), Python.list(_P(n), _F(weight)))
    _ = xa  # keep the Python object alive through the native call above
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
    if name.startswith("mojolearn.tests"):
        return False  # a test's own stand-in estimator is a user class
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
        _ = out  # keep the Python object alive through the native call above
        _ = view  # keep the Python object alive through the native call above
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
    _ = counts  # keep the Python object alive through the native call above
    var used = 0
    for i in range(n_splits):  # driver-loop: over the cv folds
        if Int(cnt[i]) > 0:
            used = i
        _ = counts  # keep the Python object alive through the native calls above
    used += 1
    var out = Python.list()
    for i in range(used):  # driver-loop: over the cv folds
        var c = Int(cnt[i])
        _ = counts  # keep the Python object alive through the native call above
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
        _ = tr32  # keep the Python object alive through the native call above
        _ = tr  # keep the Python object alive through the native call above
        trees_cast(trees_address(te), "<i8", trees_address(te32), "<i4", nte, context)
        _ = te32  # keep the Python object alive through the native call above
        _ = te  # keep the Python object alive through the native call above
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


# ------------------------------------------------------------------ Voting


def _voting_weights(model: PythonObject) raises -> PythonObject:
    """`VotingClassifier._weights`: one float per fitted member."""
    var b = trees_builtins()
    var m = len(_attr(model, "estimators_", Python.list()))
    var given = _attr(model, "weights", PythonObject(None))
    var w = Python.list()
    if given is None:
        for _ in range(m):  # driver-loop: over members
            _ = py_call(w.append, PythonObject(1.0))
    else:
        var items = py_call(b.list, given)
        for i in range(len(items)):  # driver-loop: over the user weights (one per member)
            _ = py_call(w.append, py_call_typed(b.float, items[i]))
    if len(w) != m:
        raise Error("weights has " + String(len(w)) + " entries, there are " + String(m) + " estimators")
    return w


def _voting_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                classifier: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var owner = _owner(model)
    if classifier:
        var voting = _attr(model, "voting", PythonObject(None))
        if not py_contains(Python.tuple("hard", "soft"), voting):
            raise Error("voting must be 'hard' or 'soft'")
    _refuse_jobs(model, True, "the members fit one after another.")
    var Xa = _x_array(X, context)
    var target = y
    if classifier:
        var labels = trees_encode_labels(y, context)
        _set(model, "classes_", labels.classes)
        _set(model, "n_classes_", PythonObject(len(labels.classes)))
        target = labels.codes
    var active = _active(_attr(model, "estimators", PythonObject(None)), owner)
    var weights = _member_weights(active.names, active.ests, sample_weight, _rows(Xa), context)
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    for i in range(len(active.ests)):  # driver-loop: over members
        var e = _clone(active.ests[i])
        _fit_member(e, Xa, target, weights)
        _ = py_call(fitted.append, e)
    _set(model, "named_estimators_", _named(active.names, fitted))
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
    _set(model, "_fitted", PythonObject(True))
    if classifier:
        _ = _voting_weights(model)
    _ = fields
    return model


def _voting_soft(model: PythonObject, xt: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """`VotingClassifier._soft`: the weighted average of the members'
    probabilities, class-major (n * k)."""
    var n = _rows(Xa)
    var k = Int(py=_attr(model, "n_classes_", PythonObject(0)))
    var acc = _class_zero(xt, n * k, context)
    var w = _voting_weights(model)
    var members = _attr(model, "estimators_", Python.list())
    for i in range(len(members)):  # driver-loop: over members
        var e = members[i]
        _acc_cols_w(xt, acc, py_call(e.predict_proba, Xa), _sub_cols(e, context), n, k, Float64(py=w[i]), context)
    _scale(xt, acc, _fsum(w, context))
    return acc


def _voting_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    if String(_attr(model, "voting", PythonObject(""))) == "hard":
        raise Error("AttributeError: predict_proba is not available when voting='hard'")
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var k = Int(py=_attr(model, "n_classes_", PythonObject(0)))
    return trees_reshape(_voting_soft(model, xt, Xa, context), Python.tuple(_rows(Xa), k), context)


def _voting_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var k = Int(py=_attr(model, "n_classes_", PythonObject(0)))
    var acc: PythonObject
    if String(_attr(model, "voting", PythonObject(""))) == "soft":
        acc = _voting_soft(model, xt, Xa, context)
    else:
        acc = _zeros(n * k, context)
        var w = _voting_weights(model)
        var members = _attr(model, "estimators_", Python.list())
        for i in range(len(members)):  # driver-loop: over members
            _acc_votes(xt, acc, _i32_vec(py_call(members[i].predict, Xa), "codes", context), n, k, Float64(py=w[i]))
    return trees_decode_labels(_attr(model, "classes_", PythonObject(None)), _argmax(xt, acc, n, k, context), context)


def _voting_transform(model: PythonObject, X: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var members = _attr(model, "estimators_", Python.list())
    var m = len(members)
    var soft = classifier and String(_attr(model, "voting", PythonObject(""))) == "soft"
    var k = Int(py=_attr(model, "n_classes_", PythonObject(0))) if classifier else 1
    var width = m * k if soft else m
    var out = _zeros(n * width, context)
    var rows = _arange(xt, n, context)
    for j in range(m):  # driver-loop: over members
        var e = members[j]
        if soft:
            var p = _zeros(n * k, context)
            _acc_cols_w(xt, p, py_call(e.predict_proba, Xa), _sub_cols(e, context), n, k, 1.0, context)
            var blk = trees_f32(trees_reshape(p, Python.tuple(n, k), context), 2, "C", "p", context)
            _ = _place(xt, out, n, width, blk, rows, j * k)
            _ = blk^
        else:
            var blk = _out2d(py_call(e.predict, Xa), n, context)
            _ = _place(xt, out, n, width, blk, rows, j)
            _ = blk^
    return trees_reshape(out, Python.tuple(n, width), context)


def _voting_regress(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`VotingRegressor.predict`: the weighted average of the members'
    predictions."""
    var b = trees_builtins()
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var members = _attr(model, "estimators_", Python.list())
    var m = len(members)
    var given = _attr(model, "weights", PythonObject(None))
    var w = Python.list()
    if given is None:
        for _ in range(m):  # driver-loop: over members
            _ = py_call(w.append, PythonObject(1.0))
    else:
        var items = py_call(b.list, given)
        for i in range(len(items)):  # driver-loop: over the user weights (one per member)
            _ = py_call(w.append, py_call_typed(b.float, items[i]))
    if len(w) != m:
        raise Error("weights has " + String(len(w)) + " entries, there are " + String(m) + " estimators")
    var acc = _zeros(n, context)
    for i in range(m):  # driver-loop: over members
        _acc_w(xt, acc, py_call(members[i].predict, Xa), n, Float64(py=w[i]), context)
    _scale(xt, acc, _fsum(w, context))
    return acc


# ---------------------------------------------------------------- Stacking


def _stacking_validate(model: PythonObject, context: PythonObject) raises:
    _refuse_jobs(model, True, "the members fit one after another.")
    var method = _attr(model, "stack_method", PythonObject("auto"))
    if not py_contains(Python.tuple("auto", "predict_proba", "decision_function", "predict"), method):
        raise Error("stack_method must be auto, predict_proba, decision_function or predict")
    _ = _cv_value(_attr(model, "cv", PythonObject(None)), context)


def _stack_method(model: PythonObject, est: PythonObject, classifier: Bool) raises -> PythonObject:
    """`_StackingBase._method`."""
    var method = _attr(model, "stack_method", PythonObject("auto"))
    if String(method) != "auto":
        return method
    if classifier:
        if _has(est, "predict_proba"):
            return PythonObject("predict_proba")
        if _has(est, "decision_function"):
            return PythonObject("decision_function")
    return PythonObject("predict")


def _stack_out(model: PythonObject, xt: PythonObject, est: PythonObject, method: PythonObject, Xs: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`_StackingBase._out`: a binary predict_proba keeps its second column."""
    var out = py_call(py_call(trees_builtins().getattr, est, method), Xs)
    if String(method) == "predict_proba" and _truthy(_attr(model, "_binary", PythonObject(False))):
        var p = _out2d(out, _rows(Xs), context)
        var col = _one_col(xt, p, 1, context)
        _ = p^
        return col
    return out


def _stacking_fit(model: PythonObject, X: PythonObject, y_full: PythonObject, codes: Bool, splits: PythonObject,
                  final: PythonObject, sample_weight: PythonObject, classifier: Bool,
                  context: PythonObject) raises -> PythonObject:
    """`_StackingBase._fit_stack`. `y_full` is the int32 codes (classifier)
    or the float32 targets (regressor); a fold's targets are gathered."""
    var owner = _owner(model)
    var xt = _xt(py_fields(model), context)
    var Xa = _x_array(X, context)
    var n = _rows(Xa)
    var d = _cols(Xa)
    var active = _active(_attr(model, "estimators", PythonObject(None)), owner)
    _require_member(final, owner)
    var all_names = py_call(trees_builtins().list, active.names)
    var all_ests = py_call(trees_builtins().list, active.ests)
    _ = py_call(all_names.append, PythonObject("final_estimator"))
    _ = py_call(all_ests.append, final)
    var sw = _member_weights(all_names, all_ests, sample_weight, n, context)
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    for i in range(len(active.ests)):  # driver-loop: over members
        var e = _clone(active.ests[i])
        _fit_member(e, Xa, y_full, sw)
        _ = py_call(fitted.append, e)
    var methods = Python.list()
    for i in range(len(fitted)):  # driver-loop: over members
        _ = py_call(methods.append, _stack_method(model, fitted[i], classifier))
    _set(model, "stack_method_", methods)
    var head = _gather(xt, trees_address(Xa), n, d, _arange(xt, n if n < 2 else 2, context), _arange(xt, d, context), context)
    _ = Xa  # keep the Python object alive through the native call above
    var widths = Python.list()
    var total = 0
    for i in range(len(fitted)):  # driver-loop: over members
        var blk = _out2d(_stack_out(model, xt, fitted[i], methods[i], head, context), 2, context)
        _ = Xa  # keep the Python object alive through the native call above
        _ = py_call(widths.append, PythonObject(blk.cols))
        total += blk.cols
        _ = blk^
    var width = total + (d if _truthy(_attr(model, "passthrough", PythonObject(False))) else 0)
    var meta = _zeros(n * width, context)
    var cols = _arange(xt, d, context)
    for f in range(len(splits)):  # driver-loop: over the cv folds
        var tr = splits[f][0]
        var te = splits[f][1]
        var col0 = 0
        for i in range(len(active.ests)):  # driver-loop: over members
            var e = _clone(active.ests[i])
            var y_tr: PythonObject
            if codes:
                y_tr = _gather_codes(xt, y_full, tr, context)
            else:
                y_tr = _gather_vec(xt, trees_address(y_full), len(y_full), tr, context)
            var w_tr = PythonObject(None)
            if sw is not None:
                w_tr = _gather_vec(xt, trees_address(sw), n, tr, context)
                _ = sw  # keep the Python object alive through the native call above
            _fit_member(e, _gather(xt, trees_address(Xa), n, d, tr, cols, context), y_tr, w_tr)
            _ = Xa  # keep the Python object alive through the native call above
            var Xte = _gather(xt, trees_address(Xa), n, d, te, cols, context)
            _ = Xa  # keep the Python object alive through the native call above
            var blk = _out2d(_stack_out(model, xt, e, methods[i], Xte, context), len(te), context)
            _ = Xa  # keep the Python object alive through the native call above
            var got = _place(xt, meta, n, width, blk, te, col0)
            _ = blk^
            if got != Int(py=widths[i]):
                raise Error("a member's output width changed between folds (a class missing from a fold)")
            col0 += got
    if _truthy(_attr(model, "passthrough", PythonObject(False))):
        var xb = TreesMatrix(Xa, trees_address(Xa), n, d, True)
        _ = Xa  # keep the Python object alive through the native call above
        _ = _place(xt, meta, n, width, xb, _arange(xt, n, context), total)
        _ = Xa  # keep the Python object alive through the native call above
        _ = xb^
        _ = Xa  # keep the Python object alive through the native call above
    _set(model, "_widths_", widths)
    _set(model, "named_estimators_", _named(active.names, fitted))
    var final_ = _clone(final)
    _set(model, "final_estimator_", final_)
    var meta32 = _f32_array(trees_reshape(meta, Python.tuple(n, width), context), 2, "meta", context)
    _fit_member(final_, meta32, y_full, sw)
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _stacking_transform(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_StackingBase.transform`: the meta features (float64)."""
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var d = _cols(Xa)
    var widths = _attr(model, "_widths_", Python.list())
    var total = 0
    for i in range(len(widths)):  # driver-loop: over members
        total += Int(py=widths[i])
    var passthrough = _truthy(_attr(model, "passthrough", PythonObject(False)))
    var width = total + (d if passthrough else 0)
    var meta = _zeros(n * width, context)
    var rows = _arange(xt, n, context)
    var col0 = 0
    var members = _attr(model, "estimators_", Python.list())
    var methods = _attr(model, "stack_method_", Python.list())
    for i in range(len(members)):  # driver-loop: over members
        var blk = _out2d(_stack_out(model, xt, members[i], methods[i], Xa, context), n, context)
        col0 += _place(xt, meta, n, width, blk, rows, col0)
        _ = blk^
    if passthrough:
        var xb = TreesMatrix(Xa, trees_address(Xa), n, d, True)
        _ = Xa  # keep the Python object alive through the native call above
        _ = _place(xt, meta, n, width, xb, rows, col0)
        _ = Xa  # keep the Python object alive through the native call above
        _ = xb^
        _ = Xa  # keep the Python object alive through the native call above
    return trees_reshape(meta, Python.tuple(n, width), context)


def _stacking_classifier_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    _stacking_validate(model, context)
    var labels = trees_encode_labels(y, context)
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    _set(model, "_binary", PythonObject(k == 2))
    var xt = _xt(py_fields(model), context)
    var splits = _splits(xt, _attr(model, "cv", PythonObject(None)), X, y, labels.n, labels.codes, True, k, context)
    var final = _attr(model, "final_estimator", PythonObject(None))
    if final is None:
        final = py_call(_import("mojolearn.linear_model", "LogisticRegression"))
    return _stacking_fit(model, X, labels.codes, True, splits, final, sample_weight, True, context)


def _stacking_regressor_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    _stacking_validate(model, context)
    var y32 = _f32_array(y, 1, "y", context)
    var xt = _xt(py_fields(model), context)
    var splits = _splits(xt, _attr(model, "cv", PythonObject(None)), X, y, _rows(y32), PythonObject(None), True, 0,
                         context)
    var final = _attr(model, "final_estimator", PythonObject(None))
    if final is None:
        final = py_call(_import("mojolearn.linear_model", "Ridge"))
    return _stacking_fit(model, X, y32, False, splits, final, sample_weight, False, context)


def _stacking_meta32(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    return _f32_array(_stacking_transform(model, X, context), 2, "meta", context)


def _stacking_predict(model: PythonObject, X: PythonObject, classifier: Bool, context: PythonObject) raises -> PythonObject:
    var final_ = _attr(model, "final_estimator_", PythonObject(None))
    var Xm = _stacking_meta32(model, X, context)
    if not classifier:
        return py_call(final_.predict, Xm)
    var codes = _i32_vec(py_call(final_.predict, Xm), "codes", context)
    return trees_decode_labels(_attr(model, "classes_", PythonObject(None)), codes, context)


def _stacking_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var final_ = _attr(model, "final_estimator_", PythonObject(None))
    var k = len(_attr(model, "classes_", Python.list()))
    var Xm = _stacking_meta32(model, X, context)
    var n = _rows(Xm)
    var xt = _xt(py_fields(model), context)
    var acc = _class_zero(xt, n * k, context)
    _acc_cols_w(xt, acc, py_call(final_.predict_proba, Xm), _sub_cols(final_, context), n, k, 1.0, context)
    return trees_reshape(acc, Python.tuple(n, k), context)


# ------------------------------------------------------------- MultiOutput


def _multioutput_y(Y: PythonObject) raises:
    """`_trees_multioutput_y`: scikit-learn's refusal of a 1-D target."""
    var b = trees_builtins()
    var nd = _attr(Y, "ndim", PythonObject(None))
    if nd is None and Bool(py=py_call(b.isinstance, Y, Python.tuple(b.list, b.tuple))):
        var nested = len(Y) > 0 and Bool(py=py_call(b.isinstance, Y[0], Python.tuple(b.list, b.tuple)))
        nd = PythonObject(2 if nested else 1)
    if nd is not None and Int(py=py_call(b.int, nd)) < 2:
        raise Error("y must have at least two dimensions for multi-output regression but has only one.")


def _ridge_multi_entry(model: PythonObject, est: PythonObject, sample_weight: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`_multiout_ridge_entry`: the estimators binding table when it has the
    one-program ridge (FAST + Apple builds) and `est` is a plain Ridge, else
    None."""
    var ridge = _import("mojolearn.linear_model", "Ridge")
    var plain = py_type(est) is ridge
    if plain:
        # Ridge checks its parameters at fit; this route skips Ridge.fit.
        _ = py_call(est._validate_params)
    if sample_weight is not None or not plain or _truthy(_attr(est, "normalize", PythonObject(False))):
        return PythonObject(None)
    try:
        var table = trees_table(context, _mode(py_fields(est), context), "_mojolearn_estimators")
        if trees_entry(table, "ridge_fit_multi") is None:
            return PythonObject(None)
        return table
    except:
        return PythonObject(None)


def _multioutput_ridge_fit(model: PythonObject, table: PythonObject, Xa: PythonObject, Ya: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    """`MultiOutputRegressor._fit_ridge_multi`: every target in one ridge
    program (glm/impl/ridge_multi.mojo)."""
    var n = _rows(Xa)
    var d = _cols(Xa)
    var m = _cols(Ya)
    var est = _attr(model, "estimator", PythonObject(None))
    var intercept = _truthy(_attr(est, "fit_intercept", PythonObject(True)))
    var xm: PythonObject
    var work_x = Xa
    if intercept:
        # `_column_means(b, Xa, None)` (`_means`: lm_col_sums, lm_means_finish)
        # then `_center` (lm_center).
        var sums = _zeros(d, context)
        _ = py_call_typed(trees_entry(table, "lm_col_sums"), _A(Xa), _A(sums), Python.list(_P(n), _P(d)))
        var m64 = _zeros(d, context)
        xm = trees_vector(d, "<f4", context)
        _ = py_call_typed(trees_entry(table, "lm_means_finish"), _A(sums), _A(m64), _A(xm),
                          Python.list(_P(d), _P(n), _P(0)), _F(0.0))
        work_x = trees_empty(Python.tuple(n, d), "<f4", context)
        _ = py_call_typed(trees_entry(table, "lm_center"), _A(Xa), _A(xm), _A(work_x), Python.list(_P(n), _P(d)))
    else:
        xm = trees_vector(d, "<f4", context)
    var coef = trees_vector(m * d, "<f4", context)
    var ymean = trees_vector(m, "<f4", context)
    var icpt32 = trees_vector(m, "<f4", context)
    var params = Python.list(_P(n), _P(d), _P(m), py_call_typed(trees_builtins().float, _attr(est, "alpha", PythonObject(1.0))),
                             _P(1 if intercept else 0))
    if intercept:
        _ = py_call(params.append, _A(xm))
        _ = py_call(params.append, _A(icpt32))
    _ = py_call_typed(trees_entry(table, "ridge_fit_multi"), _A(work_x), _A(Ya), _A(coef), _A(ymean), params)
    var fitted = Python.list()
    var ym = _F32P(unsafe_from_address=trees_address(ymean))
    _ = ymean  # keep the Python object alive through the native call above
    var ic = _F32P(unsafe_from_address=trees_address(icpt32))
    _ = icpt32  # keep the Python object alive through the native call above
    for j in range(m):  # driver-loop: over the output columns (one fitted ridge each)
        var e = _clone(est)
        _set(e, "solver_", PythonObject("eig"))
        _set(e, "coef_", _copy_block(coef, j * d, (j + 1) * d, "<f4", context))
        _set(e, "_x_mean", xm)
        _set(e, "_y_mean", PythonObject(Float64(ym[j]) if intercept else 0.0))
        _ = ymean  # keep the Python object alive through the native call above
        _set(e, "intercept_", PythonObject(Float64(ic[j]) if intercept else 0.0))
        _ = icpt32  # keep the Python object alive through the native call above
        _set(e, "n_features_in_", PythonObject(d))
        _ = py_call(fitted.append, e)
    _set(model, "estimators_", fitted)
    _set(model, "_multi_ridge", Python.tuple(coef, icpt32, m))
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _multioutput_regressor_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                               context: PythonObject) raises -> PythonObject:
    _refuse_jobs(model, False, "the columns fit one after another.")
    var xt = _xt(py_fields(model), context)
    var Xa = _x_array(X, context)
    _multioutput_y(y)
    var Ya = _f32_array(y, 2, "y", context)
    if _rows(Ya) != _rows(Xa):
        raise Error("y has " + String(_rows(Ya)) + " rows, X has " + String(_rows(Xa)))
    var est = _attr(model, "estimator", PythonObject(None))
    _require_member(est, _owner(model))
    var weights = _member_weights(Python.list(PythonObject(None)), Python.list(est), sample_weight, _rows(Xa), context)
    _set(model, "_multi_ridge", PythonObject(None))
    var table = _ridge_multi_entry(model, est, weights, context)
    if table is not None and _cols(Ya) > 0:
        return _multioutput_ridge_fit(model, table, Xa, Ya, context)
    var n = _rows(Xa)
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    var rows = _arange(xt, n, context)
    for j in range(_cols(Ya)):  # driver-loop: over the output columns (one member each)
        var e = _clone(est)
        var col = _codes_array(Python.list(PythonObject(j)), context)
        var yj = trees_reshape(_gather(xt, trees_address(Ya), n, _cols(Ya), rows, col, context), Python.tuple(n), context)
        _ = Ya  # keep the Python object alive through the native call above
        _fit_member(e, Xa, yj, weights)
        _ = Ya  # keep the Python object alive through the native call above
        _ = py_call(fitted.append, e)
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
    _set(model, "_fitted", PythonObject(True))
    return model


def _multioutput_regressor_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var members = _attr(model, "estimators_", Python.list())
    var m = len(members)
    var mr = _attr(model, "_multi_ridge", PythonObject(None))
    if mr is not None:
        var outputs = Int(py=mr[2])
        var out = trees_empty(Python.tuple(n, outputs), "<f4", context)
        var table = trees_table(context, _mode(py_fields(members[0]), context), "_mojolearn_estimators")
        _ = py_call_typed(trees_entry(table, "ridge_predict_multi"), _A(Xa), _A(mr[0]), _A(mr[1]), _A(out),
                          Python.list(_P(n), _P(_cols(Xa)), _P(outputs)))
        return out
    var out = _zeros(n * m, context)
    var rows = _arange(xt, n, context)
    for j in range(m):  # driver-loop: over members
        var blk = _out2d(py_call(members[j].predict, Xa), n, context)
        _ = _place(xt, out, n, m, blk, rows, j)
        _ = blk^
    return trees_reshape(out, Python.tuple(n, m), context)


def _has_buffer(value: PythonObject) raises -> Bool:
    """`_buffer._has_buffer`: `memoryview(value)` succeeds."""
    try:
        var view = py_call(trees_builtins().memoryview, value)
        _ = py_call(view.release)
        return True
    except:
        return False


def _label_columns(Y: PythonObject, xt: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_label_words(Y)` then its native transpose: a numeric 2-D label
    buffer as `(T, n, m)`, T the (m, n) 8-byte words (float64 for a float
    dtype, int64 for an integer one) whose rows are the label columns; None
    for anything else (bool, str, object, nested lists, uint64), which keeps
    the Python rows."""
    var b = trees_builtins()
    var dtype = _attr(Y, "dtype", PythonObject(None))
    var nd = _attr(Y, "ndim", PythonObject(None))
    if nd is None or not Bool(py=nd == PythonObject(2)):
        return PythonObject(None)
    var kind = _attr(dtype, "kind", PythonObject(None))
    if kind is None:
        if py_type(dtype) is not b.str:
            return PythonObject(None)
        kind = py_call(py_attr(py_call(dtype.lstrip, "<>=|"), "__getitem__"), py_call(b.slice, PythonObject(0), PythonObject(1)))
    var k = String(kind)
    if k != "f" and k != "i" and k != "u":
        return PythonObject(None)
    var target = String("<f8") if k == "f" else String("<i8")
    try:
        var view = trees_view(Y, "Y", context)
        if view.ndim() != 2:
            return PythonObject(None)
        if k != "f" and view.typestr != "<i1" and view.typestr != "<i4" and view.typestr != "<i8" and view.typestr != "|i1" and view.typestr != "|u1" and view.typestr != "<u1" and view.typestr != "<u2" and view.typestr != "<u4":
            return PythonObject(None)
        var n = view.shape[0]
        var m = view.shape[1]
        var T = trees_vector(m * n, target, context)
        if view.c_order():
            var W = trees_vector(n * m, target, context)
            trees_cast(view.addr, view.typestr, trees_address(W), target, n * m, context)
            _ = W  # keep the Python object alive through the native call above
            _ = view  # keep the Python object alive through the native call above
            _ = py_call_typed(_x(xt, "x_trees_transpose_f64"), _A(W), _A(T), Python.list(_P(n), _P(m)))
        elif view.f_order():
            # a column-major buffer's storage is already the (m, n) columns
            trees_cast(view.addr, view.typestr, trees_address(T), target, n * m, context)
            _ = T  # keep the Python object alive through the native call above
            _ = view  # keep the Python object alive through the native call above
        else:
            return PythonObject(None)
        _ = view^
        return Python.tuple(T, n, m)
    except:
        return PythonObject(None)


def _multioutput_classifier_fit(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    _refuse_jobs(model, False, "the columns fit one after another.")
    var Xa = _x_array(X, context)
    var n = _rows(Xa)
    var nd = _attr(y, "ndim", PythonObject(None))
    if nd is not None and Bool(py=nd == PythonObject(1)):
        raise Error("y must have at least two dimensions for multi-output classification but has only one.")
    var target = y
    var listed = Bool(py=py_call(b.isinstance, y, Python.tuple(b.list, b.tuple)))
    if not listed and not _has_buffer(y) and Bool(py=py_call(b.callable, _attr(y, "__array__", PythonObject(None)))):
        # A pandas DataFrame has no buffer and iterates its column names:
        # its own `__array__` hands back the host block.
        target = py_call(y.__array__)
    _multioutput_y(target)
    var est = _attr(model, "estimator", PythonObject(None))
    _require_member(est, _owner(model))
    var weights = _member_weights(Python.list(PythonObject(None)), Python.list(est), sample_weight, n, context)
    var fitted = Python.list()
    var classes = Python.list()
    var columns = _label_columns(target, _xt(py_fields(model), context), context)
    if columns is not None:
        var T = columns[0]
        var rows_n = Int(py=columns[1])
        var m = Int(py=columns[2])
        if rows_n != n:
            raise Error("Y has " + String(rows_n) + " rows, X has " + String(n))
        _set(model, "estimators_", fitted)
        _set(model, "classes_", classes)
        for j in range(m):  # driver-loop: over the output columns (one member each)
            var labels = trees_encode_labels(_copy_block(T, j * rows_n, (j + 1) * rows_n, String(T.dtype), context), context)
            var e = _clone(est)
            _fit_member(e, Xa, labels.codes, weights)
            _ = py_call(fitted.append, e)
            _ = py_call(classes.append, labels.classes)
    else:
        var rows = Python.list()
        if _has(target, "tolist"):
            rows = py_call(target.tolist)  # cpu-route: label objects (str/mixed) no Array holds, one output column at a time
        else:
            rows = py_call(b.list, py_call(b.map, b.list, target))  # cpu-route: label objects (str/mixed) no Array holds
        if len(rows) != n:
            raise Error("y has " + String(len(rows)) + " rows, X has " + String(n))
        if len(rows) > 0 and not _has(rows[0], "__len__"):
            raise Error("y must have at least two dimensions for multi-output classification but has only one.")
        var m = len(rows[0])
        _set(model, "estimators_", fitted)
        _set(model, "classes_", classes)
        var getter = Python.import_module("operator").itemgetter
        for j in range(m):  # driver-loop: over the output columns (one member each)
            var column = py_call(b.list, py_call(b.map, py_call(getter, PythonObject(j)), rows))  # cpu-route: label objects (str/mixed) no Array holds, one output column at a time
            var labels = trees_encode_labels(column, context)
            var e = _clone(est)
            _fit_member(e, Xa, labels.codes, weights)
            _ = py_call(fitted.append, e)
            _ = py_call(classes.append, labels.classes)
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
    _set(model, "_fitted", PythonObject(True))
    return model


def _multioutput_classifier_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = trees_builtins()
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var members = _attr(model, "estimators_", Python.list())
    var classes = _attr(model, "classes_", Python.list())
    var cols = Python.list()
    var stacked = True
    var all_int = True
    for j in range(len(members)):  # driver-loop: over members
        var col = trees_decode_labels(classes[j], _i32_vec(py_call(members[j].predict, Xa), "codes", context), context)
        _ = py_call(cols.append, col)
        if py_type(col) is not context["array_class"]:
            stacked = False
        elif String(col.dtype) != "<i8":
            all_int = False
            if String(col.dtype) != "<f8":
                stacked = False
    if stacked:
        var kind = String("<i8") if all_int else String("<f8")
        var addrs = Python.list()
        var keep = Python.list()
        for j in range(len(cols)):  # driver-loop: over members
            var col = cols[j]
            if String(col.dtype) != kind:
                var wide = trees_vector(n, kind, context)
                trees_cast(trees_address(col), String(col.dtype), trees_address(wide), kind, n, context)
                _ = wide  # keep the Python object alive through the native call above
                _ = col  # keep the Python object alive through the native call above
                col = wide
            _ = py_call(keep.append, col)
            _ = py_call(addrs.append, _A(col))
        var out = trees_vector(n * len(cols), kind, context)
        _ = py_call_typed(_x(xt, "x_trees_stack_w64"), addrs, _A(out), Python.list(_P(n)))
        _ = keep
        return trees_reshape(out, Python.tuple(n, len(cols)), context)
    # label objects (str, mixed) no Array holds: row tuples of them
    var lists = Python.list()
    for j in range(len(cols)):  # driver-loop: over members
        var col = cols[j]
        _ = py_call(lists.append, py_call(col.tolist) if _has(col, "tolist") else col)
    var ints = True
    for j in range(len(lists)):  # driver-loop: over members (each column's first label)
        if len(lists[j]) > 0 and not Bool(py=py_call(b.isinstance, lists[j][0], b.int)):
            ints = False
    var zipped = py_call(b.list, py_call(b.map, b.list, _call_kw(b.zip, py_call(b.tuple, lists), Python.dict())))  # cpu-route: label objects (str/mixed) no Array holds
    return py_call(context["array_class"].from_list, zipped, PythonObject("<i8" if ints else "<f8"))


def _multioutput_classifier_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var members = _attr(model, "estimators_", Python.list())
    var classes = _attr(model, "classes_", Python.list())
    var out = Python.list()
    for j in range(len(members)):  # driver-loop: over members
        var k = len(classes[j])
        var acc = _zeros(n * k, context)
        _acc_cols_w(xt, acc, py_call(members[j].predict_proba, Xa), _sub_cols(members[j], context), n, k, 1.0, context)
        _ = py_call(out.append, trees_reshape(acc, Python.tuple(n, k), context))
    return out


def _multioutput_classifier_score(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                                  context: PythonObject) raises -> PythonObject:
    """`MultiOutputClassifier.score`: the (weighted) fraction of rows whose
    every output matches."""
    var b = trees_builtins()
    var pred = _multioutput_classifier_predict(model, X, context)
    var truth = trees_view(y, "y", context)
    var shown = py_call(b.tuple, pred.shape)
    var truth_shape = Python.list()
    for k in range(truth.ndim()):  # driver-loop: over array rank metadata
        _ = py_call(truth_shape.append, PythonObject(truth.shape[k]))
    if not Bool(py=py_call(b.tuple, truth_shape) == shown):
        raise Error("y has shape " + _repr(py_call(b.tuple, truth_shape)) + ", the predictions have shape " + _repr(shown))
    var n = _rows(pred)
    var m = _cols(pred)
    # One mismatch flag per cell (the base binding's elementwise compare at
    # one exact dtype), one all-match flag per row (x_neighbors
    # `xn_row_all_zero`), then the (weighted) accuracy of those flags.
    var pdt = String(pred.dtype)
    var tdt = truth.typestr
    var t_int = tdt == "<i4" or tdt == "<i8" or tdt == "<u4" or tdt == "<u1" or tdt == "<u2" or tdt == "<i1"
    var p_int = pdt == "<i8"
    var size = n * m
    var eq = py_call(context["array_type"], PythonObject("B"), py_call(context["native_bytes_type"], size))
    var equal = trees_helper(context, "equal_elements")
    var tc = trees_vector(size, "<i8" if t_int else "<f8", context)
    trees_cast(truth.addr, tdt, trees_address(tc), "<i8" if t_int else "<f8", size, context)
    _ = tc  # keep the Python object alive through the native call above
    _ = truth  # keep the Python object alive through the native call above
    if t_int == p_int:
        _ = py_call_typed(equal, _A(tc), _A(pred), _P(3 if p_int else 1), _P(size), py_call(eq.buffer_info)[0])
    else:
        var ints = tc if t_int else pred
        var flts = pred if t_int else tc
        _ = py_call_typed(equal, _A(ints), _A(flts), _P(9), _P(size), py_call(eq.buffer_info)[0])
    _ = truth^
    var flip = py_call(b.bytes.maketrans, py_call(b.bytes, Python.list(PythonObject(0), PythonObject(1))),
                       py_call(b.bytes, Python.list(PythonObject(1), PythonObject(0))))
    var miss_u1 = py_call(context["array_type"], PythonObject("B"), py_call(py_call(eq.tobytes).translate, flip))
    var miss = trees_vector(size, "<i4", context)
    _cast_u1(miss_u1, miss, size, context)
    var hit = trees_vector(n, "<i4", context)
    var xn = trees_table(context, _mode(py_fields(model), context), "_mojolearn_x_neighbors")
    _ = py_call_typed(trees_entry(xn, "xn_row_all_zero"), Python.list(_A(miss), _A(hit)), Python.list(_P(n), _P(m)),
                      Python.list())
    var store = py_call(py_attr(py_call(context["array_type"], PythonObject("i"), Python.list(PythonObject(1))), "__mul__"), PythonObject(n))
    var ones = classical_wrap_array(store, Python.tuple(n), "<i4", "C", context)
    var mode = py_get(py_fields(model), "numeric_mode")
    return PythonObject(trees_accuracy(ones, hit, sample_weight, mode, context))


def _cast_u1(source: PythonObject, out_buf: PythonObject, size: Int, context: PythonObject) raises:
    """`astype("<i4")` of a uint8 flag store."""
    trees_cast(Int(py=py_call(source.buffer_info)[0]), "<u1", trees_address(out_buf), "<i4", size, context)


# --------------------------------------------------------------- OneVsRest


def _indicator(xt: PythonObject, codes: PythonObject, j: Int, context: PythonObject) raises -> PythonObject:
    """`OneVsRestClassifier._indicator`: the 0/1 targets of class j."""
    var n = len(codes)
    var yj = trees_vector(n, "<i4", context)
    _ = py_call_typed(_x(xt, "x_trees_indicator_codes"), _A(codes), _A(yj), _P(0), Python.list(_P(n), _P(j)))
    return yj


def _ovr_fit(model: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    _refuse_jobs(model, True, "the classes fit one after another.")
    var xt = _xt(py_fields(model), context)
    var Xa = _x_array(X, context)
    var labels = trees_encode_labels(y, context)
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var est = _attr(model, "estimator", PythonObject(None))
    _require_member(est, _owner(model))
    var targets = Python.list()
    if k == 2:
        _ = py_call(targets.append, labels.codes)
    else:
        for j in range(k):  # driver-loop: over the classes (one native indicator each)
            _ = py_call(targets.append, _indicator(xt, labels.codes, j, context))
    var fitted = Python.list()
    _set(model, "estimators_", fitted)
    for i in range(len(targets)):  # driver-loop: over members
        var e = _clone(est)
        _ = py_call(e.fit, Xa, targets[i])
        _ = py_call(fitted.append, e)
    _set(model, "n_features_in_", PythonObject(_cols(Xa)))
    _set(model, "_fitted", PythonObject(True))
    return model


def _proba_positive(xt: PythonObject, e: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OneVsRestClassifier._proba_positive`: predict_proba's column for
    code 1 (zeros when the member never saw it), float32 (n,)."""
    var n = _rows(Xa)
    var p = _out2d(py_call(e.predict_proba, Xa), n, context)
    var at = _index_of_one(e)
    if at < 0:
        _ = p^
        return trees_vector(n, "<f4", context)
    var col = _one_col(xt, p, at, context)
    _ = p^
    return trees_reshape(col, Python.tuple(n), context)


def _positive(xt: PythonObject, e: PythonObject, Xa: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OneVsRestClassifier._positive`: decision_function when the member
    has one, else predict_proba's column for code 1."""
    if _has(e, "decision_function"):
        return _f32_array(py_call(e.decision_function, Xa), 1, "score", context)
    return _proba_positive(xt, e, Xa, context)


def _ovr_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var classes = _attr(model, "classes_", PythonObject(None))
    var k = len(classes)
    var members = _attr(model, "estimators_", Python.list())
    if k == 2:
        return trees_decode_labels(classes, _i32_vec(py_call(members[0].predict, Xa), "codes", context), context)
    var acc = _class_zero(xt, n * k, context)
    var rows = _arange(xt, n, context)
    for j in range(len(members)):  # driver-loop: over members
        var blk = _out2d(_positive(xt, members[j], Xa, context), n, context)
        _ = _place(xt, acc, n, k, blk, rows, j)
        _ = blk^
    return trees_decode_labels(classes, _argmax(xt, acc, n, k, context), context)


def _ovr_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var k = len(_attr(model, "classes_", Python.list()))
    var members = _attr(model, "estimators_", Python.list())
    var acc = _class_zero(xt, n * k, context)
    var rows = _arange(xt, n, context)
    if k == 2:
        # `_trees_binary_proba`: rows (1 - p, p) of the positive column
        var p = _f32_array(_proba_positive(xt, members[0], Xa, context), 1, "p", context)
        var out = trees_vector(2 * n, "<f8", context)
        _ = py_call_typed(_x(xt, "x_trees_binary_proba"), _A(p), _A(out), Python.list(_P(n)))
        return trees_reshape(out, Python.tuple(n, 2), context)
    for j in range(len(members)):  # driver-loop: over members
        var blk = _out2d(_proba_positive(xt, members[j], Xa, context), n, context)
        _ = _place(xt, acc, n, k, blk, rows, j)
        _ = blk^
    _ = py_call_typed(_x(xt, "x_trees_normalize_rows"), _A(acc), Python.list(_P(n), _P(k)))
    return trees_reshape(acc, Python.tuple(n, k), context)


# ------------------------------------------------------------- Calibration


def _calibrated_validate(model: PythonObject, context: PythonObject) raises:
    var method = _attr(model, "method", PythonObject("sigmoid"))
    if not py_contains(Python.tuple("sigmoid", "isotonic"), method):
        raise Error("method must be 'sigmoid' or 'isotonic'")
    _refuse_jobs(model, False, "the folds fit one after another.")
    var ensemble = _attr(model, "ensemble", PythonObject(True))
    if not py_contains(Python.tuple(True, False), ensemble):
        _refuse("ensemble=" + _repr(ensemble), "True or False.")
    _ = _cv_value(_attr(model, "cv", PythonObject(None)), context)


def _cal_scores(model: PythonObject, xt: PythonObject, e: PythonObject, Xa: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV._scores`: (n, c) float64 scores, c = 1 when
    binary, else one column per class."""
    var n = _rows(Xa)
    var k = len(_attr(model, "classes_", Python.list()))
    var out: PythonObject
    if _has(e, "decision_function"):
        out = py_call(e.decision_function, Xa)
    else:
        out = py_call(e.predict_proba, Xa)
    var blk = _out2d(out, n, context)
    if k == 2:
        var acc = _zeros(n, context)
        if blk.cols == 1:
            _ = py_call_typed(_x(xt, "x_trees_accumulate"), _A(acc), _P(blk.addr), Python.list(_P(n), _F(1.0)))
            _ = blk  # keep the Python object alive through the native call above
        else:
            var at = _index_of_one(e)
            if at < 0:
                raise Error("1 is not in list")
            var col = _one_col(xt, blk, at, context)
            _ = py_call_typed(_x(xt, "x_trees_accumulate"), _A(acc), _A(col), Python.list(_P(n), _F(1.0)))
        _ = blk^
        return trees_reshape(acc, Python.tuple(n, 1), context)
    var acc = _class_zero(xt, n * k, context)
    _ = py_call_typed(_x(xt, "x_trees_accumulate_cols"), _A(acc), _P(blk.addr), _A(_sub_cols(e, context)),
                      Python.list(_P(n), _P(k), _P(len(_attr(e, "classes_", Python.list()))), _F(1.0)))
    _ = blk^
    return trees_reshape(acc, Python.tuple(n, k), context)


def _fit_calibrators(model: PythonObject, xt: PythonObject, S: PythonObject, codes: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV._fit_calibrators`: one calibrator per class
    column, the column and its 0/1 target made natively."""
    var n = _rows(S)
    var c = _cols(S)
    var sigmoid = String(_attr(model, "method", PythonObject("sigmoid"))) == "sigmoid"
    var St = S
    if c > 1:
        St = trees_vector(c * n, "<f8", context)
        _ = py_call_typed(_x(xt, "x_trees_transpose_f64"), _A(S), _A(St), Python.list(_P(n), _P(c)))
    var yj = trees_vector(n, "<i4", context)
    var y64 = PythonObject(None)
    if not sigmoid:
        y64 = trees_vector(n, "<f8", context)
    var cals = Python.list()
    for j in range(c):  # driver-loop: over the class columns (one calibrator each)
        var cls = 1 if c == 1 else j
        var f = trees_address(St) + 8 * j * n
        _ = St  # keep the Python object alive through the native call above
        _ = py_call_typed(_x(xt, "x_trees_indicator_codes"), _A(codes), _A(yj),
                          _A(y64) if y64 is not None else _P(0), Python.list(_P(n), _P(cls)))
        if sigmoid:
            var ab = trees_vector(2, "<f8", context)
            _ = py_call_typed(_x(xt, "x_trees_platt_fit"), _P(f), _A(yj), _A(ab), Python.list(_P(n)))
            _ = St  # keep the Python object alive through the native call above
            _ = py_call(cals.append, Python.tuple(PythonObject("sigmoid"),
                                                  Python.tuple(PythonObject(_f64_word(ab, 0)), PythonObject(_f64_word(ab, 1)))))
        else:
            var kx = trees_vector(n, "<f8", context)
            var ky = trees_vector(n, "<f8", context)
            var m = Int(py=py_call_typed(_x(xt, "x_trees_isotonic_fit"), _P(f), _A(y64), _A(kx), _A(ky), Python.list(_P(n))))
            _ = St  # keep the Python object alive through the native call above
            _ = py_call(cals.append, Python.tuple(PythonObject("isotonic"),
                                                  Python.tuple(_copy_block(kx, 0, m, "<f8", context),
                                                               _copy_block(ky, 0, m, "<f8", context), PythonObject(m))))
    return cals


def _calibrated_probs(model: PythonObject, xt: PythonObject, e: PythonObject, cals: PythonObject, Xa: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`CalibratedClassifierCV._calibrated`: the flat (n, k) probabilities of
    one (member, calibrators) pair."""
    var n = _rows(Xa)
    var k = len(_attr(model, "classes_", Python.list()))
    var S = _cal_scores(model, xt, e, Xa, context)
    var c = _cols(S)
    var w = 2 if k == 2 else k
    var acc = trees_vector(n * w, "<f8", context)
    for j in range(len(cals)):  # driver-loop: over the class columns (one calibrator each)
        var entry = cals[j]
        var col = 1 if k == 2 else j
        var par = entry[1]
        if String(entry[0]) == "sigmoid":
            _ = py_call_typed(_x(xt, "x_trees_platt_apply_strided"), _A(S), _A(acc),
                              Python.list(_P(n), par[0], par[1], _P(n * c), _P(c), _P(j), _P(n * w), _P(w), _P(col)))
        else:
            _ = py_call_typed(_x(xt, "x_trees_isotonic_predict_strided"), _A(par[0]), _A(par[1]), _A(S), _A(acc),
                              Python.list(par[2], _P(n), _P(n * c), _P(c), _P(j), _P(n * w), _P(w), _P(col)))
    if k == 2:
        _ = py_call_typed(_x(xt, "x_trees_complement_pairs"), _A(acc), Python.list(_P(n)))
        return acc
    _ = py_call_typed(_x(xt, "x_trees_normalize_rows"), _A(acc), Python.list(_P(n), _P(k)))
    return acc


def _calibrated_fit(model: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    _calibrated_validate(model, context)
    var xt = _xt(py_fields(model), context)
    var Xa = _x_array(X, context)
    var n = _rows(Xa)
    var d = _cols(Xa)
    var labels = trees_encode_labels(y, context)
    _set(model, "classes_", labels.classes)
    var k = len(labels.classes)
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var base = _attr(model, "estimator", PythonObject(None))
    if base is None:
        base = py_call(_import("mojolearn.svm", "LinearSVC"))
    _require_member(base, _owner(model))
    _set(model, "_cal_fast", PythonObject(None))
    var ensemble = _truthy(_attr(model, "ensemble", PythonObject(True)))
    var splits = _splits(xt, _attr(model, "cv", PythonObject(None)), X, y, n, labels.codes, not ensemble, k, context)
    var cols = _arange(xt, d, context)
    var pairs = Python.list()
    _set(model, "calibrated_classifiers_", pairs)
    if ensemble:
        for f in range(len(splits)):  # driver-loop: over the cv folds
            var tr = splits[f][0]
            var te = splits[f][1]
            var e = _clone(base)
            _ = py_call(e.fit, _gather(xt, trees_address(Xa), n, d, tr, cols, context),
                        _gather_codes(xt, labels.codes, tr, context))
            _ = Xa  # keep the Python object alive through the native call above
            var Xte = _gather(xt, trees_address(Xa), n, d, te, cols, context)
            _ = Xa  # keep the Python object alive through the native call above
            var cals = _fit_calibrators(model, xt, _cal_scores(model, xt, e, Xte, context),
                                        _gather_codes(xt, labels.codes, te, context), context)
            _ = Xa  # keep the Python object alive through the native call above
            _ = py_call(pairs.append, Python.tuple(e, cals))
    else:
        var c = 1 if k == 2 else k
        var S = _zeros(n * c, context)
        for f in range(len(splits)):  # driver-loop: over the cv folds
            var tr = splits[f][0]
            var te = splits[f][1]
            var e = _clone(base)
            _ = py_call(e.fit, _gather(xt, trees_address(Xa), n, d, tr, cols, context),
                        _gather_codes(xt, labels.codes, tr, context))
            _ = Xa  # keep the Python object alive through the native call above
            var Xte = _gather(xt, trees_address(Xa), n, d, te, cols, context)
            _ = Xa  # keep the Python object alive through the native call above
            var blk = trees_f32(_cal_scores(model, xt, e, Xte, context), 2, "C", "scores", context)
            _ = Xa  # keep the Python object alive through the native call above
            _ = _place(xt, S, n, c, blk, te, 0)
            _ = blk^
        var e = _clone(base)
        _ = py_call(e.fit, Xa, labels.codes)
        _ = py_call(pairs.append, Python.tuple(e, _fit_calibrators(model, xt, trees_reshape(S, Python.tuple(n, c), context),
                                                                   labels.codes, context)))
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "_fitted", PythonObject(True))
    return model


def _calibrated_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var Xa = _check_x(model, X, context)
    var xt = _xt(py_fields(model), context)
    var n = _rows(Xa)
    var k = len(_attr(model, "classes_", Python.list()))
    var acc = _class_zero(xt, n * k, context)
    var pairs = _attr(model, "calibrated_classifiers_", Python.list())
    for i in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
        var p = _calibrated_probs(model, xt, pairs[i][0], pairs[i][1], Xa, context)
        _acc_w(xt, acc, p, n * k, 1.0, context)
    _scale(xt, acc, Float64(len(pairs)))
    return trees_reshape(acc, Python.tuple(n, k), context)


def _calibrated_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var p = _calibrated_proba(model, X, context)
    var xt = _xt(py_fields(model), context)
    var classes = _attr(model, "classes_", PythonObject(None))
    return trees_decode_labels(classes, _argmax(xt, p, _rows(p), len(classes), context), context)


# ------------------------------------------------------------------- score


def _regressor_score(model: PythonObject, prediction: PythonObject, y: PythonObject, sample_weight: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`RegressorScoreMixin.score`: the float32 R^2 of the prediction."""
    var mode = py_get(py_fields(model), "numeric_mode")
    var truth = trees_f32(y, -1, "C", "y", context)
    var pred = trees_f32(prediction, -1, "C", "prediction", context)
    if len(_attr(pred.keep, "shape", Python.tuple(0))) == 2 or len(_attr(truth.keep, "shape", Python.tuple(0))) == 2:
        # 2-D targets: the metrics family's public r2_score (its multioutput
        # program), on the same float32 blocks.
        var r2 = _import("mojolearn._metrics_impl", "r2_score")
        var kwargs = Python.dict()
        py_setitem(kwargs, PythonObject("sample_weight"), sample_weight)
        py_setitem(kwargs, PythonObject("numeric_mode"), mode)
        var result = _call_kw(r2, Python.tuple(truth.keep, pred.keep), kwargs)
        _ = truth^
        _ = pred^
        return result
    var result = trees_r2(truth.keep, pred.keep, sample_weight, mode, context)
    _ = truth^
    _ = pred^
    return PythonObject(result)


def _meta_predict(model: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "voting_classifier":
        return _voting_predict(model, X, context)
    if kind == "voting_regressor":
        return _voting_regress(model, X, context)
    if kind == "stacking_classifier":
        return _stacking_predict(model, X, True, context)
    if kind == "stacking_regressor":
        return _stacking_predict(model, X, False, context)
    if kind == "multioutput_regressor":
        return _multioutput_regressor_predict(model, X, context)
    if kind == "multioutput_classifier":
        return _multioutput_classifier_predict(model, X, context)
    if kind == "ovr":
        return _ovr_predict(model, X, context)
    if kind == "calibrated":
        return _calibrated_predict(model, X, context)
    raise Error("unknown meta-estimator schema " + kind)


def _meta_fit(model: PythonObject, kind: String, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
              context: PythonObject) raises -> PythonObject:
    if kind == "voting_classifier":
        return _voting_fit(model, X, y, sample_weight, True, context)
    if kind == "voting_regressor":
        return _voting_fit(model, X, y, sample_weight, False, context)
    if kind == "stacking_classifier":
        return _stacking_classifier_fit(model, X, y, sample_weight, context)
    if kind == "stacking_regressor":
        return _stacking_regressor_fit(model, X, y, sample_weight, context)
    if kind == "multioutput_regressor":
        return _multioutput_regressor_fit(model, X, y, sample_weight, context)
    if kind == "multioutput_classifier":
        return _multioutput_classifier_fit(model, X, y, sample_weight, context)
    if kind == "ovr":
        return _ovr_fit(model, X, y, context)
    if kind == "calibrated":
        return _calibrated_fit(model, X, y, context)
    raise Error("unknown meta-estimator schema " + kind)


def _meta_transform(model: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "voting_classifier":
        return _voting_transform(model, X, True, context)
    if kind == "voting_regressor":
        return _voting_transform(model, X, False, context)
    if kind == "stacking_classifier" or kind == "stacking_regressor":
        return _stacking_transform(model, X, context)
    raise Error("AttributeError: '" + _owner(model) + "' object has no attribute 'transform'")


def _fit_params_weight(fit_params: PythonObject) raises -> PythonObject:
    """`fit(X, y, **fit_params)`: only `sample_weight` is a fit parameter."""
    var b = trees_builtins()
    var keys = py_call(b.list, fit_params)
    for i in range(len(keys)):  # small-loop(keyword arguments): the caller's fit parameters
        if String(keys[i]) != "sample_weight":
            raise Error("TypeError: fit() got an unexpected keyword argument '" + String(keys[i]) + "'")
    return py_get(fit_params, PythonObject("sample_weight"))


def meta_operation(model: PythonObject, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """The meta-estimators' public methods (`trees_meta_*`)."""
    var kind = meta_kind(model, context)
    if operation == "trees_meta_fit":
        return _meta_fit(model, kind, args[0], args[1], args[2], context)
    if operation == "trees_meta_predict":
        return _meta_predict(model, kind, args[0], context)
    if operation == "trees_meta_predict_proba":
        if kind == "voting_classifier":
            return _voting_predict_proba(model, args[0], context)
        if kind == "stacking_classifier":
            return _stacking_predict_proba(model, args[0], context)
        if kind == "multioutput_classifier":
            return _multioutput_classifier_proba(model, args[0], context)
        if kind == "ovr":
            return _ovr_predict_proba(model, args[0], context)
        if kind == "calibrated":
            return _calibrated_proba(model, args[0], context)
        raise Error("AttributeError: '" + _owner(model) + "' object has no attribute 'predict_proba'")
    if operation == "trees_meta_transform":
        return _meta_transform(model, kind, args[0], context)
    if operation == "trees_meta_fit_transform":
        var fitted = _meta_fit(model, kind, args[0], args[1], _fit_params_weight(args[2]), context)
        _ = fitted
        return _meta_transform(model, kind, args[0], context)
    if operation == "trees_meta_score":
        var X = args[0]
        var y = args[1]
        var sample_weight = args[2]
        if kind == "multioutput_classifier":
            return _multioutput_classifier_score(model, X, y, sample_weight, context)
        var prediction = _meta_predict(model, kind, X, context)
        if kind.endswith("_regressor"):
            return _regressor_score(model, prediction, y, sample_weight, context)
        var mode = py_get(py_fields(model), "numeric_mode")
        return PythonObject(trees_accuracy(y, prediction, sample_weight, mode, context))
    raise Error("unknown meta-estimator operation " + operation)
