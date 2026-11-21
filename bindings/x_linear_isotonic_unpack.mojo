# SPDX-License-Identifier: Apache-2.0
# UNPACK of IsotonicRegression's operations (fit, fit_transform, transform,
# predict; python/mojolearn/_expansion_linear.py): the refusals in the
# order and words the Python made them, the one feature column through the
# one admission, the joined knots of a fitted model, and the typed requests
# of x_linear/door_run.mojo (`XlFitRequest`), one function per public
# operation. RUN (`xl_isotonic_fit_run`, `xl_fit_run`) holds no Python
# object; PACK is bindings/x_linear_isotonic_pack.mojo.
from x_linear.door_run import ALGO_ISOTONIC, ALGO_ISOTONIC_PREDICT
from bindings.x_linear_common_unpack import XlFit, xl_fit_build, _xl_or
from std.python import Python, PythonObject
from bindings.x_linear_route_unpack import XLinearOps, xl_ops_require
from bindings.x_linear_common_unpack import (
    _b, _truth, _float, _eq, _xl_join, _xl_vector, _xl_column, _xl_words, _xl_with_weights,
)
from bindings.py_balanced import py_type, py_fields, py_contains
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from core.host_bytes_equal import host_bytes_equal
from providers.python.array_metadata_pyapi_door import array_slot_once
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.py_balanced import py_item
from bindings.classical_labels_door import labels_address


struct XlIsotonicFit(Movable):
    """One admitted IsotonicRegression.fit: the fit call, whether the
    direction is asked of the binding (`increasing='auto'`), and the
    address of the n targets alone (the fit's own target words carry the
    weights after them); `y` owns those targets."""
    var fit: XlFit
    var auto: Bool
    var targets: Int
    var y: PythonObject

    def __init__(out self, var fit: XlFit, auto: Bool, targets: Int, var y: PythonObject):
        self.fit = fit^
        self.auto = auto
        self.targets = targets
        self.y = y^


struct XlIsotonicTransform(Movable):
    """One admitted IsotonicRegression.transform / predict: the predict
    call over the joined knots (the call's targets), and whether the word
    after the predictions is the out-of-bounds flag (`'raise'`)."""
    var fit: XlFit
    var raise_oob: Bool

    def __init__(out self, var fit: XlFit, raise_oob: Bool):
        self.fit = fit^
        self.raise_oob = raise_oob


def _xl_isotonic_check(fields: PythonObject) raises:
    var nk = x_linear_keys()
    if not py_contains(Python.tuple("nan", "clip", "raise"), py_item(fields, nk[].k_out_of_bounds)):
        raise Error("mojolearn IsotonicRegression: out_of_bounds must be 'nan', 'clip' or 'raise'")


#: The joined isotonic knots of a fitted model (lane fx-rows-rest,
#: 2026-11-18): one prepared-registry record per model (outside its
#: __dict__, dropped with the model), the tuple
#:   (X_thresholds_ object, y_thresholds_ object, joined float32 Array,
#:    knots per array, X words address, y words address, joined address)
comptime _XL_ISO_KNOTS_SLOT = "x_linear_isotonic_knots_v1"


def _xl_own_f32(value: PythonObject, address: Int, count: Int, context: PythonObject) raises -> Bool:
    """`value` is an exact mojolearn float32 Array of `count` words whose
    own words sit at `address` (its slots read natively; no method runs)."""
    var nk = x_linear_keys()
    if py_type(value) is not py_item(context, nk[].k_array_class):
        return False
    var b = _b()
    var dtype = array_slot_once(value, "dtype", context)
    var size = array_slot_once(value, "size", context)
    var addr = array_slot_once(value, "_addr", context)
    if py_type(dtype) is not b.str or py_type(size) is not b.int or py_type(addr) is not b.int:
        return False
    return String(py=dtype) == "<f4" and Int(py=size) == count and Int(py=addr) == address


def _xl_isotonic_knots(model: PythonObject, fields: PythonObject,
                       context: PythonObject) raises -> Tuple[PythonObject, Int, Int, Int]:
    """The joined knots [X_thresholds_ | y_thresholds_] of a fitted
    IsotonicRegression: (float32 Array, knots k, its address, its words).

    Every predict admitted both fitted arrays again and joined them into a
    new Array (two admissions through the NumPy view, an allocation and two
    copies: about 50 us of a 0.34 ms predict of 100,000 queries, call
    profile nvc-prof-20261118T175049Z-49fb08). The joined Array is now kept
    in the model's record and served while it is provably the join this
    call would build: the two stored attributes are the SAME objects
    (a replaced X_thresholds_ / y_thresholds_ misses), each is still an
    exact float32 Array of the same length at the same address, and its
    words are bitwise equal to the kept copy (an in-place edit through any
    view misses; the compare reads the 2 k knot words once, across the
    host's cores, where the join copied them). A miss runs the old path and
    replaces the record; arrays the door would cast or copy (another dtype,
    a caller's ndarray) are never recorded. Same words to the kernel: no bit
    moves. Nothing is frozen and no user-visible attribute changes."""
    var nk = x_linear_keys()
    var xs = py_item(fields, nk[].k_X_thresholds_)
    var ys = py_item(fields, nk[].k_y_thresholds_)
    var mods = NativeModules()
    var held = prepared_get(mods, _XL_ISO_KNOTS_SLOT, model)
    if held is not None and held[0] is xs and held[1] is ys:
        var count = Int(py=held[3])
        var xa = Int(py=held[4])
        var ya = Int(py=held[5])
        if _xl_own_f32(xs, xa, count, context) and _xl_own_f32(ys, ya, count, context):
            var joined = held[2]
            var joined_at = Int(py=held[6])
            var same = (host_bytes_equal(joined_at, xa, 4 * count)
                        and host_bytes_equal(joined_at + 4 * count, ya, 4 * count))
            _ = xs  # keep the Python object alive through the compare above
            _ = ys  # keep the Python object alive through the compare above
            if same:
                return (joined, count, joined_at, 2 * count)
    var xt = _xl_words(xs, "X_thresholds_", context)
    var yt = _xl_words(ys, "y_thresholds_", context)
    var k = len(xs)
    var x_count = xt.rows * xt.cols
    var y_count = yt.rows * yt.cols
    var thr = _xl_join(xt.address, x_count, yt.address, y_count, context)
    var base = labels_address(thr)
    if (k > 0 and x_count == k and y_count == k and _xl_own_f32(xs, xt.address, k, context)
            and _xl_own_f32(ys, yt.address, k, context)):
        # Nothing is stored for a model that takes no weak reference.
        prepared_put(mods, _XL_ISO_KNOTS_SLOT, model,
                     Python.tuple(xs, ys, thr, PythonObject(k), PythonObject(xt.address), PythonObject(yt.address),
                                  PythonObject(base)))
    _ = yt  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
    return (thr, k, base, x_count + y_count)


def xl_isotonic_fit_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                           context: PythonObject) raises -> XlIsotonicFit:
    """IsotonicRegression.fit(X, y, sample_weight): the column, the
    targets (then the checked weights), the bounds; the binding sorts the
    positive-weight rows by (x, y, row) and pools them."""
    var nk = x_linear_keys()
    _xl_isotonic_check(fields)
    xl_ops_require(entries)
    var x = _xl_column(args[0], "X", context)
    var n = x.rows
    var y = _xl_vector(args[1], n, context)
    var increasing = py_item(fields, nk[].k_increasing)
    var auto = _eq(increasing, PythonObject("auto"))
    var rising = False
    if not auto:
        rising = _truth(increasing)
    var has_w = 0
    var yy = _xl_with_weights(y, args[2], n, has_w, context)
    var y_min = py_item(fields, nk[].k_y_min)
    var y_max = py_item(fields, nk[].k_y_max)
    var ip: List[Int] = [1 if rising else 0, 0 if y_min is None else 1, 0 if y_max is None else 1, has_w]
    var fp: List[Float64] = [_xl_or(y_min, 0.0), _xl_or(y_max, 0.0)]
    var fit = xl_fit_build(entries, ALGO_ISOTONIC, x.address, n, n, 1, x.array, yy.address, yy.rows, yy.array, ip^,
                           fp^, 3 + 2 * n, 6 * n, 3 * n, context)
    return XlIsotonicFit(fit^, auto, y.address, y.array)


def _xl_isotonic_transform_build(model: PythonObject, fields: PythonObject, t_addr: Int, n: Int,
                                 t_keep: PythonObject, entries: XLinearOps,
                                 context: PythonObject) raises -> XlIsotonicTransform:
    """The predict call over a column already read: the joined thresholds
    (the model's record, `_xl_isotonic_knots`), the interp1d-linear predict
    on the binding; out_of_bounds='raise' is the flag word after the
    predictions."""
    var nk = x_linear_keys()
    var knots = _xl_isotonic_knots(model, fields, context)
    var oob_mode = py_item(fields, nk[].k_out_of_bounds)
    var raise_oob = _eq(oob_mode, PythonObject("raise"))
    var oob = 2 if raise_oob else (1 if _eq(oob_mode, PythonObject("clip")) else 0)
    var ip: List[Int] = [knots[1], oob]
    var fp: List[Float64] = [_float(py_item(fields, nk[].k_X_min_)), _float(py_item(fields, nk[].k_X_max_))]
    # Both bindings write every prediction word (host `isotonic_predict`,
    # one query per team slot; device `_iso_predict_grid`, one thread per
    # query): only the out-of-bounds word at [n] starts as zero (the host
    # column ORs into it).
    var fit = xl_fit_build(entries, ALGO_ISOTONIC_PREDICT, t_addr, n, n, 1, t_keep, knots[2], knots[3], knots[0],
                           ip^, fp^, n + 1 if raise_oob else n, 1, 1, context, written=n)
    return XlIsotonicTransform(fit^, raise_oob)


def xl_isotonic_transform_unpack(model: PythonObject, source: PythonObject, entries: XLinearOps,
                                 context: PythonObject) raises -> XlIsotonicTransform:
    """IsotonicRegression.transform(T) / predict(T)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    if not py_contains(fields, nk[].k_X_thresholds_):
        raise Error("RuntimeError: mojolearn IsotonicRegression: call fit first")
    xl_ops_require(entries)
    var t = _xl_column(source, "T", context)
    return _xl_isotonic_transform_build(model, fields, t.address, t.rows, t.array, entries, context)


def xl_isotonic_refit_transform_unpack(model: PythonObject, fields: PythonObject, fitted: XlIsotonicFit,
                                       entries: XLinearOps, context: PythonObject) raises -> XlIsotonicTransform:
    """The second half of fit_transform, `self.fit(X, y,
    sample_weight).transform(X)`: the column the fit already converted,
    after the fit's PACK published the thresholds."""
    return _xl_isotonic_transform_build(model, fields, fitted.fit.request.x, fitted.fit.request.n, fitted.fit.x,
                                        entries, context)
