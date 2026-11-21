# SPDX-License-Identifier: Apache-2.0
# Complete native AutoARIMA search: exact option admission, candidate plan,
# compiled differencing/fit/search/IC selection, grouping and finished state.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_input, classical_typed, classical_address
from bindings.classical_pandas_door import classical_admit_value
from bindings.sequential_admission_door import sequential_integer
from bindings.sequential_arrays_door import sequential_array
from bindings.tsa_autoarima_native_door import _new_arima
from bindings.tsa_arima_native_door import arima_fit_routed
from bindings.tsa_arima_route_unpack import autoarima_route
from core.arima_run import (
    arima_fast_search_mode_run, arima_ndiffs_run, arima_order_caps_run, arima_order_search_device_run,
    arima_order_search_fit_run, arima_order_search_run, arima_route_searches, arima_search_switches_run,
)
from core.arima_entry_table import ARIMA_SWITCH_ORDER_BATCH, ARIMA_SWITCH_SEARCH_REUSE
from providers.python.autoarima_lifecycle_pyapi_door import _fields, _integer, _integer_object, _number, _packed, _buffer
from providers.python.array_metadata_pyapi_door import ArrayPin
from arima.autoarima_search_run import _penalty
from providers.python.autoarima_owner_pyapi_unpack import autoarima_commit
from bindings.py_optional import py_present
from bindings.tsa_keys_unpack import tsa_keys
from bindings.py_balanced import py_item


struct _Candidate(Copyable, Movable):
    # Private operation-owned metadata: the search loops never need Python
    # tuple indexing or integer conversion to recover a candidate's order.
    var p: Int
    var q: Int
    var seasonal_p: Int
    var seasonal_q: Int
    var period: Int
    var intercept: Int
    # The boxed scalar the compiled IC fold (a base helper called through
    # its Python entry) takes, boxed once per candidate; the grouped kernel
    # reads `penalty_value`.
    var penalty: PythonObject
    var penalty_value: Float64

    def __init__(out self, p: Int, q: Int, seasonal_p: Int, seasonal_q: Int,
                 period: Int, intercept: Int, penalty: Float64) raises:
        self.p = p
        self.q = q
        self.seasonal_p = seasonal_p
        self.seasonal_q = seasonal_q
        self.period = period
        self.intercept = intercept
        self.penalty = PythonObject(penalty)
        self.penalty_value = penalty


def _array(count: Int, code: Int, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var key = Python.tuple("f" if code < 2 else "i", 4 if code == 0 or code == 2 else 8)
    return sequential_array(Python.tuple(max(count, 1)), py_item(context, nk[].k_numeric_dtypes)[key], context)


def _base(context: PythonObject, name: String) raises -> PythonObject:
    return context["base_" + name]


def _cast(value: PythonObject, source: Int, target: Int, count: Int, context: PythonObject) raises -> PythonObject:
    var out = _array(count, target, context)
    if count:
        var refused = Int(py=py_call_typed(_base(context, "cast_elements"), classical_address(value), source,
                                           classical_address(out), target, count))
        if refused:
            raise Error("ValueError: AutoARIMA native result conversion refused a value")
    return out


def _take(source: PythonObject, indices: PythonObject, rows: Int, count: Int, width: Int,
          code: Int, context: PythonObject) raises -> PythonObject:
    var out = _array(count * width, code, context)
    if count:
        _ = py_call_typed(_base(context, "gather_rows_bytes"), classical_address(source), classical_address(out),
                          classical_address(indices), rows, count, width * (4 if code == 0 or code == 2 else 8))
    return out


def _put(output: PythonObject, indices: PythonObject, values: PythonObject, rows: Int, count: Int,
         bytes_per_row: Int, context: PythonObject) raises:
    _ = py_call_typed(_base(context, "scatter_rows_bytes"), classical_address(values), classical_address(indices),
                      count, bytes_per_row, classical_address(output), rows)


def _counts(words: PythonObject, code: Int, count: Int, classes: Int, context: PythonObject) raises -> PythonObject:
    var out = _array(classes, 3, context)
    _ = py_call_typed(_base(context, "bincount_i64"), classical_address(words), code, count, classes,
                      classical_address(out), 0)
    return out


def _where(words: PythonObject, want: Int, rows: Int, count: Int, context: PythonObject) raises -> PythonObject:
    var out = _array(count, 3, context)
    var scratch = _array(rows, 3, context)
    var got = Int(py=py_call_typed(_base(context, "select_fold_i64"), classical_address(words), rows, want,
                                  classical_address(out), classical_address(scratch)))
    # The native selection writes this scratch allocation through a raw
    # address. Keep its ndarray owner alive until those writes finish.
    _ = scratch
    if got != count:
        raise Error("RuntimeError: AutoARIMA: select_fold_i64 disagrees with the counts")
    return out


def _options(name: String, value: PythonObject, low: Int, high: Int, context: PythonObject) raises -> List[Int]:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var options = Python.list()
    if sequential_integer(value, context) or cls is py_type(PythonObject(True)):
        options = Python.list(value)
    elif cls is b.list or cls is b.tuple or cls is b.range:
        options = py_call(b.list, value)
    else:
        var admitted_pin = ArrayPin()
        var admitted = _buffer(value, "order options", context, admitted_pin)
        if len(admitted.shape) != 1 or admitted.is_array:
            raise Error("TypeError: AutoARIMA order options require a native vector, range, list or tuple")
        options = py_call(admitted_pin.keep.tolist)
    var result = List[Int]()
    for i in range(len(options)):  # driver-loop: over candidate order option metadata
        var word = _integer_object(options[i], context)
        if Bool(py=word >= PythonObject(low)) and Bool(py=word <= PythonObject(high)):
            result.append(Int(py=word))
    if len(result) == 0:
        raise Error("ValueError: AutoARIMA: no valid option for " + name)
    return result^


def _string(value: PythonObject, name: String) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: AutoARIMA " + name + " requires an exact native string")
    return String(value)


def autoarima_search(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = tsa_keys()
    var fields = _fields(model, py_item(context, nk[].k_autoarima_class))
    var criterion = _string(args[8], "ic")
    if criterion != "aic" and criterion != "aicc" and criterion != "bic":
        raise Error("ValueError: AutoARIMA: ic must be aic, aicc or bic")
    if _string(args[9], "test") != "kpss":
        raise Error("ValueError: AutoARIMA: test must be kpss")
    var method = _string(args[13], "method")
    if method != "auto" and method != "ml":
        raise Error("NotImplementedError: AutoARIMA: this ARIMA fits by ml only")
    if _number(args[11], context) != 1e-8 or _integer(args[14], context) != 0:
        raise Error("NotImplementedError: AutoARIMA: h and truncate are not carried by mojolearn.ARIMA")
    var maxiter = _integer(args[12], context)
    var period = 0 if args[0] is None else _integer(args[0], context)
    if period <= 1:
        period = 0
    var seasonal_d = 0
    if period:
        var options = _options("D", args[2], 0, 1, context)
        if len(options) > 1:
            raise Error("NotImplementedError: AutoARIMA: seasonal_test over multiple D options is not implemented; pass D as one value")
        seasonal_d = options[0]
    var ds = _options("d", args[1], 0, 2 - seasonal_d, context)
    var ps = _options("p", args[3], 0, period - 1 if period else 4, context)
    var qs = _options("q", args[4], 0, period - 1 if period else 4, context)
    var Ps = _options("P", args[5], 0, 4 if period else 0, context)
    var Qs = _options("Q", args[6], 0, 4 if period else 0, context)
    var auto_k = py_type(args[7]) is py_type(PythonObject("")) and String(args[7]) == "auto"
    var explicit_k = List[Int]()
    if not auto_k:
        explicit_k = _options("k", args[7], 0, 1, context)
    var y_pin = ArrayPin()
    var y = _packed(py_item(fields, nk[].k__endog), "endog", context, y_pin)
    if len(y.shape) != 2 or y.shape[0] <= 0 or y.shape[1] <= 0:
        raise Error("ValueError: AutoARIMA endog must be a nonempty matrix")
    var batch = y.shape[0]
    var n_obs = y.shape[1]
    if _integer(py_item(fields, nk[].k_batch_size), context) != batch or _integer(py_item(fields, nk[].k_n_obs), context) != n_obs:
        raise Error("ValueError: AutoARIMA stored dimensions disagree with endog")
    # The AutoARIMA's one route (its device, the default numeric mode its
    # per-group models carry) serves the whole search. Its search words are
    # read once per search: the fast-search arms, the search capabilities
    # and the two switches an identity trace turns off.
    var call = autoarima_route(model, fields, context, native_loaded_module("builtins"))
    var route = call.route
    var device = py_get(fields, nk[].k_device)
    var fast_mode = arima_fast_search_mode_run(route)
    var caps = arima_order_caps_run(route)
    var switches = arima_search_switches_run(route)
    var py = Python()
    var device_ic = (caps & 4) != 0
    var ic_code = 0 if device_ic else 1
    var d32 = _array(batch, 2, context)
    if len(ds) == 1:
        # A constant option is one metadata row gathered at zero indices.
        var scalar = classical_input(Python.list(ds[0]), context, "d option")
        var scalar32 = _cast(scalar, 3, 2, 1, context)
        var zeros = _array(batch, 3, context)
        _ = py_call_typed(_base(context, "gather_rows_bytes"), classical_address(scalar32), classical_address(d32),
                          classical_address(zeros), 1, batch, 4)
        # Raw addresses do not keep the source row or the zero index buffer
        # alive: without these, both can be freed before the gather reads them.
        _ = scalar32
        _ = zeros
    else:
        for i in range(len(ds)):  # driver-loop: over the bounded differencing options
            if ds[i] != i:
                raise Error("NotImplementedError: AutoARIMA: a d option list must be 0, 1, ..., d_max")
        if fast_mode & 32 and seasonal_d == 0:
            if not arima_route_searches(route):
                raise Error("AttributeError: the ARIMA binding has no arima_ndiffs_kpss")
            var y_address = y.source.addr
            var d32_address = Int(py=classical_address(d32))
            var d_max = ds[len(ds) - 1]
            var written = 0
            try:
                with GILReleased(py):
                    written = arima_ndiffs_run(route, y_address, d32_address, batch, n_obs, d_max)
            finally:
                _ = d32  # keep the Python object alive through the native call above
                _ = call.owner  # the route owner pins the image through the call
            if written != batch:
                raise Error("RuntimeError: AutoARIMA: incomplete ndiffs output")
        else:
            var tsa = py_item(context, nk[].k_tsa_bindings)[py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)]
            if py_type(tsa) is py_type(PythonObject("")):
                raise Error(String(tsa))
            _ = py_call_typed(py_item(tsa, nk[].k_select_d), y.source.addr, classical_address(d32),
                              Python.list(batch, n_obs, seasonal_d, period, ds[len(ds) - 1], 0.05))
    var dser = _cast(d32, 2, 3, batch, context)
    var dcounts = _counts(d32, 2, batch, 3, context)
    # These options are invariant across differencing groups. Keep the native
    # decision once; d only changes the candidate penalty and intercept options.
    var plain = period == 0 and seasonal_d == 0
    for p in ps:  # driver-loop: over AR candidate metadata once per search
        plain = plain and p != 4
    for q in qs:  # driver-loop: over MA candidate metadata once per search
        plain = plain and q != 4
    var dcount_words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(dcounts)))
    var models = Python.list()
    var ids_all = Python.list()
    var reuse_all = Python.list()
    var fitted = Python.list()
    var order_out = classical_typed(Python.tuple(batch, 8), "<i8", context)
    var ic_out = classical_typed(Python.tuple(batch), "<f8", context)
    for d in range(3):  # driver-loop: over the three differencing groups
        var count = Int(dcount_words[d])
        if count == 0:
            continue
        var ids = _where(d32, d, batch, count, context)
        var subflat = _take(y_pin.keep, ids, batch, count, n_obs, 0, context)
        var sub = py_call(subflat.reshape, Python.tuple(count, n_obs))
        var ks = explicit_k.copy()
        if auto_k:
            ks = List[Int]()
            if d + seasonal_d > 1:
                ks.append(0)
            else:
                if fast_mode & 8:
                    ks.append(0)
                ks.append(1)
        var grid = List[_Candidate]()
        var table = Python.list()
        # The device search's plan words after its seven fixed ones, and one
        # criterion penalty per order (typed: no Python list per search).
        var packed = List[Int]()
        var pens = List[Float64]()
        for p in ps:  # driver-loop: over AR candidate metadata
            for q in qs:  # driver-loop: over MA candidate metadata
                for P in Ps:  # driver-loop: over seasonal AR candidate metadata
                    for Q in Qs:  # driver-loop: over seasonal MA candidate metadata
                        for k in ks:  # driver-loop: over intercept candidate metadata
                            if p + q + P + Q + k == 0:
                                continue
                            var s = period if P + seasonal_d + Q else 0
                            var penalty = _penalty(p + q + P + Q + k + 1, criterion, n_obs, d, seasonal_d, s)
                            var candidate = _Candidate(p, q, P, Q, s, k, penalty)
                            _ = py_call(table.append, Python.list(p, d, q, P, seasonal_d, Q, s, k))
                            packed.append(p)
                            packed.append(q)
                            packed.append(P)
                            packed.append(Q)
                            packed.append(k)
                            pens.append(penalty)
                            grid.append(candidate^)
        var trials = len(grid)
        if trials == 0:
            raise Error("ValueError: AutoARIMA: no (p, q, P, Q, k) order to try")
        var best = _array(count, 3, context)
        var best_ic = _array(count, ic_code, context)
        var ic_k = _array(count, ic_code, context)
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var grouped_ll = Optional[PythonObject]()
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var search_fit = Optional[PythonObject]()
        var chosen = False
        if n_obs > 2 and (switches & ARIMA_SWITCH_ORDER_BATCH) != 0 and arima_route_searches(route):
            if caps & 1 and (plain or caps & 2):
                var best32 = _array(count, 2, context)
                var out = _array(count if device_ic else trials * count, 0, context)
                var plan: List[Int] = [count, n_obs, d, seasonal_d, period, maxiter, 1 if device_ic else 0]
                for word in packed:  # driver-loop: over the candidate plan words of this group
                    plan.append(word)
                var sub_address = Int(py=classical_address(sub))
                var out_address = Int(py=classical_address(out))
                var best32_address = Int(py=classical_address(best32))
                var written = 0
                try:
                    with GILReleased(py):
                        written = arima_order_search_device_run(route, sub_address, out_address, best32_address, plan, pens)
                finally:
                    # best32 is only read back on the device-IC route; keep its
                    # owner alive through the native write on every route.
                    _ = best32
                    _ = out
                    _ = sub
                    _ = call.owner  # the route owner pins the image through the call
                var expected = count if device_ic else trials * count
                if written and written != expected:
                    raise Error("RuntimeError: AutoARIMA: incomplete grouped search output")
                if written:
                    if device_ic:
                        chosen = True
                        best = _cast(best32, 2, 3, count, context)
                        best_ic = out
                    else:
                        grouped_ll = py_present(out)
            elif plain:
                var compact: List[Int] = [count, n_obs, d, maxiter]
                var offsets = Python.list()
                var off = 0
                for trial in range(trials):  # driver-loop: over candidate search plan metadata
                    ref order = grid[trial]
                    compact.append(order.p)
                    compact.append(order.q)
                    compact.append(order.intercept)
                    _ = py_call(offsets.append, off)
                    off += count * (3 * (order.p + order.q + order.intercept + 1) + 1)
                grouped_ll = _array(trials * count, 0, context)
                var written = 0
                var sub_address = Int(py=classical_address(sub))
                var ll_address = Int(py=classical_address(grouped_ll.value()))
                if (switches & ARIMA_SWITCH_SEARCH_REUSE) != 0:
                    var fitf = _array(off, 0, context)
                    var fiti = _array(2 * count * trials, 2, context)
                    var fitf_address = Int(py=classical_address(fitf))
                    var fiti_address = Int(py=classical_address(fiti))
                    try:
                        with GILReleased(py):
                            written = arima_order_search_fit_run(route, sub_address, ll_address, fitf_address,
                                                                 fiti_address, compact)
                    finally:
                        _ = fitf
                        _ = fiti
                        _ = sub
                        _ = call.owner  # the route owner pins the image through the call
                    search_fit = Python.tuple(fitf, fiti, offsets, count, grouped_ll.value())
                else:
                    try:
                        with GILReleased(py):
                            written = arima_order_search_run(route, sub_address, ll_address, compact)
                    finally:
                        _ = sub
                        _ = call.owner  # the route owner pins the image through the call
                if written == 0:
                    grouped_ll = Optional[PythonObject]()
                    search_fit = Optional[PythonObject]()
                elif written != trials * count:
                    raise Error("RuntimeError: AutoARIMA: incomplete grouped likelihood output")
        if not chosen:
            # All owners stay live through this loop, and these private output
            # buffers are not resized or replaced by an individual candidate.
            # Resolve their addresses and the IC binding once per group.
            var ic_entry = _base(context, "ic_running_min_f32" if device_ic else "ic_running_min_f64")
            var ic_k_address = classical_address(ic_k)
            var best_ic_address = classical_address(best_ic)
            var best_address = classical_address(best)
            var grouped_address = 0 if not grouped_ll else Int(py=classical_address(grouped_ll.value()))
            var arima_class = py_item(context, nk[].k_arima_class)
            for trial in range(trials):  # driver-loop: over candidate order fits and compiled IC folds
                ref order = grid[trial]
                var llf = PythonObject(None)
                var address = 0
                if not grouped_ll:
                    var m = _new_arima(Python.tuple(order.p, d, order.q),
                        Python.tuple(order.seasonal_p, seasonal_d, order.seasonal_q, order.period), order.intercept, maxiter,
                        device, context)
                    _ = arima_fit_routed(m, Python.tuple(sub, PythonObject(None)), context, call)
                    var state = _fields(m, arima_class)
                    llf = _cast(py_item(state, nk[].k_llf_), 1, 0, count, context)
                    address = Int(py=classical_address(llf))
                else:
                    address = grouped_address + trial * count * 4
                _ = py_call_typed(ic_entry, address, count, order.penalty, trial,
                                  ic_k_address, best_ic_address, best_address)
                _ = llf
            # Raw addresses do not keep Python-owned allocations alive.
            _ = ic_k
            _ = grouped_ll
        # sub is passed to the grouped kernels by raw address only.
        _ = sub
        var table_array = classical_input(table, context, "order table")
        var selected_orders = _take(table_array, best, trials, count, 8, 3, context)
        _put(order_out, ids, selected_orders, batch, count, 64, context)
        var wide_ic = _cast(best_ic, 0, 1, count, context) if device_ic else best_ic
        _put(ic_out, ids, wide_ic, batch, count, 8, context)
        var bcounts = _counts(best, 3, count, trials, context)
        var best32 = _cast(best, 3, 2, count, context)
        var bcount_words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(bcounts)))
        for trial in range(trials):  # driver-loop: over selected candidate model groups
            var selected_count = Int(bcount_words[trial])
            if selected_count == 0:
                continue
            var local = _where(best32, trial, count, selected_count, context)
            var selected = _take(ids, local, count, selected_count, 1, 3, context)
            ref order = grid[trial]
            _ = py_call(models.append, Python.tuple(Python.tuple(order.p, d, order.q),
                         Python.tuple(order.seasonal_p, seasonal_d, order.seasonal_q, order.period), order.intercept))
            _ = py_call(ids_all.append, selected)
            _ = py_call(reuse_all.append, PythonObject(None) if not search_fit else Python.tuple(search_fit.value(), trial, local))
            _ = py_call(fitted.append, PythonObject(None))
        _ = bcounts
    _ = dcounts
    # Commit finished search state only after every native candidate group returns.
    var updates = Python.dict()
    py_setitem(updates, nk[].k_d_, dser)
    py_setitem(updates, nk[].k_models, models)
    py_setitem(updates, nk[].k__ids, ids_all)
    py_setitem(updates, nk[].k__reuse, reuse_all)
    py_setitem(updates, nk[].k__fitted, fitted)
    py_setitem(updates, nk[].k__search_maxiter, maxiter)
    py_setitem(updates, nk[].k__ic, criterion)
    py_setitem(updates, nk[].k__root_check, (fast_mode & 16) != 0)
    py_setitem(updates, nk[].k_order_, order_out)
    py_setitem(updates, nk[].k_ic_, ic_out)
    autoarima_commit(fields, updates, context)
    _ = y
    _ = y_pin^  # the door's pin held endog through every RUN read
    _ = call.owner  # the route owner pinned the image through every search call
    return model
