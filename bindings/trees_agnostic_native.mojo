# SPDX-License-Identifier: Apache-2.0
# Whole KernelExplainer and PermutationExplainer operations. The xtrees
# agnostic kernels (providers/python/xtrees_api_pyapi.mojo, xtrees/agnostic*)
# own every numerical step; this owner admits the model, background, settings
# and rows, evaluates the explained mojolearn estimator through its compiled
# model-selection plan (bindings/prep_native_estimator.mojo; never its Python
# predict), drives the chunk loops and returns the shaped float64 values.
# A Python callable model is refused: native code never calls back into it.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields, _mode
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_attributes import classical_check_layout
from bindings.prep_native_estimator import prep_estimator_admit, prep_estimator_has, prep_estimator_call
from bindings.trees_public_inputs import trees_table, trees_call, trees_view, trees_vector, trees_address, trees_reshape
from bindings.classical_pandas import classical_admit_value
from bindings.trees_public_forest import forest_effective_mode, _engine, _ordered_export, _resident, _family
from bindings.trees_public_boost import _seed
from bindings.trees_explainer_native import _get, _int, _entry, _snapshot, _replace

comptime _I64 = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]

#: xtrees/agnostic.mojo AGN_BUDGET: synthetic words per explainer chunk.
comptime _AGN_BUDGET = 1 << 25
#: MOJOLEARN_IDN_PSHAP_RESIDENT: 2^28 float32 words (1 GiB of synthetic rows
#: on the device), a memory bound that fits every supported GPU.
comptime _AGN_BUDGET_RESIDENT = 1 << 28
#: x_trees_fast_switches bits (providers/python/xtrees_api_pyapi.mojo
#: XTREES_FAST_SWITCHES; python/mojolearn/_expansion_trees.py documents the
#: first three): MOJOLEARN_KSHAP_FAST_BATCH, MOJOLEARN_AGN_IDN_SYN_POOL,
#: MOJOLEARN_PSHAP_DELTA, MOJOLEARN_IDN_SHAP_DEVICE_MODEL (an IDENTICAL GPU
#: build: Kernel/Permutation SHAP over a built-in flat forest evaluate the
#: model on the device, xtrees/agnostic_device.mojo `model_kernel`; moves no
#: bit) and MOJOLEARN_KSHAP/PSHAP_FAST_DEVICE_MODEL (the FAST twins).
comptime _KSHAP_FAST_BATCH = 8
comptime _AGN_IDN_SYN_POOL = 16
comptime _PSHAP_DELTA = 32
comptime _AGN_IDN_DEVICE_MODEL = 64
comptime _KSHAP_FAST_DEVICE_MODEL = 1024
comptime _PSHAP_FAST_DEVICE_MODEL = 2048


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _name(permutation: Bool) -> String:
    return "PermutationExplainer" if permutation else "KernelExplainer"


def _switches(table: PythonObject) raises -> Int:
    var query = _entry(table, "x_trees_fast_switches", False)
    if query is None:
        return 0
    var bits = py_call(query)
    if py_type(bits) is not _b().int:
        raise Error("TypeError: x_trees_fast_switches must return an exact integer")
    return Int(py=bits)


def _call(table: PythonObject, name: String, args: PythonObject) raises -> PythonObject:
    """`entry(*args)` of a connected compiled x_trees entry, keeping the
    class of a refusal it raises."""
    return trees_call(_entry(table, name), args)


def _ints(*values: Int) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: fixed binding parameter words
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _chunk(per_row: Int, n: Int) -> Int:
    """Rows per chunk: as many as keep a chunk's synthetic matrix under the
    budget (xtrees/agnostic.mojo AGN_BUDGET words), at least one."""
    return max(1, min(n, _AGN_BUDGET // max(1, per_row)))


def _class_value(model: PythonObject, name: String) raises -> PythonObject:
    """A class attribute found on the model's MRO namespaces (no descriptor
    or metaclass protocol), or None."""
    var mro = py_call(_b().type.__dict__["__mro__"].__get__, py_type(model))
    for i in range(len(mro)):  # driver-loop: the model class ancestry
        var namespace = _class_fields(mro[i])
        if py_contains(namespace, name):
            return namespace[name]
    return PythonObject(None)


def _method(model: PythonObject, context: PythonObject) raises -> String:
    """The explained method: a classifier's predict_proba, else predict."""
    var kind = _class_value(model, "_estimator_type")
    if (kind is not None and py_type(kind) is _b().str and String(kind) == "classifier"
            and prep_estimator_has(model, "predict_proba", context)):
        return "predict_proba"
    if not prep_estimator_has(model, "predict", context):
        raise Error("TypeError: the explained mojolearn estimator has no native predict")
    return "predict"


def _evaluate(model: PythonObject, method: String, x: PythonObject, n: Int,
              context: PythonObject) raises -> PythonObject:
    """The model on rows x (an (n, d) Array) as a float32 (n, k) Array."""
    var out = prep_estimator_call(model, method, Python.tuple(x), context)
    classical_admit_value(out, context)
    var view = trees_view(out, "output", context)
    var rank = len(view.shape)
    if rank != 1 and rank != 2:
        raise Error("ValueError: the explained model output must be 1-D or 2-D")
    if view.shape[0] != n:
        raise Error("ValueError: the explained model returned a different row count")
    var k = view.shape[1] if rank == 2 else 1
    _ = view
    # A fresh snapshot table per output: a freed output's identity may be reused.
    var flat = _snapshot(out, "<f4", rank, "output", context, Python.dict())
    _ = out
    return trees_reshape(flat, Python.tuple(n, k), context)


def _shape(phi: PythonObject, n: Int, d: Int, k: Int, context: PythonObject) raises -> PythonObject:
    if n == 0:
        var empty = trees_vector(0, "<f8", context)
        return trees_reshape(empty, Python.tuple(0, d) if k == 1 else Python.tuple(0, d, k), context)
    return trees_reshape(phi, Python.tuple(n, d) if k == 1 else Python.tuple(n, d, k), context)


def _device_forest(model: PythonObject, mode: String, permutation: Bool, table: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """MOJOLEARN_IDN_SHAP_DEVICE_MODEL (and its FAST twins): (arrays, k,
    n_trees, rf_input) when the model is a built-in flat forest whose
    predict is the strict increasing-tree device kernel; else None (the
    model runs through its native predict per chunk). rf_input: the random
    forest's predict flushes the compared feature, ExtraTrees' does not."""
    var bits = _switches(table)
    var fast = mode == "fast"
    if fast:
        if (bits & (_PSHAP_FAST_DEVICE_MODEL if permutation else _KSHAP_FAST_DEVICE_MODEL)) == 0:
            return PythonObject(None)
    elif (bits & _AGN_IDN_DEVICE_MODEL) == 0:
        return PythonObject(None)
    var m = model
    var outer = _fields(model)
    var inner = _get(outer, "_random_inner")
    if inner is not None:
        m = inner
    var cls = py_type(m)
    var keys = py_call(_b().list, context["trees_classes"])
    var kind = String("")
    for i in range(len(keys)):  # driver-loop: connected exact tree classes
        if cls is keys[i]:
            kind = String(context["trees_classes"][keys[i]])
    var rf_input = 0
    if kind == "rf_classifier" or kind == "rf_regressor":
        rf_input = 1
    elif kind != "et_classifier" and kind != "et_regressor":
        return PythonObject(None)
    var fields = _fields(m)
    if not py_contains(fields, "_offsets"):
        return PythonObject(None)
    try:
        var forest_mode = forest_effective_mode(fields, context)
        var native = trees_table(context, forest_mode, "_mojolearn_trees" if _family(kind) == "et" else "_mojolearn_rf")
        if fast:
            if forest_mode != "fast" and forest_mode != "identical":
                return PythonObject(None)
        elif forest_mode != "identical" or not (_engine(fields, forest_mode) == "sequential" and _ordered_export(native)):
            return PythonObject(None)
        _ = _resident(m, fields, native, forest_mode, context)
    except:
        # The forest has no resident route here: its native predict serves.
        return PythonObject(None)
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var arrays = Python.list()
    for i in range(5):  # driver-loop: the five flat model arrays
        _ = py_call(arrays.append, fields[names[i]])
    return Python.tuple(py_call(_b().tuple, arrays), _int(fields["_num_outputs"], "output count", True),
                        _int(fields["_n_trees"], "tree count", True), rf_input)


def _model_load(dev: PythonObject, bg: PythonObject, table: PythonObject) raises -> Bool:
    """Uploads the explained forest and the background for one shap_values
    call (`x_trees_agn_model_load`); the caller releases it."""
    if dev is None:
        return False
    var arrays = dev[0]
    var addresses = Python.list()
    for i in range(5):  # driver-loop: one address per model array
        _ = py_call(addresses.append, trees_address(arrays[i]))
    _ = _call(table, "x_trees_agn_model_load", Python.list(addresses, PythonObject(trees_address(bg)),
              _ints(Int(py=dev[2]), Int(py=arrays[1].size), Int(py=dev[1]), Int(py=bg.shape[1]),
                    Int(py=bg.shape[0]), Int(py=dev[3]))))
    _ = arrays
    return True


def _pool_release(table: PythonObject) raises:
    var release = _entry(table, "x_trees_agn_pool_release", False)
    if release is not None:
        _ = py_call(release)


def _construct(fields: PythonObject, args: PythonObject, permutation: Bool,
               context: PythonObject) raises:
    var model = args[0]
    var data = args[1]
    var random_state = args[2]
    var link = args[3]
    var requested = args[4]
    var name = _name(permutation)
    if not permutation:
        if py_type(link) is not _b().str or (String(link) != "identity" and String(link) != "logit"):
            raise Error("NotImplementedError: " + name + " supports the identity and logit links, by name")
    # Only connected mojolearn estimators run here; their predict is a
    # compiled operation. A Python callable or foreign model is refused.
    prep_estimator_admit(model, name, context)
    var seed = _seed(random_state, context)
    _ = seed
    var model_fields = _fields(model)
    if requested is None:
        requested = _get(model_fields, "numeric_mode")
    var mode = _mode(requested if requested is not None else context["backend_fields"]["_DEFAULT_MODE"])
    var table = trees_table(context, mode, "_mojolearn_x_trees")
    var method = _method(model, context)
    var bg = _snapshot(data, "<f4", 2, "data", context, Python.dict())
    var nb = Int(py=bg.shape[0])
    var d = Int(py=bg.shape[1])
    if nb < 1:
        raise Error("ValueError: " + name + " requires at least one background row")
    var out = _evaluate(model, method, bg, nb, context)
    var k = Int(py=out.shape[1])
    var ev = trees_vector(k, "<f8", context)
    _ = _call(table, "x_trees_block_mean", Python.list(PythonObject(trees_address(out)), PythonObject(trees_address(ev)),
              _ints(1, nb, k)))
    _ = out
    if not permutation and String(link) == "logit":
        # shap KernelExplainer: expected_value = link(fnull); the link meets
        # each output after the background mean.
        _ = _call(table, "x_trees_logit", Python.list(PythonObject(trees_address(ev)), _ints(k)))
    var expected = PythonObject(Float64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(ev))[0])) if k == 1 else ev
    var dev = _device_forest(model, mode, permutation, table, context)
    if dev is not None:
        var fitted = _get(model_fields, "n_features_in_")
        if Int(py=dev[1]) != k or fitted is None or py_type(fitted) is not _b().int or Int(py=fitted) != d:
            dev = PythonObject(None)
    var changes = Python.dict()
    var keys = Python.tuple("model", "data", "random_state", "numeric_mode", "_bg", "_fnull", "_method",
                            "_dev", "n_outputs_", "expected_value", "n_features_in_")
    var values = Python.tuple(model, data, random_state, requested, bg, ev,
                              PythonObject(method), dev, PythonObject(k), expected, PythonObject(d))
    for i in range(len(keys)):  # driver-loop: fixed explainer state fields
        py_setitem(changes, keys[i], values[i])
    if not permutation:
        py_setitem(changes, "link", link)
    _replace(fields, changes, context)


struct _State(Copyable, Movable):
    var model: PythonObject
    var method: String
    var bg: PythonObject
    var table: PythonObject
    var mode: String
    var seed: Int
    var nb: Int
    var d: Int
    var k: Int
    var dev: PythonObject

    def __init__(out self, model: PythonObject, method: String, bg: PythonObject, table: PythonObject,
                 mode: String, seed: Int, nb: Int, d: Int, k: Int, dev: PythonObject):
        self.model = model
        self.method = method
        self.bg = bg
        self.table = table
        self.mode = mode
        self.seed = seed
        self.nb = nb
        self.d = d
        self.k = k
        self.dev = dev


def _state(fields: PythonObject, name: String, context: PythonObject) raises -> _State:
    var bg = _get(fields, "_bg")
    var method = _get(fields, "_method")
    if bg is None or method is None:
        raise Error("RuntimeError: " + name + " has not been initialized by the native owner")
    var requested = _get(fields, "numeric_mode")
    var mode = _mode(requested if requested is not None else context["backend_fields"]["_DEFAULT_MODE"])
    var table = trees_table(context, mode, "_mojolearn_x_trees")
    var model = _get(fields, "model")
    prep_estimator_admit(model, name, context)
    return _State(model, String(method), bg, table, mode, _seed(_get(fields, "random_state"), context),
                  Int(py=bg.shape[0]), Int(py=bg.shape[1]), _int(_get(fields, "n_outputs_"), "output count", True),
                  _get(fields, "_dev"))


def _rows(x_source: PythonObject, s: _State, name: String, context: PythonObject) raises -> PythonObject:
    var x = _snapshot(x_source, "<f4", 2, "X", context, Python.dict())
    if Int(py=x.shape[1]) != s.d:
        raise Error("ValueError: X has " + String(Int(py=x.shape[1])) + " features, data has " + String(s.d))
    return x


def _model_rows(s: _State, syn: PythonObject, rows: Int, context: PythonObject) raises -> PythonObject:
    """The model on a chunk's synthetic matrix: float32 (rows, k)."""
    var out = _evaluate(s.model, s.method, trees_reshape(syn, Python.tuple(rows, s.d), context), rows, context)
    if Int(py=out.shape[1]) != s.k:
        raise Error("ValueError: the explained model output width changed after construction")
    return out


def _kernel_values(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var name = String("KernelExplainer")
    var nsamples = args[1]
    var l1 = args[2]
    var b = _b()
    var l1_ok = ((py_type(l1) is b.str and String(l1) == "auto") or (py_type(l1) is b.bool and not Bool(py=l1))
                 or (py_type(l1) is b.int and Int(py=l1) == 0))
    if not l1_ok:
        raise Error("NotImplementedError: KernelExplainer carries no l1 feature selection; l1_reg must be 'auto', False or 0")
    var s = _state(fields, name, context)
    var link = _get(fields, "link")
    var logit = py_type(link) is b.str and String(link) == "logit"
    var x = _rows(args[0], s, name, context)
    var n = Int(py=x.shape[0])
    var d = s.d
    var k = s.k
    var nb = s.nb
    var M = d
    var ns = 0
    if py_type(nsamples) is b.str and String(nsamples) == "auto":
        ns = 2 * M + 2048
    elif py_type(nsamples) is b.int:
        ns = Int(py=nsamples)
    else:
        raise Error("TypeError: nsamples must be an int or 'auto'")
    if M <= 30:
        ns = min(ns, (1 << M) - 2)
    var h = max(1, M // 2)
    var size_off = trees_vector(h + 1, "<i8", context)
    var size_w = trees_vector(h, "<f8", context)
    var cdf = trees_vector(h, "<f8", context)
    var sched = trees_vector(6, "<i8", context)
    if M > 1:
        _ = _call(s.table, "x_trees_kshap_schedule", Python.list(Python.tuple(trees_address(size_off), trees_address(size_w), trees_address(cdf)),
                  PythonObject(trees_address(sched)), _ints(M, max(ns, 0))))
    var sp = _I64(unsafe_from_address=trees_address(sched))
    var m = Int(sp[0])
    var nfixed = Int(sp[1])
    var nfull = Int(sp[2])
    var npaired = Int(sp[3])
    var L = Int(sp[4])
    var wbits = Int(sp[5])
    _ = sched
    var taddr = Python.tuple(trees_address(size_off), trees_address(size_w), trees_address(cdf))
    var phi = trees_vector(max(n * d * k, 1), "<f8", context)
    if n == 0:
        return _shape(phi, 0, d, k, context)
    var fx = _evaluate(s.model, s.method, x, n, context)   # the explained rows, one model call
    if Int(py=fx.shape[1]) != k:
        raise Error("ValueError: the explained model output width changed after construction")
    var x0 = trees_address(x)
    var f0 = trees_address(fx)
    var p0 = trees_address(phi)
    var fnull = _get(fields, "_fnull")
    var nl = trees_address(fnull)
    var bg0 = trees_address(s.bg)
    var link_word = 1 if logit else 0
    var R = _chunk(m * nb * d, n)
    var bits = _switches(s.table)
    var fast = s.mode == "fast"
    if m > 0 and fast and (bits & _KSHAP_FAST_BATCH) != 0 and s.dev is None:
        # MOJOLEARN_KSHAP_FAST_BATCH: one host buffer for every chunk's
        # synthetic rows, each chunk's linked background means into one
        # float64 (n, m, k) block, then the solve over as many rows per call
        # as the budget holds (the same words as the per-chunk solve).
        var ey = trees_vector(n * m * k, "<f8", context)
        var e0 = trees_address(ey)
        var syn = trees_vector(R * m * nb * d, "<f4", context)
        var r0 = 0
        while r0 < n:  # driver-loop: one binding call per row chunk
            var rows = min(R, n - r0)
            if rows < R:
                syn = trees_vector(rows * m * nb * d, "<f4", context)
            _ = _call(s.table, "x_trees_kshap_synth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0), taddr,
                      PythonObject(trees_address(syn)), _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits)))
            var out = _model_rows(s, syn, rows * m * nb, context)
            _ = _call(s.table, "x_trees_kshap_means", Python.list(PythonObject(trees_address(out)),
                      PythonObject(e0 + 8 * r0 * m * k), _ints(rows, nb, m, k, link_word)))
            _ = out
            r0 += rows
        _ = syn
        var S = _chunk(m * d, n)
        var s0 = 0
        while s0 < n:  # driver-loop: solve-batch loop (one binding call per batch)
            var rows = min(S, n - s0)
            _ = _call(s.table, "x_trees_kshap_solve_ey", Python.list(PythonObject(e0 + 8 * s0 * m * k), PythonObject(f0 + 4 * s0 * k),
                      PythonObject(nl), taddr, PythonObject(p0 + 8 * s0 * d * k),
                      _ints(rows, nb, d, nfixed, m, nfull, L, npaired, s0, s.seed, wbits, k, link_word)))
            s0 += rows
        _ = ey
    elif _model_load(s.dev, s.bg, s.table):
        # The forest on the device: masks -> model -> means -> solve per
        # chunk, nothing crossing but the chunk's rows and phi.
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one binding call per row chunk
                var rows = min(R, n - r0)
                _ = _call(s.table, "x_trees_kshap_solve_model", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(f0 + 4 * r0 * k),
                          PythonObject(nl), taddr, PythonObject(p0 + 8 * r0 * d * k),
                          _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits, k, link_word)))
                r0 += rows
        finally:
            _ = _call(s.table, "x_trees_agn_model_release", Python.list())
    else:
        var reuse = (bits & _AGN_IDN_SYN_POOL) != 0   # one synthetic buffer for every chunk
        var syn = PythonObject(None)
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one binding call per row chunk
                var rows = min(R, n - r0)
                var y_addr = 0
                var out = PythonObject(None)
                if m > 0:
                    if not reuse or syn is None or Int(py=syn.size) != rows * m * nb * d:
                        syn = PythonObject(None)
                        syn = trees_vector(rows * m * nb * d, "<f4", context)
                    _ = _call(s.table, "x_trees_kshap_synth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0), taddr,
                              PythonObject(trees_address(syn)),
                              _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits)))
                    out = _model_rows(s, syn, rows * m * nb, context)
                    y_addr = trees_address(out)
                    if not reuse:
                        syn = PythonObject(None)
                _ = _call(s.table, "x_trees_kshap_solve", Python.list(PythonObject(y_addr), PythonObject(f0 + 4 * r0 * k),
                          PythonObject(nl), taddr, PythonObject(p0 + 8 * r0 * d * k),
                          _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits, k, link_word)))
                _ = out
                r0 += rows
        finally:
            if reuse:
                _pool_release(s.table)   # the pooled device buffer does not outlive the explanation
        _ = syn
    _ = x
    _ = fx
    _ = fnull
    _ = size_off
    _ = size_w
    _ = cdf
    return _shape(phi, n, d, k, context)


def _linear_block(s: _State, context: PythonObject) raises -> PythonObject:
    """MOJOLEARN_IDN_PSHAP_RESIDENT: ([w | b] float32 block, link) when the
    binding carries the switch and the explained model's predict is this
    library's linear `link(X W^T + b)`; else None. Byte copies only."""
    var ok = _entry(s.table, "x_trees_pshap_linear_ok", False)
    if ok is None:
        return PythonObject(None)
    var flag = py_call(ok)
    if py_type(flag) is not _b().int or Int(py=flag) != 1:
        return PythonObject(None)
    var linear_predict = py_get(context, "trees_linear_predict")
    if linear_predict is None or _class_value(s.model, "predict") is not linear_predict:
        return PythonObject(None)
    var fields = _fields(s.model)
    var fitted = _get(fields, "n_features_in_")
    if fitted is None or py_type(fitted) is not _b().int or Int(py=fitted) != s.d:
        return PythonObject(None)
    var link = _get(fields, "_link")
    if link is None:
        link = _class_value(s.model, "_link")
    if link is None or py_type(link) is not _b().int:
        return PythonObject(None)
    var coef = _get(fields, "coef_")
    var intercept = _get(fields, "intercept_")
    if coef is None or intercept is None:
        return PythonObject(None)
    var snaps = Python.dict()
    var c_view = trees_view(coef, "coef_", context)
    var c_rank = len(c_view.shape)
    _ = c_view
    var c = _snapshot(coef, "<f4", c_rank, "coef_", context, snaps)
    var bias = PythonObject(None)
    if py_type(intercept) is _b().float or py_type(intercept) is _b().int:
        bias = _snapshot(Python.list(intercept), "<f4", 1, "intercept_", context, snaps)
    else:
        var i_view = trees_view(intercept, "intercept_", context)
        var i_rank = len(i_view.shape)
        _ = i_view
        bias = _snapshot(intercept, "<f4", i_rank, "intercept_", context, snaps)
    var kk = Int(py=bias.size)
    var d = Int(py=c.size) // max(kk, 1)
    if kk != s.k or d != s.d or Int(py=c.size) != kk * d:
        return PythonObject(None)
    var wb = trees_vector(kk * (d + 1), "<f4", context)
    var o = _F32(unsafe_from_address=trees_address(wb))
    var cp = _F32(unsafe_from_address=trees_address(c))
    var bp = _F32(unsafe_from_address=trees_address(bias))
    for j in range(kk):  # driver-loop: over the fitted output rows (model structure)
        for f in range(d):  # driver-loop: one coefficient row copy
            o[j * (d + 1) + f] = cp[j * d + f]
        o[j * (d + 1) + d] = bp[j]
    _ = c
    _ = bias
    return Python.tuple(wb, link)


def _permutation_values(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var name = String("PermutationExplainer")
    var b = _b()
    var s = _state(fields, name, context)
    var x = _rows(args[0], s, name, context)
    var n = Int(py=x.shape[0])
    var d = s.d
    var k = s.k
    var nb = s.nb
    if py_type(args[1]) is not b.int:
        raise Error("TypeError: npermutations must be an int")
    var npm = Int(py=args[1])
    if npm < 1:
        raise Error("ValueError: npermutations must be >= 1")
    var phi = trees_vector(max(n * d * k, 1), "<f8", context)
    if n == 0:
        return _shape(phi, 0, d, k, context)
    var mm = npm * (2 * d + 1)
    var x0 = trees_address(x)
    var p0 = trees_address(phi)
    var bg0 = trees_address(s.bg)
    var R = _chunk(mm * nb * d, n)
    if _model_load(s.dev, s.bg, s.table):
        # The forest on the device: permutations -> model -> means ->
        # marginals per chunk, nothing crossing but the chunk's rows and phi.
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one binding call per row chunk
                var rows = min(R, n - r0)
                _ = _call(s.table, "x_trees_pshap_values_model", Python.list(PythonObject(x0 + 4 * r0 * d),
                          PythonObject(p0 + 8 * r0 * d * k), _ints(rows, nb, d, npm, r0, s.seed, k)))
                r0 += rows
        finally:
            _ = _call(s.table, "x_trees_agn_model_release", Python.list())
        _ = x
        return _shape(phi, n, d, k, context)
    var lin = _linear_block(s, context)
    if lin is not None:
        # IDN_PSHAP_RESIDENT: synthetic rows, the linear predict, means and
        # marginals on the device per chunk; only x rows and phi cross.
        var wb = lin[0]
        var Rl = max(1, min(n, _AGN_BUDGET_RESIDENT // max(1, mm * nb * d)))
        var r0 = 0
        while r0 < n:  # driver-loop: one binding call per row chunk
            var rows = min(Rl, n - r0)
            _ = _call(s.table, "x_trees_pshap_values_linear", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(wb)), PythonObject(p0 + 8 * r0 * d * k),
                      _ints(rows, nb, d, npm, r0, s.seed, k, Int(py=lin[1]))))
            r0 += rows
        _ = wb
        _ = x
        return _shape(phi, n, d, k, context)
    var bits = _switches(s.table)
    var fast = s.mode == "fast"
    var idn_pool = (bits & _AGN_IDN_SYN_POOL) != 0
    var reuse = (fast and (bits & _KSHAP_FAST_BATCH) != 0) or idn_pool
    if fast and (bits & _PSHAP_DELTA) != 0:
        # MOJOLEARN_PSHAP_DELTA: per chunk, the device builds only the
        # synthetic rows that differ from their previous coalition's into the
        # front of one reused buffer (their count into `tot`), the model runs
        # on that prefix, and the device expands the outputs back to every
        # (coalition, background row). Three Int64 words per (coalition,
        # background row): bound them as if d were at least 16.
        var Rd = min(R, _chunk(mm * nb * max(d, 16), n))
        var tot = trees_vector(1, "<i8", context)
        var t0 = trees_address(tot)
        var syn = trees_vector(Rd * mm * nb * d, "<f4", context)
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(Rd, n - r0)
            _ = _call(s.table, "x_trees_pshap_dsynth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(syn)), PythonObject(t0), _ints(rows, nb, d, npm, r0, s.seed)))
            var nv = Int(_I64(unsafe_from_address=t0)[0])
            if nv < 0 or nv > rows * mm * nb:
                raise Error("RuntimeError: x_trees_pshap_dsynth returned an invalid row count")
            # The first nv rows of syn, copied once into the model's input.
            var front = trees_vector(nv * d, "<f4", context)
            var src = _F32(unsafe_from_address=trees_address(syn))
            var dst = _F32(unsafe_from_address=trees_address(front))
            for i in range(nv * d):  # driver-loop: contiguous prefix byte copy
                dst[i] = src[i]
            var out = _model_rows(s, front, nv, context)
            _ = front
            _ = _call(s.table, "x_trees_pshap_dvalues", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(out)), PythonObject(p0 + 8 * r0 * d * k), PythonObject(t0),
                      _ints(rows, nb, d, npm, r0, s.seed, k)))
            _ = out
            r0 += rows
        _ = syn
        _ = tot
        _ = x
        return _shape(phi, n, d, k, context)
    var syn = PythonObject(None)
    try:
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(R, n - r0)
            if not reuse or syn is None or Int(py=syn.size) != rows * mm * nb * d:
                syn = PythonObject(None)
                syn = trees_vector(rows * mm * nb * d, "<f4", context)
            _ = _call(s.table, "x_trees_pshap_synth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(syn)), _ints(rows, nb, d, npm, r0, s.seed)))
            var out = _model_rows(s, syn, rows * mm * nb, context)
            if not reuse:
                syn = PythonObject(None)
            _ = _call(s.table, "x_trees_pshap_values", Python.list(PythonObject(trees_address(out)),
                      PythonObject(p0 + 8 * r0 * d * k), _ints(rows, nb, d, npm, r0, s.seed, k)))
            _ = out
            r0 += rows
    finally:
        if idn_pool:
            _pool_release(s.table)   # the pooled device buffer does not outlive the explanation
    _ = syn
    _ = x
    return _shape(phi, n, d, k, context)


def trees_agnostic_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    classical_check_layout(model, context)
    var fields = _fields(model)
    if operation == "trees_agnostic_kernel_init":
        _construct(fields, args, False, context)
        return PythonObject(None)
    if operation == "trees_agnostic_permutation_init":
        _construct(fields, args, True, context)
        return PythonObject(None)
    if operation == "trees_agnostic_kernel_values":
        return _kernel_values(fields, args, context)
    if operation == "trees_agnostic_permutation_values":
        return _permutation_values(fields, args, context)
    raise Error("unknown model-agnostic explainer operation " + operation)
