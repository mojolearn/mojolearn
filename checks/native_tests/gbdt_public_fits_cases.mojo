# SPDX-License-Identifier: Apache-2.0
"""GradientBoosting fits through the native fit, without the Python binding.

Ported from the Python tests (lane tmig-trees, 2026-11-14; the Python
functions are deleted in the same commit series):

- `test_gbdt_multiclass_stochastic.py` (whole file),
- `test_gbdt_fast_binary_classes.py` (whole file; the `identical` arm: the
  `fast` arm is Apple-only and was skipped on every other install),
- `test_gbdt_release_parity.py::test_heldout_cursor_starts_at_zero`,
  `::test_explicit_average_matches_default_for_supported_losses`,
- `test_gbdt_query_rmse.py::test_queries_of_one_row_train_nothing`,
- `test_gbdt_pair_logit.py::test_learns_on_uneven_queries`,
  `::test_explicit_unit_pairs_equal_the_generated_pairs`,
  `::test_pairs_weight_changes_the_model`,
- `test_gbdt_ranking_bootstrap.py` (whole file),
- `test_gbdt_yeti_rank.py::test_same_seed_same_bytes_other_seed_other_bytes`,
- the host-only units (the CPU suite runs them): the `*_fits_on_the_host_when_built`
  functions of `test_cpu_training_e3_gbdt.py`, `test_cpu_training_e3_gbdt_trees.py`,
  `test_cpu_training_gbdt_losses.py` and `test_cpu_training_gbdt_ordered.py`.

A case builds the fit request the public `GradientBoosting.fit` builds
(`bindings/trees_public_gbdt_fit_door.mojo` `_params` and `gbdt_fit`: the
35 slots, the class weights, the tails and the four option strings) from the
original test's options and the constructor's defaults (`gbdt_initialize`:
1000 / 100 iterations, the Bayesian bootstrap on SymmetricTree, the score
function, the automatic learning rate from CatBoost's GPU table through
`gbdt_auto_rate`, the search strength through `gbdt_validate_search_options`),
and calls the fit image `I` (the host fit in the CPU suite, the device fit in
the GPU suite). Predictions are the host model-text reader
(`gbdt/host/gbdt_host_entries_impl.mojo`), the reader the public predict
uses on the host column; the GPU suite reads the device-fitted text with it.
Inputs are the original NumPy streams, stored by `tools/gbdt_native_fixtures.py`
under `checks/native_tests/fixtures/gbdt/` (column-major X). Every option,
assertion and tolerance is the original test's.
"""
from std.math import isfinite, log, min, max
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_helpers_impl import argmax_rows_f32, argmax_rows_f64
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome, GBDT_ARG_NO_INT
from gbdt.gbdt_entry import GbdtImage
from gbdt.trees_gbdt_fit_run import gbdt_auto_rate, GbdtSearchFacts, gbdt_validate_search_options
from gbdt.mojolearn_gbdt_host_run import (
    gbdt_predict_multi_run,
    gbdt_fit_ordered_rmse_run,
    gbdt_fit_two_level_feature_freq_run,
)
from gbdt.host.gbdt_host_entries_impl import (
    _parse_model,
    gbdt_host_predict_into,
    gbdt_host_sigmoid_pair_into,
    gbdt_host_binary_margins_check,
    gbdt_host_binary_prediction_into,
)

comptime FIXTURE_DIR = "checks/native_tests/fixtures/gbdt/"
comptime UNSET = -1.0
#: units every column runs (CPU and GPU suites)
comptime GBDT_COMMON_UNITS = 24
#: units only the host column runs (the original `*_on_the_host_when_built` tests)
comptime GBDT_HOST_UNITS = 20

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


# ----------------------------------------------------------------- fixtures


def _read_words(name: String, count: Int) raises -> List[UInt32]:
    var values = List[UInt32](length=count, fill=UInt32(0))
    with open(FIXTURE_DIR + name, "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * count, "gbdt fixture extent: " + name)
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def word(index: Int) {imm bp, imm vp}:
            var bits = UInt32(0)
            for offset in range(4):  # Fixed-width little-endian word decode.
                bits |= UInt32(bp[index * 4 + offset]) << UInt32(offset * 8)
            vp.unsafe_store(index, bits)
        host_parallelize(word, count)
        _ = bytes^
    return values^


def load_f32(name: String, count: Int) raises -> List[Float32]:
    var words = _read_words(name, count)
    var values = List[Float32](length=count, fill=Float32(0))
    for i in range(count):  # small-loop(one fixture's words, inside one unit's task)
        values[i] = bitcast[DType.float32](words[i])
    return values^


def load_u32(name: String, count: Int) raises -> List[UInt32]:
    return _read_words(name, count)


# ------------------------------------------------------------------ options


struct GbdtOpts(Copyable, Movable):
    """`GradientBoosting.__init__`'s arguments; -1 / "" is None."""
    var loss: String
    var n_estimators: Int
    var max_depth: Int
    var learning_rate: Float64
    var l2_leaf_reg: Float64
    var border_count: Int
    var random_state: Int
    var loss_variance_power: Float64
    var leaf_method: String
    var leaf_iterations: Int
    var bootstrap: String
    var bagging_temperature: Float64
    var subsample: Float64
    var cat_features: List[Int]
    var one_hot_features: List[Int]
    var od_type: String
    var od_wait: Int
    var use_best_model: Int
    var best_model_min_trees: Int
    var score_function: String
    var nan_mode: String
    var random_strength: Float64
    var pointwise: Bool
    var boost_from_average: Int
    var class_weights: List[Float64]
    var grow_policy: String
    var max_leaves: Int
    var min_data_in_leaf: Int
    var min_split_gain: Float64
    var min_child_hessian: Float64
    var feature_fraction: Float64

    def __init__(out self, loss: String, n_estimators: Int, max_depth: Int = 6):
        self.loss = loss
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = UNSET
        self.l2_leaf_reg = UNSET
        self.border_count = 128
        self.random_state = 0
        self.loss_variance_power = UNSET
        self.leaf_method = ""
        self.leaf_iterations = -1
        self.bootstrap = ""
        self.bagging_temperature = 1.0
        self.subsample = UNSET
        self.cat_features = List[Int]()
        self.one_hot_features = List[Int]()
        self.od_type = ""
        self.od_wait = -1
        self.use_best_model = -1
        self.best_model_min_trees = 1
        self.score_function = ""
        self.nan_mode = "Min"
        self.random_strength = UNSET
        self.pointwise = False
        self.boost_from_average = -1
        self.class_weights = List[Float64]()
        self.grow_policy = "SymmetricTree"
        self.max_leaves = -1
        self.min_data_in_leaf = 1
        self.min_split_gain = UNSET
        self.min_child_hessian = UNSET
        self.feature_fraction = 1.0

    def deterministic(mut self):
        """The tests' pinned configuration: `bootstrap_type='No'`, `random_strength=0`."""
        self.bootstrap = "No"
        self.random_strength = 0.0


struct GbdtInputs(Movable):
    """One fit's arrays (column-major X); empty lists are the absent inputs."""
    var x: List[Float32]
    var n: Int
    var d: Int
    var y: List[Float32]
    var weights: List[Float32]
    var eval_x: List[Float32]
    var eval_y: List[Float32]
    var group_sizes: List[UInt32]
    var pairs: List[UInt32]
    var pair_weights: List[Float32]

    def __init__(out self, var x: List[Float32], n: Int, d: Int, var y: List[Float32]):
        self.x = x^
        self.n = n
        self.d = d
        self.y = y^
        self.weights = List[Float32]()
        self.eval_x = List[Float32]()
        self.eval_y = List[Float32]()
        self.group_sizes = List[UInt32]()
        self.pairs = List[UInt32]()
        self.pair_weights = List[Float32]()


struct _Slots(Movable):
    """The params list's two views and flags, as `gbdt_fit_params_py` reads
    a Python int (both views) and a Python float (no integer view)."""
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]

    def __init__(out self):
        self.ints = List[Int]()
        self.floats = List[Float64]()
        self.bad = List[UInt8]()

    def i(mut self, value: Int):
        self.ints.append(value)
        self.floats.append(Float64(value))
        self.bad.append(UInt8(0))

    def f(mut self, value: Float64):
        self.ints.append(0)
        self.floats.append(value)
        self.bad.append(UInt8(GBDT_ARG_NO_INT))


def _symmetric(o: GbdtOpts) -> Bool:
    return o.grow_policy == "SymmetricTree"


def resolved_estimators(o: GbdtOpts) -> Int:
    if o.n_estimators >= 0:
        return o.n_estimators
    return 1000 if _symmetric(o) else 100


def _bootstrap(o: GbdtOpts) -> String:
    if o.bootstrap != "":
        return o.bootstrap
    return "Bayesian" if _symmetric(o) else ""


def _score(o: GbdtOpts) -> String:
    if o.score_function != "":
        return o.score_function
    return "NewtonL2" if o.grow_policy == "Lossguide" else "Cosine"


def _score_code(name: String) raises -> Int:
    if name == "Cosine":
        return 1
    if name == "NewtonL2":
        return 2
    if name == "NewtonCosine":
        return 3
    if name == "L2":
        return 6
    raise Error("score_function " + name)


def _method_code(name: String) raises -> Int:
    if name == "":
        return -1
    if name == "Gradient":
        return 0
    if name == "Newton":
        return 1
    if name == "Exact":
        return 2
    if name == "Simple":
        return 3
    raise Error("leaf_estimation_method " + name)


def _grow_code(name: String) raises -> Int:
    if name == "SymmetricTree":
        return 0
    if name == "Depthwise":
        return 1
    if name == "Lossguide":
        return 2
    raise Error("grow_policy " + name)


def _gpu_rate_row(loss: String, use_best: Bool, bfa: Bool) -> List[Float64]:
    """`ensemble._GPU_AUTO_LEARNING_RATE`; empty where CatBoost has no row."""
    if loss == "Logloss":
        if use_best and bfa:
            return [0.04, -3.226, -0.488, 0.758]
        if bfa:
            return [0.427, -7.316, -0.907, 2.354]
        if use_best:
            return [-0.085, -2.055, -0.414, 0.427]
        return [-0.055, -3.01, -0.896, 2.366]
    if loss == "MultiClass":
        if bfa:
            return List[Float64]()
        if use_best:
            return [0.101, -2.95, -0.437, 1.136]
        return [0.204, -4.144, -0.833, 2.889]
    if loss == "RMSE":
        if use_best and bfa:
            return [0.108, -3.525, -0.285, 0.058]
        if bfa:
            return [0.131, -4.114, -0.597, 1.693]
        if use_best:
            return [0.051, -3.001, -0.449, 0.859]
        return [0.047, -3.034, -0.591, 1.554]
    return List[Float64]()


def resolved_learning_rate(o: GbdtOpts, n_rows: Int, n_eval: Int, eval_constant: Bool) raises -> Float64:
    """`_resolved_learning_rate` of the fit door."""
    if o.learning_rate >= 0.0:
        return o.learning_rate
    if not _symmetric(o):
        return 0.03
    if o.l2_leaf_reg >= 0.0 or o.leaf_method != "" or o.leaf_iterations >= 0:
        return 0.03
    var use_best = (n_eval > 0 and not eval_constant) if o.use_best_model < 0 else o.use_best_model == 1
    var bfa_loss = o.loss == "RMSE" or o.loss == "MAE" or o.loss == "Quantile" or o.loss == "MAPE"
    var bfa = bfa_loss if o.boost_from_average < 0 else o.boost_from_average == 1
    var row = _gpu_rate_row(o.loss, use_best, bfa)
    if len(row) == 0:
        return 0.03
    return gbdt_auto_rate(row[0], row[1], row[2], row[3], Float64(n_rows), Float64(resolved_estimators(o)))


def _strength(o: GbdtOpts) raises -> Float64:
    var score = _score(o)
    var facts = GbdtSearchFacts(
        o.pointwise, _symmetric(o), String(""),
        score == "L2" or score == "NewtonL2", String(""),
        o.random_strength < 0.0, o.random_strength, String(""),
        1.0, 0.0)
    return gbdt_validate_search_options(facts)


def _eval_constant(values: List[Float32]) -> Bool:
    if len(values) == 0:
        return False
    var lo = values[0]
    var hi = values[0]
    for v in values:  # small-loop(eval targets of one fit, inside one unit's task)
        lo = min(lo, v)
        hi = max(hi, v)
    return lo == hi


def fit[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs) raises -> GbdtFitOutcome:
    """`GradientBoosting.fit` over the fit image `I`, column-major X."""
    var n = inp.n
    var d = inp.d
    var n_eval = len(inp.eval_y)
    var constant = _eval_constant(inp.eval_y)
    var estimators = resolved_estimators(o)
    var iters = o.leaf_iterations
    if iters < 0 and _symmetric(o) and estimators < 200 and d < 20:
        iters = 1
    var l2 = o.l2_leaf_reg
    if l2 < 0.0:
        l2 = 0.0 if o.loss == "YetiRank" else 3.0
    var has_flags = len(o.cat_features) > 0 or len(o.one_hot_features) > 0
    var flags = List[UInt32](length=d if has_flags else 1, fill=UInt32(0))
    for f in o.cat_features:  # small-loop(the cat_features argument)
        flags[f] = flags[f] | UInt32(1)
    for f in o.one_hot_features:  # small-loop(the one_hot_features argument)
        flags[f] = flags[f] | UInt32(2)
    var s = _Slots()
    s.i(n)
    s.i(d)
    s.i(len(inp.weights))
    s.i(d if has_flags else 0)
    s.i(o.border_count)
    s.i(estimators)
    s.i(o.max_depth)
    s.f(resolved_learning_rate(o, n, n_eval, constant))
    s.f(l2)
    s.i(o.random_state)
    s.i(_score_code(_score(o)))
    s.f(UNSET)  # loss_alpha
    s.f(UNSET)  # loss_q
    s.f(UNSET)  # loss_delta
    s.f(o.loss_variance_power)
    s.f(UNSET)  # loss_border
    s.i(iters)
    s.i(_method_code(o.leaf_method))
    s.f(o.bagging_temperature)
    s.f(o.subsample)
    s.i(n_eval)
    s.f(UNSET)  # od_pvalue
    s.i(o.od_wait)
    s.i(o.use_best_model)
    s.i(o.best_model_min_trees)
    s.f(_strength(o))
    s.i(1 if o.pointwise else 0)
    s.i(200000)  # border_build_max_samples
    s.i(-1)  # permutation_count
    s.i(-1)  # ctr_estimation_permutation_id
    s.i(o.boost_from_average)
    s.i(_grow_code(o.grow_policy))
    s.i(o.max_leaves)
    s.i(o.min_data_in_leaf)
    s.i(len(o.class_weights))
    for w in o.class_weights:  # small-loop(the class_weights argument)
        s.f(w)
    var tail = 0
    if o.feature_fraction != 1.0:
        s.f(o.min_split_gain)
        s.f(o.min_child_hessian)
        s.f(o.feature_fraction)
        tail = 3
    elif o.min_child_hessian >= 0.0:
        s.f(o.min_split_gain)
        s.f(o.min_child_hessian)
        tail = 2
    elif o.min_split_gain >= 0.0:
        s.f(o.min_split_gain)
        tail = 1
    var unit_weights = List[Float32]()
    if len(inp.group_sizes) > 0:
        # `_padded_tail`: the three tail defaults, so the group words land in place
        if tail < 1:
            s.f(-1.0)
        if tail < 2:
            s.f(-1.0)
        if tail < 3:
            s.f(1.0)
        s.i(Int(inp.group_sizes.unsafe_ptr()))
        s.i(len(inp.group_sizes))
        if o.loss == "PairLogit":
            if len(inp.pairs) == 0:
                s.i(Int(inp.group_sizes.unsafe_ptr()))
                s.i(-1)
                s.i(Int(inp.group_sizes.unsafe_ptr()))
            else:
                var n_pairs = len(inp.pairs) // 2
                var pw = Int(inp.pair_weights.unsafe_ptr())
                if len(inp.pair_weights) == 0:
                    unit_weights = List[Float32](length=n_pairs, fill=Float32(1.0))
                    pw = Int(unit_weights.unsafe_ptr())
                s.i(Int(inp.pairs.unsafe_ptr()))
                s.i(n_pairs)
                s.i(pw)
    var x_addr = Int(inp.x.unsafe_ptr())
    var y_addr = Int(inp.y.unsafe_ptr())
    var w_addr = Int(inp.weights.unsafe_ptr()) if len(inp.weights) > 0 else x_addr
    var ex_addr = Int(inp.eval_x.unsafe_ptr()) if n_eval > 0 else x_addr
    var ey_addr = Int(inp.eval_y.unsafe_ptr()) if n_eval > 0 else y_addr
    var strs: List[String] = [o.loss, _bootstrap(o), o.od_type, o.nan_mode]
    var args = GbdtFitArgs(x_addr, y_addr, w_addr, Int(flags.unsafe_ptr()), ex_addr, ey_addr,
                           s.ints^, s.floats^, s.bad^, strs^)
    var outcome = I.fit(False, args)
    _ = args^
    _ = flags^
    _ = unit_weights^
    _ = inp
    return outcome^


# ---------------------------------------------------------------- prediction


def model_dim(text: String) raises -> Int:
    return _parse_model(text).dim


def predict(text: String, x: List[Float32], n: Int) raises -> List[Float32]:
    """`GradientBoosting.predict` on the host reader: raw float32, row-major
    `[row * dim + k]`."""
    var dim = model_dim(text)
    var out = List[Float32](length=n * dim, fill=Float32(0))
    var op = _F32P(unsafe_from_address=Int(out.unsafe_ptr()))
    if dim == 1:
        var m = _parse_model(text)
        var nf = len(m.fold_counts)
        var nt = len(m.tree_offsets) - 1
        gbdt_host_predict_into(m^, Int(x.unsafe_ptr()), n, nf, nt, op)
    else:
        var width = gbdt_predict_multi_run(text, Int(x.unsafe_ptr()), n, 0, op)
        require(width == dim, "predict wrote width " + String(width))
    _ = x
    return out^


def predict_proba(text: String, loss: String, x: List[Float32], n: Int) raises -> List[Float64]:
    """`GradientBoosting.predict_proba`: softmax (MultiClass, dim + 1
    columns), sigmoids (MultiClassOneVsAll), the float64 `[1 - p, p]` pair
    (Logloss). Float32 results are carried exactly in float64."""
    if loss == "MultiClass" or loss == "MultiClassOneVsAll":
        var dim = model_dim(text)
        var width = dim + 1 if loss == "MultiClass" else dim
        var p32 = List[Float32](length=n * width, fill=Float32(0))
        var wrote = gbdt_predict_multi_run(text, Int(x.unsafe_ptr()), n, 1 if loss == "MultiClass" else 2,
                                           _F32P(unsafe_from_address=Int(p32.unsafe_ptr())))
        require(wrote == width, "predict_proba wrote width " + String(wrote))
        var out = List[Float64](length=n * width, fill=0.0)
        for i in range(n * width):  # small-loop(one prediction block, inside one unit's task)
            out[i] = Float64(p32[i])
        _ = x
        return out^
    require(loss == "Logloss", "predict_proba on " + loss)
    var raw32 = predict(text, x, n)
    var raw = List[Float64](length=n, fill=0.0)
    for i in range(n):  # small-loop(one prediction block, inside one unit's task)
        raw[i] = Float64(raw32[i])
    var out = List[Float64](length=2 * n, fill=0.0)
    gbdt_host_sigmoid_pair_into(_F64P(unsafe_from_address=Int(raw.unsafe_ptr())),
                                _F64P(unsafe_from_address=Int(out.unsafe_ptr())), n)
    _ = raw^
    return out^


def same_f32(a: List[Float32], b: List[Float32]) -> Bool:
    """`a.tobytes() == b.tobytes()`."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(one prediction block, inside one unit's task)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def same_f64(a: List[Float64], b: List[Float64]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(one block, inside one unit's task)
        if bitcast[DType.uint64](a[i]) != bitcast[DType.uint64](b[i]):
            return False
    return True


def all_finite_f32(a: List[Float32]) -> Bool:
    for v in a:  # small-loop(one block, inside one unit's task)
        if not isfinite(v):
            return False
    return True


def all_finite_f64(a: List[Float64]) -> Bool:
    for v in a:  # small-loop(one block, inside one unit's task)
        if not isfinite(v):
            return False
    return True


def any_nonzero(a: List[Float32]) -> Bool:
    for v in a:  # small-loop(one block, inside one unit's task)
        if v != Float32(0):
            return True
    return False


def column(x: List[Float32], n: Int, j: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(one fixture column, inside one unit's task)
        out[i] = x[j * n + i]
    return out^


def require_fails_with(err: String, needle: String, what: String) raises:
    require(needle in err, what + ": " + err)


# ------------------------------------------------------------ common units


def _fixed(name: String, n: Int, d: Int, yname: String) raises -> GbdtInputs:
    return GbdtInputs(load_f32(name, n * d), n, d, load_f32(yname, n))


def multiclass_stochastic[I: GbdtImage](loss: String) raises:
    """`test_multiclass_stochastic_fits_are_repeatable_and_bootstrap_changes_models`, one loss."""
    var inp = _fixed("ms-x.f32", 240, 6, "ms-y.f32")
    var heldout = load_f32("ms-heldout.f32", 37 * 6)
    var boots: List[String] = ["Bayesian", "Bernoulli", "Poisson", "No"]
    var zero_models = List[String]()
    for b in range(4):  # small-loop(the four bootstrap types of the original grid)
        for s in range(2):  # small-loop(the two random strengths of the original grid)
            var o = GbdtOpts(loss, 4, 3)
            o.bootstrap = boots[b]
            o.random_strength = Float64(s)
            o.class_weights = [1.0, 2.0, 0.5]
            if boots[b] == "Bernoulli" or boots[b] == "Poisson":
                o.subsample = 0.66
            var first = fit[I](o, inp)
            var second = fit[I](o, inp)
            require(first.text == second.text, "multiclass " + loss + " " + boots[b] + " model_ differs")
            var p = predict_proba(first.text, loss, heldout, 37)
            require(len(p) == 37 * 3 and all_finite_f64(p), "multiclass proba shape/finite")
            for v in p:  # small-loop(one proba block)
                require(v >= 0.0 and v <= 1.0, "multiclass proba outside [0, 1]")
            require(same_f64(p, predict_proba(second.text, loss, heldout, 37)), "multiclass proba repeat")
            require(same_f32(predict(first.text, heldout, 37), predict(second.text, heldout, 37)),
                    "multiclass predict repeat")
            if loss == "MultiClass":
                for r in range(37):  # small-loop(heldout rows)
                    var total = p[3 * r] + p[3 * r + 1] + p[3 * r + 2]
                    require(abs(total - 1.0) <= 1e-6 + 1e-7, "softmax rows sum to 1")
            if s == 0:
                zero_models.append(first.text)
    for a in range(4):  # small-loop(the four strength-zero models)
        for b in range(a + 1, 4):  # small-loop(the four strength-zero models)
            require(zero_models[a] != zero_models[b], "bootstrap types gave the same model: " + loss)
    _ = heldout^


def fast_binary_classes[I: GbdtImage](k: Int) raises:
    """`test_predict_classes_is_the_argmax_of_predict_proba`, `identical`."""
    var loss = String("Logloss") if k == 2 else String("MultiClass")
    var inp = _fixed("fb-x.f32", 40, 4, "fb-y" + String(k) + ".f32")
    var o = GbdtOpts(loss, 4, 2)
    o.deterministic()
    var model = fit[I](o, inp)
    var proba = predict_proba(model.text, loss, inp.x, 40)
    require(len(proba) == 40 * k, "proba shape (40, k)")
    # predict_classes under identical: the argmax of predict_proba through
    # the package's row argmax (float64 pairs for Logloss, float32 softmax)
    var classes = List[Int64](length=40, fill=Int64(-1))
    if k == 2:
        argmax_rows_f64(_F64P(unsafe_from_address=Int(proba.unsafe_ptr())), 40, 2,
                        _I64P(unsafe_from_address=Int(classes.unsafe_ptr())))
    else:
        var p32 = List[Float32](length=40 * k, fill=Float32(0))
        for i in range(40 * k):  # small-loop(one proba block)
            p32[i] = Float32(proba[i])
        argmax_rows_f32(_F32P(unsafe_from_address=Int(p32.unsafe_ptr())), 40, k,
                        _I64P(unsafe_from_address=Int(classes.unsafe_ptr())))
        _ = p32^
    for r in range(40):  # small-loop(rows)
        var best = 0
        for c in range(1, k):  # small-loop(classes)
            if proba[r * k + c] > proba[r * k + best]:
                best = c
        require(classes[r] == Int64(best), "predict_classes is not the first argmax of predict_proba")


def heldout_cursor[I: GbdtImage]() raises:
    """`test_heldout_cursor_starts_at_zero`."""
    var inp = _fixed("rp-x.f32", 600, 6, "rp-x2.f32")
    _ = fit[I](GbdtOpts("RMSE", 3), inp)  # an unrelated fit first
    inp.y = load_f32("rp-y.f32", 600)
    inp.eval_x = inp.x.copy()
    inp.eval_y = inp.y.copy()
    var o = GbdtOpts("Logloss", 6, 3)
    o.learning_rate = 0.1
    o.deterministic()
    o.boost_from_average = 0
    o.use_best_model = 0
    var model = fit[I](o, inp)
    require(len(model.test) == len(model.learn), "held-out curve extent")
    for i in range(len(model.learn)):  # small-loop(iterations)
        require(abs(model.test[i] - model.learn[i]) <= 1e-7 + 2e-6 * abs(model.learn[i]),
                "held-out curve differs from the learn curve at " + String(i))


def explicit_average[I: GbdtImage]() raises:
    """`test_explicit_average_matches_default_for_supported_losses`."""
    var inp = _fixed("rp-x.f32", 600, 6, "rp-yavg.f32")
    var losses: List[String] = ["RMSE", "MAE", "Quantile", "MAPE"]
    for loss in losses:  # small-loop(the four losses of the original parametrization)
        var o = GbdtOpts(loss, 3, 3)
        o.deterministic()
        var default = fit[I](o, inp)
        o.boost_from_average = 1
        var explicit = fit[I](o, inp)
        require(same_f32(predict(explicit.text, inp.x, 600), predict(default.text, inp.x, 600)),
                "boost_from_average=True predict differs: " + loss)
        require(same_f64(explicit.learn, default.learn), "boost_from_average=True loss curve differs: " + loss)


def queries_of_one_row[I: GbdtImage]() raises:
    """`test_queries_of_one_row_train_nothing`."""
    var inp = _fixed("qr-x.f32", 256, 4, "qr-rel.f32")
    var o = GbdtOpts("QueryRMSE", 10, 4)
    o.deterministic()
    var no_groups = predict(fit[I](o, inp).text, inp.x, 256)
    inp.group_sizes = List[UInt32](length=256, fill=UInt32(1))
    var one_each = predict(fit[I](o, inp).text, inp.x, 256)
    for v in no_groups:  # small-loop(rows)
        require(v == Float32(0), "QueryRMSE without groups trained something")
    require(same_f32(no_groups, one_each), "one-row queries differ from no groups")


def _pair_model(n_estimators: Int = 8) -> GbdtOpts:
    var o = GbdtOpts("PairLogit", n_estimators, 4)
    o.deterministic()
    o.leaf_iterations = 10
    return o^


def _pair_inputs() raises -> GbdtInputs:
    var inp = _fixed("pl-x.f32", 240, 4, "pl-rel.f32")
    inp.group_sizes = load_u32("pl-sizes.u32", 28)
    return inp^


def pair_logit[I: GbdtImage](kind: Int) raises:
    """`test_learns_on_uneven_queries` (0), `test_explicit_unit_pairs_equal_the_generated_pairs`
    (1), `test_pairs_weight_changes_the_model` (2)."""
    var inp = _pair_inputs()
    var o = _pair_model()
    if kind == 0:
        var m = fit[I](o, inp)
        var pred = predict(m.text, inp.x, 240)
        require(len(pred) == 240 and all_finite_f32(pred), "PairLogit predictions")
        require(m.learn[len(m.learn) - 1] < m.learn[0], "PairLogit loss did not fall")
        require(abs(m.learn[0] - log(Float64(2.0))) < 0.1, "a pair's loss starts at log 2")
        return
    var generated = predict(fit[I](o, inp).text, inp.x, 240)
    inp.pairs = load_u32("pl-pairs.u32", 2 * 1894)
    var explicit = predict(fit[I](o, inp).text, inp.x, 240)
    if kind == 1:
        for i in range(240):  # small-loop(rows)
            require(abs(generated[i] - explicit[i]) <= Float32(1e-6), "explicit pairs differ from generated")
        return
    inp.pair_weights = load_f32("pl-pair-weights.f32", 1894)
    var weighted = predict(fit[I](o, inp).text, inp.x, 240)
    require(not same_f32(explicit, weighted), "pairs_weight did not change the model")


def _rank_inputs() raises -> GbdtInputs:
    var inp = _fixed("rb-x.f32", 96, 4, "rb-y.f32")
    inp.group_sizes = List[UInt32](length=12, fill=UInt32(8))
    return inp^


def _rank_opts(loss: String) -> GbdtOpts:
    var o = GbdtOpts(loss, 5, 3)
    o.learning_rate = 0.1
    o.random_state = 19
    return o^


def _rank_loss(i: Int) -> String:
    return "QueryRMSE" if i == 0 else ("PairLogit" if i == 1 else "YetiRank")


def ranking_defaults[I: GbdtImage](loss: String) raises:
    """`test_ranking_defaults_train_and_reproduce`."""
    var inp = _rank_inputs()
    var o = _rank_opts(loss)
    var a = fit[I](o, inp)
    var b = fit[I](o, inp)
    var pa = predict(a.text, inp.x, 96)
    var pb = predict(b.text, inp.x, 96)
    require(all_finite_f32(pa) and all_finite_f64(a.learn), "ranking defaults finite: " + loss)
    require(any_nonzero(pa), "ranking defaults trained nothing: " + loss)
    require(same_f32(pa, pb), "ranking defaults predictions repeat: " + loss)
    require(same_f64(a.learn, b.learn), "ranking defaults curve repeats: " + loss)


def ranking_temperature[I: GbdtImage](loss: String) raises:
    """`test_ranking_bootstrap_temperature_zero_and_nonzero`."""
    var inp = _rank_inputs()
    var plain = _rank_opts(loss)
    plain.deterministic()
    var zero = _rank_opts(loss)
    zero.bagging_temperature = 0.0
    zero.random_strength = 0.0
    var sampled = _rank_opts(loss)
    sampled.bagging_temperature = 1.0
    sampled.random_strength = 0.0
    var pp = predict(fit[I](plain, inp).text, inp.x, 96)
    require(same_f32(pp, predict(fit[I](zero, inp).text, inp.x, 96)), "temperature 0 is not the plain model: " + loss)
    require(not same_f32(predict(fit[I](sampled, inp).text, inp.x, 96), pp), "temperature 1 is the plain model: " + loss)


def ranking_discrete[I: GbdtImage](loss: String, bootstrap: String) raises:
    """`test_ranking_discrete_bootstrap`."""
    var inp = _rank_inputs()
    var o = _rank_opts(loss)
    o.bootstrap = bootstrap
    o.subsample = 0.7
    var pred = predict(fit[I](o, inp).text, inp.x, 96)
    require(all_finite_f32(pred) and any_nonzero(pred), "ranking " + bootstrap + " bootstrap: " + loss)


def yeti_seed[I: GbdtImage]() raises:
    """`test_same_seed_same_bytes_other_seed_other_bytes` (YetiRank)."""
    var inp = _pair_inputs()
    var o = GbdtOpts("YetiRank", 10, 4)
    o.deterministic()
    var a = predict(fit[I](o, inp).text, inp.x, 240)
    var b = predict(fit[I](o, inp).text, inp.x, 240)
    o.random_state = 7
    var c = predict(fit[I](o, inp).text, inp.x, 240)
    require(same_f32(a, b), "YetiRank same seed differs")
    require(not same_f32(a, c), "YetiRank other seed is the same model")


def common_unit[I: GbdtImage](unit: Int) raises:
    if unit == 0:
        multiclass_stochastic[I]("MultiClass")
    elif unit == 1:
        multiclass_stochastic[I]("MultiClassOneVsAll")
    elif unit <= 4:
        fast_binary_classes[I](unit)  # k = 2, 3, 4
    elif unit == 5:
        heldout_cursor[I]()
    elif unit == 6:
        explicit_average[I]()
    elif unit == 7:
        queries_of_one_row[I]()
    elif unit <= 10:
        pair_logit[I](unit - 8)
    elif unit <= 13:
        ranking_defaults[I](_rank_loss(unit - 11))
    elif unit <= 16:
        ranking_temperature[I](_rank_loss(unit - 14))
    elif unit <= 22:
        ranking_discrete[I](_rank_loss((unit - 17) // 2), "Bernoulli" if (unit - 17) % 2 == 0 else "Poisson")
    else:
        yeti_seed[I]()


# -------------------------------------------------------------- host units


def _twice[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, what: String) raises -> GbdtFitOutcome:
    """Two fits: the same model text and the same `predict(X)` bytes."""
    var a = fit[I](o, inp)
    var b = fit[I](o, inp)
    require(a.text == b.text, "two host fits returned different bytes: " + what)
    require(same_f32(predict(a.text, inp.x, inp.n), predict(b.text, inp.x, inp.n)),
            "two host fits predicted different bytes: " + what)
    return a^


def _refuses[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, what: String) raises:
    var refused = False
    try:
        _ = fit[I](o, inp)
    except error:
        require_fails_with(String(error), "no CPU implementation of", what)
        refused = True
    require(refused, what + " did not refuse on the host binding")


def _e3() raises -> GbdtInputs:
    return _fixed("e3-x.f32", 600, 6, "e3-yc.f32")


def host_unit[I: GbdtImage](unit: Int) raises:
    if unit == 0:
        # test_gradient_boosting_fits_on_the_host_when_built: Logloss twice, predict_proba
        var inp = _e3()
        var m = _twice[I](GbdtOpts("Logloss", 3, 3), inp, "Logloss")
        var proba = predict_proba(m.text, "Logloss", inp.x, 600)
        require(len(proba) == 1200, "proba shape (600, 2)")
        for v in proba:  # small-loop(one proba block)
            require(v >= 0.0 and v <= 1.0, "Logloss proba outside [0, 1]")
    elif unit <= 2:
        # the NaN modes
        var inp = GbdtInputs(load_f32("e3-xn.f32", 3600), 600, 6, load_f32("e3-yc.f32", 600))
        var o = GbdtOpts("Logloss", 3, 3)
        o.nan_mode = "Min" if unit == 1 else "Max"
        _ = _twice[I](o, inp, "nan_mode=" + o.nan_mode)
    elif unit == 3:
        # the classifier adapter's transforms (GradientBoostingClassifier: Logloss, same defaults)
        var inp = _e3()
        var m = fit[I](GbdtOpts("Logloss", 3, 3), inp)
        var margins = predict(m.text, inp.x, 600)
        var mp = _F32P(unsafe_from_address=Int(margins.unsafe_ptr()))
        gbdt_host_binary_margins_check(mp, 600)
        var pp = List[Float32](length=1200, fill=Float32(-1))
        gbdt_host_binary_prediction_into[True, DType.float32](
            mp, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(pp.unsafe_ptr())), 600)
        var labels = List[Int32](length=600, fill=Int32(-1))
        gbdt_host_binary_prediction_into[False, DType.int32](
            mp, MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(labels.unsafe_ptr())), 600)
        for v in pp:  # small-loop(one proba block)
            require(v >= Float32(0) and v <= Float32(1), "adapter float32 proba")
        for r in range(600):  # small-loop(rows)
            require(labels[r] == (Int32(1) if margins[r] > Float32(0) else Int32(0)), "strict raw > 0")
        _ = margins^
    elif unit == 4:
        # the two by-name refusals
        var nan = GbdtInputs(load_f32("e3-xn.f32", 3600), 600, 6, load_f32("e3-x0.f32", 600))
        _refuses[I](GbdtOpts("RMSE", 2, 3), nan, "an RMSE fit on an X carrying NaN")
        var o = GbdtOpts("Logloss", 2, 3)
        o.grow_policy = "Lossguide"
        o.score_function = "Cosine"
        _refuses[I](o, _e3(), "a Lossguide Cosine fit")
    elif unit <= 6:
        # test_non_symmetric_fits_on_the_host_when_built (its save/load stays in Python)
        var o = GbdtOpts("Logloss", 3)
        if unit == 5:
            o.max_depth = 3
            o.grow_policy = "Depthwise"
        else:
            o.max_leaves = 8
            o.grow_policy = "Lossguide"
        var m = _twice[I](o, _e3(), o.grow_policy)
        require("\nntree 0 " in m.text, "non-symmetric model text: " + o.grow_policy)
    elif unit <= 13:
        host_losses[I](unit - 7)
    else:
        host_ordered[I](unit - 14)


def host_losses[I: GbdtImage](case: Int) raises:
    """`test_the_five_configurations_fit_on_the_host_when_built`."""
    var reg = GbdtInputs(load_f32("e3-x.f32", 3600), 600, 6, load_f32("e3-yr.f32", 600))
    if case == 0:
        var losses: List[String] = ["Quantile", "MAPE", "Poisson", "Tweedie"]
        for loss in losses:  # small-loop(the four pointwise losses of the original)
            var o = GbdtOpts(loss, 3, 3)
            if loss == "Tweedie":
                o.loss_variance_power = 1.5
            _ = _twice[I](o, reg, loss)
    elif case == 1:
        var o = GbdtOpts("MAE", 3, 3)
        o.leaf_method = "Exact"
        o.bootstrap = "Poisson"
        o.subsample = 0.6
        _ = _twice[I](o, reg, "MAE Exact Poisson")
    elif case == 2:
        var o = GbdtOpts("Logloss", 3)
        o.max_leaves = 8
        o.grow_policy = "Lossguide"
        o.score_function = "NewtonCosine"
        o.min_child_hessian = 1.0
        o.min_split_gain = 0.01
        o.min_data_in_leaf = 8
        o.feature_fraction = 0.5
        o.random_strength = 1.0
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
        o.leaf_method = "Gradient"
        o.leaf_iterations = 3
        _ = _twice[I](o, _e3(), "Lossguide NewtonCosine")
    elif case <= 4:
        var loss = String("MultiClass") if case == 3 else String("MultiClassOneVsAll")
        var three = GbdtInputs(load_f32("e3-x.f32", 3600), 600, 6, load_f32("e3-y3.f32", 600))
        var o = GbdtOpts(loss, 3, 3)
        if case == 3:
            o.class_weights = [1.0, 2.0, 0.5]
        var m = _twice[I](o, three, loss)
        require(len(predict_proba(m.text, loss, three.x, 600)) == 600 * 3, "proba shape (600, 3): " + loss)
    elif case == 5:
        var o = GbdtOpts("Quantile", 2, 3)
        o.bootstrap = "Bayesian"
        _ = _twice[I](o, reg, "Quantile Bayesian")
    else:
        var o = GbdtOpts("Quantile", 2, 3)
        o.grow_policy = "Depthwise"
        _refuses[I](o, reg, "a Depthwise Quantile fit")


def _trees_record(text: String) raises -> Int:
    """`re.search(r"^trees (\\d+)$", text, re.M)`."""
    for raw in text.split("\n"):  # small-loop(model text records)
        var line = String(raw)
        if line.startswith("trees "):
            return Int(String(line.removeprefix("trees ")))
    raise Error("no trees record")


def host_ordered[I: GbdtImage](case: Int) raises:
    """`test_ordered_and_feature_freq_fit_on_the_host_when_built`, its native
    arms (the save/load round trip and the CTR model's predictions, read by
    the host CTR door, stay in Python)."""
    var x = load_f32("od-x.f32", 7200)
    var y = load_f32("od-y.f32", 1200)
    if case == 0:
        # OrderedRMSE(n_estimators=3, max_depth=3), twice; weighted refusal
        var perm = load_u32("od-perm.u32", 1200)
        var texts = List[String]()
        for _ in range(2):  # small-loop(the original's two fits)
            texts.append(gbdt_fit_ordered_rmse_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(x.unsafe_ptr()),
                Int(perm.unsafe_ptr()), 1200, 6, 0, 1200, 3, 3, 128, Float32(0.03), Float32(3.0)))
        require(texts[0] == texts[1], "two ordered host fits returned different bytes")
        require(same_f32(predict(texts[0], x, 1200), predict(texts[1], x, 1200)), "ordered predictions differ")
        var ones = List[Float32](length=1200, fill=Float32(1))
        var refused = False
        try:
            _ = gbdt_fit_ordered_rmse_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(ones.unsafe_ptr()),
                Int(perm.unsafe_ptr()), 1200, 6, 1200, 1200, 2, 2, 128, Float32(0.03), Float32(3.0))
        except error:
            require_fails_with(String(error), "no CPU implementation of", "weighted OrderedRMSE")
            refused = True
        require(refused, "a weighted OrderedRMSE fit did not refuse on the host binding")
        _ = ones^
        _ = perm^
        return
    var coded = load_f32("od-coded.f32", 7200)
    var yc = load_f32("od-yc.f32", 1200)
    if case == 1:
        # ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7), twice; weighted refusal
        var sources: List[UInt32] = [0, 1]
        var sp = _U32P(unsafe_from_address=Int(sources.unsafe_ptr()))
        var a = gbdt_fit_two_level_feature_freq_run(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), sp,
                                                    1200, 6, 0, 2, Float32(0.03), Float32(3.0))
        var b = gbdt_fit_two_level_feature_freq_run(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), sp,
                                                    1200, 6, 0, 2, Float32(0.03), Float32(3.0))
        require(a == b, "two FeatureFreq host fits returned different bytes")
        require(same_f32(predict(a, coded, 1200), predict(b, coded, 1200)), "FeatureFreq predictions differ")
        var refused = False
        try:
            _ = gbdt_fit_two_level_feature_freq_run(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), sp,
                                                    1200, 6, 1200, 2, Float32(0.03), Float32(3.0))
        except error:
            require_fails_with(String(error), "no CPU implementation of", "weighted FeatureFreq")
            refused = True
        require(refused, "a weighted FeatureFreq fit did not refuse on the host binding")
        _ = sources^
        return
    if case == 2:
        # the pointwise eval arm, and its by-name refusal without weights
        var inp = GbdtInputs(x.copy(), 1200, 6, yc.copy())
        inp.eval_x = load_f32("od-xe.f32", 1800)
        inp.eval_y = load_f32("od-ye.f32", 300)
        var o = GbdtOpts("Logloss", 4, 3)
        o.score_function = "L2"
        o.pointwise = True
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 0.5
        o.boost_from_average = 1
        o.od_type = "Iter"
        o.od_wait = 2
        o.use_best_model = 1
        _refuses[I](o, inp, "an unweighted pointwise fit")
        inp.weights = load_f32("od-wts.f32", 1200)
        _ = _twice[I](o, inp, "pointwise eval arm")
        return
    var inp = GbdtInputs(coded.copy(), 1200, 6, yc.copy())
    var o = GbdtOpts("Logloss", 3, 3)
    o.deterministic()
    if case == 3:
        # the one-hot arm
        o.cat_features = [0]
        o.one_hot_features = [1]
        var m = _twice[I](o, inp, "one-hot")
        require("type cat" in m.text and "ctr_" not in m.text, "one-hot model text")
        return
    o.cat_features = [1]
    o.n_estimators = 2
    var a = fit[I](o, inp)
    var b = fit[I](o, inp)
    require(a.text == b.text, "two CTR host fits returned different bytes")
    inp.eval_x = load_f32("od-exctr.f32", 1800)
    inp.eval_y = load_f32("od-eyctr.f32", 300)
    if case == 4:
        # CTR eval rows leave the learn model unchanged
        o.use_best_model = 0
        var heldout = fit[I](o, inp)
        require(heldout.text == a.text, "CTR eval rows changed the learn model")
        require(len(heldout.test) == 2 and all_finite_f64(heldout.test), "CTR eval curve")
        return
    # the best model on the CTR eval curve
    o.n_estimators = 4
    o.use_best_model = 1
    o.best_model_min_trees = 2
    var best = fit[I](o, inp)
    require(len(best.test) == 4, "best-model eval curve extent")
    var arg = 1
    for i in range(2, 4):  # small-loop(curve[1:])
        if best.test[i] < best.test[arg]:
            arg = i
    require(_trees_record(best.text) == arg + 1, "best model keeps argmin(curve[1:]) + 2 trees")
