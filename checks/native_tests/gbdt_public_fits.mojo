# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
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
  `::test_pairs_validation` (its unit-`pairs_weight` arm; lane tests-to-mojo-trees, 2026-11-16),
- `test_gbdt_ranking_bootstrap.py` (whole file),
- `test_gbdt_yeti_rank.py::test_same_seed_same_bytes_other_seed_other_bytes`,
  `::test_learns_on_uneven_queries` (lane tests-to-mojo-trees, 2026-11-16),
- the host-only units (the CPU suite runs them): the `*_fits_on_the_host_when_built`
  functions of `test_cpu_training_e3_gbdt.py`, `test_cpu_training_e3_gbdt_trees.py`,
  `test_cpu_training_gbdt_losses.py` and `test_cpu_training_gbdt_ordered.py`.
- `checks/gbdt_feature_fraction_binding.py` (the fit arms; lane py2mojo-C,
  2026-11-22) and `checks/gbdt_querywise_pointwise_check.py` (the identical
  arms): common units 26-36, inputs stored by
  `tools/py2mojo_native_fixtures_b4.py` under `fixtures/gbdt_ff/` and
  `fixtures/gbdt_qw/`. What stays in Python is named beside each unit.

A case builds the fit request the public `GradientBoosting.fit` builds
(`bindings/trees_public_gbdt_fit_door.mojo` `_params` and `gbdt_fit`: the
36 slots, the class weights, the tails and the four option strings) from the
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
from std.math import isfinite, log, log2, min, max
from std.math import nan as math_nan
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
    gbdt_fit_native_run,
)
from gbdt.host.gbdt_host_entries_impl import (
    _parse_model,
    gbdt_host_model_dim,
    gbdt_host_predict_ctr_into,
    gbdt_host_text_has_ctr,
    gbdt_host_predict_into,
    gbdt_host_sigmoid_pair_into,
    gbdt_host_binary_margins_check,
    gbdt_host_binary_prediction_into,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gbdt.host.gbdt_oracle import GBDT_ORACLE_HOST_SABOTAGE
from gbdt.mojolearn_gbdt_run import gbdt_fit_native_run as gbdt_fit_native_run_gpu
from std.testing import TestSuite
from std.time import perf_counter_ns
from checks.native_tests.testsuite_cases import check_cases, require
from std.sys.compile import is_defined


comptime FIXTURE_DIR = "checks/native_tests/fixtures/gbdt/"
comptime UNSET = -1.0
#: units every column runs (CPU and GPU suites); the last is `fit_loop_`
comptime GBDT_COMMON_UNITS = 26 + FF_UNITS + QW_UNITS + 1
#: units only the host column runs (the original `*_on_the_host_when_built` tests)
comptime GBDT_HOST_UNITS = 55

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]


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
    #: "Plain" (the four option strings) or "Ordered" (the eight:
    #: GreedyLogSum borders, `fold_len_multiplier` 2.0 as its float64 bits,
    #: `fold_permutation_block` unset), as `gbdt_fit_options` builds them
    var boosting_type: String
    #: lane gbdt-cpu-rest-4-pwl (2026-11-23): the loss parameters, the
    #: detector's p-value (UNSET is None) and `feature_border_type` ("" is
    #: the four option strings, GreedyLogSum; a name is the fifth string)
    var loss_alpha: Float64
    var loss_q: Float64
    var loss_delta: Float64
    var od_pvalue: Float64
    var border_type: String
    #: the largest feature combination (lane gbdt-cpu-rest-4-cat,
    #: 2026-11-23); 1 builds none, as every unit before it fitted
    var max_ctr_complexity: Int
    #: MultiRMSE's target dimension (lane gbdt-pw-multiclass, 2026-11-23):
    #: above 1 the request carries the +4 tail and `y` holds that many
    #: dim-major planes, as the public fit builds it for a 2-D target
    var target_dim: Int

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
        self.boosting_type = "Plain"
        self.loss_alpha = UNSET
        self.loss_q = UNSET
        self.loss_delta = UNSET
        self.od_pvalue = UNSET
        self.border_type = ""
        self.max_ctr_complexity = 1
        self.target_dim = 1

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

    def request(deinit self, x: Int, y: Int, weights: Int, flags: Int, eval_x: Int, eval_y: Int,
                var strs: List[String]) raises -> GbdtFitArgs:
        return GbdtFitArgs(x, y, weights, flags, eval_x, eval_y, self.ints^, self.floats^, self.bad^, strs^)


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
    # the reprs the public door passes (`_repr(grow)`, `_repr(score)`,
    # `str(float(random_strength))`), so a refusal reads as the public fit's
    # (lane gbdt-units-11: these were empty, and the refusal printed
    # "grow_policy= requires ...")
    var facts = GbdtSearchFacts(
        o.pointwise, _symmetric(o), "'" + o.grow_policy + "'",
        score == "L2" or score == "NewtonL2", "'" + score + "'",
        o.random_strength < 0.0, o.random_strength, String(o.random_strength),
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
    s.f(o.loss_alpha)
    s.f(o.loss_q)
    s.f(o.loss_delta)
    s.f(o.loss_variance_power)
    s.f(UNSET)  # loss_border
    s.i(iters)
    s.i(_method_code(o.leaf_method))
    s.f(o.bagging_temperature)
    s.f(o.subsample)
    s.i(n_eval)
    s.f(o.od_pvalue)
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
    s.i(o.max_ctr_complexity)  # 1: no feature combinations
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
    if o.target_dim > 1:
        # the target-dimension tail: the three tail defaults, then the
        # dimension (`bindings/_mojolearn_gbdt.mojo`'s +4 tail)
        if tail < 1:
            s.f(-1.0)
        if tail < 2:
            s.f(-1.0)
        if tail < 3:
            s.f(1.0)
        s.i(o.target_dim)
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
    if o.boosting_type != "Ordered" and o.border_type.byte_length() > 0:
        strs.append(o.border_type)
    if o.boosting_type == "Ordered":
        # `gbdt_fit_options`' Ordered tail (bindings/gbdt_config_run.mojo)
        strs.append(String("GreedyLogSum"))
        strs.append(String("Ordered"))
        strs.append(String(bitcast[DType.uint64](Float64(2.0))))
        strs.append(String(0))
    var args = s^.request(x_addr, y_addr, w_addr, Int(flags.unsafe_ptr()), ex_addr, ey_addr, strs^)
    var outcome = I.fit(False, args)
    _ = args^
    _ = flags^
    _ = unit_weights^
    _ = inp
    return outcome^


# ---------------------------------------------------------------- prediction


def model_dim(text: String) raises -> Int:
    return gbdt_host_model_dim(text)


def predict(text: String, x: List[Float32], n: Int) raises -> List[Float32]:
    """`GradientBoosting.predict` on the host reader: raw float32, row-major
    `[row * dim + k]`."""
    var dim = model_dim(text)
    var out = List[Float32](length=n * dim, fill=Float32(0))
    var op = _F32P(unsafe_from_address=Int(out.unsafe_ptr()))
    if gbdt_host_text_has_ctr(text):
        # a CTR model reads the RAW columns (the public predict's input) and
        # expands them on the host (lane gbdt-units-6)
        _ = gbdt_host_predict_ctr_into(text, Int(x.unsafe_ptr()), n, 0, op)
    elif dim == 1:
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
    (1), `test_pairs_weight_changes_the_model` (2), `test_pairs_validation`'s
    "no pairs_weight is unit weights" (3: two trees, explicit pairs)."""
    var inp = _pair_inputs()
    if kind == 3:
        var o2 = _pair_model(2)
        inp.pairs = load_u32("pl-pairs.u32", 2 * 1894)
        var base = predict(fit[I](o2, inp).text, inp.x, 240)
        inp.pair_weights = List[Float32](length=1894, fill=Float32(1))
        var unit = predict(fit[I](o2, inp).text, inp.x, 240)
        require(same_f32(base, unit), "unit pairs_weight changed the PairLogit model")
        return
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


def _ndcg(pred: List[Float32], rel: List[Float32], sizes: List[UInt32]) -> Float64:
    """The Python test's `_ndcg`: mean per-query NDCG, Base type, all
    positions; rows ranked by descending prediction, ties lower grade first
    (a stable sort, as Python's `sorted`). Queries are the consecutive runs of
    `sizes` (the fixture's group ids ascend)."""
    var total = 0.0
    var begin = 0
    for q in range(len(sizes)):  # small-loop(queries of one fit, inside one unit's task)
        var size = Int(sizes[q])
        var order = List[Int](length=size, fill=0)
        var ideal_rel = List[Float64](length=size, fill=0.0)
        for i in range(size):  # small-loop(rows of one query)
            order[i] = i
            ideal_rel[i] = Float64(rel[begin + i])
        for i in range(1, size):  # small-loop(stable insertion sort of one query)
            var key = order[i]
            var kp = Float64(pred[begin + key])
            var kt = Float64(rel[begin + key])
            var j = i - 1
            while j >= 0:
                var op = Float64(pred[begin + order[j]])
                var ot = Float64(rel[begin + order[j]])
                if op < kp or (op == kp and ot > kt):
                    order[j + 1] = order[j]
                    j -= 1
                else:
                    break
            order[j + 1] = key
        for i in range(1, size):  # small-loop(descending sort of one query's grades)
            var v = ideal_rel[i]
            var j = i - 1
            while j >= 0 and ideal_rel[j] < v:
                ideal_rel[j + 1] = ideal_rel[j]
                j -= 1
            ideal_rel[j + 1] = v
        var dcg = 0.0
        var ideal = 0.0
        for k in range(size):  # small-loop(positions of one query)
            var decay = 1.0 / log2(Float64(k) + 2.0)
            dcg += Float64(rel[begin + order[k]]) * decay
            ideal += ideal_rel[k] * decay
        total += dcg / ideal if ideal > 0.0 else 1.0
        begin += size
    return total / Float64(len(sizes))


def yeti_learns[I: GbdtImage]() raises:
    """`test_gbdt_yeti_rank.py::test_learns_on_uneven_queries`: YetiRank(10
    trees, depth 4, bootstrap No, random_strength 0) on the PairLogit
    fixture's uneven queries predicts 240 finite values, writes no loss
    value (every `loss_curve_` entry is 0), and ranks better than a constant."""
    var inp = _pair_inputs()
    var o = GbdtOpts("YetiRank", 10, 4)
    o.deterministic()
    var m = fit[I](o, inp)
    var pred = predict(m.text, inp.x, 240)
    require(len(pred) == 240 and all_finite_f32(pred), "YetiRank predictions")
    for v in m.learn:  # small-loop(the learn curve of one fit)
        require(v == 0.0, "YetiRank wrote a loss value")
    var zeros = List[Float32](length=240, fill=Float32(0))
    require(_ndcg(pred, inp.y, inp.group_sizes) > _ndcg(zeros, inp.y, inp.group_sizes),
            "YetiRank ranks no better than a constant")


# ---- checks/gbdt_feature_fraction_binding.py (lane py2mojo-C, 2026-11-22) ----
# The `identical` arm (the `fast` arm is Apple-only). Kept in Python: the
# "explicit feature_fraction=1. is the default" arm (the door sends no tail
# for 1.0, so both requests are the same words: a door decision, not a fit
# result), the pickle fingerprints and the two adapters (public boundary).
# Both columns fit every arm (lane gbdt-cpu-rest, 2026-11-23: the host
# samples each tree's features on the symmetric loop and the non-symmetric
# driver, gbdt/host/gbdt_oracle.mojo `_sample_tree_folds`), as the original
# did.
comptime FF_DIR = "../gbdt_ff/"
comptime FF_UNITS = 7


def _ff_inputs(logloss: Bool) raises -> GbdtInputs:
    return GbdtInputs(load_f32(FF_DIR + "ff-x.f32", 256 * 8), 256, 8,
                      load_f32(FF_DIR + ("ff-yb.f32" if logloss else "ff-y.f32"), 256))


def _ff_opts(policy: String, loss: String) -> GbdtOpts:
    """`options(mode, policy, loss)` of the original."""
    var o = GbdtOpts(loss, 3, 2)
    o.learning_rate = 0.25
    o.border_count = 8
    o.random_state = 17
    o.grow_policy = policy
    o.max_leaves = 4 if policy == "Lossguide" else -1
    o.bootstrap = "Bayesian"
    o.bagging_temperature = 0.0
    o.random_strength = 0.0
    return o^


def feature_fraction[I: GbdtImage](unit: Int) raises:
    """Units 0-5: policy x loss: a fraction of 0.5 and of the least positive
    float64 fits twice to the same model text and predictions, changes the
    default model and predicts finite values. Unit 6: the counted class
    weights, the three tails, the eval set and the sample weights in one
    Lossguide Logloss fit: finite predictions and a test loss curve."""
    if unit == 6:
        var inp = _ff_inputs(True)
        var o = _ff_opts("Lossguide", "Logloss")
        o.score_function = "NewtonL2"
        o.class_weights = [1.0, 2.0]
        o.min_split_gain = 0.0
        o.min_child_hessian = 0.0
        o.feature_fraction = 0.5
        inp.weights = List[Float32](length=256, fill=Float32(0))
        for r in range(256):  # small-loop(np.linspace(.5, 1.5, 256, dtype=float32))
            inp.weights[r] = Float32(0.5 + Float64(r) / 255.0)
        for j in range(8):  # small-loop(eval_set=(x[:32], target[:32]), column-major)
            for r in range(32):  # small-loop(the 32 eval rows)
                inp.eval_x.append(inp.x[j * 256 + r])
        for r in range(32):  # small-loop(the 32 eval targets)
            inp.eval_y.append(inp.y[r])
        # both columns fit it: the host binding trains an eval set and sample
        # weights on Lossguide since lane gbdt-gaps (2026-11-23), through the
        # bodies the device calls, so the host row asserts what the device
        # row does (it refused by name until then), and fits twice to the
        # same bytes
        var m = _twice[I](o, inp, "weighted Lossguide eval-set fit")
        require(all_finite_f32(predict(m.text, inp.x, 256)), "weighted feature_fraction fit: nonfinite predict")
        require(len(m.test) > 0, "weighted feature_fraction fit: no test loss curve")
        return
    var policies: List[String] = ["SymmetricTree", "Depthwise", "Lossguide"]
    var policy = policies[unit // 2]
    var loss = String("RMSE") if unit % 2 == 0 else String("Logloss")
    var key = policy + "/" + loss
    var inp = _ff_inputs(loss == "Logloss")
    var default = fit[I](_ff_opts(policy, loss), inp)
    var fractions: List[Float64] = [0.5, bitcast[DType.float64](UInt64(1))]
    for fraction in fractions:  # small-loop(the two fractions of the original)
        var o = _ff_opts(policy, loss)
        o.feature_fraction = fraction
        var what = key + " feature_fraction " + String(fraction)
        var first = _twice[I](o, inp, what)
        require(first.text != default.text, what + ": feature_fraction had no model effect")
        require(all_finite_f32(predict(first.text, inp.x, 256)), what + ": nonfinite predict")


# ---- checks/gbdt_querywise_pointwise_check.py (lane py2mojo-C, 2026-11-22) ----
# The grouped losses under use_pointwise_searcher, identical mode. Kept in
# Python: the npz/vendor records and the cross-vendor comparison (the
# release verification), and the NDCG it only records.
comptime QW_DIR = "../gbdt_qw/"
comptime QW_N = 344
comptime QW_D = 7
comptime QW_PAIRS = 2866
comptime QW_UNITS = 4


def _qw_inputs() raises -> GbdtInputs:
    var inp = GbdtInputs(load_f32(QW_DIR + "qw-x.f32", QW_N * QW_D), QW_N, QW_D, load_f32(QW_DIR + "qw-y.f32", QW_N))
    inp.group_sizes = load_u32(QW_DIR + "qw-sizes.u32", 40)
    return inp^


def _qw_opts(loss: String) -> GbdtOpts:
    """`fit(loss, tag)`'s parameters: 5 trees, depth 3, rate 0.1, seed 19,
    the pointwise searcher, Plain boosting, no bootstrap, no score noise."""
    var o = GbdtOpts(loss, 5, 3)
    o.learning_rate = 0.1
    o.random_state = 19
    o.pointwise = True
    o.bootstrap = "No"
    o.random_strength = 0.0
    return o^


struct _QwFit(Movable):
    var pred: List[Float32]
    var curve: List[Float64]

    def __init__(out self, var pred: List[Float32], var curve: List[Float64]):
        self.pred = pred^
        self.curve = curve^


def _qw_fit[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, tag: String) raises -> _QwFit:
    """One `fit`: finite predictions and loss curve, not constant."""
    var m = fit[I](o, inp)
    var pred = predict(m.text, inp.x, inp.n)
    var what = o.loss + "-" + tag
    require(all_finite_f32(pred) and all_finite_f64(m.learn), what + ": nonfinite output")
    var lo = pred[0]
    var hi = pred[0]
    for v in pred:  # small-loop(np.ptp over one prediction block)
        lo = min(lo, v)
        hi = max(hi, v)
    require(hi != lo, what + ": constant predictions")
    return _QwFit(pred^, m.learn.copy())


def querywise[I: GbdtImage](unit: Int) raises:
    """Units 0-2: QueryRMSE, PairLogit, YetiRank: the repeat, the improving
    curve (not YetiRank), Bayesian temperature 0 is the plain model, 0.7 and
    score noise and other group boundaries are not, NewtonCosine fits.
    Unit 3: PairLogit explicit pairs, unit and weighted, differ; the Simple
    leaf method is refused by name."""
    var inp = _qw_inputs()
    if unit == 3:
        var o = _qw_opts("PairLogit")
        inp.pairs = load_u32(QW_DIR + "qw-pairs.u32", 2 * QW_PAIRS)
        var unit_pred = _qw_fit[I](o, inp, "explicit-unit").pred.copy()
        inp.pair_weights = load_f32(QW_DIR + "qw-pair-weights.f32", QW_PAIRS)
        var weighted = _qw_fit[I](o, inp, "explicit-weighted").pred.copy()
        require(not same_f32(unit_pred, weighted), "explicit pair weights ignored")
        inp.pairs = List[UInt32]()
        inp.pair_weights = List[Float32]()
        # Both columns (lane bm-trees, 2026-11-22): the host and device fits admit the leaf method of the
        # grouped pointwise search in one place, gbdt/options/catboost_options.mojo
        # (lane gbdt-cpu-rest-4-simple, 2026-11-23: Simple is `simple_leaves_check`'s, before any
        # tree, so its words are that body's; `querywise_pointwise_leaf_check` keeps Exact).
        var simple = _qw_opts("PairLogit")
        simple.leaf_method = "Simple"
        var refused = False
        try:
            _ = fit[I](simple, inp)
        except error:
            require_fails_with(String(error), "use_pointwise_searcher searches the structure only", "querywise pointwise Simple")
            refused = True
        require(refused, "querywise pointwise Simple was silently accepted")
        return
    var loss = _rank_loss(unit)
    var o = _qw_opts(loss)
    var base = _qw_fit[I](o, inp, "plain")
    var again = _qw_fit[I](o, inp, "repeat")
    require(same_f32(base.pred, again.pred), loss + ": repeat predictions")
    require(same_f64(base.curve, again.curve), loss + ": repeat curve")
    if loss != "YetiRank":
        require(base.curve[len(base.curve) - 1] < base.curve[0], loss + ": loss did not improve")
    var zero = _qw_opts(loss)
    zero.bootstrap = "Bayesian"
    zero.bagging_temperature = 0.0
    require(same_f32(_qw_fit[I](zero, inp, "bayesian-zero").pred, base.pred), loss + ": zero bootstrap changed model")
    var sampled = _qw_opts(loss)
    sampled.bootstrap = "Bayesian"
    sampled.bagging_temperature = 0.7
    require(not same_f32(_qw_fit[I](sampled, inp, "bayesian").pred, base.pred), loss + ": bootstrap ignored")
    var noisy = _qw_opts(loss)
    noisy.random_strength = 1.0
    require(not same_f32(_qw_fit[I](noisy, inp, "score-noise").pred, base.pred), loss + ": pointwise score noise ignored")
    var newton = _qw_opts(loss)
    newton.score_function = "NewtonCosine"
    _ = _qw_fit[I](newton, inp, "newton-score")
    inp.group_sizes = load_u32(QW_DIR + "qw-alt-sizes.u32", 69)
    require(not same_f32(_qw_fit[I](o, inp, "different-groups").pred, base.pred), loss + ": grouping ignored")


def fit_loop_rounds[I: GbdtImage]() raises:
    """`fit_loop_` (lane ab-loop-mode): with the cap K as `n_estimators` and
    no eval set (no overfitting detector), every boosting loop the public fit
    takes (Plain SymmetricTree, Depthwise, Lossguide, Ordered, MultiClass)
    reports K rounds, as many as the model's trees and its learn curve, and a
    time of at least zero."""
    comptime K = 5
    var reg = _fixed("rp-x.f32", 600, 6, "rp-yavg.f32")
    var multi = _fixed("ms-x.f32", 240, 6, "ms-y.f32")
    var arms: List[String] = ["Plain", "Depthwise", "Lossguide", "Ordered", "MultiClass"]
    for arm in arms:  # small-loop(the five boosting loops)
        var o = GbdtOpts("RMSE", K, 3)
        if arm == "Depthwise" or arm == "Lossguide":
            o = _ns_opts("RMSE", arm, "L2")
        elif arm == "Ordered":
            o = _ordered_opts("RMSE")
        elif arm == "MultiClass":
            o = GbdtOpts("MultiClass", K, 3)
        o.n_estimators = K
        o.deterministic()
        var model: GbdtFitOutcome
        if arm == "MultiClass":
            model = fit[I](o, multi)
        else:
            model = fit[I](o, reg)
        require(model.fit_loop.recorded, "fit_loop_ not recorded: " + arm)
        require(model.fit_loop.iterations == K,
                "fit_loop_ rounds " + String(model.fit_loop.iterations) + " != " + String(K) + ": " + arm)
        if arm != "Depthwise" and arm != "Lossguide":
            # the symmetric texts carry the `trees` record
            require(_trees_record(model.text) == K, "fit_loop_ rounds differ from the model's trees: " + arm)
        require(len(model.learn) == K, "fit_loop_ rounds differ from the learn curve: " + arm)
        require(model.fit_loop.loop_ns >= 0 and model.fit_loop.milliseconds() >= 0.0,
                "fit_loop_ negative time: " + arm)
        require(not model.stopped_early, "a fit without an eval set stopped early: " + arm)


def common_unit[I: GbdtImage](unit: Int) raises:
    if unit == 26 + FF_UNITS + QW_UNITS:
        fit_loop_rounds[I]()
    elif unit >= 26 + FF_UNITS:
        querywise[I](unit - 26 - FF_UNITS)
    elif unit >= 26:
        feature_fraction[I](unit - 26)
    elif unit == 0:
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
    elif unit == 23:
        yeti_seed[I]()
    elif unit == 24:
        pair_logit[I](3)
    else:
        yeti_learns[I]()


# -------------------------------------------------------------- host units


def _twice[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, what: String) raises -> GbdtFitOutcome:
    """Two fits: the same model text and the same `predict(X)` bytes."""
    var a = fit[I](o, inp)
    var b = fit[I](o, inp)
    require(a.text == b.text, "two host fits returned different bytes: " + what)
    require(same_f32(predict(a.text, inp.x, inp.n), predict(b.text, inp.x, inp.n)),
            "two host fits predicted different bytes: " + what)
    return a^


def _twice_text[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, what: String) raises -> GbdtFitOutcome:
    """Two fits of a model with CTR columns: the same model text, learn and
    held-out curves, no `predict(X)` (lane gbdt-cpu-rest-4-cat-ns,
    2026-11-23). Since lane gbdt-units-6 the host predict expands CTR
    columns (`gbdt_host_predict_ctr_into`), so `_twice` takes CTR models
    too; this form stays for the units that compare curves."""
    var a = fit[I](o, inp)
    var b = fit[I](o, inp)
    require(a.text == b.text, "two host fits returned different bytes: " + what)
    require(same_f64(a.learn, b.learn) and same_f64(a.test, b.test),
            "two host fits returned different curves: " + what)
    return a^


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
        # the two former by-name refusals, real fits since lane
        # gbdt-cpu-rest (2026-11-23): NaN binned by the shared grid on every
        # numeric arm, Cosine under Lossguide on the non-symmetric driver
        var nan = GbdtInputs(load_f32("e3-xn.f32", 3600), 600, 6, load_f32("e3-x0.f32", 600))
        var mn = _twice[I](GbdtOpts("RMSE", 2, 3), nan, "an RMSE fit on an X carrying NaN")
        require(all_finite_f32(predict(mn.text, nan.x, 600)), "RMSE NaN fit: nonfinite predict")
        var o = GbdtOpts("Logloss", 2, 3)
        o.grow_policy = "Lossguide"
        o.score_function = "Cosine"
        var mc = _twice[I](o, _e3(), "a Lossguide Cosine fit")
        require("\nntree 0 " in mc.text, "Lossguide Cosine: non-symmetric model text")
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
    elif unit <= 19:
        host_ordered[I](unit - 14)
    elif unit == 20:
        host_rest_gaps[I]()
    elif unit == 21:
        host_pointwise_arms[I](0)
    elif unit == 24:
        host_pointwise_rest[I](0)
    elif unit == 25:
        host_symmetric_rest[I](0)
    elif unit == 26:
        host_nonsym_rest[I](0)
    elif unit == 22:
        host_ordered_weights[I]()
    elif unit == 23:
        host_cat_rest[I]()
    elif unit == 27:
        host_tail_rest[I](0)
    elif unit == 28:
        host_simple_rest[I](0)
    elif unit == 29:
        host_pointwise_multiclass[I](0)
    # lane gbdt-units-7: the heavy units' later parts, each its own unit so
    # the host cores run them side by side (a unit is serial inside: its
    # fits nest under `check_cases`' task, `_gbdt_units`' nested rule)
    elif unit <= 32:
        host_pointwise_arms[I](unit - 29)
    elif unit <= 37:
        host_pointwise_rest[I](unit - 32)
    elif unit <= 43:
        host_symmetric_rest[I](unit - 37)
    elif unit <= 48:
        host_nonsym_rest[I](unit - 43)
    elif unit <= 50:
        host_tail_rest[I](unit - 48)
    elif unit <= 53:
        host_simple_rest[I](unit - 50)
    elif unit == 54:
        host_pointwise_multiclass[I](1)
    else:
        raise Error("gbdt host unit " + String(unit) + " has no body on this branch")


def _cat_rest_inputs(yname: String) raises -> GbdtInputs:
    """e3's six float columns, then three coded columns (lane
    gbdt-cpu-rest-4-cat): 6 holds five categories (a CTR column under
    cat_features, above one_hot_max_size 2), 7 holds two (one-hot under
    cat_features), 8 holds four codes (a plain one_hot_features column).
    Rows 0 to 4 carry every code, so each column is densely coded."""
    var x = load_f32("e3-x.f32", 3600)
    var extra = List[Float32](length=1800, fill=Float32(0))
    for r in range(600):  # small-loop(three appended columns)
        var a = abs(x[r])
        var b = x[600 + r]
        var c5 = Int(a * Float32(7.0)) % 5
        var c2 = 1 if b > Float32(0.0) else 0
        var c4 = Int(abs(b) * Float32(5.0)) % 4
        if r < 5:
            c5 = r
            c2 = r % 2
            c4 = r % 4
        extra[r] = Float32(c5)
        extra[600 + r] = Float32(c2)
        extra[1200 + r] = Float32(c4)
    for r in range(1800):  # small-loop(the appended columns into the fixture)
        x.append(extra[r])
    return GbdtInputs(x^, 600, 9, load_f32(yname, 600))


def _cat_rest_opts(loss: String) -> GbdtOpts:
    """A SymmetricTree fit at rate 0.1 with every work-changing option set."""
    var o = GbdtOpts(loss, 6, 3)
    o.learning_rate = 0.1
    o.l2_leaf_reg = 3.0
    o.random_state = 7
    o.score_function = "Cosine"
    o.deterministic()
    # the host unit's time budget (lane gbdt-units-6): 16 borders still
    # give every policy and path its bins; no unit here reads the default 128
    o.border_count = 16
    return o^


def host_cat_rest[I: GbdtImage]() raises:
    """Categorical and one-hot columns on the SymmetricTree fits the host
    refused by name until lane gbdt-cpu-rest-4-cat (2026-11-23), each fit
    twice to the same bytes: Quantile with a CTR, a one-hot and a plain
    one_hot_features column (four permutations; the columns move the
    model), CrossEntropy with feature combinations (max_ctr_complexity 2),
    MultiClass with CTR and one-hot columns and an eval set under the Iter
    detector, MultiClassOneVsAll with sample weights and feature
    combinations. Then `train`'s refusals in its words: QueryRMSE with
    cat_features, YetiRank with one_hot_features, feature_fraction below 1
    with a one_hot_features column. The categorical fits compare text and
    curves (`_twice_text`: the host reader refuses a model with CTR tables).
    Last, the Depthwise and Lossguide units of sub-lane
    gbdt-cpu-rest-4-cat-ns (`host_cat_rest_ns`)."""
    var reg = _cat_rest_inputs("e3-yr.f32")
    var q = _cat_rest_opts("Quantile")
    q.leaf_method = "Gradient"
    q.leaf_iterations = 1
    var mq0 = _twice[I](q, reg, "symmetric Quantile, numeric columns")
    var qc = q.copy()
    qc.cat_features = [6, 7]
    qc.one_hot_features = [8]
    var mq = _twice_text[I](qc, reg, "symmetric Quantile with CTR and one-hot columns")
    require(mq.text != mq0.text, "symmetric Quantile: the categorical columns had no model effect")
    require("ctr_table " in mq.text, "symmetric Quantile: no CTR table in the model text")
    var clf = _cat_rest_inputs("e3-yc.f32")
    var ce = _cat_rest_opts("CrossEntropy")
    ce.cat_features = [6]
    ce.max_ctr_complexity = 2
    _ = _twice_text[I](ce, clf, "symmetric CrossEntropy with feature combinations")
    var mc3 = _cat_rest_inputs("e3-y3.f32")
    mc3.eval_x = mc3.x.copy()
    mc3.eval_y = mc3.y.copy()
    var mc = _cat_rest_opts("MultiClass")
    mc.n_estimators = 10
    mc.cat_features = [6, 7]
    mc.od_type = "Iter"
    mc.od_wait = 2
    var mm = _twice_text[I](mc, mc3, "MultiClass with CTR and one-hot columns and an eval set")
    require(len(mm.test) > 0, "categorical MultiClass: no held-out curve")
    require("ctr_table " in mm.text, "categorical MultiClass: no CTR table in the model text")
    var ova_in = _cat_rest_inputs("e3-y3.f32")
    ova_in.weights = List[Float32](length=600, fill=Float32(0))
    for r in range(600):  # small-loop(np.linspace(.5, 1.5, 600, dtype=float32))
        ova_in.weights[r] = Float32(0.5 + Float64(r) / 599.0)
    var ova = _cat_rest_opts("MultiClassOneVsAll")
    ova.cat_features = [6]
    ova.one_hot_features = [8]
    ova.max_ctr_complexity = 2
    _ = _twice_text[I](ova, ova_in, "weighted MultiClassOneVsAll with feature combinations")
    # `train`'s refusals, its words, on both columns
    var grouped = _cat_rest_inputs("e3-yr.f32")
    grouped.group_sizes = List[UInt32](length=60, fill=UInt32(10))
    var qr = _cat_rest_opts("QueryRMSE")
    qr.cat_features = [6]
    var refused = String("")
    try:
        _ = fit[I](qr, grouped)
    except error:
        refused = String(error)
    require_fails_with(refused, "with cat_features is not implemented here", "QueryRMSE with cat_features")
    var yr = _cat_rest_opts("YetiRank")
    yr.one_hot_features = [8]
    refused = String("")
    try:
        _ = fit[I](yr, grouped)
    except error:
        refused = String(error)
    require_fails_with(refused, "with one_hot_features is not implemented", "YetiRank with one_hot_features")
    var ff = _cat_rest_opts("RMSE")
    ff.boost_from_average = 0
    ff.leaf_iterations = 1
    ff.one_hot_features = [8]
    ff.feature_fraction = 0.5
    refused = String("")
    try:
        _ = fit[I](ff, reg)
    except error:
        refused = String(error)
    require_fails_with(refused, "one_hot input refused", "feature_fraction with a one_hot_features column")
    host_cat_rest_ns[I]()


def _ns_opts(loss: String, policy: String, score: String) -> GbdtOpts:
    """A Depthwise / Lossguide fit with every option it reads set."""
    var o = GbdtOpts(loss, 4, 3)
    o.deterministic()
    o.learning_rate = 0.1
    o.random_state = 7
    o.grow_policy = policy
    o.score_function = score
    if policy == "Lossguide":
        o.max_leaves = 8
    # the host unit's time budget (lane gbdt-units-6): 16 borders still
    # give every policy and path its bins; no unit here reads the default 128
    o.border_count = 16
    return o^


def _ns_refused[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, needle: String, what: String) raises:
    """`fit` raises, in the device's words."""
    var refused = String("")
    try:
        _ = fit[I](o, inp)
    except error:
        refused = String(error)
    require(refused.byte_length() > 0, what + " was silently accepted")
    require_fails_with(refused, needle, what)
    require(not ("no CPU implementation" in refused), what + ": a CPU-gap refusal: " + refused)


def host_nonsym_rest[I: GbdtImage](part: Int) raises:
    """The Depthwise / Lossguide combinations the host refused by name until
    lane gbdt-cpu-rest-4-ns (2026-11-23), each twice to the same bytes, and
    the ones the device refuses, refused here in its words."""
    var cls = _e3()
    var reg = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    # the four leafwise calcers under both policies
    if part <= 1:
        var scores: List[String] = ["L2", "NewtonL2", "NewtonCosine", "Cosine"]
        var policies: List[String] = ["Depthwise", "Lossguide"]
        for pi in range(part, part + 1):  # small-loop(this part's policy: part 0 Depthwise, 1 Lossguide)
            for si in range(4):  # small-loop(the four scores)
                var o = _ns_opts("Logloss", policies[pi], scores[si])
                var m = _twice[I](o, cls, policies[pi] + " Logloss " + scores[si])
                require("\nntree 0 " in m.text, policies[pi] + " " + scores[si] + ": non-symmetric model text")
    if part == 2:
        var l2 = fit[I](_ns_opts("Logloss", "Depthwise", "L2"), cls)
        var nl2 = fit[I](_ns_opts("Logloss", "Depthwise", "NewtonL2"), cls)
        require(l2.text != nl2.text, "Depthwise L2 and NewtonL2 grew the same model (the planes did not move)")
        # Depthwise's growth knobs: together (each fit twice to the same bytes),
        # then EACH at a value that must bite on 600 rows at depth 3 (lane
        # gbdt-units-6: at 0.01 / 1 / 8 none of the three can stop a split of
        # this tree, whose leaves hold ~75 rows of unit Hessian, so the old
        # comparison could not tell a wired knob from a dropped one):
        # min_data_in_leaf 300 makes a root child terminal (the children sum
        # to 600), min_child_hessian 250 rejects every depth-2 split (the RMSE
        # Newton plane is one per row and no depth-3 child holds 250 rows),
        # min_split_gain 1e30 rejects every split
    if part == 3:
        var knobs = _ns_opts("RMSE", "Depthwise", "NewtonCosine")
        knobs.min_split_gain = 0.01
        knobs.min_child_hessian = 1.0
        knobs.min_data_in_leaf = 8
        _ = _twice[I](knobs, reg, "Depthwise RMSE min_split_gain / min_child_hessian / min_data_in_leaf")
        var plain = fit[I](_ns_opts("RMSE", "Depthwise", "NewtonCosine"), reg)
        var k_leaf = _ns_opts("RMSE", "Depthwise", "NewtonCosine")
        k_leaf.min_data_in_leaf = 300
        require(fit[I](k_leaf, reg).text != plain.text, "Depthwise growth knobs changed nothing (min_data_in_leaf)")
        var k_hess = _ns_opts("RMSE", "Depthwise", "NewtonCosine")
        k_hess.min_child_hessian = 250.0
        require(fit[I](k_hess, reg).text != plain.text, "Depthwise growth knobs changed nothing (min_child_hessian)")
        var k_gain = _ns_opts("RMSE", "Depthwise", "NewtonCosine")
        k_gain.min_split_gain = 1e30
        require(fit[I](k_gain, reg).text != plain.text, "Depthwise growth knobs changed nothing (min_split_gain)")
        # Gradient leaves on Logloss, Poisson bootstrap, random_strength
    if part == 4:
        var gp = _ns_opts("Logloss", "Depthwise", "Cosine")
        gp.leaf_method = "Gradient"
        gp.bootstrap = "Poisson"
        gp.subsample = 0.7
        gp.random_strength = 1.0
        _ = _twice[I](gp, cls, "Depthwise Logloss Gradient leaves, Poisson, random_strength 1")
        # RMSE Gradient leaves at three iterations
        var rg = _ns_opts("RMSE", "Lossguide", "NewtonL2")
        rg.leaf_method = "Gradient"
        rg.leaf_iterations = 3
        rg.bootstrap = "Poisson"
        _ = _twice[I](rg, reg, "Lossguide RMSE Gradient leaves at 3 iterations")
        # boost_from_average on Logloss and CrossEntropy: a bias record
    if part == 5:
        var bl = _ns_opts("Logloss", "Depthwise", "Cosine")
        bl.boost_from_average = 1
        var mbl = _twice[I](bl, cls, "Depthwise Logloss boost_from_average")
        require("\nbias " in mbl.text, "Depthwise Logloss boost_from_average: no bias record")
        var bc = _ns_opts("CrossEntropy", "Lossguide", "NewtonCosine")
        bc.boost_from_average = 1
        var mbc = _twice[I](bc, cls, "Lossguide CrossEntropy boost_from_average (Newton planes)")
        require("\nbias " in mbc.text, "Lossguide CrossEntropy boost_from_average: no bias record")
        # class weights on a pointwise loss
        var cw = _ns_opts("CrossEntropy", "Depthwise", "Cosine")
        cw.class_weights = [1.0, 3.0]
        var mcw = _twice[I](cw, cls, "Depthwise CrossEntropy class weights")
        var ucw = fit[I](_ns_opts("CrossEntropy", "Depthwise", "Cosine"), cls)
        require(mcw.text != ucw.text, "Depthwise CrossEntropy: class weights changed nothing")
        # refused on both columns, in the device's words
        var ex = _ns_opts("Logloss", "Depthwise", "Cosine")
        ex.leaf_method = "Exact"
        _ns_refused[I](ex, cls, "Only MAPE, MAE and Quantile", "Depthwise Logloss Exact")
        var sm = _ns_opts("RMSE", "Lossguide", "NewtonL2")
        sm.leaf_method = "Simple"
        _ns_refused[I](sm, reg, "grow_policy Depthwise / Lossguide searches the structure only", "Lossguide RMSE Simple")
        var mvs = _ns_opts("Logloss", "Lossguide", "NewtonL2")
        mvs.bootstrap = "MVS"
        _ns_refused[I](mvs, cls, "MVS is not reachable", "Lossguide MVS")
        var rcw = _ns_opts("RMSE", "Depthwise", "Cosine")
        rcw.class_weights = [1.0, 2.0]
        _ns_refused[I](rcw, reg, "class weights take effect only", "Depthwise RMSE class weights")
        var mch = _ns_opts("Poisson", "Depthwise", "NewtonL2")
        mch.min_child_hessian = 1.0
        _ns_refused[I](mch, reg, "min_child_hessian currently supports", "Depthwise Poisson min_child_hessian")
        var pb = _ns_opts("Poisson", "Lossguide", "NewtonL2")
        pb.boost_from_average = 1
        _ns_refused[I](pb, reg, "You can use boost_from_average only", "Lossguide Poisson boost_from_average")
        var pws = _ns_opts("Logloss", "Depthwise", "Cosine")
        pws.pointwise = True
        _ns_refused[I](pws, cls, "an OBLIVIOUS searcher", "the pointwise searcher on Depthwise")
        var d0 = _ns_opts("Logloss", "Depthwise", "Cosine")
        d0.max_depth = 0
        _ns_refused[I](d0, cls, "max_leaves must be at least 2 under Depthwise", "Depthwise at depth 0")


def _pw_opts(loss: String, score: String, boot: String) -> GbdtOpts:
    """A pointwise-searcher fit at the given score and bootstrap, every
    option it reads set."""
    var o = GbdtOpts(loss, 4, 3)
    o.learning_rate = 0.1
    o.l2_leaf_reg = 3.0
    o.random_state = 7
    o.pointwise = True
    o.score_function = score
    o.bootstrap = boot
    o.random_strength = 0.0
    o.boost_from_average = 1
    o.leaf_method = "Newton"
    o.leaf_iterations = 1
    # the host unit's time budget (lane gbdt-units-6): 16 borders still
    # give every policy and path its bins; no unit here reads the default 128
    o.border_count = 16
    return o^


def host_pointwise_arms[I: GbdtImage](part: Int) raises:
    """The pointwise searcher's arms the host refused by name until lane
    gbdt-rest-3 (2026-11-23), each twice to the same bytes: NewtonL2 with
    Bernoulli, Cosine with Poisson, score noise and Gradient leaves on
    Logloss; NewtonCosine with score noise and Cosine at feature_fraction
    0.5 on RMSE; QueryRMSE on the grouped pointwise searcher at
    feature_fraction 0.5. Each moves against the L2 Bayesian fit the host
    trained before. Class weights on Logloss (lane gbdt-one-body) move the
    model; with RMSE they are refused in `train`'s words. Then a NaN in a one_hot_features column that is not
    categorical is refused in `train`'s words (the binarize's), as on the
    device."""
    var clf = _e3()
    var reg = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    if part == 0:
        var base = _pw_opts("Logloss", "L2", "Bayesian")
        var mb = _twice[I](base, clf, "pointwise Logloss L2 Bayesian")
        var o1 = _pw_opts("Logloss", "NewtonL2", "Bernoulli")
        o1.subsample = 0.7
        o1.leaf_iterations = 3
        var m1 = _twice[I](o1, clf, "pointwise Logloss NewtonL2 Bernoulli")
        require(m1.text != mb.text, "pointwise NewtonL2 Bernoulli: the L2 Bayesian model")
        var o2 = _pw_opts("Logloss", "Cosine", "Poisson")
        o2.subsample = 0.66
        o2.random_strength = 1.0
        o2.leaf_method = "Gradient"
        o2.leaf_iterations = 4
        var m2 = _twice[I](o2, clf, "pointwise Logloss Cosine Poisson noise Gradient")
        require(m2.text != mb.text, "pointwise Cosine Poisson: the L2 Bayesian model")
        # lane gbdt-one-body (2026-11-23): class weights on this searcher, the
        # device's one weight column; with RMSE refused in `train`'s words
    if part == 1:
        var ocw = _pw_opts("Logloss", "Cosine", "No")
        var mcw0 = _twice[I](ocw, clf, "pointwise Logloss Cosine")
        var ocw2 = ocw.copy()
        ocw2.class_weights = [1.0, 2.5]
        var mcw = _twice[I](ocw2, clf, "class-weighted pointwise Logloss Cosine")
        require(mcw.text != mcw0.text, "pointwise Logloss class weights had no model effect")
        var rcw = _pw_opts("RMSE", "Cosine", "No")
        rcw.class_weights = [1.0, 2.0]
        var cw_refused = String("")
        try:
            _ = fit[I](rcw, reg)
        except error:
            cw_refused = String(error)
        require_fails_with(cw_refused, "class weights take effect only with a classification loss",
                           "pointwise RMSE class weights")
    if part == 2:
        var o3 = _pw_opts("RMSE", "NewtonCosine", "No")
        o3.random_strength = 1.0
        var m3 = _twice[I](o3, reg, "pointwise RMSE NewtonCosine noise")
        require(all_finite_f32(predict(m3.text, reg.x, 600)), "pointwise NewtonCosine: nonfinite predict")
        var o4 = _pw_opts("RMSE", "Cosine", "No")
        var m4 = _twice[I](o4, reg, "pointwise RMSE Cosine")
        var o4f = o4.copy()
        o4f.feature_fraction = 0.5
        var m4f = _twice[I](o4f, reg, "pointwise RMSE Cosine at feature_fraction 0.5")
        require(m4f.text != m4.text, "pointwise feature_fraction 0.5: the unsampled model")
    if part == 3:
        var grouped = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
        grouped.group_sizes = List[UInt32](length=60, fill=UInt32(10))
        var q = GbdtOpts("QueryRMSE", 4, 3)
        q.learning_rate = 0.1
        q.l2_leaf_reg = 3.0
        q.random_state = 7
        q.pointwise = True
        q.score_function = "Cosine"
        q.bootstrap = "No"
        q.random_strength = 0.0
        q.leaf_method = "Newton"
        q.leaf_iterations = 1
        var mq = _twice[I](q, grouped, "grouped pointwise QueryRMSE")
        var qf = q.copy()
        qf.feature_fraction = 0.5
        var mqf = _twice[I](qf, grouped, "grouped pointwise QueryRMSE at feature_fraction 0.5")
        require(mqf.text != mq.text, "grouped pointwise feature_fraction 0.5: the unsampled model")
        # a NaN in a plain one-hot column (the three class codes appended as
        # column 6, flagged one_hot_features): both columns refuse it
        var xo = load_f32("e3-x.f32", 3600)
        var codes = load_f32("e3-y3.f32", 600)
        for r in range(600):  # small-loop(one appended column)
            xo.append(codes[r])
        xo[3600 + 5] = math_nan[DType.float32]()
        var oh_in = GbdtInputs(xo^, 600, 7, load_f32("e3-yr.f32", 600))
        var oh = GbdtOpts("RMSE", 4, 3)
        oh.deterministic()
        oh.boost_from_average = 0
        oh.leaf_iterations = 1
        oh.one_hot_features = [6]
        var refused = String("")
        try:
            _ = fit[I](oh, oh_in)
        except error:
            refused = String(error)
        require_fails_with(refused, "There are NaNs in feature number", "NaN in a plain one-hot column")


def _pw_refused[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, needle: String, what: String) raises:
    """`o` is refused on this column with `needle` in its words."""
    var refused = String("")
    try:
        _ = fit[I](o, inp)
    except error:
        refused = String(error)
    require_fails_with(refused, needle, what)


def _pw_class0_blind_inputs() -> GbdtInputs:
    """600 rows, 6 uniform columns from a fixed LCG (column-major). Every
    third row is class 0, whatever its columns say; every other row is class
    1 when column 1 is above 0.5, else class 2. Class 0's gradient therefore
    carries no signal for the deciding column: the first class's plane is
    blind to it."""
    var n = 600
    var d = 6
    var x = List[Float32](length=n * d, fill=Float32(0))
    var state = UInt64(0x9E3779B97F4A7C15)
    for c in range(d):  # small-loop(the fixture's six columns)
        for r in range(n):  # small-loop(one fixture column)
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            x[c * n + r] = Float32(Int(state >> UInt64(40))) / Float32(16777216.0)
    var y = List[Float32](length=n, fill=Float32(0))
    for r in range(n):  # small-loop(the fixture's labels)
        if r % 3 == 0:
            y[r] = Float32(0)
        elif x[1 * n + r] > Float32(0.5):
            y[r] = Float32(1)
        else:
            y[r] = Float32(2)
    return GbdtInputs(x^, n, d, y^)


def host_pointwise_multiclass[I: GbdtImage](part: Int) raises:
    """MultiClass, MultiClassOneVsAll and MultiRMSE on the pointwise searcher (lane
    gbdt-pw-multiclass, 2026-11-23), each twice to the same bytes, at rate
    0.1, L2 and Cosine: the split score sums every class's term. QUALITY:
    on `_pw_class0_blind_inputs` (class 0's gradient blind to the deciding
    column 1) the fit separates classes 1 and 2 on column 1, so at least
    90% of the class-1/2 rows are predicted right. The old search, on the
    first class's gradient alone, never saw column 1 and split on noise:
    this assertion fails on it."""
    var inp = _pw_class0_blind_inputs()
    var scores: List[String] = ["L2", "Cosine"]
    if part == 0:
        var losses: List[String] = ["MultiClass", "MultiClassOneVsAll"]
        for li in range(2):  # small-loop(the two multi-output losses)
            for si in range(2):  # small-loop(the two score functions)
                var o = _pw_opts(losses[li], scores[si], "No")
                o.n_estimators = 8
                o.learning_rate = 0.3
                o.max_depth = 3
                o.boost_from_average = 0
                var what = "pointwise " + losses[li] + " " + scores[si]
                var m = _twice[I](o, inp, what)
                var proba = predict_proba(m.text, losses[li], inp.x, inp.n)
                var width = len(proba) // inp.n
                require(width == 3, what + ": " + String(width) + " probability columns")
                var right = 0
                var rows = 0
                for r in range(inp.n):  # small-loop(the fixture's rows)
                    var label = Int(inp.y[r])
                    if label == 0:
                        continue
                    rows += 1
                    var best = 0
                    for k in range(1, width):  # small-loop(three classes)
                        if proba[r * width + k] > proba[r * width + best]:
                            best = k
                    if best == label:
                        right += 1
                require(
                    Float64(right) >= 0.9 * Float64(rows),
                    what + ": " + String(right) + " of " + String(rows)
                    + " class-1/2 rows right; the split ignored the deciding column",
                )
    if part == 1:
        # MultiRMSE: target dimension 0 is noise blind to every column, dimension
        # 1 is +1 / -1 on column 1. Summed over both dimensions the search splits
        # on column 1 and dimension 1's mean squared error falls far below its
        # variance (1); searched on dimension 0 alone it splits on noise and stays
        # near 1.
        var n = inp.n
        var y2 = List[Float32](length=2 * n, fill=Float32(0))
        var state = UInt64(0xD1B54A32D192ED03)
        for r in range(n):  # small-loop(the fixture's rows)
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            y2[r] = Float32(Int(state >> UInt64(40))) / Float32(16777216.0) - Float32(0.5)
            y2[n + r] = Float32(1) if inp.x[1 * n + r] > Float32(0.5) else Float32(-1)
        var reg = GbdtInputs(inp.x.copy(), n, inp.d, y2^)
        for si in range(2):  # small-loop(the two score functions)
            var o = _pw_opts("MultiRMSE", scores[si], "No")
            o.n_estimators = 8
            o.learning_rate = 0.3
            o.max_depth = 3
            o.boost_from_average = 0
            o.target_dim = 2
            var what = "pointwise MultiRMSE " + scores[si]
            var m = _twice[I](o, reg, what)
            var raw = predict(m.text, reg.x, n)
            require(len(raw) == 2 * n, what + ": " + String(len(raw)) + " predictions")
            var sse = Float64(0)
            for r in range(n):  # small-loop(the fixture's rows)
                var e = Float64(raw[r * 2 + 1]) - Float64(reg.y[n + r])
                sse += e * e
            require(
                sse / Float64(n) < 0.25,
                what + ": dimension 1 MSE " + String(sse / Float64(n))
                + "; the split ignored the deciding column",
            )


def host_pointwise_rest[I: GbdtImage](part: Int) raises:
    """The pointwise searcher's arms the host refused by name until lane
    gbdt-cpu-rest-4-pwl (2026-11-23), each twice to the same bytes: the
    one-output losses beside Logloss and RMSE (CrossEntropy, MAE and MAPE
    at their Exact default, Quantile, LogLinQuantile, Poisson, Lq,
    Expectile, Tweedie, Huber), `boost_from_average` unset (resolved as
    `train` resolves it: a bias on MAE, none on Logloss), class weights on
    CrossEntropy, a Median border grid, an eval set under IncToDec and
    under no detector, a one-hot column, a CTR column and `max_depth` 0.
    Then the device's refusals in its words: Depthwise and
    `boost_from_average=True` on Poisson."""
    var clf = _e3()
    var reg = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    # the one-output losses on the pointwise searcher; the 0/1 target serves
    # the losses that need a probability or a non-negative target
    if part == 0:
        var ce = _pw_opts("CrossEntropy", "Cosine", "Bayesian")
        ce.leaf_iterations = 2
        var mce = _twice[I](ce, clf, "pointwise CrossEntropy")
        require("\nbias " in mce.text, "pointwise CrossEntropy with boost_from_average: no bias record")
        var ce0 = ce.copy()
        ce0.boost_from_average = -1
        var mce0 = _twice[I](ce0, clf, "pointwise CrossEntropy, boost_from_average unset")
        require(not ("\nbias " in mce0.text), "CrossEntropy's unset boost_from_average resolved True")
        var cew = ce0.copy()
        cew.class_weights = [1.0, 2.5]
        var mcew = _twice[I](cew, clf, "class-weighted pointwise CrossEntropy")
        require(mcew.text != mce0.text, "pointwise CrossEntropy class weights had no model effect")
        var mae = _pw_opts("MAE", "L2", "No")
        mae.leaf_method = ""
        mae.leaf_iterations = -1
        mae.boost_from_average = -1
        var mmae = _twice[I](mae, reg, "pointwise MAE (Exact, boost_from_average unset)")
        require("\nbias " in mmae.text, "MAE's unset boost_from_average resolved False")
    if part == 1:
        var q = _pw_opts("Quantile", "NewtonL2", "Bernoulli")
        q.subsample = 0.7
        q.loss_alpha = 0.3
        q.leaf_method = "Exact"
        q.leaf_iterations = -1
        _ = _twice[I](q, reg, "pointwise Quantile alpha 0.3 Exact")
        var mape = _pw_opts("MAPE", "Cosine", "No")
        mape.leaf_method = ""
        mape.leaf_iterations = -1
        _ = _twice[I](mape, clf, "pointwise MAPE")
        var llq = _pw_opts("LogLinQuantile", "Cosine", "No")
        llq.boost_from_average = 0
        llq.leaf_method = "Gradient"
        llq.leaf_iterations = 2
        _ = _twice[I](llq, clf, "pointwise LogLinQuantile")
    if part == 2:
        var pois = _pw_opts("Poisson", "NewtonCosine", "Poisson")
        pois.subsample = 0.66
        pois.random_strength = 1.0
        pois.boost_from_average = -1
        pois.leaf_iterations = 3
        _ = _twice[I](pois, clf, "pointwise Poisson")
        var lq = _pw_opts("Lq", "L2", "No")
        lq.boost_from_average = 0
        lq.loss_q = 3.0
        _ = _twice[I](lq, reg, "pointwise Lq q 3")
        var ex = _pw_opts("Expectile", "Cosine", "Bayesian")
        ex.boost_from_average = 0
        ex.loss_alpha = 0.3
        _ = _twice[I](ex, reg, "pointwise Expectile alpha 0.3")
        var tw = _pw_opts("Tweedie", "Cosine", "No")
        tw.boost_from_average = 0
        tw.loss_variance_power = 1.5
        _ = _twice[I](tw, clf, "pointwise Tweedie 1.5")
        var hu = _pw_opts("Huber", "NewtonL2", "No")
        hu.boost_from_average = 0
        hu.loss_delta = 1.0
        _ = _twice[I](hu, reg, "pointwise Huber delta 1")
        # Logloss with boost_from_average unset: False, as `train` resolves it
    if part == 3:
        var ll = _pw_opts("Logloss", "Cosine", "No")
        var mll = _twice[I](ll, clf, "pointwise Logloss")
        var ll0 = ll.copy()
        ll0.boost_from_average = -1
        var mll0 = _twice[I](ll0, clf, "pointwise Logloss, boost_from_average unset")
        require(not ("\nbias " in mll0.text), "Logloss's unset boost_from_average resolved True")
        require(mll0.text != mll.text, "pointwise Logloss: the unset start is the log-odds start")
        # a Median border grid
        var rm = _pw_opts("RMSE", "Cosine", "No")
        var mrm = _twice[I](rm, reg, "pointwise RMSE")
        var rmed = rm.copy()
        rmed.border_type = "Median"
        var mmed = _twice[I](rmed, reg, "pointwise RMSE on Median borders")
        require(mmed.text != mrm.text, "pointwise Median borders: the GreedyLogSum model")
        # an eval set under IncToDec, and under no detector
    if part == 4:
        var held = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
        held.eval_x = held.x.copy()
        held.eval_y = held.y.copy()
        var itd = _pw_opts("Huber", "Cosine", "No")
        itd.boost_from_average = 0
        itd.loss_delta = 1.0
        itd.n_estimators = 12
        itd.od_type = "IncToDec"
        itd.od_pvalue = 0.01
        itd.od_wait = 2
        var mitd = _twice[I](itd, held, "pointwise Huber with an eval set under IncToDec")
        require(len(mitd.test) > 0, "pointwise IncToDec: no held-out curve")
        var nod = _pw_opts("RMSE", "L2", "No")
        nod.n_estimators = 6
        var mnod = _twice[I](nod, held, "pointwise RMSE with an eval set, no detector")
        require(len(mnod.test) == 6, "pointwise eval set without a detector: the curve is not one point a tree")
        # a one-hot column (the three class codes appended as column 6) and a
        # CTR column (the same codes as cat_features)
    if part == 5:
        var xo = load_f32("e3-x.f32", 3600)
        var codes = load_f32("e3-y3.f32", 600)
        for r in range(600):  # small-loop(one appended column)
            xo.append(codes[r])
        var oh_in = GbdtInputs(xo.copy(), 600, 7, load_f32("e3-yr.f32", 600))
        var oh = _pw_opts("RMSE", "Cosine", "No")
        oh.one_hot_features = [6]
        var moh = _twice[I](oh, oh_in, "pointwise RMSE with a one-hot column")
        require("one_hot 1" in moh.text, "pointwise one-hot: no one-hot feature record")
        var cat_in = GbdtInputs(xo^, 600, 7, load_f32("e3-yc.f32", 600))
        var cat = _pw_opts("Logloss", "Cosine", "Bayesian")
        cat.cat_features = [6]
        var mcat = _twice[I](cat, cat_in, "pointwise Logloss with a CTR column")
        require("ctr_table" in mcat.text, "pointwise CTR: no CTR table record")
        # depth 0: one leaf a tree
        var d0 = _pw_opts("RMSE", "Cosine", "No")
        d0.max_depth = 0
        var md0 = _twice[I](d0, reg, "pointwise RMSE at max_depth 0")
        require("depth 0" in md0.text, "pointwise max_depth 0: a tree with splits")
        # the device's refusals, its words
        var dw = _pw_opts("RMSE", "Cosine", "No")
        dw.grow_policy = "Depthwise"
        _pw_refused[I](dw, reg, "OBLIVIOUS searcher", "pointwise Depthwise")
        # (lane gbdt-pw-multiclass: MultiClass on this searcher trains; its
        # unit is host_pointwise_multiclass)
        var pb = _pw_opts("Poisson", "Cosine", "No")
        _pw_refused[I](pb, clf, "You can use boost_from_average only", "pointwise Poisson boost_from_average")


def _sym_opts(loss: String, score: String) -> GbdtOpts:
    """A greedy SymmetricTree fit at rate 0.1 with every work-changing
    option set."""
    var o = GbdtOpts(loss, 4, 3)
    o.learning_rate = 0.1
    o.l2_leaf_reg = 3.0
    o.random_state = 7
    o.score_function = score
    o.leaf_iterations = 1
    o.boost_from_average = 0
    o.deterministic()
    # the host unit's time budget (lane gbdt-units-6): 16 borders still
    # give every policy and path its bins; no unit here reads the default 128
    o.border_count = 16
    return o^


def _sym_refused[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, needle: String, what: String) raises:
    """`o` must refuse with `needle` (the device's words), never fit."""
    var refused = False
    try:
        _ = fit[I](o, inp)
    except error:
        require_fails_with(String(error), needle, what)
        refused = True
    require(refused, what + " was silently accepted")


def _l2_cosine_inputs(kind: Int) -> GbdtInputs:
    """A 400-row, two-column fixture on which the L2 and Cosine calcers
    choose DIFFERENT root splits (lane gbdt-units-6), so a fit that ignores
    `score_function` fails the comparison. Column 0 isolates rows 0..19, all
    positive; column 1 splits rows 0..199 (115 positive) from 200..399 (95
    positive). At the start (p = 0.5, der = y - 0.5, w = 1) and
    `l2_leaf_reg` 1000 the sums are column 0: (W, S) = (20, 10), (380, 0);
    column 1: (200, 15), (200, -5). L2 = sum S^2 / (W + 1000): column 1
    0.208 > column 0 0.098. Cosine = that over sqrt(sum W S^2 / (W +
    1000)^2): column 0 2.236 > column 1 1.118. So L2 splits column 1 and
    Cosine column 0. `kind` 0: Logloss labels 0/1; 1: RMSE targets +/-0.5
    (the same derivatives at a zero start); 2: MultiClass labels 0/1 (two
    classes, the pinned class doubles both scores, the order holds)."""
    var n = 400
    var x = List[Float32](length=2 * n, fill=Float32(0))
    var y = List[Float32](length=n, fill=Float32(0))
    for r in range(n):  # small-loop(the fixture's rows)
        var pos: Bool
        if r < 20:
            x[r] = Float32(1)
            pos = True
        elif r < 200:
            pos = r < 115
        else:
            pos = r < 295
        if r < 200:
            x[n + r] = Float32(1)
        if kind == 1:
            y[r] = Float32(0.5) if pos else Float32(-0.5)
        else:
            y[r] = Float32(1) if pos else Float32(0)
    return GbdtInputs(x^, n, 2, y^)


def _l2_cosine_opts(loss: String, score: String) -> GbdtOpts:
    """`_sym_opts` at `l2_leaf_reg` 1000 and depth 1 (one split a tree: the
    root `_l2_cosine_inputs` makes the two calcers disagree on)."""
    var o = _sym_opts(loss, score)
    o.l2_leaf_reg = 1000.0
    o.max_depth = 1
    return o^


def _newton_cosine_inputs() -> GbdtInputs:
    """A 400-row, two-column Logloss fixture on which Cosine and NewtonCosine
    choose DIFFERENT root splits at `l2_leaf_reg` 250 (lane gbdt-units-7).
    At the zero start every row's Newton plane is `0.25` (p = 0.5) and its
    derivative +/-0.5, so NewtonCosine at l2 is twice Cosine at 4 * l2 =
    1000 and the ranking can flip. Column 0 isolates rows 0..19 (all
    positive): (W, S) = (20, 10), (380, 0); its Cosine is 2.236 at every
    l2. Column 1 isolates rows 20..59 (36 positive): (40, 16), (360, -6);
    its Cosine is 2.380 at l2 250 (beats column 0) and 2.124 at l2 1000
    (loses). So Cosine splits column 1 and NewtonCosine column 0. Labels:
    rows 0..55 positive, 56..59 negative, 60..213 positive, 214..399
    negative (210 positive in all)."""
    var n = 400
    var x = List[Float32](length=2 * n, fill=Float32(0))
    var y = List[Float32](length=n, fill=Float32(0))
    for r in range(n):  # small-loop(the fixture's rows)
        if r < 20:
            x[r] = Float32(1)
        elif r < 60:
            x[n + r] = Float32(1)
        if r < 56 or (r >= 60 and r < 214):
            y[r] = Float32(1)
    return GbdtInputs(x^, n, 2, y^)


def _newton_cosine_opts(score: String) -> GbdtOpts:
    """`_sym_opts` at `l2_leaf_reg` 250 and depth 1 (the root
    `_newton_cosine_inputs` makes the two scores disagree on)."""
    var o = _sym_opts("Logloss", score)
    o.l2_leaf_reg = 250.0
    o.max_depth = 1
    return o^


def host_symmetric_rest[I: GbdtImage](part: Int) raises:
    """The greedy SymmetricTree and Ordered combinations the host refused
    by name until lane gbdt-cpu-rest-4-sym (2026-11-23), each fit twice to
    the same bytes: the L2, NewtonL2 and NewtonCosine scores on Logloss,
    RMSE, MAE and MultiClass (L2), Gradient leaves on Logloss and RMSE,
    RMSE Newton leaves at three iterations, boost_from_average on Logloss,
    CrossEntropy and Ordered Logloss, class weights on a pointwise loss;
    then the combinations both columns refuse, in the device's words."""
    var clf = _e3()
    var reg = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    var mc_in = _fixed("e3-x.f32", 600, 6, "e3-y3.f32")
    # the four scores on Logloss: each its own model
    if part == 0:
        var cos = _twice[I](_sym_opts("Logloss", "Cosine"), clf, "symmetric Logloss Cosine")
        var l2 = _twice[I](_sym_opts("Logloss", "L2"), clf, "symmetric Logloss L2")
        var nl2 = _twice[I](_sym_opts("Logloss", "NewtonL2"), clf, "symmetric Logloss NewtonL2")
        var ncos = _twice[I](_sym_opts("Logloss", "NewtonCosine"), clf, "symmetric Logloss NewtonCosine")
        # L2 against Cosine on the fixture where their root splits provably
        # differ (lane gbdt-units-6: on e3 the two calcers rank the splits alike
        # at l2 3 and 600 rows, so equal models proved nothing either way)
        var sc_clf = _l2_cosine_inputs(0)
        var dl2 = fit[I](_l2_cosine_opts("Logloss", "L2"), sc_clf)
        var dcos = fit[I](_l2_cosine_opts("Logloss", "Cosine"), sc_clf)
        require(dl2.text != dcos.text, "symmetric Logloss L2: the Cosine model")
        require(nl2.text != l2.text, "symmetric Logloss NewtonL2: the L2 model (first-order planes)")
        # NewtonCosine against Cosine on the fixture where their root splits
        # provably differ (lane gbdt-units-7). On e3 the planes are NOT the
        # culprit: the host search pass writes `w * p(1 - p)` under the Newton
        # scores (`_logloss_search_pass(second_order)`, the device's
        # `launch_approximate[False, True]`), but at p ~ 0.5 that plane is the
        # first-order one times 0.25, and Cosine on a plane scaled by c is
        # Cosine at `l2 / c` times a constant -- at l2 3 on 600 rows the
        # ranking does not move, so the two models were equal by arithmetic.
    if part == 1:
        var sn = _newton_cosine_inputs()
        var dncos = fit[I](_newton_cosine_opts("NewtonCosine"), sn)
        var dcos2 = fit[I](_newton_cosine_opts("Cosine"), sn)
        require(dncos.text != dcos2.text, "symmetric Logloss NewtonCosine: the Cosine model (first-order planes)")
        # RMSE: der2 is 1, so at unit weight the Newton planes are the
        # first-order ones and the Newton scores give the same bytes
    if part == 2:
        var rl2 = _twice[I](_sym_opts("RMSE", "L2"), reg, "symmetric RMSE L2")
        var rnl2 = _twice[I](_sym_opts("RMSE", "NewtonL2"), reg, "symmetric RMSE NewtonL2")
        var sc_reg = _l2_cosine_inputs(1)
        var drl2 = fit[I](_l2_cosine_opts("RMSE", "L2"), sc_reg)
        var drcos = fit[I](_l2_cosine_opts("RMSE", "Cosine"), sc_reg)
        require(drl2.text != drcos.text, "symmetric RMSE L2: the Cosine model")
        require(rnl2.text == rl2.text, "symmetric RMSE NewtonL2 at unit weight differs from L2")
        # a pointwise loss and the multiclass loss under L2
    if part == 3:
        var mae = _sym_opts("MAE", "L2")
        mae.leaf_method = "Exact"
        _ = _twice[I](mae, reg, "symmetric MAE L2 Exact")
        _ = _twice[I](_sym_opts("MultiClass", "L2"), mc_in, "symmetric MultiClass L2")
        var sc_mc = _l2_cosine_inputs(2)
        var dmcl2 = fit[I](_l2_cosine_opts("MultiClass", "L2"), sc_mc)
        var dmccos = fit[I](_l2_cosine_opts("MultiClass", "Cosine"), sc_mc)
        require(dmcl2.text != dmccos.text, "symmetric MultiClass L2: the Cosine model")
        # Gradient leaves and the RMSE walker at three iterations
    if part == 4:
        var lg = _sym_opts("Logloss", "Cosine")
        lg.leaf_method = "Gradient"
        lg.leaf_iterations = 4
        var mlg = _twice[I](lg, clf, "symmetric Logloss Gradient leaves")
        var cos_newton = fit[I](_sym_opts("Logloss", "Cosine"), clf)
        require(mlg.text != cos_newton.text, "symmetric Logloss Gradient: the Newton model")
        var rg = _sym_opts("RMSE", "Cosine")
        rg.leaf_method = "Gradient"
        _ = _twice[I](rg, reg, "symmetric RMSE Gradient leaves")
        var r3 = _sym_opts("RMSE", "Cosine")
        r3.leaf_method = "Newton"
        r3.leaf_iterations = 3
        var mr3 = _twice[I](r3, reg, "symmetric RMSE Newton at three iterations")
        require(all_finite_f32(predict(mr3.text, reg.x, 600)), "symmetric RMSE three iterations: nonfinite predict")
        # boost_from_average: the log-odds start, written as the bias
    if part == 5:
        var lb = _sym_opts("Logloss", "Cosine")
        lb.boost_from_average = 1
        var mlb = _twice[I](lb, clf, "symmetric Logloss boost_from_average")
        require("\nbias " in mlb.text, "symmetric Logloss boost_from_average: no bias record")
        var ce = _sym_opts("CrossEntropy", "Cosine")
        ce.boost_from_average = 1
        var mce = _twice[I](ce, clf, "symmetric CrossEntropy boost_from_average")
        require("\nbias " in mce.text, "symmetric CrossEntropy boost_from_average: no bias record")
        var ob = _ordered_opts("Logloss")
        ob.boost_from_average = 1
        var mob = _twice[I](ob, clf, "Ordered Logloss boost_from_average")
        require("\nbias " in mob.text, "Ordered Logloss boost_from_average: no bias record")
        # class weights on a pointwise loss: the one weight column
    if part == 6:
        var q = _sym_opts("Quantile", "Cosine")
        q.leaf_method = "Exact"
        var mq = _twice[I](q, clf, "symmetric Quantile")
        var qw = q.copy()
        qw.class_weights = [1.0, 3.0]
        var mqw = _twice[I](qw, clf, "class-weighted symmetric Quantile")
        require(mqw.text != mq.text, "symmetric Quantile class weights had no model effect")
        # refused on both columns, in the device's words
        var ex = _sym_opts("Logloss", "Cosine")
        ex.leaf_method = "Exact"
        _sym_refused[I](ex, clf, "Only MAPE, MAE and Quantile are supported for Exact", "symmetric Logloss Exact")
        var mvs = _sym_opts("Logloss", "Cosine")
        mvs.bootstrap = "MVS"
        _sym_refused[I](mvs, clf, "MVS is not reachable", "symmetric MVS")
        var mcn = _sym_opts("MultiClass", "NewtonCosine")
        _sym_refused[I](mcn, mc_in, "doesn't support second derivatives", "MultiClass NewtonCosine")
        var pb = _sym_opts("Poisson", "Cosine")
        pb.boost_from_average = 1
        _sym_refused[I](pb, reg, "You can use boost_from_average only", "symmetric Poisson boost_from_average")
        var opb = _ordered_opts("Poisson")
        opb.boost_from_average = 1
        _sym_refused[I](opb, reg, "You can use boost_from_average only", "Ordered Poisson boost_from_average")


def host_tail_rest[I: GbdtImage](part: Int) raises:
    """The last combinations the host refused by name until lane
    gbdt-cpu-rest-4-tail (2026-11-23), each fit twice to the same bytes:
    the multi-output losses' leaf walk (`device_multi_walk_estimate`) under
    Newton at three iterations and Gradient at its default (10) and at two,
    on MultiClass and MultiClassOneVsAll; Lossguide past depth 16 and
    border_count 255; then `train`'s limits on both columns, in its words
    (`check_tree_limits`): border_count 0 and 256, max_depth 17 under
    SymmetricTree and Depthwise, and Simple leaves on MultiClass in
    `simple_leaves_check`'s words (lane gbdt-cpu-rest-4-simple; were the
    walker's)."""
    var mc_in = _fixed("e3-x.f32", 600, 6, "e3-y3.f32")
    var clf = _e3()
    if part == 0:
        var mc1 = _twice[I](_sym_opts("MultiClass", "Cosine"), mc_in, "MultiClass Newton one iteration")
        var n3 = _sym_opts("MultiClass", "Cosine")
        n3.leaf_method = "Newton"
        n3.leaf_iterations = 3
        var mn3 = _twice[I](n3, mc_in, "MultiClass Newton three iterations")
        require(mn3.text != mc1.text, "MultiClass Newton at three iterations: the one-step model")
        require(all_finite_f32(predict(mn3.text, mc_in.x, 600)), "MultiClass Newton three iterations: nonfinite predict")
        var gd = _sym_opts("MultiClass", "Cosine")
        gd.leaf_method = "Gradient"
        gd.leaf_iterations = -1
        var mgd = _twice[I](gd, mc_in, "MultiClass Gradient at its default count")
        require(mgd.text != mc1.text, "MultiClass Gradient: the Newton model")
        var g2 = gd.copy()
        g2.leaf_iterations = 2
        var mg2 = _twice[I](g2, mc_in, "MultiClass Gradient at two iterations")
        require(mg2.text != mgd.text, "MultiClass Gradient at two iterations: the default-count model")
    if part == 1:
        var ova = _sym_opts("MultiClassOneVsAll", "Cosine")
        var mova = _twice[I](ova, mc_in, "MultiClassOneVsAll Newton one iteration")
        var ovg = ova.copy()
        ovg.leaf_method = "Gradient"
        ovg.leaf_iterations = -1
        var movg = _twice[I](ovg, mc_in, "MultiClassOneVsAll Gradient")
        require(movg.text != mova.text, "MultiClassOneVsAll Gradient: the Newton model")
        var ovn = ova.copy()
        ovn.leaf_method = "Newton"
        ovn.leaf_iterations = 2
        var movn = _twice[I](ovn, mc_in, "MultiClassOneVsAll Newton two iterations")
        require(movn.text != mova.text, "MultiClassOneVsAll Newton at two iterations: the one-step model")
        # depth and borders: Lossguide's depth is uncapped, 255 borders train
    if part == 2:
        var lg = GbdtOpts("Logloss", 3, 20)
        lg.grow_policy = "Lossguide"
        lg.max_leaves = 8
        lg.score_function = "Cosine"
        var mlg = _twice[I](lg, clf, "Lossguide at max_depth 20")
        require("\nntree 0 " in mlg.text, "Lossguide max_depth 20: non-symmetric model text")
        var b255 = _sym_opts("Logloss", "Cosine")
        b255.border_count = 255
        _ = _twice[I](b255, clf, "symmetric Logloss at border_count 255")
        # `train`'s limits, refused on both columns in its words
        var b0 = _sym_opts("Logloss", "Cosine")
        b0.border_count = 0
        _sym_refused[I](b0, clf, "border_count must be in 1..255", "border_count 0")
        var b256 = _sym_opts("Logloss", "Cosine")
        b256.border_count = 256
        _sym_refused[I](b256, clf, "border_count must be in 1..255", "border_count 256")
        var d17 = _sym_opts("Logloss", "Cosine")
        d17.max_depth = 17
        _sym_refused[I](d17, clf, "Maximum tree depth is 16", "SymmetricTree max_depth 17")
        var dw17 = GbdtOpts("Logloss", 3, 17)
        dw17.grow_policy = "Depthwise"
        dw17.score_function = "Cosine"
        _sym_refused[I](dw17, clf, "Maximum tree depth is 16", "Depthwise max_depth 17")
        var simple = _sym_opts("MultiClass", "Cosine")
        simple.leaf_method = "Simple"
        _sym_refused[I](simple, mc_in, "the multi-output losses", "MultiClass Simple leaves")


def host_simple_rest[I: GbdtImage](part: Int) raises:
    """`leaf_estimation_method='Simple'`, the last NO_CPU_PATH clause until
    lane gbdt-cpu-rest-4-simple (2026-11-23), each fit twice to the same
    bytes: the greedy SymmetricTree searcher's own leaves on Logloss, RMSE
    (where they ARE the Newton-1 leaves, DEVIATION 64, so the two texts are
    equal), Quantile and QueryRMSE, and over CTR columns (four permutations,
    only the learn cursor moves) on Logloss and Quantile. Then the
    admission both columns call (`simple_leaves_check`,
    `simple_leaf_iterations_check`), in its words: MultiClass, Depthwise,
    Ordered, the pointwise searcher, three iterations; and YetiRank in its
    own words."""
    var clf = _e3()
    var reg = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    var qwi = _qw_inputs()
    if part == 0:
        var ll = _sym_opts("Logloss", "Cosine")
        ll.leaf_method = "Simple"
        var mll = _twice[I](ll, clf, "symmetric Logloss Simple")
        var lln = ll.copy()
        lln.leaf_method = "Newton"
        require(mll.text != fit[I](lln, clf).text, "symmetric Logloss Simple: the Newton model")
        require(all_finite_f32(predict(mll.text, clf.x, 600)), "symmetric Logloss Simple: nonfinite predict")
        var rm = _sym_opts("RMSE", "NewtonL2")
        rm.leaf_method = "Simple"
        var mrm = _twice[I](rm, reg, "symmetric RMSE Simple")
        var rmn = rm.copy()
        rmn.leaf_method = "Newton"
        require(mrm.text == fit[I](rmn, reg).text, "symmetric RMSE Simple is not the Newton-1 model (DEVIATION 64)")
    if part == 1:
        var qt = _sym_opts("Quantile", "Cosine")
        qt.leaf_method = "Simple"
        var mqt = _twice[I](qt, reg, "symmetric Quantile Simple")
        # (lane gbdt-units-6) Simple is compared with Quantile's own Exact
        # default: one Gradient step IS the searcher's leaf here
        # (`sum(w * der) / (sum w + l2)` from the same planes; Quantile's
        # alpha-0.5 derivatives are +/-0.5, on the fixed-point grid, so the
        # snapped plane is the exact one), the way RMSE Simple is Newton-1
        # above, so `!= Gradient` could only fail
        var qte = qt.copy()
        qte.leaf_method = "Exact"
        qte.leaf_iterations = -1
        require(mqt.text != fit[I](qte, reg).text, "symmetric Quantile Simple: the Exact model")
    if part == 2:
        var qw = _qw_opts("QueryRMSE")
        qw.pointwise = False
        qw.leaf_method = "Simple"
        qw.leaf_iterations = 1
        var mqw = _twice[I](qw, qwi, "greedy QueryRMSE Simple")
        var qwn = qw.copy()
        qwn.leaf_method = "Newton"
        require(mqw.text != fit[I](qwn, qwi).text, "greedy QueryRMSE Simple: the Newton model")
    if part == 3:
        var cat_clf = _cat_rest_inputs("e3-yc.f32")
        var lc = _cat_rest_opts("Logloss")
        lc.cat_features = [6, 7]
        lc.leaf_method = "Simple"
        lc.leaf_iterations = 1
        var mlc = _twice_text[I](lc, cat_clf, "symmetric Logloss Simple over CTR columns")
        var lcn = lc.copy()
        lcn.leaf_method = "Newton"
        require(mlc.text != fit[I](lcn, cat_clf).text, "symmetric Logloss Simple over CTR columns: the Newton model")
        var cat_reg = _cat_rest_inputs("e3-yr.f32")
        var qc = _cat_rest_opts("Quantile")
        qc.cat_features = [6]
        qc.leaf_method = "Simple"
        qc.leaf_iterations = 1
        _ = _twice_text[I](qc, cat_reg, "symmetric Quantile Simple over a CTR column")
        # refused before any tree on both columns, in the one admission's words
        var mc = _sym_opts("MultiClass", "Cosine")
        mc.leaf_method = "Simple"
        _sym_refused[I](mc, _fixed("e3-x.f32", 600, 6, "e3-y3.f32"), "the multi-output losses", "MultiClass Simple")
        var dw = _ns_opts("Logloss", "Depthwise", "Cosine")
        dw.leaf_method = "Simple"
        dw.leaf_iterations = 1
        _sym_refused[I](dw, clf, "grow_policy Depthwise / Lossguide searches the structure only", "Depthwise Logloss Simple")
        var od = _ordered_opts("RMSE")
        od.leaf_method = "Simple"
        _sym_refused[I](od, reg, "boosting_type='Ordered' searches the structure only", "Ordered RMSE Simple")
        var pw = _sym_opts("Logloss", "Cosine")
        pw.pointwise = True
        pw.leaf_method = "Simple"
        _sym_refused[I](pw, clf, "use_pointwise_searcher searches the structure only", "pointwise Logloss Simple")
        var it3 = _sym_opts("Logloss", "Cosine")
        it3.leaf_method = "Simple"
        it3.leaf_iterations = 3
        _sym_refused[I](it3, clf, "can't be greater, than 1 for Simple", "Logloss Simple at three iterations")
        var yr = _qw_opts("YetiRank")
        yr.pointwise = False
        yr.leaf_method = "Simple"
        _sym_refused[I](yr, qwi, "changing the leaf_estimation_method parameter is prohibited", "YetiRank Simple")


def _ordered_opts(loss: String) -> GbdtOpts:
    """An Ordered fit at rate 0.1 with every work-changing option set."""
    var o = GbdtOpts(loss, 4, 3)
    o.boosting_type = "Ordered"
    o.learning_rate = 0.1
    o.l2_leaf_reg = 3.0
    o.score_function = "Cosine"
    o.leaf_iterations = 1
    o.deterministic()
    # the host unit's time budget (lane gbdt-units-6): 16 borders still
    # give every policy and path its bins; no unit here reads the default 128
    o.border_count = 16
    return o^


def host_ordered_weights[I: GbdtImage]() raises:
    """Sample and class weights under Ordered boosting, which the host
    refused by name until lane gbdt-ordered-weights (2026-11-23), each fit
    twice to the same bytes: weighted Logloss (unit weights give the
    unweighted bytes, the device's ones column; a ramp moves the model),
    class weights on Logloss, weighted RMSE under NewtonCosine with
    boost_from_average and the Bayesian bootstrap, weighted MAE (the
    weighted median start), and class weights with RMSE refused in
    `train`'s words."""
    var clf = _fixed("e3-x.f32", 600, 6, "e3-yc.f32")
    var ll = _ordered_opts("Logloss")
    var plain = _twice[I](ll, clf, "unweighted Ordered Logloss")
    clf.weights = List[Float32](length=600, fill=Float32(1))
    var ones = _twice[I](ll, clf, "unit-weighted Ordered Logloss")
    require(ones.text == plain.text, "unit-weighted Ordered Logloss differs from the unweighted fit")
    for r in range(600):  # small-loop(np.linspace(.5, 1.5, 600, dtype=float32))
        clf.weights[r] = Float32(0.5 + Float64(r) / 599.0)
    var ramp = _twice[I](ll, clf, "weighted Ordered Logloss")
    require(ramp.text != plain.text, "Ordered Logloss sample weights had no model effect")
    clf.weights = List[Float32]()
    var cw = _ordered_opts("Logloss")
    cw.class_weights = [1.0, 2.0]
    var mcw = _twice[I](cw, clf, "class-weighted Ordered Logloss")
    require(mcw.text != plain.text, "Ordered Logloss class weights had no model effect")
    var reg = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    reg.weights = List[Float32](length=600, fill=Float32(0))
    for r in range(600):  # small-loop(np.linspace(.5, 1.5, 600, dtype=float32))
        reg.weights[r] = Float32(0.5 + Float64(r) / 599.0)
    var rmse = _ordered_opts("RMSE")
    rmse.score_function = "NewtonCosine"
    rmse.boost_from_average = 1
    rmse.bootstrap = "Bayesian"
    rmse.bagging_temperature = 1.0
    var mr = _twice[I](rmse, reg, "weighted Ordered RMSE")
    require(all_finite_f32(predict(mr.text, reg.x, 600)), "weighted Ordered RMSE: nonfinite predict")
    var mae = _ordered_opts("MAE")
    mae.boost_from_average = 1
    _ = _twice[I](mae, reg, "weighted Ordered MAE")
    var bad = _ordered_opts("RMSE")
    bad.class_weights = [1.0, 2.0]
    var refused = False
    try:
        _ = fit[I](bad, reg)
    except error:
        require_fails_with(String(error), "class weights take effect only with a classification loss",
                           "Ordered RMSE class weights")
        refused = True
    require(refused, "Ordered RMSE class weights were silently accepted")


def host_rest_gaps[I: GbdtImage]() raises:
    """The fits the host refused by name until lane gbdt-cpu-rest-2
    (2026-11-23), each twice to the same bytes: an eval set with the Iter
    detector on symmetric Quantile and on MultiClass, feature_fraction 0.5
    on symmetric MAE Exact and on MultiClassOneVsAll, the pointwise searcher
    with RMSE, and NaN in the float columns beside a CTR column."""
    var reg = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    reg.eval_x = reg.x.copy()
    reg.eval_y = reg.y.copy()
    var q = GbdtOpts("Quantile", 12, 3)
    q.deterministic()
    q.leaf_method = "Gradient"
    q.leaf_iterations = 1
    q.od_type = "Iter"
    q.od_wait = 2
    q.use_best_model = 1
    var mq = _twice[I](q, reg, "symmetric Quantile with an eval set")
    require(len(mq.test) > 0, "symmetric Quantile: no held-out curve")
    var mc3 = _fixed("e3-x.f32", 600, 6, "e3-y3.f32")
    mc3.eval_x = mc3.x.copy()
    mc3.eval_y = mc3.y.copy()
    var mc = GbdtOpts("MultiClass", 8, 3)
    mc.deterministic()
    mc.od_type = "Iter"
    mc.od_wait = 2
    var mm = _twice[I](mc, mc3, "MultiClass with an eval set")
    require(len(mm.test) > 0, "MultiClass: no held-out curve")
    var plain = _fixed("e3-x.f32", 600, 6, "e3-yr.f32")
    var mae = GbdtOpts("MAE", 4, 3)
    mae.deterministic()
    mae.leaf_method = "Exact"
    mae.feature_fraction = 0.5
    _ = _twice[I](mae, plain, "symmetric MAE Exact at feature_fraction 0.5")
    var ova_in = _fixed("e3-x.f32", 600, 6, "e3-y3.f32")
    var ova = GbdtOpts("MultiClassOneVsAll", 4, 3)
    ova.deterministic()
    ova.feature_fraction = 0.5
    _ = _twice[I](ova, ova_in, "MultiClassOneVsAll at feature_fraction 0.5")
    var pw = GbdtOpts("RMSE", 4, 3)
    pw.pointwise = True
    pw.score_function = "L2"
    pw.bootstrap = "Bayesian"
    pw.bagging_temperature = 0.5
    pw.random_strength = 0.0
    pw.boost_from_average = 1
    pw.leaf_method = "Newton"
    pw.leaf_iterations = 1
    var mp = _twice[I](pw, plain, "the pointwise searcher with RMSE")
    require(all_finite_f32(predict(mp.text, plain.x, 600)), "pointwise RMSE: nonfinite predict")
    # NaN in the six float columns beside a CTR column (the three class
    # codes, above one_hot_max_size 2)
    var xn = load_f32("e3-xn.f32", 3600)
    var codes = load_f32("e3-y3.f32", 600)
    for r in range(600):  # small-loop(one appended column)
        xn.append(codes[r])
    var cat_in = GbdtInputs(xn^, 600, 7, load_f32("e3-yr.f32", 600))
    var cat = GbdtOpts("RMSE", 4, 3)
    cat.deterministic()
    cat.boost_from_average = 0
    cat.leaf_iterations = 1
    cat.cat_features = [6]
    # (`_twice_text`: the host reader refuses a CTR model's predict, lane
    # gbdt-cpu-rest-4-cat-ns)
    var mcat = _twice_text[I](cat, cat_in, "RMSE with NaN beside a CTR column")
    require(mcat.text.byte_length() > 0, "NaN beside CTR: empty model text")



def _cat_ns_inputs(yname: String) raises -> GbdtInputs:
    """e3's six float columns, then its three class codes (column 6: a CTR
    column under cat_features, three categories above one_hot_max_size 2)
    and code 2 against the rest (column 7: a one-hot column under
    cat_features, or one_hot_features)."""
    var x = load_f32("e3-x.f32", 3600)
    var codes = load_f32("e3-y3.f32", 600)
    for r in range(600):  # small-loop(one appended column)
        x.append(codes[r])
    for r in range(600):  # small-loop(one appended column)
        x.append(Float32(1.0) if codes[r] == Float32(2.0) else Float32(0.0))
    return GbdtInputs(x^, 600, 8, load_f32(yname, 600))


def host_cat_rest_ns[I: GbdtImage]() raises:
    """Depthwise and Lossguide over categorical columns, Plain boosting,
    which the host refused by name until lane gbdt-cpu-rest-4-cat-ns
    (2026-11-23), each fit twice to the same bytes: Depthwise Logloss over a
    CTR column, Lossguide RMSE over a one-hot and a CTR column with sample
    weights, Depthwise Quantile over a CTR column with an eval set, Lossguide
    CrossEntropy over a one_hot_features column; and max_ctr_complexity 2
    under Depthwise refused in `fit_with_test`'s words."""
    var clf = _cat_ns_inputs("e3-yc.f32")
    var dl = GbdtOpts("Logloss", 4, 3)
    dl.deterministic()
    dl.learning_rate = 0.1
    dl.grow_policy = "Depthwise"
    dl.cat_features = [6]
    var mdl = _twice_text[I](dl, clf, "Depthwise Logloss over a CTR column")
    require("\nntree 0 " in mdl.text, "Depthwise Logloss CTR: non-symmetric model text")
    require("\nctr_table " in mdl.text, "Depthwise Logloss CTR: no ctr_table record")
    var reg = _cat_ns_inputs("e3-yr.f32")
    reg.weights = List[Float32](length=600, fill=Float32(0))
    for r in range(600):  # small-loop(np.linspace(.5, 1.5, 600, dtype=float32))
        reg.weights[r] = Float32(0.5 + Float64(r) / 599.0)
    var lr = GbdtOpts("RMSE", 4, 4)
    lr.deterministic()
    lr.learning_rate = 0.1
    lr.grow_policy = "Lossguide"
    lr.max_leaves = 8
    lr.cat_features = [6, 7]
    var mlr = _twice_text[I](lr, reg, "weighted Lossguide RMSE over a one-hot and a CTR column")
    require(" type cat " in mlr.text and " type ctr " in mlr.text,
            "Lossguide RMSE: the one-hot and CTR feature records")
    require(all_finite_f64(mlr.learn), "Lossguide RMSE CTR: nonfinite learn curve")
    var qin = _cat_ns_inputs("e3-yr.f32")
    qin.eval_x = qin.x.copy()
    qin.eval_y = qin.y.copy()
    var q = GbdtOpts("Quantile", 4, 3)
    q.deterministic()
    q.learning_rate = 0.1
    q.grow_policy = "Depthwise"
    q.cat_features = [6]
    var mq = _twice_text[I](q, qin, "Depthwise Quantile over a CTR column with an eval set")
    require(len(mq.test) > 0, "Depthwise Quantile CTR: no held-out curve")
    var ce = GbdtOpts("CrossEntropy", 4, 4)
    ce.deterministic()
    ce.learning_rate = 0.1
    ce.grow_policy = "Lossguide"
    ce.max_leaves = 8
    ce.one_hot_features = [7]
    var mce = _twice[I](ce, clf, "Lossguide CrossEntropy over a one_hot_features column")
    require(" type cat " in mce.text, "Lossguide CrossEntropy: no one-hot feature record")
    var bad = GbdtOpts("Logloss", 2, 3)
    bad.deterministic()
    bad.grow_policy = "Depthwise"
    bad.cat_features = [6]
    bad.max_ctr_complexity = 2
    var refused = False
    try:
        _ = fit[I](bad, clf)
    except error:
        require_fails_with(
            String(error),
            "max_ctr_complexity above 1 (feature combinations) is implemented"
            " for grow_policy='SymmetricTree'; got Depthwise",
            "max_ctr_complexity 2 under Depthwise",
        )
        refused = True
    require(refused, "max_ctr_complexity 2 under Depthwise was silently accepted")

def host_losses[I: GbdtImage](case_index: Int) raises:
    """`test_the_five_configurations_fit_on_the_host_when_built`."""
    var reg = GbdtInputs(load_f32("e3-x.f32", 3600), 600, 6, load_f32("e3-yr.f32", 600))
    if case_index == 0:
        var losses: List[String] = ["Quantile", "MAPE", "Poisson", "Tweedie"]
        for loss in losses:  # small-loop(the four pointwise losses of the original)
            var o = GbdtOpts(loss, 3, 3)
            if loss == "Tweedie":
                o.loss_variance_power = 1.5
            _ = _twice[I](o, reg, loss)
    elif case_index == 1:
        var o = GbdtOpts("MAE", 3, 3)
        o.leaf_method = "Exact"
        o.bootstrap = "Poisson"
        o.subsample = 0.6
        _ = _twice[I](o, reg, "MAE Exact Poisson")
    elif case_index == 2:
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
    elif case_index <= 4:
        var loss = String("MultiClass") if case_index == 3 else String("MultiClassOneVsAll")
        var three = GbdtInputs(load_f32("e3-x.f32", 3600), 600, 6, load_f32("e3-y3.f32", 600))
        var o = GbdtOpts(loss, 3, 3)
        if case_index == 3:
            o.class_weights = [1.0, 2.0, 0.5]
        var m = _twice[I](o, three, loss)
        require(len(predict_proba(m.text, loss, three.x, 600)) == 600 * 3, "proba shape (600, 3): " + loss)
    elif case_index == 5:
        var o = GbdtOpts("Quantile", 2, 3)
        o.bootstrap = "Bayesian"
        _ = _twice[I](o, reg, "Quantile Bayesian")
    else:
        # a real fit since lane gbdt-cpu-rest (2026-11-23): the pointwise
        # losses on the non-symmetric driver
        var o = GbdtOpts("Quantile", 2, 3)
        o.grow_policy = "Depthwise"
        var m = _twice[I](o, reg, "a Depthwise Quantile fit")
        require("\nntree 0 " in m.text, "Depthwise Quantile: non-symmetric model text")
        require(all_finite_f32(predict(m.text, reg.x, 600)), "Depthwise Quantile: nonfinite predict")


def _trees_record(text: String) raises -> Int:
    """`re.search(r"^trees (\\d+)$", text, re.M)`."""
    for raw in text.split("\n"):  # small-loop(model text records)
        var line = String(raw)
        if line.startswith("trees "):
            return Int(String(line.removeprefix("trees ")))
    raise Error("no trees record")


def host_ordered[I: GbdtImage](case_index: Int) raises:
    """`test_ordered_and_feature_freq_fit_on_the_host_when_built`, its native
    arms (the save/load round trip and the CTR model's predictions, read by
    the host CTR door, stay in Python)."""
    var x = load_f32("od-x.f32", 7200)
    var y = load_f32("od-y.f32", 1200)
    if case_index == 0:
        # OrderedRMSE(n_estimators=3, max_depth=3), twice; weighted, twice
        # (a real fit since lane gbdt-cpu-rest, 2026-11-23), and unit weights
        # give the unweighted bytes (the device's ones fill)
        var perm = load_u32("od-perm.u32", 1200)
        var texts = List[String]()
        for _ in range(2):  # small-loop(the original's two fits)
            texts.append(gbdt_fit_ordered_rmse_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(x.unsafe_ptr()),
                Int(perm.unsafe_ptr()), 1200, 6, 0, 1200, 3, 3, 16, Float32(0.03), Float32(3.0)))
        require(texts[0] == texts[1], "two ordered host fits returned different bytes")
        require(same_f32(predict(texts[0], x, 1200), predict(texts[1], x, 1200)), "ordered predictions differ")
        var ones = List[Float32](length=1200, fill=Float32(1))
        var unit_w = gbdt_fit_ordered_rmse_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(ones.unsafe_ptr()),
            Int(perm.unsafe_ptr()), 1200, 6, 1200, 1200, 3, 3, 16, Float32(0.03), Float32(3.0))
        require(unit_w == texts[0], "unit-weighted OrderedRMSE differs from the unweighted fit")
        var ramp = List[Float32](length=1200, fill=Float32(0))
        for r in range(1200):  # small-loop(np.linspace(.5, 1.5, 1200, dtype=float32))
            ramp[r] = Float32(0.5 + Float64(r) / 1199.0)
        var wa = gbdt_fit_ordered_rmse_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(ramp.unsafe_ptr()),
            Int(perm.unsafe_ptr()), 1200, 6, 1200, 1200, 2, 2, 16, Float32(0.03), Float32(3.0))
        var wb = gbdt_fit_ordered_rmse_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(ramp.unsafe_ptr()),
            Int(perm.unsafe_ptr()), 1200, 6, 1200, 1200, 2, 2, 16, Float32(0.03), Float32(3.0))
        require(wa == wb, "two weighted ordered host fits returned different bytes")
        require(all_finite_f32(predict(wa, x, 1200)), "weighted OrderedRMSE: nonfinite predict")
        # KEEP-ALIVE (lane gbdt-units-6): `y` is read by ADDRESS inside the
        # calls; its last named use was `wb`'s argument expression, so Mojo
        # freed it BEFORE that call read it, the call's own first allocation
        # (the permutation copy, the same 4800 bytes) took the block, and
        # `wb` fitted the permutation's words as targets: the weighted pair
        # differed (unit 14). The fit itself is deterministic.
        _ = y
        _ = x
        _ = ones^
        _ = ramp^
        _ = perm^
        return
    var coded = load_f32("od-coded.f32", 7200)
    var yc = load_f32("od-yc.f32", 1200)
    if case_index == 1:
        # ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7),
        # twice; weighted, twice (a real fit since lane gbdt-cpu-rest,
        # 2026-11-23), and a weight that moves the model
        var sources: List[UInt32] = [0, 1]
        var sp = _U32P(unsafe_from_address=Int(sources.unsafe_ptr()))
        var a = gbdt_fit_two_level_feature_freq_run(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), sp,
                                                    1200, 6, 0, 2, Float32(0.03), Float32(3.0))
        var b = gbdt_fit_two_level_feature_freq_run(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), sp,
                                                    1200, 6, 0, 2, Float32(0.03), Float32(3.0))
        require(a == b, "two FeatureFreq host fits returned different bytes")
        require(same_f32(predict(a, coded, 1200), predict(b, coded, 1200)), "FeatureFreq predictions differ")
        var ramp = List[Float32](length=1200, fill=Float32(0))
        for r in range(1200):  # small-loop(np.linspace(.5, 1.5, 1200, dtype=float32))
            ramp[r] = Float32(0.5 + Float64(r) / 1199.0)
        var wa = gbdt_fit_two_level_feature_freq_run(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), sp,
                                                     1200, 6, 1200, 2, Float32(0.03), Float32(3.0),
                                                     Int(ramp.unsafe_ptr()))
        var wb = gbdt_fit_two_level_feature_freq_run(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), sp,
                                                     1200, 6, 1200, 2, Float32(0.03), Float32(3.0),
                                                     Int(ramp.unsafe_ptr()))
        require(wa == wb, "two weighted FeatureFreq host fits returned different bytes")
        require(wa != a, "FeatureFreq sample weights had no model effect")
        _ = y  # read by address by the calls above (its last named use is the weighted call's argument)
        # KEEP-ALIVE (lane gbdt-units-6): `coded` is read by address too; its
        # last named use was `wb`'s argument, so `wb` read freed memory as
        # its categorical codes ("categorical feature 0 row 0 is outside the
        # UInt32 code range", unit 15)
        _ = coded
        _ = ramp^
        _ = sources^
        return
    if case_index == 2:
        # the pointwise eval arm, and (lane gbdt-cpu-rest, 2026-11-23) the
        # same fit without weights, a real fit: the device's unweighted arm
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
        _ = _twice[I](o, inp, "an unweighted pointwise fit")
        inp.weights = load_f32("od-wts.f32", 1200)
        _ = _twice[I](o, inp, "pointwise eval arm")
        return
    var inp = GbdtInputs(coded.copy(), 1200, 6, yc.copy())
    var o = GbdtOpts("Logloss", 3, 3)
    o.deterministic()
    if case_index == 3:
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
    if case_index == 4:
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


# ---- The host column (was gbdt_public_fits_cpu_suite.mojo) ----
# The GradientBoosting fit cases on the host column, across host cores.
#
# Every unit (`gbdt_public_fits_cases.mojo`: the common units, then the
# host-only units of the `*_on_the_host_when_built` tests) is an independent
# fixture with its own inputs and fits; they run across the host cores with a
# private failure slot each. Each host fit keeps its own feature-parallel work.
struct _HostFit(GbdtImage):
    """The host image's fit (`bindings/_mojolearn_gbdt_host.mojo` `_GbdtHostImage`)."""
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        require(not row_major, "the host fit is column-major")
        return gbdt_fit_native_run(args)

    @staticmethod
    def row_major() -> Bool:
        return False

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return model_dim(text)

    # The predict, resident and ordered/feature-freq operations of GbdtImage
    # (lane dw-trees-boost, 2026-11-21): this check image fits only, so each
    # refuses in words.
    @staticmethod
    def predict(text: String, x: Int, n_rows: Int, dst: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict")

    @staticmethod
    def predict_multi(text: String, x: Int, n_rows: Int, dst: Int, mode: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict_multi")

    @staticmethod
    def binary(probabilities: Bool, raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no binary prediction")

    @staticmethod
    def sigmoid_pair(raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_sigmoid_pair")

    @staticmethod
    def resident() -> Bool:
        return False

    @staticmethod
    def resident_prepare(text: String) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_prepare")

    @staticmethod
    def resident_predict(handle: Int, x: Int, n_rows: Int, dst: Int, mode: Int, row_major: Bool) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_predict")

    @staticmethod
    def ordered_fit(x: Int, y: Int, weights: Int, permutation: Int, n_rows: Int, n_features: Int, n_weights: Int,
                    n_permutation: Int, n_estimators: Int, max_depth: Int, border_count: Int,
                    learning_rate: Float32, l2_leaf_reg: Float32) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_ordered_rmse")

    @staticmethod
    def feature_freq_fit(x: Int, y: Int, weights: Int, sources: Int, n_rows: Int, n_features: Int, n_weights: Int,
                         n_sources: Int, learning_rate: Float32, l2_leaf_reg: Float32,
                         random_seed: Int) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_two_level_feature_freq")

    @staticmethod
    def importances(text: String, x: Int, n_rows: Int, n_input: Int, row_major: Bool, weights: Int,
                    n_weights: Int, dst: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt feature importances")


# ---- The device column (was gbdt_public_fits_gpu_suite.mojo) ----
# The common GradientBoosting fit cases on the device column.
#
# One unit at a time on the physical GPU (each device fit owns the device for
# its whole graph; its kernels stay parallel). The host-only units are the CPU
# suite's. Predictions read the device-fitted model text with the host reader
# (`gbdt_public_fits_cases.predict`); the device-resident predict has its own
# check (`gbdt_resident_ns_predict_gpu`).
struct _DeviceFit(GbdtImage):
    """The GPU image's column-major fit (`bindings/_mojolearn_gbdt.mojo` `_GbdtGpuImage`)."""
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        require(not row_major, "the check fits column-major")
        return gbdt_fit_native_run_gpu(False, args)

    @staticmethod
    def row_major() -> Bool:
        return False

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return model_dim(text)

    # The predict, resident and ordered/feature-freq operations of GbdtImage
    # (lane dw-trees-boost, 2026-11-21): this check image fits only, so each
    # refuses in words.
    @staticmethod
    def predict(text: String, x: Int, n_rows: Int, dst: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict")

    @staticmethod
    def predict_multi(text: String, x: Int, n_rows: Int, dst: Int, mode: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict_multi")

    @staticmethod
    def binary(probabilities: Bool, raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no binary prediction")

    @staticmethod
    def sigmoid_pair(raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_sigmoid_pair")

    @staticmethod
    def resident() -> Bool:
        return False

    @staticmethod
    def resident_prepare(text: String) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_prepare")

    @staticmethod
    def resident_predict(handle: Int, x: Int, n_rows: Int, dst: Int, mode: Int, row_major: Bool) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_predict")

    @staticmethod
    def ordered_fit(x: Int, y: Int, weights: Int, permutation: Int, n_rows: Int, n_features: Int, n_weights: Int,
                    n_permutation: Int, n_estimators: Int, max_depth: Int, border_count: Int,
                    learning_rate: Float32, l2_leaf_reg: Float32) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_ordered_rmse")

    @staticmethod
    def feature_freq_fit(x: Int, y: Int, weights: Int, sources: Int, n_rows: Int, n_features: Int, n_weights: Int,
                         n_sources: Int, learning_rate: Float32, l2_leaf_reg: Float32,
                         random_seed: Int) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_two_level_feature_freq")

    @staticmethod
    def importances(text: String, x: Int, n_rows: Int, n_input: Int, row_major: Bool, weights: Int,
                    n_weights: Int, dst: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt feature importances")


def _host_unit(unit: Int) raises:
    # every unit prints its seconds (lane gbdt-units-7), pass or fail
    var t0 = perf_counter_ns()
    try:
        host_unit[_HostFit](unit)
    finally:
        print("gbdt_public_fits host unit", unit, "seconds", Float64(perf_counter_ns() - t0) / 1e9)


def _common_host(unit: Int) raises:
    var t0 = perf_counter_ns()
    try:
        common_unit[_HostFit](unit)
    finally:
        print("gbdt_public_fits common unit", unit, "seconds", Float64(perf_counter_ns() - t0) / 1e9)


def _cpu_cases() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime assert not GBDT_ORACLE_HOST_SABOTAGE, "a sabotage build loaded outside the gate"
    check_cases("common", GBDT_COMMON_UNITS, _common_host)
    check_cases("host", GBDT_HOST_UNITS, _host_unit)


def _gpu_cases() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():  # discovery instantiates every module function
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        var failed = String("")
        var count = 0
        for unit in range(GBDT_COMMON_UNITS):  # driver-loop: one device graph at a time (see the docstring)
            try:
                common_unit[_DeviceFit](unit)
            except error:
                count += 1
                failed += "\n      common row " + String(unit) + ": " + String(error)
        if count > 0:
            raise Error("common: " + String(count) + " of " + String(GBDT_COMMON_UNITS) + " cases failed" + failed)


def test_cases() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _cpu_cases()
    else:
        _gpu_cases()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
