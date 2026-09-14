# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_gbdt` family, GradientBoosting on the
gbdt-symmetric lane (workstream E batch 3, 2026-09-14; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 gbdt and the batch
3 section).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`gbdt/host/gbdt_oracle.mojo::gbdt_host_fit`, the device trainer
(`gbdt/train.mojo::train`, `gbdt/methods/doc_parallel_boosting.mojo::
fit_with_test`, the greedy symmetric searcher and the Newton leaf estimator)
restated on the host, so the model text is meant to be the GPU columns'
bytes. The predict entry is `core/gbdt_host_predict.mojo::gbdt_host_predict`,
the walk the forest host binding runs and the forest host gate holds to the
recorded GPU predictions, over the model text parsed here.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/ensemble.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_gbdt": "_mojolearn_gbdt_host"`):
`gbdt_fit` (the same eight arguments, the same 35-slot params list with its
counted class-weight tail and optional float tail, the same four strings,
the same five-element return), `gbdt_predict` and `gbdt_model_dim` (the
model text in, as `bindings/_mojolearn_gbdt.mojo:410-450`), `gbdt_sigmoid`
(its body), `gbdt_vendor` answering "cpu" and `gbdt_numeric_mode`.

ABSENT, and so refused BY NAME through `_HostBinding`: `gbdt_predict_multi`
(a multi-dimensional model), `gbdt_fit_ordered_rmse`,
`gbdt_fit_two_level_feature_freq`, `gbdt_binary_probabilities` and
`gbdt_binary_classes` (the adapters' device transforms),
`gbdt_per_round_paths` and the two `*_parallel_available` probes.

REFUSED BY NAME INSIDE `gbdt_fit`, with the sentence
`cpu_identity_gate_check.py` requires ("no CPU implementation of"), every
parameter value the oracle does not restate: see `_refuse` and the oracle's
module docstring. Every other GBDT lane of tools/identity_break.py therefore
reads REFUSED on a CPU-only install, as it did before this binding existed.

The sabotage arm (`gbdt_host_sabotage`) is
`gbdt/host/gbdt_oracle.mojo::GBDT_ORACLE_HOST_SABOTAGE`: the Newton walker's
Hessian regularizer is one larger, so every leaf of every tree moves.
"""
from std.math import isfinite
from std.memory import bitcast
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, read_f32, u32_ptr
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, identical_exp64
from core.gbdt_host_predict import gbdt_host_predict
from gbdt.data.quantization import (
    NAN_TREATMENT_AS_FALSE,
    NAN_TREATMENT_AS_IS,
    NAN_TREATMENT_AS_TRUE,
)
from gbdt.host.gbdt_oracle import (
    GBDT_LOGLOSS_NEWTON_ITERATIONS,
    GBDT_ORACLE_HOST_SABOTAGE,
    GbdtHostParams,
    gbdt_host_fit,
    gbdt_host_model_text,
)
from gbdt.options.data_processing_options import nan_mode_from_name


#: `SCORE_FUNCTION_COSINE` and `LEAF_ESTIMATION_NEWTON`
#: (`gbdt/options/catboost_options.mojo:130`, `:270`), the codes the wrapper
#: sends in slots 10 and 17.
comptime GBDT_HOST_SCORE_COSINE = 1
comptime GBDT_HOST_LEAF_NEWTON = 1
#: `DEFAULT_TARGET_BORDER` (`gbdt/options/loss_description.mojo:72`).
comptime GBDT_HOST_DEFAULT_BORDER = Float32(0.5)
#: The deepest tree the host binding grows; the model text reader refuses
#: past 31 and `1 << depth` leaves are materialized per tree.
comptime GBDT_HOST_MAX_DEPTH = 16


def gbdt_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gbdt_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def gbdt_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives in
    a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "gbdt host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_gbdt_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `gbdt_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def gbdt_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary adds 1.0 to the Newton walker's Hessian
    regularizer on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's
    negative control; `gbdt/host/gbdt_oracle.mojo::GBDT_ORACLE_HOST_SABOTAGE`)."""
    return PythonObject(GBDT_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def gbdt_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def gbdt_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _refuse(what: String) raises:
    """The by-name refusal. The sentence starts with the phrase the CPU
    identity gate's column check keys on, so a lane this binding does not
    cover reads REFUSED and never a hash."""
    raise Error(
        "no CPU implementation of _mojolearn_gbdt.gbdt_fit for " + what
        + "; the gbdt host binding trains the gbdt-symmetric lane only"
        " (SymmetricTree, Logloss, Cosine, Newton leaves, no bootstrap,"
        " weights, categoricals, eval set or NaN), see"
        " gbdt/host/gbdt_oracle.mojo"
    )


def gbdt_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    weights_addr: PythonObject,
    cat_flags_addr: PythonObject,
    eval_x_addr: PythonObject,
    eval_y_addr: PythonObject,
    params: PythonObject,
    strs: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit_binding` (`bindings/_mojolearn_gbdt.mojo:169-407`): the same
    slot checks in the same words, then the refusals of what the host fit
    does not restate, then the host fit and the model text. Returns
    `[model_text, best_iteration, stopped_early, learn_losses,
    test_losses]`."""
    if len(params) < 35:
        raise Error(
            "gbdt_fit: params must hold at least 35 values, got "
            + String(len(params))
        )
    var n_class_weights = Int(py=params[34])
    if n_class_weights < 0:
        raise Error(
            "gbdt_fit: n_class_weights must not be negative, got "
            + String(n_class_weights)
        )
    var fixed_and_weights = 35 + n_class_weights
    if len(params) != fixed_and_weights and len(params) != fixed_and_weights + 1 and len(params) != fixed_and_weights + 2 and len(params) != fixed_and_weights + 3:
        raise Error(
            "gbdt_fit: params must hold 35 + n_class_weights, optionally min_split_gain, min_child_hessian, then feature_fraction values ("
            + String(35 + n_class_weights)
            + ") values, got "
            + String(len(params))
        )
    if len(strs) != 4:
        raise Error(
            "gbdt_fit: strs must hold [loss, bootstrap_type, od_type,"
            " nan_mode], got " + String(len(strs))
        )
    var xp = f32_ptr(Int(py=x_addr))
    var yp = f32_ptr(Int(py=y_addr))
    _ = f32_ptr(Int(py=weights_addr))
    _ = u32_ptr(Int(py=cat_flags_addr))
    _ = f32_ptr(Int(py=eval_x_addr))
    _ = f32_ptr(Int(py=eval_y_addr))

    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_flags = Int(py=params[3])
    var grow_code = Int(py=params[31])
    if grow_code != 0 and grow_code != 1 and grow_code != 2:
        raise Error(
            "gbdt_fit: grow_policy code must be 0 (SymmetricTree), 1"
            " (Depthwise) or 2 (Lossguide), got " + String(grow_code)
        )
    var min_split_gain = Float64(-1)
    if len(params) >= fixed_and_weights + 1:
        min_split_gain = Float64(py=params[fixed_and_weights])
    var min_child_hessian = Float64(-1)
    if len(params) >= fixed_and_weights + 2:
        min_child_hessian = Float64(py=params[fixed_and_weights + 1])
    var feature_fraction = Float64(1)
    if len(params) == fixed_and_weights + 3:
        feature_fraction = Float64(py=params[fixed_and_weights + 2])

    var border_count = Int(py=params[4])
    var n_estimators = Int(py=params[5])
    var max_depth = Int(py=params[6])
    var learning_rate = Float32(Float64(py=params[7]))
    var l2_leaf_reg = Float32(Float64(py=params[8]))
    var random_seed = UInt64(Int(py=params[9]))
    var score_function = Int(py=params[10])
    var loss = String(py=strs[0])
    var loss_border = Float32(Float64(py=params[15]))
    var leaf_iterations = Int(py=params[16])
    var leaf_method = Int(py=params[17])
    var bootstrap_type = String(py=strs[1])
    var n_eval_rows = Int(py=params[20])
    var od_type = String(py=strs[2])
    var od_pvalue = Float64(py=params[21])
    var od_wait = Int(py=params[22])
    var use_best_model = Int(py=params[23])
    var best_model_min_trees = Int(py=params[24])
    var nan_mode_name = String(py=strs[3])
    var random_strength = Float32(Float64(py=params[25]))
    var use_pointwise = Int(py=params[26]) != 0
    var border_build_max_samples = Int(py=params[27])
    var permutation_count = Int(py=params[28])
    var ctr_permutation_id = Int(py=params[29])
    var boost_from_average = Int(py=params[30])
    var max_leaves = Int(py=params[32])
    var min_data_in_leaf = Int(py=params[33])

    # `gbdt_fit`'s own checks (`gbdt/estimator.mojo:446-464`)
    if n_rows <= 0:
        raise Error("gbdt_fit: n_rows must be positive")
    if n_features <= 0:
        raise Error("gbdt_fit: n_features must be positive")
    if n_weights != 0 and n_weights != n_rows:
        raise Error(
            "gbdt_fit: n_weights must be 0 or n_rows, got " + String(n_weights)
        )
    if n_flags != 0 and n_flags != n_features:
        raise Error(
            "gbdt_fit: n_flags must be 0 or n_features, got " + String(n_flags)
        )
    if n_eval_rows < 0:
        raise Error(
            "gbdt_fit: n_eval_rows must not be negative, got "
            + String(n_eval_rows)
        )

    # ---- what the host fit does not restate, refused by name ----
    if loss != String("Logloss"):
        _refuse("loss='" + loss + "'")
    if grow_code != 0:
        _refuse("grow_policy code " + String(grow_code) + " (Depthwise or Lossguide)")
    if use_pointwise:
        _refuse("use_pointwise_searcher=True")
    if score_function != GBDT_HOST_SCORE_COSINE:
        _refuse("score_function code " + String(score_function) + " (only Cosine)")
    if leaf_method != -1 and leaf_method != GBDT_HOST_LEAF_NEWTON:
        _refuse("leaf_estimation_method code " + String(leaf_method) + " (only Newton)")
    if bootstrap_type != String("") and bootstrap_type != String("No"):
        _refuse("bootstrap_type='" + bootstrap_type + "'")
    if n_weights != 0:
        _refuse("sample_weight")
    if n_class_weights != 0:
        _refuse("class_weights")
    if n_flags != 0:
        _refuse("cat_features or one_hot_features")
    if n_eval_rows != 0:
        _refuse("eval_set")
    if od_type.byte_length() > 0 or od_pvalue >= 0.0 or od_wait >= 0:
        _refuse("the overfitting detector (od_type, od_pvalue, od_wait)")
    if random_strength != Float32(0.0):
        _refuse("random_strength=" + String(random_strength))
    if boost_from_average == 1:
        _refuse("boost_from_average=True")
    if feature_fraction != 1.0:
        _refuse("feature_fraction=" + String(feature_fraction))
    if border_count < 1 or border_count > 255:
        _refuse("border_count=" + String(border_count) + " (1 to 255)")
    if max_depth < 0 or max_depth > GBDT_HOST_MAX_DEPTH:
        _refuse("max_depth=" + String(max_depth) + " (0 to 16)")

    # ---- `train`'s own raises on the rest (`gbdt/train.mojo:944-1149`,
    # `:1580-1605`, `:1733-1758`) ----
    # `child_hessian_threshold` (`gbdt/options/child_hessian.mojo:8-14`)
    if not isfinite(min_child_hessian) or (
        min_child_hessian < 0 and min_child_hessian != -1
    ) or min_child_hessian > Float64(Float32.MAX_FINITE):
        raise Error(
            "min_child_hessian must be -1 (disabled) or finite nonnegative"
            " and <= Float32.MAX_FINITE"
        )
    if min_child_hessian >= 0:
        raise Error("min_child_hessian requires Depthwise or Lossguide")
    if not isfinite(min_split_gain) or (
        min_split_gain < 0 and min_split_gain != -1
    ):
        raise Error("min_split_gain must be -1 (disabled) or finite and nonnegative")
    if min_split_gain >= 0:
        raise Error("min_split_gain requires Depthwise or Lossguide")
    if min_data_in_leaf != 1:
        raise Error(
            "min_data_in_leaf=" + String(min_data_in_leaf) + " does nothing"
            " under grow_policy=SymmetricTree: CatBoost guards its leaf-size"
            " test with `Policy != SymmetricTree`"
            " (greedy_search_helper.cpp:685) and discards the value; refused"
            " here rather than accepted and ignored. It is live under"
            " Depthwise and Lossguide."
        )
    if max_leaves >= 0 and max_leaves != (1 << max_depth):
        raise Error(
            "max_leaves option works only with lossguide tree growing"
            " (catboost_options.cpp:998): under grow_policy="
            + String("SymmetricTree") + " CatBoost pins it to 1 << depth == "
            + String(1 << max_depth) + ", got " + String(max_leaves)
        )
    var perm_count = permutation_count
    if perm_count == -1:
        perm_count = 4
    if perm_count < 1:
        raise Error(
            "Permutation count should be positive, got " + String(perm_count)
        )
    perm_count = 1
    var est_perm = ctr_permutation_id
    if est_perm == -1:
        est_perm = perm_count - 1
    if est_perm < 0 or est_perm >= perm_count:
        raise Error(
            "ctr_estimation_permutation_id " + String(est_perm)
            + " is outside the " + String(perm_count)
            + " permutations this fit builds"
        )
    if boost_from_average != -1 and boost_from_average != 0:
        raise Error(
            "boost_from_average must be -1 (their data-dependent"
            " default), 0 or 1; got " + String(boost_from_average)
        )
    if use_best_model == 1:
        raise Error(
            "use_best_model=1 needs an eval set: pass eval_x_colmajor"
            " and eval_y, or leave it unset."
        )
    elif use_best_model != 0 and use_best_model != -1:
        raise Error(
            "use_best_model must be -1 (unset), 0 or 1, got "
            + String(use_best_model)
        )
    if best_model_min_trees < 1:
        raise Error(
            "best_model_min_trees must be at least 1, got "
            + String(best_model_min_trees)
        )
    var nan_mode = nan_mode_from_name(nan_mode_name)

    var iterations = GBDT_LOGLOSS_NEWTON_ITERATIONS
    if leaf_iterations >= 0:
        iterations = leaf_iterations
    var border = GBDT_HOST_DEFAULT_BORDER
    if loss_border >= Float32(0.0):
        border = loss_border

    var p = GbdtHostParams(
        border_count, border_build_max_samples, n_estimators, max_depth,
        learning_rate, l2_leaf_reg, random_seed, nan_mode, border, iterations,
    )
    var x_address = Int(py=x_addr)
    var y_address = Int(py=y_addr)
    _ = xp
    _ = yp
    var text = String("")
    var losses = List[Float64]()
    var best_iteration = 0
    var stopped_early = False
    var has_nan = False
    with GILReleased(Python()):
        var x = read_f32(x_address, n_rows * n_features)
        for i in range(n_rows * n_features):
            if x[i] != x[i]:
                has_nan = True
                break
        if not has_nan:
            var y = read_f32(y_address, n_rows)
            var model = gbdt_host_fit(x, y, n_rows, n_features, p)
            text = gbdt_host_model_text(model)
            losses = model.losses.copy()
            best_iteration = model.best_iteration
            stopped_early = model.stopped_early
    if has_nan:
        _refuse("an X carrying NaN (the nan_mode Min and Max arms)")

    var learn = Python.list()
    for i in range(len(losses)):
        learn.append(PythonObject(losses[i]))
    var test = Python.list()
    var out = Python.list()
    out.append(PythonObject(text))
    out.append(PythonObject(best_iteration))
    out.append(PythonObject(stopped_early))
    out.append(learn)
    out.append(test)
    return out


# ===========================================================================
# THE MODEL TEXT, READ (the records `gbdt_host_model_text` and the GPU
# binding's `model_text` write; `gbdt/models/model_text.mojo:706` is the
# reader this follows, for the oblivious float-only shape)
# ===========================================================================


@fieldwise_init
struct _HostModelArrays(Movable):
    var fold_counts: List[Int32]
    var one_hot: List[Int32]
    var nan_treatment: List[Int32]
    var border_offsets: List[Int32]
    var borders: List[Float32]
    var tree_offsets: List[Int32]
    var split_feature: List[Int32]
    var split_bin: List[Int32]
    var split_take_bin: List[Int32]
    var leaf_offsets: List[Int32]
    var leaves: List[Float32]
    var dim: Int
    var bias: Float64


def _fields(line: String) raises -> List[String]:
    """`_fields` (`model_text.mojo:686-695`)."""
    var out = List[String]()
    for piece in line.split(" "):
        var s = String(piece)
        if s.byte_length() > 0:
            out.append(s)
    return out^


def _hex_value(code: Int) raises -> Int:
    if code >= 48 and code <= 57:
        return code - 48
    if code >= 97 and code <= 102:
        return code - 87
    if code >= 65 and code <= 70:
        return code - 55
    raise Error("not a hex digit: byte value " + String(code))


def _token_bits(tok: String, digits: Int) raises -> UInt64:
    """The hex half of `<decimal>/<hex bits>`, the half both readers load."""
    var parts = tok.split("/")
    if len(parts) != 2:
        raise Error("malformed float token '" + tok + "'")
    var s = String(parts[1])
    if s.byte_length() != digits:
        raise Error(
            "expected " + String(digits) + " hex digits, got '" + s + "'"
        )
    var v = UInt64(0)
    for i in range(digits):
        v = (v << UInt64(4)) | UInt64(_hex_value(ord(String(s[byte=i]))))
    return v


def _need(t: List[String], count: Int, kind: String) raises:
    if len(t) < count:
        raise Error("a `" + kind + "` record is short")


def _parse_model(text: String) raises -> _HostModelArrays:
    var n_features = -1
    var n_trees = -1
    var n_losses = -1
    var header_seen = 0
    var features_seen = 0
    var trees_seen = 0
    var losses_seen = 0
    var arrays = _HostModelArrays(
        List[Int32](), List[Int32](), List[Int32](), List[Int32](),
        List[Float32](), List[Int32](), List[Int32](), List[Int32](),
        List[Int32](), List[Int32](), List[Float32](), -1, Float64(0.0),
    )
    arrays.border_offsets.append(Int32(0))
    arrays.tree_offsets.append(Int32(0))
    arrays.leaf_offsets.append(Int32(0))
    var cur_depth = 0
    var cur_splits = 0
    var cur_leaves = 0
    var cur_dim = 1
    for raw in text.split("\n"):
        var line = String(raw)
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var t = _fields(line)
        if len(t) == 0:
            continue
        var kind = t[0]
        if kind == String("format"):
            _need(t, 3, "format")
            if header_seen != 0 or t[1] != String("mojolearn-model"):
                raise Error("not a mojolearn-model file")
            if Int(t[2]) != 2:
                raise Error("format version " + t[2] + ", this reader is version 2")
            header_seen = 1
        elif kind == String("features"):
            _need(t, 3, "features")
            if header_seen != 1:
                raise Error("`features` must follow `format`")
            n_features = Int(t[1])
            header_seen = 2
        elif kind == String("trees"):
            _need(t, 2, "trees")
            if header_seen != 2:
                raise Error("`trees` must follow `features`")
            n_trees = Int(t[1])
            header_seen = 3
        elif kind == String("losses"):
            _need(t, 2, "losses")
            if header_seen != 3:
                raise Error("`losses` must follow `trees`")
            n_losses = Int(t[1])
            header_seen = 4
        elif kind == String("bias"):
            _need(t, 2, "bias")
            arrays.bias = bitcast[DType.float64](_token_bits(t[1], 16))
        elif kind == String("feature"):
            _need(t, 12, "feature")
            if header_seen != 4 or Int(t[1]) != features_seen:
                raise Error("a `feature` record out of order")
            var folds = Int(t[3])
            var flag = Int(t[5])
            if t[6] != String("type") or (t[7] != String("float") and t[7] != String("cat")):
                _refuse_predict("a feature of type " + t[7])
            var treat: Int32
            if t[9] == String("as_is"):
                treat = Int32(NAN_TREATMENT_AS_IS)
            elif t[9] == String("as_false"):
                treat = Int32(NAN_TREATMENT_AS_FALSE)
            elif t[9] == String("as_true"):
                treat = Int32(NAN_TREATMENT_AS_TRUE)
            else:
                raise Error("unknown nan treatment '" + t[9] + "'")
            var nb = Int(t[11])
            if len(t) != 12 + nb:
                raise Error("feature " + t[1] + " declares " + t[11] + " borders")
            for b in range(nb):
                arrays.borders.append(
                    bitcast[DType.float32](UInt32(_token_bits(t[12 + b], 8)))
                )
            arrays.fold_counts.append(Int32(folds))
            arrays.one_hot.append(Int32(1) if flag != 0 else Int32(0))
            arrays.nan_treatment.append(treat)
            arrays.border_offsets.append(Int32(len(arrays.borders)))
            features_seen += 1
        elif kind == String("tree"):
            _need(t, 8, "tree")
            if features_seen != n_features or Int(t[1]) != trees_seen:
                raise Error("a `tree` record out of order")
            if trees_seen > 0 and (
                cur_splits != cur_depth or cur_leaves != (1 << cur_depth) * cur_dim
            ):
                raise Error("tree " + String(trees_seen - 1) + " is incomplete")
            cur_depth = Int(t[3])
            cur_dim = Int(t[5])
            if Int(t[7]) != 0:
                _refuse_predict("a tree carrying leaf weights")
            if cur_depth < 0 or cur_depth > 31 or cur_dim < 1:
                raise Error("tree depth or dim is not sane")
            if arrays.dim == -1:
                arrays.dim = cur_dim
            elif arrays.dim != cur_dim:
                raise Error("tree " + t[1] + " has a different dim")
            cur_splits = 0
            cur_leaves = 0
            trees_seen += 1
            arrays.tree_offsets.append(
                arrays.tree_offsets[len(arrays.tree_offsets) - 1] + Int32(cur_depth)
            )
            arrays.leaf_offsets.append(
                arrays.leaf_offsets[len(arrays.leaf_offsets) - 1]
                + Int32((1 << cur_depth) * cur_dim)
            )
        elif kind == String("split"):
            _need(t, 5, "split")
            if Int(t[1]) != trees_seen - 1 or Int(t[2]) != cur_splits:
                raise Error("a `split` record out of order")
            arrays.split_feature.append(Int32(Int(t[3])))
            arrays.split_bin.append(Int32(Int(t[4])))
            var take_bin = len(t) >= 7 and t[5] == String("split_type") and t[6] == String("take_bin")
            arrays.split_take_bin.append(Int32(1) if take_bin else Int32(0))
            cur_splits += 1
        elif kind == String("leaf"):
            _need(t, 4, "leaf")
            if Int(t[1]) != trees_seen - 1 or Int(t[2]) != cur_leaves:
                raise Error("a `leaf` record out of order")
            arrays.leaves.append(bitcast[DType.float32](UInt32(_token_bits(t[3], 8))))
            cur_leaves += 1
        elif kind == String("loss"):
            losses_seen += 1
        elif (
            kind == String("ntree") or kind == String("node")
            or kind == String("weight") or kind == String("ctr_columns")
            or kind == String("ctr_table") or kind == String("ctr_entry")
            or kind == String("tensor_ctr_registry")
        ):
            _refuse_predict("a model with `" + kind + "` records")
        else:
            raise Error("unknown record keyword '" + kind + "'")
    if header_seen != 4 or features_seen != n_features or trees_seen != n_trees:
        raise Error("the model text is incomplete")
    if losses_seen != n_losses:
        raise Error("the model text declares " + String(n_losses) + " losses")
    if trees_seen > 0 and (
        cur_splits != cur_depth or cur_leaves != (1 << cur_depth) * cur_dim
    ):
        raise Error("tree " + String(trees_seen - 1) + " is incomplete")
    if arrays.dim == -1:
        arrays.dim = 1
    return arrays^


def _refuse_predict(what: String) raises:
    raise Error(
        "no CPU implementation of _mojolearn_gbdt.gbdt_predict for " + what
        + "; the gbdt host binding reads oblivious float-only models (HostGBDT"
        " in python/mojolearn/_gbdt_host.py serves the rest)"
    )


def gbdt_predict_binding(
    model: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_predict_binding` (`bindings/_mojolearn_gbdt.mojo:410-439`): RAW
    approxes for a one-dimensional model, `params` is `[n_rows]`, the
    feature count comes from the model, returns rows written."""
    if len(params) != 1:
        raise Error(
            "gbdt_predict: params must hold [n_rows], got "
            + String(len(params))
        )
    var text = String(py=model)
    var x_address = Int(py=x_addr)
    var op = f32_ptr(Int(py=out_addr))
    _ = f32_ptr(x_address)
    var n_rows = Int(py=params[0])
    if n_rows <= 0:
        raise Error("gbdt_predict: n_rows must be positive")
    var m = _parse_model(text)
    if m.dim != 1:
        raise Error(
            "predict_floats is one-dimensional; this model has"
            " approx_dim " + String(m.dim)
            + ". Use predict_multi_floats, which returns"
            " [row * approx_dim + dim]."
        )
    var n_features = len(m.fold_counts)
    var n_trees = len(m.tree_offsets) - 1
    with GILReleased(Python()):
        var x = read_f32(x_address, n_rows * n_features)
        var out = List[Float32](length=n_rows, fill=Float32(0.0))
        var node_left = List[Int32](length=1, fill=Int32(0))
        var node_right = List[Int32](length=1, fill=Int32(0))
        var split_feature = m.split_feature.copy()
        var split_bin = m.split_bin.copy()
        var split_take_bin = m.split_take_bin.copy()
        if len(split_feature) == 0:
            split_feature.append(Int32(0))
            split_bin.append(Int32(0))
            split_take_bin.append(Int32(0))
        var borders = m.borders.copy()
        if len(borders) == 0:
            borders.append(Float32(0.0))
        var leaves = m.leaves.copy()
        if len(leaves) == 0:
            leaves.append(Float32(0.0))
        gbdt_host_predict(
            x, n_rows, n_features,
            rebind[MutPointer[Int32, MutUntrackedOrigin]](m.border_offsets.unsafe_ptr()),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](borders.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](m.fold_counts.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](m.one_hot.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](m.nan_treatment.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](m.tree_offsets.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](split_feature.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](split_bin.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](split_take_bin.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](node_left.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](node_right.unsafe_ptr()),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](m.leaf_offsets.unsafe_ptr()),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](leaves.unsafe_ptr()),
            n_trees, 1, False, len(m.split_feature), len(m.leaves), m.bias,
            out,
        )
        # KEEP-ALIVE. Every pointer above is an untracked `unsafe_ptr()`, which
        # is not a use of its list, and Mojo frees a value at its LAST USE: the
        # last use of `m` was the `m.bias` argument and of each local copy its
        # `unsafe_ptr()`, so all of them were freed before the walk ran and
        # `gbdt_host_predict` read the allocator's free-list word through
        # `tree_offsets_p[0]`, refusing every fixture on all seven runners of
        # gate run 34889886781 ("tree_offsets must start at 0 and end at
        # n_splits"). `_parse_model` makes that check true by construction
        # (one `split` per level, `cur_splits == cur_depth` enforced per tree),
        # so only a dangling read could fail it. A use after the call keeps
        # every list alive until the walk has returned.
        _ = split_feature^
        _ = split_bin^
        _ = split_take_bin^
        _ = node_left^
        _ = node_right^
        _ = borders^
        _ = leaves^
        _ = x^
        _ = m^
        for i in range(n_rows):
            op[i] = out[i]
    return PythonObject(n_rows)


def gbdt_model_dim_binding(model: PythonObject) raises -> PythonObject:
    """`gbdt_model_dim` (`bindings/_mojolearn_gbdt.mojo:442-450`): 1 for an
    empty ensemble, else the trees' common dim."""
    return PythonObject(_parse_model(String(py=model)).dim)


def gbdt_sigmoid_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """`gbdt_sigmoid_binding` (`bindings/_mojolearn_gbdt.mojo:133-148`),
    its body: `1 / (1 + identical_exp64(-raw))` in double."""
    var rp = f64_ptr(Int(py=raw_addr))
    var op = f64_ptr(Int(py=out_addr))
    var count = Int(py=n)
    for i in range(count):
        var r = rp.unsafe_load(i)
        op.unsafe_store(i, 1.0 / (1.0 + identical_exp64(-r)))
    return PythonObject(count)


@export
def PyInit__mojolearn_gbdt_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_gbdt_host")
        module.def_function[gbdt_host_numeric_mode_binding]("gbdt_host_numeric_mode")
        module.def_function[gbdt_host_vendor_binding]("gbdt_host_vendor")
        module.def_function[gbdt_host_column_binding]("gbdt_host_column")
        module.def_function[gbdt_host_sabotage_binding]("gbdt_host_sabotage")
        module.def_function[gbdt_vendor_binding]("gbdt_vendor")
        module.def_function[gbdt_numeric_mode_binding]("gbdt_numeric_mode")
        module.def_function[gbdt_fit_binding]("gbdt_fit")
        module.def_function[gbdt_predict_binding]("gbdt_predict")
        module.def_function[gbdt_model_dim_binding]("gbdt_model_dim")
        module.def_function[gbdt_sigmoid_binding]("gbdt_sigmoid")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_gbdt_host: ", error))
