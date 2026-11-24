# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/_mojolearn_trees_host.mojo` (lane bm-trees, 2026-11-22):
the host ExtraTrees fit (`et_host_fit_result`, what `ExtraTrees*.fit`,
`DecisionTree*(splitter='random')` and `RandomTreesEmbedding` fit on the CPU
column), its export registry, the sequential predict walk and the image's
forest entry operations (`EtHostForest`), as typed Mojo. Imports no Python
and never touches the GIL.

ONE BODY, TWO ENTRIES (the gbdt host image's pattern, lane fx-gil): the
binding's published entry `et_host_forest_entry_v1` releases the GIL around
the whole dispatch of a long operation (`forest_host_entry_releases`: the
fit, the sequential predict and the strict predict, the bodies that released
it before) and runs it with `ENTRY_OP_GIL_FREE`; `et_host_forest_entry_native`
below is the same dispatch for a caller that holds no GIL (a Mojo check that
publishes it with `publish_et_host_forest`). The registries an operation
reaches each have their own lock word (core/forest_export_run.mojo,
core/forest_host_strict_run.mojo, core/forest_host_groves_run.mojo)."""
from core.weight_compact_host import (
    host_gather_colmajor,
    host_gather_f32,
    host_nonzero_weight_rows,
)
from core.forest_export_run import ForestExportRegistry
from core.forest_export_binding_impl import validate_forest_export_destinations
from core.mojolearn_forest_host_impl import forest_host_vote_into
from core.mojolearn_rf_host_impl import rf_host_col_major
from core.mojolearn_trees_host_impl import et_export_into, et_forest_lists
from core.hostptr import f32_ptr, i32_ptr, read_f32
from core.forest_host_strict_run import (
    forest_host_strict_predict, forest_strict_prepare_host_native, forest_strict_release_host_native,
)
from core.forest_host_groves_run import (
    forest_prepare_host_native, forest_predict_resident_host_native, forest_release_host_native,
)
from core.forest_fit_request import ForestFitArgs, XF_CAP_STRICT, forest_export_verify_lists, fit_loop_put
from core.forest_entry import ForestImage, forest_entry_dispatch, forest_entry_publish
from core.forest_entry_table import FOREST_TABLE_ET
from core.native_descriptor_abi import NativeDescriptorError
from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
)
from extratrees.host_estimator import (
    fit_extra_trees_classifier_host_exact,
    fit_extra_trees_regressor_host_exact,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_GAMMA,
    CRITERION_INVERSE_GAUSSIAN,
    CRITERION_MSE,
    CRITERION_POISSON,
)
from core.shared_slot import SharedSlot


comptime N_FIT_PARAMS = 22
"""`bindings/_mojolearn_trees.mojo::N_FIT_PARAMS`, the same 22 slots in the
same order (that docstring is the contract; slot 20 is `device`, 21 the
criterion code)."""

# V2 (lane fx-gil, 2026-11-25): the registry gained its lock word.
comptime ET_HOST_EXPORTS = SharedSlot[ForestExportRegistry[FitResult], "MojolearnETFitExportHostV4", 3]


def _config_from(
    params: ForestFitArgs, base: ExtraTreesConfig
) raises -> ExtraTreesConfig:
    """`bindings/_mojolearn_trees.mojo::_config_from`, the same slots written
    OVER `base` in the same order (DEVIATION 458's lesson: every slot
    overwrites its field after the base is taken)."""
    var config = base.copy()
    config.n_estimators = Int32(params.i(3))
    config.max_depth = Int32(params.i(4))
    config.min_samples_split = Int32(params.i(5))
    config.min_samples_leaf = Int32(params.i(6))
    config.min_weight_fraction_leaf = params.f(7)
    config.max_features_spec = params.i(8)
    config.max_features_fraction = params.f(9)
    config.min_impurity_decrease = Float32(params.f(10))
    config.bootstrap = params.i(11) != 0
    config.oob_score = params.i(12) != 0
    config.random_state = UInt64(params.i(13))
    config.warm_start = params.i(14) != 0
    config.ccp_alpha = params.f(15)
    config.has_class_weight = params.i(16) != 0
    config.has_monotonic_cst = params.i(17) != 0
    config.max_samples = params.i(18)
    config.max_leaf_nodes = Int32(params.i(19))
    config.criterion = Int32(params.i(21))
    return config^


def _read_x_col_major(
    x_addr: Int, n_rows: Int, n_features: Int, row_major: Bool
) raises -> List[Float32]:
    """The design as the trainer reads it, COLUMN-major. A ROW-major X (the
    `_rowmajor` entries, DEVIATION 2637) is transposed here on the host:
    the same bytes the GPU binding's pinned stage holds, moved and folded
    through nothing."""
    var flat = read_f32(x_addr, n_rows * n_features)
    return rf_host_col_major(flat^, n_rows, n_features, row_major)


def et_host_retain(var result: FitResult) raises -> Int:
    """`bindings/_mojolearn_trees.mojo::_retain_et_export`: the fitted
    forest kept under a handle until `forest_export_release`."""
    var trees = len(result.forest.trees)
    var nodes = 0
    for tree in result.forest.trees:  # driver-loop: over fitted trees, one node count each (export shape bookkeeping)
        nodes += tree.num_nodes()
        if len(tree.vector_leaf) != tree.num_nodes() * Int(result.forest.num_outputs):
            raise Error("fitted ET leaf storage differs from export dimensions")
    var meta = et_legacy_meta(result)
    var outputs = Int(result.forest.num_outputs)
    if trees != Int(result.forest.n_trees):
        raise Error("fitted ET tree metadata differs from export dimensions")
    return ET_HOST_EXPORTS.get_or_create_ptr()[].insert_native(
        result^, trees, nodes, outputs, meta^
    )


def et_legacy_meta(result: FitResult) -> List[Int64]:
    """The six metadata words `_forest_out` reports (and the export keeps)."""
    var meta: List[Int64] = [Int64(result.forest.n_trees), Int64(result.forest.num_outputs),
        Int64(1 if result.depth_cap_bound else 0), Int64(result.plan.params.max_depth),
        Int64(result.plan.max_features_count), Int64(result.plan.n_sampled_rows)]
    return meta^


def et_host_fit_result[CLASSIFIER: Bool, ROWMAJOR: Bool](params: ForestFitArgs) raises -> FitResult:
    """`_et_classifier_fit` / `_et_regressor_fit` of the GPU binding: the
    same slot checks in the same words, then the host fit. `tree_start` is
    the GPU binding's global tree ID offset (the shard fits). No GIL: the
    binding releases it around this call when its caller holds it."""
    var tree_start = params.tree_start
    var weights_addr = params.weights
    comptime entry = "et_classifier_fit" if CLASSIFIER else "et_regressor_fit"
    if len(params) != N_FIT_PARAMS:
        raise Error(
            entry + ": params must hold "
            + String(N_FIT_PARAMS)
            + " values, got "
            + String(len(params))
        )
    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_classes = params.i(2)
    comptime if not CLASSIFIER:
        if n_classes != 0:
            raise Error("et_regressor_fit: n_classes (slot 2) must be 0")
    if params.f(20) != Float64(1):
        raise Error(
            entry + ": device (slot 20) must be 1, the value the wrapper"
            " sends; the host binding fits the same forest the GPU fits"
        )
    var config: ExtraTreesConfig
    comptime if CLASSIFIER:
        config = _config_from(params, ExtraTreesConfig())
        if config.criterion != CRITERION_GINI and config.criterion != CRITERION_ENTROPY:
            raise Error(
                "et_classifier_fit: criterion (slot 21) must be GINI (0) or"
                " ENTROPY (1); got " + String(config.criterion)
            )
    else:
        config = _config_from(params, ExtraTreesConfig().for_regression())
        if config.criterion != CRITERION_MSE and config.criterion != CRITERION_POISSON and (
        config.criterion != CRITERION_GAMMA and config.criterion != CRITERION_INVERSE_GAUSSIAN
    ):
            raise Error(
                "et_regressor_fit: criterion (slot 21) must be MSE (2), POISSON (4), GAMMA (5) or INVERSE_GAUSSIAN (6); got "
                + String(config.criterion)
            )
    if n_rows < 1 or n_features < 1:
        raise Error(entry + ": n_rows and n_features must be >= 1")
    var x_address = params.x
    var y_address = params.y
    var result: FitResult
    var x = _read_x_col_major(x_address, n_rows, n_features, ROWMAJOR)
    var y = read_f32(y_address, n_rows)
    # lane sk-kernels / sk-final: the per-row weights (0: none)
    var w = List[Float32]()
    if weights_addr != 0:
        w = read_f32(weights_addr, n_rows)
    # lane sk-weq-trees: without bootstrap the zero-weight rows leave
    # the fit, the device's `DeviceDataset.drop_zero_weight_rows` (same
    # rows, same ascending order); the plan then sees the kept count, as
    # the device fit's does (`device_count_nonzero_weights`)
    var n_fit = n_rows
    if len(w) > 0 and not config.bootstrap:
        var rows = host_nonzero_weight_rows(w)
        if len(rows) > 0 and len(rows) < n_rows:
            x = host_gather_colmajor(x, n_rows, n_features, rows)
            y = host_gather_f32(y, rows)
            w = host_gather_f32(w, rows)
            n_fit = len(rows)
    comptime if CLASSIFIER:
        result = fit_extra_trees_classifier_host_exact(
            x, y, Int32(n_fit), Int32(n_features), Int32(n_classes), config,
            tree_start, w^,
        )
    else:
        result = fit_extra_trees_regressor_host_exact(
            x, y, Int32(n_fit), Int32(n_features), config, tree_start,
            w^,
        )
    return result^


def et_host_fit_handle[CLASSIFIER: Bool, ROWMAJOR: Bool](params: ForestFitArgs) raises -> Int:
    """One fit kept in ET_HOST_EXPORTS (the forest entry's XF_FIT): the
    export handle. The fit loop words the caller asked for (XF_W_LOOP) are
    written first."""
    var result = et_host_fit_result[CLASSIFIER, ROWMAJOR](params)
    fit_loop_put(params.loop, result.forest.fit_loop)
    return et_host_retain(result^)


def et_host_export_into(id: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
    """XF_EXPORT_INTO: the retained forest into the caller's five arrays."""
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    registry[].validate(id, trees, nodes, outputs)
    validate_forest_export_destinations(dst[0], dst[1], dst[2], dst[3], dst[4])
    var op = i32_ptr(dst[0])
    var cp = i32_ptr(dst[1])
    var tp = f32_ptr(dst[2])
    var lp = i32_ptr(dst[3])
    # THE LOCK (core/forest_export_run.mojo): the entry is taken out, read with
    # no lock held and put back on every path.
    var entry = registry[].take(id)
    var failed = False
    var words = String("")
    try:
        et_export_into(entry.model, op, cp, tp, lp, dst[4], entry.outputs)
    except problem:
        failed = True
        words = String(problem)
    registry[].put_back(id, entry^)
    if failed:
        raise Error(words)


def et_host_export_verify(id: Int, dst: List[Int]) raises -> Int:
    """XF_EXPORT_VERIFY (MOJOLEARN_FOREST_EXPORT=verify): the diagnostic
    lists `forest_export_legacy` returns, against the exported arrays."""
    var registry = ET_HOST_EXPORTS.get_or_create_ptr()
    # THE LOCK (core/forest_export_run.mojo): the entry is taken out, read with
    # no lock held and put back on every path.
    var entry = registry[].take(id)
    var offsets = List[Int64]()
    var colid = List[Int64]()
    var quesval = List[Float64]()
    var left_child = List[Int64]()
    var leaves = List[Float64]()
    var same = True
    var failed = False
    var words = String("")
    try:
        var legacy = et_legacy_meta(entry.model)
        if len(legacy) != len(entry.meta):
            same = False
        else:
            for k in range(len(legacy)):  # small-loop(meta: six export metadata words)
                if legacy[k] != entry.meta[k]:
                    same = False
        if same:
            et_forest_lists(entry.model, offsets, colid, quesval, left_child, leaves)
    except problem:
        failed = True
        words = String(problem)
    registry[].put_back(id, entry^)
    if failed:
        raise Error(words)
    if not same:
        return -1
    return forest_export_verify_lists(offsets, colid, quesval, left_child, leaves, dst)


def et_host_predict_walk(model: List[Int], x_address: Int, out_addr: Int, n_rows: Int, n_features: Int,
                         n_trees: Int, num_outputs: Int) raises -> Int:
    """`et_predict`'s contract (XF_PREDICT): the forest's averaged vote per
    row of a ROW-major `x`, `out` is `n_rows * num_outputs` float32; returns
    rows written. The walk is `core/forest_host_predict.mojo`'s, which
    rebuilds the trees from the five arrays and refuses a malformed node
    instead of reading past it."""
    if n_trees < 1 or num_outputs < 1:
        raise Error("et_predict: n_trees and num_outputs must be >= 1")
    if n_rows < 0 or n_features < 1:
        raise Error("et_predict: n_rows must be >= 0 and n_features >= 1")
    var offsets_p = i32_ptr(model[0])
    var colid_p = i32_ptr(model[1])
    var quesval_p = f32_ptr(model[2])
    var left_p = i32_ptr(model[3])
    var leaves_p = f32_ptr(model[4])
    var op = f32_ptr(out_addr)
    var n_nodes = Int(offsets_p[n_trees])
    if Int(offsets_p[0]) != 0 or n_nodes < n_trees:
        raise Error("et_predict: tree_offsets must start at 0 and hold at least one node per tree")
    if n_rows == 0:
        return 0
    var rows = read_f32(x_address, n_rows * n_features)
    forest_host_vote_into(
        False, rows, offsets_p, colid_p, quesval_p, left_p,
        leaves_p, op, n_rows, n_features, n_trees, num_outputs, n_nodes,
    )
    return n_rows


struct EtHostForest(ForestImage):
    """This image's forest entry operations (core/forest_entry.mojo; lane
    calltable-forest). The host column scans X itself, keeps its trees in
    the strict snapshot and has no ordered resident word or resident
    labels. Every body runs bare (see the module docstring)."""

    @staticmethod
    def caps() -> Int:
        return XF_CAP_STRICT

    @staticmethod
    def dt_default_bins() -> Int:
        return 128

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        if classifier:
            if row_major:
                return et_host_fit_handle[True, True](args)
            return et_host_fit_handle[True, False](args)
        if row_major:
            return et_host_fit_handle[False, True](args)
        return et_host_fit_handle[False, False](args)

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        raise Error("RuntimeError: the extra-trees binding has no DART member fit")

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        ET_HOST_EXPORTS.get_or_create_ptr()[].sizes_into(handle, out_addr)

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        et_host_export_into(handle, dst, trees, nodes, outputs)

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        return et_host_export_verify(handle, dst)

    @staticmethod
    def export_release(handle: Int) raises:
        ET_HOST_EXPORTS.get_or_create_ptr()[].release(handle)

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool, free: Bool) raises -> Int:
        return et_host_predict_walk(model, x, out_addr, rows, cols, trees, outputs)

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        return forest_prepare_host_native[False](model, features, trees, outputs, ordered)

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        return forest_predict_resident_host_native[False](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        raise Error("RuntimeError: this forest binding has no resident label prediction")

    @staticmethod
    def resident_release(handle: Int) raises:
        forest_release_host_native[False](handle)

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        return forest_strict_prepare_host_native[False](model, features, trees, outputs)

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int,
                       free: Bool) raises -> Int:
        return forest_host_strict_predict[False](handle, x, out_addr, rows, cols, outputs)

    @staticmethod
    def strict_release(handle: Int) raises:
        forest_strict_release_host_native[False](handle)

    @staticmethod
    def session_open(x: Int, rows: Int, cols: Int, row_major: Bool, share_tables: Bool) raises -> Int:
        raise Error("RuntimeError: the extra-trees host binding has no forest data session")

    @staticmethod
    def session_close(handle: Int) raises:
        raise Error("RuntimeError: the extra-trees host binding has no forest data session")

    @staticmethod
    def fit_session(classifier: Bool, args: ForestFitArgs) raises -> Int:
        raise Error("RuntimeError: the extra-trees host binding has no forest data session")

    @staticmethod
    def ada_open(handle: Int, w: Int, y: Int, cw: Int, n: Int, k: Int) raises:
        raise Error("RuntimeError: the extra-trees host binding has no AdaBoost device rounds")

    @staticmethod
    def fit_ada_round(classifier: Bool, args: ForestFitArgs) raises -> Int:
        raise Error("RuntimeError: the extra-trees host binding has no AdaBoost device rounds")

    @staticmethod
    def stage_input(x: Int, rows: Int, cols: Int) raises -> Int:
        """XF_STAGE_INPUT: a host image has no device input to stage."""
        return 0

    @staticmethod
    def unstage_input() raises:
        pass


    @staticmethod
    def fit_rows(classifier: Bool, args: ForestFitArgs) raises -> Int:
        # XF_FIT_ROWS (lane vdf-scope-trees): the ExtraTrees host image keeps no device store
        # (no XF_CAP_DEV_ROWS), so no caller sends it.
        raise Error("RuntimeError: mojolearn: this forest image fits no device-store rows (XF_FIT_ROWS)")

    @staticmethod
    def resident_rows(mode: Int, handle: Int, store_id: Int, idx: Int, out_addr: Int, rows: Int,
                      cols: Int, outputs: Int) raises -> Int:
        # XF_RESIDENT_ROWS (lane vdf-scope-trees2a): the ExtraTrees host image keeps no device store
        # (no XF_CAP_DEV_ROWS), so no caller sends it.
        raise Error("RuntimeError: mojolearn: this forest image predicts no device-store rows (XF_RESIDENT_ROWS)")

def et_host_forest_entry_native(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The image's forest entry for a caller that holds no GIL (a Mojo check;
    the binding publishes `et_host_forest_entry_v1`, which releases)."""
    return forest_entry_dispatch[EtHostForest](operation, aa, na, pp, np, ff, nf, result, context, sink)


def publish_et_host_forest() raises:
    """Publish `et_host_forest_entry_native` under the ET table's cpu key, for
    a process with no `_mojolearn_trees_host` loaded (a Mojo check). The
    binding publishes its own GIL-releasing entry under the same key."""
    forest_entry_publish["MojolearnForestEntryOwnerEtHostNativeV1"](FOREST_TABLE_ET, "cpu", et_host_forest_entry_native)
