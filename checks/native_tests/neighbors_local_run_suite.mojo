# SPDX-License-Identifier: Apache-2.0
"""Neighbors local door RUN (`neighbors/neighbors_local_run.mojo`) without
Python: LocalOutlierFactor, NearestCentroid and KNNImputer.

Each fixture is one typed check, plan or host buffer step: accepted and
refused values, the refusal wording, plan fields and buffer contents.
Independent fixtures run across the host's cores.
"""
from core.host_parallel import host_parallelize
from neighbors.neighbors_local_run import (
    LOF_RESULT_SCORES, LOF_RESULT_DECISION, LOF_RESULT_LABELS, LOF_AUTO_OFFSET,
    local_check_n_neighbors, local_copy_f32, local_write_i32,
    lof_refuse_metric, lof_refuse_metric_params, lof_check_contamination, lof_check_fit_rows,
    lof_fit_neighbors, lof_plan, lof_refuse_fit_predict_novelty, lof_check_novelty,
    nc_refuse_metric, nc_check_label_rows, nc_check_classes, nc_check_priors_ndim, nc_check_priors_size,
    nc_check_variance, nc_refuse_decision_metric, nc_fused, nc_proba_mode,
    imp_refuse_missing_values, imp_refuse_metric, imp_refuse_weights, imp_round_f32, imp_cells_length,
    imp_colmiss_length, imp_read_count, imp_fit_columns, imp_check_features, imp_cells_operation, imp_keep_plan,
)

comptime NEIGHBORS_LOCAL_RUN_FIXTURES = 22


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def fixture(unit: Int) raises:
    if unit == 0:
        # accepted values pass every check
        local_check_n_neighbors(1)
        lof_check_contamination(0.5)
        lof_check_contamination(0.01)
        lof_check_fit_rows(2, 3)
        lof_check_novelty(True, "predict")
        nc_check_label_rows(4, 4)
        nc_check_classes(2)
        nc_check_priors_ndim(1)
        nc_check_priors_size(3, 3)
        nc_check_variance(False)
        imp_check_features(5, 5)
        require(LOF_RESULT_SCORES == 0 and LOF_RESULT_DECISION == 1 and LOF_RESULT_LABELS == 2, "result kinds")
        require(LOF_AUTO_OFFSET == -1.5, "auto offset")
    elif unit == 1:
        try:
            local_check_n_neighbors(0)
            raise Error("accepted")
        except e:
            require(String(e) == "n_neighbors must be >= 1", "n_neighbors refusal wording")
    elif unit == 2:
        try:
            lof_refuse_metric("'cosine'", "2")
            raise Error("accepted")
        except e:
            require(String(e) == "NotImplementedError: LocalOutlierFactor: metric='cosine' (p=2) is not "
                    + "implemented; euclidean (or minkowski with p=2) only", "metric refusal wording")
        try:
            lof_refuse_metric_params()
            raise Error("accepted")
        except e:
            require(String(e) == "NotImplementedError: LocalOutlierFactor: metric_params is not implemented",
                    "metric_params refusal wording")
    elif unit == 3:
        var refused: List[Float64] = [0.0, 0.51, -1.0]
        for i in range(len(refused)):
            try:
                lof_check_contamination(refused[i])
                raise Error("accepted")
            except e:
                require(String(e) == "contamination must be 'auto' or in (0, 0.5]", "contamination refusal")
    elif unit == 4:
        try:
            lof_check_fit_rows(1, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "Found array with 1 sample(s) (shape=(1, 3)) while a minimum of 2 is required.",
                    "fit rows refusal wording")
    elif unit == 5:
        require(lof_fit_neighbors(20, 10) == 9, "k is capped at n - 1")
        require(lof_fit_neighbors(3, 10) == 3, "k below the cap is kept")
    elif unit == 6:
        var plan = lof_plan(7, 8, 3, 2, True, False, True)
        require(plan.rows == 7 and plan.reference_rows == 8 and plan.features == 3 and plan.neighbors == 2,
                "plan shape")
        require(plan.exclude_self and not plan.unfused and plan.old_items, "fused plan keeps old items")
        var unfused = lof_plan(7, 8, 3, 2, False, True, True)
        require(not unfused.exclude_self and unfused.unfused and not unfused.old_items,
                "the unfused arm drops old items")
    elif unit == 7:
        try:
            lof_refuse_fit_predict_novelty()
            raise Error("accepted")
        except e:
            require(String(e) == "AttributeError: fit_predict is not available when novelty=True; use predict on new data",
                    "fit_predict novelty refusal")
        try:
            lof_check_novelty(False, "score_samples")
            raise Error("accepted")
        except e:
            require(String(e) == "AttributeError: score_samples is not available when novelty=False; use fit_predict",
                    "novelty refusal")
    elif unit == 8:
        try:
            nc_refuse_metric()
            raise Error("accepted")
        except e:
            require(String(e) == "NearestCentroid: metric must be 'euclidean' or 'manhattan'", "nc metric refusal")
        try:
            nc_refuse_decision_metric()
            raise Error("accepted")
        except e:
            require(String(e) == "AttributeError: decision_function is available for metric='euclidean' only",
                    "nc decision refusal")
    elif unit == 9:
        try:
            nc_check_label_rows(3, 4)
            raise Error("accepted")
        except e:
            require(String(e) == "X and y have different numbers of rows", "label rows refusal")
        try:
            nc_check_classes(1)
            raise Error("accepted")
        except e:
            require(String(e) == "The number of classes has to be greater than one; got 1 class", "classes refusal")
    elif unit == 10:
        try:
            nc_check_priors_ndim(2)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: priors must be 1-D, got 2-D", "priors ndim refusal")
        try:
            nc_check_priors_size(2, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "priors must have one entry per class", "priors size refusal")
    elif unit == 11:
        try:
            nc_check_variance(True)
            raise Error("accepted")
        except e:
            require(String(e) == "All features have zero variance. Division by zero.", "variance refusal")
    elif unit == 12:
        require(nc_fused(True, 3) and not nc_fused(True, 0) and not nc_fused(False, 3), "fused needs rows")
        require(nc_proba_mode(True, True) == 1 and nc_proba_mode(False, True) == 2
                and nc_proba_mode(False, False) == 0, "probability modes")
    elif unit == 13:
        try:
            imp_refuse_missing_values()
            raise Error("accepted")
        except e:
            require(String(e) == "NotImplementedError: KNNImputer: missing_values must be a number or NaN (float input)",
                    "missing_values refusal")
        try:
            imp_refuse_metric()
            raise Error("accepted")
        except e:
            require(String(e) == "NotImplementedError: KNNImputer: metric must be 'nan_euclidean'", "imputer metric")
        try:
            imp_refuse_weights()
            raise Error("accepted")
        except e:
            require(String(e) == "NotImplementedError: KNNImputer: weights must be 'uniform' or 'distance'",
                    "imputer weights")
    elif unit == 14:
        try:
            imp_check_features(4, 5)
            raise Error("accepted")
        except e:
            require(String(e) == "X has a different number of features than during fit", "features refusal")
    elif unit == 15:
        require(imp_round_f32(0.1) == Float64(Float32(0.1)), "missing value rounds once to float32")
        require(imp_round_f32(2.0) == 2.0, "exact value kept")
    elif unit == 16:
        require(imp_cells_length(3, 4, False) == 12 and imp_cells_length(0, 4, False) == 1
                and imp_cells_length(3, 4, True) == 1, "cell list length")
        require(imp_colmiss_length(0) == 1 and imp_colmiss_length(5) == 5, "column count length")
        require(imp_cells_operation(True) == "knn_impute_cells" and imp_cells_operation(False) == "knn_impute_tiled",
                "cells operation arm")
    elif unit == 17:
        var info: List[Int32] = [7]
        require(imp_read_count(Int(info.unsafe_ptr())) == 7, "info count")
        _ = info
    elif unit == 18:
        # counts per column over n = 3 rows: none missing, some, all
        var counts: List[Int32] = [0, 2, 3, 1]
        var columns = imp_fit_columns(Int(counts.unsafe_ptr()), 4, 3)
        _ = counts
        require(len(columns.valid) == 4 and columns.valid[0] and columns.valid[1] and not columns.valid[2]
                and columns.valid[3], "valid columns")
        require(len(columns.missing) == 3 and columns.missing[0] == 1 and columns.missing[1] == 2
                and columns.missing[2] == 3, "missing columns")
    elif unit == 19:
        var present: List[Bool] = [True, False, True]
        var plan = imp_keep_plan(present)
        require(len(plan.keep) == 2 and plan.keep[0] == 0 and plan.keep[1] == 2, "kept columns")
        require(len(plan.flags) == 3 and plan.flags[0] == 0 and plan.flags[1] == 1 and plan.flags[2] == 0,
                "empty flags")
        require(not plan.all_valid, "a dropped column")
        var all_present: List[Bool] = [True, True]
        require(imp_keep_plan(all_present).all_valid, "every column valid")
    elif unit == 20:
        var src: List[Float32] = [1.0, -2.5, 3.25]
        var dst = List[Float32](length=3, fill=0)
        local_copy_f32(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 3)
        _ = src
        require(dst[0] == 1.0 and dst[1] == -2.5 and dst[2] == 3.25, "float32 copy")
    else:
        var values: List[Int] = [4, 0, 9]
        var dst = List[Int32](length=3, fill=0)
        local_write_i32(values, Int(dst.unsafe_ptr()))
        require(dst[0] == 4 and dst[1] == 0 and dst[2] == 9, "int32 words")


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_LOCAL_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_local_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_LOCAL_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_local_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_local_run:", NEIGHBORS_LOCAL_RUN_FIXTURES, "typed check, plan and buffer fixtures")
