# SPDX-License-Identifier: Apache-2.0
"""Every multi-GPU driver is one native call (source contracts).

docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md. Each driver's body is a single
`<module>._NATIVE(...)` call naming its door operation (argument checks and
the plain method's public conversion through `_parallel_pool`'s shell
function aside); the door dispatches that operation and, for a cooperative
driver, binds the device plan in its family's binding around the plain owner
and releases it in a `finally`. The family's GPU binding registers the plan
entries. Read without importing the package or loading a binding; the
numerical contract (equal bytes to the one-device call at one and two
devices) is test_parallel_public_drivers.py and the par-* identity lanes.
"""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "python" / "mojolearn"

#: (driver module, function or Class.method, door operation, door file,
#:  GPU binding source registering the plan entries or None)
DRIVERS = (
    ("parallel_classical", "fit_kernel_method", "family_km_par_fit", "bindings/kernel_methods_native.mojo",
     "bindings/_mojolearn_kernel_methods.mojo"),
    ("parallel_classical", "apply_kernel_method", "family_km_par_apply", "bindings/kernel_methods_native.mojo",
     "bindings/_mojolearn_kernel_methods.mojo"),
    ("parallel_classical", "fit_cholesky", "family_chol_par_fit", "bindings/linalg_native.mojo",
     "bindings/_mojolearn_gp.mojo"),
    ("parallel_classical", "solve_cholesky", "family_chol_par_solve", "bindings/linalg_native.mojo",
     "bindings/_mojolearn_gp.mojo"),
    ("parallel_classical", "fit_svm", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_svm.mojo"),
    ("parallel_classical", "predict_svm", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_svm.mojo"),
    ("parallel_classical", "fit_gaussian_process", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_gp.mojo"),
    ("parallel_classical", "predict_gaussian_process", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_gp.mojo"),
    ("parallel_classical", "fit_logistic", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_estimators.mojo"),
    ("parallel_classical", "fit_coordinate_descent", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_solver.mojo"),
    ("parallel_classical", "fit_gram_estimator", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_estimators.mojo"),
    ("parallel_classical", "fit_kmeans", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn.mojo"),
    ("parallel_classical", "fit_dbscan", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_estimators.mojo"),
    ("parallel_classical", "fit_gaussian_mixture", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_mixture.mojo"),
    ("parallel_classical", "predict_gaussian_mixture", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_mixture.mojo"),
    ("parallel_classical", "fit_hdbscan", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_hdbscan.mojo"),
    ("parallel_classical", "bootstrap", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_resample.mojo"),
    ("parallel_classical", "permutation_test", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_resample.mojo"),
    ("parallel_classical", "monte_carlo_integrate", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_resample.mojo"),
    ("parallel_graph", "fit_graph", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_metrics.mojo"),
    ("parallel_graph", "transform_umap", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_metrics.mojo"),
    ("parallel_ensemble", "fit_boosting", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_gbdt.mojo"),
    ("parallel_ensemble", "fit_feature_freq", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_gbdt.mojo"),
    ("parallel_ensemble", "fit_ordered_rmse", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_gbdt.mojo"),
    ("parallel_ensemble", "fit_isolation_forest", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_svm.mojo"),
    ("parallel_ensemble", "score_isolation_forest", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_svm.mojo"),
    ("parallel_ensemble", "ParallelForestPredictor.predict", "family_par_run", "bindings/family_native.mojo",
     "bindings/_mojolearn_rf.mojo"),
    ("parallel_ensemble", "ParallelForestPredictor.predict_proba", "family_par_run",
     "bindings/family_native.mojo", "bindings/_mojolearn_trees.mojo"),
    ("parallel_classical", "fit_arima", "fit_arima", "bindings/parallel_forecast_native.mojo", None),
    ("parallel_classical", "fit_exponential_smoothing", "fit_exponential_smoothing",
     "bindings/parallel_forecast_native.mojo", None),
)

#: (driver name in `family_par_run`, plain owner operations it may run)
PAR_RUN = (
    ("svm_fit", ("family_svc_svc_fit", "family_svc_svr_fit")),
    ("svm_predict", ("family_svc_svc_predict", "family_svc_svc_decision", "family_svc_svr_predict")),
    ("gp_fit", ("family_gp_gpr_fit",)),
    ("gp_predict", ("family_gp_gpr_predict",)),
    ("glm_fit", ("fit",)),
    ("solver_fit", ("cd_fit",)),
    ("gram_fit", ("ols_fit", "ridge_fit", "fit")),
    ("kmeans_fit", ("family_kmeans_fit",)),
    ("dbscan_fit", ("family_cfb_dbscan_fit",)),
    ("gmm_fit", ("family_mix_gmm_fit",)),
    ("gmm_predict", ("family_mix_gmm_score_samples", "family_mix_gmm_predict_proba", "family_mix_gmm_predict")),
    ("hdbscan_fit", ("family_cfa_hdbscan_fit",)),
    ("resample", ("family_rs_bootstrap", "family_rs_permutation_test", "family_rs_monte_carlo_integrate")),
    ("graph_fit", ("family_cfb_agglo_fit", "family_cfb_sc_fit", "family_mani_umap_fit")),
    ("umap_transform", ("family_mani_umap_transform",)),
    ("gbdt_fit", ("trees_gbdt_fit", "trees_gbdt_adapter_classifier_fit", "trees_gbdt_adapter_regressor_fit",
                  "trees_gbdt_feature_freq_fit")),
    ("ordered_rmse_fit", ("trees_gbdt_ordered_fit",)),
    ("iforest_fit", ("trees_iforest_fit",)),
    ("iforest_score", ("trees_iforest_score_samples", "trees_iforest_decision_function", "trees_iforest_predict")),
    ("forest_predict", ("trees_public_forest_predict", "trees_public_forest_proba")),
)


def _read(relative):
    return (REPO / relative).read_text(encoding="utf-8")


def _function(module, name):
    tree = ast.parse((PKG / (module + ".py")).read_text(encoding="utf-8"))
    owner, _, method = name.partition(".")
    for node in tree.body:
        if method and isinstance(node, ast.ClassDef) and node.name == owner:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return item
        if not method and isinstance(node, ast.FunctionDef) and node.name == owner:
            return node
    raise AssertionError(module + "." + name + " not found")


def _native_ops(function):
    ops = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            target = node.func
            label = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if label.startswith("_native_parallel"):
                # the parallel owners' `(operation, args)` entries
                assert isinstance(node.args[0], ast.Constant), ast.dump(node)
                ops.append(node.args[0].value)
            elif label.startswith(("_NATIVE", "_native_", "_FAMILY")):
                assert isinstance(node.args[1], ast.Constant), ast.dump(node)
                ops.append(node.args[1].value)
    return ops


def test_each_driver_is_one_native_call_with_no_pool_or_loop():
    for module, name, op, _, _ in DRIVERS:
        function = _function(module, name)
        assert _native_ops(function) == [op], (module, name)
        names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
        assert "DevicePool" not in names, (module, name)
        loops = [n for n in ast.walk(function) if isinstance(n, (ast.For, ast.While, ast.comprehension))]
        assert not loops, (module, name)


def test_each_door_dispatches_its_operation():
    for module, name, op, door, _ in DRIVERS:
        assert 'operation == "' + op + '"' in _read(door), (op, door)


def test_sharded_drivers_keep_the_partition_and_the_negative_control():
    common = _read("bindings/parallel_shard_common.mojo")
    shift = common[common.index("def shard_read_shift("):common.index("def shard_run(")]
    # `_parallel_pool.driver_read_shift`, word for word
    assert "if first < 1 or n_devices < 2 or index % n_devices == 0:" in shift
    assert '"MOJOLEARN_PAR_DRIVER_SABOTAGE"' in shift and '"MOJOLEARN_HOST_ALLOW_SABOTAGE"' in shift
    ranges = common[common.index("def shard_ranges("):common.index("def shard_devices(")]
    assert "out.append(min(i + w, count))" in ranges and "i += w" in ranges
    forecast = _read("bindings/parallel_forecast_native.mojo")
    for driver in ("_fit_arima", "_fit_hw"):
        body = forecast[forecast.index("def " + driver + "("):]
        body = body[:body.index("\n\ndef ")]
        # every shard reads from its shifted start and is placed at its true offset
        assert "shard_read_shift(r, lo, n_devices)" in body
        assert "shard_rows(data, lo - shift, hi - shift, context)" in body


def test_each_door_releases_its_plan_in_a_finally():
    for _, _, _, door, _ in DRIVERS:
        source = _read(door)
        start = 0
        while True:
            at = source.find("plan_enter(", start)
            if at < 0:
                break
            body = source[at:]
            tail = body[:body.index("\n\ndef ")] if "\n\ndef " in body else body
            assert tail.index("try:") < tail.index("finally:") < tail.index("plan_exit("), door
            start = at + 1


def test_each_gpu_binding_registers_the_plan_entries():
    for _, _, _, _, binding in DRIVERS:
        if binding is None:
            continue
        source = _read(binding)
        assert 'm.def_function[device_plan_enter_binding]("device_plan_enter")' in source, binding
        assert 'm.def_function[device_plan_exit_binding]("device_plan_exit")' in source, binding


def test_par_run_drivers_name_only_their_spec_owners():
    specs = _read("bindings/parallel_plan_specs.mojo")
    body = specs[specs.index("def par_spec_admits("):specs.index("def par_spec_families(")]
    for driver, owners in PAR_RUN:
        block = body[body.index('if driver == "' + driver + '":'):]
        block = block[:block.index("\n    if ") if "\n    if " in block[1:] else len(block)]
        for owner in owners:
            assert '"' + owner + '"' in block, (driver, owner)
        # every driver's shell names the driver and one of its owners
        found = False
        for module, name, op, _, _ in DRIVERS:
            if op != "family_par_run":
                continue
            text = ast.unparse(_function(module, name))
            if "'" + driver + "'" in text:
                found = True
        assert found, driver


def test_worker_serves_no_migrated_operation():
    worker = PKG / "_parallel_worker.py"
    if not worker.exists():
        return
    source = worker.read_text(encoding="utf-8")
    for op in ("'km_fit'", "'km_apply'", "'cholesky_fit'", "'cholesky_solve'", "'svm_fit'", "'svm_predict'",
               "'gp_fit'", "'gp_predict'", "'gram_fit'", "'solver_fit'", "'glm_fit'", "'kmeans_fit'",
               "'dbscan_fit'", "'gmm_fit'", "'resample'", "'hdbscan_fit'", "'graph_fit'", "'iforest_fit'",
               "'forest_prepare'", "'forest_predict'", "'forest_release'"):
        assert "operation == " + op not in source and "operation in (" + op not in source, op
