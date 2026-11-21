# SPDX-License-Identifier: Apache-2.0
"""The tree estimators' device route as native words (core/trees_route.mojo): which
table set an operation runs on, when a fit records its table set, and the
three refusals. No Python import. The plan is a metadata decision over a
handful of words; the cases run in one pass (nothing numerical to split)."""
from core.linear_route import LINEAR_DEVICE_GPU, LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL
from core.trees_route import (
    TREES_FITTED_UNRECORDED, TREES_FITTED_INSTALL, TREES_FITTED_CPU, TREES_ROUTE_RUNS, TREES_ROUTE_GPU_ABSENT,
    TREES_ROUTE_FITTED_ON_CPU, TREES_ROUTE_FITTED_ON_INSTALL, trees_route_fits, trees_route_lifecycle,
    trees_route_strict, trees_route_plan,
)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def operation_kinds() raises:
    # Every fit of the family's operation table builds fitted state; a
    # question about fitted state does not.
    require(trees_route_fits("trees_public_forest_fit") and trees_route_fits("trees_iforest_fit_predict")
        and trees_route_fits("trees_gbdt_adapter_classifier_fit") and trees_route_fits("trees_meta_fit_transform")
        and trees_route_fits("trees_public_gbdt_adapter_load"), "a fit is not seen as a fit")
    require(not trees_route_fits("trees_public_forest_is_fitted") and not trees_route_fits("trees_public_forest_predict")
        and not trees_route_fits("trees_forest_get_params"), "a query is seen as a fit")
    require(trees_route_strict("trees_public_forest_validate_params") and trees_route_strict("trees_gbdt_fit")
        and not trees_route_strict("trees_forest_get_params") and not trees_route_strict("trees_public_forest_set_params"),
        "the operations that check the stored device are wrong")
    require(trees_route_lifecycle("trees_forest_get_params") and trees_route_lifecycle("trees_public_forest_set_params")
        and trees_route_lifecycle("trees_forest_construct") and trees_route_lifecycle("trees_gbdt_init")
        and trees_route_lifecycle("trees_public_forest_refresh") and trees_route_lifecycle("trees_device_vendor_used")
        and trees_route_lifecycle("trees_public_forest_is_fitted"), "a parameter operation is not seen as one")
    require(not trees_route_lifecycle("trees_public_forest_predict") and not trees_route_lifecycle("trees_iforest_score_samples")
        and not trees_route_lifecycle("trees_forest_getstate"), "an operation on fitted state is seen as a parameter operation")


def gpu_installation() raises:
    # Default device: the device set, recorded by the first fit only.
    var first = trees_route_plan(LINEAR_DEVICE_INSTALL, TREES_FITTED_UNRECORDED, False, True, False)
    require(not first.cpu and first.record and first.refusal == TREES_ROUTE_RUNS, "the first default fit is not recorded")
    var again = trees_route_plan(LINEAR_DEVICE_INSTALL, TREES_FITTED_INSTALL, False, True, False)
    require(not again.cpu and not again.record, "a second default fit writes the marker again")
    var gpu = trees_route_plan(LINEAR_DEVICE_GPU, TREES_FITTED_INSTALL, False, False, False)
    require(not gpu.cpu and gpu.refusal == TREES_ROUTE_RUNS, "an explicit gpu query does not run on the device set")
    # device='cpu': the CPU set, recorded when the marker does not say so.
    var cpu_fit = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_INSTALL, False, True, False)
    require(cpu_fit.cpu and cpu_fit.record and cpu_fit.refusal == TREES_ROUTE_RUNS, "a cpu fit is not recorded")
    var cpu_refit = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_CPU, False, True, False)
    require(cpu_refit.cpu and not cpu_refit.record, "a second cpu fit writes the marker again")
    var cpu_query = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_CPU, False, False, False)
    require(cpu_query.cpu and cpu_query.refusal == TREES_ROUTE_RUNS, "a cpu query of a cpu model does not run")
    var back = trees_route_plan(LINEAR_DEVICE_INSTALL, TREES_FITTED_CPU, False, True, False)
    require(not back.cpu and back.record, "a device fit of a cpu-fitted model is not recorded")
    # An unfitted estimator's query runs: its door says it is not fitted.
    var unfitted = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_UNRECORDED, False, False, False)
    require(unfitted.cpu and unfitted.refusal == TREES_ROUTE_RUNS, "an unfitted cpu estimator's query is refused here")
    # Fitted state never crosses table sets; parameter operations still run.
    var crossed = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_INSTALL, False, False, False)
    require(crossed.refusal == TREES_ROUTE_FITTED_ON_INSTALL, "device-set state reached the CPU set")
    var other = trees_route_plan(LINEAR_DEVICE_INSTALL, TREES_FITTED_CPU, False, False, False)
    require(other.refusal == TREES_ROUTE_FITTED_ON_CPU, "CPU-set state reached the device set")
    var explicit = trees_route_plan(LINEAR_DEVICE_GPU, TREES_FITTED_CPU, False, False, False)
    require(explicit.refusal == TREES_ROUTE_FITTED_ON_CPU, "CPU-set state reached the device set through an explicit gpu")
    var params = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_INSTALL, False, False, True)
    require(params.cpu and params.refusal == TREES_ROUTE_RUNS and not params.record, "a parameter operation is refused")


def cpu_only_installation() raises:
    var fit = trees_route_plan(LINEAR_DEVICE_INSTALL, TREES_FITTED_UNRECORDED, True, True, False)
    require(not fit.cpu and fit.record and fit.refusal == TREES_ROUTE_RUNS, "a CPU-only default fit is wrong")
    var cpu = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_INSTALL, True, True, False)
    require(not cpu.cpu and not cpu.record and cpu.refusal == TREES_ROUTE_RUNS,
        "device='cpu' on a CPU-only process does not run the process's own set")
    var gpu = trees_route_plan(LINEAR_DEVICE_GPU, TREES_FITTED_UNRECORDED, True, True, False)
    require(gpu.refusal == TREES_ROUTE_GPU_ABSENT, "an explicit gpu fit ran on a CPU-only process")
    var gpu_params = trees_route_plan(LINEAR_DEVICE_GPU, TREES_FITTED_UNRECORDED, True, False, True)
    require(gpu_params.refusal == TREES_ROUTE_RUNS, "get_params of an explicit gpu is refused on a CPU-only process")
    # A model pickled on a GPU installation after a cpu fit: its marker names
    # a set this process does not have; queries run, a fit replaces it.
    var moved = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_CPU, True, False, False)
    require(not moved.cpu and moved.refusal == TREES_ROUTE_RUNS, "an unpickled cpu model is refused on a CPU-only process")
    var refit = trees_route_plan(LINEAR_DEVICE_CPU, TREES_FITTED_CPU, True, True, False)
    require(refit.record and not refit.cpu, "a CPU-only fit does not replace a foreign marker")


def run_checks() raises:
    operation_kinds()
    gpu_installation()
    cpu_only_installation()
