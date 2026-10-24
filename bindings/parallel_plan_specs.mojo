# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cooperative multi-GPU drivers `family_par_run` serves
(bindings/family_native.mojo; docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md):
for each driver, the plain owner operations it may run, the owner families
its old worker's environment enabled (core/device_plan.mojo bits) and the
refusal for a binding built without the plan entries (the old worker's
capability refusal, word for word)."""
from core.device_plan import DEVICE_PLAN_WORKER


def par_spec_admits(driver: String, owner: String) -> Bool:
    """True when `driver` may run the plain owner operation `owner`."""
    if driver == "svm_fit":
        return owner == "family_svc_svc_fit" or owner == "family_svc_svr_fit"
    if driver == "svm_predict":
        return (owner == "family_svc_svc_predict" or owner == "family_svc_svc_decision"
                or owner == "family_svc_svr_predict")
    if driver == "gp_fit":
        return owner == "family_gp_gpr_fit"
    if driver == "gp_predict":
        return owner == "family_gp_gpr_predict"
    if driver == "glm_fit":
        return owner == "fit"
    if driver == "solver_fit":
        return owner == "cd_fit"
    return False


def par_spec_families(driver: String) -> Int:
    """The owner families the driver's old cooperative worker enabled: its
    environment set every `DEVICE_COUNT_VARIABLES` count (not Cholesky)."""
    return DEVICE_PLAN_WORKER


def par_spec_rebuild(driver: String) -> String:
    """The refusal for a binding without the plan entries."""
    if driver == "svm_fit" or driver == "svm_predict":
        return "rebuild SVM binding for distributed kernel rows"
    if driver == "gp_fit" or driver == "gp_predict":
        return "rebuild GP binding for distributed covariance rows"
    if driver == "glm_fit":
        return "rebuild estimators binding for parallel GLM gradients"
    if driver == "solver_fit":
        return "rebuild solver binding for parallel dot leaves"
    return "rebuild the binding for multi-GPU drivers"
