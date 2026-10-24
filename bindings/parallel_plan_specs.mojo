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
    if driver == "gram_fit":
        return owner == "ols_fit" or owner == "ridge_fit" or owner == "fit"
    if driver == "kmeans_fit":
        return owner == "family_kmeans_fit"
    if driver == "dbscan_fit":
        return owner == "family_cfb_dbscan_fit"
    if driver == "gmm_fit":
        return owner == "family_mix_gmm_fit"
    if driver == "gmm_predict":
        return (owner == "family_mix_gmm_score_samples" or owner == "family_mix_gmm_predict_proba"
                or owner == "family_mix_gmm_predict")
    if driver == "hdbscan_fit":
        return owner == "family_cfa_hdbscan_fit"
    if driver == "resample":
        return (owner == "family_rs_bootstrap" or owner == "family_rs_permutation_test"
                or owner == "family_rs_monte_carlo_integrate")
    if driver == "graph_fit":
        return owner == "family_cfb_agglo_fit" or owner == "family_cfb_sc_fit" or owner == "family_mani_umap_fit"
    if driver == "umap_transform":
        return owner == "family_mani_umap_transform"
    if driver == "gbdt_fit":
        return (owner == "trees_gbdt_fit" or owner == "trees_gbdt_adapter_classifier_fit"
                or owner == "trees_gbdt_adapter_regressor_fit" or owner == "trees_gbdt_feature_freq_fit")
    if driver == "ordered_rmse_fit":
        return owner == "trees_gbdt_ordered_fit"
    if driver == "iforest_fit":
        return owner == "trees_iforest_fit"
    if driver == "iforest_score":
        return (owner == "trees_iforest_score_samples" or owner == "trees_iforest_decision_function"
                or owner == "trees_iforest_predict")
    if driver == "forest_predict":
        return owner == "trees_public_forest_predict" or owner == "trees_public_forest_proba"
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
    if driver == "gram_fit":
        return "rebuild estimators binding for parallel Gram chunks"
    if driver == "kmeans_fit":
        return "rebuild base binding for cooperative KMeans"
    if driver == "dbscan_fit":
        return "rebuild estimators binding for parallel DBSCAN neighborhoods"
    if driver == "gmm_fit" or driver == "gmm_predict":
        return "rebuild mixture binding for row-sharded GaussianMixture E-steps"
    if driver == "hdbscan_fit":
        return "rebuild HDBSCAN binding for distributed neighbor and distance rows"
    if driver == "resample":
        return "rebuild resample binding for distributed replicate ranges"
    if driver == "graph_fit" or driver == "umap_transform":
        return "rebuild graph binding for native multi-GPU rows"
    if driver == "gbdt_fit" or driver == "ordered_rmse_fit":
        return "rebuild GBDT binding for feature-parallel training"
    if driver == "iforest_fit" or driver == "iforest_score":
        return "rebuild SVM binding for parallel IsolationForest"
    if driver == "forest_predict":
        return "rebuild RF/ET binding for pooled forest inference"
    return "rebuild the binding for multi-GPU drivers"
