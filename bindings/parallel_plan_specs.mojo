# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cooperative multi-GPU drivers `family_par_run` serves
(bindings/family_native.mojo; docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md):
for each driver, the plain owner operations it may run, the owner families
its old worker's environment enabled (core/device_plan.mojo bits) and the
refusal for a binding built without the plan entries (the old worker's
capability refusal, word for word).

Lane bw2-parallel (2026-11-20): drivers and owner operations are integer
codes (`PARALLEL_DRIVER_OPN_*`, `PARALLEL_OWNER_OPN_*`). The driver door
resolves the caller's two str objects with one lookup each in the parallel
key table (bindings/parallel_keys_unpack.mojo `ops_par_driver`,
`ops_par_owner`, filled from the names below); this file stays Python-free.
A name the tables do not hold is -1, which no driver admits."""
from core.device_plan import DEVICE_PLAN_WORKER


comptime PARALLEL_DRIVER_OPN_SVM_FIT = 0
comptime PARALLEL_DRIVER_OPN_SVM_PREDICT = 1
comptime PARALLEL_DRIVER_OPN_GP_FIT = 2
comptime PARALLEL_DRIVER_OPN_GP_PREDICT = 3
comptime PARALLEL_DRIVER_OPN_GLM_FIT = 4
comptime PARALLEL_DRIVER_OPN_SOLVER_FIT = 5
comptime PARALLEL_DRIVER_OPN_GRAM_FIT = 6
comptime PARALLEL_DRIVER_OPN_KMEANS_FIT = 7
comptime PARALLEL_DRIVER_OPN_DBSCAN_FIT = 8
comptime PARALLEL_DRIVER_OPN_GMM_FIT = 9
comptime PARALLEL_DRIVER_OPN_GMM_PREDICT = 10
comptime PARALLEL_DRIVER_OPN_HDBSCAN_FIT = 11
comptime PARALLEL_DRIVER_OPN_RESAMPLE = 12
comptime PARALLEL_DRIVER_OPN_GRAPH_FIT = 13
comptime PARALLEL_DRIVER_OPN_UMAP_TRANSFORM = 14
comptime PARALLEL_DRIVER_OPN_GBDT_FIT = 15
comptime PARALLEL_DRIVER_OPN_ORDERED_RMSE_FIT = 16
comptime PARALLEL_DRIVER_OPN_IFOREST_FIT = 17
comptime PARALLEL_DRIVER_OPN_IFOREST_SCORE = 18
comptime PARALLEL_DRIVER_OPN_FOREST_PREDICT = 19
comptime PARALLEL_DRIVER_COUNT = 20

comptime PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_FIT = 0
comptime PARALLEL_OWNER_OPN_FAMILY_SVC_SVR_FIT = 1
comptime PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_PREDICT = 2
comptime PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_DECISION = 3
comptime PARALLEL_OWNER_OPN_FAMILY_SVC_SVR_PREDICT = 4
comptime PARALLEL_OWNER_OPN_FAMILY_GP_GPR_FIT = 5
comptime PARALLEL_OWNER_OPN_FAMILY_GP_GPR_PREDICT = 6
comptime PARALLEL_OWNER_OPN_FIT = 7
comptime PARALLEL_OWNER_OPN_CD_FIT = 8
comptime PARALLEL_OWNER_OPN_OLS_FIT = 9
comptime PARALLEL_OWNER_OPN_RIDGE_FIT = 10
comptime PARALLEL_OWNER_OPN_FAMILY_KMEANS_FIT = 11
comptime PARALLEL_OWNER_OPN_FAMILY_CFB_DBSCAN_FIT = 12
comptime PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_FIT = 13
comptime PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_SCORE_SAMPLES = 14
comptime PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_PREDICT_PROBA = 15
comptime PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_PREDICT = 16
comptime PARALLEL_OWNER_OPN_FAMILY_CFA_HDBSCAN_FIT = 17
comptime PARALLEL_OWNER_OPN_FAMILY_RS_BOOTSTRAP = 18
comptime PARALLEL_OWNER_OPN_FAMILY_RS_PERMUTATION_TEST = 19
comptime PARALLEL_OWNER_OPN_FAMILY_RS_MONTE_CARLO_INTEGRATE = 20
comptime PARALLEL_OWNER_OPN_FAMILY_CFB_AGGLO_FIT = 21
comptime PARALLEL_OWNER_OPN_FAMILY_CFB_SC_FIT = 22
comptime PARALLEL_OWNER_OPN_FAMILY_MANI_UMAP_FIT = 23
comptime PARALLEL_OWNER_OPN_FAMILY_MANI_UMAP_TRANSFORM = 24
comptime PARALLEL_OWNER_OPN_TREES_GBDT_FIT = 25
comptime PARALLEL_OWNER_OPN_TREES_GBDT_ADAPTER_CLASSIFIER_FIT = 26
comptime PARALLEL_OWNER_OPN_TREES_GBDT_ADAPTER_REGRESSOR_FIT = 27
comptime PARALLEL_OWNER_OPN_TREES_GBDT_FEATURE_FREQ_FIT = 28
comptime PARALLEL_OWNER_OPN_TREES_GBDT_ORDERED_FIT = 29
comptime PARALLEL_OWNER_OPN_TREES_IFOREST_FIT = 30
comptime PARALLEL_OWNER_OPN_TREES_IFOREST_SCORE_SAMPLES = 31
comptime PARALLEL_OWNER_OPN_TREES_IFOREST_DECISION_FUNCTION = 32
comptime PARALLEL_OWNER_OPN_TREES_IFOREST_PREDICT = 33
comptime PARALLEL_OWNER_OPN_TREES_PUBLIC_FOREST_PREDICT = 34
comptime PARALLEL_OWNER_OPN_TREES_PUBLIC_FOREST_PROBA = 35
comptime PARALLEL_OWNER_COUNT = 36


def par_spec_driver_name(driver: Int) -> StaticString:
    """The driver's name (table fill, refusal text)."""
    if driver == PARALLEL_DRIVER_OPN_SVM_FIT:
        return "svm_fit"
    if driver == PARALLEL_DRIVER_OPN_SVM_PREDICT:
        return "svm_predict"
    if driver == PARALLEL_DRIVER_OPN_GP_FIT:
        return "gp_fit"
    if driver == PARALLEL_DRIVER_OPN_GP_PREDICT:
        return "gp_predict"
    if driver == PARALLEL_DRIVER_OPN_GLM_FIT:
        return "glm_fit"
    if driver == PARALLEL_DRIVER_OPN_SOLVER_FIT:
        return "solver_fit"
    if driver == PARALLEL_DRIVER_OPN_GRAM_FIT:
        return "gram_fit"
    if driver == PARALLEL_DRIVER_OPN_KMEANS_FIT:
        return "kmeans_fit"
    if driver == PARALLEL_DRIVER_OPN_DBSCAN_FIT:
        return "dbscan_fit"
    if driver == PARALLEL_DRIVER_OPN_GMM_FIT:
        return "gmm_fit"
    if driver == PARALLEL_DRIVER_OPN_GMM_PREDICT:
        return "gmm_predict"
    if driver == PARALLEL_DRIVER_OPN_HDBSCAN_FIT:
        return "hdbscan_fit"
    if driver == PARALLEL_DRIVER_OPN_RESAMPLE:
        return "resample"
    if driver == PARALLEL_DRIVER_OPN_GRAPH_FIT:
        return "graph_fit"
    if driver == PARALLEL_DRIVER_OPN_UMAP_TRANSFORM:
        return "umap_transform"
    if driver == PARALLEL_DRIVER_OPN_GBDT_FIT:
        return "gbdt_fit"
    if driver == PARALLEL_DRIVER_OPN_ORDERED_RMSE_FIT:
        return "ordered_rmse_fit"
    if driver == PARALLEL_DRIVER_OPN_IFOREST_FIT:
        return "iforest_fit"
    if driver == PARALLEL_DRIVER_OPN_IFOREST_SCORE:
        return "iforest_score"
    if driver == PARALLEL_DRIVER_OPN_FOREST_PREDICT:
        return "forest_predict"
    return ""


def par_spec_owner_name(owner: Int) -> StaticString:
    """The owner operation's name (table fill only)."""
    if owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_FIT:
        return "family_svc_svc_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVR_FIT:
        return "family_svc_svr_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_PREDICT:
        return "family_svc_svc_predict"
    if owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_DECISION:
        return "family_svc_svc_decision"
    if owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVR_PREDICT:
        return "family_svc_svr_predict"
    if owner == PARALLEL_OWNER_OPN_FAMILY_GP_GPR_FIT:
        return "family_gp_gpr_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_GP_GPR_PREDICT:
        return "family_gp_gpr_predict"
    if owner == PARALLEL_OWNER_OPN_FIT:
        return "fit"
    if owner == PARALLEL_OWNER_OPN_CD_FIT:
        return "cd_fit"
    if owner == PARALLEL_OWNER_OPN_OLS_FIT:
        return "ols_fit"
    if owner == PARALLEL_OWNER_OPN_RIDGE_FIT:
        return "ridge_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_KMEANS_FIT:
        return "family_kmeans_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_CFB_DBSCAN_FIT:
        return "family_cfb_dbscan_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_FIT:
        return "family_mix_gmm_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_SCORE_SAMPLES:
        return "family_mix_gmm_score_samples"
    if owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_PREDICT_PROBA:
        return "family_mix_gmm_predict_proba"
    if owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_PREDICT:
        return "family_mix_gmm_predict"
    if owner == PARALLEL_OWNER_OPN_FAMILY_CFA_HDBSCAN_FIT:
        return "family_cfa_hdbscan_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_RS_BOOTSTRAP:
        return "family_rs_bootstrap"
    if owner == PARALLEL_OWNER_OPN_FAMILY_RS_PERMUTATION_TEST:
        return "family_rs_permutation_test"
    if owner == PARALLEL_OWNER_OPN_FAMILY_RS_MONTE_CARLO_INTEGRATE:
        return "family_rs_monte_carlo_integrate"
    if owner == PARALLEL_OWNER_OPN_FAMILY_CFB_AGGLO_FIT:
        return "family_cfb_agglo_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_CFB_SC_FIT:
        return "family_cfb_sc_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_MANI_UMAP_FIT:
        return "family_mani_umap_fit"
    if owner == PARALLEL_OWNER_OPN_FAMILY_MANI_UMAP_TRANSFORM:
        return "family_mani_umap_transform"
    if owner == PARALLEL_OWNER_OPN_TREES_GBDT_FIT:
        return "trees_gbdt_fit"
    if owner == PARALLEL_OWNER_OPN_TREES_GBDT_ADAPTER_CLASSIFIER_FIT:
        return "trees_gbdt_adapter_classifier_fit"
    if owner == PARALLEL_OWNER_OPN_TREES_GBDT_ADAPTER_REGRESSOR_FIT:
        return "trees_gbdt_adapter_regressor_fit"
    if owner == PARALLEL_OWNER_OPN_TREES_GBDT_FEATURE_FREQ_FIT:
        return "trees_gbdt_feature_freq_fit"
    if owner == PARALLEL_OWNER_OPN_TREES_GBDT_ORDERED_FIT:
        return "trees_gbdt_ordered_fit"
    if owner == PARALLEL_OWNER_OPN_TREES_IFOREST_FIT:
        return "trees_iforest_fit"
    if owner == PARALLEL_OWNER_OPN_TREES_IFOREST_SCORE_SAMPLES:
        return "trees_iforest_score_samples"
    if owner == PARALLEL_OWNER_OPN_TREES_IFOREST_DECISION_FUNCTION:
        return "trees_iforest_decision_function"
    if owner == PARALLEL_OWNER_OPN_TREES_IFOREST_PREDICT:
        return "trees_iforest_predict"
    if owner == PARALLEL_OWNER_OPN_TREES_PUBLIC_FOREST_PREDICT:
        return "trees_public_forest_predict"
    if owner == PARALLEL_OWNER_OPN_TREES_PUBLIC_FOREST_PROBA:
        return "trees_public_forest_proba"
    return ""


def par_spec_admits(driver: Int, owner: Int) -> Bool:
    """True when `driver` may run the plain owner operation `owner`."""
    if driver == PARALLEL_DRIVER_OPN_SVM_FIT:
        return owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_FIT or owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVR_FIT
    if driver == PARALLEL_DRIVER_OPN_SVM_PREDICT:
        return (owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_PREDICT
                or owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVC_DECISION
                or owner == PARALLEL_OWNER_OPN_FAMILY_SVC_SVR_PREDICT)
    if driver == PARALLEL_DRIVER_OPN_GP_FIT:
        return owner == PARALLEL_OWNER_OPN_FAMILY_GP_GPR_FIT
    if driver == PARALLEL_DRIVER_OPN_GP_PREDICT:
        return owner == PARALLEL_OWNER_OPN_FAMILY_GP_GPR_PREDICT
    if driver == PARALLEL_DRIVER_OPN_GLM_FIT:
        return owner == PARALLEL_OWNER_OPN_FIT
    if driver == PARALLEL_DRIVER_OPN_SOLVER_FIT:
        return owner == PARALLEL_OWNER_OPN_CD_FIT
    if driver == PARALLEL_DRIVER_OPN_GRAM_FIT:
        return (owner == PARALLEL_OWNER_OPN_OLS_FIT
                or owner == PARALLEL_OWNER_OPN_RIDGE_FIT
                or owner == PARALLEL_OWNER_OPN_FIT)
    if driver == PARALLEL_DRIVER_OPN_KMEANS_FIT:
        return owner == PARALLEL_OWNER_OPN_FAMILY_KMEANS_FIT
    if driver == PARALLEL_DRIVER_OPN_DBSCAN_FIT:
        return owner == PARALLEL_OWNER_OPN_FAMILY_CFB_DBSCAN_FIT
    if driver == PARALLEL_DRIVER_OPN_GMM_FIT:
        return owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_FIT
    if driver == PARALLEL_DRIVER_OPN_GMM_PREDICT:
        return (owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_SCORE_SAMPLES
                or owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_PREDICT_PROBA
                or owner == PARALLEL_OWNER_OPN_FAMILY_MIX_GMM_PREDICT)
    if driver == PARALLEL_DRIVER_OPN_HDBSCAN_FIT:
        return owner == PARALLEL_OWNER_OPN_FAMILY_CFA_HDBSCAN_FIT
    if driver == PARALLEL_DRIVER_OPN_RESAMPLE:
        return (owner == PARALLEL_OWNER_OPN_FAMILY_RS_BOOTSTRAP
                or owner == PARALLEL_OWNER_OPN_FAMILY_RS_PERMUTATION_TEST
                or owner == PARALLEL_OWNER_OPN_FAMILY_RS_MONTE_CARLO_INTEGRATE)
    if driver == PARALLEL_DRIVER_OPN_GRAPH_FIT:
        return (owner == PARALLEL_OWNER_OPN_FAMILY_CFB_AGGLO_FIT
                or owner == PARALLEL_OWNER_OPN_FAMILY_CFB_SC_FIT
                or owner == PARALLEL_OWNER_OPN_FAMILY_MANI_UMAP_FIT)
    if driver == PARALLEL_DRIVER_OPN_UMAP_TRANSFORM:
        return owner == PARALLEL_OWNER_OPN_FAMILY_MANI_UMAP_TRANSFORM
    if driver == PARALLEL_DRIVER_OPN_GBDT_FIT:
        return (owner == PARALLEL_OWNER_OPN_TREES_GBDT_FIT
                or owner == PARALLEL_OWNER_OPN_TREES_GBDT_ADAPTER_CLASSIFIER_FIT
                or owner == PARALLEL_OWNER_OPN_TREES_GBDT_ADAPTER_REGRESSOR_FIT
                or owner == PARALLEL_OWNER_OPN_TREES_GBDT_FEATURE_FREQ_FIT)
    if driver == PARALLEL_DRIVER_OPN_ORDERED_RMSE_FIT:
        return owner == PARALLEL_OWNER_OPN_TREES_GBDT_ORDERED_FIT
    if driver == PARALLEL_DRIVER_OPN_IFOREST_FIT:
        return owner == PARALLEL_OWNER_OPN_TREES_IFOREST_FIT
    if driver == PARALLEL_DRIVER_OPN_IFOREST_SCORE:
        return (owner == PARALLEL_OWNER_OPN_TREES_IFOREST_SCORE_SAMPLES
                or owner == PARALLEL_OWNER_OPN_TREES_IFOREST_DECISION_FUNCTION
                or owner == PARALLEL_OWNER_OPN_TREES_IFOREST_PREDICT)
    if driver == PARALLEL_DRIVER_OPN_FOREST_PREDICT:
        return (owner == PARALLEL_OWNER_OPN_TREES_PUBLIC_FOREST_PREDICT
                or owner == PARALLEL_OWNER_OPN_TREES_PUBLIC_FOREST_PROBA)
    return False


def par_spec_families(driver: Int) -> Int:
    """The owner families the driver's old cooperative worker enabled: its
    environment set every `DEVICE_COUNT_VARIABLES` count (not Cholesky)."""
    return DEVICE_PLAN_WORKER


def par_spec_rebuild(driver: Int) -> StaticString:
    """The refusal for a binding without the plan entries."""
    if driver == PARALLEL_DRIVER_OPN_SVM_FIT or driver == PARALLEL_DRIVER_OPN_SVM_PREDICT:
        return "rebuild SVM binding for distributed kernel rows"
    if driver == PARALLEL_DRIVER_OPN_GP_FIT or driver == PARALLEL_DRIVER_OPN_GP_PREDICT:
        return "rebuild GP binding for distributed covariance rows"
    if driver == PARALLEL_DRIVER_OPN_GLM_FIT:
        return "rebuild estimators binding for parallel GLM gradients"
    if driver == PARALLEL_DRIVER_OPN_SOLVER_FIT:
        return "rebuild solver binding for parallel dot leaves"
    if driver == PARALLEL_DRIVER_OPN_GRAM_FIT:
        return "rebuild estimators binding for parallel Gram chunks"
    if driver == PARALLEL_DRIVER_OPN_KMEANS_FIT:
        return "rebuild base binding for cooperative KMeans"
    if driver == PARALLEL_DRIVER_OPN_DBSCAN_FIT:
        return "rebuild estimators binding for parallel DBSCAN neighborhoods"
    if driver == PARALLEL_DRIVER_OPN_GMM_FIT or driver == PARALLEL_DRIVER_OPN_GMM_PREDICT:
        return "rebuild mixture binding for row-sharded GaussianMixture E-steps"
    if driver == PARALLEL_DRIVER_OPN_HDBSCAN_FIT:
        return "rebuild HDBSCAN binding for distributed neighbor and distance rows"
    if driver == PARALLEL_DRIVER_OPN_RESAMPLE:
        return "rebuild resample binding for distributed replicate ranges"
    if driver == PARALLEL_DRIVER_OPN_GRAPH_FIT or driver == PARALLEL_DRIVER_OPN_UMAP_TRANSFORM:
        return "rebuild graph binding for native multi-GPU rows"
    if driver == PARALLEL_DRIVER_OPN_GBDT_FIT or driver == PARALLEL_DRIVER_OPN_ORDERED_RMSE_FIT:
        return "rebuild GBDT binding for feature-parallel training"
    if driver == PARALLEL_DRIVER_OPN_IFOREST_FIT or driver == PARALLEL_DRIVER_OPN_IFOREST_SCORE:
        return "rebuild SVM binding for parallel IsolationForest"
    if driver == PARALLEL_DRIVER_OPN_FOREST_PREDICT:
        return "rebuild RF/ET binding for pooled forest inference"
    return "rebuild the binding for multi-GPU drivers"
