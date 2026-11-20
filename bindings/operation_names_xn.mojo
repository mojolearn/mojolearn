# SPDX-License-Identifier: Apache-2.0
# The `family_xn_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# NOT YET CODED: the owner still receives the operation NAME (the table's own
# String, never built per call) and matches it. The lane that converts this
# owner gives each name a leaf code here (`comptime <FAMILY>_OPN_<NAME upper> =
# n`, numbered by sub-owner so a group is one range test) and appends it beside
# the name (`leaves.append(...)`, the form of
# bindings/operation_names_gp.mojo); the owner then takes `operation: Int`.
# Add every name a caller can pass: a name that is missing here still runs,
# through the door's String path (one failed lookup, then the old route).


def operation_names_xn(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_xn_ann_cagra_fit")
    names.append("family_xn_ann_cagra_search")
    names.append("family_xn_ann_getstate")
    names.append("family_xn_ann_ivf_fit")
    names.append("family_xn_ann_pq_fit")
    names.append("family_xn_ann_pq_search")
    names.append("family_xn_ann_rabitq_fit")
    names.append("family_xn_ann_rabitq_search")
    names.append("family_xn_ann_refine")
    names.append("family_xn_ann_sq_fit")
    names.append("family_xn_ann_sq_search")
    names.append("family_xn_graph_connected_components")
    names.append("family_xn_graph_louvain_fit")
    names.append("family_xn_graph_louvain_fit_predict")
    names.append("family_xn_graph_pagerank_fit")
    names.append("family_xn_graph_svgp_fit")
    names.append("family_xn_graph_svgp_predict")
    names.append("family_xn_graph_svgp_predict_f")
    names.append("family_xn_graph_svgp_predict_y")
    names.append("family_xn_graph_svgp_score")
    names.append("family_xn_kde_fit")
    names.append("family_xn_kde_score")
    names.append("family_xn_kde_score_samples")
    names.append("family_xn_knn_clf_classes")
    names.append("family_xn_knn_clf_fit")
    names.append("family_xn_knn_clf_predict")
    names.append("family_xn_knn_clf_predict_proba")
    names.append("family_xn_knn_clf_score")
    names.append("family_xn_knn_dist_params")
    names.append("family_xn_knn_getstate")
    names.append("family_xn_knn_nn_fit")
    names.append("family_xn_knn_refusals")
    names.append("family_xn_knn_reg_fit")
    names.append("family_xn_knn_reg_predict")
    names.append("family_xn_knn_reg_score")
    names.append("family_xn_knn_release")
    names.append("family_xn_knn_setstate")
    names.append("family_xn_loc_imp_fit")
    names.append("family_xn_loc_imp_fit_transform")
    names.append("family_xn_loc_imp_transform")
    names.append("family_xn_loc_lof_decision_function")
    names.append("family_xn_loc_lof_fit")
    names.append("family_xn_loc_lof_fit_predict")
    names.append("family_xn_loc_lof_predict")
    names.append("family_xn_loc_lof_score_samples")
    names.append("family_xn_loc_nc_decision_function")
    names.append("family_xn_loc_nc_fit")
    names.append("family_xn_loc_nc_predict")
    names.append("family_xn_loc_nc_predict_log_proba")
    names.append("family_xn_loc_nc_predict_proba")
    names.append("family_xn_loc_nc_score")
    names.append("family_xn_names_achi2")
    names.append("family_xn_names_imputer")
    names.append("family_xn_names_kpca")
    names.append("family_xn_names_schi2")
    names.append("family_xn_pcs_feature_names")
    names.append("family_xn_pcs_fit")
    names.append("family_xn_pcs_fit_transform")
    names.append("family_xn_pcs_transform")
    names.append("family_xn_radius_fit")
    names.append("family_xn_radius_refusals")
    names.append("family_xn_schi2_set_weights")
    names.append("family_xn_schi2_weights")
    names.append("family_xn_semi_achi2_fit")
    names.append("family_xn_semi_achi2_fit_transform")
    names.append("family_xn_semi_achi2_transform")
    names.append("family_xn_semi_lp_fit")
    names.append("family_xn_semi_lp_predict")
    names.append("family_xn_semi_lp_predict_proba")
    names.append("family_xn_semi_lp_score")
    names.append("family_xn_semi_schi2_fit")
    names.append("family_xn_semi_schi2_fit_transform")
    names.append("family_xn_semi_schi2_transform")
