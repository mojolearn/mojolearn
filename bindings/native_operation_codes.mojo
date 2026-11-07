# SPDX-License-Identifier: Apache-2.0
# The one table of integer operation codes for the classical owner's door
# (lane ne1-h-labels, plan NATIVE_ENTRY_POINTS_AFTER_0866 section H, lesson
# 12: dispatch on integer codes through a table, not string chains).
#
# The Python shells (`_expansion_{neighbors,trees,metrics}.py`,
# `_metrics_impl.py`, `_forest_host.py`, `_gbdt_host.py`) passed each
# operation as a str; `classical_native_call` copied it into a Mojo String and
# matched it against a chain of exact and prefix tests, then
# bindings/family_native.mojo ran about 32 more prefix tests before the
# family owner. Now a shell reads this table once at import
# (python/mojolearn/_native_ops.py, door operation "operation_codes") and
# passes the integer code; the door reads the route, the family route and
# the name from the table (filled once per process) and calls the owner
# directly. The family owners still receive the operation name (a String
# built once here, never from the Python object), so their own suffix
# matches are unchanged.
#
# Codes are the index in `_operation_names` (sorted); they hold within one
# build only, and Python never hard-codes them. Add a name here when a shell
# passes a new `family_*`, `trees_*` or `host_tree_*` operation by code.
from std.ffi import _Global

comptime OP_ROUTE_FAMILY = 1
comptime OP_ROUTE_TREES = 2
comptime OP_ROUTE_HOST_TREE = 3

# Family owner routes (`family_route`), in the priority order of the former
# prefix chain of bindings/family_native.mojo.
comptime FR_UNKNOWN = 0
comptime FR_DENSITY = 1
comptime FR_PAR_PUBLIC = 2
comptime FR_PAR_RUN = 3
comptime FR_KMEANS = 4
comptime FR_KM = 5
comptime FR_SVM = 6
comptime FR_XC = 7
comptime FR_METRIC = 8
comptime FR_XN = 9
comptime FR_CHOL = 10
comptime FR_MSEL = 11
comptime FR_SVC = 12
comptime FR_GP = 13
comptime FR_MCLS = 14
comptime FR_MEXP = 15
comptime FR_HDBSCAN_PUBLIC = 16
comptime FR_CFA = 17
comptime FR_CFB = 18
comptime FR_MIX = 19
comptime FR_MANI = 20
comptime FR_LOW = 21
comptime FR_LINALG_OPS = 22
# The estimator-argument binding step that runs before the owner.
comptime FR_PRE_NONE = 0
comptime FR_PRE_ESTIMATORS = 1
comptime FR_PRE_SVM = 2
comptime FR_PRE_BASE = 3
comptime FR_PRE_X_NEIGHBORS = 4
# A packed family route word: owner + 64 * pre + 1024 * cluster flag.
comptime FR_PRE_SCALE = 64
comptime FR_CLUSTER_FLAG = 1024


def family_route_owner(route: Int) -> Int:
    return route % FR_PRE_SCALE


def family_route_pre(route: Int) -> Int:
    return (route // FR_PRE_SCALE) % (FR_CLUSTER_FLAG // FR_PRE_SCALE)


def family_route_cluster(route: Int) -> Bool:
    return route >= FR_CLUSTER_FLAG


def _owner_row(mut prefixes: List[String], mut owners: List[Int], mut exact: List[Bool],
               prefix: String, owner: Int, is_exact: Bool):
    prefixes.append(prefix)
    owners.append(owner)
    exact.append(is_exact)


def _family_owner_rows(mut prefixes: List[String], mut owners: List[Int], mut exact: List[Bool]):
    """The family owner prefixes (or exact names), in the priority order of
    the former prefix chain of bindings/family_native.mojo: the first row
    that matches names the owner. Straight-line appends, no branch."""
    _owner_row(prefixes, owners, exact, "family_density_", FR_DENSITY, False)
    _owner_row(prefixes, owners, exact, "family_par_public", FR_PAR_PUBLIC, True)
    _owner_row(prefixes, owners, exact, "family_par_run", FR_PAR_RUN, True)
    _owner_row(prefixes, owners, exact, "family_kmeans_", FR_KMEANS, False)
    _owner_row(prefixes, owners, exact, "family_km_", FR_KM, False)
    _owner_row(prefixes, owners, exact, "family_svm_", FR_SVM, False)
    _owner_row(prefixes, owners, exact, "family_xc_", FR_XC, False)
    _owner_row(prefixes, owners, exact, "family_metric_", FR_METRIC, False)
    _owner_row(prefixes, owners, exact, "family_xn_", FR_XN, False)
    _owner_row(prefixes, owners, exact, "family_chol_", FR_CHOL, False)
    _owner_row(prefixes, owners, exact, "family_msel_", FR_MSEL, False)
    _owner_row(prefixes, owners, exact, "family_svc_", FR_SVC, False)
    _owner_row(prefixes, owners, exact, "family_gp_", FR_GP, False)
    _owner_row(prefixes, owners, exact, "family_mcls_", FR_MCLS, False)
    _owner_row(prefixes, owners, exact, "family_mexp_", FR_MEXP, False)
    _owner_row(prefixes, owners, exact, "family_hdbscan_public_", FR_HDBSCAN_PUBLIC, False)
    _owner_row(prefixes, owners, exact, "family_cfa_", FR_CFA, False)
    _owner_row(prefixes, owners, exact, "family_cfb_", FR_CFB, False)
    _owner_row(prefixes, owners, exact, "family_mix_", FR_MIX, False)
    _owner_row(prefixes, owners, exact, "family_mani_", FR_MANI, False)
    _owner_row(prefixes, owners, exact, "family_low_", FR_LOW, False)
    _owner_row(prefixes, owners, exact, "family_lin_", FR_LINALG_OPS, False)
    _owner_row(prefixes, owners, exact, "family_rs_", FR_LINALG_OPS, False)
    _owner_row(prefixes, owners, exact, "family_pg_", FR_LINALG_OPS, False)


def _family_owner_of(operation: String) -> Int:
    """The owner of one `family_*` name: the first matching row of
    `_family_owner_rows` (lane ne-fix-hang3, lesson 12: a loop over a
    runtime row list instead of a 25-branch prefix chain in one function;
    same tests in the same order)."""
    var prefixes = List[String]()
    var owners = List[Int]()
    var exact = List[Bool]()
    _family_owner_rows(prefixes, owners, exact)
    for i in range(len(prefixes)):  # driver-loop: over the fixed owner rows of one name
        if exact[i]:
            if operation == prefixes[i]:
                return owners[i]
        elif operation.startswith(prefixes[i]):
            return owners[i]
    return FR_UNKNOWN


def family_route(operation: String) -> Int:
    """The packed route word of one `family_*` operation name: the same
    tests, in the same order, the family dispatcher ran on every call (now
    run once per name for coded calls, at table fill)."""
    var pre = FR_PRE_NONE
    if operation.startswith("family_svc_linear_") or operation.startswith("family_svm_linear_"):
        pre = FR_PRE_ESTIMATORS
    elif operation.startswith("family_svc_svc_") or operation.startswith("family_svc_svr_"):
        pre = FR_PRE_SVM
    elif operation.startswith("family_kmeans_"):
        pre = FR_PRE_BASE
    if (operation.startswith("family_xn_loc_") or operation.startswith("family_xn_semi_")
            or operation.startswith("family_xn_graph_") or operation.startswith("family_svc_ocsvm_")):
        pre = FR_PRE_X_NEIGHBORS
    var cluster = ((operation.startswith("family_cfa_") and not operation.startswith("family_cfa_hdbscan_"))
                   or (operation.startswith("family_xc_") and operation != "family_xc_dbscan_predict"
                       and operation != "family_xc_agglo_predict") or operation.startswith("family_mix_bgmm_"))
    var owner = _family_owner_of(operation)
    return owner + FR_PRE_SCALE * pre + (FR_CLUSTER_FLAG if cluster else 0)


def _operation_names(mut names: List[String]):
    """Every operation a shell passes by code, sorted; the code is the
    index. Four straight-line groups in sorted order (lane ne-fix-hang3:
    no single function holds all the name literals)."""
    _operation_names_metrics(names)
    _operation_names_neighbors(names)
    _operation_names_host_tree(names)
    _operation_names_trees(names)


def _operation_names_metrics(mut names: List[String]):
    """Names of the metric owners (`family_mcls_*`, `family_metric_*`, `family_mexp_*`), sorted."""
    names.append("family_mcls_accuracy")
    names.append("family_mcls_accuracy_fraction")
    names.append("family_mcls_adjusted_rand")
    names.append("family_mcls_completeness")
    names.append("family_mcls_confusion")
    names.append("family_mcls_entropy")
    names.append("family_mcls_fowlkes_mallows")
    names.append("family_mcls_hcv")
    names.append("family_mcls_homogeneity")
    names.append("family_mcls_kl")
    names.append("family_mcls_log_loss")
    names.append("family_mcls_mutual_info")
    names.append("family_mcls_pr_curve")
    names.append("family_mcls_prf")
    names.append("family_mcls_r2")
    names.append("family_mcls_rand")
    names.append("family_mcls_regression_error")
    names.append("family_mcls_roc_auc")
    names.append("family_mcls_silhouette_samples")
    names.append("family_mcls_silhouette_score")
    names.append("family_mcls_trustworthiness")
    names.append("family_mcls_v_measure")
    names.append("family_metric_d2_absolute")
    names.append("family_metric_d2_pinball")
    names.append("family_metric_d2_tweedie")
    names.append("family_metric_explained_variance")
    names.append("family_metric_gamma")
    names.append("family_metric_mape")
    names.append("family_metric_max_error")
    names.append("family_metric_medae")
    names.append("family_metric_msle")
    names.append("family_metric_pinball")
    names.append("family_metric_poisson")
    names.append("family_metric_r2_options")
    names.append("family_metric_regression_error")
    names.append("family_metric_regression_report")
    names.append("family_metric_rmsle")
    names.append("family_metric_tweedie")
    names.append("family_mexp_cls_balanced_accuracy")
    names.append("family_mexp_cls_fbeta")
    names.append("family_mexp_cls_hamming")
    names.append("family_mexp_cls_jaccard")
    names.append("family_mexp_cls_kappa")
    names.append("family_mexp_cls_likelihood_ratios")
    names.append("family_mexp_cls_matthews")
    names.append("family_mexp_cls_mlcm")
    names.append("family_mexp_cls_prfs")
    names.append("family_mexp_cls_report")
    names.append("family_mexp_cls_zero_one")
    names.append("family_mexp_cluster_ami")
    names.append("family_mexp_cluster_calinski_harabasz")
    names.append("family_mexp_cluster_contingency")
    names.append("family_mexp_cluster_davies_bouldin")
    names.append("family_mexp_cluster_nmi")
    names.append("family_mexp_cluster_pair_confusion")
    names.append("family_mexp_curve_auc")
    names.append("family_mexp_curve_average_precision")
    names.append("family_mexp_curve_det")
    names.append("family_mexp_curve_ranking_report")
    names.append("family_mexp_curve_roc")
    names.append("family_mexp_curve_top_k")
    names.append("family_mexp_rank_brier")
    names.append("family_mexp_rank_coverage")
    names.append("family_mexp_rank_d2_brier")
    names.append("family_mexp_rank_d2_log_loss")
    names.append("family_mexp_rank_dcg")
    names.append("family_mexp_rank_hinge")
    names.append("family_mexp_rank_loss")
    names.append("family_mexp_rank_lrap")
    names.append("family_mexp_rank_ndcg")


def _operation_names_neighbors(mut names: List[String]):
    """Names of the neighbor owners (`family_svc_ocsvm_*`, `family_xn_*`), sorted."""
    names.append("family_svc_ocsvm_decision")
    names.append("family_svc_ocsvm_fit")
    names.append("family_svc_ocsvm_fit_predict")
    names.append("family_svc_ocsvm_predict")
    names.append("family_svc_ocsvm_score_samples")
    names.append("family_xn_graph_connected_components")
    names.append("family_xn_graph_louvain_fit")
    names.append("family_xn_graph_louvain_fit_predict")
    names.append("family_xn_graph_pagerank_fit")
    names.append("family_xn_graph_svgp_fit")
    names.append("family_xn_graph_svgp_predict")
    names.append("family_xn_graph_svgp_predict_f")
    names.append("family_xn_graph_svgp_predict_y")
    names.append("family_xn_graph_svgp_score")
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


def _operation_names_host_tree(mut names: List[String]):
    """Names of the host tree owner (`host_tree_*`), sorted."""
    names.append("host_tree_count")
    names.append("host_tree_ctr_dim")
    names.append("host_tree_ctr_predict")
    names.append("host_tree_engine")
    names.append("host_tree_getstate")
    names.append("host_tree_groves")
    names.append("host_tree_hash")
    names.append("host_tree_init_forest")
    names.append("host_tree_init_gbdt")
    names.append("host_tree_parse")
    names.append("host_tree_predict")
    names.append("host_tree_predict_pair")
    names.append("host_tree_proba")
    names.append("host_tree_setstate")
    names.append("host_tree_vote")


def _operation_names_trees(mut names: List[String]):
    """Names of the public tree ensembles (`trees_*`), sorted."""
    names.append("trees_agnostic_kernel_init")
    names.append("trees_agnostic_kernel_values")
    names.append("trees_agnostic_permutation_init")
    names.append("trees_agnostic_permutation_values")
    names.append("trees_bagging_members")
    names.append("trees_bagging_members_set")
    names.append("trees_bagging_members_slice")
    names.append("trees_dart_construct")
    names.append("trees_explainer_close")
    names.append("trees_explainer_init")
    names.append("trees_explainer_values")
    names.append("trees_public_boost_apply")
    names.append("trees_public_boost_decision_function")
    names.append("trees_public_boost_fit")
    names.append("trees_public_boost_fit_transform")
    names.append("trees_public_boost_predict")
    names.append("trees_public_boost_predict_proba")
    names.append("trees_public_boost_score")
    names.append("trees_public_boost_transform")
    names.append("trees_public_forest_depth")
    names.append("trees_public_forest_fit")
    names.append("trees_public_forest_n_leaves")
    names.append("trees_public_forest_predict")
    names.append("trees_public_forest_proba")
    names.append("trees_public_meta_fit")
    names.append("trees_public_meta_fit_transform")
    names.append("trees_public_meta_predict")
    names.append("trees_public_meta_predict_proba")
    names.append("trees_public_meta_score")
    names.append("trees_public_meta_transform")


struct OperationTable(Movable):
    """The names, routes and family route words, filled once per process."""
    var names: List[String]
    var routes: List[Int]
    var family_routes: List[Int]

    def __init__(out self):
        # Empty: the global's init_fn does no work (lane ne-fix-hang3: the
        # names and the route words are filled by `fill` at first use in
        # `operation_table`, outside the `_Global` initializer).
        self.names = List[String]()
        self.routes = List[Int]()
        self.family_routes = List[Int]()

    def fill(mut self):
        """The names, routes and family route words (once per process)."""
        _operation_names(self.names)
        for i in range(len(self.names)):  # driver-loop: over the fixed operation names, once per process
            if self.names[i].startswith("trees_"):
                self.routes.append(OP_ROUTE_TREES)
                self.family_routes.append(FR_UNKNOWN)
            elif self.names[i].startswith("host_tree_"):
                self.routes.append(OP_ROUTE_HOST_TREE)
                self.family_routes.append(FR_UNKNOWN)
            else:
                self.routes.append(OP_ROUTE_FAMILY)
                self.family_routes.append(family_route(self.names[i]))


# Shared by every Mojo extension in the process; bump the version whenever a
# name is added, removed or reordered (the codes are indices).
comptime OPERATION_TABLE = _Global[
    StorageType=OperationTable, name="MojolearnOperationCodesV2",
    init_fn=OperationTable.__init__,
]


def operation_table() raises -> UnsafePointer[OperationTable, MutAnyOrigin]:
    """The process-wide table, filled at first use. Every caller is a
    CPython entry and holds the GIL, so the fill runs once."""
    var table = OPERATION_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if len(table[].names) == 0:
        table[].fill()
    return table
