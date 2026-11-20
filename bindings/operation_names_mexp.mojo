# SPDX-License-Identifier: Apache-2.0
# The `family_mexp_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-metrics, 2026-11-20): each name has a leaf code,
# numbered by sub-owner so `metrics_exp_operation`
# (bindings/metrics_exp_native_unpack.mojo) picks the owner with one range test: cls 0-10
# (bindings/metrics_exp_cls_door.mojo), curve 11-16 (metrics_exp_curve_door), rank 17-25
# (metrics_exp_rank_door), cluster 26-31 (metrics_exp_cluster_door).
# The family door resolves a caller's str (or a shell's table code) to the leaf
# with one lookup and never builds the name; a `family_mexp_*` name the table does
# not hold keeps its String for its unchanged error text only. The appends keep
# their order (the table's indices do not move); the leaves are not in append
# order. A constant is `MEXP_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime MEXP_OPN_FAMILY_MEXP_CLS_PRFS = 0
comptime MEXP_OPN_FAMILY_MEXP_CLS_FBETA = 1
comptime MEXP_OPN_FAMILY_MEXP_CLS_JACCARD = 2
comptime MEXP_OPN_FAMILY_MEXP_CLS_BALANCED_ACCURACY = 3
comptime MEXP_OPN_FAMILY_MEXP_CLS_MATTHEWS = 4
comptime MEXP_OPN_FAMILY_MEXP_CLS_MLCM = 5
comptime MEXP_OPN_FAMILY_MEXP_CLS_HAMMING = 6
comptime MEXP_OPN_FAMILY_MEXP_CLS_ZERO_ONE = 7
comptime MEXP_OPN_FAMILY_MEXP_CLS_KAPPA = 8
comptime MEXP_OPN_FAMILY_MEXP_CLS_LIKELIHOOD_RATIOS = 9
comptime MEXP_OPN_FAMILY_MEXP_CLS_REPORT = 10
comptime MEXP_OPN_FAMILY_MEXP_CURVE_RANKING_REPORT = 11
comptime MEXP_OPN_FAMILY_MEXP_CURVE_ROC = 12
comptime MEXP_OPN_FAMILY_MEXP_CURVE_DET = 13
comptime MEXP_OPN_FAMILY_MEXP_CURVE_AUC = 14
comptime MEXP_OPN_FAMILY_MEXP_CURVE_AVERAGE_PRECISION = 15
comptime MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K = 16
comptime MEXP_OPN_FAMILY_MEXP_RANK_BRIER = 17
comptime MEXP_OPN_FAMILY_MEXP_RANK_D2_LOG_LOSS = 18
comptime MEXP_OPN_FAMILY_MEXP_RANK_D2_BRIER = 19
comptime MEXP_OPN_FAMILY_MEXP_RANK_HINGE = 20
comptime MEXP_OPN_FAMILY_MEXP_RANK_DCG = 21
comptime MEXP_OPN_FAMILY_MEXP_RANK_NDCG = 22
comptime MEXP_OPN_FAMILY_MEXP_RANK_COVERAGE = 23
comptime MEXP_OPN_FAMILY_MEXP_RANK_LRAP = 24
comptime MEXP_OPN_FAMILY_MEXP_RANK_LOSS = 25
comptime MEXP_OPN_FAMILY_MEXP_CLUSTER_CONTINGENCY = 26
comptime MEXP_OPN_FAMILY_MEXP_CLUSTER_PAIR_CONFUSION = 27
comptime MEXP_OPN_FAMILY_MEXP_CLUSTER_NMI = 28
comptime MEXP_OPN_FAMILY_MEXP_CLUSTER_AMI = 29
comptime MEXP_OPN_FAMILY_MEXP_CLUSTER_CALINSKI_HARABASZ = 30
comptime MEXP_OPN_FAMILY_MEXP_CLUSTER_DAVIES_BOULDIN = 31


def operation_names_mexp(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_mexp_cls_balanced_accuracy")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_BALANCED_ACCURACY)
    names.append("family_mexp_cls_fbeta")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_FBETA)
    names.append("family_mexp_cls_hamming")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_HAMMING)
    names.append("family_mexp_cls_jaccard")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_JACCARD)
    names.append("family_mexp_cls_kappa")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_KAPPA)
    names.append("family_mexp_cls_likelihood_ratios")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_LIKELIHOOD_RATIOS)
    names.append("family_mexp_cls_matthews")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_MATTHEWS)
    names.append("family_mexp_cls_mlcm")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_MLCM)
    names.append("family_mexp_cls_prfs")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_PRFS)
    names.append("family_mexp_cls_report")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_REPORT)
    names.append("family_mexp_cls_zero_one")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLS_ZERO_ONE)
    names.append("family_mexp_cluster_ami")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLUSTER_AMI)
    names.append("family_mexp_cluster_calinski_harabasz")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLUSTER_CALINSKI_HARABASZ)
    names.append("family_mexp_cluster_contingency")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLUSTER_CONTINGENCY)
    names.append("family_mexp_cluster_davies_bouldin")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLUSTER_DAVIES_BOULDIN)
    names.append("family_mexp_cluster_nmi")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLUSTER_NMI)
    names.append("family_mexp_cluster_pair_confusion")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CLUSTER_PAIR_CONFUSION)
    names.append("family_mexp_curve_auc")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CURVE_AUC)
    names.append("family_mexp_curve_average_precision")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CURVE_AVERAGE_PRECISION)
    names.append("family_mexp_curve_det")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CURVE_DET)
    names.append("family_mexp_curve_ranking_report")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CURVE_RANKING_REPORT)
    names.append("family_mexp_curve_roc")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CURVE_ROC)
    names.append("family_mexp_curve_top_k")
    leaves.append(MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K)
    names.append("family_mexp_rank_brier")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_BRIER)
    names.append("family_mexp_rank_coverage")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_COVERAGE)
    names.append("family_mexp_rank_d2_brier")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_D2_BRIER)
    names.append("family_mexp_rank_d2_log_loss")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_D2_LOG_LOSS)
    names.append("family_mexp_rank_dcg")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_DCG)
    names.append("family_mexp_rank_hinge")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_HINGE)
    names.append("family_mexp_rank_loss")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_LOSS)
    names.append("family_mexp_rank_lrap")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_LRAP)
    names.append("family_mexp_rank_ndcg")
    leaves.append(MEXP_OPN_FAMILY_MEXP_RANK_NDCG)
