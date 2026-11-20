# SPDX-License-Identifier: Apache-2.0
# The `family_mexp_*` operation names of the classical owner's code table
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


def operation_names_mexp(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
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
