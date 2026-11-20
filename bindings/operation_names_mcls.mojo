# SPDX-License-Identifier: Apache-2.0
# The `family_mcls_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-metrics, 2026-11-20): each name has a leaf code,
# numbered by sub-owner so `metrics_cls_operation`
# (bindings/metrics_cls_native_unpack.mojo) picks the owner with one range test: classify 0-4
# (bindings/mcls_classify_door.mojo), ranking 5-6 (mcls_ranking_door), cluster 7-21 (mcls_cluster_door).
# The family door resolves a caller's str (or a shell's table code) to the leaf
# with one lookup and never builds the name; a `family_mcls_*` name the table does
# not hold keeps its String for its unchanged error text only. The appends keep
# their order (the table's indices do not move); the leaves are not in append
# order. A constant is `MCLS_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime MCLS_OPN_FAMILY_MCLS_ACCURACY = 0
comptime MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION = 1
comptime MCLS_OPN_FAMILY_MCLS_CONFUSION = 2
comptime MCLS_OPN_FAMILY_MCLS_PRF = 3
comptime MCLS_OPN_FAMILY_MCLS_LOG_LOSS = 4
comptime MCLS_OPN_FAMILY_MCLS_ROC_AUC = 5
comptime MCLS_OPN_FAMILY_MCLS_PR_CURVE = 6
comptime MCLS_OPN_FAMILY_MCLS_RAND = 7
comptime MCLS_OPN_FAMILY_MCLS_ADJUSTED_RAND = 8
comptime MCLS_OPN_FAMILY_MCLS_ENTROPY = 9
comptime MCLS_OPN_FAMILY_MCLS_MUTUAL_INFO = 10
comptime MCLS_OPN_FAMILY_MCLS_FOWLKES_MALLOWS = 11
comptime MCLS_OPN_FAMILY_MCLS_HOMOGENEITY = 12
comptime MCLS_OPN_FAMILY_MCLS_COMPLETENESS = 13
comptime MCLS_OPN_FAMILY_MCLS_V_MEASURE = 14
comptime MCLS_OPN_FAMILY_MCLS_HCV = 15
comptime MCLS_OPN_FAMILY_MCLS_KL = 16
comptime MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SCORE = 17
comptime MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SAMPLES = 18
comptime MCLS_OPN_FAMILY_MCLS_TRUSTWORTHINESS = 19
comptime MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR = 20
comptime MCLS_OPN_FAMILY_MCLS_R2 = 21


def operation_names_mcls(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_mcls_accuracy")
    leaves.append(MCLS_OPN_FAMILY_MCLS_ACCURACY)
    names.append("family_mcls_accuracy_fraction")
    leaves.append(MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION)
    names.append("family_mcls_adjusted_rand")
    leaves.append(MCLS_OPN_FAMILY_MCLS_ADJUSTED_RAND)
    names.append("family_mcls_completeness")
    leaves.append(MCLS_OPN_FAMILY_MCLS_COMPLETENESS)
    names.append("family_mcls_confusion")
    leaves.append(MCLS_OPN_FAMILY_MCLS_CONFUSION)
    names.append("family_mcls_entropy")
    leaves.append(MCLS_OPN_FAMILY_MCLS_ENTROPY)
    names.append("family_mcls_fowlkes_mallows")
    leaves.append(MCLS_OPN_FAMILY_MCLS_FOWLKES_MALLOWS)
    names.append("family_mcls_hcv")
    leaves.append(MCLS_OPN_FAMILY_MCLS_HCV)
    names.append("family_mcls_homogeneity")
    leaves.append(MCLS_OPN_FAMILY_MCLS_HOMOGENEITY)
    names.append("family_mcls_kl")
    leaves.append(MCLS_OPN_FAMILY_MCLS_KL)
    names.append("family_mcls_log_loss")
    leaves.append(MCLS_OPN_FAMILY_MCLS_LOG_LOSS)
    names.append("family_mcls_mutual_info")
    leaves.append(MCLS_OPN_FAMILY_MCLS_MUTUAL_INFO)
    names.append("family_mcls_pr_curve")
    leaves.append(MCLS_OPN_FAMILY_MCLS_PR_CURVE)
    names.append("family_mcls_prf")
    leaves.append(MCLS_OPN_FAMILY_MCLS_PRF)
    names.append("family_mcls_r2")
    leaves.append(MCLS_OPN_FAMILY_MCLS_R2)
    names.append("family_mcls_rand")
    leaves.append(MCLS_OPN_FAMILY_MCLS_RAND)
    names.append("family_mcls_regression_error")
    leaves.append(MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR)
    names.append("family_mcls_roc_auc")
    leaves.append(MCLS_OPN_FAMILY_MCLS_ROC_AUC)
    names.append("family_mcls_silhouette_samples")
    leaves.append(MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SAMPLES)
    names.append("family_mcls_silhouette_score")
    leaves.append(MCLS_OPN_FAMILY_MCLS_SILHOUETTE_SCORE)
    names.append("family_mcls_trustworthiness")
    leaves.append(MCLS_OPN_FAMILY_MCLS_TRUSTWORTHINESS)
    names.append("family_mcls_v_measure")
    leaves.append(MCLS_OPN_FAMILY_MCLS_V_MEASURE)
