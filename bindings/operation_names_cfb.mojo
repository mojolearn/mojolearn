# SPDX-License-Identifier: Apache-2.0
# The `family_cfb_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-cluster, the form of bindings/operation_names_gp.mojo): each
# name has a leaf code, numbered by sub-owner so each prefix test of the
# owner's chain is one range test: DBSCAN 0-3 (3 is the x_cluster DBSCAN predict the family door routes here:
# no append in this file, its name is the XC table entry), Agglomerative 4-8,
# spectral 9-18 (SpectralClustering 9-12, SpectralEmbedding 13-16).
# The family door resolves a caller's str to the code with ONE dictionary
# lookup and never builds the name; a name this file does not hold keeps
# its String, for its unchanged error text only. A constant is
# `CFB_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime CFB_OPN_FAMILY_CFB_DBSCAN_FIT = 0
comptime CFB_OPN_FAMILY_CFB_DBSCAN_FIT_PREDICT = 1
comptime CFB_OPN_FAMILY_CFB_DBSCAN_PREDICT = 2
comptime CFB_OPN_FAMILY_XC_DBSCAN_PREDICT = 3
comptime CFB_OPN_FAMILY_CFB_AGGLO_FIT = 4
comptime CFB_OPN_FAMILY_CFB_AGGLO_FIT_PREDICT = 5
comptime CFB_OPN_FAMILY_CFB_AGGLO_PREDICT = 6
comptime CFB_OPN_FAMILY_CFB_AGGLO_GETSTATE = 7
comptime CFB_OPN_FAMILY_CFB_AGGLO_SETSTATE = 8
comptime CFB_OPN_FAMILY_CFB_SC_FIT = 9
comptime CFB_OPN_FAMILY_CFB_SC_FIT_PREDICT = 10
comptime CFB_OPN_FAMILY_CFB_SC_PREDICT = 11
comptime CFB_OPN_FAMILY_CFB_SC_PREDICT_EMBEDDING = 12
comptime CFB_OPN_FAMILY_CFB_SE_FIT = 13
comptime CFB_OPN_FAMILY_CFB_SE_FIT_TRANSFORM = 14
comptime CFB_OPN_FAMILY_CFB_SE_VALIDATE = 15
comptime CFB_OPN_FAMILY_CFB_SE_KNN_AFFINITY = 16
comptime CFB_OPN_FAMILY_CFB_SPECTRAL_EMBEDDING = 17
comptime CFB_OPN_FAMILY_CFB_COO_TRIPLES = 18
# `vendor_used()` of AgglomerativeClustering and of SpectralClustering /
# SpectralEmbedding: metadata, no input, served before any lifecycle step.
comptime CFB_OPN_FAMILY_CFB_AGGLO_VENDOR_USED = 19
comptime CFB_OPN_FAMILY_CFB_SPECTRAL_VENDOR_USED = 20

def operation_names_cfb(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_cfb_agglo_fit")
    leaves.append(CFB_OPN_FAMILY_CFB_AGGLO_FIT)
    names.append("family_cfb_agglo_fit_predict")
    leaves.append(CFB_OPN_FAMILY_CFB_AGGLO_FIT_PREDICT)
    names.append("family_cfb_agglo_getstate")
    leaves.append(CFB_OPN_FAMILY_CFB_AGGLO_GETSTATE)
    names.append("family_cfb_agglo_predict")
    leaves.append(CFB_OPN_FAMILY_CFB_AGGLO_PREDICT)
    names.append("family_cfb_agglo_setstate")
    leaves.append(CFB_OPN_FAMILY_CFB_AGGLO_SETSTATE)
    names.append("family_cfb_agglo_vendor_used")
    leaves.append(CFB_OPN_FAMILY_CFB_AGGLO_VENDOR_USED)
    names.append("family_cfb_coo_triples")
    leaves.append(CFB_OPN_FAMILY_CFB_COO_TRIPLES)
    names.append("family_cfb_dbscan_fit")
    leaves.append(CFB_OPN_FAMILY_CFB_DBSCAN_FIT)
    names.append("family_cfb_dbscan_fit_predict")
    leaves.append(CFB_OPN_FAMILY_CFB_DBSCAN_FIT_PREDICT)
    names.append("family_cfb_dbscan_predict")
    leaves.append(CFB_OPN_FAMILY_CFB_DBSCAN_PREDICT)
    names.append("family_cfb_sc_fit")
    leaves.append(CFB_OPN_FAMILY_CFB_SC_FIT)
    names.append("family_cfb_sc_fit_predict")
    leaves.append(CFB_OPN_FAMILY_CFB_SC_FIT_PREDICT)
    names.append("family_cfb_sc_predict")
    leaves.append(CFB_OPN_FAMILY_CFB_SC_PREDICT)
    names.append("family_cfb_sc_predict_embedding")
    leaves.append(CFB_OPN_FAMILY_CFB_SC_PREDICT_EMBEDDING)
    names.append("family_cfb_se_fit")
    leaves.append(CFB_OPN_FAMILY_CFB_SE_FIT)
    names.append("family_cfb_se_fit_transform")
    leaves.append(CFB_OPN_FAMILY_CFB_SE_FIT_TRANSFORM)
    names.append("family_cfb_se_knn_affinity")
    leaves.append(CFB_OPN_FAMILY_CFB_SE_KNN_AFFINITY)
    names.append("family_cfb_se_validate")
    leaves.append(CFB_OPN_FAMILY_CFB_SE_VALIDATE)
    names.append("family_cfb_spectral_embedding")
    leaves.append(CFB_OPN_FAMILY_CFB_SPECTRAL_EMBEDDING)
    names.append("family_cfb_spectral_vendor_used")
    leaves.append(CFB_OPN_FAMILY_CFB_SPECTRAL_VENDOR_USED)
