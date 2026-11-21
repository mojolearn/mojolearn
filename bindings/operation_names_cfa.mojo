# SPDX-License-Identifier: Apache-2.0
# The `family_cfa_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-cluster, the form of bindings/operation_names_gp.mojo): each
# name has a leaf code, numbered by sub-owner so each prefix test of the
# owner's chain is one range test: MiniBatchKMeans 0-1, BisectingKMeans 2, the centers fits 3-4, MeanShift 5-6,
# OPTICS 7-8, AffinityPropagation 9-10, HDBSCAN 11-12.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup and never builds the name; a name this file does not hold keeps
# its String, for its unchanged error text only. A constant is
# `CFA_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime CFA_OPN_FAMILY_CFA_MBK_FIT = 0
comptime CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT = 1
comptime CFA_OPN_FAMILY_CFA_BISECT_FIT = 2
comptime CFA_OPN_FAMILY_CFA_FIT_PREDICT = 3
comptime CFA_OPN_FAMILY_CFA_FIT_TRANSFORM = 4
comptime CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT = 5
comptime CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT = 6
comptime CFA_OPN_FAMILY_CFA_OPTICS_FIT = 7
comptime CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT = 8
comptime CFA_OPN_FAMILY_CFA_AP_FIT = 9
comptime CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT = 10
comptime CFA_OPN_FAMILY_CFA_HDBSCAN_FIT = 11

def operation_names_cfa(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_cfa_ap_fit")
    leaves.append(CFA_OPN_FAMILY_CFA_AP_FIT)
    names.append("family_cfa_ap_fit_predict")
    leaves.append(CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT)
    names.append("family_cfa_bisect_fit")
    leaves.append(CFA_OPN_FAMILY_CFA_BISECT_FIT)
    names.append("family_cfa_fit_predict")
    leaves.append(CFA_OPN_FAMILY_CFA_FIT_PREDICT)
    names.append("family_cfa_fit_transform")
    leaves.append(CFA_OPN_FAMILY_CFA_FIT_TRANSFORM)
    names.append("family_cfa_hdbscan_fit")
    leaves.append(CFA_OPN_FAMILY_CFA_HDBSCAN_FIT)
    names.append("family_cfa_mbk_fit")
    leaves.append(CFA_OPN_FAMILY_CFA_MBK_FIT)
    names.append("family_cfa_mbk_partial_fit")
    leaves.append(CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT)
    names.append("family_cfa_meanshift_fit")
    leaves.append(CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT)
    names.append("family_cfa_meanshift_fit_predict")
    leaves.append(CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT)
    names.append("family_cfa_optics_fit")
    leaves.append(CFA_OPN_FAMILY_CFA_OPTICS_FIT)
    names.append("family_cfa_optics_fit_predict")
    leaves.append(CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT)
