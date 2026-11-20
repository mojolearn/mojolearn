# SPDX-License-Identifier: Apache-2.0
# The `family_xc_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-cluster, the form of bindings/operation_names_gp.mojo): each
# name has a leaf code, numbered by sub-owner so each prefix test of the
# owner's chain is one range test: x_cluster queries 1-6 (the public cluster door's flagged names), the
# AgglomerativeClustering reference predict 7 and the DBSCAN predict 8 (the
# family door routes 8 to the fit-b owner as CFB_OPN_FAMILY_XC_DBSCAN_PREDICT).
# The family door resolves a caller's str to the code with ONE dictionary
# lookup and never builds the name; a name this file does not hold keeps
# its String, for its unchanged error text only. A constant is
# `XC_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime XC_OPN_FAMILY_XC_INITIALIZE = 0
comptime XC_OPN_FAMILY_XC_PREDICT = 1
comptime XC_OPN_FAMILY_XC_VARIANTS_PREDICT = 2
comptime XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM = 3
comptime XC_OPN_FAMILY_XC_VARIANTS_SCORE = 4
comptime XC_OPN_FAMILY_XC_BISECT_PREDICT = 5
comptime XC_OPN_FAMILY_XC_AP_PREDICT = 6
comptime XC_OPN_FAMILY_XC_AGGLO_PREDICT = 7
comptime XC_OPN_FAMILY_XC_DBSCAN_PREDICT = 8

def operation_names_xc(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_xc_agglo_predict")
    leaves.append(XC_OPN_FAMILY_XC_AGGLO_PREDICT)
    names.append("family_xc_ap_predict")
    leaves.append(XC_OPN_FAMILY_XC_AP_PREDICT)
    names.append("family_xc_bisect_predict")
    leaves.append(XC_OPN_FAMILY_XC_BISECT_PREDICT)
    names.append("family_xc_dbscan_predict")
    leaves.append(XC_OPN_FAMILY_XC_DBSCAN_PREDICT)
    names.append("family_xc_initialize")
    leaves.append(XC_OPN_FAMILY_XC_INITIALIZE)
    names.append("family_xc_predict")
    leaves.append(XC_OPN_FAMILY_XC_PREDICT)
    names.append("family_xc_variants_predict")
    leaves.append(XC_OPN_FAMILY_XC_VARIANTS_PREDICT)
    names.append("family_xc_variants_score")
    leaves.append(XC_OPN_FAMILY_XC_VARIANTS_SCORE)
    names.append("family_xc_variants_transform")
    leaves.append(XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM)
