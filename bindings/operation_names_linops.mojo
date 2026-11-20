# SPDX-License-Identifier: Apache-2.0
# The `family_lin_* / family_rs_* / family_pg_*` operation names of the classical owner's code table
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


def operation_names_linops(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_lin_dequantize_int15")
    names.append("family_lin_dequantize_int8")
    names.append("family_lin_eigh")
    names.append("family_lin_from_bf16")
    names.append("family_lin_from_f16")
    names.append("family_lin_matmul")
    names.append("family_lin_matmul_bf16")
    names.append("family_lin_matmul_fused")
    names.append("family_lin_matmul_int15")
    names.append("family_lin_matmul_int8")
    names.append("family_lin_numeric_mode")
    names.append("family_lin_profile")
    names.append("family_lin_qr")
    names.append("family_lin_quantize_int15")
    names.append("family_lin_quantize_int8")
    names.append("family_lin_require_identical")
    names.append("family_lin_result_item")
    names.append("family_lin_result_new")
    names.append("family_lin_svd")
    names.append("family_lin_svdvals")
    names.append("family_lin_to_bf16")
    names.append("family_rs_bootstrap")
    names.append("family_rs_monte_carlo_integrate")
    names.append("family_rs_permutation_test")
    names.append("family_rs_resample")
    names.append("family_rs_resample_indices")
    names.append("family_rs_result_initialize")
