# SPDX-License-Identifier: Apache-2.0
# The `family_lin_* / family_rs_* / family_pg_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-linalg, 2026-11-20; the form of bindings/operation_names_gp.mojo):
# each name has a leaf code, numbered by sub-owner (QR/SVD results 0-1, decompositions 2-5, low-bit and GEMM 6-20, resampling 21-26) so a prefix test
# of the owner's chain is one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown name of these prefixes is not in the table and keeps its
# String, for its unchanged error text only (`linalg_ops_unknown_operation`).
# A constant is `LINOPS_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Add every name a caller
# can pass: a name missing here is refused as unknown. No `family_pg_*` name exists
# (the route word keeps the prefix; any such name is refused as unknown, as before).

comptime LINOPS_OPN_FAMILY_LIN_RESULT_NEW = 0
comptime LINOPS_OPN_FAMILY_LIN_RESULT_ITEM = 1
comptime LINOPS_OPN_FAMILY_LIN_QR = 2
comptime LINOPS_OPN_FAMILY_LIN_EIGH = 3
comptime LINOPS_OPN_FAMILY_LIN_SVDVALS = 4
comptime LINOPS_OPN_FAMILY_LIN_SVD = 5
comptime LINOPS_OPN_FAMILY_LIN_NUMERIC_MODE = 6
comptime LINOPS_OPN_FAMILY_LIN_PROFILE = 7
comptime LINOPS_OPN_FAMILY_LIN_REQUIRE_IDENTICAL = 8
comptime LINOPS_OPN_FAMILY_LIN_MATMUL = 9
comptime LINOPS_OPN_FAMILY_LIN_MATMUL_FUSED = 10
comptime LINOPS_OPN_FAMILY_LIN_MATMUL_BF16 = 11
comptime LINOPS_OPN_FAMILY_LIN_TO_BF16 = 12
comptime LINOPS_OPN_FAMILY_LIN_FROM_BF16 = 13
comptime LINOPS_OPN_FAMILY_LIN_FROM_F16 = 14
comptime LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT8 = 15
comptime LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8 = 16
comptime LINOPS_OPN_FAMILY_LIN_MATMUL_INT8 = 17
comptime LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT15 = 18
comptime LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT15 = 19
comptime LINOPS_OPN_FAMILY_LIN_MATMUL_INT15 = 20
comptime LINOPS_OPN_FAMILY_RS_RESULT_INITIALIZE = 21
comptime LINOPS_OPN_FAMILY_RS_RESAMPLE = 22
comptime LINOPS_OPN_FAMILY_RS_BOOTSTRAP = 23
comptime LINOPS_OPN_FAMILY_RS_PERMUTATION_TEST = 24
comptime LINOPS_OPN_FAMILY_RS_MONTE_CARLO_INTEGRATE = 25
comptime LINOPS_OPN_FAMILY_RS_RESAMPLE_INDICES = 26


def operation_names_linops(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_lin_result_new")
    leaves.append(LINOPS_OPN_FAMILY_LIN_RESULT_NEW)
    names.append("family_lin_result_item")
    leaves.append(LINOPS_OPN_FAMILY_LIN_RESULT_ITEM)
    names.append("family_lin_qr")
    leaves.append(LINOPS_OPN_FAMILY_LIN_QR)
    names.append("family_lin_eigh")
    leaves.append(LINOPS_OPN_FAMILY_LIN_EIGH)
    names.append("family_lin_svdvals")
    leaves.append(LINOPS_OPN_FAMILY_LIN_SVDVALS)
    names.append("family_lin_svd")
    leaves.append(LINOPS_OPN_FAMILY_LIN_SVD)
    names.append("family_lin_numeric_mode")
    leaves.append(LINOPS_OPN_FAMILY_LIN_NUMERIC_MODE)
    names.append("family_lin_profile")
    leaves.append(LINOPS_OPN_FAMILY_LIN_PROFILE)
    names.append("family_lin_require_identical")
    leaves.append(LINOPS_OPN_FAMILY_LIN_REQUIRE_IDENTICAL)
    names.append("family_lin_matmul")
    leaves.append(LINOPS_OPN_FAMILY_LIN_MATMUL)
    names.append("family_lin_matmul_fused")
    leaves.append(LINOPS_OPN_FAMILY_LIN_MATMUL_FUSED)
    names.append("family_lin_matmul_bf16")
    leaves.append(LINOPS_OPN_FAMILY_LIN_MATMUL_BF16)
    names.append("family_lin_to_bf16")
    leaves.append(LINOPS_OPN_FAMILY_LIN_TO_BF16)
    names.append("family_lin_from_bf16")
    leaves.append(LINOPS_OPN_FAMILY_LIN_FROM_BF16)
    names.append("family_lin_from_f16")
    leaves.append(LINOPS_OPN_FAMILY_LIN_FROM_F16)
    names.append("family_lin_quantize_int8")
    leaves.append(LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT8)
    names.append("family_lin_dequantize_int8")
    leaves.append(LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8)
    names.append("family_lin_matmul_int8")
    leaves.append(LINOPS_OPN_FAMILY_LIN_MATMUL_INT8)
    names.append("family_lin_quantize_int15")
    leaves.append(LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT15)
    names.append("family_lin_dequantize_int15")
    leaves.append(LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT15)
    names.append("family_lin_matmul_int15")
    leaves.append(LINOPS_OPN_FAMILY_LIN_MATMUL_INT15)
    names.append("family_rs_result_initialize")
    leaves.append(LINOPS_OPN_FAMILY_RS_RESULT_INITIALIZE)
    names.append("family_rs_resample")
    leaves.append(LINOPS_OPN_FAMILY_RS_RESAMPLE)
    names.append("family_rs_bootstrap")
    leaves.append(LINOPS_OPN_FAMILY_RS_BOOTSTRAP)
    names.append("family_rs_permutation_test")
    leaves.append(LINOPS_OPN_FAMILY_RS_PERMUTATION_TEST)
    names.append("family_rs_monte_carlo_integrate")
    leaves.append(LINOPS_OPN_FAMILY_RS_MONTE_CARLO_INTEGRATE)
    names.append("family_rs_resample_indices")
    leaves.append(LINOPS_OPN_FAMILY_RS_RESAMPLE_INDICES)
