# SPDX-License-Identifier: Apache-2.0
# The `family_gp_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (the worked example of the prefix route): each name has a leaf code,
# numbered by sub-owner (kernel, regressor, classifier) so `gp_operation`
# picks the owner with one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown `family_gp_*` name is not in the table and keeps its
# String, for its unchanged error text only (`gp_unknown_operation`).
# A constant is `GP_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime GP_OPN_FAMILY_GP_KERNEL_INITIALIZE = 0
comptime GP_OPN_FAMILY_GP_KERNEL_ADD = 1
comptime GP_OPN_FAMILY_GP_KERNEL_MUL = 2
comptime GP_OPN_FAMILY_GP_KERNEL_REPR = 3
comptime GP_OPN_FAMILY_GP_KERNEL_N_DIMS = 4
comptime GP_OPN_FAMILY_GP_KERNEL_THETA = 5
comptime GP_OPN_FAMILY_GP_KERNEL_BOUNDS = 6
comptime GP_OPN_FAMILY_GP_GPR_FIT = 7
comptime GP_OPN_FAMILY_GP_GPR_PREDICT = 8
comptime GP_OPN_FAMILY_GP_GPR_SAMPLE_Y = 9
comptime GP_OPN_FAMILY_GP_GPR_LML = 10
comptime GP_OPN_FAMILY_GP_GPR_SCORE = 11
comptime GP_OPN_FAMILY_GP_GPC_LML = 12
comptime GP_OPN_FAMILY_GP_GPC_FIT = 13
comptime GP_OPN_FAMILY_GP_GPC_PREDICT = 14
comptime GP_OPN_FAMILY_GP_GPC_PREDICT_PROBA = 15
comptime GP_OPN_FAMILY_GP_GPC_LATENT = 16
comptime GP_OPN_FAMILY_GP_GPC_SCORE = 17
# The device read-backs (lane dw-gp-metrics-manifold): above every owner's
# range, tested first by `gp_operation`.
comptime GP_OPN_FAMILY_GP_GPR_VENDOR_USED = 18
comptime GP_OPN_FAMILY_GP_GPC_VENDOR_USED = 19


def operation_names_gp(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_gp_kernel_initialize")
    leaves.append(GP_OPN_FAMILY_GP_KERNEL_INITIALIZE)
    names.append("family_gp_kernel_add")
    leaves.append(GP_OPN_FAMILY_GP_KERNEL_ADD)
    names.append("family_gp_kernel_mul")
    leaves.append(GP_OPN_FAMILY_GP_KERNEL_MUL)
    names.append("family_gp_kernel_repr")
    leaves.append(GP_OPN_FAMILY_GP_KERNEL_REPR)
    names.append("family_gp_kernel_n_dims")
    leaves.append(GP_OPN_FAMILY_GP_KERNEL_N_DIMS)
    names.append("family_gp_kernel_theta")
    leaves.append(GP_OPN_FAMILY_GP_KERNEL_THETA)
    names.append("family_gp_kernel_bounds")
    leaves.append(GP_OPN_FAMILY_GP_KERNEL_BOUNDS)
    names.append("family_gp_gpr_fit")
    leaves.append(GP_OPN_FAMILY_GP_GPR_FIT)
    names.append("family_gp_gpr_predict")
    leaves.append(GP_OPN_FAMILY_GP_GPR_PREDICT)
    names.append("family_gp_gpr_sample_y")
    leaves.append(GP_OPN_FAMILY_GP_GPR_SAMPLE_Y)
    names.append("family_gp_gpr_lml")
    leaves.append(GP_OPN_FAMILY_GP_GPR_LML)
    names.append("family_gp_gpr_score")
    leaves.append(GP_OPN_FAMILY_GP_GPR_SCORE)
    names.append("family_gp_gpc_lml")
    leaves.append(GP_OPN_FAMILY_GP_GPC_LML)
    names.append("family_gp_gpc_fit")
    leaves.append(GP_OPN_FAMILY_GP_GPC_FIT)
    names.append("family_gp_gpc_predict")
    leaves.append(GP_OPN_FAMILY_GP_GPC_PREDICT)
    names.append("family_gp_gpc_predict_proba")
    leaves.append(GP_OPN_FAMILY_GP_GPC_PREDICT_PROBA)
    names.append("family_gp_gpc_latent")
    leaves.append(GP_OPN_FAMILY_GP_GPC_LATENT)
    names.append("family_gp_gpc_score")
    leaves.append(GP_OPN_FAMILY_GP_GPC_SCORE)
    names.append("family_gp_gpr_vendor_used")
    leaves.append(GP_OPN_FAMILY_GP_GPR_VENDOR_USED)
    names.append("family_gp_gpc_vendor_used")
    leaves.append(GP_OPN_FAMILY_GP_GPC_VENDOR_USED)
