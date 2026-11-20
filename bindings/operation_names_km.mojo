# SPDX-License-Identifier: Apache-2.0
# The `family_km_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-svm, 2026-11-20; the form of bindings/operation_names_gp.mojo):
# each name has a leaf code, numbered by sub-owner (state, cooperative, KernelRidge, Nystroem, RBFSampler) so a prefix test
# of the owner's chain is one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown `family_km_*` name is not in the table and keeps its
# String, for its unchanged error text only (`kernel_methods_unknown_operation`).
# A constant is `KM_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Add every name a caller
# can pass: a name missing here is refused as unknown.

comptime KM_OPN_FAMILY_KM_GETSTATE = 0
comptime KM_OPN_FAMILY_KM_SETSTATE = 1
comptime KM_OPN_FAMILY_KM_PAR_FIT = 2
comptime KM_OPN_FAMILY_KM_PAR_APPLY = 3
comptime KM_OPN_FAMILY_KM_KRR_FIT = 4
comptime KM_OPN_FAMILY_KM_KRR_PREDICT = 5
comptime KM_OPN_FAMILY_KM_KRR_SCORE = 6
comptime KM_OPN_FAMILY_KM_NYS_FIT = 7
comptime KM_OPN_FAMILY_KM_NYS_FIT_TRANSFORM = 8
comptime KM_OPN_FAMILY_KM_NYS_TRANSFORM = 9
comptime KM_OPN_FAMILY_KM_RBF_FIT = 10
comptime KM_OPN_FAMILY_KM_RBF_FIT_TRANSFORM = 11
comptime KM_OPN_FAMILY_KM_RBF_TRANSFORM = 12


def operation_names_km(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_km_getstate")
    leaves.append(KM_OPN_FAMILY_KM_GETSTATE)
    names.append("family_km_setstate")
    leaves.append(KM_OPN_FAMILY_KM_SETSTATE)
    names.append("family_km_par_fit")
    leaves.append(KM_OPN_FAMILY_KM_PAR_FIT)
    names.append("family_km_par_apply")
    leaves.append(KM_OPN_FAMILY_KM_PAR_APPLY)
    names.append("family_km_krr_fit")
    leaves.append(KM_OPN_FAMILY_KM_KRR_FIT)
    names.append("family_km_krr_predict")
    leaves.append(KM_OPN_FAMILY_KM_KRR_PREDICT)
    names.append("family_km_krr_score")
    leaves.append(KM_OPN_FAMILY_KM_KRR_SCORE)
    names.append("family_km_nys_fit")
    leaves.append(KM_OPN_FAMILY_KM_NYS_FIT)
    names.append("family_km_nys_fit_transform")
    leaves.append(KM_OPN_FAMILY_KM_NYS_FIT_TRANSFORM)
    names.append("family_km_nys_transform")
    leaves.append(KM_OPN_FAMILY_KM_NYS_TRANSFORM)
    names.append("family_km_rbf_fit")
    leaves.append(KM_OPN_FAMILY_KM_RBF_FIT)
    names.append("family_km_rbf_fit_transform")
    leaves.append(KM_OPN_FAMILY_KM_RBF_FIT_TRANSFORM)
    names.append("family_km_rbf_transform")
    leaves.append(KM_OPN_FAMILY_KM_RBF_TRANSFORM)
