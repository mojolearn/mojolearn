# SPDX-License-Identifier: Apache-2.0
# The `family_svm_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-svm, 2026-11-20; the form of bindings/operation_names_gp.mojo):
# each name has a leaf code, numbered by sub-owner (linear queries, GaussianMixture sample) so a prefix test
# of the owner's chain is one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown `family_svm_*` name is not in the table and keeps its
# String, for its unchanged error text only (`svm_mixture_unknown_operation`).
# A constant is `SVM_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Add every name a caller
# can pass: a name missing here is refused as unknown.

comptime SVM_OPN_FAMILY_SVM_LINEAR_SVC_DECISION = 0
comptime SVM_OPN_FAMILY_SVM_LINEAR_SVR_PREDICT = 1
comptime SVM_OPN_FAMILY_SVM_GMM_SAMPLE = 2


def operation_names_svm(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_svm_linear_svc_decision")
    leaves.append(SVM_OPN_FAMILY_SVM_LINEAR_SVC_DECISION)
    names.append("family_svm_linear_svr_predict")
    leaves.append(SVM_OPN_FAMILY_SVM_LINEAR_SVR_PREDICT)
    names.append("family_svm_gmm_sample")
    leaves.append(SVM_OPN_FAMILY_SVM_GMM_SAMPLE)
