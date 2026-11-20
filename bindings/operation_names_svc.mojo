# SPDX-License-Identifier: Apache-2.0
# The `family_svc_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-svm, 2026-11-20; the form of bindings/operation_names_gp.mojo):
# each name has a leaf code, numbered by sub-owner (pickle state, linear, SVC, SVR, OneClassSVM) so a prefix test
# of the owner's chain is one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown `family_svc_*` name is not in the table and keeps its
# String, for its unchanged error text only (`svc_unknown_operation`).
# A constant is `SVC_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Add every name a caller
# can pass: a name missing here is refused as unknown.

comptime SVC_OPN_FAMILY_SVC_PUBLIC_SVC_SETSTATE = 0
comptime SVC_OPN_FAMILY_SVC_PUBLIC_SVR_SETSTATE = 1
comptime SVC_OPN_FAMILY_SVC_LINEAR_SVC_FIT = 2
comptime SVC_OPN_FAMILY_SVC_LINEAR_SVC_PREDICT = 3
comptime SVC_OPN_FAMILY_SVC_LINEAR_SVC_SCORE = 4
comptime SVC_OPN_FAMILY_SVC_LINEAR_SVR_FIT = 5
comptime SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE = 6
comptime SVC_OPN_FAMILY_SVC_SVC_COEF = 7
comptime SVC_OPN_FAMILY_SVC_SVC_DECISION = 8
comptime SVC_OPN_FAMILY_SVC_SVC_FIT = 9
comptime SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA = 10
comptime SVC_OPN_FAMILY_SVC_SVC_PREDICT = 11
comptime SVC_OPN_FAMILY_SVC_SVC_PROBA = 12
comptime SVC_OPN_FAMILY_SVC_SVC_SCORE = 13
comptime SVC_OPN_FAMILY_SVC_SVC_VALIDATE = 14
comptime SVC_OPN_FAMILY_SVC_SVR_COEF = 15
comptime SVC_OPN_FAMILY_SVC_SVR_FIT = 16
comptime SVC_OPN_FAMILY_SVC_SVR_PREDICT = 17
comptime SVC_OPN_FAMILY_SVC_SVR_SCORE = 18
comptime SVC_OPN_FAMILY_SVC_SVR_VALIDATE = 19
comptime SVC_OPN_FAMILY_SVC_OCSVM_DECISION = 20
comptime SVC_OPN_FAMILY_SVC_OCSVM_FIT = 21
comptime SVC_OPN_FAMILY_SVC_OCSVM_FIT_PREDICT = 22
comptime SVC_OPN_FAMILY_SVC_OCSVM_PREDICT = 23
comptime SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES = 24


def operation_names_svc(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_svc_public_svc_setstate")
    leaves.append(SVC_OPN_FAMILY_SVC_PUBLIC_SVC_SETSTATE)
    names.append("family_svc_public_svr_setstate")
    leaves.append(SVC_OPN_FAMILY_SVC_PUBLIC_SVR_SETSTATE)
    names.append("family_svc_linear_svc_fit")
    leaves.append(SVC_OPN_FAMILY_SVC_LINEAR_SVC_FIT)
    names.append("family_svc_linear_svc_predict")
    leaves.append(SVC_OPN_FAMILY_SVC_LINEAR_SVC_PREDICT)
    names.append("family_svc_linear_svc_score")
    leaves.append(SVC_OPN_FAMILY_SVC_LINEAR_SVC_SCORE)
    names.append("family_svc_linear_svr_fit")
    leaves.append(SVC_OPN_FAMILY_SVC_LINEAR_SVR_FIT)
    names.append("family_svc_linear_svr_score")
    leaves.append(SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE)
    names.append("family_svc_svc_coef")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_COEF)
    names.append("family_svc_svc_decision")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_DECISION)
    names.append("family_svc_svc_fit")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_FIT)
    names.append("family_svc_svc_log_proba")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA)
    names.append("family_svc_svc_predict")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_PREDICT)
    names.append("family_svc_svc_proba")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_PROBA)
    names.append("family_svc_svc_score")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_SCORE)
    names.append("family_svc_svc_validate")
    leaves.append(SVC_OPN_FAMILY_SVC_SVC_VALIDATE)
    names.append("family_svc_svr_coef")
    leaves.append(SVC_OPN_FAMILY_SVC_SVR_COEF)
    names.append("family_svc_svr_fit")
    leaves.append(SVC_OPN_FAMILY_SVC_SVR_FIT)
    names.append("family_svc_svr_predict")
    leaves.append(SVC_OPN_FAMILY_SVC_SVR_PREDICT)
    names.append("family_svc_svr_score")
    leaves.append(SVC_OPN_FAMILY_SVC_SVR_SCORE)
    names.append("family_svc_svr_validate")
    leaves.append(SVC_OPN_FAMILY_SVC_SVR_VALIDATE)
    names.append("family_svc_ocsvm_decision")
    leaves.append(SVC_OPN_FAMILY_SVC_OCSVM_DECISION)
    names.append("family_svc_ocsvm_fit")
    leaves.append(SVC_OPN_FAMILY_SVC_OCSVM_FIT)
    names.append("family_svc_ocsvm_fit_predict")
    leaves.append(SVC_OPN_FAMILY_SVC_OCSVM_FIT_PREDICT)
    names.append("family_svc_ocsvm_predict")
    leaves.append(SVC_OPN_FAMILY_SVC_OCSVM_PREDICT)
    names.append("family_svc_ocsvm_score_samples")
    leaves.append(SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES)
