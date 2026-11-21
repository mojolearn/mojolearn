# SPDX-License-Identifier: Apache-2.0
# The `family_mix_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-svm, 2026-11-20; the form of bindings/operation_names_gp.mojo):
# each name has a leaf code, numbered by sub-owner (GaussianMixture, BayesianGaussianMixture, GaussianMixture public) so a prefix test
# of the owner's chain is one range test and each owner switches on the code.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup (bindings/native_operation_names_unpack.mojo) and never builds the
# name; an unknown `family_mix_*` name is not in the table and keeps its
# String, for its unchanged error text only (`mixture_unknown_operation`).
# A constant is `MIX_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name. Add every name a caller
# can pass: a name missing here is refused as unknown.

comptime MIX_OPN_FAMILY_MIX_GMM_FIT = 0
comptime MIX_OPN_FAMILY_MIX_GMM_FIT_PREDICT = 1
comptime MIX_OPN_FAMILY_MIX_GMM_SCORE_SAMPLES = 2
comptime MIX_OPN_FAMILY_MIX_GMM_PREDICT_PROBA = 3
comptime MIX_OPN_FAMILY_MIX_GMM_PREDICT = 4
comptime MIX_OPN_FAMILY_MIX_GMM_BIC = 5
comptime MIX_OPN_FAMILY_MIX_GMM_AIC = 6
comptime MIX_OPN_FAMILY_MIX_BGMM_FIT = 7
comptime MIX_OPN_FAMILY_MIX_BGMM_FIT_PREDICT = 8
comptime MIX_OPN_FAMILY_MIX_BGMM_PREDICT = 9
comptime MIX_OPN_FAMILY_MIX_BGMM_PREDICT_PROBA = 10
comptime MIX_OPN_FAMILY_MIX_BGMM_SCORE_SAMPLES = 11
comptime MIX_OPN_FAMILY_MIX_BGMM_SCORE = 12
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT = 13
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT_PREDICT = 14
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE_SAMPLES = 15
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT_PROBA = 16
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT = 17
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BIC = 18
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_AIC = 19
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE = 20
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_INITIALIZE = 21
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BINDING = 22
comptime MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SAMPLE = 23
# `vendor_used()` of GaussianMixture and BayesianGaussianMixture, which take
# `device=` (lane dw-neighbors-cluster): answered from the estimator's route,
# before the public range test of `mixture_operation`.
comptime MIX_OPN_FAMILY_MIX_VENDOR_USED = 24


def operation_names_mix(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_mix_gmm_fit")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_FIT)
    names.append("family_mix_gmm_fit_predict")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_FIT_PREDICT)
    names.append("family_mix_gmm_score_samples")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_SCORE_SAMPLES)
    names.append("family_mix_gmm_predict_proba")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PREDICT_PROBA)
    names.append("family_mix_gmm_predict")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PREDICT)
    names.append("family_mix_gmm_bic")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_BIC)
    names.append("family_mix_gmm_aic")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_AIC)
    names.append("family_mix_bgmm_fit")
    leaves.append(MIX_OPN_FAMILY_MIX_BGMM_FIT)
    names.append("family_mix_bgmm_fit_predict")
    leaves.append(MIX_OPN_FAMILY_MIX_BGMM_FIT_PREDICT)
    names.append("family_mix_bgmm_predict")
    leaves.append(MIX_OPN_FAMILY_MIX_BGMM_PREDICT)
    names.append("family_mix_bgmm_predict_proba")
    leaves.append(MIX_OPN_FAMILY_MIX_BGMM_PREDICT_PROBA)
    names.append("family_mix_bgmm_score_samples")
    leaves.append(MIX_OPN_FAMILY_MIX_BGMM_SCORE_SAMPLES)
    names.append("family_mix_bgmm_score")
    leaves.append(MIX_OPN_FAMILY_MIX_BGMM_SCORE)
    names.append("family_mix_gmm_public_fit")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT)
    names.append("family_mix_gmm_public_fit_predict")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT_PREDICT)
    names.append("family_mix_gmm_public_score_samples")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE_SAMPLES)
    names.append("family_mix_gmm_public_predict_proba")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT_PROBA)
    names.append("family_mix_gmm_public_predict")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT)
    names.append("family_mix_gmm_public_bic")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BIC)
    names.append("family_mix_gmm_public_aic")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_AIC)
    names.append("family_mix_gmm_public_score")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE)
    names.append("family_mix_gmm_public_initialize")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_INITIALIZE)
    names.append("family_mix_gmm_public_binding")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BINDING)
    names.append("family_mix_gmm_public_sample")
    leaves.append(MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SAMPLE)
    names.append("family_mix_vendor_used")
    leaves.append(MIX_OPN_FAMILY_MIX_VENDOR_USED)
