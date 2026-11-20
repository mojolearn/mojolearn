# SPDX-License-Identifier: Apache-2.0
# The `family_mix_*` operation names of the classical owner's code table
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


def operation_names_mix(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_mix_bgmm_fit")
    names.append("family_mix_bgmm_fit_predict")
    names.append("family_mix_bgmm_predict")
    names.append("family_mix_bgmm_predict_proba")
    names.append("family_mix_bgmm_score")
    names.append("family_mix_bgmm_score_samples")
    names.append("family_mix_gmm_aic")
    names.append("family_mix_gmm_bic")
    names.append("family_mix_gmm_fit")
    names.append("family_mix_gmm_fit_predict")
    names.append("family_mix_gmm_predict")
    names.append("family_mix_gmm_predict_proba")
    names.append("family_mix_gmm_public_aic")
    names.append("family_mix_gmm_public_bic")
    names.append("family_mix_gmm_public_binding")
    names.append("family_mix_gmm_public_fit")
    names.append("family_mix_gmm_public_fit_predict")
    names.append("family_mix_gmm_public_initialize")
    names.append("family_mix_gmm_public_predict")
    names.append("family_mix_gmm_public_predict_proba")
    names.append("family_mix_gmm_public_sample")
    names.append("family_mix_gmm_public_score")
    names.append("family_mix_gmm_public_score_samples")
    names.append("family_mix_gmm_score_samples")
