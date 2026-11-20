# SPDX-License-Identifier: Apache-2.0
# The `family_km_*` operation names of the classical owner's code table
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


def operation_names_km(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_km_getstate")
    names.append("family_km_krr_fit")
    names.append("family_km_krr_predict")
    names.append("family_km_krr_score")
    names.append("family_km_nys_fit")
    names.append("family_km_nys_fit_transform")
    names.append("family_km_nys_transform")
    names.append("family_km_par_apply")
    names.append("family_km_par_fit")
    names.append("family_km_rbf_fit")
    names.append("family_km_rbf_fit_transform")
    names.append("family_km_rbf_transform")
    names.append("family_km_setstate")
