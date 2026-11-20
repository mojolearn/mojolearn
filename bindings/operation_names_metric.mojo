# SPDX-License-Identifier: Apache-2.0
# The `family_metric_*` operation names of the classical owner's code table
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


def operation_names_metric(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_metric_d2_absolute")
    names.append("family_metric_d2_pinball")
    names.append("family_metric_d2_tweedie")
    names.append("family_metric_explained_variance")
    names.append("family_metric_gamma")
    names.append("family_metric_mape")
    names.append("family_metric_max_error")
    names.append("family_metric_medae")
    names.append("family_metric_msle")
    names.append("family_metric_pinball")
    names.append("family_metric_poisson")
    names.append("family_metric_r2_options")
    names.append("family_metric_regression_error")
    names.append("family_metric_regression_report")
    names.append("family_metric_rmsle")
    names.append("family_metric_tweedie")
