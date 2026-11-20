# SPDX-License-Identifier: Apache-2.0
# The `family_metric_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-metrics, 2026-11-20): each name has a leaf code,
# `metrics_reg_operation` (bindings/metrics_reg_door.mojo) switches on the code;
# the codes follow its chain.
# The family door resolves a caller's str (or a shell's table code) to the leaf
# with one lookup and never builds the name; a `family_metric_*` name the table does
# not hold keeps its String for its unchanged error text only. The appends keep
# their order (the table's indices do not move); the leaves are not in append
# order. A constant is `METRIC_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime METRIC_OPN_FAMILY_METRIC_REGRESSION_REPORT = 0
comptime METRIC_OPN_FAMILY_METRIC_REGRESSION_ERROR = 1
comptime METRIC_OPN_FAMILY_METRIC_R2_OPTIONS = 2
comptime METRIC_OPN_FAMILY_METRIC_MSLE = 3
comptime METRIC_OPN_FAMILY_METRIC_RMSLE = 4
comptime METRIC_OPN_FAMILY_METRIC_MAPE = 5
comptime METRIC_OPN_FAMILY_METRIC_PINBALL = 6
comptime METRIC_OPN_FAMILY_METRIC_MEDAE = 7
comptime METRIC_OPN_FAMILY_METRIC_MAX_ERROR = 8
comptime METRIC_OPN_FAMILY_METRIC_EXPLAINED_VARIANCE = 9
comptime METRIC_OPN_FAMILY_METRIC_TWEEDIE = 10
comptime METRIC_OPN_FAMILY_METRIC_POISSON = 11
comptime METRIC_OPN_FAMILY_METRIC_GAMMA = 12
comptime METRIC_OPN_FAMILY_METRIC_D2_TWEEDIE = 13
comptime METRIC_OPN_FAMILY_METRIC_D2_PINBALL = 14
comptime METRIC_OPN_FAMILY_METRIC_D2_ABSOLUTE = 15


def operation_names_metric(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_metric_d2_absolute")
    leaves.append(METRIC_OPN_FAMILY_METRIC_D2_ABSOLUTE)
    names.append("family_metric_d2_pinball")
    leaves.append(METRIC_OPN_FAMILY_METRIC_D2_PINBALL)
    names.append("family_metric_d2_tweedie")
    leaves.append(METRIC_OPN_FAMILY_METRIC_D2_TWEEDIE)
    names.append("family_metric_explained_variance")
    leaves.append(METRIC_OPN_FAMILY_METRIC_EXPLAINED_VARIANCE)
    names.append("family_metric_gamma")
    leaves.append(METRIC_OPN_FAMILY_METRIC_GAMMA)
    names.append("family_metric_mape")
    leaves.append(METRIC_OPN_FAMILY_METRIC_MAPE)
    names.append("family_metric_max_error")
    leaves.append(METRIC_OPN_FAMILY_METRIC_MAX_ERROR)
    names.append("family_metric_medae")
    leaves.append(METRIC_OPN_FAMILY_METRIC_MEDAE)
    names.append("family_metric_msle")
    leaves.append(METRIC_OPN_FAMILY_METRIC_MSLE)
    names.append("family_metric_pinball")
    leaves.append(METRIC_OPN_FAMILY_METRIC_PINBALL)
    names.append("family_metric_poisson")
    leaves.append(METRIC_OPN_FAMILY_METRIC_POISSON)
    names.append("family_metric_r2_options")
    leaves.append(METRIC_OPN_FAMILY_METRIC_R2_OPTIONS)
    names.append("family_metric_regression_error")
    leaves.append(METRIC_OPN_FAMILY_METRIC_REGRESSION_ERROR)
    names.append("family_metric_regression_report")
    leaves.append(METRIC_OPN_FAMILY_METRIC_REGRESSION_REPORT)
    names.append("family_metric_rmsle")
    leaves.append(METRIC_OPN_FAMILY_METRIC_RMSLE)
    names.append("family_metric_tweedie")
    leaves.append(METRIC_OPN_FAMILY_METRIC_TWEEDIE)
