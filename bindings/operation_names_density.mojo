# SPDX-License-Identifier: Apache-2.0
# The `family_density_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-cluster, the form of bindings/operation_names_gp.mojo): each
# name has a leaf code, numbered by sub-owner so each prefix test of the
# owner's chain is one range test: DBSCAN 0-5, KernelDensity 6-13 (each: initialize, getstate, setstate first;
# the fits and queries last).
# The family door resolves a caller's str to the code with ONE dictionary
# lookup and never builds the name; a name this file does not hold keeps
# its String, for its unchanged error text only. A constant is
# `DENSITY_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime DENSITY_OPN_FAMILY_DENSITY_DBSCAN_INITIALIZE = 0
comptime DENSITY_OPN_FAMILY_DENSITY_DBSCAN_GETSTATE = 1
comptime DENSITY_OPN_FAMILY_DENSITY_DBSCAN_SETSTATE = 2
comptime DENSITY_OPN_FAMILY_DENSITY_DBSCAN_FIT = 3
comptime DENSITY_OPN_FAMILY_DENSITY_DBSCAN_FIT_PREDICT = 4
comptime DENSITY_OPN_FAMILY_DENSITY_DBSCAN_PREDICT = 5
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_INITIALIZE = 6
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_GETSTATE = 7
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_SETSTATE = 8
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_SAMPLE = 9
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_RELEASE = 10
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_FIT = 11
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_SCORE = 12
comptime DENSITY_OPN_FAMILY_DENSITY_KDE_SCORE_SAMPLES = 13

def operation_names_density(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_density_dbscan_fit")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_DBSCAN_FIT)
    names.append("family_density_dbscan_fit_predict")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_DBSCAN_FIT_PREDICT)
    names.append("family_density_dbscan_getstate")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_DBSCAN_GETSTATE)
    names.append("family_density_dbscan_initialize")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_DBSCAN_INITIALIZE)
    names.append("family_density_dbscan_predict")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_DBSCAN_PREDICT)
    names.append("family_density_dbscan_setstate")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_DBSCAN_SETSTATE)
    names.append("family_density_kde_fit")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_FIT)
    names.append("family_density_kde_getstate")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_GETSTATE)
    names.append("family_density_kde_initialize")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_INITIALIZE)
    names.append("family_density_kde_release")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_RELEASE)
    names.append("family_density_kde_sample")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_SAMPLE)
    names.append("family_density_kde_score")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_SCORE)
    names.append("family_density_kde_score_samples")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_SCORE_SAMPLES)
    names.append("family_density_kde_setstate")
    leaves.append(DENSITY_OPN_FAMILY_DENSITY_KDE_SETSTATE)
