# SPDX-License-Identifier: Apache-2.0
# The `family_kmeans_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-cluster, the form of bindings/operation_names_gp.mojo): each
# name has a leaf code, numbered by sub-owner so each prefix test of the
# owner's chain is one range test: the fits 0-2, the queries 3-4, the pickle state 5-6.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup and never builds the name; a name this file does not hold keeps
# its String, for its unchanged error text only. A constant is
# `KMEANS_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime KMEANS_OPN_FAMILY_KMEANS_FIT = 0
comptime KMEANS_OPN_FAMILY_KMEANS_FIT_PREDICT = 1
comptime KMEANS_OPN_FAMILY_KMEANS_FIT_TRANSFORM = 2
comptime KMEANS_OPN_FAMILY_KMEANS_PREDICT = 3
comptime KMEANS_OPN_FAMILY_KMEANS_TRANSFORM = 4
comptime KMEANS_OPN_FAMILY_KMEANS_GETSTATE = 5
comptime KMEANS_OPN_FAMILY_KMEANS_SETSTATE = 6

def operation_names_kmeans(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_kmeans_fit")
    leaves.append(KMEANS_OPN_FAMILY_KMEANS_FIT)
    names.append("family_kmeans_fit_predict")
    leaves.append(KMEANS_OPN_FAMILY_KMEANS_FIT_PREDICT)
    names.append("family_kmeans_fit_transform")
    leaves.append(KMEANS_OPN_FAMILY_KMEANS_FIT_TRANSFORM)
    names.append("family_kmeans_getstate")
    leaves.append(KMEANS_OPN_FAMILY_KMEANS_GETSTATE)
    names.append("family_kmeans_predict")
    leaves.append(KMEANS_OPN_FAMILY_KMEANS_PREDICT)
    names.append("family_kmeans_setstate")
    leaves.append(KMEANS_OPN_FAMILY_KMEANS_SETSTATE)
    names.append("family_kmeans_transform")
    leaves.append(KMEANS_OPN_FAMILY_KMEANS_TRANSFORM)
