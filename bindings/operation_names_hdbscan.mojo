# SPDX-License-Identifier: Apache-2.0
# The `family_hdbscan_public_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# CODED (lane bw2-cluster, the form of bindings/operation_names_gp.mojo): each
# name has a leaf code, numbered by sub-owner so each prefix test of the
# owner's chain is one range test: initialize 0, binding 1, the fits 2-3, the queries 4-6.
# The family door resolves a caller's str to the code with ONE dictionary
# lookup and never builds the name; a name this file does not hold keeps
# its String, for its unchanged error text only. A constant is
# `HDBSCAN_OPN_<operation name, upper case>`, the form
# tools/boundary_keys_plain.py reads back as the name.

comptime HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_INITIALIZE = 0
comptime HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_BINDING = 1
comptime HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_FIT = 2
comptime HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_FIT_PREDICT = 3
comptime HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT = 4
comptime HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR = 5
comptime HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS = 6

def operation_names_hdbscan(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_hdbscan_public_all_points_membership_vectors")
    leaves.append(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS)
    names.append("family_hdbscan_public_approximate_predict")
    leaves.append(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT)
    names.append("family_hdbscan_public_binding")
    leaves.append(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_BINDING)
    names.append("family_hdbscan_public_fit")
    leaves.append(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_FIT)
    names.append("family_hdbscan_public_fit_predict")
    leaves.append(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_FIT_PREDICT)
    names.append("family_hdbscan_public_initialize")
    leaves.append(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_INITIALIZE)
    names.append("family_hdbscan_public_membership_vector")
    leaves.append(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR)
