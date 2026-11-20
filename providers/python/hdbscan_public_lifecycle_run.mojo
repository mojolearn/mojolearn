# SPDX-License-Identifier: Apache-2.0
# RUN part of providers/python/hdbscan_public_lifecycle_pyapi.mojo: typed
# Mojo only (no Python imports).
from bindings.operation_names_hdbscan import (
    HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT, HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR)


def _query_slot(operation: Int) -> String:
    """The per-model record slot of a query (lane bw2-cluster: a static
    literal per leaf code; it was a String concatenated per call)."""
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT:
        return "cluster_hdbscan_approximate_predict_v2"
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR:
        return "cluster_hdbscan_membership_vector_v2"
    return "cluster_hdbscan_all_points_membership_vectors_v2"


def _query_method(operation: Int) -> String:
    """The public method of a query, for its refusal texts (a static
    literal; it was a slice of the operation name per call)."""
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT:
        return "approximate_predict"
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR:
        return "membership_vector"
    return "all_points_membership_vectors"
