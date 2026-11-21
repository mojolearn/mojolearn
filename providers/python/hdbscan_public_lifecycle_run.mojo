# SPDX-License-Identifier: Apache-2.0
# RUN part of HDBSCAN's public lifecycle: typed Mojo only (no Python imports).
# UNPACK is providers/python/hdbscan_public_lifecycle_pyapi_unpack.mojo and
# bindings/hdbscan_fit_boundary_unpack.mojo; PACK is
# providers/python/hdbscan_public_lifecycle_pyapi_pack.mojo and
# bindings/hdbscan_fit_boundary_pack.mojo. Every refusal that is a comparison
# on a number is here, on Mojo values, with the public wording; the numerical
# work is the typed driver call of core/hdbscan_query_api.mojo and
# core/hdbscan_fit_api.mojo.
from std.math import isinf
from core.kmeans_variants_api import VariantsSpan
from core.hdbscan_fit_api import HdbscanFitOptions, HdbscanFitRequest
from core.hdbscan_query_api import (
    HdbscanQueryDimensions, HdbscanQueryBuffers, HdbscanQueryRequest,
    HDBSCAN_QUERY_APPROXIMATE, HDBSCAN_QUERY_MEMBERSHIP, HDBSCAN_QUERY_ALL_POINTS,
)
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


def hdbscan_query_kind(operation: Int) -> Int:
    """The provider's query code of a public query's leaf code."""
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT:
        return HDBSCAN_QUERY_APPROXIMATE
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR:
        return HDBSCAN_QUERY_MEMBERSHIP
    return HDBSCAN_QUERY_ALL_POINTS


def hdbscan_fitted_dimensions(selected: Int, columns: Int) raises:
    if selected < 0 or columns < 1:
        raise Error("ValueError: HDBSCAN fitted dimensions are invalid")


def hdbscan_batch_size(method: String, batch: Int) raises:
    if batch <= 0:
        raise Error("mojolearn " + method + ": batch_size must be > 0, got " + String(batch))


def hdbscan_training_shape(ndim: Int, width: Int, columns: Int) raises:
    """`width` is read only for a matrix (the caller passes -1 otherwise)."""
    if ndim != 2 or width != columns:
        raise Error("ValueError: HDBSCAN training storage differs from its fitted shape")


def hdbscan_training_rows(rows: Int) raises:
    if rows < 2:
        raise Error("ValueError: HDBSCAN training storage requires at least two rows")


def hdbscan_query_width(method: String, d: Int, columns: Int) raises:
    if d != columns:
        raise Error("mojolearn " + method + ": points_to_predict has " + String(d) + " features, the clusterer was fit on " + String(columns))


def hdbscan_query_neighbors(neighbors: Int, rows: Int) raises:
    if neighbors < 1 or neighbors > rows:
        raise Error("ValueError: HDBSCAN min_samples must fit its training row count")


def hdbscan_query_request(dimensions: HdbscanQueryDimensions, buffers: HdbscanQueryBuffers,
                          query_rows: Int, batch: Int) -> HdbscanQueryRequest:
    """One whole-input request (row_start 0); the provider batches inside."""
    return HdbscanQueryRequest(dimensions.copy(), buffers.copy(), query_rows, 0, batch)


def hdbscan_fit_rows(n: Int, d: Int) raises:
    if n < 2:
        # scikit-learn's check_array wording (ensure_min_samples=2); the
        # mutual-reachability tree needs two points
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required.")


def hdbscan_epsilon_refused(epsilon: Float64) -> Bool:
    """cluster_selection_epsilon outside scikit-learn's range (finite, >= 0).
    The refusal text quotes the Python value, so UNPACK raises it."""
    return not (epsilon >= 0.0) or (isinf(epsilon) and epsilon > 0)


def hdbscan_fit_request(source: VariantsSpan, labels: VariantsSpan, core_distances: VariantsSpan,
                        probabilities: VariantsSpan, options: HdbscanFitOptions) -> HdbscanFitRequest:
    return HdbscanFitRequest(source, labels, core_distances, probabilities, options.copy())
