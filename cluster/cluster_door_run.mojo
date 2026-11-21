# SPDX-License-Identifier: Apache-2.0
# RUN part of the x_cluster query door (MiniBatchKMeans / BisectingKMeans
# predict, transform, score; MeanShift predict; the bisection tree descent;
# AffinityPropagation predict): typed Mojo only (no Python imports). UNPACK is
# bindings/cluster_public_native_unpack.mojo, PACK and the operator are
# bindings/cluster_public_native_pack.mojo. The operation classification, the
# refusals that compare numbers and the request words are here; UNPACK calls a
# refusal at the place the compare stood, so the refusal order is unchanged.
# The numerical work is the typed driver call of core/kmeans_variants_api.mojo
# (KMeansVariantsDriver).
from core.kmeans_variants_api import VariantsSpan, VariantsQueryRequest, BisectPredictRequest
from bindings.operation_names_xc import (
    XC_OPN_FAMILY_XC_VARIANTS_PREDICT, XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM, XC_OPN_FAMILY_XC_VARIANTS_SCORE,
)

# The query kinds of the GPU binding's `x_cluster_resident_query`
# (x_cluster/resident.mojo XC_RES_*).
comptime XC_RES_PREDICT = 0
comptime XC_RES_SCORE = 1
comptime XC_RES_TRANSFORM = 2


# ----------------------------------------------------------- classification
def cluster_query_variants(operation: Int) -> Bool:
    """`operation` is one of the three centers queries of `_CentersMixin`."""
    return operation >= XC_OPN_FAMILY_XC_VARIANTS_PREDICT and operation <= XC_OPN_FAMILY_XC_VARIANTS_SCORE


def cluster_query_kind(operation: Int) -> Int:
    """The query kind of a centers query's leaf code."""
    if operation == XC_OPN_FAMILY_XC_VARIANTS_PREDICT:
        return XC_RES_PREDICT
    if operation == XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM:
        return XC_RES_TRANSFORM
    return XC_RES_SCORE


# ---------------------------------------------------------------- refusals
def cluster_features_text(d: Int, owner: String, fitted: Int) -> String:
    """`_XCluster._input_like_fit`: the feature-count refusal."""
    return "mojolearn: X has " + String(d) + " features, " + owner + " was fitted with " + String(fitted)


def cluster_node_words(byte_count: Int) raises -> Int:
    """The bisection tree's int32 word count from its C-order byte length."""
    if byte_count % 4 != 0:
        raise Error("ValueError: bytes length not a multiple of item size")
    return byte_count // 4


def cluster_unknown_operation(operation: String) raises:
    """A `family_xc_*` name the operation table does not hold (the shells
    pass table names only): the text the name chain raised."""
    raise Error("RuntimeError: unknown x_cluster operation " + operation)


def cluster_unknown_code(operation: Int) raises:
    raise Error("RuntimeError: unknown x_cluster operation code " + String(operation))


# ------------------------------------------------------------ request words
def cluster_query_request(source: VariantsSpan, centers: VariantsSpan, n: Int, k: Int, d: Int,
                          kind: Int) -> VariantsQueryRequest:
    """The centers query's typed request (score asks for the inertia)."""
    return VariantsQueryRequest(source, centers, n, k, d, kind == XC_RES_SCORE)


def cluster_bisect_request(source: VariantsSpan, centers: VariantsSpan, nodes_address: Int, words: Int,
                           n: Int, d: Int) -> BisectPredictRequest:
    """The tree descent's typed request; an empty tree has no address."""
    return BisectPredictRequest(source, centers, VariantsSpan(nodes_address if words else 0, words), n, d)


def cluster_resident_bytes(clusters: Int, features: Int) -> Int:
    """The float32 bytes a resident centers copy holds on the device."""
    return clusters * features * 4
