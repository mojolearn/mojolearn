# SPDX-License-Identifier: Apache-2.0
"""Typed rules of the public parallel boundary (RUN): every refusal of
`parallel_public_operation` and of the pooled forest predictor that is a
comparison on a word or a number, and the fixed codes of a plan's recipe and
route. `bindings/parallel_public_native_door.mojo` unpacks the Python values
into these arguments, calls the member's compiled entry and packs the
result; no Python object reaches this file. A few words per driver call:
nothing to split across cores."""

comptime PUBLIC_PREDICTOR_MAX_DEVICES = 64

# A plan's `check` word (python/mojolearn/_native_parallel_public.py plans).
comptime PUBLIC_CHECK_NONE = 0
comptime PUBLIC_CHECK_GRAM = 1
comptime PUBLIC_CHECK_MATRIX = 2
comptime PUBLIC_CHECK_KMEANS = 3
comptime PUBLIC_CHECK_GMM_PREDICT = 4
comptime PUBLIC_CHECK_POSTERIOR = 5
comptime PUBLIC_CHECK_IFOREST = 6
comptime PUBLIC_CHECK_UMAP_TRANSFORM = 7
comptime PUBLIC_CHECK_RESAMPLE = 8

# A recipe item's kind.
comptime PUBLIC_ITEM_OTHER = 0
comptime PUBLIC_ITEM_ARG = 1
comptime PUBLIC_ITEM_CONSTANT = 2
comptime PUBLIC_ITEM_MODEL = 3
comptime PUBLIC_ITEM_BINDING = 4

# Which mode word a plan's policy or a binding item's route selects.
comptime PUBLIC_MODE_DEFAULT = 0    # the process default
comptime PUBLIC_MODE_EFFECTIVE = 1  # the model's word, else the process default
comptime PUBLIC_MODE_MODEL = 2      # the model's word as stored
comptime PUBLIC_MODE_LITERAL = 3    # the item's own word (a binding item only)

# A plan's route.
comptime PUBLIC_ROUTE_DIRECT = 0
comptime PUBLIC_ROUTE_SHARD = 1
comptime PUBLIC_ROUTE_COOPERATIVE = 2


def public_check_code(check: String) -> Int:
    if check == "gram":
        return PUBLIC_CHECK_GRAM
    if check == "matrix":
        return PUBLIC_CHECK_MATRIX
    if check == "kmeans":
        return PUBLIC_CHECK_KMEANS
    if check == "gmm_predict":
        return PUBLIC_CHECK_GMM_PREDICT
    if check == "posterior":
        return PUBLIC_CHECK_POSTERIOR
    if check == "iforest":
        return PUBLIC_CHECK_IFOREST
    if check == "umap_transform":
        return PUBLIC_CHECK_UMAP_TRANSFORM
    if check == "resample":
        return PUBLIC_CHECK_RESAMPLE
    return PUBLIC_CHECK_NONE


def public_check_admits_x(check: Int) -> Bool:
    """The checks whose first argument is admitted as a 2-D X."""
    return (check == PUBLIC_CHECK_GRAM or check == PUBLIC_CHECK_MATRIX or check == PUBLIC_CHECK_KMEANS
            or check == PUBLIC_CHECK_GMM_PREDICT)


def public_item_code(kind: String) -> Int:
    if kind == "arg":
        return PUBLIC_ITEM_ARG
    if kind == "constant":
        return PUBLIC_ITEM_CONSTANT
    if kind == "model":
        return PUBLIC_ITEM_MODEL
    if kind == "binding":
        return PUBLIC_ITEM_BINDING
    return PUBLIC_ITEM_OTHER


def public_policy_code(policy: String) -> Int:
    """A plan's mode policy: "default", "effective", else the model's word."""
    if policy == "default":
        return PUBLIC_MODE_DEFAULT
    if policy == "effective":
        return PUBLIC_MODE_EFFECTIVE
    return PUBLIC_MODE_MODEL


def public_binding_mode_code(route: String) -> Int:
    """A binding item's mode route: "model" (the effective word), "default",
    else the item's own word."""
    if route == "model":
        return PUBLIC_MODE_EFFECTIVE
    if route == "default":
        return PUBLIC_MODE_DEFAULT
    return PUBLIC_MODE_LITERAL


def public_route_code(route: String) -> Int:
    if route == "direct":
        return PUBLIC_ROUTE_DIRECT
    if route == "shard":
        return PUBLIC_ROUTE_SHARD
    return PUBLIC_ROUTE_COOPERATIVE


def public_identical_admit(present: Bool, exact: Bool, word: String, refusal: String) raises:
    """A selected mode word must be absent or exactly `identical`. `present`:
    a value was selected; `exact`: it is an exact str (`word` is its text)."""
    if present and (not exact or word != "identical"):
        raise Error("ValueError: " + refusal)


def public_gram_columns_admit(columns: Int) raises:
    if columns < 1:
        raise Error("ValueError: parallel Gram requires at least one feature")


def public_gram_pca_admit(dense_pca: Bool, rows: Int, columns: Int) raises:
    """`dense_pca`: the class is PCA and its solver is one of the dense ones."""
    if dense_pca and rows < columns:
        raise Error("ValueError: parallel full PCA currently requires the tall TSQR route")


def public_gram_weight_admit(weighted: Bool, decomposition: Bool) raises:
    if weighted and decomposition:
        raise Error("TypeError: fit() got an unexpected keyword argument 'sample_weight'")


def public_kmeans_metric_admit(exact: Bool, metric: String) raises:
    if not exact or metric != "euclidean":
        raise Error("ValueError: parallel KMeans requires metric='euclidean'")


def public_kmeans_oversampling_admit(factor: Float64) raises:
    if factor != 2.0:
        raise Error("ValueError: parallel KMeans requires the default oversampling_factor")


def public_module_refusal_is_optional(error: String) -> Bool:
    """`plan_bindings`' optional-module refusals: an absent, incomplete or
    unselectable module is skipped; any other stored refusal (a readable mode
    mismatch, an invalid native export) stops the operation."""
    return error.startswith("ImportError:") or error.startswith("AttributeError:") or error.startswith("ValueError:")


def public_predictor_route_admit(mode: String, engine: String) raises:
    if mode != "identical":
        raise Error("ValueError: ParallelForestPredictor requires IDENTICAL numeric mode")
    if engine != "parallel_groves":
        raise Error("ValueError: ParallelForestPredictor requires inference_engine='parallel_groves'")


def public_predictor_devices_admit(count: Int) raises:
    if count > PUBLIC_PREDICTOR_MAX_DEVICES:
        raise Error("ValueError: ParallelForestPredictor supports at most 64 devices")


struct ForestPredictorState(Movable):
    """The typed state of one pooled forest predictor: its lifecycle word
    and whether the copied forest is a classifier."""
    var closed: Bool
    var classifier: Bool

    def __init__(out self, classifier: Bool):
        self.closed = False
        self.classifier = classifier


def public_predictor_open_admit(closed: Bool) raises:
    if closed:
        raise Error("RuntimeError: pooled forest predictor is closed")


def public_predictor_proba_admit(proba: Bool, classifier: Bool) raises:
    if proba and not classifier:
        raise Error("ValueError: predict_proba requires a classifier")
