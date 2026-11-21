# SPDX-License-Identifier: Apache-2.0
"""RUN of the one-class SVM door (bindings/svc_ocsvm_native_door.mojo): the
typed fit and query over the x_neighbors native driver. Plain words and
addresses in, native stores out; no Python object. Each function is called
with the GIL released. The driver calls, their operands and their launch
order are the ones the door made before the split (core/ocsvm_plan.mojo)."""
from core.neighbors_native_driver import XnDriver
from core.ocsvm_resident_api import OcResidentDriver
from core.ocsvm_plan import (OcMatrix, OcWeights, OcFitState, OcQueryState, oc_f32_address, oc_i32_address,
    oc_take_rows, oc_take_cols, oc_kernel, oc_variance, oc_kernel_product, oc_fused_labels, oc_product,
    oc_decision, oc_sign)
from core.admitted_convert import admitted_convert_vector

# How the fit's dual problem is solved.
comptime OC_SOLVE_GRAM = 0        # on a Gram matrix (precomputed, or built here by one kernel call)
comptime OC_SOLVE_RESIDENT = 1    # the module's resident solve over X (no Gram matrix)
comptime OC_SOLVE_DESCRIPTOR = 2  # the descriptor's resident solve over X (no Gram matrix)

# How a query's scores are produced.
comptime OC_QUERY_FUSED_LABELS = 0     # kernel, product, offset and sign in one launch (labels out)
comptime OC_QUERY_KERNEL_PRODUCT = 1   # kernel and product in one launch (scores out)
comptime OC_QUERY_GRAM_PRECOMPUTED = 2 # the support columns of a precomputed Gram matrix, then the product
comptime OC_QUERY_GRAM_KERNEL = 3      # the Gram matrix by one kernel call, then the product


def _f32_scalar(value: Float64) -> Float64:
    """A Python double rounded once to float32."""
    return Float64(Float32(value))


struct OcFitRequest(Copyable, Movable):
    """A fit, as UNPACK read it. `x` addresses rows x features float32 words
    (the square Gram matrix when `precomputed`); `weights` addresses the
    per-row weights when `weighted`. `gamma` is the given value (or
    1 / features); with `gamma_scale` it is read from the variance of X
    here. `kernel` is the kernel code; `cap` the iteration cap."""
    var x: Int
    var rows: Int
    var features: Int
    var precomputed: Bool
    var weighted: Bool
    var weights: Int
    var nu: Float64
    var degree: Int
    var coef0: Float64
    var gamma_scale: Bool
    var gamma: Float64
    var cap: Int
    var tol: Float64
    var kernel: Int
    var solve: Int
    var resident_fast: Bool

    def __init__(out self):
        self.x = 0
        self.rows = 0
        self.features = 0
        self.precomputed = False
        self.weighted = False
        self.weights = 0
        self.nu = 0.0
        self.degree = 0
        self.coef0 = 0.0
        self.gamma_scale = False
        self.gamma = 0.0
        self.cap = 0
        self.tol = 0.0
        self.kernel = 0
        self.solve = OC_SOLVE_GRAM
        self.resident_fast = False


struct OcFitRun(Movable):
    """A finished fit for PACK: the kept rows and their weights, the solved
    state with its compacted support, the float32 gamma used (0 for a
    precomputed kernel) and rho."""
    var weights: OcWeights
    var state: OcFitState
    var gamma: Float64
    var rho: Float64

    def __init__(out self):
        self.weights = OcWeights()
        self.state = OcFitState(0)
        self.gamma = 0.0
        self.rho = 0.0

    def __init__(out self, var weights: OcWeights, var state: OcFitState, gamma: Float64, rho: Float64):
        self.weights = weights^
        self.state = state^
        self.gamma = gamma
        self.rho = rho


def oc_fit_run(driver: XnDriver, resident: Optional[OcResidentDriver], r: OcFitRequest) raises -> OcFitRun:
    """The fit: the rows with a positive weight, gamma, the Gram matrix when
    the solve needs one, the alpha initialisation, the solve, the support."""
    var n = r.rows
    var d = r.features
    var weights = OcWeights()
    if r.weighted:
        weights.compact(driver, r.weights, n)
    else:
        weights.unit(n)
    var m = weights.count
    var q = OcMatrix(0, 0)
    var xw = OcMatrix(n, d, r.x)
    var gamma = Float64(0)
    var has_gram = False
    if r.precomputed:
        if not weights.selected:
            q = OcMatrix(n, d, r.x)
        else:
            var selected_rows = oc_take_rows(driver, r.x, n, d, oc_i32_address(weights.rows), m)
            q = oc_take_cols(driver, selected_rows.address(), m, d, oc_i32_address(weights.rows), m)
            _ = selected_rows
        has_gram = True
    else:
        gamma = r.gamma
        if r.gamma_scale:
            var value = oc_variance(driver, r.x, n * d)
            gamma = 1.0 / (Float64(d) * value) if value != 0 else 1.0
        gamma = _f32_scalar(gamma)
        if weights.selected:
            xw = oc_take_rows(driver, r.x, n, d, oc_i32_address(weights.rows), m)
        if r.solve == OC_SOLVE_GRAM:
            q = oc_kernel(driver, xw.address(), xw.address(), m, m, d, r.kernel, r.degree, gamma, r.coef0)
            has_gram = True
    var state = OcFitState(m)
    state.initialize(driver, oc_f32_address(weights.values), r.nu)
    state.prepare_metadata()
    if has_gram:
        state.solve_gram(driver, q.address(), oc_f32_address(weights.values), r.cap, r.tol)
    elif r.solve == OC_SOLVE_RESIDENT:
        state.solve_resident(resident.value(), xw.address(), oc_f32_address(weights.values),
            d, r.kernel, r.degree, r.cap, r.resident_fast, gamma, r.coef0, r.tol)
    else:
        state.solve_descriptor(driver, xw.address(), oc_f32_address(weights.values),
            d, r.kernel, r.degree, r.cap, gamma, r.coef0, r.tol)
    var rho = Float64(state.info[0])
    state.compact_support(driver)
    # The Gram matrix and the kept rows are read by the solve above.
    _ = xw
    _ = q
    return OcFitRun(weights^, state^, gamma, rho)


struct OcQueryRequest(Copyable, Movable):
    """A query, as UNPACK read it: `query` addresses rows x features float32
    words, `references` the support vectors (reference_rows x features),
    `coefficients` the supports dual coefficients; `support` / `support_count`
    the int32 support columns of a precomputed kernel. `offset` is read by
    the fused labels, the decision and the labels."""
    var path: Int
    var query: Int
    var references: Int
    var coefficients: Int
    var rows: Int
    var supports: Int
    var features: Int
    var reference_rows: Int
    var kernel: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64
    var offset: Float64
    var support: Int
    var support_count: Int

    def __init__(out self):
        self.path = OC_QUERY_GRAM_KERNEL
        self.query = 0
        self.references = 0
        self.coefficients = 0
        self.rows = 0
        self.supports = 0
        self.features = 0
        self.reference_rows = 0
        self.kernel = 0
        self.degree = 0
        self.gamma = 0.0
        self.coef0 = 0.0
        self.offset = 0.0
        self.support = 0
        self.support_count = 0


def oc_query_run(driver: XnDriver, r: OcQueryRequest, dst: Int) raises:
    """The query's scores into `dst` (rows float32 words), or its labels
    (rows int32 words) on the fused path."""
    if r.path == OC_QUERY_FUSED_LABELS:
        oc_fused_labels(driver, r.query, r.references, r.coefficients, dst, r.rows, r.supports, r.features,
            r.kernel, r.degree, r.gamma, r.coef0, r.offset)
        return
    if r.path == OC_QUERY_KERNEL_PRODUCT:
        oc_kernel_product(driver, r.query, r.references, r.coefficients, dst, r.rows, r.supports, r.features,
            r.kernel, r.degree, r.gamma, r.coef0)
        return
    var gram = OcMatrix(0, 0)
    if r.path == OC_QUERY_GRAM_PRECOMPUTED:
        gram = oc_take_cols(driver, r.query, r.rows, r.features, r.support, r.support_count)
    else:
        gram = oc_kernel(driver, r.query, r.references, r.rows, r.reference_rows, r.features,
            r.kernel, r.degree, r.gamma, r.coef0)
    oc_product(driver, gram.address(), r.coefficients, dst, gram.rows, gram.columns)
    _ = gram


def oc_decision_run(driver: XnDriver, r: OcQueryRequest, dst: Int) raises:
    """decision_function: the scores in private storage, then scores - offset
    into `dst` (rows float32 words). The request's path is never the fused
    labels (UNPACK asks for them on predict only)."""
    var scores = OcQueryState(r.rows, False, True)
    oc_query_run(driver, r, scores.address())
    oc_decision(driver, scores.address(), dst, r.rows, r.offset)
    _ = scores


def oc_labels_run(driver: XnDriver, r: OcQueryRequest, dst: Int) raises:
    """predict: the fused labels, or the scores, the decision and its sign;
    then the int32 labels widened into `dst` (rows int64 words)."""
    var n = r.rows
    var fused = r.path == OC_QUERY_FUSED_LABELS
    var first = OcQueryState(n, fused, True)
    oc_query_run(driver, r, first.address())
    var decision = OcQueryState(0, False, False)
    var labels = OcQueryState(0, True, False)
    var label_address = first.address()
    if not fused:
        decision = OcQueryState(n, False, True)
        oc_decision(driver, first.address(), decision.address(), n, r.offset)
        labels = OcQueryState(n, True, True)
        if n > 0:
            oc_sign(driver, decision.address(), labels.address(), n)
        label_address = labels.address()
    if n > 0:
        _ = admitted_convert_vector(label_address, "<i4", dst, "<i8", n)
    _ = first
    _ = decision
    _ = labels
