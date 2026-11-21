# SPDX-License-Identifier: Apache-2.0
"""NearestCentroid RUN: typed requests in, typed results out.

The boundary (bindings/neighbors_local_native_door.mojo) admits the public
arguments into these requests, releases the GIL for each function here and
publishes the results. Nothing in this module imports Python. The kernels are
the ones of core/nearest_centroid_plan.mojo, in the order the door ran them.
"""
from core.neighbors_native_driver import XnDriver
from core.nearest_centroid_plan import (
    NcLabelCodes, NcFitPlan, NcQueryPlan, nc_uniform, nc_normalize_priors, nc_narrow_priors,
    nc_binary_decision, nc_scratch_address,
)
from core.admitted_weights import admitted_fill
from neighbors.neighbors_local_run import nc_split_ops, nc_refuse_decision_metric, nc_fused, nc_proba_mode

comptime _F64 = MutPointer[Float64, MutUntrackedOrigin]

comptime NC_PRIORS_EMPIRICAL = 0
comptime NC_PRIORS_UNIFORM = 1
comptime NC_PRIORS_GIVEN = 2


struct NcFitRequest(Copyable, Movable):
    # Addresses of admitted storage the boundary keeps alive: float32 rows,
    # int32 label codes, and the four published arrays (float64 priors;
    # float32 standard deviations, centroids and deviations).
    var x: Int
    var labels: Int
    var prior: Int
    var stats: Int
    var centroids: Int
    var deviations: Int
    var rows: Int
    var features: Int
    var classes: Int
    var euclidean: Bool
    var priors: Int
    var shrink: Float64

    def __init__(out self, x: Int, labels: Int, rows: Int, features: Int, classes: Int,
                 euclidean: Bool, priors: Int, shrink: Float64):
        self.x = x
        self.labels = labels
        self.prior = 0
        self.stats = 0
        self.centroids = 0
        self.deviations = 0
        self.rows = rows
        self.features = features
        self.classes = classes
        self.euclidean = euclidean
        self.priors = priors
        self.shrink = shrink


struct NcFitOutcome(Copyable, Movable):
    # `status` is the deferred finite word of the statistics entry (0 = every
    # word finite); `constant` is the zero-variance answer. The boundary turns
    # either into the public refusal after it published the priors.
    var status: Int
    var constant: Bool

    def __init__(out self):
        self.status = 0
        self.constant = False


def nc_fit_count_run(driver: XnDriver, request: NcFitRequest, mut plan: NcFitPlan) raises:
    """The class counts of the label codes, into the plan."""
    plan.count(driver, request.labels)


def nc_fit_priors_run(request: NcFitRequest) raises:
    """Uniform priors are written; given priors (already cast to float64 at
    `request.prior`) are checked and normalized. Empirical priors are the
    label family's ratio step, which the boundary runs on the plan's counts."""
    if request.priors == NC_PRIORS_UNIFORM:
        admitted_fill[DType.float64](_F64(unsafe_from_address=request.prior), request.classes,
                                     1.0 / Float64(request.classes))
    elif request.priors == NC_PRIORS_GIVEN:
        nc_normalize_priors(request.prior, request.classes)


def nc_fit_run(driver: XnDriver, request: NcFitRequest, mut plan: NcFitPlan) raises -> NcFitOutcome:
    """Statistics, the variance test and the shrunken centroids. Stops at the
    first refusal and reports it as a value."""
    var outcome = NcFitOutcome()
    plan.split = nc_split_ops()
    plan.euclidean = request.euclidean
    outcome.status = plan.statistics(driver, request.x, request.labels, request.stats)
    if outcome.status != 0:
        return outcome
    var summary = plan.median_std(driver, request.stats)
    if summary[1]:
        outcome.constant = plan.constant(driver, request.x)
        if outcome.constant:
            return outcome
    plan.shrink(driver, request.x, request.stats, request.centroids, request.deviations, summary[0], request.shrink)
    return outcome


struct NcQueryRequest(Copyable, Movable):
    # Addresses of admitted storage the boundary keeps alive: the float32
    # query rows, the fitted float32 centroids and standard deviations and the
    # fitted float64 priors. `fused` and `wide` are the image's capabilities.
    var query: Int
    var centroids: Int
    var std: Int
    var prior: Int
    var rows: Int
    var features: Int
    var classes: Int
    var euclidean: Bool
    var fused: Bool
    var wide: Bool

    def __init__(out self, query: Int, centroids: Int, std: Int, prior: Int, rows: Int, features: Int,
                 classes: Int, euclidean: Bool, fused: Bool, wide: Bool):
        self.query = query
        self.centroids = centroids
        self.std = std
        self.prior = prior
        self.rows = rows
        self.features = features
        self.classes = classes
        self.euclidean = euclidean
        self.fused = fused
        self.wide = wide


def _nc_scored(request: NcQueryRequest, narrow: List[Float32]) raises -> NcQueryPlan:
    """The discriminant plan: the float64 priors narrowed into `narrow`,
    which the caller keeps alive through its kernels."""
    var address = nc_scratch_address(narrow)
    nc_narrow_priors(request.prior, address, request.classes)
    return NcQueryPlan(request.query, request.centroids, request.std, address,
                       request.rows, request.features, request.classes)


def nc_labels_run(driver: XnDriver, request: NcQueryRequest, mut codes: NcLabelCodes) raises -> Int:
    """Predicted class codes into `codes`. Returns the deferred finite word
    of the fused entry (0 on the unfused entries, which defer nothing)."""
    var fused = nc_fused(request.fused, request.rows)
    var dst = codes.address()
    var status = 0
    if nc_uniform(request.prior, request.classes):
        # Uniform priors: nearest centroid by distance; the entry reads no
        # standard deviation or prior, so the query address stands in.
        var plan = NcQueryPlan(request.query, request.centroids, request.query, request.query,
                               request.rows, request.features, request.classes)
        if fused:
            status = plan.fused(driver, "nc_labels", dst, 0 if request.euclidean else 1)
        else:
            plan.distance_labels(driver, dst, request.euclidean)
        return status
    if not request.euclidean:
        nc_refuse_decision_metric()
    var narrow = List[Float32](length=request.classes, fill=0)
    var plan = _nc_scored(request, narrow)
    if fused:
        status = plan.fused(driver, "nc_labels", dst, 2)
    else:
        plan.score_labels(driver, dst)
    _ = narrow^  # the narrowed priors outlive the kernels that read them
    return status


def nc_decision_run(driver: XnDriver, request: NcQueryRequest, dst: Int) raises -> Int:
    """Discriminant scores at `dst`: `rows` float32 words for two classes
    (the binary margin), else `rows * classes`. Returns the finite word; on a
    non-zero word the binary margin is not written."""
    var narrow = List[Float32](length=request.classes, fill=0)
    var plan = _nc_scored(request, narrow)
    var status = List[Int32](length=1, fill=0)
    var word = 0
    if request.classes == 2:
        var scores = List[Float32](length=request.rows * 2, fill=0)
        plan.decision(driver, nc_scratch_address(scores), Int(status.unsafe_ptr()))
        word = Int(status[0])
        if word == 0:
            nc_binary_decision(driver, nc_scratch_address(scores), dst, request.rows)
        _ = scores^
    else:
        plan.decision(driver, dst, Int(status.unsafe_ptr()))
        word = Int(status[0])
    _ = status^
    _ = narrow^
    return word


def nc_proba_wide(request: NcQueryRequest, logarithmic: Bool) -> Bool:
    """True when the probabilities of this request are float64 words."""
    return nc_proba_mode(logarithmic, request.wide) == 2


def nc_proba_run(driver: XnDriver, request: NcQueryRequest, dst: Int, logarithmic: Bool) raises -> Int:
    """Class probabilities (or their logarithm) at `dst`; the word width is
    `nc_proba_wide`. Returns the fused entry's finite word."""
    var mode = nc_proba_mode(logarithmic, request.wide)
    var narrow = List[Float32](length=request.classes, fill=0)
    var plan = _nc_scored(request, narrow)
    var status = 0
    if nc_fused(request.fused, request.rows):
        status = plan.fused(driver, "nc_proba", dst, mode)
    else:
        plan.probabilities(driver, dst, mode)
    _ = narrow^
    return status
