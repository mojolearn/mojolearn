# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""A second identical call uploads nothing fitted.

Per family that keeps fitted state on the device, the fitted operand goes up
once, two identical queries follow, and the count of host-to-device copies
of FITTED words (core/resident_upload_count.mojo, noted at each family's one
fitted upload site; compiled only under
`-D MOJOLEARN_RESIDENT_UPLOAD_COUNT=1`, the define of the native checks' GPU
build) must not move over either query, and the two queries must return the
same words. Each case first requires the count to MOVE at the fit (a counter
that is not wired cannot pass), and a build without the define SKIPS the
counted cases: it never reports them passed.

  0 KDE                  kde_fit_prepare, kde_score_samples_resident twice
  1 neighbors index      knn_index_prepare, knn_index_search twice
  2 KMeans query         kmeans_centers_prepare, kmeans_predict_resident twice
  3 optimizer state      opt_resident_open, a state load, opt_resident_step
                         twice with the same parameters and gradients: the
                         state (the moments) never goes up again; a second
                         load is counted
  4 Samba registry       SambaResidentSession, samba_resident_forward twice:
                         the session's own registry upload count (not under
                         the define) stays, and moves by one after
                         registry_epoch_bump
  5 x_linear decision    xl_decision_device_run twice with one weight block.
                         THE BLOCK IS UPLOADED ON EVERY CALL (a resident
                         copy was timed at noise and deleted, 2026-11-19):
                         one fitted copy per call is pinned, and a third
                         call with a changed block must upload once and
                         must not be served the old words
  6 isotonic predict     the knots ride the one upload that carries the
                         call's zeroed status words (x_linear/device.mojo
                         `_iso_predict_grid`), so one fitted copy per call
                         is the whole cost and no separate copy exists to
                         keep: the case pins exactly one per call

Not here, and why: the decomposition kit (PLS, CCA, ALS, MinCovDet,
dictionary learning) and the graph layers decide a resident hit on Python
objects in their doors' UNPACK (the frozen fitted Array and its address;
the edge_index key); the typed RUN below them has no notion of a second
call, so a native check cannot ask the question. The decomposition door's
miss upload carries the count (RUC_DECOMP_KIT) for a counted build.

What a sabotage changes here: a query that builds its resident entry again
or reloads the index, the centers, the training rows or the weights (cases
0 to 2); a step that takes the moments from the host again (3); a registry
key that never reads current (4: the count grows at the second forward) or
never expires (4: it does not grow after the epoch bump); a resident weight
block served after its words changed (5, arm); a second fitted copy added
to the isotonic predict (6).

Every transfer in this file is the product's own (the checks hand over host
lists by address); it queues no copy.
"""
from max.gpu.host import DeviceContext

from core.resident_upload_count import (
    RESIDENT_UPLOAD_COUNT,
    RUC_ISOTONIC,
    RUC_KDE,
    RUC_KMEANS_CENTERS,
    RUC_KNN_INDEX,
    RUC_OPTIMIZER_STATE,
    RUC_X_LINEAR,
    resident_upload_copies,
)
from core.weight_epoch import registry_epoch_bump
from kde.resident_fit import kde_fit_prepare, kde_fit_release, kde_score_samples_resident
from neighbors.resident_index import knn_index_prepare, knn_index_release, knn_index_search
from cluster.resident_centers import kmeans_centers_prepare, kmeans_centers_release, kmeans_predict_resident
from sequence.ops import OPT_ADAM
from sequence.opt_resident import opt_resident_close, opt_resident_move, opt_resident_open, opt_resident_step
from training.samba_resident import (
    SAMBA_LAYER_ATTENTION,
    SambaResidentSession,
    samba_registry_layout,
    samba_resident_forward,
)
from transformer.impl.llama.modeling_llama import LlamaDims
from x_linear.dispatch import ALGO_ISOTONIC_PREDICT, LINK_IDENTITY
from x_linear.ops_entry_device import xl_decision_device_run, xl_fit_device_run

from checks.native_tests.core_ops_common import uniform
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require


comptime RSC_CASES = 7
#: The one case that reads a counter the product keeps in every build.
comptime _CASE_SAMBA = 4

comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32 = MutPointer[Int32, MutUntrackedOrigin]
comptime _U32 = MutPointer[UInt32, MutUntrackedOrigin]


def _f32(values: List[Float32]) -> _F32:
    return _F32(unsafe_from_address=Int(values.unsafe_ptr()))


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": extent")
    for i in range(len(a)):
        require(a[i].to_bits() == b[i].to_bits(), what + ": the second call's word " + String(i) + " differs")


def _same_u32(a: List[UInt32], b: List[UInt32], what: String) raises:
    require(len(a) == len(b), what + ": extent")
    for i in range(len(a)):
        require(a[i] == b[i], what + ": the second call's word " + String(i) + " differs")


def _fit_counted(before: Int, family: Int, what: String) raises -> Int:
    """The count after the fit; it must have moved (the counter is wired)."""
    var now = resident_upload_copies(family)
    require(now > before, what + ": the fit uploaded its operand and no fitted copy was counted (counter not wired)")
    return now


def _no_fitted_upload(at_fit: Int, family: Int, what: String) raises:
    var now = resident_upload_copies(family)
    require(now == at_fit, what + ": " + String(now - at_fit) + " host-to-device copies of fitted words since the fit")


def _kde() raises:
    var rows = 211
    var features = 5
    var queries = 37
    var train = uniform(rows * features, 71, -1.0, 1.0)
    var weights = uniform(rows, 72, 0.5, 1.5)
    var query = uniform(queries * features, 73, -1.0, 1.0)
    var first = List[Float32](length=queries, fill=0)
    var second = List[Float32](length=queries, fill=0)
    var before = resident_upload_copies(RUC_KDE)
    var handle = kde_fit_prepare(_f32(train), rows, features, Float32(0.25), String("gaussian"), String("euclidean"),
                                 weights, True)
    try:
        var at_fit = _fit_counted(before, RUC_KDE, "KDE")
        _ = kde_score_samples_resident(handle, _f32(query), queries, features, Float32(0.25), String("gaussian"),
                                       String("euclidean"), _f32(first))
        _no_fitted_upload(at_fit, RUC_KDE, "KDE first score")
        _ = kde_score_samples_resident(handle, _f32(query), queries, features, Float32(0.25), String("gaussian"),
                                       String("euclidean"), _f32(second))
        _no_fitted_upload(at_fit, RUC_KDE, "KDE second score")
    except error:
        kde_fit_release(handle)
        raise error
    kde_fit_release(handle)
    _same_f32(first, second, "KDE scores")
    _ = train
    _ = weights
    _ = query


def _knn() raises:
    var rows = 301
    var features = 6
    var queries = 29
    var k = 4
    var index = uniform(rows * features, 81, -1.0, 1.0)
    var query = uniform(queries * features, 82, -1.0, 1.0)
    var d1 = List[Float32](length=queries * k, fill=0)
    var d2 = List[Float32](length=queries * k, fill=0)
    var i1 = List[UInt32](length=queries * k, fill=0)
    var i2 = List[UInt32](length=queries * k, fill=0)
    var before = resident_upload_copies(RUC_KNN_INDEX)
    var handle = knn_index_prepare(_f32(index), rows, features)
    try:
        var at_fit = _fit_counted(before, RUC_KNN_INDEX, "neighbors index")
        _ = knn_index_search(handle, _f32(index), rows, _f32(query), queries, features, k, _f32(d1),
                             _U32(unsafe_from_address=Int(i1.unsafe_ptr())))
        _no_fitted_upload(at_fit, RUC_KNN_INDEX, "neighbors first search")
        _ = knn_index_search(handle, _f32(index), rows, _f32(query), queries, features, k, _f32(d2),
                             _U32(unsafe_from_address=Int(i2.unsafe_ptr())))
        _no_fitted_upload(at_fit, RUC_KNN_INDEX, "neighbors second search")
    except error:
        knn_index_release(handle)
        raise error
    knn_index_release(handle)
    _same_f32(d1, d2, "neighbors distances")
    _same_u32(i1, i2, "neighbors indices")
    _ = index
    _ = query


def _kmeans() raises:
    var clusters = 7
    var features = 5
    var rows = 113
    var centers = uniform(clusters * features, 91, -1.0, 1.0)
    var x = uniform(rows * features, 92, -1.0, 1.0)
    var l1 = List[UInt32](length=rows, fill=0)
    var l2 = List[UInt32](length=rows, fill=0)
    var before = resident_upload_copies(RUC_KMEANS_CENTERS)
    var handle = kmeans_centers_prepare(_f32(centers), clusters, features)
    try:
        var at_fit = _fit_counted(before, RUC_KMEANS_CENTERS, "KMeans centers")
        # metric 0: METRIC_L2_EXPANDED, the public 'euclidean' (cluster/impl/kmeans_params.mojo)
        kmeans_predict_resident(handle, _f32(x), rows, features, clusters,
                                _U32(unsafe_from_address=Int(l1.unsafe_ptr())), 0)
        _no_fitted_upload(at_fit, RUC_KMEANS_CENTERS, "KMeans first predict")
        kmeans_predict_resident(handle, _f32(x), rows, features, clusters,
                                _U32(unsafe_from_address=Int(l2.unsafe_ptr())), 0)
        _no_fitted_upload(at_fit, RUC_KMEANS_CENTERS, "KMeans second predict")
    except error:
        kmeans_centers_release(handle)
        raise error
    kmeans_centers_release(handle)
    _same_u32(l1, l2, "KMeans labels")
    _ = centers
    _ = x


def _optimizer() raises:
    var n = 1024
    var parameters = uniform(n, 101, -1.0, 1.0)
    var gradients = uniform(n, 102, -0.1, 0.1)
    var moment = uniform(n, 103, 0.0, 0.01)
    var scalars = List[Float32](length=3, fill=0)
    # Adam: lr, beta1, beta2, eps, weight_decay, f7
    var fp: List[Float64] = [0.001, 0.9, 0.999, 0.00000001, 0.0, 0.0]
    var opened = opt_resident_open([n, OPT_ADAM, 0], fp)
    var handle = opened[0]
    try:
        # a state load (load_state_dict): the one time the moments go up
        var before = resident_upload_copies(RUC_OPTIMIZER_STATE)
        require(opt_resident_move(handle, 0, Int(moment.unsafe_ptr()), True) == 1, "Adam keeps state slot 0")
        var at_fit = _fit_counted(before, RUC_OPTIMIZER_STATE, "optimizer state")
        var addresses: List[Int] = [Int(parameters.unsafe_ptr()), Int(gradients.unsafe_ptr()), Int(scalars.unsafe_ptr())]
        require(opt_resident_step(handle, addresses, [1, OPT_ADAM, 0, 1, 0, n], fp) == n, "optimizer step extent")
        _no_fitted_upload(at_fit, RUC_OPTIMIZER_STATE, "optimizer first step")
        require(opt_resident_step(handle, addresses, [1, OPT_ADAM, 0, 2, 1, n], fp) == n, "optimizer step extent")
        _no_fitted_upload(at_fit, RUC_OPTIMIZER_STATE, "optimizer second step")
        # the counter can move again: a second load is one more copy
        _ = opt_resident_move(handle, 0, Int(moment.unsafe_ptr()), True)
        require(resident_upload_copies(RUC_OPTIMIZER_STATE) == at_fit + 1, "a second state load is one fitted copy")
    except error:
        _ = opt_resident_close(handle)
        raise error
    _ = opt_resident_close(handle)
    _ = parameters
    _ = gradients
    _ = moment
    _ = scalars


def _samba(ctx: DeviceContext) raises:
    var vocab = 64
    var dm = 32
    var heads = 2
    var kv_heads = 1
    var hd = 16
    var inter = 64
    var b = 1
    var l = 4
    var kinds: List[Int] = [SAMBA_LAYER_ATTENTION]
    var offsets = List[Int]()
    var slot = List[Int]()
    var ldims = LlamaDims(dm, heads, kv_heads, hd, inter)
    var total = samba_registry_layout(kinds, vocab, dm, ldims, inter, True, offsets, slot)
    # every tensor in (0.05, 0.25): finite, the norm weights positive
    var registry = uniform(total, 111, 0.05, 0.25)
    var ids = List[Int32](length=b * l, fill=0)
    for i in range(b * l):
        ids[i] = Int32((7 * i + 3) % vocab)
    var first = List[Float32](length=b * l * vocab, fill=0)
    var second = List[Float32](length=b * l * vocab, fill=0)
    var third = List[Float32](length=b * l * vocab, fill=0)
    var session = SambaResidentSession(ctx, kinds, vocab, dm, heads, kv_heads, hd, inter, True, Float32(1.0e-6), total,
                                       _f32(registry))
    var ip = _I32(unsafe_from_address=Int(ids.unsafe_ptr()))
    var u1 = samba_resident_forward(session, _f32(registry), ip, _f32(first), b, l)
    var u2 = samba_resident_forward(session, _f32(registry), ip, _f32(second), b, l)
    require(u2 == u1, "Samba: the second forward uploaded the registry again (" + String(u1) + " -> " + String(u2) + ")")
    _same_f32(first, second, "Samba logits")
    # the key must expire when the registry's bytes may have changed
    registry_epoch_bump(Int(registry.unsafe_ptr()))
    var u3 = samba_resident_forward(session, _f32(registry), ip, _f32(third), b, l)
    require(u3 == u2 + 1, "Samba: a forward after the registry's epoch moved must upload it once ("
            + String(u2) + " -> " + String(u3) + ")")
    _same_f32(first, third, "Samba logits after the reload")
    _ = session^
    _ = registry
    _ = ids


def _x_linear() raises:
    var rows = 97
    var features = 6
    var targets = 3
    var x = uniform(rows * features, 121, -1.0, 1.0)
    var block = uniform(targets * (features + 1), 122, -0.5, 0.5)
    var first = List[Float32](length=rows * targets, fill=0)
    var second = List[Float32](length=rows * targets, fill=0)
    var third = List[Float32](length=rows * targets, fill=0)
    var xa = Int(x.unsafe_ptr())
    var c0 = resident_upload_copies(RUC_X_LINEAR)
    _ = xl_decision_device_run(xa, Int(block.unsafe_ptr()), rows, features, targets, LINK_IDENTITY,
                               Int(first.unsafe_ptr()), 0)
    var c1 = _fit_counted(c0, RUC_X_LINEAR, "x_linear weight block")
    require(c1 == c0 + 1, "x_linear: the first decision call uploads its weight block once")
    _ = xl_decision_device_run(xa, Int(block.unsafe_ptr()), rows, features, targets, LINK_IDENTITY,
                               Int(second.unsafe_ptr()), 0)
    var c2 = resident_upload_copies(RUC_X_LINEAR)
    _same_f32(first, second, "x_linear decision")
    # The known cost (module docstring, case 5). A third copy per call would be new.
    # TOMBSTONE: MOJOLEARN_XLINEAR_FAST_WB_RESIDENT (the arm's clause: the second identical call uploads nothing; NOISE) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
    # Restore: git apply experiments/removed/MOJOLEARN_XLINEAR_FAST_WB_RESIDENT.patch; record in docs/TOMBSTONES.md.
    require(c2 == c1 + 1, "x_linear: the block is uploaded once per call, counted "
            + String(c2 - c1))
    # a changed block must go up, and its words must be the ones used
    var other = uniform(targets * (features + 1), 123, 0.5, 1.5)
    _ = xl_decision_device_run(xa, Int(other.unsafe_ptr()), rows, features, targets, LINK_IDENTITY,
                               Int(third.unsafe_ptr()), 0)
    require(resident_upload_copies(RUC_X_LINEAR) == c2 + 1, "x_linear: a changed weight block is one fitted copy")
    var moved = 0
    for i in range(rows * targets):
        if third[i].to_bits() != first[i].to_bits():
            moved += 1
    require(2 * moved > rows * targets, "x_linear: a call with another weight block returned the first block's words on "
            + String(rows * targets - moved) + " of " + String(rows * targets) + " outputs")
    _ = other
    _ = x
    _ = block


def _isotonic() raises:
    var knots = 33
    var rows = 257
    # [X thresholds ascending | y thresholds ascending]
    var thresholds = List[Float32](length=2 * knots, fill=0)
    for i in range(knots):
        thresholds[i] = Float32(i) * Float32(0.25)
        thresholds[knots + i] = Float32(i) * Float32(0.5) + Float32(1.0)
    var x = uniform(rows, 131, 0.0, 8.0)
    var first = List[Float32](length=rows, fill=0)
    var second = List[Float32](length=rows, fill=0)
    var ip: List[Int32] = [Int32(knots), Int32(1)]          # k, out_of_bounds = clip
    var fp: List[Float32] = [Float32(0.0), Float32(8.0)]    # X_min_, X_max_
    var c0 = resident_upload_copies(RUC_ISOTONIC)
    _ = xl_fit_device_run(ALGO_ISOTONIC_PREDICT, Int(x.unsafe_ptr()), Int(thresholds.unsafe_ptr()), rows, 1, rows,
                          2 * knots, rows, 1, 1, ip, fp, Int(first.unsafe_ptr()))
    var c1 = _fit_counted(c0, RUC_ISOTONIC, "isotonic knots")
    _ = xl_fit_device_run(ALGO_ISOTONIC_PREDICT, Int(x.unsafe_ptr()), Int(thresholds.unsafe_ptr()), rows, 1, rows,
                          2 * knots, rows, 1, 1, ip, fp, Int(second.unsafe_ptr()))
    var c2 = resident_upload_copies(RUC_ISOTONIC)
    require(c1 == c0 + 1 and c2 == c1 + 1,
            "isotonic predict: one upload carries the knots with the call's status words; counted "
            + String(c1 - c0) + " then " + String(c2 - c1) + " fitted copies per call")
    _same_f32(first, second, "isotonic predictions")
    _ = x
    _ = thresholds


def _case(ctx: DeviceContext, which: Int) raises:
    if which == 0:
        _kde()
    elif which == 1:
        _knn()
    elif which == 2:
        _kmeans()
    elif which == 3:
        _optimizer()
    elif which == _CASE_SAMBA:
        _samba(ctx)
    elif which == 5:
        _x_linear()
    else:
        _isotonic()


# The device column only. The resident families run in order, one device queue each in turn. The families whose
# fitted copies are counted need `-D MOJOLEARN_RESIDENT_UPLOAD_COUNT=1` (the native checks' GPU build passes it);
# without it `main()` skips that test and only the samba case runs.
comptime _COUNTED_CASES = RSC_CASES - 1


def _counted_case(row: Int) -> Int:
    """Row `row` of the counted table: every case but samba, in the old order."""
    return row if row < _CASE_SAMBA else row + 1


def test_counted_uploads() raises:
    var ctx = DeviceContext()
    def counted(row: Int) raises {imm ctx}:
        _case(ctx, _counted_case(row))
    check_cases_serial("counted_uploads", _COUNTED_CASES, counted)


def test_samba() raises:
    var ctx = DeviceContext()
    _case(ctx, _CASE_SAMBA)


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    comptime if not RESIDENT_UPLOAD_COUNT:
        suite.skip[test_counted_uploads]()
    suite^.run()
