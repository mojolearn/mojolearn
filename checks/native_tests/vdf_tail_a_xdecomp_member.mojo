# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE MEMBER WIRING of THE RESIDENT ROUTE for the XDecomp and manifold members (lane vdf-tail-a, 2026-11-24): the
XDecomp member (core/xdecomp_member.mojo `xdecomp_member_v2`) driven as a cross-validation drives it, over the routes
the door builds (providers/python/member_xdecomp_boundary_door.mojo `_table_route` / `_store_route` / `_mani_route`;
built here natively from the same entries, since that UNPACK reads a Python context), inside one real `resident()`
scope (bindings/arena_cache_run.mojo `arena_scope_new`) with a pin, over two folds of one X. The pattern of
checks/native_tests/vdf_scope_neural6_member.mojo (PCA / TruncatedSVD), for the kinds the x_decomp operations entry
runs: the per-fold member calls of the msel search RUN's units, which the RUN itself cannot make for a member with
no score (as there).

Per configuration, three arms over the same folds; every output compared bit for bit:
  scoped:  the whole host X (and the whole scores matrix) plus int32 row numbers, scope + pin, on the GPU route
           (`entry` x_decomp_device_v1, XDECOMP_ROUTE_POOL, the x_decomp store `x_decomp_store_v1` in `route.fifth`
           with SEQ_ROUTE_STORE)
  plain:   each fold's rows gathered here and handed as host words, no scope, the same GPU route (the host-input GPU
           path)
  host:    the plain arm on the host column's entry (x_decomp_host_v1, no pool, no store), the CPU image's code;
           compared in an identical build only (`fast` promises no bits across columns)

    case 0  GaussianRandomProjection (kind 5): fit (shape), transform of the test rows (XD_RP_PROJECT over the fold's
            pool id), inverse_transform of the test rows of a fixed scores matrix
    case 1  FastICA (kind 7, whitened, parallel, logcosh): fit with the sources (fit_output TRANSFORM, `x_dev`),
            transform and inverse_transform of the test rows
    case 2  FactorAnalysis (kind 8): fit (`fa_fit_run` over the pool id), transform and score_samples of the test
            rows
    case 3  Isomap (kind 16, the manifold kit `MRun.adopt`: the fold's pool id handed to the run, kept as _fit_X):
            fit with embedding_ (fit_output TRANSFORM), transform of the test rows
In every case: the member answers MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE on the store route; the first scoped fit
keeps the pin and the second fold's fit finds X resident (`pin_kept` 0); after the states are freed and the scope
closed the x_decomp pool holds exactly the ids it held before (no leak of a taken fold or a kept _fit_X).

Fail-proof edits (each must fail its case): in core/seq_member_rows.mojo `member_rows_open` take `n - 1` rows (every
case); in `_mani_x` skip `held.taken = -1` (case 3: the fold id is freed twice or leaks); in `rp_project_run` read
the X words from the host address regardless of the pool word (case 0 faults: null X).

The image key words of the manifold kit (words [3], the binding module's address in the door) are fixed small
integers here: the run matches images by them and this check opens one image per arm.
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_ref_call
from core.member_abi import (
    MemberRef, MemberCall, MemberMatrix, member_call, MEMBER_I32,
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM,
    MEMBER_OP_SCORE_SAMPLES, MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE, MEMBER_CAN_ROW_INDEX,
)
from core.seq_member_rows import SEQ_ROUTE_STORE
from core.xdecomp_member import (
    xdecomp_member_v2, XDECOMP_ROUTE_POOL, XDECOMP_KIND_GRP, XDECOMP_KIND_ICA, XDECOMP_KIND_FA, XDECOMP_KIND_ISOMAP,
    XDECOMP_RP_PARAM_COUNT, XDECOMP_ICA_PARAM_COUNT, XDECOMP_FA_PARAM_COUNT, XDECOMP_ISOMAP_PARAM_COUNT,
)
from x_decomp.descriptor_schema import XD_CAPABILITIES
from x_decomp.descriptor_device import x_decomp_device_v1
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.member_store import x_decomp_store_v1
from x_decomp.resident import pool_live_count
from x_prep.transport_abi import xp_vendor_word
from bindings.arena_cache_run import arena_scope_new, arena_scope_close

comptime CASES = 4
comptime _ROWS = 48
comptime _D = 5
comptime _NC = 3
comptime _FOLDS = 2
comptime _TEST = _ROWS // _FOLDS
comptime _TRAIN = _ROWS - _TEST

# what a configuration runs after its fit, on the test rows
comptime _Q_TRANSFORM = 1
comptime _Q_INVERSE = 2
comptime _Q_SCORE_SAMPLES = 4


def _values(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture values)
        out[i] = Float32((i * 7 + salt * 5 + (i * i) % 11) % 17 - 8) / Float32(8)
    return out^


def _test_rows(fold: Int) -> List[Int32]:
    var out = List[Int32](capacity=_TEST)
    for i in range(_TEST):  # small-loop(fixture: the fold's test rows, every other row)
        out.append(Int32(2 * i + fold))
    return out^


def _train_rows(fold: Int) -> List[Int32]:
    # The other fold's rows, in reverse order (a gather, never a slice).
    var out = List[Int32](capacity=_TRAIN)
    for i in range(_TRAIN):  # small-loop(fixture: the fold's train rows)
        out.append(Int32(2 * (_TRAIN - 1 - i) + (1 - fold)))
    return out^


def _gather(whole: List[Float32], row_words: Int, idx: List[Int32]) -> List[Float32]:
    var out = List[Float32](length=len(idx) * row_words, fill=Float32(0))
    for r in range(len(idx)):  # small-loop(oracle: the fold's rows)
        for c in range(row_words):  # small-loop(oracle: one row's words)
            out[r * row_words + c] = whole[Int(idx[r]) * row_words + c]
    return out^


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: word compare)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]),
                what + ": word " + String(i) + " differs")


def _member() -> MemberRef:
    return MemberRef(EntryRef(xdecomp_member_v2, 0), 0)


def _entry(device: Bool) -> EntryRef:
    return EntryRef(x_decomp_device_v1, 0) if device else EntryRef(x_decomp_host_v1, 0)


def _route(device: Bool) -> DeviceRoute:
    """`_table_route`'s route: the image's x_decomp entry; on the GPU XDECOMP_ROUTE_POOL and the store
    (`_store_route`: `route.fifth`, SEQ_ROUTE_STORE)."""
    var route = DeviceRoute(1 if device else 0)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.entry = _entry(device)
    if not device:
        route.vendor = 0
        route.flags = 0
        return route^
    route.vendor = xp_vendor_word(String(COMPILED_VENDOR))
    route.flags = XDECOMP_ROUTE_POOL
    route.fifth = EntryRef(x_decomp_store_v1, 0)
    route.has_fifth = True
    route.flags |= SEQ_ROUTE_STORE
    return route^


def _params(unit: Int, device: Bool) raises -> List[Float64]:
    if unit == 0:
        # kind, n_components, eps, density (Gaussian 0), compute_inverse_components, seed
        var rp: List[Float64] = [Float64(XDECOMP_KIND_GRP), Float64(_NC), 0.1, 0.0, 1.0, 20261124.0]
        require(len(rp) == XDECOMP_RP_PARAM_COUNT, "random projection words")
        return rp^
    if unit == 1:
        # kind, n_components, whiten unit-variance, parallel, logcosh, alpha, max_iter, tol, seed, no w_init
        var ica: List[Float64] = [Float64(XDECOMP_KIND_ICA), Float64(_NC), 1.0, 0.0, 0.0, 1.0, 40.0, 1.0e-4,
                                  20261124.0, 0.0, 0.0, 0.0]
        require(len(ica) == XDECOMP_ICA_PARAM_COUNT, "FastICA words")
        return ica^
    if unit == 2:
        # kind, n_components, max_iter, rotation None, tol, noise_variance_init None
        var fa: List[Float64] = [Float64(XDECOMP_KIND_FA), 2.0, 40.0, 0.0, 1.0e-2, -1.0]
        require(len(fa) == XDECOMP_FA_PARAM_COUNT, "FactorAnalysis words")
        return fa^
    # Isomap: kind, the image's capability word, ordered-LLE 0, the image key (module note), n_neighbors 6,
    # euclidean (0, power 2), the dense eigen solver, n_components, no radius
    var caps = entry_ref_call(_entry(device), XD_CAPABILITIES, List[Int](), List[Int](), List[Float64]())
    var iso: List[Float64] = [Float64(XDECOMP_KIND_ISOMAP), Float64(caps), 0.0, 2.0 if device else 1.0, 6.0, 0.0,
                              2.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    require(len(iso) == XDECOMP_ISOMAP_PARAM_COUNT, "Isomap words")
    return iso^


def _fit_transform(unit: Int) -> Bool:
    return unit == 1 or unit == 3


def _queries(unit: Int) -> Int:
    if unit == 0 or unit == 1:
        return _Q_TRANSFORM | _Q_INVERSE
    if unit == 2:
        return _Q_TRANSFORM | _Q_SCORE_SAMPLES
    return _Q_TRANSFORM


def _width(unit: Int) -> Int:
    return 2 if unit == 2 or unit == 3 else _NC


struct _Arm(Movable):
    var outputs: List[List[Float32]]  # per fold, in call order: [fit output], transform, [inverse], [score_samples]
    var pin_kept: List[Int]

    def __init__(out self):
        self.outputs = List[List[Float32]]()
        self.pin_kept = List[Int]()


def _call(route: DeviceRoute, state: Int, scope: Int, pin: Int) -> MemberCall:
    var call = MemberCall(route)
    call.state = state
    call.scope = scope
    call.pin = pin
    return call^


def _rows_view(mut call: MemberCall, whole: List[Float32], gathered: List[Float32], words: Int, idx: List[Int32],
               indexed: Bool):
    if indexed:
        call.x = MemberMatrix(Int(whole.unsafe_ptr()), _ROWS, words)
        call.row_index = MemberMatrix(Int(idx.unsafe_ptr()), len(idx), 1, MEMBER_I32)
    else:
        call.x = MemberMatrix(Int(gathered.unsafe_ptr()), len(idx), words)


def _arm(unit: Int, route: DeviceRoute, params: List[Float64], x: List[Float32], scores: List[Float32],
         indexed: Bool, scope: Int, pin: Int) raises -> _Arm:
    """For each fold: fit on the train rows (with the fit's output where the kind writes one), then the kind's
    queries on the test rows. `indexed`: the whole matrices plus int32 row numbers; else the rows gathered here."""
    var arm = _Arm()
    var nc = _width(unit)
    var queries = _queries(unit)
    for fold in range(_FOLDS):  # driver-loop: over the folds (a search's units)
        var train = _train_rows(fold)
        var test = _test_rows(fold)
        var new = MemberCall(route)
        new.params = Int(params.unsafe_ptr())
        new.param_count = len(params)
        _ = member_call(_member(), MEMBER_OP_NEW, new)
        var state = new.state
        var x_train = _gather(x, _D, train)
        var x_test = _gather(x, _D, test)
        var s_test = _gather(scores, nc, test)
        var fitted = List[Float32](length=_TRAIN * nc, fill=Float32(0))
        var transformed = List[Float32](length=_TEST * nc, fill=Float32(0))
        var inverse = List[Float32](length=_TEST * _D, fill=Float32(0))
        var samples = List[Float32](length=_TEST, fill=Float32(0))
        try:
            var fit = _call(route, state, scope, pin)
            _rows_view(fit, x, x_train, _D, train, indexed)
            if _fit_transform(unit):
                fit.fit_output = MEMBER_OP_TRANSFORM
                fit.dst = MemberMatrix(Int(fitted.unsafe_ptr()), _TRAIN, nc)
            _ = member_call(_member(), MEMBER_OP_FIT, fit)
            arm.pin_kept.append(fit.pin_kept)
            if (queries & _Q_TRANSFORM) != 0:
                var query = _call(route, state, scope, pin)
                _rows_view(query, x, x_test, _D, test, indexed)
                query.dst = MemberMatrix(Int(transformed.unsafe_ptr()), _TEST, nc)
                _ = member_call(_member(), MEMBER_OP_TRANSFORM, query)
            if (queries & _Q_INVERSE) != 0:
                var back = _call(route, state, scope, pin)
                _rows_view(back, scores, s_test, nc, test, indexed)
                back.dst = MemberMatrix(Int(inverse.unsafe_ptr()), _TEST, _D)
                _ = member_call(_member(), MEMBER_OP_INVERSE_TRANSFORM, back)
            if (queries & _Q_SCORE_SAMPLES) != 0:
                var sc = _call(route, state, scope, pin)
                _rows_view(sc, x, x_test, _D, test, indexed)
                sc.dst = MemberMatrix(Int(samples.unsafe_ptr()), _TEST, 1)
                _ = member_call(_member(), MEMBER_OP_SCORE_SAMPLES, sc)
        finally:
            var free = MemberCall(route)
            free.state = state
            _ = member_call(_member(), MEMBER_OP_FREE, free)
        if _fit_transform(unit):
            arm.outputs.append(fitted^)
        else:
            _ = len(fitted)
        arm.outputs.append(transformed^)
        if (queries & _Q_INVERSE) != 0:
            arm.outputs.append(inverse^)
        else:
            _ = len(inverse)
        if (queries & _Q_SCORE_SAMPLES) != 0:
            arm.outputs.append(samples^)
        else:
            _ = len(samples)
        _ = len(train) + len(test) + len(x_train) + len(x_test) + len(s_test)
    _ = len(params) + len(x) + len(scores)
    return arm^


def _same_arms(a: _Arm, b: _Arm, what: String) raises:
    require(len(a.outputs) == len(b.outputs), what + ": output counts differ")
    for i in range(len(a.outputs)):  # small-loop(oracle: per fold and output)
        _same(a.outputs[i], b.outputs[i], what + " output " + String(i))


def _configuration(unit: Int, label: String) raises:
    var x = _values(_ROWS * _D, 3)
    var scores = _values(_ROWS * _width(unit), 11)
    var gpu = _route(True)
    var gpu_params = _params(unit, True)
    var probe = MemberCall(gpu)
    var caps = member_call(_member(), MEMBER_OP_CAPABILITIES, probe)
    require((caps & (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)) == (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE),
            label + ": the member answers MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE on the store route")
    var plain = _arm(unit, gpu, gpu_params, x, scores, False, -1, 0)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var host = _route(False)
        var host_params = _params(unit, False)
        var host_arm = _arm(unit, host, host_params, x, scores, False, -1, 0)
        _same_arms(plain, host_arm, label + ": GPU host-input vs the host column")
    var pool_before = pool_live_count()
    var scope = arena_scope_new(0)
    var scoped = _Arm()
    try:
        scoped = _arm(unit, gpu, gpu_params, x, scores, True, scope, 1)
    finally:
        _ = arena_scope_close(scope)
    _same_arms(scoped, plain, label + ": scoped store route vs GPU host-input")
    require(scoped.pin_kept[0] == 1, label + ": the first scoped fit publishes X with the caller's pin")
    require(scoped.pin_kept[1] == 0, label + ": the second fold's fit finds X resident (one upload)")
    require(pool_live_count() == pool_before, label + ": the x_decomp pool holds no leaked id")
    _ = len(x) + len(scores) + len(gpu_params)


def _case(unit: Int) raises:
    if unit == 0:
        _configuration(0, "GaussianRandomProjection")
    elif unit == 1:
        _configuration(1, "FastICA")
    elif unit == 2:
        _configuration(2, "FactorAnalysis")
    else:
        _configuration(3, "Isomap")


def test_vdf_tail_a_xdecomp_member() raises:
    check_cases_serial("vdf_tail_a_xdecomp_member", CASES, _case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_vdf_tail_a_xdecomp_member]()
    suite^.run()
