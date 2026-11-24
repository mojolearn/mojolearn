# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE MEMBER WIRING of THE RESIDENT ROUTE (lane vdf-scope-neural6, 2026-11-24): the PCA / TruncatedSVD member
(core/decomposition_member.mojo `decomposition_member_v2`) driven as a cross-validation drives it, over the routes the
door builds (providers/python/member_decomposition_boundary_door.mojo `_pca_admit` / `_tsvd_admit`, `_fold_route`,
`_xd_store_route`; built here natively from the same entries, since that UNPACK reads a Python context), inside one
real `resident()` scope (bindings/arena_cache_run.mojo `arena_scope_new`) with a pin, over two folds of one X.

Per configuration, three arms over the same folds; every output compared bit for bit:
  scoped:  the whole host X (and the whole scores matrix) plus int32 row numbers, scope + pin, on the GPU route with
           the fold entries (`route.fourth`), the fold store (`route.fifth`) and the x_decomp store (`route.third`)
  plain:   each fold's rows gathered here and handed as host words, no scope, the same GPU route (the host-input GPU
           path)
  host:    the plain arm on the host column's entries (PCAHostKernels / TSVDHostKernels, the decomposition provider
           on HostExec / NoRDev / RES False), the CPU image's code
Outputs per fold: the fit's projection (`fit_output` = MEMBER_OP_TRANSFORM, the fit_transform route), the transform of
the test rows, and the inverse_transform of the test rows of a fixed scores matrix.

    case 0  PCA covariance fit, unwhitened (fold fit + projection, fold transform, fold inverse)
    case 1  PCA covariance fit, whitened (fold whitened projection and transform, and the whitened inverse on a fold:
            decomposition/pca_fold.mojo `pca_whiten_inverse_transform_fold`)
    case 2  PCA randomized fit (`_rows_open_xd`: the fold taken in the x_decomp store, the estimators driver reading
            the pool id, the projection copied across contexts: decomposition/impl/pca_owner.mojo `pca_project_pool`)
    case 3  TruncatedSVD Gram fit (fold fit, the fold lease `_fold_lease` on transform, fold inverse)
    case 4  TruncatedSVD randomized fit (`_rows_open_xd` over the decomposition provider, `x_dev`)
In every case: the member answers MEMBER_CAN_SCOPE on the scoped route; the first scoped fit keeps the pin and the
second fold's fit finds X resident (`pin_kept` 0); after the states are freed and the scope closed, the fold store
and the x_decomp pool hold exactly the ids they held before (no leak of a taken fold, an index slot or a lease).

What this does not cover: the msel search RUN itself (core/msel_search_run.mojo) cannot drive a decomposition member
(no score is published, scoring None has nothing to call, and the RUN has no pipeline candidate); this check makes
the same member calls the RUN's units make. The Python half of `_xd_store_route` (the factory read from the bridge
context) is import-time glue; its native half (the EntryRef in `route.third`) is what is built here.

Fail-proof edits (each must fail its case): in `_rows_open_xd` take `n - 1` rows (cases 2, 4); in
`pca_project_pool` copy `words - n_features` words (case 2); in `pca_whiten_inverse_transform_fold` gather with
`n_features` columns (case 1); in `_rows_close` skip `arena_cache_free` of `xd_taken` (cases 2, 4: the pool leaks).
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from checks.vendor import COMPILED_VENDOR
from core.device_route import DeviceRoute
from core.entry_table import EntryRef
from core.member_abi import (
    MemberRef, MemberCall, MemberMatrix, member_call, MEMBER_I32,
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM,
    MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE, MEMBER_CAN_ROW_INDEX,
)
from core.decomposition_member import (
    decomposition_member_v2, DECOMP_ROUTE_FUSED, DECOMP_ROUTE_RANDOMIZED, DECOMP_ROUTE_RESIDENT,
    DECOMP_ROUTE_FOLD, DECOMP_ROUTE_XD_STORE,
)
from core.pca_native_api import pca_vendor_code
from core.pca_native_device import pca_device_native_v1, PCADeviceKernels
from core.pca_native_host import pca_host_native_v1, PCAHostKernels
from core.pca_native_fold import pca_fold_native_v1
from core.tsvd_native_api import TSVD_CAP_FUSED, TSVD_CAP_RESIDENT
from core.tsvd_native_device import tsvd_device_native_v1, TSVDDeviceKernels
from core.tsvd_native_host import tsvd_host_native_v1, TSVDHostKernels
from core.tsvd_native_fold import tsvd_fold_native_v1
from core.tsvd_native_decomp import tsvd_decomp_native_v1
from core.msel_fold_transport import msel_fold_store_v1
from core.msel_device_x import msel_fold_store
from x_decomp.device import DevExec
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.member_store import x_decomp_store_v1
from x_decomp.resident import pool_live_count
from bindings.arena_cache_run import arena_scope_new, arena_scope_close

comptime CASES = 5
comptime _ROWS = 48
comptime _D = 6
comptime _NC = 3
comptime _FOLDS = 2
comptime _TEST = _ROWS // _FOLDS
comptime _TRAIN = _ROWS - _TEST


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
    return MemberRef(EntryRef(decomposition_member_v2, 0), 0)


def _gpu_vendor() -> Int:
    # `_route_of`: the connection's PCA-style vendor code minus one.
    return pca_vendor_code(String(COMPILED_VENDOR)) - 1


def _pca_route(device: Bool, randomized: Bool) -> DeviceRoute:
    """`_pca_admit`'s route: the PCA entry and its capability word; on the GPU the fold entries (`_fold_route`)
    and, for a randomized fit, the x_decomp store (`_xd_store_route`)."""
    var route = DeviceRoute(1 if device else 0)
    route.mode = 1
    if not device:
        route.vendor = 0
        route.entry = EntryRef(pca_host_native_v1, 0)
        route.flags = PCAHostKernels.capabilities()
        return route^
    route.vendor = _gpu_vendor()
    route.entry = EntryRef(pca_device_native_v1, 0)
    route.flags = PCADeviceKernels.capabilities()
    route.fourth = EntryRef(pca_fold_native_v1, 0)
    route.has_fourth = True
    route.fifth = EntryRef(msel_fold_store_v1, 0)
    route.has_fifth = True
    route.flags |= DECOMP_ROUTE_FOLD
    if randomized:
        route.third = EntryRef(x_decomp_store_v1, 0)
        route.has_third = True
        route.flags |= DECOMP_ROUTE_XD_STORE
    return route^


def _tsvd_route(device: Bool, randomized: Bool) -> DeviceRoute:
    """`_tsvd_admit`'s route: the TruncatedSVD entry, the decomposition provider in `second`, the fused /
    resident / randomized bits; on the GPU the fold entries and, randomized, the x_decomp store."""
    var route = DeviceRoute(1 if device else 0)
    route.mode = 1
    var caps = TSVDDeviceKernels.capabilities() if device else TSVDHostKernels.capabilities()
    var flags = DECOMP_ROUTE_FUSED if (caps & TSVD_CAP_FUSED) != 0 else 0
    if device and (caps & TSVD_CAP_RESIDENT) != 0:
        flags |= DECOMP_ROUTE_RESIDENT
    flags |= DECOMP_ROUTE_RANDOMIZED
    if not device:
        route.vendor = 0
        route.entry = EntryRef(tsvd_host_native_v1, 0)
        route.second = EntryRef(tsvd_decomp_native_v1[HostExec, NoRDev, False], 0)
        route.has_second = True
        route.flags = flags
        return route^
    route.vendor = _gpu_vendor()
    route.entry = EntryRef(tsvd_device_native_v1, 0)
    route.second = EntryRef(tsvd_decomp_native_v1[DevExec, DevRDev, True], 0)
    route.has_second = True
    route.fourth = EntryRef(tsvd_fold_native_v1, 0)
    route.has_fourth = True
    route.fifth = EntryRef(msel_fold_store_v1, 0)
    route.has_fifth = True
    flags |= DECOMP_ROUTE_FOLD
    if randomized:
        route.third = EntryRef(x_decomp_store_v1, 0)
        route.has_third = True
        flags |= DECOMP_ROUTE_XD_STORE
    route.flags = flags
    return route^


def _params(tsvd: Bool, randomized: Bool, whiten: Bool) -> List[Float64]:
    # kind, n_components, keep mode 0, fraction, solver (0 covariance / Gram, 2 randomized), whiten,
    # n_oversamples, power iterations, seed
    var out: List[Float64] = [1.0 if tsvd else 0.0, Float64(_NC), 0.0, 0.0, 2.0 if randomized else 0.0,
                              1.0 if whiten else 0.0, 4.0, 2.0, 20261124.0]
    return out^


struct _Arm(Movable):
    var projection: List[List[Float32]]
    var transform: List[List[Float32]]
    var inverse: List[List[Float32]]
    var pin_kept: List[Int]

    def __init__(out self):
        self.projection = List[List[Float32]]()
        self.transform = List[List[Float32]]()
        self.inverse = List[List[Float32]]()
        self.pin_kept = List[Int]()


def _call(route: DeviceRoute, state: Int, scope: Int, pin: Int) -> MemberCall:
    var call = MemberCall(route)
    call.state = state
    call.scope = scope
    call.pin = pin
    return call^


def _arm(route: DeviceRoute, params: List[Float64], x: List[Float32], scores: List[Float32], indexed: Bool,
         scope: Int, pin: Int) raises -> _Arm:
    """For each fold: fit on the train rows (with the fit's projection), transform the test rows, inverse_transform
    the test rows of `scores`. `indexed`: the whole matrices plus int32 row numbers; else the rows gathered here."""
    var arm = _Arm()
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
        var s_test = _gather(scores, _NC, test)
        var projection = List[Float32](length=_TRAIN * _NC, fill=Float32(0))
        var transformed = List[Float32](length=_TEST * _NC, fill=Float32(0))
        var inverse = List[Float32](length=_TEST * _D, fill=Float32(0))
        try:
            var fit = _call(route, state, scope, pin)
            if indexed:
                fit.x = MemberMatrix(Int(x.unsafe_ptr()), _ROWS, _D)
                fit.row_index = MemberMatrix(Int(train.unsafe_ptr()), _TRAIN, 1, MEMBER_I32)
            else:
                fit.x = MemberMatrix(Int(x_train.unsafe_ptr()), _TRAIN, _D)
            fit.fit_output = MEMBER_OP_TRANSFORM
            fit.dst = MemberMatrix(Int(projection.unsafe_ptr()), _TRAIN, _NC)
            _ = member_call(_member(), MEMBER_OP_FIT, fit)
            arm.pin_kept.append(fit.pin_kept)
            var query = _call(route, state, scope, pin)
            if indexed:
                query.x = MemberMatrix(Int(x.unsafe_ptr()), _ROWS, _D)
                query.row_index = MemberMatrix(Int(test.unsafe_ptr()), _TEST, 1, MEMBER_I32)
            else:
                query.x = MemberMatrix(Int(x_test.unsafe_ptr()), _TEST, _D)
            query.dst = MemberMatrix(Int(transformed.unsafe_ptr()), _TEST, _NC)
            _ = member_call(_member(), MEMBER_OP_TRANSFORM, query)
            var back = _call(route, state, scope, pin)
            if indexed:
                back.x = MemberMatrix(Int(scores.unsafe_ptr()), _ROWS, _NC)
                back.row_index = MemberMatrix(Int(test.unsafe_ptr()), _TEST, 1, MEMBER_I32)
            else:
                back.x = MemberMatrix(Int(s_test.unsafe_ptr()), _TEST, _NC)
            back.dst = MemberMatrix(Int(inverse.unsafe_ptr()), _TEST, _D)
            _ = member_call(_member(), MEMBER_OP_INVERSE_TRANSFORM, back)
        finally:
            var free = MemberCall(route)
            free.state = state
            _ = member_call(_member(), MEMBER_OP_FREE, free)
        arm.projection.append(projection^)
        arm.transform.append(transformed^)
        arm.inverse.append(inverse^)
        _ = len(train) + len(test) + len(x_train) + len(x_test) + len(s_test)
    _ = len(params) + len(x) + len(scores)
    return arm^


def _same_arms(a: _Arm, b: _Arm, what: String) raises:
    for fold in range(_FOLDS):  # small-loop(oracle: per fold)
        var at = what + " fold " + String(fold)
        _same(a.projection[fold], b.projection[fold], at + " fit projection")
        _same(a.transform[fold], b.transform[fold], at + " transform")
        _same(a.inverse[fold], b.inverse[fold], at + " inverse_transform")


def _configuration(tsvd: Bool, randomized: Bool, whiten: Bool, label: String) raises:
    var x = _values(_ROWS * _D, 3)
    var scores = _values(_ROWS * _NC, 11)
    var params = _params(tsvd, randomized, whiten)
    var gpu = _tsvd_route(True, randomized) if tsvd else _pca_route(True, randomized)
    var host = _tsvd_route(False, randomized) if tsvd else _pca_route(False, randomized)
    var probe = MemberCall(gpu)
    var caps = member_call(_member(), MEMBER_OP_CAPABILITIES, probe)
    require((caps & (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)) == (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE),
            label + ": the member answers MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE on the fold route")
    var plain = _arm(gpu, params, x, scores, False, -1, 0)
    var host_arm = _arm(host, params, x, scores, False, -1, 0)
    _same_arms(plain, host_arm, label + ": GPU host-input vs the host column")
    var folds_before = msel_fold_store()[].live_count()
    var pool_before = pool_live_count()
    var scope = arena_scope_new(0)
    var scoped = _Arm()
    try:
        scoped = _arm(gpu, params, x, scores, True, scope, 1)
    finally:
        _ = arena_scope_close(scope)
    _same_arms(scoped, plain, label + ": scoped fold route vs GPU host-input")
    require(scoped.pin_kept[0] == 1, label + ": the first scoped fit publishes X with the caller's pin")
    require(scoped.pin_kept[1] == 0, label + ": the second fold's fit finds X resident (one upload)")
    require(msel_fold_store()[].live_count() == folds_before, label + ": the fold store holds no leaked slot")
    require(pool_live_count() == pool_before, label + ": the x_decomp pool holds no leaked id")
    _ = len(x) + len(scores) + len(params)


def _case(unit: Int) raises:
    if unit == 0:
        _configuration(False, False, False, "PCA covariance")
    elif unit == 1:
        _configuration(False, False, True, "PCA covariance whitened")
    elif unit == 2:
        _configuration(False, True, False, "PCA randomized")
    elif unit == 3:
        _configuration(True, False, False, "TruncatedSVD Gram")
    else:
        _configuration(True, True, False, "TruncatedSVD randomized")


def test_vdf_scope_neural6_member() raises:
    check_cases_serial("vdf_scope_neural6_member", CASES, _case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_vdf_scope_neural6_member]()
    suite^.run()
