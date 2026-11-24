# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE SCOPED FOLD of the GaussianMixture member (lane vdf-scope-cluster2b, core/mixture_member.mojo `_rows`,
core/member_scope_rows.mojo, mixture/mixture_store.mojo).

A model-selection fold handed as the whole host X plus int32 row numbers, inside a resident scope with X pinned,
must give the SAME BYTES as the same fold without a scope (the host gather) and as the fold's rows handed as plain
host words; the CPU column runs the host-gather and plain arms of the host image (it answers no MEMBER_CAN_SCOPE),
and the verify's digest compares the columns. Claims, per column:
  device: scoped == host-index == plain: the fitted weights / means / precision factors (bits), the fit_predict
          labels, the test fold's predict, predict_proba (bits), score_samples (bits) and score (bits); the
          first scoped call keeps the pin (`pin_kept` 1), every later call over the same X finds it resident
          (`pin_kept` 0: one upload per search); the store holds no taken slot after the arm.
  host:   host-index == plain, the same quantities.
Data: a SplitMix64 stream (uniform [0, 4)), 96 rows x 3, folds 0..63 / 64..95 as int32 row numbers; three
components, k-means initialization, seed 7.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.device_route import DeviceRoute, device_vendor_code
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberRef, MemberCall, MemberMatrix, member_call, MEMBER_F32, MEMBER_I32,
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA,
    MEMBER_OP_SCORE_SAMPLES, MEMBER_OP_SCORE, MEMBER_OP_EXPORT, MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE,
    MEMBER_CAN_ROW_INDEX, MEMBER_BLOCK_FAMILY,
)
from core.mixture_member import mixture_member_v2
from core.mixture_native_api import GmmDriver, MIXTURE_NATIVE_CAPABILITIES, MIXTURE_CAP_DEV_STORE
from mixture.host.gmm_native_host import gmm_native_host_v1
from mixture.gmm_native_device import gmm_native_device_v1
from mixture.mixture_store import mixture_store_v1
from bindings.arena_cache_run import arena_scope_new, arena_scope_close, arena_cache_live
from x_prep.transport_abi import XP_OP_COUNT

comptime _ROWS = 96
comptime _D = 3
comptime _TRAIN = 64
comptime _TEST = _ROWS - _TRAIN
comptime _K = 3
comptime SCOPE_FIXTURES = 1


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform4(seed: Int) -> List[Float32]:
    var state = UInt64(seed)
    var out = List[Float32](capacity=_ROWS * _D)
    for _ in range(_ROWS * _D):  # driver-loop: over the fixture's words (small)
        out.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
    return out^


def _member() -> MemberRef:
    return MemberRef(EntryRef(mixture_member_v2, 0), 0)


def _route(device: Bool) raises -> DeviceRoute:
    var route = DeviceRoute(1 if device else 0)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = device_vendor_code(String(COMPILED_VENDOR)) if device else 0
    route.entry = EntryRef(gmm_native_device_v1 if device else gmm_native_host_v1, 0)
    route.flags = GmmDriver(route.entry.call).metadata(MIXTURE_NATIVE_CAPABILITIES)
    if device:
        # What providers/python/member_mixture_boundary_door.mojo hands the member: the store's transport entry.
        route.second = EntryRef(mixture_store_v1, entry_ref_word(route.mode, route.vendor, XP_OP_COUNT))
        route.has_second = True
    return route^


def _new(route: DeviceRoute) raises -> Int:
    # n_components 3, max_iter 100, tol 1e-3, reg_covar 1e-6, random_state 7, init_params 'kmeans'
    var params: List[Float64] = [Float64(_K), 100.0, 1e-3, 1e-6, bitcast[DType.float64](Int64(7)), 0.0]
    var call = MemberCall(route)
    call.params = Int(params.unsafe_ptr())
    call.param_count = len(params)
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(params)
    return call.state


struct _Arm(Movable):
    var weights: List[Float32]
    var means: List[Float32]
    var precisions: List[Float32]
    var labels: List[Int32]
    var predicted: List[Int32]
    var proba: List[Float32]
    var samples: List[Float32]
    var score: Float64
    var pin_kept: List[Int]

    def __init__(out self):
        self.weights = List[Float32]()
        self.means = List[Float32]()
        self.precisions = List[Float32]()
        self.labels = List[Int32](length=_TRAIN, fill=0)
        self.predicted = List[Int32](length=_TEST, fill=0)
        self.proba = List[Float32](length=_TEST * _K, fill=0.0)
        self.samples = List[Float32](length=_TEST, fill=0.0)
        self.score = 0.0
        self.pin_kept = List[Int]()


def _block(route: DeviceRoute, state: Int, block: Int, count: Int, mut dst: List[Float32]) raises:
    var export = MemberCall(route)
    export.state = state
    export.block = block
    _ = member_call(_member(), MEMBER_OP_EXPORT, export)
    var view = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=export.dst.data)
    for i in range(count):  # driver-loop: over one exported block's words (small)
        dst.append(view[i])


def _arm(route: DeviceRoute, x: List[Float32], indexed: Bool, scope: Int, pin: Int) raises -> _Arm:
    """Fit on the train fold (labels by fit_predict), then predict, predict_proba, score_samples and score the
    test fold. `indexed`: the whole X plus int32 row numbers; else the fold's rows as plain host words."""
    var arm = _Arm()
    var state = _new(route)
    var train_index = List[Int32](capacity=_TRAIN)
    for i in range(_TRAIN):  # driver-loop: over the fixture's train rows (small)
        train_index.append(Int32(i))
    var test_index = List[Int32](capacity=_TEST)
    for i in range(_TRAIN, _ROWS):  # driver-loop: over the fixture's test rows (small)
        test_index.append(Int32(i))
    var base = Int(x.unsafe_ptr())
    for step in range(5):  # driver-loop: over the five operations of one arm
        var call = MemberCall(route)
        call.state = state
        call.scope = scope
        call.pin = pin
        var first = 0 if step == 0 else _TRAIN
        var count = _TRAIN if step == 0 else _TEST
        if indexed:
            call.x = MemberMatrix(base, _ROWS, _D)
            call.row_index = MemberMatrix(Int((train_index if step == 0 else test_index).unsafe_ptr()), count, 1,
                                          MEMBER_I32)
        else:
            call.x = MemberMatrix(base + first * _D * 4, count, _D)
        if step == 0:
            call.fit_output = MEMBER_OP_PREDICT
            call.dst = MemberMatrix(Int(arm.labels.unsafe_ptr()), count, 1, MEMBER_I32)
            _ = member_call(_member(), MEMBER_OP_FIT, call)
        elif step == 1:
            call.dst = MemberMatrix(Int(arm.predicted.unsafe_ptr()), count, 1, MEMBER_I32)
            _ = member_call(_member(), MEMBER_OP_PREDICT, call)
        elif step == 2:
            call.dst = MemberMatrix(Int(arm.proba.unsafe_ptr()), count, _K)
            _ = member_call(_member(), MEMBER_OP_PREDICT_PROBA, call)
        elif step == 3:
            call.dst = MemberMatrix(Int(arm.samples.unsafe_ptr()), count, 1)
            _ = member_call(_member(), MEMBER_OP_SCORE_SAMPLES, call)
        else:
            _ = member_call(_member(), MEMBER_OP_SCORE, call)
            arm.score = call.value
        arm.pin_kept.append(call.pin_kept)
    _block(route, state, MEMBER_BLOCK_FAMILY, _K, arm.weights)
    _block(route, state, MEMBER_BLOCK_FAMILY + 1, _K * _D, arm.means)
    _block(route, state, MEMBER_BLOCK_FAMILY + 3, _K * _D * _D, arm.precisions)
    var free = MemberCall(route)
    free.state = state
    _ = member_call(_member(), MEMBER_OP_FREE, free)
    _ = len(train_index) + len(test_index) + len(x)
    return arm^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": length")
    for i in range(len(a)):  # driver-loop: over one result's words (small)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": bits")


def _same(a: _Arm, b: _Arm, what: String) raises:
    _same_f32(a.weights, b.weights, what + ": weights")
    _same_f32(a.means, b.means, what + ": means")
    _same_f32(a.precisions, b.precisions, what + ": precision factors")
    for i in range(_TRAIN):  # driver-loop: over the train labels (small)
        require(a.labels[i] == b.labels[i], what + ": fit_predict labels")
    for i in range(_TEST):  # driver-loop: over the test labels (small)
        require(a.predicted[i] == b.predicted[i], what + ": predict")
    _same_f32(a.proba, b.proba, what + ": predict_proba")
    _same_f32(a.samples, b.samples, what + ": score_samples")
    require(bitcast[DType.uint64](a.score) == bitcast[DType.uint64](b.score), what + ": score bits")


def fixture(device: Bool) raises:
    var x = _uniform4(20261124)
    var route = _route(device)
    var probe = MemberCall(route)
    var mask = member_call(_member(), MEMBER_OP_CAPABILITIES, probe)
    require((mask & MEMBER_CAN_ROW_INDEX) != 0, "the GaussianMixture member reads a row index")
    var plain = _arm(route, x, False, -1, 0)
    var host_index = _arm(route, x, True, -1, 0)
    _same(host_index, plain, "host row index vs plain fold rows")
    if not device:
        require((mask & MEMBER_CAN_SCOPE) == 0, "the host image keeps its host path (no MEMBER_CAN_SCOPE)")
        return
    require((route.flags & MIXTURE_CAP_DEV_STORE) != 0, "the GPU mixture image answers MIXTURE_CAP_DEV_STORE")
    require((mask & MEMBER_CAN_SCOPE) != 0, "the GaussianMixture member on the GPU image answers MEMBER_CAN_SCOPE")
    var live_before = arena_cache_live(route.second)
    var scope = arena_scope_new(0)
    try:
        var scoped = _arm(route, x, True, scope, 1)
        _same(scoped, host_index, "scoped device fold vs host row index")
        require(scoped.pin_kept[0] == 1, "the first scoped call publishes X with the caller's pin")
        for i in range(1, len(scoped.pin_kept)):  # driver-loop: over the arm's later calls (small)
            require(scoped.pin_kept[i] == 0, "later calls find X resident (one upload)")
        require(arena_cache_live(route.second) == live_before + 1, "only the scope's X stays in the store")
    finally:
        _ = arena_scope_close(scope)
    require(arena_cache_live(route.second) == live_before, "the scope's close frees X from the store")
    _ = len(x)


def _host_fixture(unit: Int) raises:
    fixture(False)


def test_mixture_member_scope() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN == COLUMN_CPU
        check_cases("mixture_member_scope", SCOPE_FIXTURES, _host_fixture)
    else:
        comptime assert TARGET_COLUMN != COLUMN_CPU
        fixture(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
