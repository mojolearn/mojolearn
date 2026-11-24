# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE SCOPED FOLD of the KernelDensity member (lane vdf-scope-cluster2b, core/density_member.mojo
`_kde_fit_indexed` / `_kde_query_rows`, core/estimators_store.mojo, kde/resident_fit.mojo `kde_fit_prepare_device`
and `kde_score_samples_resident_device`).

A model-selection fold handed as the whole host X plus int32 row numbers, inside a resident scope with X pinned,
must give the SAME BYTES as the same fold without a scope (the host gather) and as the fold's rows handed as plain
host words; the CPU column runs the host-gather and plain arms of the host image (it answers no MEMBER_CAN_SCOPE),
and the verify's digest compares the columns. Claims, per column:
  device: scoped == host-index == plain: score_samples of the test fold (bits), score (bits), the exported
          training rows (bits); the scoped fit keeps the pin (`pin_kept` 1), the scores find X resident
          (`pin_kept` 0: one upload per search); only the scope's X stays in the store, and its close frees it.
  host:   host-index == plain, the same quantities.
Data: a SplitMix64 stream (uniform [0, 4)), 96 rows x 3, folds 0..63 / 64..95 as int32 row numbers; gaussian
kernel, bandwidth 0.75; metrics euclidean, cosine and sqeuclidean (lane vdf-scope-cluster4: the cosine zero-row
rule and the sqeuclidean bound run on the device over the slot, so those two keep the resident-store path).
Refusal words (lane vdf-scope-cluster5): under cosine and sqeuclidean a NaN in the test fold (row 70, column 1)
refuses with the host pass's words, "kde: query contains NaN at row 6, column 1 (DEVIATION 604)", on every arm.
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
    MemberRef, MemberCall, MemberMatrix, member_call, MEMBER_I32,
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_SCORE_SAMPLES, MEMBER_OP_SCORE, MEMBER_OP_EXPORT,
    MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE, MEMBER_CAN_ROW_INDEX, MEMBER_BLOCK_FAMILY,
)
from core.density_member import density_member_v2
from core.kde_native_api import KDENativeDriver, kde_native_v1, KDE_CAPABILITIES, KDE_CAP_DEV_STORE
from core.kde_native_host import KDEHostKernels
from core.kde_native_device import KDEDeviceKernels
from core.estimators_store import estimators_store_v1
from bindings.arena_cache_run import arena_scope_new, arena_scope_close, arena_cache_live
from x_prep.transport_abi import XP_OP_COUNT

comptime _ROWS = 96
comptime _D = 3
comptime _TRAIN = 64
comptime _TEST = _ROWS - _TRAIN
comptime SCOPE_FIXTURES = 3  # unit 0 euclidean, 1 cosine, 2 sqeuclidean


def _metric(unit: Int) -> Float64:
    """The member's metric code of fixture `unit` (core/density_member.mojo `_kde_metric_name`)."""
    return 7.0 if unit == 1 else (2.0 if unit == 2 else 0.0)


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
    return MemberRef(EntryRef(density_member_v2, 0), 0)


def _route(device: Bool) raises -> DeviceRoute:
    var route = DeviceRoute(1 if device else 0)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = device_vendor_code(String(COMPILED_VENDOR)) if device else 0
    route.entry = EntryRef(kde_native_v1[KDEDeviceKernels] if device else kde_native_v1[KDEHostKernels], 0)
    route.flags = KDENativeDriver(route.entry.call).metadata(KDE_CAPABILITIES)
    if device:
        # What providers/python/member_density_boundary_door.mojo hands the member: the store's transport entry.
        route.second = EntryRef(estimators_store_v1, entry_ref_word(route.mode, route.vendor, XP_OP_COUNT))
        route.has_second = True
    return route^


def _new(route: DeviceRoute, metric: Float64 = 0.0) raises -> Int:
    # KernelDensity: bandwidth 0.75, kernel 'gaussian', metric `metric` (0 euclidean, 7 cosine, 2 sqeuclidean)
    var params: List[Float64] = [0.0, 0.75, 0.0, metric]
    var call = MemberCall(route)
    call.params = Int(params.unsafe_ptr())
    call.param_count = len(params)
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(params)
    return call.state


struct _Arm(Movable):
    var train: List[Float32]
    var samples: List[Float32]
    var score: Float64
    var pin_kept: List[Int]

    def __init__(out self):
        self.train = List[Float32]()
        self.samples = List[Float32](length=_TEST, fill=0.0)
        self.score = 0.0
        self.pin_kept = List[Int]()


def _arm(route: DeviceRoute, x: List[Float32], metric: Float64, indexed: Bool, scope: Int, pin: Int) raises -> _Arm:
    """Fit on the train fold, then score_samples and score the test fold. `indexed`: the whole X plus int32 row
    numbers; else the fold's rows as plain host words."""
    var arm = _Arm()
    var state = _new(route, metric)
    var train_index = List[Int32](capacity=_TRAIN)
    for i in range(_TRAIN):  # driver-loop: over the fixture's train rows (small)
        train_index.append(Int32(i))
    var test_index = List[Int32](capacity=_TEST)
    for i in range(_TRAIN, _ROWS):  # driver-loop: over the fixture's test rows (small)
        test_index.append(Int32(i))
    var base = Int(x.unsafe_ptr())
    for step in range(3):  # driver-loop: over the three operations of one arm
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
            _ = member_call(_member(), MEMBER_OP_FIT, call)
        elif step == 1:
            call.dst = MemberMatrix(Int(arm.samples.unsafe_ptr()), count, 1)
            _ = member_call(_member(), MEMBER_OP_SCORE_SAMPLES, call)
        else:
            _ = member_call(_member(), MEMBER_OP_SCORE, call)
            arm.score = call.value
        arm.pin_kept.append(call.pin_kept)
    var export = MemberCall(route)
    export.state = state
    export.block = MEMBER_BLOCK_FAMILY + 4
    _ = member_call(_member(), MEMBER_OP_EXPORT, export)
    var view = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=export.dst.data)
    for i in range(_TRAIN * _D):  # driver-loop: over the exported training rows (small)
        arm.train.append(view[i])
    var free = MemberCall(route)
    free.state = state
    _ = member_call(_member(), MEMBER_OP_FREE, free)
    _ = len(train_index) + len(test_index) + len(x)
    return arm^


def _same(a: _Arm, b: _Arm, what: String) raises:
    for i in range(len(a.train)):  # driver-loop: over the training rows' words (small)
        require(bitcast[DType.uint32](a.train[i]) == bitcast[DType.uint32](b.train[i]), what + ": training rows")
    for i in range(_TEST):  # driver-loop: over the test scores (small)
        require(bitcast[DType.uint32](a.samples[i]) == bitcast[DType.uint32](b.samples[i]), what + ": score_samples")
    require(bitcast[DType.uint64](a.score) == bitcast[DType.uint64](b.score), what + ": score bits")


def _refusal(route: DeviceRoute, x: List[Float32], metric: Float64, indexed: Bool, scope: Int, pin: Int) -> String:
    """The words an arm whose test fold holds a NaN refuses with ("" when it does not refuse)."""
    try:
        _ = _arm(route, x, metric, indexed, scope, pin)
    except error:
        return String(error)
    return String("")


def _nan_words(device: Bool, metric: Float64) raises:
    """Lane vdf-scope-cluster5: the device row rules keep the host pass's finite words."""
    if metric == 0.0:
        return
    var x = _uniform4(20261124)
    x[70 * _D + 1] = Float32.NAN
    var route = _route(device)
    var want = "kde: query contains NaN at row 6, column 1 (DEVIATION 604)"
    var plain = _refusal(route, x, metric, False, -1, 0)
    require(want in plain, "plain fold rows refuse a NaN query in the host pass's words: " + plain)
    var host_index = _refusal(route, x, metric, True, -1, 0)
    require(want in host_index, "the host row index refuses a NaN query in the host pass's words: " + host_index)
    if not device:
        return
    var scope = arena_scope_new(0)
    try:
        var scoped = _refusal(route, x, metric, True, scope, 1)
        require(want in scoped, "the scoped device fold refuses a NaN query in the host pass's words: " + scoped)
    finally:
        _ = arena_scope_close(scope)
    _ = len(x)


def fixture(device: Bool, metric: Float64) raises:
    var x = _uniform4(20261124)
    var route = _route(device)
    var probe = MemberCall(route)
    probe.state = _new(route)
    var mask = member_call(_member(), MEMBER_OP_CAPABILITIES, probe)
    var free = MemberCall(route)
    free.state = probe.state
    _ = member_call(_member(), MEMBER_OP_FREE, free)
    require((mask & MEMBER_CAN_ROW_INDEX) != 0, "a KernelDensity state reads a row index")
    var plain = _arm(route, x, metric, False, -1, 0)
    var host_index = _arm(route, x, metric, True, -1, 0)
    _same(host_index, plain, "host row index vs plain fold rows")
    if not device:
        require((mask & MEMBER_CAN_SCOPE) == 0, "the host image keeps its host path (no MEMBER_CAN_SCOPE)")
        return
    require((route.flags & KDE_CAP_DEV_STORE) != 0, "the GPU estimators image answers KDE_CAP_DEV_STORE")
    require((mask & MEMBER_CAN_SCOPE) != 0, "a KernelDensity state on the GPU image answers MEMBER_CAN_SCOPE")
    var live_before = arena_cache_live(route.second)
    var scope = arena_scope_new(0)
    try:
        var scoped = _arm(route, x, metric, True, scope, 1)
        _same(scoped, host_index, "scoped device fold vs host row index")
        require(scoped.pin_kept[0] == 1, "the scoped fit publishes X with the caller's pin")
        require(scoped.pin_kept[1] == 0 and scoped.pin_kept[2] == 0, "the scores find X resident (one upload)")
        require(arena_cache_live(route.second) == live_before + 1, "only the scope's X stays in the store")
    finally:
        _ = arena_scope_close(scope)
    require(arena_cache_live(route.second) == live_before, "the scope's close frees X from the store")
    _ = len(x)


def _host_fixture(unit: Int) raises:
    fixture(False, _metric(unit))
    _nan_words(False, _metric(unit))


def test_kde_member_scope() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN == COLUMN_CPU
        check_cases("kde_member_scope", SCOPE_FIXTURES, _host_fixture)
    else:
        comptime assert TARGET_COLUMN != COLUMN_CPU
        for unit in range(SCOPE_FIXTURES):  # driver-loop: over the three metrics
            fixture(True, _metric(unit))
            _nan_words(True, _metric(unit))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
