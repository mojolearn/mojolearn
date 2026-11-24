# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE SCOPED FOLD of the KMeans member (lane vdf-scope-cluster, core/kmeans_member.mojo `_scoped_indexed`).

A model-selection fold handed as the whole host X plus int32 row numbers, inside a resident scope with X pinned,
must give the SAME BYTES as the same fold without a scope (the host gather, `_host_indexed`) and as the fold's rows
handed as plain host words; the CPU column runs the host-gather and plain arms of the host image (its fold image
answers no MEMBER_CAN_SCOPE), and the verify's digest compares the columns. Claims, per column:
  device: scoped fit == host-index fit == plain fit (centers bits, labels, inertia bits); scoped predict /
          transform of the test fold == host-index ones; the first scoped call keeps the pin (`pin_kept` 1), a
          second call over the same X finds it resident (`pin_kept` 0: one upload per search).
  host:   host-index fit == plain fit; host-index predict / transform == plain ones.
Data: a SplitMix64 stream (uniform [0, 4)), 96 rows x 3, folds 0..63 / 64..95 as a permutation-sized int32 index.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.device_route import DeviceRoute
from core.entry_table import EntryRef
from core.member_abi import (
    MemberRef, MemberCall, MemberMatrix, member_call, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_TRANSFORM, MEMBER_OP_EXPORT,
    MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE, MEMBER_BLOCK_FAMILY,
)
from core.kmeans_member import kmeans_member_v2
from core.kmeans_native_api import KMeansNativeDriver, KMEANS_CAPABILITIES
from core.kmeans_native_host import kmeans_host_native_v1
from core.kmeans_native_device import kmeans_device_native_v1
from bindings.arena_cache_run import arena_scope_new, arena_scope_close

comptime _ROWS = 96
comptime _D = 3
comptime _TRAIN = 64
comptime _K = 4
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
    return MemberRef(EntryRef(kmeans_member_v2, 0), 0)


def _route(device: Bool) raises -> DeviceRoute:
    var route = DeviceRoute(1 if device else 0)
    route.entry = EntryRef(kmeans_device_native_v1 if device else kmeans_host_native_v1, 0)
    route.flags = KMeansNativeDriver(route.entry.call).metadata(KMEANS_CAPABILITIES)
    return route^


def _new(route: DeviceRoute) raises -> Int:
    # n_clusters 4, k-means++, n_init 1, max_iter 300, tol 1e-4, seed 7, euclidean, oversampling 2
    var params: List[Float64] = [Float64(_K), 0.0, 1.0, 300.0, 1e-4, bitcast[DType.float64](Int64(7)), 0.0, 2.0]
    var call = MemberCall(route)
    call.params = Int(params.unsafe_ptr())
    call.param_count = len(params)
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(params)
    return call.state


struct _Arm(Movable):
    var centers: List[Float32]
    var labels: List[Int32]
    var predicted: List[Int32]
    var distances: List[Float32]
    var pin_kept: List[Int]

    def __init__(out self):
        self.centers = List[Float32]()
        self.labels = List[Int32](length=_TRAIN, fill=0)
        self.predicted = List[Int32](length=_ROWS - _TRAIN, fill=0)
        self.distances = List[Float32](length=(_ROWS - _TRAIN) * _K, fill=0.0)
        self.pin_kept = List[Int]()


def _arm(route: DeviceRoute, x: List[Float32], indexed: Bool, scope: Int, pin: Int) raises -> _Arm:
    """Fit on the train fold (labels by fit_predict), then predict and transform the test fold. `indexed`: the
    whole X plus int32 row numbers; else the fold's rows as plain host words."""
    var arm = _Arm()
    var state = _new(route)
    var train_index = List[Int32](capacity=_TRAIN)
    for i in range(_TRAIN):  # driver-loop: over the fixture's train rows (small)
        train_index.append(Int32(i))
    var test_index = List[Int32](capacity=_ROWS - _TRAIN)
    for i in range(_TRAIN, _ROWS):  # driver-loop: over the fixture's test rows (small)
        test_index.append(Int32(i))
    var base = Int(x.unsafe_ptr())
    for step in range(3):  # driver-loop: over the three operations of one arm
        var call = MemberCall(route)
        call.state = state
        call.scope = scope
        call.pin = pin
        var first = 0 if step == 0 else _TRAIN
        var count = _TRAIN if step == 0 else _ROWS - _TRAIN
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
        else:
            call.dst = MemberMatrix(Int(arm.distances.unsafe_ptr()), count, _K)
            _ = member_call(_member(), MEMBER_OP_TRANSFORM, call)
        arm.pin_kept.append(call.pin_kept)
    var export = MemberCall(route)
    export.state = state
    export.block = MEMBER_BLOCK_FAMILY
    _ = member_call(_member(), MEMBER_OP_EXPORT, export)
    var view = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=export.dst.data)
    for i in range(_K * _D):  # driver-loop: over the exported centers (small)
        arm.centers.append(view[i])
    var free = MemberCall(route)
    free.state = state
    _ = member_call(_member(), MEMBER_OP_FREE, free)
    _ = len(train_index) + len(test_index) + len(x)
    return arm^


def _same(a: _Arm, b: _Arm, what: String) raises:
    for i in range(len(a.centers)):  # driver-loop: over the centers' words (small)
        require(bitcast[DType.uint32](a.centers[i]) == bitcast[DType.uint32](b.centers[i]), what + ": centers bits")
    for i in range(_TRAIN):  # driver-loop: over the train labels (small)
        require(a.labels[i] == b.labels[i], what + ": fit labels")
    for i in range(_ROWS - _TRAIN):  # driver-loop: over the test labels (small)
        require(a.predicted[i] == b.predicted[i], what + ": predict")
    for i in range(len(a.distances)):  # driver-loop: over the test distances (small)
        require(bitcast[DType.uint32](a.distances[i]) == bitcast[DType.uint32](b.distances[i]), what + ": transform bits")


def fixture(device: Bool) raises:
    var x = _uniform4(20261124)
    var route = _route(device)
    var plain = _arm(route, x, False, -1, 0)
    var host_index = _arm(route, x, True, -1, 0)
    _same(host_index, plain, "host row index vs plain fold rows")
    if not device:
        return
    var probe = MemberCall(route)
    require((member_call(_member(), MEMBER_OP_CAPABILITIES, probe) & MEMBER_CAN_SCOPE) != 0,
            "the KMeans member on its fold image answers MEMBER_CAN_SCOPE")
    var scope = arena_scope_new(0)
    try:
        var scoped = _arm(route, x, True, scope, 1)
        _same(scoped, host_index, "scoped device fold vs host row index")
        require(scoped.pin_kept[0] == 1, "the first scoped call publishes X with the caller's pin")
        require(scoped.pin_kept[1] == 0 and scoped.pin_kept[2] == 0, "later calls find X resident (one upload)")
    finally:
        _ = arena_scope_close(scope)
    _ = len(x)


def _host_fixture(unit: Int) raises:
    fixture(False)


def test_kmeans_member_scope() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN == COLUMN_CPU
        check_cases("kmeans_member_scope", SCOPE_FIXTURES, _host_fixture)
    else:
        comptime assert TARGET_COLUMN != COLUMN_CPU
        fixture(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
