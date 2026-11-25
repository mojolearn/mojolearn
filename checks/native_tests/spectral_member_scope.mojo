# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE SCOPED FOLD of the SpectralClustering member (lane vdf-scope-cluster4, core/spectral_member.mojo `_fit` /
`_fit_rows` / `_spectral_store`, core/metrics_store.mojo, core/metrics_store_ref.mojo, spectral/estimator.mojo
`spectral_fit_predict_dataset_resident_keep`, spectral/impl/preprocessing/detail/spectral_embedding.mojo
`create_connectivity_graph_resident`).

A model-selection fold handed as the whole host X plus int32 row numbers, inside a resident scope with X pinned,
must give the SAME BYTES as the same fold without a scope (the host gather) and as the fold's rows handed as plain
host words; the CPU column runs the host-gather and plain arms of the host image (no MEMBER_CAN_SCOPE), and the
verify's digest compares the columns. Claims, per column:
  device: scoped == host-index == plain: the fit's labels, the embedding block, and the test fold's predicted
          labels (bits); the scoped fit keeps the pin (`pin_kept` 1); only the scope's X stays in the store, and
          its close frees it.
  host:   host-index == plain, the same quantities.
Data: a SplitMix64 stream (uniform [0, 4)), 96 rows x 3, folds 0..63 / 64..95 as int32 row numbers;
affinity 'nearest_neighbors' (8), 3 clusters, 3 components, 2 initialisations, seed 7, assign_labels 'kmeans',
prediction data kept.
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
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_EXPORT,
    MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE, MEMBER_CAN_ROW_INDEX, MEMBER_BLOCK_FAMILY,
)
from core.spectral_member import spectral_member_v2
from core.spectral_native_api import SpectralDriver, SPECTRAL_CAPABILITIES, SPECTRAL_NATIVE_COUNT
from core.spectral_native_provider import spectral_native_v1
from core.spectral_native_host import SpectralHostKernels
from core.spectral_native_device import SpectralDeviceKernels
from core.metrics_store import metrics_store_v1, METRICS_STORE_API
from core.metrics_store_ref import metrics_store_publish, metrics_store_find
from bindings.arena_cache_run import arena_scope_new, arena_scope_close, arena_cache_live

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
    return MemberRef(EntryRef(spectral_member_v2, 0), 0)


def _route(device: Bool) raises -> DeviceRoute:
    var route = DeviceRoute(1 if device else 0)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = device_vendor_code(String(COMPILED_VENDOR)) if device else 0
    # What providers/python/member_spectral_boundary_door.mojo hands the member: the fit and the predict
    # entries of one image (here the same), the predict entry's capabilities.
    # the device entry only in a GPU column build: a CPU column build would elaborate the device kernels
    # ("target 'unknown' is not supported by this build")
    var entry = spectral_native_v1[SpectralHostKernels]
    comptime if not CPU_COLUMN_BUILD:
        if device:
            entry = spectral_native_v1[SpectralDeviceKernels]
    route.entry = EntryRef(entry, entry_ref_word(route.mode, route.vendor, SPECTRAL_NATIVE_COUNT))
    route.second = route.entry
    route.has_second = True
    route.flags = SpectralDriver(entry).metadata(SPECTRAL_CAPABILITIES)
    comptime if not CPU_COLUMN_BUILD:
        if device:
            # What bindings/_mojolearn_metrics.mojo publishes at module initialization.
            metrics_store_publish(route.mode, route.vendor, Int(METRICS_STORE_API.get_or_create_ptr()[].entries.unsafe_ptr()),
                                  metrics_store_v1)
    return route^


def _new(route: DeviceRoute) raises -> Int:
    var seed = bitcast[DType.float64](Int64(7))
    var params: List[Float64] = [0.0, 3.0, Float64(_K), 2.0, 8.0, 0.0, seed, -1.0, 0.0, 1.0, 0.0, 1.0]
    var call = MemberCall(route)
    call.params = Int(params.unsafe_ptr())
    call.param_count = len(params)
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(params)
    return call.state


struct _Arm(Movable):
    var labels: List[Int32]
    var predicted: List[Int32]
    var embedding: List[UInt32]
    var pin_kept: List[Int]

    def __init__(out self):
        self.labels = List[Int32](length=_TRAIN, fill=0)
        self.predicted = List[Int32](length=_TEST, fill=0)
        self.embedding = List[UInt32]()
        self.pin_kept = List[Int]()


def _arm(route: DeviceRoute, x: List[Float32], indexed: Bool, scope: Int, pin: Int) raises -> _Arm:
    """fit (with its labels) on the train fold, then predict the test fold. `indexed`: the whole X plus int32
    row numbers; else the fold's rows as plain host words."""
    var arm = _Arm()
    var state = _new(route)
    var train_index = List[Int32](capacity=_TRAIN)
    for i in range(_TRAIN):  # driver-loop: over the fixture's train rows (small)
        train_index.append(Int32(i))
    var test_index = List[Int32](capacity=_TEST)
    for i in range(_TRAIN, _ROWS):  # driver-loop: over the fixture's test rows (small)
        test_index.append(Int32(i))
    var base = Int(x.unsafe_ptr())
    for step in range(2):  # driver-loop: over the two operations of one arm
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
        else:
            call.dst = MemberMatrix(Int(arm.predicted.unsafe_ptr()), count, 1, MEMBER_I32)
            _ = member_call(_member(), MEMBER_OP_PREDICT, call)
        arm.pin_kept.append(call.pin_kept)
    var export = MemberCall(route)
    export.state = state
    export.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(_member(), MEMBER_OP_EXPORT, export)
    var view = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=export.dst.data)
    for i in range(export.dst.rows * export.dst.cols):  # driver-loop: over the embedding's words (small)
        arm.embedding.append(view[i])
    var free = MemberCall(route)
    free.state = state
    _ = member_call(_member(), MEMBER_OP_FREE, free)
    _ = len(train_index) + len(test_index) + len(x)
    return arm^


def _same(a: _Arm, b: _Arm, what: String) raises:
    for i in range(_TRAIN):  # driver-loop: over the fit's labels (small)
        require(a.labels[i] == b.labels[i], what + ": fit labels")
    for i in range(_TEST):  # driver-loop: over the predicted labels (small)
        require(a.predicted[i] == b.predicted[i], what + ": predicted labels")
    require(len(a.embedding) == len(b.embedding) and len(a.embedding) == _TRAIN * _K, what + ": embedding size")
    for i in range(len(a.embedding)):  # driver-loop: over the embedding's words (small)
        require(a.embedding[i] == b.embedding[i], what + ": embedding bits")


def fixture(device: Bool) raises:
    var x = _uniform4(20261124)
    var route = _route(device)
    var probe = MemberCall(route)
    var mask = member_call(_member(), MEMBER_OP_CAPABILITIES, probe)
    require((mask & MEMBER_CAN_ROW_INDEX) != 0, "a SpectralClustering member reads a row index")
    var plain = _arm(route, x, False, -1, 0)
    var host_index = _arm(route, x, True, -1, 0)
    _same(host_index, plain, "host row index vs plain fold rows")
    if not device:
        require((mask & MEMBER_CAN_SCOPE) == 0, "the host image keeps its host path (no MEMBER_CAN_SCOPE)")
        return
    require((mask & MEMBER_CAN_SCOPE) != 0, "a SpectralClustering member on the GPU image answers MEMBER_CAN_SCOPE")
    var store = metrics_store_find(route.mode, route.vendor)
    require(Bool(store), "the GPU metrics image's store is published for the route's tier and vendor")
    var live_before = arena_cache_live(store.value())
    var scope = arena_scope_new(0)
    try:
        var scoped = _arm(route, x, True, scope, 1)
        _same(scoped, host_index, "scoped device fold vs host row index")
        require(scoped.pin_kept[0] == 1, "the scoped fit publishes X with the caller's pin")
        require(arena_cache_live(store.value()) == live_before + 1, "only the scope's X stays in the store")
    finally:
        _ = arena_scope_close(scope)
    require(arena_cache_live(store.value()) == live_before, "the scope's close frees X from the store")
    _ = len(x)


def _host_fixture(unit: Int) raises:
    fixture(False)


def test_spectral_member_scope() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN == COLUMN_CPU
        check_cases("spectral_member_scope", SCOPE_FIXTURES, _host_fixture)
    else:
        comptime assert TARGET_COLUMN != COLUMN_CPU
        fixture(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
