# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE SCOPED FOLD of the UMAP member (lane vdf-scope-cluster7, core/manifold_member.mojo `_fit` / `_transform` /
`_store` / `_own_rows`, the slot form of core/neighbors_entry_table_umap.mojo, umap/ops_entry_device.mojo,
umap/sparse_estimator.mojo `sparse_fit_transform_device`, umap/transform.mojo `transform_device`,
neighbors/estimator.mojo `knn_self_search_slot` / `knn_search_slot_queries`, neighbors/zero_row_device.mojo).

A model-selection fold handed as the whole host X plus int32 row numbers, inside a resident scope with X pinned,
must give the SAME BYTES as the same fold without a scope (the host gather) and as the fold's rows handed as plain
host words; the CPU column runs the host-gather and plain arms of the host image (no MEMBER_CAN_SCOPE). Claims, per
column and per metric (euclidean on the legacy fit, cosine on the extended fit):
  device: scoped == host-index == plain: the fit_transform embedding, the test fold's transform, the exported
          training rows (the scoped state reads its fitted-rows slot home once) and the training fold's own
          transform (the same-input reuse, made on the device for the scoped state); the scoped fit keeps the
          pin; only the scope's X stays in the store after the states are freed, and the scope's close frees it.
          A cosine fold with an all-zero row is refused in the host scan's words on every arm.
  host:   host-index == plain, the same quantities and the same refusal.
Data: a SplitMix64 stream (uniform [0, 4)), 96 rows x 3, folds 0..63 / 64..95 as int32 row numbers; UMAP
n_neighbors 5, 2 components, 10 epochs, seed 0, spectral start.
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
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_TRANSFORM, MEMBER_OP_EXPORT,
    MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE, MEMBER_CAN_ROW_INDEX, MEMBER_BLOCK_FAMILY,
)
from core.manifold_member import manifold_member_v2, MANIFOLD_KIND_UMAP, MANIFOLD_PARAM_COUNT, MF_INIT_WORD, MF_EXACT_WORD
from core.neighbors_entry_table_umap import UMAP_OPERATION_COUNT
from umap.ops_entry_host import umap_ops_host_v1
from umap.ops_entry_device import umap_ops_device_v1
from core.metrics_store import metrics_store_v1, METRICS_STORE_API
from core.metrics_store_ref import metrics_store_publish, metrics_store_find
from bindings.arena_cache_run import arena_scope_new, arena_scope_close, arena_cache_live

comptime _ROWS = 96
comptime _D = 3
comptime _TRAIN = 64
comptime _TEST = _ROWS - _TRAIN
comptime _NC = 2
comptime SCOPE_FIXTURES = 2
comptime _METRIC_EUCLIDEAN = -1
comptime _METRIC_COSINE = 2


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
    return MemberRef(EntryRef(manifold_member_v2, 0), 0)


def _route(device: Bool) raises -> DeviceRoute:
    var route = DeviceRoute(1 if device else 0)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = device_vendor_code(String(COMPILED_VENDOR)) if device else 0
    # What the UMAP member door hands the member: the UMAP operations entry of the route's image.
    # the device entry only in a GPU column build: a CPU column build would elaborate the device kernels
    # ("target 'unknown' is not supported by this build")
    var entry = umap_ops_host_v1
    comptime if not CPU_COLUMN_BUILD:
        if device:
            entry = umap_ops_device_v1
    route.entry = EntryRef(entry, entry_ref_word(route.mode, route.vendor, UMAP_OPERATION_COUNT))
    comptime if not CPU_COLUMN_BUILD:
        if device:
            # What bindings/_mojolearn_metrics.mojo publishes at module initialization.
            metrics_store_publish(route.mode, route.vendor, Int(METRICS_STORE_API.get_or_create_ptr()[].entries.unsafe_ptr()),
                                  metrics_store_v1)
    return route^


def _words(metric: Int) -> List[Float64]:
    var words: List[Float64] = [Float64(MANIFOLD_KIND_UMAP), 5.0, Float64(_NC), 10.0, 0.1, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 5.0, Float64(metric), 2.0, 0.0, 0.0, -1.0, 0.0, 0.5, Float64(GLOBAL_NUMERIC_MODE)]
    words.resize(MANIFOLD_PARAM_COUNT, 0.0)
    words[MF_INIT_WORD] = 0.0
    words[MF_EXACT_WORD] = bitcast[DType.float64](Int64(5))
    words[MF_EXACT_WORD + 1] = bitcast[DType.float64](Int64(10))
    words[MF_EXACT_WORD + 2] = bitcast[DType.float64](Int64(0))
    return words^


def _new(route: DeviceRoute, words: List[Float64]) raises -> Int:
    var call = MemberCall(route)
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(words)
    return call.state


struct _Arm(Movable):
    var fitted: List[UInt32]
    var test: List[UInt32]
    var again: List[UInt32]
    var rows: List[UInt32]
    var pin_kept: List[Int]

    def __init__(out self):
        self.fitted = List[UInt32](length=_TRAIN * _NC, fill=0)
        self.test = List[UInt32](length=_TEST * _NC, fill=0)
        self.again = List[UInt32](length=_TRAIN * _NC, fill=0)
        self.rows = List[UInt32]()
        self.pin_kept = List[Int]()


def _index(first: Int, count: Int) -> List[Int32]:
    var index = List[Int32](capacity=count)
    for i in range(first, first + count):  # driver-loop: over the fixture's fold rows (small)
        index.append(Int32(i))
    return index^


def _arm(route: DeviceRoute, metric: Int, x: List[Float32], indexed: Bool, scope: Int, pin: Int) raises -> _Arm:
    """fit_transform on the train fold, transform the test fold, transform the train fold again (the reuse),
    then export the training rows. `indexed`: the whole X plus int32 row numbers; else plain host fold words."""
    var arm = _Arm()
    var words = _words(metric)
    var state = _new(route, words)
    var train_index = _index(0, _TRAIN)
    var test_index = _index(_TRAIN, _TEST)
    var base = Int(x.unsafe_ptr())
    for step in range(3):  # driver-loop: over the three operations of one arm
        var call = MemberCall(route)
        call.state = state
        call.scope = scope
        call.pin = pin
        var on_test = step == 1
        var first = _TRAIN if on_test else 0
        var count = _TEST if on_test else _TRAIN
        if indexed:
            call.x = MemberMatrix(base, _ROWS, _D)
            call.row_index = MemberMatrix(Int((test_index if on_test else train_index).unsafe_ptr()), count, 1,
                                          MEMBER_I32)
        else:
            call.x = MemberMatrix(base + first * _D * 4, count, _D)
        var dst = arm.fitted.unsafe_ptr() if step == 0 else (arm.test.unsafe_ptr() if on_test else arm.again.unsafe_ptr())
        call.dst = MemberMatrix(Int(dst), count, _NC)
        if step == 0:
            call.fit_output = MEMBER_OP_TRANSFORM
            _ = member_call(_member(), MEMBER_OP_FIT, call)
        else:
            _ = member_call(_member(), MEMBER_OP_TRANSFORM, call)
        arm.pin_kept.append(call.pin_kept)
    var export = MemberCall(route)
    export.state = state
    export.block = MEMBER_BLOCK_FAMILY
    _ = member_call(_member(), MEMBER_OP_EXPORT, export)
    var view = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=export.dst.data)
    for i in range(export.dst.rows * export.dst.cols):  # driver-loop: over the rows block's words (small)
        arm.rows.append(view[i])
    var free = MemberCall(route)
    free.state = state
    _ = member_call(_member(), MEMBER_OP_FREE, free)
    _ = len(train_index) + len(test_index) + len(x) + len(words)
    return arm^


def _same(a: _Arm, b: _Arm, x: List[Float32], what: String) raises:
    for i in range(_TRAIN * _NC):  # driver-loop: over the fit's embedding words (small)
        require(a.fitted[i] == b.fitted[i], what + ": fit_transform bits")
        require(a.again[i] == b.again[i], what + ": train-fold transform bits")
        require(a.again[i] == a.fitted[i], what + ": the train fold's transform reuses the fitted embedding")
    for i in range(_TEST * _NC):  # driver-loop: over the test fold's words (small)
        require(a.test[i] == b.test[i], what + ": test-fold transform bits")
    require(len(a.rows) == _TRAIN * _D and len(b.rows) == _TRAIN * _D, what + ": rows block size")
    var xw = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    for i in range(_TRAIN * _D):  # driver-loop: over the rows block's words (small)
        require(a.rows[i] == xw[i] and b.rows[i] == xw[i], what + ": exported training rows are the fold's words")


def _refusal(route: DeviceRoute, x: List[Float32], indexed: Bool, scope: Int, pin: Int) -> String:
    try:
        _ = _arm(route, _METRIC_COSINE, x, indexed, scope, pin)
    except e:
        return String(e)
    return String("")


def fixture(device: Bool, metric: Int) raises:
    var x = _uniform4(20261124 + metric)
    var route = _route(device)
    var probe = MemberCall(route)
    var mask = member_call(_member(), MEMBER_OP_CAPABILITIES, probe)
    require((mask & MEMBER_CAN_ROW_INDEX) != 0, "a UMAP member reads a row index")
    var plain = _arm(route, metric, x, False, -1, 0)
    var host_index = _arm(route, metric, x, True, -1, 0)
    _same(host_index, plain, x, "host row index vs plain fold rows")
    # A zero row in the train fold under cosine: the same refusal words on every arm.
    var zeroed = x.copy()
    if metric == _METRIC_COSINE:
        for f in range(_D):  # driver-loop: over one fixture row (small)
            zeroed[7 * _D + f] = 0.0
        var want = _refusal(route, zeroed, False, -1, 0)
        require("index row 7 is all zeros" in want, "cosine: the host scan refuses the zero row")
        require(_refusal(route, zeroed, True, -1, 0) == want, "cosine refusal: host row index vs plain")
    if not device:
        require((mask & MEMBER_CAN_SCOPE) == 0, "the host image keeps its host path (no MEMBER_CAN_SCOPE)")
        return
    require((mask & MEMBER_CAN_SCOPE) != 0, "a UMAP member on the GPU metrics image answers MEMBER_CAN_SCOPE")
    var store = metrics_store_find(route.mode, route.vendor)
    require(Bool(store), "the GPU metrics image's store is published for the route's tier and vendor")
    var live_before = arena_cache_live(store.value())
    var scope = arena_scope_new(0)
    try:
        var scoped = _arm(route, metric, x, True, scope, 1)
        _same(scoped, host_index, x, "scoped device fold vs host row index")
        require(scoped.pin_kept[0] == 1, "the scoped fit publishes X with the caller's pin")
        require(arena_cache_live(store.value()) == live_before + 1,
                "only the scope's X stays in the store (the fitted-rows and fold slots were freed)")
    finally:
        _ = arena_scope_close(scope)
    require(arena_cache_live(store.value()) == live_before, "the scope's close frees X from the store")
    if metric == _METRIC_COSINE:
        var want = _refusal(route, zeroed, False, -1, 0)
        var scope2 = arena_scope_new(0)
        try:
            require(_refusal(route, zeroed, True, scope2, 1) == want,
                    "cosine refusal on the device fold: the host scan's words")
        finally:
            _ = arena_scope_close(scope2)
    _ = len(x) + len(zeroed)


def _host_fixture(unit: Int) raises:
    fixture(False, _METRIC_EUCLIDEAN if unit == 0 else _METRIC_COSINE)


def test_manifold_member_scope() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN == COLUMN_CPU
        check_cases("manifold_member_scope", SCOPE_FIXTURES, _host_fixture)
    else:
        comptime assert TARGET_COLUMN != COLUMN_CPU
        fixture(True, _METRIC_EUCLIDEAN)
        fixture(True, _METRIC_COSINE)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
