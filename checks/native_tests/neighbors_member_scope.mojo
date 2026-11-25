# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD INDEX and THE FOLD QUERIES of the k-NN member (lane vdf-scope-cluster2a, core/neighbors_member.mojo
`_fit_indexed` / `_indexed_query`).

A model-selection fold handed as the whole host X and y plus int32 row numbers must give the SAME BYTES as the fold's
rows handed as plain host words, without a scope (the host gather) and, on the device column, inside a resident scope
with X pinned (the index built on the device from the fold, NB_OP_INDEX_PREPARE_FOLD; the queries gathered on the
device by the resident entry). The verify's digest compares the columns. Claims, per column:
  host:   classifier host-index fit + predict + predict_proba == plain ones (codes, probability bits); regressor
          host-index fit + predict == plain (value bits).
  device: the same, and scoped == host-index; the member answers MEMBER_CAN_SCOPE; the first scoped call keeps the
          pin (`pin_kept` 1) and later calls of the same search find X resident (`pin_kept` 0).
Each claim holds for euclidean and (lane vdf-scope-cluster7) cosine, whose fold queries run on the device with the
zero-row refusals as device reductions.
Data: a SplitMix64 stream (uniform [0, 4)), 96 rows x 3; class codes 0..2 from the stream; folds 0..63 / 64..95.
"""
from std.testing import TestSuite
from std.memory import bitcast
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.device_route import DeviceRoute, device_vendor_code
from checks.vendor import COMPILED_VENDOR
from core.entry_table import EntryRef
from core.member_abi import (
    MemberRef, MemberCall, MemberMatrix, member_call, MEMBER_I32,
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA,
    MEMBER_OP_CAPABILITIES, MEMBER_CAN_SCOPE, MEMBER_CAN_ROW_INDEX,
)
from core.neighbors_member import neighbors_member_v2
from neighbors.ops_entry_host import neighbors_ops_host_v1
from neighbors.ops_entry_device import neighbors_ops_device_v1
from bindings.arena_cache_run import arena_scope_new, arena_scope_close

comptime _ROWS = 96
comptime _D = 3
comptime _TRAIN = 64
comptime _TEST = _ROWS - _TRAIN
comptime _CLASSES = 3
comptime SCOPE_FIXTURES = 1


def _next(mut state: UInt64) -> UInt64:
    state += 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform4(seed: Int, count: Int) -> List[Float32]:
    var state = UInt64(seed)
    var out = List[Float32](capacity=count)
    for _ in range(count):  # driver-loop: over the fixture's words (small)
        out.append(Float32(Float64(_next(state) >> 11) * (1.0 / 9007199254740992.0) * 4.0))
    return out^


def _codes(seed: Int) -> List[Int32]:
    var state = UInt64(seed)
    var out = List[Int32](capacity=_ROWS)
    for _ in range(_ROWS):  # driver-loop: over the fixture's rows (small)
        out.append(Int32(Int(_next(state) % UInt64(_CLASSES))))
    return out^


def _member() -> MemberRef:
    return MemberRef(EntryRef(neighbors_member_v2, 0), 0)


def _route(device: Bool) raises -> DeviceRoute:
    var route = DeviceRoute(1 if device else 0)
    # the vendor word a resolved route carries (core/device_route.mojo: 0 CPU); the member answers MEMBER_CAN_SCOPE
    # from it (core/neighbors_member.mojo `route.vendor != 0`), and a hand-built route's default word is -1
    route.vendor = device_vendor_code(String(COMPILED_VENDOR)) if device else 0
    # the device entry only in a GPU column build: a CPU column build would elaborate the device kernels
    # ("target 'unknown' is not supported by this build")
    route.entry = EntryRef(neighbors_ops_host_v1, 0)
    comptime if not CPU_COLUMN_BUILD:
        if device:
            route.entry = EntryRef(neighbors_ops_device_v1, 0)
    return route^


def _new(route: DeviceRoute, regressor: Bool, metric: Int) raises -> Int:
    # kind, n_neighbors 5, the metric (1 euclidean, 2 cosine), metric argument 2, distance weights (1), query_tile 0
    var params: List[Float64] = [1.0 if regressor else 0.0, 5.0, Float64(metric), 2.0, 1.0, 0.0]
    var call = MemberCall(route)
    call.params = Int(params.unsafe_ptr())
    call.param_count = len(params)
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = len(params)
    return call.state


struct _Arm(Movable):
    var codes: List[Int32]
    var proba: List[Float32]
    var values: List[Float32]
    var pin_kept: List[Int]

    def __init__(out self):
        self.codes = List[Int32](length=_TEST, fill=0)
        self.proba = List[Float32](length=_TEST * _CLASSES, fill=0.0)
        self.values = List[Float32](length=_TEST, fill=0.0)
        self.pin_kept = List[Int]()


def _arm(route: DeviceRoute, x: List[Float32], codes: List[Int32], targets: List[Float32], regressor: Bool,
         indexed: Bool, scope: Int, pin: Int, metric: Int) raises -> _Arm:
    """Fit on the train fold, then predict (and for the classifier predict_proba) the test fold. `indexed`: the
    whole X and y plus int32 row numbers; else the fold's rows as plain host words."""
    var arm = _Arm()
    var state = _new(route, regressor, metric)
    var train_index = List[Int32](capacity=_TRAIN)
    for i in range(_TRAIN):  # driver-loop: over the fixture's train rows (small)
        train_index.append(Int32(i))
    var test_index = List[Int32](capacity=_TEST)
    for i in range(_TRAIN, _ROWS):  # driver-loop: over the fixture's test rows (small)
        test_index.append(Int32(i))
    var base = Int(x.unsafe_ptr())
    var y_base = Int(targets.unsafe_ptr()) if regressor else Int(codes.unsafe_ptr())
    var steps = 2 if regressor else 3
    for step in range(steps):  # driver-loop: over the operations of one arm
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
            var y_rows = _ROWS if indexed else count
            call.y = (MemberMatrix(y_base, y_rows, 1) if regressor
                      else MemberMatrix(y_base, y_rows, 1, MEMBER_I32))
            call.classes = 0 if regressor else _CLASSES
            _ = member_call(_member(), MEMBER_OP_FIT, call)
        elif step == 1:
            call.dst = (MemberMatrix(Int(arm.values.unsafe_ptr()), count, 1) if regressor
                        else MemberMatrix(Int(arm.codes.unsafe_ptr()), count, 1, MEMBER_I32))
            _ = member_call(_member(), MEMBER_OP_PREDICT, call)
        else:
            call.dst = MemberMatrix(Int(arm.proba.unsafe_ptr()), count, _CLASSES)
            _ = member_call(_member(), MEMBER_OP_PREDICT_PROBA, call)
        arm.pin_kept.append(call.pin_kept)
    var free = MemberCall(route)
    free.state = state
    _ = member_call(_member(), MEMBER_OP_FREE, free)
    _ = len(train_index) + len(test_index) + len(x) + len(codes) + len(targets)
    return arm^


def _same(a: _Arm, b: _Arm, what: String) raises:
    for i in range(_TEST):  # driver-loop: over the test rows (small)
        require(a.codes[i] == b.codes[i], what + ": predicted codes")
        require(bitcast[DType.uint32](a.values[i]) == bitcast[DType.uint32](b.values[i]), what + ": regression bits")
    for i in range(len(a.proba)):  # driver-loop: over the probabilities (small)
        require(bitcast[DType.uint32](a.proba[i]) == bitcast[DType.uint32](b.proba[i]), what + ": proba bits")


def fixture(device: Bool) raises:
    var x = _uniform4(20261124, _ROWS * _D)
    var targets = _uniform4(20261126, _ROWS)
    var codes = _codes(20261127)
    var route = _route(device)
    var probe = MemberCall(route)
    var caps = member_call(_member(), MEMBER_OP_CAPABILITIES, probe)
    require((caps & MEMBER_CAN_ROW_INDEX) != 0, "the k-NN member answers MEMBER_CAN_ROW_INDEX")
    require(((caps & MEMBER_CAN_SCOPE) != 0) == device, "MEMBER_CAN_SCOPE on the device column only")
    for which in range(4):  # driver-loop: over the classifier and the regressor, euclidean and cosine
        var regressor = which % 2 == 1
        # Cosine (lane vdf-scope-cluster7): the fit keeps the gathered points, the fold queries are gathered on the
        # device and the zero-row refusals run there (neighbors/zero_row_device.mojo).
        var metric = 1 if which < 2 else 2
        var what = (String("regressor ") if regressor else String("classifier ")) + (
            String("euclidean ") if metric == 1 else String("cosine "))
        var plain = _arm(route, x, codes, targets, regressor, False, -1, 0, metric)
        var host_index = _arm(route, x, codes, targets, regressor, True, -1, 0, metric)
        _same(host_index, plain, what + "host row index vs plain fold rows")
        if not device:
            continue
        var scope = arena_scope_new(0)
        try:
            var scoped = _arm(route, x, codes, targets, regressor, True, scope, 1, metric)
            _same(scoped, host_index, what + "device fold (index and queries) vs host row index")
            var first = 0 if metric == 1 else 1  # a cosine fit gathers on the host: its first put is the predict's
            require(scoped.pin_kept[first] == 1, what + "the first scoped put publishes X with the caller's pin")
            for i in range(len(scoped.pin_kept)):  # driver-loop: over the arm's calls (small)
                if i != first:
                    require(scoped.pin_kept[i] == 0, what + "the other calls find X resident (one upload)")
        finally:
            _ = arena_scope_close(scope)
    _ = len(x) + len(targets) + len(codes)


def _host_fixture(unit: Int) raises:
    fixture(False)


def test_neighbors_member_scope() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN == COLUMN_CPU
        check_cases("neighbors_member_scope", SCOPE_FIXTURES, _host_fixture)
    else:
        comptime assert TARGET_COLUMN != COLUMN_CPU
        fixture(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
