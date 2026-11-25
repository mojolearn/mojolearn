# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE SCALER PROGRAM ROUTES ON A FOLD (lane vdf-scope-neural6, 2026-11-24): the StandardScaler / MinMaxScaler member
(core/scaler_member.mojo `scaler_member_v2`) over a model-selection fold kept in the preprocessing member store, on the
paths that run the scaler cores' x_prep programs (core/scaler_core.mojo): a NaN X (fit and transform), sample weights
(fit), a fitted all-NaN column (the per-column NaN restore). Those programs now read the fold from an x_prep store copy
of the member slot (core/scaler_slot_runs.mojo, the ranges runner's slot input) instead of a host gather.

Three arms over two folds of one X, each output compared bit for bit:
  scoped: the whole X (and weights) plus int32 row numbers in a real `resident()` scope with a pin, on the GPU route
          (scaler kernel entry, GPU x_prep transport, preprocessing member store, SCALER_ROUTE_DIRECT | _STORE)
  plain:  each fold's rows gathered here as host words, no scope, the same GPU route (the host-input GPU path)
  host:   the plain arm on the host column's entries (ScalerHostKernels, the host x_prep transport; the route the
          host images publish, checks/native_tests/py2mojo_host_members.mojo `host_route`)
Outputs per fold: the transform of the test rows (it reads every fitted statistic) and its inverse_transform.

    case 0  StandardScaler, NaN words in X (fit: the weighted-statistics program; transform: the NaN fill and restore)
    case 1  StandardScaler with sample weights (finite X: the program route through the weights)
    case 2  MinMaxScaler, NaN words in X (the fill program, the refusing kernel on the filled block)
    case 3  StandardScaler with a column that is NaN in every row (the fitted all-NaN column: the restore program on a
            finite transform)
In every case the x_prep store holds exactly the slots it held before (each copy freed) and the scope closes clean.

Fail-proof edits (each must fail its case): in `scaler_slot_to_xprep` copy `words - 1` words (cases 0-3); in
`ScalerTableCalls.run_slot` bring back from `out_lo + 1` (cases 0, 2); in `_transform_restore` pass `x_slot` -1 on
the slot route (case 3: the restore reads unset host words).
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberRef, MemberCall, MemberMatrix, member_call, MEMBER_I32,
    MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM,
)
from core.scaler_member import scaler_member_v2, SCALER_ROUTE_DIRECT, SCALER_ROUTE_STORE
from core.scaler_native_api import scaler_native_v1, SCALER_OP_COUNT
from core.scaler_native_device import ScalerDeviceKernels
from core.scaler_native_host import ScalerHostKernels
from core.pca_native_api import pca_vendor_code
from core.fold_x_device import FOLD_X_STORE
from x_prep.transport_abi import XP_OP_COUNT
from x_prep.transport_device import x_prep_transport_device_v1
from x_prep.transport_host import x_prep_transport_host_v1
from preprocessing.prep_member_store import prep_member_store_v1
from bindings.arena_cache_run import arena_scope_new, arena_scope_close

comptime CASES = 4
comptime _ROWS = 40
comptime _D = 4
comptime _FOLDS = 2
comptime _TEST = _ROWS // _FOLDS
comptime _TRAIN = _ROWS - _TEST


def _nan() -> Float32:
    return bitcast[DType.float32](UInt32(0x7FC00000))


def _x(nan_cells: Bool, nan_column: Bool) -> List[Float32]:
    var out = List[Float32](length=_ROWS * _D, fill=Float32(0))
    for i in range(_ROWS * _D):  # small-loop(fixture values)
        out[i] = Float32((i * 5 + (i * i) % 7) % 19 - 9) / Float32(4)
        if nan_cells and (i * 3) % 11 == 4:
            out[i] = _nan()
        if nan_column and i % _D == 1:
            out[i] = _nan()
    return out^


def _weights() -> List[Float32]:
    var out = List[Float32](length=_ROWS, fill=Float32(0))
    for i in range(_ROWS):  # small-loop(fixture weights)
        out[i] = Float32(1 + i % 3) / Float32(2)
    return out^


def _test_rows(fold: Int) -> List[Int32]:
    var out = List[Int32](capacity=_TEST)
    for i in range(_TEST):  # small-loop(fixture: every other row)
        out.append(Int32(2 * i + fold))
    return out^


def _train_rows(fold: Int) -> List[Int32]:
    var out = List[Int32](capacity=_TRAIN)
    for i in range(_TRAIN):  # small-loop(fixture: the other rows, reversed)
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
    return MemberRef(EntryRef(scaler_member_v2, 0), 0)


def _route(device: Bool) -> DeviceRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(1 if device else 0)
    route.mode = mode
    if not device:
        route.vendor = 0
        route.entry = EntryRef(scaler_native_v1[ScalerHostKernels], entry_ref_word(mode, 0, SCALER_OP_COUNT))
        route.second = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
        route.has_second = True
        route.flags = 0
        return route^
    var vendor = pca_vendor_code(String(COMPILED_VENDOR)) - 1
    route.vendor = vendor
    route.entry = EntryRef(scaler_native_v1[ScalerDeviceKernels], entry_ref_word(mode, vendor, SCALER_OP_COUNT))
    route.second = EntryRef(x_prep_transport_device_v1, entry_ref_word(mode, vendor, XP_OP_COUNT))
    route.has_second = True
    route.fifth = EntryRef(prep_member_store_v1, entry_ref_word(mode, vendor, XP_OP_COUNT))
    route.has_fifth = True
    route.flags = SCALER_ROUTE_DIRECT | SCALER_ROUTE_STORE
    return route^


def _arm(route: DeviceRoute, params: List[Float64], x: List[Float32], w: List[Float32], weighted: Bool,
         indexed: Bool, scope: Int, pin: Int) raises -> List[List[Float32]]:
    """Per fold: fit on the train rows (with weights), transform the test rows, inverse_transform that output."""
    var outs = List[List[Float32]]()
    for fold in range(_FOLDS):  # driver-loop: over the folds (a search's units)
        var train = _train_rows(fold)
        var test = _test_rows(fold)
        var x_train = _gather(x, _D, train)
        var w_train = _gather(w, 1, train)
        var x_test = _gather(x, _D, test)
        var forward = List[Float32](length=_TEST * _D, fill=Float32(0))
        var back = List[Float32](length=_TEST * _D, fill=Float32(0))
        var new = MemberCall(route)
        new.params = Int(params.unsafe_ptr())
        new.param_count = len(params)
        _ = member_call(_member(), MEMBER_OP_NEW, new)
        var state = new.state
        try:
            var fit = MemberCall(route)
            fit.state = state
            fit.scope = scope
            fit.pin = pin
            if indexed:
                fit.x = MemberMatrix(Int(x.unsafe_ptr()), _ROWS, _D)
                fit.row_index = MemberMatrix(Int(train.unsafe_ptr()), _TRAIN, 1, MEMBER_I32)
                if weighted:
                    fit.weights = MemberMatrix(Int(w.unsafe_ptr()), _ROWS, 1)
            else:
                fit.x = MemberMatrix(Int(x_train.unsafe_ptr()), _TRAIN, _D)
                if weighted:
                    fit.weights = MemberMatrix(Int(w_train.unsafe_ptr()), _TRAIN, 1)
            _ = member_call(_member(), MEMBER_OP_FIT, fit)
            var query = MemberCall(route)
            query.state = state
            query.scope = scope
            query.pin = pin
            if indexed:
                query.x = MemberMatrix(Int(x.unsafe_ptr()), _ROWS, _D)
                query.row_index = MemberMatrix(Int(test.unsafe_ptr()), _TEST, 1, MEMBER_I32)
            else:
                query.x = MemberMatrix(Int(x_test.unsafe_ptr()), _TEST, _D)
            query.dst = MemberMatrix(Int(forward.unsafe_ptr()), _TEST, _D)
            _ = member_call(_member(), MEMBER_OP_TRANSFORM, query)
            var inverse = MemberCall(route)
            inverse.state = state
            inverse.x = MemberMatrix(Int(forward.unsafe_ptr()), _TEST, _D)
            inverse.dst = MemberMatrix(Int(back.unsafe_ptr()), _TEST, _D)
            _ = member_call(_member(), MEMBER_OP_INVERSE_TRANSFORM, inverse)
        finally:
            var free = MemberCall(route)
            free.state = state
            _ = member_call(_member(), MEMBER_OP_FREE, free)
        outs.append(forward^)
        outs.append(back^)
        _ = len(train) + len(test) + len(x_train) + len(w_train) + len(x_test)
    _ = len(params) + len(x) + len(w)
    return outs^


def _configuration(minmax: Bool, nan_cells: Bool, nan_column: Bool, weighted: Bool, label: String) raises:
    var x = _x(nan_cells, nan_column)
    var w = _weights()
    # kind, with_mean / clip, with_std, feature_range lower, upper, numeric mode
    var params: List[Float64] = [1.0 if minmax else 0.0, 0.0 if minmax else 1.0, 1.0, -1.0 if minmax else 0.0,
                                 2.0 if minmax else 1.0, Float64(Int(GLOBAL_NUMERIC_MODE))]
    var gpu = _route(True)
    var plain = _arm(gpu, params, x, w, weighted, False, -1, 0)
    var host = _arm(_route(False), params, x, w, weighted, False, -1, 0)
    for i in range(len(plain)):  # small-loop(oracle: per fold and output)
        _same(plain[i], host[i], label + ": GPU host-input vs the host column, output " + String(i))
    var slots_before = FOLD_X_STORE.get_or_create_ptr()[].live_count()
    var scope = arena_scope_new(0)
    var scoped = List[List[Float32]]()
    try:
        scoped = _arm(gpu, params, x, w, weighted, True, scope, 1)
    finally:
        _ = arena_scope_close(scope)
    for i in range(len(plain)):  # small-loop(oracle: per fold and output)
        _same(scoped[i], plain[i], label + ": scoped fold (x_prep slot programs) vs GPU host-input, output " + String(i))
    require(FOLD_X_STORE.get_or_create_ptr()[].live_count() == slots_before,
            label + ": every x_prep copy of a fold slot is freed")
    _ = len(x) + len(w) + len(params)


def _case(unit: Int) raises:
    if unit == 0:
        _configuration(False, True, False, False, "StandardScaler NaN X")
    elif unit == 1:
        _configuration(False, False, False, True, "StandardScaler sample_weight")
    elif unit == 2:
        _configuration(True, True, False, False, "MinMaxScaler NaN X")
    else:
        _configuration(False, False, True, False, "StandardScaler all-NaN column")


def test_vdf_scope_neural6_scaler() raises:
    check_cases_serial("vdf_scope_neural6_scaler", CASES, _case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_vdf_scope_neural6_scaler]()
    suite^.run()
