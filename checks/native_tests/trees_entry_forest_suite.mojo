# SPDX-License-Identifier: Apache-2.0
"""The forest table's entry ABI, reached natively (lane calltable-forest,
2026-11-11).

A probe image (`_ProbeForest`) implements `ForestImage` with echo bodies, so
these cases check what the table and the ABI carry for every operation,
independent of a fit or a kernel: the capability bits, the tier, the
default bin count, the fit request (shape bits, addresses, the int/float
views and their flags, criterion and the trailing words, the largest weight
through its word), the DART member request, the export sizes and metadata,
the copy into five arrays, the verify verdict, the release, the sequential
vote, the resident and strict snapshots, the native release the resident
table calls, and refusals coming back in the image's words. The real
bodies are the four forest bindings' own behind the same dispatch.
Ordered preconditions first (one process table): a caller refuses before
publication, then the probe publishes under the cpu keys of both families.

The cases are independent units and run across the native host worker
pool. Run in a fresh process (the first check needs an unpublished table).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.trees_entry_table import TreesEntryKey
from core.forest_entry_table import ForestEntryKey, FOREST_TABLE_RF, FOREST_TABLE_ET
from core.forest_fit_request import (
    ForestFitArgs, FOREST_ARG_NO_INT, FOREST_ARG_NO_FLOAT, XF_CAP_STRICT, XF_CAP_DART, XF_CAP_DEVICE_SCAN,
    XF_RESIDENT_RELEASE, XF_STRICT_RELEASE, forest_export_sizes_into, forest_export_verify_lists,
)
from core.forest_entry import ForestImage, forest_entry_dispatch, forest_entry_publish
from core.forest_calls import (
    xf_caps, xf_numeric_mode, xf_dt_default_bins, xf_fit, xf_fit_dart, xf_export_sizes, xf_export_into,
    xf_export_verify, xf_export_release, xf_predict, xf_resident_prepare, xf_resident_predict, xf_resident_labels,
    xf_strict_prepare, xf_strict_predict, xf_snapshot_release,
)

comptime TREES_FOREST_CASES = 12
comptime _PROBE_CAPS = XF_CAP_STRICT | XF_CAP_DART | XF_CAP_DEVICE_SCAN


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _word(address: Int, k: Int, value: Int):
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)[k] = value


struct _ProbeForest(ForestImage):
    """Echo bodies: each answer is a checkable function of the request."""

    @staticmethod
    def caps() -> Int:
        return _PROBE_CAPS

    @staticmethod
    def dt_default_bins() -> Int:
        return 256

    @staticmethod
    def fit(classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
        if args.criterion == 99:
            raise Error("rf_classifier_fit: probe refusal")
        # handle = shape bits + 10 * (x + y + weights + tree_start + oob + score + importance + criterion
        #          + int view sum + int(weight_max) + int(float view sum))
        var total = args.x + args.y + args.weights + args.tree_start + args.oob + args.score + args.importance
        total += args.criterion + Int(args.weight_max)
        for k in range(len(args)):  # small-loop(the request slots)
            total += args.i(k) + Int(args.f(k))
        return (1 if classifier else 0) + (2 if row_major else 0) + 10 * total

    @staticmethod
    def fit_dart(args: ForestFitArgs) raises -> Int:
        return args.session * 1000 + args.cls * 100 + args.x + args.y + args.criterion

    @staticmethod
    def export_sizes(handle: Int, out_addr: Int) raises:
        var meta: List[Int64] = [Int64(handle), Int64(3), Int64(7)]
        forest_export_sizes_into(out_addr, handle, handle * 2, 3, meta)

    @staticmethod
    def export_into(handle: Int, dst: List[Int], trees: Int, nodes: Int, outputs: Int) raises:
        if trees != handle or nodes != handle * 2 or outputs != 3:
            raise Error("forest_export: probe sizes differ")
        for k in range(5):  # small-loop(the five model arrays)
            _word(dst[k], 0, handle + k)

    @staticmethod
    def export_verify(handle: Int, dst: List[Int]) raises -> Int:
        var offsets: List[Int64] = [1, 2]
        var colid: List[Int64] = [3]
        var quesval: List[Float64] = [0.5]
        var left: List[Int64] = [4]
        var leaves: List[Float64] = [1.25, -2.0]
        return forest_export_verify_lists(offsets, colid, quesval, left, leaves, dst)

    @staticmethod
    def export_release(handle: Int) raises:
        if handle != 5:
            raise Error("unknown or released fitted forest export handle")

    @staticmethod
    def predict(model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int, outputs: Int,
                regression: Bool) raises -> Int:
        _word(out_addr, 0, model[0] + model[1] + model[2] + model[3] + model[4] + x + cols + trees + outputs
              + (1000 if regression else 0))
        return rows

    @staticmethod
    def resident_prepare(model: List[Int], features: Int, trees: Int, outputs: Int, ordered: Int) raises -> Int:
        return model[4] + features + trees + outputs + 100 * (ordered + 1)

    @staticmethod
    def resident_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        _word(out_addr, 0, handle + x + cols + outputs)
        return rows

    @staticmethod
    def resident_labels(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        _word(out_addr, 0, -(handle + x + cols + outputs))
        return rows

    @staticmethod
    def resident_release(handle: Int) raises:
        if handle != 41:
            raise Error("unknown or released resident forest handle")

    @staticmethod
    def strict_prepare(model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
        return model[0] * features + trees * outputs

    @staticmethod
    def strict_predict(handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int, outputs: Int) raises -> Int:
        _word(out_addr, 0, handle * 2 + x)
        return rows + cols + outputs

    @staticmethod
    def strict_release(handle: Int) raises:
        if handle != 43:
            raise Error("unknown or released strict forest handle")


def _probe_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return forest_entry_dispatch[_ProbeForest](operation, aa, na, pp, np, ff, nf, result, context, sink)


def _key(family: Int) -> ForestEntryKey:
    return ForestEntryKey(family, TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _args(var ints: List[Int], var floats: List[Float64], var bad: List[UInt8], criterion: Int) raises -> ForestFitArgs:
    return ForestFitArgs(1, 2, ints^, floats^, bad^, criterion)


def case(unit: Int) raises:
    var rf = _key(FOREST_TABLE_RF)
    var et = _key(FOREST_TABLE_ET)
    if unit == 0:
        # XF_CAPS, XF_NUMERIC_MODE, XF_DT_DEFAULT_BINS in both families
        require(xf_caps(rf) == _PROBE_CAPS and xf_caps(et) == _PROBE_CAPS, "capability bits lost")
        require(xf_numeric_mode(rf) == Int(GLOBAL_NUMERIC_MODE), "tier differs")
        require(xf_dt_default_bins(et) == 256, "default bins lost")
        return
    if unit == 1:
        # XF_FIT: shape bits, addresses, both views, criterion
        var ints: List[Int] = [7, 0, 9]
        var floats: List[Float64] = [7.0, 0.25, 9.0]
        var bad: List[UInt8] = [0, 0, 0]
        var handle = xf_fit(rf, True, False, _args(ints^, floats^, bad^, 1))
        # 1 + 2 + criterion 1 + ints 16 + floats int-part 16 = 36
        require(handle == 1 + 10 * 36, "fit request: " + String(handle))
        return
    if unit == 2:
        # XF_FIT: the trailing words and the largest weight's word
        var ints: List[Int] = [4]
        var floats: List[Float64] = [4.5]
        var bad: List[UInt8] = [0]
        var args = _args(ints^, floats^, bad^, -1)
        args.weights = 100
        args.tree_start = 20
        args.oob = 3000
        args.score = 40000
        args.importance = 500000
        args.weight_max = 6000000.75
        var handle = xf_fit(et, False, True, args)
        var total = 1 + 2 + 100 + 20 + 3000 + 40000 + 500000 - 1 + 6000000 + 4 + 4
        require(handle == 2 + 10 * total, "fit trailing words: " + String(handle))
        return
    if unit == 3:
        # the int/float view flags refuse a missing reading in the image's words
        try:
            var ints: List[Int] = [0]
            var floats: List[Float64] = [0.5]
            var bad: List[UInt8] = [UInt8(FOREST_ARG_NO_INT)]
            _ = xf_fit(rf, True, False, _args(ints^, floats^, bad^, 0))
            raise Error("a slot without an integer reading did not refuse")
        except problem:
            require("params[0] must be an integer" in String(problem), String(problem))
        try:
            var ints: List[Int] = [3]
            var floats: List[Float64] = [0.0]
            var bad: List[UInt8] = [UInt8(FOREST_ARG_NO_FLOAT)]
            _ = xf_fit(rf, True, False, _args(ints^, floats^, bad^, 0))
            raise Error("a slot without a float reading did not refuse")
        except problem:
            require("params[0] must be a number" in String(problem), String(problem))
        try:
            _ = xf_fit(rf, True, False, _args(List[Int](), List[Float64](), List[UInt8](), 99))
            raise Error("a refusing fit did not refuse")
        except problem:
            require("rf_classifier_fit: probe refusal" in String(problem), String(problem))
        return
    if unit == 4:
        # XF_FIT_DART: session, class and the bag addresses
        var ints: List[Int] = [1]
        var floats: List[Float64] = [1.0]
        var bad: List[UInt8] = [0]
        var args = _args(ints^, floats^, bad^, 2)
        args.session = 7
        args.cls = 3
        require(xf_fit_dart(rf, args) == 7000 + 300 + 1 + 2 + 2, "dart request lost")
        return
    if unit == 5:
        # XF_EXPORT_SIZES and XF_EXPORT_INTO
        var sizes = xf_export_sizes(rf, 5)
        require(sizes.trees == 5 and sizes.nodes == 10 and sizes.outputs == 3, "export sizes lost")
        require(len(sizes.meta) == 3 and sizes.meta[0] == 5 and sizes.meta[2] == 7, "export metadata lost")
        var cells = List[Int](length=5, fill=0)
        var dst = List[Int]()
        for k in range(5):  # small-loop(the five model arrays): one word each
            dst.append(Int(cells.unsafe_ptr()) + 8 * k)
        xf_export_into(rf, sizes, dst)
        for k in range(5):  # small-loop(the five model arrays)
            require(cells[k] == 5 + k, "export into array " + String(k))
        _ = cells
        return
    if unit == 6:
        # XF_EXPORT_VERIFY: equal, then a differing float array, through the image
        var offsets: List[Int32] = [1, 2]
        var colid: List[Int32] = [3]
        var quesval: List[Float32] = [0.5]
        var left: List[Int32] = [4]
        var leaves: List[Float32] = [1.25, -2.0]
        var dst: List[Int] = [Int(offsets.unsafe_ptr()), Int(colid.unsafe_ptr()), Int(quesval.unsafe_ptr()),
                              Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
        require(xf_export_verify(et, 5, dst) == 0, "equal export did not verify")
        leaves[1] = -2.5
        require(xf_export_verify(et, 5, dst) == 5, "a differing leaves array was not named")
        _ = offsets
        _ = colid
        _ = quesval
        _ = left
        _ = leaves
        return
    if unit == 7:
        # XF_EXPORT_RELEASE and its refusal
        xf_export_release(rf, 5)
        try:
            xf_export_release(rf, 6)
            raise Error("an unknown export handle did not refuse")
        except problem:
            require("unknown or released fitted forest export handle" in String(problem), String(problem))
        return
    if unit == 8:
        # XF_PREDICT: model addresses, X, out, dims and the regression flag
        var out = List[Int](length=1, fill=0)
        var model: List[Int] = [1, 2, 3, 4, 5]
        var rows = xf_predict(rf, model, 10, Int(out.unsafe_ptr()), 6, 20, 30, 40, True)
        require(rows == 6 and out[0] == 15 + 10 + 20 + 30 + 40 + 1000, "sequential vote request lost")
        _ = out
        return
    if unit == 9:
        # XF_RESIDENT_PREPARE / PREDICT / LABELS, the ordered word
        var model: List[Int] = [1, 2, 3, 4, 5]
        require(xf_resident_prepare(et, model, 6, 7, 8, -1) == 5 + 6 + 7 + 8, "default ordered word lost")
        require(xf_resident_prepare(et, model, 6, 7, 8, 1) == 5 + 6 + 7 + 8 + 200, "ordered word lost")
        var out = List[Int](length=1, fill=0)
        require(xf_resident_predict(et, 41, 9, Int(out.unsafe_ptr()), 4, 3, 2) == 4 and out[0] == 41 + 9 + 3 + 2,
                "resident predict request lost")
        require(xf_resident_labels(et, 41, 9, Int(out.unsafe_ptr()), 4, 3, 2) == 4 and out[0] == -(41 + 9 + 3 + 2),
                "resident labels request lost")
        _ = out
        return
    if unit == 10:
        # XF_STRICT_PREPARE / PREDICT
        var model: List[Int] = [3, 0, 0, 0, 0]
        require(xf_strict_prepare(rf, model, 5, 2, 7) == 15 + 14, "strict prepare request lost")
        var out = List[Int](length=1, fill=0)
        require(xf_strict_predict(rf, 43, 1, Int(out.unsafe_ptr()), 4, 5, 6) == 15 and out[0] == 87,
                "strict predict request lost")
        _ = out
        return
    # unit 11: the releases the resident table calls (ints [handle]) and their refusals
    xf_snapshot_release(rf, XF_RESIDENT_RELEASE, 41)
    xf_snapshot_release(rf, XF_STRICT_RELEASE, 43)
    try:
        xf_snapshot_release(rf, XF_STRICT_RELEASE, 44)
        raise Error("an unknown strict handle did not refuse")
    except problem:
        require("unknown or released strict forest handle" in String(problem), String(problem))


def run_checks() raises:
    try:
        _ = xf_caps(_key(FOREST_TABLE_RF))
        raise Error("an unpublished forest table did not refuse")
    except problem:
        require("provider is not published" in String(problem), String(problem))
    forest_entry_publish["MojolearnForestEntryOwnerProbeRfV1"](FOREST_TABLE_RF, "cpu", _probe_entry)
    forest_entry_publish["MojolearnForestEntryOwnerProbeEtV1"](FOREST_TABLE_ET, "cpu", _probe_entry)
    var failures = List[Int](length=TREES_FOREST_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL trees forest entry case", unit, String(problem))
    host_parallelize(task, TREES_FOREST_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "trees forest entry cases failed: " + String(failed))
    print("PASS trees forest entry", TREES_FOREST_CASES)
