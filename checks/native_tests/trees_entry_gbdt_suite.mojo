# SPDX-License-Identifier: Apache-2.0
"""The gbdt table's entry ABI, reached natively (lane
calltable-gbdt-forest-dart, 2026-11-11).

A probe image (`_ProbeImage`) implements `GbdtImage` with an echo fit, so
these cases check what the table and the ABI carry, independent of a fit:
the request (six addresses, the int/float views and their flags, the
strings), the outcome transfer (text, best iteration, stop flag, both
curves, freed in the image), the row-major capability, the tier, the model
dimension and refusals coming back in the image's words. The real fits
are the gbdt bindings' own bodies behind the same dispatch. Ordered
preconditions first (one process table): a caller refuses before
publication, then the probe publishes under the cpu key.

The cases are independent units and run across the native host worker
pool. Run in a fresh process (the first check needs an unpublished table).
"""
from core.entry_table import entry_ref_same
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.trees_entry_table import TreesEntryKey
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome, GBDT_ARG_NO_INT, GBDT_ARG_NO_FLOAT, GBDT_ENTRY_ABI, XG_OP_COUNT
from gbdt.gbdt_entry import GbdtImage, gbdt_entry_dispatch, gbdt_entry_publish
from core.gbdt_entry_table import gbdt_table_entry
from gbdt.fit_calls import xg_ref, xg_row_major, xg_numeric_mode, xg_model_dim, xg_fit

comptime TREES_GBDT_CASES = 5


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct _ProbeImage(GbdtImage):
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        if args.strs[0] == "Refuse":
            raise Error("gbdt_fit: probe refusal")
        # echo: the text joins the strings and the address words; the
        # curves are the float view and the int view; best = the slot count
        var text = String("row_major=") + String(row_major)
        for k in range(len(args.strs)):  # small-loop(the request strings)
            text += "|" + args.strs[k]
        text += "|" + String(args.x + args.y + args.weights + args.cat_flags + args.eval_x + args.eval_y)
        var learn = args.floats.copy()
        var test = List[Float64]()
        for k in range(len(args)):  # small-loop(the request slots)
            test.append(Float64(args.i(k)))
        return GbdtFitOutcome(text^, len(args), row_major, learn^, test^)

    @staticmethod
    def row_major() -> Bool:
        return True

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return text.byte_length()


def _probe_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return gbdt_entry_dispatch[_ProbeImage](operation, aa, na, pp, np, ff, nf, result, context, sink)


def _key() -> TreesEntryKey:
    return TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _args(var strs: List[String], var ints: List[Int], var floats: List[Float64], var bad: List[UInt8]) raises -> GbdtFitArgs:
    return GbdtFitArgs(1, 2, 3, 4, 5, 6, ints^, floats^, bad^, strs^)


def run_case(unit: Int) raises:
    var key = xg_ref(_key())
    if unit == 0:
        require(xg_row_major(key), "row-major capability lost")
        require(xg_numeric_mode(key) == Int(GLOBAL_NUMERIC_MODE), "tier differs")
        return
    if unit == 1:
        require(xg_model_dim(key, String("abcdé")) == 6, "model dim: text bytes not carried")
        return
    if unit == 2:
        var strs: List[String] = ["RMSE", "", "Iter", "Min"]
        var ints: List[Int] = [7, 0, 9]
        var floats: List[Float64] = [7.0, 0.25, 9.0]
        var bad: List[UInt8] = [0, 0, 0]
        var outcome = xg_fit(key, False, _args(strs^, ints^, floats^, bad^))
        require(outcome.text == "row_major=False|RMSE||Iter|Min|21", "fit text: " + outcome.text)
        require(outcome.best_iteration == 3 and not outcome.stopped_early, "fit words")
        require(len(outcome.learn) == 3 and outcome.learn[1] == 0.25, "learn curve not carried")
        return
    if unit == 3:
        var strs: List[String] = ["RMSE", "Bayesian", "IncToDec", "Max", "GreedyLogSum"]
        var ints: List[Int] = [4, 5]
        var floats: List[Float64] = [4.0, 5.5]
        var bad: List[UInt8] = [0, UInt8(GBDT_ARG_NO_FLOAT)]
        var outcome = xg_fit(key, True, _args(strs^, ints^, floats^, bad^))
        require(outcome.text == "row_major=True|RMSE|Bayesian|IncToDec|Max|GreedyLogSum|21", outcome.text)
        require(outcome.stopped_early and len(outcome.test) == 2 and outcome.test[0] == 4.0 and outcome.test[1] == 5.0,
                "test curve not carried")
        return
    # unit 4: refusals come back in the image's words
    try:
        var strs: List[String] = ["Refuse", "", "", ""]
        _ = xg_fit(key, False, _args(strs^, List[Int](), List[Float64](), List[UInt8]()))
        raise Error("a refusing fit did not refuse")
    except problem:
        require("gbdt_fit: probe refusal" in String(problem), String(problem))
    try:
        var strs: List[String] = ["RMSE", "", "", ""]
        var ints: List[Int] = [0]
        var floats: List[Float64] = [0.5]
        var bad: List[UInt8] = [UInt8(GBDT_ARG_NO_INT)]
        _ = xg_fit(key, False, _args(strs^, ints^, floats^, bad^))
        raise Error("a slot without an integer reading did not refuse")
    except problem:
        require("params[0] must be an integer" in String(problem), String(problem))


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = xg_ref(_key())
    var second = xg_ref(_key())
    require(entry_ref_same(first, second), "gbdt: a second resolution returned another entry")
    try:
        _ = gbdt_table_entry(_key(), 0, String(GBDT_ENTRY_ABI), XG_OP_COUNT + 1)
        raise Error("gbdt: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "gbdt: unexpected refusal: " + String(problem))


def run_checks() raises:
    try:
        _ = xg_numeric_mode(xg_ref(_key()))
        raise Error("an unpublished gbdt table did not refuse")
    except problem:
        require("provider is not published" in String(problem), String(problem))
    gbdt_entry_publish["MojolearnGbdtEntryOwnerProbeV1"]("cpu", _probe_entry)
    ref_resolution()
    var failures = List[Int](length=TREES_GBDT_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL trees gbdt entry case", unit, String(problem))
    host_parallelize(task, TREES_GBDT_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "trees gbdt entry cases failed: " + String(failed))
    print("PASS trees gbdt entry", TREES_GBDT_CASES)
