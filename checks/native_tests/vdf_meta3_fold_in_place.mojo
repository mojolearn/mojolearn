# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE FOLD IN PLACE of the meta-estimators' public doors (lane vdf-meta3, 2026-11-24; core/meta_fold_rows.mojo,
bindings/trees_public_meta_door.mojo `_meta_x_fit`, `_meta_take`, `_meta_compose`, `_fit_units_drawn`,
`_stack_place_indexed`), in a process with no Python.

A model-selection fold (`DeviceRows`: a host base and int64 row numbers) reaches VotingClassifier / Regressor,
StackingClassifier / Regressor, OneVsRestClassifier and MultiOutputClassifier.fit. Each member then reads the WHOLE
base with a row draw: the fold's rows (Voting, OvR, MultiOutput) or a cv split of the fold composed inside it
(Stacking's inner folds; CalibratedClassifierCV and Stacking over a host X read their cv splits as draws of X the
same way). The member's targets and weights are the fold's own words put at the base rows. A member whose kind
reads a row index takes the drawn rows ON THE DEVICE out of the one resident base (core/member_fold_x.mojo); before,
the door gathered the fold on the host and every member uploaded its copy. The device-row path must give the bytes
of that host-row path, which is the CPU column's input.

Host column (CPU_COLUMN_BUILD):
    case 0   `meta_rows_index`: the int32 draw == the fold's rows; `inv` names, for every base row in the fold, a
             fold position holding that row, and 0 elsewhere; a row number outside the base (-1, then base_rows)
             is refused in META_ROWS_OUTSIDE's words. 40000 base rows (several task chunks), a fold with a repeat
    case 1   `meta_rows_compose` (an inner split inside the fold) == the oracle `draw[inner[i]]`, and
             `meta_rows_take_u32` over int32 codes and float32 targets: the word a draw reads at every base row of
             the composed draw == the fold's own word at that inner position (bitwise)

Device column (the same bytes as the host-row path):
    case 0   the Gram Ridge fit (the linear member's fold entry body) over the base put once and the COMPOSED rows
             (outer fold, inner split) taken on the device, with the targets read through the draw out of the
             base-row words (`meta_rows_take_u32`) == the same fit over the host double gather (the fold gathered
             out of the base, then its inner split) with the fold's own targets: coef, column means and the binary64
             y mean bit for bit
    case 1   the prediction entry body over the composed device rows == over the host double gather
The host-gather fit is the estimator's own host-input route, whose device and CPU columns the release's one
verification compares; equal bytes here put the in-place route in that same column.

FAIL-PROOF EDITS (each must fail the named case): in `meta_rows_index` write `v[r] = Int64(i + 1)` (host 1, device
0); drop the zero pass (host 0); test `r > base_rows` (host 0, the refusal of base_rows); in `meta_rows_compose` read
`w[Int(s[i]) + 1]` (host 1, device 0 and 1); in `meta_rows_take_u32` read `s[b]` (host 1, device 0).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, CPU_COLUMN_BUILD, require
from core.meta_fold_rows import meta_rows_index, meta_rows_compose, meta_rows_take_u32, META_ROWS_OUTSIDE
from max.gpu.host import DeviceContext
from core.gram_multi_gpu import gram_plan_current
from core.neural_context import process_ctx
from glm.impl.gram_solve import linear_gram_fit_host
from core.p3_linear_impl import ols_predict_device
from core.fold_x_device import FOLD_X_STORE, fold_x_word

comptime NB = 40000
comptime FD = 4
comptime DN = 900


def _outer(base_rows: Int) -> List[Int64]:
    """The fold: every third row from the top down, then a repeat of row 7 and the last row."""
    var rows = List[Int64]()
    for i in range(base_rows - 1, -1, -3):  # small-loop(fixture: fold rows)
        rows.append(Int64(i))
    rows.append(Int64(7))
    rows.append(Int64(7))
    rows.append(Int64(base_rows - 1))
    return rows^


def _inner(n: Int) -> List[Int32]:
    """A cv split of the fold: its positions 1, 3, 5, ... then 0 (out of order)."""
    var rows = List[Int32]()
    for i in range(1, n, 2):  # small-loop(fixture: split positions)
        rows.append(Int32(i))
    rows.append(Int32(0))
    return rows^


def _x(n: Int, d: Int) -> List[Float32]:
    var x = List[Float32](length=n * d, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for j in range(d):  # small-loop(fixture columns)
            x[i * d + j] = Float32(((i * (j + 3) * 7) % 23) - 11) * Float32(0.25) + Float32(j)
    return x^


def _index(outer: List[Int64], base_rows: Int) raises -> Tuple[List[Int32], List[Int64]]:
    var n = len(outer)
    var draw = List[Int32](length=n, fill=Int32(-5))
    var inv = List[Int64](length=base_rows, fill=Int64(-9))
    meta_rows_index(Int(outer.unsafe_ptr()), n, base_rows, Int(draw.unsafe_ptr()), Int(inv.unsafe_ptr()))
    return (draw^, inv^)


def _host_case(case: Int) raises:
    var outer = _outer(NB)
    var n = len(outer)
    var got = _index(outer, NB)
    var draw = got[0].copy()
    var inv = got[1].copy()
    if case == 0:
        var in_fold = List[UInt8](length=NB, fill=UInt8(0))
        for i in range(n):  # small-loop(oracle: the draw and the rows in the fold)
            require(Int(draw[i]) == Int(outer[i]), "draw word " + String(i) + " is not the fold's row")
            in_fold[Int(outer[i])] = UInt8(1)
        for b in range(NB):  # small-loop(oracle: the inverse map)
            var p = Int(inv[b])
            if in_fold[b] != UInt8(0):
                require(p >= 0 and p < n and Int(outer[p]) == b, "inv of base row " + String(b) + " is wrong")
            else:
                require(p == 0, "inv of base row " + String(b) + " outside the fold is not 0")
        for edge in range(2):  # small-loop(fixture: -1, then base_rows)
            var bad = outer.copy()
            bad[3] = Int64(-1) if edge == 0 else Int64(NB)
            var refused = False
            try:
                _ = _index(bad, NB)
            except error:
                refused = String(error) == META_ROWS_OUTSIDE
            require(refused, "a row outside the base was not refused in its words")
        return
    # case 1: compose and take
    var inner = _inner(n)
    var m = len(inner)
    var composed = List[Int32](length=m, fill=Int32(-1))
    meta_rows_compose(Int(inner.unsafe_ptr()), m, Int(draw.unsafe_ptr()), Int(composed.unsafe_ptr()))
    var codes = List[Int32](length=n, fill=Int32(0))
    var target = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(fixture: the fold's own words, a function of the base row)
        codes[i] = Int32(Int(outer[i]) % 5)
        target[i] = Float32(Int(outer[i]) % 13) * Float32(0.375) - Float32(2.0)
    var codes_base = List[Int32](length=NB, fill=Int32(-7))
    var target_base = List[Float32](length=NB, fill=0.0)
    meta_rows_take_u32(Int(codes.unsafe_ptr()), Int(inv.unsafe_ptr()), NB, Int(codes_base.unsafe_ptr()))
    meta_rows_take_u32(Int(target.unsafe_ptr()), Int(inv.unsafe_ptr()), NB, Int(target_base.unsafe_ptr()))
    for i in range(m):  # small-loop(oracle: the composed draw and the words it reads)
        var p = Int(inner[i])
        require(Int(composed[i]) == Int(outer[p]), "composed row " + String(i) + " differs")
        var b = Int(composed[i])
        require(codes_base[b] == codes[p], "code read at composed row " + String(i) + " differs")
        require(target_base[b].to_bits() == target[p].to_bits(), "target read at composed row " + String(i)
                + " differs")


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _host_case(case)

        check_cases_serial("vdf_meta3_fold_in_place host", 2, body)


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(word compare)
        require(a[i].to_bits() == b[i].to_bits(), label + ": word " + String(i) + " differs")


def _device_case(case: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        comptime _CTX = "MojolearnEstimatorsContextIdenticalV3"
        var x = _x(DN, FD)
        var outer = _outer(DN)
        var n = len(outer)
        var got = _index(outer, DN)
        var draw = got[0].copy()
        var inv = got[1].copy()
        var inner = _inner(n)
        var m = len(inner)
        var composed = List[Int32](length=m, fill=Int32(0))
        meta_rows_compose(Int(inner.unsafe_ptr()), m, Int(draw.unsafe_ptr()), Int(composed.unsafe_ptr()))
        var composed64 = List[Int64](length=m, fill=Int64(0))
        for i in range(m):  # small-loop(fixture: the store reads int64 row numbers)
            composed64[i] = Int64(composed[i])
        # The host-row path: the fold gathered out of the base, then its inner split, with the fold's targets.
        var fold_x = List[Float32](length=n * FD, fill=0.0)
        var y_fold = List[Float32](length=n, fill=0.0)
        for r in range(n):  # small-loop(oracle: the door's former fold gather)
            y_fold[r] = Float32(Int(outer[r]) % 17) * Float32(0.5) - Float32(3.0)
            for c in range(FD):  # small-loop(oracle gather)
                fold_x[r * FD + c] = x[Int(outer[r]) * FD + c]
        var host_x = List[Float32](length=m * FD, fill=0.0)
        var host_y = List[Float32](length=m, fill=0.0)
        for r in range(m):  # small-loop(oracle: the member's former gather of its split)
            var p = Int(inner[r])
            host_y[r] = y_fold[p]
            for c in range(FD):  # small-loop(oracle gather)
                host_x[r * FD + c] = fold_x[p * FD + c]
        # The device-row path: the fold's targets at the base rows, read through the composed draw.
        var y_base = List[Float32](length=DN, fill=0.0)
        meta_rows_take_u32(Int(y_fold.unsafe_ptr()), Int(inv.unsafe_ptr()), DN, Int(y_base.unsafe_ptr()))
        var dev_y = List[Float32](length=m, fill=0.0)
        for r in range(m):  # small-loop(fixture: the member set's target selection by the draw)
            dev_y[r] = y_base[Int(composed[r])]
        var store_ctx = DeviceContext()
        var store = FOLD_X_STORE.get_or_create_ptr()
        var base = store[].put(store_ctx, Int(x.unsafe_ptr()), DN * FD)
        var slot = store[].take_rows(store_ctx, base, FD, Int(composed64.unsafe_ptr()), m)
        var ctx = process_ctx[_CTX]()
        var plan = gram_plan_current()
        var failure = String("")
        try:
            if case == 0:
                var coef_a = List[Float32](length=FD, fill=0.0)
                var mean_a = List[Float32](length=FD, fill=0.0)
                var ym_a = List[Float64](length=1, fill=0.0)
                var coef_b = List[Float32](length=FD, fill=0.0)
                var mean_b = List[Float32](length=FD, fill=0.0)
                var ym_b = List[Float64](length=1, fill=0.0)
                var sa = linear_gram_fit_host(plan, ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=fold_x_word(slot)),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(dev_y.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef_a.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(mean_a.unsafe_ptr())),
                    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(ym_a.unsafe_ptr())),
                    m, FD, Float32(0.5), True)
                var sb = linear_gram_fit_host(plan, ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(host_x.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(host_y.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef_b.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(mean_b.unsafe_ptr())),
                    MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(ym_b.unsafe_ptr())),
                    m, FD, Float32(0.5), True)
                require(sa == sb, "Gram status differs")
                _same(coef_a, coef_b, "coef")
                _same(mean_a, mean_b, "column means")
                require(ym_a[0].to_bits() == ym_b[0].to_bits(), "y mean differs")
            else:
                var coef: List[Float32] = [Float32(0.25), Float32(-1.5), Float32(2.0), Float32(0.125)]
                var out_a = List[Float32](length=m, fill=0.0)
                var out_b = List[Float32](length=m, fill=0.0)
                var ra = ols_predict_device(ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=fold_x_word(slot)),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef.unsafe_ptr())),
                    Int(out_a.unsafe_ptr()), m, FD, Float32(0.75), True)
                var rb = ols_predict_device(ctx,
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(host_x.unsafe_ptr())),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef.unsafe_ptr())),
                    Int(out_b.unsafe_ptr()), m, FD, Float32(0.75), True)
                require(ra == rb and ra == 0, "finite status differs")
                _same(out_a, out_b, "predictions")
        except error:
            failure = String(error)
        store[].free(store_ctx, slot)
        store[].free(store_ctx, base)
        _ = x^
        _ = composed64^
        _ = host_x^
        _ = host_y^
        _ = dev_y^
        _ = y_base^
        _ = inv^
        if failure.byte_length() > 0:
            raise Error(failure)


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        def body(case: Int) raises:
            _device_case(case)

        check_cases_serial("vdf_meta3_fold_in_place device", 2, body)


def main() raises:
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
