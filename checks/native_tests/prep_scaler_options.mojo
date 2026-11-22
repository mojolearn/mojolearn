# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""StandardScaler and MinMaxScaler options against scikit-learn: NaN in fit and transform, StandardScaler
sample_weight, partial_fit for both, an all-NaN feature (lane py2mojo-A, 2026-11-22; ported from
python/mojolearn/tests/test_x_prep_scaler_options.py `test_standard_nan_fit_and_transform`,
`test_standard_sample_weight`, `test_standard_partial_fit`, `test_standard_all_nan_feature`,
`test_minmax_nan_and_partial_fit`).

Fits and transforms run the scaler member (core/scaler_member.mojo) over the host column's kernels and x_prep
transport (checks/native_tests/py2mojo_host_members.mojo `HostMember`). StandardScaler.partial_fit runs its typed RUN,
bindings/scaler_lifecycle_run.mojo `scaler_standard_partial_fit_run` (the batch fit core and the merge program the door
calls), over the same entries. MinMaxScaler.partial_fit has no typed RUN: `minmax_partial` below repeats the door's
composition (bindings/minmax_native_door.mojo `minmax_partial_fit`: the batch fit core, stage 268 over the old and new
extrema, the kernel fit of the 4 x d extrema block, `scaler_minmax_keep_extrema` with the column NaN flags).

The inputs are the original's `_data` (257 x 5, seed 3, a constant 2.5 column, NaN every 7th row of feature 0 and
every 11th from row 1 of feature 2; 600 rows for partial_fit) and the weights it drew, with scikit-learn 1.9.1's
statistics, in `fixtures/prep_scalers/` (tools/py2mojo_native_fixtures_b1.py). Tolerance rtol = atol = 2e-5;
n_samples_seen_ exact for counts, rtol 1e-6 for weight sums.

    unit 0      NaN fit: mean_, var_, scale_, per-feature n_samples_seen_, the constant column's var 0 / scale 1;
                transform and inverse_transform keep the NaN cells, finite cells equal the NaN-free transform's bits
    units 1-2   sample_weight on X and on the NaN X (zero weights every 11th row)
    unit 3      unit weights give the unweighted mean_; the scalar weight 2.0 (broadcast at UNPACK) the same
    units 4-7   partial_fit over rows 0-100, 100-350, 350-600: X, NaN X, weighted X, weighted NaN X
    unit 8      the first partial_fit batch is fit, bit for bit (both run the batch fit core)
    unit 9      with_std=False partial_fit: mean_ is NumPy's; neither flag: the count is 600
    unit 10     an all-NaN feature: mean_ and scale_ NaN, count 0, its transform column NaN and no other; a later
                batch that sees it takes the batch's mean_
    units 11-12 MinMaxScaler(feature_range=(-1, 2)) NaN fit, clip False / True: the five rows and the transform
    unit 13     the NaN fit's extrema equal the fit of X with NaN filled by the column minimum, bit for bit
    unit 14     MinMax partial_fit: three batches equal one fit of the whole X, bit for bit; n_samples_seen_ 257
    unit 15     MinMax partial_fit with an all-NaN column poisons its scale_ (NaN), as np.minimum does

copy=False, the inf refusal, save / load and the None attributes stay in the Python file (public boundary).
Every unit is independent and the case table runs across host cores.
"""
from std.math import isnan
from std.memory import bitcast
from core.member_abi import MEMBER_OP_INVERSE_TRANSFORM
from core.scaler_core import (
    ScalerTableCalls, ScalerStandardFit, scaler_standard_fit_core, scaler_minmax_fit_core, scaler_minmax_keep_extrema,
    SCALER_SEEN_INT, SCALER_SEEN_FLOAT, SCALER_SEEN_INT_ROW, SCALER_SEEN_FLOAT_ROW,
)
from core.scaler_native_api import MinMaxFitRequest
from core.preprocessing_program import prep_program_stage
from bindings.scaler_lifecycle_run import (
    ScalerSeenView, ScalerStandardMerged, scaler_standard_partial_fit_run,
    SCALER_SEEN_SOURCE_SCALAR, SCALER_SEEN_SOURCE_I64, SCALER_SEEN_SOURCE_F32,
)
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import (
    HostMember, MEMBER_SCALER, PREP_SCALERS_DIR, host_route, all_close, rows_close, same_bits, require, take, take32,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime N = 257
comptime D = 5
comptime RT = 2e-5
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def calls() -> ScalerTableCalls:
    var route = host_route()
    return ScalerTableCalls(route.entry, route.second, False)


def std_params(with_mean: Bool, with_std: Bool) -> List[Float64]:
    var words: List[Float64] = [0.0, 1.0 if with_mean else 0.0, 1.0 if with_std else 0.0, 0.0, 0.0, 1.0]
    return words^


def mm_params(clip: Bool) -> List[Float64]:
    var words: List[Float64] = [1.0, 1.0 if clip else 0.0, 0.0, -1.0, 2.0, 1.0]
    return words^


def fixture(name: String, count: Int) raises -> List[Float32]:
    return fixture_f32(PREP_SCALERS_DIR + name, count)


def reference(name: String, count: Int) raises -> List[Float64]:
    return fixture_f64(PREP_SCALERS_DIR + name, count)


# ---- a StandardScaler fitted state as the door keeps it (rows and n_samples_seen_)
struct StdState(Movable):
    var stats: List[Float32]
    var kind: Int
    var scalar: Float64
    var i64: List[Int64]
    var f32: List[Float32]

    def __init__(out self, fit: ScalerStandardFit, d: Int):
        self.stats = fit.stats.copy()
        self.kind = fit.seen_kind
        self.scalar = fit.seen_scalar
        self.i64 = List[Int64]()
        self.f32 = List[Float32]()
        if fit.seen_kind == SCALER_SEEN_INT_ROW:
            self.i64 = List[Int64](length=d, fill=0)
            for j in range(d):  # small-loop(fixture: the published int64 counts)
                self.i64[j] = Int64(fit.seen[j])
        elif fit.seen_kind == SCALER_SEEN_FLOAT_ROW:
            self.f32 = fit.seen.copy()

    def merge(mut self, merged: ScalerStandardMerged):
        if merged.kept:
            self.stats = merged.stats.copy()
        self.kind = merged.seen_kind
        self.scalar = merged.seen_scalar if merged.seen_kind == SCALER_SEEN_FLOAT else Float64(merged.seen_integer)
        self.i64 = merged.seen_i64.copy()
        self.f32 = merged.seen_f32.copy()

    def view(self) -> ScalerSeenView:
        if self.kind == SCALER_SEEN_INT:
            return ScalerSeenView(SCALER_SEEN_SOURCE_SCALAR, self.scalar, Int64(Int(self.scalar)), True, 0)
        if self.kind == SCALER_SEEN_FLOAT:
            return ScalerSeenView(SCALER_SEEN_SOURCE_SCALAR, self.scalar, Int64(0), False, 0)
        if self.kind == SCALER_SEEN_INT_ROW:
            return ScalerSeenView(SCALER_SEEN_SOURCE_I64, 0, Int64(0), False, Int(self.i64.unsafe_ptr()))
        return ScalerSeenView(SCALER_SEEN_SOURCE_F32, 0, Int64(0), False, Int(self.f32.unsafe_ptr()))

    def seen(self, j: Int) -> Float64:
        if self.kind == SCALER_SEEN_INT_ROW:
            return Float64(self.i64[j])
        if self.kind == SCALER_SEEN_FLOAT_ROW:
            return Float64(self.f32[j])
        return self.scalar


def std_fit(x: List[Float32], n: Int, d: Int, with_mean: Bool, with_std: Bool, w: List[Float32]) raises -> StdState:
    var fit = scaler_standard_fit_core(calls(), F32P(unsafe_from_address=Int(x.unsafe_ptr())), n, d, with_mean,
        with_std, Int(w.unsafe_ptr()) if len(w) > 0 else 0)
    var state = StdState(fit, d)
    _ = fit^
    _ = x
    _ = w
    return state^


def std_partial(mut state: StdState, x: List[Float32], n: Int, d: Int, with_mean: Bool, with_std: Bool,
                w: List[Float32]) raises:
    var kept = with_mean or with_std
    var mean = Int(state.stats.unsafe_ptr()) if kept else 0
    var var_ = mean + 4 * d if with_std else 0
    var merged = scaler_standard_partial_fit_run(calls(), Int(x.unsafe_ptr()), n, d, d, with_mean, with_std,
        Int(w.unsafe_ptr()) if len(w) > 0 else 0, state.view(), mean, var_)
    state.merge(merged)
    _ = merged^
    _ = x
    _ = w


def std_compare(stats: List[Float32], seen: StdState, want: List[Float64], d: Int, counts: Bool, label: String) raises:
    rows_close(take32(stats, 0, 3 * d), take(want, 0, 3 * d), RT, RT, label + ": mean_, var_, scale_")
    for j in range(d):  # small-loop(oracle: n_samples_seen_ per feature)
        if counts:
            require(seen.seen(j) == want[3 * d + j], label + ": n_samples_seen_ feature " + String(j) + " "
                + String(seen.seen(j)) + " expected " + String(want[3 * d + j]))
        else:
            all_close(Float32(seen.seen(j)), want[3 * d + j], 1e-6, 0.0, label + ": n_samples_seen_ feature "
                + String(j))


def member_state(member: HostMember, d: Int) raises -> StdState:
    """The member's fitted blocks as the door's fields (rows, n_samples_seen_)."""
    var fit = ScalerStandardFit(d)
    fit.stats = member.block(1)
    var scalars = member.scalars()
    fit.seen_kind = Int(scalars[1])
    fit.seen_scalar = scalars[2]
    if fit.seen_kind == SCALER_SEEN_INT_ROW or fit.seen_kind == SCALER_SEEN_FLOAT_ROW:
        fit.seen = member.block(2)
    var state = StdState(fit, d)
    _ = fit^
    return state^


def slice_rows(x: List[Float32], first: Int, last: Int, d: Int) -> List[Float32]:
    return take32(x, first * d, (last - first) * d)


# ---- the units
def nan_case() raises:
    var x = fixture("opt-x.f32", N * D)
    var xn = fixture("opt-xn.f32", N * D)
    var member = HostMember(MEMBER_SCALER, std_params(True, True))
    try:
        member.fit(xn, N, D)
        var state = member_state(member, D)
        std_compare(state.stats, state, reference("opt-nan-stats.f64", 4 * D), D, True, "NaN fit")
        require(state.stats[D + 3] == 0.0 and state.stats[2 * D + 3] == 1.0, "NaN fit: constant column var/scale")
        var t = member.transform(xn, N)
        rows_close(t, reference("opt-nan-transform.f64", N * D), RT, RT, "NaN transform")
        var back = member.apply(MEMBER_OP_INVERSE_TRANSFORM, t, N)
        var clean = member.transform(x, N)
        for i in range(N * D):  # small-loop(oracle: the NaN cells and the finite cells' bits)
            require(isnan(back[i]) == isnan(xn[i]), "NaN inverse_transform: NaN pattern word " + String(i))
            if not isnan(xn[i]):
                require(bitcast[DType.uint32](t[i]) == bitcast[DType.uint32](clean[i]),
                    "NaN transform: finite word " + String(i) + " " + String(t[i]) + " NaN-free " + String(clean[i]))
    finally:
        member.free()


def weight_case(k: Int) raises:
    var data = fixture("opt-x.f32" if k == 0 else "opt-xn.f32", N * D)
    var w = fixture("opt-w.f32", N)
    var member = HostMember(MEMBER_SCALER, std_params(True, True))
    try:
        member.fit_with(data, N, D, w)
        var state = member_state(member, D)
        var label = String("sample_weight on ") + ("X" if k == 0 else "NaN X")
        std_compare(state.stats, state, reference("opt-w" + String(k) + "-stats.f64", 4 * D), D, False, label)
        require(state.stats[2 * D + 3] == 1.0, label + ": constant column scale_ " + String(state.stats[2 * D + 3]))
    finally:
        member.free()


def unit_weight_case() raises:
    var x = fixture("opt-x.f32", N * D)
    var ones = List[Float32](length=N, fill=1.0)
    var twos = List[Float32](length=N, fill=2.0)
    var a = HostMember(MEMBER_SCALER, std_params(True, True))
    var b = HostMember(MEMBER_SCALER, std_params(True, True))
    try:
        a.fit_with(x, N, D, ones)
        b.fit_with(x, N, D, twos)
        var unit = take32(a.block(1), 0, D)
        rows_close(unit, reference("opt-unit-mean.f64", D), RT, RT, "unit weights mean_")
        var wide = List[Float64](length=D, fill=0)
        for j in range(D):  # small-loop(oracle: the unit-weight mean)
            wide[j] = Float64(unit[j])
        rows_close(take32(b.block(1), 0, D), wide, RT, RT, "scalar weight 2.0 mean_")
    finally:
        a.free()
        b.free()


def partial_case(k: Int) raises:
    var n = 600
    var data = fixture("opt-p.f32" if k % 2 == 0 else "opt-pn.f32", n * D)
    var pw = fixture("opt-pw.f32", n)
    var weighted = k >= 2
    var bounds: List[Int] = [0, 100, 350, 600]
    var none = List[Float32]()
    var state = std_fit(slice_rows(data, 0, 100, D), 100, D, True, True, take32(pw, 0, 100) if weighted else none.copy())
    for s in range(1, 3):  # small-loop(oracle: the two later batches, in order)
        var a = bounds[s]
        var b = bounds[s + 1]
        std_partial(state, slice_rows(data, a, b, D), b - a, D, True, True,
            take32(pw, a, b - a) if weighted else none.copy())
    std_compare(state.stats, state, reference("opt-p" + String(k) + "-stats.f64", 4 * D), D, not weighted,
        String("partial_fit case ") + String(k))


def first_batch_case() raises:
    var xn = fixture("opt-xn.f32", N * D)
    var head = slice_rows(xn, 0, 100, D)
    var none = List[Float32]()
    var first = std_fit(head, 100, D, True, True, none)
    var member = HostMember(MEMBER_SCALER, std_params(True, True))
    try:
        member.fit(head, 100, D)
        same_bits(first.stats, member.block(1), "first partial_fit batch against fit")
    finally:
        member.free()


def flags_case() raises:
    var n = 600
    var p = fixture("opt-p.f32", n * D)
    var none = List[Float32]()
    var nm = std_fit(slice_rows(p, 0, 100, D), 100, D, True, False, none)
    std_partial(nm, slice_rows(p, 100, n, D), n - 100, D, True, False, none)
    rows_close(take32(nm.stats, 0, D), reference("opt-p-mean.f64", D), RT, RT, "with_std=False partial_fit mean_")
    var neither = std_fit(slice_rows(p, 0, 100, D), 100, D, False, False, none)
    std_partial(neither, slice_rows(p, 100, n, D), n - 100, D, False, False, none)
    require(neither.kind == SCALER_SEEN_INT and neither.scalar == Float64(n),
        "with_mean=with_std=False partial_fit: n_samples_seen_ " + String(neither.scalar))


def all_nan_case() raises:
    var x = fixture("opt-x.f32", N * D)
    for i in range(N):  # small-loop(fixture: feature 1 all NaN)
        x[i * D + 1] = bitcast[DType.float32](UInt32(0x7FC00000))
    var member = HostMember(MEMBER_SCALER, std_params(True, True))
    try:
        member.fit(x, N, D)
        var state = member_state(member, D)
        require(isnan(state.stats[1]) and isnan(state.stats[2 * D + 1]), "all-NaN feature: mean_ / scale_ not NaN")
        require(state.seen(1) == 0.0, "all-NaN feature: n_samples_seen_ " + String(state.seen(1)))
        var filled = x.copy()
        for i in range(N):  # small-loop(fixture: np.nan_to_num)
            if isnan(filled[i * D + 1]):
                filled[i * D + 1] = 0.0
        var t = member.transform(filled, N)
        for i in range(N * D):  # small-loop(oracle: only the unseen column is NaN)
            require(isnan(t[i]) == (i % D == 1), "all-NaN feature: transform word " + String(i) + " = " + String(t[i]))
        var y = fixture("opt-y.f32", N * D)
        var none = List[Float32]()
        std_partial(state, y, N, D, True, True, none)
        all_close(state.stats[1], reference("opt-y-mean.f64", D)[1], RT, RT, "all-NaN feature: first sighting mean_")
    finally:
        member.free()


def mm_nan_case(k: Int) raises:
    var xn = fixture("opt-xn.f32", N * D)
    var member = HostMember(MEMBER_SCALER, mm_params(k == 1))
    try:
        member.fit(xn, N, D)
        var label = String("MinMax NaN fit clip ") + String(k == 1)
        rows_close(member.block(1), reference("opt-mm" + String(k) + "-stats.f64", 5 * D), RT, RT,
            label + ": data_min_, data_max_, data_range_, scale_, min_")
        var t = member.transform(xn, N)
        rows_close(t, reference("opt-mm" + String(k) + "-transform.f64", N * D), RT, RT, label + ": transform")
    finally:
        member.free()


def mm_rows(x: List[Float32], n: Int) raises -> List[Float32]:
    var member = HostMember(MEMBER_SCALER, mm_params(False))
    try:
        member.fit(x, n, D)
        return member.block(1)
    finally:
        member.free()


def mm_filled_case() raises:
    var a = mm_rows(fixture("opt-mm-fill.f32", N * D), N)
    var b = mm_rows(fixture("opt-xn.f32", N * D), N)
    var names: List[String] = ["data_min_", "data_max_", "data_range_", "scale_", "min_"]
    for r in range(5):  # small-loop(oracle: the published rows but data_range_)
        if r != 2:
            same_bits(take32(a, r * D, D), take32(b, r * D, D), "NaN fit against the filled fit: " + names[r])


def minmax_partial(rows: List[Float32], x: List[Float32], n: Int, d: Int) raises -> List[Float32]:
    """bindings/minmax_native_door.mojo `minmax_partial_fit` on typed values (feature_range (-1, 2))."""
    var c = calls()
    var fresh = scaler_minmax_fit_core(c, F32P(unsafe_from_address=Int(x.unsafe_ptr())), n, d, -1.0, 2.0)
    var arena = List[Float32](length=9 * d, fill=0)
    for j in range(d):  # small-loop(fixture: the old and new extrema rows into the arena)
        arena[j] = rows[j]
        arena[d + j] = rows[d + j]
        arena[2 * d + j] = fresh[j]
        arena[3 * d + j] = fresh[d + j]
    var program = List[Int32]()
    prep_program_stage(program, 268, d, 0, d, 2 * d, 3 * d, d, 4 * d, 8 * d)
    c.run(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)
    var output = List[Float32](length=5 * d, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    _ = c.minmax_fit(MinMaxFitRequest(Int(ap + 4 * d), Int(output.unsafe_ptr()), 4, d, -1.0, 2.0, False))
    var merged = scaler_minmax_keep_extrema(c, F32P(unsafe_from_address=Int(output.unsafe_ptr())), Int(ap + 8 * d), d)
    _ = arena^
    _ = output^
    _ = program^
    _ = fresh^
    _ = x
    return merged^


def mm_partial_case() raises:
    var x = fixture("opt-x.f32", N * D)
    var member = HostMember(MEMBER_SCALER, mm_params(False))
    try:
        member.fit(slice_rows(x, 0, 50, D), 50, D)
        var rows = member.block(1)
        var seen = Int(member.scalars()[1])
        var cuts: List[Int] = [50, 120, N]
        for s in range(2):  # small-loop(oracle: the two later batches, in order)
            rows = minmax_partial(rows, slice_rows(x, cuts[s], cuts[s + 1], D), cuts[s + 1] - cuts[s], D)
            seen += cuts[s + 1] - cuts[s]
        same_bits(rows, mm_rows(x, N), "MinMax partial_fit over three batches against one fit")
        require(seen == N, "MinMax partial_fit n_samples_seen_ " + String(seen))
    finally:
        member.free()


def mm_poison_case() raises:
    var x = fixture("opt-x.f32", N * D)
    var rows = mm_rows(slice_rows(x, 20, N, D), N - 20)
    var merged = minmax_partial(rows, fixture("opt-mm-z.f32", 20 * D), 20, D)
    require(isnan(merged[3 * D + 4]), "MinMax partial_fit all-NaN column: scale_ " + String(merged[3 * D + 4]))


def options_case(unit: Int) raises:
    if unit == 0:
        nan_case()
    elif unit <= 2:
        weight_case(unit - 1)
    elif unit == 3:
        unit_weight_case()
    elif unit <= 7:
        partial_case(unit - 4)
    elif unit == 8:
        first_batch_case()
    elif unit == 9:
        flags_case()
    elif unit == 10:
        all_nan_case()
    elif unit <= 12:
        mm_nan_case(unit - 11)
    elif unit == 13:
        mm_filled_case()
    elif unit == 14:
        mm_partial_case()
    else:
        mm_poison_case()


def test_scaler_options() raises:
    check_cases("scaler_options", 16, options_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
