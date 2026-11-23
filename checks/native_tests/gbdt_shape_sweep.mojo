# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""GradientBoosting over a sweep of table shapes, and a missing value at
predict in a column that had none at fit.

Lane fx-gbdt-misaligned (2026-11-20). A published NVIDIA wheel returned
`CUDA_ERROR_MISALIGNED_ADDRESS` from a SymmetricTree Logloss fit on some
column subsets of one table and not on others of the same rows. CUDA reports
that fault at a later drain and it then stays on the context, so the unit
that FIRST fails names the shape; every unit after it fails too and says
nothing. Each unit is one fit, its failure text carries the shape, and the
device column runs the units in order on one context.

Units, in order:

- the sweep: 5 column mixes x column counts 1..33 x 2 odd row counts. The
  mixes choose the compressed-index policies a layout holds: 0 wide columns
  (one-byte groups; 128 borders on odd column counts, 254 on even, so the
  7-bit and 8-bit arms both run), 1 seven-valued columns (half-byte only),
  2 two-valued columns (binary only), 3 the three kinds in turn, 4 sparse
  counts whose later columns are constant on a small sample (zero folds
  beside live columns);
- four larger fits at the reported scale (24,033 rows; 1, 3, 21 and 63
  sparse-count columns; 4 trees: the layout, the compressed index and the
  device buffers the fault read are built once per fit, before the first
  tree, so 4 trees run every per-tree drain of them as 20 did);
- two fits whose every column is constant (an empty layout): the fit may
  refuse in words, but it may not fault the device, and an ordinary fit
  after it must succeed;
- missing values at predict: a fit on columns without a NaN, then a predict
  with NaN in one column. Under `nan_mode='Min'` the prediction equals the
  prediction with -inf in those cells, under `'Max'` with +inf; under
  `'Forbidden'` the predict refuses. The device column also runs the
  device-resident predict on the same rows.

Lane fx-gbdt-constcol (2026-11-20) added the neighbours of the unit that
faulted (one constant column: a layout with no histogram cell and, before
the fix, no compressed-index column, whose zero-length device buffers the
level's no-winner fallback read). They follow the units above, so the
earlier unit numbers stand:

- one constant column first, in the middle and last among 1..33 wide
  columns: these fits must succeed;
- every column constant under the other growth policies and another loss
  (Depthwise, Lossguide, SymmetricTree RMSE; 1 and 3 columns);
- a column that is constant only after binning (two adjacent float values),
  alone and beside wide columns;
- a per-tree column sample over a table whose late columns are constant, so
  some trees draw only constant columns;
- a table with no rows.

For a table without any candidate split, and for the table without rows,
the fit may refuse in words; it may not fault the device, and an ordinary
fit on the same context must succeed afterwards. The table without rows is
the last unit.

The host column runs the same units, one task per unit across the cores.
The device column runs them in order on one context because a device fault
stays on the context: run in order, the first failing unit names the shape.
"""
from std.math import inf, nan
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gbdt.gbdt_entry import GbdtImage
from gbdt.resident_model import RESIDENT_RAW, ResidentGbdtModel
from checks.native_tests.gbdt_public_fits import (
    GbdtInputs,
    GbdtOpts,
    _DeviceFit,
    _HostFit,
    all_finite_f32,
    fit,
    predict,
    same_f32,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from std.sys.compile import is_defined


comptime SWEEP_MIXES = 5
comptime SWEEP_MAX_COLUMNS = 33
comptime SWEEP_ROW_COUNTS = 2
comptime SWEEP_UNITS = SWEEP_MIXES * SWEEP_MAX_COLUMNS * SWEEP_ROW_COUNTS
comptime LARGE_UNITS = 4
comptime CONSTANT_UNITS = 2
comptime NAN_UNITS = 3
#: one constant column at 3 places among 1..SWEEP_MAX_COLUMNS wide columns
comptime AMONG_PLACES = 3
comptime AMONG_UNITS = AMONG_PLACES * SWEEP_MAX_COLUMNS
#: every column constant: 3 configurations x {1, 3} columns
comptime POLICY_UNITS = 6
#: constant after binning (alone, beside wide columns)
comptime BINNED_UNITS = 2
#: a per-tree column sample that can draw only constant columns
comptime SAMPLED_UNITS = 2
#: no rows
comptime EMPTY_UNITS = 1
comptime NEIGHBOUR_UNITS = AMONG_UNITS + POLICY_UNITS + BINNED_UNITS + SAMPLED_UNITS + EMPTY_UNITS
comptime FIRST_UNITS = SWEEP_UNITS + LARGE_UNITS + CONSTANT_UNITS + NAN_UNITS
comptime SHAPE_UNITS = FIRST_UNITS + NEIGHBOUR_UNITS

comptime MIX_WIDE = 0
comptime MIX_HALF = 1
comptime MIX_BINARY = 2
comptime MIX_CYCLE = 3
comptime MIX_SPARSE = 4
#: every column constant (the two empty-layout units only)
comptime MIX_CONSTANT = 5

comptime LARGE_ROWS = 24033

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def _hash(row: Int, column: Int) -> Int:
    return (row * 2654435761 + column * 97003 + 12345) % 10000


def _cell(mix: Int, row: Int, column: Int) -> Float32:
    """One value of the generated table; a pure function of its position."""
    var h = _hash(row, column)
    var kind = mix
    if mix == MIX_CYCLE:
        kind = column % 3
    if kind == MIX_WIDE:
        return Float32(h) / Float32(10000.0) - Float32(0.5)
    if kind == MIX_HALF:
        return Float32(h % 7)
    if kind == MIX_BINARY:
        return Float32(1.0) if h % 5 == 0 else Float32(0.0)
    if kind == MIX_SPARSE:
        # the share of nonzero cells halves from one column to the next and
        # starts again every 14 columns, so a small sample leaves the late
        # columns of each run constant
        var threshold = 10000 >> (column % 14)
        if h < threshold:
            return Float32(1 + h % 3)
        return Float32(0.0)
    return Float32(0.0)


def _table(mix: Int, n: Int, d: Int) -> GbdtInputs:
    """Column-major X and a two-class label that follows column 0's hash,
    with every fifth row flipped."""
    var x = List[Float32](length=n * d, fill=Float32(0.0))
    var y = List[Float32](length=n, fill=Float32(0.0))
    for column in range(d):  # small-loop(d: at most 63 generated columns of one unit, inside one unit's task)
        for row in range(n):  # small-loop(n: rows of one generated column, inside one unit's task)
            x[column * n + row] = _cell(mix, row, column)
    for row in range(n):  # small-loop(n: labels of one generated table, inside one unit's task)
        var high = _hash(row, 0) < 5000
        var flip = row % 5 == 0
        y[row] = Float32(1.0) if high != flip else Float32(0.0)
    return GbdtInputs(x^, n, d, y^)


def _options(trees: Int, d: Int) -> GbdtOpts:
    """The reported call: Logloss, SymmetricTree, depth 6, learning rate
    0.05, every other option the constructor's default."""
    var o = GbdtOpts("Logloss", trees, 6)
    o.learning_rate = 0.05
    o.border_count = 254 if d % 2 == 0 else 128
    return o^


def _shape(mix: Int, n: Int, d: Int) -> String:
    return "mix " + String(mix) + ", " + String(n) + " rows, " + String(d) + " columns"


def _fit_and_score[I: GbdtImage](mix: Int, n: Int, d: Int, trees: Int) raises:
    var inp = _table(mix, n, d)
    var text = String("")
    try:
        var model = fit[I](_options(trees, d), inp)
        text = model.text
    except error:
        raise Error("gbdt shape sweep: the fit failed at " + _shape(mix, n, d) + ": " + String(error))
    var scores = predict(text, inp.x, n)
    require(len(scores) == n, "gbdt shape sweep: predict extent at " + _shape(mix, n, d))
    require(all_finite_f32(scores), "gbdt shape sweep: a non-finite score at " + _shape(mix, n, d))


def sweep_unit[I: GbdtImage](unit: Int) raises:
    var per_mix = SWEEP_MAX_COLUMNS * SWEEP_ROW_COUNTS
    var mix = unit // per_mix
    var rest = unit - mix * per_mix
    var d = 1 + rest // SWEEP_ROW_COUNTS
    var n = 301 if rest % SWEEP_ROW_COUNTS == 0 else 1003
    _fit_and_score[I](mix, n, d, 2)


comptime LARGE_TREES = 4  # was 20 (check-trim, 2026-11-23): the scale is in the rows and columns, not the tree count


def large_unit[I: GbdtImage](unit: Int) raises:
    var widths = [1, 3, 21, 63]
    _fit_and_score[I](MIX_SPARSE, LARGE_ROWS, widths[unit], LARGE_TREES)


def constant_unit[I: GbdtImage](unit: Int) raises:
    """Every column constant. A refusal in words is accepted; a device fault
    is not, and the context must still fit an ordinary table afterwards."""
    var d = 1 if unit == 0 else 3
    var inp = _table(MIX_CONSTANT, 301, d)
    _fit_or_refuse[I](_options(2, d), inp, String(d) + " constant columns")


def _no_device_fault(error_text: String, what: String) raises:
    """A refusal in words passes; a device fault does not."""
    require(
        error_text.find("CUDA") < 0 and error_text.find("HIP") < 0 and error_text.find("Metal") < 0,
        "gbdt shape sweep: " + what + " faulted the device: " + error_text,
    )


def _fit_or_refuse[I: GbdtImage](o: GbdtOpts, inp: GbdtInputs, what: String) raises:
    """One fit that may refuse in words, then an ordinary fit that must
    succeed on the same context."""
    try:
        _ = fit[I](o, inp)
    except error:
        _no_device_fault(String(error), what)
    _fit_and_score[I](MIX_WIDE, 301, 4, 2)


def _table_with_constant(n: Int, wide: Int, place: Int) -> GbdtInputs:
    """`wide` wide columns and one constant column at index `place`."""
    var d = wide + 1
    var x = List[Float32](length=n * d, fill=Float32(3.0))
    var y = List[Float32](length=n, fill=Float32(0.0))
    for column in range(d):  # small-loop(d: at most 34 generated columns of one unit, inside one unit's task)
        if column == place:
            continue
        for row in range(n):  # small-loop(n: rows of one generated column, inside one unit's task)
            x[column * n + row] = _cell(MIX_WIDE, row, column)
    for row in range(n):  # small-loop(n: labels of one generated table, inside one unit's task)
        var high = _hash(row, 0) < 5000
        var flip = row % 5 == 0
        y[row] = Float32(1.0) if high != flip else Float32(0.0)
    return GbdtInputs(x^, n, d, y^)


def among_unit[I: GbdtImage](unit: Int) raises:
    """One constant column first, in the middle or last among wide columns."""
    var which = unit // SWEEP_MAX_COLUMNS
    var wide = 1 + unit % SWEEP_MAX_COLUMNS
    var place = 0
    if which == 1:
        place = wide // 2
    elif which == 2:
        place = wide
    var n = 301
    var d = wide + 1
    var inp = _table_with_constant(n, wide, place)
    var shape = String(wide) + " wide columns and a constant column at " + String(place)
    var text = String("")
    try:
        var model = fit[I](_options(2, d), inp)
        text = model.text
    except error:
        raise Error("gbdt shape sweep: the fit failed at " + shape + ": " + String(error))
    var scores = predict(text, inp.x, n)
    require(len(scores) == n, "gbdt shape sweep: predict extent at " + shape)
    require(all_finite_f32(scores), "gbdt shape sweep: a non-finite score at " + shape)


def policy_unit[I: GbdtImage](unit: Int) raises:
    """Every column constant under the other growth policies and RMSE."""
    var which = unit // 2
    var d = 1 if unit % 2 == 0 else 3
    var inp = _table(MIX_CONSTANT, 301, d)
    var o = _options(2, d)
    var what = String("Depthwise")
    if which == 0:
        o.max_depth = 3
        o.grow_policy = "Depthwise"
    elif which == 1:
        o.max_leaves = 8
        o.grow_policy = "Lossguide"
        what = String("Lossguide")
    else:
        o.loss = "RMSE"
        what = String("SymmetricTree RMSE")
    _fit_or_refuse[I](o, inp, what + ", " + String(d) + " constant columns")


def binned_unit[I: GbdtImage](unit: Int) raises:
    """A column holding two adjacent float values: distinct in the table,
    and with no float between them to put a border at."""
    var n = 301
    var d = 1 if unit == 0 else 3
    var inp = _table(MIX_WIDE, n, d)
    var low = Float32(1.0)
    var high = Float32(1.0) + Float32(1.1920929e-07)
    for row in range(n):  # small-loop(n: rows of one column of one unit)
        inp.x[row] = high if _hash(row, 7) < 5000 else low
    _fit_or_refuse[I](_options(2, d), inp, "two adjacent values in column 0 of " + String(d))


def sampled_unit[I: GbdtImage](unit: Int) raises:
    """A per-tree column sample over sparse counts whose late columns are
    constant on 301 rows: some trees draw only constant columns."""
    var d = 14
    var inp = _table(MIX_SPARSE, 301, d)
    var o = _options(8, d)
    o.feature_fraction = 0.1
    var what = String("SymmetricTree")
    if unit == 1:
        o.max_leaves = 8
        o.grow_policy = "Lossguide"
        what = String("Lossguide")
    _fit_or_refuse[I](o, inp, what + " with feature_fraction 0.1 over constant columns")


def empty_unit[I: GbdtImage]() raises:
    """No rows: the fit refuses in words."""
    var inp = _table(MIX_WIDE, 0, 4)
    var refused = False
    try:
        _ = fit[I](_options(2, 4), inp)
    except error:
        _no_device_fault(String(error), "a table with no rows")
        refused = True
    require(refused, "gbdt shape sweep: a fit on a table with no rows returned a model")
    _fit_and_score[I](MIX_WIDE, 301, 4, 2)


def neighbour_unit[I: GbdtImage](unit: Int) raises:
    if unit < AMONG_UNITS:
        among_unit[I](unit)
        return
    var at = unit - AMONG_UNITS
    if at < POLICY_UNITS:
        policy_unit[I](at)
        return
    at -= POLICY_UNITS
    if at < BINNED_UNITS:
        binned_unit[I](at)
        return
    at -= BINNED_UNITS
    if at < SAMPLED_UNITS:
        sampled_unit[I](at)
        return
    empty_unit[I]()


def _resident_raw(text: String, x: List[Float32], n: Int) raises -> List[Float32]:
    """The device-resident predict (`gbdt_resident_ns_predict._resident`)."""
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():  # discovery instantiates every module function
        raise Error("gbdt shape sweep: the resident predict runs on the device column only")
    else:
        var model = ResidentGbdtModel(text)
        var scores = List[Float32](length=n, fill=Float32(0.0))
        var out64 = List[Float64](length=1, fill=0.0)
        var codes = List[Int64](length=1, fill=0)
        var width = model.predict_into(
            _F32P(unsafe_from_address=Int(x.unsafe_ptr())),
            n,
            _F32P(unsafe_from_address=Int(scores.unsafe_ptr())),
            _F64P(unsafe_from_address=Int(out64.unsafe_ptr())),
            _I64P(unsafe_from_address=Int(codes.unsafe_ptr())),
            RESIDENT_RAW,
            False,
        )
        require(width == 1, "gbdt shape sweep: the resident predict wrote width " + String(width))
        _ = out64^
        _ = codes^
        _ = model^
        _ = x
        return scores^


def nan_unit[I: GbdtImage](unit: Int) raises:
    """A NaN at predict in a column whose learn rows had none."""
    var n = 1003
    var d = 5
    var hit = 2
    var inp = _table(MIX_WIDE, n, d)
    var o = _options(4, d)
    if unit == 1:
        o.nan_mode = "Max"
    elif unit == 2:
        o.nan_mode = "Forbidden"
    var model = fit[I](o, inp)
    var missing = inp.x.copy()
    var stand_in = inp.x.copy()
    var low = Float32(-inf[DType.float32]())
    var high = Float32(inf[DType.float32]())
    for row in range(n):  # small-loop(n: rows of one column of one unit)
        if row % 3 == 0:
            missing[hit * n + row] = nan[DType.float32]()
            stand_in[hit * n + row] = high if unit == 1 else low
    if unit == 2:
        var refused = False
        try:
            _ = predict(model.text, missing, n)
        except error:
            require(
                String(error).find("no NaNs in the learn dataset") >= 0,
                "gbdt shape sweep: nan_mode='Forbidden' refused in other words: " + String(error),
            )
            refused = True
        require(refused, "gbdt shape sweep: nan_mode='Forbidden' accepted a NaN at predict")
        return
    var got = predict(model.text, missing, n)
    var want = predict(model.text, stand_in, n)
    require(all_finite_f32(got), "gbdt shape sweep: a NaN at predict gave a non-finite score")
    require(
        same_f32(got, want),
        "gbdt shape sweep: a NaN at predict is not the " + o.nan_mode + " side of every split on its column",
    )
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():
        var device_got = _resident_raw(model.text, missing, n)
        var device_want = _resident_raw(model.text, stand_in, n)
        require(
            same_f32(device_got, device_want),
            "gbdt shape sweep: the device predict of a NaN is not the " + o.nan_mode + " side",
        )


def shape_unit[I: GbdtImage](unit: Int) raises:
    if unit < SWEEP_UNITS:
        sweep_unit[I](unit)
        return
    var at = unit - SWEEP_UNITS
    if at < LARGE_UNITS:
        large_unit[I](at)
        return
    at -= LARGE_UNITS
    if at < CONSTANT_UNITS:
        constant_unit[I](at)
        return
    at -= CONSTANT_UNITS
    if at < NAN_UNITS:
        nan_unit[I](at)
        return
    neighbour_unit[I](at - NAN_UNITS)


def _host_shape(unit: Int) raises:
    print("gbdt_shape_sweep unit", unit, "start")
    try:
        shape_unit[_HostFit](unit)
    finally:
        print("gbdt_shape_sweep unit", unit, "done")


def _cpu_cases() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    check_cases("gbdt_shape_sweep", SHAPE_UNITS, _host_shape)


def _gpu_cases() raises:
    """One device fit at a time, in unit order: the first failing unit is the
    shape that faulted (a device fault stays on the context)."""
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():  # discovery instantiates every module function
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        var failed = String("")
        var count = 0
        for unit in range(SHAPE_UNITS):  # driver-loop: over units, ordered GPU work on one context
            try:
                shape_unit[_DeviceFit](unit)
            except error:
                count += 1
                failed += "\n      gbdt_shape_sweep row " + String(unit) + ": " + String(error)
        if count > 0:
            raise Error("gbdt_shape_sweep: " + String(count) + " of " + String(SHAPE_UNITS) + " cases failed" + failed)


def test_cases() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _cpu_cases()
    else:
        _gpu_cases()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
