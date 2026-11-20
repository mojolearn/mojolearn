# SPDX-License-Identifier: Apache-2.0
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
  sparse-count columns; 20 trees);
- two fits whose every column is constant (an empty layout): the fit may
  refuse in words, but it may not fault the device, and an ordinary fit
  after it must succeed;
- missing values at predict: a fit on columns without a NaN, then a predict
  with NaN in one column. Under `nan_mode='Min'` the prediction equals the
  prediction with -inf in those cells, under `'Max'` with +inf; under
  `'Forbidden'` the predict refuses. The device column also runs the
  device-resident predict on the same rows.

The host column runs the same units, one task per unit across the cores.
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
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime SWEEP_MIXES = 5
comptime SWEEP_MAX_COLUMNS = 33
comptime SWEEP_ROW_COUNTS = 2
comptime SWEEP_UNITS = SWEEP_MIXES * SWEEP_MAX_COLUMNS * SWEEP_ROW_COUNTS
comptime LARGE_UNITS = 4
comptime CONSTANT_UNITS = 2
comptime NAN_UNITS = 3
comptime SHAPE_UNITS = SWEEP_UNITS + LARGE_UNITS + CONSTANT_UNITS + NAN_UNITS

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


def large_unit[I: GbdtImage](unit: Int) raises:
    var widths = [1, 3, 21, 63]
    _fit_and_score[I](MIX_SPARSE, LARGE_ROWS, widths[unit], 20)


def constant_unit[I: GbdtImage](unit: Int) raises:
    """Every column constant. A refusal in words is accepted; a device fault
    is not, and the context must still fit an ordinary table afterwards."""
    var d = 1 if unit == 0 else 3
    var inp = _table(MIX_CONSTANT, 301, d)
    try:
        _ = fit[I](_options(2, d), inp)
    except error:
        var said = String(error)
        require(
            said.find("CUDA") < 0 and said.find("HIP") < 0 and said.find("Metal") < 0,
            "gbdt shape sweep: " + String(d) + " constant columns faulted the device: " + said,
        )
    _fit_and_score[I](MIX_WIDE, 301, 4, 2)


def _resident_raw(text: String, x: List[Float32], n: Int) raises -> List[Float32]:
    """The device-resident predict (`gbdt_resident_ns_predict._resident`)."""
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
    comptime if not CPU_COLUMN_BUILD:
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
    nan_unit[I](at - CONSTANT_UNITS)


def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(SHAPE_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            shape_unit[_HostFit](unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, SHAPE_UNITS)


def _run_gpu(mut report: Report) raises:
    """One device fit at a time, in unit order: the first failing unit is the
    shape that faulted (a device fault stays on the context)."""
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(SHAPE_UNITS)
    for unit in range(SHAPE_UNITS):  # driver-loop: over units, ordered GPU work on one context
        try:
            shape_unit[_DeviceFit](unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("gbdt_shape_sweep", cpu=True, gpu=True, serial="", run=run_cases)
