# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tree-CTR (feature combination) model columns ON THE DEVICE against the
host column, word for word.

    pixi run check-native-gpu combination_code_device_check

The fixture of `gbdt/checks/combination_map_check.mojo` (four raw features,
a combination of a float split, a one-hot split and two categorical
features, a Borders and a FeatureFreq table). Checked:

  * `enqueue_combination_columns` writes the two tree-CTR model columns of
    the learn pool (column-major input) equal to
    `expand_combination_columns`' bit for bit and leaves the other model
    columns alone;
  * the same on a row-major input holding unseen tuples, an unseen category
    and a NaN on the float member, with no refusal recorded;
  * a categorical value that is not a whole number is recorded as the
    refusal of its feature and row;
  * `launch_combination_tuples` (the fit side) rebuilds the map's tuple
    list from the per-row member values and group ids.
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.native_tests.kit import download, upload
from gbdt.checks.combination_map_check import (
    COMB_FIXTURE_COLUMNS,
    COMB_FIXTURE_RAW,
    combination_fixture,
    combination_predict_rows,
)
from gbdt.models.combination_map import (
    CombF32Ptr,
    expand_combination_columns,
)
from gbdt.models.kernel.combination_code import (
    ResidentCombinationPlan,
    enqueue_combination_columns,
    launch_combination_tuples,
)
from gbdt.models.kernel.resident_ctr import CTR_NO_BAD


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("combination device check: " + what)


def _f32c(values: List[Float32]) -> CombF32Ptr:
    return CombF32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def check_combination_device() raises:
    var ctx = DeviceContext()
    var n = 997
    var fx = combination_fixture(n)
    var plan = ResidentCombinationPlan(
        ctx, fx.ctrs, COMB_FIXTURE_COLUMNS, COMB_FIXTURE_RAW
    )

    # the learn pool, column-major
    var want = List[Float32](length=COMB_FIXTURE_COLUMNS * n, fill=Float32(-1.0))
    expand_combination_columns(
        fx.ctrs, _f32c(fx.x), n, COMB_FIXTURE_RAW, False,
        COMB_FIXTURE_COLUMNS, _f32c(want),
    )
    var d_x = upload[DType.float32](ctx, fx.x)
    var d_out = ctx.enqueue_create_buffer[DType.float32](COMB_FIXTURE_COLUMNS * n)
    d_out.enqueue_fill(Float32(-1.0))
    var d_bad = ctx.enqueue_create_buffer[DType.int32](1 + COMB_FIXTURE_RAW)
    d_bad.enqueue_fill(CTR_NO_BAD)
    enqueue_combination_columns(ctx, plan, d_x, n, False, d_out, d_bad)
    var got = download[DType.float32](ctx, d_out, COMB_FIXTURE_COLUMNS * n)
    for i in range(COMB_FIXTURE_COLUMNS * n):
        _need(
            bitcast[DType.uint32](got[i]) == bitcast[DType.uint32](want[i]),
            "the learn pool's tree-CTR columns equal the host column",
        )
    var bad = download[DType.int32](ctx, d_bad, 1 + COMB_FIXTURE_RAW)
    _need(bad[0] == CTR_NO_BAD, "no refusal on the learn pool")

    # five predict rows, row-major: unseen tuples, an unseen category, a NaN
    var rows = combination_predict_rows()
    var want_rows = List[Float32](length=COMB_FIXTURE_COLUMNS * 5, fill=Float32(-1.0))
    expand_combination_columns(
        fx.ctrs, _f32c(rows), 5, COMB_FIXTURE_RAW, True,
        COMB_FIXTURE_COLUMNS, _f32c(want_rows),
    )
    var d_rows = upload[DType.float32](ctx, rows)
    var d_out_rows = ctx.enqueue_create_buffer[DType.float32](COMB_FIXTURE_COLUMNS * 5)
    d_out_rows.enqueue_fill(Float32(-1.0))
    enqueue_combination_columns(ctx, plan, d_rows, 5, True, d_out_rows, d_bad)
    var got_rows = download[DType.float32](ctx, d_out_rows, COMB_FIXTURE_COLUMNS * 5)
    for i in range(COMB_FIXTURE_COLUMNS * 5):
        _need(
            bitcast[DType.uint32](got_rows[i]) == bitcast[DType.uint32](want_rows[i]),
            "the predict rows' tree-CTR columns equal the host column",
        )
    bad = download[DType.int32](ctx, d_bad, 1 + COMB_FIXTURE_RAW)
    _need(bad[0] == CTR_NO_BAD, "unseen values are not refusals")

    # a categorical value that is not a whole number: feature 1, row 1
    rows[5] = Float32(1.5)
    var d_rows_bad = upload[DType.float32](ctx, rows)
    enqueue_combination_columns(ctx, plan, d_rows_bad, 5, True, d_out_rows, d_bad)
    bad = download[DType.int32](ctx, d_bad, 1 + COMB_FIXTURE_RAW)
    _need(bad[0] == Int32(1), "the refusal names the feature")
    _need(bad[2] == Int32(1), "the refusal names the row")

    # the fit side: the tuple list from per-row member values and group ids
    var side0 = List[UInt32](length=n, fill=UInt32(0))
    var side2 = List[UInt32](length=n, fill=UInt32(0))
    var cat1 = List[UInt32](length=n, fill=UInt32(0))
    var cat3 = List[UInt32](length=n, fill=UInt32(0))
    for r in range(n):
        if fx.x[r] > Float32(0.5):
            side0[r] = UInt32(1)
        if fx.x[2 * n + r] == Float32(1.0):
            side2[r] = UInt32(1)
        cat1[r] = UInt32(Int(fx.x[n + r]))
        cat3[r] = UInt32(Int(fx.x[3 * n + r]))
    var d_groups = upload[DType.uint32](ctx, fx.group_ids)
    var g = fx.cmap.groups()
    var d_tuples = ctx.enqueue_create_buffer[DType.uint32](g * 4)
    d_tuples.enqueue_fill(UInt32(0xFFFFFFFF))
    var d_m0 = upload[DType.uint32](ctx, side0)
    var d_m1 = upload[DType.uint32](ctx, side2)
    var d_m2 = upload[DType.uint32](ctx, cat1)
    var d_m3 = upload[DType.uint32](ctx, cat3)
    launch_combination_tuples(ctx, d_m0, d_groups, n, 4, 0, g, d_tuples)
    launch_combination_tuples(ctx, d_m1, d_groups, n, 4, 1, g, d_tuples)
    launch_combination_tuples(ctx, d_m2, d_groups, n, 4, 2, g, d_tuples)
    launch_combination_tuples(ctx, d_m3, d_groups, n, 4, 3, g, d_tuples)
    var tuples = download[DType.uint32](ctx, d_tuples, g * 4)
    for i in range(g * 4):
        _need(tuples[i] == fx.cmap.tuples[i], "the device tuple list is the map's")
    # read through raw pointers above
    _ = len(want)
    _ = len(rows)
    _ = len(want_rows)
    _ = plan^
    print("  tree-CTR model columns on the device equal the host column")


def run_checks() raises:
    check_combination_device()


# Device column only: the suite needs a GPU, so the CPU build never
# instantiates it.
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("combination_code_device_check", cpu=False, gpu=True, serial="", run=run_cases)
