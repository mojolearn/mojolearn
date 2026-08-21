"""`nan_mode`: a missing value gets a bin of its own, at one end.

    pixi run check-nan-mode

CatBoost handles NaN entirely in QUANTIZATION -- there is no NaN branch in
their histogram kernels, their scorer, their partitioner or their evaluator.
`CalcQuantization` (`libs/data/quantization.cpp:321-346`) spends one of the
column's borders on a sentinel and puts NaN in the bin beyond it, and
everything downstream sees an ordinary bin index.

That makes the whole feature checkable through `train()` and
`predict_floats`, and it makes the two failure modes worth naming, because
neither one crashes:

    the sentinel without the decrement   the column gets `border_count + 1`
                                         bins where the caller asked for
                                         `border_count`, which can push it
                                         across a grid-policy step and into
                                         a different histogram kernel

    the treatment lost on save           a model that should REFUSE a NaN
                                         row silently scores it in the
                                         bottom bin instead

THE GATES

1. **THE BIN IS SEPARATE AND IT IS AT THE END THE MODE NAMES.** Built on a
   fixture where NaN rows carry a target no real value's neighbourhood can
   explain, so a tree can only fit them by isolating the NaN bin. Under
   `Min` and under `Max` the fit reaches the same low loss; a fit that
   dropped NaNs into a neighbouring bin cannot.

2. **THE BORDER BUDGET IS RESPECTED.** A column with NaNs gets `border_count`
   borders TOTAL, one of which is the sentinel, so it has exactly as many
   as the same column without NaNs. This is the assertion that catches an
   insert-without-decrement, and nothing about the loss would.

3. **THE SENTINEL IS AT THE RIGHT END.** `Min` puts `lowest()` first,
   `Max` puts `max()` last. Read off the model.

4. **`Forbidden` IS WHAT A CLEAN COLUMN RESOLVES TO**, whatever was asked:
   the treatment is `as_is` and the borders are the same as a fit that
   never heard of NaN, so no budget is spent on a bin nothing reaches.

5. **A SAVED MODEL ROUTES A NaN THE SAME WAY.** Save, load, predict on rows
   containing NaN, and every prediction must be bit-identical.

6. **A NaN ON A COLUMN THE LEARN POOL HAD NONE IN IS REFUSED**, their
   `CB_ENSURE(allowNans, "There are NaNs in test dataset ...")`.

SABOTAGES:

    N1  the two modes compared against each      that the fixture can tell
        other -- they must NOT agree row for      Min from Max at all
        row, since the NaN bin sits at
        opposite ends of the grid
    N2  the same fit with the NaN rows given      that gate 1 is measuring
        a real value instead                      the NaN handling and not
                                                  an easy target
"""

from max.gpu.host import DeviceContext

from gbdt.data.quantization import (
    NAN_TREATMENT_AS_FALSE,
    NAN_TREATMENT_AS_IS,
    NAN_TREATMENT_AS_TRUE,
)
from gbdt.models.model_text import load_model_text, model_text
from gbdt.train import TrainedModel, predict_floats, train

comptime NM_ROWS = 4000
comptime NM_FEATURES = 3
comptime NM_BORDERS = 32


def _hashed(x: Int, salt: Int) -> Int:
    var v = ((x + 1) * 2654435761 + salt * 40503) % 1000003
    if v < 0:
        v += 1000003
    return v


def _fixture(
    plant_nans: Bool,
) raises -> Tuple[List[Float32], List[Float32]]:
    """Column 0 carries NaN on one row in eight; the other two are noise.

    **The NaN rows' target is far outside the range the non-NaN rows span**,
    and it does not depend on anything else, so the only way to fit them is
    a bin that holds NaN and nothing else. If they were merely at one end of
    the value range, a fit that dropped NaN into the neighbouring bin would
    do almost as well and gate 1 would not be able to tell.

    `plant_nans=False` replaces every NaN with an ordinary value and keeps
    the same targets -- sabotage N2, which must NOT reach the same loss.
    """
    var x = List[Float32]()
    for f in range(NM_FEATURES):
        for r in range(NM_ROWS):
            if f == 0 and (r % 8) == 0:
                if plant_nans:
                    x.append(Float32(0.0) / Float32(0.0))
                else:
                    # a value inside the ordinary range, so it shares bins
                    # with real rows instead of getting one of its own
                    x.append(
                        Float32(_hashed(r, 5) % 1000) / Float32(1000.0)
                    )
            else:
                x.append(Float32(_hashed(r, 7 + f) % 1000) / Float32(1000.0))

    var y = List[Float32]()
    for r in range(NM_ROWS):
        if (r % 8) == 0:
            y.append(Float32(50.0))
        else:
            y.append(
                Float32(_hashed(r, 11) % 1000) / Float32(1000.0)
            )
    return (x^, y^)


def _mse(p: List[Float32], y: List[Float32]) -> Float64:
    var acc = Float64(0.0)
    for r in range(len(y)):
        var d = Float64(p[r]) - Float64(y[r])
        acc += d * d
    return acc / Float64(len(y))


def _fit(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    mode: String,
) raises -> TrainedModel:
    return train(
        ctx, x, y, NM_ROWS, NM_FEATURES,
        border_count=NM_BORDERS, n_estimators=20, max_depth=6,
        loss="RMSE", learning_rate=Float32(0.3),
        nan_mode=mode,
    )


def check_nan_mode() raises:
    var ctx = DeviceContext()
    var failures = 0

    var f = _fixture(True)
    var x = f[0].copy()
    var y = f[1].copy()

    print("-- gate 1: NaN gets a bin of its own, under both modes --")
    var m_min = _fit(ctx, x, y, String("Min"))
    var m_max = _fit(ctx, x, y, String("Max"))
    var loss_min = _mse(predict_floats(ctx, m_min, x, NM_ROWS), y)
    var loss_max = _mse(predict_floats(ctx, m_max, x, NM_ROWS), y)

    var clean = _fixture(False)
    var xc = clean[0].copy()
    var yc = clean[1].copy()
    var m_clean = _fit(ctx, xc, yc, String("Min"))
    var loss_clean = _mse(predict_floats(ctx, m_clean, xc, NM_ROWS), yc)

    print("  Min", loss_min, " Max", loss_max, " N2 (no NaNs)", loss_clean)
    if not (loss_clean > loss_min * 5.0):
        print(
            "  FAIL N2: the fixture fits just as well with the NaNs"
            " replaced by ordinary values, so gate 1 is not measuring NaN"
            " handling",
        )
        failures += 1
    else:
        print(
            "  ok   N2 the same targets without NaNs cost",
            loss_clean / loss_min, "x more -- the NaN bin is what fits them",
        )
    if not (loss_max < loss_clean / 5.0):
        print("  FAIL Max did not reach the same low loss as Min")
        failures += 1
    else:
        print("  ok   both modes isolate the NaN rows")

    print()
    print("-- gate 2: a NaN column spends ONE of its borders, not one more --")
    var n_min = len(m_min.borders[0])
    var n_clean = len(m_clean.borders[0])
    if n_min != n_clean:
        print(
            "  FAIL feature 0 has", n_min, "borders with NaNs and",
            n_clean, "without; the budget is", NM_BORDERS,
        )
        failures += 1
    else:
        print(
            "  ok   feature 0 has", n_min,
            "borders either way -- the sentinel came OUT of the budget",
        )

    print()
    print("-- gate 3: the sentinel is at the end the mode names --")
    var first_min = m_min.borders[0][0]
    var last_max = m_max.borders[0][len(m_max.borders[0]) - 1]
    var lowest = Float32(-3.4028234663852886e38)
    var highest = Float32(3.4028234663852886e38)
    if first_min != lowest:
        print("  FAIL Min's first border is", first_min, "not lowest()")
        failures += 1
    elif m_min.nan_treatment[0] != NAN_TREATMENT_AS_FALSE:
        print("  FAIL Min's treatment is not as_false")
        failures += 1
    else:
        print("  ok   Min: borders[0][0] is lowest(), treatment as_false")
    if last_max != highest:
        print("  FAIL Max's last border is", last_max, "not max()")
        failures += 1
    elif m_max.nan_treatment[0] != NAN_TREATMENT_AS_TRUE:
        print("  FAIL Max's treatment is not as_true")
        failures += 1
    else:
        print("  ok   Max: borders[0][-1] is max(), treatment as_true")

    print()
    print("-- gate 4: a clean column resolves to Forbidden --")
    var clean_ok = True
    for c in range(NM_FEATURES):
        if m_clean.nan_treatment[c] != NAN_TREATMENT_AS_IS:
            clean_ok = False
    # and the NaN fit's OTHER columns, which have no NaNs of their own
    for c in range(1, NM_FEATURES):
        if m_min.nan_treatment[c] != NAN_TREATMENT_AS_IS:
            clean_ok = False
    if not clean_ok:
        print(
            "  FAIL a column with no NaNs took a treatment other than"
            " as_is; nan_mode is per FEATURE and data dependent",
        )
        failures += 1
    else:
        print(
            "  ok   every column without a NaN is as_is, including in the"
            " fit whose feature 0 has them",
        )

    print()
    print("-- gate 5: a saved model routes a NaN the same way --")
    var before = predict_floats(ctx, m_max, x, NM_ROWS)
    var round_tripped = load_model_text(model_text(m_max))
    var after = predict_floats(ctx, round_tripped, x, NM_ROWS)
    var moved = 0
    for r in range(NM_ROWS):
        if before[r] != after[r]:
            moved += 1
    if moved != 0:
        print(
            "  FAIL", moved, "of", NM_ROWS,
            "predictions moved through save/load on rows containing NaN",
        )
        failures += 1
    else:
        print(
            "  ok   all", NM_ROWS,
            "predictions bit-identical through the text format",
        )

    print()
    print("-- gate 6: a NaN the learn pool never saw is refused --")
    # feature 1 has no NaN in learn, so its treatment is as_is
    var probe = x.copy()
    probe[1 * NM_ROWS + 3] = Float32(0.0) / Float32(0.0)
    var refused = False
    try:
        var _p = predict_floats(ctx, m_min, probe, NM_ROWS)
    except e:
        refused = True
    if not refused:
        print(
            "  FAIL a NaN on a column the learn pool had none in was"
            " scored rather than refused",
        )
        failures += 1
    else:
        print("  ok   refused")

    print()
    print("-- sabotages --")
    # N1: Min and Max put the NaN bin at opposite ends of the grid, so the
    # two models must disagree somewhere. If they agreed row for row, the
    # mode would not be reaching the quantizer at all.
    var p_min = predict_floats(ctx, m_min, x, NM_ROWS)
    var p_max = predict_floats(ctx, m_max, x, NM_ROWS)
    var differ = 0
    for r in range(NM_ROWS):
        if p_min[r] != p_max[r]:
            differ += 1
    if differ == 0:
        print(
            "  FAIL N1: Min and Max produce the same model on every row,"
            " so nan_mode is not reaching the quantizer",
        )
        failures += 1
    else:
        print(
            "  ok   N1", differ, "of", NM_ROWS,
            "rows differ between Min and Max",
        )

    print()
    if failures != 0:
        raise Error("nan_mode check: " + String(failures) + " failures")
    print("nan_mode: PASS")


def main() raises:
    check_nan_mode()
