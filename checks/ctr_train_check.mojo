# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`train(cat_features=...)`: the categorical path through the real surface.

    pixi run check-ctr-train

`checks/ctr_check.mojo` gates the CTR ARITHMETIC. This gates the WIRING,
which is a different failure: `PORTING_RULES.md` rule 3 -- a ported file
that no caller reaches is not done -- and this repository has shipped
fully-written, fully-commented machinery nothing called more than once.

## The analytic asymmetry, which is the whole gate

A categorical column of 500 categories carries a target that is a pure
function of its category's FREQUENCY:

    y[r] = count(code[r]) / (n + 1)

which is exactly the FeatureFreq CTR value. Two arms on the SAME data:

* **CTR arm** -- the column is declared in `cat_features`, so `train()`
  replaces it with its CTR columns. One of them IS the target, quantized
  into 16 bins by MinEntropy 15, and a depth-6 oblivious tree over 16 bins
  fits that exactly. Loss must fall to a rounding floor. Run TWICE, once
  under the implicit fallback (`TCatFeatureParams.default()`, four columns:
  three `Borders` priors and one `FeatureFreq`) and once under
  `feature_freq_only()` (one column), because a switch with one side
  unexercised is an unchecked branch (`PORTING_RULES.md` 8).
* **RAW arm** -- the same integer codes handed in as an ordinary numeric
  feature. The category CODES are assigned so that frequency is NOT
  monotone in the code (a hashed permutation), so an ordered threshold on
  the code cannot recover the frequency except by isolating individual
  codes one leaf at a time. A depth-6 tree has 64 leaves for 500
  categories, so the loss floor is far above zero and provably so.

MEASURED, 19913 rows and 500 categories, var(y) = 8.797e-07:

    CTR arm  loss 7.21e-09   loss/var 0.0082
    RAW arm  loss 4.37e-07   loss/var 0.497

a 61x separation. Neither arm needs a CatBoost fixture: the right answer is
arithmetic, the same discipline `one_hot_check.mojo` uses.

SABOTAGE, so the gate is not merely green. Making the CTR column carry the
raw CODE instead of the frequency, and ADDING it beside the categorical
column rather than replacing it, turns all three assertions red at once:
`loss/var` 0.4907 against the RAW arm's 0.4971 (so the 5x separation
assertion fires), and `4 columns for 3 inputs`.

## Applying the CTR model, and three refusals

`predict_floats` on a CTR model USED TO REFUSE, because the CTR values are
a statistic of the LEARN pool and scoring a new row needs the final tables.
The tables landed for `FeatureFreq` and then for `Borders`
(`gbdt/models/ctr_value_table.mojo`), so that line is gone rather than
annotated: both configurations now score raw rows. The refusal that remains
is for a model whose tables are missing, and it lives beside its own gate in
`checks/ctr_apply_check.mojo`.

**AND THE TWO ARMS ARE GATED DIFFERENTLY, WHICH IS THE POINT.**
`FeatureFreq` is permutation-independent (`ctr_type.cpp:44-58`), so its
apply-time table reproduces the learn column BIT FOR BIT and the applied
mse must equal the fit's loss to the last bit. `Borders` has no such
identity: the column it trained on is the ordered statistic over the CTR
estimation permutation, and the column an applied model carries is the
FULL-LEARN-SET histogram (`private/libs/algo/online_ctr.cpp:909-930`).
Pointing the bit-identity gate at Borders would be gating a property it
does not have. MEASURED on this fixture, 19913 rows and 500 categories:

    feature_freq_only()   fit 7.207e-09   applied 7.207e-09   identical
    default()             fit 6.353e-09   applied 6.648e-09   +4.6%

so the Borders arm is gated on SCORING and on still fitting, and the
bit-identity claim stays with the arm that owns it.

Three refusals, because a refusal nobody runs is an unchecked branch:

* a feature in BOTH `cat_features` and `one_hot`.
* a constant categorical column -- their
  `CB_ENSURE(uniqueValues > 1, "Error: useless catFeature found")`.
plus a POSITIVE case where a refusal used to stand:
`TCatFeatureParams.default()`, CatBoost's own GPU `simple_ctr`, which
includes the three `Borders` descriptions. It used to raise, because the CTR
estimation permutation was not ported and row order is a different
estimator rather than a slower one. The permutation landed 2026-08-21
(`gbdt/data/permutation.mojo`, `archive/reference/PORTING.md` 55) and the apply-time tables
after it, so `default()` is now what `train()` FALLS BACK TO and the
`feature_freq_only()` arm is the one passed explicitly. The ordered
statistic's VALUES are gated in `checks/ctr_device_check.mojo`, not here.
"""

from max.gpu.host import DeviceContext

from gbdt.options.catboost_options import TCatFeatureParams
from gbdt.train import TrainedModel, predict_floats, train


comptime CTRT_CATEGORIES = 500
comptime CTRT_ROWS = 20000
comptime CTRT_FEATURES = 3
comptime CTRT_TREES = 40
comptime CTRT_DEPTH = 6


def _hashed(x: Int, salt: Int) -> Int:
    var v = ((x + 1) * 2654435761 + salt * 40503) % 1000003
    if v < 0:
        v += 1000003
    return v


def _fixture() raises -> Tuple[List[Float32], List[Float32], Float64]:
    """Codes in column 0, noise in 1 and 2, `y = count(code) / (n + 1)`.

    The per-category counts are hashed rather than a ramp, and the code a
    category gets is unrelated to its count, so frequency is not monotone
    in the code and no ordered threshold on the code recovers it.
    """
    var sizes = List[Int]()
    var n = 0
    for c in range(CTRT_CATEGORIES):
        var s = 1 + _hashed(c, 5) % 79
        sizes.append(s)
        n += s

    var codes = List[Int]()
    for c in range(CTRT_CATEGORIES):
        for _ in range(sizes[c]):
            codes.append(c)
    # deterministic Fisher-Yates: scattered, so rows of one category are
    # never contiguous
    var state = UInt64(0xA24BAED4963EE407)
    for i in range(len(codes) - 1, 0, -1):
        state = state * UInt64(6364136223846793005) + UInt64(
            1442695040888963407
        )
        var j = Int((state >> 33) % UInt64(i + 1))
        var t = codes[i]
        codes[i] = codes[j]
        codes[j] = t

    var x = List[Float32]()
    for feat in range(CTRT_FEATURES):
        for r in range(n):
            if feat == 0:
                x.append(Float32(codes[r]))
            else:
                x.append(
                    Float32(_hashed(r, 11 + feat)) / Float32(1000003.0)
                )

    var y = List[Float32]()
    for r in range(n):
        y.append(Float32(sizes[codes[r]]) / (Float32(n) + Float32(1.0)))

    var mean = Float64(0.0)
    for r in range(n):
        mean += Float64(y[r])
    mean /= Float64(n)
    var variance = Float64(0.0)
    for r in range(n):
        var d = Float64(y[r]) - mean
        variance += d * d
    variance /= Float64(n)
    return (x^, y^, variance)


def _expect_raise(mut failures: List[String], raised: Bool, what: String):
    if not raised:
        failures.append(what + String(" should have raised and did not"))
    else:
        print("    refused:", what)


def check_ctr_train() raises:
    print("train(cat_features=...) end to end:")
    var ctx = DeviceContext()
    var failures = List[String]()

    var fx = _fixture()
    var x = fx[0].copy()
    var y = fx[1].copy()
    var variance = fx[2]
    var n = len(y)

    var cat_flags = List[Bool]()
    for f in range(CTRT_FEATURES):
        cat_flags.append(f == 0)

    # THE IMPLICIT FALLBACK, which is `TCatFeatureParams.default()` --
    # CatBoost's own GPU `simple_ctr`, three Borders priors plus
    # FeatureFreq. Passing nothing is the path a caller takes by default,
    # so it is the path this arm runs.
    var ctr_model = train(
        ctx, x, y, n, CTRT_FEATURES,
        border_count=128,
        n_estimators=CTRT_TREES,
        max_depth=CTRT_DEPTH,
        learning_rate=Float32(0.7),
        l2_leaf_reg=Float32(1.0),
        cat_features=cat_flags,
    )
    var ctr_loss = ctr_model.losses[len(ctr_model.losses) - 1]

    # AND THE OTHER SIDE OF THE SWITCH, explicitly.
    var freq_params: List[TCatFeatureParams] = [
        TCatFeatureParams.feature_freq_only()
    ]
    var freq_model = train(
        ctx, x, y, n, CTRT_FEATURES,
        border_count=128,
        n_estimators=CTRT_TREES,
        max_depth=CTRT_DEPTH,
        learning_rate=Float32(0.7),
        l2_leaf_reg=Float32(1.0),
        cat_features=cat_flags,
        cat_feature_params=freq_params,
    )
    var freq_loss = freq_model.losses[len(freq_model.losses) - 1]

    var raw_model = train(
        ctx, x, y, n, CTRT_FEATURES,
        border_count=128,
        n_estimators=CTRT_TREES,
        max_depth=CTRT_DEPTH,
        learning_rate=Float32(0.7),
        l2_leaf_reg=Float32(1.0),
    )
    var raw_loss = raw_model.losses[len(raw_model.losses) - 1]

    print(
        "  rows", n, " categories", CTRT_CATEGORIES,
        " var(y)", variance,
    )
    print(
        "  CTR arm, IMPLICIT FALLBACK (their GPU simple_ctr default):"
        " columns", len(ctr_model.fold_counts),
        "of which", ctr_model.ctr_column_count, "are CTR;  loss", ctr_loss,
        " loss/var", ctr_loss / variance,
    )
    print(
        "  CTR arm, feature_freq_only(): columns",
        len(freq_model.fold_counts),
        "of which", freq_model.ctr_column_count, "are CTR;  loss",
        freq_loss, " loss/var", freq_loss / variance,
    )
    print(
        "  RAW arm (same data, codes as an ordinary numeric feature):"
        " columns", len(raw_model.fold_counts),
        " loss", raw_loss, " loss/var", raw_loss / variance,
    )

    # THE FALLBACK IS CATBOOST'S OWN GPU DEFAULT: three Borders priors and
    # one FeatureFreq, so ONE categorical input becomes FOUR columns
    # (`cat_feature_options.cpp:117-129` x `binarizations_manager.cpp:415`).
    if ctr_model.ctr_column_count != 4:
        failures.append(
            String("the implicit fallback is TCatFeatureParams.default(),")
            + String(" which is three Borders priors plus FeatureFreq, so")
            + String(" one categorical input must give FOUR CTR columns;")
            + String(" got ")
            + String(ctr_model.ctr_column_count)
        )
    if len(ctr_model.fold_counts) != CTRT_FEATURES - 1 + 4:
        failures.append(
            String("the categorical column must be REPLACED by its CTR")
            + String(" columns, not joined by them: ")
            + String(len(ctr_model.fold_counts))
            + String(" columns for ")
            + String(CTRT_FEATURES)
            + String(" inputs, two of them numeric")
        )
    # and the other side of the switch is one column and replaces likewise
    if freq_model.ctr_column_count != 1:
        failures.append(
            String("feature_freq_only should give exactly one CTR column;")
            + String(" got ")
            + String(freq_model.ctr_column_count)
        )
    if len(freq_model.fold_counts) != CTRT_FEATURES:
        failures.append(
            String("the categorical column must be REPLACED by its CTR")
            + String(" column, not joined by it: ")
            + String(len(freq_model.fold_counts))
            + String(" columns for ")
            + String(CTRT_FEATURES)
            + String(" inputs")
        )

    if ctr_loss / variance >= 0.05:
        failures.append(
            String("the CTR arm did not learn a target that IS the CTR")
            + String(" value: loss ")
            + String(ctr_loss)
            + String(" against variance ")
            + String(variance)
        )
    if freq_loss / variance >= 0.05:
        failures.append(
            String("the feature_freq_only arm did not learn a target that")
            + String(" IS the CTR value: loss ")
            + String(freq_loss)
            + String(" against variance ")
            + String(variance)
        )
    if raw_loss / variance <= 5.0 * (ctr_loss / variance):
        failures.append(
            String("the RAW arm did as well as the CTR arm, so the CTR")
            + String(" column is not what carried the signal: ")
            + String(raw_loss / variance)
            + String(" against ")
            + String(ctr_loss / variance)
        )
    if raw_loss / variance <= 5.0 * (freq_loss / variance):
        failures.append(
            String("the RAW arm did as well as the feature_freq_only arm: ")
            + String(raw_loss / variance)
            + String(" against ")
            + String(freq_loss / variance)
        )

    # --- APPLYING the CTR model, which used to be a refusal ------------
    #
    # `predict_floats` REFUSED a model with CTR columns for as long as the
    # model carried no apply-time tables. It carries them now
    # (`gbdt/models/ctr_value_table.mojo`, their `ctr_data.hash_map`), so
    # the refusal has become an assertion that it SCORES: FeatureFreq is
    # permutation-independent (`ctr_type.cpp:44-58`), so the table
    # reproduces the learn column exactly and the applied model must
    # reproduce the fit's own loss. The refusal that survives is the one
    # for a model whose tables are MISSING, and
    # `checks/ctr_apply_check.mojo` holds both halves per row.
    print("  applying the CTR models to raw rows:")

    # FeatureFreq: permutation-independent, so the apply-time table
    # reproduces the learn column and the applied mse must equal the fit's.
    var freq_preds = predict_floats(ctx, freq_model, x, n)
    var fse = Float64(0.0)
    for r in range(n):
        var d = Float64(freq_preds[r]) - Float64(y[r])
        fse += d * d
    var freq_pmse = fse / Float64(n)
    var drift = freq_pmse - freq_loss
    if drift < 0:
        drift = -drift
    if drift > 1e-12 + 1e-5 * freq_loss:
        failures.append(
            String("predict_floats on the feature_freq_only model does not")
            + String(" reproduce the fit: ")
            + String(freq_pmse)
            + String(" vs ")
            + String(freq_loss)
        )
    else:
        print("    feature_freq_only: raw rows scored through the CTR"
              " table; mse", freq_pmse, "reproduces the fit's", freq_loss)

    # Borders: NO SUCH IDENTITY, and asserting one would be gating a
    # property the estimator does not have. The fit trained on the ORDERED
    # statistic over the CTR estimation permutation; the applied model
    # carries the FULL-LEARN-SET histogram
    # (`private/libs/algo/online_ctr.cpp:909-930`). What IS asserted is
    # that it scores at all -- the refusal this round lifted -- and that
    # the applied model still fits the target.
    var ctr_preds = predict_floats(ctx, ctr_model, x, n)
    var cse = Float64(0.0)
    for r in range(n):
        var d = Float64(ctr_preds[r]) - Float64(y[r])
        cse += d * d
    var ctr_pmse = cse / Float64(n)
    var gap = (ctr_pmse - ctr_loss) / ctr_loss
    print(
        "    default() (three Borders priors + FeatureFreq): applied mse",
        ctr_pmse, "against the fit's", ctr_loss, " gap", gap,
    )
    print(
        "      the gap is CatBoost's design: the fit trained on the"
        " ORDERED statistic, the applied model carries the"
        " FULL-LEARN-SET histogram"
    )
    if ctr_pmse / variance >= 0.05:
        failures.append(
            String("the applied Borders model does not fit the target:")
            + String(" mse ")
            + String(ctr_pmse)
            + String(" against variance ")
            + String(variance)
        )

    # --- the three refusals that remain ---------------------------------
    print("  refusals:")

    var both = List[Bool]()
    for f in range(CTRT_FEATURES):
        both.append(f == 0)
    var r2 = False
    try:
        _ = train(
            ctx, x, y, n, CTRT_FEATURES, n_estimators=1,
            one_hot=both, cat_features=cat_flags,
        )
    except:
        r2 = True
    _expect_raise(
        failures, r2, String("a feature in both cat_features and one_hot")
    )

    var constant = List[Float32]()
    for f in range(CTRT_FEATURES):
        for r in range(n):
            if f == 0:
                constant.append(Float32(0.0))
            else:
                constant.append(x[f * n + r])
    var r3 = False
    try:
        _ = train(
            ctx, constant, y, n, CTRT_FEATURES, n_estimators=1,
            cat_features=cat_flags,
        )
    except:
        r3 = True
    _expect_raise(
        failures, r3, String("a constant categorical column (their")
        + String(" 'useless catFeature found')")
    )

    # The public categorical surface promises exact dense integer codes.
    # Int(Float32) used to make 1.5 silently alias category 1, while a hole
    # made `max + 1` claim a category the fit never observed.
    var fractional = x.copy()
    fractional[0] = Float32(1.5)
    var r4 = False
    try:
        _ = train(
            ctx, fractional, y, n, CTRT_FEATURES, n_estimators=1,
            cat_features=cat_flags,
        )
    except:
        r4 = True
    _expect_raise(failures, r4, String("a fractional categorical code"))

    var sparse_codes = x.copy()
    for r in range(n):
        if sparse_codes[r] == Float32(1.0):
            sparse_codes[r] = Float32(500.0)
    var r5 = False
    try:
        _ = train(
            ctx, sparse_codes, y, n, CTRT_FEATURES, n_estimators=1,
            cat_features=cat_flags,
        )
    except:
        r5 = True
    _expect_raise(failures, r5, String("non-dense categorical codes"))

    # Apply shares the same exact-code validator. Unseen integer categories
    # remain valid and take the table's empty value; fractional aliases do not.
    var bad_apply = x.copy()
    bad_apply[0] = Float32(1.5)
    var r6 = False
    try:
        _ = predict_floats(ctx, freq_model, bad_apply, n)
    except:
        r6 = True
    _expect_raise(failures, r6, String("a fractional apply-time category"))

    # THE FOURTH REFUSAL IS GONE, AND ITS REMOVAL IS THE RESULT.
    #
    # `TCatFeatureParams.default()` used to raise here, because Borders is
    # permutation dependent and the CTR estimation permutation was not
    # ported. It is ported now (`gbdt/data/permutation.mojo`,
    # `archive/reference/PORTING.md` 55), so the same call TRAINS and emits FOUR columns per
    # categorical feature -- three Borders priors and one FeatureFreq --
    # which is CatBoost's own GPU `simple_ctr` default. The values, the
    # ordering and the permutation are gated by
    # `pixi run check-ctr-device`; what is asserted here is only that
    # `train()` takes the configuration at all.
    var borders_params = List[TCatFeatureParams]()
    borders_params.append(TCatFeatureParams.default())
    var borders_model = train(
        ctx, x, y, n, CTRT_FEATURES, n_estimators=1,
        cat_features=cat_flags, cat_feature_params=borders_params,
    )
    print(
        "    CatBoost's GPU default simple_ctr, passed explicitly: columns",
        len(borders_model.fold_counts), "of which ctr",
        borders_model.ctr_column_count,
    )
    if borders_model.ctr_column_count != 4:
        failures.append(
            String("their GPU simple_ctr default (three Borders priors plus")
            + String(" FeatureFreq) must give FOUR columns for one")
            + String(" categorical feature; got ")
            + String(borders_model.ctr_column_count)
        )
    # explicit and implicit must be the SAME configuration, or the fallback
    # is not what it says it is
    if borders_model.ctr_column_count != ctr_model.ctr_column_count:
        failures.append(
            String("passing TCatFeatureParams.default() explicitly gives ")
            + String(borders_model.ctr_column_count)
            + String(" CTR columns and passing nothing gives ")
            + String(ctr_model.ctr_column_count)
            + String("; the implicit fallback is supposed to BE default()")
        )

    if len(failures) > 0:
        var msg = String("train(cat_features=...) FAILED:")
        for i in range(len(failures)):
            msg += String("\n    ") + failures[i]
        raise Error(msg)
    print(
        "  the CTR column carries a signal no ordered split on the codes"
        " can reach, and every refusal fires"
    )


def main() raises:
    check_ctr_train()
