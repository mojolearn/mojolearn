"""boost_from_average against CatBoost ITSELF, bit for bit.

The port's whole claim is that `gbdt/metrics/optimal_const_for_loss.mojo`
computes THEIR `CalcOptimumConstApprox` -- so the gate is their own output:
`model.get_scale_and_bias()[1]` from a real CatBoost fit, compared against
the bias our python surface reports, on hashed fixtures for every ported
arm and for both sides of their data-dependent default.

WHAT EXACTLY IS COMPARED. CatBoost's bias is a float64; ours is a float64
parsed from the model text's bits half. The demand is `==` on the bits for
the RMSE arms and for Logloss. (Their pipeline computes the double average
through a FLOAT return -- `optimal_const_for_loss.h`'s
`CalculateWeightedTargetAverage` is `inline float` -- and the port carries
the same truncation, which is what makes bit equality possible at all.)

THE DEFAULT RULE IS GATED TOO: an UNSET option must come out non-zero for
RMSE (their `AdjustBoostFromAverageDefaultValue` list) and ZERO for Logloss
(NOT on their list -- the 2026-08-22 higgs misread this file exists to
prevent recurring).

THE SABOTAGE is the weighted arm run with the weights DROPPED from the
expectation: it must disagree, or the weighted path is decorative.

Run: pixi run -e bench check-bfa-oracle   (a GPU; CatBoost; seconds)
"""

import struct
import sys

import numpy as np

sys.path.insert(0, "python")

import catboost
import mojolearn


def splitmix(n, salt):
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        h = (np.uint64(i * 1000003 + salt) * np.uint64(0x9E3779B97F4A7C15)) \
            & np.uint64(0xFFFFFFFFFFFFFFFF)
        h = ((h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) \
            & np.uint64(0xFFFFFFFFFFFFFFFF)
        out[i] = float(h % np.uint64(10000)) / 10000.0
    return out


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def cat_bias(loss, y, w, bfa):
    kw = dict(
        iterations=1, depth=2, learning_rate=0.1, loss_function=loss,
        verbose=False, allow_writing_files=False, thread_count=2,
    )
    if bfa is not None:
        kw["boost_from_average"] = bfa
    m = catboost.CatBoost(kw)
    X = np.stack([splitmix(len(y), 7), splitmix(len(y), 11)], axis=1)
    m.fit(X.astype(np.float32), y, sample_weight=w)
    return m.get_scale_and_bias()[1]


def our_bias(loss, y, w, bfa):
    X = np.stack([splitmix(len(y), 7), splitmix(len(y), 11)], axis=1)
    kw = dict(n_estimators=1, max_depth=2, learning_rate=0.1)
    if bfa is not None:
        kw["boost_from_average"] = bfa
    m = mojolearn.GradientBoosting(loss, **kw)
    # sample weights ride through fit
    m.fit(X.astype(np.float32), y.astype(np.float32),
          sample_weight=None if w is None else w.astype(np.float32))
    return m.bias_


def main():
    n = 4096
    y_reg = (2000.0 + 10.0 * splitmix(n, 3)).astype(np.float32)
    y_bin = (splitmix(n, 5) > 0.6).astype(np.float32)
    w = (0.5 + splitmix(n, 13)).astype(np.float32)

    failures = 0

    cases = [
        ("RMSE explicit true, unweighted", "RMSE", y_reg, None, True),
        ("RMSE explicit true, weighted", "RMSE", y_reg, w, True),
        ("RMSE unset (their auto-TRUE)", "RMSE", y_reg, None, None),
        ("Logloss explicit true, unweighted", "Logloss", y_bin, None, True),
        ("Logloss explicit true, weighted", "Logloss", y_bin, w, True),
    ]
    for name, loss, y, wt, bfa in cases:
        theirs = cat_bias(loss, y, wt, bfa)
        ours = our_bias(loss, y, wt, bfa)
        same = bits(theirs) == bits(ours)
        print(f"  {name}: catboost {theirs!r} ours {ours!r} "
              f"{'BIT-EQUAL' if same else 'DIFFER'}")
        if not same:
            failures += 1

    # the default rule's OTHER side: unset Logloss must be zero on both
    theirs0 = cat_bias("Logloss", y_bin, None, None)
    ours0 = our_bias("Logloss", y_bin, None, None)
    print(f"  Logloss unset: catboost {theirs0!r} ours {ours0!r} "
          f"(both must be 0.0)")
    if theirs0 != 0.0 or ours0 != 0.0:
        failures += 1

    # THE SABOTAGE: the weighted expectation recomputed WITHOUT weights
    # must disagree with the weighted run, or weights never reached the
    # average.
    with_w = our_bias("RMSE", y_reg, w, True)
    without_w = our_bias("RMSE", y_reg, None, True)
    print(f"  SABOTAGE weighted-vs-unweighted: {with_w!r} vs {without_w!r}")
    if bits(with_w) == bits(without_w):
        print("  FAIL: dropping the weights moved nothing; the weighted"
              " path is decorative")
        failures += 1

    if failures:
        raise SystemExit(f"check-bfa-oracle: {failures} case(s) failed")
    print("  ok   every ported arm matches CatBoost's own bias to the bit,"
          " and the default rule matches their adjust list")


if __name__ == "__main__":
    main()
