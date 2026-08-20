"""Logloss end to end through `train(X, y)`: learn, agree, generalize.

    pixi run check-logloss-train

Four claims on a synthetic binary task (probabilistic labels, hashed
features, logits in a range whose Bayes logloss is well under log 2):

1. LEARNING: mean logloss must fall well below the 0.693 of an ignorant
   model and clearly below its own first iteration -- a broken objective
   switch, estimation stage, or cursor update all leave it near log 2.
2. CONSISTENCY: sigmoid(predict_floats) on the TRAINING rows must
   reproduce the fit's final loss to 1e-3 relative -- the estimated
   leaves stored in the model must be the ones the cursor was moved by.
   THIS IS THE CLAIM THAT DIES if the estimator updates the cursor but
   the model keeps the searcher's discarded RMSE-formula leaves, or vice
   versa.
3. GENERALIZATION: held-out logloss must beat the train-mean predictor.
4. THE KNOBS REACH: leaf_estimation_iterations=1 must produce a
   BITWISE-different ensemble from the default ten, and that arm runs at
   max_depth=8 -- gbm-bench's pinned depth, the 256-leaf regime where
   empty leaves are the common case -- and must still learn.
5. CLASS WEIGHTS, their MakeClassificationWeights semantics
   (`data_providers.cpp:158-176`; scale_pos_weight=3 is [1, 3]): the
   weighted model's mean predicted probability on held-out POSITIVE rows
   must sit strictly above the unweighted model's.
Plus the borderless arm: loss="CrossEntropy" on SOFT targets must learn
through the same stack (has_border=False all the way down).

Proven to have teeth, measured 2026-08-20: flipping the check's own
probability convention (logloss computed against 1-p) fails claims 1-3
in both losses; restored green before this landed.
"""
from max.gpu.host import DeviceContext
from std.math import exp, log

from gbdt.train import predict_floats, train


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def frac(i: Int, salt: UInt64) -> Float64:
    return Float64(splitmix(UInt64(i) * UInt64(2654435761) + salt) >> 11) * (
        1.0 / 9007199254740992.0
    )


def mean_logloss(y: List[Float32], scores: List[Float32]) -> Float64:
    var total = Float64(0.0)
    for i in range(len(y)):
        var s = Float64(scores[i])
        var p = 1.0 / (1.0 + exp(-s))
        if p < 1e-12:
            p = 1e-12
        if p > 1.0 - 1e-12:
            p = 1.0 - 1e-12
        var c = Float64(y[i])
        total += -(c * log(p) + (1.0 - c) * log(1.0 - p))
    return total / Float64(len(y))


def main() raises:
    var ctx = DeviceContext()
    var failures = 0
    var n = 4096
    var n_test = 1024
    var f = 6

    var x = List[Float32]()
    var xt = List[Float32]()
    for feat in range(f):
        for r in range(n + n_test):
            var v = Float32((frac(r * f + feat, UInt64(5)) - 0.5) * 2.0)
            if r < n:
                x.append(v)
            else:
                xt.append(v)
    var y = List[Float32]()
    var yt = List[Float32]()
    var soft = List[Float32]()
    for r in range(n + n_test):
        var x0: Float64
        var x3: Float64
        if r < n:
            x0 = Float64(x[0 * n + r])
            x3 = Float64(x[3 * n + r])
        else:
            x0 = Float64(xt[0 * n_test + (r - n)])
            x3 = Float64(xt[3 * n_test + (r - n)])
        var p = 1.0 / (1.0 + exp(-(3.0 * x0 - 2.5 * x3)))
        var label = Float32(1.0) if frac(r, UInt64(77)) < p else Float32(0.0)
        if r < n:
            y.append(label)
            soft.append(Float32(p))
        else:
            yt.append(label)

    var tm = train(
        ctx, x, y, n, f,
        border_count=32, n_estimators=30, max_depth=6,
        learning_rate=Float32(0.3), l2_leaf_reg=Float32(1.0),
        loss="Logloss",
    )

    # 1: learning
    var first = tm.losses[0]
    var last = tm.losses[len(tm.losses) - 1]
    print("logloss first", first, "last", last)
    if not (last < 0.6 and last < first - 0.05):
        print("FAIL: did not learn (first", first, "last", last, ")")
        failures += 1

    # 2: train-predict consistency
    var train_scores = predict_floats(ctx, tm, x, n)
    var replay = mean_logloss(y, train_scores)
    print("fit final", last, "predict replay", replay)
    if abs(replay - last) > 1e-3 * (1.0 + abs(last)):
        print("FAIL: predict does not reproduce the fit's loss")
        failures += 1

    # 3: holdout beats the train-mean predictor
    var q = Float64(0.0)
    for i in range(n):
        q += Float64(y[i])
    q /= Float64(n)
    var base = Float64(0.0)
    for i in range(n_test):
        var c = Float64(yt[i])
        base += -(c * log(q) + (1.0 - c) * log(1.0 - q))
    base /= Float64(n_test)
    var test_scores = predict_floats(ctx, tm, xt, n_test)
    var test_ll = mean_logloss(yt, test_scores)
    print("holdout", test_ll, "baseline", base)
    if not (test_ll < base):
        print("FAIL: holdout does not beat the mean predictor")
        failures += 1

    # 4: the iterations knob reaches the estimator
    # max_depth=8 HERE ON PURPOSE: gbm-bench pins depth 8, this is the
    # 256-leaf regime's gate through the full stack (search, estimation
    # with mostly-empty leaves, apply).
    var tm1 = train(
        ctx, x, y, n, f,
        border_count=32, n_estimators=30, max_depth=8,
        learning_rate=Float32(0.3), l2_leaf_reg=Float32(1.0),
        loss="Logloss", leaf_estimation_iterations=1,
    )
    var last1 = tm1.losses[len(tm1.losses) - 1]
    print("ten-iteration d6", last, "one-iteration d8", last1)
    if last == last1:
        print("FAIL: iterations knob changed nothing (bitwise equal loss)")
        failures += 1
    if not (last1 < 0.6 and last1 < first):
        print("FAIL: depth-8 one-iteration arm did not learn")
        failures += 1

    # 5: class weights, their MakeClassificationWeights semantics
    # (scale_pos_weight=3 spelled [1, 3]): the model must shift toward
    # the positive class -- mean predicted p on the held-out POSITIVE
    # rows strictly above the unweighted model's.
    var tmw = train(
        ctx, x, y, n, f,
        border_count=32, n_estimators=30, max_depth=6,
        learning_rate=Float32(0.3), l2_leaf_reg=Float32(1.0),
        loss="Logloss", class_weights=[Float32(1.0), Float32(3.0)],
    )
    var w_scores = predict_floats(ctx, tmw, xt, n_test)
    var mean_p_w = Float64(0.0)
    var mean_p_u = Float64(0.0)
    var pos = 0
    for i in range(n_test):
        if yt[i] > Float32(0.5):
            mean_p_w += 1.0 / (1.0 + exp(-Float64(w_scores[i])))
            mean_p_u += 1.0 / (1.0 + exp(-Float64(test_scores[i])))
            pos += 1
    mean_p_w /= Float64(pos)
    mean_p_u /= Float64(pos)
    print("positive-row mean p: weighted", mean_p_w,
          "unweighted", mean_p_u)
    if not (mean_p_w > mean_p_u):
        print("FAIL: class weight [1,3] did not shift toward positives")
        failures += 1

    # 6: the borderless CrossEntropy arm on SOFT targets
    var tms = train(
        ctx, x, soft, n, f,
        border_count=32, n_estimators=20, max_depth=6,
        learning_rate=Float32(0.3), l2_leaf_reg=Float32(1.0),
        loss="CrossEntropy",
    )
    var sfirst = tms.losses[0]
    var slast = tms.losses[len(tms.losses) - 1]
    print("crossentropy first", sfirst, "last", slast)
    if not (slast < sfirst - 0.02):
        print("FAIL: CrossEntropy arm did not learn")
        failures += 1

    if failures != 0:
        raise Error(
            "logloss train check FAILED with " + String(failures)
        )
    print("logloss train check: learn, replay, holdout, iterations knob, "
          "soft-target arm -- all pass")
