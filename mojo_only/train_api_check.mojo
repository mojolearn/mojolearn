"""The train-from-raw-floats surface, end to end.

Three claims, each with a failure mode this check would catch:
1. LEARNING: `train(X, y)` on data with real signal must cut the loss by
   an order of magnitude over 20 trees -- a broken grid, a broken
   quantize launch, or a broken fit all leave the loss near the
   variance.
2. CONSISTENCY: `predict_floats` on the TRAINING rows must reproduce the
   fit's final loss -- the model's grid, the quantize kernel and the
   apply must agree with what training built (this is where a borders
   layout mismatch or a wrong one-hot synthetic grid dies).
3. GENERALIZATION SHAPE: predictions on held-out rows drawn from the
   same signal must beat predicting the mean -- quantization against the
   STORED grid has to work on rows the grid never saw.
Plus the one-hot path through the same surface: a categorical column
declared through the `one_hot` flags must train and predict without a
hand-built grid.
"""

from max.gpu.host import DeviceContext

from gbdt.train import TrainedModel, predict_floats, train


def check_train_api() raises:
    print("train-from-raw-floats surface:")
    var ctx = DeviceContext()
    var n = 4096
    var n_test = 1024
    var f = 8

    # colmajor X: feature 0..6 pseudo-normal-ish signal carriers, y a
    # known function of two of them; feature 7 a 3-category code
    var x = List[Float32]()
    var xt = List[Float32]()
    for feat in range(f):
        for r in range(n + n_test):
            var v: Float32
            if feat == 7:
                v = Float32((r * 7 + feat) % 3)
            else:
                var h = (r * 2654435761 + feat * 97003) % 10000
                v = Float32(h) / Float32(10000.0) - Float32(0.5)
            if r < n:
                x.append(v)
            else:
                xt.append(v)
    # reindex: the loops above appended train rows and test rows into the
    # same per-feature stretch in order, so x is [feat][0..n) colmajor and
    # xt is [feat][0..n_test) colmajor already.
    var y = List[Float32]()
    var yt = List[Float32]()
    for r in range(n + n_test):
        var f0: Float32
        var f3: Float32
        var c: Float32
        if r < n:
            f0 = x[0 * n + r]
            f3 = x[3 * n + r]
            c = x[7 * n + r]
        else:
            f0 = xt[0 * n_test + (r - n)]
            f3 = xt[3 * n_test + (r - n)]
            c = xt[7 * n_test + (r - n)]
        var target = Float32(3.0) * f0 - Float32(2.0) * f3 + (
            Float32(1.0) if c == Float32(1.0) else Float32(0.0)
        )
        if r < n:
            y.append(target)
        else:
            yt.append(target)

    var one_hot = List[Bool]()
    for feat in range(f):
        one_hot.append(feat == 7)

    var tm = train(
        ctx, x, y, n, f,
        border_count=32,
        n_estimators=20,
        max_depth=4,
        learning_rate=Float32(0.3),
        one_hot=one_hot,
    )
    var first = tm.losses[0]
    var last = tm.losses[len(tm.losses) - 1]
    if not (last < first / 10.0):
        raise Error("train did not learn: " + String(first) + " -> "
                    + String(last))

    var preds = predict_floats(ctx, tm, x, n)
    var se = Float64(0.0)
    for r in range(n):
        var d = Float64(preds[r]) - Float64(y[r])
        se += d * d
    var pmse = se / Float64(n)
    var drift = pmse - last
    if drift < 0:
        drift = -drift
    if drift > 1e-9 + 1e-5 * last:
        raise Error("predict_floats does not reproduce the fit: "
                    + String(pmse) + " vs " + String(last))

    var tpreds = predict_floats(ctx, tm, xt, n_test)
    var tse = Float64(0.0)
    var mean = Float64(0.0)
    for r in range(n_test):
        mean += Float64(yt[r])
    mean /= Float64(n_test)
    var vse = Float64(0.0)
    for r in range(n_test):
        var d = Float64(tpreds[r]) - Float64(yt[r])
        tse += d * d
        var dv = Float64(yt[r]) - mean
        vse += dv * dv
    var test_mse = tse / Float64(n_test)
    var test_var = vse / Float64(n_test)
    if not (test_mse < test_var / 2.0):
        raise Error("held-out predictions do not beat the mean: mse "
                    + String(test_mse) + " vs variance " + String(test_var))
    print(
        "  learned", first, "->", last, "; train-predict consistent (",
        pmse, "); held-out mse", test_mse, "vs variance", test_var,
        "-- grid, quantize, one-hot column and apply agree end to end",
    )


def main() raises:
    # STANDALONE DRIVER, the last call `probe_main.mojo` makes.
    check_train_api()
