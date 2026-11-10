# SPDX-License-Identifier: Apache-2.0
"""Typed Ridge/OLS RUN checks with independent small-system expectations.

Eight Ridge cases preserve weighted/unweighted, scalar-weight, intercept and
one-feature coverage. Twelve OLS cases preserve rank deficiency, constant and
zero rank, column units, square and wide support/refusal. Native dyadic data
replaces NumPy RNG; OLS solutions are known by construction, not by another
call into the implementation. No Python objects, bindings or reference hashes.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.linear_fit_types import LinearFitRequest, LinearFitResult, LF_VENDOR, LF_MODE
from core.linear_fit_run import linear_ridge_run
from core.linear_ols_run import linear_ols_plan, linear_ols_run
from core.linear_boundary_types import LinearPredictRequest
from core.linear_boundary_run import linear_predict_run
from core.linear_model_state import LinearModelState
from core.linear_score_run import linear_score_run, linear_r2_run

comptime LINEAR_OWNER_FIXTURES = 23


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def close(actual: Float64, expected: Float64, rtol: Float64, atol: Float64, label: String) raises:
    require(abs(actual - expected) <= atol + rtol * abs(expected), label + ": " + String(actual) + " != " + String(expected))


struct LinearTestAPI(Copyable, Movable):
    var fit: NativeDescriptorCall
    var weighted: NativeDescriptorCall
    var predict: NativeDescriptorCall
    var decomp: NativeDescriptorCall
    var metrics: NativeDescriptorCall
    var vendor: Int
    var mode: Int

    def __init__(out self, fit: NativeDescriptorCall, weighted: NativeDescriptorCall,
        predict: NativeDescriptorCall, decomp: NativeDescriptorCall, metrics: NativeDescriptorCall) raises:
        self.fit = fit
        self.weighted = weighted
        self.predict = predict
        self.decomp = decomp
        self.metrics = metrics
        self.vendor = native_descriptor_call(fit, LF_VENDOR, [], [], [])
        self.mode = native_descriptor_call(fit, LF_MODE, [], [], [])


def _value(row: Int, column: Int, seed: Int) -> Float32:
    var h = UInt64(row + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(column + seed + 1) * UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    return Float32(Int(h & 31) - 16) * 0.125


def _fill(mut x: List[Float32], mut y: List[Float32], rows: Int, cols: Int,
    intercept: Bool, ridge: Bool, fixture_id: Int, seed: Int):
    var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(y.unsafe_ptr()))
    def row_task(row: Int) {imm xp, imm yp, imm cols, imm intercept, imm ridge, imm fixture_id, imm seed}:
        var target = Float64(1.25) if intercept else Float64(0)
        for col in range(cols):
            var value = _value(row, col, seed)
            if not ridge:
                if fixture_id == 6 and col == 3:
                    value = _value(row, 1, seed)
                elif fixture_id == 7 and col == 3:
                    value = 3
                elif fixture_id == 8:
                    value = 0
                elif fixture_id == 9:
                    var scale = Float32(1)
                    if col == 0:
                        scale = 0.015625
                    elif col == 2:
                        scale = 8
                    elif col == 3:
                        scale = 128
                    value *= scale
                elif fixture_id >= 10:
                    value = 1.0 if row == col else 0.0
            xp.unsafe_store(row * cols + col, value)
            target += Float64(value) * Float64(col + 1) * (Float64(0.5) if ridge else Float64(0.25))
        yp.unsafe_store(row, Float32(target))
    host_parallelize(row_task, rows)
    _ = x
    _ = y


def _ridge_oracle(x: List[Float32], y: List[Float32], weights: List[Float32],
    rows: Int, cols: Int, intercept: Bool) raises -> List[Float64]:
    """Independent weighted equations and partial-pivot Gaussian elimination.

Only 1x1/3x3 systems are solved here. Each fixture is independently scheduled;
the pivot/elimination sequence inside one tiny system is dependent work.
"""
    var mx = List[Float64](length=cols, fill=0)
    var my = Float64(0)
    var total = Float64(0)
    for row in range(rows):
        var weight = Float64(weights[row])
        total += weight
        my += weight * Float64(y[row])
        for col in range(cols):
            mx[col] += weight * Float64(x[row * cols + col])
    if intercept:
        my /= total
        for col in range(cols):
            mx[col] /= total
    else:
        my = 0
        for col in range(cols):
            mx[col] = 0
    var system = List[Float64](length=cols * (cols + 1), fill=0)
    for a in range(cols):
        for b in range(cols):
            var value = Float64(0.25) if a == b else Float64(0)
            for row in range(rows):
                value += Float64(weights[row]) * (Float64(x[row * cols + a]) - mx[a]) * (Float64(x[row * cols + b]) - mx[b])
            system[a * (cols + 1) + b] = value
        var rhs = Float64(0)
        for row in range(rows):
            rhs += Float64(weights[row]) * (Float64(x[row * cols + a]) - mx[a]) * (Float64(y[row]) - my)
        system[a * (cols + 1) + cols] = rhs
    for pivot in range(cols):
        var selected = pivot
        for row in range(pivot + 1, cols):
            if abs(system[row * (cols + 1) + pivot]) > abs(system[selected * (cols + 1) + pivot]):
                selected = row
        require(abs(system[selected * (cols + 1) + pivot]) > 1e-14, "independent Ridge system unexpectedly singular")
        if selected != pivot:
            for col in range(pivot, cols + 1):
                var old = system[pivot * (cols + 1) + col]
                system[pivot * (cols + 1) + col] = system[selected * (cols + 1) + col]
                system[selected * (cols + 1) + col] = old
        var scale = system[pivot * (cols + 1) + pivot]
        for col in range(pivot, cols + 1):
            system[pivot * (cols + 1) + col] /= scale
        for row in range(cols):
            if row != pivot:
                var factor = system[row * (cols + 1) + pivot]
                for col in range(pivot, cols + 1):
                    system[row * (cols + 1) + col] -= factor * system[pivot * (cols + 1) + col]
    var result = List[Float64](length=cols + 1, fill=0)
    var bias = my
    for col in range(cols):
        result[col] = system[col * (cols + 1) + cols]
        bias -= mx[col] * result[col]
    result[cols] = bias
    return result^


def _model(coef: Int, mean: Int, cols: Int, result: LinearFitResult) -> LinearModelState:
    var state = LinearModelState()
    state.coef = coef
    state.coef_count = cols
    state.features = cols
    state.intercept = result.intercept
    state.mean = mean
    state.ymean = result.ymean
    state.has_coef = True
    state.fitted = True
    return state^


def _fit_fixture(api: LinearTestAPI, fixture_id: Int) raises:
    var ridge = fixture_id < 8
    var cid = fixture_id if ridge else fixture_id - 8
    var cols = 3 if ridge else 4
    var rows = 24 if ridge else 29
    var intercept = True
    var weight_mode = 0
    if ridge:
        if cid == 2 or cid == 3 or cid == 7:
            cols = 1
        intercept = cid != 1 and cid != 3 and cid != 5
        weight_mode = 2 if cid == 6 else (1 if cid == 4 or cid == 5 or cid == 7 else 0)
    else:
        if cid == 2:
            cols = 1
        if cid >= 10:
            rows = 2
            cols = 2 if cid == 10 else 3
        intercept = cid != 1 and cid != 4 and cid < 10
        weight_mode = 2 if cid == 5 else (1 if cid == 3 or cid == 4 else 0)
    var x = List[Float32](length=rows * cols, fill=0)
    var y = List[Float32](length=rows, fill=0)
    _fill(x, y, rows, cols, intercept, ridge, cid, 37)
    var original_x = x.copy()
    var original_y = y.copy()
    var weights = List[Float32](length=rows, fill=1)
    for row in range(rows):
        # Vector weights include a zero first row. Dyadic values replace the
        # Python fixture's linspace; the independent equations use these words.
        weights[row] = Float32(row % 9) * 0.25 if weight_mode == 1 else Float32(2 if weight_mode == 2 else 1)
    var original_weights = weights.copy()
    var coef = List[Float32](length=cols + 2, fill=-999)
    var mean = List[Float32](length=cols + 2, fill=0)
    mean[0] = -999
    mean[cols + 1] = -999
    var ca = Int(coef.unsafe_ptr()) + 4
    var ma = Int(mean.unsafe_ptr()) + 4
    var request = LinearFitRequest(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()),
        Int(weights.unsafe_ptr()) if weight_mode else 0, ca, ma, rows, cols, intercept, 0.25 if ridge else 0.0)
    var result = LinearFitResult()
    var refused = False
    try:
        if ridge:
            result = linear_ridge_run(api.fit, api.weighted, request)
        else:
            var plan = linear_ols_plan(api.fit, rows, cols, "LinearRegression")
            result = linear_ols_run(api.fit, api.decomp, request, plan, api.vendor)
    except error:
        if ridge or cid != 11 or api.vendor != 0:
            raise error
        refused = String(error).find("lstsq_min_norm") >= 0 and String(error).find("no CPU restatement") >= 0
        require(refused, "wide CPU OLS refused for an unexpected reason: " + String(error))
    for i in range(rows * cols):
        require(bitcast[DType.uint32](x[i]) == bitcast[DType.uint32](original_x[i]), "fit mutated X")
    for i in range(rows):
        require(bitcast[DType.uint32](y[i]) == bitcast[DType.uint32](original_y[i]) and weights[i] == original_weights[i], "fit mutated target/weights")
    require(coef[0] == -999 and coef[cols + 1] == -999 and mean[0] == -999 and mean[cols + 1] == -999, "fit escaped output spans")
    if not ridge and cid == 11 and api.vendor == 0:
        require(refused, "wide CPU OLS support contract changed")
        return
    var oracle = List[Float64](length=cols + 1, fill=0)
    if ridge:
        oracle = _ridge_oracle(original_x, original_y, weights, rows, cols, intercept)
        for col in range(cols):
            close(Float64(coef[col + 1]), oracle[col], 3e-4, 3e-5, "Ridge coefficient")
        close(result.intercept, oracle[cols], 3e-4, 3e-5, "Ridge intercept")
    elif cid == 11:
        var expected: List[Float64] = [0.25, 0.5, 0]
        for col in range(cols):
            close(Float64(coef[col + 1]), expected[col], 5e-4, 3e-4, "wide OLS minimum-norm coefficient")
    var query_rows = 7 if cid < 10 or ridge else 2
    var query = List[Float32](length=query_rows * cols, fill=0)
    var known = List[Float32](length=query_rows, fill=0)
    _fill(query, known, query_rows, cols, intercept, ridge, cid, 91)
    var narrow = List[Float32](length=query_rows + 2, fill=-999)
    var wide = List[Float64](length=query_rows + 2, fill=-999)
    linear_predict_run(api.predict, LinearPredictRequest(Int(query.unsafe_ptr()), ca,
        Int(narrow.unsafe_ptr()) + 4, query_rows, cols, result.intercept, False))
    linear_predict_run(api.predict, LinearPredictRequest(Int(query.unsafe_ptr()), ca,
        Int(wide.unsafe_ptr()) + 8, query_rows, cols, result.intercept, True))
    for row in range(query_rows):
        var wanted = Float64(known[row])
        if ridge:
            wanted = oracle[cols]
            for col in range(cols):
                wanted += Float64(query[row * cols + col]) * oracle[col]
        close(Float64(narrow[row + 1]), wanted, 3e-4 if ridge else 5e-4, 3e-5 if ridge else 3e-4, "native fit/predict solution")
        close(wide[row + 1], wanted, 3e-4 if ridge else 5e-4, 3e-5 if ridge else 3e-4, "wide native fit/predict solution")
    require(narrow[0] == -999 and narrow[query_rows + 1] == -999 and wide[0] == -999 and wide[query_rows + 1] == -999, "prediction escaped output spans")
    # Fit-derived predictions on training data also retain the original
    # numerical check. Held-out points above prevent training-only overfit.
    var training = List[Float32](length=rows, fill=0)
    linear_predict_run(api.predict, LinearPredictRequest(Int(x.unsafe_ptr()), ca,
        Int(training.unsafe_ptr()), rows, cols, result.intercept, False))
    for row in range(rows):
        var wanted = Float64(y[row])
        if ridge:
            wanted = oracle[cols]
            for col in range(cols):
                wanted += Float64(x[row * cols + col]) * oracle[col]
        close(Float64(training[row]), wanted, 3e-4 if ridge else 5e-4, 3e-5 if ridge else 3e-4, "native training prediction")
    if not ridge:
        var state = _model(ca, ma, cols, result)
        var scored = linear_score_run(api.predict, api.metrics, state, Int(x.unsafe_ptr()),
            rows, cols, Int(y.unsafe_ptr()), rows, api.vendor, api.mode, "LinearRegression")
        var target_mean = Float64(0)
        for row in range(rows):
            target_mean += Float64(y[row])
        target_mean /= Float64(rows)
        var centered = Float64(0)
        var error_bound = Float64(0)
        for row in range(rows):
            var delta = Float64(y[row]) - target_mean
            centered += delta * delta
            # Derive the score bound from this fixture's original per-value
            # prediction tolerance. No independent score tolerance is widened.
            var tolerance = 3e-4 + 5e-4 * abs(Float64(y[row]))
            error_bound += tolerance * tolerance
        if centered == 0:
            require(scored == 0, "fitted constant-target score convention")
        else:
            close(scored, 1, 0, error_bound / centered + 1e-12, "fitted model score")
    # Request descriptors store integer addresses; keep their native owners
    # alive after every fit/predict/score call, not only after pointer creation.
    _ = coef
    _ = mean
    _ = x
    _ = y
    _ = weights
    _ = query
    _ = narrow
    _ = wide
    _ = training


def _score_fixture(api: LinearTestAPI, fixture_id: Int) raises:
    var x: List[Float32] = [0, 0, 1, 0, 0, 1, 1, 1]
    var coef: List[Float32] = [0.5, 1]
    var target: List[Float32] = [0.25, 0.75, 1.75, 1.75]
    var prediction: List[Float32] = [0.25, 0.75, 1.25, 1.75]
    var state = _model(Int(coef.unsafe_ptr()), 0, 2, LinearFitResult(0.25))
    if fixture_id == 0:
        # Residual sum = 1/4, centered target sum = 27/16, R2 = 23/27.
        var actual = linear_score_run(api.predict, api.metrics, state, Int(x.unsafe_ptr()),
            4, 2, Int(target.unsafe_ptr()), 4, api.vendor, api.mode, "Ridge")
        close(actual, 23.0 / 27.0, 0, 1e-14, "fit-free native score closed form")
        var direct = linear_r2_run(api.metrics, Int(prediction.unsafe_ptr()), 4,
            Int(target.unsafe_ptr()), 4, api.vendor, api.mode)
        close(direct, 23.0 / 27.0, 0, 1e-14, "prediction-buffer score closed form")
    elif fixture_id == 1:
        # Preserve the existing score API's zero-denominator convention.
        target = [2, 2, 2, 2]
        var actual = linear_r2_run(api.metrics, Int(prediction.unsafe_ptr()), 4,
            Int(target.unsafe_ptr()), 4, api.vendor, api.mode)
        require(actual == 0, "constant target R2 convention")
    else:
        var refused = False
        try:
            _ = linear_score_run(api.predict, api.metrics, state, Int(x.unsafe_ptr()),
                4, 2, Int(target.unsafe_ptr()), 3, api.vendor, api.mode, "Ridge")
        except error:
            refused = String(error).find("different shapes (3, 1) and (4, 1)") >= 0
        require(refused, "score target extent mismatch accepted")
    require(coef[0] == 0.5 and coef[1] == 1 and prediction[0] == 0.25 and prediction[3] == 1.75, "score mutated borrowed state/predictions")
    require(x[0] == 0 and x[1] == 0 and x[6] == 1 and x[7] == 1, "score mutated input")
    _ = target


def run_linear_fixture(api: LinearTestAPI, fixture_id: Int) raises:
    if fixture_id < 20:
        _fit_fixture(api, fixture_id)
    else:
        _score_fixture(api, fixture_id - 20)
