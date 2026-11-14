# SPDX-License-Identifier: Apache-2.0
"""RobustScaler fit_transform and transform programs against an independent
NaN-ignoring percentile oracle.

Ported from python/mojolearn/tests/test_robust_scaler_native_lifecycle.py
(`test_public_fit_and_transform_keep_quantiles_nan_and_constant_columns`).
The fit program is the typed RUN `x_prep.robust_native_run._robust_fit_program`
on the arena layout `bindings/robust_native_pack.mojo::robust_fit` builds for
the host column in identical mode (no radix select, no unit variance,
quantile_range (25, 75), fused fit_transform, no separate work region); the
transform program is the affine stage 3 that `robust_transform` stages. Both
run through the x_prep host program runner.

Original shape and contract: 19 and 2049 rows (one and two 2048-row blocks),
5 features, feature 0 NaN every third row, feature 1 all NaN, feature 4
constant 3; blocked statistics on and off; every centering/scaling pair. The
oracle is NumPy's nanpercentile (linear interpolation over the observed
values), zero for a feature with no observed value, and a scale below
10 * float32 eps replaced by one. Tolerances are the original rtol = atol =
2e-6; NaN positions, the untouched input and fit_transform == transform bits
are exact. Input values come from a seeded native generator (the oracle is
computed from the same values, so NumPy's PCG64 stream is not needed).

The 16 fixtures are independent units and run across host cores; each
fixture's per-feature oracle is serial over at most 2049 values.
"""
from std.math import floor, isnan
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.preprocessing_program import prep_program_stage
from x_prep.host.program import run_program_host
from x_prep.robust_native_run import _robust_fit_program

comptime ROBUST_FIXTURES = 16
comptime FEATURES = 5
comptime TOLERANCE = 2e-6


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _mix(seed: UInt64) -> UInt64:
    var z = seed + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> UInt64(31))


def _unit(seed: UInt64) -> Float64:
    return Float64(_mix(seed) >> UInt64(11)) / 9007199254740992.0


def robust_data(rows: Int) -> List[Float32]:
    """The original fixture's shape: roughly centered values (a sum of three
    seeded uniforms), feature 4 constant 3, feature 0 NaN on every third row
    and feature 1 NaN everywhere."""
    var x = List[Float32](length=rows * FEATURES, fill=0)
    for r in range(rows):  # small-loop(fixture: at most 2049 rows of seeded input)
        for c in range(FEATURES):  # small-loop(fixture: five features)
            var s = UInt64(741) * UInt64(1000003) + UInt64(r * FEATURES + c) * UInt64(3)
            x[r * FEATURES + c] = Float32((_unit(s) + _unit(s + 1) + _unit(s + 2) - 1.5) * 2.0)
        x[r * FEATURES + 4] = 3
        x[r * FEATURES + 1] = bitcast[DType.float32](UInt32(0x7FC00000))
        if r % 3 == 0:
            x[r * FEATURES + 0] = bitcast[DType.float32](UInt32(0x7FC00000))
    return x^


def nan_percentiles(x: List[Float32], rows: Int, c: Int) -> List[Float64]:
    """NumPy nanpercentile [25, 50, 75] of feature c (linear interpolation over
    the sorted observed values); zeros when the feature has none."""
    var values = List[Float64]()
    for r in range(rows):  # small-loop(oracle: one feature's observed values)
        var v = x[r * FEATURES + c]
        if not isnan(v):
            values.append(Float64(v))
    var result = List[Float64](length=3, fill=0)
    var m = len(values)
    if m == 0:
        return result^
    for i in range(1, m):  # small-loop(oracle: insertion sort of at most 2049 values)
        var v = values[i]
        var j = i - 1
        while j >= 0 and values[j] > v:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = v
    var fractions: List[Float64] = [0.25, 0.5, 0.75]
    for k in range(3):  # small-loop(oracle: three quantiles)
        var index = fractions[k] * Float64(m - 1)
        var lo = Int(floor(index))
        var hi = min(lo + 1, m - 1)
        result[k] = values[lo] + (values[hi] - values[lo]) * (index - Float64(lo))
    return result^


def close(actual: Float32, expected: Float32, label: String) raises:
    var a = Float64(actual)
    var e = Float64(expected)
    require(abs(a - e) <= TOLERANCE + TOLERANCE * abs(e),
        label + ": " + String(actual) + " expected " + String(expected))


def fixture(unit: Int) raises:
    var rows = 19 if unit < 8 else 2049
    var blocked = (unit // 4) % 2 == 1
    var centering = (unit % 4) // 2 == 1
    var scaling = unit % 2 == 1
    var label = String("rows ") + String(rows) + " blocked " + String(blocked) + " center " + String(centering) + " scale " + String(scaling)
    var n = rows
    var d = FEATURES
    var x = robust_data(rows)
    # The fit arena of robust_fit (host column, identical, no work region).
    var blocks = (n + 2047) // 2048
    var part_words = 5 * blocks * d if blocked else 0
    var sorted = n * d
    var stats = n * d + n * d
    var fractions = stats + 6 * d
    var q = fractions + 3
    var center = q + 3 * d
    var scale = center + d
    var host = scale + d + part_words
    var dest = host
    host += n * d
    var part = scale + d
    var arena = List[Float32](length=host, fill=0)
    for i in range(n * d):  # small-loop(fixture: the input rows into the arena)
        arena[i] = x[i]
    arena[fractions] = Float32(25.0 / 100.0)
    arena[fractions + 1] = 0.5
    arena[fractions + 2] = Float32(75.0 / 100.0)
    var program = List[Int32]()
    _robust_fit_program(program, False, blocked, False, True, centering, scaling, n, d, blocks,
        sorted, part, stats, fractions, q, center, scale, dest)
    run_program_host(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)
    for i in range(n * d):  # small-loop(fixture: the input is not written)
        require(bitcast[DType.uint32](arena[i]) == bitcast[DType.uint32](x[i]), label + ": fit wrote its input")
    # Independent statistics.
    var expected_center = List[Float32](length=d, fill=0)
    var expected_scale = List[Float32](length=d, fill=1)
    for c in range(d):  # small-loop(oracle: five features)
        var quantiles = nan_percentiles(x, n, c)
        expected_center[c] = Float32(quantiles[1])
        var s = Float32(quantiles[2]) - Float32(quantiles[0])
        expected_scale[c] = Float32(1) if s < Float32(10) * bitcast[DType.float32](UInt32(0x34000000)) else s
        if centering:
            close(arena[center + c], expected_center[c], label + ": center_ feature " + String(c))
        if scaling:
            close(arena[scale + c], expected_scale[c], label + ": scale_ feature " + String(c))
    for i in range(n * d):  # small-loop(oracle: the scaled rows)
        var c = i % d
        var e = x[i]
        if centering:
            e = e - expected_center[c]
        if scaling:
            e = e / expected_scale[c]
        var a = arena[dest + i]
        require(isnan(a) == isnan(x[i]), label + ": NaN position " + String(i))
        if not isnan(e):
            close(a, e, label + ": fit_transform word " + String(i))
    # The transform program of robust_transform on the fitted vectors.
    var center_at = n * d if centering else -1
    var scale_at = n * d + (d if centering else 0) if scaling else -1
    var tdest = n * d + (d if centering else 0) + (d if scaling else 0)
    var tarena = List[Float32](length=tdest + n * d, fill=0)
    for i in range(n * d):  # small-loop(fixture: the input rows into the arena)
        tarena[i] = x[i]
    for c in range(d):  # small-loop(fixture: the fitted vectors)
        if centering:
            tarena[center_at + c] = arena[center + c]
        if scaling:
            tarena[scale_at + c] = arena[scale + c]
    var tprogram = List[Int32]()
    prep_program_stage(tprogram, 3, n * d, 0, n * d, d, center_at, scale_at, tdest)
    run_program_host(Int(tarena.unsafe_ptr()), len(tarena), Int(tprogram.unsafe_ptr()), 1)
    for i in range(n * d):  # small-loop(fixture: fit_transform == transform, bit for bit)
        require(bitcast[DType.uint32](tarena[tdest + i]) == bitcast[DType.uint32](arena[dest + i]),
            label + ": transform differs from fit_transform at " + String(i))
    _ = arena^
    _ = tarena^
    _ = program^
    _ = tprogram^
    _ = x^


def run_checks() raises:
    var failures = List[Int](length=ROBUST_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL prep_robust_scaler_native_lifecycle fixture", unit, String(error))
    host_parallelize(task, ROBUST_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures
    require(failed == 0, "prep_robust_scaler_native_lifecycle failed " + String(failed) + " fixtures")
    print("PASS prep_robust_scaler_native_lifecycle:", ROBUST_FIXTURES, "fit_transform/transform programs")
