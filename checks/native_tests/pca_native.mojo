# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Typed PCA fit/query correctness using original fixtures and external oracles.

No SVD implementation, output hash, repeated fit comparison or timing. The
recorded spectra come from independent NumPy Float64 SVD. Query equations
consume the actual fitted state, so eigenvector signs need no convention.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.sys.compile import is_defined
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.pca_native_api import PCANativeDriver, PCAFoldStore, PCAFitRequest, PCAQueryRequest
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.pca_native_host import pca_host_native_v1
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.pca_native_device import pca_device_native_v1
from core.msel_device_x import msel_fold_store, msel_device_x_context
from core.pca_native_fold import pca_fold_native_v1
from checks.native_tests.pca_native_helpers import (
    PCA_TEST_GUARD, PCATestState, require, input_values, reference_values, guards, unchanged, close_values,
    write_guarded,
)

comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()
# test_pca_randomized_options (lane py2mojo-D6, 2026-11-22)
from std.math import sqrt
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32
from checks.native_tests.testsuite_serial import check_cases_serial


comptime PCA_NATIVE_FIXTURES = 10
def fit_oracle(state: PCATestState, name: String, rows: Int, features: Int,
               spectrum: Int, keep: Int, tight: Bool) raises:
    var mean = reference_values(name, "mean", features)
    var singular = reference_values(name, "singular", spectrum)
    var variance = reference_values(name, "variance", spectrum)
    var ratio = reference_values(name, "ratio", spectrum)
    var vectors = reference_values(name, "vectors", spectrum * features)
    close_values(state.mean, mean, features, 2e-5, 2e-6, "PCA mean")
    close_values(state.singular, singular, keep, 2e-4 if tight else 4e-3,
                 2e-5 if tight else 4e-4, "PCA singular values")
    close_values(state.variance, variance, keep, 4e-4 if tight else 5e-3,
                 2e-5 if tight else 5e-4, "PCA explained variance")
    close_values(state.ratio, ratio, keep, 5e-3, 5e-4, "PCA explained variance ratio")
    var failures = List[Int](length=features * features, fill=0)
    var fp = failures.unsafe_ptr()
    var cp = state.components.unsafe_ptr() + 1
    var vp = vectors.unsafe_ptr()
    def cell(index: Int) {imm fp, imm cp, imm vp, imm features, imm keep}:
        var i = index // features
        var j = index % features
        var actual = Float64(0)
        var expected = Float64(0)
        for component in range(keep):  # Independent subspace projector cell.
            actual += Float64(cp[component * features + i]) * Float64(cp[component * features + j])
            expected += vp[component * features + i] * vp[component * features + j]
        fp.unsafe_store(index, Int(not (abs(actual - expected) <= Float64(3e-4) + Float64(3e-3) * abs(expected))))
    host_parallelize(cell, features * features)
    for failure in failures:
        require(failure == 0, "PCA independent fitted-subspace projector")
    _ = vectors^
    _ = state


def projection_oracle(source: List[Float32], state: PCATestState,
                      rows: Int, features: Int, keep: Int) raises:
    var expected = List[Float64](unsafe_uninit_length=rows * keep)
    var ep = expected.unsafe_ptr()
    var xp = source.unsafe_ptr()
    var mp = state.mean.unsafe_ptr() + 1
    var cp = state.components.unsafe_ptr() + 1
    def cell(index: Int) {imm ep, imm xp, imm mp, imm cp, imm features, imm keep}:
        var row = index // keep
        var component = index % keep
        var total = Float64(0)
        for feature in range(features):  # Defining projection equation; no PCA fitting implementation.
            total += (Float64(xp[row * features + feature]) - Float64(mp[feature])) * Float64(cp[component * features + feature])
        ep.unsafe_store(index, total)
    host_parallelize(cell, rows * keep)
    close_values(state.projected, expected, rows * keep, 5e-3, 5e-4, "PCA projection")
    _ = source
    _ = state


def inverse_oracle(restored: List[Float32], source: List[Float32], state: PCATestState,
                   rows: Int, features: Int, keep: Int, reconstructs: Bool, tight: Bool) raises:
    var expected = List[Float64](unsafe_uninit_length=rows * features)
    var ep = expected.unsafe_ptr()
    var xp = source.unsafe_ptr()
    var mp = state.mean.unsafe_ptr() + 1
    var cp = state.components.unsafe_ptr() + 1
    var zp = state.projected.unsafe_ptr() + 1
    def cell(index: Int) {imm ep, imm xp, imm mp, imm cp, imm zp, imm features, imm keep, imm reconstructs}:
        var value = Float64(xp[index])
        if not reconstructs:
            var row = index // features
            var feature = index % features
            value = Float64(mp[feature])
            for component in range(keep):  # Independent inverse equation per output cell.
                value += Float64(zp[row * keep + component]) * Float64(cp[component * features + feature])
        ep.unsafe_store(index, value)
    host_parallelize(cell, rows * features)
    close_values(restored, expected, rows * features,
                 2e-4 if tight else (1e-3 if reconstructs else 5e-3),
                 2e-4 if tight else (1e-3 if reconstructs else 8e-4), "PCA inverse")
    _ = source
    _ = state


def fitted_fixture(driver: PCANativeDriver, unit: Int, host: Bool) raises:
    var name = String("exact-21x4")
    var rows = 21
    var features = 4
    var count = 2
    var dense = False
    var keep_mode = 0
    var randomized = unit == 5
    if unit == 2:
        count = 4
        keep_mode = 1
    elif unit == 3:
        name = "full-8x3"
        rows = 8
        features = 3
        count = 3
        dense = True
    elif unit == 4:
        name = "wide-4x7"
        rows = 4
        features = 7
        dense = True
    elif unit == 5:
        name = "rank-20x5"
        rows = 20
        features = 5
        count = 3
    var source = input_values(name, rows * features)
    var original = source.copy()
    var state = PCATestState(rows, features, count)
    var request = state.fit_request(Int(source.unsafe_ptr()), rows, features, count,
        dense, keep_mode, randomized, unit != 1, PCAFoldStore())
    if host and unit == 4:
        var refused = False
        try:
            _ = driver.fit(request)
        except error:
            refused = String(error).find("pca_fit_full on the CPU: the wide route") >= 0
        require(refused, "CPU full-wide refusal must remain explicit")
        state.check_guards()
        unchanged(source, original, "PCA refused input")
        return
    var answer = driver.fit(request)
    require(answer.finite == 0 and answer.in_range == 0 and isfinite(answer.noise) and answer.noise >= 0,
            "PCA fit native status/noise")
    var keep = count
    if keep_mode == 1:
        var ratios = reference_values(name, "ratio", min(rows, features))
        var total = Float64(0)
        keep = 0
        while keep < len(ratios) and total <= 0.8:
            total += ratios[keep]
            keep += 1
    require(answer.keep == keep, "PCA kept count from independent spectrum")
    if not randomized:
        fit_oracle(state, name, rows, features, min(rows, features), keep, unit == 3)
    var components = state.components.copy()
    var mean = state.mean.copy()
    var variance = state.variance.copy()
    var ratio = state.ratio.copy()
    var singular = state.singular.copy()
    if unit == 1:
        var query = state.query_request(Int(source.unsafe_ptr()), Int(state.projected.unsafe_ptr()) + 4,
            rows, features, keep, rows, False, PCAFoldStore())
        var status = driver.query(query)
        require(status.finite == 0 and status.in_range == 0, "PCA transform status")
    projection_oracle(source, state, rows, features, keep)
    var restored = List[Float32](length=rows * features + 2, fill=PCA_TEST_GUARD)
    var inverse = state.query_request(Int(state.projected.unsafe_ptr()) + 4, Int(restored.unsafe_ptr()) + 4,
        rows, features, keep, rows, True, PCAFoldStore())
    var status = driver.query(inverse)
    require(status.finite == 0 and status.in_range == 0, "PCA inverse status")
    inverse_oracle(restored, source, state, rows, features, keep, unit == 3 or randomized, unit == 3)
    guards(restored, "PCA inverse")
    state.check_guards()
    unchanged(state.components, components, "PCA fitted components")
    unchanged(state.mean, mean, "PCA fitted mean")
    unchanged(state.variance, variance, "PCA fitted variance")
    unchanged(state.ratio, ratio, "PCA fitted ratio")
    unchanged(state.singular, singular, "PCA fitted singular values")
    unchanged(source, original, "PCA input")
    _ = state^


def planted_query(driver: PCANativeDriver) raises:
    var source: List[Float32] = [1, -2, 3, 2, 0, 5, -1, -3, 7, 4, 2, -4]
    var original = source.copy()
    var state = PCATestState(4, 3, 2)
    var component_values: List[Float32] = [1, 0, 0, 0, 1, 0]
    write_guarded(state.components, component_values)
    var mean_values: List[Float32] = [1, -2, 3]
    write_guarded(state.mean, mean_values)
    var components = state.components.copy()
    var mean = state.mean.copy()
    var request = state.query_request(Int(source.unsafe_ptr()), Int(state.projected.unsafe_ptr()) + 4,
        4, 3, 2, 4, False, PCAFoldStore())
    var status = driver.query(request)
    require(status.finite == 0 and status.in_range == 0, "planted PCA transform status")
    var expected: List[Float64] = [0, 0, 1, 2, -2, -1, 3, 4]
    close_values(state.projected, expected, 8, 0, 1e-6, "planted PCA transform")
    var restored = List[Float32](length=14, fill=PCA_TEST_GUARD)
    var inverse = state.query_request(Int(state.projected.unsafe_ptr()) + 4, Int(restored.unsafe_ptr()) + 4,
        4, 3, 2, 4, True, PCAFoldStore())
    status = driver.query(inverse)
    require(status.finite == 0 and status.in_range == 0, "planted PCA inverse status")
    var restored_expected: List[Float64] = [1, -2, 3, 2, 0, 3, -1, -3, 3, 4, 2, 3]
    close_values(restored, restored_expected, 12, 0, 1e-6, "planted PCA inverse")
    var replacement: List[Float32] = [0, 1, 0, 0, 0, 1]
    var rebound = request.copy()
    rebound.components = Int(replacement.unsafe_ptr())
    status = driver.query(rebound)
    require(status.finite == 0 and status.in_range == 0, "rebound PCA transform status")
    var rebound_expected: List[Float64] = [0, 0, 2, 2, -1, 4, 4, -7]
    close_values(state.projected, rebound_expected, 8, 0, 1e-6, "PCA current component pointer")
    require(request.components == Int(state.components.unsafe_ptr()) + 4, "original query record remains independent")
    guards(restored, "planted PCA inverse")
    state.check_guards()
    unchanged(state.components, components, "planted PCA components")
    unchanged(state.mean, mean, "planted PCA mean")
    unchanged(source, original, "planted PCA input")
    _ = replacement^
    _ = state^


def fixture(driver: PCANativeDriver, unit: Int, host: Bool) raises:
    if unit == 0:
        require(driver.metadata(0) == 1, "PCA numerical fixtures require identical mode")
        require(driver.metadata(2) == (47 if host else 15), "PCA backend capabilities")
        var vendor = driver.metadata(1)
        require(vendor == 1 if host else (vendor >= 2 and vendor <= 4), "PCA backend vendor")
    elif unit <= 5:
        fitted_fixture(driver, unit, host)
    elif unit == 6:
        planted_query(driver)
    else:
        var source: List[Float32] = [1, -2, 3, 2, 0, 5, -1, -3, 7, 4, 2, -4]
        var state = PCATestState(4, 3, 2)
        if unit == 7:
            source[5] = bitcast[DType.float32](UInt32(0x7FC00042))
            var original = source.copy()
            var answer = driver.fit(state.fit_request(Int(source.unsafe_ptr()), 4, 3, 2,
                False, 0, False, False, PCAFoldStore()))
            require(answer.finite != 0 and answer.in_range == 0, "PCA deferred fit finite status")
            var component_values: List[Float32] = [1, 0, 0, 0, 1, 0]
            write_guarded(state.components, component_values)
            var mean_values: List[Float32] = [1, -2, 3]
            write_guarded(state.mean, mean_values)
            var status = driver.query(state.query_request(Int(source.unsafe_ptr()), Int(state.projected.unsafe_ptr()) + 4,
                4, 3, 2, 4, False, PCAFoldStore()))
            require(status.finite != 0 and status.in_range == 0, "PCA deferred query finite status")
            unchanged(source, original, "PCA nonfinite input")
        elif unit == 8:
            var expected = String("PCA host provider does not own resident folds") if host else String("PCA estimator provider does not own resident folds")
            var fit_refused = False
            var query_refused = False
            try:
                _ = driver.fit(state.fit_request(0, 4, 3, 2, False, 0, False, False, PCAFoldStore(2, 4, 3)))
            except error:
                fit_refused = String(error) == expected
            try:
                _ = driver.query(state.query_request(0, Int(state.projected.unsafe_ptr()) + 4,
                    4, 3, 2, 4, False, PCAFoldStore(2, 4, 3)))
            except error:
                query_refused = String(error) == expected
            require(fit_refused and query_refused, "PCA estimator/host providers must refuse foreign fold storage")
        else:
            var request = state.query_request(Int(source.unsafe_ptr()), Int(state.projected.unsafe_ptr()) + 4,
                4, 3, 2, 1, False, PCAFoldStore())
            request.whiten = True
            var refused = False
            try:
                _ = driver.query(request)
            except error:
                refused = String(error) == "PCA whitening requires positive rows/features/components and fit_rows>=2"
            require(refused, "PCA whitening fit-row refusal through actual provider")
        state.check_guards()
        _ = source^
        _ = state^


# ---- Resident fold cases (was pca_native_fold_cases.mojo) ----
# Resident PCA gathers through the actual base-context native fold provider.
comptime PCA_NATIVE_FOLD_FIXTURES = 3


def fold_fixture(unit: Int) raises:
    # Device column only. TestSuite discovery compiles every function of the module, so the host column build
    # must not instantiate this body's device contexts (the registry's run_cpu never referenced it).
    comptime if CPU_COLUMN_BUILD:
        raise Error("fold_fixture: device column only")
    else:
        var driver = PCANativeDriver(pca_fold_native_v1)
        require(driver.metadata(0) == 1 and driver.metadata(2) == 21, "PCA fold mode/capabilities")
        var source = input_values("full-8x3", 24)
        var original = source.copy()
        var indices: List[Int64] = [7, 6, 5, 4, 3, 2, 1, 0]
        if unit == 1:
            source[0] = bitcast[DType.float32](UInt32(0x7FC00042))
        elif unit == 2:
            indices[0] = 8
        var ctx = msel_device_x_context()
        var store = msel_fold_store()
        var base_id = -1
        var index_id = -1
        try:
            base_id = store[].put(ctx, Int(source.unsafe_ptr()), 24 * 4)
            index_id = store[].put(ctx, Int(indices.unsafe_ptr()), 8 * 8)
            # Each put synchronizes its copy. Changing the original host storage
            # now must not replace resident data or the admitted index order.
            source[0] = 12345.0
            indices[0] = -7
            var fold = PCAFoldStore(base_id, 8, index_id)
            var state = PCATestState(8, 3, 3)
            var answer = driver.fit(state.fit_request(0, 8, 3, 3, True, 0, False, False, fold))
            if unit == 0:
                require(answer.keep == 3 and answer.finite == 0 and answer.in_range == 0,
                        "PCA resident full-fit status")
                fit_oracle(state, "full-8x3", 8, 3, 3, 3, True)
            else:
                require((answer.finite != 0) == (unit == 1) and (answer.in_range != 0) == (unit == 2),
                        "PCA resident fit finite/range status distinction")
                var component_values: List[Float32] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
                write_guarded(state.components, component_values)
                var mean_values: List[Float32] = [0, 0, 0]
                write_guarded(state.mean, mean_values)
            var status = driver.query(state.query_request(0, Int(state.projected.unsafe_ptr()) + 4,
                8, 3, 3, 8, False, fold))
            require((status.finite != 0) == (unit == 1) and (status.in_range != 0) == (unit == 2),
                    "PCA resident query finite/range status distinction")
            if unit == 0:
                var selected = List[Float32](unsafe_uninit_length=24)
                var sp = selected.unsafe_ptr()
                var op = original.unsafe_ptr()
                def row_word(index: Int) {imm sp, imm op}:
                    sp.unsafe_store(index, op[(7 - index // 3) * 3 + index % 3])
                host_parallelize(row_word, 24)
                projection_oracle(selected, state, 8, 3, 3)
                _ = selected^
            state.check_guards()
            _ = state^
        finally:
            if index_id >= 0:
                store[].free(ctx, index_id)
            if base_id >= 0:
                store[].free(ctx, base_id)
            _ = source^
            _ = original^
            _ = indices^
            _ = ctx^


# Host column: Actual CPU PCA request/state fixtures, with independent graphs in parallel.
# Device column: Actual GPU PCA estimator and resident-fold providers.
#
# Production contexts/stores are shared, so fixture graphs submit in order.
# Device kernels retain their parallel algorithms; independent input decoding
# and oracle cells use the host pool. No timing or identity comparison.


def _host_fixture(unit: Int) raises:
    fixture(PCANativeDriver(pca_host_native_v1), unit, True)


def test_pca_native() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        check_cases("pca_native", PCA_NATIVE_FIXTURES, _host_fixture)
    else:
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
        var failures = String("")
        for unit in range(PCA_NATIVE_FIXTURES + PCA_NATIVE_FOLD_FIXTURES):  # driver-loop: over fixtures, in order
            try:
                if unit < PCA_NATIVE_FIXTURES:
                    fixture(PCANativeDriver(pca_device_native_v1), unit, False)
                else:
                    fold_fixture(unit - PCA_NATIVE_FIXTURES)
            except error:
                failures += "\n      row " + String(unit) + ": " + String(error)
        if failures.byte_length() > 0:
            raise Error("pca_native: device fixtures failed:" + failures)


# ---- randomized solver options (lane py2mojo-D6, 2026-11-22) ----
# Ported from python/mojolearn/tests/test_pca_exact_native_owner.py
# `test_randomized_complete_owner_preserves_seed_power_and_consumed_outputs`, its six parameter rows
# (rows, cols, n_components, n_oversamples, iterated_power, power_iteration_normalizer, random_state):
#     0 (24, 5, 2, 3, 'auto', 'auto', None)   1 (24, 5, 2, 0, 1, 'QR', 37)   2 (5, 24, 2, 2, 2, 'LU', -1)
#     3 (24, 5, 2, 2, 0, 'none', 2**70 + 3)   4 (24, 5, 2, 3, -1, 'auto', 37)  5 (72, 65, 3, 14, 1, 'QR', 7)
# with whiten=True, as the door builds the request (bindings/pca_fit_native_door.mojo: random_state None -> seed 0,
# an int -> its low 32 bits, 'auto' -> auto_iterations; the normalizer is admitted there and the native solver takes
# no word for it). Inputs: the original `fixture(rows, cols)` draws (seed 9021) in fixtures/pca_random/. Checks: the
# fit's projection (fit_transform) equals the transform query of the fitted state word for word (the original's
# fit_transform == transform of an equal fit); then the original's equations on the consumed state, in float64: the
# whitened projection (x - mean) C^T sqrt(rows - 1) / s to rtol 5e-3, atol 5e-4; the inverse (x - mean) C^T C + mean
# to rtol 5e-3, atol 8e-4; C C^T = I to rtol 5e-3, atol 5e-4. The Python owner, the audit of Python orchestration and
# the published attributes stay in Python. Independent rows run across the host cores.
comptime PCA_RANDOM_CASES = 6
comptime PCA_RANDOM_DIR = "checks/native_tests/fixtures/pca_random/"


def randomized_fixture(driver: PCANativeDriver, unit: Int) raises:
    var rows_of: Array[Int, 6] = [24, 24, 5, 24, 24, 72]
    var cols_of: Array[Int, 6] = [5, 5, 24, 5, 5, 65]
    var rank_of: Array[Int, 6] = [2, 2, 2, 2, 2, 3]
    var over_of: Array[Int, 6] = [3, 0, 2, 2, 3, 14]
    var power_of: Array[Int, 6] = [0, 1, 2, 0, -1, 1]
    var seed_of: Array[Int, 6] = [0, 37, 0xFFFFFFFF, 3, 37, 7]
    var rows = rows_of[unit]
    var features = cols_of[unit]
    var count = rank_of[unit]
    var label = "PCA randomized row " + String(unit)
    var source = fixture_f32(PCA_RANDOM_DIR + "x-" + String(rows) + "x" + String(features) + ".f32", rows * features)
    var original = source.copy()
    var state = PCATestState(rows, features, count)
    var request = PCAFitRequest(Int(source.unsafe_ptr()), Int(state.components.unsafe_ptr()) + 4,
        Int(state.mean.unsafe_ptr()) + 4, Int(state.variance.unsafe_ptr()) + 4, Int(state.ratio.unsafe_ptr()) + 4,
        Int(state.singular.unsafe_ptr()) + 4, Int(state.projected.unsafe_ptr()) + 4, rows, features, count,
        False, 0, True, True, over_of[unit], power_of[unit], unit == 0, seed_of[unit], True, 0.0, PCAFoldStore())
    var answer = driver.fit(request)
    require(answer.finite == 0 and answer.in_range == 0 and answer.keep == count,
            label + ": fit status / kept count " + String(answer.keep))
    var transformed = List[Float32](length=rows * count + 2, fill=PCA_TEST_GUARD)
    var status = driver.query(PCAQueryRequest(Int(source.unsafe_ptr()), Int(state.mean.unsafe_ptr()) + 4,
        Int(state.components.unsafe_ptr()) + 4, Int(state.singular.unsafe_ptr()) + 4,
        Int(transformed.unsafe_ptr()) + 4, rows, features, count, rows, False, True, PCAFoldStore()))
    require(status.finite == 0 and status.in_range == 0, label + ": transform status")
    guards(transformed, label + " transform")
    unchanged(transformed, state.projected, label + ": transform of the fitted state vs fit_transform")
    var restored = List[Float32](length=rows * features + 2, fill=PCA_TEST_GUARD)
    status = driver.query(PCAQueryRequest(Int(state.projected.unsafe_ptr()) + 4, Int(state.mean.unsafe_ptr()) + 4,
        Int(state.components.unsafe_ptr()) + 4, Int(state.singular.unsafe_ptr()) + 4,
        Int(restored.unsafe_ptr()) + 4, rows, features, count, rows, True, True, PCAFoldStore()))
    require(status.finite == 0 and status.in_range == 0, label + ": inverse status")
    guards(restored, label + " inverse")
    # the original's equations, float64, over the consumed fitted state
    var mean = List[Float64](length=features, fill=0)
    var xp = source.unsafe_ptr()
    var mp = mean.unsafe_ptr()
    def column_mean(feature: Int) {imm xp, imm mp, imm rows, imm features}:
        var total = Float64(0)
        for row in range(rows):  # the one column's mean
            total += Float64(xp[row * features + feature])
        mp[feature] = total / Float64(rows)
    host_parallelize(column_mean, features)
    var raw = List[Float64](length=rows * count, fill=0)
    var whitened = List[Float64](length=rows * count, fill=0)
    var rp = raw.unsafe_ptr()
    var wp = whitened.unsafe_ptr()
    var cp = state.components.unsafe_ptr() + 1
    var sp = state.singular.unsafe_ptr() + 1
    def projection(index: Int) {imm xp, imm mp, imm cp, imm sp, imm rp, imm wp, imm rows, imm features, imm count}:
        var row = index // count
        var component = index % count
        var total = Float64(0)
        for feature in range(features):  # the one cell's projection
            total += (Float64(xp[row * features + feature]) - mp[feature]) * Float64(cp[component * features + feature])
        rp[index] = total
        wp[index] = total * sqrt(Float64(rows - 1)) / Float64(sp[component])
    host_parallelize(projection, rows * count)
    close_values(state.projected, whitened, rows * count, 5e-3, 5e-4, label + ": whitened projection")
    var inverse = List[Float64](length=rows * features, fill=0)
    var ip = inverse.unsafe_ptr()
    def back(index: Int) {imm rp, imm mp, imm cp, imm ip, imm features, imm count}:
        var row = index // features
        var feature = index % features
        var total = mp[feature]
        for component in range(count):  # the one cell's reconstruction
            total += rp[row * count + component] * Float64(cp[component * features + feature])
        ip[index] = total
    host_parallelize(back, rows * features)
    close_values(restored, inverse, rows * features, 5e-3, 8e-4, label + ": inverse")
    for i in range(count):  # small-loop: the count x count Gram of the components
        for j in range(count):
            var total = Float64(0)
            for feature in range(features):
                total += Float64(cp[i * features + feature]) * Float64(cp[j * features + feature])
            var want = Float64(1.0) if i == j else Float64(0.0)
            require(abs(total - want) <= 5e-4 + 5e-3 * abs(want),
                    label + ": components @ components.T [" + String(i) + ", " + String(j) + "] = " + String(total))
    state.check_guards()
    unchanged(source, original, label + " input")
    _ = mean^
    _ = raw^
    _ = whitened^
    _ = inverse^
    _ = transformed^
    _ = restored^
    _ = state^


def _host_randomized(unit: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        randomized_fixture(PCANativeDriver(pca_host_native_v1), unit)


def _device_randomized(unit: Int) raises:
    comptime if not CPU_COLUMN_BUILD:
        randomized_fixture(PCANativeDriver(pca_device_native_v1), unit)


def test_pca_randomized_options() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime if CPU_COLUMN_BUILD:
        check_cases("pca_randomized_options", PCA_RANDOM_CASES, _host_randomized)
    else:
        # the production device context is shared: the fits submit in order
        check_cases_serial("pca_randomized_options", PCA_RANDOM_CASES, _device_randomized)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
