# SPDX-License-Identifier: Apache-2.0
"""Typed PCA fit/query correctness using original fixtures and external oracles.

No SVD implementation, output hash, repeated fit comparison or timing. The
recorded spectra come from independent NumPy Float64 SVD. Query equations
consume the actual fitted state, so eigenvector signs need no convention.
"""
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.pca_native_api import PCANativeDriver, PCAFoldStore, PCAFitRequest, PCAQueryRequest

comptime PCA_NATIVE_FIXTURES = 10
comptime PCA_TEST_GUARD = Float32(27182.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def input_values(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/pca_native/" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "PCA input fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def reference_values(name: String, part: String, count: Int) raises -> List[Float64]:
    var result = List[Float64](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/pca_native/" + name + "-" + part + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 8, "PCA reference fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed file decoding only.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
            rp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def guards(value: List[Float32], name: String) raises:
    require(value[0] == PCA_TEST_GUARD and value[len(value) - 1] == PCA_TEST_GUARD, name + " guards")


def unchanged(actual: List[Float32], original: List[Float32], name: String) raises:
    require(len(actual) == len(original), name + " storage extent")
    var count = len(actual)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr()
    var bp = original.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm bp}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](bp[index])))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, name + " was mutated")
    _ = actual
    _ = original


def close_values(actual: List[Float32], expected: List[Float64], count: Int,
                 rtol: Float64, atol: Float64, name: String) raises:
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr() + 1
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm ep, imm rtol, imm atol}:
        fp.unsafe_store(index, Int(not (abs(Float64(ap[index]) - ep[index]) <= atol + rtol * abs(ep[index]))))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, name + " independent oracle tolerance")
    _ = actual
    _ = expected


def write_guarded(mut dst: List[Float32], source: List[Float32]):
    var count = len(source)
    var dp = dst.unsafe_ptr() + 1
    var sp = source.unsafe_ptr()
    def cell(index: Int) {imm dp, imm sp}:
        dp.unsafe_store(index, sp[index])
    host_parallelize(cell, count)
    _ = dst
    _ = source


struct PCATestState(Movable):
    var components: List[Float32]
    var mean: List[Float32]
    var variance: List[Float32]
    var ratio: List[Float32]
    var singular: List[Float32]
    var projected: List[Float32]

    def __init__(out self, rows: Int, features: Int, count: Int):
        self.components = List[Float32](length=count * features + 2, fill=PCA_TEST_GUARD)
        self.mean = List[Float32](length=features + 2, fill=PCA_TEST_GUARD)
        self.variance = List[Float32](length=count + 2, fill=PCA_TEST_GUARD)
        self.ratio = List[Float32](length=count + 2, fill=PCA_TEST_GUARD)
        self.singular = List[Float32](length=count + 2, fill=PCA_TEST_GUARD)
        self.projected = List[Float32](length=rows * count + 2, fill=PCA_TEST_GUARD)

    def check_guards(self) raises:
        guards(self.components, "PCA components")
        guards(self.mean, "PCA mean")
        guards(self.variance, "PCA variance")
        guards(self.ratio, "PCA ratio")
        guards(self.singular, "PCA singular")
        guards(self.projected, "PCA projection")

    def fit_request(self, source: Int, rows: Int, features: Int, count: Int,
                    dense: Bool, keep_mode: Int, randomized: Bool, project: Bool,
                    fold: PCAFoldStore) -> PCAFitRequest:
        return PCAFitRequest(source, Int(self.components.unsafe_ptr()) + 4, Int(self.mean.unsafe_ptr()) + 4,
            Int(self.variance.unsafe_ptr()) + 4, Int(self.ratio.unsafe_ptr()) + 4, Int(self.singular.unsafe_ptr()) + 4,
            Int(self.projected.unsafe_ptr()) + 4 if project else 0, rows, features, count,
            dense, keep_mode, False, randomized, 2 if randomized else 10, 1 if randomized else 0,
            False, 17 if randomized else 0, True, 0.8 if keep_mode == 1 else 0.0, fold)

    def query_request(self, source: Int, dst: Int, rows: Int, features: Int,
                      count: Int, fit_rows: Int, inverse: Bool, fold: PCAFoldStore) -> PCAQueryRequest:
        return PCAQueryRequest(source, Int(self.mean.unsafe_ptr()) + 4, Int(self.components.unsafe_ptr()) + 4,
            Int(self.singular.unsafe_ptr()) + 4, dst, rows, features, count, fit_rows, inverse, False, fold)


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
