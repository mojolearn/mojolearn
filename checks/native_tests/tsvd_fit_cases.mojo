# SPDX-License-Identifier: Apache-2.0
"""Actual TSVD fit routes using original inputs and independent saved oracles.

Shared PCA test helpers provide only buffer guards/comparison/ownership;
no PCA kernel or fitted output supplies a TSVD reference.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.pca_native_api import PCAFoldStore
from core.tsvd_native_api import (
    TSVDNativeDriver, TSVDFitRequest, TSVDQueryRequest, TSVD_GRAM,
    TSVD_FUSED, TSVD_EXPLAINED, TSVD_RANDOMIZED, TSVD_TSQR,
)
from checks.native_tests.pca_native_cases import require, close_values, unchanged, guards, PCA_TEST_GUARD

comptime TSVD_FIT_FIXTURES = 10


def input_values(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/tsvd_fit/" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "TSVD input fixture extent")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def cell(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(cell, count)
        _ = bytes^
    return result^


def reference_values(name: String, part: String, count: Int) raises -> List[Float64]:
    var result = List[Float64](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/tsvd_fit/" + name + "-" + part + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 8, "TSVD reference fixture extent")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def cell(index: Int) {imm bp, imm rp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed file-word decoding only.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
            rp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(cell, count)
        _ = bytes^
    return result^


struct TSVDFitState(Movable):
    var components: List[Float32]
    var singular: List[Float32]
    var variance: List[Float32]
    var ratio: List[Float32]

    def __init__(out self, features: Int):
        self.components = List[Float32](length=2 * features + 2, fill=PCA_TEST_GUARD)
        self.singular = List[Float32](length=4, fill=PCA_TEST_GUARD)
        self.variance = List[Float32](length=4, fill=PCA_TEST_GUARD)
        self.ratio = List[Float32](length=4, fill=PCA_TEST_GUARD)

    def check_guards(self) raises:
        guards(self.components, "TSVD components")
        guards(self.singular, "TSVD singular values")
        guards(self.variance, "TSVD explained variance")
        guards(self.ratio, "TSVD explained ratio")

    def request(self, source: Int, rows: Int, features: Int, route: Int,
                fold: PCAFoldStore, count: Int = 2) -> TSVDFitRequest:
        return TSVDFitRequest(source, Int(self.components.unsafe_ptr()) + 4,
            Int(self.singular.unsafe_ptr()) + 4, Int(self.variance.unsafe_ptr()) + 4,
            Int(self.ratio.unsafe_ptr()) + 4, rows, features, count, 10, 5, False, 7, True, route, fold)


def fit_oracle(state: TSVDFitState, name: String, rows: Int, features: Int,
               with_variance: Bool) raises:
    var spectrum = min(rows, features)
    var singular = reference_values(name, "singular", spectrum)
    var vectors = reference_values(name, "vectors", spectrum * features)
    close_values(state.singular, singular, 2, 1e-3, 1e-4, "TSVD independent singular values")
    var failures = List[Int](length=features * features, fill=0)
    var fp = failures.unsafe_ptr()
    var cp = state.components.unsafe_ptr() + 1
    var vp = vectors.unsafe_ptr()
    def cell(index: Int) {imm fp, imm cp, imm vp, imm features}:
        var row = index // features
        var column = index % features
        var actual = Float64(cp[row]) * Float64(cp[column]) + Float64(cp[features + row]) * Float64(cp[features + column])
        var expected = vp[row] * vp[column] + vp[features + row] * vp[features + column]
        fp.unsafe_store(index, Int(not (abs(actual - expected) <= Float64(2e-4) + Float64(2e-3) * abs(expected))))
    host_parallelize(cell, features * features)
    for failure in failures:
        require(failure == 0, "TSVD independent leading subspace")
    if with_variance:
        var variance = reference_values(name, "variance", 2)
        var ratio = reference_values(name, "ratio", 2)
        close_values(state.variance, variance, 2, 1e-3, 1e-5, "TSVD independent variance")
        close_values(state.ratio, ratio, 2, 1e-3, 1e-5, "TSVD independent variance ratio")
    _ = vectors^
    _ = state


def query_oracle(driver: TSVDNativeDriver, source: List[Float32], state: TSVDFitState,
                 rows: Int, features: Int) raises:
    var dst = List[Float32](length=rows * 2 + 2, fill=PCA_TEST_GUARD)
    var components = state.components.copy()
    var expected = List[Float64](unsafe_uninit_length=rows * 2)
    var xp = source.unsafe_ptr()
    var cp = state.components.unsafe_ptr() + 1
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm xp, imm cp, imm ep, imm features}:
        var row = index // 2
        var component = index % 2
        var value = Float64(0)
        for column in range(features):
            value += Float64(xp[row * features + column]) * Float64(cp[component * features + column])
        ep.unsafe_store(index, value)
    host_parallelize(cell, rows * 2)
    try:
        var status = driver.query(TSVDQueryRequest(Int(source.unsafe_ptr()),
            Int(state.components.unsafe_ptr()) + 4, Int(dst.unsafe_ptr()) + 4,
            rows, features, 2, False, True, False, 0))
        require(status == 0, "TSVD fitted transform finite status")
        close_values(dst, expected, rows * 2, 2e-4, 2e-4, "TSVD original query equation tolerance")
        guards(dst, "TSVD fitted transform")
        unchanged(state.components, components, "TSVD fitted transform components")
    finally:
        _ = dst^
        _ = expected^
        _ = source
        _ = state


def fixture(estimator: TSVDNativeDriver, decomp: TSVDNativeDriver, unit: Int, host: Bool) raises:
    if unit == 0:
        require(decomp.metadata(0) == 1 and decomp.metadata(2) == (34 if host else 6),
                "TSVD decomposition mode/capability metadata")
        var vendor = decomp.metadata(1)
        require(vendor == 1 if host else (vendor >= 2 and vendor <= 4), "TSVD decomposition vendor")
        require(decomp.metadata(3) == 0 if host else (decomp.metadata(3) == 0 or decomp.metadata(3) == 2),
                "TSVD supported native Q-fix flag")
        return
    var small = unit == 4 or unit >= 6
    var name = String("boundary-6x3") if small else String("host-512x5")
    var rows = 6 if small else 512
    var features = 3 if small else 5
    var source = input_values(name, rows * features)
    if unit == 6:
        source[2] = bitcast[DType.float32](UInt32(0x7FC00042))
    elif unit == 7:
        source[2] = bitcast[DType.float32](UInt32(0x7F800000))
    var original = source.copy()
    var state = TSVDFitState(features)
    var route = TSVD_GRAM
    if unit == 2 or unit == 7 or unit == 9:
        route = TSVD_FUSED
    elif unit == 3:
        route = TSVD_EXPLAINED
        for index in range(2 * features):
            state.components[index + 1] = Float32(1 if index == 0 or index == features + 2 else 0)
    elif unit == 4 or unit == 6 or unit == 8:
        route = TSVD_RANDOMIZED
    elif unit == 5:
        route = TSVD_TSQR
    var driver = decomp.copy() if route == TSVD_RANDOMIZED or route == TSVD_TSQR else estimator.copy()
    var request = state.request(Int(source.unsafe_ptr()), rows, features, route, PCAFoldStore(), 0 if unit == 8 else 2)
    try:
        if (host and unit == 5) or unit == 8:
            var refused = False
            try:
                _ = driver.fit(request)
            except error:
                var expected = String("TruncatedSVD host decomposition provider does not provide TSQR components") if unit == 5 else String("n_components must be in [1, min(n_samples, n_features)]")
                refused = String(error) == expected
            require(refused, "TSVD exact native route/count refusal")
        else:
            var answer = driver.fit(request)
            require(answer.in_range == 0 and (answer.finite != 0) == (unit == 6 or unit == 7),
                    "TSVD fit finite/range status")
            if unit != 6 and unit != 7:
                if unit == 3:
                    var variance = reference_values(name, "selected-variance", 2)
                    var ratio = reference_values(name, "selected-ratio", 2)
                    close_values(state.variance, variance, 2, 1e-3, 1e-5, "TSVD planted component variance")
                    close_values(state.ratio, ratio, 2, 1e-3, 1e-5, "TSVD planted component ratio")
                else:
                    if unit == 5:
                        var explained = state.request(Int(source.unsafe_ptr()), rows, features, TSVD_EXPLAINED, PCAFoldStore())
                        var status = estimator.fit(explained)
                        require(status.finite == 0 and status.in_range == 0, "TSVD TSQR explained-tail status")
                    fit_oracle(state, name, rows, features, unit != 1)
                    if unit == 1:
                        require(state.variance[1] == PCA_TEST_GUARD and state.variance[2] == PCA_TEST_GUARD
                                and state.ratio[1] == PCA_TEST_GUARD and state.ratio[2] == PCA_TEST_GUARD,
                                "TSVD Gram-only route preserves unrelated outputs")
                query_oracle(estimator, source, state, rows, features)
        state.check_guards()
        unchanged(source, original, "TSVD fit input")
    finally:
        # Integer addresses carry no lifetime. Preserve every owner through
        # successful, refused and unexpected-error descriptor calls alike.
        _ = source^
        _ = original^
        _ = state^
