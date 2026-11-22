# SPDX-License-Identifier: Apache-2.0
"""PCA native-check helpers shared by the decomposition, cluster and neighbour checks: fixture decoding, guard
words, the untouched-input and oracle-tolerance comparisons, and the guarded PCA state with its request builders.

No `main()` and no `test_*`: a helper module, not a check (docs/plans/TESTSUITE_CONVERSION.md). Moved out of
`pca_native.mojo` when that check became a TestSuite file (lane ts-b5, 2026-11-22).
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.pca_native_api import PCAFoldStore, PCAFitRequest, PCAQueryRequest


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


comptime PCA_TEST_GUARD = Float32(27182.5)


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
