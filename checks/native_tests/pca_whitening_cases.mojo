# SPDX-License-Identifier: Apache-2.0
"""Original PCA whitening numerical fixtures with independent expectations.

The zero-scale and cutoff states are literal. The fitted8x2 matrix has centered,
independent axes; unit sample variance and inverse values are checked directly.
No SVD implementation, old output reference or Python arithmetic is imported.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize

comptime PCA_WHITENING_FIXTURES = 3
comptime PCA_GUARD = Float32(27182.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


trait PcaWhiteningCalls:
    @staticmethod
    def fit(x: Int, components: Int, mean: Int, explained: Int, ratio: Int,
        singular: Int, rows: Int) raises: ...
    @staticmethod
    def query(x: Int, mean: Int, components: Int, singular: Int, dst: Int,
        rows: Int, fit_rows: Int, inverse: Bool) raises: ...


def guards(value: List[Float32], name: String) raises:
    require(value[0] == PCA_GUARD and value[len(value) - 1] == PCA_GUARD, name + " guards")


def equal_words(actual: List[Float32], expected: List[Float32], count: Int, name: String) raises:
    var a = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(actual.unsafe_ptr()) + 4)
    var b = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(expected.unsafe_ptr()) + 4)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    def word(index: Int) {imm a, imm b, imm fp}:
        if a[index] != b[index]:
            fp.unsafe_store(index, 1)
    host_parallelize(word, count)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, name + " words differ")
    guards(actual, name)
    guards(expected, name + " reference")
    _ = actual
    _ = expected


def fitted_expectations(projected: List[Float32], restored: List[Float32], source: List[Float32]) raises:
    var p = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(projected.unsafe_ptr()) + 4)
    var r = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(restored.unsafe_ptr()) + 4)
    var x = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(source.unsafe_ptr()) + 4)
    var failures = List[Int](length=18, fill=0)
    var fp = failures.unsafe_ptr()
    def contract(index: Int) {imm p, imm r, imm x, imm fp}:
        if index < 2:
            var mean = Float64(0)
            for row in range(8):  # Dependent small-column moments; independent columns split outside.
                mean += Float64(p[row * 2 + index])
            mean /= 8
            var variance = Float64(0)
            for row in range(8):
                var centered = Float64(p[row * 2 + index]) - mean
                variance += centered * centered
            variance /= 7
            if not (abs(variance - 1) <= Float64(2e-5) + Float64(2e-5)):
                fp.unsafe_store(index, 1)
        else:
            var cell = index - 2
            if not (abs(Float64(r[cell]) - Float64(x[cell])) <= Float64(2e-5) + Float64(2e-5) * abs(Float64(x[cell]))):
                fp.unsafe_store(index, 1)
    host_parallelize(contract, 18)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "PCA fitted sample variance or inverse differs")
    _ = projected
    _ = restored
    _ = source


def fixture[A: PcaWhiteningCalls](fixture_id: Int) raises:
    var rows = 3 if fixture_id == 0 else (8 if fixture_id == 1 else 2)
    var fit_rows = 5 if fixture_id == 0 else rows
    var x: List[Float32] = [PCA_GUARD, 1, 2, 3, 4, -1, -2, PCA_GUARD]
    if fixture_id == 1:
        x = [PCA_GUARD, -3, -1, -3, 1, -1, -1, -1, 1, 1, -1, 1, 1, 3, -1, 3, 1, PCA_GUARD]
    elif fixture_id == 2:
        x = [PCA_GUARD, 1, 0, 0, 1, PCA_GUARD]
    var source = x.copy()
    var components: List[Float32] = [PCA_GUARD, 1, 0, 0, 1, PCA_GUARD]
    var mean: List[Float32] = [PCA_GUARD, 0, 0, PCA_GUARD]
    var singular: List[Float32] = [PCA_GUARD, 2, 0, PCA_GUARD]
    var explained = List[Float32](length=4, fill=PCA_GUARD)
    var ratio = List[Float32](length=4, fill=PCA_GUARD)
    var threshold = Float32(1e-10)
    if fixture_id == 2:
        singular[1] = bitcast[DType.float32](bitcast[DType.uint32](threshold) - UInt32(1))
        singular[2] = threshold
    var xp = Int(x.unsafe_ptr()) + 4
    var cp = Int(components.unsafe_ptr()) + 4
    var mp = Int(mean.unsafe_ptr()) + 4
    var sp = Int(singular.unsafe_ptr()) + 4
    if fixture_id == 1:
        A.fit(xp, cp, mp, Int(explained.unsafe_ptr()) + 4, Int(ratio.unsafe_ptr()) + 4, sp, rows)
    var saved_components = components.copy()
    var saved_mean = mean.copy()
    var saved_singular = singular.copy()
    var projected = List[Float32](length=rows * 2 + 2, fill=PCA_GUARD)
    A.query(xp, mp, cp, sp, Int(projected.unsafe_ptr()) + 4, rows, fit_rows, False)
    if fixture_id == 0:
        var expected: List[Float32] = [PCA_GUARD, 1, 4, 3, 8, -1, -4, PCA_GUARD]
        equal_words(projected, expected, 6, "zero singular whitening scale")
    if fixture_id < 2:
        var prefix_rows = 1 if fixture_id == 0 else 3
        var prefix = List[Float32](length=prefix_rows * 2 + 2, fill=PCA_GUARD)
        A.query(xp, mp, cp, sp, Int(prefix.unsafe_ptr()) + 4, prefix_rows, fit_rows, False)
        equal_words(prefix, projected, prefix_rows * 2, "whitening row prefix")
        if fixture_id == 0:
            var repeated = List[Float32](length=rows * 2 + 2, fill=PCA_GUARD)
            A.query(xp, mp, cp, sp, Int(repeated.unsafe_ptr()) + 4, rows, fit_rows, False)
            equal_words(repeated, projected, rows * 2, "repeated whitening call")
        var restored = List[Float32](length=rows * 2 + 2, fill=PCA_GUARD)
        A.query(Int(projected.unsafe_ptr()) + 4, mp, cp, sp, Int(restored.unsafe_ptr()) + 4, rows, fit_rows, True)
        guards(restored, "inverse")
        if fixture_id == 0:
            equal_words(restored, x, rows * 2, "zero singular inverse")
        else:
            fitted_expectations(projected, restored, x)
        _ = prefix
        _ = restored
    else:
        require(bitcast[DType.uint32](projected[1]) == bitcast[DType.uint32](Float32(1)), "below-cutoff scale must skip division")
        require(bitcast[DType.uint32](projected[4]) == bitcast[DType.uint32](Float32(1) / threshold),
            "at-cutoff scale must divide")
        require(projected[2] == 0 and projected[3] == 0, "cutoff diagonal state produced off-axis values")
    equal_words(x, source, rows * 2, "feature source")
    equal_words(components, saved_components, 4, "component state")
    equal_words(mean, saved_mean, 2, "mean state")
    equal_words(singular, saved_singular, 2, "singular state")
    guards(projected, "transform")
    guards(explained, "explained variance")
    guards(ratio, "variance ratio")
    _ = x
    _ = components
    _ = mean
    _ = singular
