# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Concurrent host decomposition fits in one process (lane r3-small, 2026-11-22).

Lane py2mojo-D6 found that two host randomized PCA fits running at the same time crashed the process (rc 139,
cpc1 2026-11-22). This check runs `PCA_CONCURRENT_FITS` fits at once across the host cores, through the native
entry the public door calls (`pca_host_native_v1`), and compares every fit's arrays word for word with the same
fit run alone first: a fit's result does not depend on what else the process is doing.

    test_randomized_concurrent   randomized PCA (whiten, projection), three shapes and option sets dealt over
                                 the fits
    test_exact_concurrent        exact (covariance) PCA, the same shapes
"""
from std.memory import bitcast
from std.testing import TestSuite
from core.host_parallel import host_parallelize
from core.pca_native_api import PCANativeDriver, PCAFoldStore, PCAFitRequest
from core.pca_native_host import pca_host_native_v1

comptime PCA_CONCURRENT_FITS = 100
comptime _SHAPES = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _matrix(rows: Int, cols: Int, salt: Int) -> List[Float32]:
    """A fixed pseudo-random matrix (64-bit LCG, top 24 bits to [-1, 1))."""
    var x = List[Float32](length=rows * cols, fill=0)
    var state = UInt64(0x9E3779B97F4A7C15) ^ UInt64(salt)
    for i in range(rows * cols):  # small-loop(fixture: at most 72 x 65 words)
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        x[i] = Float32(Int((state >> 40) & 0xFFFFFF)) / Float32(8388608.0) - Float32(1.0)
    return x^


struct FitOut(Movable):
    var components: List[Float32]
    var mean: List[Float32]
    var variance: List[Float32]
    var ratio: List[Float32]
    var singular: List[Float32]
    var projected: List[Float32]
    var noise: Float64

    def __init__(out self, rows: Int, features: Int, count: Int):
        self.components = List[Float32](length=count * features, fill=0)
        self.mean = List[Float32](length=features, fill=0)
        self.variance = List[Float32](length=count, fill=0)
        self.ratio = List[Float32](length=count, fill=0)
        self.singular = List[Float32](length=count, fill=0)
        self.projected = List[Float32](length=rows * count, fill=0)
        self.noise = 0.0


def _shape(which: Int) -> Tuple[Int, Int, Int, Int, Int, Int]:
    """(rows, features, count, oversamples, iterations, seed): the randomized option rows of the D6 test."""
    if which == 0:
        return (24, 5, 2, 3, 0, 0)
    elif which == 1:
        return (5, 24, 2, 2, 2, 0xFFFFFFFF)
    return (72, 65, 3, 14, 1, 7)


def _fit(which: Int, randomized: Bool) raises -> FitOut:
    var shape = _shape(which)
    var rows = shape[0]
    var features = shape[1]
    var count = shape[2]
    var x = _matrix(rows, features, which)
    var fitted = FitOut(rows, features, count)
    var request = PCAFitRequest(Int(x.unsafe_ptr()), Int(fitted.components.unsafe_ptr()), Int(fitted.mean.unsafe_ptr()),
        Int(fitted.variance.unsafe_ptr()), Int(fitted.ratio.unsafe_ptr()), Int(fitted.singular.unsafe_ptr()),
        Int(fitted.projected.unsafe_ptr()), rows, features, count, False, 0, True, randomized,
        shape[3] if randomized else 0, shape[4] if randomized else 0, False, shape[5] if randomized else 0, True,
        0.0, PCAFoldStore())
    var answer = PCANativeDriver(pca_host_native_v1).fit(request)
    require(answer.finite == 0 and answer.in_range == 0 and answer.keep == count,
        "shape " + String(which) + ": fit status / kept count " + String(answer.keep))
    fitted.noise = answer.noise
    _ = x^
    return fitted^


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": length")
    for i in range(len(a)):  # small-loop(at most 216 words per array)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(label + " word " + String(i) + ": " + String(a[i]) + " alone " + String(b[i]))


def _compare(got: FitOut, want: FitOut, label: String) raises:
    _same(got.components, want.components, label + " components")
    _same(got.mean, want.mean, label + " mean")
    _same(got.variance, want.variance, label + " variance")
    _same(got.ratio, want.ratio, label + " ratio")
    _same(got.singular, want.singular, label + " singular")
    _same(got.projected, want.projected, label + " projection")
    require(bitcast[DType.uint64](got.noise) == bitcast[DType.uint64](want.noise), label + " noise variance")


def _concurrent(randomized: Bool, label: String) raises:
    var alone = List[FitOut]()
    for which in range(_SHAPES):  # driver-loop: the reference fits, one at a time, before the concurrent ones
        alone.append(_fit(which, randomized))
    var failed = List[Int](length=PCA_CONCURRENT_FITS, fill=0)
    var messages = List[String](length=PCA_CONCURRENT_FITS, fill=String(""))
    var fp = failed.unsafe_ptr()
    var mp = messages.unsafe_ptr()
    var ap = alone.unsafe_ptr()
    def task(unit: Int) {imm fp, imm mp, imm ap, imm randomized, imm label}:
        try:
            var which = unit % _SHAPES
            var got = _fit(which, randomized)
            _compare(got, ap[unsafe_offset=which], label + " fit " + String(unit))
        except problem:
            mp[unsafe_offset=unit] = String(problem)
            fp.unsafe_store(unit, 1)
    host_parallelize(task, PCA_CONCURRENT_FITS)
    var count = 0
    var first = String("")
    for i in range(PCA_CONCURRENT_FITS):  # small-loop(the fits' failure words)
        if failed[i] != 0:
            if count == 0:
                first = messages[i]
            count += 1
    _ = alone^
    _ = messages^
    require(count == 0, label + ": " + String(count) + " of " + String(PCA_CONCURRENT_FITS)
        + " concurrent fits differ from the fit alone; first: " + first)


def test_randomized_concurrent() raises:
    _concurrent(True, "randomized PCA")


def test_exact_concurrent() raises:
    _concurrent(False, "exact PCA")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
