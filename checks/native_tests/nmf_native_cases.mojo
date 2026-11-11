# SPDX-License-Identifier: Apache-2.0
"""Actual NMF graphs and mathematical oracles, without reproducing a solver.

The 6x3 input, rank two, three iterations and seed seven are the original
public boundary fixture. CD/MU and all five initialization modes are kept.
Float64 matrix multiplication and the defining beta-divergence equations
check the resulting factors; no fitted native output supplies a reference.
"""
from std.math import isfinite, log, sqrt
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.nmf_native_api import NMFNativeDriver, NMFFitRequest, NMFInverseRequest
from core.nmf_native_state import NMFScratch
from x_decomp.nmf import NmfArgs
from checks.native_tests.pca_native_cases import require, guards, unchanged, close_values, PCA_TEST_GUARD

comptime NMF_NATIVE_FIXTURES = 20


def input_values() -> List[Float32]:
    var values: List[Float32] = [1, 2, 1, 3, 1, 2, 1, 1, 4, 2, 3, 1, 1, 2, 2, 4, 1, 1]
    return values^


def product(w: List[Float32], h: List[Float32]) raises -> List[Float64]:
    var result = List[Float64](unsafe_uninit_length=18)
    var rp = result.unsafe_ptr()
    var wp = w.unsafe_ptr() + 1
    var hp = h.unsafe_ptr() + 1
    def cell(index: Int) {imm rp, imm wp, imm hp}:
        var row = index // 3
        var col = index % 3
        rp.unsafe_store(index, Float64(wp[row * 2]) * Float64(hp[col])
            + Float64(wp[row * 2 + 1]) * Float64(hp[3 + col]))
    host_parallelize(cell, 18)
    _ = w
    _ = h
    return result^


def objective(source: List[Float32], reconstruction: List[Float64], beta: Float64) -> Float64:
    var terms = List[Float64](unsafe_uninit_length=18)
    var tp = terms.unsafe_ptr()
    var xp = source.unsafe_ptr()
    var rp = reconstruction.unsafe_ptr()
    def cell(index: Int) {imm tp, imm xp, imm rp, imm beta}:
        var x = Float64(xp[index])
        var y = rp[index]
        var value = (x - y) * (x - y)
        if beta == 1:
            value = 2 * (x * log(x / y) - x + y)
        elif beta == 0:
            value = 2 * (x / y - log(x / y) - 1)
        tp.unsafe_store(index, value)
    host_parallelize(cell, 18)
    # Scalar reduction of independent oracle contributions; no solver work.
    var total = Float64(0)
    for value in terms:
        total += value
    _ = source
    _ = reconstruction
    return sqrt(max(total, Float64(0)))


def factors(value: List[Float32], count: Int) raises:
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var vp = value.unsafe_ptr() + 1
    def cell(index: Int) {imm fp, imm vp}:
        fp.unsafe_store(index, Int(not isfinite(vp[index]) or vp[index] < 0))
    host_parallelize(cell, count)
    for failed in failures:
        require(failed == 0, "NMF factors must be finite and nonnegative")
    _ = value
    guards(value, "NMF factor output")


def literal_inverse_values(value: List[Float32], changed: Bool) raises:
    var failures = List[Int](length=6, fill=0)
    var fp = failures.unsafe_ptr()
    var vp = value.unsafe_ptr()
    def cell(index: Int) {imm fp, imm vp, imm changed}:
        var expected = Float32(5 + 2 * (index % 3))
        if changed and index % 3 == 0:
            expected += 0.5
        fp.unsafe_store(index, Int(vp[index] != expected))
    host_parallelize(cell, 6)
    for failed in failures:
        require(failed == 0, "NMF literal inverse fills every word from current components")
    _ = value


def fixed_components(actual: List[Float32], expected: List[Float32]) raises:
    var failures = List[Int](length=6, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr() + 1
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        fp.unsafe_store(index, Int(ap[index] != ep[index]))
    host_parallelize(cell, 6)
    for failed in failures:
        require(failed == 0, "NMF transform retains fixed H")
    _ = actual
    _ = expected


def inverse_literal(driver: NMFNativeDriver) raises:
    var source = List[Float32](length=4, fill=1)
    var components: List[Float32] = [1, 2, 3, 4, 5, 6]
    var original = source.copy()
    var scratch = NMFScratch(2, 3)
    try:
        driver.inverse(NMFInverseRequest(Int(source.unsafe_ptr()), Int(components.unsafe_ptr()), scratch.address(), 2, 3, 2))
        literal_inverse_values(scratch.values, False)
        components[0] += 0.5
        driver.inverse(NMFInverseRequest(Int(source.unsafe_ptr()), Int(components.unsafe_ptr()), scratch.address(), 2, 3, 2))
        literal_inverse_values(scratch.values, True)
        unchanged(source, original, "NMF inverse source")
    finally:
        _ = scratch^
        _ = source^
        _ = components^


def fixture(driver: NMFNativeDriver, unit: Int, host: Bool) raises:
    if unit == 0:
        require(driver.metadata(0) == 1 and driver.metadata(2) == 3, "NMF native mode and fit/inverse capabilities")
        var vendor = driver.metadata(1)
        require(vendor == 1 if host else (vendor >= 2 and vendor <= 4), "NMF native provider vendor")
        return
    if unit == 17:
        inverse_literal(driver)
        return
    var source = input_values()
    if unit == 15:
        source[0] = bitcast[DType.float32](UInt32(0x7FC00042))
    elif unit == 16:
        source[0] = -1
    var w_initial = List[Float32](length=12, fill=1)
    var h_initial = List[Float32](length=6, fill=1)
    if unit == 11 or unit == 12:
        h_initial[0] = 1
        h_initial[1] = 2
        h_initial[2] = 1
        h_initial[3] = 2
        h_initial[4] = 1
        h_initial[5] = 3
    var source_before = source.copy()
    var w_before = w_initial.copy()
    var h_before = h_initial.copy()
    var w_output = List[Float32](length=14, fill=PCA_TEST_GUARD)
    var h_output = List[Float32](length=8, fill=PCA_TEST_GUARD)
    var w_untouched = w_output.copy()
    var h_untouched = h_output.copy()
    var initialization = (unit - 1) % 5 if unit <= 10 else 4
    var solver = (unit - 1) // 5 if unit <= 10 else 1
    var beta = Float64(1 if unit == 13 or unit == 18 else (0 if unit == 14 else 2))
    var updating = unit != 11 and unit != 12
    if not updating:
        initialization = 5
        solver = unit - 11
    if unit == 18:
        solver = 0
    var args = NmfArgs(beta, updating, 3, 1e-4, 0, 0, 0, 0, False, 7)
    var request = NMFFitRequest(Int(source.unsafe_ptr()), Int(w_initial.unsafe_ptr()), Int(h_initial.unsafe_ptr()),
        Int(w_output.unsafe_ptr()) + 4, Int(h_output.unsafe_ptr()) + 4,
        2147483648 if unit == 19 else 6, 3, 2, initialization, solver, args^, True)
    try:
        if unit == 16 or unit == 18 or unit == 19:
            var expected = String("Negative values in data passed to NMF (input X)")
            if unit == 18:
                expected = "x_decomp: nmf cd takes beta 2"
            elif unit == 19:
                expected = "x_decomp: dimension out of range"
            var refused = False
            try:
                _ = driver.fit(request)
            except error:
                refused = String(error) == expected
            require(refused, "NMF provider preserves native refusal")
            unchanged(w_output, w_untouched, "NMF refused W output")
            unchanged(h_output, h_untouched, "NMF refused H output")
        else:
            var result = driver.fit(request)
            if unit == 15:
                require(result.finite == 1 and result.iterations == 0 and result.error == 0,
                        "NMF deferred finite status returns before publishing factors")
                unchanged(w_output, w_untouched, "NMF nonfinite W output")
                unchanged(h_output, h_untouched, "NMF nonfinite H output")
            else:
                require(result.finite == 0 and result.iterations >= 1 and result.iterations <= 3,
                        "NMF fit status and original iteration bound")
                factors(w_output, 12)
                factors(h_output, 6)
                var reconstruction = product(w_output, h_output)
                if updating:
                    var expected = objective(source, reconstruction, beta)
                    require(isfinite(expected) and isfinite(result.error)
                        and abs(result.error - expected) <= Float64(2e-4) + Float64(2e-4) * abs(expected),
                        "NMF reconstruction error agrees with independent beta-divergence")
                    if beta == 2:
                        require(expected < sqrt(Float64(79)), "NMF improves on zero reconstruction")
                    if initialization == 4:
                        var initial_product = List[Float64](length=18, fill=2)
                        var before = objective(source, initial_product, beta)
                        require(expected <= before + Float64(2e-4) * max(before, Float64(1)),
                                "NMF custom initialization improves its independent objective")
                else:
                    require(result.error == 0, "NMF fixed-H transform does not publish a fit error")
                    fixed_components(h_output, h_initial)
                var inverse = List[Float32](length=20, fill=PCA_TEST_GUARD)
                try:
                    driver.inverse(NMFInverseRequest(Int(w_output.unsafe_ptr()) + 4, Int(h_output.unsafe_ptr()) + 4,
                        Int(inverse.unsafe_ptr()) + 4, 6, 3, 2))
                    close_values(inverse, reconstruction, 18, 2e-4, 2e-4, "NMF mathematical inverse")
                    guards(inverse, "NMF inverse output")
                finally:
                    _ = inverse^
                    _ = reconstruction^
        unchanged(source, source_before, "NMF source")
        unchanged(w_initial, w_before, "NMF W initialization")
        unchanged(h_initial, h_before, "NMF H initialization")
        guards(w_output, "NMF W output")
        guards(h_output, "NMF H output")
    finally:
        _ = source^
        _ = w_initial^
        _ = h_initial^
        _ = w_output^
        _ = h_output^
