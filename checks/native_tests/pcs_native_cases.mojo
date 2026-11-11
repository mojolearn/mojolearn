# SPDX-License-Identifier: Apache-2.0
"""PCS numerical graphs against analytic values and full polynomial expansion.

The independent Float64 oracle enumerates feature monomials and places their
products by the sum of hash indices. It forms neither intermediate count
sketches nor circular convolutions. Its independent output cells run in
parallel. Product kernels and their existing routes are unchanged.
"""
from std.math import sqrt
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.pcs_native_api import PCSNativeDriver, PCSFitPlan, pcs_options
from checks.native_tests.pca_native_cases import require, guards, unchanged, close_values, PCA_TEST_GUARD

comptime PCS_NATIVE_FIXTURES = 12
comptime PCS_HASH_GUARD = Int32(15337)


def original_source() raises -> List[Float32]:
    var source = List[Float32](unsafe_uninit_length=360)
    with open("checks/native_tests/fixtures/pcs/source.f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 1440, "PCS original 60x6 source extent")
        var bp = bytes.unsafe_ptr()
        var sp = source.unsafe_ptr()
        def cell(index: Int) {imm bp, imm sp}:
            var at = 4 * index
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            sp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(cell, 360)
        _ = bytes^
    return source^


def hash_storage(indices: List[Int32], signs: List[Int32], total: Int, components: Int) raises:
    require(indices[0] == PCS_HASH_GUARD and indices[total + 1] == PCS_HASH_GUARD,
        "PCS index table guards")
    require(signs[0] == PCS_HASH_GUARD and signs[total + 1] == PCS_HASH_GUARD,
        "PCS sign table guards")
    var failures = List[Int](length=total, fill=0)
    var fp = failures.unsafe_ptr()
    var ip = indices.unsafe_ptr() + 1
    var sp = signs.unsafe_ptr() + 1
    def cell(index: Int) {imm fp, imm ip, imm sp, imm components}:
        fp.unsafe_store(index, Int(Int(ip[index]) < 0 or Int(ip[index]) >= components
            or (sp[index] != Int32(-1) and sp[index] != Int32(1))))
    host_parallelize(cell, total)
    for failure in failures:
        require(failure == 0, "PCS actual draw range/sign")
    _ = indices
    _ = signs


def same_hashes(actual: List[Int32], expected: List[Int32]) raises:
    require(len(actual) == len(expected), "PCS hash storage extent")
    var failures = List[Int](length=len(actual), fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        fp.unsafe_store(index, Int(ap[index] != ep[index]))
    host_parallelize(cell, len(actual))
    for failure in failures:
        require(failure == 0, "PCS query mutated borrowed hash storage")
    _ = actual
    _ = expected


def polynomial_oracle(source: List[Float32], indices: List[Int32], signs: List[Int32],
                      rows: Int, features: Int, augmented: Int, degree: Int,
                      components: Int, gamma: Float64, coef0: Float64) -> List[Float64]:
    var expected = List[Float64](length=rows * components, fill=Float64(0))
    var terms = 1
    for _ in range(degree):  # Fixed test polynomial degree, at most three.
        terms *= augmented
    var sg = sqrt(gamma)
    var sc = sqrt(coef0)
    var xp = source.unsafe_ptr()
    var ip = indices.unsafe_ptr() + 1
    var sp = signs.unsafe_ptr() + 1
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm xp, imm ip, imm sp, imm ep, imm terms, imm sg, imm sc,
                          imm features, imm augmented, imm degree, imm components}:
        var row = index // components
        var bucket = index % components
        var sum = Float64(0)
        # Independent test oracle: exhaustive feature-tensor expansion, not
        # the production sketch/convolution recurrence. At most 7^3 terms.
        for term in range(terms):
            var digits = term
            var hash_sum = 0
            var product = Float64(1)
            for factor in range(degree):
                var feature = digits % augmented
                digits //= augmented
                var table = factor * augmented + feature
                hash_sum += Int(ip[table])
                var value = sg * Float64(xp[row * features + feature]) if feature < features else sc
                product *= value * Float64(sp[table])
            if hash_sum % components == bucket:
                sum += product
        ep.unsafe_store(index, sum)
    host_parallelize(cell, rows * components)
    _ = source
    _ = indices
    _ = signs
    return expected^


def fixture(driver: PCSNativeDriver, unit: Int) raises:
    var profile = unit // 2
    var old_items = unit % 2 == 0
    var rows = 60 if profile == 0 else (0 if profile == 5 else 2)
    var features = 6 if profile == 0 else (0 if profile == 4 else 2)
    var degree = 1 if profile == 1 else (2 if profile == 2 or profile == 4 else 3)
    var components = 32 if profile == 0 else (3 if profile == 2 else (1 if profile == 3 else 4))
    var gamma = Float64(0.5) if profile == 0 else Float64(1)
    var coef0 = Float64(1) if profile == 0 or profile == 3 or profile == 5 else Float64(0)
    var source: List[Float32] = [2.0, 3.0, -1.0, 4.0]
    if profile == 0:
        source = original_source()
    elif profile == 4 or profile == 5:
        source = List[Float32](length=1, fill=Float32(0))  # Nonnull storage for a logically empty matrix.
    var original = source.copy()
    var plan = PCSFitPlan(pcs_options(degree, components), features, coef0)
    var total = degree * plan.augmented
    var indices = List[Int32](length=total + 2, fill=PCS_HASH_GUARD)
    var signs = List[Int32](length=total + 2, fill=PCS_HASH_GUARD)
    var destination = List[Float32](length=rows * components + 2, fill=PCA_TEST_GUARD)
    var ia = Int(indices.unsafe_ptr()) + 4
    var sa = Int(signs.unsafe_ptr()) + 4
    try:
        if profile == 0 or profile == 5:
            driver.require_draw()
            driver.draw(plan.draw(7, ia, sa))
        elif profile == 1:
            indices[1] = 1
            indices[2] = 1
            signs[1] = 1
            signs[2] = -1
        elif profile == 2:
            indices[1] = 0
            indices[2] = 1
            indices[3] = 2
            indices[4] = 0
            signs[1] = 1
            signs[2] = 1
            signs[3] = 1
            signs[4] = -1
        elif profile == 3:
            var ip = indices.unsafe_ptr() + 1
            var sp = signs.unsafe_ptr() + 1
            def hash_cell(index: Int) {imm ip, imm sp}:
                ip.unsafe_store(index, Int32(0))
                sp.unsafe_store(index, Int32(1))
            host_parallelize(hash_cell, total)
        # Empty augmentation has no draw and no hash-table payload.
        hash_storage(indices, signs, total, components)
        var original_indices = indices.copy()
        var original_signs = signs.copy()
        var expected: List[Float64] = []
        if profile == 1:
            expected = [0.0, -1.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0]
        elif profile == 2:
            # (x + y z)(-y + x z^2) mod (z^3 - 1) = -y^2 z + x^2 z^2.
            expected = [0.0, -9.0, 4.0, 0.0, -16.0, 1.0]
        elif profile == 3:
            # One bucket and positive signs yield (x + y + 1)^3.
            expected = [216.0, 64.0]
        else:
            expected = polynomial_oracle(source, indices, signs, rows, features, plan.augmented,
                degree, components, gamma, coef0)
        driver.query(plan.query(Int(source.unsafe_ptr()), ia, sa, Int(destination.unsafe_ptr()) + 4,
            rows, gamma, coef0, old_items))
        close_values(destination, expected, rows * components, 1e-3 if profile == 0 else 0.0,
            1e-3 if profile == 0 else 0.0, "PCS independent polynomial oracle")
        guards(destination, "PCS output")
        hash_storage(indices, signs, total, components)
        same_hashes(indices, original_indices)
        same_hashes(signs, original_signs)
        unchanged(source, original, "PCS borrowed input")
    finally:
        _ = source
        _ = original
        _ = indices
        _ = signs
        _ = destination
