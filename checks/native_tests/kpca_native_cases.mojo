# SPDX-License-Identifier: Apache-2.0
"""Actual typed KernelPCA graphs with independent mathematical references.

The four original 80x6 workloads retain their original score/eigen tolerances.
Tiny dyadic linear cases separately expose the global-mean centering defect.
Explicit Lanczos/resident cases exercise existing backend options; they do not
claim that the original 80-row public auto route selects those options.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorCall
from core.kpca_native_api import KPCANativeDriver, KPCAEigenRequest, KPCA_MODE, KPCA_VENDOR, KPCA_CAPABILITIES, KPCA_CAP_EIGEN
from core.kpca_native_plan import KPCAKernelOptions, KPCANeighborDriver, KPCAScratch, KPCA_PRECOMPUTED, KPCAAlphaRequest, KPCAQueryRequest, KPCACenteredQuery, KPCAFitPrepareRequest, KPCADenseFit
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.native_tests.pca_native_cases import require, guards, unchanged, PCA_TEST_GUARD

comptime KPCA_NATIVE_FIXTURES = 15
comptime KPCA_RESIDENT_FIXTURES = 1


def input_values(profile: String, part: String, count: Int) raises -> List[Float32]:
    var values = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/kpca/" + profile + "-" + part + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * count, "KPCA input fixture extent")
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def word(index: Int) {imm bp, imm vp}:
            var at = 4 * index
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            vp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return values^


def reference_values(profile: String, part: String, count: Int) raises -> List[Float64]:
    var values = List[Float64](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/kpca/" + profile + "-" + part + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 8 * count, "KPCA reference fixture extent")
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def word(index: Int) {imm bp, imm vp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed-width decoding, not an algorithm fold.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(8 * byte)
            vp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(word, count)
        _ = bytes^
    return values^


def compare(actual: List[Float32], offset: Int, expected: List[Float64], rows: Int,
            count: Int, expected_width: Int, rtol: Float64, atol: Float64,
            absolute: Bool, name: String) raises:
    var failures = List[Int](length=rows * count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr() + offset
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm ep, imm count, imm expected_width, imm rtol, imm atol, imm absolute}:
        var observed = Float64(ap[index])
        var wanted = ep[(index // count) * expected_width + index % count]
        if absolute:
            observed = abs(observed)
            wanted = abs(wanted)
        fp.unsafe_store(index, Int(not (abs(observed - wanted) <= atol + rtol * abs(wanted))))
    host_parallelize(cell, rows * count)
    for failure in failures:
        require(failure == 0, name + " independent oracle tolerance")
    _ = actual
    _ = expected


def refusal(driver: KPCANativeDriver, unit: Int) raises:
    var storage = List[Float32](length=10, fill=PCA_TEST_GUARD)
    var address = Int(storage.unsafe_ptr()) + 4
    var expected = "x_decomp: null float32 buffer address" if unit == 12 else "x_decomp: dimension out of range"
    var request = KPCAEigenRequest(0 if unit == 12 else address, address, address, address, address,
        address, False, 2 if unit == 12 else 2147483648, 1, 2, False, False, 1.0)
    var refused = False
    try:
        _ = driver.eigen(request)
    except error:
        require(String(error) == expected, "KPCA exact native refusal")
        refused = True
    finally:
        guards(storage, "KPCA refusal storage")
        _ = storage
    require(refused, "KPCA malformed native request admitted")


def graph(driver: KPCANativeDriver, near: KPCANeighborDriver, decomp: NativeDescriptorCall, unit: Int) raises:
    var profile = String("rbf")
    var kind = 2
    if unit == 2:
        profile = "linear"
        kind = 0
    elif unit == 3:
        profile = "poly"
        kind = 1
    elif unit == 4:
        profile = "cosine"
        kind = 5
    elif unit == 5 or unit == 6:
        profile = "rbf02"
    elif unit == 7:
        profile = "positive2"
        kind = 0
    elif unit >= 8 and unit <= 10:
        profile = "signed2"
        kind = 0
    elif unit == 14:
        profile = "zero2"
        kind = 0
    var small = unit >= 7 and unit <= 10 or unit == 14
    var rows = 2 if small else 80
    var query_rows = 2 if small else 20
    var features = 1 if small else 6
    var reference_width = 2 if small else 3
    var requested = -1 if unit == 10 else reference_width
    var expected_kept = 0 if unit == 14 else (1 if unit == 9 or unit == 10 else reference_width)
    var remove_zero = unit == 9 or unit == 14
    var resident = unit == 15
    var lanczos = unit == 11 or resident
    var precomputed = unit == 6
    var gamma = Float64(0.2) if unit == 5 or unit == 6 else Float64(1.0) / Float64(features)
    var source = input_values(profile, "source", rows * features)
    var query = input_values(profile, "query", query_rows * features)
    var original_source = source.copy()
    var original_query = query.copy()
    var columns = List[Float32](length=rows + 2, fill=PCA_TEST_GUARD)
    # Exactly one payload word: a repeated-column reduction cannot overwrite
    # a hidden oversized global-mean buffer without tripping these canaries.
    var total = List[Float32](length=3, fill=PCA_TEST_GUARD)
    var values = List[Float32](length=rows + 2, fill=PCA_TEST_GUARD)
    var vectors = List[Float32](length=rows * reference_width + 2, fill=PCA_TEST_GUARD)
    var dummy = List[Float32](length=1, fill=Float32(0))
    var pre_kernel = KPCAScratch(rows if precomputed else 0, rows)
    var pre_query = KPCAScratch(query_rows if precomputed else 0, rows)
    var centered: Optional[KPCADenseFit] = None
    var source_address = Int(source.unsafe_ptr())
    var query_address = Int(query.unsafe_ptr())
    var columns_address = Int(columns.unsafe_ptr()) + 4
    var total_address = Int(total.unsafe_ptr()) + 4
    var centered_address = Int(dummy.unsafe_ptr())
    var values_address = Int(values.unsafe_ptr()) + 4
    var vectors_address = Int(vectors.unsafe_ptr()) + 4
    var options = KPCAKernelOptions(kind, 3, gamma, 1.0)
    try:
        if precomputed:
            near.kernel(source_address, source_address, pre_kernel.address(), rows, rows, features, options)
            near.kernel(query_address, source_address, pre_query.address(), query_rows, rows, features, options)
            source_address = pre_kernel.address()
            query_address = pre_query.address()
            options.kind = KPCA_PRECOMPUTED
            features = rows
        if not resident:
            centered = Optional[KPCADenseFit](KPCADenseFit(near, decomp,
                KPCAFitPrepareRequest(source_address, columns_address, total_address, rows, features, options.copy())))
            centered_address = centered.value().address()
            var expected_centered = reference_values(profile, "centered", rows * rows)
            compare(centered.value().centered.values, 0, expected_centered, rows, rows, rows,
                0.0 if small else 2e-5, 0.0 if small else 2e-3, False, "KPCA centered training kernel")
        var kept = driver.eigen(KPCAEigenRequest(centered_address, source_address, values_address, vectors_address,
            columns_address, total_address, resident, rows, features, requested, lanczos, remove_zero, gamma))
        require(kept == expected_kept, "KPCA retained positive/null eigenmode count")
        var expected_columns = reference_values(profile, "columns", rows)
        var expected_total = reference_values(profile, "total", 1)
        var expected_values = reference_values(profile, "values", reference_width)
        compare(columns, 1, expected_columns, 1, rows, rows, 0.0 if small else 2e-5,
            0.0 if small else 2e-5, False, "KPCA column means")
        compare(total, 1, expected_total, 1, 1, 1, 0.0 if small else 2e-5,
            0.0 if small else 2e-5, False, "KPCA true global mean")
        compare(values, 1, expected_values, 1, kept, reference_width, 1e-3,
            1e-6 if small else 0.0, False, "KPCA eigenvalues")
        if kept > 0:
            query_graph(near, profile, source_address, query_address, columns_address, total_address,
                vectors_address, values_address, rows, query_rows, features, kept, reference_width, options)
        guards(columns, "KPCA columns")
        guards(total, "KPCA scalar global mean")
        guards(values, "KPCA eigenvalues")
        guards(vectors, "KPCA eigenvectors")
        unchanged(source, original_source, "KPCA borrowed source")
        unchanged(query, original_query, "KPCA borrowed query")
    finally:
        _ = source
        _ = query
        _ = original_source
        _ = original_query
        _ = columns
        _ = total
        _ = values
        _ = vectors
        _ = dummy
        _ = pre_kernel
        _ = pre_query
        _ = centered


def query_graph(near: KPCANeighborDriver, profile: String, source: Int, query: Int,
                columns: Int, total: Int, vectors: Int, values: Int, rows: Int,
                query_rows: Int, features: Int, count: Int, reference_width: Int,
                options: KPCAKernelOptions) raises:
    var fit_scores = List[Float32](length=rows * count + 2, fill=PCA_TEST_GUARD)
    var alphas = List[Float32](length=rows * count + 2, fill=PCA_TEST_GUARD)
    var fused = List[Float32](length=query_rows * count + 2, fill=PCA_TEST_GUARD)
    var decomposed = List[Float32](length=query_rows * count + 2, fill=PCA_TEST_GUARD)
    var alpha_address = Int(alphas.unsafe_ptr()) + 4
    try:
        near.alpha(KPCAAlphaRequest(vectors, values, Int(fit_scores.unsafe_ptr()) + 4, rows, count, False))
        near.alpha(KPCAAlphaRequest(vectors, values, alpha_address, rows, count, True))
        var expected_fit = reference_values(profile, "fit-scores", rows * reference_width)
        var expected_query = reference_values(profile, "query-scores", query_rows * reference_width)
        compare(fit_scores, 1, expected_fit, rows, count, reference_width, 2e-3, 2e-3, True, "KPCA fitted scores")
        var request = KPCAQueryRequest(query, source, columns, total, alpha_address,
            Int(fused.unsafe_ptr()) + 4, query_rows, rows, features, count, options.copy())
        near.fused_query(request)
        compare(fused, 1, expected_query, query_rows, count, reference_width, 2e-3, 2e-3, True, "KPCA fused query")
        var prepared = KPCACenteredQuery(near, request)
        var expected_centered = reference_values(profile, "query-centered", query_rows * rows)
        var small = rows == 2
        compare(prepared.centered.values, 0, expected_centered, query_rows, rows, rows,
            0.0 if small else 2e-5, 0.0 if small else 2e-3, False, "KPCA centered query kernel")
        prepared.finish(near, alpha_address, Int(decomposed.unsafe_ptr()) + 4, count)
        compare(decomposed, 1, expected_query, query_rows, count, reference_width, 2e-3, 2e-3, True, "KPCA decomposed query")
        guards(fit_scores, "KPCA fit scores")
        guards(alphas, "KPCA alphas")
        guards(fused, "KPCA fused query")
        guards(decomposed, "KPCA decomposed query")
        _ = prepared
    finally:
        _ = fit_scores
        _ = alphas
        _ = fused
        _ = decomposed


def fixture(driver: KPCANativeDriver, near: KPCANeighborDriver, decomp: NativeDescriptorCall,
            unit: Int, host: Bool) raises:
    if unit == 0:
        require(driver.metadata(KPCA_MODE) == Int(GLOBAL_NUMERIC_MODE), "KPCA actual provider mode")
        var vendor = driver.metadata(KPCA_VENDOR)
        require(vendor == 1 if host else vendor >= 2 and vendor <= 4, "KPCA actual provider vendor")
        require(driver.metadata(KPCA_CAPABILITIES) == KPCA_CAP_EIGEN, "KPCA eigen owning-image capability")
    elif unit == 12 or unit == 13:
        refusal(driver, unit)
    else:
        graph(driver, near, decomp, unit)
