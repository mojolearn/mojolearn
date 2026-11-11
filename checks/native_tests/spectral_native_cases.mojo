# SPDX-License-Identifier: Apache-2.0
"""Original Spectral numerical graphs over the typed production providers.

Expected relations are the original tests' independent column/diagonal and
partition oracles. No eigenvector sign or repeated-eigenvalue basis is invented.
Original inputs are saved test data, never fitted outputs or release hashes.
"""
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.dense_coo import nonzero_f32_count, nonzero_f32_fill, F32P, I32P
from core.spectral_native_api import SpectralDriver, SpectralInput, SpectralStateView, SpectralFitOptions, SpectralFitRequest, SpectralEmbeddingOptions, SpectralEmbeddingRequest
from checks.native_tests.spectral_dense_cases import require, clear, read_words, floats, unchanged

comptime SPECTRAL_NATIVE_COMMON_FIXTURES = 7
comptime SPECTRAL_NATIVE_DENSE_FIXTURES = 2
comptime SPECTRAL_NATIVE_GUARD = Float32(12521.25)


def source(name: String, count: Int) raises -> List[Float32]:
    var words = read_words(name, count)
    var values = floats(words)
    _ = words^
    return values^


def source_unchanged(name: String, values: List[Float32]) raises:
    var words = read_words(name, len(values))
    unchanged(values, words)
    _ = values
    _ = words^


def guarded(count: Int) -> List[Float32]:
    var values = List[Float32](length=count + 2, fill=bitcast[DType.float32](UInt32(0x7fc01234)))
    values[0] = SPECTRAL_NATIVE_GUARD
    values[count + 1] = SPECTRAL_NATIVE_GUARD
    return values^


def guards(values: List[Float32]) raises:
    require(len(values) >= 2 and values[0] == SPECTRAL_NATIVE_GUARD
        and values[len(values)-1] == SPECTRAL_NATIVE_GUARD, "Spectral native output guards")


def finite(values: List[Float32]) raises:
    guards(values)
    var count = len(values) - 2
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var vp = values.unsafe_ptr() + 1
    def cell(index: Int) {imm fp, imm vp}:
        fp.unsafe_store(index, Int(not isfinite(vp[index])))
    host_parallelize(cell, count)
    clear(failures, "Spectral original finite embedding assertion")
    _ = values


def same_columns(left: List[Float32], left_width: Int, right: List[Float32],
                 right_width: Int, rows: Int, offset: Int) raises -> Bool:
    require(len(left) == rows * left_width + 2 and len(right) == rows * right_width + 2,
        "Spectral compared output extents")
    var count = rows * right_width
    var mismatches = List[Int](length=count, fill=0)
    var mp = mismatches.unsafe_ptr()
    var lp = left.unsafe_ptr() + 1
    var rp = right.unsafe_ptr() + 1
    def cell(index: Int) {imm mp, imm lp, imm rp, imm left_width, imm right_width, imm offset}:
        var i = index // right_width
        var j = index % right_width
        mp.unsafe_store(index, Int(bitcast[DType.uint32](lp[i * left_width + j + offset])
            != bitcast[DType.uint32](rp[index])))
    host_parallelize(cell, count)
    var different = 0
    for value in mismatches:
        different += value
    guards(left)
    guards(right)
    _ = left
    _ = right
    return different == 0


def prepared_input(values: List[Float32], rows: Int, features: Int, route: Int,
                   mut graph_rows: List[Int32], mut graph_columns: List[Int32],
                   mut graph_values: List[Float32]) raises -> SpectralInput:
    var address = Int(values.unsafe_ptr())
    if route != 1:
        _ = values
        return SpectralInput(address, 0, 0, 0, route, rows, features, 0)
    # The existing shared dense-to-COO preparation, independently covered by
    # spectral_dense_cpu. This only constructs admitted input, never an oracle.
    var count = nonzero_f32_count(F32P(unsafe_from_address=address), rows * rows)
    graph_rows = List[Int32](length=count, fill=0)
    graph_columns = List[Int32](length=count, fill=0)
    graph_values = List[Float32](length=count, fill=0)
    require(nonzero_f32_fill(F32P(unsafe_from_address=address), rows, rows,
        I32P(unsafe_from_address=Int(graph_rows.unsafe_ptr())),
        I32P(unsafe_from_address=Int(graph_columns.unsafe_ptr())),
        F32P(unsafe_from_address=Int(graph_values.unsafe_ptr())), count) == count,
        "Spectral complete precomputed graph input")
    _ = values
    return SpectralInput(0, Int(graph_rows.unsafe_ptr()), Int(graph_columns.unsafe_ptr()),
        Int(graph_values.unsafe_ptr()), 1, rows, features, count)


def embed(driver: SpectralDriver, values: List[Float32], rows: Int, features: Int,
          route: Int, output_columns: Int, neighbors: Int, normalize: Bool, drop: Bool,
          seed: Int, tolerance: Float64, mut destination: List[Float32]) raises:
    var graph_rows = List[Int32]()
    var graph_columns = List[Int32]()
    var graph_values = List[Float32]()
    try:
        var input = prepared_input(values, rows, features, route, graph_rows, graph_columns, graph_values)
        var options = SpectralEmbeddingOptions(output_columns + Int(drop), output_columns,
            neighbors, normalize, drop, seed, tolerance)
        var result = driver.embed(SpectralEmbeddingRequest(input,
            Int(destination.unsafe_ptr() + 1), options))
        require(result == output_columns, "Spectral original output component count")
        finite(destination)
    finally:
        _ = values
        _ = destination
        _ = graph_rows^
        _ = graph_columns^
        _ = graph_values^


def fit(driver: SpectralDriver, values: List[Float32], rows: Int, features: Int,
        route: Int, clusters: Int, columns: Int, neighbors: Int, seed: Int,
        keep: Bool, state: SpectralStateView, mut labels: List[Int32],
        mut embedding: List[Float32]) raises:
    var graph_rows = List[Int32]()
    var graph_columns = List[Int32]()
    var graph_values = List[Float32]()
    try:
        var input = prepared_input(values, rows, features, route, graph_rows, graph_columns, graph_values)
        var options = SpectralFitOptions(clusters, columns, 10, neighbors, 1e-5, seed, keep)
        require(driver.fit(SpectralFitRequest(input, Int(labels.unsafe_ptr() + 1),
            Int(embedding.unsafe_ptr() + 1), state, options)) == columns,
            "Spectral original fitted component count")
        require(labels[0] == Int32(-31) and labels[len(labels)-1] == Int32(-31), "Spectral native label guards")
        finite(embedding)
    finally:
        _ = values
        _ = labels
        _ = embedding
        _ = graph_rows^
        _ = graph_columns^
        _ = graph_values^


def simple_embedding(driver: SpectralDriver, unit: Int, dense: Bool) raises:
    var n = 80 if unit == 2 else 150
    var precomputed = unit == 2
    var name = "embedding80-affinity.f32" if precomputed else "embedding150-input.f32"
    var values = source(name, n * n if precomputed else n * 2)
    var destination = guarded(n * 2)
    try:
        embed(driver, values, n, n if precomputed else 2,
            (2 if dense else 1) if precomputed else 0, 2, n // 10, True, True, 5,
            Float64(1e-2) if unit == 1 else Float64(1e-5), destination)
        source_unchanged(name, values)
    finally:
        _ = values^
        _ = destination^


def clustering_relation(driver: SpectralDriver) raises:
    var values = source("embedding150-input.f32", 150 * 2)
    var labels = List[Int32](length=152, fill=-31)
    var clustered = guarded(150 * 4)
    var dropped = guarded(150 * 3)
    var kept = guarded(150 * 4)
    try:
        fit(driver, values, 150, 2, 0, 3, 4, 10, 3, False,
            SpectralStateView(0,0,0,0), labels, clustered)
        embed(driver, values, 150, 2, 0, 3, 10, True, True, 3, 1e-5, dropped)
        embed(driver, values, 150, 2, 0, 4, 10, True, False, 3, 1e-5, kept)
        require(same_columns(clustered,4,dropped,3,150,1),
            "Original embedding equals clustering columns after dropping the trivial first column")
        require(same_columns(clustered,4,kept,4,150,0),
            "Original drop_first=False manifold embedding equals clustering embedding")
        source_unchanged("embedding150-input.f32", values)
    finally:
        _ = values^
        _ = labels^
        _ = clustered^
        _ = dropped^
        _ = kept^


def diagonal_relation(driver: SpectralDriver, dense: Bool) raises:
    var values = source("embedding150-affinity.f32", 150 * 150)
    var hollow = source("embedding150-hollow.f32", 150 * 150)
    var full_result = guarded(150 * 2)
    var hollow_result = guarded(150 * 2)
    try:
        embed(driver, values, 150, 150, 2 if dense else 1, 2, 15, True, True, 3, 1e-5, full_result)
        embed(driver, hollow, 150, 150, 2 if dense else 1, 2, 15, True, True, 3, 1e-5, hollow_result)
        require(same_columns(full_result,2,hollow_result,2,150,0),
            "Original positive affinity diagonal and explicit hollow graph have identical embedding")
        source_unchanged("embedding150-affinity.f32", values)
        source_unchanged("embedding150-hollow.f32", hollow)
    finally:
        _ = values^
        _ = hollow^
        _ = full_result^
        _ = hollow_result^


def normalization_relation(driver: SpectralDriver) raises:
    var values = source("embedding150-input.f32", 150 * 2)
    var normalized = guarded(150 * 2)
    var raw = guarded(150 * 2)
    try:
        embed(driver, values, 150, 2, 0, 2, 10, True, True, 3, 1e-5, normalized)
        embed(driver, values, 150, 2, 0, 2, 10, False, True, 3, 1e-5, raw)
        require(not same_columns(normalized,2,raw,2,150,0), "Original unnormalized Laplacian differs and stays finite")
        source_unchanged("embedding150-input.f32", values)
    finally:
        _ = values^
        _ = normalized^
        _ = raw^


def separated_partition(driver: SpectralDriver) raises:
    var values = source("host120-input.f32", 120 * 3)
    var labels = List[Int32](length=122, fill=-31)
    var embedding = guarded(120 * 2)
    try:
        fit(driver, values, 120, 3, 0, 2, 2, 8, 5, False, SpectralStateView(0,0,0,0), labels, embedding)
        require(labels[1] != labels[61], "Original two separated blobs have distinct cluster labels")
        var failures = List[Int](length=120, fill=0)
        var fp = failures.unsafe_ptr()
        var lp = labels.unsafe_ptr() + 1
        def row(index: Int) {imm fp, imm lp}:
            var label = lp[index]
            fp.unsafe_store(index, Int(label < Int32(0) or label > Int32(1)
                or label != lp[0 if index < 60 else 60]))
        host_parallelize(row, 120)
        clear(failures, "Original120-row planted partition, independent of label permutation")
        source_unchanged("host120-input.f32", values)
    finally:
        _ = values^
        _ = labels^
        _ = embedding^


def fixture(driver: SpectralDriver, unit: Int) raises:
    if unit < 3:
        simple_embedding(driver, unit, False)
    elif unit == 3:
        clustering_relation(driver)
    elif unit == 4:
        diagonal_relation(driver, False)
    elif unit == 5:
        normalization_relation(driver)
    elif unit == 6:
        separated_partition(driver)
    elif unit == 7:
        simple_embedding(driver, 2, True)
    else:
        diagonal_relation(driver, True)
