# SPDX-License-Identifier: Apache-2.0
"""Original full Spectral prediction workloads and supported state refusals.

Training-label agreement and extension-versus-fit embedding remain report-only
in the Python controller; neither becomes a native correctness gate here.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.spectral_native_api import SpectralDriver, SpectralStateView, SpectralQueryShape, SpectralQueryRequest
from checks.native_tests.spectral_native_cases import source, source_unchanged, guarded, finite, guards, fit, same_columns
from checks.native_tests.spectral_dense_cases import require, clear

comptime SPECTRAL_PREDICTION_FIXTURES = 6


def labels_valid(labels: List[Int32]) raises:
    require(labels[0] == Int32(-31) and labels[len(labels)-1] == Int32(-31), "Spectral prediction label guards")
    var count = len(labels) - 2
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var lp = labels.unsafe_ptr() + 1
    def row(index: Int) {imm fp, imm lp}:
        fp.unsafe_store(index, Int(lp[index] < Int32(0) or lp[index] > Int32(2)))
    host_parallelize(row, count)
    clear(failures, "Original Spectral prediction labels lie in0..2")
    _ = labels


def labels_equal(left: List[Int32], right: List[Int32], reverse: Bool) raises:
    require(len(left) == len(right), "Spectral compared label extents")
    var count = len(left) - 2
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var lp = left.unsafe_ptr() + 1
    var rp = right.unsafe_ptr() + 1
    def row(index: Int) {imm fp, imm lp, imm rp, imm count, imm reverse}:
        fp.unsafe_store(index, Int(lp[index] != rp[count-1-index if reverse else index]))
    host_parallelize(row, count)
    clear(failures, "Original Spectral same-graph label relationship")
    _ = left
    _ = right


def query(driver: SpectralDriver, input: Int, training: Int, state: SpectralStateView,
          shape: SpectralQueryShape, mut labels: List[Int32], mut embedding: List[Float32]) raises:
    try:
        require(driver.query(SpectralQueryRequest(input, training, state,
            Int(labels.unsafe_ptr()+1), Int(embedding.unsafe_ptr()+1), shape)) == 0,
            "Spectral prediction completed")
        labels_valid(labels)
        finite(embedding)
    finally:
        _ = labels
        _ = embedding


def original_single_rows[Device: Bool](driver: SpectralDriver, input: Int, training: Int,
        state: SpectralStateView, expected: List[Int32], n: Int, variant: Int) raises:
    var width = n if variant == 1 else 2
    var count = 19 if variant == 0 else (11 if variant == 1 else 4)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ep = expected.unsafe_ptr() + 1
    var entry = driver.entry
    def task(unit: Int) {imm fp, imm ep, imm entry, imm input, imm training, imm state, imm width, imm n, imm variant}:
        try:
            var index = unit * 7
            if variant == 0 and unit == 18:
                index = 5
            elif variant == 1:
                index = unit * 11
            elif variant == 2:
                index = 0 if unit == 0 else (120 if unit == 1 else (7 if unit == 2 else 127))
            var labels = List[Int32](length=3, fill=-31)
            var embedding = guarded(3)
            var local = SpectralDriver(entry)
            query(local, input + index * width * 4, training, state,
                SpectralQueryShape(n,1,width,3,3,10 if variant == 1 else 8,1 if variant == 1 else 0), labels, embedding)
            require(labels[1] == ep[index], "Original single row predicts the batch label")
            _ = labels^
            _ = embedding^
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL spectral original single query", variant, unit, String(error))
    comptime if Device:
        # One complete query graph at a time on the physical GPU; its kernels
        # remain parallel. This is the documented shared-context resource rule.
        for unit in range(count):
            task(unit)
    else:
        host_parallelize(task, count)
    clear(failures, "Original selected rows queried alone")
    _ = expected


def prefix(values: List[Float32], count: Int, reverse_rows: Bool = False) -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    var vp = values.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm vp, imm rp, imm count, imm reverse_rows}:
        var selected = ((count // 2 - 1 - index // 2) * 2 + index % 2) if reverse_rows else index
        rp.unsafe_store(index, vp[selected])
    host_parallelize(cell, count)
    _ = values
    return result^


def refused_query(driver: SpectralDriver, values: List[Float32], training: Int,
                  state: SpectralStateView, kind: Int) raises:
    var width = 2 if kind == 3 else 120
    var rows = 3 if kind == 3 else 2
    var input = prefix(values, rows * width)
    if kind == 4:
        input[3] = -0.5
    elif kind == 5:
        input[120] = bitcast[DType.float32](UInt32(0x7f800000))
    var labels = List[Int32](length=rows+2, fill=-31)
    var embedding = guarded(rows*3)
    var expected = "DEVIATION 2860" if kind == 3 else ("negative" if kind == 4 else "non-finite")
    try:
        var refused = False
        try:
            query(driver, Int(input.unsafe_ptr()), training, state,
                SpectralQueryShape(120,rows,width,3,3,8 if kind == 3 else 10,0 if kind == 3 else 1), labels, embedding)
        except error:
            require(String(error).find(expected) >= 0, "Original Spectral native refusal names its reason: " + String(error))
            refused = True
        require(refused, "Spectral invalid retained eigenvalue/affinity accepted")
        guards(embedding)
        var failures = List[Int](length=rows*3, fill=0)
        var fp = failures.unsafe_ptr()
        var ep = embedding.unsafe_ptr()+1
        def cell(index: Int) {imm fp, imm ep}:
            fp.unsafe_store(index, Int(bitcast[DType.uint32](ep[index]) != UInt32(0x7fc01234)))
        host_parallelize(cell, rows*3)
        clear(failures, "Spectral refusal does not publish embedding values")
        for label in labels:
            require(label == Int32(-31), "Spectral refusal does not publish labels")
    finally:
        _ = values
        _ = input^
        _ = labels^
        _ = embedding^


def fixture[Device: Bool](driver: SpectralDriver, kind: Int) raises:
    var precomputed = kind == 1 or kind >= 4
    var n = 240 if kind == 2 else 120
    var width = n if precomputed else 2
    var name = "predict120-affinity.f32" if precomputed else ("predict240-duplicated.f32" if kind == 2 else "predict120-input.f32")
    var values = source(name,n*width)
    var fitted_labels = List[Int32](length=n+2, fill=-31)
    var fitted_embedding = guarded(n*3)
    var plain_labels = List[Int32](length=n+2, fill=-31)
    var plain_embedding = guarded(n*3)
    var eigenvalues = guarded(3)
    var eigenvectors = guarded(n*3)
    var diagonal = guarded(n)
    var centroids = guarded(9)
    var predicted_labels = List[Int32](length=n+2, fill=-31)
    var predicted_embedding = guarded(n*3)
    var state = SpectralStateView(Int(eigenvalues.unsafe_ptr()+1), Int(eigenvectors.unsafe_ptr()+1),
        Int(diagonal.unsafe_ptr()+1), Int(centroids.unsafe_ptr()+1))
    var address = Int(values.unsafe_ptr())
    var neighbors = 10 if precomputed else 8
    try:
        var route = (2 if Device else 1) if precomputed else 0
        fit(driver,values,n,width,route,3,3,neighbors,3,True,state,fitted_labels,fitted_embedding)
        labels_valid(fitted_labels)
        finite(eigenvalues)
        finite(eigenvectors)
        finite(diagonal)
        finite(centroids)
        if kind == 0:
            for column in range(3):  # Fixed three retained eigenvalue checks from the original fixture.
                require(abs(Float32(1.0)+eigenvalues[column+1]) > Float32(1e-3),
                    "Original fitted affinity eigenvalues clear the Nyström threshold")
            var seen = List[Int](length=3, fill=0)
            var sp = seen.unsafe_ptr()
            var lp = fitted_labels.unsafe_ptr()+1
            def class_present(cls: Int) {imm sp, imm lp}:
                var count = 0
                for row in range(120):
                    count += Int(lp[row] == Int32(cls))
                sp.unsafe_store(cls, Int(count == 0))
            host_parallelize(class_present,3)
            clear(seen,"Original fixture fits three classes")
        if kind == 3:
            eigenvalues[2] = -1.0
        if kind >= 3:
            refused_query(driver,values,0 if precomputed else address,state,kind)
        else:
            if kind < 2:
                fit(driver,values,n,width,route,3,3,neighbors,3,False,SpectralStateView(0,0,0,0),plain_labels,plain_embedding)
                labels_equal(fitted_labels,plain_labels,False)
                require(same_columns(fitted_embedding,3,plain_embedding,3,n,0),
                    "Original prediction_data retention leaves fit labels and embedding unchanged")
            query(driver,address,0 if precomputed else address,state,
                SpectralQueryShape(n,n,width,3,3,neighbors,1 if precomputed else 0),predicted_labels,predicted_embedding)
            original_single_rows[Device](driver,address,0 if precomputed else address,state,predicted_labels,n,kind)
            if kind == 0:
                var reversed = prefix(values,n*2,True)
                var labels = List[Int32](length=n+2,fill=-31)
                var embedding = guarded(n*3)
                query(driver,Int(reversed.unsafe_ptr()),address,state,SpectralQueryShape(n,n,2,3,3,8,0),labels,embedding)
                labels_equal(predicted_labels,labels,True)
                _ = reversed^
                _ = labels^
                _ = embedding^
            elif kind == 2:
                var failures = List[Int](length=120,fill=0)
                var fp = failures.unsafe_ptr()
                var lp = predicted_labels.unsafe_ptr()+1
                def row(index: Int) {imm fp, imm lp}:
                    fp.unsafe_store(index,Int(lp[index] != lp[index+120]))
                host_parallelize(row,120)
                clear(failures,"Original duplicated training rows predict the same labels")
            if kind < 2:
                var query_name = "predict120-query-affinity.f32" if precomputed else "predict120-far.f32"
                var far = source(query_name,2*width)
                var labels = List[Int32](length=4,fill=-31)
                var embedding = guarded(6)
                if precomputed:
                    var failures = List[Int](length=n,fill=0)
                    var fp = failures.unsafe_ptr()
                    var qp = far.unsafe_ptr()
                    def zero_degree(index: Int) {imm fp, imm qp}:
                        fp.unsafe_store(index,Int(qp[index] != Float32(0)))
                    host_parallelize(zero_degree,n)
                    clear(failures,"Original far precomputed query has zero degree")
                query(driver,Int(far.unsafe_ptr()),0 if precomputed else address,state,
                    SpectralQueryShape(n,2,width,3,3,neighbors,1 if precomputed else 0),labels,embedding)
                source_unchanged(query_name,far)
                _ = far^
                _ = labels^
                _ = embedding^
        guards(eigenvalues)
        guards(eigenvectors)
        guards(diagonal)
        guards(centroids)
        source_unchanged(name,values)
    finally:
        _ = values^
        _ = fitted_labels^
        _ = fitted_embedding^
        _ = plain_labels^
        _ = plain_embedding^
        _ = eigenvalues^
        _ = eigenvectors^
        _ = diagonal^
        _ = centroids^
        _ = predicted_labels^
        _ = predicted_embedding^
