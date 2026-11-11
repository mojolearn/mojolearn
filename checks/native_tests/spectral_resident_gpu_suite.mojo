# SPDX-License-Identifier: Apache-2.0
"""Ordered actual resident graphs and native last-owner lifetime checks.

One graph runs on the physical GPU at a time, including registry observations.
This is not a concurrent shared-DeviceContext submission test. Every numerical
kernel and original180/120-row workload is retained without a host substitute.
"""
from std.memory import ArcPointer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.spectral_native_api import SpectralDriver, SpectralStateView, SpectralQueryShape, SpectralPrepareRequest, SpectralResidentQueryRequest, SPECTRAL_RESIDENT_COUNT
from core.spectral_native_state import SpectralResidentState
from core.spectral_native_device import SpectralDeviceKernels
from core.spectral_native_provider import spectral_native_v1
from checks.native_tests.spectral_native_cases import source, source_unchanged, guarded, finite, fit
from checks.native_tests.spectral_prediction_cases import query, labels_equal, labels_valid
from checks.native_tests.spectral_dense_cases import require

comptime SPECTRAL_RESIDENT_GPU_FIXTURES = 2


def prepare_owned(driver: SpectralDriver, precomputed: Bool, mut expected_labels: List[Int32],
                  mut expected_embedding: List[Float32]) raises -> ArcPointer[SpectralResidentState]:
    var n = 120 if precomputed else 180
    var width = n if precomputed else 3
    var rows = 2 if precomputed else 36
    var name = "predict120-affinity.f32" if precomputed else "resident180-input.f32"
    var query_name = "predict120-query-affinity.f32" if precomputed else "resident180-query.f32"
    var values = source(name,n*width)
    var queries = source(query_name,rows*width)
    var labels = List[Int32](length=n+2,fill=-31)
    var embedding = guarded(n*3)
    var eigenvalues = guarded(3)
    var eigenvectors = guarded(n*3)
    var diagonal = guarded(n)
    var centroids = guarded(9)
    var state = SpectralStateView(Int(eigenvalues.unsafe_ptr()+1),Int(eigenvectors.unsafe_ptr()+1),
        Int(diagonal.unsafe_ptr()+1),Int(centroids.unsafe_ptr()+1))
    var address = Int(values.unsafe_ptr())
    try:
        fit(driver,values,n,width,2 if precomputed else 0,3,3,10,3 if precomputed else 1,
            True,state,labels,embedding)
        labels_valid(labels)
        finite(eigenvalues)
        finite(eigenvectors)
        finite(diagonal)
        finite(centroids)
        var shape = SpectralQueryShape(n,rows,width,3,3,10,1 if precomputed else 0)
        query(driver,Int(queries.unsafe_ptr()),0 if precomputed else address,state,shape,
            expected_labels,expected_embedding)
        source_unchanged(name,values)
        source_unchanged(query_name,queries)
        var handle = driver.prepare(SpectralPrepareRequest(0 if precomputed else address,state,shape))
        require(handle > 0,"Actual Spectral registry returns a positive handle")
        return ArcPointer[SpectralResidentState](SpectralResidentState(driver.copy(),handle))
    finally:
        # All caller-owned fit/state inputs die before the caller's resident
        # query. The actual registry must own its copies and device buffers.
        _ = values^
        _ = queries^
        _ = labels^
        _ = embedding^
        _ = eigenvalues^
        _ = eigenvectors^
        _ = diagonal^
        _ = centroids^
        _ = expected_labels
        _ = expected_embedding


def fixture(driver: SpectralDriver, precomputed: Bool) raises:
    var n = 120 if precomputed else 180
    var width = n if precomputed else 3
    var rows = 2 if precomputed else 36
    var name = "predict120-query-affinity.f32" if precomputed else "resident180-query.f32"
    var queries = source(name,rows*width)
    var expected_labels = List[Int32](length=rows+2,fill=-31)
    var expected_embedding = guarded(rows*3)
    var labels = List[Int32](length=rows+2,fill=-31)
    var embedding = guarded(rows*3)
    var again_labels = List[Int32](length=rows+2,fill=-31)
    var again_embedding = guarded(rows*3)
    var before = driver.metadata(SPECTRAL_RESIDENT_COUNT)
    try:
        var cache = prepare_owned(driver,precomputed,expected_labels,expected_embedding)
        var handle = cache[].handle
        require(driver.metadata(SPECTRAL_RESIDENT_COUNT) == before+1,"One native resident handle prepared")
        var active = cache.copy()
        _ = cache^
        require(driver.metadata(SPECTRAL_RESIDENT_COUNT) == before+1,"Cache retirement keeps active native query alive")
        var shape = SpectralQueryShape(n,rows,width,3,3,10,1 if precomputed else 0)
        require(driver.query_resident(SpectralResidentQueryRequest(handle,Int(queries.unsafe_ptr()),
            Int(labels.unsafe_ptr()+1),Int(embedding.unsafe_ptr()+1),shape)) == 0,"Actual resident query completed")
        labels_valid(labels)
        finite(embedding)
        labels_equal(expected_labels,labels,False)
        require(driver.query_resident(SpectralResidentQueryRequest(handle,Int(queries.unsafe_ptr()),
            Int(again_labels.unsafe_ptr()+1),Int(again_embedding.unsafe_ptr()+1),shape)) == 0,"Original repeated resident query completed")
        labels_valid(again_labels)
        finite(again_embedding)
        labels_equal(labels,again_labels,False)
        require(driver.metadata(SPECTRAL_RESIDENT_COUNT) == before+1,"Repeated query reuses the existing native handle")
        _ = active^
        require(driver.metadata(SPECTRAL_RESIDENT_COUNT) == before,"Last native owner releases registry handle")
        driver.release(handle)
        require(driver.metadata(SPECTRAL_RESIDENT_COUNT) == before,"Existing released-handle cleanup remains quiet")
        var refused = False
        try:
            _ = driver.query_resident(SpectralResidentQueryRequest(handle,Int(queries.unsafe_ptr()),
                Int(labels.unsafe_ptr()+1),Int(embedding.unsafe_ptr()+1),shape))
        except error:
            require(String(error).find("unknown or released resident spectral prediction handle") >= 0,
                "Released Spectral handle refuses by name")
            refused = True
        require(refused,"Released Spectral resident handle reused")
        source_unchanged(name,queries)
    finally:
        _ = queries^
        _ = expected_labels^
        _ = expected_embedding^
        _ = labels^
        _ = embedding^
        _ = again_labels^
        _ = again_embedding^


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var driver = SpectralDriver(spectral_native_v1[SpectralDeviceKernels])
    var failed = 0
    for unit in range(SPECTRAL_RESIDENT_GPU_FIXTURES):
        try:
            fixture(driver,unit != 0)
        except error:
            failed += 1
            print("FAIL spectral_resident_gpu fixture",unit,String(error))
    require(failed == 0,"spectral_resident_gpu failed " + String(failed) + " fixtures")
    print("PASS spectral_resident_gpu:",SPECTRAL_RESIDENT_GPU_FIXTURES,"actual native resident ownership graphs")
