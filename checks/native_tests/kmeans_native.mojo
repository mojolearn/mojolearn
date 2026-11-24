# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Native KMeans fixtures: independent small geometric oracles through the actual KMeans provider. The host
column runs its owners across host workers; the device column exercises one shared device context in order."""
from std.memory import bitcast
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from core.kmeans_native_api import (
    KMeansNativeDriver, KMeansFoldStore, KMeansFitRequest, KMeansQueryRequest,
)
from core.kmeans_native_host import kmeans_host_native_v1
from core.kmeans_native_device import kmeans_device_native_v1


# The column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).


# ---- The cases (was kmeans_kernel_cases.mojo) ----
def _near(value: Float64, expected: Float64) raises:
    var delta = value - expected
    require(delta > -1e-5 and delta < 1e-5, "geometric KMeans oracle mismatch")


def fixture(driver: KMeansNativeDriver, unit: Int) raises:
    var centers: List[Float32] = [0.0, 0.0, 10.0, 0.0]
    if unit == 0 or unit == 4 or unit == 8:
        var source: List[Float32] = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 9.0, 0.0, 10.0, 0.0, 11.0, 0.0]
        var labels = List[UInt32](length=6, fill=UInt32(99))
        var distances = List[Float32](length=12, fill=Float32(-99.0))
        if unit == 4:
            source[0] = bitcast[DType.float32](UInt32(0x7FC00000))
        var result = driver.fit(KMeansFitRequest(Int(source.unsafe_ptr()), Int(centers.unsafe_ptr()),
            Int(labels.unsafe_ptr()), Int(source.unsafe_ptr()),
            Int(distances.unsafe_ptr()) if unit == 8 else 0, 6, 2, 2, 0, 10, 7, 1, 2, 0,
            True, 1e-4, 0.0, KMeansFoldStore()))
        if unit == 4:
            require(result.finite != 0, "fit must report nonfinite input before training")
            for label in labels:
                require(label == UInt32(99), "nonfinite fit must not publish labels")
        else:
            require(result.finite == 0 and result.range_error == 0 and result.iterations > 0, "fit status")
            _near(result.inertia, 4.0)
            for i in range(6):
                require(labels[i] == UInt32(0 if i < 3 else 1), "two separated clusters")
            if unit == 8:
                var expected: List[Float32] = [1.0, 121.0, 0.0, 100.0, 1.0, 81.0,
                    81.0, 1.0, 100.0, 0.0, 121.0, 1.0]
                for i in range(12):
                    _near(Float64(distances[i]), Float64(expected[i]))
        _near(Float64(centers[0]), 0.0)
        _near(Float64(centers[1]), 0.0)
        _near(Float64(centers[2]), 10.0)
        _near(Float64(centers[3]), 0.0)
        _ = source
        _ = labels
        _ = centers
        _ = distances
        return
    var source: List[Float32] = [-1.0, 0.0, 5.0, 0.0, 11.0, 0.0]
    var labels = List[UInt32](length=5, fill=UInt32(99))
    var distances = List[Float32](length=8, fill=Float32(-99.0))
    var transform = unit == 2 or unit == 3 or unit == 7
    var metric = 1 if unit == 3 else 0
    if unit == 5:
        source[0] = bitcast[DType.float32](UInt32(0x7FC00000))
    var handle = -1
    try:
        if unit >= 6:
            handle = driver.prepare(Int(centers.unsafe_ptr()), 2, 2)
            # The resident operation must use its own cached device copy.
            # Changing the original host words cannot trigger a new upload.
            centers[0] = 99.0
        var destination = (Int(distances.unsafe_ptr()) if transform else Int(labels.unsafe_ptr())) + 4
        var result = driver.query(KMeansQueryRequest(Int(source.unsafe_ptr()),
            0 if handle >= 0 else Int(centers.unsafe_ptr()), destination,
            3, 2, 2, metric, True, transform, handle, KMeansFoldStore()))
        if unit == 5:
            require(result.finite != 0, "query must report nonfinite input")
            require(result.range_error == 0, "ordinary query has no fold range error")
            # The existing GPU path reads labels and finite status together.
            # Private scratch may be written; the public boundary rejects the
            # status before returning it. Do not add a scan/sync to prevent it.
        else:
            require(result.finite == 0 and result.range_error == 0, "query status")
            if transform:
                var expected: List[Float32] = [1.0, 121.0, 25.0, 25.0, 121.0, 1.0]
                if metric == 1:
                    expected = [1.0, 11.0, 5.0, 5.0, 11.0, 1.0]
                for i in range(6):
                    _near(Float64(distances[i + 1]), Float64(expected[i]))
            else:
                require(labels[1] == UInt32(0) and labels[2] == UInt32(0) and labels[3] == UInt32(1),
                    "nearest center and first-center tie")
        require(labels[0] == UInt32(99) and labels[4] == UInt32(99), "query label span guards")
        require(distances[0] == -99.0 and distances[7] == -99.0, "query distance span guards")
    finally:
        if handle >= 0:
            driver.release(handle)
        _ = source
        _ = centers
        _ = labels
        _ = distances


# ---- The host column (was kmeans_native_cpu_suite.mojo) ----
comptime KMEANS_CPU_FIXTURES = 6


def _host_fixture(unit: Int) raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN == COLUMN_CPU
        fixture(KMeansNativeDriver(kmeans_host_native_v1), unit)


# ---- The device column (was kmeans_native_gpu_suite.mojo) ----
# Native GPU KMeans fixtures; one shared device context is exercised in order.
comptime KMEANS_GPU_FIXTURES = 9


def _device_fixtures_in_order() raises:
    comptime if not CPU_COLUMN_BUILD:
        comptime assert TARGET_COLUMN != COLUMN_CPU
        var failed = 0
        for unit in range(KMEANS_GPU_FIXTURES):  # one shared device context, in order
            try:
                fixture(KMeansNativeDriver(kmeans_device_native_v1), unit)
            except error:
                failed += 1
                print("FAIL kmeans_native gpu fixture", unit, String(error))
        if failed:
            raise Error("native KMeans GPU failed " + String(failed) + " of " + String(KMEANS_GPU_FIXTURES) + " fixtures")


# ---- The fit loop (`KMeans.fit_loop_`; lane ab-loop-mode) ----
comptime FIT_LOOP_CAP = 3
comptime FIT_LOOP_RESTARTS = 2


def fit_loop_case(driver: KMeansNativeDriver) raises:
    """The fit loop counts the executed Lloyd steps, summed over the restarts.

    `tol` must be positive (`KMeansParams.validate`, cuVS `RAFT_EXPECTS` at
    `detail/kmeans.cuh:825-835`: "invalid parameter (tol<=0)"), so the two
    stopping rules are pinned with valid tolerances instead of tol 0:

    1. Exhaustion: one explicit start (INIT_ARRAY, which runs one restart) on
       the points x = 0..9 from centers (0, 1). The first three Lloyd steps
       move the centers to (0, 5), (1, 6), (1.5, 6.5): squared shifts 16, 2
       and 0.5, all far above tol 1e-4, and no point sits on a midpoint.
       So the fit runs FIT_LOOP_CAP steps, `n_iter` is max_iter + 1 (cuVS),
       and the loop reports FIT_LOOP_CAP steps.
    2. Restarts: random seeding (init 1) with FIT_LOOP_RESTARTS restarts and
       tol 1e30, above any squared shift of these rows. KMeans fits never
       take the lazy shift read, so every restart stops at its first step:
       `n_iter` is 1 and the loop reports one step per restart."""
    var line = List[Float32](length=20, fill=Float32(0.0))
    for i in range(10):
        line[2 * i] = Float32(i)
    var start: List[Float32] = [0.0, 0.0, 1.0, 0.0]
    var line_labels = List[UInt32](length=10, fill=UInt32(99))
    var capped = driver.fit(KMeansFitRequest(Int(line.unsafe_ptr()), Int(start.unsafe_ptr()),
        Int(line_labels.unsafe_ptr()), Int(line.unsafe_ptr()), 0, 10, 2, 2, 0, FIT_LOOP_CAP, 7,
        1, 2, 0, True, 1e-4, 0.0, KMeansFoldStore()))
    _ = line
    _ = start
    _ = line_labels
    require(capped.finite == 0 and capped.range_error == 0, "capped fit status")
    require(capped.loop.recorded, "a fit records its loop")
    require(capped.iterations == FIT_LOOP_CAP + 1, "n_iter on exhaustion is max_iter + 1 (cuVS)")
    require(capped.loop.iterations == FIT_LOOP_CAP, "an exhausted fit reports max_iter Lloyd steps")
    require(capped.loop.loop_ns >= 0 and capped.loop.milliseconds() >= 0.0, "fit loop time is not negative")

    var source: List[Float32] = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 9.0, 0.0, 10.0, 0.0, 11.0, 0.0]
    var centers = List[Float32](length=4, fill=Float32(0.0))
    var labels = List[UInt32](length=6, fill=UInt32(99))
    var result = driver.fit(KMeansFitRequest(Int(source.unsafe_ptr()), Int(centers.unsafe_ptr()),
        Int(labels.unsafe_ptr()), Int(source.unsafe_ptr()), 0, 6, 2, 2, 0, FIT_LOOP_CAP, 7,
        FIT_LOOP_RESTARTS, 1, 0, True, 1e30, 0.0, KMeansFoldStore()))
    _ = source
    _ = centers
    _ = labels
    require(result.finite == 0 and result.range_error == 0, "fit loop fit status")
    require(result.loop.recorded, "a fit records its loop")
    require(result.iterations == 1, "n_iter is the kept restart's converged step")
    require(result.loop.iterations == FIT_LOOP_RESTARTS,
        "fit loop iterations are the executed Lloyd steps summed over the restarts")
    require(result.loop.loop_ns >= 0 and result.loop.milliseconds() >= 0.0, "fit loop time is not negative")


def test_fit_loop() raises:
    comptime if CPU_COLUMN_BUILD:
        fit_loop_case(KMeansNativeDriver(kmeans_host_native_v1))
    else:
        fit_loop_case(KMeansNativeDriver(kmeans_device_native_v1))


def test_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("fixture", KMEANS_CPU_FIXTURES, _host_fixture)
    else:
        _device_fixtures_in_order()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
