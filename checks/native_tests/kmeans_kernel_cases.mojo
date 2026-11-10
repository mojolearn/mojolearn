# SPDX-License-Identifier: Apache-2.0
"""Independent small geometric oracles through the actual KMeans provider."""
from std.memory import bitcast
from core.kmeans_native_api import (
    KMeansNativeDriver, KMeansFoldStore, KMeansFitRequest, KMeansQueryRequest,
)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
    var labels = List[UInt32](length=3, fill=UInt32(99))
    var distances = List[Float32](length=6, fill=Float32(-99.0))
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
        var destination = Int(distances.unsafe_ptr()) if transform else Int(labels.unsafe_ptr())
        var result = driver.query(KMeansQueryRequest(Int(source.unsafe_ptr()),
            0 if handle >= 0 else Int(centers.unsafe_ptr()), destination,
            3, 2, 2, metric, True, transform, handle, KMeansFoldStore()))
        if unit == 5:
            require(result.finite != 0, "query must report nonfinite input")
            for label in labels:
                require(label == UInt32(99), "nonfinite query must not publish labels")
        else:
            require(result.finite == 0 and result.range_error == 0, "query status")
            if transform:
                var expected: List[Float32] = [1.0, 121.0, 25.0, 25.0, 121.0, 1.0]
                if metric == 1:
                    expected = [1.0, 11.0, 5.0, 5.0, 11.0, 1.0]
                for i in range(6):
                    _near(Float64(distances[i]), Float64(expected[i]))
            else:
                require(labels[0] == UInt32(0) and labels[1] == UInt32(0) and labels[2] == UInt32(1),
                    "nearest center and first-center tie")
    finally:
        if handle >= 0:
            driver.release(handle)
        _ = source
        _ = centers
        _ = labels
        _ = distances
