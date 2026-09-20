"""Exactness and scale gate for high-cardinality host k-NN labels."""
from std.time import perf_counter_ns

from core.knn_host_predict import host_monotonic


def main() raises:
    comptime n = 262144
    comptime n_classes = 256
    var uniq = List[Int32](capacity=n_classes)
    for i in range(n_classes):
        uniq.append(Int32(i * 3 - 200))
    var y = List[Int32](capacity=n)
    for i in range(n):
        y.append(uniq[(i * 131 + 17) % n_classes])

    var t0 = perf_counter_ns()
    var got = host_monotonic(y, n, uniq)
    var elapsed_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var hash = UInt64(1469598103934665603)
    for row in range(n):
        var want = Int32(-1)
        # The device path's ascending linear scan is the semantic reference.
        for c in range(n_classes):
            if y[row] == uniq[c]:
                want = Int32(c)
                break
        if got[row] != want:
            raise Error(
                "host label encoding mismatch at row " + String(row)
            )
        hash = (hash ^ UInt64(UInt32(got[row]))) * UInt64(1099511628211)
    print(
        "HOST_LABEL_ENCODING_OK",
        "rows",
        n,
        "classes",
        n_classes,
        "hash",
        hash,
        "candidate_ms",
        elapsed_ms,
    )
