"""Bitwise gate and price for query-row parallel host RBC k-NN."""
from std.memory import bitcast
from std.time import perf_counter_ns

from core.knn_host_predict import (
    KNN_HOST_DIST_L2_SQRT_UNEXPANDED,
    host_rbc_knn_search,
)


def main() raises:
    comptime ni = 8000
    comptime nq = 128
    comptime d = 16
    comptime k = 8
    var x = List[Float32](length=ni * d, fill=Float32(0.0))
    var q = List[Float32](length=nq * d, fill=Float32(0.0))
    for r in range(ni):
        for f in range(d):
            x[r * d + f] = Float32(((r * 17 + f * 29) % 251) - 125) / Float32(
                64.0
            )
    for r in range(nq):
        for f in range(d):
            q[r * d + f] = Float32(
                ((r * 31 + f * 13 + 7) % 241) - 120
            ) / Float32(64.0)
    var si = List[Int32](length=nq * k, fill=Int32(-1))
    var sd = List[Float32](length=nq * k, fill=Float32(0.0))
    var pi = List[Int32](length=nq * k, fill=Int32(-1))
    var pd = List[Float32](length=nq * k, fill=Float32(0.0))

    var t0 = perf_counter_ns()
    host_rbc_knn_search(
        x,
        ni,
        q,
        nq,
        d,
        k,
        KNN_HOST_DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
        si,
        sd,
        1,
    )
    var serial_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    t0 = perf_counter_ns()
    host_rbc_knn_search(
        x,
        ni,
        q,
        nq,
        d,
        k,
        KNN_HOST_DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
        pi,
        pd,
    )
    var parallel_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var hash = UInt64(1469598103934665603)
    for cell in range(nq * k):
        if si[cell] != pi[cell] or bitcast[DType.uint32](sd[cell]) != bitcast[
            DType.uint32
        ](pd[cell]):
            raise Error(
                "host RBC parallel row split moved output at cell "
                + String(cell)
            )
        hash = (hash ^ UInt64(UInt32(pi[cell]))) * UInt64(1099511628211)
        hash = (hash ^ UInt64(bitcast[DType.uint32](pd[cell]))) * UInt64(
            1099511628211
        )
    print(
        "HOST_RBC_PARALLEL_OK",
        "cells",
        nq * k,
        "hash",
        hash,
        "serial_ms",
        serial_ms,
        "parallel_ms",
        parallel_ms,
    )
