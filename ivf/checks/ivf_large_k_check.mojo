# SPDX-License-Identifier: Apache-2.0
"""Public IVF search beyond256 against an independent integer-distance order."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from ivf.checks.ivf_check import _plant_index, _search
from ivf.checks.ivf_large_probe_check import check_large_probes


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("large IVF k gate requires IDENTICAL")
    comptime N = 1057
    var x = List[Float32]()
    var labels = List[UInt32]()
    var centers = List[Float32]()
    var queries: List[Float32] = [0, 0, 3, 4]
    for row in range(N):
        x.append(Float32(row % 17))
        x.append(Float32((row // 17) % 7))
        labels.append(UInt32((row * 7) % 4))
    for _ in range(8):
        centers.append(Float32(0))
    with DeviceContext() as ctx:
        var index = _plant_index(ctx, x, labels, centers, N, 2, 4)
        var sizes: List[Int] = [257, 1024]
        for k in sizes:
            var result = _search(ctx, index, queries, 2, k, 4)
            for qi in range(2):
                var expected = List[Int]()
                var distances = List[Int]()
                for row in range(N):
                    var dx = row % 17 - (0 if qi == 0 else 3)
                    var dy = (row // 17) % 7 - (0 if qi == 0 else 4)
                    var distance = dx * dx + dy * dy
                    expected.append(row)
                    distances.append(distance)
                    var pos = row
                    while pos > 0 and distances[expected[pos - 1]] > distance:
                        expected[pos] = expected[pos - 1]
                        pos -= 1
                    expected[pos] = row
                for rank in range(k):
                    var cell = qi * k + rank
                    var row = expected[rank]
                    if result.indices[cell] != UInt32(row) or bitcast[DType.uint32](result.distances[cell]) != bitcast[DType.uint32](Float32(distances[row])):
                        print("IVF_LARGE_K_FAIL", k, qi, rank, result.indices[cell], row)
                        raise Error("IVF large k changed exact distance or tie order")
            print("IVF_LARGE_K_PASS", k, "cells", 2 * k)
        var refused = False
        try:
            _ = _search(ctx, index, queries, 2, 1025, 4)
        except e:
            if "bounded rank capacity" not in String(e):
                raise Error("IVF1025 reached the wrong refusal: " + String(e))
            refused = True
        if not refused:
            raise Error("IVF1025 exceeded bounded rank profile")
    print("IVF LARGE K PASS", "selected_cells", 2562, "refusal", 1025)
    check_large_probes()
