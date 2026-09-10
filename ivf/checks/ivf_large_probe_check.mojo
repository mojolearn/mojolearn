# SPDX-License-Identifier: Apache-2.0
"""Exercise the extended coarse selector with257 tied lists and one query."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from ivf.checks.ivf_check import _plant_index, _search


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("large probe gate requires IDENTICAL")
    comptime N = 1057
    comptime LISTS = 257
    var x = List[Float32]()
    var labels = List[UInt32]()
    var centers = List[Float32]()
    var queries: List[Float32] = [0, 0]
    for row in range(N):
        x.append(Float32(0))
        x.append(Float32(0))
        labels.append(UInt32(row % LISTS))
    for _ in range(LISTS * 2):
        centers.append(Float32(0))
    with DeviceContext() as ctx:
        var index = _plant_index(ctx, x, labels, centers, N, 2, LISTS)
        var result = _search(ctx, index, queries, 1, 257, LISTS)
        if result.n_candidates[0] != N:
            raise Error("coarse257 selection lost a probed list")
        for rank in range(257):
            if result.indices[rank] != UInt32(rank) or bitcast[DType.uint32](result.distances[rank]) != UInt32(0):
                raise Error("coarse257/final257 tied selection lost original-index order")
    print("IVF LARGE PROBE PASS", "probes", LISTS, "selected_cells", 257)
