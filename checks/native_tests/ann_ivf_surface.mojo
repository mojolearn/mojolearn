# SPDX-License-Identifier: Apache-2.0
"""IVF-Flat sample-filter short-fill on the device and in the host column.

Ported from the filter case of `arm_short_fill` in
`python/mojolearn/tests/test_ivf_surface.py`: 1,024 eight-feature rows, 32
queries, 8 lists all probed, n_neighbors 5, and a filter keeping rows 3, 500
and 1000. Every query counts 3 candidates, its slots 0..2 are exactly the
kept rows, and slots 3..4 are cuVS's kOutOfBoundsRecord (+inf, id -1). The
host search (`ivf/host/ivf_host.mojo::host_ivf_search`) over the same built
index gives the same claims. The rows are standard normal (Box-Muller over a
SplitMix64 stream regenerated here; the claims hold for any data).

The other numerical arms of that file are already Mojo gates in
`ivf/checks/ivf_check.mojo` (`pixi run check-ivf`): probing every list is
brute force, L2SqrtExpanded is the root of L2Expanded, the planted
short-fill, the filter against the oracle, and two builds and searches
writing the same card.

One fixture on the one physical device; the build and search keep their own
kernel parallelism.
"""
from std.memory import bitcast
from std.math import cos, log, sqrt
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from ivf.host.ivf_host import IvfHostIndex, host_ivf_search
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import ivf_flat_build
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import IvfFlatIndexParams, IvfFlatSearchParams
from ivf.impl.neighbors.ivf_flat.ivf_flat_search import IVF_EXPAND_TPB, ivf_flat_search_traced
from neighbors.checks.pinned_distance_tile import PINNED_TILE_TPB
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was ann_ivf_surface_gpu_suite.mojo) ----
def _next(mut state: UInt64) -> UInt64:
    """SplitMix64."""
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def normal_rows(seed: Int, count: Int) -> List[Float32]:
    var state = UInt64(seed) * UInt64(0x2545F4914F6CDD1D) + UInt64(1)
    var values = List[Float32](length=count, fill=0)
    for i in range(count):
        var u1 = (Float64(Int(_next(state) >> 11)) + 1.0) * (1.0 / 9007199254740992.0)
        var u2 = Float64(Int(_next(state) >> 11)) * (1.0 / 9007199254740992.0)
        values[i] = Float32(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2))
    return values^


def short_filled(distances: List[Float32], indices: List[UInt32], n_candidates: List[Int32],
                 queries: Int, k: Int, column: String) raises:
    var inf_bits = UInt32(0x7F800000)
    var wrong = 0
    for qi in range(queries):
        if n_candidates[qi] != 3:
            wrong += 1
        var seen = 0
        for s in range(3):
            var id = indices[qi * k + s]
            seen |= (1 if id == 3 else 0) | (2 if id == 500 else 0) | (4 if id == 1000 else 0)
        if seen != 7:
            wrong += 1
        for s in range(3, k):
            if indices[qi * k + s] != UInt32(0xFFFFFFFF) or bitcast[DType.uint32](distances[qi * k + s]) != inf_bits:
                wrong += 1
    require(wrong == 0, column + ": " + String(wrong)
        + " filter short-fill claims fail (3 candidates, the kept rows first, then (+inf, -1))")


def _run_gpu(mut report: Report) raises:
    comptime N = 1024
    comptime D = 8
    comptime Q = 32
    comptime K = 5
    comptime LISTS = 8
    var x = normal_rows(0, N * D)
    var q = normal_rows(1, Q * D)
    var keep = List[Int32](length=N, fill=0)
    keep[3] = 1
    keep[500] = 1
    keep[1000] = 1
    var ctx = DeviceContext()
    var trace = IdentityTrace.disabled()
    var params = IvfFlatIndexParams.default()
    params.n_lists = LISTS
    params.seed = UInt64(1)
    var index = ivf_flat_build(ctx, trace, params, x, N, D)
    var sp = IvfFlatSearchParams(LISTS)
    var got = ivf_flat_search_traced(ctx, trace, index, sp, q, Q, K, PINNED_TILE_TPB, IVF_EXPAND_TPB, False, keep)
    short_filled(got.distances, got.indices, got.n_candidates, Q, K, "device")
    var host = host_ivf_search(
        IvfHostIndex(
            LISTS, D, N, index.metric, index.centers.copy(), index.center_norms.copy(),
            index.list_offsets.copy(), index.list_indices.copy(), index.list_data.copy(),
        ),
        q, Q, K, LISTS, False, keep,
    )
    short_filled(host.distances, host.indices, host.n_candidates, Q, K, "host")
    _ = x^
    _ = q^
    _ = keep^
    _ = index^
    _ = got^
    _ = host^
    print("PASS ann_ivf_surface_gpu: sample-filter short-fill on the device and the host column")


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("ann_ivf_surface", cpu=False, gpu=True, serial="", run=run_cases)
