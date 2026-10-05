"""FAST-only batched IVF-Flat list scan (Apple).

The pinned search serves one query at a time from the host: gather the
probed lists' vectors into a host buffer, upload them, run the distance
tile and the selector, download, and move to the next query -- a host
round trip per query (20,000 queries over 100k rows: 22 s on the M4).

`fast_ivf_scan_kernel` does steps 3-5 for every query in one launch: one
SIMD group per query walks its probed lists straight from the device copy
of `list_data`, each lane keeps its own ascending top-`KM` of
(squared distance, original id) in registers, and the group merges the 32
lane lists with `k` rounds of a shuffle argmin. (distance, id) is a total
order, so the result does not depend on the probe order or on which lane
saw a candidate. Distances are the direct sum of squared differences
(FAST arithmetic; the root, when the metric wants one, is taken on the
host after the order is fixed, as in the pinned path).
"""

from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx, lane_id
from std.gpu.primitives.warp import shuffle_xor, shuffle_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined

comptime FIVF_QPB = 4
"""Queries (SIMD groups) per block."""
comptime FIVF_DIM_TILE = 256
"""The query tile of the default instantiation: FIVF_QPB * 256 floats = 4 KB
of threadgroup memory, so small dims keep their occupancy."""
comptime FIVF_MAX_DIM = FIVF_DIM_TILE if is_defined[
    "MOJOLEARN_LEGACY_NARROW_FIVF_DIM"
]() else (32 * 1024) // (FIVF_QPB * 4)
"""Widest query the batched FAST scan takes: the query tiles of FIVF_QPB
SIMD groups filling 32 KB of threadgroup memory (Apple's limit, under
NVIDIA's and AMD's), 2048 floats. Was 256 (admitted istella's 220);
removed as benchmark-tuned on 2026-10-04, replacement UNMEASURED. A dim
above FIVF_DIM_TILE launches the 32 KB instantiation. `-D
MOJOLEARN_LEGACY_NARROW_FIVF_DIM` restores the 256 cap."""


@always_inline
def _less(a: Float32, ai: UInt32, b: Float32, bi: UInt32) -> Bool:
    return a < b or (a == b and ai < bi)


def fast_ivf_scan_kernel[KM: Int, DM: Int = FIVF_DIM_TILE](
    queries: MutPointer[Float32, MutAnyOrigin],
    list_data: MutPointer[Float32, MutAnyOrigin],
    list_offsets: MutPointer[Int32, MutAnyOrigin],
    list_indices: MutPointer[UInt32, MutAnyOrigin],
    probe_idx: MutPointer[UInt32, MutAnyOrigin],
    out_dist: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
    keep: MutPointer[Int32, MutAnyOrigin],
    keep_len: Int32,
    n_queries: Int32,
    dim_in: Int32,
    n_probes_in: Int32,
    k_in: Int32,
):
    """`keep` (lane ivf-filter-fix, 2026-10-01): the sample filter, one
    int32 per original row, 0 removing the row; `keep_len` 0 is no filter.
    A removed candidate is never scored, selected or counted, as
    `filter_candidate_slots` drops it before the per-query path scores it,
    so a filtered search is this same kernel over the kept set and an
    all-ones filter is the unfiltered search bit for bit."""
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var q = Int(block_idx.x) * FIVF_QPB + warp
    var dim = Int(dim_in)
    var s_q = stack_allocation[
        FIVF_QPB * DM, Float32, address_space=AddressSpace.SHARED
    ]()
    var qv = s_q + warp * DM
    var active = q < Int(n_queries)
    if active:
        var j = lane
        while j < dim:
            qv[j] = queries[q * dim + j]
            j += WARP_SIZE
    barrier()
    if not active:
        return
    var td = InlineArray[Float32, KM](fill=Float32.MAX)
    var ti = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var n_probes = Int(n_probes_in)
    for p in range(n_probes):
        var l = Int(probe_idx[q * n_probes + p])
        var s = Int(list_offsets[l])
        var e = Int(list_offsets[l + 1])
        var pos = s + lane
        while pos < e:
            if keep_len != 0 and keep[Int(list_indices[pos])] == 0:
                pos += WARP_SIZE
                continue
            var row = list_data + pos * dim
            var d = Float32(0)
            for j in range(dim):
                var t = row[j] - qv[j]
                d += t * t
            var id = list_indices[pos]
            if _less(d, id, td[KM - 1], ti[KM - 1]):
                var cd = d
                var ci = id
                comptime for j in range(KM):
                    if _less(cd, ci, td[j], ti[j]):
                        var sd = td[j]
                        var si = ti[j]
                        td[j] = cd
                        ti[j] = ci
                        cd = sd
                        ci = si
            pos += WARP_SIZE
    var k = Int(k_in)
    for r in range(k):
        var bd = td[0]
        var bi = ti[0]
        # Every lane of the warp, 64 on AMD (lane/neural-net-experiment).
        comptime for sh in [32, 16, 8, 4, 2, 1]:
            comptime if sh < WARP_SIZE:
                var od = shuffle_xor(bd, UInt32(sh))
                var oi = shuffle_xor(bi, UInt32(sh))
                if _less(od, oi, bd, bi):
                    bd = od
                    bi = oi
        if td[0] == bd and ti[0] == bi:
            comptime for j in range(KM - 1):
                td[j] = td[j + 1]
                ti[j] = ti[j + 1]
            td[KM - 1] = Float32.MAX
            ti[KM - 1] = UInt32.MAX
        if lane == 0:
            out_dist[q * k + r] = bd
            out_idx[q * k + r] = bi
