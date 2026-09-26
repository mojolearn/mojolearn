"""IDENTICAL batched IVF-Flat list scan (Apple; lane/apple-identical-neural,
2026-09-26).

The pinned search serves one query at a time from the host (gather the
probed lists into a candidate row, upload, `pinned_distance_tile_kernel`,
the identical radix selector, download): at 1M rows that is minutes. This
kernel does the same scoring and selection for every query in one launch,
one SIMD group per query, and returns the same k (distance, id) pairs in
the same order:

  * each candidate's squared distance is `pinned_distance_tile_kernel`'s
    arithmetic term for term: the dot `acc = ftz(identical_mul_add(ftz(q_f),
    ftz(y_f), acc))` over f ascending from +0.0, then
    `ftz(identical_mul_add(-2, acc, ftz(ftz(q_norm) + ftz(y_norm))))`,
    clamped to +0.0 when <= 0 (the norms are the ones the pinned path
    computes, `compute_row_norms`);
  * the pinned selector keys on `(twiddle_in(distance) << 32) | position in
    the candidate row`, and that row is ordered by original index
    (DEVIATION 1786), so the key is `(distance bits, original index)`; this
    kernel keeps each lane's ascending top-KM by that same key and merges
    the 32 lane lists, so the selected set and its order do not depend on
    which lane saw a candidate or in what probe order.
"""

from std.gpu import WARP_SIZE, block_idx, thread_idx, lane_id
from std.gpu.primitives.warp import shuffle_xor
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_mul_add

comptime IIVF_QPB = 4
comptime IIVF_MAX_DIM = 256


@always_inline
def _key(d: Float32) -> UInt32:
    """`twiddle_in(d, select_min=True)`: unsigned order == float order."""
    var bits = bitcast[DType.uint32](d)
    if (bits & UInt32(0x80000000)) != 0:
        return bits ^ UInt32(0xFFFFFFFF)
    return bits ^ UInt32(0x80000000)


@always_inline
def _kless(a: UInt32, ai: UInt32, b: UInt32, bi: UInt32) -> Bool:
    return a < b or (a == b and ai < bi)


def identical_ivf_scan_kernel[KM: Int](
    queries: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    list_data: MutPointer[Float32, MutAnyOrigin],
    list_norm: MutPointer[Float32, MutAnyOrigin],
    list_offsets: MutPointer[Int32, MutAnyOrigin],
    list_indices: MutPointer[UInt32, MutAnyOrigin],
    probe_idx: MutPointer[UInt32, MutAnyOrigin],
    out_dist: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
    n_queries: Int32,
    dim_in: Int32,
    n_probes_in: Int32,
    k_in: Int32,
):
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var q = Int(block_idx.x) * IIVF_QPB + warp
    var dim = Int(dim_in)
    var s_q = stack_allocation[IIVF_QPB * IIVF_MAX_DIM, Float32, address_space=AddressSpace.SHARED]()
    var qv = s_q + warp * IIVF_MAX_DIM
    var active = q < Int(n_queries)
    if active:
        var j = lane
        while j < dim:
            qv[j] = ftz(queries[q * dim + j])
            j += WARP_SIZE
    barrier()
    if not active:
        return
    var qn = ftz(q_norm[q])
    var tk = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var td = InlineArray[Float32, KM](fill=Float32(0))
    var ti = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var n_probes = Int(n_probes_in)
    for p in range(n_probes):
        var l = Int(probe_idx[q * n_probes + p])
        var s = Int(list_offsets[l])
        var e = Int(list_offsets[l + 1])
        var pos = s + lane
        while pos < e:
            var row = list_data + pos * dim
            var acc = Float32(0.0)
            for f in range(dim):
                acc = ftz(identical_mul_add(qv[f], ftz(row[f]), acc))
            var d = ftz(identical_mul_add(Float32(-2.0), acc, ftz(qn + ftz(list_norm[pos]))))
            if d <= Float32(0.0):
                d = Float32(0.0)
            var key = _key(d)
            var id = list_indices[pos]
            if _kless(key, id, tk[KM - 1], ti[KM - 1]):
                var ck = key
                var cd = d
                var ci = id
                comptime for j in range(KM):
                    if _kless(ck, ci, tk[j], ti[j]):
                        var sk = tk[j]
                        var sd = td[j]
                        var si = ti[j]
                        tk[j] = ck
                        td[j] = cd
                        ti[j] = ci
                        ck = sk
                        cd = sd
                        ci = si
            pos += WARP_SIZE
    var k = Int(k_in)
    for r in range(k):
        var bk = tk[0]
        var bd = td[0]
        var bi = ti[0]
        comptime for sh in [16, 8, 4, 2, 1]:
            var ok = shuffle_xor(bk, UInt32(sh))
            var od = shuffle_xor(bd, UInt32(sh))
            var oi = shuffle_xor(bi, UInt32(sh))
            if _kless(ok, oi, bk, bi):
                bk = ok
                bd = od
                bi = oi
        if tk[0] == bk and ti[0] == bi:
            comptime for j in range(KM - 1):
                tk[j] = tk[j + 1]
                td[j] = td[j + 1]
                ti[j] = ti[j + 1]
            tk[KM - 1] = UInt32.MAX
            ti[KM - 1] = UInt32.MAX
        if lane == 0:
            out_dist[q * k + r] = bd
            out_idx[q * k + r] = bi
