# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MiniBatchKMeans, FAST on Apple: X uploaded straight from the caller's
array (lane/apple-fast-mbkspeed, 2026-10-03). Switch: `MBK_ZEROCOPY`
below, DEFAULT on FAST+Apple since the M3 A/B (lane/apple-fast-mbkspeed
d844f180a, n=1: minibatch-kmeans istella 351 -> 256 ms, taxi 51.1 -> 48.5 ms,
silhouette identical .1182 / .138); `-D MOJOLEARN_MBK_ZEROCOPY_OFF` turns it
off (the old `-D MOJOLEARN_MBK_ZEROCOPY` is harmless); taken by bindings/_mojolearn_x_cluster.mojo for ENTRY_MINIBATCH.

Cause: the binding copies X into a FRESH host list (`read_f32`,
bindings/_mojolearn_x_cluster.mojo `call_binding`) before `ops.put` uploads
it. At the board shape (Istella 1M x 220 float32, 880 MB) that list's first
touch is ~21 ms per 64 MB of page faults (memory
metal-transfer-costs-on-apple: ~290 ms on the M4), most of the fit
(minibatch-kmeans Istella 358 ms on the M3 vs scikit-learn 124 ms), while
an upload from a raw host pointer runs at ~2 ms per 64 MB.

Here X is uploaded from the caller's pointer (`ctx.enqueue_copy(src_ptr=)`,
no host list), and the only host reads of X are the init-sized row gathers
`minibatch_fit` already does (the validation and init samples). Everything
else is `minibatch_fit`'s FAST path word for word (the same seeded draws in
the same order, `ops.minibatch_fast` for the steps, `nearest_all` for the
labels), so a fit returns what the copying path returns. Unit weights and
tol <= 0 only (the board's shape; the device steps take nothing else):
anything else returns False and the binding takes the copying path.
IDENTICAL compiles none of this.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_cluster.bodies import SplitMix64
from x_cluster.common import greedy_kmeans_pp, nearest_all, sum_f64, weighted_draw
from x_cluster.device_ops import DeviceOps
from x_cluster.minibatch import MiniBatchParams
from x_cluster.minibatch_fast import MINIBATCH_FAST_DEV, MBK_CLS2_POOL, MBK_W2_LABRG, MBF_RG_MAXK, mbk_labels_rg
from core.device_pool import pool_give, pool_take
from x_cluster.out import ClusterOut

comptime MBK_ZEROCOPY = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and MINIBATCH_FAST_DEV
    and not is_defined["MOJOLEARN_MBK_ZEROCOPY_OFF"]()
)  # default since the M3 A/B (docstring above); MINIBATCH_FAST_DEV is required

comptime XPtr = MutPointer[Float32, MutUntrackedOrigin]


@always_inline
def _gather_ptr(xp: XPtr, d: Int, idx: List[Int]) -> List[Float32]:
    """`common.gather_rows` on the caller's array."""
    var out = List[Float32](capacity=len(idx) * d)
    for r in idx:
        for f in range(d):
            out.append(xp[r * d + f])
    return out^


def minibatch_entry_ptr(
    mut ops: DeviceOps, xp: XPtr, nx: Int, a: List[Float32], ip: List[Int], fp: List[Float64],
    mut out: ClusterOut,
) raises -> Bool:
    """`entries.minibatch_entry` with X read from `xp` (nx values, alive for
    the call). False (nothing done, `out` untouched) outside the switch or
    for weights, tol > 0, or a shape the device steps refuse before any
    work (the binding then copies X and runs `minibatch_entry`)."""
    comptime if MBK_ZEROCOPY:
        var n = ip[0]
        var d = ip[1]
        var has_w = len(ip) > 11 and ip[11] != 0
        if has_w or fp[0] > 0 or n * d != nx:
            return False
        var p = MiniBatchParams(
            k=ip[2], max_iter=ip[3], batch_size=ip[4], tol=fp[0], max_no_improvement=ip[5],
            init_size=ip[6], n_init=ip[7], reassignment_ratio=fp[1], seed=UInt64(ip[9]),
            has_init=ip[8] != 0, init_random=len(ip) > 10 and ip[10] != 0,
        )
        var k = p.k
        if k < 1 or k > n:
            raise Error("MiniBatchKMeans: n_samples=" + String(n) + " should be >= n_clusters=" + String(k))
        if p.batch_size < 1:
            raise Error("MiniBatchKMeans: batch_size must be >= 1")
        if p.reassignment_ratio < 0:
            raise Error("reassignment_ratio should be >= 0")
        var batch = p.batch_size if p.batch_size < n else n
        # minibatch_fast_steps' shape caps, checked before any draw so a
        # refusal leaves nothing done
        if k * d > 4096 or k > 256 or batch > 4096:
            return False
        var n_c = k * d if p.has_init else 0
        var centers = List[Float32](capacity=n_c)
        for t in range(n_c):
            centers.append(a[t])
        var init_size = p.init_size
        if init_size <= 0:
            init_size = 3 * batch
            if init_size < k:
                init_size = 3 * k
        elif init_size < k:
            init_size = 3 * k
        if init_size > n:
            init_size = n
        var rng = SplitMix64(p.seed)
        # X straight from the caller's array: no host list, no synchronize
        # here (the stream orders every later kernel behind the copy; the
        # first read below synchronizes, and `xp` outlives the call)
        var xbuf: DeviceBuffer[DType.float32]
        comptime if MBK_CLS2_POOL:
            # lane/apple-fast-gap-cls2: a pooled buffer, returned below
            xbuf = pool_take["MojoXClusterCls2MbkX"](ops.ctx, nx)
        else:
            xbuf = ops.ctx.enqueue_create_buffer[DType.float32](nx)
        ops.ctx.enqueue_copy(dst_buf=xbuf, src_ptr=xp)
        ops.f.append(xbuf^)
        var xs = len(ops.f) - 1

        var vidx = List[Int](capacity=init_size)
        for _t in range(init_size):
            vidx.append(rng.below(n))
        var vslot = ops.put(_gather_ptr(xp, d, vidx))
        var best = List[Float32]()
        var best_inertia = Float64(0)
        var n_init = 1 if p.has_init else p.n_init
        for it in range(n_init):
            var cand: List[Float32]
            if p.has_init:
                cand = centers.copy()
            else:
                var iidx = List[Int](capacity=init_size)
                for _t in range(init_size):
                    iidx.append(rng.below(n))
                if p.init_random:
                    var taken = List[Bool](length=init_size, fill=False)
                    var picks = List[Int]()
                    for _c in range(k):
                        var cum = List[Float64](capacity=init_size)
                        var acc = Float64(0)
                        for t in range(init_size):
                            if not taken[t]:
                                acc = acc + Float64(1)
                            cum.append(acc)
                        var r = weighted_draw(cum, rng)
                        taken[r] = True
                        picks.append(iidx[r])
                    cand = _gather_ptr(xp, d, picks)
                else:
                    cand = greedy_kmeans_pp(ops, _gather_ptr(xp, d, iidx), init_size, d, k, rng)
            var vl = List[Int32]()
            var vd = List[Float32]()
            nearest_all(ops, vslot, init_size, cand, k, d, vl, vd)
            var inertia = sum_f64(vd, init_size)
            if it == 0 or inertia < best_inertia:
                best_inertia = inertia
                best = cand^

        var c = best^
        var w = List[Float32](length=k, fill=Float32(0))
        var n_steps = (p.max_iter * n) // batch
        var steps_done = 0
        if n_steps > 0:
            if not ops.minibatch_fast(
                xs, n, d, k, batch, n_steps, p.max_no_improvement, p.reassignment_ratio, p.seed, rng, c, w,
                steps_done,
            ):
                raise Error("MiniBatchKMeans: the device steps refused a shape the zero-copy path admitted")
        var labels = List[Int32]()
        var dist = List[Float32]()
        var labeled = False
        comptime if MBK_W2_LABRG:
            # lane/apple-fast-w2-clres, opt-in: the all-rows labelling as a
            # 32-thread group per row (x_cluster/minibatch_fast.mojo)
            if k <= MBF_RG_MAXK:
                var cs = ops.put(c)
                var ls = ops.zeros_i(n)
                var ds = ops.zeros(n)
                labeled = mbk_labels_rg(ops.ctx, ops._fp(xs), n, ops._fp(cs), k, d, ops._ip(ls), ops._fp(ds))
                if labeled:
                    ops.get_if(ls, n, ds, n, labels, dist)
        if not labeled:
            nearest_all(ops, xs, n, c, k, d, labels, dist)
        comptime if MBK_CLS2_POOL:
            # nearest_all read its labels back (a synchronize): no launch
            # still reads X; the slot's later entries are not used again
            ops.ctx.synchronize()
            pool_give["MojoXClusterCls2MbkX"](ops.f.pop(xs))
        var n_iter = (steps_done * batch + n - 1) // n
        var inertia = sum_f64(dist, n)
        out.f.append(c^)
        out.f.append(w^)
        out.i.append(labels^)
        out.s.append(inertia)
        out.s.append(Float64(steps_done))
        out.s.append(Float64(n_iter))
        return True
    return False
