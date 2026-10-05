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
no host list), and X is never read on the host: the validation and init
samples and the picked init centers are gathered from the resident X on
the device (`_gather_slot`, `_kpp_indices_slot`; lane cpu3-core). Everything
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
from x_cluster.common import nearest_all, sum_f64
from x_cluster.post_bodies import FM_VAL
from x_cluster.device_ops import DeviceOps
from x_cluster.minibatch import MiniBatchParams
from x_cluster.minibatch_fast import MINIBATCH_FAST_DEV, MBK_CLS2_POOL, MBK_W2_LABRG, MBF_RG_MAXK, mbk_labels_rg
from core.device_pool import pool_give, pool_take
from core.device_scan import device_first_nonfinite
from x_cluster.out import ClusterOut

comptime MBK_ZEROCOPY = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and MINIBATCH_FAST_DEV
    and not is_defined["MOJOLEARN_MBK_ZEROCOPY_OFF"]()
)  # default since the M3 A/B (docstring above); MINIBATCH_FAST_DEV is required

#: lane/apple-fast-s-linalg (2026-10-04), default OFF, FAST + Apple only
#: (`-D MOJOLEARN_MBK_FAST_DEVSCAN`). What: MiniBatchKMeans.fit refuses NaN/inf
#: with the base binding's `all_finite_f32` (python/mojolearn/_expansion_cluster.py
#: `_f32`), a single-thread host loop with an early-exit branch over every
#: value: 220 million floats (880 MB) at the board's istella 1M x 220, on the
#: host, before this path uploads X anyway. With the define, Python skips that
#: scan and sets ip[12] = 1; this path scans the uploaded device copy
#: (`core/device_scan.mojo device_first_nonfinite`, one device read) and
#: raises the same message; the binding's copying fallback scans its host
#: list instead. Expect: minibatch-kmeans istella 145 -> ~80-110 ms (the host
#: scan is ~1 cycle a float); no arithmetic change.
comptime MBK_FAST_DEVSCAN = MBK_ZEROCOPY and is_defined["MOJOLEARN_MBK_FAST_DEVSCAN"]()
#: the message Python maps back to its ValueError
comptime MBK_NONFINITE_MSG = "mojolearn: X contains NaN or infinity"

comptime XPtr = MutPointer[Float32, MutUntrackedOrigin]


def _gather_slot(mut ops: DeviceOps, xs: Int, d: Int, idx: List[Int32]) raises -> Int:
    """The rows `idx` of the resident X slot `xs`, gathered on the device into
    a new slot (`common.gather_rows`' words, no host read of X)."""
    var m = len(idx)
    var ids = ops.put_i(idx)
    var dst = ops.alloc(m * d)
    ops.gather_rows(xs, d, ids, m, dst)
    return dst


def _kpp_indices_slot(mut ops: DeviceOps, xs: Int, m: Int, d: Int, k: Int, mut rng: SplitMix64) raises -> List[Int32]:
    """`common.greedy_kmeans_pp_indices` (unit weights) over the resident
    slot `xs` (m x d): the same device steps in the same order and the same
    draws, so the same picks, without a host copy of the sample."""
    from std.math import log

    var n_trials = 2 + Int(log(Float64(k)))
    var picks = List[Int32](capacity=k)
    var first = rng.below(m)
    picks.append(Int32(first))
    var ids = ops.zeros_i(n_trials)
    ops.set_i(ids, [Int32(first)])
    var cslot = ops.alloc(n_trials * d)
    ops.gather_rows(xs, d, ids, 1, cslot)
    var closest = ops.alloc(m)
    ops.sqdist(cslot, 1, xs, m, d, closest)
    var pot = ops.sum_ff(closest, -1, -1, m, FM_VAL)
    var dc_s = ops.alloc(n_trials * m)
    for _c in range(1, k):  # small-loop(k: centers picked): k <= 256, each pick launches device steps
        var vs = List[Float64](capacity=n_trials)
        for _t in range(n_trials):  # small-loop(n_trials: candidate uniforms): 2 + log(k) draws per pick
            vs.append(rng.unit() * pot)
        ops.kpp_search(closest, -1, m, vs, ids)
        ops.gather_rows(xs, d, ids, n_trials, cslot)
        ops.sqdist(cslot, n_trials, xs, m, d, dc_s)
        var pots = ops.kpp_pots(dc_s, closest, -1, n_trials, m)
        var best = 0
        for t in range(1, n_trials):  # small-loop(n_trials: candidate potentials): 2 + log(k) words per pick
            if pots[t] < pots[best]:
                best = t
        ops.kpp_take(dc_s, closest, best, m)
        pot = pots[best]
        picks.append(ops.get_i(ids, n_trials)[best])
    return picks^


def minibatch_entry_ptr(
    mut ops: DeviceOps, xp: XPtr, nx: Int, a: List[Float32], ip: List[Int], fp: List[Float64],
    mut out: ClusterOut,
) raises -> Bool:
    """`entries.minibatch_entry` with X read from `xp` (nx values, alive for
    the call). False (nothing done, `out` untouched) outside the switch or
    for weights (tol > 0 only under the GENERAL_OFF define), or a shape the device steps refuse before any
    work (the binding then copies X and runs `minibatch_entry`)."""
    comptime if MBK_ZEROCOPY:
        var n = ip[0]
        var d = ip[1]
        var has_w = len(ip) > 11 and ip[11] != 0
        # lane/no-bench-tuning-2: tol > 0 runs here too (the device steps
        # stop on the squared center shift); weights take `minibatch_entry`,
        # whose weighted init and draws run the same device steps
        var tol_ok = fp[0] <= 0 or not is_defined["MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF"]()
        if has_w or not tol_ok or n * d != nx:
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
        for t in range(n_c):  # small-loop(n_c: user init center words): k * d <= 4096 checked above
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
        comptime if MBK_FAST_DEVSCAN:
            # Python skipped its host scan (ip[12] = 1): scan the device copy
            if len(ip) > 12 and ip[12] != 0:
                if device_first_nonfinite(ops.ctx, xbuf, nx) >= 0:
                    ops.ctx.synchronize()
                    comptime if MBK_CLS2_POOL:
                        pool_give["MojoXClusterCls2MbkX"](xbuf^)
                    raise Error(MBK_NONFINITE_MSG)
        ops.f.append(xbuf^)
        var xs = len(ops.f) - 1

        # the validation sample: seeded row ids on the host (the serial
        # SplitMix64 stream of `minibatch_fit`), the rows gathered on the device
        var vidx = List[Int32](capacity=init_size)
        for _t in range(init_size):  # small-loop(init_size: seeded sample row ids): serial seeded draws, no data read
            vidx.append(Int32(rng.below(n)))
        var vslot = _gather_slot(ops, xs, d, vidx)
        var best = List[Float32]()
        var best_inertia = Float64(0)
        var n_init = 1 if p.has_init else p.n_init
        for it in range(n_init):  # small-loop(n_init: init restarts): each restart launches device steps
            var cand: List[Float32]
            if p.has_init:
                cand = centers.copy()
            else:
                var iidx = List[Int32](capacity=init_size)
                for _t in range(init_size):  # small-loop(init_size: seeded sample row ids): serial seeded draws, no data read
                    iidx.append(Int32(rng.below(n)))
                var picks: List[Int32]
                if p.init_random:
                    # `weighted_draw` over the 0/1 table of untaken sample
                    # positions, without the table: u * free (free = the
                    # count of untaken positions, an exact Float64 integer)
                    # searched side left lands on the (floor + 1)-th
                    # untaken position, found by stepping over the taken
                    # ones in ascending order (the same pick, word for word)
                    picks = List[Int32](capacity=k)
                    var tk = List[Int](capacity=k)
                    for c in range(k):  # small-loop(k: centers picked): k <= 256 checked above
                        var free = init_size - c
                        var r = Int(rng.unit() * Float64(free))
                        if r > free - 1:
                            r = free - 1
                        var pos = 0
                        for q in range(len(tk)):  # small-loop(tk: taken sample positions): at most k <= 256 entries
                            if tk[q] <= r:
                                r += 1
                                pos = q + 1
                        tk.insert(pos, r)
                        picks.append(iidx[r])
                    # the k picked rows of X, gathered on the device; the
                    # k x d centers come back (k * d <= 4096 words)
                    var csl = _gather_slot(ops, xs, d, picks)
                    cand = ops.get(csl, k * d)
                else:
                    # kmeans++ on the device-gathered sample, its picked
                    # rows gathered there too (`greedy_kmeans_pp`'s words)
                    var isl = _gather_slot(ops, xs, d, iidx)
                    picks = _kpp_indices_slot(ops, isl, init_size, d, k, rng)
                    var csl = _gather_slot(ops, isl, d, picks)
                    cand = ops.get(csl, k * d)
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
                steps_done, p.tol, List[Float64](),
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
