# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column of the public all-rows k-NN graph (`mojolearn.knn_graph`,
lane sc-ann-1, 2026-11-28): the exact graph (`cagra_host.knn_rows_host`, the
cell's words) or the IVF graph of `x_ann/cagra_fast_knn.mojo::ivf_graph_enqueue`
with the same arithmetic, step for step, so the CPU column holds the GPU
columns' bits under IDENTICAL:

  * every distance is `tsne_core.ts_sqdist2` (the exact graph's chain);
  * the k-means sample and seeds are the stride rows shifted by the plan's
    offset; each of IVFG_ITERS Lloyd steps assigns every sample to its
    nearest centroid under (distance, index), then each centroid becomes the
    flushed sum of its samples in ascending sample order (the device's stable
    sort order) over one `identical_div`; an empty list keeps its centroid;
  * every row goes to its nearest centroid; a list's rows are ascending (the
    stable sort); a list's probe set is itself and its n_probes - 1 nearest
    other lists by centroid under (distance, index);
  * a row's neighbors are the k nearest other rows of its probe pool under
    (distance, index): the selection is a function of the candidate SET, so
    the device's offer order does not matter.
Parallel: every sample, row and list is its own unit (`ann_tasks`); the one
serial pass per step is the stable counting placement of rows into lists
(a prefix scan, O(n))."""

from checks.numerics import ftz, identical_div, identical_sqrt
from x_ann.tsne_core import F32P, I32P, ts_sqdist2, ts_knn_offer
from x_ann.host.ann_host_cells import ann_span, ann_task_count, ann_tasks
from x_ann.host.cagra_host import knn_rows_host
from x_ann.host.ivf_pq_host import fp, ip
from x_ann.knn_graph_plan import KnnGraphPlan, KG_METHOD_EXACT, KG_METHOD_IVF, IVFG_SAMPLE, IVFG_ITERS


def _kg_nearest(q: F32P, nq: Int, c: F32P, nc: Int, d: Int, mut lab: List[Int32]):
    """lab[i] = the nearest row of c to row i of q under (distance, index)."""
    var lp = ip(lab)
    var tasks = ann_task_count(nq, nc * d)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, nq)
        for i in range(span[0], span[1]):
            var best = 0
            var bd = ts_sqdist2(q, i, c, 0, d)
            for j in range(1, nc):
                var dv = ts_sqdist2(q, i, c, j, d)
                if dv < bd:
                    bd = dv
                    best = j
            lp.unsafe_store(i, Int32(best))

    ann_tasks(task, tasks)


def _kg_buckets(lab: List[Int32], n: Int, nlist: Int, mut start: List[Int], mut members: List[Int]):
    """The stable placement of 0..n-1 by label: list c's members are
    members[start[c] : start[c + 1]], ascending (a prefix scan, serial by
    nature: O(n + nlist))."""
    for c in range(nlist + 1):  # driver-loop: over lists (clear the counts)
        start[c] = 0
    for i in range(n):  # serial by nature: the counting pass of a stable placement
        start[Int(lab[i]) + 1] += 1
    for c in range(nlist):  # serial by nature: the prefix scan of the counts
        start[c + 1] += start[c]
    var fill = List[Int](length=nlist, fill=0)
    for i in range(n):  # serial by nature: the stable scatter (rows ascending in each list)
        var c = Int(lab[i])
        members[start[c] + fill[c]] = i
        fill[c] += 1


def _kg_lloyd(xs: F32P, ns: Int, d: Int, mut cent: List[Float32], nlist: Int, start: List[Int],
              members: List[Int]):
    """Each centroid with members: their flushed sum in member order over one
    `identical_div` (the device's `cg_lloyd_update_kernel`)."""
    var cp = fp(cent)
    var sp = start.unsafe_ptr()
    var mp = members.unsafe_ptr()
    var tasks = ann_task_count(nlist, (ns // nlist + 1) * d)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, nlist)
        for c in range(span[0], span[1]):
            var lo = sp[c]
            var hi = sp[c + 1]
            if hi > lo:
                var cnt = Float32(hi - lo)
                for f in range(d):
                    var acc = Float32(0.0)
                    for m in range(lo, hi):
                        acc = ftz(acc + ftz(xs.unsafe_load(mp[m] * d + f)))
                    cp.unsafe_store(c * d + f, identical_div(acc, cnt))

    ann_tasks(task, tasks)
    _ = start.unsafe_ptr()
    _ = members.unsafe_ptr()


def _kg_stride(x: F32P, n: Int, d: Int, m: Int, off: Int, mut dst: List[Float32]):
    """dst row r = x row ((r n) / m + off) mod n (`cg_gather_stride_kernel`)."""
    for r in range(m):  # driver-loop: over sample rows (a copy, O(m d))
        var row = ((r * n) // m + off) % n
        for c in range(d):
            dst[r * d + c] = x.unsafe_load(row * d + c)


def ivf_graph_host(x: F32P, n: Int, d: Int, k: Int, nlist: Int, n_probes: Int, seed_off: Int,
                   nn_d: F32P, nn_i: I32P) raises -> Bool:
    """`ivf_graph_enqueue` on the CPU: the same graph words. False (nothing
    written) when a probe pool holds fewer than k + 1 rows."""
    if nlist < 1 or n_probes < 1 or n_probes > nlist or nlist > n or k < 1 or k >= n:
        raise Error("IVF graph: need 1 <= n_probes <= n_lists <= n_samples and 1 <= k < n_samples")
    var ns = nlist * IVFG_SAMPLE
    if ns > n:
        ns = n
    var off = seed_off % n
    var xs = List[Float32](length=ns * d, fill=Float32(0.0))
    var cent = List[Float32](length=nlist * d, fill=Float32(0.0))
    _kg_stride(x, n, d, ns, off, xs)
    _kg_stride(x, n, d, nlist, off, cent)
    var big = n if n > ns else ns
    var lab = List[Int32](length=big, fill=Int32(0))
    var start = List[Int](length=nlist + 1, fill=0)
    var members = List[Int](length=big, fill=0)
    for _ in range(IVFG_ITERS):  # driver-loop: over Lloyd iterations
        _kg_nearest(fp(xs), ns, fp(cent), nlist, d, lab)
        _kg_buckets(lab, ns, nlist, start, members)
        _kg_lloyd(fp(xs), ns, d, cent, nlist, start, members)
    # every row to its list
    _kg_nearest(x, n, fp(cent), nlist, d, lab)
    _kg_buckets(lab, n, nlist, start, members)
    # probe sets: itself, then its n_probes - 1 nearest other lists
    var pm1 = n_probes - 1
    var probes = List[Int32](length=nlist * n_probes, fill=Int32(0))
    var cnd = List[Float32](length=nlist * pm1 + 1, fill=Float32(0.0))
    var cni = List[Int32](length=nlist * pm1 + 1, fill=Int32(0))
    var pp = ip(probes)
    var cdp = fp(cnd)
    var cip = ip(cni)
    var ctp = fp(cent)
    var lt = ann_task_count(nlist, nlist * d)

    def probe_task(t: Int) {imm}:
        var span = ann_span(t, lt, nlist)
        for c in range(span[0], span[1]):
            pp.unsafe_store(c * n_probes, Int32(c))
            var filled = 0
            if pm1 > 0:
                for o in range(nlist):
                    if o == c:
                        continue
                    filled = ts_knn_offer(ts_sqdist2(ctp, c, ctp, o, d), o, c * pm1, pm1, filled, cdp, cip)
                for t2 in range(pm1):
                    pp.unsafe_store(c * n_probes + 1 + t2, cip.unsafe_load(c * pm1 + t2))

    ann_tasks(probe_task, lt)
    for c in range(nlist):  # driver-loop: over lists (the pool-size refusal)
        var pool = 0
        for t in range(n_probes):
            var p = Int(probes[c * n_probes + t])
            pool += start[p + 1] - start[p]
        if pool < k + 1:
            return False
    # the graph: each row against its list's probe pool
    var sp = start.unsafe_ptr()
    var mp = members.unsafe_ptr()
    var lp = lab.unsafe_ptr()
    var rt = ann_task_count(n, n_probes * (n // nlist + 1) * d)

    def row_task(t: Int) {imm}:
        var span = ann_span(t, rt, n)
        for i in range(span[0], span[1]):
            var c = Int(lp[i])
            var base = i * k
            var filled = 0
            for t2 in range(n_probes):
                var p = Int(pp.unsafe_load(c * n_probes + t2))
                for m in range(sp[p], sp[p + 1]):
                    var j = mp[m]
                    if j == i:
                        continue
                    filled = ts_knn_offer(ts_sqdist2(x, i, x, j, d), j, base, k, filled, nn_d, nn_i)

    ann_tasks(row_task, rt)
    _ = start^
    _ = members^
    _ = lab^
    _ = probes^
    _ = cnd^
    _ = cni^
    _ = cent^
    _ = xs^
    return True


def knn_graph_host(x: F32P, n: Int, d: Int, k: Int, plan: KnnGraphPlan, nn_d: F32P, nn_i: I32P,
                   root: Bool) raises -> Int:
    """The planned graph into the caller's n x k arrays; the method run
    (KG_METHOD_IVF, or KG_METHOD_EXACT when asked, planned or refused by a
    short probe pool). `root`: the euclidean roots of the kept squared
    distances (`identical_sqrt`, as the device's root pass)."""
    var used = KG_METHOD_EXACT
    if plan.method == KG_METHOD_IVF:
        if ivf_graph_host(x, n, d, k, plan.n_lists, plan.n_probes, plan.seed_off, nn_d, nn_i):
            used = KG_METHOD_IVF
    if used == KG_METHOD_EXACT:
        knn_rows_host(x, n, d, k, nn_d, nn_i)
    if root:
        var total = n * k
        var tasks = ann_task_count(total, 8)

        def root_task(t: Int) {imm}:
            var span = ann_span(t, tasks, total)
            for e in range(span[0], span[1]):
                nn_d.unsafe_store(e, identical_sqrt(nn_d.unsafe_load(e)))

        ann_tasks(root_task, tasks)
    return used
