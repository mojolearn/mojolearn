from experiments.classical_identical_ideas.graph_controls import C37_ROW_PANELS, C37_PANEL_ROWS
from experiments.classical_identical_ideas.graph_controls import C39_RETAIN_STATE
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MiniBatchKMeans (lane/algos-cluster). Reference: scikit-learn
`sklearn/cluster/_kmeans.py` (MiniBatchKMeans.fit :2056-2210,
`_mini_batch_step` :1566-1684, `_mini_batch_convergence`) and
`_k_means_minibatch.pyx::update_center_dense`.

One generic driver over `ClusterOps`: the batch assignment (the n x k
distance work) is the device's; the batch draws, the center update in
sklearn's order (`c * w`, `+ x` in batch order, `w += wsum`, `* (1 / w)`),
the random reassignment and the early stop are host code from this one
source in both bindings.

THE BATCH ORDER COMES FROM THE SEED: `SplitMix64(seed)` draws the validation
rows, then per init the init rows and the k-means++ picks, then per step the
batch rows (uniform with replacement, unit weights) and any reassignment.
Not scikit-learn's Mersenne Twister stream, so a fit agrees with sklearn's
at a tolerance, never bit for bit (NOT_IMPLEMENTED.tsv)."""
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_mul, identical_mul64
from checks.numerics import NUMERIC_IDENTICAL
from x_cluster.bodies import SPLITMIX_GAMMA, SplitMix64, splitmix_at
from x_cluster.common import gather_rows, greedy_kmeans_pp, nearest_all, sum_f64, weighted_draw
from x_cluster.post_bodies import FM_VAL, FOLD_CHUNK
from x_cluster.minibatch_fast import MINIBATCH_FAST_DEV
from x_cluster.ops import ClusterOps


# Lane cluster-apple3, FAST only, OPT-IN while unproven
# (`-D MOJOLEARN_MINIBATCH_ONE_PASS=1`): `update_center_dense` for every
# center in ONE walk of the batch. Each center still gets `c * w`, then its
# rows in batch order, then `w += wsum` and the scale, so every center's
# chain of operations is the per-center loop's and no bit moves; the batch is
# walked twice, not 2 k times.
comptime MINIBATCH_ONE_PASS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_MINIBATCH_ONE_PASS"]()
)


# fam2-cluster (2026-10-04), IDENTICAL, default ON, BOTH COLUMNS (this is the
# one driver): THE STEP LOOP IN GROUPS. A step uploaded its batch's row ids,
# launched, then waited for the batch's distances and the k counts (one
# drain a step; the host drew the ids and summed the distances). Now
# MB_GROUP steps are enqueued between two waits: the unit-weight batch ids
# are drawn where the data is (`ops.mb_draw`: draw t + 1 of the fit's
# splitmix64 stream by its counter, `% n`, the stream's own values), the
# batch inertia is the float-float fold of the distances left in a history
# slot (`ops.fold_at`), and each step's centers and counts are copied into
# history slots (`ops.copy_at`). One read returns the group's inertias,
# counts and centers and the host replays `_mini_batch_convergence` and the
# reassignment test step by step, exactly as before. A step that reassigns
# or stops discards the steps enqueued after it: the centers, the counts and
# the stream go back to that step's (the history has them), so every step
# sees the state the step-by-step loop gave it.
# BITS: the batch inertia was one ascending Float64 chain on the host; it is
# now `post_bodies`' float-float fold (the same on the device and the host
# column, which run this same loop). The early stop compares EWAs of it, so
# a fit can stop a step earlier or later than before; centers of a fit that
# runs the same steps are the same words.
# `-D MOJOLEARN_IDN_MINIBATCH_GROUP_OFF=1` (or the master) restores the
# per-step loop; the define must reach the host-column build too.
# CANDIDATE ARMS for the group (default 16):
# `-D MOJOLEARN_IDN_MINIBATCH_GROUP8=1`, `..._GROUP32=1`, `..._GROUP64=1`.
comptime IDN_MINIBATCH_GROUP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)
comptime MB_GROUP = (
    8 if is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP8"]()
    else (
        32 if is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP32"]()
        else (64 if is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP64"]() else 16)
    )
)


def mb_fold_scratch(n: Int) -> Int:
    """Floats `ops.fold_at` needs in each scratch slot for a fold of n."""
    var total = 0
    var nch = (n + FOLD_CHUNK - 1) // FOLD_CHUNK
    while nch > 1:
        total += nch
        nch = (nch + FOLD_CHUNK - 1) // FOLD_CHUNK
    return total if total > 0 else 1


@fieldwise_init
struct MiniBatchParams(Copyable, Movable):
    var k: Int
    var max_iter: Int
    var batch_size: Int
    var tol: Float64
    var max_no_improvement: Int  # < 0 means None
    var init_size: Int  # <= 0 means the default
    var n_init: Int
    var reassignment_ratio: Float64
    var seed: UInt64
    var has_init: Bool  # centers passed in (init array): n_init is 1
    var init_random: Bool  # init='random': k distinct rows of the init subset


@fieldwise_init
struct MiniBatchResult(Copyable, Movable):
    var inertia: Float64
    var n_steps: Int
    var n_iter: Int


def minibatch_fit[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, p: MiniBatchParams,
    mut centers: List[Float32], mut labels: List[Int32], mut counts: List[Float32],
    weights: List[Float32] = List[Float32](),
) raises -> MiniBatchResult:
    """Fit; `centers` holds the init array when `p.has_init`, and is
    overwritten with the fitted centers. `weights` (empty: unit) are
    sklearn's sample_weight: they weigh the k-means++ potentials, the
    validation inertia and the batch draw (`choice(p=w / sum(w))`); the batch
    itself updates with unit weights, as sklearn's `unit_sample_weight`."""
    var weighted = len(weights) > 0
    var cum_w = List[Float64]()
    if weighted:
        var acc = Float64(0)
        for t in range(n):
            if not (weights[t] >= Float32(0)):
                raise Error("MiniBatchKMeans: sample_weight must be non-negative")
            acc = acc + Float64(weights[t])
            cum_w.append(acc)
        if not (acc > 0):
            raise Error("MiniBatchKMeans: sample_weight must have a positive sum")
    var k = p.k
    if k < 1 or k > n:
        raise Error("MiniBatchKMeans: n_samples=" + String(n) + " should be >= n_clusters=" + String(k))
    if p.batch_size < 1:
        raise Error("MiniBatchKMeans: batch_size must be >= 1")
    if p.reassignment_ratio < 0:
        raise Error("reassignment_ratio should be >= 0")
    var batch = p.batch_size if p.batch_size < n else n
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
    var xs = ops.put(x)

    # validation rows, then n_init inits scored on them
    var vidx = List[Int](capacity=init_size)
    for _t in range(init_size):
        vidx.append(rng.below(n))
    var vslot = ops.put(gather_rows(x, d, vidx))
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
            var wi = List[Float32]()
            if weighted:
                for t in iidx:
                    wi.append(weights[t])
            if p.init_random:
                # choice(init_size, k, replace=False, p=w/sum(w)): sequential
                # weighted draws over the rows not yet taken (unit: uniform)
                var taken = List[Bool](length=init_size, fill=False)
                var picks = List[Int]()
                for _c in range(k):
                    var cum = List[Float64](capacity=init_size)
                    var acc = Float64(0)
                    for t in range(init_size):
                        if not taken[t]:
                            acc = acc + (Float64(wi[t]) if weighted else Float64(1))
                        cum.append(acc)
                    if not (acc > 0):
                        raise Error("MiniBatchKMeans: fewer positive-weight rows than n_clusters in the init sample")
                    var r = weighted_draw(cum, rng)
                    taken[r] = True
                    picks.append(iidx[r])
                cand = gather_rows(x, d, picks)
            else:
                cand = greedy_kmeans_pp(ops, gather_rows(x, d, iidx), init_size, d, k, rng, wi)
        var vl = List[Int32]()
        var vd = List[Float32]()
        nearest_all(ops, vslot, init_size, cand, k, d, vl, vd)
        var inertia = sum_f64(vd, init_size)
        if weighted:
            inertia = Float64(0)
            for t in range(init_size):
                inertia = inertia + identical_mul64(Float64(weights[vidx[t]]), Float64(vd[t]))
        if it == 0 or inertia < best_inertia:
            best_inertia = inertia
            best = cand^

    var c = best^
    var w = List[Float32](length=k, fill=Float32(0))
    var ewa = Float64(0)
    var have_ewa = False
    var ewa_min = Float64(0)
    var have_min = False
    var no_improvement = 0
    var since_reassign = 0
    var n_steps = (p.max_iter * n) // batch
    var bslot = ops.zeros(batch * d)
    var cslot = ops.zeros(k * d)
    var lslot = ops.zeros_i(batch)
    var dslot = ops.zeros(batch)
    # lane/neural-pass133 (2026-10-02): the centers and counts live in their
    # slots; a step uploads the batch's row ids (not its rows), gathers them
    # from `xs`, assigns and updates the centers with `ops.mb_update`
    # (`minibatch_step`'s unit-weight update_center_dense, the same words),
    # and reads back the batch distances and the k counts for the early stop
    # and the reassignment, which stay host code (a reassignment patches the
    # slots).
    var islot = ops.zeros_i(batch)
    var wslot = ops.zeros(k)
    ops.set(cslot, c)
    var steps_done = 0
    # lane/apple-fast-cluster (2026-10-02), FAST on Apple, ON by default
    # (`-D MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF=1` turns it off; see
    # x_cluster/minibatch_fast.mojo for the M3 A/B): MINIBATCH_FAST_DEV runs the
    # steps resident on the device (x_cluster/minibatch_fast.mojo: no upload,
    # read-back or host center fold per step; the loop below pays all three
    # every step). `c`, `w` and `steps_done` come back as the loop would
    # leave them, the loop is skipped (n_steps = 0) and `cslot` holds the
    # final centers for the readback after the loop (else it would hand back
    # the centers uploaded above). lane/no-bench-tuning-2 (2026-10-04): was
    # unit weights and tol <= 0 only (the board's configuration); weighted
    # draws and the tol stop now run on the device too
    # (`-D MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF=1` restores the gate).
    comptime if MINIBATCH_FAST_DEV:
        var general = not is_defined["MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF"]()
        if general or (not weighted and p.tol <= 0):
            if ops.minibatch_fast(
                xs, n, d, k, batch, n_steps, p.max_no_improvement, p.reassignment_ratio, p.seed, rng, c, w,
                steps_done, p.tol, cum_w,
            ):
                n_steps = 0
                ops.set(cslot, c)
    comptime if IDN_MINIBATCH_GROUP:
        var ch = ops.zeros(MB_GROUP * k * d)
        var wh = ops.zeros(MB_GROUP * k)
        var eh = ops.zeros(2 * MB_GROUP)
        var fsn = mb_fold_scratch(batch)
        var fth = ops.alloc(fsn)
        var ftl = ops.alloc(fsn)
        var gstep = 0
        var stop = False
        # the group grows 1, 2, 4, .. MB_GROUP while no step reassigns and
        # falls back to 1 when one does (the first steps of a fit with many
        # centers reassign often; a discarded step is wasted device work)
        var gsz = 1
        while gstep < n_steps and not stop:
            var g_n = n_steps - gstep
            if g_n > gsz:
                g_n = gsz
            gsz = gsz * 2
            if gsz > MB_GROUP:
                gsz = MB_GROUP
            # the stream after each step's batch draws, and (weighted) its rows
            var states = List[UInt64](capacity=g_n)
            var grows = List[List[Int]]()
            for g in range(g_n):
                if weighted:
                    var wrows = List[Int](capacity=batch)
                    var wi32 = List[Int32](capacity=batch)
                    for _t in range(batch):
                        var r = weighted_draw(cum_w, rng)
                        wrows.append(r)
                        wi32.append(Int32(r))
                    ops.set_i(islot, wi32)
                    grows.append(wrows^)
                else:
                    ops.mb_draw(islot, batch, n, rng.state)
                    rng.state = rng.state + UInt64(batch) * SPLITMIX_GAMMA
                states.append(rng.state)
                ops.mb_assign(xs, d, islot, batch, cslot, k, lslot, dslot, bslot)
                ops.mb_update(bslot, batch, lslot, cslot, wslot, k, d)
                ops.fold_at(dslot, batch, FM_VAL, eh, 2 * g, fth, ftl)
                ops.copy_at(cslot, k * d, ch, g * k * d)
                ops.copy_at(wslot, k, wh, g * k)
            # the one wait of the group
            var got = ops.gets([eh, wh, ch], [2 * g_n, g_n * k, g_n * k * d])
            for g in range(g_n):
                var step = gstep + g
                # _random_reassign(): counted BEFORE the step, as sklearn evaluates the argument
                since_reassign += batch
                var any_empty = False
                for j in range(k):
                    if w[j] == Float32(0):
                        any_empty = True
                var reassign = False
                if any_empty or since_reassign >= 10 * k:
                    since_reassign = 0
                    reassign = True
                var batch_inertia = Float64(got[0][2 * g]) + Float64(got[0][2 * g + 1])
                for j in range(k):
                    w[j] = got[1][g * k + j]
                var c_new = List[Float32](capacity=k * d)
                for t in range(k * d):
                    c_new.append(got[2][g * k * d + t])
                var to = List[Bool]()
                var nre = 0
                if reassign and p.reassignment_ratio > 0:
                    to = mb_reassign_marks(w, k, batch, p.reassignment_ratio)
                    for j in range(k):
                        if to[j]:
                            nre += 1
                var rolled = False
                if nre > 0:
                    # the step-by-step order of the stream: this step's batch
                    # draws, then its reassignment draws
                    rng.state = states[g]
                    var rows = List[Int](capacity=batch)
                    if weighted:
                        for t in range(batch):
                            rows.append(grows[g][t])
                    else:
                        var st0 = states[g] - UInt64(batch) * SPLITMIX_GAMMA
                        for t in range(batch):
                            rows.append(Int(splitmix_at(st0, UInt64(t + 1)) % UInt64(n)))
                    mb_reassign_apply(c_new, w, to, nre, x, rows, batch, k, d, rng)
                    ops.set(cslot, c_new)
                    ops.set(wslot, w)
                    rolled = True
                    gsz = 1
                var diff = Float64(0)
                if p.tol > 0:
                    for t in range(k * d):
                        var e = Float64(c_new[t]) - Float64(c[t])
                        diff = diff + identical_mul64(e, e)
                c = c_new^
                steps_done = step + 1
                # _mini_batch_convergence
                var bi = batch_inertia / Float64(batch)
                if step > 0:
                    if not have_ewa:
                        ewa = bi
                        have_ewa = True
                    else:
                        var alpha = identical_mul64(Float64(batch), 2.0) / Float64(n + 1)
                        if alpha > 1:
                            alpha = 1
                        ewa = identical_mul64(ewa, 1 - alpha) + identical_mul64(bi, alpha)
                    if p.tol > 0 and diff <= p.tol:
                        stop = True
                    else:
                        if not have_min or ewa < ewa_min:
                            no_improvement = 0
                            ewa_min = ewa
                            have_min = True
                        else:
                            no_improvement += 1
                        if p.max_no_improvement >= 0 and no_improvement >= p.max_no_improvement:
                            stop = True
                if (stop or rolled) and g + 1 < g_n:
                    # the steps enqueued after this one are discarded: the
                    # device and the stream go back to this step's state
                    if not rolled:
                        rng.state = states[g]
                        comptime if C39_RETAIN_STATE:
                            ops.copy_from(ch,g*k*d,k*d,cslot)
                            ops.copy_from(wh,g*k,k,wslot)
                        else:
                            ops.set(cslot, c)
                            ops.set(wslot, w)
                    break
                if stop:
                    break
            gstep = steps_done
        n_steps = 0
    for step in range(n_steps):
        var bidx = List[Int](capacity=batch)
        for _t in range(batch):
            if weighted:
                bidx.append(weighted_draw(cum_w, rng))
            else:
                bidx.append(rng.below(n))
        # _random_reassign(): counted BEFORE the step, as sklearn evaluates the argument
        since_reassign += batch
        var any_empty = False
        for j in range(k):
            if w[j] == Float32(0):
                any_empty = True
        var reassign = False
        if any_empty or since_reassign >= 10 * k:
            since_reassign = 0
            reassign = True
        var c_new = List[Float32]()
        var batch_inertia: Float64
        var bi32 = List[Int32](capacity=batch)
        for t in range(batch):
            bi32.append(Int32(bidx[t]))
        ops.set_i(islot, bi32)
        ops.mb_assign(xs, d, islot, batch, cslot, k, lslot, dslot, bslot)
        ops.mb_update(bslot, batch, lslot, cslot, wslot, k, d)
        var got = ops.gets([dslot, wslot], [batch, k])
        batch_inertia = sum_f64(got[0], batch)
        w = got[1].copy()
        var need_c = p.tol > 0
        var to = List[Bool]()
        var nre = 0
        if reassign and p.reassignment_ratio > 0:
            to = mb_reassign_marks(w, k, batch, p.reassignment_ratio)
            for j in range(k):
                if to[j]:
                    nre += 1
        if need_c or nre > 0:
            c_new = ops.get(cslot, k * d)
        if nre > 0:
            mb_reassign_apply(c_new, w, to, nre, x, bidx, batch, k, d, rng)
            ops.set(cslot, c_new)
            ops.set(wslot, w)
        if not need_c:
            c_new = c.copy()
        var diff = Float64(0)
        if p.tol > 0:
            for t in range(k * d):
                var e = Float64(c_new[t]) - Float64(c[t])
                diff = diff + identical_mul64(e, e)
        c = c_new^
        steps_done = step + 1
        # _mini_batch_convergence
        var bi = batch_inertia / Float64(batch)
        if step == 0:
            continue
        if not have_ewa:
            ewa = bi
            have_ewa = True
        else:
            var alpha = identical_mul64(Float64(batch), 2.0) / Float64(n + 1)
            if alpha > 1:
                alpha = 1
            ewa = identical_mul64(ewa, 1 - alpha) + identical_mul64(bi, alpha)
        if p.tol > 0 and diff <= p.tol:
            break
        if not have_min or ewa < ewa_min:
            no_improvement = 0
            ewa_min = ewa
            have_min = True
        else:
            no_improvement += 1
        if p.max_no_improvement >= 0 and no_improvement >= p.max_no_improvement:
            break

    c = ops.get(cslot, k * d)
    var dist = List[Float32]()
    nearest_all(ops, xs, n, c, k, d, labels, dist)
    centers = c^
    counts = w^
    var n_iter = (steps_done * batch + n - 1) // n
    var inertia = sum_f64(dist, n)
    if weighted:
        inertia = Float64(0)
        for t in range(n):
            inertia = inertia + identical_mul64(Float64(weights[t]), Float64(dist[t]))
    return MiniBatchResult(inertia, steps_done, n_iter)


def mb_reassign_marks(w: List[Float32], k: Int, batch: Int, ratio: Float64) -> List[Bool]:
    """sklearn `_mini_batch_step`'s low-count centers to reassign (at most
    half the batch: the stable ascending sort by weight keeps `[half:]`)."""
    var wmax = w[0]
    for j in range(1, k):
        if w[j] > wmax:
            wmax = w[j]
    var thr = identical_mul64(Float64(ratio), Float64(wmax))
    var to = List[Bool](length=k, fill=False)
    var nre = 0
    for j in range(k):
        if Float64(w[j]) < thr:
            to[j] = True
            nre += 1
    var half = batch // 2
    if 2 * nre > batch:
        # np.argsort(weight_sums)[half:] stay: a stable ascending sort by weight
        var order = List[Int](capacity=k)
        for j in range(k):
            order.append(j)
        for a in range(1, k):
            var b = a
            while b > 0 and w[order[b - 1]] > w[order[b]]:
                var tmp = order[b - 1]
                order[b - 1] = order[b]
                order[b] = tmp
                b -= 1
        for q in range(half, k):
            to[order[q]] = False
    return to^


def mb_reassign_apply(
    mut c_new: List[Float32], mut w: List[Float32], to: List[Bool], nre: Int, src: List[Float32],
    rows: List[Int], batch: Int, k: Int, d: Int, mut rng: SplitMix64,
):
    """The reassignment of the marked centers: `choice(batch, nre,
    replace=False)` by a partial Fisher-Yates, each marked center takes its
    picked batch row (row `rows[t]` of `src`) and the smallest kept count."""
    var pool = List[Int](capacity=batch)
    for t in range(batch):
        pool.append(t)
    var picked = List[Int](capacity=nre)
    for q in range(nre):
        var r = q + rng.below(batch - q)
        var tmp = pool[q]
        pool[q] = pool[r]
        pool[r] = tmp
        picked.append(pool[q])
    var q = 0
    var wmin = Float32(0)
    var have = False
    for j in range(k):
        if not to[j] and (not have or w[j] < wmin):
            wmin = w[j]
            have = True
    for j in range(k):
        if to[j]:
            var r0 = rows[picked[q]] * d
            for f in range(d):
                c_new[j * d + f] = src[r0 + f]
            q += 1
            w[j] = wmin


def minibatch_step[O: ClusterOps](
    mut ops: O, bx: List[Float32], bw: List[Float32], batch: Int, k: Int, d: Int,
    c: List[Float32], mut w: List[Float32], mut rng: SplitMix64, reassign: Bool, ratio: Float64,
    bslot: Int, cslot: Int, lslot: Int, dslot: Int, mut c_new: List[Float32],
) raises -> Float64:
    """sklearn `_mini_batch_step` on the batch `bx` (weights `bw`, empty:
    unit): the device assignment, `update_center_dense` per center, then the
    random reassignment of low-count centers when `reassign`. Returns the
    batch inertia (unweighted, as the fit's early stop reads it); `c_new` is
    the updated centers, `w` the counts, updated in place."""
    var bweighted = len(bw) > 0
    ops.set(bslot, bx)
    ops.set(cslot, c)
    ops.nearest(bslot, batch, cslot, k, d, lslot, dslot)
    var bl = List[Int32]()
    var bd = List[Float32]()
    ops.get_if(lslot, batch, dslot, batch, bl, bd)
    var batch_inertia = sum_f64(bd, batch)
    # update_center_dense, per center
    c_new = c.copy()
    var one_pass = False
    comptime if MINIBATCH_ONE_PASS:
        one_pass = True
        for t in range(batch):
            if Int(bl[t]) < 0 or Int(bl[t]) >= k:
                one_pass = False
    if one_pass:
        var ws = List[Float32](length=k, fill=Float32(0))
        for t in range(batch):
            var j = Int(bl[t])
            ws[j] = ftz(ws[j] + (bw[t] if bweighted else Float32(1)))
        for j in range(k):
            if ws[j] > Float32(0):
                for f in range(d):
                    c_new[j * d + f] = ftz(identical_mul(c[j * d + f], w[j]))
        for t in range(batch):
            var j = Int(bl[t])
            if ws[j] > Float32(0):
                for f in range(d):
                    if bweighted:
                        c_new[j * d + f] = ftz(c_new[j * d + f] + ftz(identical_mul(ftz(bx[t * d + f]), bw[t])))
                    else:
                        c_new[j * d + f] = ftz(c_new[j * d + f] + ftz(bx[t * d + f]))
        for j in range(k):
            if ws[j] > Float32(0):
                w[j] = ftz(w[j] + ws[j])
                var alpha = ftz(identical_div(Float32(1), w[j]))
                for f in range(d):
                    c_new[j * d + f] = ftz(identical_mul(c_new[j * d + f], alpha))
    for j in range(0 if not one_pass else k, k):
        var wsum = Float32(0)
        for t in range(batch):
            if Int(bl[t]) == j:
                wsum = ftz(wsum + (bw[t] if bweighted else Float32(1)))
        if wsum > Float32(0):
            for f in range(d):
                c_new[j * d + f] = ftz(identical_mul(c[j * d + f], w[j]))
            comptime if C37_ROW_PANELS:
                for f in range(d):
                    var acc=c_new[j*d+f]
                    for first in range(0,batch,C37_PANEL_ROWS):
                        var part=Float32(0)
                        for t in range(first,min(first+C37_PANEL_ROWS,batch)):
                            if Int(bl[t])==j:
                                var value=ftz(bx[t*d+f])
                                if bweighted:
                                    value=ftz(identical_mul(value,bw[t]))
                                part=ftz(part+value)
                        acc=ftz(acc+part)
                    c_new[j*d+f]=acc
            else:
                for t in range(batch):
                    if Int(bl[t]) == j:
                        for f in range(d):
                            if bweighted:
                                c_new[j * d + f] = ftz(c_new[j * d + f] + ftz(identical_mul(ftz(bx[t * d + f]), bw[t])))
                            else:
                                c_new[j * d + f] = ftz(c_new[j * d + f] + ftz(bx[t * d + f]))
            w[j] = ftz(w[j] + wsum)
            var alpha = ftz(identical_div(Float32(1), w[j]))
            for f in range(d):
                c_new[j * d + f] = ftz(identical_mul(c_new[j * d + f], alpha))
    if reassign and ratio > 0:
        var to = mb_reassign_marks(w, k, batch, ratio)
        var nre = 0
        for j in range(k):
            if to[j]:
                nre += 1
        if nre > 0:
            var rows = List[Int](capacity=batch)
            for t in range(batch):
                rows.append(t)
            mb_reassign_apply(c_new, w, to, nre, bx, rows, batch, k, d, rng)
    return batch_inertia


def minibatch_partial[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, k: Int, weights: List[Float32],
    first: Bool, init_mode: Int, init_size: Int, batch_eff: Int, ratio: Float64,
    mut rng: SplitMix64, mut since_reassign: Int,
    mut centers: List[Float32], mut counts: List[Float32], mut labels: List[Int32],
) raises -> Float64:
    """sklearn `MiniBatchKMeans.partial_fit` (cluster/_kmeans.py): on the
    first call the centers start from `_init_centroids` on at most
    `init_size` drawn rows (init_mode 0 k-means++, 1 random, 2 the array
    already in `centers`) and the counts at zero; then ONE `_mini_batch_step`
    on all of `x` with its sample weights, the reassignment counter advanced
    by the fit's batch size; then the labels and the weighted inertia of `x`.
    The stream `rng` and the counter carry over between calls."""
    from checks.numerics import identical_mul64

    var weighted = len(weights) > 0
    if first:
        if k < 1 or k > n:
            raise Error("MiniBatchKMeans: n_samples=" + String(n) + " should be >= n_clusters=" + String(k))
        if init_mode != 2:
            var m = init_size if init_size < n else n
            var iidx = List[Int](capacity=m)
            if m < n:
                for _t in range(m):
                    iidx.append(rng.below(n))
            else:
                for t in range(n):
                    iidx.append(t)
            var wi = List[Float32]()
            if weighted:
                for t in iidx:
                    wi.append(weights[t])
            if init_mode == 1:
                var taken = List[Bool](length=m, fill=False)
                var picks = List[Int]()
                for _c in range(k):
                    var cum = List[Float64](capacity=m)
                    var acc = Float64(0)
                    for t in range(m):
                        if not taken[t]:
                            acc = acc + (Float64(wi[t]) if weighted else Float64(1))
                        cum.append(acc)
                    if not (acc > 0):
                        raise Error("MiniBatchKMeans: fewer positive-weight rows than n_clusters in the init sample")
                    var r = weighted_draw(cum, rng)
                    taken[r] = True
                    picks.append(iidx[r])
                centers = gather_rows(x, d, picks)
            else:
                centers = greedy_kmeans_pp(ops, gather_rows(x, d, iidx), m, d, k, rng, wi)
        counts = List[Float32](length=k, fill=Float32(0))
        since_reassign = 0
    # _random_reassign()
    since_reassign += batch_eff
    var any_empty = False
    for j in range(k):
        if counts[j] == Float32(0):
            any_empty = True
    var reassign = False
    if any_empty or since_reassign >= 10 * k:
        since_reassign = 0
        reassign = True
    var bslot = ops.zeros(n * d)
    var cslot = ops.zeros(k * d)
    var lslot = ops.zeros_i(n)
    var dslot = ops.zeros(n)
    var c_new = List[Float32]()
    _ = minibatch_step(ops, x, weights, n, k, d, centers, counts, rng, reassign, ratio, bslot, cslot, lslot, dslot, c_new)
    centers = c_new^
    var xs = ops.put(x)
    var dist = List[Float32]()
    nearest_all(ops, xs, n, centers, k, d, labels, dist)
    var inertia = Float64(0)
    for t in range(n):
        if weighted:
            inertia = inertia + identical_mul64(Float64(weights[t]), Float64(dist[t]))
        else:
            inertia = inertia + Float64(dist[t])
    return inertia
