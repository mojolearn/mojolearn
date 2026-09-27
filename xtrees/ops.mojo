# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TREES LANE'S ENSEMBLE GLUE: the arithmetic between tree fits.

Pass 1 of the algorithm expansion (2026-09-27). Every tree is fitted through
the existing forest entry points (`_mojolearn_rf`, `_mojolearn_gbdt`); what
lives here is what an ensemble does between two fits -- draw rows, gather
them, reweight them, vote -- as HOST code with one fixed order:

  * every reduction is sequential in index order (no tree, no threads);
  * every product that meets an add is `identical_mul64` (the builds contract
    `a * b + c` into an FMA otherwise; checks/numerics.mojo);
  * exp / log / pow are the pinned binary64 polynomials (`identical_exp64`,
    `identical_log64`, `identical_pow64`), not the platform libm;
  * every tie is broken by the lower index (argmax: first max; the weighted
    median: sort by (value, estimator index));
  * the RNG is SplitMix64 as a counter (`draw(base, k)`), so a draw depends on
    (seed, stream, k) and nothing else.

The GPU binding and the host binding import these same functions, so the CPU
column and every GPU column run one spelling. Moving them onto the device is
pass 2 (speed).

THE SEAMS (IDENTITY_PATHS rows 160-165; the gate is
xtrees/checks/glue_check.mojo, one sabotage arm each):
  DEVIATION 5600  draws: SplitMix64 as a counter, index = draw mod n.
  DEVIATION 5601  products meeting an add: `identical_mul64`.
  DEVIATION 5602  folds: sequential in index order.
  DEVIATION 5603  exp / log / pow: the pinned binary64 polynomials.
  DEVIATION 5604  ties: the lower index; stable sorts.
  DEVIATION 5605  a zero row normalises to uniform 1 / k, never 0 / 0.
"""
from checks.numerics import identical_mul64, identical_exp64, identical_log64, identical_pow64

comptime GOLDEN: UInt64 = 0x9E3779B97F4A7C15


@always_inline
def mix64(z_in: UInt64) -> UInt64:
    """SplitMix64's finalizer (Steele, Lea, Flood 2014)."""
    var z = z_in
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def stream_base(seed: Int, stream: Int) -> UInt64:
    return mix64(mix64(UInt64(seed) + GOLDEN) + UInt64(stream) * GOLDEN + 1)


@always_inline
def draw(base: UInt64, k: Int) -> UInt64:
    return mix64(base + UInt64(k + 1) * GOLDEN)


@always_inline
def unit(r: UInt64) -> Float64:
    """[0, 1) from the top 53 bits, exact."""
    return Float64(r >> 11) * (1.0 / 9007199254740992.0)


def sample_indices(
    res: MutPointer[Int32, MutUntrackedOrigin], n_pool: Int, n_draw: Int, replace: Bool,
    seed: Int, stream: Int,
) raises:
    """`n_draw` row indices from [0, n_pool): with replacement, draw k is
    `draw % n_pool`; without, a partial Fisher-Yates shuffle whose position k
    swaps with k + draw % (n_pool - k)."""
    if n_pool <= 0 or n_draw < 0 or (not replace and n_draw > n_pool):
        raise Error("x_trees sample_indices: need n_pool > 0 and 0 <= n_draw (<= n_pool without replacement)")
    var base = stream_base(seed, stream)
    if replace:
        for k in range(n_draw):
            # DEVIATION 5600: the index is the counter draw mod n.
            res[unsafe_offset=k] = Int32(Int(draw(base, k) % UInt64(n_pool)))
        return
    var perm = List[Int32](capacity=n_pool)
    for i in range(n_pool):
        perm.append(Int32(i))
    for k in range(n_draw):
        var j = k + Int(draw(base, k) % UInt64(n_pool - k))
        var t = perm[k]
        perm[k] = perm[j]
        perm[j] = t
        res[unsafe_offset=k] = perm[k]


def weighted_sample(
    w: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    res: MutPointer[Int32, MutUntrackedOrigin], n_draw: Int, seed: Int, stream: Int,
) raises:
    """`n_draw` indices drawn with replacement with probability w[i] / sum(w)
    (numpy's `choice(p=...)` question): the cumulative sum in index order,
    then the first i with cdf[i] > u * total. A zero-weight row is never
    drawn."""
    var cdf = List[Float64](length=n, fill=0.0)
    var total: Float64 = 0.0
    for i in range(n):
        if not (w[unsafe_offset=i] >= 0.0):
            raise Error("x_trees weighted_sample: weights must be nonnegative")
        total = total + w[unsafe_offset=i]
        cdf[i] = total
    if not (total > 0.0):
        raise Error("x_trees weighted_sample: weights must have a positive total")
    var base = stream_base(seed, stream)
    for k in range(n_draw):
        var u = identical_mul64(unit(draw(base, k)), total)
        var lo = 0
        var hi = n - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cdf[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        while lo > 0 and w[unsafe_offset=lo] == 0.0:  # u landed on a flat step at the end
            lo -= 1
        res[unsafe_offset=k] = Int32(lo)


def gather_f32(
    src: MutPointer[Float32, MutUntrackedOrigin], n_src_rows: Int, n_src_cols: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], n_rows: Int,
    cols: MutPointer[Int32, MutUntrackedOrigin], n_cols: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """dst[r, c] = src[rows[r], cols[c]], both row-major. A copy, no arithmetic."""
    for r in range(n_rows):
        var i = Int(rows[unsafe_offset=r])
        if i < 0 or i >= n_src_rows:
            raise Error("x_trees gather: row index out of range")
        for c in range(n_cols):
            var j = Int(cols[unsafe_offset=c])
            if j < 0 or j >= n_src_cols:
                raise Error("x_trees gather: column index out of range")
            dst[unsafe_offset=r * n_cols + c] = src[unsafe_offset=i * n_src_cols + j]


def gather_i32(
    src: MutPointer[Int32, MutUntrackedOrigin], n_src: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], n_rows: Int,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    for r in range(n_rows):
        var i = Int(rows[unsafe_offset=r])
        if i < 0 or i >= n_src:
            raise Error("x_trees gather: row index out of range")
        dst[unsafe_offset=r] = src[unsafe_offset=i]


def accumulate(
    acc: MutPointer[Float64, MutUntrackedOrigin], x: MutPointer[Float32, MutUntrackedOrigin],
    n: Int, weight: Float64,
):
    """acc[i] += weight * x[i], the product pinned (never fused into the add)."""
    for i in range(n):
        # DEVIATION 5601: the product is pinned, never fused into the add.
        acc[unsafe_offset=i] = acc[unsafe_offset=i] + identical_mul64(weight, Float64(x[unsafe_offset=i]))


def accumulate_cols(
    acc: MutPointer[Float64, MutUntrackedOrigin], x: MutPointer[Float32, MutUntrackedOrigin],
    cols: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int, ks: Int, weight: Float64,
) raises:
    """acc[i, cols[c]] += weight * x[i, c]: a sub-model's ks columns added into
    the ensemble's k (an estimator that saw only some classes)."""
    for c in range(ks):
        var j = Int(cols[unsafe_offset=c])
        if j < 0 or j >= k:
            raise Error("x_trees accumulate_cols: column out of range")
    for i in range(n):
        for c in range(ks):
            var j = i * k + Int(cols[unsafe_offset=c])
            acc[unsafe_offset=j] = acc[unsafe_offset=j] + identical_mul64(weight, Float64(x[unsafe_offset=i * ks + c]))


def accumulate_onehot(
    acc: MutPointer[Float64, MutUntrackedOrigin], codes: MutPointer[Int32, MutUntrackedOrigin],
    n: Int, k: Int, on: Float64, off: Float64,
) raises:
    """acc[i, c] += on if codes[i] == c else off (a weighted vote)."""
    for i in range(n):
        var code = Int(codes[unsafe_offset=i])
        if code < 0 or code >= k:
            raise Error("x_trees accumulate_onehot: code out of range")
        for c in range(k):
            acc[unsafe_offset=i * k + c] = acc[unsafe_offset=i * k + c] + (on if c == code else off)


def argmax_rows(
    x: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int,
    res: MutPointer[Int32, MutUntrackedOrigin],
):
    """First maximum of each row (lower index wins a tie)."""
    for i in range(n):
        var best = 0
        # DEVIATION 5604: strict >, so a tie keeps the lower index.
        for c in range(1, k):
            if x[unsafe_offset=i * k + c] > x[unsafe_offset=i * k + best]:
                best = c
        res[unsafe_offset=i] = Int32(best)


def argmax_rows_f32(
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, k: Int,
    res: MutPointer[Int32, MutUntrackedOrigin],
):
    for i in range(n):
        var best = 0
        for c in range(1, k):
            if x[unsafe_offset=i * k + c] > x[unsafe_offset=i * k + best]:
                best = c
        res[unsafe_offset=i] = Int32(best)


def scale_f64(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, divisor: Float64):
    for i in range(n):
        x[unsafe_offset=i] = x[unsafe_offset=i] / divisor


def scale_to_f32(
    x: MutPointer[Float64, MutUntrackedOrigin], n: Int, factor: Float64,
    dst: MutPointer[Float32, MutUntrackedOrigin],
):
    """dst[i] = Float32(x[i] * factor): one binary64 product, one rounding."""
    for i in range(n):
        dst[unsafe_offset=i] = Float32(identical_mul64(x[unsafe_offset=i], factor))


def put_f32(
    dst: MutPointer[Float32, MutUntrackedOrigin], offset: Int,
    src: MutPointer[Float32, MutUntrackedOrigin], n: Int,
):
    """dst[offset + i] = src[i]: a copy."""
    for i in range(n):
        dst[unsafe_offset=offset + i] = src[unsafe_offset=i]


def softmax_rows(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int):
    """In place: exp(x - max) / sum, the sum in class order."""
    for i in range(n):
        var m = x[unsafe_offset=i * k]
        for c in range(1, k):
            if x[unsafe_offset=i * k + c] > m:
                m = x[unsafe_offset=i * k + c]
        var s: Float64 = 0.0
        for c in range(k):
            var e = identical_exp64(x[unsafe_offset=i * k + c] - m)
            x[unsafe_offset=i * k + c] = e
            s = s + e
        for c in range(k):
            x[unsafe_offset=i * k + c] = x[unsafe_offset=i * k + c] / s


# ------------------------------------------------------------- AdaBoost
def samme_step(
    w: MutPointer[Float64, MutUntrackedOrigin], pred: MutPointer[Int32, MutUntrackedOrigin],
    y: MutPointer[Int32, MutUntrackedOrigin], n: Int, n_classes: Int,
    learning_rate: Float64, last: Bool, stats: MutPointer[Float64, MutUntrackedOrigin],
):
    """sklearn `AdaBoostClassifier._boost_discrete` (SAMME,
    ensemble/_weight_boosting.py). `w` sums to one on entry. Writes
    stats = [status, estimator_weight, estimator_error, sum of new w]:
    status 0 continue, 1 perfect fit (weight 1, stop), 2 worse than chance
    (discard, stop). Updates `w` in place (not on the last iteration)."""
    var err: Float64 = 0.0
    var tot: Float64 = 0.0
    for i in range(n):
        tot = tot + w[unsafe_offset=i]
        if pred[unsafe_offset=i] != y[unsafe_offset=i]:
            err = err + w[unsafe_offset=i]
    err = err / tot
    if err <= 0.0:
        stats[unsafe_offset=0] = 1.0
        stats[unsafe_offset=1] = 1.0
        stats[unsafe_offset=2] = 0.0
        stats[unsafe_offset=3] = tot
        return
    var k = Float64(n_classes)
    if err >= 1.0 - 1.0 / k:
        stats[unsafe_offset=0] = 2.0
        stats[unsafe_offset=1] = 0.0
        stats[unsafe_offset=2] = err
        stats[unsafe_offset=3] = tot
        return
    var alpha = identical_mul64(
        learning_rate, identical_log64((1.0 - err) / err) + identical_log64(k - 1.0))
    var s: Float64 = 0.0
    if not last:
        # DEVIATION 5603: the pinned exp.
        var boost = identical_exp64(alpha)
        for i in range(n):
            if pred[unsafe_offset=i] != y[unsafe_offset=i] and w[unsafe_offset=i] > 0.0:
                w[unsafe_offset=i] = identical_mul64(w[unsafe_offset=i], boost)
            s = s + w[unsafe_offset=i]
    else:
        s = tot
    stats[unsafe_offset=0] = 0.0
    stats[unsafe_offset=1] = alpha
    stats[unsafe_offset=2] = err
    stats[unsafe_offset=3] = s


def r2_step(
    w: MutPointer[Float64, MutUntrackedOrigin], pred: MutPointer[Float32, MutUntrackedOrigin],
    y: MutPointer[Float32, MutUntrackedOrigin], n: Int, loss: Int,
    learning_rate: Float64, last: Bool, stats: MutPointer[Float64, MutUntrackedOrigin],
):
    """sklearn `AdaBoostRegressor._boost` (AdaBoost.R2, Drucker 1997).
    loss 0 linear, 1 square, 2 exponential. stats as `samme_step`'s, status 2
    meaning estimator_error >= 0.5."""
    var emax: Float64 = 0.0
    for i in range(n):
        if w[unsafe_offset=i] > 0.0:
            var e = abs(Float64(pred[unsafe_offset=i]) - Float64(y[unsafe_offset=i]))
            if e > emax:
                emax = e
    var err: Float64 = 0.0
    var ev = List[Float64](length=n, fill=0.0)
    for i in range(n):
        if w[unsafe_offset=i] > 0.0:
            var e = abs(Float64(pred[unsafe_offset=i]) - Float64(y[unsafe_offset=i]))
            if emax != 0.0:
                e = e / emax
            if loss == 1:
                e = identical_mul64(e, e)
            elif loss == 2:
                e = 1.0 - identical_exp64(-e)
            ev[i] = e
            err = err + identical_mul64(w[unsafe_offset=i], e)
    if err <= 0.0:
        stats[unsafe_offset=0] = 1.0
        stats[unsafe_offset=1] = 1.0
        stats[unsafe_offset=2] = 0.0
        return
    if err >= 0.5:
        stats[unsafe_offset=0] = 2.0
        stats[unsafe_offset=1] = 0.0
        stats[unsafe_offset=2] = err
        return
    var beta = err / (1.0 - err)
    var alpha = identical_mul64(learning_rate, identical_log64(1.0 / beta))
    var s: Float64 = 0.0
    for i in range(n):
        if not last and w[unsafe_offset=i] > 0.0:
            w[unsafe_offset=i] = identical_mul64(w[unsafe_offset=i], identical_pow64(beta, identical_mul64(1.0 - ev[i], learning_rate)))
        s = s + w[unsafe_offset=i]
    stats[unsafe_offset=0] = 0.0
    stats[unsafe_offset=1] = alpha
    stats[unsafe_offset=2] = err
    stats[unsafe_offset=3] = s


def weighted_median(
    preds: MutPointer[Float32, MutUntrackedOrigin], weights: MutPointer[Float64, MutUntrackedOrigin],
    n: Int, m: Int, res: MutPointer[Float32, MutUntrackedOrigin],
):
    """sklearn `AdaBoostRegressor._get_median_predict`: per row, the
    estimators sorted by prediction (ties by estimator index), the first
    whose cumulative weight reaches half the total. `preds` is estimator
    major (m rows of n)."""
    var order = List[Int](length=m, fill=0)
    for i in range(n):
        for j in range(m):
            order[j] = j
        # insertion sort, stable: m is the estimator count
        for a in range(1, m):
            var key = order[a]
            var kv = preds[unsafe_offset=key * n + i]
            var b = a - 1
            while b >= 0 and preds[unsafe_offset=order[b] * n + i] > kv:
                order[b + 1] = order[b]
                b -= 1
            order[b + 1] = key
        var total: Float64 = 0.0
        for j in range(m):
            total = total + weights[unsafe_offset=order[j]]
        var half = identical_mul64(0.5, total)
        var c: Float64 = 0.0
        var pick = order[m - 1]
        for j in range(m):
            c = c + weights[unsafe_offset=order[j]]
            if c >= half:
                pick = order[j]
                break
        res[unsafe_offset=i] = preds[unsafe_offset=pick * n + i]


# ------------------------------------------------------ flat trees (apply)
# The forest arrays `_forest_out` exports (ensemble/flatnode.mojo): per tree
# t the nodes offsets[t] .. offsets[t+1]; a node is a leaf iff left == -1;
# its children are left and left + 1 (tree-relative); `x[colid] <= quesval`
# goes LEFT (decisiontree.cuh:379, equality left).
def apply_trees(
    offsets: MutPointer[Int32, MutUntrackedOrigin], colid: MutPointer[Int32, MutUntrackedOrigin],
    quesval: MutPointer[Float32, MutUntrackedOrigin], left: MutPointer[Int32, MutUntrackedOrigin],
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, t0: Int, t1: Int,
    res: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """res[i * (t1 - t0) + (t - t0)] = the tree-relative leaf node row i
    reaches in tree t."""
    var nt = t1 - t0
    for t in range(t0, t1):
        var lo = Int(offsets[unsafe_offset=t])
        var count = Int(offsets[unsafe_offset=t + 1]) - lo
        if count < 1:
            raise Error("x_trees apply: empty tree")
        for i in range(n):
            var node = 0
            var steps = 0
            while left[unsafe_offset=lo + node] != -1:
                var c = Int(colid[unsafe_offset=lo + node])
                if c < 0 or c >= d:
                    raise Error("x_trees apply: split column out of range")
                var l = Int(left[unsafe_offset=lo + node])
                if l < 1 or l + 1 >= count:
                    raise Error("x_trees apply: child out of range")
                if x[unsafe_offset=i * d + c] <= quesval[unsafe_offset=lo + node]:
                    node = l
                else:
                    node = l + 1
                steps += 1
                if steps > count:
                    raise Error("x_trees apply: cycle in tree")
            res[unsafe_offset=i * nt + (t - t0)] = Int32(node)


def gradients(
    score: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Float32, MutUntrackedOrigin], n: Int,
    kind: Int, g: MutPointer[Float64, MutUntrackedOrigin], h: MutPointer[Float64, MutUntrackedOrigin],
    target: MutPointer[Float32, MutUntrackedOrigin],
):
    """LightGBM's objective gradients (regression_objective.hpp L2: g = s - y,
    h = 1; binary_objective.hpp with sigmoid 1: p = 1 / (1 + exp(-s)),
    g = p - y, h = p (1 - p)). target = -g as float32, the tree's fit target."""
    for i in range(n):
        var s = score[unsafe_offset=i]
        var yi = Float64(y[unsafe_offset=i])
        var gi: Float64
        var hi: Float64
        if kind == 0:
            gi = s - yi
            hi = 1.0
        else:
            var p = 1.0 / (1.0 + identical_exp64(-s))
            gi = p - yi
            hi = identical_mul64(p, 1.0 - p)
        g[unsafe_offset=i] = gi
        h[unsafe_offset=i] = hi
        target[unsafe_offset=i] = Float32(-gi)


def leaf_newton(
    nodes: MutPointer[Int32, MutUntrackedOrigin], g: MutPointer[Float64, MutUntrackedOrigin],
    h: MutPointer[Float64, MutUntrackedOrigin], n: Int, n_nodes: Int, reg_lambda: Float64,
    values: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """values[node] = -sum(g) / (sum(h) + lambda) over the rows in that leaf,
    sums in row order (LightGBM `CalculateSplittedLeafOutput` with no L1, no
    max_delta_step); 0 for a node no row reaches."""
    var sg = List[Float64](length=n_nodes, fill=0.0)
    var sh = List[Float64](length=n_nodes, fill=0.0)
    for i in range(n):
        var k = Int(nodes[unsafe_offset=i])
        if k < 0 or k >= n_nodes:
            raise Error("x_trees leaf_newton: node out of range")
        sg[k] = sg[k] + g[unsafe_offset=i]
        sh[k] = sh[k] + h[unsafe_offset=i]
    for k in range(n_nodes):
        var den = sh[k] + reg_lambda
        values[unsafe_offset=k] = Float32(-sg[k] / den) if den > 0.0 else Float32(0.0)


def tree_score_add(
    nodes: MutPointer[Int32, MutUntrackedOrigin], values: MutPointer[Float32, MutUntrackedOrigin],
    n: Int, weight: Float64, acc: MutPointer[Float64, MutUntrackedOrigin],
):
    """acc[i] += weight * values[nodes[i]], the product pinned."""
    for i in range(n):
        acc[unsafe_offset=i] = acc[unsafe_offset=i] + identical_mul64(
            weight, Float64(values[unsafe_offset=Int(nodes[unsafe_offset=i])]))


def uniform(res: MutPointer[Float64, MutUntrackedOrigin], n: Int, seed: Int, stream: Int):
    """n draws in [0, 1) from the counter RNG."""
    var base = stream_base(seed, stream)
    for k in range(n):
        res[unsafe_offset=k] = unit(draw(base, k))


def onehot_leaves(
    nodes: MutPointer[Int32, MutUntrackedOrigin], tree_base: MutPointer[Int32, MutUntrackedOrigin],
    node_col: MutPointer[Int32, MutUntrackedOrigin], n: Int, n_trees: Int, n_cols: Int,
    res: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """res[i, node_col[tree_base[t] + nodes[i, t]]] = 1 (res zeroed by the
    caller): the one-hot leaf embedding."""
    for i in range(n):
        for t in range(n_trees):
            var c = Int(node_col[unsafe_offset=Int(tree_base[unsafe_offset=t]) + Int(nodes[unsafe_offset=i * n_trees + t])])
            if c < 0 or c >= n_cols:
                raise Error("x_trees onehot_leaves: a row reached a node that is not a leaf column")
            res[unsafe_offset=i * n_cols + c] = 1.0


def transpose_f32(
    src: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, dst: MutPointer[Float32, MutUntrackedOrigin],
):
    """dst[j, i] = src[i, j]: a row-major n x d copied to column-major."""
    for i in range(n):
        for j in range(d):
            dst[unsafe_offset=j * n + i] = src[unsafe_offset=i * d + j]


# ------------------------------------------------------ wrappers' helpers
def normalize_rows(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int):
    """x[i, :] /= sum (class order); a row summing to 0 becomes uniform 1/k
    (sklearn's calibration rule; OneVsRest would divide 0/0, refused here
    as a computed NaN)."""
    for i in range(n):
        var s: Float64 = 0.0
        for c in range(k):
            s = s + x[unsafe_offset=i * k + c]
        for c in range(k):
            # DEVIATION 5605: a zero row is uniform, never 0 / 0.
            if s > 0.0:
                x[unsafe_offset=i * k + c] = x[unsafe_offset=i * k + c] / s
            else:
                x[unsafe_offset=i * k + c] = 1.0 / Float64(k)


def scatter(
    dst: MutPointer[Float64, MutUntrackedOrigin], n_dst_cols: Int,
    src: MutPointer[Float32, MutUntrackedOrigin], m: Int, c: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], col0: Int,
):
    """dst[rows[r], col0 + j] = src[r, j]: a sub-model's output block placed
    into the stacked matrix. A copy."""
    for r in range(m):
        var i = Int(rows[unsafe_offset=r])
        for j in range(c):
            dst[unsafe_offset=i * n_dst_cols + col0 + j] = Float64(src[unsafe_offset=r * c + j])


@always_inline
def _log1pexp(x: Float64) -> Float64:
    """log(1 + exp(x)) without overflow: x + log(1 + exp(-x)) for x >= 0."""
    if x >= 0.0:
        return x + identical_log64(1.0 + identical_exp64(-x))
    return identical_log64(1.0 + identical_exp64(x))


def _platt_value(f: MutPointer[Float64, MutUntrackedOrigin], t: List[Float64], n: Int, a: Float64, b: Float64) -> Float64:
    var v: Float64 = 0.0
    for i in range(n):
        var z = identical_mul64(f[unsafe_offset=i], a) + b
        # -T log p - (1 - T) log(1 - p), p = 1 / (1 + exp(z))
        v = v + identical_mul64(t[i], _log1pexp(z)) + identical_mul64(1.0 - t[i], _log1pexp(-z))
    return v


def platt_fit(
    f: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Int32, MutUntrackedOrigin], n: Int,
    ab: MutPointer[Float64, MutUntrackedOrigin],
):
    """Platt scaling (sklearn calibration.py `_sigmoid_calibration`: the
    targets T = (N+ + 1)/(N+ + 2) and 1/(N- + 2), the start B =
    log((N- + 1)/(N+ + 1))), minimised by Newton with backtracking (Lin, Lin,
    Weng 2007) where sklearn runs L-BFGS on the same objective. Writes
    ab = [A, B]; P(y=1 | f) = 1 / (1 + exp(A f + B))."""
    var prior1: Float64 = 0.0
    for i in range(n):
        if y[unsafe_offset=i] > 0:
            prior1 = prior1 + 1.0
    var prior0 = Float64(n) - prior1
    var hi = (prior1 + 1.0) / (prior1 + 2.0)
    var lo = 1.0 / (prior0 + 2.0)
    var t = List[Float64](length=n, fill=0.0)
    for i in range(n):
        t[i] = hi if y[unsafe_offset=i] > 0 else lo
    var a: Float64 = 0.0
    var b = identical_log64((prior0 + 1.0) / (prior1 + 1.0))
    var fval = _platt_value(f, t, n, a, b)
    for _ in range(100):
        var h11: Float64 = 1e-12
        var h22: Float64 = 1e-12
        var h21: Float64 = 0.0
        var g1: Float64 = 0.0
        var g2: Float64 = 0.0
        for i in range(n):
            var fi = f[unsafe_offset=i]
            var z = identical_mul64(fi, a) + b
            var p: Float64
            var q: Float64
            if z >= 0.0:
                var e = identical_exp64(-z)
                p = e / (1.0 + e)
                q = 1.0 / (1.0 + e)
            else:
                var e = identical_exp64(z)
                p = 1.0 / (1.0 + e)
                q = e / (1.0 + e)
            var d2 = identical_mul64(p, q)
            h11 = h11 + identical_mul64(identical_mul64(fi, fi), d2)
            h22 = h22 + d2
            h21 = h21 + identical_mul64(fi, d2)
            var d1 = t[i] - p
            g1 = g1 + identical_mul64(fi, d1)
            g2 = g2 + d1
        if abs(g1) < 1e-5 and abs(g2) < 1e-5:
            break
        var det = identical_mul64(h11, h22) - identical_mul64(h21, h21)
        var da = -(identical_mul64(h22, g1) - identical_mul64(h21, g2)) / det
        var db = -(identical_mul64(h11, g2) - identical_mul64(h21, g1)) / det
        var gd = identical_mul64(g1, da) + identical_mul64(g2, db)
        var step: Float64 = 1.0
        var moved = False
        while step >= 1e-10:
            var na = a + identical_mul64(step, da)
            var nb = b + identical_mul64(step, db)
            var nf = _platt_value(f, t, n, na, nb)
            if nf < fval + identical_mul64(identical_mul64(0.0001, step), gd):
                a = na
                b = nb
                fval = nf
                moved = True
                break
            step = step / 2.0
        if not moved:
            break
    ab[unsafe_offset=0] = a
    ab[unsafe_offset=1] = b


def platt_apply(
    f: MutPointer[Float64, MutUntrackedOrigin], n: Int, a: Float64, b: Float64,
    res: MutPointer[Float64, MutUntrackedOrigin],
):
    """res[i] = 1 / (1 + exp(A f + B)), written without overflow."""
    for i in range(n):
        var z = identical_mul64(f[unsafe_offset=i], a) + b
        if z >= 0.0:
            var e = identical_exp64(-z)
            res[unsafe_offset=i] = e / (1.0 + e)
        else:
            res[unsafe_offset=i] = 1.0 / (1.0 + identical_exp64(z))


def isotonic_fit(
    x: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    kx: MutPointer[Float64, MutUntrackedOrigin], ky: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """sklearn IsotonicRegression(increasing=True).fit, unit weights: sort by
    (x, y, index), merge equal x into their weighted mean (`_make_unique`),
    pool adjacent violators (`_inplace_contiguous_isotonic_regression`), and
    write the knots. Returns the knot count."""
    if n < 1:
        raise Error("x_trees isotonic_fit: no rows")
    # stable merge sort of the row order by (x, y)
    var idx = List[Int](length=n, fill=0)
    var tmp = List[Int](length=n, fill=0)
    for i in range(n):
        idx[i] = i
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                var xi = x[unsafe_offset=idx[i]]
                var xj = x[unsafe_offset=idx[j]]
                var take_right = xj < xi or (xj == xi and y[unsafe_offset=idx[j]] < y[unsafe_offset=idx[i]])
                if take_right:
                    tmp[k] = idx[j]
                    j += 1
                else:
                    tmp[k] = idx[i]
                    i += 1
                k += 1
            while i < mid:
                tmp[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                tmp[k] = idx[j]
                j += 1
                k += 1
            lo = hi
        for q in range(n):
            idx[q] = tmp[q]
        width *= 2
    # unique x: mean y, weight = count
    var ux = List[Float64]()
    var uy = List[Float64]()
    var uw = List[Float64]()
    var r = 0
    while r < n:
        var xv = x[unsafe_offset=idx[r]]
        var s: Float64 = 0.0
        var c: Float64 = 0.0
        while r < n and x[unsafe_offset=idx[r]] == xv:
            s = s + y[unsafe_offset=idx[r]]
            c = c + 1.0
            r += 1
        ux.append(xv)
        uy.append(s / c)
        uw.append(c)
    # PAV: blocks of (sum w*y, sum w), merged while decreasing
    var m = len(ux)
    var bsum = List[Float64]()
    var bw = List[Float64]()
    var bstart = List[Int]()
    for q in range(m):
        bsum.append(identical_mul64(uw[q], uy[q]))
        bw.append(uw[q])
        bstart.append(q)
        while len(bsum) > 1 and bsum[len(bsum) - 2] / bw[len(bw) - 2] >= bsum[len(bsum) - 1] / bw[len(bw) - 1]:
            var s2 = bsum.pop()
            var w2 = bw.pop()
            _ = bstart.pop()
            bsum[len(bsum) - 1] = bsum[len(bsum) - 1] + s2
            bw[len(bw) - 1] = bw[len(bw) - 1] + w2
    var nb = len(bsum)
    for q in range(nb):
        var end = bstart[q + 1] if q + 1 < nb else m
        var v = bsum[q] / bw[q]
        for p in range(bstart[q], end):
            kx[unsafe_offset=p] = ux[p]
            ky[unsafe_offset=p] = v
    return m


def isotonic_predict(
    kx: MutPointer[Float64, MutUntrackedOrigin], ky: MutPointer[Float64, MutUntrackedOrigin], m: Int,
    t: MutPointer[Float64, MutUntrackedOrigin], n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
):
    """np.interp over the knots with out_of_bounds='clip'."""
    for i in range(n):
        var v = t[unsafe_offset=i]
        if m == 1 or v <= kx[unsafe_offset=0]:
            res[unsafe_offset=i] = ky[unsafe_offset=0]
            continue
        if v >= kx[unsafe_offset=m - 1]:
            res[unsafe_offset=i] = ky[unsafe_offset=m - 1]
            continue
        var lo = 0
        var hi = m - 1
        while hi - lo > 1:
            var mid = (lo + hi) // 2
            if kx[unsafe_offset=mid] <= v:
                lo = mid
            else:
                hi = mid
        var x0 = kx[unsafe_offset=lo]
        var y0 = ky[unsafe_offset=lo]
        var slope = (ky[unsafe_offset=hi] - y0) / (kx[unsafe_offset=hi] - x0)
        res[unsafe_offset=i] = identical_mul64(slope, v - x0) + y0
