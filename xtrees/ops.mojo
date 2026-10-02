# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TREES LANE'S ENSEMBLE GLUE: the arithmetic between tree fits.

Pass 1 of the algorithm expansion (2026-09-27). Every tree is fitted through
the existing forest entry points (`_mojolearn_rf`, `_mojolearn_gbdt`); what
lives here is what an ensemble does between two fits -- draw rows, gather
them, reweight them, vote -- as HOST code with one fixed order:

  * every reduction is sequential in index order (no tree, no threads);
    the only threaded bodies are the pure moves `transpose_f32` and
    `gather_f32` (DEVIATION 5606: one load and one store per cell, tasks own
    disjoint destination rows, no arithmetic, so no order can move a bit);
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
  DEVIATION 5602  folds: sequential in index order (weighted_sample's cdf:
                  sequential per WS_CHUNK, the chunks joined by a fixed
                  Hillis-Steele tree, the device's order; w2-trees).
  DEVIATION 5603  exp / log / pow: the pinned binary64 polynomials.
  DEVIATION 5604  ties: the lower index; stable sorts.
  DEVIATION 5605  a zero row normalises to uniform 1 / k, never 0 / 0.
"""
from std.sys.compile import is_defined
from checks.numerics import identical_mul64, identical_exp64, identical_log64, identical_pow64

#: The host gate's negative control (`-D MOJOLEARN_HOST_SABOTAGE=1`, host builds
#: only): `scale_f64` divides by a perturbed divisor, so every vote average moves.
comptime XTREES_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

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


#: `weighted_sample`'s scan chunk: each chunk's cumulative sum is sequential
#: in index order; the chunk totals are joined by `ws_tree_scan`'s fixed tree.
#: `xtrees/ops_device.mojo::weighted_sample_device` uses the same chunk, so
#: the CPU column and every GPU column build one cdf.
comptime WS_CHUNK = 256


def ws_tree_scan(mut tot: List[Float64]):
    """Inclusive scan of the chunk totals in place, by the fixed Hillis-Steele
    tree: pass s (s = 1, 2, 4, ...) sets t[j] = t_prev[j] + t_prev[j - s]
    for j >= s. The device runs the same passes, one launch each."""
    var m = len(tot)
    var s = 1
    while s < m:
        var prev = tot.copy()
        for j in range(s, m):
            tot[j] = prev[j] + prev[j - s]
        s *= 2


def weighted_sample(
    w: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    res: MutPointer[Int32, MutUntrackedOrigin], n_draw: Int, seed: Int, stream: Int,
) raises:
    """`n_draw` indices drawn with replacement with probability w[i] / sum(w)
    (numpy's `choice(p=...)` question): the cumulative sum, then the first i
    with cdf[i] > u * total. A zero-weight row is never drawn.

    THE CDF (cpu-gpu-cleanup w2-trees; the old one was one sequential sum):
    rows fall in chunks of `WS_CHUNK`; inside a chunk the sum is sequential
    in index order; the chunk totals are scanned by `ws_tree_scan`'s fixed
    tree; cdf[i] = scan[chunk - 1] + local[i] (chunk 0 adds nothing); the
    total is cdf[n - 1]. This is the CPU column's spelling of
    `ops_device.weighted_sample_device`, operation for operation."""
    for i in range(n):
        if not (w[unsafe_offset=i] >= 0.0):
            raise Error("x_trees weighted_sample: weights must be nonnegative")
    var n_chunks = (n + WS_CHUNK - 1) // WS_CHUNK
    var cdf = List[Float64](length=n, fill=0.0)
    var tot = List[Float64](length=n_chunks, fill=0.0)
    for c in range(n_chunks):
        var run: Float64 = 0.0
        for i in range(c * WS_CHUNK, min((c + 1) * WS_CHUNK, n)):
            run = run + w[unsafe_offset=i]
            cdf[i] = run
        tot[c] = run
    ws_tree_scan(tot)
    for i in range(WS_CHUNK, n):
        cdf[i] = tot[i // WS_CHUNK - 1] + cdf[i]
    var total = cdf[n - 1]
    if not (total > 0.0):
        raise Error("x_trees weighted_sample: weights must have a positive total")
    var base = stream_base(seed, stream)
    var cp = cdf.unsafe_ptr()
    for k in range(n_draw):
        # Draw k is a pure function of (base, k, cdf, w); the device runs one
        # thread per draw with this body (`ws_draw_kernel`).
        var u = identical_mul64(unit(draw(base, k)), total)
        var lo = 0
        var hi = n - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cp[unsafe_offset=mid] > u:
                hi = mid
            else:
                lo = mid + 1
        while lo > 0 and w[unsafe_offset=lo] == 0.0:  # u landed on a flat step at the end
            lo -= 1
        res[unsafe_offset=k] = Int32(lo)
    _ = cdf^


def gather_f32(
    src: MutPointer[Float32, MutUntrackedOrigin], n_src_rows: Int, n_src_cols: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], n_rows: Int,
    cols: MutPointer[Int32, MutUntrackedOrigin], n_cols: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """dst[r, c] = src[rows[r], cols[c]], both row-major. A copy, no arithmetic.

    The CPU column's loop. Every index is range-checked first, so a bad
    index raises before any cell is written. A GPU install runs
    `xtrees/ops_device.mojo::gather_f32_device` (cpu-gpu-cleanup t-gbdt;
    DEVIATION 5606's host-pool copy is gone)."""
    for r in range(n_rows):
        var i = Int(rows[unsafe_offset=r])
        if i < 0 or i >= n_src_rows:
            raise Error("x_trees gather: row index out of range")
    for c in range(n_cols):
        var j = Int(cols[unsafe_offset=c])
        if j < 0 or j >= n_src_cols:
            raise Error("x_trees gather: column index out of range")
    for r in range(n_rows):
        var i = Int(rows[unsafe_offset=r])
        for c in range(n_cols):
            dst.unsafe_store(r * n_cols + c, src.unsafe_load(i * n_src_cols + Int(cols[unsafe_offset=c])))


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


def accumulate_rows(
    acc: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int, x: MutPointer[Float64, MutUntrackedOrigin],
    rows: MutPointer[Int32, MutUntrackedOrigin], m: Int,
) raises:
    """acc[rows[r], c] += x[r, c] for r in order: a sub-model's output on a
    row subset (its out-of-bag rows) added into the ensemble's n x k sums."""
    for r in range(m):
        var i = Int(rows[unsafe_offset=r])
        if i < 0 or i >= n:
            raise Error("x_trees accumulate_rows: row out of range")
        for c in range(k):
            acc[unsafe_offset=i * k + c] = acc[unsafe_offset=i * k + c] + x[unsafe_offset=r * k + c]


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


def scale_f64(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, divisor_in: Float64):
    var divisor = divisor_in
    comptime if XTREES_HOST_SABOTAGE:
        divisor = divisor * 1.0000001
    for i in range(n):
        x[unsafe_offset=i] = x[unsafe_offset=i] / divisor


def scale_to_f32(
    x: MutPointer[Float64, MutUntrackedOrigin], n: Int, factor: Float64,
    dst: MutPointer[Float32, MutUntrackedOrigin],
):
    """dst[i] = Float32(x[i] * factor): one binary64 product, one rounding."""
    for i in range(n):
        dst[unsafe_offset=i] = Float32(identical_mul64(x[unsafe_offset=i], factor))


from std.memory import bitcast

#: limbs of `exact_sum_f32`: 32-bit places 0..9 cover the 277 bits a float32
#: magnitude spans in units of 2^-149 (the smallest subnormal).
comptime EXACT_SUM_LIMBS = 10


def exact_sum_f32(
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, mut limbs: List[Int64],
) -> Bool:
    """The EXACT sum of n float32 values as an integer count of 2^-149,
    returned in `EXACT_SUM_LIMBS` signed 32-bit places (limb i weighs
    2^(32 i); limbs may be negative or exceed 32 bits, the caller adds
    `limb_i << 32 i` as exact integers). No rounding happens anywhere, so the
    order of the adds cannot matter; the caller rounds ONCE (`_portable_math.
    _scaled_integer(total, -149)`, the rounding `_portable_math.fsum` does).
    Returns False, leaving the limbs partial, when an entry is NaN or
    infinite or n exceeds 2^30 (a limb could then reach 2^63): the caller
    takes the Python fsum, which owns those cases."""
    limbs = List[Int64](length=EXACT_SUM_LIMBS, fill=0)
    if n > (1 << 30):
        return False
    for i in range(n):
        var bits = bitcast[DType.uint32](x[unsafe_offset=i])
        var e = Int((bits >> 23) & 0xFF)
        if e == 0xFF:
            return False
        var m = UInt64(bits & 0x7FFFFF)
        var shift = 0
        if e != 0:
            m |= UInt64(1) << 23
            shift = e - 1
        if m == 0:
            continue
        var v = m << UInt64(shift % 32)
        var k = shift // 32
        var lo = Int64(v & 0xFFFFFFFF)
        var hi = Int64(v >> 32)
        if (bits >> 31) != 0:
            lo = -lo
            hi = -hi
        limbs[k] += lo
        limbs[k + 1] += hi
    return True


def put_f32(
    dst: MutPointer[Float32, MutUntrackedOrigin], offset: Int,
    src: MutPointer[Float32, MutUntrackedOrigin], n: Int,
):
    """dst[offset + i] = src[i]: a copy."""
    for i in range(n):
        dst[unsafe_offset=offset + i] = src[unsafe_offset=i]


def check_weights_f32(w: MutPointer[Float32, MutUntrackedOrigin], n: Int) -> Int:
    """A sample-weight vector's status: 1 if an entry is not finite or is
    negative (NaN fails both tests), 2 if none is positive, else 0. Reads
    only; the refusals the Python loop raised, in one native pass."""
    var any_pos = False
    for i in range(n):
        var v = w[unsafe_offset=i]
        if not (v >= Float32(0) and v <= Float32(3.4028234663852886e38)):
            return 1
        if v > Float32(0):
            any_pos = True
    return 0 if any_pos else 2


def mul_f32(
    a: MutPointer[Float32, MutUntrackedOrigin], b: MutPointer[Float32, MutUntrackedOrigin], n: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
):
    """dst[i] = a[i] * b[i] in float32: ONE correctly rounded product, the
    same bits as the exact binary64 product rounded once to float32 (a
    product of two float32 values is exact in binary64). No add meets it, so
    nothing can contract it into an FMA."""
    for i in range(n):
        dst[unsafe_offset=i] = a[unsafe_offset=i] * b[unsafe_offset=i]


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
    reaches in tree t.

    The CPU column's loop, first row first. A GPU install runs
    `xtrees/ops_device.mojo::apply_trees_device`, one thread per (row, tree)
    with the same compares and the same refusals (cpu-gpu-cleanup t-gbdt;
    DEVIATION 5608's host-pool walk is gone)."""
    var nt = t1 - t0
    for t in range(t0, t1):
        var lo = Int(offsets[unsafe_offset=t])
        var count = Int(offsets[unsafe_offset=t + 1]) - lo
        if count < 1:
            raise Error("x_trees apply: empty tree")
    for t in range(t0, t1):
        var lo = Int(offsets[unsafe_offset=t])
        var count = Int(offsets[unsafe_offset=t + 1]) - lo
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
    target: MutPointer[Float32, MutUntrackedOrigin], k: Int = 1,
):
    """LightGBM's objective gradients (regression_objective.hpp L2: g = s - y,
    h = 1; binary_objective.hpp with sigmoid 1: p = 1 / (1 + exp(-s)),
    g = p - y, h = p (1 - p); multiclass_objective.hpp MulticlassSoftmax,
    kind 2: p = softmax over the row's k scores, g = p - [y == c],
    h = (k / (k - 1)) p (1 - p)). target = -g as float32, the tree's fit
    target. kind 2 is CLASS-MAJOR: score, g, h and target at [c * n + i]."""
    if kind == 2:
        var factor = Float64(k) / (Float64(k) - 1.0)
        var rec = List[Float64](length=k, fill=0.0)
        for i in range(n):
            # Common::Softmax: the max, exp(x - max), the sum in class order.
            var m = score[unsafe_offset=i]
            for c in range(1, k):
                if score[unsafe_offset=c * n + i] > m:
                    m = score[unsafe_offset=c * n + i]
            var s: Float64 = 0.0
            for c in range(k):
                var e = identical_exp64(score[unsafe_offset=c * n + i] - m)
                rec[c] = e
                s = s + e
            var yi = Int(y[unsafe_offset=i])
            for c in range(k):
                var p = rec[c] / s
                var gi = p - 1.0 if yi == c else p
                g[unsafe_offset=c * n + i] = gi
                h[unsafe_offset=c * n + i] = identical_mul64(identical_mul64(factor, p), 1.0 - p)
                target[unsafe_offset=c * n + i] = Float32(-gi)
        return
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


def _newton_values(
    sg: List[Float64], sh: List[Float64], n_nodes: Int, reg_lambda: Float64, l1: Float64,
    max_delta_step: Float64, values: MutPointer[Float32, MutUntrackedOrigin],
):
    """LightGBM `CalculateSplittedLeafOutput`: -ThresholdL1(sum g, l1) /
    (sum h + lambda), clipped to +-max_delta_step when that is > 0; 0 where
    the denominator is not positive (a node no row reaches)."""
    for k in range(n_nodes):
        var den = sh[k] + reg_lambda
        if not den > 0.0:
            values[unsafe_offset=k] = Float32(0.0)
            continue
        var s = sg[k]
        if l1 > 0.0:
            # FeatureHistogram::ThresholdL1: Sign(s) * max(0, |s| - l1).
            var a = (s if s >= 0.0 else -s) - l1
            var reg = a if a > 0.0 else 0.0
            s = reg if s > 0.0 else (-reg if s < 0.0 else 0.0)
        var ret = -s / den
        if max_delta_step > 0.0 and (ret if ret >= 0.0 else -ret) > max_delta_step:
            ret = max_delta_step if ret > 0.0 else -max_delta_step
        values[unsafe_offset=k] = Float32(ret)


def leaf_newton(
    nodes: MutPointer[Int32, MutUntrackedOrigin], g: MutPointer[Float64, MutUntrackedOrigin],
    h: MutPointer[Float64, MutUntrackedOrigin], n: Int, n_nodes: Int, reg_lambda: Float64,
    values: MutPointer[Float32, MutUntrackedOrigin], l1: Float64 = 0.0, max_delta_step: Float64 = 0.0,
) raises:
    """values[node] = `_newton_values` of the sums over the rows in that leaf,
    sums in row order."""
    var sg = List[Float64](length=n_nodes, fill=0.0)
    var sh = List[Float64](length=n_nodes, fill=0.0)
    for i in range(n):
        var k = Int(nodes[unsafe_offset=i])
        if k < 0 or k >= n_nodes:
            raise Error("x_trees leaf_newton: node out of range")
        sg[k] = sg[k] + g[unsafe_offset=i]
        sh[k] = sh[k] + h[unsafe_offset=i]
    _newton_values(sg, sh, n_nodes, reg_lambda, l1, max_delta_step, values)


def leaf_newton_rows(
    nodes: MutPointer[Int32, MutUntrackedOrigin], rows: MutPointer[Int32, MutUntrackedOrigin], m: Int,
    g: MutPointer[Float64, MutUntrackedOrigin], h: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    n_nodes: Int, reg_lambda: Float64, l1: Float64, max_delta_step: Float64,
    values: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """leaf_newton over rows[0 .. m) only, in that order (the bagged rows);
    nodes / g / h are indexed by the full row (n of them)."""
    var sg = List[Float64](length=n_nodes, fill=0.0)
    var sh = List[Float64](length=n_nodes, fill=0.0)
    for r in range(m):
        var i = Int(rows[unsafe_offset=r])
        if i < 0 or i >= n:
            raise Error("x_trees leaf_newton_rows: row out of range")
        var k = Int(nodes[unsafe_offset=i])
        if k < 0 or k >= n_nodes:
            raise Error("x_trees leaf_newton_rows: node out of range")
        sg[k] = sg[k] + g[unsafe_offset=i]
        sh[k] = sh[k] + h[unsafe_offset=i]
    _newton_values(sg, sh, n_nodes, reg_lambda, l1, max_delta_step, values)


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
    """dst[j, i] = src[i, j]: a row-major n x d copied to column-major. The
    CPU column's loop; a GPU install runs `ops_device.transpose_f32_device`
    (cpu-gpu-cleanup t-gbdt)."""
    for i in range(n):
        for j in range(d):
            dst.unsafe_store(j * n + i, src.unsafe_load(i * d + j))


# ------------------------------------------------------ wrappers' helpers
def logit(x: MutPointer[Float64, MutUntrackedOrigin], n: Int):
    """In place: x = log(x / (1 - x)) (shap `links.logit`), one IEEE division
    and the pinned binary64 log."""
    for i in range(n):
        var v = x[unsafe_offset=i]
        x[unsafe_offset=i] = identical_log64(v / (1.0 - v))


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
    platt_apply_strided(f, 1, n, a, b, res, 1)


def platt_apply_strided(
    f: MutPointer[Float64, MutUntrackedOrigin], fs: Int, n: Int, a: Float64, b: Float64,
    res: MutPointer[Float64, MutUntrackedOrigin], rs: Int,
):
    """`platt_apply` reading f[i * fs] and writing res[i * rs] (lane
    py-misc-prep: a column of a row-major score block straight into a
    column of the probability block; the same operations per element)."""
    for i in range(n):
        var z = identical_mul64(f[unsafe_offset=i * fs], a) + b
        if z >= 0.0:
            var e = identical_exp64(-z)
            res[unsafe_offset=i * rs] = e / (1.0 + e)
        else:
            res[unsafe_offset=i * rs] = 1.0 / (1.0 + identical_exp64(z))


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
    isotonic_predict_strided(kx, ky, m, t, 1, n, res, 1)


def isotonic_predict_strided(
    kx: MutPointer[Float64, MutUntrackedOrigin], ky: MutPointer[Float64, MutUntrackedOrigin], m: Int,
    t: MutPointer[Float64, MutUntrackedOrigin], ts: Int, n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
    rs: Int,
):
    """`isotonic_predict` reading t[i * ts] and writing res[i * rs] (lane py-misc-prep)."""
    for i in range(n):
        var v = t[unsafe_offset=i * ts]
        if m == 1 or v <= kx[unsafe_offset=0]:
            res[unsafe_offset=i * rs] = ky[unsafe_offset=0]
            continue
        if v >= kx[unsafe_offset=m - 1]:
            res[unsafe_offset=i * rs] = ky[unsafe_offset=m - 1]
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
        res[unsafe_offset=i * rs] = identical_mul64(slope, v - x0) + y0


def complement_pairs(x: MutPointer[Float64, MutUntrackedOrigin], n: Int):
    """x[2 i] = 1 - x[2 i + 1]: a binary calibrator's (1 - p, p) rows, one
    IEEE subtract each (the Python `1.0 - x` it replaces)."""
    for i in range(n):
        x[unsafe_offset=2 * i] = 1.0 - x[unsafe_offset=2 * i + 1]


def indicator_codes(
    codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, cls: Int,
    out_i: MutPointer[Int32, MutUntrackedOrigin], out_f: MutPointer[Float64, MutUntrackedOrigin], want_f: Bool,
):
    """out_i[r] = 1 if codes[r] == cls else 0 (and out_f the same as 1.0 / 0.0)."""
    for r in range(n):
        var hit = Int(codes[unsafe_offset=r]) == cls
        out_i[unsafe_offset=r] = Int32(1) if hit else Int32(0)
        if want_f:
            out_f[unsafe_offset=r] = 1.0 if hit else 0.0


def column_f64(
    src: MutPointer[Float64, MutUntrackedOrigin], n: Int, c: Int, j: Int, dst: MutPointer[Float64, MutUntrackedOrigin],
):
    """dst[r] = src[r, j] of a row-major (n, c) block (a copy)."""
    for r in range(n):
        dst[unsafe_offset=r] = src[unsafe_offset=r * c + j]
