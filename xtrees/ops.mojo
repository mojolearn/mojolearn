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
from xtrees.fold_order import FOLD_CHUNK, fold_chunks, fold_chunk_size, fold_tree_host

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
    `draw % n_pool`; without (lane cpu2-l5-trees, a parallel law replacing
    the serial partial Fisher-Yates), key_i = draw(base, i) >> 11 for every
    i in [0, n_pool), the selected rows are the n_draw smallest (key_i, i)
    pairs, written in ASCENDING index order. Integer keys, so the selection
    is order-free: `ops_device_elem.sample_indices_device` finds the same
    set by a parallel radix select. Here: the same 8-bit digit passes
    serially (the n_draw-th smallest key K* and how many of its ties to
    take), then one walk in index order."""
    if n_pool <= 0 or n_draw < 0 or (not replace and n_draw > n_pool):
        raise Error("x_trees sample_indices: need n_pool > 0 and 0 <= n_draw (<= n_pool without replacement)")
    var base = stream_base(seed, stream)
    if replace:
        for k in range(n_draw):
            # DEVIATION 5600: the index is the counter draw mod n.
            res[unsafe_offset=k] = Int32(Int(draw(base, k) % UInt64(n_pool)))
        return
    if n_draw == 0:
        return
    var keys = List[UInt64](capacity=n_pool)
    for i in range(n_pool):
        keys.append(draw(base, i) >> 11)
    var prefix = UInt64(0)
    var want = n_draw
    var shift = 48
    while shift >= 0:
        var hist = List[Int](length=256, fill=0)
        var sh = UInt64(shift)
        for i in range(n_pool):
            var key = keys[i]
            if (key >> (sh + 8)) == (prefix >> (sh + 8)):
                hist[Int((key >> sh) & UInt64(0xFF))] += 1
        var cum = 0
        for b in range(256):
            if cum < want and want <= cum + hist[b]:
                prefix |= UInt64(b) << sh
                want -= cum
                break
            cum += hist[b]
        shift -= 8
    var k = 0
    var ties = 0
    for i in range(n_pool):
        var key = keys[i]
        var take = key < prefix
        if key == prefix:
            take = ties < want
            ties += 1
        if take:
            res[unsafe_offset=k] = Int32(i)
            k += 1


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
# cpu2-l5-trees (2026-10-04): every n-sized float64 sum of the AdaBoost
# steps runs in xtrees/fold_order.mojo's fixed order (chunk partials, then
# the pairwise tree), the order of `ops_device_boost`'s kernels, so a GPU
# install runs these steps on the device with the host column's words. Up to
# FOLD_CHUNK rows the order IS the old sequential one.
def samme_alpha(err: Float64, k: Float64, learning_rate: Float64) -> Float64:
    """SAMME's estimator weight lr * (log((1 - err) / err) + log(k - 1))."""
    return identical_mul64(learning_rate, identical_log64((1.0 - err) / err) + identical_log64(k - 1.0))


def samme_step(
    w: MutPointer[Float64, MutUntrackedOrigin], pred: MutPointer[Int32, MutUntrackedOrigin],
    y: MutPointer[Int32, MutUntrackedOrigin], n: Int, n_classes: Int,
    learning_rate: Float64, last: Bool, stats: MutPointer[Float64, MutUntrackedOrigin],
):
    """sklearn `AdaBoostClassifier._boost_discrete` (SAMME,
    ensemble/_weight_boosting.py). `w` sums to one on entry. Writes
    stats = [status, estimator_weight, estimator_error, sum of new w]:
    status 0 continue, 1 perfect fit (weight 1, stop), 2 worse than chance
    (discard, stop). Updates `w` in place (not on the last iteration). The
    error, the total and the new sum fold in the fixed order."""
    var m = fold_chunks(n)
    var p = List[Float64](length=2 * m, fill=0.0)
    for c in range(m):
        var e: Float64 = 0.0
        var t: Float64 = 0.0
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, n)):
            t = t + w[unsafe_offset=i]
            if pred[unsafe_offset=i] != y[unsafe_offset=i]:
                e = e + w[unsafe_offset=i]
        p[2 * c] = e
        p[2 * c + 1] = t
    fold_tree_host(p, m, 2)
    var tot = p[1]
    var err = p[0] / tot
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
    var alpha = samme_alpha(err, k, learning_rate)
    var s: Float64
    if not last:
        # DEVIATION 5603: the pinned exp.
        var boost = identical_exp64(alpha)
        var q = List[Float64](length=m, fill=0.0)
        for c in range(m):
            var run: Float64 = 0.0
            for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, n)):
                if pred[unsafe_offset=i] != y[unsafe_offset=i] and w[unsafe_offset=i] > 0.0:
                    w[unsafe_offset=i] = identical_mul64(w[unsafe_offset=i], boost)
                run = run + w[unsafe_offset=i]
            q[c] = run
        fold_tree_host(q, m, 1)
        s = q[0]
    else:
        s = tot
    stats[unsafe_offset=0] = 0.0
    stats[unsafe_offset=1] = alpha
    stats[unsafe_offset=2] = err
    stats[unsafe_offset=3] = s


@always_inline
def r2_error(p: Float32, t: Float32, emax: Float64, loss: Int) -> Float64:
    """AdaBoost.R2's per-row loss: |p - t| / emax, then squared (1) or
    1 - exp(-e) (2)."""
    var e = abs(Float64(p) - Float64(t))
    if emax != 0.0:
        e = e / emax
    if loss == 1:
        e = identical_mul64(e, e)
    elif loss == 2:
        e = 1.0 - identical_exp64(-e)
    return e


def r2_step(
    w: MutPointer[Float64, MutUntrackedOrigin], pred: MutPointer[Float32, MutUntrackedOrigin],
    y: MutPointer[Float32, MutUntrackedOrigin], n: Int, loss: Int,
    learning_rate: Float64, last: Bool, stats: MutPointer[Float64, MutUntrackedOrigin],
):
    """sklearn `AdaBoostRegressor._boost` (AdaBoost.R2, Drucker 1997).
    loss 0 linear, 1 square, 2 exponential. stats as `samme_step`'s, status 2
    meaning estimator_error >= 0.5. The error and the new sum fold in the
    fixed order; the reweighting factor beta ** ((1 - e) lr) is spelled
    exp(((1 - e) lr) log(beta)) with the pinned exp and log (cpu2-l5-trees:
    the device has no pow; the same spelling on every column)."""
    var emax: Float64 = 0.0
    for i in range(n):
        if w[unsafe_offset=i] > 0.0:
            var e = abs(Float64(pred[unsafe_offset=i]) - Float64(y[unsafe_offset=i]))
            if e > emax:
                emax = e
    var m = fold_chunks(n)
    var p = List[Float64](length=m, fill=0.0)
    for c in range(m):
        var run: Float64 = 0.0
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, n)):
            if w[unsafe_offset=i] > 0.0:
                run = run + identical_mul64(w[unsafe_offset=i], r2_error(pred[unsafe_offset=i], y[unsafe_offset=i], emax, loss))
        p[c] = run
    fold_tree_host(p, m, 1)
    var err = p[0]
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
    var lb = identical_log64(beta)
    var q = List[Float64](length=m, fill=0.0)
    for c in range(m):
        var run: Float64 = 0.0
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, n)):
            if not last and w[unsafe_offset=i] > 0.0:
                var e = r2_error(pred[unsafe_offset=i], y[unsafe_offset=i], emax, loss)
                var t = identical_mul64(1.0 - e, learning_rate)
                w[unsafe_offset=i] = identical_mul64(w[unsafe_offset=i], identical_exp64(identical_mul64(t, lb)))
            run = run + w[unsafe_offset=i]
        q[c] = run
    fold_tree_host(q, m, 1)
    stats[unsafe_offset=0] = 0.0
    stats[unsafe_offset=1] = alpha
    stats[unsafe_offset=2] = err
    stats[unsafe_offset=3] = q[0]


def median_total(weights: MutPointer[Float64, MutUntrackedOrigin], m: Int) -> Float64:
    """The estimator weights' total, in estimator order (m-sized: one scalar
    for every row)."""
    var total: Float64 = 0.0
    for j in range(m):
        total = total + weights[unsafe_offset=j]
    return total


@always_inline
def median_pick(
    preds: MutPointer[Float32, MutUntrackedOrigin], weights: MutPointer[Float64, MutUntrackedOrigin], n: Int, m: Int,
    i: Int, half: Float64,
) -> Int:
    """Row i's weighted-median estimator: the smallest key (prediction, then
    estimator index) whose cumulative weight -- the weights of every key at
    or below it, added in estimator order -- reaches `half`; none: the
    largest key."""
    var pick = -1
    var pick_v = Float32(0)
    var last = 0
    var last_v = preds[unsafe_offset=i]
    for j in range(m):
        var v = preds[unsafe_offset=j * n + i]
        if j > 0 and not (v < last_v):
            last = j
            last_v = v
        var c: Float64 = 0.0
        for l in range(m):
            var u = preds[unsafe_offset=l * n + i]
            if u < v or (u == v and l <= j):
                c = c + weights[unsafe_offset=l]
        if c >= half and (pick < 0 or v < pick_v or (v == pick_v and j < pick)):
            pick = j
            pick_v = v
    return pick if pick >= 0 else last


def weighted_median(
    preds: MutPointer[Float32, MutUntrackedOrigin], weights: MutPointer[Float64, MutUntrackedOrigin],
    n: Int, m: Int, res: MutPointer[Float32, MutUntrackedOrigin],
):
    """sklearn `AdaBoostRegressor._get_median_predict`: per row, over the
    estimators ordered by (prediction, estimator index), the first whose
    cumulative weight reaches half the total. `preds` is estimator major (m
    rows of n). cpu2-l5-trees: the total is one estimator-order sum for every
    row and each estimator's cumulative weight is the sum of the weights at
    or below its key in estimator order (`median_pick`), a per-row law with
    no sort, the device's (one thread per row) on every column."""
    var half = identical_mul64(0.5, median_total(weights, m))
    for i in range(n):
        res[unsafe_offset=i] = preds[unsafe_offset=median_pick(preds, weights, n, m, i, half) * n + i]


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


def tree_shape(left: MutPointer[Int32, MutUntrackedOrigin], cnt: Int, res: MutPointer[Int32, MutUntrackedOrigin]) raises:
    """res[0] = the depth and res[1] = the leaf count of one fitted tree's
    flat nodes (lane fam2-forests, `x_trees_tree_shape`): children of a node
    sit at left and left + 1 and come after it, a leaf has left == -1. Model
    metadata for `get_depth` / `get_n_leaves`, the walk
    python/mojolearn/_expansion_trees.py used to run over `tolist()` rows;
    integers, the same on every column."""
    var depth = List[Int32](length=max(cnt, 1), fill=0)
    var best = 0
    var leaves = 0
    for i in range(cnt):
        var c = Int(left[unsafe_offset=i])
        if c == -1:
            leaves += 1
        else:
            if c <= i or c + 1 >= cnt:
                raise Error("x_trees tree_shape: child out of range")
            var dch = depth[i] + Int32(1)
            depth[c] = dch
            depth[c + 1] = dch
            if Int(dch) > best:
                best = Int(dch)
    res[unsafe_offset=0] = Int32(best)
    res[unsafe_offset=1] = Int32(leaves)


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


def leaf_sums(
    nodes: MutPointer[Int32, MutUntrackedOrigin], rows: MutPointer[Int32, MutUntrackedOrigin], use_rows: Bool,
    m: Int, g: MutPointer[Float64, MutUntrackedOrigin], h: MutPointer[Float64, MutUntrackedOrigin], n_nodes: Int,
) -> List[Float64]:
    """The per-leaf g / h sums (cpu2-l5-trees, xtrees/fold_order.mojo's
    order): list positions r in [0, m) (row i = rows[r], or r) fall in chunks
    of `fold_chunk_size(m, 2 n_nodes)`; a chunk's partial for node k adds its
    rows of that node in position order from +0.0; the partials fold by the
    pairwise tree. Returns 2 n_nodes words: [sum g, sum h] per node. The
    device runs one unit per (chunk, node) with the same statements
    (`ops_device_boost.leaf_newton_device`). Indices are checked by the
    caller."""
    var w = 2 * n_nodes
    var cs = fold_chunk_size(m, w)
    var nc = max(1, (m + cs - 1) // cs)
    var p = List[Float64](length=nc * w, fill=0.0)
    for c in range(nc):
        for r in range(c * cs, min((c + 1) * cs, m)):
            var i = Int(rows[unsafe_offset=r]) if use_rows else r
            var k = Int(nodes[unsafe_offset=i])
            p[c * w + 2 * k] = p[c * w + 2 * k] + g[unsafe_offset=i]
            p[c * w + 2 * k + 1] = p[c * w + 2 * k + 1] + h[unsafe_offset=i]
    fold_tree_host(p, nc, w)
    return p^


def _newton_values_rec(
    p: List[Float64], n_nodes: Int, reg_lambda: Float64, l1: Float64, max_delta_step: Float64,
    values: MutPointer[Float32, MutUntrackedOrigin],
):
    var sg = List[Float64](length=n_nodes, fill=0.0)
    var sh = List[Float64](length=n_nodes, fill=0.0)
    for k in range(n_nodes):
        sg[k] = p[2 * k]
        sh[k] = p[2 * k + 1]
    _newton_values(sg, sh, n_nodes, reg_lambda, l1, max_delta_step, values)


def leaf_newton(
    nodes: MutPointer[Int32, MutUntrackedOrigin], g: MutPointer[Float64, MutUntrackedOrigin],
    h: MutPointer[Float64, MutUntrackedOrigin], n: Int, n_nodes: Int, reg_lambda: Float64,
    values: MutPointer[Float32, MutUntrackedOrigin], l1: Float64 = 0.0, max_delta_step: Float64 = 0.0,
) raises:
    """values[node] = `_newton_values` of the sums over the rows in that leaf,
    folded in `leaf_sums`' fixed order."""
    for i in range(n):
        var k = Int(nodes[unsafe_offset=i])
        if k < 0 or k >= n_nodes:
            raise Error("x_trees leaf_newton: node out of range")
    var p = leaf_sums(nodes, nodes, False, n, g, h, n_nodes)
    _newton_values_rec(p, n_nodes, reg_lambda, l1, max_delta_step, values)


def leaf_newton_rows(
    nodes: MutPointer[Int32, MutUntrackedOrigin], rows: MutPointer[Int32, MutUntrackedOrigin], m: Int,
    g: MutPointer[Float64, MutUntrackedOrigin], h: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    n_nodes: Int, reg_lambda: Float64, l1: Float64, max_delta_step: Float64,
    values: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """leaf_newton over rows[0 .. m) only (the bagged rows), folded in
    `leaf_sums`' fixed order over the list positions; nodes / g / h are
    indexed by the full row (n of them)."""
    for r in range(m):
        var i = Int(rows[unsafe_offset=r])
        if i < 0 or i >= n:
            raise Error("x_trees leaf_newton_rows: row out of range")
        var k = Int(nodes[unsafe_offset=i])
        if k < 0 or k >= n_nodes:
            raise Error("x_trees leaf_newton_rows: node out of range")
    var p = leaf_sums(nodes, rows, True, m, g, h, n_nodes)
    _newton_values_rec(p, n_nodes, reg_lambda, l1, max_delta_step, values)


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


def bag_rows(res: MutPointer[Int32, MutUntrackedOrigin], n: Int, seed: Int, stream: Int, frac: Float64) -> Int:
    """The rows `i` (ascending) whose draw `unit(draw(base, i)) < frac`; none
    kept: the one row with the smallest (draw, i). Returns the count (>= 1
    for n >= 1). The CPU column's loop; a GPU install runs
    `ops_device.bag_rows_device` (cpu-gpu-cleanup t-gbdt)."""
    var base = stream_base(seed, stream)
    var k = 0
    var best = 0
    var best_key = UInt64.MAX
    for i in range(n):
        var r = draw(base, i)
        if unit(r) < frac:
            res[unsafe_offset=k] = Int32(i)
            k += 1
        if (r >> 11) < best_key:
            best_key = r >> 11
            best = i
    if k == 0 and n > 0:
        res[unsafe_offset=0] = Int32(best)
        k = 1
    return k


def unseen_rows(
    rows: MutPointer[Int32, MutUntrackedOrigin], m: Int, n: Int, res: MutPointer[Int32, MutUntrackedOrigin],
) raises -> Int:
    """The rows of `[0, n)` absent from `rows[0, m)`, ascending (sklearn's
    negated `indices_to_mask`); returns the count. The CPU column's loop; a
    GPU install runs `ops_device.unseen_rows_device` (cpu-gpu-cleanup t-gbdt)."""
    var seen = List[Bool](length=n, fill=False)
    for r in range(m):
        var i = Int(rows[unsafe_offset=r])
        if i < 0 or i >= n:
            raise Error("x_trees unseen_rows: row out of range")
        seen[i] = True
    var k = 0
    for i in range(n):
        if not seen[i]:
            res[unsafe_offset=k] = Int32(i)
            k += 1
    return k


def transpose_f64(
    src: MutPointer[Float64, MutUntrackedOrigin], n: Int, d: Int, dst: MutPointer[Float64, MutUntrackedOrigin],
):
    """dst (d x n) = src (n x d)^T, a copy of each word. The CPU column's
    loop; a GPU install runs `ops_device.transpose_f64_device`."""
    for i in range(n):
        for j in range(d):
            dst.unsafe_store(j * n + i, src.unsafe_load(i * d + j))


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


@always_inline
def platt_target(yi: Int32, hi: Float64, lo: Float64) -> Float64:
    return hi if yi > 0 else lo


trait PlattSums:
    """The two n-sized sums Platt's Newton driver needs, in
    xtrees/fold_order.mojo's fixed order: the objective at (a, b) and the
    five gradient / Hessian sums [sum f^2 d2, sum d2, sum f d2, sum f d1,
    sum d1]. `PlattHost` is the host column's; `ops_device_boost.PlattDevice`
    holds f and y on the device and folds there (the same words)."""

    def value(mut self, a: Float64, b: Float64) raises -> Float64:
        ...

    def grad(mut self, a: Float64, b: Float64, mut out: List[Float64]) raises:
        ...


struct PlattHost(PlattSums):
    var f: MutPointer[Float64, MutUntrackedOrigin]
    var y: MutPointer[Int32, MutUntrackedOrigin]
    var n: Int
    var hi: Float64
    var lo: Float64

    def __init__(out self, f: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Int32, MutUntrackedOrigin],
                 n: Int, hi: Float64, lo: Float64):
        self.f = f
        self.y = y
        self.n = n
        self.hi = hi
        self.lo = lo

    def value(mut self, a: Float64, b: Float64) raises -> Float64:
        var m = fold_chunks(self.n)
        var p = List[Float64](length=m, fill=0.0)
        for c in range(m):
            var run: Float64 = 0.0
            for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, self.n)):
                var t = platt_target(self.y[unsafe_offset=i], self.hi, self.lo)
                var z = identical_mul64(self.f[unsafe_offset=i], a) + b
                # -T log p - (1 - T) log(1 - p), p = 1 / (1 + exp(z))
                run = run + identical_mul64(t, _log1pexp(z)) + identical_mul64(1.0 - t, _log1pexp(-z))
            p[c] = run
        fold_tree_host(p, m, 1)
        return p[0]

    def grad(mut self, a: Float64, b: Float64, mut out: List[Float64]) raises:
        var m = fold_chunks(self.n)
        var p = List[Float64](length=5 * m, fill=0.0)
        for c in range(m):
            var s0: Float64 = 0.0
            var s1: Float64 = 0.0
            var s2: Float64 = 0.0
            var s3: Float64 = 0.0
            var s4: Float64 = 0.0
            for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, self.n)):
                var fi = self.f[unsafe_offset=i]
                var z = identical_mul64(fi, a) + b
                var pp: Float64
                var q: Float64
                if z >= 0.0:
                    var e = identical_exp64(-z)
                    pp = e / (1.0 + e)
                    q = 1.0 / (1.0 + e)
                else:
                    var e = identical_exp64(z)
                    pp = 1.0 / (1.0 + e)
                    q = e / (1.0 + e)
                var d2 = identical_mul64(pp, q)
                var d1 = platt_target(self.y[unsafe_offset=i], self.hi, self.lo) - pp
                s0 = s0 + identical_mul64(identical_mul64(fi, fi), d2)
                s1 = s1 + d2
                s2 = s2 + identical_mul64(fi, d2)
                s3 = s3 + identical_mul64(fi, d1)
                s4 = s4 + d1
            p[5 * c] = s0
            p[5 * c + 1] = s1
            p[5 * c + 2] = s2
            p[5 * c + 3] = s3
            p[5 * c + 4] = s4
        fold_tree_host(p, m, 5)
        for q in range(5):
            out[q] = p[q]


def platt_drive[S: PlattSums](mut sums: S, prior1: Float64, n: Int) raises -> Tuple[Float64, Float64]:
    """Platt scaling's Newton with backtracking over `sums` (the one driver of
    every column: only the sums differ in where they run). prior1 = the
    positive count."""
    var prior0 = Float64(n) - prior1
    var a: Float64 = 0.0
    var b = identical_log64((prior0 + 1.0) / (prior1 + 1.0))
    var fval = sums.value(a, b)
    var gs = List[Float64](length=5, fill=0.0)
    for _ in range(100):
        sums.grad(a, b, gs)
        var h11 = 1e-12 + gs[0]
        var h22 = 1e-12 + gs[1]
        var h21 = gs[2]
        var g1 = gs[3]
        var g2 = gs[4]
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
            var nf = sums.value(na, nb)
            if nf < fval + identical_mul64(identical_mul64(0.0001, step), gd):
                a = na
                b = nb
                fval = nf
                moved = True
                break
            step = step / 2.0
        if not moved:
            break
    return (a, b)


def platt_priors(y: MutPointer[Int32, MutUntrackedOrigin], n: Int) -> Tuple[Float64, Float64, Float64]:
    """(positive count, T+, T-): the count is an integer, exact in any order."""
    var cnt = 0
    for i in range(n):
        if y[unsafe_offset=i] > 0:
            cnt += 1
    return platt_targets(cnt, n)


def platt_targets(cnt: Int, n: Int) -> Tuple[Float64, Float64, Float64]:
    var prior1 = Float64(cnt)
    var prior0 = Float64(n) - prior1
    return (prior1, (prior1 + 1.0) / (prior1 + 2.0), 1.0 / (prior0 + 2.0))


def platt_fit(
    f: MutPointer[Float64, MutUntrackedOrigin], y: MutPointer[Int32, MutUntrackedOrigin], n: Int,
    ab: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """Platt scaling (sklearn calibration.py `_sigmoid_calibration`: the
    targets T = (N+ + 1)/(N+ + 2) and 1/(N- + 2), the start B =
    log((N- + 1)/(N+ + 1))), minimised by Newton with backtracking (Lin, Lin,
    Weng 2007) where sklearn runs L-BFGS on the same objective. Writes
    ab = [A, B]; P(y=1 | f) = 1 / (1 + exp(A f + B)). cpu2-l5-trees: the
    objective and gradient sums fold in xtrees/fold_order.mojo's order (the
    Hessian diagonal's 1e-12 is added to the folded sum), the device's."""
    var pr = platt_priors(y, n)
    var sums = PlattHost(f, y, n, pr[1], pr[2])
    var res = platt_drive(sums, pr[0], n)
    ab[unsafe_offset=0] = res[0]
    ab[unsafe_offset=1] = res[1]


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
    # cpu2-l5-trees: the unique-x means and the pooling in the device's
    # order (`iso_seg_scan`, `iso_pav_round`): no sequential chain over n.
    var xs = List[Float64](length=n, fill=0.0)
    var v = List[Float64](length=n, fill=0.0)
    var first = List[Bool](length=n, fill=False)
    for p in range(n):
        xs[p] = x[unsafe_offset=idx[p]]
        v[p] = y[unsafe_offset=idx[p]]
        first[p] = p == 0 or not (xs[p] == xs[p - 1])
    iso_seg_scan(v, first, n)
    var ux = List[Float64]()
    var uy = List[Float64]()
    var uw = List[Float64]()
    var start = 0
    for p in range(n):
        if first[p]:
            start = p
        if p == n - 1 or first[p + 1]:
            var c = Float64(p - start + 1)
            ux.append(xs[start])
            uy.append(v[p] / c)
            uw.append(c)
    var m = len(ux)
    var bs = List[Float64](length=m, fill=0.0)
    var bw = List[Float64](length=m, fill=0.0)
    var blk = List[Int](length=m, fill=0)
    for q in range(m):
        bs[q] = identical_mul64(uw[q], uy[q])
        bw[q] = uw[q]
        blk[q] = q
    while iso_pav_round(bs, bw, blk, m):
        pass
    for q in range(m):
        kx[unsafe_offset=q] = ux[q]
        ky[unsafe_offset=q] = bs[blk[q]] / bw[blk[q]]
    return m


def iso_seg_scan(mut v: List[Float64], first: List[Bool], n: Int):
    """In place: the inclusive SEGMENTED sum of v (a segment starts where
    `first`), by the fixed Hillis-Steele passes (s = 1, 2, 4, ... while
    s < n: v[j] = v_prev[j - s] + v_prev[j] when j - s is in j's segment),
    the device's `iso_seg_pass_kernel` operation for operation. A segment's
    last entry holds its sum."""
    var seg = List[Int](length=n, fill=0)
    var cur = 0
    for j in range(n):
        if first[j]:
            cur = j
        seg[j] = cur
    var s = 1
    while s < n:
        var prev = v.copy()
        for j in range(s, n):
            if j - s >= seg[j]:
                v[j] = prev[j - s] + prev[j]
        s *= 2


def iso_pav_round(mut bs: List[Float64], mut bw: List[Float64], mut blk: List[Int], m: Int) -> Bool:
    """One pooling round over the blocks (sum w y, sum w): every maximal run
    of adjacent violators (mean[b] >= mean[b + 1], the pool-adjacent-
    violators test) becomes one block, its sums by `iso_seg_scan`; `blk`
    (each knot's block) follows. Returns False when nothing violated (the
    blocks are then the isotonic fit: pooling adjacent violators in any
    order reaches the one solution). The device's `iso_pav_round_device`."""
    var nb = len(bs)
    var first = List[Bool](length=nb, fill=True)
    var hit = False
    for b in range(1, nb):
        var viol = bs[b - 1] / bw[b - 1] >= bs[b] / bw[b]
        first[b] = not viol
        if viol:
            hit = True
    if not hit:
        return False
    iso_seg_scan(bs, first, nb)
    iso_seg_scan(bw, first, nb)
    var nid = List[Int](length=nb, fill=0)
    var nbs = List[Float64]()
    var nbw = List[Float64]()
    var g = -1
    for b in range(nb):
        if first[b]:
            g += 1
        nid[b] = g
        if b == nb - 1 or first[b + 1]:
            nbs.append(bs[b])
            nbw.append(bw[b])
    for q in range(m):
        blk[q] = nid[blk[q]]
    bs = nbs^
    bw = nbw^
    return True


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


def stack_w64(cols: List[Int], n: Int, dst: MutPointer[UInt64, MutUntrackedOrigin]):
    """dst (n x m, row-major 8-byte words) with column j = the n words at
    address cols[j] (lane apple-fast-py2mojo-trees: MultiOutputClassifier's
    `zip(*cols)` transpose; the host column of `glue_device.stack_w64_device`)."""
    var m = len(cols)
    for j in range(m):
        var src = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=cols[j])
        for r in range(n):
            dst[unsafe_offset=r * m + j] = src[unsafe_offset=r]


def binary_proba(p: MutPointer[Float32, MutUntrackedOrigin], n: Int, res: MutPointer[Float64, MutUntrackedOrigin]):
    """res (n x 2) rows (1 - p, p), p widened exactly: the Python
    `[[1.0 - v, v] for v in p.tolist()]` (lane apple-fast-py2mojo-trees; the
    host column of `glue_device.binary_proba_device`)."""
    for r in range(n):
        var w = Float64(p[unsafe_offset=r])
        res[unsafe_offset=2 * r + 1] = w
        res[unsafe_offset=2 * r] = 1.0 - w


def folds_serial(
    codes: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int, n_splits: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], counts: MutPointer[Int32, MutUntrackedOrigin],
) -> Int:
    """`folds_device.device_folds` on the host column (lane
    apple-fast-py2mojo-trees): sklearn's unshuffled StratifiedKFold (k > 0)
    or KFold (k == 0), the same fold sizes and fold row lists, integers
    only. Returns the same status word (2: a code outside [0, k); 1:
    n_splits above every class count; 0)."""
    for i in range(n_splits + 1):
        counts[unsafe_offset=i] = 0
    var folds = List[Int](length=n, fill=0)
    if k > 0:
        var hist = List[Int](length=k, fill=0)
        var enc = List[Int](length=k, fill=-1)
        var n_enc = 0
        for r in range(n):
            var c = Int(codes[unsafe_offset=r])
            if c < 0 or c >= k:
                counts[unsafe_offset=n_splits] = 2
                return 2
            if hist[c] == 0:
                enc[c] = n_enc
                n_enc += 1
            hist[c] += 1
        var cmax = 0
        for c in range(k):
            cmax = max(cmax, hist[c])
        if n_splits > cmax:
            counts[unsafe_offset=n_splits] = 1
            return 1
        var count_enc = List[Int](length=n_enc, fill=0)
        for c in range(k):
            if enc[c] >= 0:
                count_enc[enc[c]] = hist[c]
        var start = List[Int](length=n_enc, fill=0)
        for e in range(1, n_enc):
            start[e] = start[e - 1] + count_enc[e - 1]
        var seen = List[Int](length=n_enc, fill=0)
        var ns = n_splits
        for r in range(n):
            var e = enc[Int(codes[unsafe_offset=r])]
            var j = seen[e]
            seen[e] += 1
            var s = start[e]
            var cnt = count_enc[e]
            var sm = s % ns
            var cum = 0
            var f = 0
            while f < ns - 1:
                var p0 = s + ((f - sm + ns) % ns)
                if p0 < s + cnt:
                    cum += (s + cnt - 1 - p0) // ns + 1
                if j < cum:
                    break
                f += 1
            folds[r] = f
    else:
        var q = n // n_splits
        var rem = n % n_splits
        for r in range(n):
            if r < rem * (q + 1):
                folds[r] = r // (q + 1)
            else:
                folds[r] = rem + (r - rem * (q + 1)) // q
    for r in range(n):
        counts[unsafe_offset=folds[r]] += 1
    for i in range(n_splits):
        var base = i * n
        var cnt = Int(counts[unsafe_offset=i])
        var out_pos = 0
        var in_pos = 0
        for r in range(n):
            if folds[r] == i:
                rows[unsafe_offset=base + (n - cnt) + in_pos] = Int32(r)
                in_pos += 1
            else:
                rows[unsafe_offset=base + out_pos] = Int32(r)
                out_pos += 1
    return 0


# lane apple-fast-py2mojo-trees: the host column of glue_device's DART / RTE
# bookkeeping (the same integers, compares and word copies).


def class_counts(y: MutPointer[Float32, MutUntrackedOrigin], n: Int, k: Int,
                 counts: MutPointer[Int32, MutUntrackedOrigin]) raises:
    for c in range(k):
        counts[unsafe_offset=c] = 0
    for r in range(n):
        var v = y[unsafe_offset=r]
        var c = Int(v) if (v >= 0.0 and v < Float32(k)) else -1
        if c < 0 or Float32(c) != v:
            raise Error("x_trees class_counts: a class code outside [0, n_classes)")
        counts[unsafe_offset=c] = counts[unsafe_offset=c] + 1


def remap_cols(colid: MutPointer[Int32, MutUntrackedOrigin], nn: Int,
               cols: MutPointer[Int32, MutUntrackedOrigin], m: Int) raises:
    for g in range(nn):
        var v = Int(colid[unsafe_offset=g])
        if v >= m:
            raise Error("x_trees remap_cols: a split column outside the sampled columns")
    for g in range(nn):
        var v = Int(colid[unsafe_offset=g])
        if v >= 0:
            colid[unsafe_offset=g] = cols[unsafe_offset=v]


def positive_codes(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, codes: MutPointer[Int32, MutUntrackedOrigin]):
    for r in range(n):
        codes[unsafe_offset=r] = Int32(1) if x[unsafe_offset=r] > 0.0 else Int32(0)


def spread_leaves(vals: MutPointer[Float32, MutUntrackedOrigin], offs: MutPointer[Int32, MutUntrackedOrigin],
                  t: Int, nn: Int, k: Int, dst: MutPointer[Float32, MutUntrackedOrigin]):
    for j in range(t):
        var c = j % k
        for g in range(Int(offs[unsafe_offset=j]), Int(offs[unsafe_offset=j + 1])):
            for q in range(k):
                dst[unsafe_offset=g * k + q] = vals[unsafe_offset=g] if q == c else Float32(0.0)


# lane cpu2-l5-trees: host twins of `ops_device_elem.mojo`'s new device
# entries (the CPU column; a GPU build runs the device spelling).


def margin2(
    acc: MutPointer[Float64, MutUntrackedOrigin], n: Int, mode: Int,
    dst_f: MutPointer[Float64, MutUntrackedOrigin], dst_i: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """Two-class vote rows (n x 2) to the SAMME margin d = acc[2i+1] -
    acc[2i], one IEEE binary64 subtraction per row. mode 0: dst_f[i] = d;
    mode 1: dst_i[i] = 1 if d > 0 else 0 (a NaN gives 0); mode 2: dst_f
    pairs (-(d/2), d/2). `dst_f` and `dst_i` may alias (the binding passes
    one address); only the mode's one is written."""
    if mode < 0 or mode > 2:
        raise Error("x_trees_margin2: mode must be 0, 1 or 2")
    for i in range(n):
        var d = acc[unsafe_offset=2 * i + 1] - acc[unsafe_offset=2 * i]
        if mode == 0:
            dst_f[unsafe_offset=i] = d
        elif mode == 1:
            dst_i[unsafe_offset=i] = Int32(1) if d > 0 else Int32(0)
        else:
            var h = d / 2
            dst_f[unsafe_offset=2 * i] = -h
            dst_f[unsafe_offset=2 * i + 1] = h


def normalized_weights(
    w: MutPointer[Float32, MutUntrackedOrigin], n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
) -> Int:
    """AdaBoost's initial weights: res[i] = float64(w[i]) / total, the total
    in xtrees/fold_order.mojo's fixed order (CHUNK partials from +0.0 in row
    order, then the pairwise TREE; it was one sequential chain). Returns 0;
    1 when an entry is not finite or is negative; 2 when the total is not
    positive (res untouched)."""
    if n <= 0:
        return 2
    var m = fold_chunks(n)
    var p = List[Float64](length=m, fill=0.0)
    for c in range(m):
        var run: Float64 = 0.0
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, n)):
            var v = Float64(w[unsafe_offset=i])
            if not (v >= 0 and v <= 1.7976931348623157e308):
                return 1
            run = run + v
        p[c] = run
    fold_tree_host(p, m, 1)
    var total = p[0]
    if not (total > 0):
        return 2
    for i in range(n):
        res[unsafe_offset=i] = Float64(w[unsafe_offset=i]) / total
    return 0


def iota_i32(res: MutPointer[Int32, MutUntrackedOrigin], n: Int):
    """res[i] = i."""
    for i in range(n):
        res[unsafe_offset=i] = Int32(i)


def fill_class_major_f64(
    inits: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int, res: MutPointer[Float64, MutUntrackedOrigin],
):
    """res[c * n + i] = inits[c] for c < k, i < n (a word copy)."""
    for c in range(k):
        for i in range(n):
            res[unsafe_offset=c * n + i] = inits[unsafe_offset=c]
