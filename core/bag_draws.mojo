# SPDX-License-Identifier: Apache-2.0
"""Bagging's rows as scikit-learn 1.9.1 draws and hands them to a member (lane w3f3-h, 2026-11-24).

scikit-learn 1.9.1 (sklearn/ensemble/_bagging.py):
  * :82-93 `_generate_bagging_indices`: without sample_weight the rows are drawn uniformly; with it,
    `random_state.choice(n_samples, max_samples, replace=bootstrap, p=sample_weight / sum(sample_weight))`.
  * :467 / sklearn/ensemble/_bootstrap.py:44-59 `_get_n_samples_bootstrap`: a float max_samples is a fraction of
    `sample_weight.sum()` when weighted (`max(int(max_samples * sum), 1)`), and :468-472 refuses more than n rows
    without replacement.
  * :164-172: a member that takes sample_weight (every tree member Bagging admits) is fitted on the WHOLE y with
    `sample_weight = np.bincount(indices, minlength=n_samples)` (the caller's sample_weight only drove the draw),
    so its classes_ is every class of y. Only a member that takes no sample_weight is fitted on the gathered rows.

Both Bagging routes call this module: the public door (bindings/trees_public_boost_door.mojo `_bagging_fit_bags`)
and the META member (core/ensemble_member.mojo `_fit_bag`). The draws run on the host, one law on every column
(the weighted draw with replacement is `xtrees/ops.mojo::weighted_sample`, the CPU column's spelling of the device
one; the draw without replacement is the exponential-clock law below, an order-free selection). No Python.
"""
from std.memory import bitcast
from checks.numerics import identical_log64
from xtrees.ops import stream_base, draw, unit, weighted_sample
from core.host_parallel import host_parallelize

#: Rows per task of the clocks (each row's clock is independent: any split gives the same keys).
comptime _CLOCK_CHUNK = 4096


struct BagWeights(Movable):
    """The caller's sample weights as the draws read them: float64 words,
    their total (fixed index order) and how many are positive."""
    var w64: List[Float64]
    var total: Float64
    var positive: Int

    def __init__(out self):
        self.w64 = List[Float64]()
        self.total = 0.0
        self.positive = 0


def bag_weights(w: Int, n: Int) raises -> BagWeights:
    """The float32 weights at `w` (n words) widened once per fit. Refused in
    numpy's `choice(p=...)` words: a negative weight, a zero total."""
    var out = BagWeights()
    out.w64 = List[Float64](length=max(n, 1), fill=0.0)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=w)
    var total = 0.0
    var positive = 0
    for i in range(n):  # native data loop (one pass over the fit's weights, once per fit)
        var value = Float64(src[i])
        if not (value >= 0.0):
            raise Error("ValueError: probabilities are not non-negative")
        out.w64[i] = value
        total += value
        if value > 0.0:
            positive += 1
    if not (total > 0.0):
        raise Error("ValueError: probabilities contain NaN")
    out.total = total
    out.positive = positive
    return out^


def bag_weighted_count(fraction: Float64, total: Float64, n: Int, bootstrap: Bool) raises -> Int:
    """A float max_samples of a weighted fit: `max(int(fraction * sum(w)), 1)`
    rows, refused without replacement past n (scikit-learn's words)."""
    var k = max(Int(fraction * total), 1)
    if not bootstrap and k > n:
        raise Error("ValueError: Effective max_samples=" + String(k) + " must be <= n_samples=" + String(n)
                    + " to be able to sample without replacement.")
    return k


def bag_draw_weighted(weights: BagWeights, n: Int, dst: Int, n_draw: Int, replace: Bool, seed: Int,
                      stream: Int) raises:
    """`n_draw` int32 row numbers into `dst`, row i drawn with probability
    w[i] / sum(w) (numpy's `choice(n, n_draw, replace, p)` question). With
    replacement: `weighted_sample` (the cdf law). Without: each positive
    row's exponential clock e_i = -log(u_i) / w_i (u_i in (0, 1] from the
    row's own counter draw), the n_draw smallest (e_i, i) taken, written in
    ascending row order (successive weighted sampling without replacement,
    Efraimidis-Spirakis). The clocks are independent and the selection is
    the radix select `xtrees/ops.mojo::sample_indices` uses: order-free."""
    var res = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst)
    if n_draw <= 0:
        return
    if replace:
        weighted_sample(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(weights.w64.unsafe_ptr())),
                        n, res, n_draw, seed, stream)
        return
    if n_draw > weights.positive:
        raise Error("ValueError: Fewer non-zero entries in p than size")
    var base = stream_base(seed, stream)
    var keys = List[UInt64](length=n, fill=~UInt64(0))
    var keys_at = Int(keys.unsafe_ptr())
    var w_at = Int(weights.w64.unsafe_ptr())
    var chunks = (n + _CLOCK_CHUNK - 1) // _CLOCK_CHUNK

    def _clocks(c: Int) {imm keys_at, imm w_at, imm base, imm n}:
        var dst = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=keys_at)
        var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=w_at)
        for i in range(c * _CLOCK_CHUNK, min((c + 1) * _CLOCK_CHUNK, n)):  # small-loop(one chunk of row clocks)
            var wi = w[i]
            if wi > 0.0:
                var u = 1.0 - unit(draw(base, i))
                var e = -identical_log64(u) / wi
                if not (e > 0.0):
                    e = 0.0  # u == 1: the earliest clock (never -0.0, whose bits sort last)
                dst[i] = UInt64(bitcast[DType.uint64](e))

    host_parallelize(_clocks, chunks)
    # The n_draw-th smallest key K* and how many of its ties to take, by
    # 8-bit digits from the top (positive float64 bits sort as the values).
    var prefix = UInt64(0)
    var want = n_draw
    var shift = 56
    while shift >= 0:
        var hist = List[Int](length=256, fill=0)
        var sh = UInt64(shift)
        for i in range(n):  # native data loop (one digit histogram pass)
            var key = keys[i]
            var same = True
            if shift < 56:
                same = (key >> (sh + 8)) == (prefix >> (sh + 8))
            if same:
                hist[Int((key >> sh) & UInt64(0xFF))] += 1
        var cum = 0
        for b in range(256):  # small-loop(the 256 digit buckets)
            if cum < want and want <= cum + hist[b]:
                prefix |= UInt64(b) << sh
                want -= cum
                break
            cum += hist[b]
        shift -= 8
    var k = 0
    var ties = 0
    for i in range(n):  # native data loop (the selected rows, ascending)
        var key = keys[i]
        var take = key < prefix
        if key == prefix:
            take = ties < want
            ties += 1
        if take:
            res[k] = Int32(i)
            k += 1
    _ = len(keys)


def bag_counts_into(dst: Int, draw_at: Int, n_draw: Int, n: Int, class_rows: Int):
    """The member's sample_weight into the n zeroed float32 words at `dst`:
    `np.bincount(draw, minlength=n)` (exact), times the tree's own
    class_weight rows at `class_rows` (0: none) as the tree's fit multiplies
    them (one float32 product per row). One unit per call: callers run the
    units on the host cores."""
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst)
    var picks = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=draw_at)
    for i in range(n_draw):  # small-loop(one pass over one unit's draw)
        out[Int(picks[i])] += 1.0
    if class_rows != 0:
        var cw = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=class_rows)
        for r in range(n):  # small-loop(one pass over one unit's rows)
            out[r] = out[r] * cw[r]


def bag_counts_units(dst_at: List[Int], draw_at: List[Int], draw_n: List[Int], n: Int, class_rows: Int):
    """`bag_counts_into` for every unit u (its n words at `dst_at[u]`, its
    draw at `draw_at[u]`, `draw_n[u]` rows), the units on the host cores."""
    var dsts = Int(dst_at.unsafe_ptr())
    var draws = Int(draw_at.unsafe_ptr())
    var counts = Int(draw_n.unsafe_ptr())

    def _unit(u: Int) {imm dsts, imm draws, imm counts, imm n, imm class_rows}:
        bag_counts_into(MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=dsts)[u],
                        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=draws)[u],
                        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=counts)[u], n, class_rows)

    host_parallelize(_unit, len(dst_at))
    _ = len(dst_at)  # read through the tasks above
    _ = len(draw_at)
    _ = len(draw_n)

