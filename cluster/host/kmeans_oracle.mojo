# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""k-means TRAINING on the host, for a box with no GPU (workstream E batch 2,
the kmeans lane, 2026-09-14).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and no GPU binding imports this file. Every kernel of the fit is spelled a
SECOND time here from the device source, with the same arithmetic order, in the
order `cluster/estimator.mojo::kmeans_fit` then
`cluster/impl/kmeans.mojo::fit_predict` reach them at the shipped default
(k-means|| init, L2 expanded, `n_init` 1, `inertia_check` off). The
arithmetic leaves are `checks/numerics.mojo`'s (`ftz`, `identical_mul_add`,
`identical_sqrt`), the fixed-point scale is `checks/fixed_point.mojo`'s own
`choose_scale` (host code with no import at all), and the fold widths are
read from `checks/kernel_matrix.mojo` the way the kernels read them.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `HostRngReplay`          `HostRng`, `cluster/impl/detail/kmeans.mojo:113`:
                           splitmix64, `next_index` is `u % n`, `next_unit`
                           is `Float64(u >> 11) * 2^-53`.
  `host_plan_sum_scale`    `plan_sum_scale`, `cluster/estimator.mojo:143`:
                           one sequential Float64 chain `column += abs(x)`
                           per column over the rows in order, the worst
                           column under a strict `>`, `choose_scale(worst,
                           n_samples)`. The host pool split there is
                           scheduling and reorders no chain.
  `host_row_norm`          `row_norm_kernel`, `core/row_norms.mojo:70`:
                           NORM_TPB strided lane chains `acc =
                           ftz(fma(v, v, acc))`, the halving tree of
                           `pinned_block_sum`, `ftz` of the total, the
                           clamp and `identical_sqrt` when asked.
  `host_assign`            `fused_distance_nn_kernel`,
                           `cluster/impl/distance/fused_distance_nn/
                           simt_kernel.mojo:246`: per cell one ascending
                           chain `acc = ftz(fma(ftz(x), ftz(y), acc))` over
                           the features (the k tiles continue the chain and
                           a zero-padded tail adds exactly nothing), the
                           epilog `ftz(fma(-2, ftz(acc), ftz(ftz(xn) +
                           ftz(yn))))`, the positivity clamp and the
                           self-neighbor guard (`d * d < 1e-6 and xn == yn`
                           on the RAW norms), then `raft::argmin_op`'s total
                           order (lowest value, then lowest key), a
                           selection no fold shape can move. The `sqrt` of
                           the L2-sqrt metric is `identical_sqrt` on the
                           kernel's final write (DEVIATION 2715); the
                           Python surface reaches it through
                           `metric="l2_sqrt_expanded"` (the kmeans-sqrt
                           lane).
  `host_sum_device`        `_sum_device`, `kmeans.mojo:157`:
                           `sum_partials_kernel` over `min(256, ceil(n /
                           TPB))` blocks, each lane's grid-strided chain
                           `acc = ftz(acc + v)` with the map inside
                           (`ftz(ftz(v) * ftz(b))`, or the flushed square
                           of the flushed difference), the halving tree per
                           block, then `finish_sum_kernel`'s lane chains
                           over the partials and one more halving tree.
  `host_inclusive_scan`    the three-stage device scan of
                           `cluster/checks/plus_plus.mojo` (`chunk_sums_
                           kernel`, `scan_chunk_offsets_kernel`, `write_
                           inclusive_scan_kernel`): PLUS_PLUS_TPB-wide
                           chunks folded by the halving tree, the chunk
                           offsets by one exclusive block scan, the chunk
                           interiors by one inclusive block scan each, then
                           `(base + carry) + inc`. THE BLOCK SCAN IS THE
                           LIBRARY'S, `max.gpu.primitives.block.prefix_sum`
                           (max/v26.5.0, max/mojo/max/gpu/primitives/
                           block.mojo:672 and mojo/stdlib/std/gpu/
                           primitives/warp.mojo:1084): a Hillis-Steele scan
                           inside each warp (`res += shuffle_up(res, 1, 2,
                           4, ...)`), the warp totals scanned by warp 0 the
                           same way, and the previous warps' inclusive
                           prefix added. That shape depends on the warp
                           width, 32 on Apple and NVIDIA and 64 on AMD's
                           CDNA3; `host_block_prefix_sum` replays the
                           32-wide shape. The scan feeds `binary_search_
                           kernel` alone, and the three 2026-09-14 GPU
                           columns agree on every k-means cell, so the
                           64-wide AMD scan differs from the 32-wide one in
                           no bit that reaches a draw on those fixtures;
                           this file claims the 32-wide bits and says so.
  `host_binary_search`     `binary_search_kernel`, `plus_plus.mojo:266`:
                           `target = u * csum[n - 1]`, the lower bound
                           under a strict `<`, clamped to `n - 1`.
  `host_kmeans_plus_plus`  `kmeans_plus_plus`, `kmeans.mojo:242`: `n_trials
                           = 2 + ceil(log(k))`, the first centroid from
                           `next_index`, `n_trials` uniforms per pick
                           narrowed to Float32, the scan and the search, the
                           gathered rows and their norms, `gemm_nt` (the
                           pinned cell, `core/classical_host_predict.mojo::
                           host_gemm_nt`), `candidate_cost_kernel`'s lane
                           chains `acc = ftz(acc + min(d, cur))` and halving
                           tree, the greedy argmin in Float64 under a
                           strict `<`, `adopt_candidate_min_kernel`.
  `host_init_scalable`     `init_scalable_kmeans_plus_plus`, `kmeans.mojo:
                           470`: the 2^24-row refusal in its words, the
                           flagged seed point, `psi` through
                           `host_sum_device`, `min(8, ceil(log(psi)))`
                           rounds of `sample_flags_kernel` (`scalable_
                           uniform`, the counter hash of the round seed and
                           the index, `scalable_keep`'s `lk * dist / psi >
                           u`), the flag scan and the stable scatter, the
                           float count histogram, then step 8: the classic
                           k-means++ over the candidates and the weighted
                           Lloyd pass under fresh defaults with the inner
                           scales chosen as there (`choose_scale(worst)`
                           with no row count, `choose_scale(n_samples)`).
  `host_init_random`       `init_random`, `kmeans.mojo:198`: `next_index`
                           draws with rejection of a repeat, row copies.
  `host_fit_main`          `kmeans_fit_main_traced`, `kmeans.mojo:929`:
                           `params.validate()` in its words, the row norms
                           once, the restart loop (`n_init` collapses to 1
                           on an explicit start), the Lloyd iteration in
                           cuVS's order (assign, accumulate, finalize,
                           shift, copy back, test), the quantized Int32
                           scatter-add `Int32(x * w * scale)` whose sum is
                           order independent (`cluster/checks/reduce_by_
                           key.mojo`), `finalize_centroids_kernel`'s two
                           flushed quotients and the empty-cluster keep,
                           the shift as `host_sum_device` SQDIFF, `shift <
                           tol` on the host, the post-loop assignment and
                           the weighted inertia, the best restart kept.
  `host_kmeans_fit`        `kmeans_fit`, `cluster/estimator.mojo:212`, then
                           `fit_predict`, `cluster/impl/kmeans.mojo:141`:
                           the guards in their words, the two scales, the
                           unit or supplied weights, the fit, then the
                           FRESH assignment against the returned centroids
                           with the estimator's own row norms (squared for
                           both L2 metrics, rooted for cosine alone,
                           DEVIATION 2716).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` makes every quantized
centroid-sum cell carry ONE EXTRA UNIT (`q + 1` in `host_accumulate`), so
every finalized centroid moves by `1 / sum_scale` per coordinate on every
iteration and the `centers` hash of every fixture moves; a chain walked in
another order would not reliably move an argmin or an Int32 sum, which is
why this family's arm is a unit and not an order. Read back by
`core_host_sabotage`.

That arm is in `host_accumulate`, which ONLY THE FIT walks. It therefore
says nothing about a SAVED model's `predict` and `transform`, which is what
`bench/results/classical_host/` records and what a CPU-only install actually
runs. The arms for that side are `KMEANS_PREDICT_HOST_SABOTAGE` (one bit into
every predicted label AND one unit into every transform cell) and
`KMEANS_TRANSFORM_HOST_SABOTAGE` (the transform half alone, which the family
define also selects); see their comments for why the label arm has a define
of its own.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the kmeans lane is the measurement.
"""
from std.math import fma
from std.math import ceil, log
from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined

from checks.fixed_point import choose_scale
from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_NVIDIA,
    K_LIB_PLUS_PLUS,
    K_LIB_REDUCE_BY_KEY,
    K_LIB_ROW_NORM,
    TARGET_COLUMN,
    lib_block_size_for,
)
from checks.numerics import ftz, identical_mul_add, identical_sqrt
from core.classical_host_predict import host_gemm_nt


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime KMEANS_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: THE SAVED-MODEL NEGATIVE CONTROL (lane/classical-host-recordings,
#: 2026-09-16). `KMEANS_ORACLE_HOST_SABOTAGE` above sits in `host_accumulate`,
#: which only the FIT walks, so it cannot move a saved model's `predict` or
#: `transform` by one bit. That was measured, not assumed: the k-means
#: recording's `check --expect-mismatch --every-fixture` built with
#: `-D MOJOLEARN_HOST_SABOTAGE=1` read `SABOTAGE NOT CAUGHT ON FIXTURES` with
#: ALL 54 of them in `unmoved`, which is a gate that could not fail dressed as
#: a gate that passed.
#:
#: THERE ARE TWO ARMS HERE AND THEY ARE NOT THE SAME DEFINE, FOR A REASON THAT
#: WAS ALSO MEASURED. `tools/identity_break.py`'s `_km_probe` asserts, before
#: it hashes anything, that `predict` over the TRAINING rows is `labels_` bit
#: for bit. An arm that moves `predict` therefore makes that assertion raise,
#: and the lane's infer and model cells read REFUSED rather than DIVERGENT.
#: A first version of this control ORed the family define into the label arm
#: and turned lane/kmeans-save's clean `DIVERGENT=18` into
#: `ONE-COLUMN ... REFUSED=12`, which is precisely the reading that lane was
#: honest about not being a control at all.
#:
#:  * `MOJOLEARN_KMEANS_PREDICT_SABOTAGE` moves BOTH: one bit into every
#:    label and one unit into every transform cell. It is the strongest arm
#:    and it is what a saved-model gate should be run against.
#:  * the FAMILY define, `MOJOLEARN_HOST_SABOTAGE`, moves the TRANSFORM half
#:    only. That is enough for every recorded cell to move, because the cell
#:    hashes the (predict, transform) pair, and it leaves `predict` equal to
#:    `labels_` so the identity harness's own assertion still holds and its
#:    cells read DIVERGENT rather than REFUSED.
#:
#: One unit, not an order, in both: an argmin walked in another order need not
#: move, which is why this family's arms are units
#: (`core/labeled_reference_host_predict.mojo` uses the same shape for the
#: transductive predicts).
comptime KMEANS_PREDICT_HOST_SABOTAGE = is_defined[
    "MOJOLEARN_KMEANS_PREDICT_SABOTAGE"
]()
comptime KMEANS_TRANSFORM_HOST_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]() or KMEANS_PREDICT_HOST_SABOTAGE
)

#: The three fold widths, read as the kernels read them. Each is a
#: classified float fold (`lib_block_bounds_a_float_fold`), so under
#: IDENTICAL every column resolves it to the same width; the asserts in
#: `host_kmeans_fit` hold that against the three GPU columns.
comptime NORM_TPB = lib_block_size_for[K_LIB_ROW_NORM, TARGET_COLUMN]()
comptime REDUCE_BY_KEY_TPB = lib_block_size_for[K_LIB_REDUCE_BY_KEY, TARGET_COLUMN]()
comptime PLUS_PLUS_TPB = lib_block_size_for[K_LIB_PLUS_PLUS, TARGET_COLUMN]()

#: The warp width the library block scan is replayed at (module docstring,
#: `host_inclusive_scan`).
comptime SCAN_WARP = 32

#: `sum_partials_kernel`'s block cap (`_sum_device`, `kmeans.mojo:181`).
comptime SUM_PARTIAL_BLOCKS = 256

#: `fused_distance_nn_kernel`'s identity element and clamp precision.
comptime FUSED_MAX = Float32(3.4028234663852886e38)
comptime FUSED_CLAMP_PRECISION = Float32(1.0e-6)

comptime SUM_MODE_PLAIN = 0
comptime SUM_MODE_PRODUCT = 1
comptime SUM_MODE_SQDIFF = 2

comptime INIT_KMEANS_PLUS_PLUS = 0
comptime INIT_RANDOM = 1
comptime INIT_ARRAY = 2

comptime METRIC_L2_EXPANDED = 0
comptime METRIC_L2_SQRT_EXPANDED = 1

#: `KMeansParams.default()`, the fields the fit reads (`kmeans_params.mojo`).
comptime DEFAULT_MAX_ITER = 300
comptime DEFAULT_TOL = Float64(1e-4)
comptime DEFAULT_OVERSAMPLING = Float64(2.0)

#: `init_scalable_kmeans_plus_plus`'s Float32 selection scan bound.
comptime SCALABLE_ROW_LIMIT = 1 << 24


# ===========================================================================
# THE STAGE CARD (`core/identity_trace.mojo`'s format, written by host
# code): MOJOLEARN_IDENTITY_TRACE=<path> makes the host fit emit the same
# tags the device fit emits (`fit.x_norm`, `restartNN.init.centroids`,
# `restartNN.iterMM.{centroid_norm,labels,min_dist,sums_i32,weight_i32,
# new_centroids,shift}`, `restartNN.final.{labels,centroids,inertia}`,
# `fit.{centroids,labels}`, the k-means|| recluster under
# `restart00.init.par.`), each an FNV-1a64 over the little-endian bytes of
# the buffer, so tools/identity_trace_diff.py can name the FIRST stage on
# which a host fit and a GPU fit disagree. One getenv per fit; every
# record returns on one boolean when unset.
# ===========================================================================

comptime TRACE_FNV_OFFSET: UInt64 = 0xCBF29CE484222325
comptime TRACE_FNV_PRIME: UInt64 = 0x100000001B3


def _trace_hex16(v: UInt64) -> String:
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(16):
        var nib = Int((v >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _trace_pad2(v: Int) -> String:
    """`_pad2`, `kmeans.mojo`."""
    if v < 10:
        return String("0") + String(v)
    return String(v)


struct KMeansHostTrace(Movable):
    var enabled: Bool
    var path: String
    var seq: Int

    def __init__(out self):
        self.path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
        self.enabled = self.path != ""
        self.seq = 0

    def header(mut self, what: String) raises:
        if not self.enabled:
            return
        with open(self.path, "a") as fh:
            fh.write(String("# ") + what + "\n")

    def _emit(mut self, tag: String, dtype: String, count: Int, h: UInt64) raises:
        if self.seq == 0:
            with open(self.path, "a") as vh:
                vh.write("# format: mojolearn-identity-trace v1\n")
        var line = (
            String(self.seq) + "\t" + tag + "\t" + dtype + "\t"
            + String(count) + "\t" + _trace_hex16(h) + "\n"
        )
        with open(self.path, "a") as fh:
            fh.write(line)
        self.seq += 1

    def _record_words(
        mut self, tag: String, dtype: String, words: List[UInt32]
    ) raises:
        var h = TRACE_FNV_OFFSET
        for i in range(len(words)):
            var u = words[i]
            for b in range(4):
                h = (h ^ UInt64((u >> UInt32(8 * b)) & UInt32(0xFF))) * TRACE_FNV_PRIME
        self._emit(tag, dtype, len(words), h)

    def record_f32(mut self, tag: String, v: List[Float32]) raises:
        if not self.enabled:
            return
        var words = List[UInt32]()
        for i in range(len(v)):
            words.append(bitcast[DType.uint32](v[i]))
        self._record_words(tag, String("f32"), words)

    def record_u32(mut self, tag: String, v: List[UInt32]) raises:
        if not self.enabled:
            return
        self._record_words(tag, String("u32"), v.copy())

    def record_i32(mut self, tag: String, v: List[Int32]) raises:
        if not self.enabled:
            return
        var words = List[UInt32]()
        for i in range(len(v)):
            words.append(bitcast[DType.uint32](v[i]))
        self._record_words(tag, String("i32"), words)

    def record_scalar_f32(mut self, tag: String, v: Float32) raises:
        if not self.enabled:
            return
        var one = List[Float32]()
        one.append(v)
        self.record_f32(tag, one)


@fieldwise_init
struct HostRngReplay(Copyable, Movable):
    """`HostRng`, `cluster/impl/detail/kmeans.mojo:113`, verbatim."""

    var state: UInt64

    def next_u64(mut self) -> UInt64:
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def next_index(mut self, n: Int) -> Int:
        return Int(self.next_u64() % UInt64(n))

    def next_unit(mut self) -> Float64:
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)


def _halving[block: Int](mut red: List[Float32]) -> Float32:
    """`pinned_block_sum`'s IDENTICAL arm (`core/pinned_reduce.mojo`): the
    halving tree `red[t] += red[t + step]`, `step = block/2 .. 1`."""
    var step = block // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return red[0]


def host_row_norm(
    a: List[Float32], row: Int, d: Int, take_sqrt: Bool
) -> Float32:
    """`row_norm_kernel` for one row (module docstring)."""
    var red = List[Float32](length=NORM_TPB, fill=Float32(0.0))
    for t in range(NORM_TPB):
        var acc = Float32(0.0)
        var col = t
        while col < d:
            var v = ftz(a[row * d + col])
            acc = ftz(identical_mul_add(v, v, acc))
            col += NORM_TPB
        red[t] = acc
    var total = ftz(_halving[NORM_TPB](red))
    if take_sqrt:
        if total <= Float32(0.0):
            total = Float32(0.0)
        total = ftz(identical_sqrt(total))
    return total


def host_row_norms(
    a: List[Float32], n_rows: Int, d: Int, take_sqrt: Bool
) -> List[Float32]:
    var out = List[Float32](length=n_rows, fill=Float32(0.0))
    for row in range(n_rows):
        out[row] = host_row_norm(a, row, d, take_sqrt)
    return out^


def host_norms_take_sqrt(metric: Int) -> Bool:
    """`centroid_norms_take_sqrt`, `kmeans_common.mojo`, and the same answer.

    DO NOT SIMPLIFY THIS AWAY, AND DO NOT INVERT IT. It is False for every
    metric this tree admits. It is a named function rather than a literal
    `False` at each of its call sites because the flag it feeds is exactly
    where a measured defect already lived, and the measurement has to travel
    with it.

    DEVIATION 2716. The flag used to follow `metric_is_sqrt`, so under
    L2SqrtExpanded the final assignment computed `||x|| + ||c||^2 - 2 x.c`:
    the row constant was wrong, the value went negative wherever `||x||^2`
    exceeded `||x||`, the clamp made it 0, and the tie went to the lowest
    key. Measured on the M4 at 1eea14f80: 9,675 of 20,000 `labels_` on the
    `wide` fixture and 4 on `base` were not the argmin to the returned
    centers, ON EVERY COLUMN ALIKE, so the record read IDENTICAL on a wrong
    answer. No identity check can catch that; only an argmin check can.

    The rooted arm existed for a cosine metric, which divides by the norms.
    That metric was deleted on 2026-09-18 (lane/kmeans-cosine-capability).
    Both surviving metrics want SQUARED norms; the root belongs to the
    reduction's OUTPUT alone, which is `host_metric_is_sqrt`.
    """
    return False


def host_metric_is_sqrt(metric: Int) -> Bool:
    """`metric_is_sqrt`, `kmeans_common.mojo`."""
    return metric != METRIC_L2_EXPANDED


def host_assign(
    x: List[Float32],
    n: Int,
    x_norm: List[Float32],
    c: List[Float32],
    k: Int,
    c_norm: List[Float32],
    d: Int,
    is_sqrt: Bool,
    mut labels: List[UInt32],
    mut min_dist: List[Float32],
):
    """`min_cluster_and_distance_compute` through the fused kernel (module
    docstring): `labels[row]`, `min_dist[row]` for every row."""
    for row in range(n):
        var val = FUSED_MAX
        var key = UInt32(0xFFFFFFFF)
        var xn = x_norm[row]
        for col in range(k):
            var acc = Float32(0.0)
            for p in range(d):
                acc = ftz(
                    identical_mul_add(ftz(x[row * d + p]), ftz(c[col * d + p]), acc)
                )
            var yn = c_norm[col]
            var dist = ftz(
                identical_mul_add(
                    Float32(-2.0), ftz(acc), ftz(ftz(xn) + ftz(yn))
                )
            )
            if dist <= Float32(0.0) or (
                dist * dist < FUSED_CLAMP_PRECISION and xn == yn
            ):
                dist = Float32(0.0)
            if dist < val or (dist == val and UInt32(col) < key):
                val = dist
                key = UInt32(col)
        if is_sqrt:
            # DEVIATION 2715: the kernel's root is `identical_sqrt` now. The
            # host libm root is correctly rounded too, so this moves no bit;
            # it keeps the host spelling the same as the kernel's.
            val = identical_sqrt(val)
        min_dist[row] = val
        labels[row] = key


def host_checked_label(labels: List[UInt32], row: Int, k: Int) raises -> Int:
    """`labels[row]` as an index into `k` clusters, or a raised error.

    `host_assign` leaves its sentinel 0xFFFFFFFF when no distance of the
    row compares below FUSED_MAX: a NaN or an infinity from an input, a
    centroid or an overflowing square. Used as an index that label read
    past the end of a `k` or `k x d` list, and the bounds assert ABORTED
    the whole process (gate run 35636551982, spectral-precomputed under
    the sabotage host set: index 17179869180 = 4 x 0xFFFFFFFF). Valid
    inputs always produce a label below `k`, so this moves no answer; it
    turns the abort into an error the Python caller sees and records."""
    var label = Int(labels[row])
    if label < 0 or label >= k:
        raise Error(
            "kmeans host: row "
            + String(row)
            + " has no nearest centroid (label "
            + String(label)
            + " of "
            + String(k)
            + "); a distance was not finite, so the input or the centroids"
            " carry a NaN, an infinity or an overflowing value"
        )
    return label


def host_sum_device(
    a: List[Float32], b: List[Float32], n: Int, mode: Int
) -> Float32:
    """`_sum_device` (module docstring). `b` is read on the PRODUCT and
    SQDIFF modes only."""
    var blocks = (n + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB
    if blocks > SUM_PARTIAL_BLOCKS:
        blocks = SUM_PARTIAL_BLOCKS
    if blocks < 1:
        blocks = 1
    var stride = blocks * REDUCE_BY_KEY_TPB
    var partials = List[Float32](length=blocks, fill=Float32(0.0))
    for blk in range(blocks):
        var red = List[Float32](length=REDUCE_BY_KEY_TPB, fill=Float32(0.0))
        for t in range(REDUCE_BY_KEY_TPB):
            var acc = Float32(0.0)
            var i = blk * REDUCE_BY_KEY_TPB + t
            while i < n:
                var v = a[i]
                if mode == SUM_MODE_PRODUCT:
                    v = ftz(ftz(v) * ftz(b[i]))
                elif mode == SUM_MODE_SQDIFF:
                    var dd = ftz(ftz(v) - ftz(b[i]))
                    v = ftz(dd * dd)
                acc = ftz(acc + v)
                i += stride
            red[t] = acc
        partials[blk] = _halving[REDUCE_BY_KEY_TPB](red)
    var red2 = List[Float32](length=REDUCE_BY_KEY_TPB, fill=Float32(0.0))
    for t in range(REDUCE_BY_KEY_TPB):
        var acc = Float32(0.0)
        var i = t
        while i < blocks:
            acc = ftz(acc + partials[i])
            i += REDUCE_BY_KEY_TPB
        red2[t] = acc
    return _halving[REDUCE_BY_KEY_TPB](red2)


def host_block_prefix_sum(
    vals: List[Float32], exclusive: Bool
) -> List[Float32]:
    """`max.gpu.primitives.block.prefix_sum[block_size=PLUS_PLUS_TPB]` at a
    32-wide warp (module docstring, `host_inclusive_scan`). `vals` holds
    one value per lane of the block."""
    comptime n_warps = PLUS_PLUS_TPB // SCAN_WARP
    var res = vals.copy()
    for w in range(n_warps):
        var base = w * SCAN_WARP
        # warp.prefix_sum: `res += shuffle_up(res, offset)` for lanes at
        # or past the offset, offsets 1, 2, 4, ...; every lane reads the
        # previous step's values.
        var offset = 1
        while offset < SCAN_WARP:
            var snap = List[Float32](length=SCAN_WARP, fill=Float32(0.0))
            for l in range(SCAN_WARP):
                snap[l] = res[base + l]
            for l in range(SCAN_WARP):
                if l >= offset:
                    res[base + l] = snap[l] + snap[l - offset]
            offset *= 2
        if exclusive:
            var snap2 = List[Float32](length=SCAN_WARP, fill=Float32(0.0))
            for l in range(SCAN_WARP):
                snap2[l] = res[base + l]
            for l in range(SCAN_WARP):
                if l == 0:
                    res[base] = Float32(0.0)
                else:
                    res[base + l] = snap2[l - 1]
    # Step 2: the last lane of each warp stores the warp's INCLUSIVE sum.
    var warp_mem = List[Float32](length=SCAN_WARP, fill=Float32(0.0))
    for w in range(n_warps):
        var last = w * SCAN_WARP + SCAN_WARP - 1
        var inclusive_warp_sum = res[last]
        if exclusive:
            inclusive_warp_sum += vals[last]
        warp_mem[w] = inclusive_warp_sum
    # Step 3: warp 0 scans the warp sums, inclusive, the same way. Lanes
    # past `n_warps` hold nothing a lower lane can read.
    var offset2 = 1
    while offset2 < SCAN_WARP:
        var snap3 = warp_mem.copy()
        for l in range(SCAN_WARP):
            if l >= offset2:
                warp_mem[l] = snap3[l] + snap3[l - offset2]
        offset2 *= 2
    # Step 4: add the previous warps' prefix.
    for w in range(1, n_warps):
        for l in range(SCAN_WARP):
            res[w * SCAN_WARP + l] = res[w * SCAN_WARP + l] + warp_mem[w - 1]
    return res^


def host_inclusive_scan(a: List[Float32], n: Int) -> List[Float32]:
    """The three-stage device scan (module docstring): `csum[i]` is the
    running total of `a` up to and including `i`, in the device's bits."""
    var chunk = PLUS_PLUS_TPB
    var n_chunks = (n + chunk - 1) // chunk
    # chunk_sums_kernel
    var totals = List[Float32](length=n_chunks, fill=Float32(0.0))
    for c in range(n_chunks):
        var begin = c * chunk
        var end = begin + chunk
        if end > n:
            end = n
        var red = List[Float32](length=PLUS_PLUS_TPB, fill=Float32(0.0))
        for t in range(PLUS_PLUS_TPB):
            var acc = Float32(0.0)
            var i = begin + t
            while i < end:
                acc += a[i]
                i += PLUS_PLUS_TPB
            red[t] = acc
        totals[c] = _halving[PLUS_PLUS_TPB](red)
    # scan_chunk_offsets_kernel: one block, a per-lane slice of the totals,
    # an exclusive block scan of the slice sums, the running write.
    var per = (n_chunks + PLUS_PLUS_TPB - 1) // PLUS_PLUS_TPB
    var lane_sum = List[Float32](length=PLUS_PLUS_TPB, fill=Float32(0.0))
    for t in range(PLUS_PLUS_TPB):
        var begin = t * per
        if begin > n_chunks:
            begin = n_chunks
        var end = begin + per
        if end > n_chunks:
            end = n_chunks
        var s = Float32(0.0)
        for i in range(begin, end):
            s += totals[i]
        lane_sum[t] = s
    var lane_offset = host_block_prefix_sum(lane_sum, True)
    var offsets = List[Float32](length=n_chunks, fill=Float32(0.0))
    for t in range(PLUS_PLUS_TPB):
        var begin = t * per
        if begin > n_chunks:
            begin = n_chunks
        var end = begin + per
        if end > n_chunks:
            end = n_chunks
        var running = lane_offset[t]
        for i in range(begin, end):
            offsets[i] = running
            running += totals[i]
    # write_inclusive_scan_kernel: one block per chunk, an inclusive block
    # scan per PLUS_PLUS_TPB step of the chunk, plus base and carry.
    var csum = List[Float32](length=n, fill=Float32(0.0))
    for c in range(n_chunks):
        var begin = c * chunk
        var base = offsets[c]
        var carry = Float32(0.0)
        var c0 = 0
        while c0 < chunk:
            var vals = List[Float32](length=PLUS_PLUS_TPB, fill=Float32(0.0))
            for t in range(PLUS_PLUS_TPB):
                var i = begin + c0 + t
                if i < n and c0 + t < chunk:
                    vals[t] = a[i]
            var inc = host_block_prefix_sum(vals, False)
            for t in range(PLUS_PLUS_TPB):
                var i = begin + c0 + t
                if i < n and c0 + t < chunk:
                    csum[i] = base + carry + inc[t]
            carry = carry + inc[PLUS_PLUS_TPB - 1]
            c0 += PLUS_PLUS_TPB
    return csum^


def host_binary_search(csum: List[Float32], n: Int, u: Float32) -> Int:
    """`binary_search_kernel` for one trial (module docstring)."""
    var total = csum[n - 1]
    var target = u * total
    var lo = 0
    var hi = n
    while hi > lo:
        var mid = (lo + hi) // 2
        if csum[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo >= n:
        lo = n - 1
    return lo


def _candidate_distance(
    z: List[Float32],
    i: Int,
    n_trials: Int,
    trial: Int,
    x_norm: List[Float32],
    cn: Float32,
) -> Float32:
    """The clamped expanded distance of `candidate_cost_kernel` and
    `adopt_candidate_min_kernel` (`plus_plus.mojo:50, 93`)."""
    var dd = ftz(
        identical_mul_add(
            Float32(-2.0),
            ftz(z[i * n_trials + trial]),
            ftz(ftz(x_norm[i]) + ftz(cn)),
        )
    )
    if dd <= Float32(0.0):
        dd = Float32(0.0)
    return dd


def host_kmeans_plus_plus(
    x: List[Float32],
    x_norm: List[Float32],
    n: Int,
    d: Int,
    k: Int,
    is_sqrt: Bool,
    mut centroids: List[Float32],
    mut rng: HostRngReplay,
):
    """`kmeans_plus_plus`, the classic sequential variant (module
    docstring). Writes `k` rows of `centroids`."""
    var n_trials = 2 + Int(ceil(log(Float64(k))))
    # Step 1.
    var first = rng.next_index(n)
    for p in range(d):
        centroids[p] = x[first * d + p]
    var c_norm = List[Float32](length=1, fill=Float32(0.0))
    c_norm[0] = host_row_norm(centroids, 0, d, False)
    var labels = List[UInt32](length=n, fill=UInt32(0))
    var min_dist = List[Float32](length=n, fill=Float32(0.0))
    host_assign(x, n, x_norm, centroids, 1, c_norm, d, is_sqrt, labels, min_dist)

    var picked = 1
    while picked < k:
        # Step 3: `n_trials` host uniforms, the device draw over d^2.
        var u01 = List[Float32](length=n_trials, fill=Float32(0.0))
        for t in range(n_trials):
            u01[t] = Float32(rng.next_unit())
        var csum = host_inclusive_scan(min_dist, n)
        var candidates = List[Float32](length=n_trials * d, fill=Float32(0.0))
        for t in range(n_trials):
            var sel = host_binary_search(csum, n, u01[t])
            for p in range(d):
                candidates[t * d + p] = x[sel * d + p]
        var cand_norm = host_row_norms(candidates, n_trials, d, False)
        var z = host_gemm_nt(x, candidates, n, n_trials, d)
        # candidate_cost_kernel: one block per trial, lanes stride the rows.
        var cost = List[Float32](length=n_trials, fill=Float32(0.0))
        for t in range(n_trials):
            var red = List[Float32](length=PLUS_PLUS_TPB, fill=Float32(0.0))
            for lane in range(PLUS_PLUS_TPB):
                var acc = Float32(0.0)
                var i = lane
                while i < n:
                    var dd = _candidate_distance(z, i, n_trials, t, x_norm, cand_norm[t])
                    var cur = min_dist[i]
                    acc = ftz(acc + (dd if dd < cur else cur))
                    i += PLUS_PLUS_TPB
                red[lane] = acc
            cost[t] = _halving[PLUS_PLUS_TPB](red)
        # Step 4, the greedy argmin on the host, in Float64.
        var best = 0
        var best_cost = Float64(cost[0])
        for t in range(1, n_trials):
            var cc = Float64(cost[t])
            if cc < best_cost:
                best_cost = cc
                best = t
        # adopt_candidate_min_kernel
        for i in range(n):
            var dd = _candidate_distance(z, i, n_trials, best, x_norm, cand_norm[best])
            if dd < min_dist[i]:
                min_dist[i] = dd
        for p in range(d):
            centroids[picked * d + p] = candidates[best * d + p]
        picked += 1


def host_round_seed_as_the_device_reassembles_it(round_seed: UInt64) -> UInt64:
    """THE DEVICE'S SEED, NOT THE HOST'S DRAW. `init_scalable_kmeans_plus_
    plus` hands `sample_flags_kernel` the round seed as two Int32 halves
    (`kmeans.mojo`, `seed_lo = round_seed.cast[uint32]().cast[int32]()`,
    `seed_hi = (round_seed >> 32)...`) and the kernel rebuilds
    `(hi.cast[uint32]().cast[uint64]() << 32) | lo.cast[uint32]().cast[uint64]()`
    (`scalable_init.mojo:73-75`). On every GPU column the low half's
    `int32 -> uint32 -> uint64` chain SIGN-EXTENDS, so a low half at or
    above 2^31 fills the high word with ones and the OR keeps them: the
    seed the device hashes is `(hi << 32) | sext64(lo)`. Measured on the
    Apple M4 (2026-09-14, the base fixture, seed 3): round 0's seed had a
    positive low half and the host's whole-seed hash selected the same 19
    rows; round 1's had a negative low half and the host selected 16
    different rows until this function spelled the device's value. The
    three GPU columns agree on every k-means cell, so the three vendors
    share the reassembly. A host with the whole draw is the one that is
    wrong here."""
    var lo = round_seed & UInt64(0xFFFFFFFF)
    var hi = round_seed >> 32
    var lo_ext = lo
    if lo >= UInt64(0x80000000):
        lo_ext = lo | UInt64(0xFFFFFFFF00000000)
    return (hi << 32) | lo_ext


def host_scalable_uniform(seed: UInt64, i: Int) -> Float32:
    """`scalable_uniform`, `cluster/checks/scalable_init.mojo:65`, over the
    seed the device reassembled (`host_round_seed_as_the_device_
    reassembles_it`)."""
    var z = seed + UInt64(i) * 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(z >> 40) * Float32(5.9604644775390625e-08)


def host_scalable_keep(
    dist: Float32, psi: Float32, lk: Float32, u: Float32
) -> Bool:
    """`scalable_keep`, `scalable_init.mojo:86`."""
    var prob = lk * dist / psi
    return prob > u


def host_init_random(
    x: List[Float32],
    n: Int,
    d: Int,
    n_clusters: Int,
    mut centroids: List[Float32],
    mut rng: HostRngReplay,
):
    """`init_random` (module docstring), into the first `n_clusters` rows
    of `centroids`."""
    var chosen = List[Int]()
    while len(chosen) < n_clusters:
        var candidate = rng.next_index(n)
        var seen = False
        for i in range(len(chosen)):
            if chosen[i] == candidate:
                seen = True
        if not seen:
            chosen.append(candidate)
    for j in range(n_clusters):
        for p in range(d):
            centroids[j * d + p] = x[chosen[j] * d + p]


def host_init_scalable(
    x: List[Float32],
    x_norm: List[Float32],
    n: Int,
    d: Int,
    k: Int,
    metric: Int,
    oversampling_factor: Float64,
    mut centroids: List[Float32],
    mut rng: HostRngReplay,
    mut trace: KMeansHostTrace,
    tag_prefix: String,
) raises:
    """`init_scalable_kmeans_plus_plus` (module docstring)."""
    if n >= SCALABLE_ROW_LIMIT:
        raise Error(
            "scalable k-means++ selection scan counts in Float32 and is"
            " exact only below 2^24 rows; got "
            + String(n)
        )
    var is_sqrt = host_metric_is_sqrt(metric)
    # Step 1: one uniform point, flagged.
    var c_idx = rng.next_index(n)
    var is_centroid = List[Int32](length=n, fill=Int32(0))
    is_centroid[c_idx] = Int32(1)
    var cand = List[Float32](length=d, fill=Float32(0.0))
    for p in range(d):
        cand[p] = x[c_idx * d + p]
    var cand_count = 1
    var labels = List[UInt32](length=n, fill=UInt32(0))
    var min_dist = List[Float32](length=n, fill=Float32(0.0))

    # Step 2: psi = phi_X(C), an UNWEIGHTED sum.
    var cand_norm = host_row_norms(cand, cand_count, d, False)
    host_assign(x, n, x_norm, cand, cand_count, cand_norm, d, is_sqrt, labels, min_dist)
    var psi = Float64(host_sum_device(min_dist, min_dist, n, SUM_MODE_PLAIN))

    var niter = 0
    if psi > 0.0:
        var lp = ceil(log(psi))
        if lp >= 8.0:
            niter = 8
        elif lp > 0.0:
            niter = Int(lp)

    # Step 3, each round against the WHOLE current candidate set.
    var flags = List[Float32](length=n, fill=Float32(0.0))
    for _iter in range(niter):
        cand_norm = host_row_norms(cand, cand_count, d, False)
        host_assign(x, n, x_norm, cand, cand_count, cand_norm, d, is_sqrt, labels, min_dist)
        psi = Float64(host_sum_device(min_dist, min_dist, n, SUM_MODE_PLAIN))
        # Step 4: one round seed, hashed per sample; flags, scan, scatter.
        var round_seed = host_round_seed_as_the_device_reassembles_it(
            rng.next_u64()
        )
        var lk = Float32(oversampling_factor * Float64(k))
        var psi32 = Float32(psi)
        for i in range(n):
            var u = host_scalable_uniform(round_seed, i)
            var keep = is_centroid[i] == Int32(0) and host_scalable_keep(
                min_dist[i], psi32, lk, u
            )
            flags[i] = Float32(1.0) if keep else Float32(0.0)
        var csum = host_inclusive_scan(flags, n)
        var n_selected = Int(csum[n - 1])
        if n_selected > 0:
            # Step 5: append in index order (`DeviceSelect::If`'s stability).
            var sel = List[UInt32](length=n_selected, fill=UInt32(0))
            for i in range(n):
                if flags[i] != Float32(0.0):
                    var pos = Int(csum[i]) - 1
                    sel[pos] = UInt32(i)
                    is_centroid[i] = Int32(1)
            for j in range(n_selected):
                var row = Int(sel[j])
                for p in range(d):
                    cand.append(x[row * d + p])
            cand_count += n_selected

    if cand_count > k:
        # Step 7: w_x = points nearest each candidate, a float histogram.
        var weight = List[Float32](length=cand_count, fill=Float32(0.0))
        cand_norm = host_row_norms(cand, cand_count, d, False)
        host_assign(x, n, x_norm, cand, cand_count, cand_norm, d, is_sqrt, labels, min_dist)
        for i in range(n):
            var lab = host_checked_label(labels, i, cand_count)
            weight[lab] = weight[lab] + Float32(1.0)
        # Step 8: UNWEIGHTED classic k-means++ over the candidates under the
        # outer params, then Lloyd over the weighted candidates under fresh
        # defaults with only `n_clusters` copied.
        var cand_norm_rows = host_row_norms(cand, cand_count, d, False)
        host_kmeans_plus_plus(
            cand, cand_norm_rows, cand_count, d, k, is_sqrt, centroids, rng
        )
        var worst = Float64(0.0)
        for f in range(d):
            var column = Float64(0.0)
            for c in range(cand_count):
                # explicit fma: the default build fused this product into the sum (lane/pinned-mul-contract-free)
                column = fma(Float64(abs(cand[c * d + f])), Float64(weight[c]), column)
            if column > worst:
                worst = column
        var inner_sum_scale = Float32(choose_scale(worst))
        var inner_weight_scale = Float32(choose_scale(Float64(n)))
        var cand_labels = List[UInt32](length=cand_count, fill=UInt32(0))
        _ = host_fit_main(
            cand,
            cand_count,
            d,
            weight,
            k,
            centroids,
            cand_labels,
            INIT_ARRAY,
            UInt64(0),
            1,
            DEFAULT_MAX_ITER,
            DEFAULT_TOL,
            METRIC_L2_EXPANDED,
            DEFAULT_OVERSAMPLING,
            inner_sum_scale,
            inner_weight_scale,
            trace,
            tag_prefix + "init.par.",
        )
    elif cand_count < k:
        var n_random = k - cand_count
        host_init_random(x, n, d, n_random, centroids, rng)
        for j in range(cand_count * d):
            centroids[n_random * d + j] = cand[j]
    else:
        for j in range(k * d):
            centroids[j] = cand[j]


def host_validate_params(
    metric: Int, n_clusters: Int, tol: Float64, oversampling_factor: Float64
) raises:
    """`KMeansParams.validate`, `kmeans_params.mojo`, in its words.

    The sentence must stay byte-equal to the device's
    (`python/mojolearn/tests/test_cpu_training_misc.py` reads both files and
    compares), so that a CPU-only install refuses in the same words a GPU
    install does.
    """
    if metric != METRIC_L2_EXPANDED and metric != METRIC_L2_SQRT_EXPANDED:
        raise Error(
            "kmeans supports only the L2Expanded (0) and L2SqrtExpanded"
            " (1) distance metrics; got metric="
            + String(metric)
        )
    if n_clusters <= 0:
        raise Error("invalid parameter (n_clusters<=0)")
    if tol <= 0.0:
        raise Error("invalid parameter (tol<=0)")
    if oversampling_factor < 0.0:
        raise Error("invalid parameter (oversampling_factor<0)")


def host_accumulate(
    x: List[Float32],
    n: Int,
    d: Int,
    labels: List[UInt32],
    weights: List[Float32],
    k: Int,
    sum_scale: Float32,
    weight_scale: Float32,
    mut sums_i32: List[Int32],
    mut weight_i32: List[Int32],
) raises:
    """`launch_accumulate_centroid_sums` and `launch_accumulate_weight_per_
    cluster` (`cluster/checks/reduce_by_key.mojo`): the quantized Int32
    scatter-add, whose sum is order independent, walked in row order.

    Every label is checked against `k` before it addresses a cell
    (`host_checked_label`), so a row with no nearest centroid raises a
    Python-visible error instead of aborting the process."""
    for row in range(n):
        _ = host_checked_label(labels, row, k)
    for c in range(k * d):
        sums_i32[c] = Int32(0)
    for c in range(k):
        weight_i32[c] = Int32(0)
    for gid in range(n * d):
        var row = gid // d
        var f = gid - row * d
        var label = Int(labels[row])
        var w = weights[row]
        var q = Int32(x[gid] * w * sum_scale)
        comptime if KMEANS_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM: one extra unit per cell. Wrong on purpose;
            # see KMEANS_ORACLE_HOST_SABOTAGE.
            q = q + Int32(1)
        sums_i32[label * d + f] = sums_i32[label * d + f] + q
    for row in range(n):
        var label = Int(labels[row])
        var q = Int32(weights[row] * weight_scale)
        weight_i32[label] = weight_i32[label] + q


@fieldwise_init
struct KMeansHostFit(Copyable, Movable):
    var inertia: Float64
    var n_iter: Int


def host_fit_main(
    x: List[Float32],
    n: Int,
    d: Int,
    weights: List[Float32],
    k: Int,
    mut centroids: List[Float32],
    mut labels: List[UInt32],
    init: Int,
    seed: UInt64,
    n_init_in: Int,
    max_iter: Int,
    tol: Float64,
    metric: Int,
    oversampling_factor: Float64,
    sum_scale: Float32,
    weight_scale: Float32,
    mut trace: KMeansHostTrace,
    tag_prefix: String,
) raises -> KMeansHostFit:
    """`kmeans_fit_main_traced` (module docstring). `centroids` is in-out:
    read as the start on INIT_ARRAY, the best restart on return."""
    host_validate_params(metric, k, tol, oversampling_factor)
    if trace.enabled and tag_prefix == "":
        trace.header(
            String("kmeans n=") + String(n) + " d=" + String(d) + " k="
            + String(k) + " metric=" + String(metric) + " seed="
            + String(seed) + " n_init=" + String(n_init_in) + " max_iter="
            + String(max_iter)
        )
    var cd = k * d
    var is_sqrt = host_metric_is_sqrt(metric)
    var x_norm = host_row_norms(x, n, d, False)
    trace.record_f32(tag_prefix + "fit.x_norm", x_norm)
    var rng = HostRngReplay(seed)
    var best_inertia = Float64(1.0e308)
    var best_iter = 0
    var n_init = n_init_in
    if init == INIT_ARRAY:
        n_init = 1
    var cur = List[Float32](length=cd, fill=Float32(0.0))
    var new_c = List[Float32](length=cd, fill=Float32(0.0))
    var c_norm = List[Float32](length=k, fill=Float32(0.0))
    var min_dist = List[Float32](length=n, fill=Float32(0.0))
    var sums_i32 = List[Int32](length=cd, fill=Int32(0))
    var weight_i32 = List[Int32](length=k, fill=Int32(0))

    for _seed_iter in range(n_init):
        var restart_tag = tag_prefix + "restart" + _trace_pad2(_seed_iter) + "."
        if init == INIT_ARRAY:
            for j in range(cd):
                cur[j] = centroids[j]
        elif init == INIT_RANDOM:
            host_init_random(x, n, d, k, cur, rng)
        elif init == INIT_KMEANS_PLUS_PLUS and oversampling_factor != 0.0:
            host_init_scalable(
                x, x_norm, n, d, k, metric, oversampling_factor, cur, rng,
                trace, restart_tag,
            )
        else:
            host_kmeans_plus_plus(x, x_norm, n, d, k, is_sqrt, cur, rng)
        trace.record_f32(restart_tag + "init.centroids", cur)

        var n_current_iter = max_iter + 1
        var it = 1
        while it <= max_iter:
            var it_tag = restart_tag + "iter" + _trace_pad2(it) + "."
            c_norm = host_row_norms(cur, k, d, host_norms_take_sqrt(metric))
            host_assign(x, n, x_norm, cur, k, c_norm, d, is_sqrt, labels, min_dist)
            trace.record_f32(it_tag + "centroid_norm", c_norm)
            trace.record_u32(it_tag + "labels", labels)
            trace.record_f32(it_tag + "min_dist", min_dist)
            host_accumulate(
                x, n, d, labels, weights, k, sum_scale, weight_scale,
                sums_i32, weight_i32,
            )
            # finalize_centroids_kernel
            for idx in range(cd):
                var cluster = idx // d
                var w = ftz(Float32(weight_i32[cluster]) / weight_scale)
                if w == Float32(0.0):
                    new_c[idx] = cur[idx]
                else:
                    var s = ftz(Float32(sums_i32[idx]) / sum_scale)
                    new_c[idx] = ftz(s / w)
            trace.record_i32(it_tag + "sums_i32", sums_i32)
            trace.record_i32(it_tag + "weight_i32", weight_i32)
            trace.record_f32(it_tag + "new_centroids", new_c)
            var shift = host_sum_device(cur, new_c, cd, SUM_MODE_SQDIFF)
            for j in range(cd):
                cur[j] = new_c[j]
            trace.record_scalar_f32(it_tag + "shift", shift)
            # check_convergence with inertia_check off: the shift alone.
            if Float64(shift) < tol:
                n_current_iter = it
                break
            it += 1

        # The post-loop assignment and the weighted inertia.
        c_norm = host_row_norms(cur, k, d, host_norms_take_sqrt(metric))
        host_assign(x, n, x_norm, cur, k, c_norm, d, is_sqrt, labels, min_dist)
        var cost32 = host_sum_device(min_dist, weights, n, SUM_MODE_PRODUCT)
        var iter_cost = Float64(cost32)
        trace.record_u32(restart_tag + "final.labels", labels)
        trace.record_f32(restart_tag + "final.centroids", cur)
        trace.record_scalar_f32(restart_tag + "final.inertia", cost32)
        if iter_cost < best_inertia:
            best_inertia = iter_cost
            best_iter = n_current_iter
            for j in range(cd):
                centroids[j] = cur[j]
    trace.record_f32(tag_prefix + "fit.centroids", centroids)
    trace.record_u32(tag_prefix + "fit.labels", labels)
    return KMeansHostFit(best_inertia, best_iter)


@fieldwise_init
struct KMeansHostResult(Copyable, Movable):
    """`KMeansFitResult`, `cluster/estimator.mojo`."""

    var inertia: Float64
    var n_iter: Int
    var sum_scale: Float64
    var weight_scale: Float64


def host_plan_sum_scale(x: List[Float32], n: Int, d: Int) raises -> Float64:
    """`plan_sum_scale` (module docstring)."""
    var totals = List[Float64](length=d, fill=Float64(0.0))
    for r in range(n):
        var row = r * d
        for f in range(d):
            totals[f] = totals[f] + Float64(abs(x[row + f]))
    var worst = Float64(0.0)
    for f in range(d):
        var column = totals[f]
        if column > worst:
            worst = column
    return choose_scale(worst, n)


def host_kmeans_validate(
    n: Int, d: Int, k: Int, n_weights: Int
) raises:
    """`kmeans_fit`'s guards, `cluster/estimator.mojo`, in its words and
    order; called before any address is read."""
    if n < 1 or d < 1 or k < 1:
        raise Error(
            "kmeans_fit needs n_samples, n_features and n_clusters >= 1: got "
            + String(n)
            + ", "
            + String(d)
            + ", "
            + String(k)
        )
    if n_weights != 0 and n_weights != n:
        raise Error(
            "kmeans_fit needs n_weights == 0 (unit weights) or == n_samples:"
            " got "
            + String(n_weights)
            + " for "
            + String(n)
            + " samples"
        )
    if k > n:
        raise Error(
            "kmeans_fit cannot form more clusters than samples: "
            + String(k)
            + " > "
            + String(n)
        )


def host_kmeans_fit(
    x: List[Float32],
    n: Int,
    d: Int,
    k: Int,
    mut centroids: List[Float32],
    mut labels: List[UInt32],
    weights_in: List[Float32],
    n_weights: Int,
    max_iter: Int,
    tol: Float64,
    seed: UInt64,
    n_init: Int,
    init: Int,
    metric: Int,
    oversampling_factor: Float64 = DEFAULT_OVERSAMPLING,
) raises -> KMeansHostResult:
    """`kmeans_fit` then `fit_predict` (module docstring). `centroids` is
    `k x d`, read first on INIT_ARRAY; `labels` is `n`, the assignment
    against the FINAL centroids."""
    comptime assert NORM_TPB == lib_block_size_for[K_LIB_ROW_NORM, COLUMN_APPLE](), (
        "kmeans host: the row norm fold width differs from the Apple column's"
    )
    comptime assert NORM_TPB == lib_block_size_for[K_LIB_ROW_NORM, COLUMN_NVIDIA](), (
        "kmeans host: the row norm fold width differs from the NVIDIA column's"
    )
    comptime assert NORM_TPB == lib_block_size_for[K_LIB_ROW_NORM, COLUMN_AMD](), (
        "kmeans host: the row norm fold width differs from the AMD column's"
    )
    comptime assert REDUCE_BY_KEY_TPB == lib_block_size_for[K_LIB_REDUCE_BY_KEY, COLUMN_APPLE](), (
        "kmeans host: the reduce-by-key fold width differs from the Apple column's"
    )
    comptime assert REDUCE_BY_KEY_TPB == lib_block_size_for[K_LIB_REDUCE_BY_KEY, COLUMN_NVIDIA](), (
        "kmeans host: the reduce-by-key fold width differs from the NVIDIA column's"
    )
    comptime assert REDUCE_BY_KEY_TPB == lib_block_size_for[K_LIB_REDUCE_BY_KEY, COLUMN_AMD](), (
        "kmeans host: the reduce-by-key fold width differs from the AMD column's"
    )
    comptime assert PLUS_PLUS_TPB == lib_block_size_for[K_LIB_PLUS_PLUS, COLUMN_APPLE](), (
        "kmeans host: the k-means++ fold width differs from the Apple column's"
    )
    comptime assert PLUS_PLUS_TPB == lib_block_size_for[K_LIB_PLUS_PLUS, COLUMN_NVIDIA](), (
        "kmeans host: the k-means++ fold width differs from the NVIDIA column's"
    )
    comptime assert PLUS_PLUS_TPB == lib_block_size_for[K_LIB_PLUS_PLUS, COLUMN_AMD](), (
        "kmeans host: the k-means++ fold width differs from the AMD column's"
    )
    comptime assert (NORM_TPB & (NORM_TPB - 1)) == 0 and (
        REDUCE_BY_KEY_TPB & (REDUCE_BY_KEY_TPB - 1)
    ) == 0 and (PLUS_PLUS_TPB & (PLUS_PLUS_TPB - 1)) == 0, (
        "kmeans host: the halving tree needs a power-of-two block"
    )
    comptime assert PLUS_PLUS_TPB % SCAN_WARP == 0, (
        "kmeans host: the block scan needs a warp multiple"
    )
    host_kmeans_validate(n, d, k, n_weights)

    var sum_scale = host_plan_sum_scale(x, n, d)
    var weight_bound = Float64(n)
    if n_weights != 0:
        weight_bound = Float64(0.0)
        for r in range(n):
            weight_bound += Float64(abs(weights_in[r]))
    var weight_scale = choose_scale(weight_bound, n)

    var weights = List[Float32](length=n, fill=Float32(1.0))
    if n_weights != 0:
        for r in range(n):
            weights[r] = weights_in[r]

    var trace = KMeansHostTrace()
    var result = host_fit_main(
        x, n, d, weights, k, centroids, labels, init, seed, n_init, max_iter,
        tol, metric, oversampling_factor, Float32(sum_scale),
        Float32(weight_scale), trace, String(""),
    )

    # `host_fit_main` ends EVERY restart with this exact assignment against
    # that restart's final centroids.  When there is only one effective
    # restart, its returned centroids and the labels already in `labels` are
    # therefore the requested fit_predict answer; repeating the O(n*k*d)
    # pass cannot change a bit.  INIT_ARRAY forces one effective restart in
    # host_fit_main even if the caller supplied a larger n_init.
    #
    # With multiple effective restarts, `labels` belongs to the LAST one
    # while `centroids` belongs to the BEST one, so the fresh assignment is
    # still required.  Keep that distinction explicit: it is the proof that
    # makes this elision safe rather than an assumption about defaults.
    if n_init > 1 and init != INIT_ARRAY:
        var x_norm = host_row_norms(x, n, d, host_norms_take_sqrt(metric))
        var c_norm = host_row_norms(centroids, k, d, host_norms_take_sqrt(metric))
        var min_dist = List[Float32](length=n, fill=Float32(0.0))
        host_assign(
            x, n, x_norm, centroids, k, c_norm, d, host_metric_is_sqrt(metric),
            labels, min_dist,
        )
    return KMeansHostResult(result.inertia, result.n_iter, sum_scale, weight_scale)


def host_kmeans_predict(
    x: List[Float32],
    n: Int,
    d: Int,
    centroids: List[Float32],
    k: Int,
    metric: Int,
    mut labels: List[UInt32],
) raises:
    """`kmeans_predict`, `cluster/estimator.mojo`: the metric refused by
    name as the fit refuses it, then `host_kmeans_fit`'s own final
    assignment, statement for statement (the same row norms, the same
    `host_assign`), so on the training rows and the returned centroids it
    is `labels_` by construction. cuML's `KMeans.predict` is
    `_predict_labels_inertia` keeping the labels (`kmeans.pyx:1071-1082`),
    one cuVS assignment pass."""
    if n < 1 or d < 1 or k < 1:
        raise Error(
            "kmeans_predict needs n_samples, n_features and n_clusters >= 1: got "
            + String(n)
            + ", "
            + String(d)
            + ", "
            + String(k)
        )
    host_validate_params(metric, k, 1e-4, DEFAULT_OVERSAMPLING)
    var x_norm = host_row_norms(x, n, d, host_norms_take_sqrt(metric))
    var c_norm = host_row_norms(centroids, k, d, host_norms_take_sqrt(metric))
    var min_dist = List[Float32](length=n, fill=Float32(0.0))
    host_assign(
        x, n, x_norm, centroids, k, c_norm, d, host_metric_is_sqrt(metric),
        labels, min_dist,
    )
    comptime if KMEANS_PREDICT_HOST_SABOTAGE:
        # THE SABOTAGE ARM: one bit into every label. Wrong on purpose; see
        # KMEANS_PREDICT_HOST_SABOTAGE.
        for row in range(n):
            labels[row] = labels[row] ^ UInt32(1)


def host_kmeans_transform(
    x: List[Float32],
    n: Int,
    d: Int,
    centroids: List[Float32],
    k: Int,
    metric: Int,
    mut dist_out: List[Float32],
) raises:
    """`kmeans_transform`, `cluster/estimator.mojo`: the metric refused by
    name as the fit refuses it, `host_kmeans_predict`'s row norms, then every
    cell of `cluster/impl/detail/kmeans_transform.mojo::
    kmeans_transform_kernel` statement for statement: `host_assign`'s cell,
    with the root taken per cell (cuVS `kmeans_transform`,
    `detail/kmeans.cuh:1178-1219`, through `l2_exp_distance_op`). `out` is
    row-major `n x k`."""
    if n < 1 or d < 1 or k < 1:
        raise Error(
            "kmeans_transform needs n_samples, n_features and n_clusters >= 1: got "
            + String(n)
            + ", "
            + String(d)
            + ", "
            + String(k)
        )
    host_validate_params(metric, k, 1e-4, DEFAULT_OVERSAMPLING)
    var x_norm = host_row_norms(x, n, d, host_norms_take_sqrt(metric))
    var c_norm = host_row_norms(centroids, k, d, host_norms_take_sqrt(metric))
    var is_sqrt = host_metric_is_sqrt(metric)
    for row in range(n):
        var xn = x_norm[row]
        for col in range(k):
            var acc = Float32(0.0)
            for p in range(d):
                acc = ftz(
                    identical_mul_add(ftz(x[row * d + p]), ftz(centroids[col * d + p]), acc)
                )
            var yn = c_norm[col]
            var dist = ftz(
                identical_mul_add(
                    Float32(-2.0), ftz(acc), ftz(ftz(xn) + ftz(yn))
                )
            )
            if dist <= Float32(0.0) or (
                dist * dist < FUSED_CLAMP_PRECISION and xn == yn
            ):
                dist = Float32(0.0)
            if is_sqrt:
                dist = identical_sqrt(dist)
            comptime if KMEANS_TRANSFORM_HOST_SABOTAGE:
                # THE SABOTAGE ARM: one unit into every cell. Wrong on
                # purpose; see KMEANS_TRANSFORM_HOST_SABOTAGE.
                dist = dist + Float32(1.0)
            dist_out[row * k + col] = dist
