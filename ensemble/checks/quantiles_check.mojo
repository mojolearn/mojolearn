# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does `quantiles.mojo` compute cuML's quantiles, or something that
looks like them?

    tools/with_build_lock.sh pixi run mojo run -I . \\
        ensemble/checks/quantiles_check.mojo

THE RULE THIS FILE IS BUILT AROUND, and the scar behind it:

    A check whose expected value is the same in EVERY cell verifies the
    total and nothing about placement.

A histogram kernel in this repository once read `0 wrong of 512` on
uniform data and `490 wrong of 512` on hashed data -- same kernel, same
parameters -- after two earlier checks had reported it correct at
exactly the failing configuration. Quantiles are unusually good at
hiding under that: a per-column bug is INVISIBLE if every column has the
same distribution, and a sort bug is invisible if the fixture is already
sorted or has few distinct values. So every column here has a DIFFERENT
distribution, chosen for a different failure, and every comparison is
per cell.

THE THREE ARMS
---------------
ARM 1 -- the generator, against RAFT's own source.
  `PCGenerator` is the highest-risk construct in the port: one wrong bit
  moves a sampled row, a moved row moves a quantile, and a moved
  quantile moves every split threshold built on it. Nothing downstream
  would notice, so nothing downstream is trusted to. The expected values
  below were produced by compiling
  `raft/random/detail/rng_device.cuh:546-683` and `:208-231` plus
  `raft/util/integer_utils.hpp:207` VERBATIM with clang++ -- the same
  device this repository used for the Mersenne twister in
  `tools/permutation_oracle/`. Constants, shifts and call ORDER are
  therefore checked against their code, not against our reading of it.

  Four sub-arms, because the generator has four reachable behaviours:
  the raw `next_u32` stream, the `next_u64` word pairing (swapping the
  two halves passes every distributional test and fails here), the
  `skipahead` walk (which their call site exercises only with
  `offset = 0`, i.e. never), and the REJECTION LOOP inside
  `custom_next`, which their call site cannot reach at all -- see below.

ARM 2 -- the segmented sort, per cell, against a host sort.
  Four segments with four different distributions and a segment length
  that spans more than one block, so the block-sum scan is exercised
  rather than skipped. Compared slot for slot, not for sortedness:
  sortedness compares each output only with its neighbour and any
  permutation of an equal-key run satisfies it.

ARM 3 -- the whole pipeline, per cell, against a host reference.
  Seven columns of ADVERSARIAL CARDINALITY, and three shapes chosen to
  reach the branches rather than the common case:

    3a  n_rows 20, max_n_bins 8  -> sample_count == global_rows, so the
        RNG IS NEVER CONSTRUCTED (`quantiles.cuh:58`) and every row is
        used in order. On a small dataset THIS is the default arm.
    3b  n_rows 5000, max_n_bins 8 -> sample_count 32 != global_rows, the
        `PCGenerator` arm, 32 draws WITH REPLACEMENT.
    3c  n_rows 5000, max_n_bins 200 -> sample_count 800, so each sort
        segment spans two blocks and the block-sum scan is live inside
        the pipeline and not only in arm 2.
    3d  n_rows 1200, max_n_bins 1100 -> block is capped at 1024
        (`quantiles.cuh:271`) so the bin loop at `:92` takes more than
        one grid-stride iteration, which at 8/32/200 bins it never does;
        and `sample_count < max_n_bins`, i.e. `bin_width < 1`, where
        several bins interpolate to the SAME sorted index and the
        `unique` collapse carries the result.

THE COLUMNS, and what each one is for
--------------------------------------
  0  constant                 -> `n_bins == 1`. The whole `unique`
                                 collapse in one cell.
  1  {-2.0, -0.0, +0.0}       -> THREE distinct bit patterns, TWO
                                 distinct values. The sort orders `-0.0`
                                 strictly before `+0.0` (bit order) and
                                 `thrust::unique` then collapses them
                                 because `-0.0 == +0.0`. A port that
                                 deduped on bits, or that sorted with a
                                 `<` that calls them equal, differs here
                                 and nowhere else.
  2  exactly max_n_bins       -> `n_bins == max_n_bins` with no collapse
     distinct values             at all: the boundary between the two
                                 regimes, on the far side.
  3  all distinct, hashed     -> the placement arm. Every cell has a
                                 different expected value, so a bin
                                 index off by one is visible AS A CELL
                                 and not as a count.
  4  packing boundaries       -> ±inf, ±3.4e38, ±1e-38, ±0.0. Exercises
                                 the float-to-unsigned twiddle at the
                                 exponent extremes, where a sign-only
                                 flip would sort the negatives backwards.
  5  heavy duplicates         -> a handful of values, each repeated
                                 many times, at scattered magnitudes.
  6  contains NaN             -> NaN sorts above +inf by bit order and
                                 NEVER collapses, because `NaN != NaN`.
                                 Both are cuML's behaviour by
                                 construction (`thrust::unique` uses
                                 `==`); both are checked rather than
                                 assumed.

WHAT IS NOT CHECKED HERE, said plainly
---------------------------------------
  * The distributed arm. It is not ported (DEVIATION 108) and
    `compute_quantiles` raises on it; the raise is checked, the
    collectives are not, because there are none.
  * Any timing. Out of scope for this round by instruction.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator

from ensemble.decisiontree.batched_levelalgo.quantiles import (
    PCGenerator,
    compute_quantiles,
    custom_next_uniform_int_u64,
    quantile_bin_index,
)
from core.segmented_sort import segmented_sort_keys_f32


# ===========================================================================
# ARM 1 -- the generator against RAFT's own source, compiled by clang++
# ===========================================================================
#
# PROVENANCE OF EVERY NUMBER BELOW. `rng_device.cuh:546-683` (PCGenerator),
# `:208-231` (the uint64 `custom_next`) and `integer_utils.hpp:207`
# (`wmul_64bit`) were copied character for character into a standalone
# .cpp, with exactly two edits: `HDI` -> `inline`, and the `__CUDA_ARCH__`
# arm of `wmul_64bit` dropped so it builds for the host. Built with
# `clang++ -O0 -std=c++17` on this machine and run. These are its stdout.
#
# THE ORACLE IS NOT IN THE REPOSITORY. It is a scratchpad artifact, and
# the numbers are transcribed here so this file stands alone. If the
# generator is ever touched, rebuild it rather than adjusting these.


def _expect_u32_stream() -> List[UInt32]:
    """`PCGenerator g(42, 7, 0); g.next_u32() x8`."""
    return [
        UInt32(1956239935),
        UInt32(1010964048),
        UInt32(2769188248),
        UInt32(3076816759),
        UInt32(888960798),
        UInt32(435942894),
        UInt32(3750715939),
        UInt32(926258306),
    ]


def _expect_u64_stream() -> List[UInt64]:
    """`PCGenerator g(0, 0, 0); g.next_u64() x4`.

    The pairing arm: `next_u64` is `a | (b << 32)` with `a` drawn FIRST
    (`rng_device.cuh:613-615`). A port that swapped them produces a
    perfectly good stream that disagrees with every one of these.
    """
    return [
        UInt64(4007188917454456712),
        UInt64(6925731248493736891),
        UInt64(15130594009555676044),
        UInt64(10520772042555918893),
    ]


def _expect_skipahead() -> List[UInt64]:
    """`PCGenerator g(123456789, 3, 5); g.next_u64() x4`.

    `offset = 5` so the `skipahead` while-loop runs THREE times with two
    odd bits. `quantiles.cuh:63` always passes `offset = 0`, where the
    loop body never executes -- this sub-arm is the only reach the
    walk gets, and without it `skipahead` could be `pass` and the
    pipeline would not care.
    """
    return [
        UInt64(12950541207357205721),
        UInt64(3831298223129847893),
        UInt64(18224208963890975960),
        UInt64(2004067133548302545),
    ]


def _expect_rejection() -> List[UInt64]:
    """`custom_next` with `diff = 2^63 + 1`, subsequences 0..7, seed 2026.

    THIS IS THE ONLY REACH THE REJECTION LOOP GETS, and it is here
    because their call site cannot reach it: `s` is `global_rows`, so
    `t = 2^64 mod s` and the loop is entered with probability
    `t / 2^64`, which is under 3e-17 for any row count that fits in
    memory. At `s = 2^63 + 1`, `t = 2^63 - 1` and the rate is about one
    half -- the instrumented oracle took 7 rejections across these 8
    draws. An unreached branch is an unchecked branch even when the
    reason it is unreached is arithmetic.
    """
    return [
        UInt64(8429057709363667418),
        UInt64(1602622916730613810),
        UInt64(860687880735702580),
        UInt64(3906246198907172489),
        UInt64(6827112983213114151),
        UInt64(1756816584889289153),
        UInt64(5951624068092817341),
        UInt64(4987640998476887167),
    ]


def _expect_rows(seed: Int, global_rows: Int) -> List[UInt64]:
    """`quantiles.cuh:57-66` for `sample_idx` 0..15, from the oracle.

    Note `seed=0, rows=1000` and `seed=0, rows=100000` share their draw
    `x` and differ only in the scaling -- 217 against 21723 -- which is
    a property of Lemire's method and a cheap sanity read on the table.
    """
    if seed == 0 and global_rows == 1000:
        return [
            UInt64(217), UInt64(94), UInt64(992), UInt64(697),
            UInt64(720), UInt64(15), UInt64(255), UInt64(766),
            UInt64(651), UInt64(598), UInt64(46), UInt64(808),
            UInt64(717), UInt64(602), UInt64(899), UInt64(826),
        ]
    if seed == 0 and global_rows == 100000:
        return [
            UInt64(21723), UInt64(9476), UInt64(99258), UInt64(69768),
            UInt64(72010), UInt64(1544), UInt64(25507), UInt64(76621),
            UInt64(65117), UInt64(59892), UInt64(4654), UInt64(80858),
            UInt64(71733), UInt64(60287), UInt64(89922), UInt64(82684),
        ]
    if seed == 42 and global_rows == 1000:
        return [
            UInt64(755), UInt64(896), UInt64(304), UInt64(227),
            UInt64(849), UInt64(317), UInt64(699), UInt64(235),
            UInt64(474), UInt64(750), UInt64(248), UInt64(863),
            UInt64(511), UInt64(459), UInt64(897), UInt64(448),
        ]
    return [
        UInt64(75535), UInt64(89653), UInt64(30427), UInt64(22743),
        UInt64(84975), UInt64(31736), UInt64(69938), UInt64(23538),
        UInt64(47409), UInt64(75031), UInt64(24871), UInt64(86303),
        UInt64(51164), UInt64(45949), UInt64(89797), UInt64(44822),
    ]


def reference_bin_index(bin: Int, sample_count: Int, max_n_bins: Int) -> Int:
    """`quantiles.cuh:90` and `:94-95`, transcribed a SECOND TIME.

    THIS FUNCTION EXISTS BECAUSE A SABOTAGE CAUGHT ITS ABSENCE. The
    first version of this file imported `quantile_bin_index` from
    `quantiles.mojo` and used it to build the expected values. Deleting
    their `- 1` from the port then ran the check GREEN: the sabotage
    moved BOTH sides of the comparison by the same amount, so the
    comparison could not see it. A check that shares a mechanism with
    the thing it checks verifies that the mechanism is consistent with
    itself and nothing else.

    So it is transcribed again, from their source rather than from ours,
    and spelled differently on purpose: `Int(x)` truncates toward zero,
    which for the strictly positive `x` here is `floor`, and the
    half-test is written as an addition rather than a branch on `floor`.
    Same value at every point, no shared line.

        double bin_width = static_cast<double>(sample_count) / max_n_bins;
        int idx          = int(round((bin + 1) * bin_width)) - 1;
        idx              = min(max(0, idx), sample_count - 1);
    """
    var bin_width = Float64(sample_count) / Float64(max_n_bins)
    var x = Float64(bin + 1) * bin_width
    var truncated = Float64(Int(x))
    var rounded = truncated
    if x - truncated >= Float64(0.5):
        rounded = truncated + Float64(1.0)
    var idx = Int(rounded) - 1
    if idx < 0:
        idx = 0
    if idx > sample_count - 1:
        idx = sample_count - 1
    return idx


def check_bin_index_table() raises -> Int:
    """The two independent transcriptions, point for point.

    Swept over shapes their dispatch reaches and shapes it does not:
    `sample_count / max_n_bins` exactly representable (a power of two),
    a ratio of 1, ratios above and below 1, and the `12`/`2` pair whose
    exact rational lands ON a half-integer where the double lands just
    below it -- the case that rules out the exact-integer resolution in
    DEVIATION 110.
    """
    var counts = [512, 512, 2, 9, 800, 128, 20, 32]
    var bins = [128, 512, 12, 6, 200, 128, 8, 8]
    var wrong = 0
    var total = 0
    for a in range(len(counts)):
        for b in range(bins[a]):
            total += 1
            if quantile_bin_index(b, counts[a], bins[a]) != (
                reference_bin_index(b, counts[a], bins[a])
            ):
                wrong += 1
    print("  bin-index table vs second transcription wrong:", wrong,
          "of", total)
    return wrong


def sampled_row(seed: UInt64, sample_idx: Int, global_rows: UInt64) -> UInt64:
    """`quantiles.cuh:57-66`, on the host, for the reference pipeline."""
    var gen = PCGenerator.init_pcg(seed, UInt64(sample_idx), UInt64(0))
    return custom_next_uniform_int_u64(gen, UInt64(0), global_rows)


def check_pcgenerator() raises -> Int:
    print("ARM 1 -- PCGenerator vs RAFT compiled by clang++")
    var wrong = 0

    var e32 = _expect_u32_stream()
    var g32 = PCGenerator.init_pcg(UInt64(42), UInt64(7), UInt64(0))
    var bad32 = 0
    for i in range(len(e32)):
        if g32.next_u32() != e32[i]:
            bad32 += 1
    print("  next_u32 stream wrong:      ", bad32, "of", len(e32))
    wrong += bad32

    var e64 = _expect_u64_stream()
    var g64 = PCGenerator.init_pcg(UInt64(0), UInt64(0), UInt64(0))
    var bad64 = 0
    for i in range(len(e64)):
        if g64.next_u64() != e64[i]:
            bad64 += 1
    print("  next_u64 word pairing wrong:", bad64, "of", len(e64))
    wrong += bad64

    var esk = _expect_skipahead()
    var gsk = PCGenerator.init_pcg(
        UInt64(123456789), UInt64(3), UInt64(5)
    )
    var badsk = 0
    for i in range(len(esk)):
        if gsk.next_u64() != esk[i]:
            badsk += 1
    print("  skipahead(5) walk wrong:    ", badsk, "of", len(esk))
    wrong += badsk

    var erj = _expect_rejection()
    var s = (UInt64(1) << 63) + UInt64(1)
    var badrj = 0
    for i in range(len(erj)):
        var g = PCGenerator.init_pcg(UInt64(2026), UInt64(i), UInt64(0))
        if custom_next_uniform_int_u64(g, UInt64(0), s) != erj[i]:
            badrj += 1
    print("  rejection-loop draws wrong: ", badrj, "of", len(erj))
    wrong += badrj

    var seeds = [0, 0, 42, 42]
    var rows = [1000, 100000, 1000, 100000]
    var badrow = 0
    var totrow = 0
    for a in range(4):
        var expect = _expect_rows(seeds[a], rows[a])
        for k in range(len(expect)):
            totrow += 1
            if sampled_row(UInt64(seeds[a]), k, UInt64(rows[a])) != expect[k]:
                badrow += 1
    print("  sampler draws wrong:        ", badrow, "of", totrow)
    wrong += badrow
    return wrong


# ===========================================================================
# The fixture
# ===========================================================================


def hashed(i: Int) -> UInt32:
    """Xorshift, so neighbouring rows are unrelated."""
    var x = UInt32(i * 2654435761 + 0x9E3779B9)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


def boundary_value(k: Int) -> Float32:
    """Eight values at the packing boundaries, in NO sorted order."""
    var t = k % 8
    if t == 0:
        return Float32(0.0)
    if t == 1:
        return Float32(-1.0) / Float32(0.0)  # -inf
    if t == 2:
        return Float32(1e-38)
    if t == 3:
        return Float32(-3.4e38)
    if t == 4:
        return Float32(-0.0)
    if t == 5:
        return Float32(1.0) / Float32(0.0)  # +inf
    if t == 6:
        return Float32(3.4e38)
    return Float32(-1e-38)


def fixture_value(row: Int, col: Int, n_distinct_col2: Int) -> Float32:
    """Column `col`, row `row`. See the module docstring for what each
    column is for. Nothing here is uniform across columns and nothing
    is uniform across rows within a column except column 0, which is
    uniform ON PURPOSE."""
    if col == 0:
        return Float32(3.5)
    if col == 1:
        var t = Int(hashed(row) % UInt32(3))
        if t == 0:
            return Float32(-2.0)
        if t == 1:
            return Float32(-0.0)
        return Float32(0.0)
    if col == 2:
        return Float32(row % n_distinct_col2) * Float32(0.125) - Float32(1.0)
    if col == 3:
        return Float32(Int(hashed(row) % UInt32(1000000))) * Float32(1e-3)
    if col == 4:
        return boundary_value(Int(hashed(row) % UInt32(8)))
    if col == 5:
        var t5 = Int(hashed(row + 7919) % UInt32(4))
        if t5 == 0:
            return Float32(-1e20)
        if t5 == 1:
            return Float32(-1e-20)
        if t5 == 2:
            return Float32(7.0)
        return Float32(1e20)
    var t6 = Int(hashed(row + 104729) % UInt32(5))
    if t6 == 0:
        return Float32(0.0) / Float32(0.0)  # NaN
    return Float32(t6) * Float32(11.0) - Float32(20.0)


comptime N_COLS = 7


def build_fixture(n_rows: Int, n_distinct_col2: Int) -> List[Float32]:
    """Column-major, `data[col * n_rows + row]` -- their `row_major =
    false` layout (`quantiles.cuh:76`)."""
    var d = List[Float32]()
    for c in range(N_COLS):
        for r in range(n_rows):
            d.append(fixture_value(r, c, n_distinct_col2))
    return d^


# ===========================================================================
# The host reference order
# ===========================================================================


def total_order_key(v: Float32) -> UInt32:
    """The order a radix sort on float bits produces, written from the
    IEEE layout rather than copied from `segmented_sort.mojo`: a
    different spelling of the same function, so a typo in one shows up
    against the other.

    Negatives descend in unsigned bit order, so they are REVERSED by
    subtracting from all-ones; non-negatives only need the sign bit set
    so they land above every negative. `-0.0` (0x80000000) maps to
    0x7FFFFFFF and `+0.0` (0x00000000) to 0x80000000, so `-0.0` sorts
    strictly first -- and then compares EQUAL in the collapse.
    """
    var b = bitcast[DType.uint32](v)
    if b >= UInt32(0x80000000):
        return UInt32(0xFFFFFFFF) - b
    return b + UInt32(0x80000000)


def host_sort(values: List[Float32]) -> List[Float32]:
    """Insertion sort by `total_order_key`. Deliberately the dumbest
    correct thing: it shares no structure, no bit windowing and no pass
    count with the device implementation, so the two cannot fail the
    same way."""
    var out = List[Float32]()
    var keys = List[UInt32]()
    for i in range(len(values)):
        var v = values[i]
        var k = total_order_key(v)
        var j = len(out)
        out.append(v)
        keys.append(k)
        while j > 0 and keys[j - 1] > k:
            out[j] = out[j - 1]
            keys[j] = keys[j - 1]
            j -= 1
        out[j] = v
        keys[j] = k
    return out^


def flush_subnormal(v: Float32) -> Float32:
    """Map every zero and every subnormal to `+0.0`, leaving normals,
    infinities and NaN alone.

    THIS IS NOT A MODELLING CHOICE, IT IS A MEASURED PROPERTY OF THE
    DEVICE, and it is the single most surprising thing this lane found.
    Measured this session on the Apple GPU with a two-kernel probe:

      * a float32 subnormal SURVIVES a host->device->kernel->host round
        trip bit for bit, through `enqueue_copy`, through a Float32 load
        and store, and through a `bitcast`. Memory is exact.
      * the same subnormal COMPARES EQUAL to `+0.0`, to `-0.0` and to a
        DIFFERENT subnormal (`0x006CE3EE == 0x00000001` returned true),
        and `subnormal + (-0.0)` returned `+0.0`. Arithmetic and
        comparison flush.
      * the smallest NORMAL, `0x00800000`, behaves correctly in all of
        the above, so the boundary is exactly the subnormal range.

    The consequence for this port is confined to ONE line: the
    `thrust::unique` comparison in `computeQuantilesBatchedKernel`
    (`quantiles.cuh:104`). The sort is unaffected -- it compares INTEGER
    keys, never floats, which is why arm 2 matches cell for cell on the
    same column that arm 3 diverges on. See the hardware row in
    `quantiles.mojo`.
    """
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7F800000)) == UInt32(0):
        return Float32(0.0)
    return v


def same_bits(a: Float32, b: Float32) -> Bool:
    """Cell comparison is on BITS. `-0.0 == +0.0` and `NaN != NaN` are
    exactly the two places a value comparison would lie about placement.
    """
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


# ===========================================================================
# ARM 2 -- the segmented sort, per cell
# ===========================================================================


def check_segmented_sort(ctx: DeviceContext) raises -> Int:
    """Four segments, four distributions, segment length spanning more
    than one 512-wide block so the block-sum scan is live."""
    comptime SEG = 1300
    comptime NSEG = N_COLS
    print("ARM 2 -- segmented sort, per cell,", NSEG, "segments of", SEG)

    var host = List[Float32]()
    for s in range(NSEG):
        for i in range(SEG):
            host.append(fixture_value(i, s, 8))

    var h_in = ctx.enqueue_create_host_buffer[DType.float32](NSEG * SEG)
    for i in range(NSEG * SEG):
        h_in.unsafe_ptr().unsafe_store(i, host[i])
    var d_in = ctx.enqueue_create_buffer[DType.float32](NSEG * SEG)
    var d_out = ctx.enqueue_create_buffer[DType.float32](NSEG * SEG)
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=h_in.unsafe_ptr())

    var wa = ctx.enqueue_create_buffer[DType.uint32](NSEG * SEG)
    var wb = ctx.enqueue_create_buffer[DType.uint32](NSEG * SEG)
    var off = ctx.enqueue_create_buffer[DType.int32](NSEG * SEG)
    var wide = (SEG + 511) // 512
    var bsum = ctx.enqueue_create_buffer[DType.int32](NSEG * wide)
    ctx.synchronize()
    # Freed-at-enqueue UAF guard (perf-lane find, 2026-08-22): `h_in`'s
    # last use above was the enqueue itself; keep-alive AFTER the sync.
    _ = h_in^

    segmented_sort_keys_f32(
        ctx, NSEG, SEG, d_in, d_out, wa, wb, off, bsum
    )
    ctx.synchronize()

    var got = ctx.enqueue_create_host_buffer[DType.float32](NSEG * SEG)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()

    var wrong = 0
    for s in range(NSEG):
        var seg_in = List[Float32]()
        for i in range(SEG):
            seg_in.append(host[s * SEG + i])
        var want = host_sort(seg_in)
        var bad = 0
        for i in range(SEG):
            if not same_bits(
                got.unsafe_ptr().unsafe_load(s * SEG + i), want[i]
            ):
                bad += 1
        print("    segment", s, "cells wrong:", bad, "of", SEG)
        wrong += bad

    # Printed beside the tally so the difference between them is visible
    # rather than argued: sortedness alone compares each output only with
    # its neighbour and is satisfied by any permutation of a tie run.
    var unsorted = 0
    for s in range(NSEG):
        for i in range(SEG - 1):
            var a = total_order_key(
                got.unsafe_ptr().unsafe_load(s * SEG + i)
            )
            var b = total_order_key(
                got.unsafe_ptr().unsafe_load(s * SEG + i + 1)
            )
            if a > b:
                unsorted += 1
    print("    (sortedness alone, out of order:", unsorted, ")")
    return wrong


# ===========================================================================
# ARM 3 -- the whole pipeline, per cell
# ===========================================================================


def reference_pipeline(
    data: List[Float32],
    n_rows: Int,
    max_n_bins: Int,
    oversampling_factor: Int,
    seed: UInt64,
    ftz: Bool,
) -> Tuple[List[Float32], List[Int32], Int]:
    """`quantiles.cuh` from the top, on the host, written straight from
    their source rather than from `quantiles.mojo`.

    Returns `(quantiles_array col-major, n_bins_array, sample_count)`.

    `ftz` selects which machine the reference is FOR. `False` is cuML on
    an IEEE float32 device -- what their code computes. `True` adds the
    measured subnormal flush of THIS device to the `unique` comparison
    and nowhere else. Both are run and both are reported, because the
    difference between them is not a defect in this port and must not be
    hidden inside one number.
    """
    var global_rows = UInt64(n_rows)
    var budget = UInt64(max_n_bins) * UInt64(oversampling_factor)
    var sc = Int(global_rows if global_rows < budget else budget)

    # `:57-66` -- ONE row sample, shared by every column.
    var rows = List[Int]()
    for k in range(sc):
        if UInt64(sc) != global_rows:
            rows.append(Int(sampled_row(seed, k, global_rows)))
        else:
            rows.append(k)

    var quantiles = List[Float32]()
    var n_bins = List[Int32]()
    for c in range(N_COLS):
        var col_vals = List[Float32]()
        for k in range(sc):
            col_vals.append(data[c * n_rows + rows[k]])
        var srt = host_sort(col_vals)

        # `:92-97`
        var picked = List[Float32]()
        for b in range(max_n_bins):
            picked.append(srt[reference_bin_index(b, sc, max_n_bins)])

        # `:104-106` -- thrust::unique, keeping the first of each
        # consecutive run, compared with `==` and therefore collapsing
        # `-0.0` with `+0.0` and never collapsing NaN.
        var uniq = List[Float32]()
        uniq.append(picked[0])
        for r in range(1, max_n_bins):
            var lhs = picked[r]
            var rhs = uniq[len(uniq) - 1]
            var differs: Bool
            if ftz:
                differs = flush_subnormal(lhs) != flush_subnormal(rhs)
            else:
                differs = lhs != rhs
            if differs:
                uniq.append(picked[r])
        n_bins.append(Int32(len(uniq)))
        for b in range(max_n_bins):
            if b < len(uniq):
                quantiles.append(uniq[b])
            else:
                # positions past `n_bins` are untouched tail in theirs
                quantiles.append(picked[b])
    return (quantiles^, n_bins^, sc)


def check_pipeline(
    ctx: DeviceContext,
    label: String,
    n_rows: Int,
    max_n_bins: Int,
    seed: UInt64,
) raises -> Int:
    var data = build_fixture(n_rows, max_n_bins)
    var expected = reference_pipeline(
        data, n_rows, max_n_bins, 4, seed, True
    )
    var want_q = expected[0].copy()
    var want_n = expected[1].copy()
    var sc = expected[2]
    # The same reference for an IEEE device, i.e. what cuML computes.
    var ieee = reference_pipeline(data, n_rows, max_n_bins, 4, seed, False)
    var ieee_n = ieee[1].copy()

    var arm = "IDENTITY (no RNG)" if sc == n_rows else "PCGenerator"
    print(
        "ARM 3 --",
        label,
        ": n_rows",
        n_rows,
        "max_n_bins",
        max_n_bins,
        "-> sample_count",
        sc,
        ",",
        arm,
    )

    var h = ctx.enqueue_create_host_buffer[DType.float32](len(data))
    for i in range(len(data)):
        h.unsafe_ptr().unsafe_store(i, data[i])
    var d = ctx.enqueue_create_buffer[DType.float32](len(data))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    # Freed-at-enqueue UAF guard: keep-alive AFTER the sync.
    _ = h^

    var result = compute_quantiles(
        ctx, d, max_n_bins, n_rows, N_COLS, 4, seed, False
    )
    var view = result.view()

    var got_q = ctx.enqueue_create_host_buffer[DType.float32](
        N_COLS * max_n_bins
    )
    var got_n = ctx.enqueue_create_host_buffer[DType.int32](N_COLS)
    ctx.enqueue_copy(
        dst_ptr=got_q.unsafe_ptr(), src_buf=result.quantiles_array
    )
    ctx.enqueue_copy(
        dst_ptr=got_n.unsafe_ptr(), src_buf=result.n_bins_array
    )
    ctx.synchronize()
    _ = view

    var wrong = 0
    for c in range(N_COLS):
        var gn = Int(got_n.unsafe_ptr().unsafe_load(c))
        var wn = Int(want_n[c])
        var nbad = 0 if gn == wn else 1
        # Only the first `n_bins` cells are the answer; the tail past it
        # is `unique`'s undefined remainder in their code too.
        var cbad = 0
        var lim = gn if gn < wn else wn
        for b in range(lim):
            if not same_bits(
                got_q.unsafe_ptr().unsafe_load(c * max_n_bins + b),
                want_q[c * max_n_bins + b],
            ):
                cbad += 1
        var ftz_note = ""
        if Int(ieee_n[c]) != wn:
            ftz_note = (
                " | SUBNORMAL FLUSH: cuML on IEEE would give n_bins "
                + String(Int(ieee_n[c]))
            )
        print(
            "    col",
            c,
            ": n_bins got",
            gn,
            "want",
            wn,
            "| quantile cells wrong",
            cbad,
            "of",
            lim,
            ftz_note,
        )
        wrong += nbad + cbad
    return wrong


def check_distributed_raises(ctx: DeviceContext) raises -> Int:
    """DEVIATION 108 must be a raise, not a silence."""
    var d = ctx.enqueue_create_buffer[DType.float32](8)
    var raised = False
    try:
        var r = compute_quantiles(
            ctx, d, 4, 8, 1, 4, UInt64(0), False, 2, 0
        )
        _ = r^
    except e:
        raised = True
    print("  comm_size=2 raises:", raised)
    return 0 if raised else 1


def main() raises:
    comptime assert has_accelerator(), "quantiles_check requires a GPU"
    var ctx = DeviceContext()
    var wrong = 0

    wrong += check_pcgenerator()
    wrong += check_bin_index_table()
    print()
    wrong += check_segmented_sort(ctx)
    print()
    wrong += check_pipeline(ctx, "3a", 20, 8, UInt64(0))
    print()
    wrong += check_pipeline(ctx, "3b", 5000, 8, UInt64(42))
    print()
    wrong += check_pipeline(ctx, "3c", 5000, 200, UInt64(7))
    print()
    # 3d crosses `std::min(1024, max_n_bins)` (`quantiles.cuh:271`), so
    # the bin loop at `:92` takes MORE THAN ONE grid-stride iteration --
    # at 8, 32, 200 bins it takes exactly one and the `bin +=
    # blockDim.x` step is dead. It also puts `sample_count` BELOW
    # `max_n_bins`, i.e. `bin_width < 1`, where their interpolation
    # hands the same sorted index to several bins in a row and the
    # `unique` collapse is doing most of the work.
    wrong += check_pipeline(ctx, "3d", 1200, 1100, UInt64(3))
    print()
    print("DEVIATION 108 --")
    wrong += check_distributed_raises(ctx)

    print()
    if wrong == 0:
        print("quantiles_check: ALL CELLS MATCH")
    else:
        print("quantiles_check: FAILURES:", wrong)
