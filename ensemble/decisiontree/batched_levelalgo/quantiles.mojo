# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS cuML's random forest. Per-file provenance is in this file's own docstring and in NOTICE.
"""The per-feature split candidates the whole forest is built from.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/quantiles.cuh` and
`quantiles.h` at rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), checked out read-only at
`~/CascadeProjects/upstream/cuml-v26.08.00`. The one construct that is
NOT theirs -- RAFT's `PCGenerator` -- mirrors
`cpp/include/raft/random/detail/rng_device.cuh` at rapidsai/raft
`661a3b840c3300f95f053812a560c952c9d049a4` (branch-25.08), which is what
cuML 26.08 pins.

WHAT THIS FILE IS FOR, and how often it runs
---------------------------------------------
`computeQuantiles` is called ONCE PER FOREST, from
`randomforest.cuh:318-325`, before the tree loop opens:

    auto quantile_result = DT::computeQuantiles(handle,
                                                input,
                                                this->rf_params.tree_params.max_n_bins,
                                                n_rows,
                                                n_cols,
                                                4,
                                                rf_params.seed,
                                                input_row_major);
    auto quantiles       = quantile_result.view();

Not per tree, not per node. Every tree in the forest and every node in
every tree shares one `Quantiles` view, so the bins a split can be made
at are a property of the DATASET, not of the tree. That is why
`n_bins_array` is a real array and not a constant: a feature with fewer
distinct sampled values than `max_n_bins` gets fewer candidates, and
`builder.cuh` reads that count per column.

THE DISPATCH ANSWER, written down first as the charter requires
----------------------------------------------------------------
Their call site above passes `oversampling_factor = 4` as a HARD-CODED
literal -- the default in the signature (`quantiles.cuh:152`) is never
what selects it, the call site is. `max_n_bins` defaults to 128
(`cuml/tree/decisiontree.hpp:85`, "maximum number of bins; default
128"). So for a default RF fit:

    sample_count = min(global_rows, 128 * 4) = min(n_rows, 512)

and that `min` is a BRANCH, not a clamp, because `sample_count` is then
compared against `global_rows` inside the sampler kernel
(`quantiles.cuh:58`):

  * `n_rows <= 512`  ->  `sample_count == global_rows`  ->  the RNG IS
    NEVER CONSTRUCTED and `global_row = sample_idx`, i.e. every row is
    used, in order.
  * `n_rows > 512`   ->  `sample_count != global_rows`  ->  512 draws
    from `PCGenerator`, WITH REPLACEMENT (nothing dedupes them).

Both arms ship here and the check reaches both, because
`PORTING_RULES.md:231` ("A NON-DEFAULT PATH IS AN UNCHECKED PATH") cuts
the other way too: on a small dataset the RNG arm is the unreached one.

The second dispatch question -- distributed or not -- is answered by
`quantiles.cuh:164`:

    bool distributed = raft::resource::comms_initialized(handle) &&
                       handle.get_comms().get_size() > 1;

A single-process fit has no comms initialised, so `distributed` is
false, `comm_size` is 1 and `rank` is 0. See DEVIATION 108.

THE PIPELINE, four steps
-------------------------
  1. `detail::sampleOwnedColumnsKernel` (`:41-79`) draws the shared row
     sample and writes it column-major.
  2. `thrust::inclusive_scan` (`:183`) over the per-rank row counts.
     (Ordered before 1 in their source; it produces `global_rows`, which
     1 needs.)
  3. `cub::DeviceSegmentedRadixSort::SortKeys` (`:244` sizing, `:258`
     real) sorts each column's sample as its own segment.
  4. `computeQuantilesBatchedKernel` (`:84-111`) interpolates the
     quantile positions out of the sorted segment and collapses
     duplicates with `thrust::unique(thrust::seq, ...)` (`:104`).

ONE ROW SAMPLE IS SHARED BY EVERY COLUMN. `sampleOwnedColumnsKernel`
derives `global_row` from `sample_idx` alone -- `col` appears only in
the addressing (`:74-76`) -- so column 7's k-th sample and column 3's
k-th sample are the SAME ROW of the input. Their docstring says so
(`:124-126`, "A deterministic global row sample is drawn once with
replacement and shared across feature columns"). A port that seeded per
`(col, sample_idx)` would still produce plausible quantiles and would
not be their function.

================== DEVIATION BLOCK (whole file) ==================
DEVIATION 108. THE DISTRIBUTED ARM IS NOT PORTED. Theirs branches on
`raft::resource::comms_initialized(handle)` (`quantiles.cuh:164`) and,
when true, does an `allgather` of the per-rank row counts (`:177`) and a
SUM `allreduce` of the sparse sample buffer (`:231-235`). Ours fixes
`comm_size = 1` and `rank = 0`.

WHAT IT COSTS, priced rather than waved through: nothing on any fit this
repository can currently run, and it is not a precision or an ordering
change -- it is a whole configuration that is absent. The price is
paid in reach, not in bits: a multi-process fit would silently quantise
against one rank's rows instead of raising. That is why
`compute_quantiles` takes `comm_size` and `rank` as parameters and
RAISES on `comm_size != 1` rather than ignoring them, and why the
`rank_row_offsets` buffer, the `lower_bound` over it and the
`sample_rank == rank` guard are all still here doing their one-rank
work: the shape their kernel has is kept, so that adding the two
collectives later is an addition and not a rewrite. There is no
counterpart in Mojo/MAX for a device-side NCCL collective to port
against, which is the structural reason this is a DECLINE and not a
gap in effort.

DEVIATION 109. `thrust::inclusive_scan` (`quantiles.cuh:183-186`) runs
on the HOST here. Theirs is `rmm::exec_policy(stream)`, i.e. on the
device, over `comm_size + 1` elements.

WHAT IT COSTS: at `comm_size == 1` that is a scan of TWO elements,
`[0, n_rows] -> [0, n_rows]`, and their very next line
(`:187-189`) copies the last element back to the host and
`sync_stream`s on it anyway -- so the value crosses to the host in their
code too, two lines later, unconditionally. Doing the two-element scan
on the host removes one launch and one round trip and changes no value:
integer addition of one number to zero. The structural reason not to
reach for a device scan here is the harder one -- MAX ships none
(`VENDOR_LIBS.md`), a general one is being written by another lane in
`core/` this round, and depending on another lane's unfinished work is
what the lane charter forbids. The scan is written as the loop it is,
inside this file, and if `core/` lands a scan the two-element case
should still not use it.

DEVIATION 110. `double bin_width` and `round(...)` (`quantiles.cuh:90`,
`:94`) are computed on the HOST in Float64 into an index table that the
kernel reads, rather than in the kernel. Theirs:

    double bin_width = static_cast<double>(sample_count) / max_n_bins;
    int idx          = int(round((bin + 1) * bin_width)) - 1;
    idx              = min(max(0, idx), sample_count - 1);

Metal has no float64 in a kernel (`mojolearn hardware limits`,
re-confirmed by this lane's compile of the same expression), so the
literal transcription does not exist as an option. The three candidate
resolutions and why this one:

  * Float32 in the kernel: WRONG, and cheaply shown to be. `bin_width`
    lands in `(0, 4]` for their call site but `(bin + 1)` runs to
    `max_n_bins`, so the product reaches 24 bits of significand at
    `max_n_bins = 2^24`; well before that, a Float32 product that lands
    a half-ULP on the other side of a `.5` boundary picks the NEIGHBOURING
    SORTED SAMPLE, which is a different split threshold, not a rounding
    of one.
  * Exact integer, `idx = (2*(bin+1)*sample_count + max_n_bins) /
    (2*max_n_bins) - 1`: this is round-half-away-from-zero on the exact
    rational, which is what their expression APPROXIMATES but is not
    what it computes -- theirs rounds `sample_count / max_n_bins` to a
    double first and multiplies second, two roundings, and the two
    disagree wherever the exact rational sits on or within a double ULP
    of a half-integer. Constructible: `sample_count = 2`,
    `max_n_bins = 12` puts `(bin+1) * fl(1/6)` at
    0.4999999999999999722 where the exact value is 0.5.
  * HOST Float64 (this one): bit-identical to theirs, because the host
    HAS float64 and the expression is transcribed into it character for
    character.

WHAT IT COSTS: the index table depends on `(bin, sample_count,
max_n_bins)` and on NOTHING ELSE -- no data, no column, no seed -- which
is the property that makes this legal. It is `max_n_bins` Int32 values,
computed and uploaded ONCE PER FOREST beside the four device buffers the
function already allocates, and read by the kernel with the same
`col_data[idx]` gather their line `:96` does. The kernel loses a
multiply and gains a load. C's `round` is round-half-away-from-zero and
Mojo's `round` is not documented to be, so the half case is written out
as `floor(x); if x - floor(x) >= 0.5: +1` on a strictly positive `x`,
rather than trusting a stdlib rounding mode -- this repository has
already been bitten once by assuming a Mojo stdlib numeric matched libm
(`Mojo's log breaks ported tie-breaks`).

DEVIATION 111. `cub::DeviceSegmentedRadixSort::SortKeys`
(`quantiles.cuh:244`, `:258`) is hand-written as
`ensemble/original/segmented_sort.mojo`.

WHAT IT COSTS: CUB is OPEN, so under the charter the correct move is to
port the kernel rather than substitute a vendor primitive -- and there
is no primitive to substitute in any case, since MAX ships no device
sort (`VENDOR_LIBS.md`, checked 2026-08-20). The implementation is not a
new design: it is `gbdt/gpu_util/kernel/segmented_sort.mojo`, this
repository's already-checked port of CatBoost's own
`cub::DeviceSegmentedRadixSort::SortPairs` wrapper, with the value
payload dropped because their call is `SortKeys`. Sorted ORDER is
identical to CUB's for every input including `-0.0`/`+0.0` and NaN,
because CUB's float-to-unsigned twiddle is transcribed and all 32 bits
are used, exactly as their `0, 8 * sizeof(T)` asks. A radix reorder
moves values and never sums them, so there is no arithmetic here to
differ across vendors.

THE DUPLICATION IS A LANE ARTIFACT, not a design choice: `gbdt/` is
owned by another session this round and the lane charter forbids
importing across it. It should be collapsed to one file at merge.

=========== DEVIATION 123 / 403 -- THE SUBNORMAL HARDWARE ROW ===========
What follows is a fifth, real, OUTPUT-CHANGING divergence from cuML.
It was first recorded here without a number; the builder's trace site
already cites it as DEVIATION 123, and DEVIATION 403 (2026-08-22) is
the half of it this lane can act on -- see THE IDENTICAL-MODE ALIGNMENT
at the end of this block.

SUBNORMAL FLOAT32 FEATURE VALUES COLLAPSE INTO ZERO IN `n_bins_array`.

MEASURED this session on the Apple GPU, with a two-kernel probe, not
inferred:

  * a float32 subnormal SURVIVES host -> device -> kernel -> host bit
    for bit, through `enqueue_copy`, through a Float32 load and store
    and through a `bitcast`. MEMORY IS EXACT.
  * the same subnormal COMPARES EQUAL to `+0.0`, to `-0.0`, and to a
    DIFFERENT subnormal: `0x006CE3EE == 0x00000001` returned true. And
    `subnormal + (-0.0)` returned `+0.0`. ARITHMETIC AND COMPARISON
    FLUSH TO ZERO.
  * the smallest NORMAL, `0x00800000`, is correct in all of the above,
    so the boundary is exactly the subnormal range.

WHERE IT BITES, and it is exactly one line: the `thrust::unique`
comparison at `quantiles.cuh:104`. Every subnormal sampled value, and
every zero, is one equivalence class to that comparison on this device
and up to three classes (`-0.0`, `+0.0`, and each distinct subnormal) to
cuML on an IEEE device. So a feature carrying subnormals gets a SMALLER
`n_bins_array[col]` here than on CUDA. The measured spread in
`quantiles_check.mojo` is `n_bins` 5 against 6, 4 against 6, and 5
against 7 on the boundary-value column across the three shapes.

WHERE IT DOES NOT BITE, checked rather than assumed: the SORT. It
compares INTEGER keys and never floats, so the sorted segment is
bit-exact including every subnormal -- arm 2 of the check matches cell
for cell on the same column arm 3 diverges on. The `quantiles_array`
VALUES are likewise exact: they are gathered from the sorted array and
stored without arithmetic. Only the COUNT moves.

THE PRICE: on a real feature this is invisible -- a column whose sampled
values include a float32 subnormal (|x| < 1.18e-38 and nonzero) alongside
a zero is a column whose split candidates near zero were already
degenerate. It cannot be fixed by writing different Mojo: the flush is
in the hardware's compare, and the only escape would be to reimplement
`unique`'s `==` on integer bit patterns, WHICH WOULD BE A DIFFERENT
FUNCTION FROM THEIRS -- `-0.0` and `+0.0` would then stop collapsing,
which cuML does collapse. Copying them exactly on a device that cannot
compare exactly is not available, so the divergence is declared rather
than engineered away.

NOT A CANDIDATE FOR THE `kernel_matrix.mojo` NUMERIC/SCHEDULING TEST as
it stands: that table's rows are knobs this project chooses. This is a
vendor property nobody chose, and it belongs in the table as a
CAPABILITY row. The lane charter forbids editing that file, so the row
is stated here and the orchestrator should move it.

THE IDENTICAL-MODE ALIGNMENT (DEVIATION 403, 2026-08-22). The paragraph
above establishes that cuML's exact behavior is not reproducible on
Metal; what IS available is ONE behavior on every vendor, which is what
`NUMERIC_IDENTICAL` promises. IDENTITY_PATHS row 10's construction
(`original/numerics.ftz`) does exactly this: the `unique` comparison in
`compute_quantiles_batched_kernel` compares `ftz(cur) != ftz(prev)`.
Under FAST, `ftz` is a comptime no-op and this file's behavior is
unchanged bit for bit -- Apple keeps its hardware flush, CUDA keeps
cuML's IEEE compare, and the divergence above stands as documented.
Under IDENTICAL, every vendor flushes the two operands to signed zero
before comparing: on Apple that is bitwise inert (flushing what the
hardware compare would have flushed -- the measured model above
reproduced all observed divergences bit for bit), and on a
denormal-honoring backend it aligns the count to Apple's. The STORED
representative is unaffected either way: the sort is integer-keyed and
bit-exact, so the first element of each run is the same value on every
vendor, and only the run BOUNDARIES move. Result: `n_bins_array` is the
same bytes on Metal, CUDA and HIP under IDENTICAL, at the price of
differing from cuML's CUDA count on subnormal-bearing columns -- the
same price Apple's hardware already charged, now stated once for all
vendors instead of varying by vendor.
==================================================================
"""

from std.math import ceildiv, floor
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, thread_idx

from core.launch_log import log_launch
from core.segmented_sort import segmented_sort_keys_f32
from original.numerics import ftz


# ===========================================================================
# `quantiles.h` -- the view every builder kernel reads through
# ===========================================================================


@fieldwise_init
struct Quantiles[dtype: DType](Copyable, Movable):
    """`ML::DT::Quantiles<DataT, IdxT>`, `quantiles.h:11-17`.

    Two pointers and nothing else, exactly as theirs. `IdxT` is `int`
    throughout cuML's RF, so `n_bins_array` is Int32.
    """

    # `quantiles.h:14` -- quantiles computed for each feature, col-major
    var quantiles_array: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin]
    # `quantiles.h:16` -- the number of bins used for each feature
    var n_bins_array: MutPointer[Int32, MutUntrackedOrigin]


# ===========================================================================
# `raft/random/detail/rng_device.cuh` -- the generator the sample hangs on
# ===========================================================================


@always_inline
def wmul_64bit(a: UInt64, b: UInt64) -> Tuple[UInt64, UInt64]:
    """`raft::wmul_64bit` (`util/integer_utils.hpp:207-233`), returning
    `(hi, lo)`.

    THEIR FILE HAS TWO IMPLEMENTATIONS AND ONLY ONE OF THEM RUNS ON THE
    DEVICE. Under `__CUDA_ARCH__` (`:209-211`) it is two PTX
    instructions:

        asm("mul.hi.u64 %0, %1, %2;" : "=l"(res_hi) : "l"(a), "l"(b));
        asm("mul.lo.u64 %0, %1, %2;" : "=l"(res_lo) : "l"(a), "l"(b));

    i.e. the exact 128-bit product. The `#else` arm (`:213-233`) is a
    32x32 decomposition with explicit carry counting, compiled only for
    the host. `sampleOwnedColumnsKernel` is device code, so THE PATH
    THEIR DISPATCH TAKES IS THE ASM ONE, and what is transcribed here is
    that: the exact 128-bit product, by the standard decomposition
    (Mojo has no 128-bit integer and no PTX on Metal). The `#else` arm
    computes the same function, so this agrees with both.
    """
    var a_lo = a & UInt64(0xFFFFFFFF)
    var a_hi = a >> 32
    var b_lo = b & UInt64(0xFFFFFFFF)
    var b_hi = b >> 32

    var t0 = a_lo * b_lo
    var t1 = a_hi * b_lo
    var t2 = a_lo * b_hi
    var t3 = a_hi * b_hi

    var mid = (t0 >> 32) + (t1 & UInt64(0xFFFFFFFF)) + (t2 & UInt64(0xFFFFFFFF))
    var lo = (t0 & UInt64(0xFFFFFFFF)) | (mid << 32)
    var hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32)
    return (hi, lo)


@fieldwise_init
struct PCGenerator(Copyable, Movable):
    """`raft::random::PCGenerator`, `rng_device.cuh:546-683`.

    Transcribed constant for constant and shift for shift. This is the
    highest-risk construct in the file: `sampleOwnedColumnsKernel` turns
    one 64-bit draw into one row index, so ONE WRONG BIT HERE MOVES A
    ROW, and a moved row moves a quantile, and a moved quantile moves
    every split threshold that uses it. There is no downstream check
    that would notice.

    Its verification is therefore not left to the pipeline check:
    `quantiles_check.mojo` compares draw for draw against RAFT's own
    source compiled by clang++ as a host oracle, in the manner of
    `tools/permutation_oracle/`.
    """

    var pcg_state: UInt64
    var inc: UInt64

    @staticmethod
    @always_inline
    def init_pcg(seed: UInt64, subsequence: UInt64, offset: UInt64) -> Self:
        """`_init_pcg` (`rng_device.cuh:671-680`).

        Reached through the three-argument public ctor (`:569-572`) --
        which is the one `quantiles.cuh:63` calls:

            raft::random::PCGenerator gen(seed, uint64_t(sample_idx),
                                          uint64_t(0));

        NOT the `DeviceState` ctor at `:557`, so there is no
        `base_subsequence` in play. Their body:

            pcg_state = uint64_t(0);
            inc       = (subsequence << 1u) | 1u;
            uint32_t discard;
            next(discard);
            pcg_state += seed;
            next(discard);
            skipahead(offset);

        `next(uint32_t&)` resolves to `next_u32` (`:659`), so the two
        discards are 32-bit draws, not 64-bit ones. Getting that
        overload wrong doubles the warm-up and silently reseeds the
        whole forest.
        """
        var g = Self(pcg_state=UInt64(0), inc=(subsequence << 1) | UInt64(1))
        _ = g.next_u32()
        g.pcg_state += seed
        _ = g.next_u32()
        g.skipahead(offset)
        return g^

    @always_inline
    def skipahead(mut self, offset_in: UInt64):
        """`skipahead` (`rng_device.cuh:576-592`), Brown's arbitrary
        stride. `quantiles.cuh:63` passes `offset = 0`, so the `while`
        body never runs and the tail reduces to
        `pcg_state = pcg_state * 1 + 0`. Kept whole because it is theirs
        and because a nonzero offset must not become a silent no-op.
        """
        var g = UInt64(1)
        var h = UInt64(6364136223846793005)
        var c = UInt64(0)
        var f = self.inc
        var offset = offset_in
        while offset != UInt64(0):
            if (offset & UInt64(1)) != UInt64(0):
                g = g * h
                c = c * h + f
            f = f * (h + UInt64(1))
            h = h * h
            offset >>= 1
        self.pcg_state = self.pcg_state * g + c

    @always_inline
    def next_u32(mut self) -> UInt32:
        """`next_u32` (`rng_device.cuh:599-608`).

            uint64_t oldstate   = pcg_state;
            pcg_state           = oldstate * 6364136223846793005ULL + inc;
            uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            uint32_t rot        = oldstate >> 59u;
            ret = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

        Two truncations are load-bearing and are written out rather than
        left to a coincidence of widths: `xorshifted` and `rot` are
        assigned into `uint32_t` from 64-bit expressions, so the high
        halves are DISCARDED before the rotate. `(-rot) & 31` is unary
        minus on an unsigned, i.e. `2^32 - rot` wrapped, and at
        `rot == 0` it yields 0 -- which is what stops the C++ from being
        a shift by 32.
        """
        var oldstate = self.pcg_state
        self.pcg_state = oldstate * UInt64(6364136223846793005) + self.inc
        var xorshifted = UInt32(
            (((oldstate >> 18) ^ oldstate) >> 27) & UInt64(0xFFFFFFFF)
        )
        var rot = UInt32((oldstate >> 59) & UInt64(0xFFFFFFFF))
        return (xorshifted >> rot) | (
            xorshifted << ((~rot + UInt32(1)) & UInt32(31))
        )

    @always_inline
    def next_u64(mut self) -> UInt64:
        """`next_u64` (`rng_device.cuh:609-617`), low word FIRST.

            a = next_u32(); b = next_u32();
            ret = uint64_t(a) | (uint64_t(b) << 32);

        Swapping the two calls is a different generator that passes
        every distributional test.
        """
        var a = self.next_u32()
        var b = self.next_u32()
        return UInt64(a) | (UInt64(b) << 32)


@always_inline
def custom_next_uniform_int_u64(
    mut gen: PCGenerator, start: UInt64, diff: UInt64
) -> UInt64:
    """`raft::random::custom_next` for
    `UniformIntDistParams<OutType, uint64_t>`, `rng_device.cuh:208-231`.

    THE uint64 OVERLOAD, not the uint32 one at `:184-206`, because
    `quantiles.cuh:59` declares

        raft::random::UniformIntDistParams<std::uint64_t, std::uint64_t>

    and it is the SECOND template argument that selects between them.
    The two differ in more than width: the uint32 arm keeps `m` as a
    single `uint64_t` and returns `m >> 32`, the uint64 arm needs a
    128-bit product and returns its high half.

        uint64_t x = 0;  gen.next(x);
        uint64_t s = params.diff;
        wmul_64bit(m_hi, m_lo, x, s);
        if (m_lo < s) {
          uint64_t t = (-s) % s;          // (2^64 - s) mod s
          while (m_lo < t) { gen.next(x); wmul_64bit(m_hi, m_lo, x, s); }
        }
        *val = OutType(m_hi) + params.start;

    This is Lemire's nearly-divisionless bounded draw. `params.end` is
    set by their caller (`quantiles.cuh:61`) and NEVER READ by this
    function -- only `diff` and `start` are.

    THE REJECTION LOOP IS UNREACHABLE AT THEIR CALL SITE, and that is
    worth writing down rather than discovering later: `s` is
    `global_rows`, so `t = 2^64 mod s`, which for any row count that
    fits in memory is under `s` and the loop is entered with probability
    `t / 2^64 < 3e-17`. It is transcribed anyway, and the check reaches
    it directly with `diff = 2^63 + 1` where the rejection rate is about
    one half -- an unreached branch is an unchecked branch even when the
    reason it is unreached is arithmetic.
    """
    var x = gen.next_u64()
    var s = diff
    var m = wmul_64bit(x, s)
    var m_hi = m[0]
    var m_lo = m[1]
    if m_lo < s:
        var t = (~s + UInt64(1)) % s
        while m_lo < t:
            x = gen.next_u64()
            var mm = wmul_64bit(x, s)
            m_hi = mm[0]
            m_lo = mm[1]
    return m_hi + start


# ===========================================================================
# `quantiles.cuh` -- the two kernels
# ===========================================================================

#: `int n_threads = 256` (`quantiles.cuh:200`).
comptime SAMPLE_BLOCK = 256

#: `std::min(1024, max_n_bins)` (`quantiles.cuh:271`) -- the cap only.
comptime QUANTILE_BLOCK_CAP = 1024


def sample_owned_columns_kernel(
    out_buf: MutPointer[Float32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    rank_row_offsets: MutPointer[UInt64, MutAnyOrigin],
    u64_params: MutPointer[UInt64, MutAnyOrigin],
    comm_size: Int32,
    sample_count: Int32,
    rank: Int32,
    n_rows: Int32,
    n_cols: Int32,
    row_major: Int32,
):
    """`detail::sampleOwnedColumnsKernel`, `quantiles.cuh:41-79`.

    `u64_params` carries their two 64-bit scalar arguments,
    `global_rows` at `[0]` and `seed` at `[1]`. A Metal kernel argument
    must be Int32 in this repository, and the repository's own precedent
    for a 64-bit kernel scalar is a device buffer
    (`gbdt/gpu_util/kernel/bootstrap.mojo:116`, where the seeds arrive
    as `MutPointer[UInt64]`). No value changes; this is spelling.
    """
    var col = Int(block_idx.x)
    var sample_idx = Int(block_idx.y) * Int(block_dim.x) + Int(thread_idx.x)
    if sample_idx >= Int(sample_count):
        return

    var global_rows = u64_params.unsafe_load(0)
    var seed = u64_params.unsafe_load(1)

    # `:57-66`. The identity arm is not an optimisation of the RNG arm:
    # when the budget covers the data, EVERY row is used exactly once
    # and in order, and no generator is constructed at all.
    var global_row = UInt64(sample_idx)
    if UInt64(Int(sample_count)) != global_rows:
        var gen = PCGenerator.init_pcg(seed, UInt64(sample_idx), UInt64(0))
        global_row = custom_next_uniform_int_u64(
            gen, UInt64(0), global_rows
        )

    # `:68-70` -- `::cuda::std::lower_bound(rank_row_offsets + 1,
    # rank_row_offsets + comm_size + 1, global_row + 1)`, then
    # `sample_end - (rank_row_offsets + 1)`. lower_bound returns the
    # first position whose element is NOT LESS than the value.
    var lo = 1
    var hi = Int(comm_size) + 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if rank_row_offsets.unsafe_load(mid) < global_row + UInt64(1):
            lo = mid + 1
        else:
            hi = mid
    var sample_rank = lo - 1

    # `:71`
    var local_begin = rank_row_offsets.unsafe_load(Int(rank))
    if sample_rank == Int(rank):
        # `:73-76`
        var local_row = Int(global_row - local_begin)
        var v: Float32
        if row_major != Int32(0):
            v = data.unsafe_load(local_row * Int(n_cols) + col)
        else:
            v = data.unsafe_load(col * Int(n_rows) + local_row)
        out_buf.unsafe_store(col * Int(sample_count) + sample_idx, v)


def compute_quantiles_batched_kernel(
    quantiles: MutPointer[Float32, MutAnyOrigin],
    n_bins: MutPointer[Int32, MutAnyOrigin],
    sorted_data: MutPointer[Float32, MutAnyOrigin],
    bin_idx: MutPointer[Int32, MutAnyOrigin],
    max_n_bins: Int32,
    sample_count: Int32,
):
    """`computeQuantilesBatchedKernel`, `quantiles.cuh:84-111`.

    `bin_idx` is DEVIATION 110's host-computed table; it holds exactly
    the `idx` their `:94-95` computes, already clamped by their
    `min(max(0, idx), sample_count - 1)`. Everything else is theirs.
    """
    var col = Int(block_idx.x)
    # `:88-89`, the int64 casts are theirs
    var col_q = col * Int(max_n_bins)
    var col_d = col * Int(sample_count)

    # `:92-97`. Grid-stride over bins, so `max_n_bins` above the block
    # size still works -- which is what their `min(1024, max_n_bins)`
    # launch shape requires.
    var bin = Int(thread_idx.x)
    while bin < Int(max_n_bins):
        var idx = Int(bin_idx.unsafe_load(bin))
        quantiles.unsafe_store(
            col_q + bin, sorted_data.unsafe_load(col_d + idx)
        )
        bin += Int(block_dim.x)

    # `:99`
    barrier()

    # `:101-107` -- `thrust::unique(thrust::seq, col_quantiles,
    # col_quantiles + max_n_bins)` then `n_bins[col] = new_last -
    # col_quantiles`. `thrust::seq` is explicit in their source, with
    # the comment "to explicitly disable cuda dynamic parallelism here"
    # (`:103`), so this IS a sequential loop in one thread and porting
    # it as a parallel compaction would be inventing.
    #
    # `unique` removes CONSECUTIVE duplicates and keeps the first of each
    # run; on a sorted range that is the distinct count. Two consequences
    # are theirs and are preserved: `-0.0` and `+0.0` compare EQUAL and
    # collapse even though the sort ordered them apart, and NaN compares
    # unequal to itself and never collapses.
    #
    # DEVIATION 403 -- the operands pass through `numerics.ftz` so that
    # under NUMERIC_IDENTICAL every vendor applies the SAME denormal
    # policy to this comparison (flush, Apple's hardware behavior). Under
    # NUMERIC_FAST `ftz` is a comptime no-op and this line is the
    # transcription. See the deviation block's subnormal hardware row.
    if Int(thread_idx.x) == 0:
        var w = 1
        for r in range(1, Int(max_n_bins)):
            var prev = quantiles.unsafe_load(col_q + w - 1)
            var cur = quantiles.unsafe_load(col_q + r)
            if ftz(cur) != ftz(prev):
                quantiles.unsafe_store(col_q + w, cur)
                w += 1
        n_bins.unsafe_store(col, Int32(w))

    # `:109`
    barrier()


# ===========================================================================
# `quantiles.cuh:113-279` -- the result and the host driver
# ===========================================================================


struct QuantileResult(Movable):
    """`ML::DT::QuantileResult<T>`, `quantiles.cuh:113-120`.

    Theirs owns two `rmm::device_uvector`s and hands out a `Quantiles`
    view; the `view() &&= delete` on `:119` is a C++ way of saying the
    view must not outlive the buffers. Mojo's origin system says the
    same thing, so the `= delete` has no counterpart to write.
    """

    var quantiles_array: DeviceBuffer[DType.float32]
    var n_bins_array: DeviceBuffer[DType.int32]

    def __init__(
        out self,
        var quantiles_array: DeviceBuffer[DType.float32],
        var n_bins_array: DeviceBuffer[DType.int32],
    ):
        self.quantiles_array = quantiles_array^
        self.n_bins_array = n_bins_array^

    def view(self) -> Quantiles[DType.float32]:
        """`quantiles.cuh:118`."""
        return Quantiles[DType.float32](
            quantiles_array=rebind[
                MutPointer[Scalar[DType.float32], MutUntrackedOrigin]
            ](self.quantiles_array.unsafe_ptr()),
            n_bins_array=rebind[
                MutPointer[Int32, MutUntrackedOrigin]
            ](self.n_bins_array.unsafe_ptr()),
        )


def quantile_bin_index(bin: Int, sample_count: Int, max_n_bins: Int) -> Int:
    """`quantiles.cuh:90, 94-95`, evaluated on the host in Float64.

        double bin_width = static_cast<double>(sample_count) / max_n_bins;
        int idx          = int(round((bin + 1) * bin_width)) - 1;
        idx              = min(max(0, idx), sample_count - 1);

    See DEVIATION 110 for why this is here and not in the kernel. The
    rounding is written out because C's `round` is round-half-away-from-
    zero and Mojo's `round` is not documented to be; `(bin + 1) *
    bin_width` is strictly positive at every call, so `floor` plus a
    half-test IS round-half-away there. `int(...)` in C++ truncates
    toward zero, and `round` has already produced an integral value, so
    the truncation is exact and not a second rounding.
    """
    var bin_width = Float64(sample_count) / Float64(max_n_bins)
    var x = Float64(bin + 1) * bin_width
    var r = floor(x)
    if x - r >= Float64(0.5):
        r = r + Float64(1.0)
    var idx = Int(r) - 1
    if idx < 0:
        idx = 0
    if idx > sample_count - 1:
        idx = sample_count - 1
    return idx


def compute_quantiles(
    ctx: DeviceContext,
    mut data: DeviceBuffer[DType.float32],
    max_n_bins: Int,
    n_rows: Int,
    n_cols: Int,
    oversampling_factor: Int = 4,
    seed: UInt64 = UInt64(0),
    row_major: Bool = False,
    comm_size: Int = 1,
    rank: Int = 0,
) raises -> QuantileResult:
    """`ML::DT::computeQuantiles`, `quantiles.cuh:146-279`.

    Their `raft::handle_t` carries the stream; ours is the
    `DeviceContext`. `oversampling_factor = 4` is the default in their
    signature (`:152`) AND the literal their only caller passes
    (`randomforest.cuh:322`), so it is the same number twice and not a
    default this port chose.

    `comm_size` and `rank` are ours, and exist so that DEVIATION 108 is
    a raise rather than a silence.

    `data` is `const T*` in their signature (`:149`) and is never
    written here either; it is spelled `mut` because a MAX device
    buffer hands out a mutable pointer or none at all. Spelling, not
    value -- the same note `dataset.mojo` records for its integer
    widths.
    """
    # `:157-161`, RAFT_EXPECTS
    if max_n_bins <= 0:
        raise Error("max_n_bins must be positive")
    if n_rows <= 0:
        raise Error("n_rows must be positive")
    if n_cols <= 0:
        raise Error("n_cols must be positive")
    if oversampling_factor <= 0:
        raise Error("oversampling_factor must be positive")
    # DEVIATION 108: the arm that is not ported refuses to be entered.
    if comm_size != 1 or rank != 0:
        raise Error(
            "DEVIATION 108: the distributed arm of computeQuantiles"
            " (quantiles.cuh:175-182, :227-238) is NOT PORTED; comm_size"
            " must be 1 and rank 0, got comm_size="
            + String(comm_size)
            + " rank="
            + String(rank)
        )

    # `:169-189`. `rank_row_offsets[0] = 0` (their `cudaMemsetAsync`,
    # `:172`), `rank_row_offsets[1] = n_rows` (their `raft::copy` of
    # `local_row_count`, `:181`), then the inclusive scan. DEVIATION 109.
    var n_offsets = comm_size + 1
    var h_offsets = ctx.enqueue_create_host_buffer[DType.uint64](n_offsets)
    h_offsets.unsafe_ptr().unsafe_store(0, UInt64(0))
    h_offsets.unsafe_ptr().unsafe_store(1, UInt64(n_rows))
    var acc = UInt64(0)
    for i in range(n_offsets):
        acc += h_offsets.unsafe_ptr().unsafe_load(i)
        h_offsets.unsafe_ptr().unsafe_store(i, acc)
    # `:187-190`
    var global_rows = h_offsets.unsafe_ptr().unsafe_load(n_offsets - 1)
    if global_rows == UInt64(0):
        raise Error("global row count must be positive")

    var d_offsets = ctx.enqueue_create_buffer[DType.uint64](n_offsets)
    log_launch("xfer_quantiles_offsets")
    ctx.enqueue_copy(dst_buf=d_offsets, src_ptr=h_offsets.unsafe_ptr())

    # `:193-194` -- `narrow_cast<int>(min(global_rows,
    # checked_mul<uint64>(max_n_bins, oversampling_factor)))`.
    var budget = UInt64(max_n_bins) * UInt64(oversampling_factor)
    if budget // UInt64(oversampling_factor) != UInt64(max_n_bins):
        raise Error("max_n_bins * oversampling_factor overflows")
    var sample_count_u = global_rows if global_rows < budget else budget
    if sample_count_u > UInt64(2147483647):
        raise Error("sample_count does not narrow to int")
    var sample_count = Int(sample_count_u)

    # `:196-198`, `:206-207`
    var total_sample_values = sample_count * n_cols
    if total_sample_values // n_cols != sample_count:
        raise Error("sample_count * n_cols overflows")
    var sampled_columns = ctx.enqueue_create_buffer[DType.float32](
        total_sample_values
    )
    var sorted_samples = ctx.enqueue_create_buffer[DType.float32](
        total_sample_values
    )
    var quantiles_array = ctx.enqueue_create_buffer[DType.float32](
        n_cols * max_n_bins
    )
    var n_bins_array = ctx.enqueue_create_buffer[DType.int32](n_cols)

    # `:210-213` -- every position this rank does not own stays zero.
    sampled_columns.enqueue_fill(Float32(0.0))

    # Their two 64-bit kernel scalars, as a buffer. See the kernel.
    var h_u64 = ctx.enqueue_create_host_buffer[DType.uint64](2)
    h_u64.unsafe_ptr().unsafe_store(0, global_rows)
    h_u64.unsafe_ptr().unsafe_store(1, seed)
    var d_u64 = ctx.enqueue_create_buffer[DType.uint64](2)
    log_launch("xfer_quantiles_seed")
    ctx.enqueue_copy(dst_buf=d_u64, src_ptr=h_u64.unsafe_ptr())

    # `:214-225` -- `dim3 sample_grid(n_cols, ceil(sample_count /
    # n_threads))`, block `n_threads = 256`.
    log_launch("quantiles_sample_columns")
    ctx.enqueue_function[sample_owned_columns_kernel](
        sampled_columns.unsafe_ptr(),
        data.unsafe_ptr(),
        d_offsets.unsafe_ptr(),
        d_u64.unsafe_ptr(),
        Int32(comm_size),
        Int32(sample_count),
        Int32(rank),
        Int32(n_rows),
        Int32(n_cols),
        Int32(1) if row_major else Int32(0),
        grid_dim=(n_cols, ceildiv(sample_count, SAMPLE_BLOCK), 1),
        block_dim=(SAMPLE_BLOCK, 1, 1),
    )

    # `:240-268`. DEVIATION 111. Their segment bounds are
    # `col * sample_count`, so the segments are uniform.
    var sort_a = ctx.enqueue_create_buffer[DType.uint32](total_sample_values)
    var sort_b = ctx.enqueue_create_buffer[DType.uint32](total_sample_values)
    var sort_off = ctx.enqueue_create_buffer[DType.int32](total_sample_values)
    var blocks_wide = ceildiv(sample_count, 512)
    var sort_bsum = ctx.enqueue_create_buffer[DType.int32](
        n_cols * blocks_wide
    )
    segmented_sort_keys_f32(
        ctx,
        n_cols,
        sample_count,
        sampled_columns,
        sorted_samples,
        sort_a,
        sort_b,
        sort_off,
        sort_bsum,
    )

    # DEVIATION 110: their `:90` and `:94-95`, on the host.
    var h_bin_idx = ctx.enqueue_create_host_buffer[DType.int32](max_n_bins)
    for bin in range(max_n_bins):
        h_bin_idx.unsafe_ptr().unsafe_store(
            bin, Int32(quantile_bin_index(bin, sample_count, max_n_bins))
        )
    var d_bin_idx = ctx.enqueue_create_buffer[DType.int32](max_n_bins)
    log_launch("xfer_quantiles_bin_idx")
    ctx.enqueue_copy(dst_buf=d_bin_idx, src_ptr=h_bin_idx.unsafe_ptr())

    # `:271-272` -- grid `n_cols`, block `min(1024, max_n_bins)`.
    var quantile_block = max_n_bins
    if quantile_block > QUANTILE_BLOCK_CAP:
        quantile_block = QUANTILE_BLOCK_CAP
    log_launch("quantiles_batched")
    ctx.enqueue_function[compute_quantiles_batched_kernel](
        quantiles_array.unsafe_ptr(),
        n_bins_array.unsafe_ptr(),
        sorted_samples.unsafe_ptr(),
        d_bin_idx.unsafe_ptr(),
        Int32(max_n_bins),
        Int32(sample_count),
        grid_dim=(n_cols, 1, 1),
        block_dim=(quantile_block, 1, 1),
    )

    # `:275`
    ctx.synchronize()

    # NO CUML COUNTERPART, AND NOT OPTIONAL. Mojo destroys a value at its
    # LAST USE, not at the end of scope, and a buffer handed to a kernel as
    # a raw pointer is NOT used by that launch -- so every scratch buffer
    # above is dead at the `.unsafe_ptr()` that enqueued it, and the next
    # `enqueue_create_buffer` is free to land on it while the kernel that
    # reads it is still queued. Measured once already in this port: a
    # kernel read `n_bins` as -8388609, the bit pattern of `Split::Min()`,
    # through a freed quantiles pointer -- AFTER a green run. These
    # keep-alives must stay AFTER the `synchronize()` above; moving them
    # before it, or deleting them as dead code, restores the hazard.
    _ = h_offsets^
    _ = d_offsets^
    _ = h_u64^
    _ = d_u64^
    _ = sampled_columns^
    _ = sorted_samples^
    _ = sort_a^
    _ = sort_b^
    _ = sort_off^
    _ = sort_bsum^
    _ = h_bin_idx^
    _ = d_bin_idx^

    # `:278`
    return QuantileResult(quantiles_array^, n_bins_array^)
