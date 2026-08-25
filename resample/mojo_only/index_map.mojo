"""The position-to-draw map: which observation replicate `r` position `i` picks.

**THIS FILE IS THE LANE.** Everything else in `resample/` is a fold over what
these four functions return, and every identity claim the lane makes reduces
to one sentence stated here: the value drawn at position `(r, i)` is a PURE
FUNCTION of `(seed, kind, r, i)` and of nothing else. Not of how many
replicates were requested, not of how many threads ran, not of which thread
got there first, not of what was drawn before it.

NO UPSTREAM. cuML, cuVS and RAFT ship no bootstrap, no permutation test and
no Monte Carlo integrator, so `PORTING_RULES.md`'s "COPY, DO NOT IMPROVE"
governs nothing in this file. What IS ported, and is ported without a change
of any kind, is the GENERATOR underneath it: `core/philox.mojo` holds RAFT's
`PhiloxGenerator` (cuRAND's `curandStatePhilox4_32_10_t`) and RAFT's Lemire
range reduction (`custom_next` for `UniformIntDistParams<OutType, uint32_t>`,
`rng_device.cuh:175-196`), both transcribed line by line and held to an
oracle built by compiling their generator. This file spends that generator at
positions of its own choosing; it does not reimplement one bit of it.

================= DEVIATION BLOCK =================

DEVIATION 1690. THE RESAMPLE INDEX IS A POSITION, NOT A STREAM DRAW. THIS IS
THE LANE'S ONE DESIGN DECISION AND EVERY OTHER PROPERTY FOLLOWS FROM IT.

WHAT SciPy DOES. `scipy.stats.bootstrap` reaches
`_resampling.py::_bootstrap_resample`, which is one call:

    i = rng_integers(rng, 0, n, (n_resamples, n))

`rng` is a `numpy.random.Generator` -- a SEQUENTIAL stream. The `(r, i)`
entry of that array is the `(r * n + i)`-th value the stream has produced, so
replicate `r` depends on how many draws preceded it, which depends on
`n_resamples`, on `batch`, and on whether the caller passed a
`bootstrap_result` to extend. `permutation_test` is worse: each replicate is
one `rng.permutation(n_obs)` call, a Fisher-Yates shuffle whose position `i`
depends on every swap before it, and the number of stream words a shuffle
consumes is itself a function of `n_obs`.

WHAT THIS FILE DOES. Position `(r, i)` gets its OWN generator, initialised at
its own subsequence:

    subsequence = (UInt64(r) << 32) | UInt64(i)          `position_subsequence`
    gen         = PhiloxState.init(key, subsequence, 0)
    index       = custom_next_uniform_int_u32(gen, 0, n)

`skipahead_sequence` puts the subsequence in `ctr.z`/`ctr.w`, the HIGH half of
Philox's 128-bit counter (`core/philox.mojo::_incr_hi`, and the warning there
about which half). The per-position rejection loop walks `ctr.x`/`ctr.y`, 2^64
counter blocks away. `position_subsequence` is injective on
`(r, i) in [0, 2^32) x [0, 2^32)`, so two positions never share a stream and
no position can run into its neighbour's.

WHY THE DEPARTURE IS REQUIRED AND NOT PREFERRED. A sequential stream has to be
walked in order. On one CPU that is free; across a GPU it is either a
serialisation or a partition of the stream into per-thread chunks, and the
partition is exactly the thing that made cuML's Random Forest bootstrap a
function of the GPU MODEL (`core/philox.mojo`, DEVIATION 184: RAFT's stride is
`gridDim.x * blockDim.x` with `n_blocks = 4 * getMultiProcessorCount()`, so an
A100 and an H100 draw different rows from one seed). There is no partition of
a sequential stream that is simultaneously parallel, machine-independent and
independent of the batch size. A counter-based generator has no stream to
partition, which is the entire reason Random123 exists.

WHAT IT BUYS, all four gated in `resample_check.mojo`:

  (a) EMBARRASSINGLY PARALLEL WITH NO CROSS-THREAD RNG STATE. No atomic, no
      shuffle, no shared counter, nothing to order.
  (b) BATCH INVARIANCE. Replicate 7 computed alone is bit-identical to
      replicate 7 computed inside a batch of a million.
      (`check_batch_invariance`, and it is the strongest evidence this lane
      can produce.)
  (c) PREFIX STABILITY. A run of 100000 resamples agrees with a run of 10000
      on the first 10000 replicates, bit for bit. A researcher who extends a
      run keeps every number already published from it.
      (`check_prefix_stability`.)
  (d) CROSS-VENDOR IDENTITY. Integer arithmetic all the way to the drawn
      index -- multiply-high, xor, add, compare -- so the map itself carries
      no float at all and cannot round differently anywhere. What remains for
      the identity ledger is the FOLD over the drawn values, which is
      `metrics/mojo_only/pinned_sum.mojo`'s pinned tree, not this file's.

PRICE, and it is a real one, paid on every drawn index. A sequential stream
amortises one Philox block evaluation (ten rounds) over FOUR draws.
`PhiloxState.init` runs `skipahead_sequence` and `skipahead`, and each of them
regenerates the cached block, so a fresh per-position generator costs TWO
block evaluations -- twenty rounds -- for one draw. That is 8x the RNG
arithmetic of the streamed spelling, before the Lemire rejection test. It is
also the reason `monte_carlo_integrate` is the lane's arithmetic-intensity
argument: the draws are the work, they are register-resident integer
arithmetic, and they never touch memory.

NO TIMING HAS BEEN TAKEN. Nothing in this lane has been run.

DEVIATION 1691. THE DERIVED KEY IS FNV-1a64 OVER `(seed, kind)`, NOT THE SEED.
cuML derives its per-tree Random Forest seed the same way -- `rs =
fnv1a32(fnv1a32(fnv1a32_basis, seed), tree_id)`, `randomforest.cuh:119-123` --
and the reason is the same: two DIFFERENT uses of one seed must not share a
stream. Here the uses are the bootstrap draw, the permutation key and the
Monte Carlo coordinate. Without the kind byte, `bootstrap(x, seed=0)` and
`monte_carlo_integrate(f, seed=0)` would consume the same counter positions
and their answers would be correlated in a way no caller expects.

The hash is FNV-1a64 with the constants imported from
`core/identity_trace.mojo` rather than re-chosen, for that file's own stated
reason ("a second hash function in one repository is a second thing to get
wrong"). It is 16 byte-steps of integer arithmetic on the HOST, once per run;
the device is handed the finished key as two Int32 halves and never hashes
anything.

DEVIATION 1692. WE DO NOT PORT RAFT'S LAUNCH GEOMETRY, BECAUSE WE DO NOT HAVE
ONE. `core/philox.mojo::launch_uniform_int` writes `ptr[i]` as the
`(i / RNG_STRIDE)`-th draw of subsequence `i mod RNG_STRIDE`, with
`RNG_STRIDE` frozen at 110592 (DEVIATION 184). That mapping exists to
reproduce cuML's row sample. It is a STREAM-PARTITION map and it is exactly
what (a)-(d) above give up: `ptr[i]` under it depends on `RNG_STRIDE`, which
is a launch property. This lane calls `PhiloxState.init` and
`custom_next_uniform_int_u32` directly and never calls `launch_uniform_int`,
so `RNG_STRIDE` does not appear anywhere in `resample/`.

CONSEQUENCE, STATED SO NOBODY LOOKS FOR IT: a `resample/` bootstrap sample
does NOT equal an `ensemble/` Random Forest bootstrap sample from the same
seed, and it is not meant to. They are different draws for different jobs and
they live behind different `kind` bytes besides.

DEVIATION 1693. THE UNIFORM FLOAT IS RAFT'S `next_float`, AND ITS DIVISION IS
EXACT. `rng_device.cuh:481-487`:

    uint32_t val = next_u32() >> 8;
    ret = static_cast<float>(val) / float(uint32_t(1) << 24);

`val < 2^24` so `Float32(val)` is exact; `2^24` is exact; a quotient of two
exactly-representable float32 values whose ratio is a 24-bit significand
times a power of two is exact under every rounding mode. So this line moves
no bit on any vendor and needs neither `identical_div` nor `ftz` (the
smallest nonzero result is `2^-24`, nowhere near subnormal). Recorded rather
than assumed: `check_index_map_is_positional` compares the device value to
the host value bit for bit.

The AFFINE MAP onto `[lower, upper)` is a different matter and is pinned --
see `draw_uniform_in`.
=================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx

from core.identity_trace import FNV_OFFSET, FNV_PRIME
from core.philox import (
    PhiloxState,
    custom_next_uniform_int_u32,
    philox_next_u64,
)
from mojo_only.numerics import ftz, identical_mul_add


# ===========================================================================
# The kind bytes (DEVIATION 1691)
# ===========================================================================

#: The bootstrap row draw: `draw_row_index`.
comptime RESAMPLE_KIND_BOOTSTRAP: UInt64 = 1

#: The permutation ordering key: `draw_permutation_key`.
comptime RESAMPLE_KIND_PERMUTATION: UInt64 = 2

#: The Monte Carlo coordinate: `draw_unit_float` / `draw_uniform_in`.
comptime RESAMPLE_KIND_MONTE_CARLO: UInt64 = 3

#: The jackknife has NO kind byte and draws nothing: leave-one-out is
#: deterministic. Listed here so a reader does not go looking for it.


# ===========================================================================
# Bounds, all of them refusals rather than wraps
# ===========================================================================

#: `position_subsequence` packs `r` into the high 32 bits of the subsequence
#: word. A replicate index at or above this would alias replicate `r - 2^32`,
#: silently, with a perfectly plausible answer -- exactly the failure class
#: `core/philox.mojo`'s header calls "still uniform and still wrong". Refused
#: by name instead.
comptime RESAMPLE_MAX_REPLICATES = 1 << 32

#: The same for the position within a replicate.
comptime RESAMPLE_MAX_POSITION = 1 << 32

#: The pooled length a permutation replicate may have. The rank pass keys the
#: whole pooled sample into threadgroup memory (8 bytes each, 8 KB at this
#: bound, under the 16 KB floor every column in `mojo_only/kernel_matrix.mojo`
#: declares) and ranks by COUNTING a total order, which is O(N^2) per
#: replicate. Above the bound the answer would still be correct and the cost
#: would stop being reasonable, so it RAISES and names the closure.
comptime PERM_MAX_POOLED = 1024


# ===========================================================================
# The derived key (DEVIATION 1691)
# ===========================================================================


def resample_key(seed: UInt64, kind: UInt64) -> UInt64:
    """FNV-1a64 over the eight bytes of `seed` then the eight bytes of
    `kind`, least significant byte first.

    HOST ONLY, once per run. The kernels take the result as two Int32
    halves and never hash. Byte at a time and in this order because
    `core/identity_trace.mojo::fnv1a64_bytes` is byte at a time and in
    this order, and the two must remain the same function.
    """
    var h = FNV_OFFSET
    for i in range(8):
        h = (h ^ ((seed >> UInt64(8 * i)) & UInt64(0xFF))) * FNV_PRIME
    for i in range(8):
        h = (h ^ ((kind >> UInt64(8 * i)) & UInt64(0xFF))) * FNV_PRIME
    return h


@always_inline
def key_lo(key: UInt64) -> Int32:
    """The low half of a derived key as a Metal-legal kernel argument.

    THE MASK IS NOT DECORATION, and the reason is `core/philox.mojo`'s:
    Mojo 1.0 has been MEASURED sign-extending `Int32 -> UInt32 -> UInt64`
    through a `var`, and the failure hides behind the `<< 32` on the way
    back because the bad bits shift out.
    """
    return (key & UInt64(0xFFFFFFFF)).cast[DType.uint32]().cast[DType.int32]()


@always_inline
def key_hi(key: UInt64) -> Int32:
    """The high half; see `key_lo`."""
    return ((key >> UInt64(32)) & UInt64(0xFFFFFFFF)).cast[
        DType.uint32
    ]().cast[DType.int32]()


@always_inline
def key_join(lo: Int32, hi: Int32) -> UInt64:
    """Recombine the two halves BY BIT PATTERN inside a kernel. Do not
    replace the masks with narrower casts; see `key_lo`."""
    return ((hi.cast[DType.uint64]() & 0xFFFFFFFF) << 32) | (
        lo.cast[DType.uint64]() & 0xFFFFFFFF
    )


# ===========================================================================
# The position map itself (DEVIATION 1690)
# ===========================================================================


@always_inline
def position_subsequence(r: UInt64, i: UInt64) -> UInt64:
    """`(r << 32) | i`, the Philox subsequence position `(r, i)` owns.

    INJECTIVE on `[0, 2^32) x [0, 2^32)`, which is what makes two positions
    independent rather than merely different. The subsequence lands in the
    HIGH half of the 128-bit counter (`skipahead_sequence` ->
    `_incr_hi`, `core/philox.mojo`), so consecutive positions are 2^64
    counter blocks apart and the Lemire rejection loop -- which walks the
    LOW half through `_incr` -- can never carry a position into its
    neighbour's stream.

    The caller has already refused `r` and `i` at or above 2^32
    (`validate_positions`), so the mask below cannot silently alias.
    """
    return (r << UInt64(32)) | (i & UInt64(0xFFFFFFFF))


@always_inline
def draw_row_index(key: UInt64, r: Int, i: Int, n: Int32) -> Int32:
    """The observation replicate `r` picks at position `i`, in `[0, n)`.

    RAFT's own reduction, unchanged: `custom_next_uniform_int_u32(gen, 0,
    n)` is `custom_next` for `UniformIntDistParams<OutType, uint32_t>`
    (`rng_device.cuh:175-196`), the Lemire nearly-divisionless bounded
    draw with its rejection loop, which is the arm cuML's
    `uniformInt<int>(..., 0, n_rows)` reaches. It is NOT `x % n` and the
    difference is a wrong stream on every draw where the branch fires.

    The rejection loop consumes as many words as it needs FROM THIS
    POSITION'S OWN GENERATOR, so a position that rejects twice still
    disturbs nothing outside itself. That is why the map stays pure with a
    variable-length draw in it, and it is why no draw budget had to be
    invented and then bounded.
    """
    var gen = PhiloxState.init(
        key, position_subsequence(UInt64(r), UInt64(i)), UInt64(0)
    )
    return custom_next_uniform_int_u32(gen, Int32(0), n.cast[DType.uint32]())


@always_inline
def draw_unit_float(key: UInt64, r: Int, i: Int) -> Float32:
    """A float32 uniform on `[0, 1)`: RAFT's `next_float`, DEVIATION 1693.

    Exact on every vendor. No `ftz`, no `identical_div`, no transcendental;
    see the deviation block for the proof that the division rounds nowhere.
    """
    var gen = PhiloxState.init(
        key, position_subsequence(UInt64(r), UInt64(i)), UInt64(0)
    )
    var val = gen.next_u32() >> UInt32(8)
    return Float32(Int(val)) / Float32(Int(UInt32(1) << 24))


@always_inline
def draw_uniform_in(
    key: UInt64, r: Int, i: Int, lo: Float32, span: Float32
) -> Float32:
    """`u * span + lo`, RAFT's affine map (`custom_next` for
    `UniformDistParams<OutType>`, `rng_device.cuh:163-173`).

    NOTE THE ORDER, which is theirs: multiply by the span, THEN add the
    start. Distributing it differently is a different float.

    THE ONE PINNED SEAM IN THIS FILE. `res * span + lo` is one rounding or
    two at the codegen's whim (IDENTITY_PATHS row 9), so under IDENTICAL it
    goes through `identical_mul_add` -- an explicit `fma`, one rounding,
    the same on Metal, PTX and AMDGPU -- and under FAST it is the naive
    chain the backend prefers. The result is stored through `ftz` because
    it is a value a kernel writes for another kernel to read, which is row
    10's checklist.

    `span` and `lo` are computed ONCE ON THE HOST (`span = ftz(upper -
    lower)`) and passed in, so no kernel recomputes a subtraction that two
    vendors could round apart at a subnormal.
    """
    var u = draw_unit_float(key, r, i)
    return ftz(identical_mul_add(u, span, lo))


@always_inline
def draw_permutation_key(key: UInt64, r: Int, j: Int) -> UInt64:
    """The 64-bit ordering key pooled position `j` carries in replicate `r`.

    `philox_next_u64` is RAFT's (`rng_device.cuh:455-463`): the FIRST draw
    is the LOW word, and a port that swaps them passes every distributional
    test and produces a different stream.

    SIXTY-FOUR BITS, NOT THIRTY-TWO, and the reason is the tie rule below.
    A permutation read off sorted keys is exactly uniform only where no two
    keys collide; ties are broken by position index, which biases the
    result towards the identity permutation by exactly the collision
    probability. At 32 bits and `N = 1024` that is about `N^2 / 2^33`, one
    replicate in 17 thousand carrying one biased pair; at 64 bits it is
    about `N^2 / 2^65`, which is 3e-14 and is not a thing that happens.
    The bias is bounded and stated rather than assumed away.
    """
    var gen = PhiloxState.init(
        key, position_subsequence(UInt64(r), UInt64(j)), UInt64(0)
    )
    return philox_next_u64(gen)


@always_inline
def permutation_key_lt(ka: UInt64, ja: Int, kb: UInt64, jb: Int) -> Bool:
    """The TOTAL ORDER the permutation ranks on: `(key, position)`
    ascending, position breaking every tie.

    A strict `<` on the key alone is not a total order -- two equal keys
    are unordered and their rank is then decided by whichever thread
    counted first, which is arrival order, which is the whole thing this
    lane exists to not do. Including the position makes the order total,
    the rank a pure function, and the resulting permutation a bijection
    (two positions can never claim one rank).
    """
    if ka != kb:
        return ka < kb
    return ja < jb


# ===========================================================================
# Refusals (host side, before any launch)
# ===========================================================================


def validate_positions(n_resamples: Int, n: Int) raises:
    """Every bound the position map itself carries, refused BY NAME.

    Called by every entry point in `resample/estimator.mojo` before a
    buffer is allocated, so a caller never gets an answer computed from a
    wrapped subsequence.
    """
    if n_resamples <= 0:
        raise Error(
            "resample: n_resamples must be positive; got "
            + String(n_resamples)
            + ". A zero-resample run has no bootstrap distribution, no"
            " standard error and no interval, and SciPy's n_resamples=0 arm"
            " exists only to re-interval an EXISTING bootstrap_result, which"
            " this surface does not carry."
        )
    if n <= 0:
        raise Error(
            "resample: the sample must have at least one observation; got n="
            + String(n)
        )
    if n_resamples >= RESAMPLE_MAX_REPLICATES:
        raise Error(
            "resample: n_resamples must be below 2^32 ("
            + String(RESAMPLE_MAX_REPLICATES)
            + "); got "
            + String(n_resamples)
            + ". The replicate index occupies the HIGH 32 bits of the Philox"
            " subsequence (index_map.mojo::position_subsequence), so a larger"
            " one would alias replicate r - 2^32 and return a uniform, wrong"
            " answer. To close this refusal, widen the subsequence packing to"
            " use the offset word as well and re-gate"
            " check_index_map_is_positional."
        )
    if n >= RESAMPLE_MAX_POSITION:
        raise Error(
            "resample: the sample length must be below 2^32 ("
            + String(RESAMPLE_MAX_POSITION)
            + "); got n="
            + String(n)
            + ". The position index occupies the LOW 32 bits of the Philox"
            " subsequence; see n_resamples above for the closure."
        )


def validate_pooled(n_pooled: Int) raises:
    """`PERM_MAX_POOLED`, refused by name with its closure."""
    if n_pooled > PERM_MAX_POOLED:
        raise Error(
            "permutation_test: the pooled sample length "
            + String(n_pooled)
            + " exceeds PERM_MAX_POOLED = "
            + String(PERM_MAX_POOLED)
            + ". The rank pass keys the pooled sample into threadgroup"
            " memory and ranks by counting a total order, which is O(N^2)"
            " per replicate and 8*N bytes of threadgroup memory. To close"
            " this refusal, replace the counting rank with a pinned"
            " segmented sort over the 64-bit composite key -- the same"
            " construction neighbors/mojo_only/select_radix_identical.mojo"
            " uses for its (distance, index) key -- and re-gate"
            " check_permutation_separable at the larger size."
        )


# ===========================================================================
# The kernels. Every one of them is one thread per POSITION with no shared
# state, which is what DEVIATION 1690(a) means in practice.
# ===========================================================================


def bootstrap_index_kernel(
    out_idx: MutPointer[Int32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_in: Int32,
    n_positions_in: Int32,
):
    """Materialise `n_replicates x n_positions` drawn row indices, starting
    at replicate `r_first`.

    NOT ON THE HOT PATH. The statistic kernels call `draw_row_index`
    in place and store no index at all; this kernel exists so the index map
    can be RECORDED as a card stage (`resample.index_map`) and compared
    against the host mirror, and so `check_batch_invariance` can compare a
    replicate's indices and not only its statistic.

    `r_first` is what makes the batch-invariance gate cheap: replicate 7 of
    a million is `r_first = 7, n_replicates = 1`, and it must return the
    same bytes as row 7 of `r_first = 0, n_replicates = 1000000`.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_positions = Int(n_positions_in)
    var total = Int(n_replicates_in) * n_positions
    if tid >= total:
        return
    var r = Int(r_first_in) + tid // n_positions
    var i = tid % n_positions
    out_idx.unsafe_store(
        tid, draw_row_index(key_join(lo_bits, hi_bits), r, i, n_in)
    )


def monte_carlo_point_kernel(
    out_pts: MutPointer[Float32, MutAnyOrigin],
    lower: MutPointer[Float32, MutAnyOrigin],
    span: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    i_first_in: Int32,
    n_samples_in: Int32,
    n_dims_in: Int32,
):
    """Materialise `n_samples x n_dims` coordinates of the box.

    Same role as `bootstrap_index_kernel` and the same `i_first` handle: it
    is the RECORDED form of the map, and `monte_carlo_integrate` itself
    never writes a point. Sample `i`, dimension `d` sits at position
    `(i, d)` of the SAME map -- one map, three callers, which is why
    `check_index_map_is_positional` covers all of them at once.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_dims = Int(n_dims_in)
    var total = Int(n_samples_in) * n_dims
    if tid >= total:
        return
    var i = Int(i_first_in) + tid // n_dims
    var d = tid % n_dims
    out_pts.unsafe_store(
        tid,
        draw_uniform_in(
            key_join(lo_bits, hi_bits),
            i,
            d,
            lower.unsafe_load(d),
            span.unsafe_load(d),
        ),
    )


# ===========================================================================
# The host mirrors. Serial, same arithmetic, same order.
#
# These are ORACLES, not a CPU path (PORTING_RULES 0b-ii): the map is
# integer arithmetic, so a host mirror is bit-for-bit the device's answer by
# construction rather than by tolerance, and a gate that says so is worth
# having. `resample_oracle.mojo` holds the float side.
# ===========================================================================


def bootstrap_index_host(
    key: UInt64, r_first: Int, n_replicates: Int, n: Int, n_positions: Int
) -> List[Int32]:
    """`bootstrap_index_kernel` on the host, in index order."""
    var out = List[Int32]()
    for rr in range(n_replicates):
        for i in range(n_positions):
            out.append(draw_row_index(key, r_first + rr, i, Int32(n)))
    return out^


def monte_carlo_point_host(
    key: UInt64,
    i_first: Int,
    n_samples: Int,
    lower: List[Float32],
    span: List[Float32],
) -> List[Float32]:
    """`monte_carlo_point_kernel` on the host, in index order."""
    var d = len(lower)
    var out = List[Float32]()
    for ii in range(n_samples):
        for k in range(d):
            out.append(draw_uniform_in(key, i_first + ii, k, lower[k], span[k]))
    return out^


def permutation_ranks_host(key: UInt64, r: Int, n_pooled: Int) -> List[Int32]:
    """The rank of every pooled position under replicate `r`'s total order.

    The host model of the device rank pass: the SAME `O(N^2)` count over
    the SAME `permutation_key_lt`. `ranks[j]` is where position `j` lands,
    so `ranks[j] < n_x` is the membership test the statistic reads.
    """
    var keys = List[UInt64]()
    for j in range(n_pooled):
        keys.append(draw_permutation_key(key, r, j))
    var out = List[Int32]()
    for j in range(n_pooled):
        var rank = 0
        for l in range(n_pooled):
            if permutation_key_lt(keys[l], l, keys[j], j):
                rank += 1
        out.append(Int32(rank))
    return out^
