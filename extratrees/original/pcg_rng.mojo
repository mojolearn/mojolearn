# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Counter-based keyed RNG: RAFT's `PCGenerator` and cuML's fnv1a32 key chain.

Host-side. This is the draw machinery ExtraTrees needs so that a threshold is a
pure function of `(seed, tree_id, node_id, feature_id)` rather than of the order
in which a parallel builder happened to visit candidates — the reason recorded
in DEVIATION 130 of `extratrees/DEVIATIONS.md`.

Every arithmetic line below is a transcription of an upstream, pinned:

* RAFT `661a3b840c3300f95f053812a560c952c9d049a4`
  * `cpp/include/raft/random/detail/rng_device.cuh:546` `struct PCGenerator`
    * `:576-593`  `skipahead`   -> `PCGenerator.skipahead`
    * `:599-608`  `next_u32`    -> `PCGenerator.next_u32`
    * `:609-617`  `next_u64`    -> `PCGenerator.next_u64`
    * `:619-626`  `next_i32`    -> `PCGenerator.next_i32`
    * `:628-635`  `next_i64`    -> `PCGenerator.next_i64`
    * `:637-643`  `next_float`  -> `PCGenerator.next_float`
    * `:645-651`  `next_double` -> `PCGenerator.next_double`
    * `:671-680`  `_init_pcg`   -> `PCGenerator.__init__`
  * `:173-183` `custom_next(UniformDistParams<OutType>)`             -> `uniform_float`
  * `:185-206` `custom_next(UniformIntDistParams<OutType,uint32_t>)` -> `uniform_int_u32`
  * `:208-230` `custom_next(UniformIntDistParams<OutType,uint64_t>)` -> `uniform_int_u64`
  * `cpp/include/raft/util/integer_utils.hpp:207-237` `wmul_64bit`   -> `wmul_64bit`
* cuML `00094f7e4e4b5da3a968d193a4da6085fa38f11b`
  * `cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh:100-113`
    `fnv1a32_prime` / `fnv1a32_basis` / `fnv1a32` -> `FNV1A32_PRIME`,
    `FNV1A32_BASIS`, `fnv1a32`
  * same file `:165-176`, the call site in
    `excess_sample_with_replacement_kernel` that chains `threadIdx.x`,
    `treeid`, `nodeid` into `subsequence` and constructs
    `PCGenerator(seed, subsequence, 0)` -> `key_for`

`key_for`, `SplitKey` and `uniform_threshold` at the bottom of the file are
OURS, not a port. They are marked as such.

Checked cell-for-cell against the upstreams' own arithmetic by
`extratrees/original/pcg_rng_check.mojo` against
`extratrees/tools/rng_oracle/pcg_reference.txt`.
"""

from std.math import fma
from std.sys.compile import is_defined



# cuML kernels/builder_kernels.cuh:100-101.
comptime FNV1A32_PRIME: UInt32 = 16777619
comptime FNV1A32_BASIS: UInt32 = 2166136261


def fmix32(h_in: UInt32) -> UInt32:
    """murmur3's 32-bit finalizer: a full-avalanche bijection on `UInt32`.

    OURS (well, Austin Appleby's), DEVIATION 464. It exists because a bare
    fnv1a32 chain has NO avalanche on a small final word: for `txt < 256`
    only the first byte round does anything and the trailing three rounds
    collapse to an affine map `K + perm(txt) * C mod 2^32` with
    `C = FNV1A32_PRIME^3`, whose high bits walk a coarse Weyl rotation
    (`C / 2^32` is within 2^-9 of 5/16, so consecutive inputs land in ~16
    clusters). Ranking or subsetting BY such a hash is uniform per input in
    the marginal sense but structured JOINTLY across inputs. Running the
    combined word through this finalizer restores full bit diffusion; it is
    a bijection, so it cannot introduce collisions the chain did not have.

    Integer-only, vendor-neutral, callable from host and device alike.
    """
    var h = h_in
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def fnv1a32(hash: UInt32, txt: UInt32) -> UInt32:
    """One 32-bit FNV-1a step over the four bytes of `txt`.

    cuML `kernels/builder_kernels.cuh:102-113`, byte for byte. Note the order:
    they mix the LOW byte first, and they do not mask the multiply because
    `uint32_t` arithmetic already wraps.
    """
    var h = hash
    h ^= (txt >> 0) & 0xFF
    h *= FNV1A32_PRIME
    h ^= (txt >> 8) & 0xFF
    h *= FNV1A32_PRIME
    h ^= (txt >> 16) & 0xFF
    h *= FNV1A32_PRIME
    h ^= (txt >> 24) & 0xFF
    h *= FNV1A32_PRIME
    return h


def row_sample_seed(seed: UInt64, tree_id: Int32) -> UInt32:
    """`randomforest.cuh:59-62`, the per-tree ROW-SAMPLE seed. DEVIATION 460.

        auto rs = DT::fnv1a32_basis;
        rs      = DT::fnv1a32(rs, rf_params.seed);
        rs      = DT::fnv1a32(rs, tree_id);

    Their `fnv1a32` takes `uint32_t`, so the `uint64_t seed` is folded in
    ONE round on its low 32 bits and the high half is discarded. The RF lane
    (`ensemble/decisiontree/batched_levelalgo/random_utils.mojo::
    fnv1a32_hash_seed_tree`, its DEVIATION 400) rules that their bug, not
    their design, and gives the high half its round exactly when it is
    nonzero -- every seed below 2^32 keeps the transcription's bits. This
    lane REUSES that seed contract rather than inventing a second one, so a
    bootstrap forest here and one in `ensemble/` draw the same rows from the
    same `(seed, tree_id)`; `forest_check` pins the values. Widened to
    `UInt64` by zero extension at the call, as `RngState::seed` widens it.
    """
    var rs = FNV1A32_BASIS
    rs = fnv1a32(rs, UInt32(seed & 0xFFFFFFFF))
    var hi = UInt32((seed >> 32) & 0xFFFFFFFF)
    if hi != 0:
        rs = fnv1a32(rs, hi)
    rs = fnv1a32(rs, UInt32(Int(tree_id)))
    return rs


struct PCGenerator(Copyable, Movable):
    """RAFT's PCG, `rng_device.cuh:546-683`.

    Constructed from `(seed, subsequence, offset)` exactly as their lower-level
    constructor at `:569-572`. Their other constructor, which takes a
    `DeviceState<PCGenerator>` and adds `rng_state.base_subsequence`, is not
    ported; see DEVIATION 141.
    """

    var pcg_state: UInt64
    var inc: UInt64

    def __init__(out self, seed: UInt64, subsequence: UInt64, offset: UInt64):
        """`_init_pcg`, rng_device.cuh:671-680.

        The two discarded draws are not decoration: the first advances a state
        of zero under `inc`, and the seed is added BETWEEN the two, so seed and
        subsequence enter the state at different points. A from-memory
        `pcg32_srandom_r` that seeds with `state = 0; state += seed;` before the
        first step produces a different stream.
        """
        self.pcg_state = UInt64(0)
        self.inc = (subsequence << 1) | 1
        var discard = self.next_u32()
        _ = discard
        self.pcg_state += seed
        discard = self.next_u32()
        _ = discard
        self.skipahead(offset)

    def skipahead(mut self, offset: UInt64):
        """rng_device.cuh:576-593.

        "Random Number Generation with Arbitrary Strides", F. B. Brown. Note
        `C = C * h + f` happens BEFORE `f` and `h` are updated, and that `G`
        starts at 1 and `C` at 0, so `offset == 0` leaves the state alone.
        """
        var G: UInt64 = 1
        var h: UInt64 = 6364136223846793005
        var C: UInt64 = 0
        var f: UInt64 = self.inc
        var o = offset
        while o:
            if o & 1:
                G = G * h
                C = C * h + f
            f = f * (h + 1)
            h = h * h
            o >>= 1
        self.pcg_state = self.pcg_state * G + C

    def next_u32(mut self) -> UInt32:
        """rng_device.cuh:599-608.

        The output comes from the OLD state, and the rotate distance also comes
        from the OLD state's top 5 bits. `(-rot) & 31` is unsigned negation.
        """
        var oldstate = self.pcg_state
        self.pcg_state = oldstate * 6364136223846793005 + self.inc
        var xorshifted = (((oldstate >> 18) ^ oldstate) >> 27).cast[DType.uint32]()
        var rot = (oldstate >> 59).cast[DType.uint32]()
        return (xorshifted >> rot) | (xorshifted << ((UInt32(0) - rot) & 31))

    def next_u64(mut self) -> UInt64:
        """rng_device.cuh:609-617. FIRST draw is the LOW half."""
        var a = self.next_u32()
        var b = self.next_u32()
        return a.cast[DType.uint64]() | (b.cast[DType.uint64]() << 32)

    def next_i32(mut self) -> Int32:
        """rng_device.cuh:619-626. Sign bit cleared, not shifted away."""
        var val = self.next_u32()
        return (val & 0x7FFFFFFF).cast[DType.int32]()

    def next_i64(mut self) -> Int64:
        """rng_device.cuh:628-635."""
        var val = self.next_u64()
        return (val & 0x7FFFFFFFFFFFFFFF).cast[DType.int64]()

    def next_float(mut self) -> Float32:
        """rng_device.cuh:637-643, uniform in [0, 1).

        24 bits, divided by 2**24. NOT the `x * 2.3283064e-10` multiply a
        from-memory PCG would reach for, and NOT `>> 9 | 0x3f800000`.
        """
        var val = self.next_u32() >> 8
        return val.cast[DType.float32]() / Float32(1 << 24)

    def next_double(mut self) -> Float64:
        """rng_device.cuh:645-651, uniform in [0, 1). 53 bits over 2**53."""
        var val = self.next_u64() >> 11
        return val.cast[DType.float64]() / Float64(1 << 53)


def wmul_64bit(a: UInt64, b: UInt64) -> Tuple[UInt64, UInt64]:
    """Wide 64x64 -> 128 multiply. Returns `(hi, lo)`.

    raft/util/integer_utils.hpp:207-237. See DEVIATION 140: their `__CUDA_ARCH__`
    branch is `mul.hi.u64` / `mul.lo.u64` PTX; this transcribes their portable
    `#else` branch, which is the same product.
    """
    var a_hi = (a >> 32) & 0xFFFFFFFF
    var a_lo = a & 0xFFFFFFFF
    var b_hi = (b >> 32) & 0xFFFFFFFF
    var b_lo = b & 0xFFFFFFFF

    var t0 = a_lo * b_lo
    var t1 = a_hi * b_lo
    var t2 = a_lo * b_hi
    var t3 = a_hi * b_hi

    var carry: UInt64 = 0
    var res_lo = t0
    var trial = res_lo + (t1 << 32)
    if trial < res_lo:
        carry += 1
    res_lo = trial
    trial = res_lo + (t2 << 32)
    if trial < res_lo:
        carry += 1
    res_lo = trial

    # No need to worry about carry in this addition
    var res_hi = (t1 >> 32) + (t2 >> 32) + t3 + carry
    return Tuple(res_hi, res_lo)


def uniform_int_u32(mut gen: PCGenerator, start: UInt32, diff: UInt32) -> UInt32:
    """rng_device.cuh:185-206, `custom_next` for `UniformIntDistParams<_, uint32_t>`.

    Lemire's multiply-shift with the rejection loop, drawing in `[start, start+diff)`.
    It is NOT `x % diff`, and the rejection threshold is `(-diff) % diff`, i.e.
    `(2**32 - diff) mod diff`, not `2**32 % diff`.
    """
    var s = diff
    var x = gen.next_u32()
    var m = x.cast[DType.uint64]() * s.cast[DType.uint64]()
    var l = m.cast[DType.uint32]()
    if l < s:
        var t = (UInt32(0) - s) % s  # (2^32 - s) mod s
        while l < t:
            x = gen.next_u32()
            m = x.cast[DType.uint64]() * s.cast[DType.uint64]()
            l = m.cast[DType.uint32]()
    return (m >> 32).cast[DType.uint32]() + start


def uniform_int_u64(mut gen: PCGenerator, start: UInt64, diff: UInt64) -> UInt64:
    """rng_device.cuh:208-230, `custom_next` for `UniformIntDistParams<_, uint64_t>`.

    This is the overload cuML's feature sampler actually instantiates
    (`builder_kernels.cuh:173`, `UniformIntDistParams<IdxT, uint64_t>`). Same
    Lemire scheme over the wide product, so it burns a 64-bit draw (two u32s)
    per attempt, not one.
    """
    var s = diff
    var x = gen.next_u64()
    var hi_lo = wmul_64bit(x, s)
    var m_hi = hi_lo[0]
    var m_lo = hi_lo[1]
    if m_lo < s:
        var t = (UInt64(0) - s) % s  # (2^64 - s) mod s
        while m_lo < t:
            x = gen.next_u64()
            hi_lo = wmul_64bit(x, s)
            m_hi = hi_lo[0]
            m_lo = hi_lo[1]
    return m_hi + start


@no_inline
def uniform_float(mut gen: PCGenerator, start: Float32, end: Float32) -> Float32:
    """rng_device.cuh:173-183, `custom_next` for `UniformDistParams<float>`.

    `(res * (end - start)) + start` with `res = next_float()`, so the result is
    in `[start, end)` for `start <= end`. Their expression rescales the [0,1)
    draw; it does not redraw, so `start == end` is a constant and `start > end`
    is their caller's problem, not theirs.

    **THE RESCALE IS AN EXPLICIT `fma`, AND THAT REPLACED AN UNFUSED FORM THAT
    COULD NOT BE HELD ON A GPU.** See DEVIATION 142, which was amended rather
    than kept: this used to route the multiply through an `@no_inline` helper
    so the product rounded before the add. On the HOST that worked. On the
    DEVICE it does not — six source-level barriers were measured and every one
    of them fused anyway (plain, `@no_inline`, an integer bitcast round trip,
    both together, and a private `stack_allocation` round trip, which fused on
    device AND made the host fuse). The divergence was real, not theoretical:
    9 of 105 scored cells differed between the host and device draws, worst 14
    ulps, and the device's value was ALWAYS exactly `fma(res, span, start)` and
    never a third value.

    So the choice was never "fused or unfused", it was "fused on device and
    unfused on host, or fused on both". An explicit `fma` is ONE IEEE
    operation, fixed by the source on every backend — strictly more determined
    than either — and it is also what RAFT's own expression becomes under
    nvcc's default `--fmad=true`, i.e. what the upstream actually computes on
    the hardware they ship for.
    """
    var res = gen.next_float()
    return fma(res, end - start, start)


# ===========================================================================
# Everything below this line is OURS. It is not in RAFT or cuML.
# ===========================================================================


@fieldwise_init
struct SplitKey(Copyable, Movable):
    """The two numbers a `PCGenerator` is built from, carried together.

    OURS. cuML holds these as two locals in one kernel
    (`builder_kernels.cuh:167-172`); we hand them across function boundaries,
    so they get a name.
    """

    var seed: UInt64
    var subsequence: UInt64

    def generator(self, offset: UInt64 = 0) -> PCGenerator:
        """`PCGenerator(seed, subsequence, 0)`, cuML builder_kernels.cuh:172."""
        return PCGenerator(self.seed, self.subsequence, offset)


comptime THRESHOLD_KEY_SALT: UInt32 = 0xA24BAED4
"""DEVIATION 465: the salt that makes `key_for`'s stream DISJOINT from the
excess sampler's. Fresh constant, used nowhere else in the tree (grep); its
value is arbitrary and pinned only by `tools/rng_oracle/pcg_reference.txt`."""


comptime KEY_FOR_UNSALTED = is_defined["MOJOLEARN_ET_KEY_UNSALTED"]()
"""A/B arm for DEVIATION 465: build with `-D MOJOLEARN_ET_KEY_UNSALTED=1` to
restore the pre-465 three-link chain (the one that COLLIDES with
`excess_subsequence` at `feature_id == thread_id`). Measurement arm only --
`pcg_rng_check`'s chain pins are against the salted default and go red under
this define, by design."""


def key_for(
    seed: UInt64, tree_id: UInt32, node_id: UInt32, feature_id: UInt32
) -> SplitKey:
    """OURS, and it is DEVIATION 130 in `extratrees/DEVIATIONS.md`.

    cuML chains exactly three components into the subsequence at
    `builder_kernels.cuh:167-170`::

        uint64_t subsequence(fnv1a32_basis);
        subsequence = fnv1a32(subsequence, uint32_t(threadIdx.x));
        subsequence = fnv1a32(subsequence, uint32_t(treeid));
        subsequence = fnv1a32(subsequence, uint32_t(nodeid));

    We chain FOUR, with the SAME `fnv1a32` and the same basis:
    `THRESHOLD_KEY_SALT, feature_id, tree_id, node_id` -- the last three
    deliberately mirroring their `threadIdx, treeid, nodeid`, with
    `feature_id` in the per-candidate slot their `threadIdx.x` occupies (the
    extension ExtraTrees needs because it draws a threshold per
    `(node, feature)` where their sampler draws per `(node, thread)`).

    THE SALT IS LOAD-BEARING, NOT DECORATION (DEVIATION 465). Until
    2026-08-26 this function chained only the three ids -- byte-identical to
    `excess_subsequence`'s chain in `builder_kernels.mojo` -- so for
    `feature_id == thread_id` the threshold stream and the sampler stream
    were the SAME PCG stream. On covtype that collision sat on the
    informative columns 0-7, exactly where the sampler's threads 0-7 draw.
    The comments at `builder_kernels.mojo` (`excess_selection_hash`) and
    `rescue.mojo::RESCUE_FEATURE_SALT` already asserted the streams were
    disjoint; this docstring even said "we chain FOUR" while the body
    chained three. Now it chains four. `MOJOLEARN_ET_KEY_UNSALTED` restores
    the old chain for an A/B.

    The seed is NOT hashed in; like cuML it stays the `seed` argument of the
    generator, so the subsequence is a pure function of the position in the
    forest and the seed is the only global knob.
    """
    var subsequence = FNV1A32_BASIS
    comptime if not KEY_FOR_UNSALTED:
        subsequence = fnv1a32(subsequence, THRESHOLD_KEY_SALT)
    subsequence = fnv1a32(subsequence, feature_id)
    subsequence = fnv1a32(subsequence, tree_id)
    subsequence = fnv1a32(subsequence, node_id)
    return SplitKey(seed, subsequence.cast[DType.uint64]())


def uniform_threshold(key: SplitKey, min_value: Float32, max_value: Float32) -> Float32:
    """OURS: one threshold in `[min_value, max_value)` for this split key.

    The draw itself is RAFT's `uniform_float` above, unchanged. The SHAPE of
    the expression is sklearn's `rand_uniform`
    (`sklearn/tree/_utils.pyx:57-61`)::

        return ((high - low) * <float64_t> our_rand_r(random_state) /
                <float64_t> RAND_R_MAX) + low

    -- a span-scaled uniform offset from the low end, which is what
    `_splitter.pyx`'s random splitter draws between `min_feature_value` and
    `max_feature_value`. Ours differs on three counts, all of them
    consequences of the uniform coming from PCG rather than `our_rand_r`:
    the uniform is already normalized to `[0,1)` so there is no divide by
    `RAND_R_MAX` (and so no `RAND_R_MAX`-inclusive endpoint), it is `Float32`
    rather than `float64_t` (DEVIATION 142), and it is keyed rather than drawn
    from a sequential stream (DEVIATION 130).
    """
    var gen = key.generator()
    return uniform_float(gen, min_value, max_value)


# ===========================================================================
# DEVIATION BLOCK
# ===========================================================================
#
# 140. The wide 64-bit multiply has no PTX fast path.
#
#   Theirs. `raft/util/integer_utils.hpp:207-237` picks between two
#   implementations on `__CUDA_ARCH__`: two inline-PTX instructions
#   (`mul.hi.u64`, `mul.lo.u64`) on device, and a four-partial-product
#   schoolbook expansion with an explicit carry on host.
#
#   Ours. Only the schoolbook expansion, transcribed line for line. Mojo has
#   no inline PTX and, per the repository's ALWAYS-GPU-AGNOSTIC rule, an
#   `if nvidia:` arm would be forbidden even if it did.
#
#   Why it is safe. The two branches compute the same 128-bit product; this is
#   a code-generation difference, not an arithmetic one, and the oracle
#   exercises the result through `uniform_int_u64` over seven ranges
#   (including `diff = 2**63 + 1`, which puts a bit in every position of the
#   high word) on nine streams.
#
#   Price. If this ever runs on device it will be several instructions instead
#   of two. Perf is deferred; when it stops being deferred, this is the row to
#   look at, and the place to look is the KERNEL MATRIX, not this file.
#
# ---------------------------------------------------------------------------
#
# 141. Only the three-argument constructor is ported, and there is no `half`.
#
#   Theirs. `PCGenerator` has a second constructor taking
#   `DeviceState<PCGenerator>` (`rng_device.cuh:557-560`) which forms
#   `_init_pcg(state.seed, state.base_subsequence + subsequence, subsequence)`
#   -- note it passes the subsequence AGAIN as the offset. It also has
#   `next_half` / `next(half&)` (`:653-657`, `:666`).
#
#   Ours. Only `PCGenerator(seed, subsequence, offset)`, which is the one cuML's
#   decision-tree code calls (`builder_kernels.cuh:172`), and no `half`.
#
#   Why. `DeviceState` and `Rng` are RAFT's whole-array RNG driver -- they exist
#   to give one generator per CUDA thread of a fill kernel. Nothing in this lane
#   fills an array with noise; every draw here is keyed. Porting the driver
#   would be porting a caller we do not have. `half` has no use here either:
#   thresholds are compared against feature values, which are `Float32`.
#
#   Price. Anyone who later wants RAFT's `Rng` array fills must port the
#   `DeviceState` constructor, and MUST notice that its offset argument is the
#   subsequence, not zero, or they will silently get a different stream. This
#   is an omission, not a behaviour change: no ported call site takes the
#   missing path, so nothing here can be wrong because of it.
#
# ---------------------------------------------------------------------------
#
# 142. Thresholds are Float32, and the rescale is not allowed to fuse.
#
#   Theirs, twice over. sklearn's `rand_uniform` (`_utils.pyx:57-61`) is
#   `float64_t` throughout. RAFT's `custom_next` for `UniformDistParams<OutType>`
#   (`rng_device.cuh:173-183`) is generic in `OutType` and would happily be
#   instantiated at `double`, and writes the rescale as one C++ expression,
#   `(res * (params.end - params.start)) + params.start`, which both nvcc
#   (`--fmad=true` by default) and clang (`-ffp-contract=on`) may contract into
#   a single FMA.
#
#   Ours. `Float32`, because there is no `float64` on device and a threshold
#   that changes type when it moves between host and device is worse than one
#   that is `Float32` everywhere. And the multiply is isolated in a
#   `@no_inline` helper, `_product_f32`, so the product is rounded before the
#   add: Mojo contracts multiply-then-add ACROSS statements, so writing the two
#   halves on separate lines is not enough.
#
#   What was actually measured, because this is easy to overclaim. Three runs of
#   `pcg_rng_check.mojo`:
#     * unfused Mojo vs unfused reference -- 0 of 2658 cells differ. Shipped.
#     * FUSED Mojo (the `@no_inline` removed, written as one expression) vs the
#       unfused reference -- 99 cells differ, every one of them by one ULP. So
#       Mojo does contract here, and the `@no_inline` barrier is load-bearing,
#       not defensive.
#     * fused Mojo vs a reference built WITHOUT `-ffp-contract=off` -- also 0
#       of 2658. Both toolchains lower the fused form to the same arm64
#       `fmadd`, so fused-vs-fused agrees on this box.
#
#   So unfused is a CHOICE, not the only thing that works. It is chosen because
#   the fused form's agreement is an accident of two compilers making the same
#   optimisation decision on one target, and it would have to be re-established
#   on every backend; the unfused form is determined by the source. FMA and
#   mul-then-add differ by one rounding of the product, and on a threshold
#   compared with `<=` against feature values, one rounding decides which side
#   a row falls on when the threshold lands exactly on a value -- which, for a
#   threshold drawn between the observed min and max, is not rare.
#
#   Price. One extra rounding of accuracy against a `float64` sklearn, one
#   un-inlinable call in the threshold path, and a build flag in
#   `tools/rng_oracle/build.sh` that must not be dropped. The accuracy one is a
#   quality-band question that DEVIATION 130 already put out of gate scope.
