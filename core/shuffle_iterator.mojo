# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cuda::shuffle_iterator`, the thing that decides which features a node sees.

NO CUML FILE MIRRORS THIS. It is CCCL -- NVIDIA's CUDA Core Compute
Libraries, the home of Thrust, CUB and libcu++ -- which this tree does not
mirror file for file, the same way `cluster/original/` holds ported RAFT
primitives rather than a RAFT directory. CCCL is open source, so it is a
PORT target and not a substitution target, and every construct below cites
the header it was transcribed from.

PIN. CCCL **3.4.3**, commit `9d65c77f9763cfec20452e4071128d3f0bd2625b`,
checked out read-only at `~/CascadeProjects/upstream/cccl-3.4.3`. cuML does
not pin CCCL directly: `cpp/cmake/thirdparty/get_cccl.cmake` calls
`rapids_cpm_cccl()`, which resolves through rapids-cmake; cuML v26.08.00
sets rapids-cmake 26.08 (`cmake/rapids_config.cmake` + `VERSION`), and
rapids-cmake v26.08.00's `rapids-cmake/cpm/versions.json` names CCCL 3.4.3
at that commit.

WHY THIS FILE IS WORTH ITS LENGTH. cuML picks the features every node in
every tree considers with exactly one expression
(`kernels/builder_kernels.cuh:88-92`):

    uint32_t rng_seed = fnv1a32_hash(seed, treeid, nodeid);
    cuda::shuffle_iterator<IdxT> shuffled_features(
        n, cuda::std::minstd_rand(rng_seed), sample_offset);
    column_samples[sample_idx] = shuffled_features[column_index];

A one-index error here changes every tree in the forest, and NOTHING
downstream can attribute the difference, because a wrong permutation is
still a perfectly uniform-looking permutation. That is why the reference
values in `ensemble/bench/shuffle_oracle.txt` are produced by COMPILING AND
RUNNING THEIR HEADERS (`ensemble/tools/shuffle_oracle/`), and why
`shuffle_check.mojo` compares this file against them index for index at
three separable layers -- the raw LCG, the 24 Feistel keys, and the
permutation -- so a failure says which layer failed.

THE LAYER THAT WILL BITE A REIMPLEMENTER, named in advance. It is not the
Feistel network; that has 24 rounds and one magic constant, and any error in
it scrambles the output visibly. It is `key_stream_next` below --
`uniform_int_distribution<uint32_t>{}` asking for "a random uint32". Nothing
at the call site hints that the right answer is TWO LCG draws, each reduced
by `min() == 1`, each rejection-tested against `2147418112`, packed HIGH 16
BITS FIRST. Every plausible shortcut -- one draw truncated, two draws
without subtracting `min()`, low half first, or dropping the rejection loop
because it fires about once in 33,000 draws -- produces a uniform-looking
permutation that is simply a DIFFERENT one. The rejection branch in
particular would pass a suite of a few hundred trees and then diverge
silently on somebody's dataset.

================= DEVIATION BLOCK (whole file) =================
DEVIATION 121 (CLOSED by this file; it was opened in
`kernels/builder_kernels.mojo` as "not ported, and open").

NO ALGORITHMIC DEVIATION. Every constant, every shift width, every
truncation and the do-while cycle walk are transcribed from CCCL 3.4.3 and
are held to their compiled output by `shuffle_check.mojo`.

Three SPELLING notes, none of which change a value:

1. `minstd_rand`'s `result_type` is `uint_fast32_t`, which is 32-bit on
   Darwin and 64-bit on glibc/x86-64. That selects between Schrage's
   algorithm (`linear_congruential_engine.h:117+`) and the direct
   `(a*x+c)%m` (`:95-103`). Both compute the same integer exactly, so this
   port uses UInt64 arithmetic and the direct form unconditionally. Held to
   both: the oracle was run against an explicitly 64-bit-result-type engine
   as well and produced bit-identical keys and permutations.

2. Their `__feistel_bijection` stores its 24 keys in a member array built
   in the constructor; ours computes them into a fixed-size stack array in
   the same order. Same values, same order, no heap either way.

3. Their `operator[]` is `bijection(current_ + k)` on an iterator holding
   `current_ = start` (`shuffle_iterator.h:135-141`, `:156-162`). Ours is a
   free function taking `(start + k)` directly, because there is no
   iterator protocol to satisfy here -- cuML only ever subscripts it.

ONE PRECONDITION THEIRS DOCUMENTS AND DOES NOT ENFORCE IN RELEASE BUILDS,
kept and made loud here. `random_bijection::operator()` cycle-walks with a
do-while (`random_bijection.h:73-85`) and its own header warns at `:76-77`
that a start index >= `num_elements` MAY LOOP FOREVER -- the orbit through
the Feistel permutation need not contain any value below `n`. Their
`operator[]` guards it with a `_CCCL_ASSERT` that is compiled out in
release. On a GPU that is a hang, not a wrong answer. cuML cannot reach it
(`builder.cuh:240` sets `n_sampled_cols = max(1, max_features * n_cols)`,
`decisiontree.cu:23-25` asserts `0 < max_features <= 1.0`, and the round
loop clamps `n_sampled_cols` to `n_cols - sample_offset`), so this is a
precondition and not a branch -- but it is checked on the HOST at the call
site rather than trusted, because the failure mode is a wedged device.

A DOCUMENTATION DEFECT IN THEIR HEADER, recorded so nobody uses it as a
test vector: `shuffle_iterator.h:85-89` documents
`shuffle_iterator{random_bijection{4, minstd_rand(0xDEADBEEF)}}` as
yielding `1, 3, 2, 0`. Compiled and run at this commit it yields
`2, 0, 1, 3`. The docstring is wrong at the pin; `shuffle_oracle.txt` has
the real value.
=================================================================
"""

from std.bit import bit_width

# --- cuda::std::minstd_rand ------------------------------------------------
# `linear_congruential_engine.h:403`:
#   using minstd_rand = linear_congruential_engine<uint_fast32_t, 48271, 0,
#                                                  2147483647>;
# It is minstd_rand (a = 48271), NOT minstd_rand0 (a = 16807, `:402`).
comptime LCG_A: UInt64 = 48271
comptime LCG_M: UInt64 = 2147483647  # 2^31 - 1
# `:225-226`: min() == 1 because c == 0; max() == m - 1.
comptime LCG_MIN: UInt64 = 1

# --- the key draw ----------------------------------------------------------
# `uniform_int_distribution<uint32_t>{}` has a == 0 and b == UINT32_MAX, so
# `__rp = b - a + 1` WRAPS TO ZERO in uint32 (`uniform_int_distribution.h:
# 231-232`). The `__rp == 0` branch at `:240-243` therefore runs, and the
# rejection loop at `:252-255` is never reached -- the whole draw is
# `__independent_bits_engine<minstd_rand, uint32_t>(g, 32)()`.
#
# That engine's constructor (`:65-107`) with w = 32 and
# _Rp = max() - min() + 1 = 2147483646 computes __m = bit_log2(_Rp) = 30,
# n_ = 32/30 + 1 = 2, w0_ = 16, y0_ = (2147483646 >> 16) << 16 = 2147418112.
# The rebalance test at `:83` is false, so n_ stays 2, n0_ = 2, and the
# second loop (which would use y1_/mask1_) never runs. mask0_ = 0xFFFF.
comptime IBE_ROUNDS = 2
comptime IBE_W0_MASK: UInt64 = 0xFFFF
comptime IBE_Y0: UInt64 = 2147418112

# --- cuda::__feistel_bijection --------------------------------------------
# `feistel_bijection.h:39`
comptime FEISTEL_ROUNDS = 24
# `feistel_bijection.h:87` -- the round function's multiplier. Mitchell et
# al., "Bandwidth-optimal random shuffling for GPUs", ACM TOPC 9.1 (2022).
comptime FEISTEL_M0: UInt64 = 0xD2B74407B1CE6E93


@always_inline
def lcg_seed(s: UInt32) -> UInt64:
    """`linear_congruential_engine.h:363-366`, the (M != 0, C == 0) seeding
    overload: `__x_ = __s % __M == 0 ? 1 : __s % __M;`.

    THE ZERO CASE IS NOT DECORATION. Seeds 0, 1, 2147483647, 2^31 and
    0xFFFFFFFF all reduce to state 1 and therefore produce the IDENTICAL
    stream -- verified against their compiled header, all giving first draws
    `48271, 182605794, 1291394886`. A port that skips the `== 0` rescue
    produces a degenerate all-zero stream for seed 0 and diverges for the
    others. `fnv1a32_hash` can return 0.
    """
    var x = UInt64(Int(s)) % LCG_M
    return 1 if x == 0 else x


@always_inline
def lcg_next(mut x: UInt64) -> UInt64:
    """`linear_congruential_engine.h:270-273`.

    ADVANCE THEN RETURN. `operator()` assigns the new state and returns it,
    so the first value a fresh engine yields is `a * seed % m`, never the
    seed itself. Getting this backwards shifts the entire stream by one and
    still looks random.
    """
    x = (LCG_A * x) % LCG_M
    return x


@always_inline
def key_stream_next(mut x: UInt64) -> UInt32:
    """One `uniform_int_distribution<uint32_t>{}(gen)` draw.

    THE LAYER MOST LIKELY TO BE SILENTLY WRONG -- see the module docstring.
    Two LCG draws, each reduced by `min() == 1`, each rejection-tested, and
    packed HIGH HALF FIRST:

        sp = 0
        twice:
            u = lcg_next() - 1
            while u >= 2147418112: u = lcg_next() - 1
            sp = ((sp << 16) + (u & 0xFFFF)) mod 2^32

    The rejection fires with probability about 3.05e-5 per draw
    (`(2147483646 - 2147418112) / 2147483646`), which is roughly once in
    33,000 -- often enough to matter over a forest, rare enough that a small
    test suite never sees it. It is transcribed rather than dropped for
    exactly that reason, and `shuffle_check.mojo` reaches it deliberately.
    """
    var sp = UInt64(0)
    for _ in range(IBE_ROUNDS):
        var u = lcg_next(x) - LCG_MIN
        while u >= IBE_Y0:
            u = lcg_next(x) - LCG_MIN
        sp = ((sp << 16) + (u & IBE_W0_MASK)) & 0xFFFFFFFF
    return UInt32(Int(sp))


@fieldwise_init
struct FeistelBijection(Copyable, Movable):
    """`cuda::__feistel_bijection`, `feistel_bijection.h`, wrapped in
    `cuda::random_bijection`'s cycle walk (`random_bijection.h:73-85`).

    Plain `(Copyable, Movable)` and NOT `TrivialRegisterPassable`: a
    24-element `Array` member is not itself trivially register-passable, and
    Mojo 1.0 refuses the trait on that ground ("all members of
    'TrivialRegisterPassable' struct must themselves be
    'TrivialRegisterPassable'"). It does not need to be. Their kernel
    constructs this per `sample_idx` as a kernel LOCAL
    (`builder_kernels.cuh:90-92`) and never puts one in shared memory, so it
    never crosses an address space -- which is the only thing the trait
    would have bought.
    """

    var num_elements: UInt64
    var left_bits: UInt64
    var right_bits: UInt64
    var left_mask: UInt64
    var right_mask: UInt64
    var keys: Array[UInt32, FEISTEL_ROUNDS]

    @always_inline
    def __init__(out self, num_elements: Int, seed: UInt32):
        """`feistel_bijection.h:57-72`.

        THE `max(8, ...)` FLOOR IS LOAD-BEARING AND IS THE SECOND-EASIEST
        THING TO GET WRONG. `total_bits = max(8, bit_width(max_index))`
        means the Feistel domain is 256 for EVERY `n <= 256`, not `n`
        rounded up to a power of two. Writing `bit_width(n - 1)` alone is
        right for every `n > 256` and wrong for every realistic feature
        count -- and right again in exactly the large-`n` cases somebody
        stress-testing would reach for.

        `L_bits` rounds DOWN and `R_bits` takes the remainder, so
        `R_bits - L_bits` is always 0 or 1.
        """
        self.num_elements = UInt64(max(1, num_elements))
        var max_index = self.num_elements - 1
        # `bit_width(0) == 0` in both languages, which the floor absorbs.
        var total_bits = UInt64(max(8, Int(bit_width(max_index))))
        self.left_bits = total_bits // 2
        self.right_bits = total_bits - self.left_bits
        self.left_mask = (UInt64(1) << self.left_bits) - 1
        self.right_mask = (UInt64(1) << self.right_bits) - 1
        # `:67-72` -- all 24 keys drawn up front, in index order. The
        # bijection itself consumes no randomness afterwards.
        var x = lcg_seed(seed)
        self.keys = Array[UInt32, FEISTEL_ROUNDS](fill=UInt32(0))
        for i in range(FEISTEL_ROUNDS):
            self.keys[i] = key_stream_next(x)

    @always_inline
    def _round_trip(self, val: UInt64) -> UInt64:
        """`feistel_bijection.h:80-100`, one full pass of 24 rounds.

        Transcribed with its oddities intact, because they are the
        algorithm and not blemishes:

        - `L` is taken as `val >> R_bits` and is NOT masked on entry.
        - This is not a textbook balanced Feistel: the right half is
          re-packed from `B_k`, the LOW half of the same 64-bit product
          whose HIGH half makes `F_k`.
        - `B_k << (R_bits - L_bits)` has a `uint32_t` left operand, so by
          C++ shift rules the result is 32-BIT and truncates there, not at
          64. Doing that shift in 64 bits is a real and invisible bug: it
          only differs when `B_k`'s top bit is live.
        """
        var l = (val >> self.right_bits) & 0xFFFFFFFF
        var r = val & self.right_mask
        for i in range(FEISTEL_ROUNDS):
            var product = (FEISTEL_M0 * l) & 0xFFFFFFFFFFFFFFFF
            var f_k = ((product >> 32) & 0xFFFFFFFF) ^ UInt64(
                Int(self.keys[i])
            )
            var b_k = product & 0xFFFFFFFF
            var l_prime = f_k ^ r
            # 32-bit shifts, per the note above.
            var r_prime = (
                (b_k << (self.right_bits - self.left_bits)) & 0xFFFFFFFF
            ) | (r >> self.left_bits)
            l = l_prime & self.left_mask
            r = r_prime & self.right_mask
        return (l << self.right_bits) | r

    @always_inline
    def __call__(self, index: Int) -> Int:
        """`random_bijection.h:73-85`, the cycle walk.

        A do-while: apply the Feistel AT LEAST ONCE, then keep applying
        until the value lands in `[0, num_elements)`. It terminates for any
        start below `num_elements` because the orbit returns to its start.
        For a start at or above it, their own header warns it may not
        terminate -- see the deviation block; the caller guards that.
        """
        var n = UInt64(index)
        while True:
            n = self._round_trip(n)
            if n < self.num_elements:
                break
        return Int(n)


@always_inline
def shuffled_feature(
    num_elements: Int, seed: UInt32, start: Int, k: Int
) -> Int:
    """`shuffle_iterator.h:135-141` + `:156-162`, i.e.
    `shuffled_features[k]` for an iterator built as
    `shuffle_iterator(num_elements, minstd_rand(seed), start)`.

    Rebuilds the whole bijection per call, which is what THEIR kernel does
    too: `sample_features` constructs a fresh `shuffle_iterator` inside the
    per-`sample_idx` lambda (`builder_kernels.cuh:90-92`), so every one of a
    node's `k` threads redraws all 24 keys and reruns the cycle walk. That
    is redundant work by construction and it is transcribed rather than
    hoisted -- copy, do not improve. Hoisting it per node would be a
    deviation with a measurement attached, and no measurement is being taken
    this round.
    """
    return FeistelBijection(num_elements, seed)(start + k)
