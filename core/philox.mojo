# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RAFT's Philox generator and `uniformInt`, the thing that decides which rows
every tree in the forest is trained on.

NO CUML FILE MIRRORS THIS. It is RAFT -- `cpp/include/raft/random/` -- which
this tree does not mirror file for file, the same way `ensemble/mojo_only/
shuffle_iterator.mojo` holds CCCL and `cluster/mojo_only/` holds ported RAFT
primitives rather than a RAFT directory. RAFT is Apache-2.0, so it is a PORT
target and not a substitution target, and every construct below cites the
header and line it was transcribed from.

PIN. **RAFT v26.08.00**, commit `ebf92684b8a15addcddb43f442a382d528e8bd77`,
checked out read-only at `~/CascadeProjects/upstream/raft-v26.08.00`. cuML does
not name a RAFT commit: `cpp/cmake/thirdparty/get_raft.cmake:8` asks for
version `${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00` = 26.08.00 at tag
`${rapids-cmake-checkout-tag}`, and `cmake/rapids_config.cmake` + `VERSION` +
`RAPIDS_BRANCH` set rapids-cmake to `release/26.08`. That is rapidsai/raft
v26.08.00.

RECORDED BECAUSE IT SAVES THE NEXT PERSON A CLONE: RAFT's RNG headers are
byte-identical between 25.08.00 and 26.08.00 apart from the SPDX licence-header
rewrite, one added `#include <raft/core/detail/macros.hpp>`, and `cub/cub.cuh`
being split into three narrower includes. No constant, no shift and no control
flow moved across that range.

WHY THIS FILE IS WORTH ITS LENGTH. cuML's GPU Random Forest draws every tree's
bootstrap row sample with exactly one call
(`cpp/src/randomforest/randomforest.cuh:140-142`):

    raft::random::uniformInt<int>(
        stream_resources, rng_state, selected_rows.data(),
        selected_rows.size(), 0, n_rows_);

with `rng_state = RngState(rs, GenPhilox)` and
`rs = fnv1a32(fnv1a32(fnv1a32_basis, seed), tree_id)` (`:119-123`). Those row
ids ARE which rows every tree sees. One wrong value and every tree differs, and
nothing downstream can attribute the difference, because a wrong bootstrap
sample is still a perfectly plausible bootstrap sample. So the reference values
in `ensemble/bench/philox_oracle.txt` are produced by COMPILING AND RUNNING
THEIR GENERATOR (`ensemble/tools/philox_oracle/`), and `philox_check.mojo`
compares this file against them at six separable layers.

THE LAYER THAT WILL BITE A REIMPLEMENTER, named in advance. It is NOT the
Philox rounds. Those are a published, self-announcing construction -- ten
rounds, two multiplier constants, two Weyl constants -- and any error in them
scrambles the output visibly on the first draw. It is:

  1. WHICH COUNTER WORD THE SUBSEQUENCE GOES IN. `curand_init` routes the
     subsequence through `skipahead_sequence`, which adds it to `ctr.z`/`ctr.w`
     -- the HIGH half of the 128-bit counter -- while the per-thread draw index
     walks `ctr.x`/`ctr.y`. Putting it in the low half gives a generator that
     passes every distributional test and shares state between threads that
     are 2^66 draws apart.
  2. HOW THE OUTPUT STREAM MAPS ONTO ARRAY POSITIONS. Philox emits FOUR 32-bit
     words per counter block; `curand()` hands them out one at a time, x then
     y then z then w, and only bumps the counter after the fourth. Thread
     `tid` writes indices `tid, tid+stride, tid+2*stride, ...` from ITS OWN
     generator. Get the stride, the start index, or the word order wrong and
     the array is still uniform and still wrong.
  3. THE RANGE REDUCTION. It is NOT `x % diff`. It is Lemire's
     nearly-divisionless bounded draw with a rejection loop, and the loop is
     unreachable at cuML's own call site (see `custom_next_uniform_int_u32`).

================= DEVIATION BLOCK (whole file) =================

DEVIATION 184. THE LAUNCH GEOMETRY IS OURS, BECAUSE THEIRS IS NOT A CONSTANT.
`call_rng_kernel` (`rng_impl.cuh:64-74`) launches

    auto n_threads = 256;
    auto n_blocks  = 4 * getMultiProcessorCount();

and `getMultiProcessorCount()` reads `cudaDevAttrMultiProcessorCount`
(`util/cudart_utils.hpp:301-308`). The kernel's `stride` is
`gridDim.x * blockDim.x` (`rng_device.cuh:683`) and IS the index mapping, so
**RAFT's `uniformInt` output for a fixed seed is a function of the GPU model**:
an A100 (108 SMs) and an H100 (132 SMs) give different bootstrap samples from
the same `RngState`, and cuML inherits that. There is no constant that
reproduces them everywhere, so this port fixes one:

    RNG_BLOCK_THREADS = 256          (theirs, `rng_impl.cuh:70`)
    RNG_GRID_BLOCKS   = 432          = 4 * 108
    RNG_STRIDE        = 110592

PRICE, and it is paid in both directions. What we lose: our rows equal cuML's
only on a 108-SM device (A100, A30, and anything else with that count); on any
other NVIDIA part they differ, and no choice here could have avoided that.
What we gain, and it is the reason for a fixed number rather than a queried
one: our row sample depends on `(seed, tree_id, n_rows, n_sampled_rows)` and
NOTHING ELSE -- not the machine, not the vendor, not the core count. Given this
repository's determinism claims, a device-dependent training set would have
been a defect we inherited rather than a fidelity we kept. `RNG_STRIDE` is a
`comptime` and `launch_uniform_int_ex` takes it as an argument, so reproducing
another device's stride is one call away and is what the oracle table's
`stride` column exists for.

DEVIATION 185. WE LAUNCH FEWER THREADS THAN THE STRIDE, AND THE OUTPUT IS
BIT-IDENTICAL. Theirs launches the full `n_blocks` unconditionally and derives
`stride` from `gridDim.x * blockDim.x`; ours passes `stride` as an explicit
kernel argument (always `RNG_STRIDE`) and launches only
`min(RNG_GRID_BLOCKS, ceildiv(n, RNG_BLOCK_THREADS))` blocks. This is exact,
not an approximation: the value written at index `i` depends only on `stride`
and on `i mod stride`, and a thread with `tid >= n` writes nothing. When
`n >= RNG_STRIDE` the launch is theirs exactly; when `n < RNG_STRIDE` the
threads we drop are precisely the ones whose loop body never executes. The
kernel also returns early when `tid >= n`, before constructing the generator.

PRICE: at `n = 1000` this is 4 blocks instead of 432 -- 109,568 fewer threads,
and 109,568 fewer `curand_init` calls, each of which is two full
Philox-4x32-10 evaluations (20 rounds). Structurally: 1 launch either way,
`4*n` bytes written either way, one extra branch per thread. The cost of NOT
doing it would have been ~2.2 million wasted rounds per tree at cuML's own
default `n_sampled_rows`. No timing was taken.

DEVIATION 186. `OutType(m >> 32) + params.start` (`rng_device.cuh:195`) IS A
SIGNED INT ADDITION THAT CAN OVERFLOW, and this port does it in UInt32 and
reinterprets the bits. Their expression is UB in C++ whenever
`(m >> 32) + start` leaves `int`, which needs `diff > 2^31` and therefore a
NEGATIVE `start` -- reachable through their public API
(`uniformInt<int>(..., -2, INT_MAX)` gives `diff = 2^31 + 1`) but NOT through
cuML's call site, which passes `start = 0` and `end = n_rows <= INT_MAX`. On
every two's-complement target with a compiler that does not exploit the UB the
two spellings agree bit for bit; the oracle is built with `-fwrapv` so that the
committed table records the hardware's answer and this port is held to it. Ours
is the defined spelling of the same arithmetic.

PRICE: none in behaviour at any reachable input. One extra `cast` per drawn
value, folded away.

DEVIATION 187 (A DECLINE, PRICED). THE REST OF THE `rng.cuh` SURFACE IS NOT
PORTED. Specifically:

  (a) `uniformInt`'s 8-byte arm (`rng_impl.cuh:101-106`), which selects
      `UniformIntDistParams<OutType, uint64_t>` and a 128-bit product. cuML's
      RF call is `uniformInt<int>`, so `sizeof(OutType) == 4` and the uint32
      arm is the only one it can reach. The uint64 arm is ALREADY PORTED
      elsewhere for a different call site --
      `ensemble/decisiontree/batched_levelalgo/quantiles.mojo`'s
      `custom_next_uniform_int_u64`, against `PCGenerator` -- so the missing
      piece is only the cross product (Philox x 64-bit), which nothing calls.
      Price of the decline: `launch_uniform_int` takes a
      `DeviceBuffer[DType.int32]`, so asking for a 64-bit output is a COMPILE
      error rather than a silent narrowing; 0 call sites affected today.
  (b) `RngState::advance` (`rng_state.hpp:44-49`) and the `base_subsequence`
      threading around it. `call_rng_kernel` bumps the caller's `RngState` by
      `n_blocks * n_threads` after every launch (`rng_impl.cuh:73`), so a
      REUSED `RngState` yields disjoint subsequences per call. cuML's RowSampler
      cannot reach that: it constructs a FRESH `RngState` per tree from a hashed
      seed (`randomforest.cuh:120-123`) and destroys it at the end of `sample()`,
      so `base_subsequence` is 0 at every call. `PhiloxState.init` and
      `launch_uniform_int_ex` both take `base_subsequence` and honour it -- the
      oracle table has a non-zero-base row -- but `launch_uniform_int` passes 0
      and NOTHING IN THIS PORT MUTATES A PERSISTENT RNG STATE. Price of the
      decline: a future caller that wants two independent draws from one state
      must supply distinct `base_subsequence` values itself; there is no
      stateful object to forget to advance.
  (c) The other fourteen `custom_next` overloads (`rng_device.cuh:153-422`) --
      uniform float, normal, box-Muller, gumbel, logistic, exponential,
      Rayleigh, laplace, bernoulli, scaled-bernoulli and invariant among
      them. `uniform<double>`
      is the one cuML's WEIGHTED bootstrap arm needs (`randomforest.cuh:125-138`);
      that arm is already declined, with its own price, in
      `ensemble/randomforest.mojo`, and it needs float64 on the device besides,
      which this hardware does not have. Price: 0 reachable call sites for the
      unweighted path.
  (d) `GenPC` dispatch. `RAFT_CALL_RNG_FUNC` (`rng_impl.cuh:48-61`) switches on
      `RngState::type`, and cuML pins `GenPhilox` at the RF call site
      (`randomforest.cuh:123`). Only `GenPhilox` is here. Price: none for RF;
      `PCGenerator` already exists in `quantiles.mojo` for the quantile
      sampler's own call site.

DEVIATION-NUMBER COLLISION, REPORTED RATHER THAN QUIETLY WORKED AROUND. This
lane was assigned 181-184 and could use NONE of them. Grepping the whole
repository found, at the moment of writing:

  * 181 -- `ensemble/randomforest.mojo:1241`, the unported weighted /
    zero-weight `RowSampler` arms;
  * 182 -- `extratrees/ported/decisiontree/batched_levelalgo/builder.mojo:732`
    and `:990`, scored cells into reduction candidates;
  * 183 -- `extratrees/mojo_only/device_tree_check.mojo:265`, the device's
    missing `min_impurity_decrease` gate. THIS ONE APPEARED DURING THIS
    LANE'S OWN SESSION: a grep at the start of the work showed 183 free, and a
    grep before writing this paragraph showed it taken, by a lane running
    concurrently.

So this file uses 184-187, re-grepped across the whole repository immediately
before they were spent. The lesson is not "grep harder": a range assigned up
front is not a reservation while other lanes are writing, and four collisions
in two days say the numbers need an allocator, not more diligence.

WHAT IS TRANSCRIBED FROM WHERE, since two upstreams are involved:

  * `PhiloxState` below is cuRAND's `curandStatePhilox4_32_10_t` state machine
    -- `curand_philox4x32_x.h` (the D. E. Shaw Research Random123 construction,
    BSD-3, inside an NVIDIA-proprietary header) plus the five Philox functions
    in `curand_kernel.h`. RAFT's `PhiloxGenerator` (`rng_device.cuh:426-533`)
    is a thin wrapper that calls exactly those. The cuRAND headers are NOT
    vendored here -- their licence forbids redistribution -- so the oracle
    fetches them at build time and only the NUMBERS are committed.
  * `custom_next_uniform_int_u32`, `UniformIntDistParams`, the index mapping in
    `uniform_int_kernel`, and the launch shape are RAFT (Apache-2.0), cited by
    line.
=================================================================
"""

from core.launch_log import log_launch
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext


# ===========================================================================
# cuRAND's Philox-4x32-10, the generator RAFT's `PhiloxGenerator` wraps
# ===========================================================================

#: `curand_philox4x32_x.h`, the four Random123 constants. `W32_*` are the two
#: Weyl (key-bump) constants -- the golden ratio and sqrt(3) in 2.30 fixed
#: point -- and `M4x32_*` are the two round multipliers.
comptime PHILOX_W32_0: UInt32 = 0x9E3779B9
comptime PHILOX_W32_1: UInt32 = 0xBB67AE85
comptime PHILOX_M4X32_0: UInt32 = 0xD2511F53
comptime PHILOX_M4X32_1: UInt32 = 0xCD9E8D57


@always_inline
def _mulhilo32(a: UInt32, b: UInt32) -> Tuple[UInt32, UInt32]:
    """`mulhilo32` (`curand_philox4x32_x.h`), returning `(hi, lo)`.

    THEIR FILE HAS TWO IMPLEMENTATIONS behind `NV_IF_ELSE_TARGET(NV_IS_HOST,
    ...)`: the host arm forms the 64-bit product and splits it, the device arm
    is `*hip = __umulhi(a, b); return a * b;`. They compute the same 64-bit
    product, so this is the host arm, which is also what the oracle compiles.

    THE MASKS ARE NOT DECORATION. `UInt32 -> UInt64` is a widening conversion,
    and this repository has MEASURED Mojo 1.0 sign-extending through such a
    conversion even with the intermediate bound to a `var` (see
    `kernels/builder_kernels.mojo`'s `sample_features_kernel` docstring). An
    explicit `& 0xFFFFFFFF` is arithmetic the folder cannot discard. Here the
    bug would hide completely: a sign-extended `a` still produces the right LOW
    word, so only `hi` would be wrong, and only for `a >= 2^31` -- which both
    round multipliers are.
    """
    var p = (a.cast[DType.uint64]() & 0xFFFFFFFF) * (
        b.cast[DType.uint64]() & 0xFFFFFFFF
    )
    return (UInt32((p >> 32) & 0xFFFFFFFF), UInt32(p & 0xFFFFFFFF))


@always_inline
def _philox4x32_round(
    c: SIMD[DType.uint32, 4], k: SIMD[DType.uint32, 2]
) -> SIMD[DType.uint32, 4]:
    """`_philox4x32round` (`curand_philox4x32_x.h`).

        unsigned int lo0 = mulhilo32(PHILOX_M4x32_0, ctr.x, &hi0);
        unsigned int lo1 = mulhilo32(PHILOX_M4x32_1, ctr.z, &hi1);
        uint4 ret = {hi1^ctr.y^key.x, lo1, hi0^ctr.w^key.y, lo0};

    NOTE THE CROSS. The word that multiplies with `M4x32_0` (from `ctr.x`)
    contributes its HIGH half to output word 2 and its LOW half to output
    word 3, while `M4x32_1` (from `ctr.z`) feeds words 0 and 1. Writing the
    obvious uncrossed version gives a bijection that is not Philox.
    """
    var r0 = _mulhilo32(PHILOX_M4X32_0, c[0])
    var r1 = _mulhilo32(PHILOX_M4X32_1, c[2])
    return SIMD[DType.uint32, 4](
        r1[0] ^ c[1] ^ k[0], r1[1], r0[0] ^ c[3] ^ k[1], r0[1]
    )


@always_inline
def philox4x32_10(
    ctr: SIMD[DType.uint32, 4], key: SIMD[DType.uint32, 2]
) -> SIMD[DType.uint32, 4]:
    """`curand_Philox4x32_10` (`curand_philox4x32_x.h`): TEN rounds, with the
    key bumped by the two Weyl constants BETWEEN rounds -- nine bumps, not ten.
    Their body is ten unrolled `c = _philox4x32round(c, k)` calls with the bump
    written out between consecutive pairs, and the tenth is
    `return _philox4x32round(c, k)` with no bump after it.
    """
    var c = ctr
    var k = key
    for _ in range(9):
        c = _philox4x32_round(c, k)
        k[0] = k[0] + PHILOX_W32_0
        k[1] = k[1] + PHILOX_W32_1
    return _philox4x32_round(c, k)


trait U32Stream:
    """The `GenType` half of RAFT's `template <typename GenType, ...>
    custom_next` (`rng_device.cuh:174-175`), declared here because Mojo needs
    it before the struct that conforms.

    Their `custom_next` is a template over the generator, and `gen.next(x)`
    with `uint32_t x` resolves to `next_u32` for both `PhiloxGenerator`
    (`:523`) and `PCGenerator` (`:649`). Keeping that parameterisation is what
    lets `philox_check.mojo` drive the reduction with a SCRIPTED draw sequence
    and hold the rejection loop to an exact number of consumed draws -- without
    reimplementing the loop on the test side, which would have checked nothing.
    """

    def next_u32(mut self) -> UInt32:
        ...


@fieldwise_init
struct PhiloxState(Copyable, Movable, U32Stream):
    """`curandStatePhilox4_32_10_t` and the five functions that drive it, which
    together are all of RAFT's `PhiloxGenerator` (`rng_device.cuh:426-533`):

        DI PhiloxGenerator(const DeviceState<PhiloxGenerator>& rng_state,
                           const uint64_t subsequence)
        { curand_init(rng_state.seed,
                      rng_state.base_subsequence + subsequence, 0,
                      &philox_state); }                      // `:440-443`
        DI uint32_t next_u32() { return curand(&philox_state); }  // `:449-453`
        DI void next(uint32_t& ret) { ret = next_u32(); }         // `:523`

    Everything RAFT contributes is the argument order. The generator is
    cuRAND's, and the four fields below are its four state members: the 128-bit
    counter, the 64-bit key, the cached 4-word output block, and the index of
    the next word to hand out.

    THE OFFSET IS ALWAYS ZERO on RAFT's DeviceState path, and the subsequence
    is `base_subsequence + subsequence` -- a 64-bit add that is allowed to wrap
    and is not checked.
    """

    var ctr: SIMD[DType.uint32, 4]
    var key: SIMD[DType.uint32, 2]
    var output: SIMD[DType.uint32, 4]
    var state: UInt32

    @staticmethod
    @always_inline
    def init(seed: UInt64, subsequence: UInt64, offset: UInt64) -> Self:
        """`curand_init` for Philox (`curand_kernel.h`):

            state->ctr = make_uint4(0, 0, 0, 0);
            state->key.x = (unsigned int)seed;
            state->key.y = (unsigned int)(seed>>32);
            state->STATE = 0;
            ... boxmuller fields, unused by RAFT ...
            skipahead_sequence(subsequence, state);
            skipahead(offset, state);

        The four box-Muller cache fields are not ported: RAFT's Philox
        `next(float&)` is `next_float()` (`rng_device.cuh:481-487`), NOT
        `curand_uniform`, and its `box_muller_transform` is RAFT's own
        (`rng_device.cuh:118-143`), so cuRAND's normal cache is never touched
        by anything RAFT calls. It holds no bits that feed `curand()`.

        BOTH skipaheads recompute `output`, so a fresh state's first draw is
        `philox4x32_10(ctr, key).x`. With `offset == 0` the second call is
        arithmetically a no-op on the counter and only rewrites `output` with
        the same value -- kept whole anyway, because a nonzero offset must not
        become a silent no-op.
        """
        var s = Self(
            ctr=SIMD[DType.uint32, 4](0, 0, 0, 0),
            key=SIMD[DType.uint32, 2](
                UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF)
            ),
            output=SIMD[DType.uint32, 4](0, 0, 0, 0),
            state=0,
        )
        s.skipahead_sequence(subsequence)
        s.skipahead(offset)
        return s^

    @always_inline
    def _incr(mut self):
        """`Philox_State_Incr(s)` (`curand_philox4x32_x.h`), the +1 carry chain
        across the whole 128-bit counter:

            if(++s->ctr.x) return;
            if(++s->ctr.y) return;
            if(++s->ctr.z) return;
            ++s->ctr.w;
        """
        var c = self.ctr
        c[0] = c[0] + 1
        if c[0] == 0:
            c[1] = c[1] + 1
            if c[1] == 0:
                c[2] = c[2] + 1
                if c[2] == 0:
                    c[3] = c[3] + 1
        self.ctr = c

    @always_inline
    def _incr_n(mut self, n: UInt64):
        """`Philox_State_Incr(s, n)` (`curand_philox4x32_x.h`), +n into the LOW
        half of the counter:

            unsigned int nlo = (unsigned int)(n);
            unsigned int nhi = (unsigned int)(n>>32);
            s->ctr.x += nlo;
            if( s->ctr.x < nlo ) nhi++;
            s->ctr.y += nhi;
            if(nhi <= s->ctr.y) return;
            if(++s->ctr.z) return;
            ++s->ctr.w;

        THE CARRY TEST IS `nhi <= ctr.y`, NOT `ctr.y < nhi`. It is checking
        whether the `ctr.y += nhi` wrapped, and `nhi == 0` (the common case)
        makes it trivially true, which is why the carry into `ctr.z` almost
        never fires. Inverting it would be invisible for 2^32 draws per thread.
        """
        var c = self.ctr
        var nlo = UInt32(n & 0xFFFFFFFF)
        var nhi = UInt32((n >> 32) & 0xFFFFFFFF)
        c[0] = c[0] + nlo
        if c[0] < nlo:
            nhi = nhi + 1
        c[1] = c[1] + nhi
        if not (nhi <= c[1]):
            c[2] = c[2] + 1
            if c[2] == 0:
                c[3] = c[3] + 1
        self.ctr = c

    @always_inline
    def _incr_hi(mut self, n: UInt64):
        """`Philox_State_Incr_hi(s, n)` (`curand_philox4x32_x.h`), +n into the
        HIGH half of the counter:

            s->ctr.z += nlo;
            if( s->ctr.z < nlo ) nhi++;
            s->ctr.w += nhi;

        THIS IS THE FUNCTION THAT MAKES THREADS INDEPENDENT. The subsequence
        lands in `ctr.z`/`ctr.w`, 2^64 draws apart from where the per-thread
        draw counter walks (`ctr.x`/`ctr.y`, via `_incr`). A port that used
        `_incr_n` here would give every thread a correct-looking stream, and
        threads 0 and 1 would collide after 2^32 blocks.

        Note there is no carry OUT of `ctr.w`: the counter saturates by
        wrapping, and cuRAND does not check.
        """
        var c = self.ctr
        var nlo = UInt32(n & 0xFFFFFFFF)
        var nhi = UInt32((n >> 32) & 0xFFFFFFFF)
        c[2] = c[2] + nlo
        if c[2] < nlo:
            nhi = nhi + 1
        c[3] = c[3] + nhi
        self.ctr = c

    @always_inline
    def _regen(mut self):
        """`state->output = curand_Philox4x32_10(state->ctr, state->key);`"""
        self.output = philox4x32_10(self.ctr, self.key)

    @always_inline
    def skipahead_sequence(mut self, n: UInt64):
        """`skipahead_sequence` (`curand_kernel.h`): `Philox_State_Incr_hi`
        then regenerate the cached block. Does NOT touch `STATE`."""
        self._incr_hi(n)
        self._regen()

    @always_inline
    def skipahead(mut self, n_in: UInt64):
        """`skipahead` (`curand_kernel.h`), n ELEMENTS not n blocks:

            state->STATE += (n & 3);
            n /= 4;
            if( state->STATE > 3 ){ n += 1; state->STATE -= 4; }
            Philox_State_Incr(state, n);
            state->output = curand_Philox4x32_10(state->ctr,state->key);

        `STATE` is a word index inside the 4-word block, so a skip of `n`
        elements is `n/4` blocks plus `n%4` words, with a borrow when the word
        index overflows the block. RAFT always passes 0 here
        (`rng_device.cuh:442`); transcribed whole so that a nonzero offset is
        not silently a no-op.
        """
        var n = n_in
        self.state = self.state + UInt32(n & 3)
        n = n // 4
        if self.state > 3:
            n = n + 1
            self.state = self.state - 4
        self._incr_n(n)
        self._regen()

    @always_inline
    def next_u32(mut self) -> UInt32:
        """`curand(curandStatePhilox4_32_10_t*)` (`curand_kernel.h`):

            switch(state->STATE++){
            default: ret = state->output.x; break;
            case 1:  ret = state->output.y; break;
            case 2:  ret = state->output.z; break;
            case 3:  ret = state->output.w; break;
            }
            if(state->STATE == 4){
                Philox_State_Incr(state);
                state->output = curand_Philox4x32_10(state->ctr,state->key);
                state->STATE = 0;
            }
            return ret;

        `switch(STATE++)` reads the OLD index and increments, and `default` is
        the `case 0` arm (it also catches out-of-range values, which cannot
        occur). The counter is bumped and the block regenerated only AFTER the
        fourth word has been handed out -- so a request for a number of values
        that is not a multiple of four simply leaves the tail of the last block
        unconsumed. There is no packing, and nothing to get wrong at a
        non-multiple length.
        """
        var s = self.state
        self.state = s + 1
        var ret: UInt32
        if s == 1:
            ret = self.output[1]
        elif s == 2:
            ret = self.output[2]
        elif s == 3:
            ret = self.output[3]
        else:
            ret = self.output[0]
        if self.state == 4:
            self._incr()
            self._regen()
            self.state = 0
        return ret


# ===========================================================================
# `raft/random/detail/rng_device.cuh` -- the range reduction
# ===========================================================================


@always_inline
def custom_next_uniform_int_u32[
    G: U32Stream, //
](mut gen: G, start: Int32, diff: UInt32) -> Int32:
    """`raft::random::custom_next` for
    `UniformIntDistParams<OutType, uint32_t>`, `rng_device.cuh:175-196`.

    THE uint32 ARM, not the uint64 one at `:198-221`, because it is the SECOND
    template argument that selects between them and `uniformInt`
    (`rng_impl.cuh:94-100`) picks it on `sizeof(OutType) == 4`:

        if (sizeof(OutType) == 4) {
          UniformIntDistParams<OutType, uint32_t> params;
          params.start = start;
          params.end   = end;
          params.diff  = uint32_t(params.end - params.start);
          ...

    cuML's `uniformInt<int>` (`randomforest.cuh:141`) has `OutType = int`, so
    this is the arm the Random Forest reaches, always.

    `params.end` is stored and NEVER READ by the reduction -- only `diff` and
    `start` are. Their body:

        uint32_t x = 0;
        uint32_t s = params.diff;
        gen.next(x);
        uint64_t m = uint64_t(x) * s;
        uint32_t l = uint32_t(m);
        if (l < s) {
          uint32_t t = (-s) % s;  // (2^32 - s) mod s
          while (l < t) {
            gen.next(x);
            m = uint64_t(x) * s;
            l = uint32_t(m);
          }
        }
        *val = OutType(m >> 32) + params.start;

    This is Lemire's nearly-divisionless bounded draw. It is NOT `x % diff`,
    and it is not `(x * diff) >> 32` either: the low word `l` is a rejection
    test that removes the modulo bias exactly, and dropping it gives a
    distribution that is wrong by about `diff / 2^32` and a stream that is
    wrong on every draw where the branch would have fired.

    WHEN THE REJECTION LOOP RUNS. `t = (2^32 - s) mod s = 2^32 mod s`, so entry
    has probability `t / 2^32 < s / 2^32`. AT CUML'S CALL SITE IT IS
    UNREACHABLE IN PRACTICE: `s = n_rows`, so for a ten-million-row problem the
    entry probability is under 2.4e-3 per draw -- reachable, actually, unlike
    the uint64 sibling in `quantiles.mojo` whose probability is 3e-17 -- and
    for `n_rows` a power of two it is exactly zero, because `2^32 mod 2^k == 0`.
    Both facts are load-bearing for a check: the power-of-two case enters the
    `if` and never the `while`, and only a deliberately chosen `diff` reaches
    several iterations. `philox_check.mojo` reaches both, by feeding `x == 0` --
    the universal adversary, since `l == 0` is below `s` for every legal `s` and
    below `t` whenever `t > 0`.

    THE `-s` IS UNARY MINUS ON AN UNSIGNED, i.e. `2^32 - s` wrapped, spelled
    `~s + 1` here because Mojo has no unary minus on UInt32 that means that.

    DEVIATION 186 lives on the last line: theirs is a signed `int` addition
    that can overflow (only for `diff > 2^31`, which needs a negative `start`,
    which cuML never passes); ours does the add in UInt32 and reinterprets,
    which is the same bits with no UB.
    """
    var s = diff
    var x = gen.next_u32()
    var m = (x.cast[DType.uint64]() & 0xFFFFFFFF) * (
        s.cast[DType.uint64]() & 0xFFFFFFFF
    )
    var l = UInt32(m & 0xFFFFFFFF)
    if l < s:
        var t = (~s + UInt32(1)) % s
        while l < t:
            x = gen.next_u32()
            m = (x.cast[DType.uint64]() & 0xFFFFFFFF) * (
                s.cast[DType.uint64]() & 0xFFFFFFFF
            )
            l = UInt32(m & 0xFFFFFFFF)
    var hi = UInt32((m >> 32) & 0xFFFFFFFF)
    return (hi + start.cast[DType.uint32]()).cast[DType.int32]()


# ===========================================================================
# `rngKernel` and its launch
# ===========================================================================

#: `auto n_threads = 256;` (`rng_impl.cuh:70`) -- theirs, unchanged.
comptime RNG_BLOCK_THREADS = 256

#: `auto n_blocks = 4 * getMultiProcessorCount();` (`rng_impl.cuh:71`) with the
#: multiprocessor count FROZEN AT 108. See DEVIATION 184 for why a constant and
#: why this one.
comptime RNG_GRID_BLOCKS = 4 * 108

#: `const LenType stride = gridDim.x * blockDim.x;` (`rng_device.cuh:683`).
comptime RNG_STRIDE = RNG_GRID_BLOCKS * RNG_BLOCK_THREADS  # 110592


def uniform_int_kernel(
    ptr: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
    start: Int32,
    diff_bits: Int32,
    seed_lo: Int32,
    seed_hi: Int32,
    base_lo: Int32,
    base_hi: Int32,
    stride: Int32,
):
    """`rngKernel<1>` (`rng_device.cuh:675-694`) with `ParamType =
    UniformIntDistParams<int, uint32_t>`.

        LenType tid = threadIdx.x + static_cast<LenType>(blockIdx.x) * blockDim.x;
        GenType gen(rng_state, (uint64_t)tid);
        const LenType stride = gridDim.x * blockDim.x;
        for (LenType idx = tid; idx < len; idx += stride * ITEMS_PER_CALL) {
          OutType val[ITEMS_PER_CALL];
          custom_next(gen, val, params, idx, stride);
          for (int i = 0; i < ITEMS_PER_CALL; i++)
            if ((idx + i * stride) < len) ptr[idx + i * stride] = val[i];
        }

    ITEMS_PER_CALL IS 1 FOR `uniformInt` (`rng_impl.cuh:99`, `:105`), so the
    inner loop and the `val` array collapse: this is a plain grid-stride loop
    writing one value per iteration. The 2-item form exists for `normal` and
    friends, which produce values in pairs.

    THE MAPPING, which is the whole reason this file has an oracle:

      * thread `tid` uses subsequence `base_subsequence + tid` -- ONE generator
        per thread, not one per value;
      * thread `tid` writes indices `tid, tid+stride, tid+2*stride, ...`, in
        that order, consuming CONSECUTIVE draws from its own generator;
      * so `ptr[i]` is the `(i / stride)`-th reduced draw of subsequence
        `i mod stride`, and moving `stride` moves every value.

    A `len` that is not a multiple of 4, or of the block size, needs no special
    handling anywhere: the block-of-four is internal to `curand()` (see
    `PhiloxState.next_u32`) and the tail of the last block is simply never
    consumed.

    DEVIATION 185: `stride` arrives as an ARGUMENT rather than as
    `gridDim.x * blockDim.x`, and the early return skips the generator
    construction for threads that would write nothing. Both are bit-identical
    to their launch; see the deviation block.

    THE SEED AND BASE SUBSEQUENCE ARRIVE AS Int32 PAIRS because a Metal kernel
    argument must be Int32 (this repository's standing rule), and they are
    recombined BY BIT PATTERN with explicit masks. Do not replace the masks
    with `var`s and do not replace them with narrower casts: Mojo 1.0 has been
    MEASURED sign-extending `Int32 -> UInt32 -> UInt64` through a `var`, and
    the failure hides behind the `<< 32` because the bad bits shift out.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var len = Int(n)
    if tid >= len:
        return
    var seed = (
        (seed_hi.cast[DType.uint64]() & 0xFFFFFFFF) << 32
    ) | (seed_lo.cast[DType.uint64]() & 0xFFFFFFFF)
    var base = (
        (base_hi.cast[DType.uint64]() & 0xFFFFFFFF) << 32
    ) | (base_lo.cast[DType.uint64]() & 0xFFFFFFFF)
    var diff = diff_bits.cast[DType.uint32]()

    var gen = PhiloxState.init(seed, base + UInt64(tid), UInt64(0))
    var step = Int(stride)
    var idx = tid
    while idx < len:
        ptr[unsafe_offset=idx] = custom_next_uniform_int_u32(gen, start, diff)
        idx += step


def uniform_int_host(
    seed: UInt64,
    base_subsequence: UInt64,
    stride: Int,
    n: Int,
    start: Int32,
    end: Int32,
) raises -> List[Int32]:
    """The same mapping on the host, thread by thread, for a CPU fallback and
    for the check's non-device layers.

    Their `ASSERT(end > start, "'end' must be greater than 'start'")`
    (`rng_impl.cuh:93`) is enforced here and in `launch_uniform_int_ex`.
    """
    if end <= start:
        raise Error(
            "uniform_int: 'end' must be greater than 'start' (rng_impl.cuh:93);"
            " got start=" + String(start) + " end=" + String(end)
        )
    var diff = (end.cast[DType.uint32]() - start.cast[DType.uint32]())
    var out = List[Int32]()
    for _ in range(n):
        out.append(Int32(0))
    for tid in range(stride):
        if tid >= n:
            break
        var gen = PhiloxState.init(
            seed, base_subsequence + UInt64(tid), UInt64(0)
        )
        var idx = tid
        while idx < n:
            out[idx] = custom_next_uniform_int_u32(gen, start, diff)
            idx += stride
    return out^


def launch_uniform_int_ex(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.int32],
    n: Int,
    start: Int32,
    end: Int32,
    seed: UInt64,
    base_subsequence: UInt64,
    stride: Int,
) raises:
    """`raft::random::uniformInt<int>` with the launch geometry exposed.

    `stride` must be a multiple of `RNG_BLOCK_THREADS`; it is the value the
    kernel loops by, and it is what `launch_uniform_int` fixes to `RNG_STRIDE`.
    Exposed so that another device's `4 * SM_count * 256` can be reproduced
    exactly -- which is what the oracle table's `stride` column is for -- and
    so that `base_subsequence` can be supplied by a caller that wants two
    independent draws from one seed (see DEVIATION 187b).
    """
    if end <= start:
        raise Error(
            "uniformInt: 'end' must be greater than 'start' (rng_impl.cuh:93);"
            " got start=" + String(start) + " end=" + String(end)
        )
    if stride % RNG_BLOCK_THREADS != 0 or stride <= 0:
        raise Error(
            "uniform_int: stride must be a positive multiple of "
            + String(RNG_BLOCK_THREADS)
            + " (it is gridDim.x * blockDim.x, rng_device.cuh:683); got "
            + String(stride)
        )
    if n <= 0:
        return

    # DEVIATION 185: their full grid is `stride / RNG_BLOCK_THREADS` blocks;
    # the blocks past `ceildiv(n, RNG_BLOCK_THREADS)` hold only threads whose
    # loop body never runs, so dropping them cannot change a written value.
    var full_blocks = stride // RNG_BLOCK_THREADS
    var need_blocks = ceildiv(n, RNG_BLOCK_THREADS)
    var n_blocks = full_blocks if full_blocks < need_blocks else need_blocks

    var diff = end.cast[DType.uint32]() - start.cast[DType.uint32]()

    log_launch("philox_uniform_int")
    ctx.enqueue_function[uniform_int_kernel](
        out_buf.unsafe_ptr(),
        Int32(n),
        start,
        diff.cast[DType.int32](),
        (seed & 0xFFFFFFFF).cast[DType.uint32]().cast[DType.int32](),
        (seed >> 32).cast[DType.uint32]().cast[DType.int32](),
        (base_subsequence & 0xFFFFFFFF).cast[DType.uint32]().cast[
            DType.int32
        ](),
        (base_subsequence >> 32).cast[DType.uint32]().cast[DType.int32](),
        Int32(stride),
        grid_dim=n_blocks,
        block_dim=RNG_BLOCK_THREADS,
    )
    # Mojo frees a value at its LAST USE, not at end of scope, and a
    # DeviceBuffer handed to a kernel as a raw pointer is dead at
    # `.unsafe_ptr()`. Keep a use alive past the enqueue.
    _ = out_buf.unsafe_ptr()


def launch_uniform_int(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.int32],
    n: Int,
    start: Int32,
    end: Int32,
    seed: UInt64,
) raises:
    """cuML's call, `randomforest.cuh:140-142`:

        raft::random::uniformInt<int>(
            stream_resources, rng_state, selected_rows.data(),
            selected_rows.size(), 0, n_rows_);

    `n` is `selected_rows.size()` (= `n_sampled_rows`), `start` is their `0`,
    `end` is their `n_rows_`, and `seed` is the fnv1a32 chain's `rs` widened to
    64 bits.

    NO STATE IS THREADED, AND NONE NEEDS TO BE. `RngState` carries a
    `base_subsequence` that `call_rng_kernel` advances by `n_blocks * n_threads`
    after every launch (`rng_impl.cuh:73`, `rng_state.hpp:44-49`), so a REUSED
    state gives disjoint subsequences. cuML's RowSampler cannot reach that: it
    builds a FRESH `RngState(rs, GenPhilox)` inside `sample()` from a hashed
    per-tree seed (`randomforest.cuh:120-123`) and lets it die at the end of the
    call, so `base_subsequence` is 0 on every launch and the advance is written
    into an object nobody reads again. This function therefore passes 0, which
    is bit-exact for the Random Forest. A caller that ever wants two draws off
    one seed must use `launch_uniform_int_ex` and choose the base itself. This
    was verified against their source rather than assumed; see DEVIATION 187b.

    The seed is `uint32_t` at their call site (`rs`, `randomforest.cuh:120`)
    and widens to `uint64_t` by zero extension when it enters `RngState`, so
    the high half of the key is 0 for every Random Forest draw. The parameter
    is UInt64 because `RngState::seed` is.

    `out_buf` is NOT synchronized here: the caller owns the buffer and the
    stream, exactly as `sample()` does. It also must outlive this call --
    Mojo frees a value at its last use, so hold it in a struct field or keep a
    use after the `synchronize`.

    TWO SIGNATURE NOTES FOR THE CALLER, both forced by Mojo 1.0 and neither
    optional:

      * the buffer parameter is `mut out_buf`, NOT `out`. `out` is a soft
        keyword and is rejected as an ARGUMENT name ("error: expected argument
        name"), and `mut` is required because `DeviceBuffer.unsafe_ptr()` on an
        immutably-borrowed binding yields an immutable pointer, which
        `enqueue_function` refuses ("does not match the declared function
        argument type Pointer[..., mut=True]"). Calling it from a
        `sample(mut self)` on a `DeviceBuffer` field works unchanged.
      * `end` is exclusive and `end > start` is enforced, raising rather than
        asserting -- theirs is `ASSERT` (`rng_impl.cuh:93`), which is a throw in
        RAFT too.
    """
    launch_uniform_int_ex(
        ctx, out_buf, n, start, end, seed, UInt64(0), RNG_STRIDE
    )


# ===========================================================================
# `raft::random::uniform<double>`, and the pieces `next(double&)` needs.
# `rng_device.cuh:455-463`, `:491-497`, `:511-515`; `rng_device.cuh:163-173`.
#
# WHY THIS IS HOST-ONLY, and it is the whole of DEVIATION 306. These produce
# a `double`, and this device has no float64 at all -- not a precision
# preference, an absent type. Their generator, their pairing, their divisor
# and their affine map are transcribed EXACTLY; only the machine they run on
# moves. The one caller is the weighted bootstrap, which draws
# `n_sampled_rows` values ONCE PER TREE and then does a binary search per
# draw -- work that is O(n log n) on a vector their own code also brings
# back through a `thrust::upper_bound`, so the host is not an unreasonable
# place for it. PRICE: one host pass per tree instead of one device launch,
# and `n_sampled_rows` Int32 row ids copied up instead of drawn in place.
# ===========================================================================


@always_inline
def philox_next_u64(mut gen: PhiloxState) -> UInt64:
    """`PhiloxGenerator::next_u64`, `rng_device.cuh:455-463`.

        a = next_u32(); b = next_u32();
        ret = uint64_t(a) | (uint64_t(b) << 32);

    THE FIRST DRAW IS THE LOW WORD. A port that swaps them passes every
    distributional test and produces a different stream; the oracle compares
    the pairing directly for that reason.
    """
    var a = gen.next_u32().cast[DType.uint64]() & 0xFFFFFFFF
    var b = gen.next_u32().cast[DType.uint64]() & 0xFFFFFFFF
    return a | (b << 32)


@always_inline
def philox_next_double(mut gen: PhiloxState) -> Float64:
    """`PhiloxGenerator::next_double`, `rng_device.cuh:491-497`.

        uint64_t val = next_u64() >> 11;
        ret = double(val) / double(uint64_t(1) << 53);

    53 bits over 2^53, so the result is in [0, 1) and every value is exactly
    representable. Their commented-out alternative on `:513` is
    `curand_uniform_double`, which they did NOT take -- it returns (0, 1].
    """
    var v = philox_next_u64(gen) >> 11
    return Float64(Int(v)) / Float64(Int(UInt64(1) << 53))


@always_inline
def custom_next_uniform_double(
    mut gen: PhiloxState, start: Float64, end: Float64
) -> Float64:
    """`custom_next` for `UniformDistParams<double>`,
    `rng_device.cuh:163-173`:

        OutType res; gen.next(res);
        *val = (res * (params.end - params.start)) + params.start;

    Note the ORDER: multiply by the span, THEN add the start. Distributing
    it differently is a different float.
    """
    var res = philox_next_double(gen)
    return (res * (end - start)) + start


def uniform_double_host(
    seed: UInt64,
    base_subsequence: UInt64,
    stride: Int,
    n: Int,
    start: Float64,
    end: Float64,
) -> List[Float64]:
    """`raft::random::uniform<double>`, `rng_impl.cuh:78-86`, on the host.

    Same thread-to-index mapping as `uniform_int_host`: thread `tid` owns
    subsequence `base + tid` and writes `idx = tid, tid + stride, ...`
    (`rng_device.cuh:680-694`). See DEVIATION 184 for why `stride` is pinned.
    """
    var out = List[Float64]()
    for _ in range(n):
        out.append(Float64(0.0))
    for tid in range(stride):
        if tid >= n:
            break
        var gen = PhiloxState.init(
            seed, base_subsequence + UInt64(tid), UInt64(0)
        )
        var idx = tid
        while idx < n:
            out[idx] = custom_next_uniform_double(gen, start, end)
            idx += stride
    return out^
