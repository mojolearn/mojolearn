# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RAFT's Philox generator and `uniformInt`, the thing that decides which rows every tree in the forest is trained on."""

from core.launch_log import log_launch
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext



comptime PHILOX_W32_0: UInt32 = 0x9E3779B9
comptime PHILOX_W32_1: UInt32 = 0xBB67AE85
comptime PHILOX_M4X32_0: UInt32 = 0xD2511F53
comptime PHILOX_M4X32_1: UInt32 = 0xCD9E8D57


@always_inline
def _mulhilo32(a: UInt32, b: UInt32) -> Tuple[UInt32, UInt32]:
    """`mulhilo32` (`curand_philox4x32_x.h`), returning `(hi, lo)`."""
    var p = (a.cast[DType.uint64]() & 0xFFFFFFFF) * (
        b.cast[DType.uint64]() & 0xFFFFFFFF
    )
    return (UInt32((p >> 32) & 0xFFFFFFFF), UInt32(p & 0xFFFFFFFF))


@always_inline
def _philox4x32_round(
    c: SIMD[DType.uint32, 4], k: SIMD[DType.uint32, 2]
) -> SIMD[DType.uint32, 4]:
    """`_philox4x32round` (`curand_philox4x32_x.h`)."""
    var r0 = _mulhilo32(PHILOX_M4X32_0, c[0])
    var r1 = _mulhilo32(PHILOX_M4X32_1, c[2])
    return SIMD[DType.uint32, 4](
        r1[0] ^ c[1] ^ k[0], r1[1], r0[0] ^ c[3] ^ k[1], r0[1]
    )


@always_inline
def philox4x32_10(
    ctr: SIMD[DType.uint32, 4], key: SIMD[DType.uint32, 2]
) -> SIMD[DType.uint32, 4]:
    """`curand_Philox4x32_10` (`curand_philox4x32_x.h`): TEN rounds, with the key bumped by the two Weyl constants BETWEEN rounds -- nine bumps, not ten."""
    var c = ctr
    var k = key
    for _ in range(9):
        c = _philox4x32_round(c, k)
        k[0] = k[0] + PHILOX_W32_0
        k[1] = k[1] + PHILOX_W32_1
    return _philox4x32_round(c, k)


trait U32Stream:
    """The `GenType` half of RAFT's `template <typename GenType, ...> custom_next` (`rng_device.cuh:174-175`), declared here because Mojo needs it before the struct that conforms."""

    def next_u32(mut self) -> UInt32:
        ...


@fieldwise_init
struct PhiloxState(Copyable, Movable, U32Stream):
    """`curandStatePhilox4_32_10_t` and the five functions that drive it, which together are all of RAFT's `PhiloxGenerator` (`rng_device.cuh:426-533`): DI PhiloxGenerator(const DeviceState<PhiloxGenerator>& rng_state, const uint64_t subsequence) { curand_init(rng_state.seed, rng_state.base_subsequence + subsequence, 0, &philox_state); } // `:440-443`..."""

    var ctr: SIMD[DType.uint32, 4]
    var key: SIMD[DType.uint32, 2]
    var output: SIMD[DType.uint32, 4]
    var state: UInt32

    @staticmethod
    @always_inline
    def init(seed: UInt64, subsequence: UInt64, offset: UInt64) -> Self:
        """`curand_init` for Philox (`curand_kernel.h`): state->ctr = make_uint4(0, 0, 0, 0); state->key.x = (unsigned int)seed; state->key.y = (unsigned int)(seed>>32); state->STATE = 0; ..."""
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
        """`Philox_State_Incr(s)` (`curand_philox4x32_x.h`), the +1 carry chain across the whole 128-bit counter: if(++s->ctr.x) return; if(++s->ctr.y) return; if(++s->ctr.z) return; ++s->ctr.w;"""
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
        """`Philox_State_Incr(s, n)` (`curand_philox4x32_x.h`), +n into the LOW half of the counter: unsigned int nlo = (unsigned int)(n); unsigned int nhi = (unsigned int)(n>>32); s->ctr.x += nlo; if( s->ctr.x < nlo ) nhi++; s->ctr.y += nhi; if(nhi <= s->ctr.y) return; if(++s->ctr.z) return; ++s->ctr.w; THE CARRY TEST IS `nhi <= ctr.y`, NOT `ctr.y < nhi`."""
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
        """`Philox_State_Incr_hi(s, n)` (`curand_philox4x32_x.h`), +n into the HIGH half of the counter: s->ctr.z += nlo; if( s->ctr.z < nlo ) nhi++; s->ctr.w += nhi; THIS IS THE FUNCTION THAT MAKES THREADS INDEPENDENT."""
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
        """`skipahead_sequence` (`curand_kernel.h`): `Philox_State_Incr_hi` then regenerate the cached block."""
        self._incr_hi(n)
        self._regen()

    @always_inline
    def skipahead(mut self, n_in: UInt64):
        """`skipahead` (`curand_kernel.h`), n ELEMENTS not n blocks: state->STATE += (n & 3); n /= 4; if( state->STATE > 3 ){ n += 1; state->STATE -= 4; } Philox_State_Incr(state, n); state->output = curand_Philox4x32_10(state->ctr,state->key); `STATE` is a word index inside the 4-word block, so a skip of `n` elements is `n/4` blocks plus `n%4` words,..."""
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
        """`curand(curandStatePhilox4_32_10_t*)` (`curand_kernel.h`): switch(state->STATE++){ default: ret = state->output.x; break; case 1: ret = state->output.y; break; case 2: ret = state->output.z; break; case 3: ret = state->output.w; break; } if(state->STATE == 4){ Philox_State_Incr(state); state->output =..."""
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




@always_inline
def custom_next_uniform_int_u32[
    G: U32Stream, //
](mut gen: G, start: Int32, diff: UInt32) -> Int32:
    """`raft::random::custom_next` for `UniformIntDistParams<OutType, uint32_t>`, `rng_device.cuh:175-196`."""
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



comptime RNG_BLOCK_THREADS = 256

comptime RNG_GRID_BLOCKS = 4 * 108

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
    """`rngKernel<1>` (`rng_device.cuh:675-694`) with `ParamType = UniformIntDistParams<int, uint32_t>`."""
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
    """The same mapping on the host, thread by thread, for a CPU fallback and for the check's non-device layers."""
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
    """`raft::random::uniformInt<int>` with the launch geometry exposed."""
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
    _ = out_buf.unsafe_ptr()


def launch_uniform_int(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.int32],
    n: Int,
    start: Int32,
    end: Int32,
    seed: UInt64,
) raises:
    """cuML's call, `randomforest.cuh:140-142`: raft::random::uniformInt<int>( stream_resources, rng_state, selected_rows.data(), selected_rows.size(), 0, n_rows_); `n` is `selected_rows.size()` (= `n_sampled_rows`), `start` is their `0`, `end` is their `n_rows_`, and `seed` is the fnv1a32 chain's `rs` widened to 64 bits."""
    launch_uniform_int_ex(
        ctx, out_buf, n, start, end, seed, UInt64(0), RNG_STRIDE
    )




@always_inline
def philox_next_u64(mut gen: PhiloxState) -> UInt64:
    """`PhiloxGenerator::next_u64`, `rng_device.cuh:455-463`."""
    var a = gen.next_u32().cast[DType.uint64]() & 0xFFFFFFFF
    var b = gen.next_u32().cast[DType.uint64]() & 0xFFFFFFFF
    return a | (b << 32)


@always_inline
def philox_next_double(mut gen: PhiloxState) -> Float64:
    """`PhiloxGenerator::next_double`, `rng_device.cuh:491-497`."""
    var v = philox_next_u64(gen) >> 11
    return Float64(Int(v)) / Float64(Int(UInt64(1) << 53))


@always_inline
def custom_next_uniform_double(
    mut gen: PhiloxState, start: Float64, end: Float64
) -> Float64:
    """`custom_next` for `UniformDistParams<double>`, `rng_device.cuh:163-173`: OutType res; gen.next(res); *val = (res * (params.end - params.start)) + params.start; Note the ORDER: multiply by the span, THEN add the start."""
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
    """`raft::random::uniform<double>`, `rng_impl.cuh:78-86`, on the host."""
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
