# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST candidates for the transformer block FORWARD (lane afn-attn,
2026-10-03). Every kernel and launcher here is reached only under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` plus
one `-D MOJOLEARN_AFN_ATTN_<NAME>` define (default off); IDENTICAL and every
other vendor compile main's code unchanged. The block orchestration that
decides when each candidate runs lives in modeling_llama.mojo
(`_afn_block_ok`, `_afn_attention_forward`, `_afn_mlp_and_residual2`);
this file holds the kernels and launchers only, so the import is one way.

Candidates (docs/apple-fast/notes/neural-attn.md has the launch profile):

- NORM_SG: RMSNorm with one simdgroup (32 lanes) per token row, eight rows
  per block, vector-4 loads and a shuffle butterfly for the sum of squares.
  Main's kernel is one THREAD per row with a serial 384-wide fold, so at
  M = 2048 it runs 16 blocks of 128 threads; this runs 256 blocks of 256.
  FAST permits the fold order; f32 throughout.
- ROPE_CACHE: q RoPE, k RoPE, the K/V cache append and the two cache
  copies in ONE launch and no wait (main: 2 rope launches, 1 append, 2
  device copies, 1 synchronize). Fresh prefill only (`kv.s == 0`, full
  causal), which is the board's shape and the LM forward's.
- FLASH: one-launch online-softmax attention per (batch, head, 32 query
  rows): QK^T and PV on the simdgroup matrix unit (8x8 f32), K/V tiles
  staged in threadgroup memory, running max and sum per row, causal key
  block skipping, no score stash in device memory (main's Apple forward
  allocates a fresh [B, nh, L, S] stash (100 MB at the board shape),
  waits, runs a scalar-chain kernel, waits, then a regime scan and a corner
  flag read back: three host round trips and one allocation per layer).
- GQA_TILE: FLASH with the n_rep query heads of one KV group stacked along
  the tile's row axis, so one staged K/V tile serves every head of its
  group (GROUP = n_rep in {2, 4}); at n_kv == n_heads it is FLASH's kernel
  at GROUP 1, same cost.
- FUSE_PRE: norm1 folded into the QKV projection (one GEMM launch for q, k
  and v with the row rstd and the norm weight applied while staging A;
  norm1_out never round-trips) with RoPE and the cache append in the GEMM's
  epilogue; and norm2 folded the same way into the gate/up projections.
- FUSE_MLP: the SwiGLU gate in the epilogue of ONE gate+up GEMM launch
  (two weight tiles, two accumulators per block, `silu(g) * u` written
  straight to `gated`), the residual adds in the epilogues of o_proj
  (`residual1 = x + ctx . Wo^T`) and down_proj (`residual2 = residual1 +
  gated . Wd^T`).
- ARENA (bindings/_mojolearn_transformer.mojo only): the lone block's
  workspace (stages, cache, rope, x) carved from one device arena
  (core/device_arena.mojo), no per-call thirty-buffer reset and no
  synchronize between the uploads and the forward.
- ALL: every candidate at once.
"""

from std.ffi import external_call, _Global
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.math import exp, sqrt
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.apple_air import simdgroup_load_legacy_air
from core.step_phase import step_count_launch

# ---------------------------------------------------------------------------
# The guard and the defines. Every candidate is FAST + Apple + its define
# (or the ALL define); nothing here is reachable on any other build.
# ---------------------------------------------------------------------------

comptime AFN_ATTN_ON = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)
comptime AFN_ATTN_ALL = AFN_ATTN_ON and is_defined["MOJOLEARN_AFN_ATTN_ALL"]()
comptime AFN_ATTN_NORM_SG = AFN_ATTN_ON and (
    AFN_ATTN_ALL or is_defined["MOJOLEARN_AFN_ATTN_NORM_SG"]()
)
comptime AFN_ATTN_ROPE_CACHE = AFN_ATTN_ON and (
    AFN_ATTN_ALL or is_defined["MOJOLEARN_AFN_ATTN_ROPE_CACHE"]()
)
comptime AFN_ATTN_GQA_TILE = AFN_ATTN_ON and (
    # NEVER RUN — PENDING MEASUREMENT. New candidate remains opt-in/default OFF.
# Scored FAST quality: 3/3 metrics within the existing bands; PASS.
# F07/gqa M3 2026-10-06: 3 retained public-caller timings;
# B/A range 0.7077..2.7525, mixed/regressing; retain OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/gqa.
# No combined-switch or full-board default claim from these component cases.
    AFN_ATTN_ALL or is_defined["MOJOLEARN_AFN_ATTN_GQA_TILE"]()
)
#: FLASH is also what GQA_TILE runs (at GROUP 1 when n_kv == n_heads).
comptime AFN_ATTN_FLASH = AFN_ATTN_ON and (
    # NEVER RUN — PENDING MEASUREMENT. New candidate remains opt-in/default OFF.
# Scored FAST quality: 3/3 metrics within the existing bands; PASS.
# F07/default M3 2026-10-06: 3 retained public-caller timings;
# B/A range 0.1752..0.8973, mixed/regressing; retain OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/default.
# No combined-switch or full-board default claim from these component cases.
    AFN_ATTN_ALL or AFN_ATTN_GQA_TILE or is_defined["MOJOLEARN_AFN_ATTN_FLASH"]()
)
comptime AFN_ATTN_FUSE_PRE = AFN_ATTN_ON and (
    AFN_ATTN_ALL or is_defined["MOJOLEARN_AFN_ATTN_FUSE_PRE"]()
)
comptime AFN_ATTN_FUSE_MLP = AFN_ATTN_ON and (
    AFN_ATTN_ALL or is_defined["MOJOLEARN_AFN_ATTN_FUSE_MLP"]()
)
comptime AFN_ATTN_ARENA = AFN_ATTN_ON and (
    AFN_ATTN_ALL or is_defined["MOJOLEARN_AFN_ATTN_ARENA"]()
)
#: Any candidate that changes the block forward's launches.
comptime AFN_ATTN_ANY = (
    AFN_ATTN_NORM_SG or AFN_ATTN_ROPE_CACHE or AFN_ATTN_FLASH
    or AFN_ATTN_FUSE_PRE or AFN_ATTN_FUSE_MLP
)

comptime AFN_TPB = 256
#: Apple threadgroup memory per block: every shared page asserts it fits.
comptime AFN_APPLE_TG_BYTES = 32768
comptime AFN_NEGMAX_BITS: UInt32 = 0xFF7FFFFF


# ---------------------------------------------------------------------------
# The simdgroup matrix unit: 8x8 f32 fragments. The lane -> cell map and the
# load/multiply intrinsics are gemm/checks/gemm_identical.mojo's (and the
# probe gemm/checks/apple_simdgroup_probe.mojo's); copied rather than
# imported so this lane never couples to the gemm lane's edits.
# ---------------------------------------------------------------------------

comptime _M64 = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]


@always_inline
def _sg_load_t(
    p: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    stride: Int,
) -> _M64:
    """Fragment M[r][c] = p[c * stride + r] (the transposed load)."""
    comptime if simdgroup_load_legacy_air():
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, Int64(stride), _V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, _V2(Int64(stride), 8), _V2(Int64(stride), 1), _V2(0, 0)
        )


@always_inline
def _sg_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


@always_inline
def _frag_row(lane: Int) -> Int:
    """The fragment row this lane's two cells sit in (probe's `frow`)."""
    var qd = lane // 4
    return (qd & 4) + ((lane // 2) % 4)


@always_inline
def _frag_col(lane: Int) -> Int:
    """The fragment column of this lane's first cell; its second is + 1."""
    var qd = lane // 4
    return (qd & 2) * 2 + (lane % 2) * 2


@always_inline
def _row_lo_hi(t: Int, pos0: Int, key_lo: Int, window: Int, s: Int) -> Tuple[Int, Int]:
    """The packed keys `[lo, hi]` query row `t` sees (fused_attention's
    `_row_range`, copied: same predicate as `attn_mask_kernel`)."""
    var p_q = pos0 + t
    var hi = p_q - key_lo
    if hi > s - 1:
        hi = s - 1
    var lo = 0
    if window > 0:
        lo = p_q - window + 1 - key_lo
        if lo < 0:
            lo = 0
    return (lo, hi)


# ---------------------------------------------------------------------------
# NORM_SG: one simdgroup per token row.
# ---------------------------------------------------------------------------


def afn_rms_norm_sg_kernel[RESIDUAL: Bool, WRITE_OUT: Bool](
    out_buf: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    residual: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dm_in: Int32,
    eps_in: Float32,
):
    """Row `t` = `a[t]` (`RESIDUAL`: `a[t] + b[t]`, written to `residual`);
    `sumsq[t]` = its sum of squares (lane partials of 4-wide vectors, then
    a 5-step xor butterfly); `WRITE_OUT`: `out[t] = row * rstd * weight`
    with `rstd = 1 / sqrt(sumsq / dm + eps)` (the exact reciprocal square
    root, not the hardware approximation). `dm % 4 == 0`."""
    var m = Int(m_in)
    var dm = Int(dm_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var t = Int(block_idx.x) * (AFN_TPB // 32) + sg
    if t >= m:
        return
    var base = t * dm
    var acc = SIMD[DType.float32, 4](0.0)
    var j = lane * 4
    while j < dm:
        var x4 = a.unsafe_load[width=4](base + j)
        comptime if RESIDUAL:
            x4 = x4 + b.unsafe_load[width=4](base + j)
            (residual + base + j).store(x4)
        acc = x4 * x4 + acc
        j += 128
    var s = acc.reduce_add()
    comptime for sh in range(5):
        s = s + shuffle_xor(s, UInt32(1 << sh))
    if lane == 0:
        sumsq.unsafe_store(t, s)
    comptime if WRITE_OUT:
        var rstd = Float32(1.0) / sqrt(s / Float32(dm) + eps_in)
        j = lane * 4
        while j < dm:
            var x4: SIMD[DType.float32, 4]
            comptime if RESIDUAL:
                x4 = residual.unsafe_load[width=4](base + j)
            else:
                x4 = a.unsafe_load[width=4](base + j)
            var w4 = weight.unsafe_load[width=4](j)
            (out_buf + base + j).store(x4 * rstd * w4)
            j += 128


def afn_rms_norm_sg(
    ctx: DeviceContext,
    sumsq: MutPointer[Float32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    dm: Int,
    eps: Float32,
    write_out: Bool,
) raises:
    """RMSNorm of `x` into `sumsq` (and `out_buf` when `write_out`)."""
    comptime if AFN_ATTN_ON:
        var blocks = (m + AFN_TPB // 32 - 1) // (AFN_TPB // 32)
        step_count_launch()
        if write_out:
            ctx.enqueue_function[afn_rms_norm_sg_kernel[False, True]](
                out_buf, sumsq, x, x, x, weight, Int32(m), Int32(dm), eps,
                grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[afn_rms_norm_sg_kernel[False, False]](
                out_buf, sumsq, x, x, x, weight, Int32(m), Int32(dm), eps,
                grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
    else:
        raise Error("afn_rms_norm_sg: not compiled on this build")


def afn_residual_rms_norm_sg(
    ctx: DeviceContext,
    residual: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    dm: Int,
    eps: Float32,
    write_out: Bool,
) raises:
    """`residual = a + b`, then its RMSNorm (`sumsq`, and `out_buf` when
    `write_out`)."""
    comptime if AFN_ATTN_ON:
        var blocks = (m + AFN_TPB // 32 - 1) // (AFN_TPB // 32)
        step_count_launch()
        if write_out:
            ctx.enqueue_function[afn_rms_norm_sg_kernel[True, True]](
                out_buf, sumsq, residual, a, b, weight, Int32(m), Int32(dm), eps,
                grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[afn_rms_norm_sg_kernel[True, False]](
                out_buf, sumsq, residual, a, b, weight, Int32(m), Int32(dm), eps,
                grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
    else:
        raise Error("afn_residual_rms_norm_sg: not compiled on this build")


# ---------------------------------------------------------------------------
# ROPE_CACHE: q RoPE + k RoPE + the K/V cache append (stage and carried
# cache) in one launch, fresh prefill (`s_old == 0`, full causal).
# ---------------------------------------------------------------------------


@always_inline
def _rope_cell(
    x: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    i: Int,
    d: Int,
    pos: Int,
    rd: Int,
) -> Float32:
    """`apply_rotary_pos_emb_kernel`'s cell: `x * cos + rotate_half(x) * sin`
    at absolute position `pos`; columns `d >= rd` pass through."""
    var xj = x.unsafe_load(i)
    if d >= rd:
        return xj
    var half = rd // 2
    var f = d
    if f >= half:
        f = f - half
    var cos_v = cos_tab.unsafe_load(pos * half + f)
    var sin_v = sin_tab.unsafe_load(pos * half + f)
    var rh: Float32
    if d < half:
        rh = -x.unsafe_load(i + half)
    else:
        rh = x.unsafe_load(i - half)
    return xj * cos_v + rh * sin_v


def afn_rope_cache_kernel(
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_rope: MutPointer[Float32, MutAnyOrigin],
    k_stage: MutPointer[Float32, MutAnyOrigin],
    v_stage: MutPointer[Float32, MutAnyOrigin],
    k_carry: MutPointer[Float32, MutAnyOrigin],
    v_carry: MutPointer[Float32, MutAnyOrigin],
    q_proj: MutPointer[Float32, MutAnyOrigin],
    k_proj: MutPointer[Float32, MutAnyOrigin],
    v_proj: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    hd_in: Int32,
    pos0_in: Int32,
    rd_in: Int32,
):
    """One thread per cell of q, then of k, then of v. A fresh prefill: the
    packed cache stride is `l` and slot `t` is token `t`; both the stage
    (`stages.k_cache`) and the carried cache (`kv.k`) receive the same
    words, which is what main's append plus its device copy leave."""
    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var pos0 = Int(pos0_in)
    var rd = Int(rd_in)
    var m = b * l
    var nq = m * nh * hd
    var nk = m * nkv * hd
    var i = Int(block_idx.x) * AFN_TPB + Int(thread_idx.x)
    if i < nq:
        var width = nh * hd
        var tok = i // width
        var rem = i - tok * width
        var d = rem % hd
        var t = tok - (tok // l) * l
        q_rope.unsafe_store(i, _rope_cell(q_proj, cos_tab, sin_tab, i, d, pos0 + t, rd))
        return
    i -= nq
    if i < nk:
        var width = nkv * hd
        var tok = i // width
        var rem = i - tok * width
        var kvh = rem // hd
        var d = rem - kvh * hd
        var bb = tok // l
        var t = tok - bb * l
        var v = _rope_cell(k_proj, cos_tab, sin_tab, i, d, pos0 + t, rd)
        k_rope.unsafe_store(i, v)
        var ci = ((bb * nkv + kvh) * l + t) * hd + d
        k_stage.unsafe_store(ci, v)
        k_carry.unsafe_store(ci, v)
        return
    i -= nk
    if i < nk:
        var width = nkv * hd
        var tok = i // width
        var rem = i - tok * width
        var kvh = rem // hd
        var d = rem - kvh * hd
        var bb = tok // l
        var t = tok - bb * l
        var v = v_proj.unsafe_load(i)
        var ci = ((bb * nkv + kvh) * l + t) * hd + d
        v_stage.unsafe_store(ci, v)
        v_carry.unsafe_store(ci, v)


def afn_rope_cache(
    ctx: DeviceContext,
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_rope: MutPointer[Float32, MutAnyOrigin],
    k_stage: MutPointer[Float32, MutAnyOrigin],
    v_stage: MutPointer[Float32, MutAnyOrigin],
    k_carry: MutPointer[Float32, MutAnyOrigin],
    v_carry: MutPointer[Float32, MutAnyOrigin],
    q_proj: MutPointer[Float32, MutAnyOrigin],
    k_proj: MutPointer[Float32, MutAnyOrigin],
    v_proj: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    b: Int,
    l: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    pos0: Int,
    rope_dim: Int,
) raises:
    comptime if AFN_ATTN_ON:
        var cells = b * l * (nh + 2 * nkv) * hd
        var blocks = (cells + AFN_TPB - 1) // AFN_TPB
        step_count_launch()
        ctx.enqueue_function[afn_rope_cache_kernel](
            q_rope, k_rope, k_stage, v_stage, k_carry, v_carry, q_proj, k_proj,
            v_proj, cos_tab, sin_tab, Int32(b), Int32(l), Int32(nh), Int32(nkv),
            Int32(hd), Int32(pos0), Int32(rope_dim),
            grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )
    else:
        raise Error("afn_rope_cache: not compiled on this build")


# ---------------------------------------------------------------------------
# FLASH (+ GQA_TILE): one-launch online-softmax attention on the matrix unit.
# ---------------------------------------------------------------------------

#: lane/no-bench-tuning-2 (2026-10-04): the kernel was head_dim 64 only (the
#: board's GPT-3-small 768 / 12). It is now templated on a padded head class
#: `HDP` (a multiple of 16: whole 8 x 8 context fragments over 8 simdgroups)
#: and reads the real `hd` at run time: Q, K and V columns `p >= hd` stage as
#: +0.0 (an exact zero term on the matrix unit), context columns `>= hd` are
#: never stored. The classes are the multiples of 16 whose shared page
#: (`afn_fl_page_bytes`, 272 * HDP + 9728 bytes at TQ = BK = 32) fits
#: Apple's 32 KB: 16, 32, 48, 64, 80; a head is served by the smallest class
#: >= hd (`afn_flash_head_class`), so every hd <= 80 with hd % 4 == 0 (the
#: vector-4 row loads) takes the same route, and at hd 64 (class 64, no
#: padding) the arithmetic is exactly the old kernel's. Wider heads do not fit
#: one page at BK = 32: they take main's fused launch (a capacity limit, not a
#: shape key). `-D MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF=1` restores hd == 64 only.
comptime AFN_FLASH_HD_GENERIC_OFF = is_defined["MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF"]()
comptime AFN_FL_HD = 64
"""The old single head class (kept for the OFF rule and the docs)."""
comptime AFN_FL_TQ = 32
comptime AFN_FL_BK = 32
comptime AFN_FL_NSG = AFN_TPB // 32
comptime AFN_FL_VST = AFN_FL_BK + 4  # V tile transposed: vT[c * VST + key]
comptime AFN_FL_QST = AFN_FL_TQ + 4  # Q transposed: qT[p * QST + r]
comptime AFN_FL_WST = AFN_FL_TQ + 4  # P transposed: wT[key * WST + r]
comptime AFN_FL_TST = AFN_FL_BK + 1  # scores: tile[r * TST + key]
comptime AFN_FL_NFR = AFN_FL_TQ // 8
comptime AFN_FL_NFK = AFN_FL_BK // 8
comptime AFN_FL_SPS = (AFN_FL_NFR * AFN_FL_NFK) // AFN_FL_NSG  # score fragments per simdgroup
comptime AFN_FL_WSZ = AFN_FL_BK * AFN_FL_WST
comptime AFN_FL_TSZ = AFN_FL_TQ * AFN_FL_TST


def afn_fl_kst(hdp: Int) -> Int:
    """K tile row stride: kT[key * KST + p]."""
    return hdp + 4


def afn_fl_ksz(hdp: Int) -> Int:
    return AFN_FL_BK * afn_fl_kst(hdp)


def afn_fl_vqsz(hdp: Int) -> Int:
    """The V page (V transposed), which also stages Q before the first key
    block: the larger of the two."""
    var vsz = hdp * AFN_FL_VST
    var qsz = hdp * AFN_FL_QST
    return vsz if vsz >= qsz else qsz


def afn_fl_cps(hdp: Int) -> Int:
    """Context fragments per simdgroup."""
    return (AFN_FL_NFR * (hdp // 8)) // AFN_FL_NSG


def afn_fl_page_bytes(hdp: Int) -> Int:
    """The flash kernel's shared page at head class `hdp`."""
    return 4 * (afn_fl_ksz(hdp) + afn_fl_vqsz(hdp) + AFN_FL_WSZ + AFN_FL_TSZ + 3 * AFN_FL_TQ)


comptime AFN_FL_PAGE_BYTES = afn_fl_page_bytes(AFN_FL_HD)


def afn_flash_head_class(hd: Int) -> Int:
    """The padded head class serving `hd`, or 0 when no class does (the
    caller then runs main's launches): the smallest multiple of 16 >= hd
    whose page fits `AFN_APPLE_TG_BYTES`; `hd % 4 == 0` for the vector-4
    row loads."""
    comptime if AFN_FLASH_HD_GENERIC_OFF:
        return AFN_FL_HD if hd == AFN_FL_HD else 0
    if hd <= 0 or hd % 4 != 0:
        return 0
    var c = ((hd + 15) // 16) * 16
    if c > 80 or afn_fl_page_bytes(c) > AFN_APPLE_TG_BYTES:
        return 0
    return c


def afn_flash_forward_kernel[GROUP: Int, HDP: Int](
    ctxv: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
    hd_in: Int32,
):
    """One block per (batch, head group, 32 query rows); 256 threads = 8
    simdgroups. Row `r` of the tile is head `h0 + r // TQG`, token
    `t0 + r % TQG` (`TQG = 32 / GROUP`), so at GROUP = n_rep one staged K/V
    tile serves every head of the KV group. Per key block of 32: K and V
    staged (K as [key][p], V transposed as [c][key]), scores = Q K^T on
    the matrix unit (Q fragments held in registers), scaled and masked,
    the per-row running max and sum updated by 8 lanes per row (xor
    butterflies), `p = exp(s - m_new)` written transposed, the context
    accumulators rescaled by `exp(m_old - m_new)` and advanced by P V on
    the matrix unit. Writes `ctxv`, `amax` (the row max) and `denom` (the
    row sum), what main's fused forward writes. Shared page
    `afn_fl_page_bytes(HDP)`. `HDP` is the padded head class, `hd_in` the
    real head size (columns `hd..HDP` stage as zero, never stored)."""
    comptime HD = HDP
    comptime AFN_FL_KST = afn_fl_kst(HD)
    comptime AFN_FL_KSZ = afn_fl_ksz(HD)
    comptime AFN_FL_VQSZ = afn_fl_vqsz(HD)
    comptime AFN_FL_NFC = HD // 8
    comptime AFN_FL_CPS = afn_fl_cps(HD)
    comptime TQ = AFN_FL_TQ
    comptime BK = AFN_FL_BK
    comptime TQG = TQ // GROUP
    comptime assert TQG * GROUP == TQ and TQG >= 8, "flash: GROUP in {1, 2, 4}"
    comptime assert AFN_FL_SPS * AFN_FL_NSG == AFN_FL_NFR * AFN_FL_NFK, "flash: whole score fragments"
    comptime assert HD % 16 == 0, "flash: head class a multiple of 16"
    comptime assert afn_fl_page_bytes(HD) <= AFN_APPLE_TG_BYTES, "flash: the threadgroup page fits Apple's 32 KB"
    comptime assert AFN_FL_CPS * AFN_FL_NSG == AFN_FL_NFR * AFN_FL_NFC, "flash: whole context fragments"
    comptime KST = AFN_FL_KST
    comptime VST = AFN_FL_VST
    comptime QST = AFN_FL_QST
    comptime WST = AFN_FL_WST
    comptime TST = AFN_FL_TST
    var kT = stack_allocation[AFN_FL_KSZ, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var vT = stack_allocation[AFN_FL_VQSZ, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var wT = stack_allocation[AFN_FL_WSZ, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var tile = stack_allocation[AFN_FL_TSZ, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var stm = stack_allocation[TQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var stl = stack_allocation[TQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sta = stack_allocation[TQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()

    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var frow = _frag_row(lane)
    var fcol = _frag_col(lane)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var scale = scale_in
    var hd = Int(hd_in)
    var n_rep = nh // nkv
    var ngroups = nh // GROUP
    var ntb = (l + TQG - 1) // TQG
    var raw = Int(block_idx.x)
    var tile_i = raw % ntb
    var rest = raw // ntb
    var g = rest % ngroups
    var bb = rest // ngroups
    var t0 = tile_i * TQG
    var h0 = g * GROUP
    var kvh = h0 // n_rep
    var kvbase = (bb * nkv + kvh) * s * hd
    var t_last = t0 + TQG - 1
    if t_last > l - 1:
        t_last = l - 1
    var r0 = _row_lo_hi(t0, pos0, key_lo, window, s)
    var r1 = _row_lo_hi(t_last, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK
    var negmax = bitcast[DType.float32](AFN_NEGMAX_BITS)

    # Q, transposed into the V page, then into this simdgroup's A fragments.
    for i in range(tid, (TQ * HD) // 4, AFN_TPB):
        var i4 = i * 4
        var r = i4 // HD
        var p = i4 % HD
        var hh = h0 + r // TQG
        var t = t0 + r % TQG
        var x4 = SIMD[DType.float32, 4](0.0)
        if t < l and p < hd:
            x4 = q_rope.unsafe_load[width=4]((bb * l + t) * nh * hd + hh * hd + p)
        vT[p * QST + r] = x4[0]
        vT[(p + 1) * QST + r] = x4[1]
        vT[(p + 2) * QST + r] = x4[2]
        vT[(p + 3) * QST + r] = x4[3]
    if tid < TQ:
        stm[tid] = negmax
        stl[tid] = Float32(0.0)
        sta[tid] = Float32(1.0)
    barrier()
    var fr = sg // (AFN_FL_NSG // AFN_FL_NFR)
    var qf = InlineArray[_M64, HD // 8](fill=_M64(0))
    comptime for p8 in range(HD // 8):
        qf[p8] = _sg_load_t(vT + (8 * p8) * QST + fr * 8, QST)
    barrier()

    # The softmax lanes: 8 per row, 4 keys each.
    var r_s = tid // 8
    var q8 = tid % 8
    var t_s = t0 + r_s % TQG
    var rng = _row_lo_hi(t_s, pos0, key_lo, window, s)
    var cacc = InlineArray[_M64, AFN_FL_CPS](fill=_M64(0))
    comptime SPAN = AFN_FL_NSG // AFN_FL_NFR  # simdgroups sharing one row fragment

    for kb in range(kb_lo, kb_hi + 1):
        # Stage K [key][p] and V [c][key].
        for i in range(tid, (BK * HD) // 4, AFN_TPB):
            var i4 = i * 4
            var r = i4 // HD
            var p = i4 % HD
            var j = kb * BK + r
            var k4 = SIMD[DType.float32, 4](0.0)
            var v4 = SIMD[DType.float32, 4](0.0)
            if j < s and p < hd:
                k4 = k_cache.unsafe_load[width=4](kvbase + j * hd + p)
                v4 = v_cache.unsafe_load[width=4](kvbase + j * hd + p)
            (kT + r * KST + p).store[alignment=16](k4)
            vT[p * VST + r] = v4[0]
            vT[(p + 1) * VST + r] = v4[1]
            vT[(p + 2) * VST + r] = v4[2]
            vT[(p + 3) * VST + r] = v4[3]
        barrier()
        # Scores on the matrix unit: this simdgroup's fragments.
        comptime for q in range(AFN_FL_SPS):
            var fk = (sg % SPAN) * AFN_FL_SPS + q
            var acc = _M64(0)
            comptime for p8 in range(HD // 8):
                var bf = _sg_load_t(kT + (fk * 8) * KST + 8 * p8, KST)
                acc = _sg_mma(qf[p8], bf, acc)
            comptime for e in range(2):
                tile[(fr * 8 + frow) * TST + fk * 8 + fcol + e] = acc[e]
        barrier()
        # Online softmax: the row's running max and sum.
        var sv = SIMD[DType.float32, 4](negmax)
        var vis = SIMD[DType.bool, 4](fill=False)
        comptime for c in range(4):
            var jj = q8 * 4 + c
            var j = kb * BK + jj
            if t_s < l and j >= rng[0] and j <= rng[1]:
                sv[c] = tile[r_s * TST + jj] * scale
                vis[c] = True
        var bm = sv.reduce_max()
        comptime for sh in range(3):
            bm = max(bm, shuffle_xor(bm, UInt32(1 << sh)))
        var m_old = stm[r_s]
        var l_old = stl[r_s]
        var m_new = max(m_old, bm)
        var p4 = SIMD[DType.float32, 4](0.0)
        comptime for c in range(4):
            if vis[c]:
                p4[c] = exp(sv[c] - m_new)
        var ps = p4.reduce_add()
        comptime for sh in range(3):
            ps = ps + shuffle_xor(ps, UInt32(1 << sh))
        var alpha = exp(m_old - m_new)
        comptime for c in range(4):
            wT[(q8 * 4 + c) * WST + r_s] = p4[c]
        barrier()
        if q8 == 0:
            stm[r_s] = m_new
            stl[r_s] = l_old * alpha + ps
            sta[r_s] = alpha
        barrier()
        # Rescale, then P V on the matrix unit.
        var al = sta[fr * 8 + frow]
        comptime for q in range(AFN_FL_CPS):
            cacc[q][0] = cacc[q][0] * al
            cacc[q][1] = cacc[q][1] * al
            var fc = (sg % SPAN) * AFN_FL_CPS + q
            comptime for k8 in range(AFN_FL_NFK):
                var af = _sg_load_t(wT + (k8 * 8) * WST + fr * 8, WST)
                var bf = _sg_load_t(vT + (fc * 8) * VST + k8 * 8, VST)
                cacc[q] = _sg_mma(af, bf, cacc[q])
        barrier()

    # The context, divided by the row sum; the row statistics.
    var rr = fr * 8 + frow
    var hh = h0 + rr // TQG
    var tt = t0 + rr % TQG
    if tt < l:
        var den = stl[rr]
        var inv = Float32(0.0)
        if den > Float32(0.0):
            inv = Float32(1.0) / den
        comptime for q in range(AFN_FL_CPS):
            var fc = (sg % SPAN) * AFN_FL_CPS + q
            comptime for e in range(2):
                var col = fc * 8 + fcol + e
                if col < hd:
                    ctxv.unsafe_store(
                        (bb * l + tt) * nh * hd + hh * hd + col,
                        cacc[q][e] * inv,
                    )
    if tid < TQ:
        var hs = h0 + tid // TQG
        var ts = t0 + tid % TQG
        if ts < l:
            amax.unsafe_store((bb * nh + hs) * l + ts, stm[tid])
            denom.unsafe_store((bb * nh + hs) * l + ts, stl[tid])


def afn_flash_group(n_rep: Int) -> Int:
    """The GROUP the flash launcher runs for `n_rep = n_heads / n_kv`: the
    group under GQA_TILE when it is 2 or 4, else 1 (one head per block,
    its KV head `h // n_rep`)."""
    comptime if AFN_ATTN_GQA_TILE:
        if n_rep == 2 or n_rep == 4:
            return n_rep
    return 1


def _afn_flash_launch[HDP: Int](
    ctx: DeviceContext,
    ctxv: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b: Int,
    l: Int,
    nh: Int,
    nkv: Int,
    s: Int,
    pos0: Int,
    key_lo: Int,
    window: Int,
    scale: Float32,
    hd: Int,
) raises:
    """The flash launch at head class `HDP` for each GQA group."""
    var group = afn_flash_group(nh // nkv)
    var tqg = AFN_FL_TQ // group
    var blocks = b * (nh // group) * ((l + tqg - 1) // tqg)
    step_count_launch()
    if group == 4:
        ctx.enqueue_function[afn_flash_forward_kernel[4, HDP]](
            ctxv, amax, denom, q_rope, k_cache, v_cache, Int32(b), Int32(l),
            Int32(nh), Int32(nkv), Int32(s), Int32(pos0), Int32(key_lo),
            Int32(window), scale, Int32(hd),
            grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )
    elif group == 2:
        ctx.enqueue_function[afn_flash_forward_kernel[2, HDP]](
            ctxv, amax, denom, q_rope, k_cache, v_cache, Int32(b), Int32(l),
            Int32(nh), Int32(nkv), Int32(s), Int32(pos0), Int32(key_lo),
            Int32(window), scale, Int32(hd),
            grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[afn_flash_forward_kernel[1, HDP]](
            ctxv, amax, denom, q_rope, k_cache, v_cache, Int32(b), Int32(l),
            Int32(nh), Int32(nkv), Int32(s), Int32(pos0), Int32(key_lo),
            Int32(window), scale, Int32(hd),
            grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )


struct FlashCallerAudit(Defaultable, Movable):
    var calls: Int
    var grouped_calls: Int
    def __init__(out self):
        self.calls = 0
        self.grouped_calls = 0

comptime FLASH_AUDIT = _Global[StorageType=FlashCallerAudit, name="AppleFlashCallerAudit", init_fn=FlashCallerAudit.__init__]
# NEVER RUN — PENDING MEASUREMENT. New candidate remains opt-in/default OFF.
# Scored FAST quality: 3/3 metrics within the existing bands; PASS.
# F07/default M3 2026-10-06: 3 retained public-caller timings;
# B/A range 0.1752..0.8973, mixed/regressing; retain OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/default.
# No combined-switch or full-board default claim from these component cases.
# Scored FAST quality: 3/3 metrics within the existing bands; PASS.
# F07/gqa M3 2026-10-06: 3 retained public-caller timings;
# B/A range 0.7077..2.7525, mixed/regressing; retain OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F07/gqa.
# No combined-switch or full-board default claim from these component cases.
comptime AFN_FLASH_AUDIT_ON = AFN_ATTN_ON and is_defined["MOJOLEARN_AFN_ATTN_AUDIT"]()

def afn_flash_call_count(grouped: Bool) raises -> Int:
    var audit = FLASH_AUDIT.get_or_create_ptr()
    return audit[].grouped_calls if grouped else audit[].calls


def afn_flash_forward(
    ctx: DeviceContext,
    ctxv: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b: Int,
    l: Int,
    nh: Int,
    nkv: Int,
    s: Int,
    pos0: Int,
    key_lo: Int,
    window: Int,
    scale: Float32,
    hd: Int,
) raises:
    """The one launch, at `afn_flash_head_class(hd)` (the caller checks it is
    nonzero). No wait: the kernel is enqueued on the in-order context like
    every other stage."""
    comptime if AFN_ATTN_FLASH:
        comptime if AFN_FLASH_AUDIT_ON:
            var audit = FLASH_AUDIT.get_or_create_ptr()
            audit[].calls += 1
            audit[].grouped_calls += Int(afn_flash_group(nh // nkv) > 1)
        var c = afn_flash_head_class(hd)
        if c == 16:
            _afn_flash_launch[16](ctx, ctxv, amax, denom, q_rope, k_cache, v_cache, b, l, nh, nkv, s, pos0, key_lo, window, scale, hd)
        elif c == 32:
            _afn_flash_launch[32](ctx, ctxv, amax, denom, q_rope, k_cache, v_cache, b, l, nh, nkv, s, pos0, key_lo, window, scale, hd)
        elif c == 48:
            _afn_flash_launch[48](ctx, ctxv, amax, denom, q_rope, k_cache, v_cache, b, l, nh, nkv, s, pos0, key_lo, window, scale, hd)
        elif c == 64:
            _afn_flash_launch[64](ctx, ctxv, amax, denom, q_rope, k_cache, v_cache, b, l, nh, nkv, s, pos0, key_lo, window, scale, hd)
        elif c == 80:
            _afn_flash_launch[80](ctx, ctxv, amax, denom, q_rope, k_cache, v_cache, b, l, nh, nkv, s, pos0, key_lo, window, scale, hd)
        else:
            raise Error("afn_flash_forward: head_dim " + String(hd) + " has no head class")
    else:
        raise Error("afn_flash_forward: not compiled on this build")


# ---------------------------------------------------------------------------
# FUSE_PRE / FUSE_MLP: the projection GEMM with an epilogue, on the matrix
# unit. `C[m x n] = A[m x k] . W[n x k]^T` (OP_NT, the block's layout),
# 64 x 64 output tile per block, K windows of 32, 8 simdgroups each owning
# 32 x 16 cells (4 x 2 fragments).
# ---------------------------------------------------------------------------

comptime AFN_GEMM_BM = 64
comptime AFN_GEMM_BN = 64
comptime AFN_GEMM_KB = 32
comptime AFN_GEMM_SGM = 2
comptime AFN_GEMM_SGN = 4
comptime AFN_GEMM_FM = AFN_GEMM_BM // (8 * AFN_GEMM_SGM)
comptime AFN_GEMM_FN = AFN_GEMM_BN // (8 * AFN_GEMM_SGN)
comptime AFN_GEMM_AST = AFN_GEMM_BM + 4  # A staged transposed: at[p * AST + i]
comptime AFN_GEMM_BST = AFN_GEMM_KB + 4  # W staged as rows: bt[j * BST + p]
comptime AFN_GEMM_ASZ = AFN_GEMM_KB * AFN_GEMM_AST
comptime AFN_GEMM_BSZ = AFN_GEMM_BN * AFN_GEMM_BST
comptime AFN_GEMM_CST = AFN_GEMM_BN + 1  # the epilogue's C tile: ct[i * CST + j]
comptime AFN_GEMM_CSZ = AFN_GEMM_BM * AFN_GEMM_CST
comptime AFN_GEMM_PAGE = AFN_GEMM_ASZ + 2 * AFN_GEMM_BSZ
comptime AFN_GEMM_PAGE_BYTES = 4 * (AFN_GEMM_PAGE + AFN_GEMM_BM)

#: epilogues
comptime AFN_EPI_PLAIN = 0  # c = acc
comptime AFN_EPI_RESIDUAL = 1  # c = c2 + acc
comptime AFN_EPI_SWIGLU = 2  # c = silu(acc_gate) * acc_up (two weights)
comptime AFN_EPI_ROPE_QKV = 3  # q/k/v heads: RoPE and the cache append


def afn_gemm_nt_kernel[EPI: Int, ANORM: Bool](
    c: MutPointer[Float32, MutAnyOrigin],
    c2: MutPointer[Float32, MutAnyOrigin],
    c3: MutPointer[Float32, MutAnyOrigin],
    c4: MutPointer[Float32, MutAnyOrigin],
    c5: MutPointer[Float32, MutAnyOrigin],
    c6: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    w0: MutPointer[Float32, MutAnyOrigin],
    w1: MutPointer[Float32, MutAnyOrigin],
    w2: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    nw: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    eps_in: Float32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    pos0_in: Int32,
    hd_in: Int32,
):
    """`ANORM`: A's row `i` is `a[i] * rstd[i] * nw` with `rstd[i] = 1 /
    sqrt(sumsq[i] / k + eps)` (RMSNorm folded into the staging; `k` is the
    model width; `sumsq` is read only, into the block's `rs` page).
    `EPI_PLAIN`: `c[i, j] = acc`. `EPI_RESIDUAL`: `c[i, j] = c2[i, j] +
    acc`. `EPI_SWIGLU`: `w0` gate, `w1` up, `c = silu(g) * u`.
    `EPI_ROPE_QKV`: one BN = 64 column tile per head, `n = (nh + 2 nkv) *
    64`, head tile `nt = n0 / 64`, real head size `hd_in <= 64` (columns
    `d >= hd` of the tile load zero weight rows and are never stored;
    lane/no-bench-tuning-2, was hd == 64 only; at hd 64 the arithmetic is
    the old kernel's):
    `nt < nh` is q head `nt` from `w0`, then the k heads from `w1`, then
    the v heads from `w2`. q and k take RoPE at absolute position
    `pos0 + t` (the whole head is this block's tile, so the partner column
    is in the staged C tile): q -> `c` (`q_rope`, token-major); k -> `c2`
    (`k_rope`), `c3` (`stages.k_cache`) and `c4` (`kv.k`); v -> `c5`
    (`stages.v_cache`) and `c6` (`kv.v`). Fresh prefill (`s_old == 0`): the
    packed cache stride is `l`. `k % 32 == 0`; rows past `m` and columns
    past `n` are never stored."""
    comptime BM = AFN_GEMM_BM
    comptime BN = AFN_GEMM_BN
    comptime KB = AFN_GEMM_KB
    comptime FM = AFN_GEMM_FM
    comptime FN = AFN_GEMM_FN
    comptime SGN = AFN_GEMM_SGN
    comptime AST = AFN_GEMM_AST
    comptime BST = AFN_GEMM_BST
    comptime ASZ = AFN_GEMM_ASZ
    comptime BSZ = AFN_GEMM_BSZ
    comptime CST = AFN_GEMM_CST
    comptime NF = FM * FN
    comptime DUAL = EPI == AFN_EPI_SWIGLU
    comptime assert AFN_GEMM_CSZ <= AFN_GEMM_ASZ + AFN_GEMM_BSZ, "afn gemm: the C tile fits the A and first W pages"
    comptime assert AFN_GEMM_PAGE_BYTES <= AFN_APPLE_TG_BYTES, "afn gemm: the threadgroup page fits Apple's 32 KB"
    var pg = stack_allocation[AFN_GEMM_PAGE, Scalar[DType.float32], alignment = 16, address_space = AddressSpace.SHARED]()
    var rs = stack_allocation[BM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var at = pg
    var bt = pg + ASZ
    var bt2 = pg + ASZ + BSZ
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var frow = _frag_row(lane)
    var fcol = _frag_col(lane)
    var nbn = (n + BN - 1) // BN
    var bid = Int(block_idx.x)
    var m0 = (bid // nbn) * BM
    var n0 = (bid % nbn) * BN

    # The weight rows this block multiplies.
    var wsel = w0
    var wrow0 = n0
    var hdr = Int(hd_in)
    comptime if EPI == AFN_EPI_ROPE_QKV:
        var nh = Int(nh_in)
        var nkv = Int(nkv_in)
        var nt = n0 // BN
        if nt >= nh + nkv:
            wsel = w2
            wrow0 = (nt - nh - nkv) * hdr
        elif nt >= nh:
            wsel = w1
            wrow0 = (nt - nh) * hdr
        else:
            wrow0 = nt * hdr
    comptime if ANORM:
        if tid < BM:
            var i = m0 + tid
            var r = Float32(0.0)
            if i < m:
                r = Float32(1.0) / sqrt(sumsq.unsafe_load(i) / Float32(k) + eps_in)
            rs[tid] = r
        barrier()

    var acc = InlineArray[_M64, NF](fill=_M64(0))
    var acc2 = InlineArray[_M64, NF](fill=_M64(0))
    var windows = k // KB
    for w in range(windows):
        var k0 = w * KB
        # A: (BM * KB) / 4 slots, two per thread: (row i, 4 consecutive p).
        comptime for sl in range((BM * KB) // (4 * AFN_TPB)):
            var sidx = sl * AFN_TPB + tid
            var i = sidx // (KB // 4)
            var p4 = (sidx % (KB // 4)) * 4
            var gi = m0 + i
            var x4 = SIMD[DType.float32, 4](0.0)
            if gi < m:
                x4 = a.unsafe_load[width=4](gi * k + k0 + p4)
                comptime if ANORM:
                    x4 = x4 * rs[i] * nw.unsafe_load[width=4](k0 + p4)
            at[p4 * AST + i] = x4[0]
            at[(p4 + 1) * AST + i] = x4[1]
            at[(p4 + 2) * AST + i] = x4[2]
            at[(p4 + 3) * AST + i] = x4[3]
        # W: (BN * KB) / 4 slots, two per thread: (row j, 4 consecutive p).
        comptime for sl in range((BN * KB) // (4 * AFN_TPB)):
            var sidx = sl * AFN_TPB + tid
            var j = sidx // (KB // 4)
            var p4 = (sidx % (KB // 4)) * 4
            var gj = wrow0 + j
            var y4 = SIMD[DType.float32, 4](0.0)
            comptime if EPI == AFN_EPI_ROPE_QKV:
                if j < hdr:
                    y4 = wsel.unsafe_load[width=4](gj * k + k0 + p4)
            else:
                if n0 + j < n:
                    y4 = w0.unsafe_load[width=4](gj * k + k0 + p4)
            (bt + j * BST + p4).store[alignment=16](y4)
            comptime if DUAL:
                var z4 = SIMD[DType.float32, 4](0.0)
                if n0 + j < n:
                    z4 = w1.unsafe_load[width=4](gj * k + k0 + p4)
                (bt2 + j * BST + p4).store[alignment=16](z4)
        barrier()
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_M64, FM](fill=_M64(0))
            comptime for fm in range(FM):
                af[fm] = _sg_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                var bf = _sg_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                comptime for fm in range(FM):
                    acc[fm * FN + fq] = _sg_mma(af[fm], bf, acc[fm * FN + fq])
                comptime if DUAL:
                    var bf2 = _sg_load_t(bt2 + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                    comptime for fm in range(FM):
                        acc2[fm * FN + fq] = _sg_mma(af[fm], bf2, acc2[fm * FN + fq])
        barrier()

    # The epilogue.
    comptime if EPI == AFN_EPI_ROPE_QKV:
        # Through the C tile (the RoPE partner column is in this head).
        var ct = pg
        comptime for fm in range(FM):
            comptime for fq in range(FN):
                comptime for e in range(2):
                    var li = (sgm * FM + fm) * 8 + frow
                    var lj = (sgn * FN + fq) * 8 + fcol + e
                    ct[li * CST + lj] = acc[fm * FN + fq][e]
        barrier()
        var l = Int(l_in)
        var nh = Int(nh_in)
        var nkv = Int(nkv_in)
        var pos0 = Int(pos0_in)
        var nt = n0 // BN
        var hd = hdr
        var half = hd // 2
        for cell in range(tid, BM * BN, AFN_TPB):
            var li = cell // BN
            var d = cell % BN
            var gi = m0 + li
            if gi >= m or d >= hd:
                continue
            var bb = gi // l
            var t = gi - bb * l
            var x = ct[li * CST + d]
            if nt >= nh + nkv:
                var kvh = nt - nh - nkv
                var ci = ((bb * nkv + kvh) * l + t) * hd + d
                c5.unsafe_store(ci, x)
                c6.unsafe_store(ci, x)
            else:
                var pos = pos0 + t
                var f = d
                if f >= half:
                    f = f - half
                var cos_v = cos_tab.unsafe_load(pos * half + f)
                var sin_v = sin_tab.unsafe_load(pos * half + f)
                var rh: Float32
                if d < half:
                    rh = -ct[li * CST + d + half]
                else:
                    rh = ct[li * CST + d - half]
                var y = x * cos_v + rh * sin_v
                if nt < nh:
                    c.unsafe_store(gi * (nh * hd) + nt * hd + d, y)
                else:
                    var kvh = nt - nh
                    c2.unsafe_store(gi * (nkv * hd) + kvh * hd + d, y)
                    var ci = ((bb * nkv + kvh) * l + t) * hd + d
                    c3.unsafe_store(ci, y)
                    c4.unsafe_store(ci, y)
    else:
        comptime for fm in range(FM):
            comptime for fq in range(FN):
                comptime for e in range(2):
                    var gi = m0 + (sgm * FM + fm) * 8 + frow
                    var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                    if gi < m and gj < n:
                        var v = acc[fm * FN + fq][e]
                        comptime if EPI == AFN_EPI_RESIDUAL:
                            c.unsafe_store(gi * n + gj, c2.unsafe_load(gi * n + gj) + v)
                        elif EPI == AFN_EPI_SWIGLU:
                            var u = acc2[fm * FN + fq][e]
                            var sil = v / (Float32(1.0) + exp(-v))
                            c.unsafe_store(gi * n + gj, sil * u)
                        else:
                            c.unsafe_store(gi * n + gj, v)


def afn_pre_head_ok(hd: Int) -> Bool:
    """lane/no-bench-tuning-2: the fused QKV + RoPE launch serves every even
    head size that fits one BN-column tile (the RoPE partner column must be
    in the block's staged C tile), not hd == 64 only. BN = 64 is the widest
    tile whose page (A + two W pages) fits Apple's 32 KB, so wider heads take
    main's launches (a capacity limit). `-D MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF=1`
    restores hd == 64 only."""
    comptime if AFN_FLASH_HD_GENERIC_OFF:
        return hd == AFN_FL_HD
    return hd > 0 and hd % 2 == 0 and hd <= AFN_GEMM_BN


def afn_gemm_ok(m: Int, n: Int, k: Int) -> Bool:
    """Whether the epilogue GEMM runs this shape: whole K windows."""
    return m > 0 and n > 0 and k > 0 and k % AFN_GEMM_KB == 0


def _afn_gemm_launch[EPI: Int, ANORM: Bool](
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    c2: MutPointer[Float32, MutAnyOrigin],
    c3: MutPointer[Float32, MutAnyOrigin],
    c4: MutPointer[Float32, MutAnyOrigin],
    c5: MutPointer[Float32, MutAnyOrigin],
    c6: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    w0: MutPointer[Float32, MutAnyOrigin],
    w1: MutPointer[Float32, MutAnyOrigin],
    w2: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    nw: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    eps: Float32,
    l: Int,
    nh: Int,
    nkv: Int,
    pos0: Int,
    hd: Int = AFN_GEMM_BN,
) raises:
    comptime if AFN_ATTN_ON:
        var blocks = ((m + AFN_GEMM_BM - 1) // AFN_GEMM_BM) * ((n + AFN_GEMM_BN - 1) // AFN_GEMM_BN)
        step_count_launch()
        ctx.enqueue_function[afn_gemm_nt_kernel[EPI, ANORM]](
            c, c2, c3, c4, c5, c6, a, w0, w1, w2, sumsq, nw, cos_tab, sin_tab,
            Int32(m), Int32(n), Int32(k), eps, Int32(l), Int32(nh), Int32(nkv),
            Int32(pos0), Int32(hd),
            grid_dim=(blocks, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )
    else:
        raise Error("afn gemm: not compiled on this build")


def afn_proj_plain(
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    nw: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    eps: Float32,
    anorm: Bool,
) raises:
    """`c = A . W^T`; `anorm`: A is `a` RMS-normalized by `sumsq` and `nw`."""
    if anorm:
        _afn_gemm_launch[AFN_EPI_PLAIN, True](
            ctx, c, c, c, c, c, c, a, w, w, w, sumsq, nw, nw, nw, m, n, k, eps, 1, 1, 1, 0,
        )
    else:
        _afn_gemm_launch[AFN_EPI_PLAIN, False](
            ctx, c, c, c, c, c, c, a, w, w, w, sumsq, nw, nw, nw, m, n, k, eps, 1, 1, 1, 0,
        )


def afn_proj_residual(
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    residual_in: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`c = residual_in + A . W^T`."""
    _afn_gemm_launch[AFN_EPI_RESIDUAL, False](
        ctx, c, residual_in, c, c, c, c, a, w, w, w, a, a, a, a, m, n, k, Float32(0.0),
        1, 1, 1, 0,
    )


def afn_proj_swiglu(
    ctx: DeviceContext,
    gated: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    w_gate: MutPointer[Float32, MutAnyOrigin],
    w_up: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    nw: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    eps: Float32,
    anorm: Bool,
) raises:
    """`gated = silu(A . Wg^T) * (A . Wu^T)` in one launch."""
    if anorm:
        _afn_gemm_launch[AFN_EPI_SWIGLU, True](
            ctx, gated, gated, gated, gated, gated, gated, a, w_gate, w_up, w_up, sumsq,
            nw, nw, nw, m, n, k, eps, 1, 1, 1, 0,
        )
    else:
        _afn_gemm_launch[AFN_EPI_SWIGLU, False](
            ctx, gated, gated, gated, gated, gated, gated, a, w_gate, w_up, w_up, sumsq,
            nw, nw, nw, m, n, k, eps, 1, 1, 1, 0,
        )


def afn_proj_qkv_rope_cache(
    ctx: DeviceContext,
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_rope: MutPointer[Float32, MutAnyOrigin],
    k_stage: MutPointer[Float32, MutAnyOrigin],
    k_carry: MutPointer[Float32, MutAnyOrigin],
    v_stage: MutPointer[Float32, MutAnyOrigin],
    v_carry: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    w_q: MutPointer[Float32, MutAnyOrigin],
    w_k: MutPointer[Float32, MutAnyOrigin],
    w_v: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    nw: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    k: Int,
    eps: Float32,
    l: Int,
    nh: Int,
    nkv: Int,
    pos0: Int,
    anorm: Bool,
    hd: Int,
) raises:
    """The q, k and v projections of a fresh prefill in one launch, RoPE on
    q and k, both caches written. `afn_pre_head_ok(hd)` and `rope_dim == hd`
    (the caller checks): one 64-column tile per head. `anorm`: A
    is `a` (the block input) normalized by `sumsq` and `nw` (norm1, whose
    output never round-trips); otherwise A is `norm1_out`."""
    var n = (nh + 2 * nkv) * AFN_GEMM_BN
    if anorm:
        _afn_gemm_launch[AFN_EPI_ROPE_QKV, True](
            ctx, q_rope, k_rope, k_stage, k_carry, v_stage, v_carry, a, w_q, w_k, w_v,
            sumsq, nw, cos_tab, sin_tab, m, n, k, eps, l, nh, nkv, pos0, hd,
        )
    else:
        _afn_gemm_launch[AFN_EPI_ROPE_QKV, False](
            ctx, q_rope, k_rope, k_stage, k_carry, v_stage, v_carry, a, w_q, w_k, w_v,
            sumsq, nw, cos_tab, sin_tab, m, n, k, eps, l, nh, nkv, pos0, hd,
        )
