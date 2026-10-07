# SPDX-License-Identifier: Apache-2.0
"""NI20 model launch adapters; arithmetic lives in the shared native module.

THE TILED KERNELS (lane/attention-tiled-v2, 2026-10-07). The profile
`attention-online-tile32.fp32.v2` (contract: attention_v2_model_contract.mojo,
`v2m_forward_row`, `v2m_prepare_row`, `v2m_dq_row`, `v2m_dkdv_key`) is a
per-row fold whose order is fixed by the contract and reproduced here step
for step, so the device words equal the CPU column's on NVIDIA and AMD. The
only thing these kernels change against the one-thread-per-row reference
kernels (kept below for the trace-only diagnostic cells, the Apple column and
head dims the register layout does not cover) is WHO computes each step and
WHERE the operands sit; never the operands, the operation or the order.

Forward, per query row (contract `v2m_forward_row`), mapped to `_forward_tiled`:

  contract step                              tiled kernel
  -----------------------------------------  ------------------------------------------
  logical tile = 32 keys from `lo`, ascending  the block streams 32-key DEVICE tiles
    (`first = lo + 32*m`, `last = min(first+32, hi)`)  aligned to key 0 and keeps TWO of them
                                             resident (a 64-key ring), so the one
                                             logical tile of each row that starts
                                             inside device tile `kt` is resident in
                                             full even when `lo` differs per row
                                             (sliding window); `f` and `last` are the
                                             contract's `first`/`last` for that row.
  score = mul(sum_d fma(q[d], k[key][d]), scale)  lane `kg` of the row scores keys
    (ascending d, from +0.0)                 `f + kg*KPT + i`; q in registers, k from
                                             shared; the ascending-d fma chain is the
                                             contract's.
  tile_max = fmax fold over the tile's scores  per-lane fmax partial, then the TPR
                                             partials folded with the same `identical_fmax`.
                                             `portable_fmaxf` is a total-order
                                             selection over flushed, NaN-canonical
                                             operands, so the fold shape cannot change
                                             the word (contract 5.1 says the same).
  new_max = fmax(maximum, tile_max)          every lane of the row, redundantly.
  correction = 0 if maximum == -inf else exp(maximum - new_max)   same.
  denominator = mul(denominator, correction)  same, every lane redundantly.
  output[d] = mul(output[d], correction)     lane `dg` owns d = dg + j*TPR.
  for key ascending in the tile:             weights exp(score - new_max) go to shared
    weight = exp(score - new_max)            (one word per key, written by the lane
    denominator = add(weight, denominator)   that scored it), then every lane walks
    output[d] = fma(weight, v[key][d], output[d])  the tile ASCENDING, folding the
                                             denominator and its own output lanes.
  output[d] = div(output[d], denominator)    lane `dg`, its d lanes; lane 0 stores
  store maximum, denominator                 the row's two words.

Backward (contract `v2m_prepare_row` -> `v2m_dq_row` -> `v2m_dkdv_key`):
  prepare: `v2m_normalizer` is the forward fold without V (`_prepare_tiled`
           phase A, the same mapping as above), then zdot is the ascending
           chain fma(p, dyv, zdot) over the visible keys with the FINAL max
           (phase B: p and dyv per key to shared, the chain walked ascending
           by every lane of the row). maxima/denominators are recomputed
           like the contract does, never read from the forward.
  dq:      ds = mul(mul(p, dyv - zdot), scale) per key to shared (`_dq_tiled`),
           then dq[d] = fma(ds, k[key][d], dq[d]) ascending over the visible keys
           of each device tile, lane `dg` owning d = dg + j*TPR.
  dkdv:    a block owns 32 keys of one (batch, kv head) and walks the query
           rows in the contract's canonical GQA order, head ascending then
           query ascending (`_dkdv_tiled`): per (key, query) p and ds to shared,
           then dk[d] = fma(ds, q[d], dk[d]) and dv[d] = fma(p, dy[d], dv[d])
           ascending in (head, query). Masked cells are SKIPPED like the
           contract (never folded as +0.0, which would launder a -0.0).

Cost reasoning (no board dimension anywhere): one QK^T sweep and one PV sweep
per forward row instead of the reference's two score passes per tile with the
output row in global memory; K/V read once per block from HBM and served from
shared memory to TQ rows; the device tile equals the profile's logical tile so
no row ever needs more than two resident tiles. Shared memory at HD=64:
forward 38.5 KB (TQ=32) / 42.7 KB (TQ=64), backward 34-42 KB: inside the
NVIDIA 48 KB static budget and AMD's 64 KB; over Apple's 32 KB, so the Apple
column keeps the reference kernels (Apple is never timed for IDENTICAL).
HD > 64 keeps the reference as well: a 64-key ring of K and V at HD=128 is
66 KB. `MOJOLEARN_IDN_ATTN_V2_TQ` (32 | 64) is the forward's query rows per
block; it moves no bit (the fold is per row).
"""
from std.gpu import block_dim, block_idx, thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.memory import bitcast, stack_allocation
from std.sys.compile import get_defined_int
from std.utils import StaticTuple
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import identical_div, identical_fmax, identical_mul_add, portable_expf
from transformer.impl.llama.attention_v2_model_contract import (
    ATTENTION_MODEL_V2_TILE, V2Ptr, v2m_add, v2m_mul, v2m_query_base,
    v2m_forward_row, v2m_forward_diagnostic, v2m_prepare_row,
    v2m_dq_row, v2m_dkdv_key, v2m_backward_diagnostic,
)

comptime V2_TILE = ATTENTION_MODEL_V2_TILE
comptime V2_RING = 2 * V2_TILE
comptime V2_THREADS = 256
# Forward query rows per block (legal set 32 | 64; core/six_lane_experiment_guards.mojo).
comptime ATTN_V2_TQ = get_defined_int["MOJOLEARN_IDN_ATTN_V2_TQ", 32]()
# Backward row-side blocks stay at 32 rows: their DY tile and two per-tile
# word arrays would push 64 rows past the NVIDIA static shared budget.
comptime V2_TQB = 32
comptime V2_TILED_DEVICE = TARGET_COLUMN != COLUMN_APPLE
comptime SharedF32 = UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]


@always_inline
def _v2_tiled_head_dim(depth: Int) -> Bool:
    """Head dims the register-resident Q row and the shared budget cover."""
    return depth == 16 or depth == 32 or depth == 64


@always_inline
def _v2_lo_hi(t: Int, keys: Int, own0: Int, window: Int) -> Tuple[Int, Int]:
    """`v2m_visible` spelled on the query position `t` directly (same formula)."""
    var absolute_in_span = own0 + t
    var lo = max(0, absolute_in_span - window + 1) if window > 0 else 0
    return lo, min(keys, absolute_in_span + 1)


@always_inline
def _v2_stage_rows[HD: Int, ROWS: Int](
    dst: SharedF32, src: V2Ptr, base: Int, first: Int, limit: Int, tid: Int
):
    """Rows `first .. first+ROWS` of a contiguous `[*, HD]` operand (a K or V
    head slab) into `dst` with stride HD+1; rows at or past `limit` stage
    zeros that no reader ever uses (every read is bounded by `hi <= keys`)."""
    comptime for i in range((ROWS * HD) // V2_THREADS):
        var e = tid + i * V2_THREADS
        var r = e // HD
        var d = e % HD
        var j = first + r
        var x = Float32(0.0)
        if j < limit:
            x = src.unsafe_load(base + j * HD + d)
        dst.unsafe_store(r * (HD + 1) + d, x)


@always_inline
def _v2_stage_tokens[HD: Int, ROWS: Int](
    dst: SharedF32, src: V2Ptr, batch: Int, h: Int, t_first: Int,
    length: Int, heads: Int, tid: Int
):
    """Rows `t_first .. t_first+ROWS` of head `h` of a token-major `[B,L,H,HD]`
    operand (Q, dY) into `dst` with stride HD+1; rows past `length` are zeros."""
    comptime for i in range((ROWS * HD) // V2_THREADS):
        var e = tid + i * V2_THREADS
        var r = e // HD
        var d = e % HD
        var t = t_first + r
        var x = Float32(0.0)
        if t < length:
            x = src.unsafe_load(((batch * length + t) * heads + h) * HD + d)
        dst.unsafe_store(r * (HD + 1) + d, x)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(V2_THREADS)))
def _forward_tiled[HD: Int, TQ: Int](
    q: V2Ptr, k: V2Ptr, v: V2Ptr, output: V2Ptr, maxima: V2Ptr,
    denominators: V2Ptr, length: Int32, heads: Int32, kv_heads: Int32,
    keys: Int32, own0: Int32, window: Int32, scale: Float32,
):
    """Block = TQ query rows of one (batch, head); grid (ceil(L/TQ), heads, B).
    Each row's fold is `v2m_forward_row` in the contract's order (module doc)."""
    comptime TPR = V2_THREADS // TQ  # lanes per query row
    comptime KPT = V2_TILE // TPR    # keys each lane scores per tile
    comptime DPT = HD // TPR         # output lanes each lane owns
    comptime KS = HD + 1             # padded shared stride (bank spread)
    comptime WS = V2_TILE + 1
    comptime assert TQ * TPR == V2_THREADS and KPT * TPR == V2_TILE and DPT * TPR == HD, "attention v2 tiled forward: TQ in {32, 64}, HD a multiple of 256/TQ"
    var ks = stack_allocation[V2_RING * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var vs = stack_allocation[V2_RING * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ws = stack_allocation[TQ * WS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ms = stack_allocation[V2_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var r = tid // TPR
    var lane = tid % TPR
    var l = Int(length)
    var nh = Int(heads)
    var nkv = Int(kv_heads)
    var s = Int(keys)
    var o0 = Int(own0)
    var win = Int(window)
    var t0 = Int(block_idx.x) * TQ
    var h = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var kb = (batch * nkv + h // (nh // nkv)) * s * HD
    var t = t0 + r
    var valid = t < l
    var tt = t if valid else l - 1
    var row = (batch * nh + h) * l + tt
    var qb = v2m_query_base(row, l, nh, HD)
    var vis = _v2_lo_hi(tt, s, o0, win)
    var lo = vis[0]
    var hi = vis[1]
    # The block's device tile range: lo and hi are non-decreasing in t, so the
    # first row's lo and the last valid row's hi bound every row of the block.
    var vis_first = _v2_lo_hi(t0, s, o0, win)
    var vis_last = _v2_lo_hi(min(t0 + TQ - 1, l - 1), s, o0, win)
    var tile_lo = vis_first[0] // V2_TILE
    var tile_hi = -1
    if vis_last[1] > 0:
        tile_hi = (vis_last[1] - 1) // V2_TILE
    var qr = SIMD[DType.float32, HD](0.0)
    if valid:
        comptime for d in range(HD):
            qr[d] = q.unsafe_load(qb + d)
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var maximum = neg_inf
    var denominator = Float32(0.0)
    var oacc = SIMD[DType.float32, DPT](0.0)
    var sc = SIMD[DType.float32, KPT](0.0)
    for kt in range(tile_lo, tile_hi + 1):
        # Resident window [kt, kt+1]: tile kt was staged by the previous
        # iteration (or here, on the first one); tile kt+1 replaces kt-1,
        # whose last reader finished before the barrier that ended that
        # iteration. Ring row of key j is j % 64.
        comptime for which in range(2):
            var lt = kt + which
            if (which == 1 or kt == tile_lo) and lt <= tile_hi:
                var slot = (lt % 2) * V2_TILE * KS
                _v2_stage_rows[HD, V2_TILE](ks + slot, k, kb, lt * V2_TILE, s, tid)
                _v2_stage_rows[HD, V2_TILE](vs + slot, v, kb, lt * V2_TILE, s, tid)
        barrier()
        # This row's logical tile that starts inside device tile kt: the
        # contract's `first` (= lo + 32*m) and `last`.
        var base = kt * V2_TILE
        var f = lo
        if lo < base:
            f = lo + ((base - lo + V2_TILE - 1) // V2_TILE) * V2_TILE
        var has = valid and lo < hi and f < base + V2_TILE and f < hi
        var last = min(f + V2_TILE, hi)
        var n = 0
        if has:
            n = last - f
        var pmax = neg_inf
        comptime for i in range(KPT):
            var key = f + lane * KPT + i
            if has and key < last:
                var acc = Float32(0.0)
                var kr = (key % V2_RING) * KS
                comptime for d in range(HD):
                    acc = identical_mul_add(qr[d], ks.unsafe_load(kr + d), acc)
                var sv = v2m_mul(acc, scale)
                sc[i] = sv
                pmax = identical_fmax(pmax, sv)
        ms.unsafe_store(tid, pmax)
        barrier()
        var tile_max = neg_inf
        comptime for g in range(TPR):
            tile_max = identical_fmax(tile_max, ms.unsafe_load(r * TPR + g))
        if has:
            var new_max = identical_fmax(maximum, tile_max)
            var correction = Float32(0.0) if maximum == neg_inf else portable_expf(maximum - new_max)
            denominator = v2m_mul(denominator, correction)
            comptime for j in range(DPT):
                oacc[j] = v2m_mul(oacc[j], correction)
            comptime for i in range(KPT):
                var key = f + lane * KPT + i
                if key < last:
                    ws.unsafe_store(r * WS + lane * KPT + i, portable_expf(sc[i] - new_max))
            maximum = new_max
        barrier()
        if has:
            for i in range(n):
                var w = ws.unsafe_load(r * WS + i)
                denominator = v2m_add(w, denominator)
                var vr = ((f + i) % V2_RING) * KS
                comptime for j in range(DPT):
                    oacc[j] = identical_mul_add(w, vs.unsafe_load(vr + lane + j * TPR), oacc[j])
        barrier()
    if valid:
        comptime for j in range(DPT):
            output.unsafe_store(qb + lane + j * TPR, identical_div(oacc[j], denominator))
        if lane == 0:
            maxima.unsafe_store(row, maximum)
            denominators.unsafe_store(row, denominator)


def _forward_diag(q: V2Ptr, k: V2Ptr, maxima: V2Ptr, denominators: V2Ptr,
                  scores: V2Ptr, masked: V2Ptr, exps: V2Ptr, weights: V2Ptr,
                  b: Int32, length: Int32, heads: Int32, kv_heads: Int32,
                  keys: Int32, depth: Int32, own0: Int32, window: Int32,
                  scale: Float32):
    """Trace-only [B,H,L,S] cells from the stored (max, denom); never on the
    timed route (the materialized stages are the diagnostic by definition)."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_forward_diagnostic(q, k, maxima, denominators, scores, masked,
                          exps, weights, row, Int(length), Int(heads),
                          Int(kv_heads), Int(keys), Int(depth),
                          Int(own0), Int(window), scale)


def _forward[diagnostic: Bool](q: V2Ptr, k: V2Ptr, v: V2Ptr, output: V2Ptr,
             maxima: V2Ptr, denominators: V2Ptr, scores: V2Ptr, masked: V2Ptr,
             exps: V2Ptr, weights: V2Ptr, b: Int32, length: Int32, heads: Int32,
             kv_heads: Int32, keys: Int32, depth: Int32, own0: Int32,
             window: Int32, scale: Float32):
    """Reference: one thread per row (Apple column, head dims outside the tiled set)."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_forward_row(q, k, v, output, maxima, denominators, row,
                   Int(length), Int(heads), Int(kv_heads), Int(keys),
                   Int(depth), Int(own0), Int(window), scale)
    comptime if diagnostic:
        v2m_forward_diagnostic(q, k, maxima, denominators, scores, masked,
                              exps, weights, row, Int(length), Int(heads),
                              Int(kv_heads), Int(keys), Int(depth),
                              Int(own0), Int(window), scale)


def _enqueue_forward_tiled[HD: Int](
    ctx: DeviceContext, q: V2Ptr, k: V2Ptr, v: V2Ptr, output: V2Ptr,
    maxima: V2Ptr, denominators: V2Ptr, b: Int, length: Int, heads: Int,
    kv_heads: Int, keys: Int, own0: Int, window: Int, scale: Float32,
) raises:
    comptime TQ = ATTN_V2_TQ
    ctx.enqueue_function[_forward_tiled[HD, TQ]](
        q, k, v, output, maxima, denominators,
        Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(own0), Int32(window), scale,
        grid_dim=((length + TQ - 1) // TQ, heads, b), block_dim=(V2_THREADS, 1, 1),
    )


def attention_v2_model_forward[diagnostic: Bool](
    ctx: DeviceContext, mut q: DeviceBuffer[DType.float32],
    mut k: DeviceBuffer[DType.float32], mut v: DeviceBuffer[DType.float32],
    mut output: DeviceBuffer[DType.float32], mut maxima: DeviceBuffer[DType.float32],
    mut denominators: DeviceBuffer[DType.float32], mut scores: DeviceBuffer[DType.float32],
    mut masked: DeviceBuffer[DType.float32], mut exps: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32], b: Int, length: Int, heads: Int,
    kv_heads: Int, keys: Int, depth: Int, own0: Int, window: Int, scale: Float32,
) raises:
    var qp = q.unsafe_ptr()
    var kp = k.unsafe_ptr()
    var vp = v.unsafe_ptr()
    var op = output.unsafe_ptr()
    var mp = maxima.unsafe_ptr()
    var dp = denominators.unsafe_ptr()
    var tiled = False
    comptime if V2_TILED_DEVICE:
        if b * length * heads > 0 and _v2_tiled_head_dim(depth):
            tiled = True
            if depth == 64:
                _enqueue_forward_tiled[64](ctx, qp, kp, vp, op, mp, dp, b, length, heads, kv_heads, keys, own0, window, scale)
            elif depth == 32:
                _enqueue_forward_tiled[32](ctx, qp, kp, vp, op, mp, dp, b, length, heads, kv_heads, keys, own0, window, scale)
            else:
                _enqueue_forward_tiled[16](ctx, qp, kp, vp, op, mp, dp, b, length, heads, kv_heads, keys, own0, window, scale)
            comptime if diagnostic:
                ctx.enqueue_function[_forward_diag](
                    qp, kp, mp, dp, scores.unsafe_ptr(), masked.unsafe_ptr(),
                    exps.unsafe_ptr(), weights.unsafe_ptr(),
                    Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
                    Int32(depth), Int32(own0), Int32(window), scale,
                    grid_dim=((b * heads * length + 63) // 64, 1, 1), block_dim=(64, 1, 1),
                )
    if tiled:
        return
    ctx.enqueue_function[_forward[diagnostic]](
        qp, kp, vp, op, mp, dp, scores.unsafe_ptr(),
        masked.unsafe_ptr(), exps.unsafe_ptr(), weights.unsafe_ptr(),
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=((b * heads * length + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(V2_THREADS)))
def _prepare_tiled[HD: Int](
    q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr, denominators: V2Ptr,
    zdots: V2Ptr, length: Int32, heads: Int32, kv_heads: Int32, keys: Int32,
    own0: Int32, window: Int32, scale: Float32,
):
    """`v2m_prepare_row` for a block of V2_TQB rows: phase A is `v2m_normalizer`
    (the forward fold without V), phase B the zdot chain with the final max."""
    comptime TQ = V2_TQB
    comptime TPR = V2_THREADS // TQ
    comptime KPT = V2_TILE // TPR
    comptime KS = HD + 1
    comptime WS = V2_TILE + 1
    comptime assert TQ * TPR == V2_THREADS and KPT * TPR == V2_TILE and (HD // TPR) * TPR == HD, "attention v2 tiled prepare: HD a multiple of 8"
    # ring: phase A = 64 K rows; phase B = 32 K rows then 32 V rows.
    var ring = stack_allocation[V2_RING * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[TQ * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ws = stack_allocation[TQ * WS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ds = stack_allocation[TQ * WS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ms = stack_allocation[V2_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var r = tid // TPR
    var lane = tid % TPR
    var l = Int(length)
    var nh = Int(heads)
    var nkv = Int(kv_heads)
    var s = Int(keys)
    var o0 = Int(own0)
    var win = Int(window)
    var t0 = Int(block_idx.x) * TQ
    var h = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var kb = (batch * nkv + h // (nh // nkv)) * s * HD
    var t = t0 + r
    var valid = t < l
    var tt = t if valid else l - 1
    var row = (batch * nh + h) * l + tt
    var qb = v2m_query_base(row, l, nh, HD)
    var vis = _v2_lo_hi(tt, s, o0, win)
    var lo = vis[0]
    var hi = vis[1]
    var vis_first = _v2_lo_hi(t0, s, o0, win)
    var vis_last = _v2_lo_hi(min(t0 + TQ - 1, l - 1), s, o0, win)
    var tile_lo = vis_first[0] // V2_TILE
    var tile_hi = -1
    if vis_last[1] > 0:
        tile_hi = (vis_last[1] - 1) // V2_TILE
    var qr = SIMD[DType.float32, HD](0.0)
    if valid:
        comptime for d in range(HD):
            qr[d] = q.unsafe_load(qb + d)
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var maximum = neg_inf
    var denominator = Float32(0.0)
    var sc = SIMD[DType.float32, KPT](0.0)
    # Phase A: the normalizer (forward fold, no V, no output).
    for kt in range(tile_lo, tile_hi + 1):
        comptime for which in range(2):
            var lt = kt + which
            if (which == 1 or kt == tile_lo) and lt <= tile_hi:
                _v2_stage_rows[HD, V2_TILE](ring + (lt % 2) * V2_TILE * KS, k, kb, lt * V2_TILE, s, tid)
        barrier()
        var base = kt * V2_TILE
        var f = lo
        if lo < base:
            f = lo + ((base - lo + V2_TILE - 1) // V2_TILE) * V2_TILE
        var has = valid and lo < hi and f < base + V2_TILE and f < hi
        var last = min(f + V2_TILE, hi)
        var n = 0
        if has:
            n = last - f
        var pmax = neg_inf
        comptime for i in range(KPT):
            var key = f + lane * KPT + i
            if has and key < last:
                var acc = Float32(0.0)
                var kr = (key % V2_RING) * KS
                comptime for d in range(HD):
                    acc = identical_mul_add(qr[d], ring.unsafe_load(kr + d), acc)
                var sv = v2m_mul(acc, scale)
                sc[i] = sv
                pmax = identical_fmax(pmax, sv)
        ms.unsafe_store(tid, pmax)
        barrier()
        var tile_max = neg_inf
        comptime for g in range(TPR):
            tile_max = identical_fmax(tile_max, ms.unsafe_load(r * TPR + g))
        if has:
            var new_max = identical_fmax(maximum, tile_max)
            var correction = Float32(0.0) if maximum == neg_inf else portable_expf(maximum - new_max)
            denominator = v2m_mul(denominator, correction)
            comptime for i in range(KPT):
                var key = f + lane * KPT + i
                if key < last:
                    ws.unsafe_store(r * WS + lane * KPT + i, portable_expf(sc[i] - new_max))
            maximum = new_max
        barrier()
        if has:
            for i in range(n):
                denominator = v2m_add(ws.unsafe_load(r * WS + i), denominator)
        barrier()
    # Phase B: zdot = ascending fold of fma(p, dy.v, zdot) over the visible
    # keys, p with the FINAL max and denominator (contract `v2m_prepare_row`).
    _v2_stage_tokens[HD, TQ](dys, dy, batch, h, t0, l, nh, tid)
    var zdot = Float32(0.0)
    for kt in range(tile_lo, tile_hi + 1):
        _v2_stage_rows[HD, V2_TILE](ring, k, kb, kt * V2_TILE, s, tid)
        _v2_stage_rows[HD, V2_TILE](ring + V2_TILE * KS, v, kb, kt * V2_TILE, s, tid)
        barrier()
        var base = kt * V2_TILE
        var active = valid and lo < hi and base < hi and base + V2_TILE > lo
        comptime for i in range(KPT):
            var pos = lane * KPT + i
            var key = base + pos
            if active and key >= lo and key < hi:
                var acc = Float32(0.0)
                var dacc = Float32(0.0)
                comptime for d in range(HD):
                    acc = identical_mul_add(qr[d], ring.unsafe_load(pos * KS + d), acc)
                    dacc = identical_mul_add(dys.unsafe_load(r * KS + d), ring.unsafe_load((V2_TILE + pos) * KS + d), dacc)
                var p = identical_div(portable_expf(v2m_mul(acc, scale) - maximum), denominator)
                ws.unsafe_store(r * WS + pos, p)
                ds.unsafe_store(r * WS + pos, dacc)
        barrier()
        if active:
            for pos in range(V2_TILE):
                var key = base + pos
                if key >= lo and key < hi:
                    zdot = identical_mul_add(ws.unsafe_load(r * WS + pos), ds.unsafe_load(r * WS + pos), zdot)
        barrier()
    if valid and lane == 0:
        maxima.unsafe_store(row, maximum)
        denominators.unsafe_store(row, denominator)
        zdots.unsafe_store(row, zdot)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(V2_THREADS)))
def _dq_tiled[HD: Int](
    q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr, denominators: V2Ptr,
    zdots: V2Ptr, dq: V2Ptr, length: Int32, heads: Int32, kv_heads: Int32,
    keys: Int32, own0: Int32, window: Int32, scale: Float32,
):
    """`v2m_dq_row` for a block of V2_TQB rows: per device tile, ds per visible
    key to shared, then dq[d] = fma(ds, k[key][d], dq[d]) ascending."""
    comptime TQ = V2_TQB
    comptime TPR = V2_THREADS // TQ
    comptime KPT = V2_TILE // TPR
    comptime DPT = HD // TPR
    comptime KS = HD + 1
    comptime WS = V2_TILE + 1
    comptime assert TQ * TPR == V2_THREADS and KPT * TPR == V2_TILE and DPT * TPR == HD, "attention v2 tiled dq: HD a multiple of 8"
    var ring = stack_allocation[V2_RING * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[TQ * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dss = stack_allocation[TQ * WS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var r = tid // TPR
    var lane = tid % TPR
    var l = Int(length)
    var nh = Int(heads)
    var nkv = Int(kv_heads)
    var s = Int(keys)
    var o0 = Int(own0)
    var win = Int(window)
    var t0 = Int(block_idx.x) * TQ
    var h = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var kb = (batch * nkv + h // (nh // nkv)) * s * HD
    var t = t0 + r
    var valid = t < l
    var tt = t if valid else l - 1
    var row = (batch * nh + h) * l + tt
    var qb = v2m_query_base(row, l, nh, HD)
    var vis = _v2_lo_hi(tt, s, o0, win)
    var lo = vis[0]
    var hi = vis[1]
    var vis_first = _v2_lo_hi(t0, s, o0, win)
    var vis_last = _v2_lo_hi(min(t0 + TQ - 1, l - 1), s, o0, win)
    var tile_lo = vis_first[0] // V2_TILE
    var tile_hi = -1
    if vis_last[1] > 0:
        tile_hi = (vis_last[1] - 1) // V2_TILE
    var qr = SIMD[DType.float32, HD](0.0)
    var maximum = Float32(0.0)
    var denominator = Float32(0.0)
    var zdot = Float32(0.0)
    if valid:
        comptime for d in range(HD):
            qr[d] = q.unsafe_load(qb + d)
        maximum = maxima.unsafe_load(row)
        denominator = denominators.unsafe_load(row)
        zdot = zdots.unsafe_load(row)
    _v2_stage_tokens[HD, TQ](dys, dy, batch, h, t0, l, nh, tid)
    var dacc_q = SIMD[DType.float32, DPT](0.0)
    for kt in range(tile_lo, tile_hi + 1):
        _v2_stage_rows[HD, V2_TILE](ring, k, kb, kt * V2_TILE, s, tid)
        _v2_stage_rows[HD, V2_TILE](ring + V2_TILE * KS, v, kb, kt * V2_TILE, s, tid)
        barrier()
        var base = kt * V2_TILE
        var active = valid and lo < hi and base < hi and base + V2_TILE > lo
        comptime for i in range(KPT):
            var pos = lane * KPT + i
            var key = base + pos
            if active and key >= lo and key < hi:
                var acc = Float32(0.0)
                var dacc = Float32(0.0)
                comptime for d in range(HD):
                    acc = identical_mul_add(qr[d], ring.unsafe_load(pos * KS + d), acc)
                    dacc = identical_mul_add(dys.unsafe_load(r * KS + d), ring.unsafe_load((V2_TILE + pos) * KS + d), dacc)
                var p = identical_div(portable_expf(v2m_mul(acc, scale) - maximum), denominator)
                dss.unsafe_store(r * WS + pos, v2m_mul(v2m_mul(p, dacc - zdot), scale))
        barrier()
        if active:
            for pos in range(V2_TILE):
                var key = base + pos
                if key >= lo and key < hi:
                    var dsv = dss.unsafe_load(r * WS + pos)
                    comptime for j in range(DPT):
                        dacc_q[j] = identical_mul_add(dsv, ring.unsafe_load(pos * KS + lane + j * TPR), dacc_q[j])
        barrier()
    if valid:
        comptime for j in range(DPT):
            dq.unsafe_store(qb + lane + j * TPR, dacc_q[j])


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(V2_THREADS)))
def _dkdv_tiled[HD: Int](
    q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr, denominators: V2Ptr,
    zdots: V2Ptr, dk: V2Ptr, dv: V2Ptr, length: Int32, heads: Int32,
    kv_heads: Int32, keys: Int32, own0: Int32, window: Int32, scale: Float32,
):
    """`v2m_dkdv_key` for a block of 32 keys of one (batch, kv head); grid
    (ceil(S/32), kv_heads, B). Query rows are walked in the contract's
    canonical GQA order: head ascending, then query ascending."""
    comptime KT = V2_TILE            # keys per block
    comptime QT = V2_TILE            # query rows per staged tile
    comptime TPK = V2_THREADS // KT  # lanes per key (8)
    comptime QPT = QT // TPK         # queries each lane scores per tile
    comptime DPT = HD // TPK         # dk/dv lanes each lane owns
    comptime KS = HD + 1
    comptime WS = QT + 1
    comptime assert KT * TPK == V2_THREADS and QPT * TPK == QT and DPT * TPK == HD, "attention v2 tiled dkdv: HD a multiple of 8"
    var kvs = stack_allocation[2 * KT * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var qs = stack_allocation[QT * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dys = stack_allocation[QT * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ps = stack_allocation[KT * WS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dss = stack_allocation[KT * WS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var st = stack_allocation[3 * QT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var kl = tid // TPK   # this lane's key within the block
    var lane = tid % TPK
    var l = Int(length)
    var nh = Int(heads)
    var nkv = Int(kv_heads)
    var s = Int(keys)
    var o0 = Int(own0)
    var win = Int(window)
    var k0 = Int(block_idx.x) * KT
    var kvh = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var group = batch * nkv + kvh
    var kb = group * s * HD
    var repeats = nh // nkv
    var key = k0 + kl
    _v2_stage_rows[HD, KT](kvs, k, kb, k0, s, tid)
    _v2_stage_rows[HD, KT](kvs + KT * KS, v, kb, k0, s, tid)
    # Query rows that can see a key of this block: hi(t) > k0 needs
    # t >= k0 - own0; lo(t) < k0 + KT needs, with a window,
    # t <= k0 + KT - 2 + window - own0. Both bounds are supersets; every
    # cell still checks its own visibility.
    var t_min = max(0, k0 - o0)
    var t_max = l - 1
    if win > 0:
        t_max = min(l - 1, k0 + KT - 2 + win - o0)
    var dk_acc = SIMD[DType.float32, DPT](0.0)
    var dv_acc = SIMD[DType.float32, DPT](0.0)
    if t_min <= t_max:
        var qt_lo = t_min // QT
        var qt_hi = t_max // QT
        for h in range(kvh * repeats, (kvh + 1) * repeats):
            for qt in range(qt_lo, qt_hi + 1):
                var tq0 = qt * QT
                barrier()  # the previous tile's accumulate has finished reading qs/dys/ps/dss
                _v2_stage_tokens[HD, QT](qs, q, batch, h, tq0, l, nh, tid)
                _v2_stage_tokens[HD, QT](dys, dy, batch, h, tq0, l, nh, tid)
                if tid < QT:
                    var tq = tq0 + tid
                    var m = Float32(0.0)
                    var dn = Float32(0.0)
                    var z = Float32(0.0)
                    if tq < l:
                        var qrow = (batch * nh + h) * l + tq
                        m = maxima.unsafe_load(qrow)
                        dn = denominators.unsafe_load(qrow)
                        z = zdots.unsafe_load(qrow)
                    st.unsafe_store(tid, m)
                    st.unsafe_store(QT + tid, dn)
                    st.unsafe_store(2 * QT + tid, z)
                barrier()
                comptime for i in range(QPT):
                    var qi = lane * QPT + i
                    var tq = tq0 + qi
                    if tq < l and key < s:
                        var qv = _v2_lo_hi(tq, s, o0, win)
                        if key >= qv[0] and key < qv[1]:
                            var acc = Float32(0.0)
                            var dacc = Float32(0.0)
                            comptime for d in range(HD):
                                acc = identical_mul_add(qs.unsafe_load(qi * KS + d), kvs.unsafe_load(kl * KS + d), acc)
                                dacc = identical_mul_add(dys.unsafe_load(qi * KS + d), kvs.unsafe_load((KT + kl) * KS + d), dacc)
                            var p = identical_div(portable_expf(v2m_mul(acc, scale) - st.unsafe_load(qi)), st.unsafe_load(QT + qi))
                            ps.unsafe_store(kl * WS + qi, p)
                            dss.unsafe_store(kl * WS + qi, v2m_mul(v2m_mul(p, dacc - st.unsafe_load(2 * QT + qi)), scale))
                barrier()
                if key < s:
                    for qi in range(QT):
                        var tq = tq0 + qi
                        if tq < l:
                            var qv = _v2_lo_hi(tq, s, o0, win)
                            if key >= qv[0] and key < qv[1]:
                                var p = ps.unsafe_load(kl * WS + qi)
                                var dsv = dss.unsafe_load(kl * WS + qi)
                                comptime for j in range(DPT):
                                    dk_acc[j] = identical_mul_add(dsv, qs.unsafe_load(qi * KS + lane + j * TPK), dk_acc[j])
                                    dv_acc[j] = identical_mul_add(p, dys.unsafe_load(qi * KS + lane + j * TPK), dv_acc[j])
    if key < s:
        var outbase = (group * s + key) * HD
        comptime for j in range(DPT):
            dk.unsafe_store(outbase + lane + j * TPK, dk_acc[j])
            dv.unsafe_store(outbase + lane + j * TPK, dv_acc[j])


def _backward_diag(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr,
                   denominators: V2Ptr, zdots: V2Ptr, dw: V2Ptr, dmasked: V2Ptr,
                   dscores: V2Ptr, dqk: V2Ptr, b: Int32, length: Int32,
                   heads: Int32, kv_heads: Int32, keys: Int32, depth: Int32,
                   own0: Int32, window: Int32, scale: Float32):
    """Trace-only backward cells from the stored (max, denom, zdot)."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_backward_diagnostic(q, k, v, dy, maxima, denominators, zdots,
                           dw, dmasked, dscores, dqk, row, Int(length),
                           Int(heads), Int(kv_heads), Int(keys),
                           Int(depth), Int(own0), Int(window), scale)


def _prepare(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr,
             denominators: V2Ptr, zdots: V2Ptr, b: Int32, length: Int32,
             heads: Int32, kv_heads: Int32, keys: Int32, depth: Int32,
             own0: Int32, window: Int32, scale: Float32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_prepare_row(q, k, v, dy, maxima, denominators, zdots, row,
                   Int(length), Int(heads), Int(kv_heads), Int(keys),
                   Int(depth), Int(own0), Int(window), scale)


def _dq[diagnostic: Bool](q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr,
        maxima: V2Ptr, denominators: V2Ptr, zdots: V2Ptr, dq: V2Ptr,
        dw: V2Ptr, dmasked: V2Ptr, dscores: V2Ptr, dqk: V2Ptr,
        b: Int32, length: Int32, heads: Int32, kv_heads: Int32, keys: Int32,
        depth: Int32, own0: Int32, window: Int32, scale: Float32):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(b) * Int(length) * Int(heads):
        return
    v2m_dq_row(q, k, v, dy, maxima, denominators, zdots, dq, row,
              Int(length), Int(heads), Int(kv_heads), Int(keys), Int(depth),
              Int(own0), Int(window), scale)
    comptime if diagnostic:
        v2m_backward_diagnostic(q, k, v, dy, maxima, denominators, zdots,
                               dw, dmasked, dscores, dqk, row, Int(length),
                               Int(heads), Int(kv_heads), Int(keys),
                               Int(depth), Int(own0), Int(window), scale)


def _dkdv(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr,
          denominators: V2Ptr, zdots: V2Ptr, dk: V2Ptr, dv: V2Ptr,
          b: Int32, length: Int32, heads: Int32, kv_heads: Int32, keys: Int32,
          depth: Int32, own0: Int32, window: Int32, scale: Float32):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(b) * Int(kv_heads) * Int(keys):
        return
    v2m_dkdv_key(q, k, v, dy, maxima, denominators, zdots, dk, dv, idx,
                Int(length), Int(heads), Int(kv_heads), Int(keys), Int(depth),
                Int(own0), Int(window), scale)


def _enqueue_backward_tiled[HD: Int](
    ctx: DeviceContext, q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr, maxima: V2Ptr,
    denominators: V2Ptr, zdots: V2Ptr, dq: V2Ptr, dk: V2Ptr, dv: V2Ptr,
    b: Int, length: Int, heads: Int, kv_heads: Int, keys: Int, own0: Int,
    window: Int, scale: Float32,
) raises:
    var row_blocks = (length + V2_TQB - 1) // V2_TQB
    ctx.enqueue_function[_prepare_tiled[HD]](
        q, k, v, dy, maxima, denominators, zdots,
        Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(own0), Int32(window), scale,
        grid_dim=(row_blocks, heads, b), block_dim=(V2_THREADS, 1, 1),
    )
    ctx.enqueue_function[_dq_tiled[HD]](
        q, k, v, dy, maxima, denominators, zdots, dq,
        Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(own0), Int32(window), scale,
        grid_dim=(row_blocks, heads, b), block_dim=(V2_THREADS, 1, 1),
    )
    ctx.enqueue_function[_dkdv_tiled[HD]](
        q, k, v, dy, maxima, denominators, zdots, dk, dv,
        Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(own0), Int32(window), scale,
        grid_dim=((keys + V2_TILE - 1) // V2_TILE, kv_heads, b), block_dim=(V2_THREADS, 1, 1),
    )


def attention_v2_model_backward[diagnostic: Bool](
    ctx: DeviceContext, mut q: DeviceBuffer[DType.float32],
    mut k: DeviceBuffer[DType.float32], mut v: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32], mut maxima: DeviceBuffer[DType.float32],
    mut denominators: DeviceBuffer[DType.float32], mut zdots: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.float32], mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.float32],
    mut dmasked: DeviceBuffer[DType.float32], mut dscores: DeviceBuffer[DType.float32],
    mut dqk: DeviceBuffer[DType.float32], b: Int, length: Int, heads: Int,
    kv_heads: Int, keys: Int, depth: Int, own0: Int, window: Int, scale: Float32,
) raises:
    var qp = q.unsafe_ptr()
    var kp = k.unsafe_ptr()
    var vp = v.unsafe_ptr()
    var dyp = dy.unsafe_ptr()
    var mp = maxima.unsafe_ptr()
    var dp = denominators.unsafe_ptr()
    var zp = zdots.unsafe_ptr()
    var dqp = dq.unsafe_ptr()
    var dkp = dk.unsafe_ptr()
    var dvp = dv.unsafe_ptr()
    var row_grid = (b * heads * length + 63) // 64
    var tiled = False
    comptime if V2_TILED_DEVICE:
        if b * length * heads > 0 and b * kv_heads * keys > 0 and _v2_tiled_head_dim(depth):
            tiled = True
            if depth == 64:
                _enqueue_backward_tiled[64](ctx, qp, kp, vp, dyp, mp, dp, zp, dqp, dkp, dvp, b, length, heads, kv_heads, keys, own0, window, scale)
            elif depth == 32:
                _enqueue_backward_tiled[32](ctx, qp, kp, vp, dyp, mp, dp, zp, dqp, dkp, dvp, b, length, heads, kv_heads, keys, own0, window, scale)
            else:
                _enqueue_backward_tiled[16](ctx, qp, kp, vp, dyp, mp, dp, zp, dqp, dkp, dvp, b, length, heads, kv_heads, keys, own0, window, scale)
            comptime if diagnostic:
                ctx.enqueue_function[_backward_diag](
                    qp, kp, vp, dyp, mp, dp, zp, dw.unsafe_ptr(), dmasked.unsafe_ptr(),
                    dscores.unsafe_ptr(), dqk.unsafe_ptr(),
                    Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
                    Int32(depth), Int32(own0), Int32(window), scale,
                    grid_dim=(row_grid, 1, 1), block_dim=(64, 1, 1),
                )
    if tiled:
        return
    ctx.enqueue_function[_prepare](
        qp, kp, vp, dyp, mp, dp, zp,
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=(row_grid, 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[_dq[diagnostic]](
        qp, kp, vp, dyp, mp, dp, zp, dqp,
        dw.unsafe_ptr(), dmasked.unsafe_ptr(), dscores.unsafe_ptr(), dqk.unsafe_ptr(),
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=(row_grid, 1, 1), block_dim=(64, 1, 1),
    )
    ctx.enqueue_function[_dkdv](
        qp, kp, vp, dyp, mp, dp, zp, dkp, dvp,
        Int32(b), Int32(length), Int32(heads), Int32(kv_heads), Int32(keys),
        Int32(depth), Int32(own0), Int32(window), scale,
        grid_dim=((b * kv_heads * keys + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )
