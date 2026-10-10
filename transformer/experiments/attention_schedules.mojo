# SPDX-License-Identifier: Apache-2.0
"""NN25/26/29 source-only scheduling candidates, 2026-10-06.

NN22 (eager dK/dV pairing) and NN23 (row-dot dS) were deleted on 2026-10-07:
only transformer/checks read them, never a public binding route.

No compilation, identity, quality or timing evidence exists for these drafts.
Every switch requires IDENTICAL and is disabled by MOJOLEARN_IDN_ALL_OFF.
All numeric loops run in Mojo; no vendor or benchmark dimension dispatch.
"""
from transformer.experiments.norm_profile import (
    NN24_NORM_LANES8, NN24_LANES, _sum, _square, norm_profile_dot,
)
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div,
    identical_mul, identical_mul_add, identical_rsqrt, identical_silu,
)

comptime NN_AB_ENABLED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
# TOMBSTONE: MOJOLEARN_NN22_EAGER_DKDV_PAIR (unmeasured) deleted 2026-10-07 by a1488ca2d; code recoverable at a1488ca2d^.
# Restore: git apply experiments/removed/MOJOLEARN_NN22_EAGER_DKDV_PAIR.patch; record in docs/TOMBSTONES.md.
# Each flag below is OFF: full transformer/LM/Samba train-step NVIDIA+AMD
# ratios, host/three-GPU words, task quality and sample counts are pending.
# NN25 = arms 2 and 4 of MOJOLEARN_IDN_NORM (norm_profile_contract.mojo).
comptime NN25_RMS_SPLIT_SCALE = NN_AB_ENABLED and (get_defined_int["MOJOLEARN_IDN_NORM", 0]() == 2 or get_defined_int["MOJOLEARN_IDN_NORM", 0]() == 4)
comptime NN29_RING_PAIR = NN_AB_ENABLED and is_defined["MOJOLEARN_NN29_RING_PAIR"]()


def rms_sumsq_kernel(
    sumsq: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], m_in: Int32, dm_in: Int32,
):
    """NN25: retain the incumbent ascending row fold exactly."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dm = Int(dm_in)
    if row >= Int(m_in):
        return
    var acc = Float32(0.0)
    for j in range(dm):
        var v = ftz(x.unsafe_load(row * dm + j))
        acc = ftz(identical_mul_add(v, v, acc))
    comptime if NN24_NORM_LANES8:
        acc = norm_profile_dot[NN24_LANES](x, x, row * dm, row * dm, dm)
    sumsq.unsafe_store(row, acc)


def rms_parallel_scale_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32, dm_in: Int32, eps: Float32,
):
    """NN25: one cell/thread; scalar recomputation trades work for parallelism.

    Reading the stored sumsq repeats the identical div/rsqrt graph. There
    is no scalar scratch allocation or lifetime to hide from cold timing.
    This arm may lose from repeated rsqrt; record that outcome if it does.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dm = Int(dm_in)
    if i >= Int(m_in) * dm:
        return
    var mean = ftz(identical_div(sumsq.unsafe_load(i // dm), Float32(dm)))
    var rstd = ftz(identical_rsqrt(ftz(mean + eps)))
    var inner = ftz(identical_mul(ftz(x.unsafe_load(i)), rstd))
    output.unsafe_store(i, ftz(identical_mul(ftz(weight.unsafe_load(i % dm)), inner)))


def training_swiglu_kernel(
    silu_out: MutPointer[Float32, MutAnyOrigin],
    gated: MutPointer[Float32, MutAnyOrigin],
    gate: MutPointer[Float32, MutAnyOrigin],
    up: MutPointer[Float32, MutAnyOrigin], n_in: Int32,
):
    """NN26: fuse forward activation/product, retaining backward/trace state.

    Unlike forward-only swiglu_fused_kernel, both S20 and S21 are written.
    SiLU remains the portable single quotient, never x*sigmoid(x). The
    existing paired backward can consume these unchanged saved words.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var sil = ftz(identical_silu(ftz(gate.unsafe_load(i))))
    silu_out.unsafe_store(i, sil)
    gated.unsafe_store(i, ftz(identical_mul(sil, ftz(up.unsafe_load(i)))))


def kv_ring_pair_kernel(
    ring_k: MutPointer[Float32, MutAnyOrigin],
    ring_v: MutPointer[Float32, MutAnyOrigin],
    fresh_k: MutPointer[Float32, MutAnyOrigin],
    fresh_v: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nkv_in: Int32, hd_in: Int32,
    cap_in: Int32, pos0_in: Int32,
):
    """NN29: write both ring caches in one traversal, one owner per slot.

    For calls longer than capacity, choose the last token for each slot.
    The existing launcher requires the span gather to complete before this
    enqueue on its in-order context; no cache read can overlap these writes.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var l = Int(l_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    var cap = Int(cap_in)
    var pos0 = Int(pos0_in)
    if i >= Int(b_in) * nkv * cap * hd:
        return
    var d = i % hd
    var slot = (i // hd) % cap
    var kvh = (i // (hd * cap)) % nkv
    var bb = i // (hd * cap * nkv)
    var end = pos0 + l - 1
    var pos = end - ((end % cap - slot + cap) % cap)
    if pos < pos0:
        return
    var src = (bb * l + pos - pos0) * nkv * hd + kvh * hd + d
    ring_k.unsafe_store(i, fresh_k.unsafe_load(src))
    ring_v.unsafe_store(i, fresh_v.unsafe_load(src))

# NN27 retains LlamaRopeTable's existing validated construction/lifetime;
# the new schedule shares its immutable position cells between Q and K.
# No pointer-based trust cache or skipped refusal scan is introduced.
# ONE RoPE switch (NN27 + NI23 merged, 2026-10-07): -D MOJOLEARN_IDN_ROPE=<arm>,
#   1  qk_pair   NN27: rotate Q and K in one launch (this kernel)
#   2  k_cache   NI23: rotate K while writing the prefill cache
#                (training/neural_identical_experiments.mojo)
# Same products and adds as the incumbent: no bit change.
comptime NN27_QK_ROPE_PAIR = NN_AB_ENABLED and get_defined_int["MOJOLEARN_IDN_ROPE", 0]() == 1


def qk_rope_pair_kernel(
    q_out: MutPointer[Float32, MutAnyOrigin],
    k_out: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    cos_tab: MutPointer[Float32, MutAnyOrigin],
    sin_tab: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32, l_in: Int32, nh_in: Int32, nkv_in: Int32,
    hd_in: Int32, pos0_in: Int32, rd_in: Int32,
):
    """NN27: Q and matching KV head reuse their position/frequency load.

    The grid owns Q cells; the first nkv heads also write K. GQA has
    nh>=nkv by LlamaDims validation, and every K cell has one owner.
    Both products round before addition exactly as the split rotation.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var hd = Int(hd_in)
    if i >= Int(m_in) * nh * hd:
        return
    var tok = i // (nh * hd)
    var h = (i // hd) % nh
    var d = i % hd
    var rd = Int(rd_in)
    var ki = tok * nkv * hd + h * hd + d
    if d >= rd:
        q_out.unsafe_store(i, ftz(q.unsafe_load(i)))
        if h < nkv:
            k_out.unsafe_store(ki, ftz(k.unsafe_load(ki)))
        return
    var half = rd // 2
    var f = d % half
    var ti = (Int(pos0_in) + tok % Int(l_in)) * half + f
    var c = ftz(cos_tab.unsafe_load(ti))
    var s = ftz(sin_tab.unsafe_load(ti))
    var offset = half if d < half else -half
    var qr = ftz(q.unsafe_load(i + offset))
    if d < half:
        qr = -qr
    var qa = ftz(identical_mul(ftz(q.unsafe_load(i)), c))
    var qb = ftz(identical_mul(qr, s))
    q_out.unsafe_store(i, ftz(ftz(qa) + ftz(qb)))
    if h < nkv:
        var kr = ftz(k.unsafe_load(ki + offset))
        if d < half:
            kr = -kr
        var ka = ftz(identical_mul(ftz(k.unsafe_load(ki)), c))
        var kb = ftz(identical_mul(kr, s))
        k_out.unsafe_store(ki, ftz(ftz(ka) + ftz(kb)))

comptime NN21_SCALE_MASK = NN_AB_ENABLED and is_defined["MOJOLEARN_NN21_SCALE_MASK"]()
# TOMBSTONE: MOJOLEARN_NN23_ROWDOT_DS (unmeasured) deleted 2026-10-07 by a1488ca2d; code recoverable at a1488ca2d^.
# Restore: git apply experiments/removed/MOJOLEARN_NN23_ROWDOT_DS.patch; record in docs/TOMBSTONES.md.


def score_scale_mask_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    masked: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32, l_in: Int32, s_in: Int32,
    pos0_in: Int32, key_lo_in: Int32, window_in: Int32,
    scale: Float32, mask_fill: Float32,
):
    """NN21 eager extension: same rounded score, then mandatory mask add.

    Both public intermediates are stored. Softcap, plants and sabotage use
    the incumbent split route; supported zero-bias causal/window masks only.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(cells_in):
        return
    var s = Int(s_in)
    var p = Int(pos0_in) + (i // s) % Int(l_in)
    var k = Int(key_lo_in) + i % s
    var visible = k <= p
    if Int(window_in) > 0 and k <= p - Int(window_in):
        visible = False
    var score = ftz(identical_mul(ftz(scores.unsafe_load(i)), scale))
    scores.unsafe_store(i, score)
    var fill = Float32(0.0) if visible else mask_fill
    masked.unsafe_store(i, ftz(score + fill))

