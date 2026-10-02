# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU inference for the byte-level decoder language model (DEVIATION 2610).

HOST ONLY, AND NO NEW ARITHMETIC. Every number this file produces comes from
the host FP32 oracles that the device kernels are gated against bit for bit,
called in the order `training/byte_lm.mojo::_byte_forward_loss` launches the
device kernels:

    embedding   emb_forward_oracle             identical_embedding_forward_into
    each block  transformer_block_oracle       llama_decoder_layer_forward
                (fresh cache, prefill at 0)    (prefill_cache.s = 0)
    head        gemm_oracle(OP_NT)             identical_gemm_into(..., OP_NT)
    loss        ce_forward_oracle(causal_lm)   identical_ce_forward_into

There is no final norm between the last block and the head, because the
device forward has none. The parameter registry is
`training/byte_lm_config.mojo`'s, and block weights are sliced in the order
`training/byte_lm.mojo::_block_weights` hands them to the device.

What that composition promises is a PREDICTION until the gate runs.
`tools/byte_lm_host_gate.py` (DEVIATION 2613) compares the loss bytes this
file produces against the retained Metal, CUDA and HIP captures of the same
parameters and batches. Nothing here is qualified by construction.

DEVIATION 2611. `transformer/checks/transformer_oracle.mojo` imports
`core/identity_trace.mojo`, which imports `max.gpu.host` (that file's
DEVIATION 1003). This module never creates a context or a kernel, but the
import means a CPU-only build still needs the MAX package on the box. Whether
it compiles with no accelerator target is measured on the first CPU-only box,
not assumed.

DEVIATION 2612. `MOJOLEARN_BYTE_LM_HOST_SABOTAGE` replaces the head product
with the same k-term chain folded in REVERSE order through the same seam. At
the admitted profile k = 32, which `contract_leaf_size` makes one serial
leaf, so the reversal changes the fold and nothing else. It is the gate's
negative control: a build with it defined must fail the loss comparison, or
the comparison is not reaching the arithmetic. The threaded path reverses the
fold inside the kernel it actually runs (`gemm_nt_rows(..., reverse=True)`).
"""

from std.math import max, min
from std.os import getenv
from std.sys.compile import is_defined
from std.sys.info import num_physical_cores, simd_width_of

from std.memory import bitcast, unsafe_memcpy

from core.host_lanes import ftz_lanes, host_f32_uninit
from core.host_parallel import host_parallelize

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from embedding.checks.embedding_oracle import EmbConfig, emb_forward_oracle, refuse_nonfinite
from gemm.host.gemm_host_rows import GhrPtr
from gemm.contract import GEMM_ORACLE_HOST_SABOTAGE, OP_NT
from gemm.host.identical_gemm import gemm_oracle
from training.byte_lm_config import ByteConfig
from training.byte_lm_host_kernels import (
    all_finite_span,
    ce_causal_mean_loss_fast,
    flushed_span,
    gemm_nt_rows,
    gemm_w,
    hidden_fast,
    pack_nt_span,
    pack_w_range,
    pack_w_units,
    packed_w_len,
    _residual_add,
    _silu_gated,
    _softmax_head,
    _value_sum_head,
    copy_rows,
    rms_norm_fast,
    rope_rows,
)
from training.checks.loss_oracle import CeConfig, ce_forward_oracle
from transformer.checks.transformer_fixture import (
    attention_scale,
    mask_fill,
    unmasked_fill,
    ScorePlant,
    TransformerDims,
    TransformerWeights,
)
from transformer.checks.transformer_oracle import (
    RopeTable,
    TransformerKVCache,
    build_rope_table,
    refuse_bad_weights,
    transformer_block_oracle,
)


comptime BYTE_HOST_SABOTAGE = is_defined["MOJOLEARN_BYTE_LM_HOST_SABOTAGE"]()
comptime BYTE_HOST_MAX_THREADS = 1024
comptime HOST_FW_P = simd_width_of[DType.float32]()
comptime U32P = SIMD[DType.uint32, HOST_FW_P]


def byte_host_sabotage_compiled() -> Bool:
    """Whether this binary carries the DEVIATION 2612 negative control."""
    comptime if BYTE_HOST_SABOTAGE:
        return True
    return False


def _require_identical() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("byte LM host: requires -D MOJOLEARN_NUMERIC_IDENTICAL=1")


def _slice(values: List[Float32], offsets: List[Int], j: Int) -> List[Float32]:
    """Registry tensor `j` as one block copy (lane neural-pass8): the same
    values in the same order as the element loop it replaces."""
    var n = offsets[j + 1] - offsets[j]
    var out = host_f32_uninit(n)
    if n > 0:
        unsafe_memcpy(dest=out.unsafe_ptr(), src=values.unsafe_ptr().unsafe_offset(offsets[j]), count=n)
    return out^


def byte_host_dims(config: ByteConfig) raises -> TransformerDims:
    """The block shape. RoPE is sized from the configured length, as the
    device table `LlamaRopeTable(ctx, dims, 10000, config.length)` is."""
    config.validate()
    return TransformerDims(config.d_model, config.n_heads, config.n_kv,
                           config.head_dim, config.intermediate, config.length)


def byte_host_block_weights(params: List[Float32], offsets: List[Int], block: Int,
                            dims: TransformerDims) raises -> TransformerWeights:
    """Registry order per block: norm1_w, w_q, w_k, w_v, w_o, norm2_w,
    w_gate, w_up, w_down (`training/byte_lm.mojo::byte_param_name`)."""
    var base = 1 + 9 * block
    var w = TransformerWeights(dims)
    w.norm1_w = _slice(params, offsets, base)
    w.w_q = _slice(params, offsets, base + 1)
    w.w_k = _slice(params, offsets, base + 2)
    w.w_v = _slice(params, offsets, base + 3)
    w.w_o = _slice(params, offsets, base + 4)
    w.norm2_w = _slice(params, offsets, base + 5)
    w.w_gate = _slice(params, offsets, base + 6)
    w.w_up = _slice(params, offsets, base + 7)
    w.w_down = _slice(params, offsets, base + 8)
    refuse_bad_weights(w)
    return w^


def _sabotaged_cell(a: List[Float32], b: List[Float32], i: Int, j: Int, k: Int) -> Float32:
    """DEVIATION 2612 only. One OP_NT cell with k walked DESCENDING."""
    var acc = Float32(0.0)
    var kk = k - 1
    while kk >= 0:
        acc = identical_mul_add(a[i * k + kk], b[j * k + kk], acc)
        kk -= 1
    return acc


def _sabotaged_head(a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int) -> List[Float32]:
    var out = List[Float32](capacity=m * n)
    for i in range(m):
        for j in range(n):
            out.append(_sabotaged_cell(a, b, i, j, k))
    return out^


def _validate_logits_inputs(params: List[Float32], inputs: List[Int32], batch: Int,
                            length: Int, config: ByteConfig) raises:
    _require_identical()
    config.validate()
    if batch <= 0 or length <= 0:
        raise Error("byte LM host: batch and length must be positive")
    if length > config.length:
        raise Error("byte LM host: length exceeds the configured length the RoPE table is sized for")
    var n = config.n_total()
    if len(params) != n:
        raise Error("byte LM host: expected " + String(n) + " parameters, got " + String(len(params)))
    if len(inputs) != batch * length:
        raise Error("byte LM host: ids must hold batch * length tokens")
    for t in range(len(inputs)):
        var v = Int(inputs[t])
        if v < 0 or v >= config.vocab_size:
            raise Error("byte LM host: token id outside [0, vocab) at " + String(t))


def _byte_host_hidden(params: List[Float32], inputs: List[Int32], batch: Int,
                      length: Int, config: ByteConfig) raises -> List[Float32]:
    """The last block's residual `[batch * length, d_model]`. Inputs are
    validated by the caller."""
    var offsets = config.offsets()
    var dims = byte_host_dims(config)
    var rope = build_rope_table(dims)
    var x = emb_forward_oracle(_slice(params, offsets, 0), inputs,
                               EmbConfig.llama(config.vocab_size, config.d_model))
    for layer in range(config.n_layers):
        var w = byte_host_block_weights(params, offsets, layer, dims)
        var cache = TransformerKVCache(batch, dims, length, 0)
        var st = transformer_block_oracle(w, x, batch, length, cache, rope, ScorePlant.none())
        x = st.residual2_out.copy()
    return x^


def byte_host_logits(params: List[Float32], inputs: List[Int32], batch: Int,
                     length: Int, config: ByteConfig) raises -> List[Float32]:
    """Logits `[batch * length, vocab]`, row-major, for token ids
    `[batch, length]` starting at absolute position 0. THE REFERENCE PATH:
    one thread, the oracles exactly as written.

    `length` may be shorter than the configured length and `batch` may
    differ from it. The block and head contracts carry no fold across rows,
    so a row's logits do not depend on the batch around it; the gate
    measures that at the configured shape only."""
    _validate_logits_inputs(params, inputs, batch, length, config)
    var x = _byte_host_hidden(params, inputs, batch, length, config)
    var head = _slice(params, config.offsets(), config.n_tensors() - 1)
    var m = batch * length
    comptime if BYTE_HOST_SABOTAGE:
        return _sabotaged_head(x, head, m, config.vocab_size, config.d_model)
    return gemm_oracle(x, head, OP_NT, m, config.vocab_size, config.d_model)


# ===========================================================================
# THE THREADED PATH (DEVIATIONS 2616, 2640)
# ===========================================================================
# Same bits as the reference, and gated to prove it. Since DEVIATION 2640 it
# runs `training/byte_lm_host_kernels.mojo`: the oracles' arithmetic, per
# output value in the oracles' order, without their per-cell allocation,
# with operands flushed and packed once per call and the cells of a row
# advanced as SIMD lanes. It runs on at most `threads` threads (0: one per
# physical core). Threads split only along axes the contracts already make
# independent, so no float crosses a thread boundary and no fold changes
# order:
#
#   rows        batch rows in at most `threads` contiguous chunks, each chunk
#               running its rows one after another into disjoint slices.
#               Rows share nothing in any block or head contract. One row,
#               or one thread, runs on the calling thread. (The first cut
#               also split a single row's head product across threads; on
#               the M4 at [1, 32] that measured slower, 0.70 against 0.57
#               ms, and it was removed.)
#   loss        `ce_causal_mean_loss_fast` on the calling thread, the loss
#               oracle's seams with its folds through `gemm_nt_rows`.
#
# No task starts another parallel region. Owners that tasks read through a
# pointer are transferred only AFTER the join (the step-33 race class,
# `gbdt/train.mojo`).


def byte_host_worker_count(threads: Int) raises -> Int:
    """`threads` in [1, BYTE_HOST_MAX_THREADS] as given; 0 is one per
    physical core."""
    if threads < 0 or threads > BYTE_HOST_MAX_THREADS:
        raise Error("byte LM host: threads must be in [0, " + String(BYTE_HOST_MAX_THREADS) + "]")
    if threads > 0:
        return threads
    var cores = num_physical_cores()
    if cores < 1:
        return 1
    return cores


# ===========================================================================
# THE TOKEN SPLIT (lane/neural-pass5, 2026-09-30): one batch row over many
# threads.
#
# `_threaded_rows` splits the work across BATCH ROWS, one row per task, so
# the board's `lm-infer` (batch 1, 2048 tokens) and every one-row call ran
# on ONE core of a 28- or 64-core host: 15x to 66x behind torch's CPU path.
# `hidden_par` splits INSIDE a row along the axes the contracts already make
# independent, and nothing else: a token's output cells in the norms, the
# projections, the MLP and the head (each cell its own chain over k, the
# same chain whoever runs it), and a (head, query) pair in attention (the
# scores chain over head_dim, the softmax folds over the keys, the value
# sum over the keys, each per query). Every stage runs the SAME kernels as
# `block_fast` on chunk-local lists, so no float crosses a thread boundary
# and no fold changes order; the pinned environment of `host_parallelize`
# makes a chunk's bits the calling thread's. Two regions a block, one after
# the other (no task starts another parallel region): the pre-attention
# region per token chunk (norm1, q, k, v, rope) into shared q_r, k_r, v
# tables, then a region per query chunk that runs all heads of its queries
# (packing the head's keys and values itself) and the rest of the block for
# those tokens (o_proj, residual, norm2, gate, up, SiLU, down, residual).
# The head GEMM splits its token rows the same way. Taken when the batch
# is smaller than the worker count, unless MOJOLEARN_BYTE_LM_HOST_TOKEN_SPLIT=0.
# ===========================================================================


def byte_host_token_split_enabled() -> Bool:
    return String(getenv("MOJOLEARN_BYTE_LM_HOST_TOKEN_SPLIT")) != "0"


def _chunks(n: Int, workers: Int) -> Int:
    var c = workers
    if c > n:
        c = n
    if c < 1:
        c = 1
    return c


def block_par(
    held: List[List[List[Float32]]],
    tb: Int,
    x: List[Float32],
    l: Int,
    dims: TransformerDims,
    ropes: List[RopeTable],
    workers: Int,
) raises -> List[Float32]:
    """`block_fast(held[0], tb, x, l, dims, ropes[0])` over `workers`
    threads, the section comment above. `held` and `ropes` are the caller's
    one-element wrappers (the tensors are never copied). Returns the same
    `[l, d_model]` residual."""
    var dm = dims.d_model
    var nh = dims.n_heads
    var hd = dims.head_dim
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate
    var n_rep = dims.n_rep()
    var s = l
    if len(x) != l * dm:
        raise Error("byte LM host: block input has the wrong length")
    if not all_finite_span(x, 0, len(x)):
        raise Error("byte LM host: non-finite block input x")
    var tasks = _chunks(l, workers)
    var chunk = (l + tasks - 1) // tasks

    # ---- region A: norm1, q, k, v, rope per token chunk into shared tables
    var qr = List[Float32](length=l * qw, fill=Float32(0.0))
    var kr = List[Float32](length=l * kw, fill=Float32(0.0))
    var v = List[Float32](length=l * kw, fill=Float32(0.0))
    var failed = List[Int](length=tasks, fill=0)
    var tp = held.unsafe_ptr()
    var rp = ropes.unsafe_ptr()
    var xs = List[List[Float32]]()
    xs.append(x.copy())
    var xp = xs.unsafe_ptr()
    var qrp = qr.unsafe_ptr()
    var krp = kr.unsafe_ptr()
    var vp = v.unsafe_ptr()
    var fp = failed.unsafe_ptr()
    var c_kv = dims.n_kv_heads

    def _pre_task(c: Int) {imm tp, imm rp, imm xp, imm qrp, imm krp, imm vp, imm fp, imm chunk, imm l,
                           imm dm, imm nh, imm hd, imm qw, imm kw, imm tb, imm c_kv}:
        try:
            var lo = c * chunk
            var hi = lo + chunk
            if hi > l:
                hi = l
            if lo >= hi:
                return  # a trailing chunk past the last token: nothing to do
            var rows = hi - lo
            var xc = copy_rows(xp[0], lo, hi, dm)
            var n1 = rms_norm_fast(xc, tp[][tb], rows, dm)
            var q = List[Float32](length=rows * qw, fill=Float32(0.0))
            var k = List[Float32](length=rows * kw, fill=Float32(0.0))
            var vv = List[Float32](length=rows * kw, fill=Float32(0.0))
            gemm_w(n1, tp[][tb + 1], qw, dm, 0, rows, q)
            gemm_w(n1, tp[][tb + 2], kw, dm, 0, rows, k)
            gemm_w(n1, tp[][tb + 3], kw, dm, 0, rows, vv)
            var qrc = rope_rows(q, nh, hd, lo, hi, rp[])
            var krc = rope_rows(k, c_kv, hd, lo, hi, rp[])
            for i in range(rows * qw):
                qrp.unsafe_store(lo * qw + i, qrc[i])
            for i in range(rows * kw):
                krp.unsafe_store(lo * kw + i, krc[i])
                vp.unsafe_store(lo * kw + i, vv[i])
        except:
            fp.unsafe_store(c, 1)

    if tasks == 1:
        _pre_task(0)
    else:
        host_parallelize(_pre_task, tasks)
    for c in range(tasks):
        if failed[c] != 0:
            raise Error("byte LM host: token chunk " + String(c) + " raised in the projections")

    # ---- region B: attention over all heads and the rest of the block per
    # query chunk. The mask table is the block's, read only.
    var qrs = List[List[Float32]]()
    qrs.append(qr^)
    var krs = List[List[Float32]]()
    krs.append(kr^)
    var vs = List[List[Float32]]()
    vs.append(v^)
    # The causal mask is formed inside `_softmax_head` (q_first = lo), so
    # no [l, s] mask is built or copied (lane neural-pass16).
    var scale = attention_scale(hd)
    var out = List[Float32](length=l * dm, fill=Float32(0.0))
    var ms = List[List[Float32]]()
    ms.append(List[Float32]())
    var mp = ms.unsafe_ptr()
    # EVERY HEAD'S KEYS AND VALUES PACKED ONCE (lane neural-pass16): each
    # post task packed the full key span of every head itself, so the packs
    # were written `tasks` times a layer (64 tasks x 6 heads x 2048 x 64 x 2
    # floats on a 64-core host). Now `kpacks[h]` ([hd, s], flushed) and
    # `vpacks[h]` ([s, hd], flushed) are built once, (head, key range) over
    # tasks, and every post task reads them. A pack is a copy through `ftz`;
    # the products read the same values.
    var kpacks = List[Float32](length=nh * hd * s, fill=Float32(0.0))
    var vpacks = List[Float32](length=nh * s * hd, fill=Float32(0.0))
    var kpk = kpacks.unsafe_ptr()
    var vpk = vpacks.unsafe_ptr()
    var ksrc = krs.unsafe_ptr()
    var vsrc = vs.unsafe_ptr()
    var pack_rows = nh * tasks
    def _pack_task(r: Int) {imm ksrc, imm vsrc, imm kpk, imm vpk, imm tasks, imm chunk, imm s, imm hd, imm kw, imm n_rep}:
        var h = r // tasks
        var c = r % tasks
        var kvh = h // n_rep
        var lo = c * chunk
        var hi = lo + chunk
        if hi > s:
            hi = s
        var ksp = ksrc[0].unsafe_ptr()
        var vsp = vsrc[0].unsafe_ptr()
        for j in range(lo, hi):
            for d in range(hd):
                kpk.unsafe_store(h * hd * s + d * s + j, ftz(ksp.unsafe_load(j * kw + kvh * hd + d)))
                vpk.unsafe_store(h * s * hd + j * hd + d, ftz(vsp.unsafe_load(j * kw + kvh * hd + d)))
    if pack_rows == 1:
        _pack_task(0)
    else:
        host_parallelize(_pack_task, pack_rows)
    var qtp = qrs.unsafe_ptr()
    var op = out.unsafe_ptr()
    for c in range(tasks):
        failed[c] = 0

    def _post_task(c: Int) {imm tp, imm xp, imm mp, imm qtp, imm kpk, imm vpk, imm op, imm fp, imm chunk,
                            imm l, imm s, imm dm, imm nh, imm hd, imm qw, imm kw, imm inter, imm n_rep,
                            imm tb, imm scale}:
        try:
            var lo = c * chunk
            var hi = lo + chunk
            if hi > l:
                hi = l
            if lo >= hi:
                return  # a trailing chunk past the last token: nothing to do
            var rows = hi - lo
            var mchunk = mp[0].copy()  # empty: the fill is formed in the kernel
            var ctx = List[Float32](length=rows * qw, fill=Float32(0.0))
            var qmat = List[Float32](length=rows * hd, fill=Float32(0.0))
            var kpack = List[Float32](length=hd * s, fill=Float32(0.0))
            var vpack = List[Float32](length=s * hd, fill=Float32(0.0))
            var cell = List[Float32](length=rows * s, fill=Float32(0.0))
            var aweights = List[Float32](length=rows * s, fill=Float32(0.0))
            var qsp = qtp[0].unsafe_ptr()
            var qmp = qmat.unsafe_ptr()
            var kpp = kpack.unsafe_ptr()
            var vpp = vpack.unsafe_ptr()
            for h in range(nh):
                for qi in range(rows):
                    for d in range(hd):
                        qmp.unsafe_store(qi * hd + d, qsp.unsafe_load((lo + qi) * qw + h * hd + d))
                # this head's shared packs, a block copy each
                unsafe_memcpy(dest=kpp, src=kpk.unsafe_offset(h * hd * s), count=hd * s)
                unsafe_memcpy(dest=vpp, src=vpk.unsafe_offset(h * s * hd), count=s * hd)
                gemm_nt_rows(qmat, kpack, s, hd, 0, rows, cell)
                _softmax_head(cell, mchunk, rows, s, scale, aweights, lo)
                _value_sum_head(aweights, vpack, rows, s, hd, qw, h, ctx, lo)
            var xc = copy_rows(xp[0], lo, hi, dm)
            var o = List[Float32](length=rows * dm, fill=Float32(0.0))
            gemm_w(ctx, tp[][tb + 4], dm, qw, 0, rows, o)
            var r1 = _residual_add(xc, o)
            var n2 = rms_norm_fast(r1, tp[][tb + 5], rows, dm)
            var gate = List[Float32](length=rows * inter, fill=Float32(0.0))
            var up = List[Float32](length=rows * inter, fill=Float32(0.0))
            gemm_w(n2, tp[][tb + 6], inter, dm, 0, rows, gate)
            gemm_w(n2, tp[][tb + 7], inter, dm, 0, rows, up)
            var gated = _silu_gated(gate, up)
            var down = List[Float32](length=rows * dm, fill=Float32(0.0))
            gemm_w(gated, tp[][tb + 8], dm, inter, 0, rows, down)
            var res = _residual_add(r1, down)
            for i in range(rows * dm):
                op.unsafe_store(lo * dm + i, res[i])
        except:
            fp.unsafe_store(c, 1)

    if tasks == 1:
        _post_task(0)
    else:
        host_parallelize(_post_task, tasks)
    _ = xs^
    _ = ms^
    _ = qrs^
    _ = krs^
    _ = vs^
    _ = kpacks^
    _ = vpacks^
    for c in range(tasks):
        if failed[c] != 0:
            raise Error("byte LM host: token chunk " + String(c) + " raised in attention or the MLP")
    return out^


def hidden_par(
    held: List[List[List[Float32]]],
    ropes: List[RopeTable],
    row_ids: List[Int32],
    l: Int,
    dims: TransformerDims,
    layers: Int,
    workers: Int,
) raises -> List[Float32]:
    """`hidden_fast(held[0], ropes[0], ...)` with every block through
    `block_par`."""
    var dm = dims.d_model
    var ep = held[0][0].unsafe_ptr()
    var x = List[Float32](length=l * dm, fill=Float32(0.0))
    var xp = x.unsafe_ptr()
    for t in range(l):
        var base = Int(row_ids[t]) * dm
        for j in range(dm):
            xp.unsafe_store(t * dm + j, ep.unsafe_load(base + j))
    for layer in range(layers):
        x = block_par(held, 1 + 9 * layer, x, l, dims, ropes, workers)
    return x^


def head_rows_par(
    hidden: List[Float32],
    held: List[List[List[Float32]]],
    head_index: Int,
    vocab: Int,
    dm: Int,
    l: Int,
    reverse: Bool,
    workers: Int,
) raises -> List[Float32]:
    """`gemm_nt_rows(hidden, held[0][head_index], vocab, dm, 0, l, out,
    reverse)` with the token rows split over `workers` threads. Returns
    `[l, vocab]`."""
    var tasks = _chunks(l, workers)
    var chunk = (l + tasks - 1) // tasks
    var out = List[Float32](length=l * vocab, fill=Float32(0.0))
    var failed = List[Int](length=tasks, fill=0)
    var hs = List[List[Float32]]()
    hs.append(hidden.copy())
    var hp = hs.unsafe_ptr()
    var wp = held.unsafe_ptr()
    var op = out.unsafe_ptr()
    var fp = failed.unsafe_ptr()

    def _head_task(c: Int) {imm hp, imm wp, imm op, imm fp, imm chunk, imm l, imm vocab, imm dm, imm reverse, imm head_index}:
        try:
            var lo = c * chunk
            var hi = lo + chunk
            if hi > l:
                hi = l
            if lo >= hi:
                return  # a trailing chunk past the last token: nothing to do
            var part = List[Float32](length=(hi - lo) * vocab, fill=Float32(0.0))
            gemm_w(hp[0], wp[][head_index], vocab, dm, lo, hi, part, reverse)
            for i in range((hi - lo) * vocab):
                op.unsafe_store(lo * vocab + i, part[i])
        except:
            fp.unsafe_store(c, 1)

    if tasks == 1:
        _head_task(0)
    else:
        host_parallelize(_head_task, tasks)
    _ = hs^
    for c in range(tasks):
        if failed[c] != 0:
            raise Error("byte LM host: head chunk " + String(c) + " raised")
    return out^


def _span(offsets: List[Int], j: Int, want: Int) raises -> Int:
    """The offset of registry tensor `j`, refusing a size other than `want`."""
    if offsets[j + 1] - offsets[j] != want:
        raise Error("byte LM host: tensor " + String(j) + " holds " + String(offsets[j + 1] - offsets[j])
                    + " values, expected " + String(want))
    return offsets[j]


def byte_host_fast_tensors(params: List[Float32], config: ByteConfig) raises -> List[List[Float32]]:
    """The threaded path's operands, prepared once per call in registry
    order and read straight from `params` by offset: the flushed embedding;
    per block norm1 flushed, q, k, v, o packed, norm2 flushed, gate, up, down
    packed; then the packed head.

    The reference path refuses a non-finite embedding (`emb_forward_oracle`)
    or block weight (`refuse_bad_weights`) and nothing in the head. So the
    embedding and the blocks are screened lane-wise, and a non-finite value
    there re-enters those same refusals, which raise with their own messages.
    (On the M4 the element-by-element copies and scans were about 90% of a
    one-token call.)"""
    var offsets = config.offsets()
    var dims = byte_host_dims(config)
    var dm = config.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var ff = config.intermediate
    var head_j = config.n_tensors() - 1
    var nt = config.n_tensors()
    # lane/neural-pass73 (2026-10-01): the registry tensors in order, each a
    # flushed span (kind 0: the embedding and the norms) or a weight packed
    # for `gemm_w` (kind 1, `[n x k]`), prepared over host tasks: the screen
    # and the copies are per element, so any split moves no bit. A
    # non-finite value found by any task re-enters the serial refusals.
    var kind = List[Int](length=nt, fill=0)
    var tn = List[Int](length=nt, fill=0)
    var tk = List[Int](length=nt, fill=0)
    var tlo = List[Int](length=nt, fill=0)
    var tlen = List[Int](length=nt, fill=0)
    tlo[0] = _span(offsets, 0, config.vocab_size * dm)
    tlen[0] = config.vocab_size * dm
    for layer in range(config.n_layers):
        var base = 1 + 9 * layer
        tlo[base] = _span(offsets, base, dm)
        tlen[base] = dm
        tlo[base + 5] = _span(offsets, base + 5, dm)
        tlen[base + 5] = dm
        var shapes_n = [qw, kw, kw, dm, ff, ff, dm]
        var shapes_k = [dm, dm, dm, qw, dm, dm, ff]
        var slots = [1, 2, 3, 4, 6, 7, 8]
        for q in range(7):
            var j = base + slots[q]
            kind[j] = 1
            tn[j] = shapes_n[q]
            tk[j] = shapes_k[q]
            tlo[j] = _span(offsets, j, tn[j] * tk[j])
            tlen[j] = packed_w_len(tn[j], tk[j])
    kind[head_j] = 1
    tn[head_j] = config.vocab_size
    tk[head_j] = dm
    tlo[head_j] = _span(offsets, head_j, config.vocab_size * dm)
    tlen[head_j] = packed_w_len(config.vocab_size, dm)
    var out = List[List[Float32]]()
    for j in range(nt):
        out.append(host_f32_uninit(tlen[j]))
    # Work units: a flushed span in slices of PREP_SLICE values, a packed
    # weight in runs of its pack units with about PREP_SLICE values each.
    comptime PREP_SLICE = 1 << 17
    var uj = List[Int]()
    var ulo = List[Int]()
    var uhi = List[Int]()
    for j in range(nt):
        if kind[j] == 0:
            var e = 0
            while e < tlen[j]:
                uj.append(j)
                ulo.append(e)
                uhi.append(min(e + PREP_SLICE, tlen[j]))
                e += PREP_SLICE
        else:
            var units = pack_w_units(tn[j])
            var per = max(1, units // max(1, (tn[j] * tk[j] + PREP_SLICE - 1) // PREP_SLICE))
            var u = 0
            while u < units:
                uj.append(j)
                ulo.append(u)
                uhi.append(min(u + per, units))
                u += per
    var ntasks = len(uj)
    var bad = List[Int](length=ntasks, fill=0)
    var pp = rebind[GhrPtr](params.unsafe_ptr())
    var op = out.unsafe_ptr()
    var ujp = uj.unsafe_ptr()
    var ulop = ulo.unsafe_ptr()
    var uhip = uhi.unsafe_ptr()
    var kp = kind.unsafe_ptr()
    var tnp = tn.unsafe_ptr()
    var tkp = tk.unsafe_ptr()
    var tlop = tlo.unsafe_ptr()
    var bp = bad.unsafe_ptr()

    def _prep_task(t: Int) {imm pp, imm op, imm ujp, imm ulop, imm uhip, imm kp, imm tnp, imm tkp, imm tlop, imm bp, imm head_j}:
        var j = ujp.unsafe_load(t)
        var lo = ulop.unsafe_load(t)
        var hi = uhip.unsafe_load(t)
        var src = pp.unsafe_offset(tlop.unsafe_load(j))
        var dst = rebind[GhrPtr](op[j].unsafe_ptr())
        if kp.unsafe_load(j) == 0:
            var i = lo
            while i + HOST_FW_P <= hi:
                var v = src.unsafe_load[width=HOST_FW_P](i)
                if (bitcast[DType.uint32](v) & U32P(0x7F800000)).reduce_max() == UInt32(0x7F800000):
                    bp.unsafe_store(t, 1)
                dst.unsafe_store(i, ftz_lanes(v))
                i += HOST_FW_P
            while i < hi:
                var x = src.unsafe_load(i)
                if (bitcast[DType.uint32](x) & UInt32(0x7F800000)) == UInt32(0x7F800000):
                    bp.unsafe_store(t, 1)
                dst.unsafe_store(i, ftz(x))
                i += 1
            return
        var n = tnp.unsafe_load(j)
        var k = tkp.unsafe_load(j)
        if j != head_j:
            # the screened columns of this unit range (the head is never screened)
            var c0 = lo * (n // max(1, pack_w_units(n)))
            var c1 = min(n, hi * (n // max(1, pack_w_units(n))))
            var i = c0 * k
            var e = c1 * k
            while i + HOST_FW_P <= e:
                if (bitcast[DType.uint32](src.unsafe_load[width=HOST_FW_P](i)) & U32P(0x7F800000)).reduce_max() == UInt32(0x7F800000):
                    bp.unsafe_store(t, 1)
                i += HOST_FW_P
            while i < e:
                if (bitcast[DType.uint32](src.unsafe_load(i)) & UInt32(0x7F800000)) == UInt32(0x7F800000):
                    bp.unsafe_store(t, 1)
                i += 1
        pack_w_range(src, n, k, dst, lo, hi)

    if ntasks == 1:
        _prep_task(0)
    elif ntasks > 1:
        host_parallelize(_prep_task, ntasks)
    var any_bad = False
    for t in range(ntasks):
        if bad[t] != 0:
            any_bad = True
    if any_bad or not all_finite_span(params, 0, offsets[head_j]):
        refuse_nonfinite(String("W"), _slice(params, offsets, 0))
        for layer in range(config.n_layers):
            _ = byte_host_block_weights(params, offsets, layer, dims)
        raise Error("byte LM host: a non-finite parameter that no refusal named")
    return out^


def _threaded_rows(params: List[Float32], inputs: List[Int32], batch: Int, length: Int,
                   config: ByteConfig, workers: Int) raises -> List[Float32]:
    var vocab = config.vocab_size
    var layers = config.n_layers
    var head_index = 1 + 9 * layers
    # The CPU identity gate's sabotage set is built with
    # `-D MOJOLEARN_HOST_SABOTAGE=1` alone. The reference path reaches it
    # through gemm_oracle's descending leaf, but these kernels restate the
    # fold, so without this the byte-lm-host-infer-threaded lane read
    # IDENTICAL x4 on all nine fixtures under that set (M4, 2026-09-15,
    # lane/cpu-training-host-only-lanes). The head's one-leaf reversal is
    # the same arm DEVIATION 2612 already admits.
    var reverse = byte_host_sabotage_compiled() or GEMM_ORACLE_HOST_SABOTAGE
    var held = List[List[List[Float32]]]()
    held.append(byte_host_fast_tensors(params, config))
    var ropes = List[RopeTable]()
    ropes.append(build_rope_table(byte_host_dims(config)))
    var logits = List[Float32](length=batch * length * vocab, fill=Float32(0.0))
    var tasks = workers
    if tasks > batch:
        tasks = batch
    if workers > batch and byte_host_token_split_enabled():
        # the token split: each row over every worker, rows one after another
        var dims_ts = TransformerDims(config.d_model, config.n_heads, config.n_kv, config.head_dim,
                                      config.intermediate, config.length)
        var row_ids_ts = List[Int32](length=length, fill=Int32(0))
        for r in range(batch):
            for t in range(length):
                row_ids_ts[t] = inputs[r * length + t]
            var hidden_ts = hidden_par(held, ropes, row_ids_ts, length, dims_ts, layers, workers)
            var part_ts = head_rows_par(hidden_ts, held, head_index, vocab, config.d_model, length, reverse, workers)
            for q in range(length * vocab):
                logits[r * length * vocab + q] = part_ts[q]
        _ = held^
        _ = ropes^
        return logits^
    var chunk = (batch + tasks - 1) // tasks
    var failed = List[Int](length=tasks, fill=0)
    var op = logits.unsafe_ptr()
    var fp = failed.unsafe_ptr()
    var ip = inputs.unsafe_ptr()
    var tp = held.unsafe_ptr()
    var rp = ropes.unsafe_ptr()
    var c_dm = config.d_model
    var c_heads = config.n_heads
    var c_kv = config.n_kv
    var c_hd = config.head_dim
    var c_ff = config.intermediate
    var c_len = config.length

    def _row_task(c: Int) {imm op, imm fp, imm ip, imm tp, imm rp, imm chunk, imm batch, imm length,
                           imm vocab, imm layers, imm head_index, imm reverse, imm c_dm, imm c_heads,
                           imm c_kv, imm c_hd, imm c_ff, imm c_len}:
        try:
            var lo = c * chunk
            var hi = lo + chunk
            if hi > batch:
                hi = batch
            var dims = TransformerDims(c_dm, c_heads, c_kv, c_hd, c_ff, c_len)
            var row_ids = List[Int32](length=length, fill=Int32(0))
            var part = List[Float32](length=length * vocab, fill=Float32(0.0))
            for r in range(lo, hi):
                for t in range(length):
                    row_ids[t] = ip.unsafe_load(r * length + t)
                var hidden = hidden_fast(tp[], rp[], row_ids, length, dims, layers)
                gemm_w(hidden, tp[][head_index], vocab, c_dm, 0, length, part, reverse)
                for q in range(length * vocab):
                    op.unsafe_store(r * length * vocab + q, part[q])
        except:
            fp.unsafe_store(c, 1)

    if tasks == 1:
        _row_task(0)
    else:
        host_parallelize(_row_task, tasks)
    _ = held^
    _ = ropes^
    for c in range(tasks):
        if failed[c] != 0:
            raise Error("byte LM host: threaded row chunk " + String(c) + " raised")
    return logits^


def byte_host_logits_threaded(params: List[Float32], inputs: List[Int32], batch: Int,
                              length: Int, config: ByteConfig, threads: Int = 0) raises -> List[Float32]:
    """`byte_host_logits` through the DEVIATION 2640 kernels on at most
    `threads` threads (0: one per physical core); same arguments, same bits."""
    _validate_logits_inputs(params, inputs, batch, length, config)
    return _threaded_rows(params, inputs, batch, length, config, byte_host_worker_count(threads))


def byte_host_next_threaded(params: List[Float32], inputs: List[Int32], batch: Int,
                            length: Int, config: ByteConfig,
                            threads: Int = 0) raises -> List[Int32]:
    """Greedy token after each row, through the threaded logits arithmetic.

    Only the last hidden row enters the LM head.  Earlier hidden rows still run
    unchanged because causal attention needs them, but their ``vocab`` logits
    were immediately discarded by ``LanguageModelInference.next_bytes``.
    Skipping those independent output cells changes no fold and no bit in the
    surviving row.
    """
    _validate_logits_inputs(params, inputs, batch, length, config)
    var vocab = config.vocab_size
    var layers = config.n_layers
    var head_index = 1 + 9 * layers
    var reverse = byte_host_sabotage_compiled() or GEMM_ORACLE_HOST_SABOTAGE
    var held = List[List[List[Float32]]]()
    held.append(byte_host_fast_tensors(params, config))
    var ropes = List[RopeTable]()
    ropes.append(build_rope_table(byte_host_dims(config)))
    var out = List[Int32](length=batch, fill=Int32(0))
    var workers_nb = byte_host_worker_count(threads)
    var tasks = workers_nb
    if tasks > batch:
        tasks = batch
    if workers_nb > batch and byte_host_token_split_enabled():
        # the token split (see `block_par`): the hidden rows over every
        # worker, the last row's head on the calling thread
        var dims_ts = TransformerDims(config.d_model, config.n_heads, config.n_kv, config.head_dim,
                                      config.intermediate, config.length)
        var row_ids_ts = List[Int32](length=length, fill=Int32(0))
        var last_ts = List[Float32](length=vocab, fill=Float32(0.0))
        for r in range(batch):
            for t in range(length):
                row_ids_ts[t] = inputs[r * length + t]
            var hidden_ts = hidden_par(held, ropes, row_ids_ts, length, dims_ts, layers, workers_nb)
            gemm_w(hidden_ts, held[0][head_index], vocab, config.d_model, length - 1, length, last_ts, reverse)
            var best = 0
            for j in range(1, vocab):
                if last_ts[j] > last_ts[best]:
                    best = j
            out[r] = Int32(best)
        _ = held^
        _ = ropes^
        return out^
    var chunk = (batch + tasks - 1) // tasks
    var failed = List[Int](length=tasks, fill=0)
    var op = out.unsafe_ptr()
    var fp = failed.unsafe_ptr()
    var ip = inputs.unsafe_ptr()
    var tp = held.unsafe_ptr()
    var rp = ropes.unsafe_ptr()
    var c_dm = config.d_model
    var c_heads = config.n_heads
    var c_kv = config.n_kv
    var c_hd = config.head_dim
    var c_ff = config.intermediate
    var c_len = config.length

    def _row_task(c: Int) {imm op, imm fp, imm ip, imm tp, imm rp, imm chunk, imm batch, imm length,
                           imm vocab, imm layers, imm head_index, imm reverse, imm c_dm, imm c_heads,
                           imm c_kv, imm c_hd, imm c_ff, imm c_len}:
        try:
            var lo = c * chunk
            var hi = min(lo + chunk, batch)
            var dims = TransformerDims(c_dm, c_heads, c_kv, c_hd, c_ff, c_len)
            var row_ids = List[Int32](length=length, fill=Int32(0))
            var last = List[Float32](length=vocab, fill=Float32(0.0))
            for r in range(lo, hi):
                for t in range(length):
                    row_ids[t] = ip.unsafe_load(r * length + t)
                var hidden = hidden_fast(tp[], rp[], row_ids, length, dims, layers)
                gemm_w(hidden, tp[][head_index], vocab, c_dm,
                             length - 1, length, last, reverse)
                var best = 0
                for j in range(1, vocab):
                    if last[j] > last[best]:
                        best = j
                op.unsafe_store(r, Int32(best))
        except:
            fp.unsafe_store(c, 1)

    if tasks == 1:
        _row_task(0)
    else:
        host_parallelize(_row_task, tasks)
    _ = held^
    _ = ropes^
    for c in range(tasks):
        if failed[c] != 0:
            raise Error("byte LM host: threaded next-byte chunk " + String(c) + " raised")
    return out^


def byte_host_loss(params: List[Float32], ids: List[Int32], config: ByteConfig,
                   threaded: Bool = False, threads: Int = 0) raises -> Float32:
    """Mean next-byte cross-entropy of ids `[batch, length + 1]` at the
    configured shape, split exactly as `_byte_forward_loss` splits them.
    The loss runs on the calling thread: the oracle on the reference path,
    `ce_causal_mean_loss_fast` on the threaded path."""
    config.validate()
    var b = config.batch
    var l = config.length
    if len(ids) != b * (l + 1):
        raise Error("byte LM host: loss ids must be [batch, length + 1]")
    var inputs = List[Int32](capacity=b * l)
    var targets = List[Int32](capacity=b * l)
    for bi in range(b):
        for li in range(l):
            inputs.append(ids[bi * (l + 1) + li])
            targets.append(ids[bi * (l + 1) + li + 1])
    if threaded:
        var fast_logits = byte_host_logits_threaded(params, inputs, b, l, config, threads)
        return ce_causal_mean_loss_fast(fast_logits, targets, config.vocab_size)
    var logits = byte_host_logits(params, inputs, b, l, config)
    var st = ce_forward_oracle(logits, targets, CeConfig.causal_lm(config.vocab_size))
    return st.loss[0]
