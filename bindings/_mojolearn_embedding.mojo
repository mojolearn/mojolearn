# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the Embedding lane, profile `mojolearn.identical.embedding.fp32.v1`.

The door of `embedding/checks/embedding_identical.mojo`: the forward gather
(`identical_embedding_forward_into`, seams G1 and G2) and the backward fold
(`identical_embedding_backward_into`, seams E0 to E4, PLAN_SCAN or PLAN_SORT,
contract 6.1 and 6.2, which clause (d) holds bit-identical) with the
two knobs the training binding's `embedding_forward` and
`embedding_backward` (`training/samba_ops.mojo`) do not carry:
`padding_idx` (contract section 8, DEVIATION 1311) and `accumulate`, the
microbatch CARRY (contract 7.4, DEVIATION 1309). Nothing is re-decided.

REFUSED ON THIS HOST, BY NAME, BEFORE ANY UPLOAD (the contract's refusals):
    V <= 0, d < 0, T < 0, T > EMB_MAX_POSITIONS   emb_refuse_shape, contract 3 and 8
    an id below 0 or at or past V                  emb_refuse_ids, contract 8 (never clamped)
    padding_idx outside [0, V) other than -1        here
    a NaN or an infinity in W, dY or a carried dW   refuse_nonfinite, contract 9.1

The nonfinite refusal is the host half of DEVIATION 1506: the device entry
points still do not refuse NaN, so this boundary does it on the host copy.

THE ABI IS THE GP'S: two length-checked lists, orders written out below and
mirrored in `python/mojolearn/embedding.py`. THE GIL is released around the
device call, and nothing inside the `GILReleased` block touches a
`PythonObject`.
"""

from std.os import abort
from std.ffi import _Global
from std.sys.compile import is_defined
from bindings.hostptr import f32_ptr, read_f32, read_i32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import bitcast
from core.device_zero import enqueue_fill

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from core.device_scan import device_first_nonfinite
from core.staged_download import download_f32_into
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojoNeuralEmbeddingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralEmbeddingContextFast"
# The pinned stages of this binding's downloads (core/staged_download.mojo),
# one pool per binding and tier like the context above.
comptime _STAGE_POOL = "MojoDownloadStagesEmbeddingIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoDownloadStagesEmbeddingFast"
from checks.vendor import COMPILED_VENDOR
from embedding.checks.embedding_identical import (
    PLAN_AUTO,
    emb_resolve_plan,
    identical_embedding_backward_into,
    identical_embedding_backward_prerefused_into,
    identical_embedding_forward_into,
    identical_embedding_forward_prerefused_into,
)
from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT
#: lane afn-mlp (2026-10-03): the Apple FAST atomic backward, compiled only
#: under FAST + Apple + `-D MOJOLEARN_AFN_EMB_ATOMIC_BWD`; every other build
#: runs the paths below unchanged (embedding/checks/embedding_fast_apple.mojo).
from embedding.checks.embedding_fast_apple import (
    EMB_ATOMIC_BWD,
    AFN26_EMB_RESIDENT,
    AFN26_EMB_SCRATCH,
    fast_embedding_backward_into,
    fast_embedding_forward_into,
)
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids,
    emb_refuse_shape,
    nonfinite_refusal,
)


def embedding_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL, 2
    DETERMINISTIC."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def embedding_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


# ---- transfers (lane neural-pass28, 2026-10-01) ----------------------------------
# These helpers moved every value through a pinned host buffer ONE SCALAR AT A
# TIME in a host loop, both ways (32 million stores and 32 million reads of
# write-combined memory per call at the board's 32,768 x 1,024 table), after
# copying the caller's table into a list first: forward 270 ms, backward
# 390 ms on the M4 for a gather and a scatter. Now: an upload is one raw
# host-pointer copy (1.6-2.4 ms per 64 MB here); a download is one DMA into
# pinned memory and a read out over host tasks (one thread reads pinned
# memory at ~3 GB/s, four or more at 10 GB/s and more). Copies: no bit moves.
comptime EMB_COPY_TASKS_MAX = 16

# ---- per-call overhead (lane neural-pass138, 2026-10-02) ---------------------------------
# The board's embedding cell (V = 32,768, d = 1,024, T = 32,768; one forward
# and one dense backward) read 208 ms on the L40S against torch eager's
# 2.2 ms. Per call this binding paid, besides the PCIe copies the host
# contract needs (W and dY up, Y and dW down, 128 MB each):
#   * a NEW pinned host buffer of the whole output in every download
#     (`_download_into`: 128 MB pinned, then freed, twice per cell); on CUDA
#     a pinned allocation of that size costs tens of ms (lane neural-pass40
#     measured 64 MB pinned allocations dominating the L40S layernorm cell);
#   * a host scan of all of W (forward) and all of dY (backward) for
#     non-finite values, 128 MB of host reads each.
# Now: downloads go through `core/staged_download.mojo::download_f32_into`
# (two pooled pinned stages, a chunk pipeline, the copy-out over host tasks;
# MOJOLEARN_DOWNLOAD_STAGE=0 is the raw host-pointer copy), and the non-
# finite scan runs ON THE DEVICE over the uploaded buffer
# (`core/device_scan.device_first_nonfinite`); only a hit pays the host
# scan, which raises the oracle's own message for the same first index, so
# the refusal (name, index, order: dY before the carried dW) is unchanged.
# Copies and a read-only scan: no bit moves. (The A/B switch back to the
# per-call pinned download and the host scans is removed, lane
# gap-neural-overhead2.)


# ---- resident table and pooled device scratch (lane idn-embedding, 2026-10-04) ----
# The board's cell (V = 32,768, d = 1,024, T = 32,768) still paid, per
# forward + backward: the whole 128 MB table uploaded and scanned for
# non-finite values in every forward; three zero lists built on the host and
# uploaded as the backward's run scratch, each with its own wait; a device
# +0.0 fill of dW that the seed kernel (E0) then wrote again; the ids read
# back from the device into a new pinned buffer and walked on the host a
# second time in both calls; new 128 MB device buffers for Y, dW and dY in
# every call; and the scan plan's 2 * V * T id reads. IDENTICAL tier only
# (the FAST paths are untouched):
#   EMB_RESIDENT (off: -D MOJOLEARN_EMB_RESIDENT_OFF). The Python layer
#     passes a table token (a new one whenever its `weight` is assigned).
#     The table is uploaded and scanned once per token and its device copy
#     kept in this binding's pool; later forwards gather from that copy.
#     The same bytes in the same kernel: no bit moves. Token 0 (or the
#     three-value params list) is the per-call upload and scan.
#   EMB_DEVICE_SCRATCH (off: -D MOJOLEARN_EMB_DEVICE_SCRATCH_OFF). Y, dW,
#     dY, the ids and the run scratch (counts, run_begin, perm) are device
#     buffers kept in the pool by size and never filled from the host: the
#     seed kernel writes every dW cell of a fresh gradient and both plans
#     write every scratch cell the fold reads. The uploads do not wait (the
#     caller's arrays outlive the call on the one in-order queue), and the
#     ids, refused on the host list before the upload, are not read back.
#     The same kernels on the same values: no bit moves.
#   PLAN_AUTO (embedding_identical.mojo; off: -D MOJOLEARN_EMB_AUTO_SORT_OFF).
comptime _EMB_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# E09-E10: not tested. FAST opts into token-owned tables and fully written
# scratch separately; the IDENTICAL controls retain their existing policy.
comptime EMB_RESIDENT = AFN26_EMB_RESIDENT or (_EMB_IDENTICAL and not (is_defined["MOJOLEARN_EMB_RESIDENT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()))
comptime EMB_DEVICE_SCRATCH = AFN26_EMB_SCRATCH or (_EMB_IDENTICAL and not (is_defined["MOJOLEARN_EMB_DEVICE_SCRATCH_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()))
#: buffers kept per element type (the oldest is retired past these)
comptime EMB_POOL_F32_KEEP = 3
comptime EMB_POOL_I32_KEEP = 6


struct _EmbPool(Defaultable, Movable):
    #: the resident table (0 or 1 entry) and its key
    var w: List[DeviceBuffer[DType.float32]]
    var w_token: Int
    var w_addr: Int
    var w_cells: Int
    #: pooled work buffers and their element counts
    var f: List[DeviceBuffer[DType.float32]]
    var f_n: List[Int]
    var i: List[DeviceBuffer[DType.int32]]
    var i_n: List[Int]

    def __init__(out self):
        self.w = List[DeviceBuffer[DType.float32]]()
        self.w_token = 0
        self.w_addr = 0
        self.w_cells = 0
        self.f = List[DeviceBuffer[DType.float32]]()
        self.f_n = List[Int]()
        self.i = List[DeviceBuffer[DType.int32]]()
        self.i_n = List[Int]()


comptime _EMB_POOL_NAME = "MojoEmbeddingResidentIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoEmbeddingResidentFast"
comptime _EMB_POOL = _Global[StorageType=_EmbPool, name=_EMB_POOL_NAME, init_fn=_EmbPool.__init__]


def _pool_f32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A device buffer of exactly max(n, 1) floats, contents unspecified:
    the pooled one of that size, else a new one."""
    var nb = n if n > 0 else 1
    var pool = _EMB_POOL.get_or_create_ptr()
    for k in range(len(pool[].f_n)):  # small-loop(f_n: pooled buffer sizes, at most EMB_POOL_F32_KEEP): buffer pool lookup, not data
        if pool[].f_n[k] == nb:
            _ = pool[].f_n.pop(k)
            return pool[].f.pop(k)
    return ctx.enqueue_create_buffer[DType.float32](nb)


def _give_f32(var buf: DeviceBuffer[DType.float32], n: Int) raises:
    var pool = _EMB_POOL.get_or_create_ptr()
    if len(pool[].f) >= EMB_POOL_F32_KEEP:
        _ = pool[].f_n.pop(0)
        _ = pool[].f.pop(0)
    pool[].f.append(buf^)
    pool[].f_n.append(n if n > 0 else 1)


def _pool_i32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.int32]:
    """`_pool_f32` for int32."""
    var nb = n if n > 0 else 1
    var pool = _EMB_POOL.get_or_create_ptr()
    for k in range(len(pool[].i_n)):  # small-loop(i_n: pooled buffer sizes, at most EMB_POOL_I32_KEEP): buffer pool lookup, not data
        if pool[].i_n[k] == nb:
            _ = pool[].i_n.pop(k)
            return pool[].i.pop(k)
    return ctx.enqueue_create_buffer[DType.int32](nb)


def _give_i32(var buf: DeviceBuffer[DType.int32], n: Int) raises:
    var pool = _EMB_POOL.get_or_create_ptr()
    if len(pool[].i) >= EMB_POOL_I32_KEEP:
        _ = pool[].i_n.pop(0)
        _ = pool[].i.pop(0)
    pool[].i.append(buf^)
    pool[].i_n.append(n if n > 0 else 1)


def _table_take(
    ctx: DeviceContext,
    token: Int,
    addr: Int,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises -> DeviceBuffer[DType.float32]:
    """The device copy of the caller's table, refused for non-finite values:
    the resident one when (token, address, cells) match, else a new upload
    and scan. The pool gives the buffer up for the call (`_table_give`
    returns it), so a raise in between leaves no resident table behind."""
    comptime if EMB_RESIDENT:
        if token > 0:
            var pool = _EMB_POOL.get_or_create_ptr()
            var hit = (
                len(pool[].w) > 0 and pool[].w_token == token
                and pool[].w_addr == addr and pool[].w_cells == n
            )
            pool[].w_token = 0
            if hit:
                return pool[].w.pop()
            pool[].w.clear()
    var d = _upload_f32_ptr(ctx, wp, n)
    _refuse_nonfinite_device(ctx, String("W"), d, wp, n)
    return d^


def _table_give(var d: DeviceBuffer[DType.float32], token: Int, addr: Int, n: Int) raises:
    comptime if EMB_RESIDENT:
        if token > 0:
            var pool = _EMB_POOL.get_or_create_ptr()
            pool[].w.clear()
            pool[].w.append(d^)
            pool[].w_token = token
            pool[].w_addr = addr
            pool[].w_cells = n
            return
    _ = d^


def embedding_table_release_binding(token_obj: PythonObject) raises -> PythonObject:
    """Drop the resident table of `token` (a layer that is going away or
    whose weight was replaced). Returns 1 when one was dropped."""
    var token = Int(py=token_obj)
    comptime if EMB_RESIDENT:
        var pool = _EMB_POOL.get_or_create_ptr()
        if token > 0 and pool[].w_token == token:
            pool[].w.clear()
            pool[].w_token = 0
            return PythonObject(1)
    return PythonObject(0)


def embedding_resident_binding() raises -> PythonObject:
    """1 when this build keeps the table resident by token (EMB_RESIDENT)."""
    comptime if EMB_RESIDENT:
        return PythonObject(1)
    return PythonObject(0)


def _download_out(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    dst: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises:
    if n > 0:
        download_f32_into[_STAGE_POOL](ctx, buf, n, dst)


def _refuse_nonfinite_device(
    ctx: DeviceContext,
    name: String,
    mut buf: DeviceBuffer[DType.float32],
    p: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises:
    """The non-finite refusal of the caller's `p[0:n]`, scanned on its
    device copy `buf`. A hit re-runs the host refusal on `p` (the same
    bytes), so the error is the oracle's message at the same first index."""
    var idx = device_first_nonfinite(ctx, buf, n)
    if idx < 0:
        return
    nonfinite_refusal(name, idx, p.unsafe_load(idx))
    raise Error(
        String("embedding: the device scan found a non-finite value in ") + name
        + " at flat index " + String(idx) + " that the host scan did not (a scan defect)"
    )


def _upload_f32_ptr(
    ctx: DeviceContext, p: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.float32]:
    """A device copy of the caller's n floats (at least one cell)."""
    var n_buf = n if n > 0 else 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    if n > 0:
        ctx.enqueue_copy(dst_buf=dev.create_sub_buffer[DType.float32](0, n), src_ptr=p)
    else:
        enqueue_fill(ctx, dev, Float32(0.0))
    ctx.synchronize()
    return dev^


def _zero_f32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A device buffer of n +0.0 cells (at least one), filled on the device."""
    var dev = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    enqueue_fill(ctx, dev, Float32(0.0))
    ctx.synchronize()
    return dev^


def _upload_i32(
    ctx: DeviceContext, values: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var n = len(values)
    var n_buf = n if n > 0 else 1
    var dev = ctx.enqueue_create_buffer[DType.int32](n_buf)
    if n > 0:
        ctx.enqueue_copy(dst_buf=dev.create_sub_buffer[DType.int32](0, n), src_ptr=values.unsafe_ptr())
    if n_buf > n:
        var pad = List[Int32](length=1, fill=Int32(0))
        ctx.enqueue_copy(dst_buf=dev.create_sub_buffer[DType.int32](n, 1), src_ptr=pad.unsafe_ptr())
        ctx.synchronize()
        _ = pad^
    ctx.synchronize()
    return dev^


def _refuse_nonfinite_upload(
    name: String, p: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises:
    """The non-finite refusal of `p[0:n]` when no run uploads it (an empty
    output): a device copy and the device scan, as the runs do."""
    if n <= 0:
        return
    var ctx = neural_ctx[_NEURAL_CTX]()
    var d = _upload_f32_ptr(ctx, p, n)
    _refuse_nonfinite_device(ctx, name, d, p, n)
    _ = d^
    _ = ctx^


def _zeros_i32(n: Int) -> List[Int32]:
    return List[Int32](length=n if n > 0 else 1, fill=Int32(0))


def _config(vocab: Int, width: Int, padding_idx: Int, accumulate: Bool) raises -> EmbConfig:
    if padding_idx != EMB_NO_PADDING_IDX and (padding_idx < 0 or padding_idx >= vocab):
        raise Error(
            String("embedding: padding_idx = ")
            + String(padding_idx)
            + " is outside [0, "
            + String(vocab)
            + ") REFUSED (contract 8; -1 means no padding row)"
        )
    return EmbConfig(vocab, width, padding_idx, accumulate)


def _forward_run(
    wp: MutPointer[Float32, MutUntrackedOrigin],
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    yp: MutPointer[Float32, MutUntrackedOrigin],
    token: Int = 0,
    w_addr: Int = 0,
) raises:
    var cells = n_positions * cfg.width
    if cells <= 0:
        return
    var ctx = neural_ctx[_NEURAL_CTX]()
    comptime if EMB_ATOMIC_BWD:
        # lane afn-mlp: the uploads do not wait (the caller's arrays outlive
        # the call on the one in-order queue); the W scan keeps its wait;
        # the gather; the download with the final wait. Two waits, not four.
        # E09: not tested; token/address/extent own the cached table. A
        # refused upload is never returned to the pool as valid state.
        var a_w: DeviceBuffer[DType.float32]
        comptime if AFN26_EMB_RESIDENT:
            a_w = _table_take(ctx, token, w_addr, wp, cfg.vocab * cfg.width)
        else:
            a_w = ctx.enqueue_create_buffer[DType.float32](cfg.vocab * cfg.width)
            ctx.enqueue_copy(dst_buf=a_w, src_ptr=wp)
            _refuse_nonfinite_device(ctx, String("W"), a_w, wp, cfg.vocab * cfg.width)
        # E10: not tested; all output/ID cells are overwritten before use.
        var a_ids: DeviceBuffer[DType.int32]
        var a_y: DeviceBuffer[DType.float32]
        comptime if AFN26_EMB_SCRATCH:
            a_ids = _pool_i32(ctx, n_positions)
            a_y = _pool_f32(ctx, cells)
        else:
            a_ids = ctx.enqueue_create_buffer[DType.int32](n_positions)
            a_y = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.enqueue_copy(dst_buf=a_ids, src_ptr=ids.unsafe_ptr())
        fast_embedding_forward_into(ctx, a_y, a_w, a_ids, n_positions, cfg)
        ctx.enqueue_copy(dst_ptr=yp, src_buf=a_y)
        ctx.synchronize()
        comptime if AFN26_EMB_RESIDENT:
            _table_give(a_w^, token, w_addr, cfg.vocab * cfg.width)
        else:
            _ = a_w^
        comptime if AFN26_EMB_SCRATCH:
            _give_i32(a_ids^, n_positions)
            _give_f32(a_y^, cells)
        else:
            _ = a_ids^
            _ = a_y^
        _ = ctx^
        return
    comptime if EMB_DEVICE_SCRATCH:
        # the table (resident by token, else uploaded and scanned), the ids
        # up without a wait, the gather into a pooled Y, the download (it
        # waits; the ids list is the caller's and outlives this call)
        var w_cells = cfg.vocab * cfg.width
        var r_w = _table_take(ctx, token, w_addr, wp, w_cells)
        var r_ids = _pool_i32(ctx, n_positions)
        ctx.enqueue_copy(dst_buf=r_ids, src_ptr=ids.unsafe_ptr())
        var r_y = _pool_f32(ctx, cells)
        identical_embedding_forward_prerefused_into(ctx, r_y, r_w, r_ids, n_positions, cfg)
        _download_out(ctx, r_y, yp, cells)
        _table_give(r_w^, token, w_addr, w_cells)
        _give_i32(r_ids^, n_positions)
        _give_f32(r_y^, cells)
        _ = ctx^
        return
    var d_w = _table_take(ctx, token, w_addr, wp, cfg.vocab * cfg.width)
    var d_ids = _upload_i32(ctx, ids)
    var d_y = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.synchronize()
    identical_embedding_forward_into(ctx, d_y, d_w, d_ids, n_positions, cfg)
    ctx.synchronize()
    _download_out(ctx, d_y, yp, cells)
    _table_give(d_w^, token, w_addr, cfg.vocab * cfg.width)
    _ = d_ids^
    _ = d_y^
    _ = ctx^


def _backward_run(
    dyp: MutPointer[Float32, MutUntrackedOrigin],
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    dwp: MutPointer[Float32, MutUntrackedOrigin],
    plan: Int,
) raises:
    var cells = cfg.vocab * cfg.width
    if cells <= 0:
        return
    var ctx = neural_ctx[_NEURAL_CTX]()
    comptime if EMB_ATOMIC_BWD:
        # lane afn-mlp: dY up (no wait), its scan (one wait), the carried dW
        # up when accumulating (no wait; else the device seed), the ids up,
        # the scatter-add, the padding row, the download, the final wait.
        # E10: not tested. Fresh gradients still execute the full device
        # seed, and accumulated gradients still upload the caller's dW.
        var a_dw: DeviceBuffer[DType.float32]
        var a_dy: DeviceBuffer[DType.float32]
        var a_ids: DeviceBuffer[DType.int32]
        comptime if AFN26_EMB_SCRATCH:
            a_dw = _pool_f32(ctx, cells)
            a_dy = _pool_f32(ctx, n_positions * cfg.width)
            a_ids = _pool_i32(ctx, n_positions)
        else:
            a_dw = ctx.enqueue_create_buffer[DType.float32](cells)
            a_dy = ctx.enqueue_create_buffer[DType.float32](n_positions * cfg.width)
            a_ids = ctx.enqueue_create_buffer[DType.int32](n_positions if n_positions > 0 else 1)
        if cfg.accumulate:
            ctx.enqueue_copy(dst_buf=a_dw, src_ptr=dwp)
        comptime if AFN26_EMB_SCRATCH:
            # A zero-position call owns one dummy cell but reads no dY.
            if n_positions > 0:
                ctx.enqueue_copy(dst_buf=a_dy, src_ptr=dyp)
        else:
            ctx.enqueue_copy(dst_buf=a_dy, src_ptr=dyp)
        _refuse_nonfinite_device(ctx, String("dY"), a_dy, dyp, n_positions * cfg.width)
        if cfg.accumulate:
            _refuse_nonfinite_device(ctx, String("the carried dW"), a_dw, dwp, cells)
        if n_positions > 0:
            ctx.enqueue_copy(
                dst_buf=a_ids.create_sub_buffer[DType.int32](0, n_positions),
                src_ptr=ids.unsafe_ptr(),
            )
        fast_embedding_backward_into(ctx, a_dw, a_dy, a_ids, n_positions, cfg)
        ctx.enqueue_copy(dst_ptr=dwp, src_buf=a_dw)
        ctx.synchronize()
        comptime if AFN26_EMB_SCRATCH:
            # Pool only after the consumed output's completion boundary.
            _give_f32(a_dw^, cells)
            _give_f32(a_dy^, n_positions * cfg.width)
            _give_i32(a_ids^, n_positions)
        else:
            _ = a_dw^
            _ = a_dy^
            _ = a_ids^
        _ = ctx^
        return
    comptime if EMB_DEVICE_SCRATCH:
        # the carried dW up when accumulating (else the seed kernel writes
        # every cell +0.0), dY up, their scans (dY first: the refusals'
        # order), the ids up, pooled run scratch with no host fill, the
        # launches, the download (it waits)
        var ty = n_positions * cfg.width
        var r_dw = _pool_f32(ctx, cells)
        if cfg.accumulate:
            ctx.enqueue_copy(dst_buf=r_dw, src_ptr=dwp)
        var r_dy = _pool_f32(ctx, ty)
        if ty > 0:
            ctx.enqueue_copy(dst_buf=r_dy, src_ptr=dyp)
        _refuse_nonfinite_device(ctx, String("dY"), r_dy, dyp, ty)
        if cfg.accumulate:
            _refuse_nonfinite_device(ctx, String("the carried dW"), r_dw, dwp, cells)
        var r_ids = _pool_i32(ctx, n_positions)
        if n_positions > 0:
            ctx.enqueue_copy(dst_buf=r_ids, src_ptr=ids.unsafe_ptr())
        var r_counts = _pool_i32(ctx, cfg.vocab)
        var r_begin = _pool_i32(ctx, cfg.vocab + 1)
        var r_perm = _pool_i32(ctx, n_positions)
        identical_embedding_backward_prerefused_into(
            ctx, r_dw, r_dy, r_ids, r_counts, r_begin, r_perm, n_positions, cfg, plan
        )
        _download_out(ctx, r_dw, dwp, cells)
        _give_f32(r_dw^, cells)
        _give_f32(r_dy^, ty)
        _give_i32(r_ids^, n_positions)
        _give_i32(r_counts^, cfg.vocab)
        _give_i32(r_begin^, cfg.vocab + 1)
        _give_i32(r_perm^, n_positions)
        _ = ctx^
        return
    # the carried dW starts from the caller's bits; a fresh one from +0.0
    # (filled on the device: the same +0.0 cells the host list held)
    var d_dw: DeviceBuffer[DType.float32]
    if cfg.accumulate:
        d_dw = _upload_f32_ptr(ctx, dwp, cells)
    else:
        d_dw = _zero_f32(ctx, cells)
    var d_dy = _upload_f32_ptr(ctx, dyp, n_positions * cfg.width)
    # dY first, then the carried dW: the host refusals' order
    _refuse_nonfinite_device(ctx, String("dY"), d_dy, dyp, n_positions * cfg.width)
    if cfg.accumulate:
        _refuse_nonfinite_device(ctx, String("the carried dW"), d_dw, dwp, cells)
    var d_ids = _upload_i32(ctx, ids)
    var counts = _upload_i32(ctx, _zeros_i32(cfg.vocab))
    var run_begin = _upload_i32(ctx, _zeros_i32(cfg.vocab + 1))
    var perm = _upload_i32(ctx, _zeros_i32(n_positions))
    identical_embedding_backward_into(
        ctx, d_dw, d_dy, d_ids, counts, run_begin, perm, n_positions, cfg, plan
    )
    ctx.synchronize()
    _download_out(ctx, d_dw, dwp, cells)
    _ = d_dw^
    _ = d_dy^
    _ = d_ids^
    _ = counts^
    _ = run_begin^
    _ = perm^
    _ = ctx^


def embedding_forward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`Y[t, j] = W[ids[t], j]` through seams G1 and G2. Returns `T * d`.

    `addrs`, in this exact order:

        0  weight     V * d float32, row-major, read
        1  ids        T int32, read
        2  y_out      T * d float32, WRITTEN

    `params`, in this exact order:

        0  V
        1  d
        2  T
        3  token      optional; a positive table token keeps the table's
                      device copy resident for the next forward with the
                      same token (EMB_RESIDENT); 0 or absent uploads it
    """
    if len(addrs) != 3:
        raise Error(
            "embedding_forward: addrs must contain 3 addresses (weight, ids,"
            " y_out), got " + String(len(addrs))
        )
    if len(params) != 3 and len(params) != 4:
        raise Error(
            "embedding_forward: params must contain 3 or 4 values (V, d, T[,"
            " token]), got " + String(len(params))
        )
    var vocab = Int(py=params[0])
    var width = Int(py=params[1])
    var n_positions = Int(py=params[2])
    var token = 0
    if len(params) == 4:
        token = Int(py=params[3])
    var cfg = _config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n_positions)
    var w_addr = Int(py=addrs[0])
    var wp = f32_ptr(w_addr)
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    var yp = f32_ptr(Int(py=addrs[2]))
    # the W scan runs on the device copy unless there is nothing to gather
    # (then nothing is uploaded)
    var no_run = n_positions * width <= 0
    with GILReleased(Python()):
        if no_run:
            _refuse_nonfinite_upload(String("W"), wp, vocab * width)
        _forward_run(wp, ids, n_positions, cfg, yp, token, w_addr)
    return PythonObject(n_positions * width)


def embedding_backward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`dW` through seams E0 to E4 (contract 5.1's serial ascending fold). Returns `V * d`.

    `addrs`, in this exact order:

        0  dy         T * d float32, row-major, read
        1  ids        T int32, read
        2  dw         V * d float32, WRITTEN; READ FIRST when accumulate is 1
                      (the carried accumulator, contract 7.4)

    `params`, in this exact order:

        0  V
        1  d
        2  T
        3  padding_idx   -1 for none; its row is +0.0 STORED (contract 8)
        4  accumulate    0 fresh (+0.0 fill), 1 carry the dw buffer's bits
        5  plan          optional; 0 PLAN_SCAN (the default), 1 PLAN_SORT,
                         2 PLAN_AUTO (PLAN_SORT at large V * T, else
                         PLAN_SCAN; embedding_identical.mojo). The run
                         structure's execution plan (contract 6); both
                         give the same counts, perm and dW bits
    """
    if len(addrs) != 3:
        raise Error(
            "embedding_backward: addrs must contain 3 addresses (dy, ids, dw),"
            " got " + String(len(addrs))
        )
    if len(params) != 5 and len(params) != 6:
        raise Error(
            "embedding_backward: params must contain 5 or 6 values (V, d, T,"
            " padding_idx, accumulate[, plan]), got " + String(len(params))
        )
    var vocab = Int(py=params[0])
    var width = Int(py=params[1])
    var n_positions = Int(py=params[2])
    var padding_idx = Int(py=params[3])
    var acc_code = Int(py=params[4])
    var plan = PLAN_SCAN
    if len(params) == 6:
        var plan_code = Int(py=params[5])
        if plan_code != PLAN_SCAN and plan_code != PLAN_SORT and plan_code != PLAN_AUTO:
            raise Error(
                String("embedding_backward: plan must be 0 (PLAN_SCAN), 1")
                + " (PLAN_SORT) or 2 (PLAN_AUTO), got "
                + String(plan_code)
            )
        plan = emb_resolve_plan(plan_code, vocab, n_positions)
    if acc_code != 0 and acc_code != 1:
        raise Error(
            String("embedding_backward: accumulate must be 0 or 1, got ")
            + String(acc_code)
        )
    var cfg = _config(vocab, width, padding_idx, acc_code == 1)
    emb_refuse_shape(cfg, n_positions)
    var dyp = f32_ptr(Int(py=addrs[0]))
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    var dwp = f32_ptr(Int(py=addrs[2]))
    var no_run = vocab * width <= 0
    with GILReleased(Python()):
        if no_run:
            _refuse_nonfinite_upload(String("dY"), dyp, n_positions * width)
            if cfg.accumulate:
                _refuse_nonfinite_upload(String("the carried dW"), dwp, vocab * width)
        _backward_run(dyp, ids, n_positions, cfg, dwp, plan)
    return PythonObject(vocab * width)


@export
def PyInit__mojolearn_embedding() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_embedding")
        m.def_function[embedding_vendor_binding]("embedding_vendor")
        m.def_function[embedding_numeric_mode_binding]("embedding_numeric_mode")
        m.def_function[embedding_forward_binding]("embedding_forward")
        m.def_function[embedding_backward_binding]("embedding_backward")
        m.def_function[embedding_table_release_binding]("embedding_table_release")
        m.def_function[embedding_resident_binding]("embedding_resident")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_embedding: ", e))
