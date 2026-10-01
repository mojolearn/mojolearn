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
from bindings.hostptr import f32_ptr, read_f32, read_i32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import bitcast, memcpy
from core.device_zero import enqueue_fill
from core.host_lanes import HOST_FW, U32V
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojoNeuralEmbeddingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralEmbeddingContextFast"
from checks.vendor import COMPILED_VENDOR
from embedding.checks.embedding_identical import (
    identical_embedding_backward_into,
    identical_embedding_forward_into,
)
from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids,
    emb_refuse_shape,
    refuse_nonfinite,
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


def _parallel_copy_out(dst: MutPointer[Float32, MutUntrackedOrigin], src: MutPointer[Float32, MutUntrackedOrigin], n: Int):
    """`memcpy(dst, src, n)` in contiguous chunks over host tasks: the read of pinned memory."""
    var tasks = host_predict_task_count(1 << 30)
    if tasks > EMB_COPY_TASKS_MAX:
        tasks = EMB_COPY_TASKS_MAX
    if n < (1 << 18) or tasks <= 1:
        memcpy(dest=dst, src=src, count=n)
        return
    var chunk = (n + tasks - 1) // tasks

    def _piece(t: Int) {imm dst, imm src, imm n, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, n)
        if hi > lo:
            memcpy(dest=dst + lo, src=src + lo, count=hi - lo)

    host_parallelize(_piece, tasks)


def _all_finite_ptr(p: MutPointer[Float32, MutUntrackedOrigin], n: Int) -> Bool:
    """`core.host_lanes.all_finite`'s bit test over the caller's n floats, the
    ranges over host tasks: it reads bits and computes nothing."""
    var tasks = host_predict_task_count(1 << 30)
    if tasks > EMB_COPY_TASKS_MAX:
        tasks = EMB_COPY_TASKS_MAX
    if n < (1 << 18) or tasks <= 1:
        tasks = 1
    var chunk = (n + tasks - 1) // tasks
    var bad = List[Int32](length=tasks, fill=Int32(0))
    var bp = bad.unsafe_ptr()

    def _scan(t: Int) {imm p, imm bp, imm n, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, n)
        var i = lo
        var acc = U32V(0)
        var expm = U32V(0x7F800000)
        while i + HOST_FW <= hi:
            var e = bitcast[DType.uint32](p.unsafe_load[width=HOST_FW](i)) & expm
            acc = acc | e.eq(expm).select(U32V(1), U32V(0))
            i += HOST_FW
        var found = acc.reduce_or() != UInt32(0)
        while i < hi and not found:
            if (bitcast[DType.uint32](p.unsafe_load(i)) & UInt32(0x7F800000)) == UInt32(0x7F800000):
                found = True
            i += 1
        if found:
            bp.unsafe_store(t, Int32(1))

    if tasks <= 1:
        _scan(0)
    else:
        host_parallelize(_scan, tasks)
    var ok = True
    for t in range(tasks):
        if bad[t] != Int32(0):
            ok = False
    _ = bad^
    return ok


def _refuse_nonfinite_ptr(name: String, p: MutPointer[Float32, MutUntrackedOrigin], n: Int) raises:
    """`refuse_nonfinite` over the caller's floats: the parallel bit test
    first; only a refusal pays the list copy, so the message and the index
    it names are the oracle's own."""
    if _all_finite_ptr(p, n):
        return
    var values = List[Float32](length=n if n > 0 else 1, fill=Float32(0.0))
    if n > 0:
        memcpy(dest=values.unsafe_ptr(), src=p, count=n)
    refuse_nonfinite(name, values)


def _zeros_i32(n: Int) -> List[Int32]:
    return List[Int32](length=n if n > 0 else 1, fill=Int32(0))


def _download_into(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    dst: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) raises:
    if n <= 0:
        return
    var host = ctx.enqueue_create_host_buffer[DType.float32](len(buf))
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=host, src_buf=buf)
    ctx.synchronize()
    _parallel_copy_out(dst, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(host.unsafe_ptr())), n)
    _ = host^


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
) raises:
    var cells = n_positions * cfg.width
    if cells <= 0:
        return
    var ctx = neural_ctx[_NEURAL_CTX]()
    var d_w = _upload_f32_ptr(ctx, wp, cfg.vocab * cfg.width)
    var d_ids = _upload_i32(ctx, ids)
    var d_y = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.synchronize()
    identical_embedding_forward_into(ctx, d_y, d_w, d_ids, n_positions, cfg)
    ctx.synchronize()
    _download_into(ctx, d_y, yp, cells)
    _ = d_w^
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
    # the carried dW starts from the caller's bits; a fresh one from +0.0
    # (filled on the device: the same +0.0 cells the host list held)
    var d_dw: DeviceBuffer[DType.float32]
    if cfg.accumulate:
        d_dw = _upload_f32_ptr(ctx, dwp, cells)
    else:
        d_dw = _zero_f32(ctx, cells)
    var d_dy = _upload_f32_ptr(ctx, dyp, n_positions * cfg.width)
    var d_ids = _upload_i32(ctx, ids)
    var counts = _upload_i32(ctx, _zeros_i32(cfg.vocab))
    var run_begin = _upload_i32(ctx, _zeros_i32(cfg.vocab + 1))
    var perm = _upload_i32(ctx, _zeros_i32(n_positions))
    identical_embedding_backward_into(
        ctx, d_dw, d_dy, d_ids, counts, run_begin, perm, n_positions, cfg, plan
    )
    ctx.synchronize()
    _download_into(ctx, d_dw, dwp, cells)
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
    """
    if len(addrs) != 3:
        raise Error(
            "embedding_forward: addrs must contain 3 addresses (weight, ids,"
            " y_out), got " + String(len(addrs))
        )
    if len(params) != 3:
        raise Error(
            "embedding_forward: params must contain 3 values (V, d, T), got "
            + String(len(params))
        )
    var vocab = Int(py=params[0])
    var width = Int(py=params[1])
    var n_positions = Int(py=params[2])
    var cfg = _config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n_positions)
    var wp = f32_ptr(Int(py=addrs[0]))
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    var yp = f32_ptr(Int(py=addrs[2]))
    with GILReleased(Python()):
        _refuse_nonfinite_ptr(String("W"), wp, vocab * width)
        _forward_run(wp, ids, n_positions, cfg, yp)
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
        5  plan          optional; 0 PLAN_SCAN (the default), 1 PLAN_SORT.
                         The run structure's execution plan (contract 6);
                         both give the same counts, perm and dW bits
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
        if plan_code != PLAN_SCAN and plan_code != PLAN_SORT:
            raise Error(
                String("embedding_backward: plan must be 0 (PLAN_SCAN) or 1")
                + " (PLAN_SORT), got "
                + String(plan_code)
            )
        plan = plan_code
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
    with GILReleased(Python()):
        _refuse_nonfinite_ptr(String("dY"), dyp, n_positions * width)
        if cfg.accumulate:
            _refuse_nonfinite_ptr(String("the carried dW"), dwp, vocab * width)
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
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_embedding: ", e))
