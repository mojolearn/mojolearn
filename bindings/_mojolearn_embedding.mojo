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
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE
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


def _upload_f32(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """A device copy of at least one cell (a zero-length buffer is not portable)."""
    var n = len(values)
    var n_buf = n if n > 0 else 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _upload_i32(
    ctx: DeviceContext, values: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var n = len(values)
    var n_buf = n if n > 0 else 1
    var dev = ctx.enqueue_create_buffer[DType.int32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.int32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


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
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    for i in range(n):
        dst.unsafe_store(i, host.unsafe_ptr().unsafe_load(i))
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
    weight: List[Float32],
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    yp: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var cells = n_positions * cfg.width
    if cells <= 0:
        return
    var ctx = DeviceContext()
    var d_w = _upload_f32(ctx, weight)
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
    dy: List[Float32],
    ids: List[Int32],
    dw_start: List[Float32],
    n_positions: Int,
    cfg: EmbConfig,
    dwp: MutPointer[Float32, MutUntrackedOrigin],
    plan: Int,
) raises:
    var cells = cfg.vocab * cfg.width
    if cells <= 0:
        return
    var ctx = DeviceContext()
    var d_dw = _upload_f32(ctx, dw_start)
    var d_dy = _upload_f32(ctx, dy)
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
    var weight = read_f32(Int(py=addrs[0]), vocab * width)
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("W"), weight)
    var yp = f32_ptr(Int(py=addrs[2]))
    with GILReleased(Python()):
        _forward_run(weight, ids, n_positions, cfg, yp)
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
    var dy = read_f32(Int(py=addrs[0]), n_positions * width)
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("dY"), dy)
    var dwp = f32_ptr(Int(py=addrs[2]))
    var dw_start: List[Float32]
    if cfg.accumulate:
        dw_start = read_f32(Int(py=addrs[2]), vocab * width)
        refuse_nonfinite(String("the carried dW"), dw_start)
    else:
        # Contents are irrelevant: the fresh path's seed kernel STORES +0.0 in
        # every cell (contract 5.5), which the check's poisoned buffer gates.
        dw_start = List[Float32](length=vocab * width, fill=Float32(0.0))
    with GILReleased(Python()):
        _backward_run(dy, ids, dw_start, n_positions, cfg, dwp, plan)
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
