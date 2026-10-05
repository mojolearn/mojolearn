# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST candidates for the Samba stack's training-binding ops (lane
afn-samba, 2026-10-03).

Everything in this file is reachable ONLY when
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` AND
one of the lane's build defines is set; `training/samba_ops.mojo` routes
its ops here through one guarded early return per op and keeps its own
bodies byte for byte main's for IDENTICAL and for every other vendor.

The defines (each read once, at compile time; default OFF):

  MOJOLEARN_AFN_SAMBA_FUSE          three fused binding entries: the final
                                    norm + head in one call (the forward),
                                    the final norm + head + loss + head
                                    backward + norm backward in one call
                                    (the train step's tail), and the
                                    embedding backward + tied pair add in
                                    one call. Same kernels, same operands,
                                    same order; `hn`, `dhn`, the logits'
                                    gradient and the tied pair never cross
                                    the bus, and the per-op waits between
                                    them go.
  MOJOLEARN_AFN_SAMBA_ARENA         every float scratch of an op is a view
                                    of one device arena chunk
                                    (core/device_arena.mojo) instead of a
                                    fresh Metal buffer; the arena is opened
                                    at the op's start and released after its
                                    wait.
  MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT  the non-finite refusal of inputs,
                                    weights and logits runs as one flag
                                    kernel per operand on the device and is
                                    read back with the op's one wait; the
                                    logits never come down for the host
                                    scan. The refusal still fires by name.
  MOJOLEARN_AFN_SAMBA_EMB_ATOMIC    the embedding backward is one f32
                                    atomic scatter over (position, column)
                                    in free order, after a zero fill, in
                                    place of the run-sorted fold (counts,
                                    run begins, permutation, fold; the id
                                    refusal's download and two waits).
  MOJOLEARN_AFN_SAMBA_ALL           all of the above.
  MOJOLEARN_AFN_SAMBA_HEAD_GEMM     (wave 2, lane w2-epi; also on under
                                    MOJOLEARN_AFN_EPI_ALL, NOT under
                                    _SAMBA_ALL) the tied head GEMM and its
                                    two backward GEMMs in the fused entries
                                    and the resident head loss run on the
                                    gemm lane's FAST matrix-unit kernel
                                    (gemm/afn_apple_fast.mojo) instead of
                                    `identical_gemm_into` (no workspaces);
                                    a product whose tiles cover fewer than
                                    2 x AFN_GEMM_CORES blocks (the head's
                                    dW at the board shape: 24 tiles, k =
                                    1024 tokens) splits k over grid.y into
                                    a zeroed output with f32 atomic adds.
                                    Reached only through the entries above,
                                    so an A/B pairs it with _SAMBA_FUSE.

FAST promises quality, never bits: the fold orders that change here are the
embedding scatter (free order) and the tied pair add's placement; every
accumulate stays f32.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from core.device_arena import (
    arena_active,
    arena_begin,
    arena_end,
    arena_release,
    arena_take,
)
from core.neural_context import neural_ctx
from embedding.checks.embedding_identical import (
    identical_embedding_backward_into,
    identical_embedding_forward_into,
)
from embedding.checks.embedding_oracle import EmbConfig
#: lane w2-epi (MOJOLEARN_AFN_SAMBA_HEAD_GEMM): the FAST matrix-unit kernel's
#: launcher and policies, instantiated only under that switch.
from gemm.afn_apple_fast import (
    AFN_EPI_NONE,
    AFN_GEMM_CORES,
    AFN_GEMM_KB,
    AFN_GEMM_SPLIT_MAX,
    AFN_GEMM_SPLIT_MIN_STEPS,
    AFN_ZERO_TPB,
    _afn_launch_tile,
    _afn_strides,
    afn_gemm_tile,
    afn_gemm_tile_count,
    afn_zero_kernel,
)
from gemm.checks.gemm_backward import (
    BWD_DC_LEFT,
    gemm_backward_a_call,
    gemm_backward_b_call,
    identical_gemm_backward_a_into,
    identical_gemm_backward_a_workspace_max_floats,
    identical_gemm_backward_b_into,
    identical_gemm_backward_b_workspace_max_floats,
)
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NN, OP_NT, OP_TN
from training.checks.loss_contract import (
    CeConfig,
    ce_count,
    ce_refuse_inputs,
    ce_refuse_shape,
    ce_refuse_targets,
)
from training.estimator import identical_ce_admit_call, identical_ce_loss_resident
from transformer.checks.transformer_backward import bwd_rms_norm
from transformer.impl.llama.modeling_llama import llama_rms_norm


# ===========================================================================
# THE GUARD AND THE DEFINES
# ===========================================================================

comptime AFN_SAMBA_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)
comptime AFN_SAMBA_ALL = AFN_SAMBA_APPLE_FAST and is_defined["MOJOLEARN_AFN_SAMBA_ALL"]()
comptime AFN_SAMBA_FUSE = AFN_SAMBA_ALL or (
    AFN_SAMBA_APPLE_FAST and is_defined["MOJOLEARN_AFN_SAMBA_FUSE"]()
)
comptime AFN_SAMBA_ARENA = AFN_SAMBA_ALL or (
    AFN_SAMBA_APPLE_FAST and is_defined["MOJOLEARN_AFN_SAMBA_ARENA"]()
)
comptime AFN_SAMBA_DEVICE_ADMIT = AFN_SAMBA_ALL or (
    AFN_SAMBA_APPLE_FAST and is_defined["MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT"]()
)
comptime AFN_SAMBA_EMB_ATOMIC = AFN_SAMBA_ALL or (
    AFN_SAMBA_APPLE_FAST and is_defined["MOJOLEARN_AFN_SAMBA_EMB_ATOMIC"]()
)
#: lane w2-epi (wave 2): the head GEMMs on the FAST matrix-unit kernel. The
#: gemm kernel's own guard adds the Apple column (never the CPU column).
comptime AFN_SAMBA_HEAD_GEMM = (
    AFN_SAMBA_APPLE_FAST
    and TARGET_COLUMN == COLUMN_APPLE
    and (
        is_defined["MOJOLEARN_AFN_SAMBA_HEAD_GEMM"]()
        or is_defined["MOJOLEARN_AFN_EPI_ALL"]()
    )
)
#: the standalone ops of samba_ops.mojo route here when any of these is on
comptime AFN_SAMBA_OPS = (
    AFN_SAMBA_ARENA or AFN_SAMBA_DEVICE_ADMIT or AFN_SAMBA_EMB_ATOMIC
)
#: whether an op carries a device flag buffer (admit scans or the id check)
comptime AFN_SAMBA_FLAGS = AFN_SAMBA_DEVICE_ADMIT or AFN_SAMBA_EMB_ATOMIC

comptime AFN_TPB = 256
comptime AFN_FLAG_SLOTS = 8


def _grid(n: Int) -> Int:
    var g = (n + AFN_TPB - 1) // AFN_TPB
    if g < 1:
        return 1
    return g


def _refuse_nonfinite_host(
    name: String, ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises:
    """samba_ops.mojo's host scan, for the arms that keep it."""
    for i in range(n):
        if not isfinite(ptr.unsafe_load(i)):
            raise Error(
                "mojolearn samba ops: non-finite " + name + " at flat index "
                + String(i)
            )


# ===========================================================================
# ARENA SCRATCH (MOJOLEARN_AFN_SAMBA_ARENA)
# ===========================================================================


struct AfnArena(Movable):
    """One op's arena bracket: opened at the op's start, released after its
    wait. With the define off (or arenas off) it does nothing and every
    scratch is a fresh buffer, as in samba_ops.mojo."""

    var id: Int

    def __init__(out self) raises:
        self.id = -1
        comptime if AFN_SAMBA_ARENA:
            if not arena_active():
                self.id = arena_begin()

    def __deinit__(deinit self):
        """A raise between the open and `close` still ends and releases
        the arena, so no arena is ever left active."""
        comptime if AFN_SAMBA_ARENA:
            if self.id >= 0:
                try:
                    arena_end(self.id)
                    arena_release(self.id)
                except:
                    pass

    def close(mut self) raises:
        comptime if AFN_SAMBA_ARENA:
            if self.id >= 0:
                arena_end(self.id)
                arena_release(self.id)
                self.id = -1


def afn_scratch_f32(
    own: Bool, ctx: DeviceContext, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """A scratch of `n` floats: a view of the op's own arena when `own`
    (the op's AfnArena opened it), else a fresh buffer. An arena the op
    found already open (a session's, or one a raise left open) never
    receives the op's views."""
    comptime if AFN_SAMBA_ARENA:
        if own and arena_active():
            return arena_take(ctx, n)
    return ctx.enqueue_create_buffer[DType.float32](n)


def afn_upload_f32(
    own: Bool,
    ctx: DeviceContext, ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.float32]:
    var buf = afn_scratch_f32(own, ctx, n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=ptr)
    return buf^


def afn_upload_i32(
    ctx: DeviceContext, ptr: MutPointer[Int32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=ptr)
    return buf^


# ===========================================================================
# DEVICE ADMISSION (MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT) AND THE FLAG BUFFER
# ===========================================================================


def afn_nonfinite_flag_kernel(
    flag: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    slot_in: Int32,
):
    """`flag[slot] = 1` when any `src[i]` is a NaN or an infinity; one thread
    per cell, a benign race of equal stores."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if not isfinite(src.unsafe_load(i)):
        flag.unsafe_store(Int(slot_in), Int32(1))


struct AfnAdmit(Movable):
    """An op's device refusals: up to AFN_FLAG_SLOTS named flags in one
    int32 buffer, zeroed at construction, read back with the op's wait
    (`finish` before the wait, `check` after it). Every method is a no-op
    unless a define that needs flags is on."""

    var flags: List[DeviceBuffer[DType.int32]]
    var names: List[String]
    var host: List[Int32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.flags = List[DeviceBuffer[DType.int32]]()
        self.names = List[String]()
        self.host = List[Int32](length=AFN_FLAG_SLOTS, fill=Int32(0))
        comptime if AFN_SAMBA_FLAGS:
            var f = ctx.enqueue_create_buffer[DType.int32](AFN_FLAG_SLOTS)
            f.enqueue_fill(Int32(0))
            self.flags.append(f^)

    def slot(mut self, name: String) raises -> Int:
        """Reserve the next flag for `name` (its refusal message)."""
        if len(self.names) >= AFN_FLAG_SLOTS:
            raise Error("mojolearn samba afn: more admit flags than slots")
        self.names.append(name)
        return len(self.names) - 1

    def scan(
        mut self,
        ctx: DeviceContext,
        mut buf: DeviceBuffer[DType.float32],
        n: Int,
        name: String,
    ) raises:
        """The non-finite scan of `buf[0:n]` on the device, under the admit
        define; nothing otherwise (the caller then kept the host scan)."""
        comptime if AFN_SAMBA_DEVICE_ADMIT:
            var s = self.slot("non-finite " + name)
            ctx.enqueue_function[afn_nonfinite_flag_kernel](
                self.flags[0].unsafe_ptr(), buf.unsafe_ptr(), Int32(n), Int32(s),
                grid_dim=(_grid(n), 1, 1),
                block_dim=(AFN_TPB, 1, 1),
            )

    def finish(mut self, ctx: DeviceContext) raises:
        """Enqueue the flags' readback; the caller's wait follows."""
        comptime if AFN_SAMBA_FLAGS:
            if len(self.flags) > 0:
                ctx.enqueue_copy(dst_ptr=self.host.unsafe_ptr(), src_buf=self.flags[0])

    def check(self) raises:
        """After the wait: the first raised flag, by name."""
        comptime if AFN_SAMBA_FLAGS:
            for i in range(len(self.names)):  # small-loop(names: admission flags of one op, a handful): first raised flag by name
                if self.host[i] != Int32(0):
                    raise Error("mojolearn samba ops: " + self.names[i] + " (device admit)")


# ===========================================================================
# THE EMBEDDING BACKWARD (MOJOLEARN_AFN_SAMBA_EMB_ATOMIC) AND THE PAIR ADD
# ===========================================================================


def afn_emb_scatter_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    vocab_in: Int32,
    width_in: Int32,
    slot_in: Int32,
):
    """`dw[ids[t], e] += dy[t, e]` by f32 atomic add, one thread per
    (position, column), free order. An id outside [0, vocab) raises the
    flag and contributes nothing (contract 8: never clamped)."""
    var width = Int(width_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in) * width:
        return
    var t = i // width
    var e = i - t * width
    var v = Int(ids.unsafe_load(t))
    if v < 0 or v >= Int(vocab_in):
        flag.unsafe_store(Int(slot_in), Int32(1))
        return
    _ = Atomic.fetch_add(dw.unsafe_offset(v * width + e), dy.unsafe_load(i))


def afn_pair_add_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`dst[i] = dst[i] + b[i]`: the tied pair add in place (embedding
    gradient first, head gradient second), one thread per cell."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, dst.unsafe_load(i) + b.unsafe_load(i))


def afn_embedding_backward_into(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    mut admit: AfnAdmit,
    mut ints: List[DeviceBuffer[DType.int32]],
    n_positions: Int,
    vocab: Int,
    width: Int,
    seed: Bool,
) raises:
    """The embedding gradient into `dw`: the atomic scatter under
    MOJOLEARN_AFN_SAMBA_EMB_ATOMIC (`seed` zero-fills `dw` first; False
    accumulates onto what `dw` holds), else the identical run-sorted fold
    whose int32 scratch is appended to `ints` (the caller keeps it alive
    past its wait)."""
    var cfg = EmbConfig.llama(vocab, width)
    comptime if AFN_SAMBA_EMB_ATOMIC:
        if seed:
            dw.enqueue_fill(Float32(0.0))
        var s = admit.slot(
            "embedding id outside [0, " + String(vocab) + ") REFUSED (contract 8; never clamped)"
        )
        var cells = n_positions * width
        ctx.enqueue_function[afn_emb_scatter_kernel](
            dw.unsafe_ptr(), dy.unsafe_ptr(), ids.unsafe_ptr(),
            admit.flags[0].unsafe_ptr(),
            Int32(n_positions), Int32(vocab), Int32(width), Int32(s),
            grid_dim=(_grid(cells), 1, 1),
            block_dim=(AFN_TPB, 1, 1),
        )
    else:
        var i0 = ctx.enqueue_create_buffer[DType.int32](vocab)
        var i1 = ctx.enqueue_create_buffer[DType.int32](vocab + 1)
        var i2 = ctx.enqueue_create_buffer[DType.int32](n_positions)
        identical_embedding_backward_into(
            ctx, dw, dy, ids, i0, i1, i2, n_positions, cfg
        )
        ints.append(i0^)
        ints.append(i1^)
        ints.append(i2^)


# ===========================================================================
# THE STANDALONE OPS (samba_ops.mojo routes here under AFN_SAMBA_OPS)
# ===========================================================================


def afn_embedding_forward_host(
    ctx: DeviceContext,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_positions: Int,
    vocab: Int,
    width: Int,
) raises -> Int:
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    comptime if not AFN_SAMBA_DEVICE_ADMIT:
        _refuse_nonfinite_host("embedding weight", w_ptr, vocab * width)
    var arena = AfnArena()
    var own = arena.id >= 0
    var admit = AfnAdmit(ctx)
    var cells = n_positions * width
    var w = afn_upload_f32(own, ctx, w_ptr, vocab * width)
    admit.scan(ctx, w, vocab * width, "embedding weight")
    var ids = afn_upload_i32(ctx, ids_ptr, n_positions)
    var y = afn_scratch_f32(own, ctx, cells)
    var cfg = EmbConfig.llama(vocab, width)
    identical_embedding_forward_into(ctx, y, w, ids, n_positions, cfg)
    ctx.enqueue_copy(dst_ptr=y_ptr, src_buf=y)
    admit.finish(ctx)
    ctx.synchronize()
    arena.close()
    admit.check()
    _ = w^
    _ = ids^
    _ = y^
    _ = admit^
    return cells


def afn_embedding_backward_host(
    ctx: DeviceContext,
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dy_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_positions: Int,
    vocab: Int,
    width: Int,
) raises -> Int:
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    comptime if not AFN_SAMBA_DEVICE_ADMIT:
        _refuse_nonfinite_host("embedding upstream gradient", dy_ptr, n_positions * width)
    var arena = AfnArena()
    var own = arena.id >= 0
    var admit = AfnAdmit(ctx)
    var cells = vocab * width
    var dy = afn_upload_f32(own, ctx, dy_ptr, n_positions * width)
    admit.scan(ctx, dy, n_positions * width, "embedding upstream gradient")
    var ids = afn_upload_i32(ctx, ids_ptr, n_positions)
    var dw = afn_scratch_f32(own, ctx, cells)
    var ints = List[DeviceBuffer[DType.int32]]()
    afn_embedding_backward_into(
        ctx, dw, dy, ids, admit, ints, n_positions, vocab, width, True
    )
    ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
    admit.finish(ctx)
    ctx.synchronize()
    arena.close()
    admit.check()
    _ = dy^
    _ = ids^
    _ = dw^
    _ = ints^
    _ = admit^
    return cells


def _refuse_norm_shape(m: Int, dm: Int, eps: Float32) raises:
    if m < 1 or dm < 1:
        raise Error("mojolearn samba ops: rms_norm shape must be positive")
    if not isfinite(eps) or eps < Float32(0.0):
        raise Error("mojolearn samba ops: rms_norm eps must be finite and >= 0")


def afn_rms_norm_forward_host(
    ctx: DeviceContext,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    dm: Int,
    eps: Float32,
) raises -> Int:
    _refuse_norm_shape(m, dm, eps)
    var cells = m * dm
    comptime if not AFN_SAMBA_DEVICE_ADMIT:
        _refuse_nonfinite_host("rms_norm input", x_ptr, cells)
        _refuse_nonfinite_host("rms_norm weight", w_ptr, dm)
    var arena = AfnArena()
    var own = arena.id >= 0
    var admit = AfnAdmit(ctx)
    var x = afn_upload_f32(own, ctx, x_ptr, cells)
    var w = afn_upload_f32(own, ctx, w_ptr, dm)
    admit.scan(ctx, x, cells, "rms_norm input")
    admit.scan(ctx, w, dm, "rms_norm weight")
    var y = afn_scratch_f32(own, ctx, cells)
    var sumsq = afn_scratch_f32(own, ctx, m)
    llama_rms_norm(ctx, sumsq, y, x, w, m, dm, eps)
    ctx.enqueue_copy(dst_ptr=y_ptr, src_buf=y)
    admit.finish(ctx)
    ctx.synchronize()
    arena.close()
    admit.check()
    _ = x^
    _ = w^
    _ = y^
    _ = sumsq^
    _ = admit^
    return cells


def _rms_norm_backward_resident(
    own: Bool,
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut sumsq: DeviceBuffer[DType.float32],
    mut keep: List[DeviceBuffer[DType.float32]],
    m: Int,
    dm: Int,
    eps: Float32,
) raises:
    """`bwd_rms_norm[0]` on device-resident operands with `sumsq` already
    the forward's; its scratch is appended to `keep` (alive past the
    caller's wait). The ones vector is a device fill, not a host loop."""
    var cells = m * dm
    var dot_out = afn_scratch_f32(own, ctx, m)
    var dh = afn_scratch_f32(own, ctx, cells)
    var dprod = afn_scratch_f32(own, ctx, cells)
    var rstd = afn_scratch_f32(own, ctx, m)
    var dvcoef = afn_scratch_f32(own, ctx, m)
    var ones = afn_scratch_f32(own, ctx, m)
    ones.enqueue_fill(Float32(1.0))
    bwd_rms_norm[0](
        ctx, dot_out, dx, dw, dh, dprod, rstd, dvcoef, ones, dy, x, w, sumsq,
        dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), False, m, dm, eps,
    )
    keep.append(dot_out^)
    keep.append(dh^)
    keep.append(dprod^)
    keep.append(rstd^)
    keep.append(dvcoef^)
    keep.append(ones^)


def afn_rms_norm_backward_host(
    ctx: DeviceContext,
    dx_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dy_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    dm: Int,
    eps: Float32,
) raises -> Int:
    _refuse_norm_shape(m, dm, eps)
    var cells = m * dm
    comptime if not AFN_SAMBA_DEVICE_ADMIT:
        _refuse_nonfinite_host("rms_norm input", x_ptr, cells)
        _refuse_nonfinite_host("rms_norm weight", w_ptr, dm)
        _refuse_nonfinite_host("rms_norm upstream gradient", dy_ptr, cells)
    var arena = AfnArena()
    var own = arena.id >= 0
    var admit = AfnAdmit(ctx)
    var x = afn_upload_f32(own, ctx, x_ptr, cells)
    var w = afn_upload_f32(own, ctx, w_ptr, dm)
    var dy = afn_upload_f32(own, ctx, dy_ptr, cells)
    admit.scan(ctx, x, cells, "rms_norm input")
    admit.scan(ctx, w, dm, "rms_norm weight")
    admit.scan(ctx, dy, cells, "rms_norm upstream gradient")
    var y = afn_scratch_f32(own, ctx, cells)
    var sumsq = afn_scratch_f32(own, ctx, m)
    var dx = afn_scratch_f32(own, ctx, cells)
    var dw = afn_scratch_f32(own, ctx, dm)
    var keep = List[DeviceBuffer[DType.float32]]()
    llama_rms_norm(ctx, sumsq, y, x, w, m, dm, eps)
    _rms_norm_backward_resident(own, ctx, dx, dw, dy, x, w, sumsq, keep, m, dm, eps)
    ctx.enqueue_copy(dst_ptr=dx_ptr, src_buf=dx)
    ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
    admit.finish(ctx)
    ctx.synchronize()
    arena.close()
    admit.check()
    _ = x^
    _ = w^
    _ = dy^
    _ = y^
    _ = sumsq^
    _ = dx^
    _ = dw^
    _ = keep^
    _ = admit^
    return cells


# ===========================================================================
# THE HEAD GEMMS ON THE MATRIX UNIT (MOJOLEARN_AFN_SAMBA_HEAD_GEMM, lane w2-epi)
# ===========================================================================


def _afn_head_k_split(tiles: Int, k: Int) -> Int:
    """Steps per split, or 0: the gemm lane's `afn_gemm_k_split` rule, owned
    here so it follows MOJOLEARN_AFN_SAMBA_HEAD_GEMM rather than
    MOJOLEARN_AFN_GEMM_SPLITK. Whole windows per split."""
    var target = 2 * AFN_GEMM_CORES
    if tiles >= target or k < 2 * AFN_GEMM_SPLIT_MIN_STEPS:
        return 0
    var s = (target + tiles - 1) // tiles
    s = min(s, k // AFN_GEMM_SPLIT_MIN_STEPS)
    s = min(s, AFN_GEMM_SPLIT_MAX)
    if s <= 1:
        return 0
    var per = (k + s - 1) // s
    per = ((per + AFN_GEMM_KB - 1) // AFN_GEMM_KB) * AFN_GEMM_KB
    if (k + per - 1) // per <= 1:
        return 0
    return per


def afn_head_gemm_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C[m x n] = op(A) . op(B)` on the FAST matrix-unit kernel: one launch,
    or (a short grid with a long `k`) one zero launch of the `m n` cells and
    one split launch adding every split into them. Asynchronous; every
    buffer is the caller's. True when served; False (nothing enqueued) when
    the switch is off or the shape/op is not served, and the caller runs its
    `identical_gemm_*` call."""
    comptime if not AFN_SAMBA_HEAD_GEMM:
        return False
    else:
        if m <= 0 or n <= 0 or k <= 0:
            return False
        if op != OP_NN and op != OP_NT and op != OP_TN:
            return False
        var st = _afn_strides(op, m, n, k)
        var tile = afn_gemm_tile(m, n)
        var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ap = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var bp = b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var k_split = _afn_head_k_split(afn_gemm_tile_count(tile, m, n), k)
        if k_split > 0:
            var splits = (k + k_split - 1) // k_split
            ctx.enqueue_function[afn_zero_kernel](
                cp,
                Int32(m * n),
                grid_dim=((m * n + 4 * AFN_ZERO_TPB - 1) // (4 * AFN_ZERO_TPB), 1, 1),
                block_dim=(AFN_ZERO_TPB, 1, 1),
            )
            _afn_launch_tile[DType.float32, DType.float32, True, AFN_EPI_NONE](
                ctx, tile, cp, ap, bp, cp, cp, m, n, k, st, splits, k_split
            )
        else:
            _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_NONE](
                ctx, tile, cp, ap, bp, cp, cp, m, n, k, st, 1, k
            )
        return True


def _afn_head_backward_a_into(
    ctx: DeviceContext,
    mut da: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`identical_gemm_backward_a_into`'s operand table
    (`gemm_backward_a_call`) on the matrix-unit kernel."""
    var call = gemm_backward_a_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        return afn_head_gemm_into(ctx, da, dc, b, call[1], call[2], call[3], call[0])
    return afn_head_gemm_into(ctx, da, b, dc, call[1], call[2], call[3], call[0])


def _afn_head_backward_b_into(
    ctx: DeviceContext,
    mut db: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`identical_gemm_backward_b_into`'s operand table
    (`gemm_backward_b_call`) on the matrix-unit kernel."""
    var call = gemm_backward_b_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        return afn_head_gemm_into(ctx, db, dc, a, call[1], call[2], call[3], call[0])
    return afn_head_gemm_into(ctx, db, a, dc, call[1], call[2], call[3], call[0])


def _head_loss_resident(
    own: Bool,
    ctx: DeviceContext,
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    row_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mut da: DeviceBuffer[DType.float32],
    mut dw: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    mut admit: AfnAdmit,
    mut keep: List[DeviceBuffer[DType.float32]],
    mut ints: List[DeviceBuffer[DType.int32]],
    m: Int,
    n: Int,
    k: Int,
    ignore_index: Int,
    reduction: Int,
    num_items: Int,
    label_smoothing: Float32,
) raises -> Int:
    """samba_head_loss_host's device half on resident `a` (the head input)
    and `w`: the head GEMM, the loss refusals, the loss with its gradient,
    then both head backward GEMMs into `da` and `dw`. The logits come down
    for the host scan only without the admit define. Returns `count`."""
    var cells = m * n
    var c = afn_scratch_f32(own, ctx, cells)
    # lane w2-epi (MOJOLEARN_AFN_SAMBA_HEAD_GEMM): the matrix-unit kernel,
    # no workspace; afn-samba's spelling if it declines or the switch is off.
    comptime if AFN_SAMBA_HEAD_GEMM:
        if not afn_head_gemm_into(ctx, c, a, w, m, n, k, OP_NT):
            var ws = afn_scratch_f32(own, ctx, identical_gemm_workspace_max_floats(m, n, k))
            identical_gemm_into(ctx, c, a, w, ws, m, n, k, OP_NT)
            keep.append(ws^)
    else:
        var ws = afn_scratch_f32(own, ctx, identical_gemm_workspace_max_floats(m, n, k))
        identical_gemm_into(ctx, c, a, w, ws, m, n, k, OP_NT)
        keep.append(ws^)

    identical_ce_admit_call(reduction, 1, m)
    var cfg = CeConfig(n, ignore_index, reduction, label_smoothing, num_items)
    var h_targets = List[Int32](capacity=m)
    for i in range(m):
        h_targets.append(targets_ptr.unsafe_load(i))
    comptime if AFN_SAMBA_DEVICE_ADMIT:
        ce_refuse_shape(m, cells, cfg)
        admit.scan(ctx, c, cells, "logits")
        ce_refuse_targets(h_targets, cfg)
    else:
        var h_c = ctx.enqueue_create_host_buffer[DType.float32](cells)
        ctx.enqueue_copy(dst_ptr=h_c.unsafe_ptr(), src_buf=c)
        ctx.synchronize()
        var h_logits = List[Float32](capacity=cells)
        var hp = h_c.unsafe_ptr()
        for i in range(cells):
            h_logits.append(hp.unsafe_load(i))
        _ = ce_refuse_inputs(h_logits, h_targets, cfg)
        _ = h_logits^
        _ = h_c^
    var count = ce_count(h_targets, ignore_index)
    _ = h_targets^

    ints.append(afn_upload_i32(ctx, targets_ptr, m))
    var dc = afn_scratch_f32(own, ctx, cells)
    identical_ce_loss_resident(
        ctx, loss_ptr, row_ptr, dc, c, ints[len(ints) - 1], m, count, reduction, 1, cfg,
    )
    # lane w2-epi (MOJOLEARN_AFN_SAMBA_HEAD_GEMM): dA and dW on the
    # matrix-unit kernel (dW, k = m tokens, splits when its grid is short).
    comptime if AFN_SAMBA_HEAD_GEMM:
        if not _afn_head_backward_a_into(ctx, da, dc, w, m, n, k, OP_NT):
            var ws_a = afn_scratch_f32(
                own, ctx, identical_gemm_backward_a_workspace_max_floats(OP_NT, m, n, k)
            )
            identical_gemm_backward_a_into(ctx, da, dc, w, ws_a, m, n, k, OP_NT)
            keep.append(ws_a^)
        if not _afn_head_backward_b_into(ctx, dw, dc, a, m, n, k, OP_NT):
            var ws_b = afn_scratch_f32(
                own, ctx, identical_gemm_backward_b_workspace_max_floats(OP_NT, m, n, k)
            )
            identical_gemm_backward_b_into(ctx, dw, dc, a, ws_b, m, n, k, OP_NT)
            keep.append(ws_b^)
    else:
        var ws_a = afn_scratch_f32(
            own, ctx, identical_gemm_backward_a_workspace_max_floats(OP_NT, m, n, k)
        )
        var ws_b = afn_scratch_f32(
            own, ctx, identical_gemm_backward_b_workspace_max_floats(OP_NT, m, n, k)
        )
        identical_gemm_backward_a_into(ctx, da, dc, w, ws_a, m, n, k, OP_NT)
        identical_gemm_backward_b_into(ctx, dw, dc, a, ws_b, m, n, k, OP_NT)
        keep.append(ws_a^)
        keep.append(ws_b^)
    keep.append(c^)
    keep.append(dc^)
    return count


def afn_head_loss_host(
    ctx: DeviceContext,
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    row_ptr: MutPointer[Float32, MutUntrackedOrigin],
    da_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    a_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    ignore_index: Int,
    reduction: Int,
    num_items: Int,
    label_smoothing: Float32,
) raises -> Int:
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    comptime if not AFN_SAMBA_DEVICE_ADMIT:
        _refuse_nonfinite_host("linear input", a_ptr, m * k)
        _refuse_nonfinite_host("linear weight", w_ptr, n * k)
    var arena = AfnArena()
    var own = arena.id >= 0
    var admit = AfnAdmit(ctx)
    var a = afn_upload_f32(own, ctx, a_ptr, m * k)
    var w = afn_upload_f32(own, ctx, w_ptr, n * k)
    admit.scan(ctx, a, m * k, "linear input")
    admit.scan(ctx, w, n * k, "linear weight")
    var da = afn_scratch_f32(own, ctx, m * k)
    var dw = afn_scratch_f32(own, ctx, n * k)
    var keep = List[DeviceBuffer[DType.float32]]()
    var ints = List[DeviceBuffer[DType.int32]]()
    var count = _head_loss_resident(
        own, ctx, loss_ptr, row_ptr, da, dw, a, w, targets_ptr, admit, keep, ints,
        m, n, k, ignore_index, reduction, num_items, label_smoothing,
    )
    ctx.enqueue_copy(dst_ptr=da_ptr, src_buf=da)
    ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
    admit.finish(ctx)
    ctx.synchronize()
    arena.close()
    admit.check()
    _ = a^
    _ = w^
    _ = da^
    _ = dw^
    _ = keep^
    _ = ints^
    _ = admit^
    return count


# ===========================================================================
# THE FUSED ENTRIES (MOJOLEARN_AFN_SAMBA_FUSE)
# ===========================================================================


def afn_norm_head_forward_host(
    ctx: DeviceContext,
    c_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    nw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    hw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    eps: Float32,
) raises -> Int:
    """The final RMSNorm then the head GEMM in one call: `c[m, n] =
    rms_norm(x[m, k], nw) . hw[n, k]^T`. `hn` never leaves the device.
    Returns `m * n`."""
    comptime if not AFN_SAMBA_FUSE:
        raise Error("mojolearn samba afn: fused entries are not in this build")
    else:
        _refuse_norm_shape(m, k, eps)
        if n < 1:
            raise Error("mojolearn samba ops: linear shape must be positive")
        var cells = m * k
        comptime if not AFN_SAMBA_DEVICE_ADMIT:
            _refuse_nonfinite_host("rms_norm input", x_ptr, cells)
            _refuse_nonfinite_host("rms_norm weight", nw_ptr, k)
            _refuse_nonfinite_host("linear weight", hw_ptr, n * k)
        var arena = AfnArena()
        var own = arena.id >= 0
        var admit = AfnAdmit(ctx)
        var x = afn_upload_f32(own, ctx, x_ptr, cells)
        var nw = afn_upload_f32(own, ctx, nw_ptr, k)
        var hw = afn_upload_f32(own, ctx, hw_ptr, n * k)
        admit.scan(ctx, x, cells, "rms_norm input")
        admit.scan(ctx, nw, k, "rms_norm weight")
        admit.scan(ctx, hw, n * k, "linear weight")
        var hn = afn_scratch_f32(own, ctx, cells)
        var sumsq = afn_scratch_f32(own, ctx, m)
        llama_rms_norm(ctx, sumsq, hn, x, nw, m, k, eps)
        var c = afn_scratch_f32(own, ctx, m * n)
        var keep = List[DeviceBuffer[DType.float32]]()
        # lane w2-epi (MOJOLEARN_AFN_SAMBA_HEAD_GEMM): the matrix-unit
        # kernel, no workspace; afn-samba's spelling if it declines.
        comptime if AFN_SAMBA_HEAD_GEMM:
            if not afn_head_gemm_into(ctx, c, hn, hw, m, n, k, OP_NT):
                var ws = afn_scratch_f32(own, ctx, identical_gemm_workspace_max_floats(m, n, k))
                identical_gemm_into(ctx, c, hn, hw, ws, m, n, k, OP_NT)
                keep.append(ws^)
        else:
            var ws = afn_scratch_f32(own, ctx, identical_gemm_workspace_max_floats(m, n, k))
            identical_gemm_into(ctx, c, hn, hw, ws, m, n, k, OP_NT)
            keep.append(ws^)
        ctx.enqueue_copy(dst_ptr=c_ptr, src_buf=c)
        admit.finish(ctx)
        ctx.synchronize()
        arena.close()
        admit.check()
        _ = x^
        _ = nw^
        _ = hw^
        _ = hn^
        _ = sumsq^
        _ = c^
        _ = keep^
        _ = admit^
        return m * n


def afn_tail_train_host(
    ctx: DeviceContext,
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    row_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dh_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dnw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dhw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    nw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    hw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    eps: Float32,
    ignore_index: Int,
    reduction: Int,
    num_items: Int,
    label_smoothing: Float32,
) raises -> Int:
    """The train step's tail in one call: final RMSNorm forward, head GEMM,
    the loss with its gradient, both head backward GEMMs and the norm
    backward, on the device throughout. In: `x[m, k]` (the last block's
    output), `nw[k]`, `hw[n, k]`, `targets[m]`. Out: `loss[1]`, `row[m]`,
    `dh[m, k]` (the gradient at the last block's output), `dnw[k]`,
    `dhw[n, k]`. The norm backward reuses the forward's row sums instead of
    recomputing them (same kernel, same inputs, same bits). Returns
    `count`."""
    comptime if not AFN_SAMBA_FUSE:
        raise Error("mojolearn samba afn: fused entries are not in this build")
    else:
        _refuse_norm_shape(m, k, eps)
        if n < 1:
            raise Error("mojolearn samba ops: linear shape must be positive")
        var cells = m * k
        comptime if not AFN_SAMBA_DEVICE_ADMIT:
            _refuse_nonfinite_host("rms_norm input", x_ptr, cells)
            _refuse_nonfinite_host("rms_norm weight", nw_ptr, k)
            _refuse_nonfinite_host("linear weight", hw_ptr, n * k)
        var arena = AfnArena()
        var own = arena.id >= 0
        var admit = AfnAdmit(ctx)
        var x = afn_upload_f32(own, ctx, x_ptr, cells)
        var nw = afn_upload_f32(own, ctx, nw_ptr, k)
        var hw = afn_upload_f32(own, ctx, hw_ptr, n * k)
        admit.scan(ctx, x, cells, "rms_norm input")
        admit.scan(ctx, nw, k, "rms_norm weight")
        admit.scan(ctx, hw, n * k, "linear weight")
        var hn = afn_scratch_f32(own, ctx, cells)
        var sumsq = afn_scratch_f32(own, ctx, m)
        llama_rms_norm(ctx, sumsq, hn, x, nw, m, k, eps)
        var dhn = afn_scratch_f32(own, ctx, cells)
        var dhw = afn_scratch_f32(own, ctx, n * k)
        var keep = List[DeviceBuffer[DType.float32]]()
        var ints = List[DeviceBuffer[DType.int32]]()
        var count = _head_loss_resident(
            own, ctx, loss_ptr, row_ptr, dhn, dhw, hn, hw, targets_ptr, admit, keep, ints,
            m, n, k, ignore_index, reduction, num_items, label_smoothing,
        )
        var dx = afn_scratch_f32(own, ctx, cells)
        var dnw = afn_scratch_f32(own, ctx, k)
        _rms_norm_backward_resident(own, ctx, dx, dnw, dhn, x, nw, sumsq, keep, m, k, eps)
        ctx.enqueue_copy(dst_ptr=dh_ptr, src_buf=dx)
        ctx.enqueue_copy(dst_ptr=dnw_ptr, src_buf=dnw)
        ctx.enqueue_copy(dst_ptr=dhw_ptr, src_buf=dhw)
        admit.finish(ctx)
        ctx.synchronize()
        arena.close()
        admit.check()
        _ = x^
        _ = nw^
        _ = hw^
        _ = hn^
        _ = sumsq^
        _ = dhn^
        _ = dhw^
        _ = dx^
        _ = dnw^
        _ = keep^
        _ = ints^
        _ = admit^
        return count


def afn_embedding_backward_tied_host(
    ctx: DeviceContext,
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dy_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    pair_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_positions: Int,
    vocab: Int,
    width: Int,
) raises -> Int:
    """The tied embedding gradient in one call: `dw = emb_backward(dy, ids)
    + pair` with `pair[vocab, width]` the head's weight gradient. Under
    MOJOLEARN_AFN_SAMBA_EMB_ATOMIC the pair is uploaded straight into `dw`
    and the scatter accumulates onto it (no seed, no pair-add launch);
    otherwise the identical fold runs into a scratch and one pair-add
    launch forms `dw` (embedding first, head second, as the tree did).
    Returns `vocab * width`."""
    comptime if not AFN_SAMBA_FUSE:
        raise Error("mojolearn samba afn: fused entries are not in this build")
    else:
        if n_positions < 1 or vocab < 1 or width < 1:
            raise Error("mojolearn samba ops: embedding shape must be positive")
        var cells = vocab * width
        comptime if not AFN_SAMBA_DEVICE_ADMIT:
            _refuse_nonfinite_host("embedding upstream gradient", dy_ptr, n_positions * width)
            _refuse_nonfinite_host("accumulate parts", pair_ptr, cells)
        var arena = AfnArena()
        var own = arena.id >= 0
        var admit = AfnAdmit(ctx)
        var dy = afn_upload_f32(own, ctx, dy_ptr, n_positions * width)
        admit.scan(ctx, dy, n_positions * width, "embedding upstream gradient")
        var ids = afn_upload_i32(ctx, ids_ptr, n_positions)
        var pair = afn_upload_f32(own, ctx, pair_ptr, cells)
        admit.scan(ctx, pair, cells, "accumulate parts")
        var dw = afn_scratch_f32(own, ctx, cells)
        var ints = List[DeviceBuffer[DType.int32]]()
        comptime if AFN_SAMBA_EMB_ATOMIC:
            afn_embedding_backward_into(
                ctx, pair, dy, ids, admit, ints, n_positions, vocab, width, False
            )
            ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=pair)
        else:
            afn_embedding_backward_into(
                ctx, dw, dy, ids, admit, ints, n_positions, vocab, width, True
            )
            ctx.enqueue_function[afn_pair_add_kernel](
                dw.unsafe_ptr(), pair.unsafe_ptr(), Int32(cells),
                grid_dim=(_grid(cells), 1, 1),
                block_dim=(AFN_TPB, 1, 1),
            )
            ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
        admit.finish(ctx)
        ctx.synchronize()
        arena.close()
        admit.check()
        _ = dy^
        _ = ids^
        _ = pair^
        _ = dw^
        _ = ints^
        _ = admit^
        return cells


# ===========================================================================
# THE PYTHON ENTRIES (registered by bindings/_mojolearn_training.mojo under
# AFN_SAMBA_FUSE; the (addresses, params) shape of the samba ops)
# ===========================================================================

comptime _AFN_NEURAL_CTX = (
    "MojoNeuralTrainingContextIdentical"
    if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojoNeuralTrainingContextFast"
)


def _afn_addrs(addresses: PythonObject, want: Int, name: String) raises -> List[Int]:
    if len(addresses) != want:
        raise Error(
            name + ": addresses must contain " + String(want) + " entries, got "
            + String(len(addresses))
        )
    var out = List[Int]()
    for i in range(want):  # small-loop(want: buffer addresses of one op): address slot admission
        var a = Int(py=addresses[i])
        if a == 0:
            raise Error(name + ": null buffer address at slot " + String(i))
        out.append(a)
    return out^


def _afn_params(params: PythonObject, want: Int, name: String) raises:
    if len(params) != want:
        raise Error(
            name + ": params must contain " + String(want) + " values, got "
            + String(len(params))
        )


def _afn_f32(addr: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _afn_i32(addr: Int) -> MutPointer[Int32, MutUntrackedOrigin]:
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def samba_afn_norm_head_forward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [c (m*n f32, written), x (m*k f32), norm_w (k f32),
    head_w (n*k f32)]; params = [m, n, k, eps (float)]. Returns `m * n`."""
    var a = _afn_addrs(addresses, 4, "samba_afn_norm_head_forward")
    _afn_params(params, 4, "samba_afn_norm_head_forward")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var eps = Float32(Float64(py=params[3]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_AFN_NEURAL_CTX]()
        count = afn_norm_head_forward_host(
            ctx, _afn_f32(a[0]), _afn_f32(a[1]), _afn_f32(a[2]), _afn_f32(a[3]),
            m, n, k, eps,
        )
    return PythonObject(count)


def samba_afn_tail_train_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [loss (1 f32, written), row_loss (m f32, written), dh
    (m*k f32, written), d_norm_w (k f32, written), d_head_w (n*k f32,
    written), x (m*k f32), norm_w (k f32), head_w (n*k f32), targets (m
    i32)]; params = [m, n, k, eps (float), ignore_index, reduction (1 sum,
    2 mean), num_items, label_smoothing (float)]. Returns `count`."""
    var a = _afn_addrs(addresses, 9, "samba_afn_tail_train")
    _afn_params(params, 8, "samba_afn_tail_train")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var eps = Float32(Float64(py=params[3]))
    var ignore_index = Int(py=params[4])
    var reduction = Int(py=params[5])
    var num_items = Int(py=params[6])
    var label_smoothing = Float32(Float64(py=params[7]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_AFN_NEURAL_CTX]()
        count = afn_tail_train_host(
            ctx, _afn_f32(a[0]), _afn_f32(a[1]), _afn_f32(a[2]), _afn_f32(a[3]),
            _afn_f32(a[4]), _afn_f32(a[5]), _afn_f32(a[6]), _afn_f32(a[7]),
            _afn_i32(a[8]),
            m, n, k, eps, ignore_index, reduction, num_items, label_smoothing,
        )
    return PythonObject(count)


def samba_afn_embedding_backward_tied_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [dw (vocab*width f32, written), dy (n_positions*width
    f32), ids (n_positions i32), pair (vocab*width f32, the head's weight
    gradient)]; params = [n_positions, vocab, width]. Returns
    `vocab * width`."""
    var a = _afn_addrs(addresses, 4, "samba_afn_embedding_backward_tied")
    _afn_params(params, 3, "samba_afn_embedding_backward_tied")
    var n_positions = Int(py=params[0])
    var vocab = Int(py=params[1])
    var width = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_AFN_NEURAL_CTX]()
        count = afn_embedding_backward_tied_host(
            ctx, _afn_f32(a[0]), _afn_f32(a[1]), _afn_i32(a[2]), _afn_f32(a[3]),
            n_positions, vocab, width,
        )
    return PythonObject(count)
