# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DART's boosting round on the device (lane/apple-fast-dart, 2026-10-02).

FAST on every GPU vendor (Apple first, NVIDIA and AMD since lane cpu2-l5-trees;
off: `-D MOJOLEARN_DART_DEVICE_OFF`) and, since fam2-forests, IDENTICAL on every
vendor (`IDN_DART_DEVICE`, off: `-D MOJOLEARN_IDN_DART_DEVICE_OFF`); nothing here is
instantiated otherwise (`DART_DEVICE` guards every entry body and the
binding's registration), so IDENTICAL compiles main's code unchanged.

Main's `_DARTBase._boost_loop` (python/mojolearn/_expansion_trees.py) keeps
the running score on the host and, each round, walks every row through the
new tree on the host (`apply_trees`), takes the gradients on the host, sums
the leaf Newton statistics on the host, and runs `tree_score_add` once per
dropped tree to subtract it and once more to put it back rescaled: with
drop_rate 0.1 over 200 trees that is tens of host passes over a million
rows per round. Here one session (`x_trees_dart_open`) holds X, y, the
running score, the per-tree leaf index rows and the per-tree leaf values on
the device for the whole fit, and each round is

  step   drop draws (t threads, the counter RNG of xtrees/ops.mojo, the
         decision against host-prepared 53-bit integer thresholds, so the
         drop SET is main's bit for bit) and ONE row launch that gathers the
         dropped trees' leaf values through their leaf index rows, takes them
         off the score and writes the objective's gradient, hessian and fit
         target; the target and the flags come back once (the fit needs its
         labels) -- the only sync of the round;
  add    (per class tree) the leaf index row of the new tree (row walk), the
         leaf g/h sums (chunked, folded in a fixed order: no atomics), the
         Newton leaf values, and one row launch that adds the scaled new tree
         and the dropped trees' rescale (factor x the gathered sum) back
         onto the score; the leaf values go back to the host asynchronously
         for the model.

Deviations from main's spelling under this switch (FAST tier): the device
holds the score, gradients and leaf values in float32 (Metal has no
float64), the leaf sums fold per 8192-row chunk then across chunks instead
of one row-order chain, and the dropped trees come off and go back as one
gathered sum per row instead of one add per tree. The drop set, the shrink
factors and the tree shapes are main's."""
from std.ffi import _Global
from std.gpu import block_idx, block_dim, grid_dim, thread_idx
from std.sys.compile import is_defined
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_zero import enqueue_fill
from core.neural_context import process_ctx
from xtrees.ops import stream_base, draw
from xtrees.dart_units import (
    IDN_DART_DEVICE, DART_CHUNK, F32P, I32P, I64P, U16P, dart_init_unit, dart_drop_unit, dart_row_unit, dart_apply_unit,
    dart_leaf_sum_unit, dart_newton_unit, dart_add_unit,
)

# fam2-forests (2026-10-04), IDENTICAL, every vendor, default ON: the round
# below is DART's IDENTICAL route too (`IDN_DART_DEVICE`). Before, an
# IDENTICAL GPU fit ran main's `_boost_loop`: the score, every tree walk,
# the gradients, the leaf Newton sums and one score pass per dropped tree on
# the HOST each round. The kernel bodies now live in xtrees/dart_units.mojo
# with every float operation pinned under IDENTICAL (see that file), and the
# host column runs the same units (xtrees/dart_host.mojo), so NVIDIA, AMD,
# Apple and the host produce the same words. BITS CHANGE against the old
# IDENTICAL DART (float32 state and chunked leaf sums instead of float64
# row-order chains), on all four columns together.
# `-D MOJOLEARN_IDN_DART_DEVICE_OFF` (or `MOJOLEARN_IDN_ALL_OFF`) restores
# main's loop on every column.
# The gate itself is `xtrees/dart_units.mojo: IDN_DART_DEVICE` (shared with the host twin).

# FAST default since the M3 A/B (lane/apple-fast-dart 443f4b3cc,
# istella, n=1): dart 45,837 -> 24,730 ms (-46%), acc .9487 -> .9486;
# dart-reg 45,445 -> 24,837 ms (-45%), r2 .5507 -> .5514. Off define:
# -D MOJOLEARN_DART_DEVICE_OFF. The old -D MOJOLEARN_DART_DEVICE is harmless.
# lane cpu2-l5-trees (2026-10-04): FAST on EVERY GPU vendor, not Apple only.
# Before, a FAST NVIDIA or AMD fit ran main's `_boost_loop`, the score, the
# tree walks, the gradients, the leaf sums and the per-dropped-tree passes
# as host ops each round; the round here is the FAST + Apple round
# unchanged (float32 score, chunked leaf sums), now on every vendor.
comptime DART_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_accelerator() and not is_defined["MOJOLEARN_DART_DEVICE_OFF"]()
) or IDN_DART_DEVICE

comptime TPB = 256
comptime GRID_MAX = 65535 * 16
comptime _DART_CTX = "MojoXTreesDartContext"


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _tstride() -> Int:
    return Int(grid_dim.x) * Int(block_dim.x)


def _blocks(units: Int) -> Int:
    return min((units + TPB - 1) // TPB, GRID_MAX)


# ------------------------------------------------------------------ kernels
# Each kernel is a grid-stride loop over its units; the unit bodies are
# xtrees/dart_units.mojo's (shared with the host column).
def dart_init_kernel(units: Int64, n: Int64, inits: F32P, score: F32P):
    """score[c * n + i] = inits[c]."""
    var e = _tid()
    var stride = _tstride()
    while e < Int(units):
        dart_init_unit(e, Int(n), inits, score)
        e += stride


def dart_drop_kernel(units: Int64, seed: Int64, stream: Int64, skip_thr: Int64, thr: I64P, flags: I32P):
    """flags[i] = 1 when iteration i is dropped this round: draw 0 of the
    round's stream decides the skip (`unit(draw) < skip_drop` as the exact
    integer compare of the top 53 bits against ceil(skip_drop * 2^53)),
    draw 1 + i the tree (against thr[i] = ceil(rate_i * 2^53)). The same
    draws as `uniform(.., 1 + t, drop_seed, it)` on the host."""
    var base = stream_base(Int(seed), Int(stream))
    var skip = (draw(base, 0) >> 11) < UInt64(skip_thr)
    var e = _tid()
    var stride = _tstride()
    while e < Int(units):
        dart_drop_unit(e, base, skip, thr, flags)
        e += stride


def dart_row_kernel(
    units: Int64, n: Int64, k: Int32, t: Int32, kind: Int32, node_cap: Int32, flags: I32P, coef: F32P,
    nodes: U16P, values: F32P, y: F32P, score: F32P, dsum: F32P, target: F32P, h: F32P,
):
    """Per row: dsum[c, i] = the dropped trees' coef x leaf value, taken off
    the score; then main's `gradients` (kind 0 L2, 1 logloss, 2 softmax,
    class-major) into target = -g and h."""
    var e = _tid()
    var stride = _tstride()
    while e < Int(units):
        dart_row_unit(
            e, Int(n), Int(k), Int(t), kind, Int(node_cap), flags, coef, nodes, values, y, score, dsum, target, h,
        )
        e += stride


def dart_apply_kernel(
    units: Int64, d: Int32, count: Int32, colid: I32P, quesval: F32P, left: I32P, x: F32P, row_off: Int64,
    nodes: U16P, bad: I32P,
):
    """nodes[row_off + i] = the leaf node row i reaches (main's `apply_trees`
    walk: left child l, right l + 1, leaf where left is -1). A walk that
    leaves the tree sets bad[0] and lands on node 0; the host raises."""
    var e = _tid()
    var stride = _tstride()
    while e < Int(units):
        dart_apply_unit(e, Int(d), Int(count), colid, quesval, left, x, Int(row_off), nodes, bad)
        e += stride


def dart_leaf_sum_kernel(
    units: Int64, n: Int64, n_nodes: Int32, row_off: Int64, class_off: Int64, nodes: U16P, target: F32P, h: F32P,
    part: F32P,
):
    """part[(q * n_nodes + k) * 2 + {0, 1}] = sum of g (= -target) and h over
    the rows of chunk q that reached node k, in row order."""
    var e = _tid()
    var stride = _tstride()
    while e < Int(units):
        dart_leaf_sum_unit(e, Int(n), Int(n_nodes), Int(row_off), Int(class_off), nodes, target, h, part)
        e += stride


def dart_newton_kernel(
    units: Int64, n_nodes: Int32, n_chunks: Int32, part: F32P, lam: Float32, l1: Float32, mds: Float32,
    voff: Int64, values: F32P,
):
    """values[voff + k] = main's `_newton_values` of the chunk sums folded
    in chunk order: -ThresholdL1(sum g, l1) / (sum h + lambda), clipped to
    +-max_delta_step when that is > 0, 0 where the denominator is not
    positive."""
    var e = _tid()
    var stride = _tstride()
    while e < Int(units):
        dart_newton_unit(e, Int(n_nodes), Int(n_chunks), part, lam, l1, mds, Int(voff), values)
        e += stride


def dart_add_kernel(
    units: Int64, class_off: Int64, row_off: Int64, voff: Int64, factor: Float32, shrink: Float32, nodes: U16P,
    values: F32P, dsum: F32P, score: F32P,
):
    """score[c, i] += factor * dsum[c, i] + shrink * values[leaf of row i]:
    the dropped trees back at their rescaled weight (main: each one added
    again at coef x factor) and the new tree at its shrinkage."""
    var e = _tid()
    var stride = _tstride()
    while e < Int(units):
        dart_add_unit(e, Int(class_off), Int(row_off), Int(voff), factor, shrink, nodes, values, dsum, score)
        e += stride


# ------------------------------------------------------------------ session
struct DartSession(Movable):
    var id: Int
    var n: Int
    var d: Int
    var k: Int
    var kind: Int
    var cap_iters: Int
    var node_cap: Int
    var n_chunks: Int
    var x: DeviceBuffer[DType.float32]
    var y: DeviceBuffer[DType.float32]
    var score: DeviceBuffer[DType.float32]
    var dsum: DeviceBuffer[DType.float32]
    var target: DeviceBuffer[DType.float32]
    var h: DeviceBuffer[DType.float32]
    var nodes: DeviceBuffer[DType.uint16]
    var values: DeviceBuffer[DType.float32]
    var coef: DeviceBuffer[DType.float32]
    var thr: DeviceBuffer[DType.int64]
    var flags: DeviceBuffer[DType.int32]
    var bad: DeviceBuffer[DType.int32]
    var part: DeviceBuffer[DType.float32]
    var colid: DeviceBuffer[DType.int32]
    var quesval: DeviceBuffer[DType.float32]
    var left: DeviceBuffer[DType.int32]

    def __init__(
        out self, id: Int, n: Int, d: Int, k: Int, kind: Int, cap_iters: Int, node_cap: Int, n_chunks: Int,
        var x: DeviceBuffer[DType.float32], var y: DeviceBuffer[DType.float32],
        var score: DeviceBuffer[DType.float32], var dsum: DeviceBuffer[DType.float32],
        var target: DeviceBuffer[DType.float32], var h: DeviceBuffer[DType.float32],
        var nodes: DeviceBuffer[DType.uint16], var values: DeviceBuffer[DType.float32],
        var coef: DeviceBuffer[DType.float32], var thr: DeviceBuffer[DType.int64],
        var flags: DeviceBuffer[DType.int32], var bad: DeviceBuffer[DType.int32],
        var part: DeviceBuffer[DType.float32], var colid: DeviceBuffer[DType.int32],
        var quesval: DeviceBuffer[DType.float32], var left: DeviceBuffer[DType.int32],
    ):
        self.id = id
        self.n = n
        self.d = d
        self.k = k
        self.kind = kind
        self.cap_iters = cap_iters
        self.node_cap = node_cap
        self.n_chunks = n_chunks
        self.x = x^
        self.y = y^
        self.score = score^
        self.dsum = dsum^
        self.target = target^
        self.h = h^
        self.nodes = nodes^
        self.values = values^
        self.coef = coef^
        self.thr = thr^
        self.flags = flags^
        self.bad = bad^
        self.part = part^
        self.colid = colid^
        self.quesval = quesval^
        self.left = left^


struct DartRegistry(Defaultable, Movable):
    var sessions: List[DartSession]
    var next_id: Int

    def __init__(out self):
        self.sessions = List[DartSession]()
        self.next_id = 1

    def find(self, id: Int) raises -> Int:
        for i in range(len(self.sessions)):
            if self.sessions[i].id == id:
                return i
        raise Error("x_trees dart: unknown or closed session handle")


comptime DART_SESSIONS = _Global[StorageType=DartRegistry, name="MojoXTreesDartSessions", init_fn=DartRegistry.__init__]


@always_inline
def _f(b: DeviceBuffer[DType.float32]) -> F32P:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _i(b: DeviceBuffer[DType.int32]) -> I32P:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _i64(b: DeviceBuffer[DType.int64]) -> I64P:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _u16(b: DeviceBuffer[DType.uint16]) -> U16P:
    return b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def dart_open(
    x_addr: Int, y_addr: Int, inits_addr: Int, n: Int, d: Int, k: Int, kind: Int, cap_iters: Int, node_cap: Int,
) raises -> Int:
    """Stage X (float32, row-major n x d), y (float32 n) and the k class
    starts on the device; allocate the score (k x n), the per-tree leaf
    index rows (cap_iters x k rows of n) and leaf values (node_cap each).
    Returns the handle."""
    comptime if DART_DEVICE:
        if n <= 0 or d <= 0 or k <= 0 or cap_iters <= 0 or node_cap <= 0 or node_cap > 65535:
            raise Error("x_trees dart_open: needs rows, features, classes, iterations and 1 <= node_cap <= 65535")
        var n_trees = cap_iters * k
        var n_chunks = (n + DART_CHUNK - 1) // DART_CHUNK
        var ctx = process_ctx[_DART_CTX]()
        var x = ctx.enqueue_create_buffer[DType.float32](n * d)
        ctx.enqueue_copy(dst_buf=x, src_ptr=F32P(unsafe_from_address=x_addr))
        var y = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_copy(dst_buf=y, src_ptr=F32P(unsafe_from_address=y_addr))
        var inits = ctx.enqueue_create_buffer[DType.float32](k)
        ctx.enqueue_copy(dst_buf=inits, src_ptr=F32P(unsafe_from_address=inits_addr))
        var score = ctx.enqueue_create_buffer[DType.float32](k * n)
        var dsum = ctx.enqueue_create_buffer[DType.float32](k * n)
        var target = ctx.enqueue_create_buffer[DType.float32](k * n)
        var h = ctx.enqueue_create_buffer[DType.float32](k * n)
        var nodes = ctx.enqueue_create_buffer[DType.uint16](n_trees * n)
        var values = ctx.enqueue_create_buffer[DType.float32](n_trees * node_cap)
        var coef = ctx.enqueue_create_buffer[DType.float32](n_trees)
        var thr = ctx.enqueue_create_buffer[DType.int64](cap_iters)
        var flags = ctx.enqueue_create_buffer[DType.int32](cap_iters)
        var bad = ctx.enqueue_create_buffer[DType.int32](1)
        var part = ctx.enqueue_create_buffer[DType.float32](n_chunks * node_cap * 2)
        var colid = ctx.enqueue_create_buffer[DType.int32](node_cap)
        var quesval = ctx.enqueue_create_buffer[DType.float32](node_cap)
        var left = ctx.enqueue_create_buffer[DType.int32](node_cap)
        ctx.enqueue_function[dart_init_kernel](
            Int64(k * n), Int64(n), _f(inits), _f(score),
            grid_dim=(_blocks(k * n), 1, 1), block_dim=(TPB, 1, 1),
        )
        enqueue_fill[DType.int32](ctx, bad, Int32(0))
        ctx.synchronize()
        _ = inits^
        var reg = DART_SESSIONS.get_or_create_ptr()
        var id = reg[].next_id
        reg[].next_id += 1
        reg[].sessions.append(DartSession(
            id, n, d, k, kind, cap_iters, node_cap, n_chunks, x^, y^, score^, dsum^, target^, h^, nodes^, values^,
            coef^, thr^, flags^, bad^, part^, colid^, quesval^, left^,
        ))
        return id
    else:
        raise Error("x_trees dart_open: built without DART_DEVICE (MOJOLEARN_DART_DEVICE_OFF / MOJOLEARN_IDN_DART_DEVICE_OFF)")


def dart_step(
    id: Int, coef_addr: Int, thr_addr: Int, flags_out: Int, bad_out: Int, targets: List[Int], t: Int, seed: Int,
    stream: Int, skip_thr: Int,
) raises:
    """One round's first half: the drop flags of iterations 0 .. t (draws on
    the device, coef[t * k] and thr[t] from the host), the dropped trees off
    the score, the gradients. Writes the fit target of each class to
    targets[c] (float32 n each), the flags (int32 t) and the bad word
    (int32 1), then waits: the round's one sync."""
    comptime if DART_DEVICE:
        var reg = DART_SESSIONS.get_or_create_ptr()
        var idx = reg[].find(id)
        var n = reg[].sessions[idx].n
        var k = reg[].sessions[idx].k
        if t < 0 or t > reg[].sessions[idx].cap_iters or len(targets) != k:
            raise Error("x_trees dart_step: iteration count or targets out of range")
        var ctx = process_ctx[_DART_CTX]()
        if t > 0:
            var csub = reg[].sessions[idx].coef.create_sub_buffer[DType.float32](0, t * k)
            ctx.enqueue_copy(dst_buf=csub, src_ptr=F32P(unsafe_from_address=coef_addr))
            var tsub = reg[].sessions[idx].thr.create_sub_buffer[DType.int64](0, t)
            ctx.enqueue_copy(dst_buf=tsub, src_ptr=I64P(unsafe_from_address=thr_addr))
            ctx.enqueue_function[dart_drop_kernel](
                Int64(t), Int64(seed), Int64(stream), Int64(skip_thr), _i64(reg[].sessions[idx].thr),
                _i(reg[].sessions[idx].flags),
                grid_dim=(_blocks(t), 1, 1), block_dim=(TPB, 1, 1),
            )
        ctx.enqueue_function[dart_row_kernel](
            Int64(n), Int64(n), Int32(k), Int32(t), Int32(reg[].sessions[idx].kind),
            Int32(reg[].sessions[idx].node_cap), _i(reg[].sessions[idx].flags), _f(reg[].sessions[idx].coef),
            _u16(reg[].sessions[idx].nodes), _f(reg[].sessions[idx].values), _f(reg[].sessions[idx].y),
            _f(reg[].sessions[idx].score), _f(reg[].sessions[idx].dsum), _f(reg[].sessions[idx].target),
            _f(reg[].sessions[idx].h),
            grid_dim=(_blocks(n), 1, 1), block_dim=(TPB, 1, 1),
        )
        for c in range(k):
            var sub = reg[].sessions[idx].target.create_sub_buffer[DType.float32](c * n, n)
            ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=targets[c]), src_buf=sub)
        if t > 0:
            var fsub = reg[].sessions[idx].flags.create_sub_buffer[DType.int32](0, t)
            ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=flags_out), src_buf=fsub)
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=bad_out), src_buf=reg[].sessions[idx].bad)
        ctx.synchronize()
    else:
        raise Error("x_trees dart_step: built without DART_DEVICE (MOJOLEARN_DART_DEVICE_OFF / MOJOLEARN_IDN_DART_DEVICE_OFF)")


def dart_add(
    id: Int, colid_addr: Int, quesval_addr: Int, left_addr: Int, values_out: Int, j: Int, c: Int, lo: Int,
    n_nodes: Int, shrink: Float64, factor: Float64, lam: Float64, l1: Float64, mds: Float64,
) raises:
    """One round's second half for class c's new tree j (its nodes lo ..
    lo + n_nodes of the forest arrays): the leaf index row, the Newton leaf
    values (to values_out, float32 n_nodes, landed by the next sync) and
    the score update. No wait."""
    comptime if DART_DEVICE:
        var reg = DART_SESSIONS.get_or_create_ptr()
        var idx = reg[].find(id)
        var n = reg[].sessions[idx].n
        var k = reg[].sessions[idx].k
        var node_cap = reg[].sessions[idx].node_cap
        var n_chunks = reg[].sessions[idx].n_chunks
        if j < 0 or j >= reg[].sessions[idx].cap_iters * k or c < 0 or c >= k:
            raise Error("x_trees dart_add: tree or class index out of range")
        if n_nodes < 1 or n_nodes > node_cap or lo < 0:
            raise Error("x_trees dart_add: a tree with more nodes than 2 * num_leaves - 1")
        var ctx = process_ctx[_DART_CTX]()
        var csub = reg[].sessions[idx].colid.create_sub_buffer[DType.int32](0, n_nodes)
        ctx.enqueue_copy(dst_buf=csub, src_ptr=I32P(unsafe_from_address=colid_addr + 4 * lo))
        var qsub = reg[].sessions[idx].quesval.create_sub_buffer[DType.float32](0, n_nodes)
        ctx.enqueue_copy(dst_buf=qsub, src_ptr=F32P(unsafe_from_address=quesval_addr + 4 * lo))
        var lsub = reg[].sessions[idx].left.create_sub_buffer[DType.int32](0, n_nodes)
        ctx.enqueue_copy(dst_buf=lsub, src_ptr=I32P(unsafe_from_address=left_addr + 4 * lo))
        var row_off = Int64(j * n)
        var class_off = Int64(c * n)
        var voff = Int64(j * node_cap)
        ctx.enqueue_function[dart_apply_kernel](
            Int64(n), Int32(reg[].sessions[idx].d), Int32(n_nodes), _i(reg[].sessions[idx].colid),
            _f(reg[].sessions[idx].quesval), _i(reg[].sessions[idx].left), _f(reg[].sessions[idx].x), row_off,
            _u16(reg[].sessions[idx].nodes), _i(reg[].sessions[idx].bad),
            grid_dim=(_blocks(n), 1, 1), block_dim=(TPB, 1, 1),
        )
        var units = n_nodes * n_chunks
        ctx.enqueue_function[dart_leaf_sum_kernel](
            Int64(units), Int64(n), Int32(n_nodes), row_off, class_off, _u16(reg[].sessions[idx].nodes),
            _f(reg[].sessions[idx].target), _f(reg[].sessions[idx].h), _f(reg[].sessions[idx].part),
            grid_dim=(_blocks(units), 1, 1), block_dim=(TPB, 1, 1),
        )
        ctx.enqueue_function[dart_newton_kernel](
            Int64(n_nodes), Int32(n_nodes), Int32(n_chunks), _f(reg[].sessions[idx].part), Float32(lam),
            Float32(l1), Float32(mds), voff, _f(reg[].sessions[idx].values),
            grid_dim=(_blocks(n_nodes), 1, 1), block_dim=(TPB, 1, 1),
        )
        ctx.enqueue_function[dart_add_kernel](
            Int64(n), class_off, row_off, voff, Float32(factor), Float32(shrink), _u16(reg[].sessions[idx].nodes),
            _f(reg[].sessions[idx].values), _f(reg[].sessions[idx].dsum), _f(reg[].sessions[idx].score),
            grid_dim=(_blocks(n), 1, 1), block_dim=(TPB, 1, 1),
        )
        var vsub = reg[].sessions[idx].values.create_sub_buffer[DType.float32](j * node_cap, n_nodes)
        ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=values_out), src_buf=vsub)
    else:
        raise Error("x_trees dart_add: built without DART_DEVICE (MOJOLEARN_DART_DEVICE_OFF / MOJOLEARN_IDN_DART_DEVICE_OFF)")


def dart_close(id: Int, bad_out: Int) raises:
    """Wait for every queued launch and copy (the last tree's leaf values),
    write the bad word, and free the session."""
    comptime if DART_DEVICE:
        var reg = DART_SESSIONS.get_or_create_ptr()
        var idx = reg[].find(id)
        var ctx = process_ctx[_DART_CTX]()
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=bad_out), src_buf=reg[].sessions[idx].bad)
        ctx.synchronize()
        var gone = reg[].sessions.pop(idx)
        _ = gone^
    else:
        raise Error("x_trees dart_close: built without DART_DEVICE (MOJOLEARN_DART_DEVICE_OFF / MOJOLEARN_IDN_DART_DEVICE_OFF)")
