# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S CPU GEMM (phase 5, DEVIATION 5719): `gemm_oracle`'s
answer, bit for bit, at CPU speed.

THE ORACLE'S ARITHMETIC, NOT A NEW ONE. Every output cell is the cell
`gemm.host.gemm_oracle.gemm_oracle` computes for profile
mojolearn.identical.gemm.fp32.v1: leaves at `contract_leaf_size(k)`, each
leaf the chain `acc = ftz(identical_mul_add(ftz(a), ftz(b), acc))` over p
ascending from +0.0, the leaves combined by the fixed balanced tree
(`fold_balanced_tree`: pair (2q, 2q + 1) with `ftz(left + right)`, carry an
odd tail bit for bit). What changes is only everything around that
arithmetic, and each change moves no bit:

  - Operands are flushed ONCE (`ftz` is pure and idempotent, so a value read
    from a flushed copy is the value the oracle's per-read `ftz` returns). A
    [k x n] right operand with no subnormal is read in place: a parallel
    scan decides, and a copy is made only when one exists.
  - The n cells of one output row advance together down one leaf, one SIMD
    lane per cell (`identical_mul_add_simd` is an IEEE fused multiply-add
    per lane, correctly rounded exactly as the scalar seam is), in
    registers, with the flush DEFERRED: the chain runs unflushed while a
    tracker keeps the smallest `|bits| - 1` (unsigned) of every raw result.
    The operands are flushed, so only an accumulator can be subnormal; when
    no raw result was a nonzero subnormal, `ftz` was the identity at every
    step and, by induction from the +0.0 seed, the unflushed chain IS the
    flushed chain. Otherwise the group runs again with the flush at every
    step (the check's `tiny` fixture separates the two, so it proves the
    fallback is taken where it matters). The byte
    LM host (`training/byte_lm_host_kernels.mojo`, DEVIATION 2640) runs the
    same argument, conservatively counting exact zeros as suspects; here a
    zero (ReLU gradients, zero padding) is not one, because `ftz` of a zero
    is that zero.
  - The balanced tree is evaluated as a binary counter: leaf partials are
    pushed in ascending order, two pending nodes of equal size combine
    (older + newer) as soon as both exist, and at the end the pending nodes
    combine from the smallest (newest) up. Node (d, q) of the tree covers
    leaves [q 2^d, min((q + 1) 2^d, P)); a full node is two full children,
    the tail node at each level is (full left child) + (tail) or a carry of
    its left child, which is exactly that right-to-left finish. So every
    addition is the tree's, with the same operands in the same order.
  - Rows split into contiguous tasks (`core/host_predict_threads.mojo`,
    MOJOLEARN_CPU_THREADS); when there are few rows and many leaves the
    LEAVES split instead, into aligned power-of-two chunks: each chunk is a
    subtree whose root is a node of the same tree, and the chunk roots are
    folded by the same rule. A thread count changes which thread computes a
    node, never a node.

`GEMM_ORACLE_HOST_SABOTAGE` (the host family's -D MOJOLEARN_HOST_SABOTAGE
build) routes every product through `gemm_oracle` itself, so the sabotaged
build is the oracle's sabotaged answer exactly as before this file existed.
`x_cnn/checks/gemm_host_check.mojo` compares this file with `gemm_oracle` by
bits on every operation, one leaf and many, both split modes, one thread and
many, with planted subnormals, NaN and infinities."""
from std.math import min
from std.memory import bitcast
from std.sys.info import simd_width_of

from core.host_parallel import host_parallelize

from checks.numerics import ftz, identical_mul_add, identical_mul_add_simd
from core.host_predict_threads import host_predict_task_count
from gemm.contract import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_size,
    leaf_count,
)
from gemm.host.neural_gemm import gemm_oracle
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED, NEURAL_LEAF, NEURAL_CHAINS, neural_cell, neural_partition, neural_strides
from x_cnn.ops import FP

comptime GW = simd_width_of[DType.float32]()
comptime GV = SIMD[DType.float32, GW]
comptime GU = SIMD[DType.uint32, GW]
#: SIMD accumulators advanced together per p step (one output row, CHAINS *
#: GW cells). A schedule knob: every lane runs its own cell's chain.
comptime CHAINS = 8
#: Output rows per tile: a leaf's slab of the right operand is reused across
#: these rows while it is in cache. A schedule knob.
comptime ROW_TILE = 16
#: The fewest multiply-adds worth a second thread.
comptime MIN_TASK_WORK = 1 << 17
#: The largest fold depth: P <= CONTRACT_MAX_LEAVES = 1024 leaves.
comptime MAX_LEVELS = 12
comptime SUB_LO = UInt32(0x007FFFFF)


@always_inline
def ftz_v(x: GV) -> GV:
    """`ftz` on every lane: the signed zero when the exponent field is zero
    (a subnormal flushes to its sign; a zero already is its sign)."""
    var bits = bitcast[DType.uint32](x)
    var sub = (bits & GU(0x7F800000)).eq(GU(0))
    return bitcast[DType.float32](sub.select(bits & GU(0x80000000), bits))


@always_inline
def _suspect(raw: GV) -> GU:
    """`|bits| - 1`, unsigned: below 0x007FFFFF exactly for a nonzero
    subnormal (a zero wraps to the largest word)."""
    return (bitcast[DType.uint32](raw) & GU(0x7FFFFFFF)) - GU(1)


def gemm_tasks(rows: Int, work: Int) -> Int:
    """The policy's task count for `rows` independent units, capped so each
    task has at least MIN_TASK_WORK multiply-adds."""
    var t = host_predict_task_count(rows)
    var cap = work // MIN_TASK_WORK
    if cap < 1:
        cap = 1
    return t if t < cap else cap


def parallel_tasks(total: Int, per_task_min: Int) -> Int:
    """Tasks for an element loop of `total` elements: the policy's count,
    at least `per_task_min` elements each."""
    var cap = total // per_task_min
    if cap < 1:
        cap = 1
    var t = host_predict_task_count(total)
    return t if t < cap else cap


# ------------------------------------------------------------------ operands


def _has_subnormal(p: FP, n: Int) -> Bool:
    """True when any of the n words is a nonzero subnormal (parallel scan)."""
    var tasks = parallel_tasks(n, 1 << 20)
    var chunk = (n + tasks - 1) // tasks
    var flags = List[Int32](length=tasks, fill=Int32(0))
    var fp = flags.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def _scan(c: Int) {imm p, imm fp, imm chunk, imm n}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        var acc = GU(0xFFFFFFFF)
        var i = lo
        while i + GW <= hi:
            acc = min(acc, _suspect(p.unsafe_load[width=GW](i)))
            i += GW
        var found = acc.reduce_min() < SUB_LO
        while i < hi:
            var b = bitcast[DType.uint32](p.unsafe_load(i)) & UInt32(0x7FFFFFFF)
            if b != UInt32(0) and b <= SUB_LO:
                found = True
            i += 1
        fp.unsafe_store(c, Int32(1) if found else Int32(0))

    if tasks == 1:
        _scan(0)
    else:
        host_parallelize(_scan, tasks)
    var any = False
    for c in range(tasks):
        if flags[c] != 0:
            any = True
    _ = flags^
    return any


def _flush_copy(src: FP, dst: FP, n: Int):
    """dst = ftz(src), word by word (parallel)."""
    var tasks = parallel_tasks(n, 1 << 18)
    var chunk = (n + tasks - 1) // tasks

    def _cp(c: Int) {imm src, imm dst, imm chunk, imm n}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        var i = lo
        while i + GW <= hi:
            dst.unsafe_store[width=GW](i, ftz_v(src.unsafe_load[width=GW](i)))
            i += GW
        while i < hi:
            dst.unsafe_store(i, ftz(src.unsafe_load(i)))
            i += 1

    if tasks == 1:
        _cp(0)
    else:
        host_parallelize(_cp, tasks)


def _transpose_flush(src: FP, dst: FP, rows: Int, cols: Int):
    """dst[c * rows + r] = ftz(src[r * cols + c]): a [rows x cols] matrix
    laid out [cols x rows], flushed (parallel over the destination rows)."""
    var tasks = parallel_tasks(rows * cols, 1 << 18)
    if tasks > cols:
        tasks = cols
    var chunk = (cols + tasks - 1) // tasks

    def _tp(t: Int) {imm src, imm dst, imm chunk, imm rows, imm cols}:
        var c0 = t * chunk
        var c1 = min(c0 + chunk, cols)
        # 64-row blocks: the next column reuses the same 64 source lines.
        var r0 = 0
        while r0 < rows:
            var r1 = min(r0 + 64, rows)
            for c in range(c0, c1):
                for r in range(r0, r1):
                    dst.unsafe_store(c * rows + r, ftz(src.unsafe_load(r * cols + c)))
            r0 = r1

    if tasks == 1:
        _tp(0)
    else:
        host_parallelize(_tp, tasks)


# ------------------------------------------------------------------ one leaf


@always_inline
def _leaf_group[V: Int](arow: FP, b: FP, dst: FP, n: Int, jb: Int, p0: Int, p1: Int):
    """Cells [jb, jb + V * GW) of one output row over leaf [p0, p1): V SIMD
    chains with the flush deferred, and the flush-every-step rerun when a
    raw result was a nonzero subnormal (the argument in the module doc)."""
    var acc = InlineArray[GV, V](fill=GV(0))
    var trk = GU(0xFFFFFFFF)
    for p in range(p0, p1):
        var av = GV(arow.unsafe_load(p))
        var base = p * n + jb
        comptime for v in range(V):
            acc[v] = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + v * GW), acc[v])
        comptime for v in range(V):
            trk = min(trk, _suspect(acc[v]))
    if trk.reduce_min() < SUB_LO:
        comptime for v in range(V):
            var cr = GV(0)
            for p in range(p0, p1):
                cr = ftz_v(identical_mul_add_simd[GW](
                    GV(arow.unsafe_load(p)), b.unsafe_load[width=GW](p * n + jb + v * GW), cr))
            dst.unsafe_store[width=GW](jb + v * GW, cr)
    else:
        comptime for v in range(V):
            dst.unsafe_store[width=GW](jb + v * GW, acc[v])


@always_inline
def _leaf_row(arow: FP, b: FP, dst: FP, n: Int, p0: Int, p1: Int):
    """dst[j] = the leaf [p0, p1) partial of every cell j of one output row:
    `arow` is the row's flushed A values indexed by p, `b` the flushed
    [k x n] right operand. The oracle's `oracle_leaf_partial` per cell."""
    var group = CHAINS * GW
    var jb = 0
    while jb + group <= n:
        var c0 = GV(0)
        var c1 = GV(0)
        var c2 = GV(0)
        var c3 = GV(0)
        var c4 = GV(0)
        var c5 = GV(0)
        var c6 = GV(0)
        var c7 = GV(0)
        var trk = GU(0xFFFFFFFF)
        for p in range(p0, p1):
            var av = GV(arow.unsafe_load(p))
            var base = p * n + jb
            c0 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base), c0)
            c1 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + GW), c1)
            c2 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + 2 * GW), c2)
            c3 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + 3 * GW), c3)
            c4 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + 4 * GW), c4)
            c5 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + 5 * GW), c5)
            c6 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + 6 * GW), c6)
            c7 = identical_mul_add_simd[GW](av, b.unsafe_load[width=GW](base + 7 * GW), c7)
            trk = min(
                trk,
                min(
                    min(min(_suspect(c0), _suspect(c1)), min(_suspect(c2), _suspect(c3))),
                    min(min(_suspect(c4), _suspect(c5)), min(_suspect(c6), _suspect(c7))),
                ),
            )
        if trk.reduce_min() < SUB_LO:
            # A nonzero subnormal raw result: this group again, flushed at
            # every step (the oracle's chain as written).
            var jr = jb
            while jr < jb + group:
                var cr = GV(0)
                for p in range(p0, p1):
                    cr = ftz_v(identical_mul_add_simd[GW](
                        GV(arow.unsafe_load(p)), b.unsafe_load[width=GW](p * n + jr), cr))
                dst.unsafe_store[width=GW](jr, cr)
                jr += GW
        else:
            dst.unsafe_store[width=GW](jb, c0)
            dst.unsafe_store[width=GW](jb + GW, c1)
            dst.unsafe_store[width=GW](jb + 2 * GW, c2)
            dst.unsafe_store[width=GW](jb + 3 * GW, c3)
            dst.unsafe_store[width=GW](jb + 4 * GW, c4)
            dst.unsafe_store[width=GW](jb + 5 * GW, c5)
            dst.unsafe_store[width=GW](jb + 6 * GW, c6)
            dst.unsafe_store[width=GW](jb + 7 * GW, c7)
        jb += group
    # The row's remaining cells in smaller groups (4, 2, 1 vectors), each
    # with the same deferred flush: independent chains in flight instead of
    # one flushed chain per vector (n = 16 .. 63 is every small conv's OC).
    if jb + 4 * GW <= n:
        _leaf_group[4](arow, b, dst, n, jb, p0, p1)
        jb += 4 * GW
    if jb + 2 * GW <= n:
        _leaf_group[2](arow, b, dst, n, jb, p0, p1)
        jb += 2 * GW
    if jb + GW <= n:
        _leaf_group[1](arow, b, dst, n, jb, p0, p1)
        jb += GW
    while jb < n:
        var cs = Float32(0)
        for p in range(p0, p1):
            cs = ftz(identical_mul_add(arow.unsafe_load(p), b.unsafe_load(p * n + jb), cs))
        dst.unsafe_store(jb, cs)
        jb += 1


@always_inline
def _add_rows(left: FP, right: FP, dst: FP, n: Int):
    """dst[j] = ftz(left[j] + right[j]): one arithmetic node of the tree,
    older (lower leaves) on the left."""
    var j = 0
    while j + GW <= n:
        dst.unsafe_store[width=GW](j, ftz_v(left.unsafe_load[width=GW](j) + right.unsafe_load[width=GW](j)))
        j += GW
    while j < n:
        dst.unsafe_store(j, ftz(left.unsafe_load(j) + right.unsafe_load(j)))
        j += 1


@always_inline
def _push(stack: FP, carry: FP, cnt: Int, n: Int):
    """Push the next leaf (or subtree) row `carry` onto the binary counter
    `stack` ([MAX_LEVELS x n]; slot d pending when bit d of cnt is set).
    `carry` is scratch and is consumed."""
    var d = 0
    while (cnt >> d) & 1 == 1:
        _add_rows(stack + d * n, carry, carry, n)
        d += 1
    for j in range(n):
        stack.unsafe_store(d * n + j, carry.unsafe_load(j))


@always_inline
def _finish(stack: FP, cnt: Int, dst: FP, n: Int):
    """The tree's root from the counter after `cnt` pushes: the pending
    nodes combined from the newest (smallest) up, older on the left."""
    var d = 0
    while (cnt >> d) & 1 == 0:
        d += 1
    for j in range(n):
        dst.unsafe_store(j, stack.unsafe_load(d * n + j))
    d += 1
    while (cnt >> d) != 0:
        if (cnt >> d) & 1 == 1:
            _add_rows(stack + d * n, dst, dst, n)
        d += 1


# ------------------------------------------------------------------ kernels


def _rows_mode(a: FP, b: FP, c: FP, m: Int, n: Int, k: Int, tasks: Int):
    """Rows split across tasks; every row's whole tree in one task."""
    var leaf = contract_leaf_size(k)
    var pcount = leaf_count(k, leaf)
    var chunk = (m + tasks - 1) // tasks

    def _task(t: Int) {imm a, imm b, imm c, imm n, imm k, imm m, imm chunk, imm leaf, imm pcount}:
        var lo = t * chunk
        var hi = min(lo + chunk, m)
        if pcount == 1:
            for i in range(lo, hi):
                _leaf_row(a + i * k, b, c + i * n, n, 0, k)
            return
        var stack = List[Float32](length=ROW_TILE * MAX_LEVELS * n, fill=Float32(0))
        var part = List[Float32](length=n, fill=Float32(0))
        var sp = stack.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var r0 = lo
        while r0 < hi:
            var r1 = min(r0 + ROW_TILE, hi)
            for lt in range(pcount):
                var p0 = lt * leaf
                var p1 = min(p0 + leaf, k)
                for i in range(r0, r1):
                    _leaf_row(a + i * k, b, pp, n, p0, p1)
                    _push(sp + (i - r0) * MAX_LEVELS * n, pp, lt, n)
            for i in range(r0, r1):
                _finish(sp + (i - r0) * MAX_LEVELS * n, pcount, c + i * n, n)
            r0 = r1
        _ = stack^
        _ = part^

    if tasks <= 1:
        _task(0)
    else:
        host_parallelize(_task, tasks)


def _leaves_mode(a: FP, b: FP, c: FP, m: Int, n: Int, k: Int, tasks: Int):
    """Few rows, many leaves: aligned power-of-two leaf chunks across tasks.
    Chunk q's root is the tree's node (s, q) for chunk size 2^s, so the
    chunk roots, folded in ascending order by the same counter, are the
    tree's root."""
    var leaf = contract_leaf_size(k)
    var pcount = leaf_count(k, leaf)
    var span = 1
    while span * tasks < pcount:
        span *= 2
    var chunks = (pcount + span - 1) // span
    var mn = m * n
    var roots = List[Float32](length=chunks * mn, fill=Float32(0))
    var rp = roots.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def _task(q: Int) {imm a, imm b, imm rp, imm m, imm n, imm k, imm leaf, imm pcount, imm span, imm mn}:
        var l0 = q * span
        var l1 = min(l0 + span, pcount)
        var stack = List[Float32](length=m * MAX_LEVELS * n, fill=Float32(0))
        var part = List[Float32](length=n, fill=Float32(0))
        var sp = stack.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for lt in range(l0, l1):
            var p0 = lt * leaf
            var p1 = min(p0 + leaf, k)
            for i in range(m):
                _leaf_row(a + i * k, b, pp, n, p0, p1)
                _push(sp + i * MAX_LEVELS * n, pp, lt - l0, n)
        for i in range(m):
            _finish(sp + i * MAX_LEVELS * n, l1 - l0, rp + q * mn + i * n, n)
        _ = stack^
        _ = part^

    host_parallelize(_task, chunks)
    if chunks == 1:
        for x in range(mn):
            c.unsafe_store(x, rp.unsafe_load(x))
        _ = roots^
        return
    var stack = List[Float32](length=MAX_LEVELS * n, fill=Float32(0))
    var part = List[Float32](length=n, fill=Float32(0))
    var sp = stack.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for i in range(m):
        for q in range(chunks):
            for j in range(n):
                pp.unsafe_store(j, rp.unsafe_load(q * mn + i * n + j))
            _push(sp, pp, q, n)
        _finish(sp, chunks, c + i * n, n)
    _ = stack^
    _ = part^
    _ = roots^


def gemm_host_split(a: FP, b: FP, c: FP, m: Int, n: Int, k: Int, tasks: Int, leaves: Bool):
    """The kernels on prepared operands (`a` [m x k] and `b` [k x n], both
    flushed), at an explicit task count and split mode: the check's door."""
    if k <= 0:
        for x in range(m * n):
            c.unsafe_store(x, Float32(0))
        return
    var pcount = leaf_count(k, contract_leaf_size(k))
    if leaves and pcount > 1 and tasks > 1:
        _leaves_mode(a, b, c, m, n, k, tasks)
    else:
        _rows_mode(a, b, c, m, n, k, tasks)


def gemm_host_into(
    a: FP, b: FP, c: FP, op: Int, m: Int, n: Int, k: Int, tasks_override: Int = 0, leaves_override: Int = -1
):
    """C [m x n] = op(A, B) under mojolearn.identical.gemm.fp32.v1, written
    to `c`: OP_NT A [m x k] . B [n x k]^T, OP_TN A [k x m]^T . B [k x n],
    OP_NN A [m x k] . B [k x n] (gemm_oracle's operand layouts).
    `tasks_override` > 0 and `leaves_override` (0 rows, 1 leaves) pin the
    schedule: the check's doors, never a different answer."""
    if m <= 0 or n <= 0:
        return
    comptime if GEMM_ORACLE_HOST_SABOTAGE:
        var la = List[Float32](length=m * k if m * k > 0 else 1, fill=Float32(0))
        var lb = List[Float32](length=n * k if n * k > 0 else 1, fill=Float32(0))
        for x in range(m * k):
            la[x] = a.unsafe_load(x)
        for x in range(n * k):
            lb[x] = b.unsafe_load(x)
        var r = gemm_oracle(la, lb, op, m, n, k)
        for x in range(m * n):
            c.unsafe_store(x, r[x])
        return
    comptime if NEURAL_PROFILE_CHANGED:
        var part = neural_partition[NEURAL_LEAF](k)
        var strides = neural_strides(op, m, n, k)
        var tasks = tasks_override if tasks_override > 0 else gemm_tasks(m, m * n * k)
        def selected_rows(task: Int) {imm a, imm b, imm c, imm m, imm n, imm k, imm part, imm strides, imm tasks}:
            for row in range(task * m // tasks, (task + 1) * m // tasks):
                for col in range(n):
                    c.unsafe_store(row * n + col, neural_cell[NEURAL_CHAINS](a, b, row, col, k, part[0], part[1], strides[0], strides[1], strides[2], strides[3]))
        host_parallelize(selected_rows, tasks)
        return
    var work = m * n * k
    var pcount = leaf_count(k, contract_leaf_size(k))
    var tasks = tasks_override
    var leaves = False
    if tasks <= 0:
        tasks = gemm_tasks(m, work)
        # Few rows, many leaves: split the leaves instead.
        if pcount > 1 and m < 4 * host_predict_task_count(pcount):
            var lt = gemm_tasks(pcount, work)
            if lt > tasks:
                tasks = lt
                leaves = True
    # A: [m x k], flushed.
    var aw = List[Float32]()
    var ap = a
    if op == OP_TN:
        aw = List[Float32](length=m * k, fill=Float32(0))
        ap = aw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        _transpose_flush(a, ap, k, m)
    elif _has_subnormal(a, m * k):
        aw = List[Float32](length=m * k, fill=Float32(0))
        ap = aw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        _flush_copy(a, ap, m * k)
    # B: [k x n], flushed.
    var bw = List[Float32]()
    var bp = b
    if op == OP_NT:
        bw = List[Float32](length=n * k, fill=Float32(0))
        bp = bw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        _transpose_flush(b, bp, n, k)
    elif _has_subnormal(b, n * k):
        bw = List[Float32](length=n * k, fill=Float32(0))
        bp = bw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        _flush_copy(b, bp, n * k)
    if leaves_override >= 0:
        leaves = leaves_override == 1
    if tasks > 1 and not leaves and tasks > m:
        tasks = m
    gemm_host_split(ap, bp, c, m, n, k, tasks, leaves)
    _ = aw^
    _ = bw^
