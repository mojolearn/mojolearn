# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The embedding backward the tape calls: identical_embedding_backward_into and
its prerefused and refusing entries, the sorted-run scatter fold and its
kernels. Moved unchanged out of embedding/checks/embedding_identical.mojo,
which keeps the forward gather, the id and nonfinite refusals both
directions share, the plan and the sabotage switches."""

from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT, embedding_sort_runs
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from core.step_phase import step_count_d2h, step_count_host_alloc, step_count_launch, step_count_sync
from checks.numerics import ftz
from checks.kernel_matrix import IDENTITY_FLOOR_BLOCK, TARGET_COLUMN, column_max_block_size
from embedding.checks.embedding_oracle import EMB_NO_PADDING_IDX, EmbConfig, emb_refuse_shape
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from embedding.checks.embedding_identical import (
    EMB_RUN_BEGIN_THREADS,
    EMB_TOUCHED_ROWS,
    EMB_TPB,
    EMB_UNIQUE_ROWS,
    EMB_UNIQ_TPB,
    NN49_EMB_VECTOR4,
    SAB_ACCUM_REFILLS,
    SAB_EMPTY_ROW_NEG_ZERO,
    SAB_EMPTY_ROW_SKIPPED,
    SAB_FOLD_BALANCED_TREE,
    SAB_FOLD_DESCENDING,
    SAB_FOLD_READS_LAUNCH,
    SAB_FOLD_VIA_GEMM_ONEHOT,
    SAB_NO_FLUSH_ACC,
    SAB_PAD_ROW_CONTRIBUTES,
    SAB_PAD_ROW_NEG_ZERO,
    SAB_RANK_BY_ARRIVAL,
    SAB_SEED_SEEDLESS,
    SAB_SINGLE_RUN_BYPASS,
    SAB_SORT_TIE_REVERSED,
    _grid_for,
    emb_refuse_device_ids,
    emb_refuse_device_nonfinite,
)


def emb_counts_kernel(
    counts: MutPointer[Int32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    vocab_in: Int32,
    padding_idx_in: Int32,
):
    """`counts[v]` = the number of positions carrying `v`. One thread per `v`, walking `t` ASCENDING."""
    var vocab = Int(vocab_in)
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v >= vocab:
        return
    var n_positions = Int(n_positions_in)
    var pad = Int(padding_idx_in)
    var c = Int32(0)
    comptime if SAB_PAD_ROW_CONTRIBUTES:
        pad = EMB_NO_PADDING_IDX
    if v == pad:
        counts.unsafe_store(v, Int32(0))
        return
    for t in range(n_positions):
        if Int(ids.unsafe_load(t)) == v:
            c = c + Int32(1)
    counts.unsafe_store(v, c)


def emb_run_begin_kernel(
    run_begin: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
):
    """The exclusive prefix sum of `counts`, length `V + 1`."""
    var vocab = Int(vocab_in)
    var acc = Int32(0)
    for v in range(vocab):
        run_begin.unsafe_store(v, acc)
        acc = acc + counts.unsafe_load(v)
    run_begin.unsafe_store(vocab, acc)


def emb_run_begin_block_kernel(
    run_begin: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
):
    """`emb_run_begin_kernel`'s exclusive prefix sum of `counts`, by ONE
    block of `EMB_RUN_BEGIN_THREADS` threads instead of one thread: thread
    `t` sums its contiguous chunk, thread 0 scans the chunk sums, each
    thread writes its chunk's prefixes. Integer addition is exactly
    associative (contract 6.1; the counts sum to the position count, far
    below 2^31), so every `run_begin` word equals the serial kernel's. At the
    T3 vocabulary (50,257) the serial kernel took 2.05 ms a shard on an
    H100 (lane/nvidia-step-time leg 1 nsys)."""
    comptime NT = EMB_RUN_BEGIN_THREADS
    var part = stack_allocation[NT, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var vocab = Int(vocab_in)
    var t = Int(thread_idx.x)
    var chunk = (vocab + NT - 1) // NT
    var lo = t * chunk
    var hi = min(lo + chunk, vocab)
    var sum = Int32(0)
    for v in range(lo, hi):
        sum = sum + counts.unsafe_load(v)
    part.unsafe_store(t, sum)
    barrier()
    if t == 0:
        var acc = Int32(0)
        for i in range(NT):
            var x = part.unsafe_load(i)
            part.unsafe_store(i, acc)
            acc = acc + x
        run_begin.unsafe_store(vocab, acc)
    barrier()
    var run = part.unsafe_load(t)
    for v in range(lo, hi):
        run_begin.unsafe_store(v, run)
        run = run + counts.unsafe_load(v)


def emb_perm_kernel(
    perm: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    vocab_in: Int32,
    padding_idx_in: Int32,
):
    """`perm[run_begin[v] + r]` = the `r`-th position carrying `v`, ASCENDING `t`. **NO SORT, NO KEY, NO TIE CLASS, NO STABILITY QUESTION.** Each `v` owns its own region of `perm` and appends in `t` order, so the within-run ranks come out ascending BY CONSTRUCTION rather than by arrival."""
    var vocab = Int(vocab_in)
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v >= vocab:
        return
    var n_positions = Int(n_positions_in)
    var pad = Int(padding_idx_in)
    comptime if SAB_PAD_ROW_CONTRIBUTES:
        pad = EMB_NO_PADDING_IDX
    if v == pad:
        return
    var w = Int(run_begin.unsafe_load(v))
    var hi = Int(run_begin.unsafe_load(v + 1))
    if w >= hi:
        return

    comptime if SAB_SORT_TIE_REVERSED:
        var back = hi - 1
        for t in range(n_positions):
            if Int(ids.unsafe_load(t)) == v:
                perm.unsafe_store(back, Int32(t))
                back -= 1
        return

    comptime if SAB_RANK_BY_ARRIVAL:
        var nth = Int(block_dim.x)
        var r = 0
        var phase = 0
        while phase < 2:
            for t in range(n_positions):
                if Int(ids.unsafe_load(t)) == v:
                    var side = 1 if (t // nth) % 2 == 1 else 0
                    if side == phase:
                        perm.unsafe_store(w + r, Int32(t))
                        r += 1
            phase += 1
        return

    for t in range(n_positions):
        if Int(ids.unsafe_load(t)) == v:
            perm.unsafe_store(w, Int32(t))
            w += 1


def emb_seed_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
):
    """`dW[v, j] = +0.0`, every one of the `V * d` cells. **THE FILL IS NOT AN IMPLEMENTATION DETAIL AND IT MAY NOT BE SKIPPED.** Contract 5.5 -- the empty run's value is STATED, not derived, and an implementation must WRITE it rather than leave whatever was in the buffer."""
    var cells = Int(cells_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= cells:
        return
    comptime if SAB_EMPTY_ROW_SKIPPED:
        return
    comptime if SAB_EMPTY_ROW_NEG_ZERO:
        dw.unsafe_store(i, Float32(-0.0))
        return
    dw.unsafe_store(i, Float32(0.0))


def emb_pad_row_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    width_in: Int32,
    padding_idx_in: Int32,
):
    """`dW[padding_idx, :] = +0.0`, STORED."""
    var width = Int(width_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= width:
        return
    var base = Int(padding_idx_in) * width
    comptime if SAB_PAD_ROW_NEG_ZERO:
        dw.unsafe_store(base + j, Float32(-0.0))
        return
    dw.unsafe_store(base + j, Float32(0.0))


def emb_backward_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    perm: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
    width_in: Int32,
):
    """**THE CONTRACT'S FOLD**, contract 5.1. One thread owns one `(v, j)` cell, walks its own run ASCENDING and adds."""
    var width = Int(width_in)
    if width < 1:
        return
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= vocab * width:
        return
    var v = cell // width
    var j = cell - v * width

    var lo = Int(run_begin.unsafe_load(v))
    var hi = Int(run_begin.unsafe_load(v + 1))
    if lo >= hi:
        return

    var acc = dw.unsafe_load(cell)

    comptime if SAB_SEED_SEEDLESS:
        acc = ftz(dy.unsafe_load(Int(perm.unsafe_load(lo)) * width + j))
        for r in range(lo + 1, hi):
            var t2 = Int(perm.unsafe_load(r))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(t2 * width + j)))
        dw.unsafe_store(cell, ftz(acc))
        return

    comptime if SAB_SINGLE_RUN_BYPASS:
        if hi - lo == 1:
            var t1 = Int(perm.unsafe_load(lo))
            dw.unsafe_store(cell, ftz(dy.unsafe_load(t1 * width + j)))
            return

    comptime if SAB_FOLD_DESCENDING:
        var rdx = hi - 1
        while rdx >= lo:
            var td = Int(perm.unsafe_load(rdx))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(td * width + j)))
            rdx -= 1
        dw.unsafe_store(cell, ftz(acc))
        return

    comptime if SAB_FOLD_READS_LAUNCH:
        var span = hi - lo
        var start = Int(block_dim.x) % span
        for qL in range(span):
            var rr = lo + (start + qL) % span
            var tr = Int(perm.unsafe_load(rr))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(tr * width + j)))
        dw.unsafe_store(cell, ftz(acc))
        return

    comptime if SAB_FOLD_BALANCED_TREE:
        var span2 = hi - lo
        var pairs = span2 // 2
        for qT in range(pairs):
            var ta = Int(perm.unsafe_load(lo + 2 * qT))
            var tb = Int(perm.unsafe_load(lo + 2 * qT + 1))
            var pair_sum = ftz(
                ftz(dy.unsafe_load(ta * width + j))
                + ftz(dy.unsafe_load(tb * width + j))
            )
            acc = ftz(ftz(acc) + pair_sum)
        if span2 % 2 != 0:
            var tt = Int(perm.unsafe_load(hi - 1))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(tt * width + j)))
        dw.unsafe_store(cell, ftz(acc))
        return

    for r in range(lo, hi):
        var t = Int(perm.unsafe_load(r))
        var contribution = dy.unsafe_load(t * width + j)
        comptime if SAB_NO_FLUSH_ACC:
            acc = acc + contribution
        else:
            acc = ftz(ftz(acc) + ftz(contribution))
    dw.unsafe_store(cell, ftz(acc))


def emb_backward_touched_vector4_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    perm: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32, vocab_in: Int32, width_in: Int32,
):
    var width = Int(width_in)
    var groups = (width + 3) // 4
    var task = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if width < 1 or task >= Int(n_positions_in) * groups:
        return
    var r = task // groups
    var first_feature = (task - r * groups) * 4
    if r >= Int(run_begin[Int(vocab_in)]):
        return
    var v = Int(ids[Int(perm[r])])
    var lo = Int(run_begin[v])
    if r != lo:
        return
    var hi = Int(run_begin[v + 1])
    var acc = InlineArray[Float32, 4](fill=Float32(0.0))
    comptime for lane in range(4):
        if first_feature + lane < width:
            acc[lane] = dw[v * width + first_feature + lane]
    for q in range(lo, hi):
        var t = Int(perm[q])
        comptime for lane in range(4):
            if first_feature + lane < width:
                acc[lane] = ftz(ftz(acc[lane]) + ftz(dy[t * width + first_feature + lane]))
    comptime for lane in range(4):
        if first_feature + lane < width:
            dw[v * width + first_feature + lane] = ftz(acc[lane])


def emb_backward_touched_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    perm: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    vocab_in: Int32,
    width_in: Int32,
):
    """`emb_backward_kernel`'s clean fold, one thread per (sorted slot r,
    column j); only the slot that opens a run folds it."""
    var width = Int(width_in)
    if width < 1:
        return
    var n_positions = Int(n_positions_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n_positions * width:
        return
    var r = cell // width
    var j = cell - r * width
    var vocab = Int(vocab_in)
    if r >= Int(run_begin.unsafe_load(vocab)):
        return
    var v = Int(ids.unsafe_load(Int(perm.unsafe_load(r))))
    var lo = Int(run_begin.unsafe_load(v))
    if r != lo:
        return
    var hi = Int(run_begin.unsafe_load(v + 1))
    var out = v * width + j
    var acc = dw.unsafe_load(out)
    for q in range(lo, hi):
        var t = Int(perm.unsafe_load(q))
        acc = ftz(ftz(acc) + ftz(dy.unsafe_load(t * width + j)))
    dw.unsafe_store(out, ftz(acc))


def emb_touched_count_kernel(
    part: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
):
    """part[b] = the touched rows among block b's EMB_UNIQ_TPB rows (an
    integer tree sum: exact, any order)."""
    comptime NT = EMB_UNIQ_TPB
    var sh = stack_allocation[NT, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var v = Int(block_idx.x) * NT + t
    var flag = Int32(0)
    if v < Int(vocab_in) and run_begin.unsafe_load(v + 1) > run_begin.unsafe_load(v):
        flag = Int32(1)
    sh.unsafe_store(t, flag)
    barrier()
    var stride = NT // 2
    while stride > 0:
        if t < stride:
            sh.unsafe_store(t, sh.unsafe_load(t) + sh.unsafe_load(t + stride))
        barrier()
        stride //= 2
    if t == 0:
        part.unsafe_store(Int(block_idx.x), sh.unsafe_load(0))


def emb_touched_scan_kernel(
    part: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    nparts_in: Int32,
):
    """The exclusive prefix sum of part[0:nparts] in place and the total in
    total[0], by ONE block of EMB_UNIQ_TPB threads over the per-block counts
    (V / EMB_UNIQ_TPB words: `emb_run_begin_block_kernel`'s chunked form)."""
    comptime NT = EMB_UNIQ_TPB
    var sums = stack_allocation[NT, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var n = Int(nparts_in)
    var t = Int(thread_idx.x)
    var chunk = (n + NT - 1) // NT
    var lo = t * chunk
    var hi = min(lo + chunk, n)
    var sum = Int32(0)
    for i in range(lo, hi):
        sum = sum + part.unsafe_load(i)
    sums.unsafe_store(t, sum)
    barrier()
    if t == 0:
        var acc = Int32(0)
        for i in range(NT):
            var x = sums.unsafe_load(i)
            sums.unsafe_store(i, acc)
            acc = acc + x
        total.unsafe_store(0, acc)
    barrier()
    var run = sums.unsafe_load(t)
    for i in range(lo, hi):
        var x = part.unsafe_load(i)
        part.unsafe_store(i, run)
        run = run + x


def emb_touched_rows_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    part: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
):
    """rows[part[b] + (touched rows of block b before v)] = v for every
    touched v of block b: the touched rows, ascending. The in-block rank is
    a Hillis-Steele inclusive scan of the flags over two shared buffers."""
    comptime NT = EMB_UNIQ_TPB
    var a = stack_allocation[NT, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var c = stack_allocation[NT, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var v = Int(block_idx.x) * NT + t
    var flag = Int32(0)
    if v < Int(vocab_in) and run_begin.unsafe_load(v + 1) > run_begin.unsafe_load(v):
        flag = Int32(1)
    a.unsafe_store(t, flag)
    barrier()
    var off = 1
    var src_is_a = True
    while off < NT:
        if src_is_a:
            var x = a.unsafe_load(t)
            if t >= off:
                x = x + a.unsafe_load(t - off)
            c.unsafe_store(t, x)
        else:
            var y = c.unsafe_load(t)
            if t >= off:
                y = y + c.unsafe_load(t - off)
            a.unsafe_store(t, y)
        barrier()
        src_is_a = not src_is_a
        off *= 2
    var incl = a.unsafe_load(t) if src_is_a else c.unsafe_load(t)
    if flag != Int32(0):
        rows.unsafe_store(Int(part.unsafe_load(Int(block_idx.x)) + incl - Int32(1)), Int32(v))


def emb_backward_rows_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    perm: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    width_in: Int32,
):
    """`emb_backward_kernel`'s clean fold, one thread per (touched row
    rows[k], column j)."""
    var width = Int(width_in)
    if width < 1:
        return
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_rows_in) * width:
        return
    var k = cell // width
    var j = cell - k * width
    var v = Int(rows.unsafe_load(k))
    var lo = Int(run_begin.unsafe_load(v))
    var hi = Int(run_begin.unsafe_load(v + 1))
    var out = v * width + j
    var acc = dw.unsafe_load(out)
    for r in range(lo, hi):
        var t = Int(perm.unsafe_load(r))
        acc = ftz(ftz(acc) + ftz(dy.unsafe_load(t * width + j)))
    dw.unsafe_store(out, ftz(acc))


def _emb_unique_rows_fold(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut run_begin: DeviceBuffer[DType.int32],
    mut perm: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
    block_threads: Int,
) raises -> Bool:
    """EMB_UNIQUE_ROWS: compacts the touched rows, reads their count U back
    and, when U < min(T, V), launches the fold over U * d threads and waits
    for it (the scratch lives to that wait). Returns False (nothing
    launched beyond the compaction) when U = min(T, V): the caller's launch
    then runs, the same work."""
    var nb = (cfg.vocab + EMB_UNIQ_TPB - 1) // EMB_UNIQ_TPB
    # three separate buffers (rows, the per-block counts, the total): one
    # launch never receives two mutable pointers into one buffer
    var rows = ctx.enqueue_create_buffer[DType.int32](cfg.vocab)
    var part = ctx.enqueue_create_buffer[DType.int32](nb)
    var total = ctx.enqueue_create_buffer[DType.int32](1)
    step_count_launch()
    ctx.enqueue_function[emb_touched_count_kernel](
        part.unsafe_ptr(), run_begin.unsafe_ptr(), Int32(cfg.vocab),
        grid_dim=(nb, 1, 1), block_dim=(EMB_UNIQ_TPB, 1, 1),
    )
    step_count_launch()
    ctx.enqueue_function[emb_touched_scan_kernel](
        part.unsafe_ptr(), total.unsafe_ptr(), Int32(nb),
        grid_dim=(1, 1, 1), block_dim=(EMB_UNIQ_TPB, 1, 1),
    )
    step_count_launch()
    ctx.enqueue_function[emb_touched_rows_kernel](
        rows.unsafe_ptr(), part.unsafe_ptr(), run_begin.unsafe_ptr(), Int32(cfg.vocab),
        grid_dim=(nb, 1, 1), block_dim=(EMB_UNIQ_TPB, 1, 1),
    )
    step_count_host_alloc()
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=total)
    step_count_sync()
    ctx.synchronize()
    var u = Int(h.unsafe_ptr().unsafe_load(0))
    _ = h^
    if u >= min(n_positions, cfg.vocab):
        _ = rows^
        _ = part^
        _ = total^
        return False
    if u > 0:
        step_count_launch()
        ctx.enqueue_function[emb_backward_rows_kernel](
            dw.unsafe_ptr(), dy.unsafe_ptr(), perm.unsafe_ptr(), run_begin.unsafe_ptr(),
            rows.unsafe_ptr(), Int32(u), Int32(cfg.width),
            grid_dim=(_grid_for(u * cfg.width, block_threads), 1, 1),
            block_dim=(block_threads, 1, 1),
        )
        step_count_sync()
        ctx.synchronize()
    _ = rows^
    _ = part^
    _ = total^
    return True


def emb_onehot_kernel(
    onehot: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    vocab_in: Int32,
    padding_idx_in: Int32,
):
    """SABOTAGE `EMB_FOLD_VIA_GEMM_ONEHOT` only: `A[t, v] = 1.0` where `ids[t] == v` and `v != padding_idx`, else `+0.0`, row-major `[T, V]`. The padding position is dropped at the source here too (contract section 8), so the arm differs from the normative fold ONLY in the arithmetic route."""
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_positions_in) * vocab:
        return
    var t = cell // vocab
    var v = cell - t * vocab
    var id = Int(ids.unsafe_load(t))
    if id == v and id != Int(padding_idx_in):
        onehot.unsafe_store(cell, Float32(1.0))
    else:
        onehot.unsafe_store(cell, Float32(0.0))


def emb_onehot_store_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    product: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    accumulate_in: Int32,
):
    """SABOTAGE `EMB_FOLD_VIA_GEMM_ONEHOT` only: store the GEMM product. A one-hot GEMM cannot be SEEDED from a carried `dW`, so under `accumulate` the arm is the ADD spelling of contract 7.4, `ftz(ftz(dW_prev) + ftz(product))`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(cells_in):
        return
    var got = product.unsafe_load(i)
    if accumulate_in != Int32(0):
        dw.unsafe_store(i, ftz(ftz(dw.unsafe_load(i)) + ftz(got)))
    else:
        dw.unsafe_store(i, ftz(got))


def _emb_fold_via_gemm_onehot(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
    block_threads: Int,
) raises:
    """SABOTAGE `EMB_FOLD_VIA_GEMM_ONEHOT` (contract 5.2(d), DEVIATION 1315): `dW = A^T . dY` through the certified `identical_gemm` entry point, `op = OP_TN`, `m = V`, `n = d`, `k = T`. GEMM v1 pins its `k` fold as leaves of `contract_leaf_size(T)` under a balanced tree, so the arithmetic is a function of `T`, a LAUNCH quantity, and it moves once `T > 128` gives more than one leaf. The GEMM call sits in `_emb_onehot_gemm_call` with a function-local import; the compiler still elaborates that import on every build of this file (the clean embedding check reports `gemm_identical.mojo` warnings since 2026-09-15), but only this arm's build calls it."""
    var cells = cfg.vocab * cfg.width
    var onehot = ctx.enqueue_create_buffer[DType.float32](n_positions * cfg.vocab)
    var product = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.enqueue_function[emb_onehot_kernel](
        onehot.unsafe_ptr(),
        ids.unsafe_ptr(),
        Int32(n_positions),
        Int32(cfg.vocab),
        Int32(cfg.padding_idx),
        grid_dim=(_grid_for(n_positions * cfg.vocab, block_threads), 1, 1),
        block_dim=(block_threads, 1, 1),
    )
    ctx.synchronize()
    _emb_onehot_gemm_call(ctx, product, onehot, dy, cfg.vocab, cfg.width, n_positions)
    var acc_code = Int32(0)
    if cfg.accumulate:
        acc_code = Int32(1)
    ctx.enqueue_function[emb_onehot_store_kernel](
        dw.unsafe_ptr(),
        product.unsafe_ptr(),
        Int32(cells),
        acc_code,
        grid_dim=(_grid_for(cells, block_threads), 1, 1),
        block_dim=(block_threads, 1, 1),
    )
    ctx.synchronize()
    _ = onehot^
    _ = product^


def _emb_onehot_gemm_call(
    ctx: DeviceContext,
    mut product: DeviceBuffer[DType.float32],
    mut onehot: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    vocab: Int,
    width: Int,
    n_positions: Int,
) raises:
    """The one GEMM call of the one-hot arm. On its own because with the import local to the launching function the next `enqueue_function` became an ambiguous call (the extension candidates arrived twice)."""
    from gemm.checks.gemm_identical import identical_gemm
    from gemm.contract import OP_TN

    identical_gemm(ctx, product, onehot, dy, vocab, width, n_positions, OP_TN)


def emb_run_scratch_ints(vocab: Int, n_positions: Int) -> Int:
    """Integers of run scratch `identical_embedding_backward_into` needs."""
    var need = vocab + vocab + 1 + n_positions
    if need < 1:
        return 1
    return need


def identical_embedding_backward_prerefused_into(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    mut counts: DeviceBuffer[DType.int32],
    mut run_begin: DeviceBuffer[DType.int32],
    mut perm: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
    plan: Int = PLAN_SCAN,
    block_threads: Int = EMB_TPB,
    unique_rows: Bool = False,
) raises:
    """Seams E0 through E4 for an owning caller that already refused the
    exact immutable ids by host-list admission or an owned device token scan;
    `identical_embedding_backward_into` without the
    device id read-back. `counts`, `run_begin` and `perm` need no initial
    value: both plans write every cell the fold reads. `unique_rows` opts
    into EMB_UNIQUE_ROWS (one extra wait; the same bits)."""
    _emb_backward_refuse_launch(ctx, plan, block_threads)
    _emb_backward_launch(ctx, dw, dy, ids, counts, run_begin, perm, n_positions, cfg, plan, block_threads,
                         unique_rows=unique_rows)


def identical_embedding_backward_into(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    mut counts: DeviceBuffer[DType.int32],
    mut run_begin: DeviceBuffer[DType.int32],
    mut perm: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
    plan: Int = PLAN_SCAN,
    block_threads: Int = EMB_TPB,
) raises:
    """Seams E0 through E4, enqueued. **`counts`, `run_begin` and `perm` ARE THE CALLER'S** and must be sized with `emb_run_scratch_ints` and kept alive past the caller's own `ctx.synchronize()`."""
    _emb_backward_refuse_launch(ctx, plan, block_threads)
    emb_refuse_device_ids(ctx, ids, n_positions, cfg)
    _emb_backward_launch(ctx, dw, dy, ids, counts, run_begin, perm, n_positions, cfg, plan, block_threads)


def _emb_backward_refuse_launch(ctx: DeviceContext, plan: Int, block_threads: Int) raises:
    if plan != PLAN_SCAN and plan != PLAN_SORT:
        raise Error("embedding: unknown execution plan")
    if block_threads < 1 or block_threads > IDENTITY_FLOOR_BLOCK or block_threads > column_max_block_size(TARGET_COLUMN):
        raise Error("embedding: launch threads exceed portable device bounds")


def identical_embedding_backward_refusing_into(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    mut counts: DeviceBuffer[DType.int32],
    mut run_begin: DeviceBuffer[DType.int32],
    mut perm: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
    plan: Int = PLAN_SCAN,
    block_threads: Int = EMB_TPB,
) raises:
    """THE REFUSING DEVICE BACKWARD: `identical_embedding_backward_into` behind the shape refusals, the id refusal and a NaN or infinity in `dY`, or in the carried `dW` when `accumulate`, refused by name before any kernel is launched (DEVIATION 1506, closed for this entry point on 2026-09-15)."""
    _emb_backward_refuse_launch(ctx, plan, block_threads)
    emb_refuse_shape(cfg, n_positions)
    emb_refuse_device_ids(ctx, ids, n_positions, cfg)
    emb_refuse_device_nonfinite(ctx, String("dY"), dy, n_positions, cfg.width)
    if cfg.accumulate:
        emb_refuse_device_nonfinite(ctx, String("the carried dW"), dw, cfg.vocab, cfg.width)
    _emb_backward_launch(ctx, dw, dy, ids, counts, run_begin, perm, n_positions, cfg, plan, block_threads)


def _emb_backward_launch(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    mut counts: DeviceBuffer[DType.int32],
    mut run_begin: DeviceBuffer[DType.int32],
    mut perm: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
    plan: Int,
    block_threads: Int,
    runs_ready: Bool = False,
    unique_rows: Bool = False,
) raises:
    """Seams E0 through E4, launched. Every refusal is the caller's.

    NN50's explicit immutable run owner may pass runs_ready after constructing
    the canonical sorted groups. Normal public callers always rebuild. The
    owner must retain matching IDs/vocabulary/padding and complete run buffers.
    """

    if cfg.vocab < 1 or cfg.width < 1:
        return
    var cells = cfg.vocab * cfg.width

    var fill = not cfg.accumulate
    comptime if SAB_ACCUM_REFILLS:
        fill = True
    if fill:
        step_count_launch()
        ctx.enqueue_function[emb_seed_kernel](
            dw.unsafe_ptr(),
            Int32(cells),
            grid_dim=(_grid_for(cells, block_threads), 1, 1),
            block_dim=(block_threads, 1, 1),
        )

    if n_positions < 1:
        if cfg.has_padding():
            step_count_launch()
            ctx.enqueue_function[emb_pad_row_kernel](
                dw.unsafe_ptr(),
                Int32(cfg.width),
                Int32(cfg.padding_idx),
                grid_dim=(_grid_for(cfg.width, block_threads), 1, 1),
                block_dim=(block_threads, 1, 1),
            )
        return

    if runs_ready:
        pass
    elif plan == PLAN_SORT:
        embedding_sort_runs(ctx, ids, counts, run_begin, perm, n_positions, cfg.vocab, cfg.padding_idx, block_threads)
    else:
        step_count_launch()
        ctx.enqueue_function[emb_counts_kernel](
            counts.unsafe_ptr(),
            ids.unsafe_ptr(),
            Int32(n_positions),
            Int32(cfg.vocab),
            Int32(cfg.padding_idx),
            grid_dim=(_grid_for(cfg.vocab, block_threads), 1, 1),
            block_dim=(block_threads, 1, 1),
        )

        step_count_launch()
        comptime if is_defined["MOJOLEARN_EMB_SERIAL_RUN_BEGIN"]():
            # The revert arm: the single-thread scan.
            ctx.enqueue_function[emb_run_begin_kernel](
                run_begin.unsafe_ptr(),
                counts.unsafe_ptr(),
                Int32(cfg.vocab),
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
            )
        else:
            ctx.enqueue_function[emb_run_begin_block_kernel](
                run_begin.unsafe_ptr(),
                counts.unsafe_ptr(),
                Int32(cfg.vocab),
                grid_dim=(1, 1, 1),
                block_dim=(EMB_RUN_BEGIN_THREADS, 1, 1),
            )

        step_count_launch()
        ctx.enqueue_function[emb_perm_kernel](
            perm.unsafe_ptr(),
            run_begin.unsafe_ptr(),
            ids.unsafe_ptr(),
            Int32(n_positions),
            Int32(cfg.vocab),
            Int32(cfg.padding_idx),
            grid_dim=(_grid_for(cfg.vocab, block_threads), 1, 1),
            block_dim=(block_threads, 1, 1),
        )

    var touched = False
    comptime if EMB_TOUCHED_ROWS:
        touched = n_positions < cfg.vocab
    var folded = False
    comptime if EMB_UNIQUE_ROWS:
        if unique_rows:
            folded = _emb_unique_rows_fold(ctx, dw, dy, run_begin, perm, n_positions, cfg, block_threads)
    comptime if SAB_FOLD_VIA_GEMM_ONEHOT:
        _emb_fold_via_gemm_onehot(ctx, dw, dy, ids, n_positions, cfg, block_threads)
    else:
        if folded:
            pass
        elif touched:
            step_count_launch()
            comptime if NN49_EMB_VECTOR4:
                ctx.enqueue_function[emb_backward_touched_vector4_kernel](
                    dw.unsafe_ptr(), dy.unsafe_ptr(), perm.unsafe_ptr(),
                    run_begin.unsafe_ptr(), ids.unsafe_ptr(), Int32(n_positions),
                    Int32(cfg.vocab), Int32(cfg.width),
                    grid_dim=(_grid_for(n_positions * ((cfg.width + 3) // 4), block_threads), 1, 1),
                    block_dim=(block_threads, 1, 1),
                )
            else:
                ctx.enqueue_function[emb_backward_touched_kernel](
                    dw.unsafe_ptr(), dy.unsafe_ptr(), perm.unsafe_ptr(),
                    run_begin.unsafe_ptr(), ids.unsafe_ptr(), Int32(n_positions),
                    Int32(cfg.vocab), Int32(cfg.width),
                    grid_dim=(_grid_for(n_positions * cfg.width, block_threads), 1, 1),
                    block_dim=(block_threads, 1, 1),
                )
        else:
            step_count_launch()
            ctx.enqueue_function[emb_backward_kernel](
                dw.unsafe_ptr(),
                dy.unsafe_ptr(),
                perm.unsafe_ptr(),
                run_begin.unsafe_ptr(),
                Int32(cfg.vocab),
                Int32(cfg.width),
                grid_dim=(_grid_for(cells, block_threads), 1, 1),
                block_dim=(block_threads, 1, 1),
            )

    if cfg.has_padding():
        step_count_launch()
        ctx.enqueue_function[emb_pad_row_kernel](
            dw.unsafe_ptr(),
            Int32(cfg.width),
            Int32(cfg.padding_idx),
            grid_dim=(_grid_for(cfg.width, block_threads), 1, 1),
            block_dim=(block_threads, 1, 1),
        )
