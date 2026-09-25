# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device embedding of profile `mojolearn.identical.embedding.fp32.v1`. The CALLER owns every buffer -- including `counts`, `run_begin` and `perm` -- and must keep every one of them alive past its own `ctx.synchronize()`."""

from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT, embedding_sort_runs

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
# DEVIATION 2630: the step phase timers and counters (core/step_phase.mojo;
# compiled only under -D MOJOLEARN_STEP_PHASE_TIMERS=1).
from core.step_phase import (
    step_count_d2h,
    step_count_host_alloc,
    step_count_launch,
    step_count_sync,
)

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
)
from checks.kernel_matrix import (
    IDENTITY_FLOOR_BLOCK,
    TARGET_COLUMN,
    column_max_block_size,
)
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids,
    emb_refuse_shape,
    refuse_nonfinite,
)
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier



comptime SAB_FOLD_DESCENDING = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_FOLD_DESCENDING"
]()
comptime SAB_FOLD_BALANCED_TREE = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_FOLD_BALANCED_TREE"
]()
comptime SAB_SEED_SEEDLESS = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_SEED_SEEDLESS"
]()
comptime SAB_SINGLE_RUN_BYPASS = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_SINGLE_RUN_BYPASS"
]()
comptime SAB_EMPTY_ROW_SKIPPED = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_EMPTY_ROW_SKIPPED"
]()
comptime SAB_EMPTY_ROW_NEG_ZERO = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_EMPTY_ROW_NEG_ZERO"
]()
comptime SAB_FOLD_READS_LAUNCH = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_FOLD_READS_LAUNCH"
]()
comptime SAB_RANK_BY_ARRIVAL = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_RANK_BY_ARRIVAL"
]()
comptime SAB_SORT_TIE_REVERSED = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_SORT_TIE_REVERSED"
]()
comptime SAB_PAD_ROW_CONTRIBUTES = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_PAD_ROW_CONTRIBUTES"
]()
comptime SAB_PAD_ROW_NEG_ZERO = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_PAD_ROW_NEG_ZERO"
]()
comptime SAB_NO_FLUSH_ACC = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_NO_FLUSH_ACC"
]()
comptime SAB_GATHER_NO_FLUSH = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_GATHER_NO_FLUSH"
]()
comptime SAB_GATHER_CLAMP_OOR = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_GATHER_CLAMP_OOR"
]()
comptime SAB_ACCUM_BY_ADD = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_ACCUM_BY_ADD"
]()
comptime SAB_ACCUM_REFILLS = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_ACCUM_REFILLS"
]()
# The two rows of contract 11.1 that had no switch until 2026-09-15.
# FOLD_VIA_GEMM_ONEHOT routes the backward through `identical_gemm` over a
# materialized one-hot [T, V] matrix (DEVIATION 1315's refused candidate,
# built so the difference is PRINTED); SORT_KEY_ID_ONLY_UNSTABLE is read by
# `embedding_sort.mojo`'s compare/exchange pass (PLAN_SORT only).
comptime SAB_FOLD_VIA_GEMM_ONEHOT = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_FOLD_VIA_GEMM_ONEHOT"
]()
comptime SAB_SORT_KEY_ID_ONLY_UNSTABLE = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_SORT_KEY_ID_ONLY_UNSTABLE"
]()

comptime SAB_SORT_NEGATIVE_CONTROL = is_defined["MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL"]()

comptime ANY_EMB_SABOTAGE = (
    SAB_FOLD_DESCENDING
    or SAB_FOLD_BALANCED_TREE
    or SAB_SEED_SEEDLESS
    or SAB_SINGLE_RUN_BYPASS
    or SAB_EMPTY_ROW_SKIPPED
    or SAB_EMPTY_ROW_NEG_ZERO
    or SAB_FOLD_READS_LAUNCH
    or SAB_RANK_BY_ARRIVAL
    or SAB_SORT_TIE_REVERSED
    or SAB_SORT_NEGATIVE_CONTROL
    or SAB_PAD_ROW_CONTRIBUTES
    or SAB_PAD_ROW_NEG_ZERO
    or SAB_NO_FLUSH_ACC
    or SAB_GATHER_NO_FLUSH
    or SAB_GATHER_CLAMP_OOR
    or SAB_ACCUM_BY_ADD
    or SAB_ACCUM_REFILLS
    or SAB_FOLD_VIA_GEMM_ONEHOT
    or SAB_SORT_KEY_ID_ONLY_UNSTABLE
)


def emb_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner. A check MUST print this AND the numeric mode AND the resolved block size in its header, because a sabotage arm that silently failed to compile in looks exactly like a clause that is bit-inert, and `[[reached-but-inert]]` is the standing rule that those are not the same thing."""
    comptime if SAB_FOLD_DESCENDING:
        return String("FOLD_DESCENDING")
    comptime if SAB_FOLD_BALANCED_TREE:
        return String("FOLD_BALANCED_TREE")
    comptime if SAB_SEED_SEEDLESS:
        return String("SEED_SEEDLESS")
    comptime if SAB_SINGLE_RUN_BYPASS:
        return String("SINGLE_RUN_BYPASS")
    comptime if SAB_EMPTY_ROW_SKIPPED:
        return String("EMPTY_ROW_SKIPPED")
    comptime if SAB_EMPTY_ROW_NEG_ZERO:
        return String("EMPTY_ROW_NEG_ZERO")
    comptime if SAB_FOLD_READS_LAUNCH:
        return String("FOLD_READS_LAUNCH")
    comptime if SAB_RANK_BY_ARRIVAL:
        return String("RANK_BY_ARRIVAL")
    comptime if SAB_SORT_TIE_REVERSED:
        return String("SORT_TIE_REVERSED")
    comptime if SAB_SORT_NEGATIVE_CONTROL:
        return String("SORT_NEGATIVE_CONTROL")
    comptime if SAB_PAD_ROW_CONTRIBUTES:
        return String("PAD_ROW_CONTRIBUTES")
    comptime if SAB_PAD_ROW_NEG_ZERO:
        return String("PAD_ROW_NEG_ZERO")
    comptime if SAB_NO_FLUSH_ACC:
        return String("NO_FLUSH_ACC")
    comptime if SAB_GATHER_NO_FLUSH:
        return String("GATHER_NO_FLUSH")
    comptime if SAB_GATHER_CLAMP_OOR:
        return String("GATHER_CLAMP_OOR")
    comptime if SAB_ACCUM_BY_ADD:
        return String("ACCUM_BY_ADD")
    comptime if SAB_ACCUM_REFILLS:
        return String("ACCUM_REFILLS")
    comptime if SAB_FOLD_VIA_GEMM_ONEHOT:
        return String("FOLD_VIA_GEMM_ONEHOT")
    comptime if SAB_SORT_KEY_ID_ONLY_UNSTABLE:
        return String("SORT_KEY_ID_ONLY_UNSTABLE")
    return String("none")



comptime EMB_TPB_WANT = 256


def _emb_max_tpb[column: Int]() -> Int:
    """Threads per block, resolved the way `checks/kernel_matrix.mojo::block_size_for` resolves its own."""
    comptime want = EMB_TPB_WANT
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime floored = (
        IDENTITY_FLOOR_BLOCK if identical
        and IDENTITY_FLOOR_BLOCK < want else want
    )
    comptime hard = column_max_block_size(column)
    return floored if floored < hard else hard


comptime EMB_TPB = _emb_max_tpb[TARGET_COLUMN]()


def _grid_for(count: Int, threads: Int = EMB_TPB) -> Int:
    """Blocks for a flat `count`-element launch."""
    if count < 1:
        return 1
    return (count + threads - 1) // threads




def emb_gather_kernel(
    out_y: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    width_in: Int32,
    vocab_in: Int32,
):
    """`Y[t, j] = ftz(ftz(W[ids[t], j]))`. Contract section 8 refuses an out-of-range id BY NAME on the host, before any launch, so a kernel-side clamp would be dead code that hides a data bug."""
    var n_positions = Int(n_positions_in)
    var width = Int(width_in)
    if width < 1:
        return
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n_positions * width:
        return
    var t = cell // width
    var j = cell - t * width
    var v = Int(ids.unsafe_load(t))
    comptime if SAB_GATHER_CLAMP_OOR:
        var vocab = Int(vocab_in)
        if v < 0:
            v = 0
        if v >= vocab:
            v = vocab - 1
    var src = weight.unsafe_load(v * width + j)
    comptime if SAB_GATHER_NO_FLUSH:
        out_y.unsafe_store(cell, src)
        return
    out_y.unsafe_store(cell, ftz(ftz(src)))




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




#: lane/nvidia-step-time (2026-09-25): threads of the one-block scan below.
comptime EMB_RUN_BEGIN_THREADS = 256


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
    from gemm.checks.gemm_oracle import OP_TN

    identical_gemm(ctx, product, onehot, dy, vocab, width, n_positions, OP_TN)


def emb_nonfinite_rows_kernel(
    flags: MutPointer[Int32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    width_in: Int32,
):
    """Contract 9.1 on the DEVICE, by BITS and never by a compare (Metal flushes compare operands, row 49; integer operations flush nowhere). One thread per row: `flags[r]` is `2 * j + 1` for a NaN or `2 * j` for an infinity at the row's FIRST nonfinite column `j`, else `-1`. No float is written and no thread reads another row."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(rows_in):
        return
    var width = Int(width_in)
    var base = r * width
    for j in range(width):
        var au = bitcast[DType.uint32](values.unsafe_load(base + j)) & UInt32(0x7FFFFFFF)
        if au > UInt32(0x7F800000):
            flags.unsafe_store(r, Int32(2 * j + 1))
            return
        if au == UInt32(0x7F800000):
            flags.unsafe_store(r, Int32(2 * j))
            return
    flags.unsafe_store(r, Int32(-1))


def emb_refuse_device_nonfinite(
    ctx: DeviceContext,
    name: String,
    mut values: DeviceBuffer[DType.float32],
    rows: Int,
    width: Int,
) raises:
    """DEVIATION 1506 CLOSED for the refusing entry points (2026-09-15): a NaN or an infinity in a device buffer is REFUSED BY NAME, with the first flat index in row-major order and the message `refuse_nonfinite` gives on the host, before any kernel of the profile is launched. Costs one launch over `rows * width` cells and a download of `rows` integers."""
    if rows <= 0 or width <= 0:
        return
    var flags = ctx.enqueue_create_buffer[DType.int32](rows)
    step_count_launch()
    ctx.enqueue_function[emb_nonfinite_rows_kernel](
        flags.unsafe_ptr(),
        values.unsafe_ptr(),
        Int32(rows),
        Int32(width),
        grid_dim=(_grid_for(rows), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )
    step_count_host_alloc()
    var h = ctx.enqueue_create_host_buffer[DType.int32](rows)
    step_count_sync()
    ctx.synchronize()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=flags)
    step_count_sync()
    ctx.synchronize()
    for r in range(rows):
        var f = Int(h.unsafe_ptr().unsafe_load(r))
        if f >= 0:
            var flat = r * width + f // 2
            _ = flags^
            _ = h^
            if f % 2 == 1:
                raise Error(
                    String("embedding: NaN in ")
                    + name
                    + " at flat index "
                    + String(flat)
                    + " REFUSED on the device entry point (row 39: NaN payloads"
                    + " are vendor-shaped; no stage may record one)"
                )
            raise Error(
                String("embedding: infinity in ")
                + name
                + " at flat index "
                + String(flat)
                + " REFUSED on the device entry point (contract 9.1)"
            )
    _ = flags^
    _ = h^


def emb_run_scratch_ints(vocab: Int, n_positions: Int) -> Int:
    """Integers of run scratch `identical_embedding_backward_into` needs."""
    var need = vocab + vocab + 1 + n_positions
    if need < 1:
        return 1
    return need


def emb_refuse_device_ids(
    ctx: DeviceContext,
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """Contract section 8 and 9.1 ON THE DEVICE ENTRY POINTS, which is where they were missing. IT IS AN OUT-OF-BOUNDS READ.** `emb_gather_kernel` computes `weight.unsafe_load(v * width + j)` with NO bounds branch on the normative path -- the only bounds handling in the file lives inside `SAB_GATHER_CLAMP_OOR`, a SABOTAGE arm, so a build without that define has none at all."""
    if n_positions <= 0:
        return
    comptime if SAB_GATHER_CLAMP_OOR:
        # THE SABOTAGE IS THE WHOLE WRONG SPELLING, NOT HALF OF IT: the
        # refusal is dropped and the gather kernel clamps instead. With the
        # refusal left in, the kernel's clamp could never see an
        # out-of-range id and the arm was unrunnable (2026-09-14 legs).
        return
    step_count_host_alloc()
    var h = ctx.enqueue_create_host_buffer[DType.int32](n_positions)
    step_count_sync()
    ctx.synchronize()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=ids)
    step_count_sync()
    ctx.synchronize()
    var lids = List[Int32]()
    for i in range(n_positions):
        lids.append(h.unsafe_ptr().unsafe_load(i))
    emb_refuse_ids(lids, cfg)
    _ = h


def identical_embedding_forward_into(
    ctx: DeviceContext,
    mut out_y: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """Seams G1 and G2, enqueued. `[[mojo-buffer-freed-at-last-use]]`: a `DeviceBuffer` is dead at its `.unsafe_ptr()`, so every one of these must outlive the caller's `ctx.synchronize()`."""
    emb_refuse_device_ids(ctx, ids, n_positions, cfg)
    _emb_forward_launch(ctx, out_y, weight, ids, n_positions, cfg)


def identical_embedding_forward_refusing_into(
    ctx: DeviceContext,
    mut out_y: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """THE REFUSING DEVICE FORWARD, contract sections 3, 8 and 9.1 in full: the shape refusals, the id refusal and a NaN or infinity anywhere in `W` refused by name before the gather is launched (DEVIATION 1506, closed for this entry point on 2026-09-15). `identical_embedding_forward_into` stays the caller-refused spelling the training loops use: it refuses ids and NOT nonfinite values, and its caller owns 9.1."""
    emb_refuse_shape(cfg, n_positions)
    emb_refuse_device_ids(ctx, ids, n_positions, cfg)
    emb_refuse_device_nonfinite(ctx, String("W"), weight, cfg.vocab, cfg.width)
    _emb_forward_launch(ctx, out_y, weight, ids, n_positions, cfg)


def _emb_forward_launch(
    ctx: DeviceContext,
    mut out_y: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """Seams G1 and G2, launched. Every refusal is the caller's."""
    if cfg.width < 1 or n_positions < 1:
        return
    var cells = n_positions * cfg.width
    step_count_launch()
    ctx.enqueue_function[emb_gather_kernel](
        out_y.unsafe_ptr(),
        weight.unsafe_ptr(),
        ids.unsafe_ptr(),
        Int32(n_positions),
        Int32(cfg.width),
        Int32(cfg.vocab),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )


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
) raises:
    """Seams E0 through E4, launched. Every refusal is the caller's."""

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

    if plan == PLAN_SORT:
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

    comptime if SAB_FOLD_VIA_GEMM_ONEHOT:
        _emb_fold_via_gemm_onehot(ctx, dw, dy, ids, n_positions, cfg, block_threads)
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
