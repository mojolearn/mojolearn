# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device embedding of profile `mojolearn.identical.embedding.fp32.v1`. The CALLER owns every buffer -- including `counts`, `run_begin` and `perm` -- and must keep every one of them alive past its own `ctx.synchronize()`."""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

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
    refuse_nonfinite,
)



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
    or SAB_PAD_ROW_CONTRIBUTES
    or SAB_PAD_ROW_NEG_ZERO
    or SAB_NO_FLUSH_ACC
    or SAB_GATHER_NO_FLUSH
    or SAB_GATHER_CLAMP_OOR
    or SAB_ACCUM_BY_ADD
    or SAB_ACCUM_REFILLS
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


def _grid_for(count: Int) -> Int:
    """Blocks for a flat `count`-element launch."""
    if count < 1:
        return 1
    return (count + EMB_TPB - 1) // EMB_TPB




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
    var h = ctx.enqueue_create_host_buffer[DType.int32](n_positions)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=ids)
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

    if cfg.width < 1 or n_positions < 1:
        return
    var cells = n_positions * cfg.width
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
) raises:
    """Seams E0 through E4, enqueued. **`counts`, `run_begin` and `perm` ARE THE CALLER'S** and must be sized with `emb_run_scratch_ints` and kept alive past the caller's own `ctx.synchronize()`."""
    emb_refuse_device_ids(ctx, ids, n_positions, cfg)

    if cfg.vocab < 1 or cfg.width < 1:
        return
    var cells = cfg.vocab * cfg.width

    var fill = not cfg.accumulate
    comptime if SAB_ACCUM_REFILLS:
        fill = True
    if fill:
        ctx.enqueue_function[emb_seed_kernel](
            dw.unsafe_ptr(),
            Int32(cells),
            grid_dim=(_grid_for(cells), 1, 1),
            block_dim=(EMB_TPB, 1, 1),
        )

    if n_positions < 1:
        if cfg.has_padding():
            ctx.enqueue_function[emb_pad_row_kernel](
                dw.unsafe_ptr(),
                Int32(cfg.width),
                Int32(cfg.padding_idx),
                grid_dim=(_grid_for(cfg.width), 1, 1),
                block_dim=(EMB_TPB, 1, 1),
            )
        return

    ctx.enqueue_function[emb_counts_kernel](
        counts.unsafe_ptr(),
        ids.unsafe_ptr(),
        Int32(n_positions),
        Int32(cfg.vocab),
        Int32(cfg.padding_idx),
        grid_dim=(_grid_for(cfg.vocab), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )

    ctx.enqueue_function[emb_run_begin_kernel](
        run_begin.unsafe_ptr(),
        counts.unsafe_ptr(),
        Int32(cfg.vocab),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )

    ctx.enqueue_function[emb_perm_kernel](
        perm.unsafe_ptr(),
        run_begin.unsafe_ptr(),
        ids.unsafe_ptr(),
        Int32(n_positions),
        Int32(cfg.vocab),
        Int32(cfg.padding_idx),
        grid_dim=(_grid_for(cfg.vocab), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )

    ctx.enqueue_function[emb_backward_kernel](
        dw.unsafe_ptr(),
        dy.unsafe_ptr(),
        perm.unsafe_ptr(),
        run_begin.unsafe_ptr(),
        Int32(cfg.vocab),
        Int32(cfg.width),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )

    if cfg.has_padding():
        ctx.enqueue_function[emb_pad_row_kernel](
            dw.unsafe_ptr(),
            Int32(cfg.width),
            Int32(cfg.padding_idx),
            grid_dim=(_grid_for(cfg.width), 1, 1),
            block_dim=(EMB_TPB, 1, 1),
        )
