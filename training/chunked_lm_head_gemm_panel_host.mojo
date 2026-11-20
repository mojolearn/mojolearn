# SPDX-License-Identifier: Apache-2.0
"""The host column of the chunked LM head's GEMM-panel arm
(the default profile since 2026-11-19, the only one since 2026-11-20).

The device door's words on the CPU: the same GEMM contract for every product
(`gemm_host_rows_into`, split across cores inside) and the same per-unit
bodies as the device kernels (training/chunked_lm_head_panel_math.mojo), one
unit per iteration, the units of every pass split into contiguous tasks
across all cores. Every unit is a function of its own inputs, so the task
count moves no bit. The loop over logit panels is the only serial driver: a
panel's products need the whole pool. The oracle
(training/checks/chunked_lm_head_oracle.mojo) reaches this module under the
arm, and through it the training host binding and every check; the executor
op's host arm (autograd/lm_head_v2_host.mojo) calls the `_into` entries.
Operands are not flushed here: they are read only through the GEMM, whose
leaf flushes both operands itself.
"""
from std.memory import bitcast

from checks.numerics import identical_fmax
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.host.neural_gemm import GhrPtr, gemm_host_rows_into
from training.checks.loss_contract import CE_NEG_INF_BITS
from training.chunked_lm_head_profile import LM_HEAD_V2_CHUNK
from training.chunked_lm_head_panel_math import (
    LMHEAD_GP_LEAF,
    lmhead_gp_leaf_count, lmhead_gp_stack_slots,
    lmhead_gp_leaf_body, lmhead_gp_level_body, lmhead_gp_row_loss_body,
    lmhead_gp_dlogit_body, lmhead_gp_merge_body,
)

comptime _P = MutPointer[Float32, MutAnyOrigin]
comptime _T = MutPointer[Int32, MutAnyOrigin]


def _host_max_pass(logits: _P, maxima: _P, rows: Int, n: Int, first: Bool):
    """`_chunk_max_kernel` on the host: each row's running maximum over one
    panel. A total-order selection, so the order moves no bit."""
    var tasks = host_predict_task_count_work(rows, rows * n)
    var chunk = host_predict_chunk(rows, tasks)

    def _rows(k: Int) {imm logits, imm maxima, imm chunk, imm rows, imm n, imm first}:
        var lo = k * chunk
        for row in range(lo, min(lo + chunk, rows)):
            var acc = bitcast[DType.float32](CE_NEG_INF_BITS) if first else maxima.unsafe_load(row)
            for j in range(n):
                acc = identical_fmax(acc, logits.unsafe_load(row * n + j))
            maxima.unsafe_store(row, acc)

    host_parallelize(_rows, tasks)


def _host_leaf_pass(partial: _P, target_shift: _P, logits: _P, maxima: _P, targets: _T,
                    rows: Int, n: Int, chunk0: Int, leaves: Int):
    """`_gp_leaf_kernel` on the host: one unit per (row, leaf of this panel)."""
    var units = rows * ((n + LMHEAD_GP_LEAF - 1) // LMHEAD_GP_LEAF)
    var tasks = host_predict_task_count_work(units, rows * n)
    var chunk = host_predict_chunk(units, tasks)

    def _units(k: Int) {imm partial, imm target_shift, imm logits, imm maxima, imm targets, imm chunk, imm units, imm n, imm chunk0, imm leaves}:
        var lo = k * chunk
        for unit in range(lo, min(lo + chunk, units)):
            lmhead_gp_leaf_body(unit, partial, target_shift, logits, maxima, targets, n, chunk0, leaves)

    host_parallelize(_units, tasks)


def _host_level_pass(dst: _P, src: _P, rows: Int, width: Int):
    """`_gp_level_kernel` on the host: one unit per (row, output node)."""
    var units = rows * ((width + 1) // 2)
    var tasks = host_predict_task_count_work(units, units)
    var chunk = host_predict_chunk(units, tasks)

    def _units(k: Int) {imm dst, imm src, imm chunk, imm units, imm width}:
        var lo = k * chunk
        for unit in range(lo, min(lo + chunk, units)):
            lmhead_gp_level_body(unit, dst, src, width)

    host_parallelize(_units, tasks)


def _host_row_loss_pass(denom: _P, row_loss: _P, root: _P, rows: Int):
    """`_gp_row_loss_kernel` on the host: one unit per row."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)

    def _rows(k: Int) {imm denom, imm row_loss, imm root, imm chunk, imm rows}:
        var lo = k * chunk
        for row in range(lo, min(lo + chunk, rows)):
            lmhead_gp_row_loss_body(row, denom, row_loss, root)

    host_parallelize(_rows, tasks)


def _host_dlogit_pass(logits: _P, maxima: _P, denom: _P, targets: _T, rows: Int, n: Int, chunk0: Int):
    """`_gp_dlogit_kernel` on the host: one unit per panel cell, in place."""
    var units = rows * n
    var tasks = host_predict_task_count_work(units, units)
    var chunk = host_predict_chunk(units, tasks)

    def _cells(k: Int) {imm logits, imm maxima, imm denom, imm targets, imm chunk, imm units, imm rows, imm n, imm chunk0}:
        var lo = k * chunk
        for cell in range(lo, min(lo + chunk, units)):
            lmhead_gp_dlogit_body(cell, logits, maxima, denom, targets, rows, n, chunk0)

    host_parallelize(_cells, tasks)


def _host_merge_pass(dst: _P, l: _P, r: _P, cells: Int, has_r: Int):
    """`_gp_pair_kernel` (`has_r` 1) and `_gp_root_kernel` (`has_r` 0) on the host: one unit per dHidden cell."""
    var tasks = host_predict_task_count_work(cells, cells)
    var chunk = host_predict_chunk(cells, tasks)

    def _cells(k: Int) {imm dst, imm l, imm r, imm chunk, imm cells, imm has_r}:
        var lo = k * chunk
        for cell in range(lo, min(lo + chunk, cells)):
            lmhead_gp_merge_body(cell, dst, l, r, has_r)

    host_parallelize(_cells, tasks)


def lmhead_gp_host_forward_into(
    hp: GhrPtr, wp: GhrPtr, tg: _T, rows: Int, vocab: Int, width: Int,
    mx: _P, dn: _P, rl: _P,
) raises:
    """Row maxima, denominators and row losses (each `rows` long) in the
    device forward's words (`chunked_lm_head_gp_forward_into`), over raw
    storage: the oracle's lists and the executor op's host buffers. A code
    outside [0, vocab) matches no entry; its row's `rl` cell must then hold
    +0.0 on entry. The caller folds the mean."""
    var leaves = lmhead_gp_leaf_count(vocab)
    var panel = List[Float32](length=rows * min(vocab, LM_HEAD_V2_CHUNK), fill=Float32(0.0))
    var leaf_a = List[Float32](length=rows * leaves, fill=Float32(0.0))
    var leaf_b = List[Float32](length=rows * ((leaves + 1) // 2), fill=Float32(0.0))
    var pp = rebind[GhrPtr](panel.unsafe_ptr())
    var lg = rebind[_P](panel.unsafe_ptr())
    var la = rebind[_P](leaf_a.unsafe_ptr())
    var lb = rebind[_P](leaf_b.unsafe_ptr())
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):  # driver-loop: over logit panels
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        gemm_host_rows_into(hp, wp + chunk0 * width, pp, OP_NT, rows, n, width)
        _host_max_pass(lg, mx, rows, n, chunk0 == 0)
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):  # driver-loop: over logit panels
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        gemm_host_rows_into(hp, wp + chunk0 * width, pp, OP_NT, rows, n, width)
        _host_leaf_pass(la, rl, lg, mx, tg, rows, n, chunk0, leaves)
    var level_width = leaves
    var from_a = True
    while level_width > 1:  # driver-loop: over denominator tree levels
        if from_a:
            _host_level_pass(lb, la, rows, level_width)
        else:
            _host_level_pass(la, lb, rows, level_width)
        from_a = not from_a
        level_width = (level_width + 1) // 2
    if from_a:
        _host_row_loss_pass(dn, rl, la, rows)
    else:
        _host_row_loss_pass(dn, rl, lb, rows)
    # The pointers above do not retain these owners.
    _ = panel^
    _ = leaf_a^
    _ = leaf_b^


def lmhead_gp_host_backward_into(
    hp: GhrPtr, wp: GhrPtr, tg: _T, rows: Int, vocab: Int, width: Int,
    mx: _P, dn: _P, dh: _P, dwp: GhrPtr, want_dh: Bool, want_dw: Bool,
) raises:
    """dHidden (rows * width) and dWeight (vocab * width) from the matching
    forward statistics, in the device backward's words
    (`chunked_lm_head_gp_backward_into`), over raw storage. A gradient not
    wanted is neither computed nor written (its pointer is not read)."""
    var cells = rows * width
    var panel = List[Float32](length=rows * min(vocab, LM_HEAD_V2_CHUNK), fill=Float32(0.0))
    var stack = List[Float32](
        length=(lmhead_gp_stack_slots(vocab) * cells if want_dh else 1), fill=Float32(0.0)
    )
    var pp = rebind[GhrPtr](panel.unsafe_ptr())
    var sp = rebind[GhrPtr](stack.unsafe_ptr())
    var lg = rebind[_P](panel.unsafe_ptr())
    var st = rebind[_P](stack.unsafe_ptr())
    # The device's binary counter over panel partials, slot for slot.
    var depth = 0
    var panels = 0
    for chunk0 in range(0, vocab, LM_HEAD_V2_CHUNK):  # driver-loop: over logit panels
        var n = min(LM_HEAD_V2_CHUNK, vocab - chunk0)
        gemm_host_rows_into(hp, wp + chunk0 * width, pp, OP_NT, rows, n, width)
        _host_dlogit_pass(lg, mx, dn, tg, rows, n, chunk0)
        if want_dw:
            gemm_host_rows_into(pp, hp, dwp + chunk0 * width, OP_TN, n, width, rows)
        if want_dh:
            gemm_host_rows_into(pp, wp + chunk0 * width, sp + depth * cells, OP_NN, rows, width, n)
            depth += 1
            panels += 1
            var carry = panels
            while carry % 2 == 0:  # driver-loop: over completed tree nodes
                _host_merge_pass(st + (depth - 2) * cells, st + (depth - 2) * cells, st + (depth - 1) * cells, cells, 1)
                depth -= 1
                carry //= 2
    if want_dh:
        while depth > 1:  # driver-loop: over leftover tree nodes
            _host_merge_pass(st + (depth - 2) * cells, st + (depth - 2) * cells, st + (depth - 1) * cells, cells, 1)
            depth -= 1
        _host_merge_pass(dh, st, st, cells, 0)
    # The pointers above do not retain these owners.
    _ = panel^
    _ = stack^


def lmhead_gp_host_forward(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int,
    mut maxima: List[Float32], mut denom: List[Float32], mut row_loss: List[Float32],
) raises:
    """`lmhead_gp_host_forward_into` over lists (the oracle's door);
    `row_loss` arrives zero-filled."""
    lmhead_gp_host_forward_into(
        rebind[GhrPtr](hidden.unsafe_ptr()), rebind[GhrPtr](weight.unsafe_ptr()),
        rebind[_T](targets.unsafe_ptr()), rows, vocab, width,
        rebind[_P](maxima.unsafe_ptr()), rebind[_P](denom.unsafe_ptr()), rebind[_P](row_loss.unsafe_ptr()),
    )


def lmhead_gp_host_backward(
    hidden: List[Float32], weight: List[Float32], targets: List[Int32],
    rows: Int, vocab: Int, width: Int,
    maxima: List[Float32], denom: List[Float32],
    mut d_hidden: List[Float32], mut d_weight: List[Float32],
) raises:
    """`lmhead_gp_host_backward_into` over lists (the oracle's door), both gradients."""
    lmhead_gp_host_backward_into(
        rebind[GhrPtr](hidden.unsafe_ptr()), rebind[GhrPtr](weight.unsafe_ptr()),
        rebind[_T](targets.unsafe_ptr()), rows, vocab, width,
        rebind[_P](maxima.unsafe_ptr()), rebind[_P](denom.unsafe_ptr()),
        rebind[_P](d_hidden.unsafe_ptr()), rebind[GhrPtr](d_weight.unsafe_ptr()), True, True,
    )
