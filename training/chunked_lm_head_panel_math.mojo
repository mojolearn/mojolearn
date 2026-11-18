# SPDX-License-Identifier: Apache-2.0
"""The GEMM-panel arm of the chunked LM head: its switch and the per-unit
bodies both columns run.

`-D MOJOLEARN_LMHEAD_FAST_GEMM_PANEL` (default off) moves
`chunked_lm_head_v2_loss` and `chunked_lm_head_v2_train` from the scalar
serial profile to GEMM logits and balanced-tree folds. The device kernels
(training/chunked_lm_head_v2.mojo) and the host column
(training/chunked_lm_head_gemm_panel_host.mojo, reached through the oracle)
call the same @always_inline bodies below, one unit per thread or per host
iteration, so the two columns and every vendor compute the same words. The
fold order is written in training/CHUNKED_LM_HEAD_V2.md, "GEMM-panel arm".
This module imports no device and no host-pool code.
"""
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_div, identical_exp, identical_log
from training.checks.loss_contract import neg_by_bits
from training.chunked_lm_head_profile import LM_HEAD_V2_CHUNK

#: The arm. One build decision read by the device door, the host column and
#: the oracle together.
comptime LMHEAD_FAST_GEMM_PANEL = is_defined["MOJOLEARN_LMHEAD_FAST_GEMM_PANEL"]()

#: Vocabulary ids per denominator leaf. A leaf is the unit one device thread
#: (or one host iteration) folds serially; the tree above the leaves is
#: balanced. 64 is a power of two that divides the panel width, so a panel
#: holds whole leaves and the fold does not depend on where panels end. It is
#: a cost choice (64 exponentials per thread against one tree level saved per
#: doubling), not a vocabulary size.
comptime LMHEAD_GP_LEAF = 64

#: THE NEGATIVE CONTROL of the arm on the host column: the training host
#: binding's own define (`-D MOJOLEARN_HOST_SABOTAGE=1`; no device build and
#: no check defines it). It folds every denominator leaf over its vocabulary
#: ids DESCENDING, a fold-order fault the contract forbids, with each term's
#: arithmetic unchanged. The default profile's control reverses dWeight's row
#: chain, which the arm no longer has (dWeight is a GEMM product).
comptime LMHEAD_GP_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

comptime _P = MutPointer[Float32, MutAnyOrigin]
comptime _T = MutPointer[Int32, MutAnyOrigin]


def lmhead_gp_leaf_count(vocab: Int) -> Int:
    """Denominator leaves of a vocabulary."""
    return (vocab + LMHEAD_GP_LEAF - 1) // LMHEAD_GP_LEAF


def lmhead_gp_panel_count(vocab: Int) -> Int:
    """Logit panels of a vocabulary."""
    return (vocab + LM_HEAD_V2_CHUNK - 1) // LM_HEAD_V2_CHUNK


def lmhead_gp_stack_slots(vocab: Int) -> Int:
    """dHidden partials held at once by the pairwise fold over panels:
    floor(log2(panels)) + 1."""
    var slots = 1
    var q = lmhead_gp_panel_count(vocab)
    while q > 1:  # small-loop(bits of the panel count)
        slots += 1
        q //= 2
    return slots


@always_inline
def lmhead_gp_pair(l: Float32, r: Float32) -> Float32:
    """One tree node: the flushed sum of two flushed children."""
    return ftz(ftz(l) + ftz(r))


@always_inline
def lmhead_gp_leaf_body(  # device-helper: one (row, leaf) of the denominator leaf kernel, also the host column's unit
    unit: Int,
    partial: _P, target_shift: _P, logits: _P, maxima: _P, targets: _T,
    n: Int, chunk0: Int, leaves: Int,
):
    """Unit `row * panel_leaves + leaf`: the leaf's `ftz(exp(shifted))` terms
    folded ascending from +0.0 into `partial[row * leaves + global_leaf]`.
    The leaf holding the row's target also stores its shifted logit."""
    var panel_leaves = (n + LMHEAD_GP_LEAF - 1) // LMHEAD_GP_LEAF
    var row = unit // panel_leaves
    var leaf = unit % panel_leaves
    var lo = leaf * LMHEAD_GP_LEAF
    var hi = min(lo + LMHEAD_GP_LEAF, n)
    var maximum = ftz(maxima.unsafe_load(row))
    var target = Int(targets.unsafe_load(row)) - chunk0
    var acc = Float32(0.0)
    for step in range(lo, hi):
        var j = step
        comptime if LMHEAD_GP_HOST_SABOTAGE:
            j = hi - 1 - (step - lo)
        var shifted = ftz(ftz(logits.unsafe_load(row * n + j)) - maximum)
        acc = ftz(acc + ftz(identical_exp(shifted)))
        if j == target:
            target_shift.unsafe_store(row, shifted)
    partial.unsafe_store(row * leaves + chunk0 // LMHEAD_GP_LEAF + leaf, acc)


@always_inline
def lmhead_gp_level_body(  # device-helper: one (row, node) of a denominator tree level, also the host column's unit
    unit: Int, dst: _P, src: _P, width: Int,
):
    """Unit `row * out_width + q` of the level that halves `width` nodes per
    row: node q is the pair (2q, 2q+1) of the level below, an odd tail
    carried bit for bit. Rows are packed at the level's own width."""
    var out_width = (width + 1) // 2
    var row = unit // out_width
    var q = unit % out_width
    var node = src.unsafe_load(row * width + 2 * q)
    if 2 * q + 1 < width:
        node = lmhead_gp_pair(node, src.unsafe_load(row * width + 2 * q + 1))
    dst.unsafe_store(row * out_width + q, node)


@always_inline
def lmhead_gp_row_loss_body(  # device-helper: one row of the denominator finish kernel, also the host column's unit
    row: Int, denom: _P, row_loss: _P, root: _P,
):
    """`denom[row]` = the tree's root; `row_loss[row]` (holding the target's
    shifted logit) becomes the row's negative log-probability."""
    var total = root.unsafe_load(row)
    denom.unsafe_store(row, total)
    row_loss.unsafe_store(
        row, neg_by_bits(ftz(ftz(row_loss.unsafe_load(row)) - ftz(identical_log(ftz(total)))))
    )


@always_inline
def lmhead_gp_dlogit_body(  # device-helper: one panel cell of the dlogit kernel, also the host column's unit
    cell: Int, logits: _P, maxima: _P, denom: _P, targets: _T,
    rows: Int, n: Int, chunk0: Int,
):
    """Panel cell `row * n + j`, in place: the logit becomes
    `ftz((ftz(exp(shifted) / denom) - onehot) / rows)`."""
    var row = cell // n
    var j = cell % n
    var shifted = ftz(ftz(logits.unsafe_load(cell)) - ftz(maxima.unsafe_load(row)))
    var probability = ftz(identical_div(ftz(identical_exp(shifted)), ftz(denom.unsafe_load(row))))
    var target = Float32(1.0) if chunk0 + j == Int(targets.unsafe_load(row)) else Float32(0.0)
    logits.unsafe_store(cell, ftz(identical_div(ftz(probability - target), Float32(rows))))


@always_inline
def lmhead_gp_merge_body(  # device-helper: one cell of the dHidden pair kernel, also the host column's unit
    cell: Int, dst: _P, l: _P, r: _P, has_r: Int,
):
    """One dHidden cell of a tree node over panel partials: the pair of two
    children, or (`has_r == 0`) the left child bit for bit."""
    var node = l.unsafe_load(cell)
    if has_r != 0:
        node = lmhead_gp_pair(node, r.unsafe_load(cell))
    dst.unsafe_store(cell, node)
