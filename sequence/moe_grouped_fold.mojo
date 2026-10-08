# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MoE expert products on the identical GEMM (lane gap-gemm-layers,
2026-10-08; plan docs/plans/gaps-2026-10-08.md section 8): THE HOST TWIN
AND THE SWITCH. HOST-SAFE (no device construct): `sequence/moe.mojo`'s items
import this file, `sequence/moe_grouped.mojo` holds the device side.

MOJOLEARN_IDN_MOE_GROUPED_GEMM (IDENTICAL, default off). The device runs
each expert's two products through `identical_gemm_into` (profile
mojolearn.identical.gemm.fp32.v1): gate|up = x_e . W_gu[e]^T over d, and
s_e = h_e . W_down[e]^T over f. So a cell is no longer `moe_reg`'s single
ascending chain over the whole k: it is the contract's cell, leaves of
`contract_leaf_size(k)` consecutive terms, each an ascending chain from +0.0
through `ftz(identical_mul_add(ftz(a), ftz(b), acc))`, the leaf partial
`ftz(acc)` pushed on the contract's balanced tree, the root `ftz`'d. The
host column (`op_moe_hidden`, `op_moe_out`) spells the same cell here, so the
host digest follows the device's when the host twin is built with the same
define. BITS CHANGE once (the fold order, k = D and k = F above one leaf);
NVIDIA and AMD move together, by the contract.

`_fold_push16` / `_fold_drain16` are `gemm/checks/gemm_identical.mojo`'s
`_fold_push` / `_fold_drain` character for character (the contract's tree,
`check_stack_fold_is_the_contract_tree`), spelled on an `InlineArray` so this
file imports nothing from the device GEMM.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from gemm.contract import contract_leaf_size, leaf_count
from sequence.ops import FP

comptime MOE_GROUPED_GEMM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_MOE_GROUPED_GEMM"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: the fold stack's depth: 2^12 leaves cover the contract's CONTRACT_MAX_LEAVES = 1024
comptime MOE_FOLD_LEVELS = 12
comptime MOE_FOLD_SLOTS = 16


@always_inline
def _fold_push16(mut stack: InlineArray[Float32, MOE_FOLD_SLOTS], mut occ: Int, value: Float32):
    """`gemm_identical._fold_push`: slot d holds a full block of 2^d leaves;
    an occupied level merges `ftz(ftz(stack[d]) + ftz(val))` (the earlier
    leaves on the left) and carries upward; an unpaired leaf waits."""
    var val = value
    var placed = False
    comptime for d in range(MOE_FOLD_LEVELS):
        if not placed:
            if ((occ >> d) & 1) == 1:
                val = ftz(ftz(stack[d]) + ftz(val))
                occ = occ - (1 << d)
            else:
                stack[d] = val
                occ = occ + (1 << d)
                placed = True


@always_inline
def _fold_drain16(stack: InlineArray[Float32, MOE_FOLD_SLOTS], occ: Int) -> Float32:
    """`gemm_identical._fold_drain`: the leftover slots, lowest level first;
    one slot performs no addition."""
    var have = False
    var acc = Float32(0.0)
    comptime for d in range(MOE_FOLD_LEVELS):
        if ((occ >> d) & 1) == 1:
            if have:
                acc = ftz(ftz(stack[d]) + ftz(acc))
            else:
                acc = stack[d]
                have = True
    return acc


@always_inline
def moe_contract_cell(a: FP, b: FP, k: Int) -> Float32:
    """One output cell of `A . B^T` for the rows `a[0..k)` and `b[0..k)`
    (both contiguous along k, the OP_NT operands of every MoE product):
    the contract's leaves and tree, the stored word `ftz(root)`."""
    var leaf = contract_leaf_size(k)
    var p_count = leaf_count(k, leaf)
    var stack = InlineArray[Float32, MOE_FOLD_SLOTS](fill=Float32(0.0))
    var occ = 0
    for t in range(p_count):
        var pb = t * leaf
        var pe = pb + leaf
        if pe > k:
            pe = k
        var acc = Float32(0.0)
        for p in range(pb, pe):
            acc = ftz(identical_mul_add(ftz(a.unsafe_load(p)), ftz(b.unsafe_load(p)), acc))
        _fold_push16(stack, occ, ftz(acc))
    return ftz(_fold_drain16(stack, occ))
