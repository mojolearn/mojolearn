# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LOG-DETERMINANT FOLD ORDER (DEVIATION 1639, revised lane
cgr4-device-optim 2026-10-03). GPU-free: the device kernels
(cholesky/checks/potrf.mojo `logdet_part_kernel`, `logdet_kernel`) and the
host column (cholesky/host/chol_oracle.mojo) both call these.

`log |A| = 2 * sum_j log(L[j][j])`, the sum in the blocked-then-tree order:
  * the diagonal in blocks of LOGDET_BLOCK = 256 entries; a block's partial
    is `acc = ftz(acc + ftz(identical_log(ftz(d_j))))`, j ascending, from 0;
  * the block partials by the ALIGNED BINARY TREE: the node over blocks
    [i 2^L, (i+1) 2^L) is `ftz(left + right)`, or the left child alone when
    the right one is empty. The device folds it level by level (pairs
    (2i, 2i+1) to i, an odd last one carried); one thread folds it with a
    binary counter (`logdet_tree_serial`). The same nodes from the same
    children: the same words;
  * the doubling `ftz(identical_mul(2.0, root))`.
Before this lane the whole sum was ONE thread's ascending chain over n (n up
to the GP's training rows). For n <= 256 the words are unchanged.
"""
from checks.numerics import ftz, identical_log, identical_mul

comptime LOGDET_BLOCK = 256
comptime _LD_STACK = 48


@always_inline
def logdet_blocks(n: Int) -> Int:
    return (n + LOGDET_BLOCK - 1) // LOGDET_BLOCK


@always_inline
def logdet_part(diag: MutPointer[Float32, MutAnyOrigin], n: Int, b: Int) -> Float32:
    """Block b's partial: its logs folded ascending from +0.0."""
    var acc = Float32(0.0)
    var hi = min(b * LOGDET_BLOCK + LOGDET_BLOCK, n)
    for j in range(b * LOGDET_BLOCK, hi):
        acc = ftz(acc + ftz(identical_log(ftz(diag.unsafe_load(j)))))
    return acc


@always_inline
def logdet_pair(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def logdet_double(root: Float32) -> Float32:
    return ftz(identical_mul(Float32(2.0), root))


def logdet_serial(diag: MutPointer[Float32, MutAnyOrigin], n: Int) -> Float32:
    """The whole order on one thread: the partials, the aligned tree by a
    binary counter (complete blocks merged as they close, the open right
    spine merged right to left), the doubling."""
    var vals = InlineArray[Float32, _LD_STACK](fill=Float32(0))
    var lvls = InlineArray[Int, _LD_STACK](fill=0)
    var top = 0
    for b in range(logdet_blocks(n)):
        var v = logdet_part(diag, n, b)
        var lv = 0
        while top > 0 and lvls[top - 1] == lv:
            v = logdet_pair(vals[top - 1], v)
            top -= 1
            lv += 1
        vals[top] = v
        lvls[top] = lv
        top += 1
    var r = Float32(0.0)
    if top > 0:
        r = vals[top - 1]
        var k = top - 2
        while k >= 0:
            r = logdet_pair(vals[k], r)
            k -= 1
    return logdet_double(r)
