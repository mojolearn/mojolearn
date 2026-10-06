# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the cluster family's CPU host bindings.
"""`gemm/host/gemm_oracle.mojo::gemm_oracle`, bit for bit, fast, for the
cluster family's host paths (lane cluster-cpu, 2026-09-28). HOST ONLY.

`gemm_oracle(a, b, op, m, n, k)` is, per output cell (i, j): the logical
leaves `[t L, min((t+1) L, k))` at `L = contract_leaf_size(k)`; each leaf's
partial is the ascending chain `acc = ftz(fma(ftz(A[i,p]), ftz(B[p,j]),
acc))` from `+0.0`, written through `ftz`; then `fold_balanced_tree` over
the P partials (one leaf: `ftz` of the partial). `host_gemm_oracle` runs
exactly that:
  * the leaf partials of one row i and one leaf t are a task; within it
    GEMM_W neighbouring columns j run in the lanes of one vector (for NN and
    TN the right operand's row p is contiguous in j), each lane its cell's
    own chain in p order (`cluster/host/host_cells.mojo`: `ftz_v`,
    `mul_add_v`); a ragged column tail runs the scalar statement;
  * the fold is the oracle's `fold_balanced_tree`, per cell, cells as tasks.
Leaf boundaries depend on k alone, so nothing here depends on the thread
count. NT, and any build with a gemm-oracle sabotage arm compiled in, take
the oracle's own cell (`gemm_oracle_cell`) cell by cell, so a sabotaged
build bites here exactly as it bites `gemm_oracle`. Checked against
`gemm_oracle` by `cluster/checks/host_gemm_check.mojo`. Not a numeric row."""
from gemm.host.gemm_oracle import (
    GEMM_ORACLE_SABOTAGE_ORDER_ARM,
    GEMM_ORACLE_SABOTAGE_VALUE_ARM,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_size,
    fold_balanced_tree,
    gemm_oracle_cell,
    leaf_count,
    leaf_end,
)
from checks.numerics import ftz, identical_mul_add

from cluster.host.host_cells import ftz_v, host_cells, mul_add_v
from core.host_storage import host_list_ptr

#: Output columns per vector.
comptime GEMM_W = 8


def _host_gemm_by_cell(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int
) -> List[Float32]:
    """Every cell through `gemm_oracle_cell` at the contract's leaf."""
    var c = List[Float32](length=m * n, fill=Float32(0.0))
    var leaf = contract_leaf_size(k)
    var cp = host_list_ptr(c)

    def _cell(t: Int) {imm a, imm b, imm cp, imm op, imm m, imm n, imm k, imm leaf}:
        var i = t // n
        var j = t - i * n
        cp.unsafe_store(t, gemm_oracle_cell(a, b, op, i, j, m, n, k, leaf))

    host_cells(_cell, m * n, 3 * k)
    return c^


def host_gemm_oracle(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int
) -> List[Float32]:
    """`gemm_oracle(a, b, op, m, n, k)`, bit for bit (module docstring)."""
    if m * n == 0:
        return List[Float32]()
    comptime if GEMM_ORACLE_SABOTAGE_ORDER_ARM or GEMM_ORACLE_SABOTAGE_VALUE_ARM:
        return _host_gemm_by_cell(a, b, op, m, n, k)
    if op == OP_NT or k <= 0:
        return _host_gemm_by_cell(a, b, op, m, n, k)
    var leaf = contract_leaf_size(k)
    var pc = leaf_count(k, leaf)
    # partials[(t * m + i) * n + j]
    var partials = List[Float32](length=pc * m * n, fill=Float32(0.0))
    var ap = host_list_ptr(a)
    var bp = host_list_ptr(b)
    var pp = host_list_ptr(partials)
    var tn = op == OP_TN

    def _leaf_row(task: Int) {imm ap, imm bp, imm pp, imm tn, imm m, imm n, imm k, imm leaf}:
        var t = task // m
        var i = task - t * m
        var p0 = t * leaf
        var p1 = leaf_end(t, leaf, k)
        var out = pp + (t * m + i) * n
        var j0 = 0
        while j0 + GEMM_W <= n:
            var acc = SIMD[DType.float32, GEMM_W](0)
            for p in range(p0, p1):
                var av = ap.unsafe_load(p * m + i) if tn else ap.unsafe_load(i * k + p)
                var aa = SIMD[DType.float32, GEMM_W](ftz(av))
                var bv = ftz_v[GEMM_W]((bp + p * n + j0).load[width=GEMM_W]())
                acc = ftz_v[GEMM_W](mul_add_v[GEMM_W](aa, bv, acc))
            (out + j0).store(ftz_v[GEMM_W](acc))
            j0 += GEMM_W
        for j in range(j0, n):
            var acc = Float32(0.0)
            for p in range(p0, p1):
                var av = ap.unsafe_load(p * m + i) if tn else ap.unsafe_load(i * k + p)
                acc = ftz(identical_mul_add(ftz(av), ftz(bp.unsafe_load(p * n + j)), acc))
            out.unsafe_store(j, ftz(acc))

    host_cells(_leaf_row, pc * m, 3 * n * leaf)
    if pc == 1:
        # `gemm_oracle_at_leaf`'s one-leaf path: `ftz` of the partial.
        var c1 = List[Float32](length=m * n, fill=Float32(0.0))
        for t in range(m * n):
            c1[t] = ftz(partials[t])
        return c1^
    var c = List[Float32](length=m * n, fill=Float32(0.0))
    var cp = host_list_ptr(c)

    def _fold(cell: Int) {imm pp, imm cp, imm pc, imm m, imm n}:
        var parts = List[Float32](capacity=pc)
        for t in range(pc):
            parts.append(pp.unsafe_load(t * m * n + cell))
        cp.unsafe_store(cell, fold_balanced_tree(parts))

    host_cells(_fold, m * n, 4 * pc)
    _ = partials^
    return c^
