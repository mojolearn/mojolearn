# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the cluster family's CPU host bindings.
"""`gemm/host/gemm_oracle.mojo::gemm_oracle` with its cells split over
threads, for the cluster family's host paths (lane cluster-cpu,
2026-09-28). HOST ONLY.

`gemm_oracle(a, b, op, m, n, k)` is `gemm_oracle_at_leaf` at
`contract_leaf_size(k)`: every output cell is `gemm_oracle_cell` at that
leaf (whose one-leaf case is `ftz(oracle_leaf_partial(.., 0, k))`, the
at-leaf path's hoisted spelling of the same value). A cell reads `a` and
`b` only and writes itself, so `host_gemm_oracle` computes each cell with
that very function inside `cluster/host/host_cells.mojo::host_cells`: the
same bits as `gemm_oracle` at every thread count. Not a numeric row."""
from gemm.host.gemm_oracle import contract_leaf_size, gemm_oracle_cell

from cluster.host.host_cells import host_cells
from core.host_predict_threads import host_list_ptr


def host_gemm_oracle(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int
) -> List[Float32]:
    """`gemm_oracle(a, b, op, m, n, k)`, bit for bit, cells over threads."""
    var c = List[Float32](length=m * n if m * n > 0 else 1, fill=Float32(0.0))
    if m * n == 0:
        return List[Float32]()
    var leaf = contract_leaf_size(k)
    var cp = host_list_ptr(c)

    def _cell(t: Int) {imm a, imm b, imm cp, imm op, imm m, imm n, imm k, imm leaf}:
        var i = t // n
        var j = t - i * n
        cp.unsafe_store(t, gemm_oracle_cell(a, b, op, i, j, m, n, k, leaf))

    host_cells(_cell, m * n, 3 * k)
    return c^
