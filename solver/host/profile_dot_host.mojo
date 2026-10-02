# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host values of `solver/checks/profile_dot.mojo`'s device dot (the
oracle side): `gemm_oracle_cell` at the contract's leaf size, the serial
chain, and a column copy. Moved out of the device module (cpu-gpu-cleanup
c-linear, cross-lane from n-gemm) so no GPU-reachable module imports the
host GEMM oracle."""
from gemm.contract import OP_NT, contract_leaf_size
from gemm.checks.gemm_oracle import gemm_oracle_cell, gemm_oracle_serial_cell


def profile_dot_host(a: List[Float32], b: List[Float32], k: Int) -> Float32:
    """The host value of the same cell: `gemm_oracle_cell` at the contract's
    own leaf size. `a` and `b` hold at least `k` floats from index 0."""
    return gemm_oracle_cell(a, b, OP_NT, 0, 0, 1, 1, k, contract_leaf_size(k))


def serial_dot_host(a: List[Float32], b: List[Float32], k: Int) -> Float32:
    """The SERIAL ascending chain over all `k` -- the one-leaf case, and
    what the profile returns at `k <= 128`. Reported beside the profile
    value by the oracle so a reader can see the fold move (they differ at
    `P > 1`, which is the witness that the tree is reached)."""
    return gemm_oracle_serial_cell(a, b, OP_NT, 0, 0, 1, 1, k)


def column_as_list(x: List[Float32], off: Int, k: Int) -> List[Float32]:
    """`x[off : off + k]` as its own list, for the two host dots above."""
    var out = List[Float32]()
    out.reserve(k)
    for p in range(k):
        out.append(x[off + p])
    return out^
