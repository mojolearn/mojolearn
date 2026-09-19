# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: production-facing import surface for the CPU IDENTICAL GEMM implementation.
"""The CPU implementation of ``mojolearn.identical.gemm.fp32.v1``.

Production host code should import this module.  The implementation remains in
``gemm.host.gemm_oracle`` for source compatibility with contracts, evidence,
and external Mojo imports that use its historical name.  In CPU bindings that
implementation is executable product code, not merely a test oracle.

Keeping this narrow re-export makes the distinction explicit without changing
symbols, sabotage receipts, evidence paths, or the independently compiled
device implementation in one churn-heavy rename.  Checks may continue to use
``gemm.checks.gemm_oracle`` or the historical host module when they mean the
diagnostic reference specifically.
"""

from gemm.host.gemm_oracle import (
    CONTRACT_K_LEAF_MIN,
    CONTRACT_MAX_LEAVES,
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
    contract_leaf_size,
    fold_balanced_tree,
    fold_level_base,
    fold_level_count,
    fold_level_width,
    fold_node_addr,
    fold_node_is_carry,
    fold_node_total,
    gemm_oracle,
    gemm_oracle_at_leaf,
    gemm_oracle_cell,
    gemm_oracle_right_zero_padded,
    gemm_oracle_serial,
    gemm_oracle_serial_cell,
    leaf_begin,
    leaf_count,
    leaf_end,
    op_name,
    oracle_leaf_partial,
    oracle_leaf_partial_right_zero_padded,
)
