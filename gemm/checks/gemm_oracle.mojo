# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Re-export. The IDENTICAL FP32 GEMM oracle SHIPS in the CPU host bindings
(the linalg, solver, svm and byte LM families; python/mojolearn/host_surface.py
names them), so it lives in `gemm/host/gemm_oracle.mojo` since the host
surface manifest lane, 2026-09-14. Every check, bench and contract that
imported `gemm.checks.gemm_oracle` keeps resolving through this file; new
code imports `gemm.host.gemm_oracle` directly."""

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
    gemm_oracle_right_zero_padded,
    gemm_oracle_at_leaf,
    gemm_oracle_cell,
    gemm_oracle_serial,
    gemm_oracle_serial_cell,
    leaf_begin,
    leaf_count,
    leaf_end,
    op_name,
    oracle_leaf_partial,
    oracle_leaf_partial_right_zero_padded,
)
