# SPDX-License-Identifier: Apache-2.0
"""Separate grid jobs for independent products sharing one left operand.

The job axis changes operand/output offsets only. The production FLAT body's
leaf boundaries and fold are called unchanged; contractions are never joined.
This tests launch batching separately from tiled shared-operand staging.
"""
from max.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import (
    _flat_cell_body, contract_partition, gemm_operand_strides,
)


# I03/N05 2026-10-06 L40S grouped component mixed WIN/LOSS: small shapes
# improve two layouts (~1.24x-1.31x), but middle layout loses (~0.25x);
# large shapes near parity (0.988x-1.029x), including rebound N05 versions.
# One same-process warmup/score; no full-workload default promotion. Evidence:
# overnight-ab-20261006/nvidia/{default,specific}-repair-normalized-measurements.json.
# Explicit grouped adapter only; the shared FLAT arithmetic body is unchanged.

