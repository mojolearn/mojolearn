# SPDX-License-Identifier: Apache-2.0
"""Fuse a rounded bias add with the production FLAT contract body.

Both arms explicitly materialize the product's Float32/FTZ seam. Each thread
owns one output; bias arithmetic follows that rounded product. This is an
opt-in research adapter and never changes the production GEMM dispatcher.
"""
from max.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from gemm.checks.gemm_identical import identical_gemm_flat_kernel,_flat_cell_body,contract_partition,gemm_operand_strides


# I05 2026-10-06 L40S component LOSS on both measured shapes/all layouts:
# baseline/candidate 0.405x-0.962x, one same-process warmup/score.
# Full AMD+NVIDIA workload promotion remains unqualified; default unchanged.
# Evidence: overnight-ab-20261006/nvidia/default-repair-normalized-measurements.json.
# Explicit fused adapter only; rounded product seam must pass independent bits.

