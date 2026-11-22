# SPDX-License-Identifier: Apache-2.0
"""A02 bounded one/two/four-plane staging with unchanged FP32 leaves.

Existing packed production bodies admit only one/two pages. This separate
supported adapter tests a four-plane resource choice without silently
widening those bodies' template contract. Every plane holds one ascending
contraction step for each independent output lane, and two uniform barriers
protect the whole staging group. Increasing depth amortizes those barriers
over more terms while charging its extra shared storage. It does not claim
asynchronous load overlap or replace the production tiled-reuse dispatcher.
"""
from max.gpu import block_idx,block_dim,thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext,DeviceBuffer
from gemm.checks.gemm_identical import GEMM_FOLD_SLOTS,_fold_push,_fold_drain,contract_partition,gemm_operand_strides
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add


# A02 experiment: NEVER RUN — PENDING MEASUREMENT; incumbent defaults retained.
# A02 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit depth1/2/4 staging adapter only; existing library depth stays unchanged.

