# SPDX-License-Identifier: Apache-2.0
"""NN58/NN59 neural pointwise experiments: deleted; the names stay for their importers."""
from max.gpu.host import DeviceBuffer, DeviceContext

# TOMBSTONE: MOJOLEARN_NN58_ACCUMULATE_STATUS (opt-in accumulate-with-status kernel, no A/B recorded), MOJOLEARN_NN59_DROPOUT_RESIDUAL (opt-in fused residual-dropout kernels, no A/B recorded) and the MOJOLEARN_IDN_ALL_OFF term deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at f75eb67b37.
# Restore: git apply experiments/removed/MOJOLEARN_NN58_ACCUMULATE_STATUS.patch and experiments/removed/MOJOLEARN_NN59_DROPOUT_RESIDUAL.patch
comptime NN58_ACCUMULATE_STATUS = False


def nn_dropout_residual_into[BACKWARD: Bool](ctx: DeviceContext, mut first: DeviceBuffer[DType.float32], mut second: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32], mut residual: DeviceBuffer[DType.float32], n: Int, offset: Int, p: Float32, scale: Float32, seed_lo: UInt32, seed_hi: UInt32, stream: UInt32) raises:
    """Refuses: the NN59 fused schedule is deleted (training/residual_dropout.mojo keeps the baseline)."""
    raise Error("NN59 is not enabled")
