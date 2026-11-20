# SPDX-License-Identifier: Apache-2.0
"""Shared NN54/NN57 arithmetic; host-only imports for CPU-only installations."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul

# L11 (2026-10-07): the CE token-total fold is ONE switch with arms,
# -D MOJOLEARN_IDN_CE_TOKEN_FOLD=0|1: 0 = the pinned GEMM ones-fold
# (default), 1 = NN54 128-row leaf tree. (TOMBSTONE: MOJOLEARN_IDN_CE_TOKEN_FOLD=2
# (noise) deleted 2026-10-10 by 3034789a2; code recoverable at ca25d9321.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_TOKEN_FOLD-arm2.patch) The
# arm runs on the GPU (training/checks/loss.mojo identical_ce_forward_into)
# and on the host column (training/byte_lm_host_kernels.mojo) together.
# TOMBSTONE: MOJOLEARN_IDN_CE_TOKEN_FOLD (arm 1, the NN54 128-row leaf tree; record read: EXP:2009 noise) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at 19d2549589.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_TOKEN_FOLD.patch
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old-path umbrella term; owner another lane) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at 19d2549589.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-32-neural.patch
# (Sweep 2026-11-15: IDN_CE_TOKEN_FOLD_ARM = 0, NN54_LOSS_PROFILE = False and the checked nn_reduce_host removed with
# their readers' dead arms in training/byte_lm_host_kernels.mojo, training/checks/loss.mojo, training/checks/loss_oracle.mojo.)
comptime NN57_NORM_PROFILE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN57_NORM_PROFILE"]()
)
comptime _FP = MutPointer[Float32, MutAnyOrigin]


@always_inline
def nn_reduce_leaf[LEAF: Int, SQUARE: Bool](x: _FP, n: Int, leaf: Int) -> Float32:
    comptime assert LEAF > 0
    var value = Float32(0.0)
    var first = leaf * LEAF
    for i in range(first, min(n, first + LEAF)):
        var term = ftz(x[i])
        comptime if SQUARE:
            term = ftz(identical_mul(term, term))
        value = ftz(ftz(value) + term)
    return value


@always_inline
def nn_reduce_pair(left: Float32, right: Float32) -> Float32:
    return ftz(ftz(left) + ftz(right))


def nn_reduce_host_admitted[LEAF: Int, SQUARE: Bool](x: _FP, n: Int) -> Float32:
    """Caller admitted shape; same arithmetic as the checked wrapper."""
    if n == 0:
        return Float32(0.0)
    var width = (n + LEAF - 1) // LEAF
    var values = List[Float32](length=width, fill=Float32(0.0))
    for i in range(width):
        values[i] = nn_reduce_leaf[LEAF, SQUARE](x, n, i)
    while width > 1:
        var count = (width + 1) // 2
        for i in range(count):
            var v = values[2 * i]
            if 2 * i + 1 < width:
                v = nn_reduce_pair(v, values[2 * i + 1])
            values[i] = v
        width = count
    return values[0]


