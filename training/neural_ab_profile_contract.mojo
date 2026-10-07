# SPDX-License-Identifier: Apache-2.0
"""Shared NN54/NN57 arithmetic; host-only imports for CPU-only installations."""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul

# L11 (2026-10-07): the CE token-total fold is ONE switch with arms,
# -D MOJOLEARN_IDN_CE_TOKEN_FOLD=0|1|2: 0 = the pinned GEMM ones-fold
# (default), 1 = NN54 128-row leaf tree, 2 = NI35 256-token tree v2. Both
# arms run on the GPU (training/checks/loss.mojo identical_ce_forward_into)
# and on the host column (training/byte_lm_host_kernels.mojo) together.
comptime IDN_CE_TOKEN_FOLD_ARM = get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD", 0]()
comptime NN54_LOSS_PROFILE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_CE_TOKEN_FOLD_ARM == 1
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN57_NORM_PROFILE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN57_NORM_PROFILE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
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


def nn_reduce_host[LEAF: Int, SQUARE: Bool](x: _FP, n: Int) raises -> Float32:
    """The same leaf/pair functions on the host; a version oracle, not a test."""
    comptime if SQUARE:
        comptime if not NN57_NORM_PROFILE:
            raise Error("NN57 norm profile is not enabled")
    else:
        comptime if not NN54_LOSS_PROFILE:
            raise Error("NN54 loss profile is not enabled")
    if n < 0 or n > 2147483647:
        raise Error("neural reduction count exceeds native index range")
    return nn_reduce_host_admitted[LEAF, SQUARE](x, n)


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


