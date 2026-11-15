# SPDX-License-Identifier: Apache-2.0
"""NI55--NI60 source-only graph neural and dropout experiment contracts.

Every switch is new, IDENTICAL-only and OFF without its own define. No compile,
identity, quality or timing evidence exists for these candidates. The controls
do not select graph clustering, nearest-neighbor or other classical algorithms.
"""
from checks.numerics import (
    ftz, identical_mul, identical_div,
)

# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (DELETE-OLD-PATH, EXP:2438 umbrella old path, rule e) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-35-neural.patch
# The NI55-NI60 arms below were deleted; each name stays a False constant for its importers.
# TOMBSTONE: MOJOLEARN_NI55_GRAPH_FEATURE4 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI55_GRAPH_FEATURE4.patch
comptime NI55_GRAPH_FEATURE4 = False
# TOMBSTONE: MOJOLEARN_NI57_GRAPH_TREE64 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI57_GRAPH_TREE64.patch
comptime NI57_GRAPH_TREE64 = False
# TOMBSTONE: MOJOLEARN_NI58_SAGE_FEATURE4 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI58_SAGE_FEATURE4.patch
comptime NI58_SAGE_FEATURE4 = False
# TOMBSTONE: MOJOLEARN_NI59_DROPOUT_CHANNEL (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI59_DROPOUT_CHANNEL.patch
comptime NI59_DROPOUT_CHANNEL = False
# TOMBSTONE: MOJOLEARN_NI60_DROPOUT_APPLY4 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI60_DROPOUT_APPLY4.patch
comptime NI60_DROPOUT_APPLY4 = False

# Four independent FP32 accumulators amortize graph metadata or mask reads
# without a feature-size dispatch boundary. Tails are masked by true dimensions.
comptime NEURAL_AUX_FEATURE_TILE = 4

# NI57 is a numerical version, not a physical thread tile: 64 ascending edge
# terms per logical leaf, then adjacent pair reduction with unpadded odd tails.
# The same definition is used for every CSR row and all four execution columns.
# The initial stack implementation keeps leaves serial; parallel leaf scheduling
# remains separate work and no throughput claim follows from this helper alone.
comptime GRAPH_EDGE_LEAF = 64
comptime GRAPH_FOLD_LEVELS = 32  # CSR offsets are Int32: at most 2**31-1 edges.

comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def graph_edge_term(e: Int, f: Int, n: Int, features: Int, mode: Int,
                    vals: _FP, h: _FP, csr: _IP) -> Float32:
    var column = Int(csr.unsafe_load(n + 1 + e))
    var value = ftz(h.unsafe_load(column * features + f))
    if mode == 0:
        value = ftz(identical_mul(ftz(vals.unsafe_load(e)), value))
    elif mode == 2:
        value = ftz(identical_div(value, ftz(vals.unsafe_load(e))))
    return value


@always_inline
def graph_reduce_tree64(row: Int, f: Int, n: Int, features: Int, mode: Int,
                        vals: _FP, h: _FP, csr: _IP) -> Float32:
    """Version NI57 graph sum/mean; mode-3 scaling stays on its original path."""
    var lo = Int(csr.unsafe_load(row))
    var hi = Int(csr.unsafe_load(row + 1))
    var count = hi - lo
    var stack = SIMD[DType.float32, GRAPH_FOLD_LEVELS](0.0)
    var occupied = 0
    var leaves = (count + GRAPH_EDGE_LEAF - 1) // GRAPH_EDGE_LEAF
    for leaf in range(leaves):
        var acc = Float32(0)
        var start = lo + leaf * GRAPH_EDGE_LEAF
        var end = min(start + GRAPH_EDGE_LEAF, hi)
        for edge in range(start, end):
            acc = ftz(acc + graph_edge_term(edge, f, n, features, mode, vals, h, csr))
        var level = 0
        while ((occupied >> level) & 1) != 0:
            acc = ftz(ftz(stack[level]) + ftz(acc))
            occupied -= 1 << level
            level += 1
        stack[level] = acc
        occupied += 1 << level
    var result = Float32(0)
    var have_value = False
    for level in range(GRAPH_FOLD_LEVELS):
        if ((occupied >> level) & 1) != 0:
            if have_value:
                result = ftz(ftz(stack[level]) + ftz(result))
            else:
                result = stack[level]
                have_value = True
    if mode == 1 and count > 0:
        result = ftz(identical_div(result, Float32(count)))
    return result
