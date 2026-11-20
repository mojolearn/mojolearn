# SPDX-License-Identifier: Apache-2.0
"""NI55--NI60 source-only graph neural and dropout experiment contracts.

Every switch is new, IDENTICAL-only and OFF without its own define. No compile,
identity, quality or timing evidence exists for these candidates. The controls
do not select graph clustering, nearest-neighbor or other classical algorithms.
"""

# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (DELETE-OLD-PATH, EXP:2438 umbrella old path, rule e) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-35-neural.patch
# The NI55-NI60 arms below were deleted. (Sweep 2026-11-15: their five False constants, graph_reduce_tree64,
# graph_edge_term, GRAPH_EDGE_LEAF and GRAPH_FOLD_LEVELS removed: no reader left.)
# TOMBSTONE: MOJOLEARN_NI55_GRAPH_FEATURE4 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI55_GRAPH_FEATURE4.patch
# TOMBSTONE: MOJOLEARN_NI57_GRAPH_TREE64 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI57_GRAPH_TREE64.patch
# TOMBSTONE: MOJOLEARN_NI58_SAGE_FEATURE4 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI58_SAGE_FEATURE4.patch
# TOMBSTONE: MOJOLEARN_NI59_DROPOUT_CHANNEL (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI59_DROPOUT_CHANNEL.patch
# TOMBSTONE: MOJOLEARN_NI60_DROPOUT_APPLY4 (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-35-neural; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_NI60_DROPOUT_APPLY4.patch

# Four independent FP32 accumulators amortize graph metadata or mask reads
# without a feature-size dispatch boundary. Tails are masked by true dimensions.
comptime NEURAL_AUX_FEATURE_TILE = 4
