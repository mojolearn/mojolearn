"""G01--G12: opt-in Apple FAST tree experiments, 2026-10-06.

All source is uncompiled, unverified and unmeasured. No performance or quality
claim is made. Each flag requires both Apple and FAST; no umbrella or default
promotion. See experiments/apple_fast_trees/G.json for A/B prerequisites.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime AFT_G_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)

# G01: one uniform-barrier block per feature scans 128-bin tiles.
# No performance/quality evidence; FAST prefix sums may change rounding.
comptime AFT_G01 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G01"]()
# G02: occupancy and per-partition row budget choose document replication.
# No performance/quality evidence; retain every row and the multiplier cap.
comptime AFT_G02 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G02"]()
# G03: one leader folds winners per block and broadcasts through shared memory.
# No performance/quality evidence; preserve original record order and tie rules.
comptime AFT_G03 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G03"]()
# G04: 512-thread partition reduction, without chunk atomics or extra planes.
# No performance/quality evidence; FAST reduction order may change.
comptime AFT_G04 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G04"]()
# G05: 512-thread quantization block at the original eight rows per lane.
# No performance/quality evidence; identical borders and comparison routine.
comptime AFT_G05 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G05"]()
# G06: 128-thread compressed-word packing with sixteen rows per lane.
# No performance/quality evidence; same 2048 rows/block and bit layout.
comptime AFT_G06 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G06"]()
# G07: 128-thread Apple leaf walker objective reduction and direction traversal.
# No performance/quality evidence; leave shared/fused statistics contracts intact.
comptime AFT_G07 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G07"]()
# G08: four independent cursor rows per leaf-update loop iteration.
# No performance/quality evidence; same leaves, learning rate and row coverage.
comptime AFT_G08 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G08"]()
# G09: eight-row numerator preparation tiles for exact ordered CTR sums.
# No performance/quality evidence; preserve segment signs and target predicates.
comptime AFT_G09 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G09"]()
# G10: four coalesced rows per weighted-frequency/mean-scatter CTR worker.
# No performance/quality evidence; preserve priors, index maps and prefix sums.
comptime AFT_G10 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G10"]()
# G11: 128-thread ordered batched derivative/weight reduction.
# No performance/quality evidence; same prefix tasks, eight chunks and RNG.
comptime AFT_G11 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G11"]()
# G12: four-row ordered apply tiles with cached task descriptor bounds.
# No performance/quality evidence; every row still resolves its own prefix model.
comptime AFT_G12 = AFT_G_APPLE_FAST and is_defined["MOJOLEARN_AFT_G12"]()
