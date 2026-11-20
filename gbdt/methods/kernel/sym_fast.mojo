"""Apple FAST experiment switches for the oblivious (symmetric) split
search, lane/apple-fast-sym-hist (2026-10-03).

Every switch is `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()` AND its own `-D MOJOLEARN_SYM_<NAME>` define,
so IDENTICAL (and FAST on any other vendor, and FAST on Apple without the
define) compiles main's code unchanged. Profile and mechanisms:
docs/apple-fast/notes/sym-hist.md, docs/apple-fast/ab/sym-hist.md.
"""

from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


#: the FAST + Apple guard every switch below is inside
comptime SYM_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
)

# TOMBSTONE: MOJOLEARN_SYM_HIST_ALL (NEUTRAL: rab4-symhist istella -0.1%, taxi -0.2%) umbrella of SORT_SWAP +
# RESOLVE_BLOCK deleted 2026-11-15 by lane/expdel-05-trees; code recoverable at 24c9108159.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_HIST_ALL.patch

# TOMBSTONE: MOJOLEARN_SYM_SORT_SWAP (NEUTRAL: rab7-symsortswap istella +0.3%, taxi +0.1%) deleted 2026-11-15 by
# lane/expdel-05-trees; code recoverable at 1805cb6b73.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_SORT_SWAP.patch

# TOMBSTONE: MOJOLEARN_SYM_RESOLVE_BLOCK (DELETE-EXPERIMENT, EXP:1303 NEUTRAL alone, never built or timed as its own arm, rule g) deleted 2026-11-15 by lane/expdel2-43-trees; code recoverable at e301d611ef.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_RESOLVE_BLOCK.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-trees: opt-in arm back as it was, default off, Apple fast only; default path unchanged.
#: `pw_resolve_pack_bins_kernel` folds the per-block score records ONCE
#: per threadgroup (threads stride the records, then a shared-memory tree
#: reduce under the same three-key order) instead of once per thread
#: (`pointwise_split_resolve.mojo`).
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue).
comptime SYM_RESOLVE_BLOCK = SYM_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_RESOLVE_BLOCK"]()
)

#: TOMBSTONE: MOJOLEARN_SYM_GATHER_FUSED (DROPPED-noise: symhist-gather istella/taxi noise, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (sizes + both gathers in one launch); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_GATHER_FUSED.patch; record in docs/TOMBSTONES.md.

#: TOMBSTONE: MOJOLEARN_SYM_PART_STATS_PAR (DROPPED-noise: symhist-part-stats istella/taxi noise, old base) deleted 2026-10-09
#: on lane/owed-deletions-D1 (per-partition sums on a partitions x chunks grid with atomics); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_PART_STATS_PAR.patch; record in docs/TOMBSTONES.md.

#: TOMBSTONE: MOJOLEARN_SYM_SCAN_SUB_FUSED (DROPPED-noise: symhist-scan-sub istella/taxi noise, old base) deleted 2026-10-09
#: on lane/owed-deletions-D1 (fold scan + sibling subtraction in one launch); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_SCAN_SUB_FUSED.patch; record in docs/TOMBSTONES.md.
#: TOMBSTONE: MOJOLEARN_SYM_HIST_MULT (DROPPED-noise: symhist-hist-mult istella/taxi noise, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (2x the SM count's worth of histogram blocks); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_HIST_MULT.patch; record in docs/TOMBSTONES.md.

