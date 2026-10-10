"""Apple FAST experiment switches for the oblivious (symmetric) split
search, lane/apple-fast-sym-hist (2026-10-03).

Every switch is `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()` AND its own `-D MOJOLEARN_SYM_<NAME>` define,
so IDENTICAL (and FAST on any other vendor, and FAST on Apple without the
define) compiles main's code unchanged. `-D MOJOLEARN_SYM_HIST_ALL` turns
on SORT_SWAP + RESOLVE_BLOCK (the other four are recorded DROPs; see
each switch). Profile and mechanisms:
docs/apple-fast/notes/sym-hist.md, docs/apple-fast/ab/sym-hist.md.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


#: the FAST + Apple guard every switch below is inside
comptime SYM_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)

#: all six switches at once
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue).
#: apple-fast LEDGER 2026-10-03: symhist-all noise DROP (all six, old
#: base). The umbrella now holds only SORT_SWAP + RESOLVE_BLOCK, the two
#: switches with no named verdict (the batch's 'DROPPED: symhist x2' names
#: no tags).
comptime SYM_HIST_ALL = SYM_FAST_APPLE and is_defined["MOJOLEARN_SYM_HIST_ALL"]()

#: the one-bit radix pass of the level split writes into `tmp_bins` /
#: `tmp_indices` and the handles are SWAPPED instead of copied back: the
#: two `copy_u32_kernel` launches per level go away
#: (`pointwise_optimization_subsets.mojo::split_subsets_from_desc`).
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue).
comptime SYM_SORT_SWAP = SYM_FAST_APPLE and (
    SYM_HIST_ALL or is_defined["MOJOLEARN_SYM_SORT_SWAP"]()
)

#: `pw_resolve_pack_bins_kernel` folds the per-block score records ONCE
#: per threadgroup (threads stride the records, then a shared-memory tree
#: reduce under the same three-key order) instead of once per thread
#: (`pointwise_split_resolve.mojo`).
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue).
comptime SYM_RESOLVE_BLOCK = SYM_FAST_APPLE and (
    SYM_HIST_ALL or is_defined["MOJOLEARN_SYM_RESOLVE_BLOCK"]()
)

#: TOMBSTONE: MOJOLEARN_SYM_GATHER_FUSED (DROPPED-noise: symhist-gather istella/taxi noise, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (sizes + both gathers in one launch); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_GATHER_FUSED.patch; record in docs/TOMBSTONES.md.

#: TOMBSTONE: MOJOLEARN_SYM_PART_STATS_PAR (DROPPED-noise: symhist-part-stats istella/taxi noise, old base) deleted 2026-10-09
#: on lane/owed-deletions-D1 (per-partition sums on a partitions x chunks grid with atomics); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_PART_STATS_PAR.patch; record in docs/TOMBSTONES.md.

#: on a partial pass the fold scan and the sibling subtraction are one
#: launch: the thread that scans a feature's folds for the computed child
#: also writes parent - child for the sibling
#: (`split_properties_helpers.mojo::scan_sub_pointwise_histograms_kernel`).
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue).
#: apple-fast LEDGER 2026-10-03 batchv: DROP symhist scan-sub (noise), old base;
#: recorded loser, OUT of SYM_HIST_ALL, not in the A/B table.
comptime SYM_SCAN_SUB_FUSED = SYM_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_SCAN_SUB_FUSED"]()
)

#: TOMBSTONE: MOJOLEARN_SYM_HIST_MULT (DROPPED-noise: symhist-hist-mult istella/taxi noise, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (2x the SM count's worth of histogram blocks); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_HIST_MULT.patch; record in docs/TOMBSTONES.md.

