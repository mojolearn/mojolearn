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

#: after the sort, ONE launch gathers weights and targets by the new
#: `indices` and writes the partition sizes, replacing the sizes kernel and
#: the two gathers (`pointwise_optimization_subsets.mojo::update_subsets_stats`).
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue).
#: apple-fast LEDGER 2026-10-03 batchv: DROP symhist gather (noise), old base;
#: recorded loser, OUT of SYM_HIST_ALL, not in the A/B table.
comptime SYM_GATHER_FUSED = SYM_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_GATHER_FUSED"]()
)

#: the per-partition weight / target sums run on a (partitions x chunks)
#: grid with a shared-memory reduce per block and a global float atomic
#: add per block, after a fill, instead of one 1024-thread block per
#: partition (`pointwise_scores.mojo::partition_update_chunked_kernel`).
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue). KNOWN: the per-block float atomic add
#: makes the partition sums' fold order run-dependent (FAST only; check auc
#: and run-to-run on the A/B).
#: apple-fast LEDGER 2026-10-03 batchv: DROP symhist part-stats (noise), old base;
#: recorded loser, OUT of SYM_HIST_ALL, not in the A/B table.
comptime SYM_PART_STATS_PAR = SYM_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_PART_STATS_PAR"]()
)

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

#: the pointwise histogram launchers ask CatBoost's document multiplier for
#: `SYM_HIST_MULT_FACTOR` times as many blocks as the SM heuristic would
#: (`pointwise_kernels.mojo::pw_block_multiplier`), still capped at 64 and
#: at >= 10k rows per block.
#: recovery 2026-10-04 (lane/apple-fast-rec-sym): source
#: lane/apple-fast-sym-hist@3bb4db314; never built or timed (its prebuilt M3
#: arms never reached the queue).
#: apple-fast LEDGER 2026-10-03: symhist hist-mult noise DROP, old base;
#: recorded loser, OUT of SYM_HIST_ALL, not in the A/B table.
comptime SYM_HIST_MULT = SYM_FAST_APPLE and (
    is_defined["MOJOLEARN_SYM_HIST_MULT"]()
)
comptime SYM_HIST_MULT_FACTOR = 2

#: the block the chunked partition stats kernel runs at
comptime SYM_PART_STATS_BLOCK = 256

#: the most chunks one partition is cut into (SYM_PART_STATS_PAR)
comptime SYM_PART_STATS_MAX_CHUNKS = 64

#: blocks per SM the chunked partition stats grid aims for
comptime SYM_PART_STATS_BLOCKS_PER_SM = 8
