# SPDX-License-Identifier: Apache-2.0
"""2026-10-06 tree-only IDENTICAL candidates, all default OFF.

A defines a selector below; B omits it and keeps the incumbent settings.
This module imports no device API: host/device units share the same gates.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# T16 (resident Lossguide frontier, = I17) removed 2026-10-07: lost on NVIDIA and AMD.

# T17 Lossguide exact best-first batch width, an int sweep {32, 64, 128}.
# 0 keeps the pre-T17 width 32 (or 64 under the older
# MOJOLEARN_GBDT_LG_EXACT_BATCH64 arm).
# PROMOTED: absent now means 128, the IDENTICAL default (lane/grid-flips-1,
# 2026-10-08, Andrew 13:00Z "flip all of these"). Grid run ge123e6f9
# (NVIDIA L40S sm_89 + AMD MI325X gfx942, full board data, one scored run per
# arm), width 32 -> 128:
#   gbdt-lossguide taxi    NV 4674 -> 4243 ms (0.908x)  AMD 4241 -> 3942 ms (0.929x)
#   gbdt-lossguide istella NV 10082 -> 9421 ms (0.935x) AMD 8324 -> 7638 ms (0.918x)
#   gbdt-categorical taxicat AMD 10988 -> 10558 ms (0.961x); NV cell not
#   measured, quality FAIL on that cell's second check (open, both arms).
#   Geometric mean 0.922x; quality SAME on the lossguide cells; no bits change.
# `-D MOJOLEARN_TREES_T17_BATCH=32` measures the old width (the grid's
# "32" arm); 64 stays a sweep arm; `=0` still selects the pre-T17 rule.
comptime T17_BATCH = get_defined_int["MOJOLEARN_TREES_T17_BATCH", 128]() if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 0

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T18 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T18"]()

# T19 (fused partition chain under IDENTICAL) deleted 2026-10-08 by lane/grid-losers-1: grid ge123e6f9 quality loss
# (gbdt-depthwise istella AUC -0.43%, logloss +20.7%); see docs/apple-fast/EXPERIMENTS.md. T19_DEFER below is independent.

# T20 removed 2026-10-07: it set the same DEFER_HIST_COPY_1903 constant as T19_DEFER.

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T21 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T21"]()

# T21 stream count is a scheduling arm, never an accumulator-width change:
# one int sweep `-D MOJOLEARN_TREES_T21_STREAMS=1|2|4` (absent = 1, the
# incumbent single replica stream). Replaces T21_STREAM / T21_STREAM4.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T21_STREAMS = get_defined_int["MOJOLEARN_TREES_T21_STREAMS", 1]() if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 1

# T22 document-keyed ordered storage: fold-position compressed storage, the
# original document ids passed to the dither (no bits change).
# PROMOTED to the IDENTICAL default (lane/grid-flips-1, 2026-10-08, Andrew
# 13:00Z "flip all of these"). Grid run ge123e6f9 (NVIDIA L40S sm_89 + AMD
# MI325X gfx942, full board data, one scored run per arm), off -> on:
#   gbdt-ordered istella NV 54409 -> 37059 ms (0.681x)  AMD 39277 -> 37263 ms (0.949x)
#   gbdt-ordered taxi    NV 22296 -> 22069 ms (0.990x)  AMD 24991 -> 24159 ms (0.967x)
#   With ORD_STD_GRIDFOLD also on, istella NV reads 0.675x. Quality SAME,
#   NV vs AMD output hashes MATCH. Geometric mean 0.887x. Ordered fits of
#   gbdt-categorical reach this storage too; not measured there.
# `-D MOJOLEARN_TREES_T22_OFF` restores the incumbent storage (the grid's
# "off" arm). The old opt-in define is refused in
# core/six_lane_experiment_guards.mojo.
comptime T22 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TREES_T22_OFF"]()

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T25 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T25"]()

# T27 (alias of MOJOLEARN_2030_FUSED_EST_MOVE) deleted by lane/grid-prune 2026-10-07.

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T28 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T28"]()

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T29 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T29"]()

# Tried 2026-10-08 (MOJOLEARN_TREES_T29_VERSIONED, the 'versioned' arm of T29, run ge123e6f9): the generated V1 PairLogit
# objective (gbdt/targets/kernel/tree_t29_pair.mojo + tree_t29_units.mojo, host _group_values_t29); NV/AMD gbdt-rank-pairlogit
# istella 3.43x/3.76x SLOWER -> deleted (T29 'on' stays). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T30 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T30"]()

# Independent sub-arms preserve the incumbent when omitted.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T19_DEFER = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T19_DEFER"]()
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T28_SORT = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T28_SORT"]()
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T28_PREFIX = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T28_PREFIX"]()
# C47_GBDT (8 MiB cap on the lossguide exact batch width) deleted by lane/grid-prune 2026-10-07; T17_BATCH owns the width.

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T23 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T23"]()

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T24 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T24"]()

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T26 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T26"]()

# T29_YETI (YetiRank block-parallel tasks) DELETED by lane trees-small 2026-10-07:
# a no-op on NVIDIA and AMD, where `yeti_rank.yeti_block_parallel_for` already
# takes the block kernel under IDENTICAL (its negative arm is the existing
# `MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF`). docs/apple-fast/EXPERIMENTS.md row.

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T30_ADABOOST = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T30_ADABOOST"]()

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime C45_GBDT = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_C45_GBDT"]()

# ---- Lane S3 trees-structural (2026-10-07), all default OFF ----

# ORD_STD_GRIDFOLD (gbdt-ordered): the score-noise / scale-magnitude sums over
# the full grid (ORD_STD_GRID_BLOCKS x ORD_STD_GRID_TPB lanes, a fixed lane
# count independent of device and shape) instead of 256 serial lane chains.
# BITS CHANGE (new fold order, same on every vendor); the host column
# (`gbdt_oracle_ordered`) follows the same order under the same flag.
# PROMOTED to the IDENTICAL default (lane/grid-flips-1, 2026-10-08, Andrew
# 13:00Z "flip all of these"). Grid run ge123e6f9 (NVIDIA L40S sm_89 + AMD
# MI325X gfx942, full board data, one scored run per arm), off -> on:
#   gbdt-ordered taxi    NV 22296 -> 21538 ms (0.966x)  AMD 24991 -> 23004 ms (0.921x)
#   gbdt-ordered istella NV 54409 -> 54239 ms (0.997x)  AMD 39277 -> 38241 ms (0.974x)
#   Geometric mean 0.964x. Quality SAME; NV vs AMD output hashes MATCH.
# `-D MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF` restores the 256 lane chains on
# the device and in the host column together (the grid's "off" arm). The old
# opt-in define is refused in core/six_lane_experiment_guards.mojo.
comptime ORD_STD_GRIDFOLD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF"]()

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
# LG_LEVEL_ROUNDS (gbdt-lossguide): when the leaf budget cannot bind
# (max_leaves >= 2^max_depth, a parameter test), one exact search round per
# depth level with an uncapped frontier. Bit-neutral (see the call sites).
comptime LG_LEVEL_ROUNDS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_LG_LEVEL_ROUNDS"]()

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
# CTR_SORTFREE_SUMS (gbdt-categorical, gbdt-ordered non-symmetric leaves):
# per-(permutation, leaf) sums without the per-permutation radix sort.
comptime CTR_SORTFREE_SUMS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_CTR_SORTFREE_SUMS"]()

# ======== lane fg-gbdt-ordered (2026-10-09): the Ordered fold-arm overhead ========
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
# Plan: docs/plans/flagship-gaps-20261009/read_gbdt_ordered.md, ideas F, C, A, E.
# Target: the fold arm's per-tree overhead over the plain tree (taxi 35 ms/tree
# against CatBoost's 25 ms). Every switch below is IDENTICAL-only; FAST builds
# keep their own (Apple ORD_ALL / ORDERED_BATCH_EST) paths untouched.

# F ORD_SKIP_UNSEARCHED_PERM (default ON, pure waste removal; `_OFF` restores).
# `fit_ordered` picks the structure's learn permutation as
# `rng % (learn_count - 1)` (CatBoost's modulus, dynamic_boosting.h:282-289),
# so the LAST learn permutation (`learn_count - 1`, when learn_count > 1) is
# never searched on. Its only readers are its own fold tasks, which move only
# its own fold cursors; no derivative launch, score, structure or exported
# leaf reads them (the exported leaves come from the estimation permutation,
# `est_p = perm_count - 1`, whose task and cursor are untouched). The rng draws
# are per tree, outside the tasks, so the stream is unchanged. Skipped: that
# permutation's F fold tasks (1/L of the estimation stage: F x ~11 launches and
# F x 3 small H2D per tree at L = 3), its partition sort (7 launches + 3 D2H a
# tree) and settle, and its resident buffers (F cursors = ~2n floats, the
# own-order y/w gathers = 2n floats, the partition's 2n words, F task slots).
# Its fold-order index copy (T22) was never built: the searcher caches
# `fold_cindex` per searched `permutation_id` only. BITS: none of the exported
# model (output hash unchanged). A traced run keeps the tasks (trace.enabled),
# so `*.perm.<last>.fold.*` records stay comparable with the host column; the
# host column (`gbdt_oracle_ordered`) is untouched.
comptime ORD_SKIP_UNSEARCHED_PERM = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM_OFF"]()

# C IDN_ORD_CAT_PLANES (default ON, launch removal; `_OFF` restores).
# Each learn permutation's F fold cursors live in ONE buffer at the
# concatenated fold offsets (sub-buffer views, as Apple's ORDERED_CAT_CURSORS),
# and each searchable learn permutation keeps its own-order targets and weights
# laid out the same way once per fit (2 x total floats). The per-tree fold
# derivatives are then ONE `launch_approximate` over `total` (~2n rows) plus
# ONE plane scatter instead of F of each (2F = 32 launches at 16 folds -> 2),
# and the searcher's pooled per-tree fold bins are ONE
# `fold_bins_from_table_kernel` launch from a fit-long partition-start table
# instead of 2F (create + fill + copy) and a drain (the kernel was written for
# Apple ORD_ALL and is un-gated here). Cost reasoning: launches and one drain
# per tree, independent of shape; the derivative traffic is the same 2n rows.
# BITS: none (every position runs the same per-row statement on the same
# operands; the fold bins are the same integers). Host column untouched.
comptime IDN_ORD_CAT_PLANES = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ORD_CAT_PLANES_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

# A ORD_HIST_FOLD_SKIP (default OFF, a schedule change; A/B owed).
# The fold arm's 8-bit one-byte histogram launches one block per (feature
# block x doc split, part, fold partition): grid.z = 2F. The fold partitions
# grow geometrically, so at depth d the smallest ones average far fewer
# documents per part than a block has threads, and each such block still
# zeroes its 32 KB shared Int32 histogram, reduces it and scans its writeback.
# Under this switch the leading fold partitions z whose document count is
# below `parts x ORD_HIST_FOLD_SKIP_DOCS` (a size rule: block fixed cost
# against per-document global atomics) are histogrammed by a small-fold kernel
# (`ord_small_fold_hist8_kernel`: one block per (feature block, fold
# partition, part group), every document adds its two dithered Int32 values
# per feature straight into the histogram cell with a global Int32 atomic),
# then one conversion pass writes `Float32(Int(count)) / scale` with the
# 1e-20 write guard, the one-block writeback's statement; the regular grid's
# blocks for those z return at entry. Int32 addition is exact in any order and
# wraps as the shared counters wrap, so every cell is the multiplier-1 cell.
# BITS: none. Host column untouched. Reached on the device-scale route (the
# IDENTICAL default, `submit_compute_dev`), single device, 8-bit fixed one-byte
# policy only; binary and half-byte features keep the regular grid.
comptime ORD_HIST_FOLD_SKIP = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_ORD_HIST_FOLD_SKIP"]()
# the per-part document budget below which a fold partition takes the small
# kernel (default 256 = one document per thread of the 8-bit block); an int
# sweep arm for the grid, not a shape rule
comptime ORD_HIST_FOLD_SKIP_DOCS = get_defined_int["MOJOLEARN_ORD_HIST_FOLD_SKIP_DOCS", 256]()

# E IDN_ORD_INDEX_SHARED (default OFF, a storage change; A/B owed).
# T22 keeps, per searched learn permutation, a fold-order copy of the
# compressed index (n_cols x 2n words: ~2n because the fold layout repeats
# every learn prefix). Under this switch each searched permutation keeps ONE
# permutation-order copy (n_cols x n words, half the bytes) plus one fold
# position -> permutation index map shared by every permutation (2n words,
# `make_fold_doc_indices_device` at the identity). Each level gathers its
# documents through the map (the pre-T22 gather, now into permutation order,
# so a leaf's documents stay near-contiguous), the histogram and split
# kernels read the permutation-order index at stride n, and the dither keys
# on `perm[i]` = the document id, which is T22's key. BITS: none (same words,
# same dither keys, same cells). Host column untouched. Cost: memory
# n_cols x n words less per searched permutation (istella-sized data: ~0.9 GB
# each), one 2n-position gather per level more (12 B/position). The partition
# readbacks (`_PermPartition` settle) are NOT dead before the batched leaf
# estimation (idea B) lands: `task_partition` builds each task's host leaf
# sizes from them for the oracle, so they stay.
comptime IDN_ORD_INDEX_SHARED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_ORD_INDEX_SHARED"]()
