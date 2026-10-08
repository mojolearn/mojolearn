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

# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime T29_VERSIONED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T29_VERSIONED"]()

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
