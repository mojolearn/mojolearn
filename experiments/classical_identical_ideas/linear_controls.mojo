# SPDX-License-Identifier: Apache-2.0
"""C13-C28 classical IDENTICAL opt-in controls.

Unless scoped evidence below says otherwise, switches remain NOT COMPILED —
NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Absence preserves the incumbent; measurements do not imply default promotion.
Numerical profiles are shared by the host and all three device columns.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime CLASSICAL_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# T.C13.only, measured source 6fe3cfce38fd (2026-10-07): remain opt-in/off.
# Full LassoCV Taxi/Istella, ElasticNetCV Taxi/Istella; candidate/baseline time
# ratios NVIDIA sm90: 4.0132/1.1120/0.7913/1.1144; AMD gfx942:
# 1.3997/1.1302/1.3511/1.0997. Both Taxi CV quality gates fail on both vendors.
# One excluded warmup + one scored sample per arm; same-arm NV/AMD outputs and
# complete declared model state match. Apple/host/PTX identity remains pending.
# Evidence: experiments/six_lane_integration/measurements/20261006/retained-pairs.json
# and BOARD.md. C13+C18 reproduces the saved C13 quality loss; no promotion.
# C13 split (lane/ridgecv-c13, 2026-10-07). C13_FOLD_STATS now covers RidgeCV
# only (x_linear/ridgecv.mojo, the RidgeCV route in x_linear/device.mojo and
# the host binding's RidgeCV scratch). Default ON in IDENTICAL: in the six-lane
# campaign (source 55a815e, all candidates on, one excluded warmup + one scored
# run per arm) RidgeCV full Taxi ran 2.75 s vs 29.95 s incumbent on NVIDIA
# sm90 (0.092x) and 2.38 s vs 28.03 s on AMD gfx942 (0.085x); full Istella
# 46.4 s vs 57.9 s (0.80x) and 89.9 s vs 104.0 s (0.86x). r2/rmse identical to
# the incumbent on both vendors, NVIDIA and AMD output hashes equal. C13 is the
# only candidate control on the RidgeCV route (catalog.json). Evidence:
# experiments/six_lane_integration/measurements/20261006/BOARD.md (ridge-cv rows).
# The `-D MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF` switch (per-fold passes in an
# IDENTICAL build) was tried 2026-10-08 (lane grid-act-3, run ge123e6f9): ridge-cv
# NV/AMD 1.19x/1.11x istella, 19.5x/18.0x taxi SLOWER, quality SAME. The switch is
# deleted and the define refused: the fold cache is the only IDENTICAL RidgeCV route.
# FAST builds keep the per-fold passes (C13_FOLD_STATS is False outside IDENTICAL).
# Recoverable at main bc10b8b56.
comptime C13_FOLD_STATS = CLASSICAL_IDN
# The LassoCV/ElasticNetCV fold cache (C13_CD_FOLD_STATS) was deleted on
# lane/classical-cv (2026-10-07): T.C13.only measured it 4.0x/1.4x slower on
# Taxi and quality-failing on both vendors (row in docs/apple-fast/EXPERIMENTS.md).
# lane/classical-cv (2026-10-07), NEW, opt-in, NOT MEASURED: LassoCV /
# ElasticNetCV fold statistics from fold-aligned compensated block partials
# (x_linear/enetcv_blocks.mojo). Each fold's row span is cut into
# ENETCV_FB_CHUNKS chunks (one switch, arms = chunks per fold 16|32|64, a
# fixed count, not a data shape); every row is read once per pass, each
# chunk's sums are compensated (TwoSum / Dot2), chunks merge ascending, and
# the training sets combine fold statistics ascending by the parallel-axis
# rule. Changes bits (host column and both GPU vendors together).
comptime ENETCV_FOLD_BLOCKS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS"]()
comptime ENETCV_FB_CHUNKS = get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS", 32]() if ENETCV_FOLD_BLOCKS else 32
# TOMBSTONE: MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS (quality loss) deleted 2026-10-07 by f4de82db4; code recoverable at f4de82db4^.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch; record in docs/TOMBSTONES.md.
comptime C14_GROUP_RHS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C14_GROUP_RHS"]()
# lane/classical-cv-folds (2026-10-07), NEW, opt-in, NOT MEASURED: RidgeCV's
# float-float fallback (x_linear/ridge_ff_blocks.mojo). The incumbent runs one
# thread per statistic serially over every training row (`kf_ff_unit_kernel`,
# `ridge_ff_unit_kernel`) and the float-float Cholesky of each untrusted alpha
# on ONE thread (`kf_ff_solve_kernel` at grid 1 / block 1). Here every
# float-float statistic is FOLD_BLOCK-row block partials (one thread per
# (statistic, block), the rw_gram_parts_kernel shape) folded blocks
# ascending, and the solve is one block team per alpha (`t_ff_cholesky`, the
# t_cholesky split: every entry keeps its own chain). Changes bits (the block
# fold order of the float-float sums; the host column and both GPU vendors
# together). Cost reasoning: the work n * (d+1)^2 / 2 float-float adds is
# unchanged; the thread count goes from (d+1)^2 / 2 to n / FOLD_BLOCK times
# that, with each block's threads reading the same row tile. No shape rule.
comptime RIDGECV_FF_BLOCKED = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED"]()
# lane/classical-cv-folds (2026-10-07); PROMOTED to the IDENTICAL default
# 2026-10-08 (lane/grid-act-2, grid run ge123e6f9: NVIDIA L40S sm_89 + AMD
# MI325X gfx942, full board data, one run per arm; r2 and rmse identical A vs
# B on every cell; NV hash == AMD hash on every arm). LassoCV / ElasticNetCV
# held-out scoring as (path, alpha, row-block) partials. The old
# `ecv_score_staged_kernel` is ONE block per (fold, l1_ratio) path walking
# every held-out row for every alpha. Here each block owns
# ENETCV_SCORE_BLOCKS rows of the fold's span (rows per block, legal
# 1024 | 4096, a fixed count, not a data shape), thread k folds alpha k's
# squared errors over them from zero, and a thread per (path, alpha) folds
# the block partials ascending (`fold_parts`, the kf_sq / kf_score shape).
# Changes bits (the block fold order; host column and both GPU vendors
# together). Grid ge123e6f9, NV / AMD ms, unblocked -> 4096:
#   enet-cv  istella 1052.9 -> 555.2 / 3567.8 -> 818.1 (0.348x combined),
#            taxi 122.4 -> 101.6 / 483.8 -> 359.2 (0.785x);
#   lasso-cv istella 1041.8 -> 555.9 / 3530.4 -> 848.7 (0.358x),
#            taxi 122.6 -> 97.9 / 513.2 -> 364.5 (0.753x).
# 1024 is also faster (0.541x) but 4096 wins every cell. Cost reasoning: the
# score is n_heldout * A * d multiply-adds per path; one block per path
# serializes all of it on one SM, while span / ESB_ROWS blocks per path
# spread it over the device. 4096 rows per block keeps each block's work
# (4096 * A * d) large against its launch and staging cost and makes the
# second-level fold short (span / 4096 partials per alpha), which is why it
# beats 1024 (four times the partials to write and fold for the same work).
# Absent = 4096 in IDENTICAL; -D MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS=1024
# is the smaller-block arm; -D MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF is
# the old unblocked path (value 0; an explicit =0 is refused by the guard,
# use _OFF). FAST keeps the unblocked path.
comptime ENETCV_SCORE_BLOCKS_OFF = is_defined["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF"]()
comptime _ESB_RAW = 0 if ENETCV_SCORE_BLOCKS_OFF else get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS", 4096]()
comptime ENETCV_SCORE_BLOCKS = _ESB_RAW if CLASSICAL_IDN else 0
comptime ENETCV_SCORE_BLOCKS_LEGAL = _ESB_RAW == 0 or _ESB_RAW == 1024 or _ESB_RAW == 4096
comptime C15_FACTOR_SOLVE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE"]()
# lane/classical-structural (2026-10-07): LINEAR_GRAM_SOLVE (OLS and Ridge).
# One resident centered-Gram fit (glm/impl/gram_solve.mojo) instead of OLS's
# 14-panel TSQR of a centered copy and Ridge's four PCIe crossings + untiled
# U = A V. Bits change; the host column (glm/host/gram_solve_host.mojo)
# follows the same comptime flag.
# PROMOTED to the IDENTICAL default (lane/grid-flips-1, 2026-10-08, Andrew
# 13:00Z "flip all of these"): grid run ge123e6f9 (NVIDIA L40S sm_89 + AMD
# MI325X gfx942, full board data, one scored run per arm), baseline -> gram:
#   OLS taxi     NV 34.04 -> 17.47 ms (0.513x)  AMD 14.46 -> 8.07 ms (0.558x)
#   Ridge taxi   AMD 16.32 -> 3.06 ms (0.188x); NV single-arm cell unmeasured,
#                the ridge all_on cell (gram + c02_linear_pair, which alone is
#                neutral 0.98x) read NV 49.95 -> 4.77 ms (0.096x)
#   Ridge istella AMD 1020.6 -> 1027.2 ms (1.006x, neutral)
#   OLS istella  NV 816.2 -> 1008.3 ms (1.235x)  AMD 468.9 -> 532.2 ms (1.135x)
#                SLOWER: the one cell that reads slower. The route is
#                fallback-gated: when the Gram does not factor or a pivot
#                fails the trust gate, linear_gram_fit returns status 1 and
#                the caller then runs the incumbent TSQR, paying both. Whether
#                this cell took that fallback was not isolated in the run.
#   Geometric mean over all measured cells 0.80x. Quality identical (r2 and
#   rmse within 1e-6 in every cell); NV vs AMD output hashes MATCH.
# `-D MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF` restores the incumbent
# routes (the grid's "off" arm). The old opt-in define is refused in
# core/six_lane_experiment_guards.mojo.
comptime LINEAR_GRAM_SOLVE = CLASSICAL_IDN and not is_defined["MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF"]()
comptime C16_GLM_FUSED = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C16_GLM_FUSED"]()
comptime C17_OVR = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C17_OVR"]()
comptime C17_LS_TRIALS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C17_LS_TRIALS"]()
# T.C18.residual-only, measured source 6fe3cfce38fd: remain opt-in/off, pending
# qualification. Full LassoCV Taxi/Istella, ElasticNetCV Taxi/Istella ratios
# (candidate/baseline): NVIDIA sm90 1.0088/1.0061/0.9697/1.0067;
# AMD gfx942 1.0039/1.0336/0.9464/0.9740. Mixed timing; no blanket win claimed.
# Saved quality metrics match the incumbent in all eight vendor/workload cells;
# independent quality admission and required Apple/host/PTX identity are pending.
# NV/AMD same-arm outputs and complete declared model state match. One excluded
# warmup + one scored sample per arm; combination/scope qualification incomplete.
# Evidence: experiments/six_lane_integration/measurements/20261006/retained-pairs.json
# and BOARD.md; C13+C18 quality failures do not establish C18 as their cause.
comptime C18_RESIDUAL_NEXT = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT"]()
# C19: ONE integer sweep (was ORDERED_128/ORDERED_32, two defines where 32
# silently won when both were on). SGD_PS samples per ordered launch; legal
# set {32, 128, 2048}; absent = 2048 = incumbent. Sample order, time index and
# per-sample updates are unchanged; only the launch/witness granularity moves.
# Legal set enforced in core/six_lane_experiment_guards.mojo.
comptime C19_SGD_CHUNK = get_defined_int["MOJOLEARN_CLASSICAL_C19_SGD_CHUNK", 2048]() if CLASSICAL_IDN else 2048
# TOMBSTONE: MOJOLEARN_CLASSICAL_C20_ROW_CACHE (serial shape) deleted 2026-10-07 by fc5c573f7; code recoverable at fc5c573f7^.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C20_ROW_CACHE.patch; record in docs/TOMBSTONES.md.
# C20_ROW_CACHE deleted 2026-10-07 (serial per-row host loop); refused in core/six_lane_experiment_guards.mojo.
comptime C20_PAIR_LOAD = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C20_PAIR_LOAD"]()
comptime C21_EXTREMA = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C21_EXTREMA"]()
comptime C22_TRIANGLE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C22_TRIANGLE"]()
# lane classical-decomp (2026-10-07): the old C23_CENTERED_PANELS define is
# split per algorithm family. Its PCA use (the ONE_PASS arm of
# MOJOLEARN_CLASSICAL_PCA_COV, =23) was deleted 2026-10-08 (below); its MCD use is C23_MCD. Both run
# row-parallel (core/blocked_moments.mojo); the per-cell serial kernels
# (one GPU thread per covariance cell over every row) are deleted.
# C23_MCD: MinCovDet/EllipticEnvelope `emp_cov_at` as the centered Gram
# around the candidate location, leaves of 256 rows folded by the binary
# counter: the old C23 cell's value (x_decomp/classical_cells.mojo, the host
# column), now computed in parallel. NOT MEASURED.
comptime C23_MCD = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C23_MCD"]()
# PCA covariance, ONE switch with a named arm (the old C04-over-C23 silent
# priority is gone): -D MOJOLEARN_CLASSICAL_PCA_COV=4 is the C04 arm
# (two passes: the column mean, then the centered Gram around it read
# straight from X, leaves of contract_leaf_size(n) rows, binary-counter
# fold: the old C04 cell's value, computed in parallel, no shift/unshift
# passes). Absent = the incumbent (column_mean_launch, then split-K or
# shift + gemm_tn). The arm replaces the incumbent's routing at every
# width. NOT MEASURED.
# Tried 2026-10-08 (MOJOLEARN_CLASSICAL_PCA_COV=23, the C23 one-pass Chan covariance arm, run ge123e6f9): NV/AMD pca
# istella 2.28x/1.27x SLOWER, taxi 0.90x/0.78x faster (dimension-dependent; combined 1.195x SLOWER) -> deleted
# (c04 stays; C23_MCD is separate). Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
comptime _PCA_COV_RAW = get_defined_int["MOJOLEARN_CLASSICAL_PCA_COV", 0]()
comptime PCA_COV_LEGAL = _PCA_COV_RAW == 0 or _PCA_COV_RAW == 4
comptime PCA_COV_C04 = CLASSICAL_IDN and _PCA_COV_RAW == 4
# TSVD_FUSED_STATS (new, lane classical-decomp): TruncatedSVD's
# explained_variance_ / _ratio_ in one blocked kernel: per leaf the mean and
# centered sum of squares of X's columns and of X V^T's columns (the
# projection formed in the kernel, never stored), merged by Chan's update.
# Replaces gemm_nt + two (mean, shift, square, mean) chains: about 8 passes
# over n x d down to 2 reads of each leaf. NOT MEASURED.
comptime TSVD_FUSED_STATS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_TSVD_FUSED_STATS"]()
comptime C24_PANEL8 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_PANEL8"]()
# C24 ROWS2048: TSQR leaf blocks of 2048 rows instead of 4096
# (x_decomp/tsqr_core.mojo TS_ROWS; twice the leaf blocks in flight per
# panel, a fixed row count, not a data shape). Bits change (the leaf
# boundaries move); the host column reads the same TS_ROWS.
# PROMOTED to the IDENTICAL default (lane/grid-flips-1, 2026-10-08, Andrew
# 13:00Z "flip all of these"). Grid run ge123e6f9 (NVIDIA L40S sm_89 + AMD
# MI325X gfx942, full board data, one scored run per arm), 4096 -> 2048 rows:
#   OLS taxi    NV 34.04 -> 21.52 ms (0.632x)  AMD 14.46 -> 13.78 ms (0.953x)
#   OLS istella NV 816.2 -> 733.8 ms (0.899x)  AMD 468.9 -> 432.2 ms (0.922x)
#   Geometric mean 0.84x. Quality SAME; NV vs AMD output hashes MATCH.
#   randomized-svd (rsvd_tsqr_ortho route) cells were not measured.
#   Measured with LINEAR_GRAM_SOLVE off; with it on, OLS reaches TSQR only
#   when the Gram is not trusted.
# `-D MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF` restores 4096-row leaves (the
# grid's "rows4096" arm). PANEL8 and TREE4 stay independent opt-in arms. The
# old opt-in define is refused in core/six_lane_experiment_guards.mojo.
comptime C24_ROWS2048 = CLASSICAL_IDN and not is_defined["MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF"]()
comptime C24_TREE4 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_TREE4"]()
# TOMBSTONE: MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE (slower) deleted 2026-10-08 by ab4e8e543; code recoverable at ab4e8e543^.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE.patch; record in docs/TOMBSTONES.md.
comptime C26_PRODUCTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C26_PRODUCTS"]()
comptime C26_UPDATE_FUSED = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED"]()
comptime C27_COMPONENTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_COMPONENTS"]()
comptime C28_BUCKET_SOLVES = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES"]()

# Additional independent component/residual sub-arms; the header status applies.
# TOMBSTONE: MOJOLEARN_CLASSICAL_C18_TILE64 (unmeasured) deleted 2026-10-07 by f4de82db4; code recoverable at f4de82db4^.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_CLASSICAL_C18_GRAM_PREFETCH (unmeasured) deleted 2026-10-07 by f4de82db4; code recoverable at f4de82db4^.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch; record in docs/TOMBSTONES.md.
comptime C27_FA_COMPONENTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS"]()
comptime C27_NORM_VECTOR = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_NORM_VECTOR"]()
comptime C13_LOGCV_WEIGHTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS"]()
