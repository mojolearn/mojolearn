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
# `-D MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF` restores the per-fold passes (arm B
# of the isolated confirmation A/B owed on nv and amd).
comptime C13_FOLD_STATS = CLASSICAL_IDN and not is_defined["MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF"]()
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
comptime C14_GROUP_RHS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C14_GROUP_RHS"]()
comptime C15_FACTOR_SOLVE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE"]()
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
comptime C19_ORDERED_128 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C19_ORDERED_128"]()
comptime C19_ORDERED_32 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C19_ORDERED_32"]()
comptime C20_ROW_CACHE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C20_ROW_CACHE"]()
comptime C20_PAIR_LOAD = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C20_PAIR_LOAD"]()
comptime C21_EXTREMA = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C21_EXTREMA"]()
comptime C22_TRIANGLE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C22_TRIANGLE"]()
# lane classical-decomp (2026-10-07): the old C23_CENTERED_PANELS define is
# split per algorithm family. Its PCA use is the ONE_PASS arm of
# MOJOLEARN_CLASSICAL_PCA_COV below; its MCD use is C23_MCD. Both run
# row-parallel (core/blocked_moments.mojo); the per-cell serial kernels
# (one GPU thread per covariance cell over every row) are deleted.
# C23_MCD: MinCovDet/EllipticEnvelope `emp_cov_at` as the centered Gram
# around the candidate location, leaves of 256 rows folded by the binary
# counter: the old C23 cell's value (x_decomp/classical_cells.mojo, the host
# column), now computed in parallel. NOT MEASURED.
comptime C23_MCD = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C23_MCD"]()
# PCA covariance, ONE switch with named arms (the old C04-over-C23 silent
# priority is gone): -D MOJOLEARN_CLASSICAL_PCA_COV=4 is the C04 arm
# (two passes: the column mean, then the centered Gram around it read
# straight from X, leaves of contract_leaf_size(n) rows, binary-counter
# fold: the old C04 cell's value, computed in parallel, no shift/unshift
# passes); =23 is the C23 arm (ONE blocked pass: each leaf of
# bm_onepass_leaf_rows rows centers on its own means, leaves merge by
# Chan's update in the binary-counter order; the mean comes out of the same
# pass). Absent = the incumbent (column_mean_launch, then split-K or
# shift + gemm_tn). Either arm replaces the incumbent's routing at every
# width. NOT MEASURED.
comptime _PCA_COV_RAW = get_defined_int["MOJOLEARN_CLASSICAL_PCA_COV", 0]()
comptime PCA_COV_LEGAL = _PCA_COV_RAW == 0 or _PCA_COV_RAW == 4 or _PCA_COV_RAW == 23
comptime PCA_COV_C04 = CLASSICAL_IDN and _PCA_COV_RAW == 4
comptime PCA_COV_C23 = CLASSICAL_IDN and _PCA_COV_RAW == 23
# TSVD_FUSED_STATS (new, lane classical-decomp): TruncatedSVD's
# explained_variance_ / _ratio_ in one blocked kernel: per leaf the mean and
# centered sum of squares of X's columns and of X V^T's columns (the
# projection formed in the kernel, never stored), merged by Chan's update.
# Replaces gemm_nt + two (mean, shift, square, mean) chains: about 8 passes
# over n x d down to 2 reads of each leaf. NOT MEASURED.
comptime TSVD_FUSED_STATS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_TSVD_FUSED_STATS"]()
comptime C24_PANEL8 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_PANEL8"]()
comptime C24_ROWS2048 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_ROWS2048"]()
comptime C24_TREE4 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_TREE4"]()
comptime C25_PROJECTION_REUSE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE"]()
comptime C26_PRODUCTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C26_PRODUCTS"]()
comptime C26_UPDATE_FUSED = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED"]()
comptime C27_COMPONENTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_COMPONENTS"]()
comptime C28_BUCKET_SOLVES = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES"]()

# Additional independent component/residual sub-arms; the header status applies.
comptime C27_FA_COMPONENTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS"]()
comptime C27_NORM_VECTOR = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_NORM_VECTOR"]()
comptime C13_LOGCV_WEIGHTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS"]()
