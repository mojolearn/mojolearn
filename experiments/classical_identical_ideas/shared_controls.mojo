# SPDX-License-Identifier: Apache-2.0
"""Classical IDENTICAL shared candidates. Every switch defaults OFF.
Unless scoped evidence below says otherwise: NOT COMPILED — NOT TESTED —
IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
A enables only its named define; B omits it and retains all incumbent controls.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime CLASSICAL_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# lane classical-decomp (2026-10-07): C01 is split in two.
# C01_LEAF: the regression-metrics PairSum leaf length (x_metrics/common.mojo
# LEAF), an integer sweep: -D MOJOLEARN_CLASSICAL_C01_LEAF=32|64|128, absent =
# 32. Replaces the C01_LEAF64/C01_LEAF128 pair, where 128 silently won.
comptime _C01_LEAF_RAW = get_defined_int["MOJOLEARN_CLASSICAL_C01_LEAF", 32]()
comptime C01_LEAF_LEGAL = _C01_LEAF_RAW == 32 or _C01_LEAF_RAW == 64 or _C01_LEAF_RAW == 128
comptime C01_LEAF = _C01_LEAF_RAW if CLASSICAL_IDENTICAL else 32
# C01_MEAN: core/xtdz_coalesced.mojo column_mean_launch (PCA covariance mean,
# PCA full-SVD mean, TSVD column variances) as leaf column sums (one block per
# leaf of contract_leaf_size(n) rows, sub-chains added in order) and the
# binary-counter fold over leaves (core/blocked_moments.mojo). Replaces the
# one-thread-per-column serial mean. NOT MEASURED.
comptime C01_MEAN = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C01_MEAN"]()
comptime C02_STATS_PAIR = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C02_STATS_PAIR"]()
comptime C03_FINITE_EXTREMA = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA"]()
# C04 is split (lane classical-decomp, 2026-10-07): its PCA use is the =4 arm of
# MOJOLEARN_CLASSICAL_PCA_COV (linear_controls.mojo); C04_LDA is the x_prep
# `centered_matmul` op (LDA transform, solver != eigen), behavior unchanged.
# C04_LDA (lane classical-nbda, 2026-10-07): C04's LDA half, split from
# C04_LOAD_CENTER (whose PCA half lane L2 owns). LDA `transform` (solver
# 'svd') stages `centered_matmul` (x_prep/prims.mojo, one unit per output
# cell, centring in the load) instead of center_rows -> matmul. Same words.
# Reach: LDA transform only; the board's lda-clf times fit + predict/proba,
# not transform. Bit 2 (value 4) of `x_prep_classical_shared`.
comptime C04_LDA = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C04_LDA"]()
comptime C05_PHASE_SCRATCH = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH"]()
# C06 (lane classical-kmeans, 2026-10-07): ONE control with arms, replacing
# ROWS2/ROWS4 (two defines where ROWS4 silently won). Rows per norm block,
# legal set 2|4; absent = the incumbent one-row block.
comptime C06_NORM_ROWS = get_defined_int["MOJOLEARN_CLASSICAL_C06_NORM_ROWS", 1]() if CLASSICAL_IDENTICAL else 1
comptime C06_ROWS_ON = C06_NORM_ROWS > 1
# C06 small-d arm: one thread per row while d <= C06_THREAD_MAX_D (a cost
# rule: a NORM_TPB-lane block per row leaves more than 75% of its lanes idle
# there). The thread replays the incumbent halving tree in registers, so the
# bits are the incumbent's. Independent of C06_NORM_ROWS (which keeps d above
# the bound).
comptime C06_SMALL_D_THREAD = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD"]()
# C07 (lane classical-encoders, 2026-10-07): two integer sweeps of the device
# radix sort (x_prep/dradix.mojo), replacing DIGIT4/DIGIT6/KEYS1024/KEYS4096,
# which silently overrode each other. The sort has one answer, so no arm
# moves a bit. Read only under IDENTICAL; FAST keeps 8 bits / 2048 rows.
#   MOJOLEARN_CLASSICAL_C07_RADIX_BITS = 4 | 6 | 8 (default 8): digit bits a
#     pass, so 8 | 6 | 4 passes (the pass count must be even).
#   MOJOLEARN_CLASSICAL_C07_RADIX_ROWS = 1024 | 2048 | 4096 (default 2048):
#     consecutive keys one (column, chunk) task counts and scatters.
comptime C07_RADIX_BITS = get_defined_int["MOJOLEARN_CLASSICAL_C07_RADIX_BITS", 8]() if CLASSICAL_IDENTICAL else 8
comptime C07_RADIX_ROWS = get_defined_int["MOJOLEARN_CLASSICAL_C07_RADIX_ROWS", 2048]() if CLASSICAL_IDENTICAL else 2048
# C08 (lane classical-encoders, 2026-10-07), one define per route that builds
# the fit dictionary and the fit codes in ONE program (x_prep `unique_inverse`,
# whose device form is the parallel index radix sort + run scan of
# x_prep/ddict.mojo). The old MOJOLEARN_CLASSICAL_C08_DICTIONARY spanned all
# three routes and is removed. Same categories and codes as the incumbent.
#   C08_TARGET_CODES: TargetEncoder fit / fit_transform (board: target-encoder).
#   C08_ONEHOT_FT:    OneHotEncoder.fit_transform only (the board times fit, transform).
#   C08_ORDINAL_FT:   OrdinalEncoder.fit_transform only (the board times fit, transform).
comptime C08_TARGET_CODES = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C08_TARGET_CODES"]()
comptime C08_ONEHOT_FT = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C08_ONEHOT_FT"]()
comptime C08_ORDINAL_FT = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C08_ORDINAL_FT"]()
#: derived, not a define: some route compiles the `unique_inverse` op
comptime C08_DICTIONARY = C08_TARGET_CODES or C08_ONEHOT_FT or C08_ORDINAL_FT
# C08_UNIQUE_SCAN: the device `unique_cols` op (every IDENTICAL `_fit_categories`:
# OneHot/Ordinal/TargetEncoder fit, LabelEncoder-style callers) as a parallel
# chunked run scan over all columns at once (x_prep/ddict.mojo) instead of one
# thread per column walking n sorted rows. Same words, same counts.
# Promoted 2026-10-08 (lane grid-act-3, IDENTICAL grid ge123e6f9, one run per arm,
# incumbent -> scan ms, istella / taxi): onehot NV 53.9 -> 46.7 / 41.7 -> 32.1,
# AMD 75.2 -> 21.2 / 72.1 -> 18.2; ordinal NV 65.0 -> 46.2 / 49.0 -> 30.2, AMD
# 75.2 -> 21.2 / 71.5 -> 17.5; target-encoder NV 188.3 -> 175.4 / 148.2 -> 133.0,
# AMD 142.3 -> 88.4 / 134.3 -> 78.8 (0.525x combined); output digests equal to the
# incumbent's, so no bit moves. The scan's work is a fixed row chunk per task
# across every column, so it wins wherever n rows per column outgrow one
# thread's serial walk. Default on in IDENTICAL; -D MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF
# restores the one-thread-per-column walk.
comptime C08_UNIQUE_SCAN = CLASSICAL_IDENTICAL and not is_defined["MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF"]()
comptime C09_REG_BUNDLE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C09_REG_BUNDLE"]()
comptime C10_RANK_REUSE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C10_RANK_REUSE"]()
comptime C11_DRAW_GATHER = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C11_DRAW_GATHER"]()
comptime C12_SPARSE_COUNTS = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C12_SPARSE_COUNTS"]()
# C08 independent grouped one-hot emission; immutable dictionary is unchanged.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime C08_GROUPED_OUTPUT = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C08_GROUPED_OUTPUT"]()

# Independent C02 linear sums and C56 LDA input reuse, default OFF.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime C02_LINEAR_PAIR = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR"]()
comptime C56_LDA_INPUT = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C56_LDA_INPUT"]()

# C05 independent invocation-owned OLS covariance/inverse phase storage.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime C05_OLS_PHASE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C05_OLS_PHASE"]()

# C61 (lane classical-nbda, 2026-10-07): single-pass blocked class statistics
# (MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS=1|2 for GaussianNB,
# MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS for LDA/QDA; one walk of X with
# per-block Welford M2 and a Chan merge) tried 2026-10-08, run ge123e6f9:
# gaussian-nb NV/AMD 1.58x/1.05x istella, 1.42x/1.08x taxi (arm 1),
# 1.57x/1.02x, 1.37x/1.03x (arm 2); lda-clf 1.38x/1.04x, 1.54x/1.11x SLOWER,
# quality SAME. Deleted, both defines refused; recoverable at main bc10b8b56.
