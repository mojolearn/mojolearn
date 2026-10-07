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
comptime C04_LDA = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C04_LDA"]()
comptime C05_PHASE_SCRATCH = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH"]()
comptime C06_ROWS2 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C06_ROWS2"]()
comptime C06_ROWS4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C06_ROWS4"]()
comptime C07_DIGIT4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C07_DIGIT4"]()
comptime C07_DIGIT6 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C07_DIGIT6"]()
comptime C07_KEYS1024 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C07_KEYS1024"]()
comptime C07_KEYS4096 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C07_KEYS4096"]()
comptime C08_DICTIONARY = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C08_DICTIONARY"]()
comptime C09_REG_BUNDLE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C09_REG_BUNDLE"]()
comptime C10_RANK_REUSE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C10_RANK_REUSE"]()
comptime C11_DRAW_GATHER = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C11_DRAW_GATHER"]()
comptime C12_SPARSE_COUNTS = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C12_SPARSE_COUNTS"]()
# T.C55.only, measured source 6fe3cfce38fd (2026-10-07): remain opt-in/off.
# Full GaussianNB Taxi/Istella, LDA Taxi/Istella; candidate/baseline time ratios
# NVIDIA sm90: 3.7583/1.6769/2.3931/1.2764; AMD gfx942:
# 7.2976/3.2273/3.4697/1.6871. GaussianNB Taxi and LDA Istella fail quality
# on both vendors; matching cross-vendor bits do not excuse those failures.
# One excluded warmup + one scored sample per arm. NV/AMD same-arm outputs and
# complete declared model state match; Apple/host/PTX identity remains pending.
# Evidence: experiments/six_lane_integration/measurements/20261006/retained-pairs.json
# and BOARD.md. Other affected estimators/combinations remain unqualified.
comptime C55_CLASS_GROUP = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C55_CLASS_GROUP"]()
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
