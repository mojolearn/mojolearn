# SPDX-License-Identifier: Apache-2.0
"""Classical IDENTICAL shared candidates. Every switch defaults OFF.
Unless scoped evidence below says otherwise: NOT COMPILED — NOT TESTED —
IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
A enables only its named define; B omits it and retains all incumbent controls.
"""
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime CLASSICAL_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime C01_LEAF64 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C01_LEAF64"]()
comptime C01_LEAF128 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C01_LEAF128"]()
comptime C02_STATS_PAIR = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C02_STATS_PAIR"]()
comptime C03_FINITE_EXTREMA = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA"]()
comptime C04_LOAD_CENTER = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C04_LOAD_CENTER"]()
# C04_LDA (lane classical-nbda, 2026-10-07): C04's LDA half, split from
# C04_LOAD_CENTER (whose PCA half lane L2 owns). LDA `transform` (solver
# 'svd') stages `centered_matmul` (x_prep/prims.mojo, one unit per output
# cell, centring in the load) instead of center_rows -> matmul. Same words.
# Reach: LDA transform only; the board's lda-clf times fit + predict/proba,
# not transform. Bit 2 (value 4) of `x_prep_classical_shared`.
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

# C61 (lane classical-nbda, 2026-10-07): single-pass blocked class statistics.
# The incumbent IDENTICAL class statistics read X twice per fit: csb1_part
# (block sums) + csb_fold, then csb1_ss (block squared deviations from the
# folded class mean) + csb_var. C61 reads X ONCE: one unit per (XB-row block,
# column) keeps, for every class, the block's sum (csb1_part's chain, same
# words), count and a Welford mean / M2; one unit per (class, column) folds the
# blocks ascending (sum and count exactly as csb_fold, so means are unchanged)
# and merges M2 with Chan's pairwise rule in a fixed ascending block order,
# the M2 terms summed with a compensated (two-sum) accumulator. No serial
# chain longer than one block. Variances take a new order (bits change on
# every vendor and the host column together).
# Split per estimator family (Python routes by caller):
#   C61_NB_ARM (naive Bayes; GaussianNB is the only NB fit that reads a class
#     variance, Multinomial/Complement/Bernoulli/Categorical read sums or
#     counts that are already one pass, so they are not reached):
#     -D MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS=1  single-pass class stats
#     -D MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS=2  also GaussianNB's epsilon
#        column variance from the merged class M2 (Chan across classes), so
#        the unweighted fit reads X once instead of four times
#     (arms need the value; legal set {1, 2}). Bits 4-5 (arm << 4).
#   C61_DA (discriminants): -D MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS: LDA/QDA
#     class stats with a variance (shrinkage routes) single-pass, and LDA
#     'svd' takes its within-class std from the pooled class M2 instead of
#     materialising X - mean[y] and running two column-stat passes over it.
#     Bit 6 (value 64).
comptime C61_NB_ARM = get_defined_int["MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS", 0]() if CLASSICAL_IDENTICAL else 0
comptime C61_DA = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS"]()
comptime C61_OPS = C61_NB_ARM != 0 or C61_DA
