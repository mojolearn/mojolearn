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
