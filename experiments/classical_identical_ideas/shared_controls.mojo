# SPDX-License-Identifier: Apache-2.0
"""Classical IDENTICAL shared candidates. Every switch defaults OFF.
Unless scoped evidence below says otherwise: NOT COMPILED — NOT TESTED —
IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
A enables only its named define; B omits it and retains all incumbent controls.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime CLASSICAL_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime C01_LEAF64 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C01_LEAF64"]()
comptime C01_LEAF128 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C01_LEAF128"]()
comptime C02_STATS_PAIR = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C02_STATS_PAIR"]()
comptime C03_FINITE_EXTREMA = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA"]()
comptime C04_LOAD_CENTER = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C04_LOAD_CENTER"]()
comptime C05_PHASE_SCRATCH = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH"]()
comptime C06_ROWS2 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C06_ROWS2"]()
comptime C06_ROWS4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C06_ROWS4"]()
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
comptime C08_UNIQUE_SCAN = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN"]()
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
