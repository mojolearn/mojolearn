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
comptime C13_FOLD_STATS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C13_FOLD_STATS"]()
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
# C19: ONE integer sweep (was ORDERED_128/ORDERED_32, two defines where 32
# silently won when both were on). SGD_PS samples per ordered launch; legal
# set {32, 128, 2048}; absent = 2048 = incumbent. Sample order, time index and
# per-sample updates are unchanged; only the launch/witness granularity moves.
# Legal set enforced in core/six_lane_experiment_guards.mojo.
comptime C19_SGD_CHUNK = get_defined_int["MOJOLEARN_CLASSICAL_C19_SGD_CHUNK", 2048]() if CLASSICAL_IDN else 2048
comptime C20_ROW_CACHE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C20_ROW_CACHE"]()
comptime C20_PAIR_LOAD = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C20_PAIR_LOAD"]()
comptime C21_EXTREMA = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C21_EXTREMA"]()
comptime C22_TRIANGLE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C22_TRIANGLE"]()
comptime C23_CENTERED_PANELS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C23_CENTERED_PANELS"]()
comptime C24_PANEL8 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_PANEL8"]()
comptime C24_ROWS2048 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_ROWS2048"]()
comptime C24_TREE4 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C24_TREE4"]()
comptime C25_PROJECTION_REUSE = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE"]()
comptime C26_PRODUCTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C26_PRODUCTS"]()
comptime C26_UPDATE_FUSED = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED"]()
comptime C27_COMPONENTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_COMPONENTS"]()
comptime C28_BUCKET_SOLVES = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES"]()

# Additional independent component/residual sub-arms; the header status applies.
comptime C18_TILE64 = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C18_TILE64"]()
comptime C18_GRAM_PREFETCH = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C18_GRAM_PREFETCH"]()
comptime C27_FA_COMPONENTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS"]()
comptime C27_NORM_VECTOR = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C27_NORM_VECTOR"]()
comptime C13_LOGCV_WEIGHTS = CLASSICAL_IDN and is_defined["MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS"]()
