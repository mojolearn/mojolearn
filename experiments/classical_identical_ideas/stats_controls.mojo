"""C52–C60: default-OFF classical IDENTICAL experiment switches.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
A enables one named candidate; B omits it, preserving all incumbent switches.
These controls must never select a FAST or neural runtime path.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime CLASSICAL_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# C52: ONE switch with named arms (was PAIR_128/PAIR_512, two defines where 512
# silently won when both were on). -D MOJOLEARN_C52_PAIR_ROWS=128|512 selects
# the KDE pair-combine route with that fixed leaf (a numerical profile shared
# by the device and the host oracle, not a scheduling leaf); absent = incumbent
# route. Legal set enforced in core/six_lane_experiment_guards.mojo.
comptime C52_PAIR = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C52_PAIR_ROWS"]()
comptime C52_ROWS = get_defined_int["MOJOLEARN_C52_PAIR_ROWS", 128]()
# C53: independent GMM/BGMM centered-component staging arms.
comptime C53_CENTER4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C53_CENTER4"]()
comptime C53_BGMM_STATS = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C53_BGMM_STATS"]()
# C54: reuse within kernel construction; fitted factors remain caller-owned.
comptime C54_PREDICT_TILES = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C54_PREDICT_TILES"]()
# C56_QDA_PROJECT (lane classical-nbda, 2026-10-07; was MOJOLEARN_C56_QDA_PROJECT4):
# one switch with arms, QDA `qda_dec` only (predict / predict_proba scoring):
# each unit computes W adjacent projection columns per pass over the centred
# row, so the row's centred word is formed once per W columns instead of once
# per column. Register width, a hardware choice, not a data shape.
# -D MOJOLEARN_CLASSICAL_C56_QDA_PROJECT=2|4|8 (absent: off). Each projection
# keeps ascending c and the norm keeps ascending r: same words as off.
comptime C56_QDA_PROJECT = get_defined_int["MOJOLEARN_CLASSICAL_C56_QDA_PROJECT", 0]() if CLASSICAL_IDENTICAL else 0
# C57: retain each robust candidate's mean for its covariance computation.
comptime C57_CANDIDATE_STATE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C57_CANDIDATE_STATE"]()
# C58/59: only independent classical series/trials, no neural sequence callers.
comptime C58_SHARED_PREP = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C58_SHARED_PREP"]()
comptime C58_SERIES4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C58_SERIES4"]()
comptime C58_FORECAST4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C58_FORECAST4"]()
# C58 team state budget in MiB: integer sweep -D MOJOLEARN_C58_TEAM_MIB=64|256
# (was the boolean MOJOLEARN_C58_TEAM64); absent = 256 = incumbent. A memory
# budget only: it sets series per launch slice, never per-series arithmetic.
comptime C58_TEAM_MIB = get_defined_int["MOJOLEARN_C58_TEAM_MIB", 256]() if CLASSICAL_IDENTICAL else 256
comptime C59_TRIAL_STATE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C59_TRIAL_STATE"]()
# C60: shared detrended observations across independent lag tasks.
comptime C60_LAG4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C60_LAG4"]()
# C60_DIFF_REUSE removed 2026-10-07 (lane classical-misc): dead code. It fired only
# for d_ == 2, but select_d loops d_ in range(d_max) with d_max <= 2 - D, so d_ <= 1.
