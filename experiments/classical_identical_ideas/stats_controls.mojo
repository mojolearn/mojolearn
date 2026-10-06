"""C52–C60: default-OFF classical IDENTICAL experiment switches.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
A enables one named candidate; B omits it, preserving all incumbent switches.
These controls must never select a FAST or neural runtime path.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime CLASSICAL_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# C52: independent numerical profiles, not scheduling-dependent leaves.
comptime C52_PAIR_128 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C52_PAIR_128"]()
comptime C52_PAIR_512 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C52_PAIR_512"]()
comptime C52_PAIR = C52_PAIR_128 or C52_PAIR_512
comptime C52_ROWS = 512 if C52_PAIR_512 else 128
# C53: independent GMM/BGMM centered-component staging arms.
comptime C53_CENTER4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C53_CENTER4"]()
comptime C53_BGMM_STATS = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C53_BGMM_STATS"]()
# C54: reuse within kernel construction; fitted factors remain caller-owned.
comptime C54_PREDICT_TILES = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C54_PREDICT_TILES"]()
# C56: adjacent independent projection outputs share each centered load.
comptime C56_QDA_PROJECT4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C56_QDA_PROJECT4"]()
# C57: retain each robust candidate's mean for its covariance computation.
comptime C57_CANDIDATE_STATE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C57_CANDIDATE_STATE"]()
# C58/59: only independent classical series/trials, no neural sequence callers.
comptime C58_SHARED_PREP = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C58_SHARED_PREP"]()
comptime C58_SERIES4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C58_SERIES4"]()
comptime C58_FORECAST4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C58_FORECAST4"]()
comptime C58_TEAM64 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C58_TEAM64"]()
comptime C59_TRIAL_STATE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C59_TRIAL_STATE"]()
# C60: shared detrended observations across independent lag tasks.
comptime C60_LAG4 = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C60_LAG4"]()
comptime C60_DIFF_REUSE = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C60_DIFF_REUSE"]()
