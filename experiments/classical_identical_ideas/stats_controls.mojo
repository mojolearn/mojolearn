"""C52–C60: default-OFF classical IDENTICAL experiment switches.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
A enables one named candidate; B omits it, preserving all incumbent switches.
These controls must never select a FAST or neural runtime path.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime CLASSICAL_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# C52: ONE switch with named arms (was PAIR_128/PAIR_512, two defines where 512
# silently won when both were on). -D MOJOLEARN_C52_PAIR_ROWS=128|512 selects
# the KDE pair-combine route with that fixed leaf (a numerical profile shared
# by the device and the host oracle, not a scheduling leaf); absent = incumbent
# route. Legal set enforced in core/six_lane_experiment_guards.mojo.
# TOMBSTONE: MOJOLEARN_C52_PAIR_ROWS (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at dd94b7ad88.
# Restore: git apply experiments/removed/MOJOLEARN_C52_PAIR_ROWS.patch
# (Sweep 2026-11-15: comptime C52_PAIR removed; nothing read it.)
# (Sweep 2026-11-15: comptime C52_ROWS removed; nothing read it.)
# C53: independent GMM/BGMM centered-component staging arms.
# TOMBSTONE: MOJOLEARN_C53_CENTER4 (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at f76fba175b.
# Restore: git apply experiments/removed/MOJOLEARN_C53_CENTER4.patch
# (Sweep 2026-11-15: comptime C53_CENTER4 removed; nothing read it.)
comptime C53_BGMM_STATS = CLASSICAL_IDENTICAL and is_defined["MOJOLEARN_C53_BGMM_STATS"]()
# C54: reuse within kernel construction; fitted factors remain caller-owned.
# TOMBSTONE: MOJOLEARN_C54_PREDICT_TILES (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 843366c4fd.
# Restore: git apply experiments/removed/MOJOLEARN_C54_PREDICT_TILES.patch
# (Sweep 2026-11-15: comptime C54_PREDICT_TILES removed; nothing read it.)
# C56_QDA_PROJECT (lane classical-nbda, 2026-10-07; was MOJOLEARN_C56_QDA_PROJECT4):
# one switch with arms, QDA `qda_dec` only (predict / predict_proba scoring):
# each unit computes W adjacent projection columns per pass over the centred
# row, so the row's centred word is formed once per W columns instead of once
# per column. Register width, a hardware choice, not a data shape.
# -D MOJOLEARN_CLASSICAL_C56_QDA_PROJECT=2|4|8 (absent: off). Each projection
# keeps ascending c and the norm keeps ascending r: same words as off.
# TOMBSTONE: MOJOLEARN_CLASSICAL_C56_QDA_PROJECT (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at fe708dfe4a.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C56_QDA_PROJECT.patch
# (Sweep 2026-11-15: comptime C56_QDA_PROJECT removed; nothing read it.)
# C57: retain each robust candidate's mean for its covariance computation.
# TOMBSTONE: MOJOLEARN_C57_CANDIDATE_STATE (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 5bb9e1659f.
# Restore: git apply experiments/removed/MOJOLEARN_C57_CANDIDATE_STATE.patch
# (Sweep 2026-11-15: comptime C57_CANDIDATE_STATE removed; nothing read it.)
# C58/59: only independent classical series/trials, no neural sequence callers.
# TOMBSTONE: MOJOLEARN_C58_SHARED_PREP (slower) deleted 2026-10-08 by 3978db7ce; code recoverable at 3978db7ce^.
# Restore: git apply experiments/removed/MOJOLEARN_C58_SHARED_PREP.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_C58_SERIES4 (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 236e97faa2.
# Restore: git apply experiments/removed/MOJOLEARN_C58_SERIES4.patch
# (Sweep 2026-11-15: comptime C58_SERIES4 removed; nothing read it.)
# C58_FORECAST4 (four theta/ets/garch series per thread) tried 2026-10-08, run ge123e6f9: NV/AMD 2.93x/2.58x
# synthetic, 2.81x/2.13x taxi-hourly (auto-theta), 2.2-3.1x on every theta/damped-ets cell, quality SAME. Deleted and
# refused (core/six_lane_experiment_guards.mojo); recoverable at main bc10b8b56.
# C58 team state budget in MiB: integer sweep -D MOJOLEARN_C58_TEAM_MIB=64|256
# (was the boolean MOJOLEARN_C58_TEAM64); absent = 256 = incumbent. A memory
# budget only: it sets series per launch slice, never per-series arithmetic.
# TOMBSTONE: MOJOLEARN_C58_TEAM_MIB (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at d6045cbd0a.
# Restore: git apply experiments/removed/MOJOLEARN_C58_TEAM_MIB.patch
# (Sweep 2026-11-15: comptime C58_TEAM_MIB removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C59_TRIAL_STATE (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 7935a1db35.
# Restore: git apply experiments/removed/MOJOLEARN_C59_TRIAL_STATE.patch
# (Sweep 2026-11-15: comptime C59_TRIAL_STATE removed; nothing read it.)
# C60: shared detrended observations across independent lag tasks.
# TOMBSTONE: MOJOLEARN_C60_LAG4 (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at ad78b8054f.
# Restore: git apply experiments/removed/MOJOLEARN_C60_LAG4.patch
# (Sweep 2026-11-15: comptime C60_LAG4 removed; nothing read it.)
# C60_DIFF_REUSE removed 2026-10-07 (lane classical-misc): dead code. It fired only
# for d_ == 2, but select_d loops d_ in range(d_max) with d_max <= 2 - D, so d_ <= 1.
