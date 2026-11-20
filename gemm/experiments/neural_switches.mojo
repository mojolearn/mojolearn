# SPDX-License-Identifier: Apache-2.0
"""One-switch-per-idea neural GEMM controls (lane neural-gemm-attn-dedupe, 2026-10-07).

Imports nothing, so the low-level IDENTICAL GEMM
(gemm/checks/gemm_identical.mojo) can read the same switch without importing
the neural experiment modules.

The neural GEMM schedule is the incumbent IDENTICAL GEMM.

Neural caller classes (`identical_gemm_into[ROLE=...]`): 1 projection,
2 head, 4 weight-grad; every role takes the one schedule.

The legal-arm asserts live in core/six_lane_experiment_guards.mojo, which
every binding evaluates through GLOBAL_NUMERIC_MODE.
"""
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (its terms here: every IDENTICAL switch off at once; EXP:2438 old path) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at 5cde955daf.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-25-neural.patch.

# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE (arms 1 2 4 9 10 11 15 24: no A/B won; EXP:1746 DROP) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at 9813509a01.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: NEURAL_GEMM_SCHEDULE, the SCHED_* arm names, ROLE_ALL and NEURAL_GEMM_ROLES removed with the
# schedule modules gemm/neural_dispatch.mojo read them for.)

comptime ROLE_PROJECTION = 1
comptime ROLE_HEAD = 2
comptime ROLE_WGRAD = 4
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES (role mask knob, no A/B; 7 kept) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at 7737754a09.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES.patch; record in docs/TOMBSTONES.md.

