# SPDX-License-Identifier: Apache-2.0
"""One-switch-per-idea neural GEMM controls (lane neural-gemm-attn-dedupe, 2026-10-07).

Imports only std, so the low-level IDENTICAL GEMM
(gemm/checks/gemm_identical.mojo) can read the same switch without importing
the neural experiment modules.

The neural GEMM schedule is the incumbent IDENTICAL GEMM. The arm names
below stay as constants for the dispatcher (gemm/neural_dispatch.mojo).

`-D MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES=<mask>` (default 7 = every
role): which neural caller classes take the selected schedule; the others
keep the incumbent route in the same build. Bits: 1 projection (forward and
dInput of every layer that is not the LM head), 2 head (LM head forward and
dInput), 4 weight-grad (every dWeight/dBias product). The role is a
call-site tag (`identical_gemm_into[ROLE=...]`), never a matrix size, so one
mask covers every model width.

The legal-arm asserts live in core/six_lane_experiment_guards.mojo, which
every binding evaluates through GLOBAL_NUMERIC_MODE.
"""
from std.sys.compile import get_defined_int

# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (its terms here: every IDENTICAL switch off at once; EXP:2438 old path) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at 5cde955daf.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-25-neural.patch.

comptime SCHED_INCUMBENT = 0
comptime SCHED_GEOMETRY = 1
comptime SCHED_STREAM = 2
comptime SCHED_STREAM_EXACT = 4
comptime SCHED_PAGES = 9
comptime SCHED_COST = 10
comptime SCHED_FOLD_EXACT = 11
comptime SCHED_THREADMAP = 15
comptime SCHED_PAGES_THREADMAP = 24

# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE (arms 1 2 4 9 10 11 15 24: no A/B won; EXP:1746 DROP) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at 9813509a01.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE.patch; record in docs/TOMBSTONES.md.
comptime NEURAL_GEMM_SCHEDULE = SCHED_INCUMBENT

comptime ROLE_PROJECTION = 1
comptime ROLE_HEAD = 2
comptime ROLE_WGRAD = 4
comptime ROLE_ALL = 7
comptime NEURAL_GEMM_ROLES = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES", 7]()

