# SPDX-License-Identifier: Apache-2.0
"""One-switch-per-idea neural GEMM controls (lane neural-gemm-attn-dedupe, 2026-10-07).

Imports only std and checks.numerics, so the low-level IDENTICAL GEMM
(gemm/checks/gemm_identical.mojo) can read the same switch without importing
the neural experiment modules.

`-D MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=<arm>`: the mutually exclusive neural
GEMM schedules, formerly seven defines (NN01/02/08/09/10/11/15) plus NI02. An
absent define is the incumbent IDENTICAL GEMM. Arms:
   1  geometry       NN01: an existing gemm_identical step geometry, chosen
                      by -D MOJOLEARN_IDN_NEURAL_GEMM_ARM (needs
                      MOJOLEARN_GEMM_ARM_TRIAL)
   2  stream         NN02: bounded partial-plane groups, neural callers only
   3  stream_all     NI02: bounded partial planes in gemm_identical, EVERY
                      GEMM caller (gemm lane, classical, neural)
   4  stream_exact   NN02 + NN11: stream with exact fold-state slots
   8  async          NN08: NVIDIA async operand pipeline (sync elsewhere)
   9  pages          NN09: staging depth/pad/swizzle (STAGE_* parameters)
  10  cost           NN10: cost-based plan (needs IDN_NEURAL_FILL_BLOCKS)
  11  fold_exact     NN11: standalone exact-capacity fold storage
  15  threadmap      NN15: staged output-thread mapping
  24  pages_threadmap NN09 + NN15 together

`-D MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES=<mask>` (default 7 = every
role): which neural caller classes take the selected schedule; the others
keep the incumbent route in the same build. Bits: 1 projection (forward and
dInput of every layer that is not the LM head), 2 head (LM head forward and
dInput), 4 weight-grad (every dWeight/dBias product). The role is a
call-site tag (`identical_gemm_into[ROLE=...]`), never a matrix size, so one
mask covers every model width. Arm 3 is a global GEMM-contract arm and takes
only the default mask.

The legal-arm asserts live in core/six_lane_experiment_guards.mojo, which
every binding evaluates through GLOBAL_NUMERIC_MODE.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _ALLOWED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

comptime SCHED_INCUMBENT = 0
comptime SCHED_GEOMETRY = 1
comptime SCHED_STREAM = 2
comptime SCHED_STREAM_ALL = 3
comptime SCHED_STREAM_EXACT = 4
comptime SCHED_ASYNC = 8
comptime SCHED_PAGES = 9
comptime SCHED_COST = 10
comptime SCHED_FOLD_EXACT = 11
comptime SCHED_THREADMAP = 15
comptime SCHED_PAGES_THREADMAP = 24

comptime NEURAL_GEMM_SCHEDULE_RAW = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE", 0]()
comptime NEURAL_GEMM_SCHEDULE = NEURAL_GEMM_SCHEDULE_RAW if _ALLOWED else SCHED_INCUMBENT

comptime ROLE_PROJECTION = 1
comptime ROLE_HEAD = 2
comptime ROLE_WGRAD = 4
comptime ROLE_ALL = 7
comptime NEURAL_GEMM_ROLES = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES", 7]()

