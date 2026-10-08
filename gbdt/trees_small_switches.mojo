# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""2026-10-07 lane trees-small: IDENTICAL GBDT histogram replication sweep.

One define; since lane/grid-flips-1 (2026-10-08) its IDENTICAL default is the
device arm (0, device SMs x 2), and `-D MOJOLEARN_TREES_HIST_REP_SM_OFF` is the
old pinned grid (`replication_for`, 32 SMs x 2). The arms change
how many blocks the Int32 histogram families launch, never which integer
lands in a cell: those families quantize PER ROW (`hist2_quantize` keyed by
the storage position) and add in Int32 with relaxed atomics, and the level
accumulator is zero between levels, so any partition of rows into blocks
gives the same bits (`replication_for`'s note). Read ONLY by
`greedy_search_helper.replication_int32_for`, whose callers are the Int32
families (fused two-stat 8-bit, the shared-Int32 one-byte ladder, the wide
8-bit arm, and `launch_one_byte` / `launch_hist2_one_byte` only when their
smem mode is HIST_SMEM_SHARED2_I32). The float-accumulating binary and
half-byte families, the level quantize launch, the multi-GPU shards and
every other reader of `partition_chunks_sm_for` keep the pinned count,
because there the SM count decides which rows form a rounded partial.

Cost reasoning: under IDENTICAL the replication target is
2 blocks/SM x 32 pinned SMs whatever the device, so a part with more than
32 SMs/CUs runs the root and first levels (few live leaves) at a fraction of
its block slots. The sweep scales the target with the hardware, never with
rows, features or a dataset.

NOT COMPILED -- NOT TESTED -- NOT MEASURED. Host column unchanged.

| define | kind | legal values | gates |
|---|---|---|---|
| MOJOLEARN_TREES_HIST_REP_SM | arms | absent = 0 (device SMs x 2, the default), 32 / 64 / 128 (pinned SMs x 2), 1 (device SMs x 4) | `replication_int32_for` block target |
| MOJOLEARN_TREES_HIST_REP_SM_OFF | switch | defined = the pre-flip pinned `replication_for` (32 SMs x 2) | turns the sweep off |

Lane grid-prune (2026-10-07) merged MOJOLEARN_TREES_HIST_REP_BPSM into this
define: the target is blocks-per-SM x SMs, so BPSM=4 alone equalled SM=64 and
SM=64 + BPSM=4 equalled SM=128. The one distinct BPSM point, 4 blocks per
device SM, is arm 1 (HIST_REP_DEVICE_X4; 1 is never a real SM pin). The old
define is refused in core/six_lane_experiment_guards.mojo.
"""
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

# PROMOTED: the device arm (0) is the IDENTICAL default (lane/grid-flips-1,
# 2026-10-08, Andrew 13:00Z "flip all of these"). Grid run ge123e6f9 (NVIDIA
# L40S sm_89 + AMD MI325X gfx942, full board data, one scored run per arm),
# pinned 32 -> device SMs x 2, ms NV / AMD:
#   symmetric taxi       NV 3915 -> 3717 (0.950x)   AMD 4554 -> 3964 (0.870x)
#   symmetric istella    NV 6940 -> 6763 (0.975x)   AMD 7101 -> 5966 (0.840x)
#   symmetric-1000 taxi  NV 7534 -> 7031 (0.933x)   AMD 9024 -> 7800 (0.864x)
#   symmetric-1000 ist.  NV 11737 -> 11422 (0.973x) AMD 13581 -> 11228 (0.827x)
#   depthwise taxi       NV 4181 -> 4083 (0.977x)   AMD 3928 -> 3622 (0.922x)
#   depthwise istella    NV 8998 -> 8984 (0.999x)   AMD 7527 -> 6254 (0.831x)
#   lossguide taxi       NV 4674 -> 4611 (0.987x)   AMD 4241 -> 3990 (0.941x)
#   lossguide istella    NV 10082 -> 9693 (0.961x)  AMD 8324 -> 6966 (0.837x)
#   multiclass taximc    NV 7251 -> 7028 (0.969x)   AMD 6754 -> 6043 (0.895x)
#   multiclass istellamc NV 14689 -> 14710 (1.001x) AMD 10373 -> 9969 (0.961x)
#   rank-pairlogit       NV 2985 -> 2966 (0.994x)   AMD 1656 -> 1472 (0.889x)
#   rank-yetirank        NV 3108 -> 3074 (0.989x)   AMD 1802 -> 1621 (0.899x)
#   categorical taxicat  AMD 10988 -> 10818 (0.984x); NV not measured.
#   10 faster / 0 slower cells, geometric mean 0.927x. Quality SAME where
#   scored (rank quality pending); no bits change (Int32 sums).
# Arms kept for the grid: `-D MOJOLEARN_TREES_HIST_REP_SM_OFF` (pinned32, the
# old default), `=64`, `=1` (device x4); `=32` is the pinned target through
# this route.
comptime HIST_REP_SM_ON = _IDN and not is_defined["MOJOLEARN_TREES_HIST_REP_SM_OFF"]()
#: Arm values: the device's own SM/CU count (the `sm_count` the launcher was
#: given) at 2 or at 4 blocks per SM.
comptime HIST_REP_DEVICE = 0
comptime HIST_REP_DEVICE_X4 = 1
comptime HIST_REP_SM = get_defined_int["MOJOLEARN_TREES_HIST_REP_SM", HIST_REP_DEVICE]() if HIST_REP_SM_ON else 32
comptime HIST_REP_BPSM = 4 if HIST_REP_SM == HIST_REP_DEVICE_X4 else 2
