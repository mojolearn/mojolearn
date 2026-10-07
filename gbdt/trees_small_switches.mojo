# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""2026-10-07 lane trees-small: IDENTICAL GBDT histogram replication sweeps.

Both default OFF (define absent = the incumbent pinned grid). They change
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
| MOJOLEARN_TREES_HIST_REP_SM   | int sweep | absent (pinned 32), 64, 128, 0 (device count) | `replication_int32_for` SM count |
| MOJOLEARN_TREES_HIST_REP_BPSM | int sweep | absent (2), 4 | `replication_int32_for` blocks per SM |
"""
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime HIST_REP_SM_ON = _IDN and is_defined["MOJOLEARN_TREES_HIST_REP_SM"]()
#: 0 = the device's own SM/CU count (the `sm_count` the launcher was given).
comptime HIST_REP_SM = get_defined_int["MOJOLEARN_TREES_HIST_REP_SM", 32]() if HIST_REP_SM_ON else 32

comptime HIST_REP_BPSM_ON = _IDN and is_defined["MOJOLEARN_TREES_HIST_REP_BPSM"]()
comptime HIST_REP_BPSM = get_defined_int["MOJOLEARN_TREES_HIST_REP_BPSM", 2]() if HIST_REP_BPSM_ON else 2
