# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Independent Apple FAST classical scheduling experiments, 2026-10-06.

NEVER RUN — PENDING MEASUREMENT. Source written only: uncompiled, unverified,
unmeasured; no speed or quality outcome is claimed. Every switch is default OFF.
Define presence enables an arm; omit the define for A (do not pass '=0').
No switch changes bins, sampled features, row membership or solver settings.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime AFCL_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)

# AFCL-T01: halve rows per quantized-histogram replica to shorten each
# block's serial row walk; the existing occupancy floor/cap still applies.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T01 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T01"]()

# AFCL-T02: eight features use 16 KiB of histogram shared memory instead
# of Apple's 32 KiB group, trading extra workgroups for resident capacity.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T02 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T02"]()

# AFCL-T03: 512 threads retain room for all 256 borders while reducing
# the per-feature binarizer's register footprint per threadgroup.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T03 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T03"]()

# AFCL-T04: four Apple SIMD groups per partition-metadata block instead
# of eight; all kernels retain their grid stride and visit every row.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T04 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T04"]()

# AFCL-T05: four adjacent score records per lane before advancing a
# cooperative block; comparator and final reduction order stay unchanged.
# Both arms require MOJOLEARN_SYM_RESOLVE_BLOCK and the fused search path.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T05 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T05"]()
comptime AFCL_RESOLVE_RECORDS = 4 if AFCL_T05 else 1

# AFCL-T06: halve hardware-derived leaf-statistic partial counts to reduce
# scratch/fold traffic; accumulation order may move bits, not solver effort.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T06 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T06"]()

# AFCL-T07: four SIMD groups per resident prediction block rather than
# eight; rows remain independent and each row visits trees in order.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T07 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T07"]()
comptime AFCL_PREDICT_BLOCK = 128 if AFCL_T07 else 256

# AFCL-T12: four SIMD groups per CTR elementwise block rather than eight;
# same sorted categories, prefixes, priors and output destinations.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T12 = AFCL_APPLE_FAST and is_defined["MOJOLEARN_AFCL_T12"]()
