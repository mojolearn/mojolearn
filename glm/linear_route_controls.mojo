# SPDX-License-Identifier: Apache-2.0
"""The existing linear solver switches, shared by native and address doors."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST as _OLS_FAST
from core.accelerator_target import TARGET_HAS_APPLE_GPU

#: Lane apple-fast-olsne (2026-10-03): LinearRegression.fit takes the
#: equilibrated normal equations (DEVIATION 2620) with X and y uploaded once
#: (`ols_fit_resident`) instead of the blocked TSQR under FAST on Apple.
#: Default since the M3 A/B (lane/apple-fast-olsne 9f5cf2a77, n=1): taxi
#: 286 -> 110 ms, r2 .9088 same; istella 2,921 -> 842 ms, r2 .3325 -> .3319.
#: The `_OFF` rollback was deleted 2026-11-15 (docs/TOMBSTONES.md); the old
#: MOJOLEARN_FAST_OLS_NORMAL_EQ define is now harmless. IDENTICAL
#: and the other vendors keep main's route. Read by Python
#: through `ols_normal_eq_default`.
# TOMBSTONE: MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF (old path, superseded by the measured FAST+Apple default) deleted 2026-11-15 by lane/expdel-03-linear; code recoverable at ed9ef96a34.
# Restore: git apply experiments/removed/MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF.patch; record in docs/TOMBSTONES.md.
comptime OLS_FAST_NORMAL_EQ = GLOBAL_NUMERIC_MODE == _OLS_FAST and TARGET_HAS_APPLE_GPU

#: Lane apple-fast-ridgespeed (2026-10-03): Ridge.fit with X and y uploaded
#: once (`ridge_fit_resident`) instead of the four host trips of the Python
#: centering route. FAST on Apple only; same kernels and words. Read by
#: Python through `ridge_resident_default`. Default since the M3 A/B
#: (lane/apple-fast-ridgespeed e3ef68416, n=1, r2 identical .909 / .3287:
#: taxi 57.1 -> 24.5 ms, istella 1,405 -> 724 ms).
# TOMBSTONE: MOJOLEARN_RIDGE_RESIDENT_OFF (old path; EXP:247 promoted on Apple only; NVIDIA/AMD A/B owed (FAST is Apple only)) deleted 2026-11-15 by lane/expdel2-20-linear; code recoverable at 3fcdcb78f6.
# Restore: git apply experiments/removed/MOJOLEARN_RIDGE_RESIDENT_OFF.patch; record in docs/TOMBSTONES.md.
comptime RIDGE_FAST_RESIDENT = (
    GLOBAL_NUMERIC_MODE == _OLS_FAST
    and TARGET_HAS_APPLE_GPU
)
