# SPDX-License-Identifier: Apache-2.0
"""The existing linear solver switches, shared by native and address doors."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST as _OLS_FAST
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined

#: Lane apple-fast-olsne (2026-10-03): LinearRegression.fit takes the
#: equilibrated normal equations (DEVIATION 2620) with X and y uploaded once
#: (`ols_fit_resident`) instead of the blocked TSQR under FAST on Apple.
#: Default since the M3 A/B (lane/apple-fast-olsne 9f5cf2a77, n=1): taxi
#: 286 -> 110 ms, r2 .9088 same; istella 2,921 -> 842 ms, r2 .3325 -> .3319.
#: MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF turns it off; the old
#: MOJOLEARN_FAST_OLS_NORMAL_EQ define is now harmless. IDENTICAL
#: and the other vendors keep main's route. Read by Python
#: through `ols_normal_eq_default`.
comptime OLS_FAST_NORMAL_EQ = (
    GLOBAL_NUMERIC_MODE == _OLS_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF"]()
)

#: Lane apple-fast-ridgespeed (2026-10-03): Ridge.fit with X and y uploaded
#: once (`ridge_fit_resident`) instead of the four host trips of the Python
#: centering route. FAST on Apple only; same kernels and words. Read by
#: Python through `ridge_resident_default`. Default since the M3 A/B
#: (lane/apple-fast-ridgespeed e3ef68416, n=1, r2 identical .909 / .3287:
#: taxi 57.1 -> 24.5 ms, istella 1,405 -> 724 ms). Off with
#: -D MOJOLEARN_RIDGE_RESIDENT_OFF (which also turns RIDGE_NO_U off); the
#: old -D MOJOLEARN_RIDGE_RESIDENT define is now harmless.
comptime RIDGE_FAST_RESIDENT = (
    GLOBAL_NUMERIC_MODE == _OLS_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_RIDGE_RESIDENT_OFF"]()
)
