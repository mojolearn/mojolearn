# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the time-series bindings (tsa, ARIMA, forecast, resample).

Each binding of this family imports SERIES_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    # Deleted 2026-10-07 (lane serial-cleanup): forbidden serial shape; incumbent route is parallel. Recoverable at origin/integration/switches-20261007 608a7cf4a.
    comptime assert not is_defined["MOJOLEARN_C60_DIFF_REUSE"](), "removed: C60_DIFF_REUSE was dead code (select_d never reaches d == 2)"
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    # C25_PROJECTION_REUSE: deleted 2026-10-08 above, restored 2026-10-10 (lane c25-restore-promote) from the tombstone patch after grid pass 2.
    comptime assert not is_defined["MOJOLEARN_C58_SHARED_PREP"](), "removed: MOJOLEARN_C58_SHARED_PREP retired 2026-10-08: noise, NV 0.996x / AMD 0.980x on ets (grid ge123e6f9); see EXPERIMENTS.md"
    # Lane owed-deletions-D1 (2026-10-09): DROPPED rows in docs/apple-fast/EXPERIMENTS.md whose code was still on main (docs/TOMBSTONES.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_CSS_SEARCH"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA CSS-scored order search was DROPPED-quality: rab26 synthetic -45.2% / taxi-hourly -30.5% ms but forecast_rmse 2.465 -> 3.485 / 75.71 -> 76.06; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_STEPWISE"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA Hyndman-Khandakar stepwise device search was DROPPED-slower: rab26 synthetic +51.6% / taxi-hourly +102.5% ms, taxi forecast_rmse 75.71 -> 90.54; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA Kalman groups in one concurrent round loop was DROPPED-slower: autoarima +14-16% (M3 verdicts batch 6, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_D_CONCURRENT"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA d groups in one native search call was DROPPED-slower: autoarima +5-7% (M3 verdicts batch 6, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_RANK_SORT"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple bootstrap distribution sorted by one rank launch was DROPPED-slower: bootstrap taxi 14.38 -> 13.64 ms, istella 11.59 -> 14.46 ms (M3 afc_ab_def ab1 d51f4b4bf, mixed: slower on istella); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_ONE_FOLD"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple bootstrap mean / diff_means replicates folded once per block was DROPPED-slower: bootstrap taxi 16.31 -> 13.89 ms, istella 12.84 -> 14.23 ms (M3 afc_ab_def ab1 d51f4b4bf, mixed: slower on istella); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_TSA2_KPSS"](), "removed 2026-10-03 by 7ff2caf99 (refusal added 2026-10-09, lane/owed-deletions-D1): FAST Apple one-launch kpss_fused was DROPPED-noise: kpss taxi-hourly -73%, synthetic -62%, the same gain as the kept TSA_FAST_KPSS_PACK; code at 7ff2caf99^; see docs/TOMBSTONES.md"
    # wave-2 w2-decide-neural (2026-10-11): A/B on main c8ef6c66d, one run per arm, NV L40S nv2 / AMD MI325X, digests NV==AMD
    # on every cell (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    # Promoted: the old positive define is refused; -D <NAME>_OFF restores the old path.
    comptime assert not is_defined["MOJOLEARN_IDN_ETS_TEAM"](), "promoted 2026-10-11 (lane/w2-decide-neural): the ETS(A, A|Ad, N) team is the IDENTICAL default (damped-ets 0.014-0.017x, bits change NV==AMD, forecast_rmse equal, nv2 v0099 / amd a1681); drop -D MOJOLEARN_IDN_ETS_TEAM, -D MOJOLEARN_IDN_ETS_TEAM_OFF restores op_ets"
    # Lane k1-delete-unverdicted-fast (2026-11-09): Apple FAST experiments with no verdict, never to be measured (Apple FAST
    # investment stopped); deleted (docs/TOMBSTONES.md "Unverdicted FAST experiments deleted (lane K1)").
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_K3_DF_PROBE.patch (and _PRODUCT_DF_TAIL.patch)
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_K3_DF_PROBE"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_ARIMA_FAST_K3_DF_PROBE, the private Apple FAST double-float K3 likelihood probe (`_arima_k3_df_probe`, arima/impl/fast_scalar_df.mojo), was never measured on a fit (private-kernel quality only) and never will be; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL (F16), the Apple FAST compensated Kalman likelihood and finite-difference tail (arima/impl/fast_eval_df.mojo), had no verdict row (F16 M3 one-sample B/A 1.13 / 1.04 / 1.01, left off) and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"

    return True


comptime SERIES_GUARDS_OK = _check_configuration()
