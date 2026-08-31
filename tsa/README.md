# tsa: the KPSS stationarity test and the differencing-order choice, from cuML

Part of the ARIMA/TSA lane (DEVIATIONS 670-679). **This README covers
the `tsa/` half; `arima/README.md` covers `arima/` and carries the lane's
shared material (DEVIATION 670, the hand-off list, the row text for both
directories).** COPY, DO NOT IMPROVE.

## Status: CONSTRUCTION plus one Apple device's gates; no second vendor has run this

Rung 1 of the lane. What is here:

    cuml/cpp/src/tsa/stationarity.cu                  -> tsa/impl/tsa/stationarity.mojo
    cuml/cpp/src_prims/timeSeries/stationarity.cuh    -> tsa/impl/timeSeries/stationarity.mojo
    cuml/cpp/src_prims/timeSeries/arima_helpers.cuh   -> tsa/impl/timeSeries/arima_helpers.mojo  (prepare_data only)
    cuml/cpp/src_prims/linalg/batched/matrix.cuh      -> tsa/impl/linalg/batched/matrix.mojo     (the two diff kernels only)
    cuml/python/cuml/cuml/tsa/auto_arima.pyx (the d block) -> tsa/impl/tsa/auto_arima.mojo::select_d

`tsa/DERIVATION_MAP.tsv` pins the commit (cuML 265b9da6, v26.08.00) and says
per file what is transliterated and what is partial; `tsa/NOT_IMPLEMENTED.tsv`
lists what was not ported and why, parameter by parameter. Float32 only:
cuML offers `float` and `double`, Metal has no Float64 (DEVIATION 670,
`arima/README.md`).

## Commands

    tools/with_build_lock.sh     pixi run mojo run -I . tsa/checks/stationarity_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . tsa/checks/stationarity_check.mojo
    tools/with_build_lock.sh     pixi run mojo run -I . tsa/tsa_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/tsa.card tools/with_identical_mode.sh pixi run mojo run -I . tsa/tsa_main.mojo
    python3 tools/identity_trace_diff.py /tmp/tsa.mac.card /tmp/tsa.other.card

No pixi task is registered (pixi.toml is not this lane's); the suggested
task names are in `arima/README.md` under HAND-OFF.

## The algorithm, and the two deviations

`_kpss_test` (`stationarity.cuh:191-281`) is: mean per series; center;
`s2A = sum e^2`; `lags = ceil(12 (n/100)^0.25)`; the Bartlett-weighted
autocovariance accumulator and its sum `s2B`; the cumulative sum and
`eta = sum S_t^2`; `stat = (eta/n^2) / (s2A/n + s2B)`; the p-value from
table 1 of Kwiatkowski 1992 by linear interpolation; `stationary = pvalue >
threshold`. `kpss_test` differences first (`prepare_data`, `d + D <= 2`).

**DEVIATION 671** (`tsa/impl/timeSeries/stationarity.mojo`): every
per-series sum of theirs is RAFT's `coalescedReduction` -- a per-thread
Kahan-Babuska-Neumaier chain, a logical-warp shuffle fold at a width chosen
from `n_obs`, a dispatch that reads the SM COUNT -- and the scan is Thrust's.
Ours: one block of `STATS_TPB` per series, strided partials serial
ascending (`x*x + acc` one `identical_mul_add`), `pinned_block_sum`;
the scan serial per series. Under IDENTICAL every sum is a function of the
series bits and `STATS_TPB`; under FAST the library fold decides.

**DEVIATION 672**: a constant series makes `stat = 0/0`; theirs lets the
NaN fall through every comparison to `pvalue = 0.10`, stationary. Ours
defines `stat = 0.0` at a zero denominator, which interpolates to the same
`0.10`: the same decision, no computed NaN in a recorded stage (ADDENDUM
11). Non-finite INPUTS are refused by name before any stage.

## What the gates found (2026-08-23, M4, both modes)

    check_kpss_device_equals_oracle     5 orders x 10 stages, 0 cells differ [IDENTICAL]
                                        (FAST: 2-6 of 8 per-series cells and
                                        ~10% of the per-cell stages differ in
                                        the last bits: RECORDED, the library fold)
    check_kpss_matches_float64          rel <= 1.5e-6 on every series that is
                                        not built to flush; the 2^-66 series is
                                        REPORTED (1.9e1 at d=0 -- its s2B is
                                        mostly flushed by construction)
    check_kpss_constant_series_is_defined  stat 0x00000000, stationary, d=0 and d=1
    check_kpss_refuses_by_name          n_obs <= d+sD, D with s<2, d+D>2, NaN, inf
    check_kpss_launch_invariant         elem block 256/64/128, 0/37 floats of
                                        padding poisoned 12345.678 / -0.0, a
                                        batch of 8 vs a batch of 3 in another
                                        order, run twice: 0 cells differ, BOTH modes
    check_kpss_fold_order_is_visible    a descending fold moves 7 of 8 statistics
    check_kpss_ftz_seam_is_reached      IDENTICAL: device s2A 0x05275bd4 == oracle
                                        with ftz; oracle WITHOUT ftz 0x052ec0b0
    check_select_d                      AR(1) -> 0, random walk -> 1, constant ->
                                        0, linear trend -> 1; equal to the host
                                        replay of auto_arima's loop

**A finding about the SHARED fold, handed off.** The 2^-66 series' s2B
partials cancel inside `pinned_block_sum`'s tree to a SUBNORMAL partial,
which the M4 flushes and a gradual-underflow host keeps; the unflushed
oracle disagreed with the device at `s2B` cell 3 (0x82b2ab5d vs
0x82b0b3d3). `core/pinned_reduce.mojo`'s device tree has no `ftz` between
steps, so on CUDA/HIP the partial would be KEPT and the Apple and NVIDIA
bits of `pinned_block_sum` would differ whenever a subnormal partial arises
inside the tree. The oracle now models the FTZ column (`pinned_fold_host`
flushes each step); the one-line fix to the shared tree is in
`arima/README.md` under HAND-OFF TO THE IDENTITY LANE.

## Sabotage table

| # | what was broken | result under IDENTICAL |
|---|---|---|
| a | `series_sum_kernel` stride start ROTATED by block id | NO bit moves: a halving tree is rotation-invariant (pairs `{j, j+step}` map to pairs); recorded as a property, not evidence |
| a' | `series_sum_kernel` per-thread stride folded DESCENDING | FAIL `y_means: 8 cells, 3 differ (first cell 0: 0x3c28d21d vs 0x3c28d220)` |
| b | oracle's tree swapped for a serial ascending sum of the partials | FAIL `y_means: 8 cells, 6 differ (first cell 0: 0x3d447329 vs 0x3d447325)` |
| c | `s2B_accumulation_kernel` lag loop reversed | FAIL `s2B: 8 cells, 3 differ (first cell 0: 0x3ed8ff89 vs 0x3ed8ff8a)` |
| d | `ftz` dropped from the s2B product | no bit moves on Apple (FTZ backend; the pin is inert here); `check_kpss_ftz_seam_is_reached` is the host-side evidence the seam is on the path |

All reverted; the tree is clean (`diff` against the backup printed nothing).

## ROW TEXT FOR THE IDENTITY LANE

**`tsa/` HAS NO ROW IN `IDENTITY_PATHS.md`.** This table used to head its row
`49`; row 49 there is the mamba lane's `portable_divf` division row
(DEVIATION 740, `IDENTITY_PATHS.md`), and the ledger runs 1 to 59 with no tsa
entry anywhere in it. The number is deleted rather than corrected, because
there is no number to correct it to. As in `holtwinters/README.md`, I do not
own `IDENTITY_PATHS.md` and the row number is the orchestrator's to assign;
the next free index is 60. The row text below is the hand-off and is written
to be pasted unchanged.

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| NN | tsa: KPSS stationarity test (`stationarity.cuh`) and auto_arima's `d` loop | per-series sums are `raft::linalg::coalescedReduction`: a per-thread Kahan chain whose thread->element map follows a policy table keyed on `n_obs`, a logical-warp shuffle fold, a CUB block fold past 512 rows, and a dispatch that reads `getMultiProcessorCount()`; the scan is Thrust's decoupled look-back; `0/0` yields a NaN whose payload is the vendor's | DEVIATION 671: one pinned shape per series (STATS_TPB strided partials, `pinned_block_sum`, serial scan); DEVIATION 672: `0/0 -> 0.0`, same decision; non-finite inputs refused by name; FOUND: `pinned_block_sum`'s tree keeps or flushes a subnormal partial per vendor (hand-off) | device == oracle bitwise under IDENTICAL on the M4 for 5 orders x 10 stages; launch/batch invariant both modes; no second vendor |

## HAND-OFF

See `arima/README.md` (one list for both directories).
