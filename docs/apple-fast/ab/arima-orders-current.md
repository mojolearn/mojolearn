# AutoARIMA order batching against current main (2026-10-04)

Branch `lane/apple-fast-arima-orders-current` merges current main `d4bb2b795`
into prior order-batching source/checker `eae73e1f0` (kernel `457160e23`).
Follow [EXPERIMENT_PROCESS.md](../EXPERIMENT_PROCESS.md). Candidate remains
opt-in `MOJOLEARN_ARIMA_ORDER_BATCH`, guarded by the existing Apple FAST
Kalman and likelihood-only gates. No acceptance or default change.

Hypothesis: grouped GPU likelihood evaluations amortize independent order
launches and synchronization, retaining each order's LBFGS/Jones state and
selection tie order. Old-source small quality reportedly passed 18 arrays;
it does not establish full quality or quality after this integration.

## Source integration and rebuild requirement

Main's accepted fused evaluation tail (`311d5233e`) is retained in BOTH
arms. The conflict in `fast_eval_ws.mojo` was resolved by moving main's
fused `ew_finish_kernel` dispatch into shared `FastEvalWS.finish`, used by
single-order and grouped-order fits. `prepare` skips bad-flag memset only
when the fused tail writes those flags, matching current main. The
`MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF` rollback still takes the original
separate launches. Do not set it in this comparison.

This changes runtime `.mojo` source relative to the previously compiled
order-batching binaries. Rebuild both A and B; do not use the old
`457160e23`/`eae73e1f0` prebuild or compare B against pre-fused-tail A. Arm A
has no experiment define (current-main behavior through the shared
refactor); B adds ONLY `-D MOJOLEARN_ARIMA_ORDER_BATCH`. Binding `arima` is
changed; the existing current-main `tsa` binding is also needed by quality.
The Python order-search dispatch and arima binding changes inherited from
the candidate are preserved. No host model-fit fallback was introduced.

## Manager build and queue plan

No commands below were run by this agent. Manager owns boxes and queue.
Compile both arms on M2 using the existing semaphore-aware wrapper:

```
bash ~/mojolearn-evidence/apple-fast/sync/compile_arms_m2.sh lane/apple-fast-arima-orders-current arima MOJOLEARN_ARIMA_ORDER_BATCH
```

Resolve the final full branch SHA as `SOURCE`, synchronize the M3 named
branch/ref, transfer its verified manifest and hashes, and ensure matching
FAST runtime dependencies (including `tsa`) without replacing staged arima.
Use new tags (examples below), never the old valid small-quality tag.

M3 quality CMD, in the exact SOURCE checkout:

```
~/board-0834/cache/venv/bin/python ~/mq/verified_arms.py SOURCE arima MOJOLEARN_ARIMA_ORDER_BATCH gap26-orders-current-full --stage-only
bash tools/arima_orders_full_quality.sh gap26-orders-current-full
```

Full quality uses both seeded 512/2048-observation fixtures, each containing
stationary, near-unit-root, integrated and mixed ARMA series. It captures
selected orders, IC, differencing, parameters, likelihoods, iterations,
return codes, fitted predictions and 24-step forecasts. Byte equality is
the unchanged conservative gate; RMSE/loglike summaries remain in logs.
There is no timing, opponent invocation or binding build in this helper.
It fixes the Bash 3.2 empty-array issue using positional arguments, forces
Apple FAST, and restores the original binding even on failure. A receipt
`~/mq/out/arima-quality-<tag>/FULL_PASS.json` is created only after the full
paired comparison, required fixture check and captured binding-hash check.

Only after FULL_PASS, queue the following two serialized M3 CMD jobs:

```
bash tools/arima_orders_gated_timing.sh SOURCE gap26-orders-current-full gap26-orders-current-synthetic synthetic
bash tools/arima_orders_gated_timing.sh SOURCE gap26-orders-current-full gap26-orders-current-taxi-hourly taxi-hourly
```

The timing helper verifies receipt source, exact current checkout, full
fixture names and both binary hashes against `verified-arms/SOURCE/arima`.
It then delegates to manager `verified_arms.py` and `afc_ab_def.sh`: one
scored run/arm, no opponent rerun, refusal if the target race already has
scored content. Missing, failed or stale quality exits before timing.
Read only `ARIMA_*QUALITY`, `AFC-AB`, `AFC-DEF-SUMMARY`, or bounded errors.

Quality failure stays HOLD with original threshold; build or harness
failure is not a measurement. Preserve artifacts and retry only unscored
work under explicit fresh tags. Speed/quality are both owed. Taxi-hourly's
preexisting opponent RMSE deficit remains held independently of any speed
or main-relative quality result.
