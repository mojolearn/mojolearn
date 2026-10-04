# AutoARIMA order-batching default promotion prepared (2026-10-04)

Promotion branch `lane/apple-fast-arima-orders-default` starts from validated
candidate `7ba385b30`, then merges current main `d1871643b`. Changes on main
since the measured candidate affect other algorithms/docs/tooling; no
intervening AutoARIMA kernel change. This is a reviewable promotion, not a
claim that the default has already merged or builds passed.

| M3 tag | A: fused-tail main ms | B: order batching + fused tail ms | Change | Digest (both) | Forecast RMSE (both) |
|---|---:|---:|---:|---|---:|
| gap26-orders-current-synthetic | 13780.316916992888 | 9287.286750040948 | -32.6% | 9bdde6d5440d1cb6 | 2.624119555180116 |
| gap26-orders-current-taxi-hourly | 22706.24804199906 | 13747.215082985349 | -39.5% | 345eb06e29edb8f1 | 74.65912244133813 |

One scored M3 run per arm; no opponent rerun. Raw successful define-arm
records: [arima-orders-evidence/raw-ab.txt](arima-orders-evidence/raw-ab.txt),
from manager-fetched `sync/orders-eigh-results-0925.txt`. The nested
`arm=A` label is the inner harness, not the experiment arm; `def=A/B` is
used above. Measured source is `7ba385b30`.

Full-quality gate precedes both timings:
M3 `~/mq/out/arima-quality-gap26-orders-current-full/FULL_PASS.json`,
with exact source/binary hashes checked by `arima_orders_gated_timing.sh`.
512/2048-observation fixtures cover stationary, near-unit-root, integrated
and mixed ARMA series: selected orders, IC/differencing, fitted parameters,
likelihood, iteration/return codes, fitted predictions and forecasts are
byte-identical. No quality threshold was loosened. The manager-fetched [receipt](arima-orders-evidence/FULL_PASS.json) and
[summary](arima-orders-evidence/quality-summary.txt) confirm 44 arrays exact;
the original 18-array small fixture alone was not used as the promotion gate.

Taxi-hourly's preexisting opponent-quality HOLD is preserved:
RMSE74.6591 versus stored statsforecast68.21. This optimization preserves
main quality but does not resolve that deficit. No board cell, page flag
or headline count is changed by this source promotion branch.

## Gate and source review

The single `ARIMA_ORDER_BATCH` gate now defaults true only when
`KALMAN_FAST_EVAL_WS` and `KALMAN_LL_ONLY` admit FAST + Apple, unless
`MOJOLEARN_ARIMA_ORDER_BATCH_OFF` is defined. Old positive define is harmless;
OFF wins even if both are supplied. No IDENTICAL or non-Apple behavior
changes. Disabling either existing Kalman prerequisite also disables it.

All gate consumers reviewed:

- `fast_order_search.mojo`: grouped likelihood GPU entrypoint.
- `batched_fit.mojo`: resumable single-order GPU optimizer used by final fit;
  trace-enabled calls retain the existing non-grouped path.
- `_mojolearn_arima.mojo`: capability export is false during identity tracing.
- `_x_sequence_autoarima.py`: grouped search only for existing admissible
  nonseasonal order grids, p/q<=3, nobs>2; unsupported grids retain their
  previous GPU route and order-selection tie ordering.

Accepted `ARIMA_FUSED_EVAL_TAIL` is unchanged. Both default and order-batch
OFF use shared prepare/finish with fused tail on; only its separate named
`MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF` rolls that optimization back.

## Manager actions still owed

Compile BOTH promotion arms with the existing semaphore-aware wrapper:

```
bash ~/mojolearn-evidence/apple-fast/sync/compile_arms_m2.sh lane/apple-fast-arima-orders-default arima MOJOLEARN_ARIMA_ORDER_BATCH_OFF
```

Here A/default is the measured candidate behavior; B/OFF restores the
measured baseline behavior (labels invert relative to historical A/B).
Record new hashes/source. Existing quality/timing wrappers for the opt-in
candidate hardcode the old positive define; do not accidentally reuse
those wrappers for this OFF build. Manager import/capability smoke should
observe `arima_order_batch_enabled()` true on default and false on OFF in
FAST Metal without identity tracing. New compile result does not authorize
replaying the already-scored timing tags.

Manager must confirm measurement isolation under
[EXPERIMENT_PROCESS.md](../EXPERIMENT_PROCESS.md), both builds, source review,
then merge under the user's existing authorization (no further permission request). If a scan-window overlap is unresolved, keep speed
verdict HOLD-measurement regardless of passing quality. No main merge,
SSH, builds, queue actions or tests were performed by this agent. Preserve
the raw measurements and taxi opponent hold in subsequent accounting.
