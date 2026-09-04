# ARIMA

GPU batched ARIMA fitting and forecasting, derived from cuML's Kalman-filter implementation.

The supported surface and deliberate refusals are recorded in `DERIVATION_MAP.tsv` and
`NOT_IMPLEMENTED.tsv`. `SEAMS.tsv` identifies boundaries where upstream behavior or numerical
ordering can diverge. Those machine-readable files and the checks are authoritative; historical
investigation notes live in Git history.

## Verify

```bash
pixi run check-arima
pixi run check-arima-surface
```

`arima_main.mojo` emits the lane identity card. The Python wrapper is checked separately because
API compatibility and kernel identity are different promises.

## Current focus

- keep rejection behavior explicit for unsupported orders and options;
- extend vendor evidence without changing the arithmetic schedule;
- optimize only behind the FAST mode boundary.
