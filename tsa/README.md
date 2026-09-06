# Time-series analysis

Shared GPU time-series primitives and estimators. `DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv`
are the concise source of truth for supported behavior.

```bash
mojo run -I . tsa/tsa_main.mojo
```

Window boundaries, missing values, initialization, and accumulation order should be explicit per estimator.
