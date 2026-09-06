# Metrics

GPU implementations of selected cuML metrics and the RAFT statistics primitives they depend on.

`DERIVATION_MAP.tsv` identifies the upstream source for each implementation.
`NOT_IMPLEMENTED.tsv` is the explicit refusal ledger. Integer reductions can be order independent;
floating-point reductions must use a specified schedule when identity is promised.

## Verify

```bash
pixi run check-metrics-labels
pixi run check-metrics-regression
pixi run check-metrics-silhouette
pixi run check-metrics-trust
```

New metrics need an upstream mapping, an adversarial fixture, and a documented reduction policy.
