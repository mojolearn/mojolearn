# Gaussian mixture

Cross-vendor full-covariance Gaussian-mixture estimation.

The lane implements the numerically sensitive EM path with explicit covariance regularization,
component ordering, and convergence behavior. `DERIVATION_MAP.tsv` records provenance and
`NOT_IMPLEMENTED.tsv` records refusals.

## Verify

```bash
pixi run check-mixture
pixi run mixture-main
```

Future work should prioritize adversarial coverage for collapsed components, tied likelihoods,
and ill-conditioned covariance matrices before adding more covariance models.
