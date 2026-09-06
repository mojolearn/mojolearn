# Spectral

Spectral embedding and clustering derived from cuVS and cuML.

`IDENTICAL_SPECTRAL_CONTRACT.md` defines the reproducible path, including graph construction,
eigensolver conventions, sign handling, and downstream clustering. `DERIVATION_MAP.tsv` and
`NOT_IMPLEMENTED.tsv` define provenance and scope.

## Verify

```bash
pixi run check-spectral
pixi run spectral-card
```

Eigenvectors are not uniquely represented without conventions. Any change to ordering, sign
canonicalization, degeneracy handling, or tie breaking is therefore a contract change.
