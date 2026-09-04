# Mamba

This directory implements reference-pinned Mamba 1, Mamba 2, and Mamba 3
blocks. They are low-level forward blocks with caller-owned continuation
state, not trainable estimator or complete language-model APIs.

## Current surface

| family | forward | decode/continuation | Python binding | backward |
|---|---|---|---|---|
| Mamba 1 | implemented | implemented | implemented | kernels and host oracle exist; no whole-pass composer or public API |
| Mamba 2 | implemented | implemented | implemented | routing/workspace scaffold only |
| Mamba 3 | implemented | implemented | implemented | routing/workspace scaffold only |

All forward families expose `fast`, `deterministic`, and `identical` builds.
FAST is allowed to differ from the IDENTICAL oracle; recording that difference
is not a FAST failure. Cross-vendor claims require matching cards from a named
commit and configuration.

The Python binding keeps each `DeviceContext` alive through the final buffer
download. This is required on every backend and is especially important on
AMD, where premature context destruction previously presented as a GPU memory
access fault.

## Contracts and evidence

- [Mamba 1 contract](IDENTICAL_MAMBA_CONTRACT.md)
- [Mamba 2 contract](IDENTICAL_MAMBA2_CONTRACT.md)
- [Mamba 3 contract](IDENTICAL_MAMBA3_CONTRACT.md)
- [Corpus format and cross-check](corpus/README.md)
- [Historical parity ledger](../archive/evidence/mamba/FEATURE_PARITY.md)
- [Historical backward plans](../archive/plans/mamba/)

The contracts define the profile. Historical plans and ledgers explain how it
was reached but are not the current status source.

## Backward implementation order

1. Compose the existing Mamba 1 backward kernels into one end-to-end pass.
2. Add an independent whole-pass oracle, non-finite refusal, and the MB1–MB10
   separating gates.
3. Add a native binding and Python API only after those gates close.
4. Implement Mamba 2 backward from an independent oracle.
5. Implement Mamba 3 backward last.

Do not infer Mamba 2/3 backward capability from their compile probes. Those
probes currently validate routing and workspace topology, not backward
arithmetic.
