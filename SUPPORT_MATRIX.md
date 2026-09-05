# Support and certification

This page is the current support snapshot for the mojolearn 0.5.0 release candidate. It deliberately
separates API availability, packaging, and numerical certification. A feature
being importable does not mean that every hardware column or numeric mode has
been certified.

For the enforced identity surface, see the
[identity-path ledger](IDENTITY_PATHS.md). Fresh certification artifacts name
their commit, device, mode, and limitations; stale narrative reports are not a
support source.

## Numeric modes

| Mode | Contract |
|---|---|
| `fast` | Performance-oriented. No repeatability or cross-vendor bitwise promise. |
| `deterministic` | Repeated execution on the same device and build is intended to return the same bits. It makes no cross-vendor promise. |
| `identical` | For a specifically certified fixture and configuration, raw result bits match across the recorded Metal, CUDA, and HIP legs. |

Select a process default with `mojolearn.set_numeric_mode(...)` or
`MOJOLEARN_NUMERIC_MODE`. Estimators that accept `numeric_mode=` can override
the process default. All three modes are runtime-selectable binary sets, not
separate packages.

`identical` is not a blanket claim for arbitrary shapes, parameters, hardware,
drivers, or future builds. The claim is limited to completed evidence cards.
An unrun column is **pending**, never inferred from source inspection or another
device.

## Installation support

| Platform | Status | Qualification |
|---|---|---|
| macOS arm64 / Apple silicon | Primary packaged platform | Built at the Apple M1 ISA floor. Current packaging inventory contains 15 extensions in each of three modes. A clean release-wheel run is required before claiming that every newly added extension ships correctly. |
| Linux x86-64 / NVIDIA CUDA | Supported source-build and wheel path | Device code is architecture-specific. Certification on one NVIDIA architecture does not certify another. Release 0.3.0 had an AVX-512 host-code defect; it is historical and must not be used as current evidence. |
| Linux x86-64 / AMD HIP | Supported source-build and wheel path | Measured chiefly on `gfx942`. That is not evidence for every AMD architecture. |
| CPU-only and other accelerators | Unsupported | There is no CPU implementation. |

Before publishing a release, validate the installed wheel rather than only the
source tree: import it in a clean environment, load all 15 extensions in each
included mode, and run the public smoke suite. See [release instructions](docs/PYPI_RELEASE.md).

## Capability snapshot

The table groups public surfaces by the strongest evidence currently retained.
“Three-vendor card” means at least one named fixture has matching IDENTICAL
cards on Apple, NVIDIA, and AMD; it does not extend beyond that fixture.

| Surface | Public availability | Strongest retained identity evidence | Important open work |
|---|---|---|---|
| Gradient boosting | Beta | Three-vendor cards for recorded configurations | Complete ordered boosting and categorical/CTR coverage; improve FAST performance without weakening checks. |
| Random Forest / Extra Trees | Beta | Three-vendor cards for recorded configurations | Keep sklearn `max_leaf_nodes` semantics distinct from cuML-style level-order `max_leaves`; extend NVIDIA performance coverage. |
| k-means, k-NN, DBSCAN | Beta | Three-vendor cards for recorded fixtures | Validate newer metrics and DBSCAN paths; optimize only with sabotage-sensitive identity gates. |
| PCA, truncated SVD, OLS, Ridge, logistic regression | Beta | Three-vendor matrix for recorded fixtures | Broaden shapes and public-surface wheel smoke. |
| FP32 matrix multiplication | Beta | Three-vendor frozen-profile sweep | Shapes and plans outside the recorded profile remain uncertified. |
| Isolation Forest | Beta | Three-vendor card for the recorded fixture | Re-run current bindings on NVIDIA architectures affected by earlier context-lifetime failures. |
| ARIMA filtering | Beta | Three-vendor card for the recorded filter fixture | The newer fitted-estimator surface still needs NVIDIA and AMD qualification. |
| Holt-Winters and spectral/time-series helpers | Beta | Apple–AMD cards for recorded fixtures | NVIDIA column remains pending. |
| Gaussian process | Experimental | Apple–AMD IDENTICAL card for recorded fixture | Complete current-wheel and NVIDIA qualification; no performance claim. |
| Mamba 1 | Experimental | Three-vendor operator card for recorded fixture | Forward/backward public surfaces need independent oracle and wheel validation. |
| Mamba 2 | Experimental | Pairwise Apple–NVIDIA and Apple–AMD cards, not all from one source state | Produce a same-commit three-vendor forward/backward certificate. |
| Mamba 3 | Experimental | Apple–AMD card for recorded fixture | NVIDIA is pending; forward and backward need a dedicated qualification lane. |
| Transformer block | Experimental | Three-vendor operator card for an earlier fixture | Current Python binding and backward path need NVIDIA/AMD and corpus-oracle qualification. |
| Training primitives and checkpointing | Experimental | Local correctness gates | Cross-vendor public-surface qualification remains open. |

## What counts as certification

A cross-vendor result is accepted only when all of the following are recorded:

1. The same source state, fixture bytes, numeric profile, and card schema were
   used on each claimed vendor.
2. Each hardware leg actually ran and records device and toolchain provenance.
3. Stage tags, dtypes, element counts, and raw-bit hashes agree.
4. A sabotage or alternate-spelling arm demonstrates that the check detects the
   numerical mechanism it is intended to protect.
5. Refused configurations are reported as refusals, not passes.

`python -m mojolearn verify` is a useful local check against a shipped
reference card, but one local run cannot independently establish a
cross-vendor claim. See [verification](docs/VERIFY.md) and
[conformance bundles](docs/CONFORMANCE.md).

## Current priorities

- Make FAST materially faster than IDENTICAL on representative NVIDIA work;
  benchmark FAST, IDENTICAL, and one external comparator per method.
- Treat Mamba 1/2/3 forward and backward as a dedicated lane with independent
  calculus/oracle checks and same-commit vendor cards.
- Finish clean installed-wheel validation for every extension and mode.
- Add build/source provenance to native artifacts so stale binaries cannot be
  mistaken for current results.
- Close pending NVIDIA and AMD legs for already-public APIs before adding more
  breadth.

The project-level sequencing lives in [ROADMAP.md](ROADMAP.md). Update this
page only from recorded evidence; do not turn planned or in-progress runs into
support claims.

## UMAP and Mamba closure (2026-09-05)

`UMAP.fit` and `fit_transform` expose the supported dense Euclidean 2D/3D
slice in the 0.5.0 candidate. Installed-wheel qualification is required before
publishing; no Linux wheel is inferred from the Apple build.

At `718495cd`, Apple M4, NVIDIA RTX 4090, and AMD MI300X matched all 54
native Mamba backward gradient tensors across five cases and all 186 cells
in the named UMAP fixture. See the [three-vendor record](bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json).
These source certificates do not resolve the separately recorded AMD Mamba
Python binding fault or certify other shapes and installed artifacts.
