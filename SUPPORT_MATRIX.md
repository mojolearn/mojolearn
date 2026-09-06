# Support and certification

This page covers the published mojolearn 0.5.0 artifact and the 0.6.0 source
release candidate. The [older macOS 0.6.0 candidate](bench/results/wheels/2026-09-05-umap-060-install-recovery/README.md) passed its installed qualification;
it predates OrderedRMSE and the latest sequence/serialization fixes. Publication
and final combined Linux qualification remain pending. It separates API availability, packaging, and numerical certification. A feature
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
| macOS arm64 / Apple silicon | Primary packaged platform | Built at the Apple M1 ISA floor. Current packaging inventory contains 15 extensions in each of three modes. The 0.5.0 wheel passed every Python 3.10–3.14/mode combination on Apple M4 and fresh PyPI-install API checks. The older 0.6.0 macOS candidate passed its installed checks; current OrderedRMSE/sequence/serialization bytes need a fresh build and qualification. |
| Linux x86-64 / NVIDIA CUDA | Frozen CUDA candidate built; full qualification failed | At `eb835021`, all 45 `sm_89` extensions build and 19/24 installed jobs pass; Mamba accuracy and Transformer stalls fail the full candidate. Source overlays fix Transformer, with one Mamba3 non-IDENTICAL tolerance gap remaining. Device code is architecture-specific. Certification on one NVIDIA architecture does not certify another. Release 0.3.0 had an AVX-512 host-code defect; it is historical and must not be used as current evidence. |
| Linux x86-64 / AMD HIP | Frozen HIP candidate qualified; final release pending | At `eb835021`, all 45 `gfx942` extensions and all 24 installed jobs pass. Newer sequence/serialization fixes still need fresh AMD qualification. Measured chiefly on `gfx942`. That is not evidence for every AMD architecture. |
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
| Gradient boosting | Beta | Three-vendor cards for recorded configurations | Numeric single-permutation ordered RMSE passes AMD/NVIDIA at `6dd44ac5`, with 130 bitwise-matching records. Broader categorical/CTR coverage and external parity remain. |
| Random Forest / Extra Trees | Beta | Three-vendor cards for recorded configurations | Keep sklearn `max_leaf_nodes` semantics distinct from cuML-style level-order `max_leaves`; extend NVIDIA performance coverage. |
| k-means, DBSCAN | Beta | Three-vendor cards for recorded fixtures | Broaden shapes and public-surface wheel smoke. |
| k-NN | Beta | Three-vendor cards; the completed AMD four-arm layout outputs also match retained NVIDIA outputs and Apple hashes | Selector and transposed-distance flags remain opt-in. Broaden distributions and installed/external comparisons before changing normal dispatch. |
| PCA, truncated SVD, OLS, Ridge, logistic regression | Beta | Three-vendor matrix for recorded fixtures | Broaden shapes and public-surface wheel smoke. |
| FP32 matrix multiplication | Beta | Three-vendor frozen-profile sweep | Shapes and plans outside the recorded profile remain uncertified. |
| Isolation Forest | Beta | Three-vendor card for the recorded fixture | Re-run current bindings on NVIDIA architectures affected by earlier context-lifetime failures. |
| ARIMA filtering | Beta | Three-vendor card for the recorded filter fixture | The fitted estimator passes the frozen AMD/NVIDIA installed candidates in all three modes; final release qualification remains. |
| Holt-Winters and spectral/time-series helpers | Beta | Apple–AMD cards for recorded fixtures | NVIDIA column remains pending. |
| Gaussian process | Experimental | Apple–AMD IDENTICAL card for recorded fixture | Complete current-wheel and NVIDIA qualification; no performance claim. |
| Mamba 1 | Experimental | Three-vendor operator/backward evidence at `718495cd`; baseline and L64 pass and match AMD/NVIDIA at `395d9421` | Frozen AMD installed checks pass all modes; NVIDIA IDENTICAL passes, with non-IDENTICAL sequence accuracy work remaining. Python backward remains absent. |
| Mamba 2 | Experimental | Three-vendor backward certificate at `718495cd`; baseline, L257 and incoming state pass and match AMD/NVIDIA at `395d9421` | Newer NVIDIA/AMD source forward/state checks passed, including the AMD binding fix. Frozen AMD installed checks pass all modes and NVIDIA IDENTICAL passes. Broader fixtures and Python backward remain. |
| Mamba 3 | Experimental | Baseline and L65 pass AMD/NVIDIA at `395d9421`; public gradients and complete retained diagnostics match by bits | L65 uses the explicit [compositional arithmetic contract](mamba/BACKWARD_CERTIFICATION.md), with independent whole-forward public gradients. Frozen AMD installed checks pass all modes and NVIDIA IDENTICAL passes. Corrected Apple, a remaining NVIDIA non-IDENTICAL k_last tolerance miss, final wheel checks and Python backward remain. |
| Transformer block | Experimental | Three-vendor operator card for an earlier fixture | Frozen AMD installed API checks pass. NVIDIA lifetime fix passes IDENTICAL; the subsequent full-FP32 patch passes FAST/DETERMINISTIC. A final-source full matrix, rebuilt wheel and independent corpus API oracle remain pending. |
| UMAP | Published 0.5.0: dense `fit`/`fit_transform`. Source 0.6.0 candidate: `transform` and CSR graph storage for public fitting | Original three-vendor fixtures; expanded six-case IDENTICAL inputs/embeddings and 876 native stage cells match AMD/NVIDIA at `d88c7883` | All six expanded quality cases pass in all modes on AMD/NVIDIA after the self-neighbor fix. The frozen Linux candidates pass all installed UMAP modes and match six IDENTICAL input/embedding fixtures. Publication, final combined-wheel qualification and larger-scale coverage remain pending. |
| Training primitives and checkpointing | Experimental | Local correctness gates | Cross-vendor public-surface qualification remains open. |

The [AMD k-NN layout record](bench/results/e1/2026-09-05_215006-mojolearn-e2-amd/README.md)
retains all four arms, correctness bytes, rotating timing rounds and the
comparison with earlier NVIDIA and Apple evidence. Its speedups are scoped
to those fixtures and devices, not a general dispatch or scalability claim.

The [weighted CTR slice](bench/results/e1/2026-09-05_235251-amd-catboost-fixed-partition/README.md)
passes on AMD, including the production leaf estimator on fixed occupied
zero-mass partitions at L2=0 and L2=3. This is focused native correctness;
the numeric single-permutation [ordered RMSE path](gbdt/ORDERED_RMSE.md) now
passes both GPUs at `6dd44ac5`, including fresh weighted-CTR checks. The
[comparison](bench/results/resume/2026-09-06-ordered-mamba-knn/cross-device-continued.json)
also matches all 5,440 new adversarial kNN records in all four flag arms.

The [installed comparison](bench/results/resume/2026-09-06-installed-gap-closure/installed-lane-comparison.json)
adds six UMAP fixtures (24 arrays) and the full OrderedRMSE model plus 72
prediction cells matching across AMD/NVIDIA at `eb835021`. Both complete
candidate dispositions remain visible; lane agreement does not approve the
failed NVIDIA candidate for release. GBDT/OrderedRMSE saves now persist the
effective numeric mode at `29a8c848`, tested with changed process defaults in
lightweight checks and a separate NVIDIA native wrapper overlay. That newer
wrapper is not present in the frozen candidate wheels or current PyPI releases.

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
- Build and qualify the 0.6.0 artifact with UMAP transform and CSR graph storage;
  complete publication and fresh Linux installed-wheel checks.
- Add build/source provenance to native artifacts so stale binaries cannot be
  mistaken for current results.
- Close pending NVIDIA and AMD legs for already-public APIs before adding more
  breadth.

The project-level sequencing lives in [ROADMAP.md](ROADMAP.md). Update this
page only from recorded evidence; do not turn planned or in-progress runs into
support claims.

## UMAP and Mamba closure (2026-09-05)

`UMAP.fit` and `fit_transform` expose the supported dense Euclidean 2D/3D
slice in the published macOS 0.5.0 wheel. Its installed API passed in all
three modes on Python 3.10–3.14. The downloaded PyPI artifact matched the
publication digest and passed additional API checks; see the
[post-publication record](bench/results/wheels/2026-09-05-umap-api/postpublish/results.json).
A refreshed Linux wheel and NVIDIA/AMD installed-artifact qualification remain pending.

At `718495cd`, Apple M4, NVIDIA RTX 4090, and AMD MI300X matched all 54
native Mamba backward gradient tensors across five cases and all 186 cells
in the named UMAP fixture. See the [three-vendor record](bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json).
These source certificates do not certify other shapes or installed artifacts.
The Mamba3 correctness claim from that historical certificate is superseded:
its staged reference shared a missing scale/gamma chain-rule contribution.
The [corrected AMD/NVIDIA record](bench/results/resume/2026-09-05-next-certification/corrected-backward-cross-device.json)
passes all five baseline cases and matches 54 gradient tensors, while requiring
independent whole-forward gradients for every Mamba3 public leaf. Its long
profile was RED on intermediate checks despite matching public tensors. The
[September 6 certificate](bench/results/resume/2026-09-06-ordered-mamba-knn/README.md)
now passes both profiles on AMD/NVIDIA under the documented compositional
arithmetic policy, retaining all intermediate outputs and direct differences.
The later AMD allocation fix has its own successful source Python API rerun;
see the [AMD source qualification record](bench/results/e1/2026-09-05_134041-mojolearn-e2-amd/README.md).
That later binding evidence is separate from the original native certificate.

## UMAP 0.6.0 release candidate

The source candidate implements `fit`, `fit_transform` and `transform` with
dense Euclidean input and two- or three-dimensional output. Public fitting
now stores the fuzzy graph in CSR form using O(n_samples × n_neighbors)
space. This does not add sparse input support or approximate neighbors:
exact neighbor search still performs quadratic pair comparisons.

`transform` embeds new samples against the frozen fitted embedding. Private
training and embedding copies add O(n_samples × (n_features + n_components))
storage. Changes to parameters or numeric mode require refitting; changing
query batching can change results. Supervised targets and alternate metrics
or initialization remain unsupported.

| Evidence layer | Completed evidence | Still pending |
|---|---|---|
| Published 0.5.0 macOS artifact | Dense fit/fit_transform installed-wheel checks | Transform and CSR are not in this published artifact |
| Source transform and sparse native paths before public CSR integration | Named held-out transform embeddings match Apple, NVIDIA and AMD; separate sparse native gates passed | This earlier source does not certify the later public integration |
| Integrated public CSR fit/transform source | All three numeric modes passed API and held-out quality checks on Apple, NVIDIA and AMD; named IDENTICAL held-out layouts match | Broader shapes and metrics remain outside this evidence |
| 0.6.0 release candidate artifact | macOS candidate passed Python 3.10–3.14/mode smoke and clean Python 3.12 UMAP fit/transform/quality checks in all modes | Publication and Linux installed-artifact checks |

Evidence: [Apple public integration](bench/results/umap/2026-09-05-public-sparse/README.md),
[AMD integrated API results](bench/results/e1/2026-09-05_142553-mojolearn-e2-amd/diag/followup/results.tsv),
and [earlier NVIDIA transform and sparse qualification](bench/results/e1g/2026-09-05_095508-nvidia-mamba/README.md).
The latter records frozen source `72da212b`, whose public fit remained dense;
public CSR integration landed separately at `274ba7c0`. The later AMD
integration ran source `e2cfe1ac`. Keep these revisions and source hashes with
their results rather than relabeling them as one release certificate.

The original 186-cell UMAP certificate and 54-tensor Mamba backward
certificate above retain their original counts and scope. New transform
quality/identity checks are separate evidence, not additions to those totals.
The k-NN optimization remains experimental and is not enabled in normal
wheel builds.

Latest evidence: [integrated NVIDIA qualification and native public k-NN pricing](bench/results/e1g/2026-09-05_103918-nvidia-mamba/README.md),
and [exact macOS 0.6.0 candidate qualification](bench/results/wheels/2026-09-05-umap-060/README.md).
The flag-gated k-NN selector improved paired median native request times by
2.99x/5.99x/18.86x on the named NVIDIA 32/128/1000-query fixtures; these are
not cuML comparisons or a claim that every algorithm improved.
