# Exposing the whole library on PyPI: what is there, what is missing, what it costs

Written 2026-08-24, answering: *"expose everything in mojolearn to pypi ...
that may be clean separation?"*

**Short answer to the separation question: yes, and it is already the shape
this project has.** The paper repository (`~/CascadeProjects/mlsys`) holds a
manuscript that quotes result artifacts; this repository holds the system;
PyPI holds the thing a reader can install. The paper does not ship code and
the library does not carry claims. What is NOT yet true is the "everything"
half: the released surface is a fraction of what is built and certified.

**One constraint the separation runs into.** MLSys is double-blind and
anonymization failures are desk rejected, so the main-track submission may not
name the package, link it, or cite it. The artifact paragraph in the paper is
written for the camera-ready. **The industrial track's rules permit product
names and URLs under an anonymized byline**, and that is the one decision
worth making deliberately rather than defaulting past: on the industrial track
`pip install mojolearn` can appear in the submitted paper.

## Where the surface is today

| | |
|---|---|
| Published on PyPI | **0.1.0**, macOS arm64 wheel, both binary sets (FAST and IDENTICAL) in one wheel |
| Estimators in the 0.1.0 release notes | 11 |
| Estimators exported by `python/mojolearn/__init__.py` at HEAD | **16** -- the tree is five ahead of the release: `KernelDensity`, `LogisticRegression`, `Ridge`, `KNeighborsClassifier`, `KNeighborsRegressor` |

So the first action is not new binding work at all. **Cutting 0.2.0 publishes
five estimators that are already written, already bound and already
certified**, and it costs one release run.

## What is built and NOT reachable from Python

Every row below is ported, gated, and (where the E3 rounds cover it)
bit-identical on three vendors. None of it is importable.

| family | what exists | what is missing | size |
|---|---|---|---|
| `isolation_forest/` | full port of cuML v26.08.00 Isolation Forest, `estimator.mojo` exists but holds helpers only | a host fit/score entry point, binding, `IsolationForest` class | **M** |
| `holtwinters/` | `holtwinters_fit_host` and `holtwinters_forecast_host` already exist | binding + `ExponentialSmoothing` class + serialization | **S -- the cheapest new family** |
| `solver/` | coordinate descent, `cdFit`/`cdPredict` ported, three-vendor card (20 stages) | host entry, binding, `Lasso` + `ElasticNet` | **M** |
| `svm/` | C-SVC: SMO, block solve, working-set selection, kernel matrices; three-vendor card (32 stages) | host entry, binding, `SVC` | **M/L** (fit returns support vectors + duals; serialization is new shape) |
| `hierarchy/` | single-linkage agglomerative, MST and dendrogram; three-vendor card (8 stages) | host entry, binding, `AgglomerativeClustering` | **M** |
| `metrics/` | r2, kl divergence, silhouette, trustworthiness, accuracy, rand/ARI, entropy, MI, h/c/v; three-vendor card (34 stages) | binding; these are FUNCTIONS not estimators, so a `mojolearn.metrics` submodule, not a class | **M -- and the highest ratio of surface to work** |
| `spectral/` | cuVS spectral embedding + cuML spectral clustering, its own identity contract | host entry, binding, `SpectralEmbedding` / `SpectralClustering` | **L** |
| `arima/`, `tsa/` | batched Kalman filter likelihood, KPSS, differencing order | host entries, binding, `ARIMA` | **L**, and the lane's own README says no second vendor has run it |
| `gemm/` | `mojolearn.identical.gemm.fp32.v1`, bit-identical at 62 shapes on three vendors | a functional API (`mojolearn.linalg.matmul`), not an estimator | **S/M**, and it is the one item a non-ML user would install for |
| `mamba/` | one Mamba-1 block, its own contract | not scikit-learn shaped; belongs behind a separate module or not at all in 0.2 | **defer** |

## The recommended order, and why

1. **Cut 0.2.0 with what is already bound.** Five estimators, no new code,
   and it makes the released surface match the repository's own README.
2. **`metrics` submodule.** Highest surface per unit of work, no new
   serialization shape, and it is what makes the library usable *with*
   scikit-learn rather than instead of it.
3. **`holtwinters`, then `solver` (Lasso/ElasticNet), then `hierarchy`.**
   Each is one host entry plus one class, and each already has a
   three-vendor card, so the identity table in the paper grows with the
   surface rather than after it.
4. **`isolation_forest`, then `svm`.** Both are wanted; both need more than a
   binding.
5. **`gemm` as a functional API**, if the audience is meant to include people
   who want a reproducible FP32 matrix product and no estimator at all.
6. **`spectral`, `arima`/`tsa`** last, and `arima` only after a second vendor
   has run it.
7. **`mamba`** is not a 0.2 item.

## What this does NOT need

- No change to the wheel's structure: it already carries every numeric mode
  as its own compiled binary set. (This read "both numeric modes ... selects
  on `MOJOLEARN_NUMERIC_MODE` at import" while there were two and the
  environment was the only selector. Since 2026-08-29 there are THREE --
  `fast`, `deterministic`, `identical` -- and the mode is a runtime parameter:
  `mojolearn.set_numeric_mode()` or `numeric_mode=` on an estimator, with the
  environment variable still setting the starting default.)
- No change to the identity machinery: each family above already has its
  ledger rows and, for six of them, three-vendor cards.
- No new claim in the paper. **The paper's certificate is per training stage
  and already includes families that are not on the Python surface** (the
  classical lanes are Mojo-only paths reached by the matrix). Exposing them
  makes the artifact match the paper; it does not change what the paper says.

## What a release run costs, and the standing constraint

`docs/PYPI_RELEASE.md` is the procedure: clean-worktree build at the M1 ISA
floor, smoke fits on real Apple GPU hardware, then upload. Two standing rules
bound how this gets done: **no heavy compute on the laptop** (builds go to the
ephemeral runner, one at a time), and **publishing is outward-facing and needs
Andrew's word each time** -- a version on PyPI cannot be recalled, only
yanked.
