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

Published on PyPI, macOS arm64 wheel, every numeric mode as its own compiled
binary set in one wheel. **Read `python/mojolearn/__init__.py`'s `__all__`, not
a table here** -- the count in this file went 11 -> 16 -> 24 in seven days and
every written-down number was false the day after. At 2026-08-31 it is 24
estimators plus the `metrics` and `linalg` submodules and the `kpss_test` /
`select_d` functions.

## What is built and still NOT reachable from Python

This table had ten rows on 2026-08-24. Eight of them shipped: `holtwinters`
(`ExponentialSmoothing`), `solver` (`Lasso`, `ElasticNet`), `hierarchy`
(`AgglomerativeClustering`), `metrics`, `isolation_forest` (`IsolationForest`),
`svm` (`SVC`), `gemm` (the `linalg` submodule) and spectral clustering
(`SpectralClustering`). Those rows are deleted rather than annotated. What is
left:

| family | what exists | what is missing | size |
|---|---|---|---|
| `spectral/` | cuVS spectral embedding as well as cuML spectral clustering, its own identity contract | `SpectralEmbedding` -- the CLUSTERING half shipped, the EMBEDDING half did not | **M** |
| `arima/`, `tsa/` | batched Kalman filter likelihood; KPSS and differencing order SHIPPED as `kpss_test` / `select_d`; arima's card is byte-identical on THREE vendors at one commit, `221aa141`, 139 stages (`bench/results/e1/CERT_2026-08-31.md`), and `tsa`'s is byte-identical Apple against AMD at that commit, 13 stages, NVIDIA owed | `ARIMA` still has no `fit`, because `estimate_x0` and the batched L-BFGS driver are unported | **L** |
| `mamba/` | one Mamba-1 block, its own contract | not scikit-learn shaped; belongs behind a separate module or not at all | **defer** |

The ordering rule the shipped eight followed, and the reason it worked: take
the family with the highest surface per unit of work and an existing
three-vendor card, so the identity table grows WITH the surface rather than
after it.

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
