# mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068632.svg)](https://doi.org/10.5281/zenodo.22068632)

**GPU machine learning in Mojo, on hardware the originals cannot reach.**

mojolearn ports the GPU implementations that established today's tree and
classical algorithms, CatBoost, cuVS, cuML and RAFT, into one Mojo source
that runs on Apple silicon through Metal and, from the same source, on NVIDIA
through CUDA and AMD through HIP. Every estimator has two numeric modes.
`FAST` is the upstream's shipped behavior. `IDENTICAL` produces bit-identical
models on every supported GPU vendor, with a per-stage certificate that
proves it rather than a hash that hopes so.

The scikit-learn shapes are kept. The defaults follow the upstream each
algorithm mirrors, and every place that differs from scikit-learn is named
on the class.

## Install in five minutes

Requirements are the lowest that can run it, not the machine it was built
on. Any Apple silicon Mac from the M1 up, macOS 11 or later, Python 3.10
through 3.14, numpy 1.24 or later. The wheel and every runtime library in
it are built for macOS 11 and the M1 instruction set, and the build refuses
anything newer. There is no CPU path; every estimator runs on the GPU.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install mojolearn
```

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((100_000, 20), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float32)

gb = mojolearn.GradientBoosting(loss="Logloss", n_estimators=200, max_depth=6)
gb.fit(X, y)
print(gb.predict_proba(X[:5]))

rf = mojolearn.RandomForestClassifier(n_estimators=100, max_depth=12, random_state=7)
print(rf.fit(X, y).predict(X[:5]))

km = mojolearn.KMeans(n_clusters=8).fit(X)
print(km.cluster_centers_.shape, mojolearn.numeric_mode())
```

To train the bit-identical way, set one environment variable before the
import. Nothing is rebuilt; the wheel carries both binary sets.

```sh
MOJOLEARN_NUMERIC_MODE=identical python train.py
```

`mojolearn.numeric_mode()` reports the mode that actually loaded, read back
from the binary, so a run cannot be mislabeled by accident.

## What is in 0.2

The `3 vendors` column is the one to read. It says whether that lane's
IDENTICAL card has been measured bit-identical on Apple M4, NVIDIA H100 and
AMD MI325X, or whether it has only ever run on one Apple M4. **The standing
is not uniform, and no lane inherits a neighbour's certificate.** FAST, the
default, makes no cross-vendor claim anywhere.

| estimator | mirrors | what it does | FAST | IDENTICAL | 3 vendors |
|---|---|---|---|---|---|
| `GradientBoosting` | CatBoost GPU | oblivious (symmetric) trees, 12 losses plus MultiClass, depthwise and lossguide growth, CTRs, eval sets, overfitting detector, save and load | yes | yes | yes |
| `RandomForestClassifier`, `RandomForestRegressor` | cuML | quantile-split forest, with-replacement bootstrap, gini, entropy, poisson, gamma, inverse gaussian | yes | yes | yes |
| `ExtraTreesClassifier`, `ExtraTreesRegressor` | cuML | extremely randomized trees, gini, entropy, mse | yes | yes | yes |
| `KMeans` | cuVS | k-means with k-means++ init; `n_init` defaults to cuVS's 1, not scikit-learn's 10 | yes | yes | yes |
| `NearestNeighbors` | cuVS, RAFT, FAISS | brute-force k-NN, fused L2, ball cover, top-k selection | yes | yes | yes |
| `KNeighborsClassifier`, `KNeighborsRegressor` | cuVS, RAFT, FAISS | the vote and the mean on that k-NN path | yes | yes | the k-NN path is |
| `DBSCAN` | cuML, RAFT | epsilon neighborhoods, label propagation, border and noise points | yes | yes | yes |
| `PCA`, `TruncatedSVD` | cuML, RAFT | eigen and SVD decompositions, transform and inverse transform | yes | yes | yes |
| `LinearRegression` | RAFT | ordinary least squares (`lstsqEig`) | yes | yes | yes |
| `Ridge` | cuML | ridge regression, the `eig` arm (`svdEig` + `ridgeSolve`) | yes | yes | Apple M4 only |
| `LogisticRegression` | cuML | binary L-BFGS with the Armijo line search (`qnFit`); l1, multiclass, sample and class weights refused by name | yes | yes | Apple M4 only |
| `SVC` | cuML | binary C-SVC. There is no `SVR`, and `svmType != C_SVC` raises by name | yes | yes | yes, 32 card stages |
| `Lasso`, `ElasticNet` | cuML | coordinate descent (`cd.cuh::cdFit`), cyclic and random selection | yes | yes | yes, 20 stages |
| `KernelDensity` | cuML | kernel density estimation; `bandwidth='scott'` and `'silverman'` refused by name | yes | yes | yes, 7 stages |
| `AgglomerativeClustering` | cuML, cuVS, RAFT | single linkage over RAFT's Boruvka MST | yes | yes | yes, 8 stages |
| `IsolationForest` | cuML | isolation forest, anomaly scores | yes | yes | Apple M4 only |
| `SpectralClustering` | cuML, cuVS, RAFT | kNN connectivity graph, normalized Laplacian, thick-restart Lanczos | yes | yes | Apple M4 only |
| `ExponentialSmoothing` | cuML `tsa` | Holt-Winters, additive and multiplicative | yes | yes | Apple M4 only |
| `kpss_test`, `select_d` | cuML `tsa` | stationarity test and auto_arima's choice of d | yes | yes | Apple M4 only |
| `mojolearn.metrics` | cuML, RAFT | fourteen scoring functions, scikit-learn's names with cuML's defaults and semantics | yes | yes | yes, 34 stages; the card has since grown to 61 and that leg is owed |
| `mojolearn.linalg.matmul` | -- | FP32 matrix product, profile `mojolearn.identical.gemm.fp32.v1` | yes | yes | yes, 60 stages |

Estimators save to and load from `.npz` files, and a model fitted on a Mac
loads and predicts identically on an NVIDIA or AMD box (95 of 95 models,
probabilities included, in the E2 certificate below).

Named rather than omitted, because "why is this missing" is a short and
interesting question: `ARIMA` (the batched Kalman filter, its gradient and
predict all exist; `estimate_x0` and the batched L-BFGS driver do not, so
there is no `fit`), `SVR`, `RadiusNeighbors`, `MultiClassOneVsAll` from
Python, Intel and Qualcomm GPU columns, and a CPU fallback of any kind.
Importing one of the first three raises with the line where the thing that
exists stops, rather than an `AttributeError`.

## The two numeric modes

`FAST` is the default. Histograms flush through float atomics, library
reductions follow the hardware warp width, and the last bits of a model can
move between two runs on the same device. That is CatBoost's shipped
behavior and it is the fastest the hardware goes. One vendor fact rides
with it: on NVIDIA, MAX's fp32 matrix product is a TF32 tensor-core product
by default, so the `FAST` products that go through it there (the Gram step of
`PCA`, `TruncatedSVD` and `LinearRegression`, PCA's transforms, k-NN
brute-force distances, k-means++ seeding costs; not the k-means assignment)
carry TF32 accuracy, roughly four significant digits on each operand
([VENDOR_LIBRARIES.md](VENDOR_LIBRARIES.md), "FAST products on NVIDIA are
TF32"). On Apple M1-M4 and AMD CDNA the same products are fp32. `IDENTICAL`
never calls the vendor product, on any vendor.

`IDENTICAL` pins every pathway that can move a bit. The enumeration of those
pathways is [IDENTITY_PATHS.md](IDENTITY_PATHS.md), 32 rows, each with what
`IDENTICAL` does about it and the status of that closure. The mode makes
exactly three kinds of move, pin a machine-derived parameter to a frozen
floor, replace an order-dependent operation with a fixed-point or fixed-shape
one, or refuse by name. There is no fourth move and no "usually fine".

### Identity coverage, measured on three vendors

One source tree, one commit, byte-identical inputs proven by hash, and the
device plus toolchain as the only variable. Every fit writes an identity
trace card, one hash per training stage, and cards are diffed stage by stage.

| certificate | what was compared | Apple M4 (Metal) vs NVIDIA H100 (CUDA) | Apple M4 vs AMD MI325X (HIP) |
|---|---|---|---|
| E2, the sub-feature matrix ([E2_RESULTS.md](E2_RESULTS.md)) | 99 configurations across all 13 GBDT losses, 4 bootstraps, 4 score functions, both searchers, CTRs, NaN modes, 5 bin widths, 3 depths, 13 Extra Trees and 18 Random Forest configurations, plus depthwise, lossguide, one-vs-all and feature-parallel | 93 identical on the full card, 2 identical on the host arm, 4 refused with the same message, **0 divergent** | 93, 2, 4, **0 divergent** |
| E1, one configuration per family ([E1_RESULTS.md](E1_RESULTS.md)) | Extra Trees classification, Random Forest regression, symmetric GBDT RMSE and Logloss, 99 to 302 stages per fit | ET and RF identical on every stage; GBDT RMSE predictions identical | ET, RF and GBDT RMSE identical on every stage |
| Train here, infer there | 95 models fitted on the Mac, loaded and predicted on the box | 95 of 95 prediction hashes equal | 95 of 95 |

The Logloss row in E1 diverged on a device exp and log seam that the ledger
had named before any hardware ran, and round 2 of E2 closed that class for
the whole matrix. The unsupervised estimators have their own ledger rows and
Apple to AMD cards ([UNSUPERVISED_IDENTITY.md](UNSUPERVISED_IDENTITY.md)).

Reproduce a certificate with the runbook in [E1_RUNBOOK.md](E1_RUNBOOK.md).
On the Mac, one fit and its card is

```sh
pixi run python tools/e1_traced_fit.py --fit et_clf --out cards/mac
```

and the comparison against a card from another box is
`python tools/identity_trace_diff.py cards/mac/et_clf.card cards/box/et_clf.card`.

## Hardware

| GPU | wheel | source build | certificates |
|---|---|---|---|
| Apple silicon (Metal) | yes, macOS arm64, `pip install mojolearn` | yes | E1, E2 |
| NVIDIA (CUDA) | no wheel yet | yes, `tools/e2_remote_leg.sh` | E1, E2 on H100 |
| AMD CDNA (HIP) | no wheel yet | yes, same script | E1, E2 on MI325X and MI300X |

Support is one source; validation is what the certificates say and nothing
more. Performance has been measured on exactly one machine, an M4 laptop
with 10 cores and 16 GB, and the numbers below carry that.

## Benchmarks

Method first. We run NVIDIA's gbm-bench harness unmodified except for
registering our arms and letting its imports survive a machine with no CUDA.
Their timing code, their datasets, their metrics, their parameters for their
arms. On Apple silicon none of the established libraries has a working GPU
path, so the comparison is our Metal path against their CPU path, on the
same machine, all arms of one invocation interleaved in one process because
the box drifts across thermal windows. Every arm records a sha256 of its
prediction vector beside its timing, so determinism is visible in the results
themselves. The reproduction command is

```sh
pixi run -e gbmbench bash bench/external/run_gbm_bench.sh year 500 gbdt
pixi run -e gbmbench bash bench/external/run_gbm_bench.sh covtype 100 forest
```

and the results and the machine record land in `bench/results/gbm_bench_*`.
These are the last interleaved runs of 2026-08-22 on the M4 described above.

| dataset, arm pair | ours (Metal) | theirs (CPU, same M4) | speed | accuracy |
|---|---|---|---|---|
| year, symmetric GBDT 500 rounds, vs CatBoost CPU | 11.9 s | 13.6 s | 1.14x | MAE 6.261 vs 6.263 |
| higgs, symmetric GBDT, vs CatBoost CPU | 131.5 s | 177.8 s | 1.35x | AUC 0.822 vs 0.830, an open parity gap |
| covtype, Extra Trees 100 trees, vs scikit-learn on 10 cores | 3.7 s | 5.4 s | 1.46x | 0.647 vs 0.645 |
| covtype, Random Forest, vs scikit-learn on 10 cores | 4.0 s | 5.3 s | 1.33x | 0.722 vs 0.720 |
| higgs, Extra Trees, vs scikit-learn on 10 cores | 39.0 s | 132.4 s | 3.4x | AUC 0.709 vs 0.700 |
| higgs, Random Forest, vs scikit-learn on 10 cores | 47.1 s | 310.3 s | 6.6x | AUC 0.775 vs 0.775 |

Six interleaved year runs over the day ranged from 1.14x to 1.68x as the
laptop warmed; the table shows the last one, not the best one. The CatBoost
pair compares symmetric trees only, because LightGBM has no symmetric mode.
The higgs GBDT accuracy gap is stated because it is there.

## Limitations and refusals

- GPU only. No CPU fallback exists and none is planned for this release.
- The wheel is macOS arm64. CUDA and HIP are source builds.
- Validated on one M4 for performance; correctness and identity validated on
  the M4, an H100, an MI325X and an MI300X through the certificates above.
- `IDENTICAL` refuses rather than guessing. A GPU column that misses the
  frozen identity floor, a k-NN arm that needs a warp primitive a column does
  not have, or `k > 256` on the identical k-NN selector raise with a named
  reason instead of returning a model that might not match.
- `KMeans.n_init` is 1. `GradientBoosting` defaults are CatBoost's.
- No Intel or Qualcomm GPU column has hardware to build for yet.
- Numerics are float32 on the device; no float64 device path.

## Provenance and licensing

Every line of Mojo here was written in this repository and Andrew Hendel
holds its copyright. What it mirrors is the design of the upstreams, file for
file where the toolchain allows, under the rule copy, do not improve, so that
a port can be checked against its original. That makes the ported
directories derivative works under Apache-2.0 section 4, and
[NOTICE](NOTICE) carries each upstream's attribution. `PORTED_MAP.tsv` maps
every ported file to its origin and status. This is not a clean-room
reimplementation and must not be described as one.

| directory | upstream | license |
|---|---|---|
| `gbdt/` | CatBoost, YANDEX LLC | Apache-2.0 |
| `ensemble/`, `extratrees/`, `dbscan/`, `decomposition/` | cuML, NVIDIA | Apache-2.0 |
| `cluster/`, `neighbors/` | cuVS, RAFT, NVIDIA | Apache-2.0 |
| `neighbors/` (warp select) | FAISS, Meta | MIT |
| `glm/` | RAFT, NVIDIA | Apache-2.0 |

License Apache-2.0, in [LICENSE](LICENSE). [AUTHORS.md](AUTHORS.md) records
who wrote what. Not affiliated with, endorsed by, or sponsored by YANDEX LLC,
NVIDIA, Meta, or Modular, Inc. MAX and Mojo are trademarks of Modular, Inc.
used under license.

## Citing

If mojolearn is useful in work you publish, please cite it.
[CITATION.cff](CITATION.cff) is what GitHub's **Cite this repository** button
reads. Each GitHub release is archived on Zenodo with its own DOI; the
concept DOI [10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632)
always resolves to the latest release, and 0.1.0 is
[10.5281/zenodo.22068633](https://doi.org/10.5281/zenodo.22068633).

## Where things are

- [IDENTITY_PATHS.md](IDENTITY_PATHS.md), the bit-identity ledger.
- [E1_RESULTS.md](E1_RESULTS.md), [E2_RESULTS.md](E2_RESULTS.md),
  [E1_RUNBOOK.md](E1_RUNBOOK.md), the cross-vendor certificates and how to
  reproduce them.
- [PORTING_RULES.md](PORTING_RULES.md), [PORTING.md](PORTING.md), how the
  port is done and every numbered deviation from the upstream.
- [VENDOR_LIBS.md](VENDOR_LIBS.md), where a vendor library is called instead
  of hand-written.
- [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md), the working notes that were
  this README during the port.
- [docs/PYPI_RELEASE.md](docs/PYPI_RELEASE.md), how a release is built,
  verified and published.
- [ROADMAP.md](ROADMAP.md), what comes next.
