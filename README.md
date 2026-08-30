# mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068632.svg)](https://doi.org/10.5281/zenodo.22068632)

**GPU machine learning in Mojo, on hardware the originals cannot reach.**

mojolearn ports the GPU implementations that established today's tree and
classical algorithms, CatBoost, cuVS, cuML and RAFT, into one Mojo source
that runs on Apple silicon through Metal and, from the same source, on NVIDIA
through CUDA and AMD through HIP. Every estimator accepts a numeric mode, and
the wheel carries all three. `FAST` is the upstream's shipped behavior, and it
promises speed and nothing else. `DETERMINISTIC` gives the same bits on every
run of one box. `IDENTICAL` gives the same bits on every supported GPU vendor,
with a per-stage certificate that proves it rather than a hash that hopes so.

The scikit-learn shapes are kept. The defaults follow the upstream each
algorithm mirrors, and every place that differs from scikit-learn is named
on the class.

## Install in five minutes

Requirements are the lowest that can run it, not the machine it was built
on. Any Apple silicon Mac from the M1 up, macOS 11 or later, Python 3.10
through 3.14, numpy 1.24 or later. The wheel and every runtime library in
it are built for macOS 11 and the M1 instruction set, and the build refuses
anything newer. There is no CPU path; every estimator runs on the GPU.

A Linux x86_64 wheel under the same name is designed and tooled
(`docs/LINUX_WHEEL.md`) and not yet published: one wheel carrying both a
CUDA and a HIP binary set in all three tiers, the vendor picked at import
and read back from the binary (`mojolearn.vendor()`). Until it is on PyPI,
NVIDIA and AMD are source builds.

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

### Numeric modes

ONE install carries all three. The mode is a PARAMETER in your code, not an
install option and not something you have to set from the shell: nothing is
rebuilt and nothing is reinstalled to change it, because each mode is a
separately compiled binary set inside the same wheel and `_backend.py` loads
whichever one is asked for.

```python
import mojolearn

mojolearn.set_numeric_mode("deterministic")     # process default, in code
rf = mojolearn.RandomForestClassifier(numeric_mode="identical")   # one estimator
km = mojolearn.KMeans(n_clusters=8)             # takes the process default

print(mojolearn.numeric_mode())                 # 'deterministic'
print(rf.numeric_mode_used())                   # 'identical'
```

More than one tier can be live in the same process, and that is measured
rather than assumed: all three were loaded together and called INTERLEAVED on
one product, and each returned its own arithmetic every time.

The environment variable still works and sets the STARTING default, so
anything written against the older spelling keeps running:

```sh
python train.py                                        # fast (the default)
MOJOLEARN_NUMERIC_MODE=deterministic python train.py
MOJOLEARN_NUMERIC_MODE=identical     python train.py
```

| mode | what it promises |
|---|---|
| `fast` | **Nothing but speed.** The same fit on the same box may return different bits on two runs. |
| `deterministic` | **Same box, same build, same input gives the same bits, every run.** Says nothing about a second box. |
| `identical` | All of the above, AND **the same bits on Apple Metal, NVIDIA CUDA and AMD HIP.** |

`fast` is not a broken `identical`. It promises speed and nothing else, so
asking a `fast` run a bitwise question is a category error.

**And `fast` really does move, on every vendor including Apple.** This is the
common wrong assumption -- that Metal has no float atomics, so Apple gets
determinism for free. It does not. Measured at one commit on 2026-08-29 by
`tools/repeat_run_stability.py`, which runs one fit repeatedly in one process
and compares RAW OUTPUT BYTES with no tolerance:

| column | `fast` | `deterministic` | `identical` |
|---|---|---|---|
| Apple M4, Metal | **MOVED in 8 of 10 attempts** | STABLE 10/10 | STABLE 10/10 |
| NVIDIA RTX 4090, CUDA | **MOVED -- 24 calls, 24 different answers** | STABLE | STABLE |
| AMD MI325X, HIP | **MOVED -- 6 different answers in 24 calls** | STABLE | STABLE |

So if you need a fit to reproduce, ask for `deterministic`. Do not assume you
already have it because you are on one machine, and do not assume Apple is a
special case. Full record, including what each leg cost to get:
[bench/results/stability/RESULTS.md](bench/results/stability/RESULTS.md).

**The middle tier is not the top one wearing a hat.** Under `deterministic`
the output hashes DIFFER between vendors on 10 of 12 comparable lanes, the
vendor matmul among them -- it keeps MAX `matmul`, cuBLAS and rocBLAS and
their speed, and buys none of the cross-vendor pinning that costs `identical`
up to 4.64x on the gemm lane.

`mojolearn.numeric_mode()` reports the mode that actually loaded, read back
out of the binary, so a run cannot be mislabeled by accident.

## What is in 0.2

**Every lane below runs the same pinned path under
`MOJOLEARN_NUMERIC_MODE=identical`**: fused multiply-add pinned, flush-to-zero
pinned, every reduction a fixed-order fold where the CUDA upstream uses float
atomics. That is true on Apple, on NVIDIA and on AMD alike, and it is what
IDENTICAL means.

The `identity diffed on` column is not about whether a lane is pinned. It is
about **where the pinned card has actually been compared**, and bit-identity
is a claim about two machines agreeing, so checking it takes two machines.

Most of this surface has been compared. On 2026-08-28, at one commit
(`a0a0eee`) on all three boxes, the phase-8 lane cards and the E2U cell cards
were emitted on an NVIDIA H100 (CUDA) and an AMD MI325X (HIP) and diffed
against the Apple M4's (Metal): cd 23 stages, gemm 61, iforest 124, kde 9,
linkage 10, metrics 64 and svm 35, plus the `ridge_*` and `logreg_*` cells,
byte-identical Apple-to-NVIDIA and Apple-to-AMD in every case.

Three lanes carry less than that. `spectral` and `holtwinters` have an
Apple card (`241aed6`) and an AMD MI325X card (`26eb8ba`, leg
`bench/results/e1/2026-08-28_203552-mojolearn-e2-amd`), and the two are
byte-identical, but a `numerics.mojo` commit sits between them, so no
same-commit certificate is claimed for either. `tsa` has run on the M4 only.
No NVIDIA leg has run any of the three. No lane inherits a neighbor's
certificate.

FAST, the default, is a different path with those pins compiled away. It
promises speed and makes no cross-vendor claim anywhere, by design.

**Reading the `DETERMINISTIC` column.** Every estimator accepts every mode --
that is a property of the library, not of any one algorithm -- so the
interesting question is not whether the tier is offered but where its promise
has been RUN. `tools/repeat_run_stability.py` refits sixteen lanes repeatedly
in one process and compares raw output bytes, and the parenthesis names the
vendors it has done that on. A row with no parenthesis ships the tier and
takes its pins from the same source, but has not been through that harness;
that is an untested promise and is marked as one rather than assumed.

| estimator | mirrors | what it does | FAST | DETERMINISTIC | IDENTICAL | identity diffed on |
|---|---|---|---|---|---|---|
| `GradientBoosting` | CatBoost GPU | oblivious (symmetric) trees, 12 losses plus MultiClass, depthwise and lossguide growth, CTRs, eval sets, overfitting detector, save and load | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `RandomForestClassifier`, `RandomForestRegressor` | cuML | quantile-split forest, with-replacement bootstrap, gini, entropy, poisson, gamma, inverse gaussian | yes | yes (Apple + AMD) | yes | Apple + NVIDIA + AMD |
| `ExtraTreesClassifier`, `ExtraTreesRegressor` | cuML | extremely randomized trees, gini, entropy, mse | yes | yes (Apple + AMD) | yes | Apple + NVIDIA + AMD |
| `KMeans` | cuVS | k-means with k-means++ init; `n_init` defaults to cuVS's 1, not scikit-learn's 10 | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `NearestNeighbors` | cuVS, RAFT, FAISS | brute-force k-NN, fused L2, ball cover, top-k selection | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `KNeighborsClassifier`, `KNeighborsRegressor` | cuVS, RAFT, FAISS | the vote and the mean on that k-NN path | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD, on the k-NN path |
| `DBSCAN` | cuML, RAFT | epsilon neighborhoods, label propagation, border and noise points | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `PCA`, `TruncatedSVD` | cuML, RAFT | eigen and SVD decompositions, transform and inverse transform | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `LinearRegression` | RAFT | ordinary least squares (`lstsqEig`) | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `Ridge` | cuML | ridge regression, the `eig` arm (`svdEig` + `ridgeSolve`) | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `LogisticRegression` | cuML | binary L-BFGS with the Armijo line search (`qnFit`); l1, multiclass, sample and class weights refused by name | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `SVC` | cuML | binary C-SVC. There is no `SVR`, and `svmType != C_SVC` raises by name | yes | yes | yes | Apple + NVIDIA + AMD, 35 card stages |
| `Lasso`, `ElasticNet` | cuML | coordinate descent (`cd.cuh::cdFit`), cyclic and random selection | yes | yes | yes | Apple + NVIDIA + AMD, 23 stages |
| `KernelDensity` | cuML | kernel density estimation; `bandwidth='scott'` and `'silverman'` refused by name | yes | yes | yes | Apple + NVIDIA + AMD, 9 stages |
| `AgglomerativeClustering` | cuML, cuVS, RAFT | single linkage over RAFT's Boruvka MST | yes | yes | yes | Apple + NVIDIA + AMD, 10 stages |
| `IsolationForest` | cuML | isolation forest, anomaly scores | yes | yes (Apple + AMD) | yes | Apple + NVIDIA + AMD, 124 card stages |
| `SpectralClustering` | cuML, cuVS, RAFT | kNN connectivity graph, normalized Laplacian, thick-restart Lanczos | yes | yes | yes | Apple + AMD cards (not at one commit); NVIDIA owed |
| `ExponentialSmoothing` | cuML `tsa` | Holt-Winters, additive and multiplicative | yes | yes | yes | Apple + AMD cards (not at one commit); NVIDIA owed |
| `kpss_test`, `select_d` | cuML `tsa` | stationarity test and auto_arima's choice of d | yes | yes | yes | Apple only, leg owed |
| `mojolearn.metrics` | cuML, RAFT | fourteen scoring functions, scikit-learn's names with cuML's defaults and semantics | yes | yes | yes | Apple + NVIDIA + AMD, 64 stages |
| `mojolearn.linalg.matmul` | -- | FP32 matrix product, profile `mojolearn.identical.gemm.fp32.v1` | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD, 61 stages |

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

## The numeric tiers

Three tiers, and each rung keeps the rung below it. Every one of them is in
the wheel you installed. The complete surface for choosing between them:

| | |
|---|---|
| `mojolearn.set_numeric_mode("deterministic")` | sets the PROCESS DEFAULT, in code, at any point. Estimators constructed before and after both honour it, because the mode is read off the instance at every call rather than frozen in `__init__`. |
| `Estimator(..., numeric_mode="identical")` | sets the mode for ONE estimator. Accepted by every estimator; it is injected by `__init_subclass__` rather than written into eleven constructor signatures, so it cannot drift between them. |
| `est.numeric_mode = "fast"` | the same thing after construction, including on an estimator unpickled from an older version, which falls back to the process default rather than raising. |
| `est.numeric_mode_used()` | the tier THIS instance will actually call into. |
| `mojolearn.numeric_mode()` | the process default, read back out of the loaded binary rather than out of the variable that asked for it. |
| `MOJOLEARN_NUMERIC_MODE=deterministic` | sets the STARTING default at import. The oldest spelling, still supported; everything above overrides it. |
| `mojolearn` (the console script) | prints which tiers are installed, which loaded, and every `MOJOLEARN_*` variable that is set. Run it before filing a reproducibility bug. |

`MOJOLEARN_IDENTITY_TRACE=<path>` is a separate feature and is OFF by default.
It writes a per-stage IDENTITY CARD -- a fingerprint of the PATH the
arithmetic took, which is why a `deterministic` card and a `fast` card are
expected to differ -- and it is not what any of the tables here hash. Those
hash the ANSWER.

Nothing above rebuilds or reinstalls anything, and more than one tier can be
live at once: they are separate compiled binary sets in one wheel, each with
its own runtime and device context.

| tier | what it promises |
|---|---|
| `fast` | Nothing. Speed only. The same fit on the same box may return different bits on two runs, and on the histogram lanes it measurably does. This is the default. |
| `deterministic` | Same box, same build, same input gives the same bits, every run. Says NOTHING about a second box. |
| `identical` | All of the above, and the same bits on Metal, CUDA and HIP. |

`deterministic` is the tier for a regression test, a byte-comparable model
file, or a fit reproducible from its seed, none of which need another vendor
to agree. It exists because that reproducibility used to be purchasable only
by taking `identical` whole, and `identical` is not free: measured on one M4
on 2026-08-28, it costs 4.64x on the gemm lane, 1.35x on coordinate descent,
1.19x on kernel density, and nothing at all on linkage, metrics and SVM.

**The wheel carries all three tiers.** It carried two until 2026-08-29, on
the stated grounds that the deterministic pin lane was unfinished and that a
tier shipping without its pins would be non-reproducible while calling itself
deterministic. That reason is retired, and by measurement rather than by pin
count: the tier is STABLE on Apple, NVIDIA and AMD against a `fast` arm that
moved on all three. The pin side is 15 files keyed to `PIN_DETERMINISM`, and
the class is small because this tree uses no float `atomicAdd` anywhere,
which is exactly why the middle tier is cheap.

Which tiers a given wheel carries is one variable,
`MOJOLEARN_RELEASE_MODES` in `packaging/macos/build_release_wheel.sh`, and
`packaging/macos/verify_wheel.sh` installs the finished wheel into a clean
venv under every claimed interpreter and fits every estimator family in each
of them. A tier that did not build raises from a missing-binary stub BY NAME
on use, rather than quietly serving fast arithmetic under another label.

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

`DETERMINISTIC` adds exactly one class of pin to that: the places where the
ORDER of a reduction is decided at runtime by the thread scheduler rather
than by the build. Float atomics, mutex merges, CAS retries. It leaves alone
everything that is fixed for a given build and merely differs BETWEEN
vendors -- machine constants, FMA contraction, flush-to-zero policy,
transcendentals, shape dispatch, and the vendor matmul -- because none of
those can move between two runs on one box. That division is why it is cheap,
and it is visible in the numbers: under `deterministic` the output hashes
differ across vendors on 10 of 12 comparable lanes, and `gemm-vendor` still
calls MAX `matmul`, cuBLAS or rocBLAS at full speed. All three of those were
measured run-to-run stable at 256x4096 @ 4096x128, a wide k chosen to provoke
a split-K epilogue, which is what earns them their exemption.

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
| NVIDIA (CUDA) | not yet published; the Linux wheel is designed and tooled, `docs/LINUX_WHEEL.md` | yes, `tools/e2_remote_leg.sh` | E1, E2, E2U, E3 (H100); run-to-run stability (RTX 4090) |
| AMD CDNA (HIP) | not yet published; same wheel as CUDA, vendor picked at import | yes, same script | E1, E2, E2U, E3, stability (MI325X); E1U (MI300X) |

Support is one source; validation is what the certificates say and nothing
more. The benchmark table below is from one machine, an M4 laptop with 10
cores and 16 GB. FAST timings on an NVIDIA H100 and an AMD MI325X are in
`bench/results/BOARD_2026-08-28_three-vendor.md`.

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
- The published wheel is macOS arm64. CUDA and HIP are source builds until
  the Linux wheel of `docs/LINUX_WHEEL.md` ships; its tooling is in the tree
  and unrun.
- The benchmark table is from one M4; FAST timings on an H100 and an MI325X
  are in `bench/results/BOARD_2026-08-28_three-vendor.md`. Correctness and
  identity are validated on the M4, an H100, an MI325X and an MI300X through
  the certificates above.
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
always resolves to the latest release. The per-version DOI is
[10.5281/zenodo.22171041](https://doi.org/10.5281/zenodo.22171041) for 0.2.0
and [10.5281/zenodo.22068633](https://doi.org/10.5281/zenodo.22068633) for
0.1.0.

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
