# mojolearn

[![PyPI](https://img.shields.io/pypi/v/mojolearn.svg)](https://pypi.org/project/mojolearn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22068632.svg)](https://doi.org/10.5281/zenodo.22068632)

**The same source, three GPU vendors, and the same bits.**

mojolearn is a GPU machine-learning library in Mojo whose distinguishing
result is a numeric contract: one source that runs on Apple silicon through
Metal, NVIDIA through CUDA and AMD through HIP, and, on request, returns
BIT-IDENTICAL answers on all three. Every estimator accepts a numeric mode
and the wheel carries all three. `FAST` promises speed and nothing else.
`DETERMINISTIC` gives the same bits on every run of one box. `IDENTICAL`
gives the same bits on every supported vendor, with a per-stage certificate
that proves it rather than a hash that hopes so.

**None of the CUDA libraries this draws on has such a tier**, because none of
them was ever asked to run anywhere but CUDA. The determinism ladder, the
Metal backend, the host control plane, the identity cards and the sabotage
methodology that gates them have no upstream. By line count, **between 67%
and 70% of the Mojo here has no upstream file it corresponds to**; the other
30% to 33% is the algorithmic substrate, and it is derived work documented as
such per file, with the upstream path and the pinned commit, in the
derivation tables and in each file's own header. **No upstream source is in
this repository**: the shipped library is 100% Mojo, there is no CUDA and no
C++ in it, and the only C++ in the tree is twenty test harnesses under
`tools/*_oracle/`. See [CONTRIBUTION.md](CONTRIBUTION.md) for what was built
and what it cost, with the command that measures each number. (An earlier version of this
paragraph said 84.9%. That was computed as "everything outside a `impl/`
directory" and `gbdt/`, the CatBoost mirror, has no such directory, so it
undercounted the derived side by 61,888 lines. The range above is the
corrected figure and its bounds are stated in `NOTICE`.) The algorithms come from CatBoost,
cuVS, cuML and RAFT; see `NOTICE`.

The scikit-learn shapes are kept. The defaults follow the upstream each
algorithm mirrors, and every place that differs from scikit-learn is named
on the class.

## Install in five minutes

Requirements are the lowest that can run it, not the machine it was built
on. Any Apple silicon Mac from the M1 up, macOS 11 or later, Python 3.10
through 3.14, numpy 1.24 or later. The wheel and every runtime library in
it are built for macOS 11 and the M1 instruction set, and the build refuses
anything newer. There is no CPU path; every estimator runs on the GPU.

A Linux x86_64 wheel under the same name is **published** and has been since
0.3.0 (`docs/LINUX_WHEEL.md`). One wheel carries both a CUDA and a HIP binary
set in all three tiers, the vendor is picked at import and read back from the
binary (`mojolearn.vendor()`), and the GPU architecture is picked at import
from five sets, `cuda/sm_80`, `cuda/sm_90a`, `hip/gfx942`, `hip/gfx1100` and
`hip/gfx90a`. A Blackwell device gets a clean refusal rather than a crash;
`cuda/sm_120a` is not in this release. Its host code is pinned to x86-64-v3,
which is Haswell 2013 and Zen 1 2017 onward.

**Install 0.3.1 or later on Linux, never 0.3.0.** The 0.3.0 Linux wheel's host
code carries AVX-512 with no cpuid dispatch and dies with SIGILL on any x86_64
host without it, which is every AMD Zen 1, 2 and 3 and most Intel consumer
parts. It was measured on an L40 whose host was a Zen 3 EPYC, and 0.3.1 is the
fix, verified on that same machine.

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
their speed, and buys none of the cross-vendor pinning `identical` pays for.
What that pinning costs is priced in `bench/LANES_PRICE.md`, on rented
single-tenant GPUs and with the fixture size beside every ratio.

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
| `RadiusNeighbors` | cuVS, RAFT | every neighbor inside a radius, over the random ball cover; exact, not approximate. Distances are recomputed from the finished neighbor list, not stored by the search | yes | yes | yes | Apple; the AMD leg is what closes the CSR's cross-vendor order (DEVIATION 551) |
| `DBSCAN` | cuML, RAFT | epsilon neighborhoods, label propagation, border and noise points; `sample_weight`, and `metric='manhattan'` on `algorithm='brute'` (that metric is OURS, not a port -- cuML's DBSCAN has no Manhattan arm) | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD for the unweighted L2 path, which is what the cards cover; `sample_weight` and the L1 arm landed 2026-09-01 and are Apple only, three-vendor legs OWED |
| `PCA`, `TruncatedSVD` | cuML, RAFT | eigen and SVD decompositions, transform and inverse transform | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `LinearRegression` | RAFT | ordinary least squares (`lstsqEig`) | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `Ridge` | cuML | ridge regression, the `eig` arm (`svdEig` + `ridgeSolve`) | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `LogisticRegression` | cuML | binary L-BFGS with the Armijo line search (`qnFit`); l1, multiclass, sample and class weights refused by name | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD |
| `SVC` | cuML | binary C-SVC | yes | yes | yes | Apple + NVIDIA + AMD, 35 card stages |
| `SVR` | cuML | epsilon-SVR over the same SMO solver (`SvrInit`, the `2 * n_rows` domain, `CombineCoefs`' fold); `NU_SVC` and `NU_SVR` raise by name, as they do upstream | yes | yes | yes | **Apple only.** Gated 44 of 44 at `fea6becc`, four property gates derived from the epsilon-insensitive formulation; NO cross-vendor round has carried it |
| `Lasso`, `ElasticNet` | cuML | coordinate descent (`cd.cuh::cdFit`), cyclic and random selection | yes | yes | yes | Apple + NVIDIA + AMD, 23 stages |
| `KernelDensity` | cuML | kernel density estimation; `bandwidth='scott'` and `'silverman'` refused by name | yes | yes | yes | Apple + NVIDIA + AMD, 9 stages |
| `AgglomerativeClustering` | cuML, cuVS, RAFT | single linkage over RAFT's Boruvka MST | yes | yes | yes | Apple + NVIDIA + AMD, 10 stages |
| `IsolationForest` | cuML | isolation forest, anomaly scores | yes | yes (Apple + AMD) | yes | Apple + NVIDIA + AMD, 124 card stages |
| `SpectralClustering` | cuML, cuVS, RAFT | kNN connectivity graph, normalized Laplacian, thick-restart Lanczos | yes | yes | yes | Apple + AMD byte-identical at one commit, `221aa141`, 171 stages; NVIDIA owed |
| `ExponentialSmoothing` | cuML `tsa` | Holt-Winters, additive and multiplicative | yes | yes | yes | Apple + AMD byte-identical at one commit, `221aa141`, 182 stages; NVIDIA owed |
| `kpss_test`, `select_d` | cuML `tsa` | stationarity test and auto_arima's choice of d | yes | yes | yes | Apple + AMD byte-identical at one commit, `221aa141`, 13 stages; NVIDIA owed |
| `ARIMA` | cuML `tsa` | BATCHED ARIMA, one series per row each with its own parameters: batched Kalman filter, `estimate_x0` over a Householder QR, own-written batched L-BFGS; `predict` (`end` EXCLUDED, cuML's convention) and `forecast`; exog, CSS, confidence intervals and missing observations refused by name | yes | yes | yes | the KALMAN FILTER is Apple + NVIDIA + AMD byte-identical at `221aa141`, 139 stages; the FIT is gated on ONE Apple M4 only and its Python surface gate (`check-arima-surface`) ran green there in both tiers on 2026-09-02, 88 checks 0 failed each, APPLE ONLY -- no cross-vendor claim for `fit` |
| `mojolearn.metrics` | cuML, RAFT | fourteen scoring functions, scikit-learn's names with cuML's defaults and semantics | yes | yes | yes | Apple + NVIDIA + AMD, 64 stages |
| `mojolearn.linalg.matmul` | -- | FP32 matrix product, profile `mojolearn.identical.gemm.fp32.v1` | yes | yes (Apple + NVIDIA + AMD) | yes | Apple + NVIDIA + AMD, 61 stages |

The last three rows moved up on 2026-08-31. `spectral`, `holtwinters` and
`tsa` now have Apple and AMD cards at ONE commit, `221aa141`, and they are
byte-identical: 171, 182 and 13 stages, 0 records differing. Their NVIDIA
column did not fail; it never ran, which is a different statement. The
certificate is `bench/results/e1/CERT_2026-08-31.md`.

`arima` is byte-identical on THREE vendors at that same commit, Apple M4
against AMD MI325X against NVIDIA, 139 stages -- the first lane in this
repository whose time-series card is closed on all three columns. That card
is the Kalman filter's. The `ARIMA` estimator over it joined the table on
2026-09-01 (this paragraph used to say it was "not in the table because it
has no `fit`"); the fit itself has run on one Apple M4 only and inherits
nothing from the filter's card.

Estimators save to and load from `.npz` files, and a model fitted on a Mac
loads and predicts identically on an NVIDIA or AMD box (95 of 95 models,
probabilities included, in the E2 certificate below).

Named rather than omitted, because "why is this missing" is a short and
interesting question: `AutoARIMA` (its `p/q/P/Q/k` search and
information-criterion arms are not ported; the differencing half ships as
`kpss_test` / `select_d`), `MultiClassOneVsAll` from Python, Intel and
Qualcomm GPU columns, and a CPU fallback of any kind. `RadiusNeighbors` was
on this list until 2026-08-31, `SVR` until 2026-09-01, and `ARIMA` -- the
longest-standing entry, "no `estimate_x0`, no L-BFGS, no `fit`" -- also
until 2026-09-01; all three are in the table above now.

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
by taking `identical` whole, and `identical` is not free. The price is
measured on rented single-tenant GPUs and lives in `bench/LANES_PRICE.md`,
with the fixture size beside every ratio, because three lanes changed by more
than 3x between two sizes and one changed sign; on the matrix and neighbour
lanes the cost is large and GROWS with size. The Apple figures that used to
stand here are RETRACTED -- taken on a laptop carrying 7.5 GB of swap and
never reproduced, and a contended box adds a fixed per-launch cost to BOTH
arms that pulls every ratio toward 1.0, so it understates the tax rather than
bounding it.

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
| NVIDIA (CUDA) | yes, Linux x86_64 since 0.3.0, `pip install mojolearn`; sets `sm_80` and `sm_90a`, no Blackwell | yes, `tools/e2_remote_leg.sh` | E1, E2, E2U, E3 (H100); run-to-run stability (RTX 4090) |
| AMD CDNA (HIP) | yes, same wheel as CUDA, vendor and architecture picked at import; sets `gfx942`, `gfx90a`, `gfx1100` | yes, same script | E1, E2, E2U, E3, stability (MI325X); E1U (MI300X) |

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
themselves -- but NOT in the table below: the hash patch landed at `fe25d2e3`
at 16:32 on 2026-08-22 and these runs are from 15:41, so their JSONs carry no
`hashes` field. The sentence describes the harness as it stands today, and the
rows beneath it predate that. The reproduction command is

```sh
pixi run -e gbmbench bash bench/external/run_gbm_bench.sh year 500 gbdt
pixi run -e gbmbench bash bench/external/run_gbm_bench.sh covtype 100 forest
```

and the results and the machine record land in `bench/results/gbm_bench_*`.
These are the last interleaved runs of 2026-08-22 on the M4 described above.

| dataset, arm pair | ours (Metal) | theirs (CPU, same M4) | speed | accuracy |
|---|---|---|---|---|
| year, symmetric GBDT 500 rounds, vs CatBoost CPU | 11.9 s | 13.6 s | 1.14x | MAE 6.261 vs 6.263, MSE 80.11 vs 79.98 |
| higgs, symmetric GBDT, vs CatBoost CPU | 131.5 s | 177.8 s | 1.35x | AUC 0.822 vs 0.830, a real gap, see below |
| covtype, Extra Trees 100 trees, vs scikit-learn on 10 cores | 3.7 s | 5.4 s | 1.46x | 0.647 vs 0.645 |
| covtype, Random Forest, vs scikit-learn on 10 cores | 4.0 s | 5.3 s | 1.33x | 0.722 vs 0.720 |
| higgs, Extra Trees, vs scikit-learn on 10 cores | 39.0 s | 132.4 s | 3.4x | AUC 0.709 vs 0.700 |
| higgs, Random Forest, vs scikit-learn on 10 cores | 47.1 s | 310.3 s | 6.6x | AUC 0.775 vs 0.775 |

Six interleaved year runs over the day ranged from 1.14x to 1.68x as the
laptop warmed; the table shows the last one, not the best one. The CatBoost
pair compares symmetric trees only, because LightGBM has no symmetric mode.

**INVESTIGATED AND CLOSED (2026-09-02): the higgs symmetric row is
valid; full-scale reruns on a loaded box are not.** A 2026-09-02 rerun at
`27ae6c82` measured 586.4 s at 8.0M train rows and prompted a full
investigation. Findings, in order: (1) the engine did not regress —
stage-timed walls at `dfa41bb` (this table's commit) and current HEAD are
equal to within noise on the same fixture; (2) the shipped Python
extension did not regress — an interleaved A/B at 2M rows, old extension
built at `dfa41bb` in a worktree versus HEAD's, gave 75.8/82.2 s vs
74.3/80.8 s over two rounds with the CatBoost canary stable; (3) the
decisive run: the `dfa41bb` extension at this table's exact shape (full
higgs, 500 trees) took 745.4 s on 2026-09-02 with AUC 0.82167 matching
this row's record exactly — same commit, same computation, same output,
5.7x the recorded time, and CatBoost itself ran 388.3 s against the
177.8 s recorded here (2.2x). Both arms slower with identical outputs is
a box condition, not a code change: the 2026-09-02 box carried ~11 GB of
used swap from days of uptime, where this table's runs were taken under
the quiet-box protocol. The row stands as recorded at `dfa41bb`;
full-scale (8M+) numbers taken on a swap-loaded box, including the
2026-09-02 overnight experiment's symmetric rows, must not be compared
against it. The overnight forest rows' internal ordering (RF 2.4x, ET
1.2x over sklearn at 8M) is a same-window comparison and survives its
own conditions.

THE YEAR ROW IS QUOTED ON BOTH METRICS AND THE DIRECTION REVERSES BETWEEN
THEM. Until 2026-08-31 it quoted MAE only, which is the one of the two we
win. The two internal records this table is drawn from
(`bench/results/THREE_SUITE_QUIET_2026-08-22.md:26`,
`bench/results/THREE_SUITE_2026-08-22.md:23`) both quote that row on MSE,
where we lose by 0.13. Quoting the metric that flatters us while the source
record quotes the other one is the reporting half of `never build to
datasets`, so both are now shown.

THE HIGGS GBDT ACCURACY GAP IS REAL, IT IS NOT A CONFIGURATION ARTIFACT,
AND MATCHING THE CONFIGURATION MAKES IT SLIGHTLY WORSE. Measured
2026-08-31 at the full 8.8M x 28 train split, 500 rounds, depth 8, both
arms in one process on the M4:

| CatBoost CPU config | their AUC | ours | delta | their fit |
|---|---|---|---|---|
| as gbm-bench ships it (MVS, subsample 0.8, random_strength 1.0) | 0.8303359534882081 | 0.8216098413977209 | **-0.00873** | 213.1 s |
| matched (`bootstrap_type='No'`, `random_strength=0`, `boosting_type='Plain'`) | 0.8304610825961414 | same fit, same bits | **-0.00885** | 264.7 s |

Ours fits in 129.4 s, so 1.65x and 2.05x against those two. Our arm is
fitted ONCE and scored against both, so our column cannot move between the
rows and any change is theirs.

This retires a hypothesis this repository had carried since 2026-08-22 as
"the tracked CONFIG-PARITY item, ruling pending". The suspicion was that
gbm-bench passes CatBoost no `bootstrap_type` and no `random_strength`, so
their arm ran its shipped regularization (MVS at subsample 0.8,
`random_strength` 1.0) while ours ran neither, and that this was what the
-0.0084 measured. It was not. Turning their regularization OFF moved
CatBoost UP, from 0.83034 to 0.83046, and widened the gap. The
configuration difference was real and worth stating; it is not the cause,
and an earlier draft of this very section leaned on it as though it were.

`boosting_type` is settled too, and it was the benign answer: read back off
the fitted model, CatBoost resolved it to `Plain` on its own at this size,
so the unpinned knob was never costing anything on this row.

What remains is a genuine accuracy deficit of about 0.0089 AUC at this
shape against CatBoost CPU, with the leading suspect now the one mechanism
the tree has actually measured moving this number: the Logloss Newton
leaf-estimation walk of PORTING.md item 140, where their CPU freezes at six
accepted steps and ours stalls later. Narrowing our host fold to Float32
moved AUC +0.00037 there, which is 4% of the gap and confirms the mechanism
is live without explaining the size. Depth 8 over 500 trees is where that
divergence has the most iterations to compound.

That sweep has now run (`bench/results/HIGGS_GAP_2026-09-01.md`, four
matched points on the full 8.8M split, one process, M4). The gap grows with
BOTH depth and tree count, and the two effects are very nearly separable:
depth 6 to 8 multiplies it by 1.48 at 100 trees and 1.81 at 500, while 100
to 500 trees multiplies it by 7.4 at depth 6 and 9.0 at depth 8. Boosting
iterations dominate by roughly 5x. A gap that multiplies cleanly in each
factor independently is the signature of a small per-tree error accumulating
down the boosting sequence, with depth setting how large each tree's
contribution is; it is not a threshold, a single bad split, or a
shape-specific artifact, any of which would show as a jump rather than a
clean product.

Set against that, the same learner at a matched config on a SMALLER shape
(100 trees, depth 6, 1M rows) is at parity or slightly ahead on this
dataset, on two vendors:
`bench/results/fast_speed/2026-08-26-nvidia-forest-postfix.md:58` and
`bench/results/fast_speed/2026-08-27-APPLE-trees-evening.md:79`. Both
readings are on this page because the shape is the difference between them
and nothing yet explains which way it cuts.

## Limitations and refusals

- GPU only. No CPU fallback exists and none is planned for this release.
- **0.3.0's LINUX WHEEL IS DEFECTIVE AND SHOULD NOT BE INSTALLED.** Its host
  code was compiled with AVX-512 and no runtime `cpuid` dispatch, so it dies
  with `SIGILL` on the first `fit()` on any x86-64 host without AVX-512, which
  includes every AMD Zen 1, 2 and 3 part, most Intel consumer chips and every
  Xeon before Skylake-SP. Measured 2026-08-31 on an NVIDIA L40 whose host was
  an AMD EPYC 7773X. All thirty extensions carry it. The GPU is not involved.
  **0.3.1 is the fix.** The macOS wheel of 0.3.0 is unaffected.
- Two wheels are published, macOS arm64 and Linux x86_64. The Linux wheel
  carries six GPU architecture sets in one file, `cuda/sm_80`, `cuda/sm_90a`,
  `cuda/sm_120a`, `hip/gfx942`, `hip/gfx1100` and `hip/gfx90a`, and picks one
  at import from the device. It is `manylinux_2_35`, so glibc 2.35 or later,
  which puts Ubuntu 22.04 and Debian 12 in and leaves RHEL 9 out by one minor
  version. **There is no Windows wheel and none is possible today**, because
  the Mojo toolchain has no Windows target; WSL2 works, being Linux. FOUR of
  the six sets have run on real silicon and two ship build-verified only,
  which `bench/results/wheels/LEGS_2026-08-30.md` states set by set:
  `cuda/sm_80` on an A40 (`sm_86`) and again on an L40 (`sm_89`) from real
  PyPI, `cuda/sm_90a` on an H100, `cuda/sm_120a` on an RTX 5090 from real
  PyPI, and `hip/gfx942` on an MI325X. `hip/gfx1100` and `hip/gfx90a` have
  no box to run on, because neither RunPod nor DigitalOcean rents RDNA3 or
  MI200 hardware. The within-family fallback is measured, not documented:
  the `sm_80` set ran 29 of 29 lanes in all three tiers on `sm_86` silicon
  (`c526e58b`) and again on `sm_89` from the published 0.3.1 wheel
  (`bench/results/wheels/2026-08-31_124809-nvidia`).
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
[NOTICE](NOTICE) carries each upstream's attribution. `DERIVATION_MAP.tsv` maps
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
[10.5281/zenodo.22211591](https://doi.org/10.5281/zenodo.22211591) for 0.3.1,
[10.5281/zenodo.22181752](https://doi.org/10.5281/zenodo.22181752) for 0.3.0,
[10.5281/zenodo.22171041](https://doi.org/10.5281/zenodo.22171041) for 0.2.0
and [10.5281/zenodo.22068633](https://doi.org/10.5281/zenodo.22068633) for
0.1.0.

### Do you also have to cite CatBoost, cuML, cuVS and RAFT?

Almost certainly not, and the two halves of that answer are different things
that are easy to conflate.

**If you USE mojolearn** and publish a result, cite mojolearn. You do not
inherit its bibliography, any more than citing a paper obliges you to cite
everything in its reference list. Citation is not transitive.

The one thing that is worth citing alongside is the METHOD, if your paper
describes it. A paper that says "gradient-boosted oblivious trees" would
normally cite CatBoost for the method whichever implementation it ran, and
that is ordinary methods attribution rather than anything to do with where
this library's code came from.

**If you REDISTRIBUTE mojolearn** -- fork it, vendor it, ship it inside your
own package -- Apache 2.0 section 4(d) asks you to carry a readable copy of
the attribution notices from the `NOTICE` file. `NOTICE` already contains the
CatBoost, cuVS, cuML, RAFT, FAISS, HuggingFace and other attributions, so
carrying it is the whole of that obligation. You do not have to go read those
projects and assemble your own list. That part IS transitive, deliberately:
the attribution work here exists so that nobody downstream has to repeat it.

**Just using the library, including in commercial work, requires no
attribution at all.** Apache 2.0's conditions attach to distributing the work
or a derivative of it, not to running it.

This is a description of what the licence says and of ordinary citation
convention, not legal advice; an unusual redistribution arrangement is worth
a lawyer rather than a README.

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
