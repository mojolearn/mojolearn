# The classical speed lane: our FAST path against what an NVIDIA user runs

**Nothing in this file is a measurement.** It describes a harness that was
written and has never been built or run. The first run belongs to the
orchestrator on a rented NVIDIA box, and until that happens every claim here
is about what the code *asks*, not about what any GPU *answered*.

## The question

For every classical algorithm family in this repository: how fast is our
**FAST path** against the **native NVIDIA competitor**, on an NVIDIA GPU?

FAST path means the default build, without `-D MOJOLEARN_NUMERIC_IDENTICAL=1`.
It is the non-deterministic arm. That is deliberate: the question is pure
speed against what an NVIDIA user would actually install, which is cuML,
cuVS, cuSOLVER and RAPIDS.

**This is not `bench/LANES_PRICE.md`'s question.** That file prices what
conforming to a bit-identical profile costs, by racing our FAST build against
our IDENTICAL build. A ratio from that file and a ratio from this one are
about different variables and must never appear in one table. Two lanes here
(`svm`, `metrics`) also deliberately time a *different amount of work* from
their `lanes_price` namesakes; see "Overlap with lanes_price" below.

## The two files

| file | what it is |
| --- | --- |
| `bench/speed/classical_speed_main.mojo` | our side. One Mojo driver, FAST path, one lane per process, lane chosen by `MOJOLEARN_SPEED_LANE`. |
| `tools/speed_cuml_arm.py` | the opponent. Same lane names, same shapes, same data, one lane per process, every import inside a `try`. |

`tools/fast_speed_table.py` already exists and already parses the line format
both sides emit. It expects a flat run directory of `<family>.<lane>.<arm>.log`
files. `family` is `classical` for every line either of these two files emits.

## How to run it

Both sides must see the same bytes. Two lanes-worth of mechanism:

1. **kmeans, dbscan, pca, ols, knn** generate their data from a splitmix64
   recurrence. `bench/bench_sklearn.py` already owns the vectorized twin, so
   `tools/speed_cuml_arm.py` **loads `u01` out of that file by path** rather
   than writing a third copy. No dump needed.
2. **Every other lane** has a Mojo fixture builder with no Python twin. The
   Mojo driver writes the inputs out and the Python arm reads them:

```sh
mkdir -p /tmp/speedfix

# once per lane: write the fixture
MOJOLEARN_SPEED_DUMP=/tmp/speedfix MOJOLEARN_SPEED_LANE=kde \
    pixi run mojo run -I . bench/speed/classical_speed_main.mojo

# then the race, one process each
MOJOLEARN_SPEED_LANE=kde \
    pixi run mojo run -I . bench/speed/classical_speed_main.mojo \
    > run/classical.kde.ours.log
MOJOLEARN_SPEED_DUMP=/tmp/speedfix MOJOLEARN_SPEED_LANE=kde \
    python3 tools/speed_cuml_arm.py > run/classical.kde.cuml-gpu.log

python3 tools/fast_speed_table.py run --out CLASSICAL_SPEED_RESULTS.md
```

Floats travel in the dump as their **bits**, eight hex digits per line, never
as decimals: `String(Float32)` does not round trip in this toolchain
(`[[mojo-string-float-roundtrip]]`) and a decimal dump would hand the opponent
a different dataset while looking correct. **Dump first, race after.** A
group-2 lane run with no dump present prints `FSPEED-REFUSED` rather than
inventing data.

Environment, identical on both sides:

| variable | meaning |
| --- | --- |
| `MOJOLEARN_SPEED_LANE` | required; the lane name |
| `MOJOLEARN_SPEED_ROUNDS` | timed rounds, default 5 |
| `MOJOLEARN_SPEED_SIZE` | `shipped` (default) or `smoke` |
| `MOJOLEARN_SPEED_DUMP` | fixture directory |
| `MOJOLEARN_SPEED_HASH_MAX` | Python side only; bytes hashed before giving up, default 262144 |

## What is measured

One round is one fit, or one score pass, through the lane's public entry, on
a fixture built once before the loop. The clock is around the call plus the
`synchronize()` that proves it finished: on this stack an unsynchronized
timing measures enqueue rate and nothing else. The Python side calls
`cupy.cuda.runtime.deviceSynchronize()` or `torch.cuda.synchronize()`,
whichever runtime the arm loaded.

One untimed warm-up round runs first and is printed as `FSPEED-WARMUP`. It is
never in the table and it is never hidden.

Setup, allocation, data generation and host-to-device transfer are outside
the clock **on both sides wherever the lane's entry allows it**. Ten lanes do
not allow it, and that is stated rather than buried; see the next section.

### Lanes whose timed region contains more than compute

| lane | what else is inside the clock |
| --- | --- |
| `svm` | `svc_fit` takes host lists and uploads them itself. At 240 x 2 this is negligible, and cuML's `fit` pays the same kind of cost. |
| `ivf` | host lists in, host lists out. The cuVS arm is given device arrays but its build allocates and its search downloads, so both sides carry a transfer. |
| `hdbscan` | `fit_hdbscan` takes a host list beside the device buffer. |
| `gmm` | `gaussian_mixture_fit` constructs its **own `DeviceContext`** per call (`mixture/estimator.mojo:704`). |
| `gp` | `gpr_fit_host` and `gpr_predict_host` each construct their own `DeviceContext`. |
| `krr`, `nystroem`, `rbfsampler` | same: each host entry constructs a context. |
| `resample`, `spectral` | host lists in and out. |
| `cholesky` | the ridge, the log determinant and the solve are all inside, because the torch arm's four steps are the same four. |

**A reader comparing a 3 ms `gmm` against a 3 ms sklearn arm has to know that
most of our 3 ms is a context construction.** That is a finding about the
lane's host surface, not a defect in the harness, and it is one of the more
useful things this run will produce.

`holtwinters` is the exception among the host-list lanes: it calls the
`_traced` form with a **disabled** trace, so the context is the harness's and
no queue is drained for the instrument.

### Sizes, and the honest caveat about them

Benchmarks use the **shipped** size. `MOJOLEARN_SPEED_SIZE=smoke` shrinks only
the five lanes that have a size knob, and every line carries `size=` so a
smoke number can never be mistaken for a real one.

Five lanes are large:

| lane | shipped shape |
| --- | --- |
| `kmeans` | 4,000,000 x 32, k = 64, 20 iterations |
| `pca` | 4,000,000 x 32, 8 components |
| `ols` | 4,000,000 x 32 |
| `knn` | 400,000 x 32 index, 4,000 queries, k = 10 |
| `dbscan` | 4,000 x 16 |

The rest run the fixture their own lane ships, and several of those are
**correctness fixtures of a few dozen rows**: `gp` is 12 x 3, `gmm` is 24 x 2,
`kernel_methods` is 16 x 5, `spectral` is 48 x 4, `cholesky` is 64 x 64,
`hdbscan` is 96 x 4, `linkage` is 102 x 5.

**At those sizes the number is per-call fixed cost, not throughput.** It
answers a different question from the one the big lanes answer, and it is
still the measurement. The house rule forbids picking, dropping, deferring or
tuning a benchmark dataset by whether it flatters us; those fixture builders
take a fixture id and not an `n`, so inventing a bigger one would be inventing
a dataset. The number goes in the table with the shape beside it.

Three small-fixture lanes do have real work in them and are worth more than
the others: `resample` (10,000 bootstrap resamples of 200 rows, two million
draws), `cd` (2,048 x 16, 1,000 epochs) and `kpss` (8 series of 520
observations).

## The opponents

Every arm label says the library and the device that actually ran. An arm
labeled `cuml-gpu` **is** a cuml call.

| lane | our entry point | opponent arm | library call | true NVIDIA GPU equivalent? |
| --- | --- | --- | --- | --- |
| `kmeans` | `cluster.derived.cluster.kmeans::fit` | `cuml-gpu` | `cuml.cluster.KMeans` | **yes** |
| `dbscan` | `dbscan.derived.dbscan.dbscan::dbscan_fit_impl` | `cuml-gpu` | `cuml.cluster.DBSCAN` | **yes** (this is the code we ported) |
| `pca` | `decomposition.derived.linalg.detail.pca::pca_fit` | `cuml-gpu` | `cuml.decomposition.PCA` | **yes**, if `svd_solver="covariance_eigh"` exists on the build |
| `ols` | `glm.derived.linalg.detail.lstsq::lstsq_eig` | `cuml-gpu` | `cuml.linear_model.LinearRegression(algorithm="eig")` | **yes** |
| `knn` | `neighbors.derived...knn_brute_force::brute_force_knn_impl` | `cuml-gpu` | `cuml.neighbors.NearestNeighbors(algorithm="brute")` | **yes** |
| `cd` | `solver.derived.solver.cd::cd_fit_traced` | `cuml-gpu` | `cuml.linear_model.Lasso` | **yes** |
| `kde` | `kde.derived.kde.kde::score_samples` | `cuml-gpu` | `cuml.neighbors.KernelDensity` | **yes** |
| `linkage` | `hierarchy.derived.hierarchy.linkage::single_linkage` | `cuml-gpu` | `cuml.cluster.AgglomerativeClustering(linkage="single")` | **yes** |
| `svm` | `svm.derived.svm.svc_impl::svc_fit` | `cuml-gpu` | `cuml.svm.SVC` | **yes** |
| `metrics` | eleven `metrics.derived.metrics.*` entries | `cuml-gpu` | `cuml.metrics` + `cuml.metrics.cluster` | **yes** |
| `ivf` | `ivf.estimator::ivf_flat_build_and_search_host` | `cuvs-gpu` | `cuvs.neighbors.ivf_flat.build` + `.search` | **yes** |
| `hdbscan` | `hdbscan.derived.hdbscan.runner::fit_hdbscan` | `cuml-gpu` | `cuml.cluster.HDBSCAN` | **yes** |
| `cholesky` | `cholesky.original.potrf::potrf_lower` + `trsm::cho_solve` | `torch-gpu` | `torch.linalg.cholesky` + `torch.cholesky_solve` | **yes** — this is cuSOLVER `potrf`/`potrs`, which is what the lane was ported against. RAPIDS exposes no public Cholesky estimator. |
| `krr` | `kernel_methods.estimator::kernel_ridge_fit_host` + `_predict_host` | `cuml-gpu` | `cuml.kernel_ridge.KernelRidge` | **yes** |
| `holtwinters` | `holtwinters.estimator::holtwinters_fit_host_traced` | `cuml-gpu` | `cuml.ExponentialSmoothing` | **yes** |
| `kpss` | `tsa.derived.tsa.stationarity::kpss_test` | `cuml-gpu` **or** `statsmodels-cpu` | `cuml.tsa.stationarity.kpss_test` where the build exposes it, else `statsmodels.tsa.stattools.kpss` per series | **conditional** — the arm label says which ran |
| `gmm` | `mixture.estimator::gaussian_mixture_fit` | `sklearn-cpu` | `sklearn.mixture.GaussianMixture` | **no** |
| `gp` | `gaussian_process.estimator::gpr_fit_host` + `gpr_predict_host` | `sklearn-cpu` | `sklearn.gaussian_process.GaussianProcessRegressor` | **no** |
| `nystroem` | `kernel_methods.estimator::nystroem_fit_host` + `_transform_host` | `sklearn-cpu` | `sklearn.kernel_approximation.Nystroem` | **no** |
| `rbfsampler` | `kernel_methods.estimator::rbf_sampler_fit_host` + `_transform_host` | `sklearn-cpu` | `sklearn.kernel_approximation.RBFSampler` | **no** |
| `resample` | `resample.estimator::bootstrap_host` | `scipy-cpu` | `scipy.stats.bootstrap` | **no** |
| `spectral` | `spectral.derived.cuvs...::fit_predict_dataset` | `sklearn-cpu` | `sklearn.cluster.SpectralClustering` | **no** |

### Lanes with NO honest NVIDIA opponent, stated plainly

RAPIDS ships **no** GPU implementation of any of these, at any version this
repository has looked at. The Python arm prints an explicit
`FSPEED-REFUSED lane=<l> arm=cuml-gpu reason=RAPIDS ships no ...` line before
running the fallback, so the absence is a record in the log and not an
inference from a label:

* **`gmm`** — no `cuml.GaussianMixture`. Fallback: scikit-learn CPU, same
  `covariance_type`, `max_iter`, `tol`, `reg_covar`, `init_params`, `n_init`
  and `random_state`.
* **`gp`** — no cuML Gaussian process. Fallback: scikit-learn CPU with
  `optimizer=None` (our `gpr_fit_host` implements no hyperparameter
  optimizer, DEVIATION 1761) and the same ARD length scales and ridge.
  scikit-learn works in float64 and we work in float32; that is a real
  difference in the amount of arithmetic and it cannot be turned off on their
  side.
* **`nystroem`** — no cuML Nystroem. Fallback: scikit-learn CPU. **The basis
  sample differs**: ours permutes with a pinned Philox stream and theirs with
  numpy's `RandomState`, so the two fit different rows. The *work* is the
  same shape and that is what is timed; the hashes are not comparable and the
  arm reports `hash=-`.
* **`rbfsampler`** — no cuML RBFSampler. Fallback: scikit-learn CPU. Same
  caveat about the draws; `hash=-`.
* **`resample`** — no cuML bootstrap. Fallback: SciPy CPU, `method="percentile"`,
  same `n_resamples` and `confidence_level`. Neither arm computes a
  jackknife (`with_bca_diagnostics=False` on ours, and `percentile` does not
  compute one on theirs).
* **`spectral`** — no cuML `SpectralClustering` estimator. Fallback:
  scikit-learn CPU with `affinity="nearest_neighbors"`, the same
  `n_neighbors`, `n_clusters`, `n_components` and `n_init`. Both sides build a
  kNN affinity, take a Lanczos eigendecomposition of the normalized Laplacian
  and run k-means on the embedding; the eigensolvers are two Lanczos
  implementations (ours restarts, theirs is ARPACK). Label numbering is
  arbitrary in both, so `hash=-`.
* **`cholesky`** — RAPIDS exposes no public Cholesky estimator, but this one
  is **not** a CPU fallback: `torch.linalg.cholesky` is cuSOLVER's `potrf` on
  the GPU and `torch.cholesky_solve` is its `potrs`, which is exactly the pair
  `cholesky/original/potrf.mojo` was ported against. The arm is labeled
  `torch-gpu` because torch is what is being called.
* **`kpss`** — conditional. Where `cuml.tsa.stationarity.kpss_test` exists it
  is used and the arm is `cuml-gpu`. Where it does not, the fallback is
  `statsmodels.tsa.stattools.kpss` **one series at a time**, which means the
  CPU arm's number includes a Python call per series that our single batched
  launch does not have. Both facts are printed as `FSPEED-NOTE` lines.

### Lanes NOT covered at all

* **`arima`** — `arima/derived/arima/batched_arima::batched_loglike` is one
  batched Kalman-filter log-likelihood evaluation at given parameters.
  `cuml.tsa.ARIMA` exposes no such entry: its `fit()` is a BFGS optimization
  that calls the log-likelihood many times, so racing it against one
  evaluation would compare an optimizer to a filter pass. statsmodels'
  `SARIMAX.loglike` is one evaluation but uses a different state-space
  initialization and a different parameterization, so it computes a different
  number. **No honest opponent exists, so no arm was written.** Adding one
  would need a public single-shot batched log-likelihood on cuML's side.
* **`embedding`** — `embedding/` is the embedding-bag backward kernel, a
  neural-network gradient op with no estimator surface, no `*_main.mojo` and
  no classical counterpart. It belongs to the NN slice, not this one.
* **`decomposition`'s TSVD and Jacobi drivers**, **`neighbors`' ball-cover
  driver**, **`dbscan`'s phase driver** — sub-drivers of lanes already covered
  by their principal entry. Adding them would multiply the compile surface
  without adding an algorithm family.

## Overlap with `bench/lanes_price_main.mojo`

Six lane names appear in both files. **Four are the same call and two are
not.** Do not double-count, and do not put a number from one file beside a
number from the other:

| lane | same work as `lanes_price`? |
| --- | --- |
| `cd` | yes — same entry, same fixture, same parameters |
| `kde` | yes |
| `linkage` | yes |
| `metrics` | **no** — this lane times **eleven** metrics; `lanes_price` times **twelve**. `rand_index` is dropped here because cuML ships no plain Rand index and an arm doing one more metric than the other is not a comparison. |
| `svm` | **no** — this lane times `svc_fit` **only**; `lanes_price` times `_run_device`, which is a fit plus two predicts over `n + 37` queries. |
| `gemm` | not in this file at all; the gemm lane is not classical ML and is owned elsewhere. |

## Fairness rules this harness obeys

* Parameters that mean the same thing are **set explicitly on both sides**,
  never left to two libraries' defaults: `k`, `eps`, `min_samples`,
  `n_components`, `n_neighbors`, `alpha`, `gamma`, `C`, `kernel`, `tol`,
  `max_iter`, `n_init`, `random_state`. Where a default differed there is a
  comment at the call site saying so.
* No solver is chosen by `auto` on either side. cuML's PCA `auto` is asked for
  `covariance_eigh` by name; our route is the covariance eigendecomposition
  and if that solver is missing the arm prints a note saying the ratio is now
  algorithm plus device rather than device alone.
* No dataset is picked, dropped, deferred or tuned by whether it flatters us.
  If a lane loses badly, that is the measurement.
* `cuml.DBSCAN` runs with `calc_core_sample_indices=False`, because our
  `dbscan_fit_impl` returns labels only and making them compute an extra array
  would be unfair in our favor.
* `cuml.svm.SVC`'s `cache_size` is left at their default: a cache size is a
  memory policy, not a parameter of the answer.

## The mode witness

Every `ours` header carries `mode=`, read from the **compile-time constant**
`GLOBAL_NUMERIC_MODE`, never from an environment variable and never from the
flag that was passed. Three mislabeled measurements were caught by that
witness on 2026-08-23. **Any row of the results whose mode is not `FAST` is
not an answer to this question** and `tools/fast_speed_table.py` flags it.

## The hash, and why it is allowed to move

Every timed round prints `hash=`, the FNV-1a64 of the output bytes, byte at a
time and little endian — the same function `core/identity_trace.mojo::fnv1a64_bytes`
uses for the stage cards, so a lane's hash here is comparable with its card's
final stage where the card records that buffer whole.

**Under FAST the hash may move between rounds.** If it does, an
`FSPEED-NOTE ... hash moved across rounds: <h1> <h2>` line is printed and the
run continues. That is a *report* about a non-deterministic arm, which is what
FAST is; it is not a failure. `bench/lanes_price_main.mojo` raises on the same
event under IDENTICAL, and that is the right behavior there and the wrong
behavior here.

The hash is a **within-arm determinism probe, not a cross-arm equality check**.
Two different implementations of k-means will not produce the same bits and
are not supposed to. Where the two sides genuinely compute different objects
the Python arm prints `hash=-` rather than a number that means nothing:
`metrics` (eleven scalars of two different widths), `nystroem` and
`rbfsampler` (different random draws), `spectral` (arbitrary label numbering),
`kpss` under the statsmodels fallback (a different lag-truncation rule).

The Python side also prints `hash=-` for outputs above
`MOJOLEARN_SPEED_HASH_MAX` bytes, with a note naming the size: pure-Python FNV
costs about a microsecond a byte and hashing four megabytes of k-means labels
five times would cost more than the benchmark.

## Robustness on a rented box

* **One lane per process is mandatory** so a lane's run-time failure cannot
  take the others down.
* **It does not protect against a compile failure.** Mojo compiles every
  import in `classical_speed_main.mojo` whether the selected lane uses it or
  not, and most of these lanes have never been compiled for CUDA at all. The
  per-lane import blocks in that file are separated by banners for exactly
  this reason: if the build fails inside one lane, comment out that lane's
  import block, its `run_*` function and its `elif` arm, rebuild, and the
  other twenty-one still run. **Record the amputation as a refused lane in the
  results rather than dropping it silently.**
* Every opponent import is inside a `try` and every lane body is inside a
  `try`; a failure prints `FSPEED-REFUSED` and exits zero.
* Neither side hangs unbounded. Both respect `MOJOLEARN_SPEED_ROUNDS`, and the
  shipped sizes finish well under a minute per lane.

## Deviations recorded by this lane

* **DEVIATION 1810.** `bench/bench_sklearn.py:163-166` builds the k-means
  initial centroids as `u01(c * 7919 + 1, km_cols, 5)[0]`, which indexes row
  **zero** of that block for every `c` and therefore hands scikit-learn 64
  **identical** centroids. `bench/bench_main.mojo:106` seeds centroid `c` from
  `_u01(c * 7919, f, 5)`, which is row `c * 7919`. The two sides of the
  existing k-means comparison have not been running on the same initialization.
  `tools/speed_cuml_arm.py` does **not** reproduce the defect: it uses
  `u01_at(np.arange(k) * 7919, cols, 5)`, a row-indexed variant of the same
  recurrence that is asserted against the imported `u01` at startup on every
  run. **The fix to `bench/bench_sklearn.py` is not this lane's to make** and
  is owed; every k-means ratio in `bench/results/` taken against that file
  should be re-read in this light.
