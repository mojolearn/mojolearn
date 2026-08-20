# The first window where the PCA arm is not a mix of two algorithms

Every PCA ratio recorded before this one is contaminated, and the harness said
so in a column nobody read. From `WINDOW_2026-08-20_post-lane.md`:

    | pca | 75.80 | 147.04 | 1.94x | ours faster | ... | 15/30 |

**n = 15/30.** Ours 15 samples, scikit-learn 30, where every other arm is
15/15. `bench_sklearn.py` emitted `ARM pca` TWICE per repeat, once for
`svd_solver="auto"` and once for `svd_solver="covariance_eigh"`, and
`run_bench.py:35` pools by name and takes a median. The reported scikit-learn
PCA time was therefore a median across a bimodal mix of two different
algorithms. Its own docstring said the second arm should be named
`pca_cov_eigh`; the code did not do it.

Fixed, and this is the first run after. `pca` is 15/15 and `pca_cov_eigh` is
its own arm.

## The board, 3 rounds, arms alternated per round, 15 samples per side

    arm             ours ms    sklearn ms    speedup   verdict
    kmeans           758.58       2192.59      2.89x   ours faster
    knn              701.45       1072.30      1.53x   ours faster
    pca               37.51        136.55      3.64x   ours faster
    ols               34.68        878.32     25.32x   ours faster
    dbscan             9.90          9.24      0.93x   INDISTINGUISHABLE

    ours [min, max]              sklearn [min, max]
    kmeans  [749.40, 1243.53]    [1898.36, 3758.45]
    knn     [693.05,  725.72]    [ 900.93, 1673.61]
    pca     [ 36.45,   41.30]    [ 122.65,  410.66]
    ols     [ 33.48,   46.76]    [ 800.52, 1577.10]
    dbscan  [  8.22,   16.60]    [   8.46,   31.44]

One-sided arms, which scikit-learn emits and we have no matching arm name for.
Their ratio against our single arm is computed here rather than left out:

    ols_normal_eq   150.23 ms   ->  4.33x  vs our 34.68   ALGORITHM-MATCHED
    pca_cov_eigh    126.24 ms   ->  3.37x  vs our 37.51   ALGORITHM-MATCHED
    dbscan_brute      8.97 ms   ->  0.90x  vs our  9.90   INDISTINGUISHABLE

## Which number is the honest one, per arm

**PCA: 3.64x against their default, 3.37x against the matched algorithm.** Both
are defensible and they are close together, which is the useful part: their
`auto` resolves to `covariance_eigh` at this shape, so the two arms are the
same algorithm and the gap between 136.55 and 126.24 is inside the noise. PCA
needs no caveat that DBSCAN and OLS need.

**OLS: 25.32x against their default, 4.33x against the matched algorithm.**
The 25.32x is `LinearRegression`, LAPACK `gelsd`, an SVD of the full
4,000,000 x 32 design. Ours forms a 32 x 32 Gram and eigendecomposes it. That
is a true statement about what a user who calls scikit-learn's OLS gets, and
it is NOT a hardware result. **The hardware result is 4.33x.** Quote both or
quote 4.33x; never quote 25.32x as evidence about kernels.

And the accuracy side, which a timing harness cannot see and which belongs
beside the OLS number every time: `bench/ols_conditioning.py` prices it. In
float32, which is all Metal has, the normal-equations route is 32x less
accurate than `gelsd` at condition number 10 and 834x at 100, and past 1e3
float32 cannot hold the problem at all.

**DBSCAN: no finding.** 0.93x with overlapping ranges.

## Why our side moved so far since 11:24 this morning

    arm       morning     now      what landed
    kmeans     1657.90   758.58    ee05664, d62ab0e, 62b6a65, 65327c5
    ols          62.61    34.68    50451a9, 2c2b46c, a4a7ed0, be226f8
    pca          75.80    37.51    50451a9, 10ae918
    knn         756.00   701.45    (no lane; inside noise)

Not drift and not thermal. PCA and OLS share the Gram shape, and `50451a9`
(register-tile cell ownership for the split-K Gram, raising the accumulation
loop's FMA/shared-read ratio) plus `10ae918` (PCA fuses mean-centering into
the split-K read) are both on that path. k-means took four separate lanes.

## What this run does NOT include

The working tree had uncommitted changes when `build/bench_main` was built.
All of them are on the CatBoost tree path -- `ported/gpu_data/`,
`ported/methods/`, `ported/targets/`, `ported/models/`,
`mojo_only/bootstrap_check.mojo`, `probe_main.mojo` -- and another session is
writing them. None of the five arms above reaches any of those files: they
live in `cluster/`, `neighbors/`, `decomposition/`, `dbscan/`, `glm/` and
`core/`. Stated rather than assumed, because a benchmark built from a dirty
tree is worth nothing if nobody checked which parts were dirty.

The k-NN arm still does not go through `knn_search`. It calls the kernels
directly, so the insertion sort that fixes the cross-arm ordering bug is not
in the 701.45 ms. That sort is `n_queries * k^2` host comparisons, about
400,000 here. Small against 701 ms, and still unmeasured.

## Environment

```
recorded_at=2026-08-20T13:55:27Z
model=Mac16,12   chip=Apple M4   cores=10   memory_bytes=17179869184
macos=26.5.2     power=Now drawing from 'AC Power'
thermal=no warning level recorded
python=3.14.6    numpy=2.4.4
```

Reproduce:

    pixi run mojo build -I . bench/bench_main.mojo -o build/bench_main
    pixi run -e bench python bench/run_bench.py --mojo-bin build/bench_main --rounds 3
