# GaussianMixture: row-sharded E-step

Design note for `mojolearn.parallel_classical.fit_gaussian_mixture` and
`predict_gaussian_mixture`. The contract is the same as every other driver in
this directory: IDENTICAL mode only, and a multi-device result must have the
exact bits of the one-device IDENTICAL result.

## What the estimator exposes

`GaussianMixture` implements `covariance_type='full'` with
`init_params='kmeans'` or `'random'`. `'tied'`, `'diag'`, `'spherical'`,
`'k-means++'`, `'random_from_data'`, `n_init > 1`, `warm_start` and the three
`*_init` arrays are refused by name by the estimator (see
`mixture/estimator.mojo` and `python/mojolearn/mixture.py`). The distributed
entries do not widen that surface: every refusal fires unchanged inside the
worker. There is no covariance type to add a partition for.

## Where the arithmetic reduces across rows

One EM iteration on one device is:

1. E-step, per component k: `y = X . P_k` (GEMM, `n x d` by `d x d`), `mu_k . P_k`,
   and the Mahalanobis fold `sum_j (y[i][j] - murow[j])^2`, one thread per row.
2. Weighted log probability, one cell. Logsumexp with a positional row max,
   one thread per row. Log responsibility, one cell.
3. Mean log likelihood: one thread, ascending over all `n` rows, one division.
   This is the convergence quantity.
4. M-step: `exp(log_resp)`, `nk` (column folds over rows), means and
   covariances (`OP_TN` GEMMs whose cells contract over all `n` rows), then the
   precision Cholesky per component.

Steps 1 and 2 are pure functions of one sample row and the parameters. The
GEMM fp32.v1 cell contract contracts output cell `(i, j)` over row `i` of `X`
and column `j` of `P_k` only, so a row's `y` does not depend on which rows
share its launch. Steps 3 and 4 fold across rows. Their fold trees are fixed
functions of `n` on one device.

## Partition

Steps 1 and 2 run on contiguous row ranges: rank r owns rows
`[n*r/active, n*(r+1)/active)`. Each owner receives its rows and copies of the
small parameter buffers, runs the original `gmm_e_step` with its own row
count, and its five per-row outputs (`mahal`, `wlp`, `rowmax`, `lse`,
`logresp`) are copied as bytes into the original row positions on the root.
No floating-point value from one owner is combined with a value from another.

Step 3 runs on the root with the original `meanll_kernel` over the complete
gathered `lse`. Step 4, the Cholesky, the collapse refusal and the convergence
test run on the root unchanged. The KMeans initialization runs its existing
row-tile assignment driver in the same worker
(`MOJOLEARN_KMEANS_DEVICE_COUNT`), which carries its own identity gate.

A distributed M-step would have to reproduce the single-device fold trees of
`nk`, the means GEMM and the covariance GEMM. The covariance cells can be
assigned whole to owners the way `core/gram_multi_gpu.mojo` assigns Gram
output cells, but every owner then needs every row of `X` and of the
responsibilities, which is compute partitioning with replicated data. It is
not implemented here; the root holds the full data and responsibilities.

## Tracing and refusals

When a fit is traced, the gathered stages are recorded on the root in the
original stage order, so the identity card of a multi-device fit is the same
file as the one-device card. Sabotage probes are refused by the distributed
path, and so are builds with process-global GEMM phase counters. The native
gate has a check-only build, `-D MOJOLEARN_GMM_PARALLEL_SABOTAGE=1`, that
makes every later owner read its rows one row early; the gate must fail
under it.

## Gates

- `training/checks/gmm_parallel_check.mojo`: every E-step output bit across
  ragged row counts (2 to 4097 rows, 1 to 64 features, 1 to 5 components);
  full fits of all six mixture fixtures and two larger blob sets with both
  init modes, requiring equal identity traces, fitted state, `n_iter`,
  `converged`, `lower_bound` and scoring outputs; equal collapse refusals.
- `tools/parallel_gmm_check.py`: public fits and predictions against one
  device, failed-fit publication and refusals.

No speed or memory-capacity claim is made. The root still needs the full
data, responsibilities and M-step workspace.
