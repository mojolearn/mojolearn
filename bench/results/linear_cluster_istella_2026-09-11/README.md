# linear-cluster-istella lane, 2026-09-11 (DEVIATIONS 2671, 2672)

Istella-S stage breakdown for OLS and PCA, two bit-preserving changes, and the
first DBSCAN rows against cuML. The before tree is `origin/main` (8dc33f00,
which already carries DEVIATIONS 2632 and 2633 from lane linear-cluster-speed);
the after tree is this lane. Both are built ON THE SAME POD and raced
interleaved, round by round, through `tools/classical_two_datasets.py race`
with the `ours-base` arm pointed at the before tree's `python/`
(`MOJOLEARN_CTD_BASE_PY`).

## The box

RunPod pod `1yxsotvvcbxtuu`, NVIDIA H100 80GB HBM3, driver 570.195.03, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, cuML 26.08
(cuml-cu12 26.8.0) in the image's Python 3.11, ours IDENTICAL built on the pod
with `MOJOLEARN_GPU_ARCHS=sm_90a`.

Two boxes before it were reaped unused and are named here because a rented box
that cannot fetch a dataset is a finding, not a gap: H200 `lle7doq4sqx0my`
pulled Istella at 31 kB/s and pip at 115 kB/s (hours per leg), and H100
`gvbbi4tmwi7m0r` was rejected by a qualify probe of mine that used a guessed
wheel URL and measured an error page rather than the network. Every box since
is qualified against two real files (the Istella tarball and a taxi month) and
reaped at once if it cannot serve.

## The changes

**DEVIATION 2671, the device Jacobi's phases** (`decomposition/checks/
jacobi_eigh_device.mojo`). A rotation used to close four phases with a barrier
each, the `(c, s)` pick, the column update, the row update and the eigenvector
update. At Istella-S's 220 columns that is 24,090 rotations and 96,360
barriers a sweep, and OLS runs 12 sweeps there. The column and row updates
share a cell only in the 2 x 2 block `{p, q} x {p, q}` and the eigenvector
update shares none, so the last three phases are now one: each lane does its
own column pair then its own row pair, the lane owning `k == p` does the 2 x 2
block in the old column-then-row order, and every lane does its eigenvector
pair. Two barriers per rotation instead of four. Rotation order, arithmetic
spelling, both folds and the sweep test are untouched. The four-phase kernel
stays in the file as `jacobi_eigh_kernel_four_phase` so
`check_jacobi_merged_phases_equal_four_phase` can hold the two equal at every
output cell and all three info slots.

**DEVIATION 2672, k-means host staging** (`cluster/estimator.mojo`,
`python/mojolearn/cluster.py`). The fit built its weight vector on the host (a
pinned `n_samples` buffer filled with 1.0 one row at a time on one thread) and
read its results back through two more pinned buffers copied value by value
into the caller's arrays; the Python surface then copied the label vector
twice more to change its dtype. Unit weights are now a device fill of the same
exact 1.0, supplied weights upload straight from the caller's pointer,
centroids and labels are copied from the device into the caller's memory, and
`labels_` is allocated as the int32 array the kernel writes.

Neither change moves a bit by construction: no arithmetic, no order and no
kernel geometry changes in either.

## Istella-S stage breakdown (the question this lane was opened on)

`stage_probe`, IDENTICAL, 3 reps, Istella-S 2,043,304 x 220, warm. The caller's
block is a pinned host buffer here, so the upload row is a pinned upload and
the public fit's pageable upload can only be slower.

| stage | OLS (ms) | PCA (ms) |
|---|---|---|
| device allocation | 0.006 | 0.006 to 2.4 |
| upload of the 1.8 GB design | 36.0 | 34.9 |
| Gram (`gemm_tn`) / covariance | 23.6 | 30.0 |
| `A^T b` | 2.7 | - |
| equilibration (DEVIATION 2620) | 0.06 | - |
| **device Jacobi, 220 columns** | **1656.5 (12 sweeps)** | **418.3 (3 sweeps)** |
| `ols_fit_host` total | 1744 to 1775 | - |

**The Jacobi is 94 percent of the OLS solve and 86 percent of the PCA fit, and
the GEMM is 23.6 ms of it.** The v1 GEMM profile past 128 columns is not what
costs anything here, so there is nothing in this lane for the neural session's
`gemm/` to take; the 220-column eigensolver is the whole cost.

Same matrix, both kernels, per rep:

| fit | merged, DEVIATION 2671 | four-phase | after/before | cells differing |
|---|---|---|---|---|
| OLS Gram, n = 220, 12 sweeps | 1656.5 ms | 1825.8 ms | 0.907 | 0 of 48,400 matrix, 0 of 48,400 eigenvector |
| PCA covariance, n = 220, 3 sweeps | 418.3 ms | 466.6 ms | 0.896 | 0 of 48,400 and 0 of 48,400 |
| taxi, n = 11, 4 to 5 sweeps | 0.449 to 0.566 ms | 0.401 to 0.559 ms | 1.06 to 1.12 | 0 of 121 and 0 of 121 |

At 11 columns the merged kernel is a shade slower (a rotation there is two
lanes of work, and one barrier saved does not pay for the branch), and the
whole Jacobi is half a millisecond of a 10 ms solve, so the taxi fit does not
notice either way. At 220 columns it removes about 169 ms from every OLS fit.

k-means host stages, same probe:

| stage | taxi 4,000,000 x 11 | Istella-S 2,043,304 x 220 |
|---|---|---|
| `plan_sum_scale` (host pool, DEVIATION 2633) | 28.2 to 33.1 ms | 189.2 to 220.5 ms |
| upload of X | 3.9 | 34.1 |
| weight fill (now a device fill, 2672) | 2.0 | 1.0 |
| row norms | 2.4 | 1.2 |
| `fit_predict` (21 iterations) | 121.8 to 122.6 | 823.1 to 826.9 |
| readback of centroids and labels | 0.3 to 0.6 | 0.2 to 0.3 |
| public fit | 156.4 to 159.4 | 1081 to 1146 |

## Before and after, same pod, interleaved (1 warm-up plus 5 rounds)

`ours` is this lane, `ours-base` is origin/main (8dc33f00) built on the same
pod, `cuml-gpu` is cuML 26.08. Ratios against `ours-base` are ours against
ours and are never quoted as an opponent row; the opponent column is
`ours / cuML`.

| family | dataset | cuML ms | before ms | after ms | after/before | ours/cuML after | quality after = before |
|---|---|---|---|---|---|---|---|
| LinearRegression | taxi 4,000,000 x 11 | 21.67 | 139.65 | 135.76 | 0.972 | 6.26x | R2 0.90883698 both (cuML 0.90883616) |
| LinearRegression | Istella-S 2,043,304 x 220 | 84.85 | 2529.06 | 2402.44 | 0.950 | 28.31x | R2 0.33194438 both (cuML -6473.68) |
| PCA | taxi | 19.54 | 28.46 | 26.32 | 0.925 | 1.35x | EVR sum 0.99786071 both (cuML 0.99786046) |
| PCA | Istella-S | 81.91 | 713.09 | 686.27 | 0.962 | 8.38x | EVR sum 1.00000001 both (cuML the same) |
| KMeans | taxi | 129.28 | 216.59 | 211.99 | 0.979 | 1.64x | inertia 1.2062766e8 both (cuML 1.2019161e8 at 20 iterations against our 21) |
| KMeans | Istella-S | 171.81 | see below | see below | see below | 7.56x | inertia 1.3128483e17 both (cuML 1.2855462e17) |

Every cell held ONE digest across its five rounds, and the after digest equals
the before digest in every cell: OLS taxi `fb86358654367fa0`, k-means taxi
`89520efe99a08d5f`, PCA taxi `c790338770a4c120`, k-means Istella-S
`7f720b0b76896308`, and OLS and PCA Istella-S likewise stable and equal.
cuML's k-means returned a different centroid digest in every round on both
datasets; ours held one. **cuML's OLS is wrong on Istella-S**: its `eig`
solver returns R2 -6473.68 where ours returns 0.331944, which is the ill
conditioning DEVIATION 2620's equilibration exists for, so its 84.85 ms is not
a time for the same answer.

### The verdicts (ENGINEERING_RULES section 9, geometric mean over the two)

* **LinearRegression: 0.972 and 0.950, geomean 0.9610, FLIP.** Quality equal
  on both datasets, bits equal on both.
* **PCA: 0.925 and 0.962, geomean 0.9435, FLIP.** Same.
* **KMeans: pending the pooled instances below.**

Both of those are DEVIATION 2671 alone: OLS and PCA reach the Jacobi and
k-means does not, so the k-means row isolates DEVIATION 2672 and the OLS and
PCA rows isolate 2671.

### KMeans and DEVIATION 2672: the effect is under this shape's noise

The Istella-S k-means A/B swung between race instances, in both directions:

| instance | before ms | after ms | after/before |
|---|---|---|---|
| first race | 1219.0 | 1299.4 | 1.066 |
| repeat | 1279.6 | 1215.5 | 0.950 |

The rounds inside each instance were tight (after 1267 to 1346, before 1217 to
1267 in the first; after 1211 to 1234, before 1231 to 1378 in the repeat), so
the spread is BETWEEN race instances, not within them, and it is about 80 ms
against a change the probe measures at about 5 ms of host work at this shape
(weight fill 1.0 ms, readback 0.2 to 0.3 ms). Pooled instances and the verdict
they give are below.

## DBSCAN and HDBSCAN rows (measurement only, first at this size)

1 warm-up plus 3 rounds, eps the p75 quantile of the min_samples-th nearest
neighbor distance ON THE BLOCK (the same rule and the same value for every
arm), min_samples 10, blocks standardized.

| dataset | shape | eps | cuML brute ms | ours (ball cover) ms | ours / cuML | agreement |
|---|---|---|---|---|---|---|
| taxi | 1,000,000 x 11 | 0.177 | 13631.0 | 1129.8 | 0.083x | both 2900 clusters, both noise fraction 0.216119, adjusted Rand index 0.99999999908, noise agreement 1.000 |
| Istella-S | 1,000,000 x 220 | 4.17 | pending | pending (round 0 was 337.1 s against cuML's 55.2 s) | | |

| dataset | shape | cuML HDBSCAN ms | ours | notes |
|---|---|---|---|---|
| taxi | 100,000 x 11 | 191.2 | no arm | 159 clusters, noise fraction 0.1310; min_samples 10, min_cluster_size 100, eom. **This library ships no HDBSCAN**, so there is no ratio, only cuML's row. |

`cuml-gpu-rbc` (cuML's own ball cover) refuses both datasets at 1,000,000 rows
with "An overflow occurred with the current choice of precision and the number
of samples", so cuML's rbc has no row.

**The two datasets say opposite things about our ball cover, and that is the
finding.** On the narrow table it is an order of magnitude below cuML's brute
force; on 220 features its pruning stops paying and cuML's brute force wins,
which is what a landmark bound does as dimension grows. Anyone quoting the
taxi row alone would be quoting the fixture.

## DBSCAN and HDBSCAN

New lanes in `tools/classical_two_datasets.py`, measurement only. The block is
1,000,000 rows of each dataset's train split, sentinel cleaned and standardized
by its own float64 mean and standard deviation, so one eps means the same thing
on every column. eps and min_samples come from
`MOJOLEARN_CTD_DBSCAN_<DATASET>` and are the SAME for every arm; the rule that
picked them is in `dbscan_eps.py` (a quantile of the min_samples-th nearest
neighbor distance measured on the block itself, with each candidate's cluster
count and noise share shown on a 200,000-row subsample first). Ours runs its
default ball cover, `cuml-gpu` runs cuML's default brute force, and
`cuml-gpu-rbc` asks cuML for its ball cover. HDBSCAN has a cuML arm only,
because this library ships no HDBSCAN.

## RUN OWED

DEVIATION 2671 changes a GPU kernel's phase structure, so it is the one that
needs other vendors even though it cannot move a bit by construction. Every
command below is IDENTICAL and is run from a checkout of this branch.

1. **Apple M4 (local, one deliberate run, nothing else heavy running).**

       tools/with_identical_mode.sh pixi run mojo run -I . decomposition/checks/jacobi_check.mojo
       tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo
       tools/with_identical_mode.sh pixi run check-kmeans-identity
       tools/with_identical_mode.sh pixi run check-kmeans
       tools/with_identical_mode.sh pixi run mojo run -I . decomposition/pca_main.mojo

   `check_jacobi_merged_phases_equal_four_phase` is the gate that matters: it
   holds the two-barrier kernel equal to the four-barrier one at every output
   cell on THAT box. The eigenvector sign convention and the sweep counts
   printed there are the cross-vendor witnesses.

2. **AMD (Hot Aisle MI300X, `tools/hotaisle_leg.sh`, gfx942).** The same five
   commands, plus the k-means and OLS digests from the taxi race so the
   centroid, label and coefficient hashes can be compared against the H100
   values in this file:

       MOJOLEARN_GPU_ARCHS=gfx942 sh bindings/build.sh
       MOJOLEARN_GPU_ARCHS=gfx942 sh bindings/build_estimators.sh
       python3 tools/classical_two_datasets.py race --lane kmeans --dataset taxi \
         --data $DATA --out $OUT --work $WORK --root $PWD --rounds 5 \
         --arms ours,torch-gpu --ours-python python3 --theirs-python python3

3. **The vendor scheduling row.** `K_LIB_JACOBI_EIGH` keeps its pinned block
   size of 32 in every column; 2671 changes no geometry, so no new row in
   `checks/kernel_matrix.mojo` is needed. If a vendor ever wants a different
   phase structure, that is where it would go, and the equality gate is what
   would have to be reproven.

## Files

* `stage_probe_main.mojo.txt` -- the stage probe (OLS, PCA and k-means stages,
  and the Jacobi A/B with bit counts). Build with
  `pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1`.
* `probe_bins.py` -- writes the probe's raw blocks through the shipped Python
  layer's own centering, so the probe's OLS input is the bytes the public fit
  uploads.
* `dbscan_eps.py` -- the eps rule and its candidates.
* `ab.sh` -- the race driver used on the pod.
