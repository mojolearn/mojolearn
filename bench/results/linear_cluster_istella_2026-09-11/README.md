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

## Measurements

Filled from the pod's races and probes; see `summary.tsv`, the `probe-*.log`
files and the tables below.

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
