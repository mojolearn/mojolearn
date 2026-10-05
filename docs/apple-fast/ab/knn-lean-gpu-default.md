# KNN imputer GPU count default preparation

Base1f33e9764; only the three product-file hunks of measured candidate
f9c5887a8ff3b5fe8878e0bfe3d5f553a7f741f7 are imported. No old small-lane
candidates or host lean reduction are imported. Current-main KPCA default
and scope changes in the same Python file are retained.

Source compatibility: bindings/_mojolearn_x_neighbors.mojo,
x_neighbors/nan_cells_device.mojo, core/device_pool.mojo and
x_neighbors/device_ops.mojo match the measured candidate's baseline.
Only unrelated KPCA hunks differ in _expansion_neighbors.py. Applying the
candidate's KNN hunk preserves those changes.

M3 w2-knn-lean-gpu-taxi-r1 A2.2 ->B0.8ms, one scored run per arm.
Repaired harness a8e668f4c, fixtureknn-lean-gpu-v2: 31 arrays byte-exact,
including transforms, fitted flags, column counts, total, public empty-fit
refusal, native empty count, wide all-NaN/all-present and pool reuse.
No opponent replay. The default branch does not add scored measurements.

GPU proof: each thread counts up to64 rows of one column. Grid size grows
with ceil(n/64)*d; integer atomics accumulate both per-column and total
counts on GPU. Buffers are zeroed on GPU, counts downloaded, synchronized,
then returned to the existing pool. The Python change is only selecting
a one-slot unused cell buffer from the binding's compiled capability.
No host count arithmetic is introduced. Transform still requests and
receives the full cell list. Output buffer ownership and readiness remain.

Default guard inherits NC_COLMISS_ONLY's FAST Apple scope. Rollback is
MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU_OFF; disabling the parent colmiss-only
optimization also disables this path and its Python one-slot allocation.

Manager: promote_build.sh lane/apple-fast-knn-lean-gpu-default x_neighbors
MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU_OFF (Adefault, B_OFF). Then M3 quality:
`python3 tools/knn_lean_gpu_pair.py quality SOURCE_SHA
knn-lean-gpu-default-quality-20261004 knn-imputer`.
Quality helper checks reachesA1/B0 and the same31 exact arrays. No timing
action is accepted. Pending manager compile/quality; no merge performed.
