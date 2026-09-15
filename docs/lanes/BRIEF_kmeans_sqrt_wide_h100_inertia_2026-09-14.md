# kmeans-sqrt/wide: the H100 inertia stands alone (2026-09-14 night) -- CLOSED at 9fde8f5f7

Found by the 166-lane record at 1eea14f80
(`bench/results/identity_break/2026-09-14_166-lanes/diff.three-columns.txt`), the
first record that carries the `kmeans-sqrt` lane on any GPU. One cell of 1494
training cells is DIVERGENT. Nothing here is fixed; no kernel was touched.

## CLOSED (2026-09-14 night, lane/kmeans-sqrt-inertia, commit 9fde8f5f7)

Evidence: `bench/results/identity_break/2026-09-14_kmeans-sqrt-fix/README.md`.

1. **The bits.** Wide fixture, `np.float64(m.inertia_).tobytes().hex()`, each box
   building main e2d770ba8's k-means sources (unfixed) and the fix:
   H100 sm_90a and RTX 4090 sm_89 unfixed `000000e0e4f28d41` = 62807196.0;
   MI325X and the M4 unfixed `00000000e5f28d41` = 62807200.0; all four fixed
   `00000000e5f28d41`. One float32 ulp, and 1a7e4ac5b8c0caaf / 52ea06cbbcc24144
   are exactly those two float64 values.
2. **The second NVIDIA architecture.** The RTX 4090 carries the H100's value, so
   the column that stood alone was NVIDIA, not the H100.
3. **The step.** The reduced L2SqrtExpanded distance was rooted with the stdlib
   `sqrt` at the final write of both min-reduce kernels
   (`cluster/impl/distance/fused_distance_nn/simt_kernel.mojo`,
   `cluster/impl/distance/unfused_distance_nn.mojo`), which on NVIDIA is the
   approximate PTX square root (DEVIATION 258); the fit sums those roots into
   `inertia_`. Not a reduction order or a contraction. Now `identical_sqrt`
   (DEVIATION 2715). The L2Expanded inertia never takes a root, which is why that
   arm was IDENTICAL.

**A SECOND DEFECT THIS BRIEF'S "labels agree" HID.** The labels agreed on every
column and were wrong on every column: `kmeans_fit` passed rooted row norms to
`fit_predict`'s final assignment under L2SqrtExpanded, so it computed
`||x|| + ||c||^2 - 2 x.c`, clamped most rows to 0 and gave them the lowest key.
9675 of 20000 `wide` labels and 4 of 20000 `base` labels were not the argmin to
the returned centers. cuVS takes squared norms for both L2 metrics; so does the
fix (DEVIATION 2716, the estimator and the host oracle), and
`python/mojolearn/tests/test_kmeans_metric_surface.py` now checks labels against
the argmin (it failed on the 1eea14f80 binding, 336 and 61 of 512).

After both fixes the eight kmeans lanes read `summary: IDENTICAL=72` on
apple-m4, nvidia-h100-sm_90a and amd-mi325x-gfx942 (IDENTICAL x4 with the
4090), and against the 166-lane record each column moves exactly the nine
`kmeans-sqrt` label cells (plus the NVIDIA `wide` inertia). The 166-lane
record's JSONs keep their 1eea14f80 bytes, so the GPU-column step still asserts
`DIVERGENT=1, IDENTICAL=1493` about them; the CPU gate's kmeans-sqrt fix record
step asserts the fixed columns IDENTICAL and the unfixed columns DIVERGENT on
this one row.

The text below is the brief as it was opened.

## The cell, every column

`kmeans-sqrt` is `KMeans(n_clusters=8, random_state=3, metric="l2_sqrt_expanded").fit(X)`
on the `wide` fixture (column magnitudes 1e-4 to 1e4). Its parts, both repeats
of each column (every column STABLE):

| column | centers | labels | scales | inertia |
|---|---|---|---|---|
| apple-m4 | b20229152b8003ca | cf7cc99cd1e1e019 | 600b4305f9020d70 | 52ea06cbbcc24144 |
| nvidia-h100-sm_90a | b20229152b8003ca | cf7cc99cd1e1e019 | 600b4305f9020d70 | **1a7e4ac5b8c0caaf** |
| amd-mi325x-gfx942 | b20229152b8003ca | cf7cc99cd1e1e019 | 600b4305f9020d70 | 52ea06cbbcc24144 |

The lane is not in any earlier record on main (`git grep -l kmeans-sqrt
bench/results/identity_break` finds only this record), so there is no older
column to print. The NVIDIA column stands alone: Apple and AMD agree with each
other. The centers, the labels and the two scales agree on all three, so the fit
reached the same solution; only the reported `inertia_` (the float64 host read of
the sqrt-metric objective, `metric_is_sqrt` in cluster/estimator.mojo:336) differs.
The other eight fixtures of the lane are IDENTICAL x3, including `base`, `ties`,
`hashed` and `odd`.

## Owed, in order

1. Print the inertia BITS, not the hash, on the H100 and one other column, for
   this fixture (a tiny body: the lane's fit, then `np.float64(m.inertia_).tobytes().hex()`).
   A one-ulp difference points at the final sqrt fold or its reduction order on
   sm_90a; a large one points at a different sum over the assignment distances.
2. Rerun the lane alone on a second NVIDIA architecture (RTX 4090 sm_89 or RTX
   5090 sm_120a) to learn whether the column that stands alone is the H100 or NVIDIA.
3. Read the sqrt-metric inertia path (cluster/estimator.mojo around line 336 and
   cluster/impl/detail/kmeans_common.mojo) against cuML's `L2SqrtExpanded`
   inertia for an accumulation whose order depends on the device.

Until then (and still, since the record's bytes are unchanged) the CPU identity gate's GPU-column step asserts this record as it is:
`summary: DIVERGENT=1, IDENTICAL=1493` with this one row, so a fix or a second
divergence changes the count and fails the step.
