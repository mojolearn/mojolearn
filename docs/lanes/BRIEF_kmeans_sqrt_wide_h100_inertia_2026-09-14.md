# kmeans-sqrt/wide: the H100 inertia stands alone (2026-09-14 night)

Found by the 166-lane record at 1eea14f80
(`bench/results/identity_break/2026-09-14_166-lanes/diff.three-columns.txt`), the
first record that carries the `kmeans-sqrt` lane on any GPU. One cell of 1494
training cells is DIVERGENT. Nothing here is fixed; no kernel was touched.

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

Until then the CPU identity gate's GPU-column step asserts this record as it is:
`summary: DIVERGENT=1, IDENTICAL=1493` with this one row, so a fix or a second
divergence changes the count and fails the step.
