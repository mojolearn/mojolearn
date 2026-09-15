# SpectralClustering.predict: design and blockers (2026-09-15)

Lane `lane/inference-transductive-predict`. Stages 1 and 2 (DBSCAN and
AgglomerativeClustering `predict`, DEVIATION 2740) shipped. Stage 3 did not.
`SpectralClustering.predict` still raises `NotImplementedError`. This note
records the design that would make it identical on Apple, NVIDIA, AMD and the
CPU host binding, and why it was not shipped half done.

## What the fit computes today (read at merge 21c244b51)

`spectral/impl/cluster/detail/spectral.mojo::fit_predict_dataset`:

1. `create_connectivity_graph`: identical k-NN of the data against itself
   (`knn_search`, L2SqrtExpanded, `n_neighbors` including the row itself),
   edges `(i, j, 1.0)`, then `coo_symmetrize` (`0.5 * (a + b)`), sort, and
   drop zeros. So `W_ij` is 1.0, 0.5 or 0.
2. `create_laplacian`: the normalized Laplacian `I - D^-1/2 W D^-1/2`, negated,
   with `diagonal` (the degree scaling) written to a device buffer.
3. `compute_eigenpairs`: thick-restart Lanczos (`which = LA`) gives the Ritz
   values and unit eigenvectors, then `divide_rows_kernel` divides by
   `diagonal` and a reversed gather produces `embedding_`.
4. `kmeans_fit_predict` on `embedding_` (k-means++, `n_init`, the fit's seed).
   The `centroids` buffer is freed at the end of `fit_predict_graph`.

The binding returns `labels_` and `embedding_` only.

## The design

A new row `q` gets a label in three steps. All of them are float32 on the
identical primitives, so the whole path can be made identical.

1. **Affinity.** Take the identical k-NN of `q` against the training rows
   (`knn_search`, the same arithmetic and `(distance, index)` order as the fit)
   at `k = n_neighbors`. Set `a_qi = 0.5` for each neighbor. That is the value
   a one-directional edge takes after `coo_symmetrize`, since a new row has no
   reverse edges. The degree is `d_q = sum_i a_qi`, a fixed-order fold over k
   slots.
2. **Nystrom extension.** For each embedding column `c` with Ritz value
   `theta_c` of the negated Laplacian, the normalized adjacency eigenvalue is
   `lambda_c = 1 + theta_c`. Then
   `u_c(q) = identical_div(sum_i identical_div(a_qi, sqrt(d_q) * diag_i) * u_c(i), lambda_c)`
   with the sum folded over the k neighbor slots in slot order, and
   `embedding_c(q) = identical_div(u_c(q), sqrt(d_q))`, which matches step 3
   of the fit.
3. **Assignment.** Run the fit's own final k-means assignment
   (`row_norm_kernel`, then `cluster/impl/kmeans.mojo::predict`, as
   `KMeans.predict` does) against the fit's centroids. Ties go to the lowest
   centroid index, which is k-means' rule.

The precomputed arm would take a caller-given `(n_queries, n_train)` affinity
in place of step 1.

## Why it is not shipped: the blockers

1. **The fit does not keep what predict needs.** Predict needs the Ritz
   values, the unit eigenvectors before the division, `diagonal` and the
   k-means centroids. None of them leaves `fit_predict_graph`. Two changes are
   needed:
   - Four outputs on the GPU path: `spectral/impl/cluster/detail/spectral.mojo`,
     `spectral/impl/preprocessing/detail/spectral_embedding.mojo`,
     `spectral/estimator.mojo` and `bindings/_mojolearn_metrics.mojo`.
   - The same four on the host restatement: `spectral/host/spectral_oracle.mojo`
     and `bindings/_mojolearn_metrics_host.mojo`.

   Each must be proven not to move `spectral`, `spectral-precomputed` or
   `par-graph-spectral` train cells on three committed columns. Recomputing
   `u(i) = embedding(i) * diag_i` in place of storing it is not exact, because
   a multiply does not undo a division.
2. **The affinity rule for a new row is an invented choice, not a derived
   one.** The fit's graph is symmetric and its degrees count reverse edges. A
   new row has none, so `0.5` per neighbor versus `1.0`, and whether the
   training degrees should change, are rules this library would have to
   declare as a DEVIATION. Its consequence is also unmeasured: how often the
   out-of-sample label agrees with a refit on the augmented data.
3. **No self-consistency guarantee exists.** Predict on a training row does
   not reproduce `embedding_` or `labels_`. The fit's graph symmetrization
   differs from the query's one-directional affinity. The eigen-equation also
   holds only to the Lanczos tolerance (`eigen_tol`). So the documented
   property would be a measured agreement rate, not a guarantee. The brief
   asks for the rule to be documented where it does not guarantee, and that
   needs a measurement this lane did not take.
4. **A small `lambda_c` divides.** A column whose normalized adjacency
   eigenvalue is near 0 amplifies the query projection. With `drop_first =
   false` and the smallest Laplacian eigenvalues kept, `lambda_c` is near 1 for
   the retained columns on connected graphs. That is not true for every input
   (bipartite-like graphs reach `lambda = -1`), so the rule needs a by-name
   refusal threshold, which is another declared constant.
5. **Nothing in the repo is reusable as the Nystrom step.**
   `kernel_methods/`'s `Nystroem` is an RBF kernel approximation, a basis and
   an SVD of the basis kernel. It is not a spectral out-of-sample extension,
   so steps 1 and 2 would be new kernels on the device and on the host, each
   with its own sabotage arm.

## Estimated work to ship it

- The fit outputs (blocker 1) on both stacks, plus the train-cell proof on the
  three committed columns.
- Two new kernels (affinity fold and projection) on the device and on the
  host.
- The k-means predict call on the extended embedding.
- Python `predict`, `save` and `load`, and host subclasses.
- identity_break infer and batch parts for `spectral` and `par-graph-spectral`
  (`spectral-precomputed` needs the cross-affinity input).
- A measured agreement rate on training rows, recorded as the documented
  property.
