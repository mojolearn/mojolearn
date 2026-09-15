# SpectralClustering.predict: design, blockers and results (2026-09-15)

Lane `lane/spectral-predict`, DEVIATION 2860. Stages 1 and 2 of
`lane/inference-transductive-predict` shipped DBSCAN and
AgglomerativeClustering `predict` (DEVIATION 2740). This note first recorded
why `SpectralClustering.predict` was not shipped with them. It now records the
design that shipped and how each blocker was resolved. The evidence is in
`bench/results/identity_break/2026-09-15_spectral-predict/`.

## The method

This is the Nystrom out-of-sample extension. The reference is Bengio,
Paiement, Vincent, Delalleau, Le Roux and Ouimet, "Out-of-Sample Extensions
for LLE, Isomap, MDS, Eigenmaps, and Spectral Clustering" (NIPS 2003),
section 3. Fowlkes, Belongie, Chung and Malik, "Spectral Grouping Using the
Nystrom Method" (TPAMI 2004), use the same step for spectral grouping.

For the fit's normalized affinity with eigenpairs `(mu_k, u_k)`, a new row
`x` gets

    e_k(x) = (1 / mu_k) * sum_i Ktilde(x, x_i) u_k[i] / sqrt(d(x))
    Ktilde(x, x_i) = K(x, x_i) / (sqrt(d(x)) * sqrt(d_i))
    d(x) = sum_i K(x, x_i)

- `mu_k = 1 + theta_k`, where `theta_k` is the Ritz value of the negated
  normalized Laplacian the fit solves.
- `u_k` is the fit's unit Ritz vector before its degree division.
- `sqrt(d_i)` is the fit's `diagonal` (the square root of the degree, with
  zeros set to one).
- The final division by `sqrt(d(x))` is the fit's own row scaling
  (`divide_rows_kernel`).

The row is then assigned by the fit's own final k-means pass
(`kmeans_predict`, `METRIC_L2_EXPANDED`) against the fit's centroids. Ties go
to the lowest centroid index.

The rule is stated once, in `spectral/host/spectral_predict_host.mojo` (the
CPU spelling). The device spelling is `spectral/impl/spectral_predict.mojo`.

## The blockers, resolved

1. **The fit discarded what predict needs.** `SpectralClustering(
   prediction_data=True)` calls new `_state` binding entries, which reach
   `_keep` variants of `compute_eigenpairs`, `transform_graph`,
   `fit_predict_graph` and `fit_predict_dataset` on the device, and of the
   host oracle's three fit entries. They copy out:
   - the Ritz values and the undivided Ritz vectors, reordered into embedding
     column order (a gather);
   - the downloaded `diagonal`;
   - the downloaded centroids.

   No value is recomputed. The old entries call the `_keep` variants with
   `keep = False`. The train cells are proven unchanged in the evidence
   directory.
2. **Affinity of a new row.** The kernel is the fit's own:
   - **`nearest_neighbors`.** The fit's identical k-NN of the query against
     the training rows at `n_neighbors`. Each neighbor gets `0.5`, which is
     the fit's symmetrization `0.5 * (a + b)` of an edge with no reverse edge.
     The query's row is restricted to its own edges, and the training degrees
     are unchanged.
   - **`precomputed`.** The caller passes the `(n_new, n_train)` affinity.

   (`rbf` is refused at construction, as it always was.) Every fold (the
   degree, each projection) runs over the training rows in ascending index,
   seeded `+0.0`, through `ftz`, `identical_mul_add`, `identical_div`,
   `identical_mul` and `identical_sqrt`, one query at a time.
3. **Near-zero eigenvalues.** The clustering fit drops nothing
   (`drop_first = false`), so predict drops nothing either. The trivial
   column's `mu` is near 1. A used column with `|mu_k| < 1e-3`
   (`SPECTRAL_PREDICT_MIN_ABS_EIGENVALUE`), or a NaN, is refused by name on
   both bindings, with the column and its value in the message.
4. **Training rows.** The extension reproduces `embedding_` only in exact
   arithmetic and only for the fit's own affinity. Asked as a query, a
   training row sees itself at `0.5` where the fit's graph had `1.0`, and it
   has no reverse edges. The Lanczos pairs also hold only to `eigen_tol`. The
   agreement rate is measured, not promised; see Results.

## Results

Filled in from the evidence directory.
