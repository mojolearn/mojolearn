# DBSCAN FAST on taxi times out on the M3: cause and fix

Race: `tools/classical_two_datasets.py race --lane dbscan --dataset taxi`, ours-fast,
1,000,000 standardized taxi rows x 11 columns, eps=3, min_samples=2, `algorithm='rbc'`
(`tools/classical_two_datasets.py:210`, `:236-237`, `:1773`).

## Cause

The ball-cover route materializes every eps-edge. Standardized taxi is a tight bulk
(the outliers inflate the std), so at eps=3 nearly every row neighbours nearly every
other row: about n^2 = 1e12 edges.

- `dbscan/impl/runner.mojo:618`: loop 1 counts every row's eps-neighbours in full (no
  early exit at min_samples), about 1e12 distances.
- `dbscan/impl/runner.mojo:640`: a batch over `edge_cap` (int32 CSR, 2^31 edges) is split
  in halves and recounted. At ~1e6 edges per row that is ~2,000 rows per batch, so
  there are ~500 batches and each split recounts its rows.
- `dbscan/impl/runner.mojo:817` / `:939`: loop 2 refills each batch's CSR, another ~1e12
  distances, and `:989` / `:994` `weak_cc_batched` min-propagates over ~2e9 edges per
  pass, several passes per batch, then `:1015` `merge_labels_run` once per batch.
- `dbscan/impl/runner.mojo:1057`: with more than one batch the border pass reads the
  core mask and labels back and loops over the rows on the host, refilling CSR batches.

The work is O(n^2) edges, walked several times. Nothing is wrong with the kernels; the
route has no shortcut when most pairs are edges.

## Fix (`-D MOJOLEARN_DBSCAN_FAST_DENSEBALL=1`, FAST + Apple, default off)

`dbscan/impl/denseball.mojo`, entered from `runner.mojo` right after the ball-cover index
is built (unweighted, Euclidean only). It never builds an edge list:

1. Dense balls: rows of a landmark slice with `d1 <= eps/2 * 0.999` are pairwise within
   eps, so a prefix of at least min_samples such rows is all core and one component.
2. Every other row gets an eps count with ball-cover pruning that stops at min_samples.
3. Components: lock-free union-find (CAS, smaller root wins), one threadgroup per
   unpruned landmark pair. A pair stops at its first hook and is retried the next round.
   It is done after a clean scan, or once all its core rows share a root. Rounds run
   until one hooks nothing; one flag readback per round.
4. Labels: core gets `root + 1`, which is the `weak_cc` fixed point (smallest core id + 1).
   Non-core rows get the smallest core neighbour's label (the DEVIATION 5130 border rule),
   or MAX_LABEL. Then the same `make_monotonic` + `relabel_for_skl` tail runs
   (`runner.mojo` `_dbscan_finish`). The labels match the reference route's, not only
   up to a permutation.

Edge predicate: `eps_dist_sq <= eps*eps` in Float32 (the ball-cover kernels' own). Pruning
bounds use slack, so no true edge is dropped.
