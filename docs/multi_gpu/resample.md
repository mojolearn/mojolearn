# Resampling: global replicate and sample IDs

Design note for `mojolearn.parallel_classical.bootstrap`, `permutation_test`
and `monte_carlo_integrate`, and for `MOJOLEARN_RESAMPLE_DEVICE_COUNT` in
`resample/estimator.mojo`. IDENTICAL mode only; every multi-device result must
have the exact bits of the one-device result.

## Why the ranges partition

The resample lane already addresses every draw by position. Bootstrap
replicate `r` draws its row indices from Philox counters built from the seed
key and the global replicate index, and its statistic is folded inside that
replicate (a pinned per-replicate fold, or a per-replicate sort and order
statistic). The `r_first` argument exists so that replicates
`[r_first, r_first + R)` of a run are bit-identical to the same slice of the
whole run (DEVIATION 1690(b)). Permutation `r` is the same construction over
the pooled sample. Monte Carlo sample `i` is a pure function of the key and
`i_first + i`, and the lane folds samples in fixed `PINNED_SUM_W = 256`
chunks, then folds the chunk partials on the host in chunk order.

So:

- Bootstrap and permutation: rank `k` of `a` owners computes replicates
  `[r_first + R*k/a, r_first + R*(k+1)/a)` with the original launch on its own
  device and copy of the sample; the values are copied as bytes into the
  root's distribution in global order. Everything after the distribution
  stays on the root and is unchanged: the full sort, the point estimate, the
  order-statistic interval, the standard error, the observed statistic and
  the p-value counts.
- Monte Carlo: owners take whole chunk ranges `[c0, c1)`, drawing from
  `i_first + c0 * 256`, so an owner's chunk `j` is global chunk `c0 + j` with
  the same values and fold. The root gathers the partials in chunk order and
  runs the original host fold and `volume * (sum / n)`.

The identity-card stages (`resample.index_map` window, `resample.theta` or
`resample.null`, `resample.sorted`, `resample.mc.points`,
`resample.mc.partials` and the scalars) are recorded on the root after the
gather, so a traced multi-device run writes the same card. Owners currently
run one after another; there is no throughput claim.

## Gates

- `training/checks/resample_parallel_check.mojo`: all six bootstrap
  statistics at 2 to 4099 replicates with `r_first` 0 and 5, both implemented
  methods and all alternatives; three permutation statistics; three Monte
  Carlo integrands across the 256-sample chunk boundary with `i_first` 0 and
  3. One- and two-device identity traces and every result bit must agree. A
  check-only build, `-D MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE=1`, shifts later
  owners one position and must fail.
- `tools/parallel_resample_check.py`: the public functions against one device.

`method='bca'` stays refused by name (DEVIATION 1699).
