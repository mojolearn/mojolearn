# Multi-GPU resampling by global IDs — two H100s

RunPod pod `qcybgy3aqu52m6`, two NVIDIA H100 80GB HBM3 (sm_90a), 2026-09-14
22:25-22:29Z. Source: the `git archive` of commit `b9f219a9e`
(`commit.txt`); box and Mac source SHA256 agree. Body `body.sh`; GPU work
serial. Design: `docs/multi_gpu/resample.md`.

## Result (`out/gate.txt`)

- `training/checks/resample_parallel_check.mojo` (production): 67 PASS lines,
  `PASS resample parallel gate`: 36 bootstrap cases (six statistics, 2 to 4099
  replicates, `r_first` 0 and 5, percentile and basic, three alternatives), 9
  permutation cases (three statistics, up to 4097 permutations, `r_first` 0
  and 5) and 21 Monte Carlo cases (three integrands, 1 to 100003 samples across
  the 256-sample chunk boundary, `i_first` 0 and 3). For each, the one-device
  and two-device identity traces are equal and so are the distribution,
  sorted distribution, point estimate, standard error, interval and order
  positions, or the null distribution, observed statistic, p-value and
  counts, or the integral and mean.
- The same gate built with `-D MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE=1` FAILS
  at its first case: `trace differs boot-n53-d2-s0-m0-a0-r2-f0: 2
  resample.theta f32 2 82fecf6e6cf037c4 VS 2 resample.theta f32 2
  23e6ff649c2b542b`. That run stops in the bootstrap partition; the
  permutation and Monte Carlo partitions are shown failing on their own in
  the MI300X leg, which runs the sabotage build per family
  (`MOJOLEARN_RESAMPLE_CHECK_ONLY`, added after this leg).
- `tools/parallel_resample_check.py`: `PASS 36 resample configurations and 5
  refusals` (`out/public.json`), the public `bootstrap`, `permutation_test`
  and `monte_carlo_integrate` against one device.
- `resample/checks/resample_check.mojo`, the unchanged single-device gate:
  `resample_check mode=IDENTICAL ALL OK`.

No speed claim.
