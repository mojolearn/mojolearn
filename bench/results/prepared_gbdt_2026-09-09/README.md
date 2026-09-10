# Prepared numeric GPU datasets — Apple M4, 2026-09-09

Implementation: `gbdt/prepared.mojo`; usage and scope:
[PREPARED_GBDT.md](../../../docs/lanes/PREPARED_GBDT.md).

The reusable pool retains numeric borders/NaN metadata, compressed device
index, targets, weights and context. It currently serves native Mojo callers
for RMSE, Logloss and CrossEntropy under all three growth policies. It does
not cache every per-fit workspace or expose Python handles/CTR/eval pools.

Run with:

```sh
python3 tools/gbdt_prepared_check.py --out bench/results/prepared_gbdt_2026-09-09
```

The check compares 72 ordinary/prepared fits per mode using complete model
text, prediction bits and Float64 loss bits; it also checks caller-mutation
isolation and invalid Poisson rates. Each log prints the actual compiled mode.
The shared quantizer body was additionally compared byte-for-byte to the
original training body before extraction; its arithmetic is unchanged.

## FAST repeated-fit measurement

65,537 rows × 8 numeric features, RMSE, Lossguide, 5 trees, depth 5,
32 borders, border sample cap 4,097. Three warmup pairs and six measured
pairs, alternating which arm runs first. Device synchronization surrounds
each timer; full model equality is required after every run. Local repository
build lock held for the benchmark, so cooperating builds/tests were excluded.

- One-time pool preparation: **9.319 ms**.
- Ordinary fit median: **153.567 ms**.
- Reused-pool fit median: **145.992 ms** (1.052×; 4.9% less time).

Reuse timing excludes one-time preparation. This is one synthetic local
workload, not a general fit-speed guarantee or NVIDIA/AMD measurement.
Raw samples: `fast.bench.log`.

Validation completed: FAST, IDENTICAL and DETERMINISTIC each passed all 72
comparisons, owned-snapshot and invalid-bootstrap checks (216 comparisons total).
