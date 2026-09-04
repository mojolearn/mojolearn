# FAST versus IDENTICAL price harness

`tools/lanes_price.sh` builds separate FAST and IDENTICAL binaries, alternates them within one clean
window, verifies each binary's compiled mode, and writes raw evidence beneath
`bench/results/lanes_price/`.

## What it measures

Each sample times one complete public lane entry with device synchronization before and after it.
The harness performs one untimed warm-up, records every timed round separately, and hashes the
observable output. It reports median IDENTICAL time divided by median FAST time plus the range of
paired-round ratios.

The supported lanes are:

- classical: `cd`, `kde`, `linkage`, `svm`, `metrics`, `gemm`;
- unsupervised/linalg: `kmeans`, `knn`, `dbscan`, `gram`, `nt`, `gemv`;
- trees: `gbdt`, `rf`, `et`.

Fixture definitions and environment size knobs live beside the executable driver in
`bench/lanes_price_main.mojo`; repeating them here caused earlier documentation drift.

## Run

Default lanes and sizes:

```bash
tools/lanes_price.sh
```

Select lanes explicitly:

```bash
MOJOLEARN_LANES_PRICE_LANES="kmeans knn dbscan" tools/lanes_price.sh
MOJOLEARN_LANES_PRICE_LANES="gbdt rf et" tools/lanes_price.sh
```

For a build-only smoke run:

```bash
MOJOLEARN_LANES_PRICE_SMOKE=1 tools/lanes_price.sh
```

Smoke timings are never publishable. Datacenter measurements must use representative sizes rather
than the small Apple defaults; set the documented `MOJOLEARN_LANES_PRICE_*` variables without
editing the driver.

## Interpretation

- A paired-ratio band crossing 1.0 is inconclusive; increase the workload instead of quoting it.
- Equal FAST and IDENTICAL hashes mean the selected fixture did not expose a numerical difference.
- Moving IDENTICAL hashes indicate a correctness failure, not a performance result.
- A ratio below 1.0 means FAST is slower and must be investigated with profiling and repeated runs.
- Results describe one device, fixture, commit, and session. They do not rank vendors.

Historical AMD, Apple, and NVIDIA tables remain in Git history and raw result directories. Current
claims should be generated from admitted result records rather than copied into prose.
