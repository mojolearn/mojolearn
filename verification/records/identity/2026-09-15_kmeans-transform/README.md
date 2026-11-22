# KMeans.transform: the k-means infer and batch cells (2026-09-15)

Run on the Apple M4 on one core (a shared machine), at bb937d57b on
lane/inference-new-methods. The bindings were built from that tree:
`_mojolearn` (Metal, identical) and `_mojolearn_core_host` (CPU), plus the
core host binding with `-D MOJOLEARN_HOST_SABOTAGE=1`. Fixtures were base,
ties, odd and denormal. The lanes were kmeans, kmeans-random, kmeans-array,
kmeans-weighted, kmeans-sqrt, kmeans-classic-pp, kmeans-cosine and
par-kmeans. The infer probe of every fitting lane now hashes `predict` and
`transform` on the held-out rows, and raises unless `transform(X)` at
`labels_` is each training row's minimum bit for bit. The batch part asks
`predict` and `transform`.

- `apple-m4.json`, the Metal identical column, two repeats. Train is STABLE
  on 32 of 32 cells. Infer and batch are each STABLE on 28 cells and n/a on 4
  (kmeans-cosine, whose fit is refused by name).
- `cpu-apple-m4.json`, the CPU column: a CPU-only package view with the core
  host binding, two repeats. Train, infer and batch are STABLE on 24 cells
  each. par-kmeans refuses by name (its driver needs a GPU device group).
- `diff_metal_cpu.txt`: IDENTICAL=28 on train, IDENTICAL=24 on infer and on
  batch. The only ONE-COLUMN cells are par-kmeans.
- `diff_fixrecord_owed.txt`: the Apple, H100 and MI325X columns of
  `2026-09-14_kmeans-sqrt-fix`, cut to these four fixtures, with the Metal
  and CPU columns under `--require-columns 4 --owed-json`. Train reads
  IDENTICAL=32, so no train cell of any k-means lane moved. Infer and batch
  read OWED=24 each; `owed.json` lists the 48 parts the next release record
  owes. The par-kmeans infer and batch cells read ONE-COLUMN (Metal only),
  which is why that diff exits 1; they are owed to a GPU record too.
- `diff_record5.txt`: the full fix record (nine fixtures) with Metal and CPU,
  train IDENTICAL=72.
- `diff_record_metal.txt`: against the older 166-lane record. Its 5
  DIVERGENT cells are kmeans-sqrt, where that record predates the
  kmeans-sqrt fix (the fix record agrees); its REQUIRE FAIL rows are the
  fixtures this run did not cover.
- `apple-m4.batch-sabotage.json` and `cpu-apple-m4.batch-sabotage.json`, run
  with `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`: BATCH_MOVED=28 on Metal and 24
  on CPU (`diff_metal_bsab.txt`, `diff_cpu_bsab.txt`).
- `cpu-apple-m4.host-sabotage.json`, the sabotage host build: against Metal,
  DIVERGENT=24 on train, infer and batch (`diff_metal_cpuhsab.txt`). The arm
  moves the centroid sums, and transform reads the moved centroids.
- `test_metal.log` and `test_cpu.log`: test_kmeans_transform GREEN (19
  checks) on both, test_kmeans_predict GREEN (25) on both,
  test_kmeans_metric_surface GREEN (16) on Metal. On the CPU view that last
  module's two arms call `fit`, which a CPU-only install refuses by design
  (training is the internal verifier's), so they do not run there; no
  workflow runs it on CPU.

Still owed to the next release record: the NVIDIA and AMD infer and batch
cells for the six single-device lanes, and the par-kmeans cells. No box was
rented.
