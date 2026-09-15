# KMeans.predict: the k-means infer and batch cells (2026-09-15)

Run on the Apple M4 on one core (a shared machine), at the merge of
lane/cpu-training-kmeans-predict with origin/main. Fixtures were base, ties,
odd and denormal, with two repeats. The lanes were kmeans, kmeans-random,
kmeans-array, kmeans-weighted, kmeans-sqrt, kmeans-classic-pp, kmeans-cosine
and par-kmeans.

- `apple-m4.json` is the Metal identical set. It has 32 train cells, all
  STABLE. Infer is STABLE on 28 and n/a on 4, and batch is STABLE on 28 and
  n/a on 4. The n/a cells belong to kmeans-cosine, whose fit is refused by
  name.
- `cpu-apple-m4.json` is the CPU column: a CPU-only package view with the
  core host binding. Train, infer and batch are STABLE on 24 cells each. The
  4 par-kmeans cells are refused by name because its driver needs a GPU
  device group, and par-kmeans is not a CPU-covered lane.
- `diff_metal_cpu.txt`: IDENTICAL=28 on train, IDENTICAL=24 on infer/model
  and IDENTICAL=24 on batch. The only ONE-COLUMN cells are the 4 par-kmeans
  cells.
- `diff_record5.txt`: this adds the kmeans-sqrt fix record's Apple, H100
  and MI325X columns. Train reads IDENTICAL=72, so no train cell of any
  k-means lane moved against the committed GPU columns. The recorded GPU
  infer and batch cells are n/a:no-predict, so those cells rest on the M4
  and CPU columns only.
- `apple-m4.batch-sabotage.json` and `cpu-apple-m4.batch-sabotage.json`
  were run under `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`. Every new batch
  cell reads BATCH_MOVED: 28 on Metal and 24 on CPU
  (`diff_metal_bsab.txt`, `diff_cpu_bsab.txt`).
- `cpu-apple-m4.host-sabotage.json` was run with
  `-D MOJOLEARN_HOST_SABOTAGE=1`. Against Metal, 24 train, 24 infer and 24
  batch cells read DIVERGENT (`diff_metal_cpuhsab.txt`). The existing arm
  moves the centroid sums, and predict inherits the moved centroids.

Still owed to the next release record: the NVIDIA and AMD infer and batch
cells, and the two-device par-kmeans cells. No box was rented.
