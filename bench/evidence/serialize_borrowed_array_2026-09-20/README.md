# Borrowed-array model save qualification (2026-09-20)

- Base: `origin/main` `7f83fee27`.
- Host: Apple M4, 16 GiB, macOS 26.5.2. One benchmark process at a time.
- Public boundary: deterministic `_serialize.write_npz` of a fitted-model-style
  100 MB float32 `Array` (25,000,000 elements), plus its format member. Seed
  `20260920`.
- Candidate borrows the bytes of contiguous `Array` and buffer inputs while
  streaming them. `encode_npy` remains an owning `bytes` API. Noncontiguous and
  order-converting inputs retain their required copies.

Five alternating separate-process `/usr/bin/time -l` pairs:

```
old wall_s 0.67 0.41 0.39 0.36 0.43
new wall_s 0.40 0.43 0.37 0.37 0.40
old rss_B  257310720 256131072 256262144 256180224 256262144
new rss_B  156303360 156237824 156090368 156188672 156123136
```

Median wall time was 0.41 to 0.40 seconds (2.44% lower; the win is primarily
memory). Median peak RSS was 256,262,144 to 156,188,672 bytes: 39.05% lower,
eliminating essentially one complete 100 MB transient payload.

The 4,000,000-node forest-shaped archive screen (96,009,632 bytes, nine
interleaved repetitions) had medians 63.742 ms old and 59.550 ms new (6.58%
lower). Both complete archives had identical SHA-256:

```
fceab9e015c83d170ed78eab6862fad4c00365421c159ff38117178d4cf34493
```

Load-side candidates were rejected: a direct `ZipExtFile.readinto` path slowed
the 100 MB load median from 29.440 to 37.361 ms (26.9%), and a direct
ZIP_STORED/fromfile path with CRC validation slowed it from 25.528 to 29.032 ms
(13.7%). Neither change was retained.

Independent classical screen: production IDENTICAL `MinMaxScaler.fit` and
`StandardScaler.fit` on deterministic 1,000,000 x 16 float32 input had five-run
medians 43.448 ms and 45.433 ms. Their native single-pass reductions already
operate near memory bandwidth; no exact-order >=10% candidate was identified.
Statistic hashes were respectively
`b7f6d400040bd0a3d770c6eb3ba1f4bc4ea95e1eddd08d45f13fd71ed7a3e008` and
`ed268a76d3fc3dc48a0f88260d1315fa379493040e98562e145462ab9bd11b5d`.

Gate:

```
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  python3 -m mojolearn.tests.test_numpy_free_core
```

This covers exact NPY/NPZ bytes against NumPy and the pinned legacy writer,
layouts, empty arrays, round trips, errors, and verifies that streamed
contiguous `Array` payloads are passed as borrowed memoryviews.
