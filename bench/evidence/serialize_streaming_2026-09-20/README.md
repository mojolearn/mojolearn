# Exact streaming NPZ qualification (2026-09-20)

- Base after integration: `origin/main` at `12069d165`.
- Host: Apple M4, 16 GiB, macOS 26.5.2. One benchmark process; no
  benchmark concurrency.
- Production boundary: `_serialize.write_npz`, using the existing forest-shaped
  benchmark payload at 4,000,000 nodes (six numeric arrays plus the format
  member), 96,009,632 output bytes. Seed: `20260920`.
- Change: give each ZIP_STORED member to `ZipFile.open` and write its NPY header
  and raw payload separately, instead of concatenating another complete member
  before `writestr`.

Interleaved old/new write times in milliseconds (nine repetitions):

```
old 108.844 89.163 83.259 78.279 86.977 79.939 83.925 35.029 32.569
new  74.212 88.846 82.389 72.916 89.750 82.014 67.141 28.391 25.452
```

Medians: 83.259 ms old, 74.212 ms new, 10.87% lower latency (1.122x).
The late trials benefited from filesystem caching in both arms; alternation was
reversed every round.

The complete archives were byte-for-byte equal. Both SHA-256 values were:

```
fceab9e015c83d170ed78eab6862fad4c00365421c159ff38117178d4cf34493
```

Separate-process `/usr/bin/time -l` runs (old/new alternating) reported peak
RSS bytes:

```
old 264814592  new 200572928
old 264749056  new 200949760
old 265093120  new 200671232
```

Median peak RSS fell from 264,814,592 to 200,671,232 bytes (24.22%). Separate
process wall seconds were old `[1.18, 0.76, 0.81]`, new `[0.63, 0.70, 0.87]`;
medians 0.81 to 0.70 seconds (13.58%). RSS includes interpreter, input arrays,
and package import, so this is an end-to-end process measurement.

Correctness gate after merging the base:

```
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  python3 -m mojolearn.tests.test_numpy_free_core
# 14/14 passed
```

That gate compares NPY bytes to NumPy, streamed NPY bytes to the same oracle,
NPZ bytes to the pinned 0.6 writer, round trips every supported numeric layout,
and exercises refusal behavior.
