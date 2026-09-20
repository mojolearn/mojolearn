# Exact-capacity device download qualification

Scope: IDENTICAL mode, Apple Metal, production `metrics.checks.device_io`
download helpers. Candidate preallocates the final `List` capacity before the
unchanged ascending append loop. Element order and values are unchanged.

Workload: public preprocessing native ABI, StandardScaler fit plus transform,
1,000,000 rows x 8 Float32 columns. The fixture is a deterministic uint32 LCG
mapped to `[0,1)`. Each run used a fresh process; round zero is warmup.

Baseline warmed milliseconds: 137.870, 127.964, 156.401, 155.670, 425.362,
142.302; median 149.0 ms. Exact-capacity warmed milliseconds: 121.198,
117.608, 124.143, 109.947, 126.646, 126.077; median 122.7 ms. Improvement:
17.7%. Every round on both arms produced SHA-256 prefix
`423db12a1e1923a0` for all 8,000,000 output Float32 cells.

The first completed call's process peak RSS was 293,224,448 bytes baseline
and 244,334,592 bytes candidate, 48,889,856 bytes lower. Later-process maxima
are allocator high-water marks and are not treated as live allocation.

The same candidate ran MinMaxScaler fit plus transform at the same shape for
six rounds. All output hashes were `bcf689e2272ff89e`; warmed times were
124.790, 133.075, 148.575, 142.895, and 152.811 ms.

`bindings/build_preprocessing.sh` in IDENTICAL mode passed both StandardScaler
and MinMaxScaler native ABI fit/transform/inverse gates after the change.
