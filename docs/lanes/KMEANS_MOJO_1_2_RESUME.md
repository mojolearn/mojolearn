# lane/kmeans-mojo-1-2: resume file

Branch `lane/kmeans-mojo-1-2`, cut from `trial/mojo-1-2-at-0819` (0.8.19 source
on Mojo 1.2.0.dev2026092505 / MAX 26.7.0.dev2026092505). Branch only, never
merged to main.

## Cause (FOUND, 2026-09-26)

`cluster/checks/scalable_init.mojo::scalable_uniform`, the k-means|| round seed
reassembly. The host passes the 64-bit round seed as two Int32 halves and the
kernel rebuilt it as

    (seed_hi.cast[uint32]().cast[uint64]() << 32) | seed_lo.cast[uint32]().cast[uint64]()

Under Mojo 1.0 that chain was folded into a SIGN extension on every GPU target
(the int-widening trap recorded in
`ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels.mojo`), so
the seed hashed was `(hi << 32) | sext64(lo)`. That was pinned as DEVIATION 2714
(IDENTITY_PATHS.md row 94) and the host oracle spells it explicitly
(`host_round_seed_as_the_device_reassembles_it`). Mojo 1.2 zero-extends the
chain as the types say, so whenever a round seed's low half is >= 2^31 the
device draws a different stream. On the kmeans base fixture (seed 3) round 1's
seed is 0x9cebe8a6d050dd01 (low half negative): 33 flags differ, 17 vs 16
candidates, 132 vs 134 candidates in total, and every downstream stage moves.
The host oracle is unaffected, which is why CPU kept the 0.8.19 bits.

How it was localized: stage traces (`MOJOLEARN_IDENTITY_TRACE`, extra
per-round records of min_dist, psi, flags, csum, chunk totals) of one
`KMeans(n_clusters=8, random_state=3)` fit on the base fixture, built from the
same source under Mojo 1.0 (main checkout's env, read only, private cache) and
under 1.2. First differing stage: round 1 `flags`, with identical inputs
(min_dist, psi, candidates). A numpy replay of `sample_flags_kernel` matched
1.2 with the zero-extended seed and 1.0 exactly with `(hi << 32) | sext64(lo)`.

## Fix

The kernel now spells the sign extension: mask the low half to 32 bits, OR
`0xFFFFFFFF00000000` when the Int32 is negative. Same bits as 0.8.19 under both
toolchains (no dependence on a compiler fold).

## Proof so far

- kmeans base fixture, Metal (M4): 1.2 + fix = 0.8.19 PyPI wheel bits
  (centers 2e933cd5e18553fd, labels 1248adbcbc952b93, inertia
  0x1.09dd68p+18), whole stage trace byte-equal to the 0.8.19 wheel's trace;
  1.0 + fix also unchanged.

## Next step

1. Metal column 627 cells on the nightly with the fix vs the recorded 0.8.19
   Metal column (all bindings, `verify_lanes.py --gpu-pass metal --selection
   trial/selection-metal.json`, then `identity_break.py --diff`).
2. CUDA column on one rented RTX 4090 (RunPod, cap $6), vs 0.8.19 CUDA.
3. CPU column: unaffected by construction (host oracle already spells sext);
   confirm the recorded nightly CPU column = 0.8.19.

## Spend

$0.00 so far (no rental yet).
