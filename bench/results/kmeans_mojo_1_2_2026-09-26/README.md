# k-means under Mojo 1.2: cause, fix, proof (2026-09-26)

Branch `lane/kmeans-mojo-1-2` (from `trial/mojo-1-2-at-0819`, the 0.8.19
source on Mojo 1.2.0.dev2026092505 / MAX 26.7.0.dev2026092505). Fix commit
0a60b76ae.

## Verdict

**With the fix, Mojo 1.2 moves 0 of 627 release cells.** Metal and CUDA
on the nightly equal the recorded 0.8.19 columns on all 627 cells (infer and
model 984 of 984), and equal the nightly CPU column (which never moved).

| diff (identity_break --diff) | train | infer/model | DIVERGENT |
|---|---|---|---|
| 0.8.19 Metal vs nightly+fix Metal (M4) | IDENTICAL=627 | IDENTICAL=984 | 0 |
| 0.8.19 CUDA vs nightly+fix CUDA (RTX 4090) | IDENTICAL=627 | IDENTICAL=984 | 0 |
| nightly+fix Metal vs nightly+fix CUDA | IDENTICAL=627 | IDENTICAL=984 | 0 |
| nightly CPU vs nightly+fix Metal vs nightly+fix CUDA | IDENTICAL=627 | IDENTICAL=984 | 0 |

References: 0.8.19 Metal `~/mojolearn-evidence/release-check/69a519c1522d/metal/column.json`,
0.8.19 CUDA `~/mojolearn-evidence/release/0.8.19/69a519c1522d/smoke-linux/remote/column.json`,
nightly CPU `bench/results/mojo_1_2_nightly_trial_2026-09-26/columns/nightly-cpu-m4.column.json.gz`
(on `lane/mojo-1-2-trial`). New columns are gzipped in `columns/`, diff
summaries in `diffs/`.

## Cause

`cluster/checks/scalable_init.mojo::scalable_uniform`, the k-means|| (default
init) round seed. The host splits the 64-bit round seed into two Int32 halves;
the kernel rebuilt it as `(hi.cast[uint32]().cast[uint64]() << 32) |
lo.cast[uint32]().cast[uint64]()`. **Mojo 1.0 folded the low half's
`int32 -> uint32 -> uint64` chain into a sign extension** on every GPU target
(the same fold `ensemble/decisiontree/.../builder_kernels.mojo` documents), so
the hashed seed was `(hi << 32) | sext64(lo)`. DEVIATION 2714 pinned those
bits and the host oracle spells them explicitly. **Mojo 1.2 zero-extends** the
chain, as its types say, so any round whose seed has a low half >= 2^31 draws a
different candidate set. Everything downstream of a k-means|| fit moved
(kmeans, kmeans-sqrt, kmeans-weighted, gmm, gmm-sample, ivf x3, metrics x2);
kmeans-random, kmeans-array and kmeans-classic-pp, which do not use the round
seed, did not. The CPU host oracle does its own explicit sign extension, so it
kept the 0.8.19 bits.

How it was found: stage traces (`MOJOLEARN_IDENTITY_TRACE`, plus temporary
per-round records of min_dist, psi, flags, csum inside
`init_scalable_kmeans_plus_plus`) of `KMeans(n_clusters=8, random_state=3)` on
the base fixture (20000 x 16), the same source built under Mojo 1.0 (main
checkout's env, read only, private cache) and under 1.2, on Metal. The first
differing stage was round 1's `flags` with byte-equal inputs (min_dist, psi,
candidates). Round 1's seed is 0x9cebe8a6d050dd01 (low half 0xd050dd01,
negative as Int32); rounds 0 and 2 have positive low halves and matched. A
numpy replay of `sample_flags_kernel` reproduced 1.2's flags with the
zero-extended seed and 1.0's with 0xffffffffd050dd01 exactly (0 misses, 0
extras each). Nothing else moved: MAX `prefix_sum`, reductions, distances and
the Lloyd loop were byte-equal until the candidate set differed.

## Fix

Spell the sign extension: mask the low half to 32 bits and OR
`0xFFFFFFFF00000000` when the Int32 is negative (the high half is masked too).
Arithmetic, so no cast fold can reinterpret it. Same bits as 0.8.19 under
BOTH toolchains: on the base fixture the 1.0 build with the fix and the 1.2
build with the fix each give the 0.8.19 wheel's centers 2e933cd5e18553fd,
labels 1248adbcbc952b93, inertia 0x1.09dd68p+18, and the 1.2 trace is
byte-equal to the 0.8.19 PyPI wheel's trace (1190 records).

## Runs

- Metal: all 76 bindings (23 identical GPU, 32 host, 18 fast, 3
  deterministic) built on the M4 from 0a60b76ae, one at a time, -j 1, private
  MODULAR_HOME; `verify_lanes.py --apple-pass --selection
  trial/selection-metal.json`: COMPLETE, 209 lanes, 627 cells, 531 s.
- CUDA: RunPod RTX 4090 pod re8o8tv6g4p228 via `tools/gemm_remote_leg.sh
  nvidia --payload gemm` + `trial/nvidia_body.sh` (all bindings built on the
  box, 0 failures, 1737 s; `--gpu-pass cuda --selection
  trial/selection-cuda.json`: COMPLETE, 627 cells, 172 s; gemm card: no
  divergence). 07:54Z to 08:28Z, about 35 min at $0.74/h = about $0.43.
  Terminated and verified gone (HTTP 404). Evidence:
  `~/mojolearn-evidence/e1g/2026-09-26-kmeans-fix-nvidia/`.
- CPU: not rerun. The fix touches no host code path (the host oracle has its
  own `host_round_seed_as_the_device_reassembles_it`; only its docstring
  changed), and the recorded nightly CPU column equals both fixed GPU columns
  on 627 of 627.

AMD: not tested (no AMD box in this lane).
