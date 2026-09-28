# decomp-apple: progress (Apple Metal speed, FAST and IDENTICAL)

Brief: ~/mojolearn-evidence/apple_speed_brief.md. Branch lane/decomp-apple
(off origin/main, with origin/lane/algos-decomp merged in: its unmerged GPU
speed work and bench/decomp_speed.py are this lane's base). Home Mac for
speed: m4pro-a. Files: ~/mojolearn-evidence/decomp-apple/.

## Profile (existing Apple records, before this lane)

m4pro-a / m4pro-b, 1M x 28 synth (`~/mojolearn-evidence/algos-decomp/speed/`):
the x_decomp kit's cost on Apple is per-call data movement, not arithmetic.
`ew` over 1M x 5: ~2.9 ms a call on the M4 Pro against ~0.33 ms on MI300X
(FastICA 338 ew = 0.99 s vs 0.11 s); gemm 123 calls 1.13 s vs 0.24 s; orth
9 calls 1.5 s vs 0.65 s. Every kit call uploaded its inputs into fresh
buffers, synced twice and downloaded its output.

## Found and fixed

1. **Every x_decomp fit raised TypeError on CPython 3.13** (the pixi env on
   the Macs is 3.13.15): `_M.from_input` passed a typed memoryview to
   `array.frombytes` (since 2f2c2891b). Fix: `mv.cast("B")`, the same one
   copy (ec9a05770).
2. **Isomap hang at 10k rows** (AMD box, Andrew 2026-09-28): the shortest
   paths were a dense O(n^2) scan per source (O(n^3) total, with column
   reads), and the component walk an O(n^2) Python loop. Now: compressed
   arcs built once on the host, a (distance, index) binary heap per row
   (the same pick as the scan: least distance, ties to the lower index, and
   each distance the exact minimum of the same float32 sums, so the same
   bits), the component walk over adjacency lists. Host column bit-equal to
   the old dense routine on 12 random graphs with integer ties
   (dj_check.py). Arm 5314 regenerated on the new cell. Isomap past a few
   thousand rows is still bounded by its n x n Jacobi eigh (arpack is
   refused by name); bench/decomp_speed.py caps Isomap at 3000 rows.
3. **Device-resident matrices** (x_decomp/resident.mojo): ew, gemm, colsum,
   rowsum and sqdist/pdist launch on pooled device buffers with no sync; a
   result stays on the device until Python reads it. Same kernels, same
   launch sequence (DevExec now calls the same launch_* helpers). Python
   plumbing proven on the host with a fake resident binding over the host
   kit: 22 algorithms bit-equal to the plain host path, no buffer leaked
   (mock_resident.py).

## Requests in flight

| request | what |
|---|---|
| 1790581691613-speed-decomp-112ef9d1c7 (m4pro-a) | BEFORE: micro, decomp_speed at N=200k (N2 20k, N3 1500, N4 20k), digests |
| 1790581670956-speed-decomp-edca711112 (m4pro-a) | AFTER: the same |
| 1790581701484-decomp-12f28c7ed5 | identity, 16 x-decomp lanes, e2e_host_all |
| 1790581750691-decomp-df79910175 | identity, x-decomp-spectral-rbf, e2e_host_sqdist |
| 1790581760634-decomp-df79910175 | identity, x-decomp-umap-options, e2e_p2b_options |

## Before -> after (per algorithm, IDENTICAL, m4pro-a)

(pending the two speed requests)

## FAST

x_decomp's FAST build runs the IDENTICAL cells (`_sets/identical` is
loaded in fast mode); no FAST-only path exists yet.
