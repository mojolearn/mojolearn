# ann-apple: Apple (Metal) speed, FAST and IDENTICAL

Lane `ann-apple`, branch `lane/ann-apple`, worktree `~/mojolearn-wt/ann-apple`
(brief: `~/mojolearn-evidence/apple_speed_brief.md`). Home Mac for speed
jobs: m4pro-b. Nothing here merges to main; the gate runners merge after an
NVIDIA + CPU gate.

Bench: `bench/speed/ann_cpu_speed.py` through the public classes with the GPU
route (MOJOLEARN_VENDOR unset), HIGGS from R2 (`~/datasets/gbm-bench/higgs/
higgs_speed.npz` on the Macs), standardized, float32. Shapes: IVF family 1M x
28 index, 1024 lists, 32 probes, 1000 queries, k = 10, 10 k-means
iterations (IVF-PQ: pq_dim 14, 8 bits); CAGRA 50k x 28 (degree 64 -> 32);
t-SNE 10k x 28, 300 iterations. Each cell prints a digest of its outputs, so
one run is the timing and the bit check. `MOJOLEARN_ANN_STAGES=1` prints a
per-stage split of every device driver (`x_ann/stage_timer.mojo`; off by
default, it syncs only when on).

## Baseline, IDENTICAL, m4pro-b (commit 5b622763d, request 1790580086853)

| algorithm | fit s | search s | digests (model / out) |
|---|---|---|---|
| t-SNE (10k, 300 it) | 2.373 | - | out f31f68ec8bad9247 |
| CAGRA (50k) | 14.199 | 0.073 | 54d696296c9c7c8a / 45e435db03654b4e |
| IVF-Flat | 8.946 | 0.171 | out d730b8082a1cdbfb |
| IVF-SQ | 9.466 | 0.543 | c55eedfcb6459d7b / 1d9c53fd8c13f452 |
| IVF-RaBitQ | 8.668 | 0.624 | a4f2343eb268b0a7 / 056a570709713477 |
| IVF-PQ | 21.812 | 0.767 | 6bb7a6c5fc753846 / 3b1e0c1ae73444eb |
| refine (IVF-PQ top-40 -> 10) | (IVF-PQ fit) | 0.776 | refine 0.047 s, out 3c73bf8ae59e47e4 |

## Changes (IDENTICAL: the same bits, digests below equal before and after)

1. 34d98b186 + 6b357b96f: IVF-PQ / IVF-SQ / IVF-RaBitQ device search split
   (`x_ann/ivf_scan_device.mojo`). The old kernel ran one thread per query
   that recomputed every coarse distance for every probe and scored its
   candidates serially. Now: coarse distances once per (query, list); the
   probe walk over them (`pq_probe_takes`, the cell's comparison); one
   threadgroup per (query, probe) scoring the list (PQ lookup entries once
   into threadgroup memory; `sq_candidate_dist` / `rq_candidate_est`, the
   cells' statements as shared helpers); one thread per query inserting the
   stored distances in the cell's order with `pq_insert`.
2. 3a9830f63 (`x_ann/knn_device.mojo`): t-SNE / CAGRA exact k-NN tiled through
   threadgroup memory: 64-row tiles of ftz(x), row i's ftz(x) in registers,
   the fold at a comptime width with zero padding (fma(+0, +0, acc) == acc),
   candidates ascending through the cell's own `ts_knn_offer`.
3. 4316f3eb3: t-SNE repulsion tiled the same way (`repulse_tiled_kernel`,
   `ts_repulse_pair`).
4. 5035df694: IVF-SQ range as 1024-row chunk partials joined in row order
   (`sq_lo_takes` / `sq_hi_takes`), instead of one thread per column over
   all rows.
5. `MOJOLEARN_ANN_STAGES=1` stage timings (`x_ann/stage_timer.mojo`).

Sabotage arms retargeted to the shared helpers (all apply): 5804, 5810,
5813 (now also reverses the tiled device fold), 5830, 5832, 5842, 5855.

## M3 Ultra (m3ultra), IDENTICAL, stage split, before 34c08c918 (main + timer) -> after 6b357b96f (changes 1-3)

| stage (ms) | before | after |
|---|---|---|
| CAGRA k-NN graph (50k x 28, k 64) | 1983 | 215 |
| CAGRA prune (host) | 723 | 749 |
| t-SNE k-NN + perplexity (10k) | 193 | 41 |
| t-SNE 300 iterations | 900 | 843 |
| t-SNE symmetrize (host) | 132 | 118 |
| IVF coarse k-means (cluster/, 1M, 1024 lists) | 1271-1283 | 1279-1292 |
| IVF-PQ codebooks (cluster/ k-means x 14) | 2474 | 2456 |
| IVF-SQ range | 435 | 421 (change 4 not in this run) |

| algorithm (s) | before | after | digests |
|---|---|---|---|
| t-SNE fit | 1.269 | 1.112 | f31f68ec8bad9247 both |
| CAGRA fit | 2.778 | 1.031 | 54d696296c9c7c8a / 45e435db03654b4e both |
| IVF-SQ search | 0.580 | 0.375 | 1d9c53fd8c13f452 both |
| IVF-RaBitQ search | 0.673 | 0.345 | 056a570709713477 both |
| IVF-PQ search | 0.873 | 0.355 | 3b1e0c1ae73444eb both |
