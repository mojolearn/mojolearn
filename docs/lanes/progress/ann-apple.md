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

## M4 Pro (m4pro-b, home), IDENTICAL: before origin/lane/merged 003ea19ba -> after 8ce6266d7 (changes 1-4 merged onto lane/merged)

Requests 1790584322570 (before) and 1790584324545 (after). Every digest
equal before and after (lane/merged's t-SNE digest is 2bb1d3d75ffa1885; it
moved from main's f31f68ec8bad9247 by lane/merged's own t-SNE changes, not
by this lane).

| algorithm (s) | before fit | after fit | before search | after search |
|---|---|---|---|---|
| t-SNE (10k, 300 it) | 2.476 | 1.767 | - | - |
| CAGRA (50k) | 13.627 | 1.050 | 0.052 | 0.052 |
| IVF-Flat | 8.679 | 8.662 | 0.140 | 0.141 |
| IVF-SQ | 9.122 | 8.824 | 0.532 | 0.368 |
| IVF-RaBitQ | 8.645 | 8.651 | 0.604 | 0.378 |
| IVF-PQ | 21.727 | 21.631 | 0.756 | 0.373 |
| refine (0.014 -> 0.013 s) | | | | |

After, stage split (ms): t-SNE k-NN + perplexity 92, symmetrize (host,
threaded by lane/merged) 120, 300 iterations 1472; CAGRA k-NN 862, prune
(host, threaded) 111, reverse merge 27, search 49; IVF-SQ range 1.8 (was
~430 ms on the M3 Ultra), encode 171; IVF-PQ residuals 184, encode 196.

FINDING (cluster/, not this lane's code): the IVF fits are the coarse
k-means and the PQ codebook k-means. On the M4 Pro the coarse k-means (1M x
28, 1024 lists, k-means++ then 10 Lloyd iterations, `kmeans_fit_main_traced`)
takes 8613 ms and the 14 codebook k-means (1M x 2, 256 codes each,
`kmeans_fit`) 12726 ms; the same calls take 1280 and 2460 ms on the M3 Ultra
(about 7x, more than the GPU core ratio). k-means++ with k = 1024 on 1M rows
is 1023 sequential picks, each several full passes over the rows (scan,
search, gather, GEMM, cost, adopt), already enqueued without a sync on Apple
(KMEANS_FAST_PP_NOSYNC). IDENTICAL cannot shorten it from this lane.

## M4 Pro (m4pro-b), FAST: before origin/lane/merged 003ea19ba -> after 8ce6266d7

Requests 1790584328631 (before) and 1790584331100 (after). FAST adds one
change: 5d218f95d, IVF-PQ codebooks train on at most 256 rows per code, a
seeded uniform sample (`PQ_FAST_TRAINSET`, FAISS's max_points_per_centroid
rule; `-D MOJOLEARN_PQ_FAST_TRAINSET_OFF` reverts). Digests equal before and
after for every algorithm except IVF-PQ / refine (the codebooks changed);
quality below. (t-SNE did not run in either: lane/merged's TSNE needs the
FAST `_mojolearn_estimators.so`, which these two requests did not build; a
t-SNE pair follows.)

| algorithm (s) | before fit | after fit | before search | after search |
|---|---|---|---|---|
| CAGRA (50k) | 12.465 | 0.936 | 0.070 | 0.050 |
| IVF-Flat | 2.095 | 2.063 | 0.133 | 0.139 |
| IVF-SQ | 2.692 | 2.284 | 0.519 | 0.377 |
| IVF-RaBitQ | 2.118 | 2.081 | 0.621 | 0.382 |
| IVF-PQ | 11.791 | 3.697 | 0.737 | 0.375 |
| refine (0.045 -> 0.013 s) | | | | |

After, stage split (ms): IVF-PQ coarse 2032 (FAST samples the coarse
trainset on Apple already), codebooks 1256 (was 12726 under IDENTICAL, same
Mac), residuals 184, encode 181; CAGRA k-NN 702.

### FAST quality: IVF-PQ codebook sample (paired, 5 seeds x 2 datasets)

`bench/speed/ann_fast_quality.py --algos ivf_pq` (MOJOLEARN_NUMERIC_MODE=fast,
m4pro-b, requests 1790584394142 before 34c08c918 / 1790584398594 after
6b357b96f): 200k x 28 index (the sample is active: 65536 < 200k), 500
queries, 256 lists, 16 probes, pq_dim 14, 8 bits, recall@10 against the exact
k-NN (float64), random_state = seed.

| data | seed 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| HIGGS before | 0.8512 | 0.8510 | 0.8482 | 0.8512 | 0.8498 | 0.8503 |
| HIGGS after | 0.8604 | 0.8524 | 0.8506 | 0.8614 | 0.8550 | 0.8560 |
| taxi before | 0.8186 | 0.8270 | 0.8122 | 0.8372 | 0.8346 | 0.8259 |
| taxi after | 0.8352 | 0.8486 | 0.8602 | 0.8420 | 0.8306 | 0.8433 |

Mean recall rises on both datasets (+0.006 HIGGS, +0.017 taxi); 9 of 10
pairs rise, taxi seed 4 falls 0.004 (inside the seed spread). KEPT.
