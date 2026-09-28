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
