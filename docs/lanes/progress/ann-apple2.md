# ann-apple2: Apple (Metal) speed round 2, FAST and IDENTICAL

Lane `ann-apple2`, branch `lane/ann-apple2`, worktree `~/mojolearn-wt/ann-apple2`,
forked from lane/apple-merged 037daa353 (brief:
`~/mojolearn-evidence/apple2_speed_brief.md`). Family: IVF-Flat / PQ / SQ /
RaBitQ, refine, CAGRA, t-SNE. cluster/ k-means (the IVF coarse quantizer and
the PQ codebook k-means) belongs to cluster-apple2 and is not edited here.

Bench: `bench/speed/ann_cpu_speed.py` (round 1's shapes: IVF family 1M x 28
HIGGS, 1024 lists, 32 probes, 1000 queries, k = 10; CAGRA 50k; t-SNE 10k,
300 iterations), driven by `tools/ann_apple2_ab.sh`, which builds every arm
(before, intermediate commits, after) in both tiers in ONE steward speed job
on ONE Mac and alternates the arms, then prints a stage split
(`MOJOLEARN_ANN_STAGES=1`). Every line carries its digests.

## Changes

| commit | what | tier | default |
|---|---|---|---|
| f4545110c | opt-in stage marks inside the IVF scan search; the A/B script | both | marks off unless MOJOLEARN_ANN_STAGES |
| 74d074090 | IVF-PQ/SQ/RaBitQ search: probe walk and top-k a threadgroup per query (was one thread per query); NaN row takes the cell's sequential path | both | on; `-D MOJOLEARN_ANN_SERIAL_SCAN` reverts |
| bc3c22c03 | `ftz` on the Apple GPU: one exponent test (SHARED: every family's Apple IDENTICAL kernels) | IDENTICAL | on; `-D MOJOLEARN_FTZ_TWO_TEST` reverts |
| c3841ce42 | t-SNE repulsion + tiled k-NN: nonnegative flushes as one compare, reciprocal without operand flushes, padded column groups skipped | IDENTICAL (FAST: no-op) | on, except the padded-group skip (measured slower, removed in 19e0aabac) |
| b7f433a89 | t-SNE FAST: repulsion over candidate spans, partials joined in span order | FAST | OPT-IN since 19e0aabac (`-D MOJOLEARN_TSNE_FAST_SPLIT`): no gain measured |
| 95101d105 | t-SNE symmetrize (host): counting-pass CSR + per-row merge (host-checked SAME on 4 random graphs) | both | on |
| 782ad9a09 | CAGRA detour prune on the device (integer counts, rank placement) | both | on; `-D MOJOLEARN_CAGRA_HOST_PRUNE` reverts |

| 513a3006e | IVF-PQ codebooks: subspace k-means side by side on pooled contexts | both | opt-in trial, `MOJOLEARN_ANN_PQ_CB_STREAMS=k` |

## Measurements

### A/B 1: m4pro-b (Apple M4 Pro), steward 1790604321939, job at b8ef52742

Arms in one job, alternated, two reps each (seconds, rep1/rep2). Digests
equal across all five arms for every IDENTICAL cell and every FAST cell
except FAST t-SNE (the span split, b7f433a89, moves its bits by design).
Raw: `~/mojolearn-evidence/ann-apple2/ab1_m4pro-b_1790604321939.txt`.

IDENTICAL:

| cell | f4545110c base | 74d074090 scan | bc3c22c03 ftz | c3841ce42 flush | b8ef52742 (+sym, prune) | digests |
|---|---|---|---|---|---|---|
| IVF-Flat fit | 2.349/2.304 | 2.342/2.293 | 2.248/1.997 | 2.026/1.994 | 2.042/1.997 | d730b8082a1cdbfb |
| IVF-PQ fit | 6.478/6.398 | 6.431/6.406 | 6.022/5.746 | 5.743/5.755 | 5.766/5.752 | 6bb7a6c5fc753846 |
| IVF-PQ search | 0.291/0.275 | 0.106/0.054 | 0.073/0.073 | 0.053/0.056 | 0.053/0.054 | 3b1e0c1ae73444eb |
| IVF-SQ fit | 2.451/2.440 | 2.428/2.466 | 2.141/2.143 | 2.126/2.137 | 2.129/2.147 | c55eedfcb6459d7b |
| IVF-SQ search | 0.296/0.288 | 0.064/0.083 | 0.072/0.063 | 0.083/0.085 | 0.062/0.063 | 1d9c53fd8c13f452 |
| IVF-RaBitQ fit | 2.253/2.251 | 2.257/2.256 | 1.953/1.953 | 1.944/1.949 | 1.969/1.963 | a4f2343eb268b0a7 |
| IVF-RaBitQ search | 0.275/0.275 | 0.052/0.052 | 0.069/0.052 | 0.052/0.052 | 0.052/0.052 | 056a570709713477 |
| refine search (top-40) | 0.317/0.316 | 0.084/0.109 | 0.086/0.100 | 0.083/0.084 | 0.084/0.083 | 3c73bf8ae59e47e4 |
| CAGRA fit | 0.619/0.618 | 0.627/0.620 | 0.570/0.547 | 0.621/0.583 | 0.541/0.544 | 54d696296c9c7c8a / 45e435db03654b4e |
| t-SNE fit | 0.919/0.922 | 0.920/0.918 | 0.945/0.824 | 0.796/0.781 | 0.708/0.711 | 2bb1d3d75ffa1885 |

FAST (IVF fits and searches: the same pattern, search 0.28-0.32 -> 0.05-0.09;
fits unchanged, as expected: ftz and the flushes are IDENTICAL-only):

| cell | base | scan | ftz | flush | after |
|---|---|---|---|---|---|
| IVF-PQ search | 0.307/0.283 | 0.106/0.053 | 0.052/0.052 | 0.069/0.053 | 0.053/0.054 |
| CAGRA fit | 0.405/0.408 | 0.405/0.407 | 0.415/0.403 | 0.554/0.516 | 0.474/0.466 |
| t-SNE fit | 0.650/0.634 | 0.623/0.624 | 0.624/0.624 | 0.638/0.651 | 0.645/0.632 (split) |

Reading: the scan split is the big win (IVF quantized searches 5x); the
Apple ftz spelling cuts every IDENTICAL IVF fit 5 to 13% (it reaches the
cluster/ k-means as shared code) and moves no digest; the t-SNE flushes and
the symmetrize take IDENTICAL t-SNE 0.92 -> 0.71. The k-NN padded-group
skip in c3841ce42 made CAGRA slower in both tiers (FAST 0.405 -> 0.55) and
was removed; the FAST t-SNE span split gained nothing and is opt-in now.
(The stage split and the FAST quality pass of this job did not run: a
script bug, fixed in 19e0aabac.)
