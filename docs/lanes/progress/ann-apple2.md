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
| 513a3006e | IVF-PQ codebooks: subspace k-means side by side on pooled contexts | both | REVERTED (6f817e0ed): the process died in the codebook stage |
| 399de4811 | IVF-PQ encode: subspace codebook staged in threadgroup memory | both | on; `-D MOJOLEARN_PQ_ASSIGN_UNSTAGED` reverts |
| b9cc59dcf | x_ann/io.mojo: uploads without a private copy, downloads by memcpy | both | on |
| 7b51e91e5 | tiled k-NN fold width = d rounded up to a multiple of 4 (28, was 32) | both | on |
| e0d2ff366 | CAGRA device prune: neighbor rows staged in threadgroup memory | both | on |

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

### A/B 2: m4pro-a (Apple M4 Pro), steward 1790605523624, job at 19e0aabac

Arms f4545110c (base), bc3c22c03 (scan + ftz), 19e0aabac (all to date,
padded-group skip removed, FAST split opt-in). Three runs per cell (two
reps + the stage pass). Digests equal across the arms in every cell and
tier. Raw: `~/mojolearn-evidence/ann-apple2/ab2_m4pro-a_1790605523624.txt`.

| cell (s) | base | scan + ftz | 19e0aabac | digests |
|---|---|---|---|---|
| IDENTICAL CAGRA fit | 0.687/0.684/0.685 | 0.635/0.591/0.591 | 0.572/0.510/0.505 | 54d696296c9c7c8a / 45e435db03654b4e |
| IDENTICAL IVF-PQ fit | 7.996/7.982/8.011 | 7.519/7.209/7.229 | 7.198/7.223/7.183 | 6bb7a6c5fc753846 |
| IDENTICAL IVF-PQ search | 0.328/0.348/0.329 | 0.103/0.058/0.061 | 0.057/0.061/0.061 | 3b1e0c1ae73444eb |
| IDENTICAL t-SNE fit | 1.013/1.023/1.022 | 1.051/0.911/0.913 | 0.829/0.814/0.816 | 2bb1d3d75ffa1885 |
| FAST CAGRA fit | 0.467/0.446/0.441 | 0.449/0.447/0.447 | 0.404/0.399/0.401 | same |
| FAST IVF-PQ search | 0.375/0.357/0.334 | 0.108/0.057/0.060 | 0.060/0.057/0.060 | 4ace8e5f668db63d |
| FAST t-SNE fit | 0.733/0.692/0.698 | 0.697/0.696/0.692 | 0.632/0.652/0.644 | ca01838ecfb9306d |

Stage split (ms, IDENTICAL; base -> 19e0aabac): IVF scan select 264 -> 9.7,
probe 18 -> 4.9 (score 35 is now the largest); t-SNE iterations 854 -> 704,
symmetrize 112 -> 66; CAGRA k-NN + download + host prune 526 + 16 + 112 ->
k-NN + device prune 467; IVF-PQ build coarse 2850 -> 2540 and codebooks
4760 -> 4254 (cluster/ k-means, moved by the Apple ftz spelling only),
residuals 190, encode 184 (targets of 399de4811 and b9cc59dcf).

The MOJOLEARN_ANN_PQ_CB_STREAMS=4 arm died after the residual stage in both
tiers (no traceback); the trial is reverted.

### A/B 3: m4-a (Apple M4, 10-core GPU), steward 1790607052703, job at b9cc59dcf

Arms f4545110c (base), 19e0aabac, b9cc59dcf (+ staged encode, io). Digests
equal across arms in every cell and tier. Raw:
`~/mojolearn-evidence/ann-apple2/ab3_m4-a_1790607052703.txt`.

| cell (s) | base | 19e0aabac | b9cc59dcf |
|---|---|---|---|
| IDENTICAL IVF-PQ fit | 11.706/11.616/11.636 | 10.422/10.079/10.131 | 10.032/9.910/9.910 |
| IDENTICAL IVF-PQ search | 0.305/0.303/0.310 | 0.149/0.116/0.122 | 0.115/0.114/0.120 |
| IDENTICAL IVF-SQ fit / search | 4.483 / 0.349 | 3.814 / 0.152 | 3.718 / 0.147 |
| IDENTICAL IVF-RaBitQ fit / search | 4.324 / 0.330 | 3.632 / 0.140 | 3.612 / 0.127 |
| IDENTICAL CAGRA fit | 1.229/1.231/1.233 | 0.929/0.905/0.899 | 0.891/0.888/0.902 |
| IDENTICAL t-SNE fit | 1.510/1.514/1.515 | 1.258/1.174/1.178 | 1.165/1.164/1.168 |
| FAST IVF-PQ fit | 3.349/2.809/2.832 | 3.139/2.837/2.827 | 3.033/2.696/2.633 |
| FAST IVF-SQ fit | 1.213/1.207/1.211 | 1.204/1.211/1.197 | 1.113/1.126/1.116 |
| FAST IVF-* search | 0.33-0.35 | 0.12-0.16 | 0.12-0.15 |
| FAST CAGRA fit | 0.808/0.783/0.780 | 0.710/0.704/0.716 | 0.753/0.735/0.705 |
| FAST t-SNE fit | 1.017/0.929/0.924 | 0.880/0.880/0.881 | 0.841/0.843/0.877 |

Stages (ms, IDENTICAL, 19e0aabac -> b9cc59dcf): IVF-PQ residuals 157 -> 67,
encode 267 -> 197; IVF-SQ encode 148 -> 56; IVF-RaBitQ encode 34 -> 24.
CAGRA (FAST): k-NN 513 + download 16 + host prune 218 (base) -> k-NN +
device prune 675, so the device prune itself cost ~160 ms on the M4:
e0d2ff366 stages its reads. On the M4 the scan's select is fixed but the
search is 0.12 s (score and upload are the rest).
