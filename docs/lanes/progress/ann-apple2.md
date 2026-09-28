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
| c3841ce42 | t-SNE repulsion + tiled k-NN: nonnegative flushes as one compare, reciprocal without operand flushes, padded column groups skipped | IDENTICAL (FAST: no-op) | on |
| b7f433a89 | t-SNE FAST: repulsion over candidate spans, partials joined in span order | FAST | on; `-D MOJOLEARN_TSNE_FAST_SPLIT_OFF` reverts |
| 95101d105 | t-SNE symmetrize (host): counting-pass CSR + per-row merge (host-checked SAME on 4 random graphs) | both | on |
| 782ad9a09 | CAGRA detour prune on the device (integer counts, rank placement) | both | on; `-D MOJOLEARN_CAGRA_HOST_PRUNE` reverts |

## Measurements

(pending)
