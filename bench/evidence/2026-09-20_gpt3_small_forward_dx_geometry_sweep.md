# GPT-3-small IDENTICAL forward/input-gradient GEMM geometry sweep

## Contract and workload

- FP32 `MOJOLEARN_NUMERIC_IDENTICAL`; every candidate preserves the existing
  per-cell reduction order.
- One layer is represented by eight `(m,768,768)` calls, three
  `(m,3072,768)` calls, and three `(m,768,3072)` calls, for token-row counts
  2,048, 8,192, and 32,768. These weights cover Q/K/V/O forward and input
  gradients plus the gated-MLP up/down forward and input gradients. Weight
  gradients remain outside this lane.
- Each timing is five samples interleaved with the shipped selector after
  warm-up. Comparisons below sum the per-shape medians using those call weights.
- Full output comparison, not sampling, reported `mismatches_vs_plan10 0` for
  all 99 candidate/shape cases. Output hashes were stable within each shape.

## Apple baseline

Apple M4 shipped weighted layer time and useful throughput:

| rows | weighted time | useful TFLOP/s |
|---:|---:|---:|
| 2,048 | 407.740 ms | 0.190 |
| 8,192 | 1,776.467 ms | 0.174 |
| 32,768 | 7,162.130 ms | 0.173 |

The local baseline establishes the portable workload and hashes. Candidate
qualification was narrowed only after the NVIDIA sweep; no structurally
different candidate survived to justify another long Apple matrix.

## NVIDIA geometry sweep

Device: NVIDIA L40S, driver 580.159.03. Source base: `71d7a6f69`. Ratios are
weighted candidate time / interleaved shipped time, so lower is faster.

| exact geometry | rows 2,048 | rows 8,192 | rows 32,768 |
|---|---:|---:|---:|
| kpack_hg (geometry 18) | 0.986 | 0.997 | 1.004 |
| kpack_gs (17) | 1.065 | 1.121 | 1.109 |
| kpack_hf (16) | 1.092 | 1.089 | 1.101 |
| kpack_padv (15) | 1.162 | 1.208 | 1.211 |
| kpack_pad (14) | 1.171 | 1.214 | 1.224 |
| kpack (10) | 1.747 | 1.907 | 1.875 |
| kpack_wide (11) | 2.791 | 2.842 | 2.536 |
| quarter (4) | 2.006 | 2.311 | 2.350 |
| half (2) | 3.699 | 3.898 | 3.922 |
| half_ks16 (3) | 3.678 | 3.916 | 3.911 |

Geometry 18 is the body already selected by the shipped tuned path for this
workload. Its roughly -1.4% to +0.4% ratios are timing noise from invoking the
same computation through the trial entry point. Every different geometry is
6.5% to 292% slower in the weighted layer mix and has no compensating memory
benefit (all use one workspace float).

## Disposition and resources

No production change is promoted, and no AMD rental is warranted: there is no
portable candidate to qualify. This avoids inferring an AMD plan from NVIDIA
and avoids spending vendor time on a candidate already rejected at its source
bottleneck. Raw NVIDIA logs are retained in the session evidence directory
`mojolearn-evidence/gpt3-gemm-forward-dx-nvidia/results`; the benchmark harness
now exposes the weighted workload and named exact geometries for a future
architectural change.

Owned RunPod `39b8xwhu6ewukp` was deleted after evidence retrieval (`DELETE
204`) and verified absent (`GET 404`). The unrelated attention pod was not
touched.
