# py-decomp-nbrs: progress

Lane `py-decomp-nbrs` (Python-work fix lanes, ~/mojolearn-evidence/py_work_brief.md;
audit ~/mojolearn-evidence/python_work_audit.md: decomp, neighbors, SVM, ann).
Worktree `~/mojolearn-wt/py-decomp-nbrs`, branch `lane/py-decomp-nbrs`, base
lane/apple2-merged 342469dae. Sub-lanes (own worktrees and branches, merged
here when proven): `lane/py-dn-svm` (SVC inference and Platt),
`lane/py-dn-ann` (resident ANN indexes, DEVIATION 1804; distributed IVF merge),
`lane/py-dn-kern` (KernelPCA/OCSVM/SVGP fused items; spectral zero scan and
precomputed kNN).

## Changes on this branch

| item | audit | what moved | reference arm |
|---|---|---|---|
| ParallelQueries | neighbors 1, rank 5 | the estimator (whole index) travels once per worker per `query()` (`neighbor_state`), not once per 128-row shard; the worker keeps it and its resident index handle under a per-call token | none needed (same shard cut, call and merge) |
| MinCovDet / EllipticEnvelope fast_mcd | decomp 1, rank 3 | one `x_decomp_mcd` call (x_decomp/mcd.mojo over x_decomp/kit.mojo): every C-step, trial, argsort and gather | `MOJOLEARN_XD_MCD_PYTHON=1` |
| LDA online / partial_fit | decomp 3, rank 8 | one `x_decomp_lda_online` call per pass (x_decomp/lda_online.mojo) | `MOJOLEARN_XD_LDA_PYTHON=1` |
| LDA E-step on GPU | decomp 9, rank 8 | `x_decomp_dev_lda_rows`: X, EW, Dt, Et stay on the device (was 12 GB of round trips per iteration at 1M x 1000) | the host-address entry (non-resident kit) |
| non-metric MDS | decomp 7, rank 9 | pair gather, (x, y) order, isotonic fit/predict and scatter/mirror on buffers (x_decomp/moves.mojo, direct x_linear calls) | `MOJOLEARN_XD_MDS_PYTHON=1` |

Executors: `Kit[E, S]` sends a call whose largest operand has at least
MOJOLEARN_XD_RES_DEV_MIN (default 65536) elements to the GPU executor and
smaller ones to the host executor inside the GPU binding (the same cells; the
x-decomp lanes hold GPU == CPU). NaN distances/draws/dissimilarities are
refused by the native orders (Python's tuple sort of a NaN is not a total
order); no finite input reaches one.

## Proof and timing

STOPPED by order (2026-09-28 ~19:53Z: all py-* checking stopped, jobs cancelled;
py-consolidated merges the Python lanes and runs ONE global check). What exists:

- Both x_decomp bindings (`_mojolearn_x_decomp`, `_mojolearn_x_decomp_host`)
  BUILD on nvc1 (x86, 2x A40) at this branch.
- CPU column, native entry vs its Python reference arm in ONE tree (pod nvc1,
  `sh`, `MOJOLEARN_VENDOR=cpu`, ~/mojolearn-evidence/py-decomp-nbrs/tools/ab_arms.py),
  output sha256 of the fitted attributes:

| case | native | python arm | verdict | native s | python s |
|---|---|---|---|---|---|
| MinCovDet 400x6 (n <= 500 path) | f63fea263810a423 | f63fea263810a423 | SAME | 0.19 | 0.22 |
| EllipticEnvelope 400x6 | b84bfe654140796f | b84bfe654140796f | SAME | 0.07 | 0.12 |
| MinCovDet 1200x5 (n < 1500 path) | e665ffdd95fb745a | e665ffdd95fb745a | SAME | 0.32 | 1.98 |
| MinCovDet 5000x8 (full path) | fedc98556e0495ff | fedc98556e0495ff | SAME | 0.82 | 5.65 |
| LDA online 3000x200 | d8ebf4c7c3321695 | d8ebf4c7c3321695 | SAME | 2.23 | 2.19 |
| LDA partial_fit x2 | a22d384d8394fe7b | a22d384d8394fe7b | SAME | 0.18 | 0.21 |
| MDS non-metric 300x4, 15 it | abcef8fcd816cf84 | abcef8fcd816cf84 | SAME | 0.61 | 3.78 |

NOT RUN: the GPU column, base-vs-head lane digests (x-decomp-* lanes,
par-queries-*), GPU == CPU lane checks, the dev_lda_rows sabotage arm, and
large-shape timing (bench_decomp.py). ParallelQueries is unproven.

## DEVIATION changes

- Ledger row 139 (`_expansion_decomp.py` host control flow): the data-length
  keyed sorts of MinCovDet and non-metric MDS are no longer Python (narrowing
  pending proof).

## Unproven

Everything except the CPU same-tree SAME rows above. Branches to merge
(py-consolidated): lane/py-decomp-nbrs, lane/py-dn-svm, lane/py-dn-ann,
lane/py-dn-kern (each unproven; see their progress files).
