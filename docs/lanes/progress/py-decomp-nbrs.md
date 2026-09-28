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

PENDING: the shared NVIDIA pod went down at 19:19Z with the base-columns job
(nvc1-0031) queued; nothing is built or proven yet.

## DEVIATION changes

- Ledger row 139 (`_expansion_decomp.py` host control flow): the data-length
  keyed sorts of MinCovDet and non-metric MDS are no longer Python (narrowing
  pending proof).

## Unproven

Everything above until the pod job reports.
