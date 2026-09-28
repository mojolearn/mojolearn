# linear-apple2: Apple (Metal) speed round 2, the linear family

Brief: ~/mojolearn-evidence/apple2_speed_brief.md (2026-09-28 ~13:45Z).
Branch `lane/linear-apple2` off lane/apple-merged 037daa353. Round 1:
docs/lanes/progress/linear-apple.md. Jobs go through
`tools/apple_steward.py submit --kind speed --target <mac>`; before and after
arms are in the SAME job on the SAME Mac (each arm's files checked out and its
bindings rebuilt in turn). Job scripts: ~/mojolearn-evidence/linear-apple2/.

## Leads (from round 1)

- SGD family on Metal: 20x to 30x the one-core host (row-serial pass).
- Lasso / ElasticNet (solver CD, IDENTICAL): ~10 ms per epoch at 1M x 16.
  Round 1 found the epoch bound by the 1 x 1 x 1M dot's contract leaf chains
  (1024 chains of 977 serial steps), not by the six launches per coordinate.
- The Metal compiler crash seen on the M3 Ultra at 06ef7f558: recheck at
  96a7fe158 (the estimators binding rebuilt on m3ultra-b, job 0).

## Changes

| commit | what | mode | default | shared code |
|---|---|---|---|---|
| ae036928e | SPLITK leaf kernel on Apple loads 16 steps of operands ahead of its chain | IDENTICAL | on (`-D MOJOLEARN_APPLE_LEAF_PREFETCH_OFF=1` reverts) | YES: gemm/checks/gemm_identical.mojo, every PLAN_SPLITK caller on Apple |
| 62002dea3 | SPLITK leaf launch on Apple: 32 threads per block (8 -> 32 blocks at 1M) | IDENTICAL | on (same define) | YES, same file |
| 6cdbd32ab | x_linear SGD warp form: next row prefetched, folds interleaved, dead norms skipped | both | on | no (x_linear/sgd.mojo, GPU form only) |

## Jobs

| steward id | Mac | what |
|---|---|---|
| 1790603060531-speed-linear-037daa3530 | m3ultra-b | job 0: baseline board at 037daa353, estimators rebuild (crash recheck), qn_scalar_ieee_check |
| 1790603380973-speed-linear-62002dea30 | m4-a | leaf A/B: 037daa353 / ae036928e / 62002dea3, Lasso and ElasticNet 1M |
| 1790603575578-speed-linear-6cdbd32abe | m4pro-a | SGD A/B: 037daa353 / 6cdbd32ab, board + profile + 36-case Metal vs host bits |

## Results

(pending)
