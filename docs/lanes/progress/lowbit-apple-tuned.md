# lane/lowbit-apple-tuned: progress

Lane G of the low-bit project. The brief in force is
`~/mojolearn-evidence/lowbit-units/brief_current.md`. Files of this lane:
`gemm/checks/gemm_int15_apple_tuned.mojo`, its check, the clock
`bench/gemm_int15_apple_tuned_price_main.mojo`, the job
`tools/lowbit_apple_tuned/job.sh`. Boxes: m3ultra-b first, m2pro, through
the steward only.

## Jobs

| job | commit | box | request | state |
|---|---|---|---|---|
| 1 (13 variants: FOUR and THREE, staging load, windows, tiles) | 19ad540d3 | m2pro | 1790657025264 | gate RED only on the two 512-thread variants (see below); price running |
| 1 | 19ad540d3 | m3ultra-b | 1790657019939 | queued behind the orchestrator's lowbit-apple-now |

## Findings

- 2026-09-29, job 1 on m2pro, gate: 11 of 13 variants equal the oracle,
  the flat kernel and the simulation vectors on every case. The two
  512-thread variants (`f4.sg16.kb16`, `f3.sg16.kb16`, 4 x 4 simdgroups)
  NEVER WRITE their output (every poison survives, 266 cases each): the
  launch does not run on the M2 Pro. Both sabotage arms fail where they
  must; the chunk arm also fails `check_tuned_shapes_inside_one_chunk`,
  because of the same two variants (the reach check is red for that
  reason only). Next job drops the 512-thread variants.
