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
| 1 (13 variants: FOUR and THREE, staging load, windows, tiles) | 19ad540d3 | m2pro | 1790657025264 | RED only by the two 512-thread variants (never ran on the M2 Pro); every other variant equal; both arms seen failing |
| 1 | 19ad540d3 | m3ultra-b | 1790657019939 | GREEN, every variant; the LAST M3 Ultra job (the box is released) |
| 2 (TWO per-step and deferred carry, small tiles, launch bound, 512-thread probe) | 8fcd8be81 | m2pro | 1790657914702 | RED by my f3 staging slip (fixed d69b5917b); every other variant equal; f2.t32.kb16 1.01-1.02 |
| 3 (fragments from float32 planes in device memory) | d69b5917b | m2pro | 1790658417171 | GREEN; the device-fragment lever costs (1.59 vs 1.40) |
| 4 (TWO's tiles and windows; launcher refusal) | 897fed01d | m2pro | 1790659592875 | GREEN; 512-thread tiles REFUSED; nothing beats f2.t32.kb16 on the worst row |
| 5 (prefetch) | 16bb4bc6b | m2pro | 1790660873689 | GREEN; prefetch costs (1.26) |
| 6 (best forms on every row) | efb596bcd | m2pro | 1790661638487 | running |

Nothing more goes to m3ultra-b (the orchestrator, 2026-09-29 ~05:00Z).

## Tables (complete inference call over fp32.v1, same run, qkv / mlp_up / mlp_down / lm_head t512)

M3 Ultra, job 1 (`bench/results/lowbit_apple_tuned/job1_19ad540d3/m3ultra/table_inference.md`):
best f4.t32.kb16 2.04 / 2.12 / 2.38 / 2.12; f3.t32.kb16 1.97 / 2.17 / 2.41 / 2.14;
untuned four 3.48 / 3.40 / 3.65 / 3.40. The orchestrator's lowbit-apple-now
(four-code staging on lane C's plan) read 2.90 / 2.81 / 2.99 / 2.84.

M2 Pro, job 1 (`.../job1_19ad540d3/m2pro/table_inference.md`):
best f4.t32.kb32 1.53 / 1.54 / 1.57 / 1.54; untuned four 3.57 / 3.44 / 3.50 / 3.53.
(The M2 Pro's fp32.v1 is about five times the M3 Ultra's time.)

## Findings

- 2026-09-29, job 1 on m2pro, gate: 11 of 13 variants equal the oracle,
  the flat kernel and the simulation vectors on every case. The two
  512-thread variants (`f4.sg16.kb16`, `f3.sg16.kb16`, 4 x 4 simdgroups)
  NEVER WRITE their output (every poison survives, 266 cases each): the
  launch does not run on the M2 Pro. Both sabotage arms fail where they
  must; the chunk arm also fails `check_tuned_shapes_inside_one_chunk`,
  because of the same two variants (the reach check is red for that
  reason only). Next job drops the 512-thread variants.

## Lever table, M2 Pro, complete inference call over fp32.v1 (worst of the four t512 rows)

| lever (one at a time) | best variant | worst-row ratio | job |
|---|---|---|---|
| start line: lane C's FOUR untuned | int15i64.v1.apple.four | 3.57 | 1 |
| four-code staging load, 64x64 tile | f4.w64.kb16 | 3.45 | 1 |
| 32x32 tile | f4.t32.kb16 | 1.81 | 1 |
| 32-step window | f4.t32.kb32 | 1.57 | 1 |
| THREE products (float unit) | f3.t32.kb16 | 1.75 | 1 |
| TWO products, carry every step in 16-bit halves | f2.t32.kb16 | 1.02 | 2 |
| TWO, deferred carry (runs flushed every 64 steps) | f2d.t32.kb32 | 1.43 | 2 |
| declared launch bound 128 | f2.t32.kb16.b128 | 1.03 | 4, 5 |
| fragments from float32 planes in device memory | dev.f2d.t32 | 2.18 | 3 |
| other tiles (64x32, 32x64, 16x16, 256 threads), window 8 | f2.t64x32.kb16 | 1.15 | 4 |
| prefetch of the next window | f2.t32.kb16.pf | 1.28 | 5 |

M3 Ultra (job 1 only, before the box was released): best f4.t32.kb16 2.04 / 2.12 / 2.38 / 2.12.
Form TWO was never timed on the M3 Ultra in this lane.

## Bugs found and fixed

- The handover commit staged THREE's low plane over its sums plane (job 2's gate caught it on every case; fixed d69b5917b).
- A 512-thread block of 2 x 2 fragments launches nothing and reports nothing on the M2 Pro (the pipeline's thread limit falls with registers; Metal does not answer the attribute query): the launchers now refuse a block above 256 threads when the limit cannot be read (seen refusing in job 4).
- The generic kernel pointer passed to the device simdgroup load fails in the Metal compiler; cast to AddressSpace.GLOBAL first (job 3's probes).
