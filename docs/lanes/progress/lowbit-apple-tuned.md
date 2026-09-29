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
| 2 (TWO per-step and deferred carry, 16x16 / 32x16 tiles, launch bound 128, 512-thread probe) | 8fcd8be81 | m2pro | 1790657914702 | queued |
| 3 (fragments from float32 planes in device memory; devload spelling probes) | committed, not submitted | m2pro | | waits for job 2's build |

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
