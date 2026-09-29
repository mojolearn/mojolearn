# H100 table of record: the fifteen-bit profile on the tuned plan (run 5)

Box: H100 NVL, pod nvc3. Job nvc3-0028, commit 99ae7bb66 (`tools/lowbit_int15/run5_job_h100.sh`),
2026-09-29 04:38Z to 04:46Z, exit 0. Profile `mojolearn.identical.gemm.int15i64.v1`.
Full log: `runs/h100_run5_nvc3-0028_99ae7bb66.txt`; full tables: `TIMING_h100_run5.md`.

Gates in the same job, before the clock: `gate` GREEN (3 clean arms pass, 4 sabotage arms
fail the checks they must) and `tuned_gate` GREEN (tuned passes; tuned, pieces, epilogue and
host sabotage arms each fail `check_int15_tuned_matches_oracle` and
`check_int15_tuned_planted_worst_cases`). In the timing run all 50 tuned-plan digests
equal the reference unit plan's digest at their row; the warm and timed runs printed the
same 372 digests; the sabotaged build changed every fifteen-bit digest.

Method: one job alone on the box; build, one untimed run of every arm, then the timed run
of the same binary; median of 5 timed calls. The alternation runs in two blocks, so no arm
follows a launch of hundreds of milliseconds. `over` = fifteen-bit time over fp32.v1's at
the same row IN THIS RUN; above 1 took longer.

## The complete inference call, tuned plan

Complete call = the activations to planes (parallel quantizer), Lane D's four products with
one staging, then this profile's epilogue (Int64 recombination, pinned Int64 to float32
seam, scale). Weights packed once, outside the call.

| row | m x n x k | fp32.v1 ms | fifteen-bit complete call ms | over |
|---|---|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.0715 | 0.1602 | 2.241 |
| qkv.t8 | 8 x 4096 x 4096 | 0.1020 | 0.1722 | 1.688 |
| qkv.t512 | 512 x 4096 x 4096 | 1.0324 | 0.4993 | 0.484 |
| mlp_up.t1 | 1 x 14336 x 4096 | 0.1802 | 0.1613 | 0.895 |
| mlp_up.t8 | 8 x 14336 x 4096 | 0.2725 | 0.1863 | 0.684 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.4145 | 1.6924 | 0.496 |
| mlp_down.t1 | 1 x 4096 x 14336 | 0.1693 | 0.4552 | 2.689 |
| mlp_down.t8 | 8 x 4096 x 14336 | 0.2883 | 0.4965 | 1.722 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.4251 | 1.4638 | 0.427 |
| lm_head.t1 | 1 x 128256 x 4096 | 1.2450 | 1.1134 | 0.894 |
| lm_head.t8 | 8 x 128256 x 4096 | 2.3660 | 1.2500 | 0.528 |
| lm_head.t512 | 512 x 16032 x 4096 (n capped) | 4.0654 | 1.9440 | 0.478 |

At the 512-token rows: 0.43 to 0.50. At the decode rows t1 and t8 the call takes longer
than fp32.v1 for qkv and mlp_down (1.69 to 2.69) and less for mlp_up and lm_head.

## Three products of one layer, each timed alone and added, tuned plan

Forward product, input gradient, weight gradient; each a complete product (both operands
converted per call). The sum of three operations measured one at a time, NOT a training step.

| layer | fp32.v1, three products ms | fifteen-bit, three products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9320 | 2.2425 | 0.765 |
| mlp_up.t512 | 9.7571 | 7.4621 | 0.765 |
| mlp_down.t512 | 9.9499 | 7.2616 | 0.730 |
| lm_head.t512 | 11.2257 | refused (input gradient k = 128256 > INT15_MAX_K) | refused |

Per product (over): forward 0.600 to 0.671, input gradient 0.679 to 0.742, weight
gradient 0.884 to 0.978. The weight gradient is the product the epilogue fold (task 4)
aims at.

## Beside it: run 4 (nvc3-0026, 985162978), the disturbed run

Run 4's fp32.v1 at mlp_up.t512 read 6.1283 ms against 3.4145 here and at lm_head.t512
5.5734 against 4.0654, so its 0.31 to 0.48 divided by a disturbed reference. The tuned
complete call itself moved little between the runs (qkv 0.4988 / 0.4993, mlp_down
1.4793 / 1.4638). Run 5's column is the one of record.
