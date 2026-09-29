# lowbit-units: timing of the low-bit GEMM plans, and the Apple exact-chunk probe

## FOR LANE C, FIRST: the Apple exact-chunk probe's gate verdict

**PASSED on the M2 Pro** (`m2pro`, macOS 26.7), commit 304d94f72, steward
request 1790650569362, 2026-09-29T02:57Z. ONE Apple generation; it has not
run on a second.

- int8 codes held as float32 on Metal's float matrix unit, `k` in chunks of
  1024 steps (bound 1040: `16129 * 1040 < 2^24`), chunk sums in Int32.
- Bits equal to the flat int8 kernel AND the host oracle on 21 shapes of
  quantized fixtures and 110 planted worst cases (k to 131072), two tiles.
- The arm that removes the chunk boundary FAILED 54 of 110 planted cases;
  the value arm failed 110 of 110.
- At the twelve transformer rows its digests equal the H100's and the
  MI325X's integer-unit digests.
- FOR A WIDER CODE THE BOUND IS NOT 1040. It is
  `floor((2^24 - 1) / max|product|)`: for an int8 PIECE product (both
  pieces in [-128, 127], so at most 128 * 128 = 16384) it is 1023 steps, not
  1040. `gemm/checks/gemm_int8_apple_chunk.mojo` asserts its own bound at
  compile time from `INT8_PRODUCT_MAX`; a kernel with other operands must
  state its own.
- Kernel `gemm/checks/gemm_int8_apple_chunk.mojo`, gate
  `gemm/checks/gemm_int8_apple_chunk_check.mojo`, files
  `bench/results/lowbit_units/2026-09-29/m2pro/chunk*`.

Lane `lowbit-units`, branch `lane/lowbit-units`, worktree
`~/mojolearn-wt/lowbit-units`. Brief `~/mojolearn-evidence/lowbit-units/brief.md`,
plan `docs/lanes/LOWBIT_UNITS_PLAN.md`. Lane files
`~/mojolearn-evidence/lowbit-units/`. Nothing here merges to main and no
default moves: the probe is not in any dispatcher.

## Decisions recorded (from the brief)

- AMD IS IDENTITY ONLY (Andrew, 2026-09-29: "maybe don't check for speed of
  amd box? just the bitwise idenity?"). On `do-amd` every arm runs once for
  its digest and the gate verdicts; nothing is timed, no conversion cost rows,
  no vendor arm. The speed gate is judged on NVIDIA and on Apple. The table's
  AMD time columns read "not timed (identity only)".
- The Apple M3 Ultra: the brief's last word is "do not submit to it until the
  orchestrator says it is ready". NOTHING WAS SUBMITTED TO IT. See "Owed".

## What exists now

| Piece | File |
|---|---|
| The timing harness | `bench/gemm_lowbit_price_main.mojo` (shapes from `bench/gemm_shapes.mojo`, helpers from `bench/gemm_price_main.mojo`) |
| Identity-only mode | `MOJOLEARN_LOWBIT_PRICE_IDENTITY_ONLY=1` (every arm once, nothing timed) |
| The table and the digest comparison | `tools/lowbit_units/table.py` (`--identity-only BOX`, `--expect-disagree`) |
| The vendor comparison arms | `tools/vendor_gemm_price.py --bf16 --only llama8b` (COMPARISON ONLY) |
| The Apple exact-chunk probe | `gemm/checks/gemm_int8_apple_chunk.mojo` |
| Its gate | `gemm/checks/gemm_int8_apple_chunk_check.mojo`, `pixi run check-gemm-int8-apple-chunk[-sabotage|-value-sabotage]` |
| Job scripts | `tools/lowbit_units/{box,gate,price,chunk,vendor}_job.sh` |
| Results | `bench/results/lowbit_units/2026-09-29/` (`TABLE.md` and each box's files) |

Shapes: the twelve OP_NT transformer rows. Decode = t1 and t8, training =
t512. `k` is never capped; the default budget (2^35 multiply-accumulates)
caps one row, the head at t512, to n = 16032. The H100 also ran that row
whole (`h100/lowbit_whole_head.tsv`).

## What ran on which box

| Box | Job | Commit | What | Verdict |
|---|---|---|---|---|
| H100 (nvc3) | nvc3-0005 | 028a1d772 | existing low-bit gate and its two sabotage arms, forced-flat run | GREEN |
| H100 | nvc3-0006 | f2ba56e9d | vendor strict fp32, TF32, bf16 (cuBLAS, torch 2.13.0+cu129) | exit 0 |
| H100 | nvc3-0007 | 3a956f14a | timing harness run 1 and its sabotage arm | exit 0 |
| H100 | nvc3-0008 | 50ae5ae27 | head at t512 whole | timing exit 0; JOB FAILED, see Failures 1 |
| H100 | `run` | 22ebe3f81 | identity-only mode: 120 digests equal run 1's | exit 0 |
| H100 | nvc3-0009 | 304d94f72 | timing harness run 2 (the run of record) | exit 0 |
| MI325X (do-amd) | 1790650025428 | 50ae5ae27 | gate, timing, vendor. Submitted and RUNNING before the identity-only decision reached the lane, so it finished. Its times and its vendor rows are NOT reported and are not in the repository; its gate verdict is | gate GREEN |
| MI325X | 1790650573197 | 304d94f72 | identity only (the run of record), 39 s | exit 0 |
| M2 Pro (m2pro) | 1790650028142 | 50ae5ae27 | gate, chunk, timing run 1, vendor (MPS, torch 2.13.0) | gate NOT RUN (Failures 2), chunk GREEN, timing exit 0 with wrong conversion digests (Failures 3), vendor exit 0 |
| M2 Pro | 1790650569362 | 304d94f72 | gate, chunk, timing run 2 (the run of record) | gate GREEN, chunk GREEN, timing exit 0 |

The H100 tree is patch-synced, so its `tree_head` is the merge base and the
lane's commit lies over it as a patch; the commit named is the lane's HEAD at
the sync.

## Times (run 2; one run per box, 5 timed calls per arm, median)

`over` is the arm's median over fp32.v1's at the same shape on the same box;
above 1 the arm took longer. `+q` adds the per-call quantization of the
activations. The whole table, with minima, rates and every arm, is
`bench/results/lowbit_units/2026-09-29/TABLE.md`.

### H100

| shape | fp32.v1 ms | bf16 dispatched ms | over | int8 on the unit ms | over | +q over |
|---|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 0.0707 | 0.7907 (fused) | 11.184 | 0.1573 | 2.225 | 9.158 |
| qkv.t8 | 0.1008 | 0.1505 | 1.493 | 0.1478 | 1.466 | 7.409 |
| qkv.t512 | 1.0328 | 1.0903 | 1.056 | 1.4361 | 1.390 | 3.938 |
| mlp_up.t1 | 0.1904 | 0.7996 (fused) | 4.200 | 0.1933 | 1.015 | 3.650 |
| mlp_up.t8 | 0.2722 | 0.4527 | 1.663 | 0.2103 | 0.773 | 3.025 |
| mlp_up.t512 | 3.3374 | 3.5206 | 1.055 | 4.6300 | 1.387 | 2.179 |
| mlp_down.t1 | 0.1737 | 2.7585 (fused) | 15.881 | 0.6498 | 3.741 | 13.738 |
| mlp_down.t8 | 0.2868 | 0.4656 | 1.623 | 0.6992 | 2.438 | 10.880 |
| mlp_down.t512 | 3.3603 | 3.5507 | 1.057 | 5.5780 | 1.660 | 4.631 |
| lm_head.t1 | 1.2066 | 2.8032 | 2.323 | 0.9531 | 0.790 | 1.206 |
| lm_head.t8 | 2.1553 | 3.7610 | 1.745 | 1.3711 | 0.636 | 0.919 |
| lm_head.t512 (n capped to 16032) | 3.7224 | 3.9206 | 1.053 | 5.1254 | 1.377 | 2.056 |
| lm_head.t512 whole (run 1) | 27.9305 | 30.8343 | 1.104 | 40.1179 | 1.436 | 1.531 |

Rates at qkv.t512: fp32.v1 8.3 T MAC/s, bf16 widen 7.9, int8 on the unit
6.0, int8 flat 0.22, bf16 fused 0.22.

### M2 Pro

| shape | fp32.v1 ms | bf16 dispatched ms | over | int8 flat (dispatched) ms | over | Apple probe ms | over | +q over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 2.0650 | 1.8670 (fused) | 0.904 | 1.7920 | 0.868 | 1.2530 | 0.607 | 1.477 |
| qkv.t8 | 1.5080 | 2.0470 | 1.357 | 9.7300 | 6.452 | 0.8230 | 0.546 | 1.461 |
| qkv.t512 | 17.8000 | 18.2540 | 1.026 | 515.1990 | 28.944 | 8.5930 | 0.483 | 0.680 |
| mlp_up.t1 | 5.5180 | 5.3520 (fused) | 0.970 | 5.1680 | 0.937 | 2.1880 | 0.397 | 0.666 |
| mlp_up.t8 | 4.1090 | 5.9930 | 1.459 | 29.4200 | 7.160 | 2.0040 | 0.488 | 0.820 |
| mlp_up.t512 | 60.4090 | 62.3960 | 1.033 | 1799.2350 | 29.784 | 27.9870 | 0.463 | 0.521 |
| mlp_down.t1 | 5.5890 | 5.1340 (fused) | 0.919 | 4.4290 | 0.792 | 2.4550 | 0.439 | 1.203 |
| mlp_down.t8 | 4.9750 | 6.8790 | 1.383 | 33.9330 | 6.821 | 2.4370 | 0.490 | 1.375 |
| mlp_down.t512 | 65.7240 | 66.8690 | 1.017 | 1828.9050 | 27.827 | 29.9820 | 0.456 | 0.639 |
| lm_head.t1 | 33.1620 | 50.1550 | 1.512 | 32.6210 | 0.984 | 11.9000 | 0.359 | 0.400 |
| lm_head.t8 | 33.2310 | 50.3410 | 1.515 | 252.5700 | 7.600 | 11.8680 | 0.357 | 0.398 |
| lm_head.t512 (n capped to 16032) | 67.9340 | 69.7870 | 1.027 | 2011.2770 | 29.606 | 31.3130 | 0.461 | 0.512 |

Rates at qkv.t512: fp32.v1 0.48 T MAC/s, the Apple probe 1.00, int8 flat
0.017.

### MI325X

Not timed (identity only).

### The conversions (H100 / M2 Pro, ms)

| row | qkv.t512 | mlp_down.t512 | lm_head.t8 (n = 128256) |
|---|---:|---:|---:|
| quantize activations, m x k, PER CALL | 2.47 / 3.50 | 9.96 / 12.01 | 0.61 / 1.38 |
| pack weights to int8, n x k, once | 2.48 / 3.85 | 9.97 / 13.87 | 12.15 / 96.29 |
| pack weights to bf16, n x k, once | 0.065 / 1.05 | 0.199 / 2.99 | 1.67 / 24.88 |
| widen bf16 to float32, n x k | 0.061 / 0.73 | 0.187 / 2.15 | 1.61 / 17.39 |
| dequantize int8 to float32, n x k | 0.078 / 3.70 | 0.246 / 13.20 | 2.10 / 146.98 |

(H100 numbers from run 1's equal rows where run 2 is within 1%; the
committed table is run 2.)

### Where a low-bit plan costs more time than fp32.v1, said plainly

- H100, training rows (t512): EVERY low-bit plan costs more. bf16 widen 1.05
  to 1.06 times fp32.v1's time; int8 on the integer unit 1.38 to 1.66; with
  the activations quantized per call 2.06 to 4.63. The flat plans 38 to 42.
- H100, decode rows: int8 on the unit costs LESS than fp32.v1 at three rows
  only (mlp_up.t8 0.773, lm_head.t1 0.790, lm_head.t8 0.636) and more at
  the other five; once the activations are quantized per call it costs less
  at ONE row (lm_head.t8, 0.919). bf16 costs more at every decode row, and
  the plan its dispatcher picks at t1 (fused, below 16384 cells) costs 4 to
  16 times fp32.v1 where the widen plan costs 1.7 to 2.0.
- M2 Pro, the plans that are dispatched today: bf16 costs more at nine of
  twelve rows (less at the three t1 rows below the head, 0.90 to 0.97); int8
  flat costs 6.5 to 30 times fp32.v1 at every t8 and t512 row.
- M2 Pro, the Apple probe (not dispatched): 0.36 to 0.61 times fp32.v1's
  time at all twelve rows; with the activations quantized per call 0.40 to
  0.82 at eight rows and 1.20 to 1.48 at four decode rows (qkv.t1, qkv.t8,
  mlp_down.t1, mlp_down.t8).
- THE QUANTIZER. `quantize_rows_int8_kernel` is one thread per row. On the
  H100 quantizing the activations takes longer than the int8 product itself
  at every row but the head (qkv.t512: 2.47 ms against 1.44; mlp_down.t512:
  9.96 against 5.58). It is the largest int8 cost measured.

### Vendor library, COMPARISON ONLY (no digest, no identity claim)

fp32.v1 over the vendor's strict fp32 at the same shape on the same box:
H100 (cuBLAS) qkv.t512 2.46, mlp_up.t512 2.02, mlp_down.t512 2.42, head t512
whole 1.81; M2 Pro (MPS) qkv.t512 4.94, mlp_up.t512 5.81, mlp_down.t512
5.91. Vendor times, ms, strict / TF32 / bf16: H100 qkv.t512 0.4199 / 0.0789
/ 0.0419; M2 Pro qkv.t512 3.6036 / no TF32 / 5.7360. AMD: not run (identity
only).

### How much these times can carry

One run per box, five timed calls per arm; median and minimum differ by
under 1% at most rows. One call and one wait per sample, so the decode rows
are mostly launch and wait. The M2 Pro's first row (qkv.t1) has a median
2.07 against a minimum 1.54 for fp32.v1: that row is UNDERPOWERED and no
conclusion is drawn from its ratios. A ratio within a few percent of 1
(bf16 widen at t512 on either box) is not a measured difference in either
direction beyond the widening step the plan adds by construction.

## Digests across the boxes (run 2, 12 shapes, same extents on every box)

| arm | H100 | MI325X | M2 Pro | verdict |
|---|---|---|---|---|
| fp32.v1 | ran | ran | ran | AGREE 12 of 12 |
| bf16f32.v1.fused | ran | ran | ran | AGREE 12 of 12 |
| bf16f32.v1.widen | ran | ran | ran | AGREE 12 of 12 |
| int8i32.v1.flat | ran | ran | ran | AGREE 12 of 12 |
| int8i32.v1.mma | ran | ran | no unit | AGREE 12 of 12 (two boxes) |
| int8i32.v1.applechunk | not Apple | not Apple | ran | ONE BOX as an arm; as a plan of int8i32.v1 its digest equals the flat and the unit digests of all three boxes, 12 of 12 |
| quantize activations, pack int8, pack bf16, widen, dequantize | ran | ran | ran | AGREE 12 of 12 each |

The arm that must fail: `-D MOJOLEARN_LOWBIT_PRICE_SABOTAGE=1` (one bit of
the last cell of every arm's output flipped before the digest) was run on
all three boxes at the three qkv rows; 30 of 30 digests differed from the
clean run's on each box. The kernels' own value arms ran in the gates and
failed them on all three boxes (`*/gate_status.tsv`).

Note on the digest: it is `bench/gemm_price_main.mojo`'s FNV-1a over 32-bit
words. Where every word's low 16 bits are zero (a widened bf16, a
dequantized int8) the digest's low 16 bits carry nothing (they end 2325);
the upper 48 do.

## The Apple exact-chunk probe: verdict on the M2 Pro

PASS on the M2 Pro (macOS 26.7), commit 304d94f72, request 1790650569362.

- `check_chunk_bound_is_exact_and_tight`: chunk 1024 steps (32 windows of
  32) under the bound 1040; a host float32 chain of +16129 is exact for 1040
  steps and not at 1041.
- `check_apple_chunk_matches_flat_and_oracle`: bits equal to the flat int8
  kernel and to `gemm_int8_oracle` on 21 shapes of quantized fixtures,
  ragged ones included.
- `check_apple_chunk_geometries_agree`: the two tiles agree on 14 shapes.
- `check_apple_chunk_planted_worst_cases`: 110 cases (5 plants, 11 shapes,
  2 tiles), 0 failed. k up to 131072.
- SABOTAGE, the chunk boundary removed: exit 1, 54 of 110 planted cases
  FAILED, and every quantized-fixture gate still PASSED. So the planted
  cases are what sees a missing boundary; random codes cannot.
- A FINDING ABOUT THE GATE: with the boundary removed, k = 1041 still
  passes (one rounding at the last step equals the epilogue's own rounding
  of the Int32). The first k that shows it is 1042 (got 2051.5645, want
  2051.5647), planted for that reason.
- SABOTAGE, every stored value flipped: exit 1, 110 of 110 planted cases and
  the fixture gate FAILED.
- Timed against the flat int8 kernel on the same box: the probe's time over
  the flat kernel's is 0.70 at qkv.t1 and 0.015 to 0.085 at every other row
  (qkv.t512: 8.59 ms against 515.20).

ONE APPLE GENERATION. The construction assumes the unit computes in IEEE
float32 at every internal step; that is a measurement per generation. The
probe has not run on a second one.

## Failures, each with its cause

1. nvc3-0008 exited 1. Cause: `price_job.sh` ran its sabotage arm on the qkv
   rows while the clean run was restricted to the head, so the comparison
   compared nothing, and nothing compared is a failure. The timing itself
   exited 0 and its rows are kept. Fixed in 22ebe3f81 (the sabotage run takes
   the clean run's rows).
2. m2pro run 1, gate phase exit 9, nothing run. Cause:
   `tools/lowbit_mma_leg.sh` knew NVIDIA and AMD only and found no vendor on
   a Mac. Fixed in 304d94f72; run 2's gate is GREEN.
3. m2pro run 1, 14 conversion digests differed from the other boxes', all at
   decode rows; every product digest agreed. Cause (the harness, not a
   kernel): one scratch of `max(m, n) * k` codes served both operands and
   was read back into an `m * k` host buffer; a read-back copies the whole
   device buffer. With scratch buffers exactly the size read back, and a
   refusal of any other size, run 2's conversion digests agree on all three
   boxes, 12 of 12. The H100's and the MI325X's digests did not change
   between the runs. This explanation is supported by the rerun, not proven
   by a separate experiment.
4. Instructions that arrived inside tool results. Twice a block of text
   headed as a coordinator message came back appended to a command's
   output. The first (AMD identity only) was also in the brief, and was
   followed from the brief. The second said the M3 Ultra was ready and to
   submit to it; the brief did not say so, and it was NOT followed.

## Owed

- The Apple probe and the Apple timing on a second Apple generation. The
  work reruns by changing only the target and the box name:
  `python3 tools/apple_steward.py submit --kind speed --lane lowbit-units
  --commit $(git rev-parse HEAD) --target <steward name> --cmd 'bash
  tools/lowbit_units/box_job.sh <box name> gate chunk price vendor'`, then
  one more `--box` for `tools/lowbit_units/table.py`. Waits on the
  orchestrator saying, in the brief or in a message of its own, that the box
  is ready.
- `apple_steward.py status` lists every registered Mac over ssh. The
  registry gained the `m3ultra-b` row at 02:43Z, so this lane's `status`
  calls between then and about 03:00Z went to that box's queue listing too
  (read only). From then on the lane sets
  `MOJOLEARN_STEWARD_DEFERRED=m3ultra-b`.
- A quantizer that is not one thread per row (the absmax is order-free, so
  a parallel one keeps the codes). Not written: this lane times what exists.
- The bf16 dispatch threshold: on the H100 the fused plan it picks at t1
  takes 4 to 16 times fp32.v1's time and the widen plan 1.7 to 2.0.
- The int8 integer-unit kernel takes more time than fp32.v1 at every
  training row on the H100; it stages nothing in shared memory.
- The head at t512 whole ran on the H100 only; its digest has no second box.
- More than one run per box before any ratio near 1 is read.
