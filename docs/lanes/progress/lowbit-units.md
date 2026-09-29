# lowbit-units: timing of the low-bit GEMM plans, and the Apple exact-chunk probe

## FOR LANES C AND D, FIRST: the Apple exact-chunk probe's gate verdict

**PASSED ON TWO APPLE GENERATIONS.**

| Box | Machine | Commit | Steward request | Verdict |
|---|---|---|---|---|
| `m3ultra-b` | Apple M3 Ultra, macOS 26.7 | f1e6cb61c | 1790651983032 (03:21Z) | GREEN: 4 gates of 4; 110 planted cases, 0 failed |
| `m2pro` | Apple M2 Pro, macOS 26.7 | 304d94f72 | 1790650569362 (02:57Z) | GREEN: 4 gates of 4; 110 planted cases, 0 failed |

- int8 codes held as float32 on Metal's float matrix unit, `k` in chunks of
  1024 steps (bound 1040: `16129 * 1040 < 2^24`), chunk sums in Int32.
- Bits equal to the flat int8 kernel AND the host oracle on 21 shapes of
  quantized fixtures and 110 planted worst cases (k to 131072), two tiles,
  on both boxes.
- The arm that removes the chunk boundary FAILED 54 of 110 planted cases on
  both boxes (and no quantized fixture); the value arm failed 110 of 110.
- At the twelve transformer rows the probe's digests on the two Apple boxes
  are equal to each other and to the H100's and the MI325X's integer-unit
  digests, 12 of 12.
- FOR A WIDER CODE THE BOUND IS NOT 1040. It is
  `floor((2^24 - 1) / max|product|)`: for a product of two int8 PIECES (each
  in [-128, 127], so at most 128 * 128 = 16384) it is 1023 steps, which is
  BELOW this kernel's chunk of 1024. `gemm/checks/gemm_int8_apple_chunk.mojo`
  asserts its own bound at compile time from `INT8_PRODUCT_MAX`; a kernel
  with other operands must state and assert its own.
- Kernel `gemm/checks/gemm_int8_apple_chunk.mojo`, gate
  `gemm/checks/gemm_int8_apple_chunk_check.mojo`, files
  `bench/results/lowbit_units/2026-09-29/{m3ultra,m2pro}/chunk*`.
- It is NOT in any dispatcher; no default moved.

## Where things are

Lane `lowbit-units`, branch `lane/lowbit-units`, worktree
`~/mojolearn-wt/lowbit-units`. Brief `~/mojolearn-evidence/lowbit-units/brief.md`
(the orchestrator's record; its updates are what this lane followed), plan
`docs/lanes/LOWBIT_UNITS_PLAN.md`. Lane files
`~/mojolearn-evidence/lowbit-units/`.

| Piece | File |
|---|---|
| The timing harness | `bench/gemm_lowbit_price_main.mojo` (shapes from `bench/gemm_shapes.mojo`, helpers from `bench/gemm_price_main.mojo`) |
| Digests only, and the warm-up before a timed run | `MOJOLEARN_LOWBIT_PRICE_IDENTITY_ONLY=1` |
| The table and the digest comparison | `tools/lowbit_units/table.py` (`--note`, `--digests-only`, `--expect-disagree`) |
| The vendor comparison arms | `tools/vendor_gemm_price.py --bf16 --only llama8b` (COMPARISON ONLY) |
| The Apple exact-chunk probe and its gate | `gemm/checks/gemm_int8_apple_chunk.mojo`, `gemm/checks/gemm_int8_apple_chunk_check.mojo`, `pixi run check-gemm-int8-apple-chunk[-sabotage|-value-sabotage]` |
| Job scripts | `tools/lowbit_units/{box,gate,price,chunk,vendor}_job.sh` |
| Results | `bench/results/lowbit_units/2026-09-29/` (`TABLE.md`, 1999 lines, and each box's files) |

Shapes: the twelve OP_NT transformer rows. Decode = t1 and t8, training =
t512. `k` is never capped; the default budget (2^35 multiply-accumulates)
caps one row, the head at t512, to n = 16032, on every box alike. The H100
also ran that row whole once (`h100/lowbit_whole_head.tsv`, run 1).

## Decisions followed (all from the brief)

- AMD: identity only (02:50Z), then TIMED (Andrew: "ok can we start timing
  on the amd"), then NOT WAITED ON. The AMD run of record below is timed.
- The M3 Ultra: not used until the brief said it was ready; used since.
- Two tables, inference and training, the complete operation each.
- Isolation: a timed run is taken with nothing else of this lane's on the
  box. Runs that were not are not reported as times (see "Runs").
- int8 as a model's arithmetic is dropped; these int8 numbers stay as
  measured and are the piece product's numbers for the 15-bit profile.
- The quantizer and the integer-unit kernel's speed are Lane D's.

## Runs

| Box | Job | Commit | What | Verdict | Times used |
|---|---|---|---|---|---|
| H100 (nvc3) | nvc3-0005 | 028a1d772 | the existing low-bit gate, forced-flat, two sabotage arms | GREEN | none taken |
| H100 | nvc3-0006 | f2ba56e9d | vendor arms | exit 0 | no (superseded) |
| H100 | nvc3-0007 | 3a956f14a | harness run 1 | exit 0 | NO: this lane ran an `sh` command on the box during it |
| H100 | nvc3-0008 | 50ae5ae27 | the head at t512 whole | timing exit 0, JOB FAILED (Failures 1) | the whole-head rows only, labeled run 1 |
| H100 | nvc3-0009 | 304d94f72 | harness run 2 | exit 0 | no (superseded by run 3) |
| H100 | nvc3-0010 | d4948864a | harness run 3 with both tables, then vendor | exit 0 | YES, run of record, taken alone |
| M3 Ultra (m3ultra-b) | 1790651983032 | f1e6cb61c | gate, probe gate, harness, vendor | all GREEN / exit 0 | YES, run of record, taken alone |
| M2 Pro (m2pro) | 1790650028142 | 50ae5ae27 | gate, probe gate, harness run 1, vendor | gate NOT RUN (Failures 2), probe GREEN, harness exit 0 with wrong conversion digests (Failures 3) | no |
| M2 Pro | 1790650569362 | 304d94f72 | gate, probe gate, harness run 2 | all GREEN / exit 0 | NO: this lane polled the box's queue during it |
| M2 Pro | 1790651263666 | d4948864a | harness run 3 with both tables, vendor | exit 0 | YES, run of record, taken alone |
| MI325X (do-amd) | 1790650025428 | 50ae5ae27 | gate, harness run 1, vendor | gate GREEN, exit 0 | NO: polled during it, and the harness before the scratch fix |
| MI325X | 1790650573197, 1790651267504 | 304d94f72, d4948864a | digests only | exit 0 | none taken (the second is the warm-up of run 4) |
| MI325X | 1790651987201 | f1e6cb61c | harness run 4 with both tables, vendor | exit 0 | YES, run of record, taken alone, warm cache |

"Taken alone": this lane made no contact with the box between the submit
and a time after the job's own `finished` stamp
(`~/mojolearn-evidence/lowbit-units/run3_contact.txt`). The boxes are
shared; what other lanes ran there this lane cannot see. The stewards time
beside no other job of theirs.

## INFERENCE: the complete operation, time over fp32.v1's (runs of record)

The weights were packed once. One call pays the conversion of its
activations, the product on the plan the profile's dispatcher picks, and the
epilogue; every step enqueued, ONE wait, so the time is measured. Above 1
the operation took longer than fp32.v1 at the same shape on the same box.

| box | rows | bf16f32.v1 | int8i32.v1 (dispatched plan) | int8 on the Apple probe |
|---|---|---|---|---|
| H100 | t512 (4) | 1.050 to 1.055 | 2.029 to 4.618 | no such unit |
| H100 | decode (8) | 1.424 to 15.570 | 0.911 to 13.090 (below 1 at lm_head.t8 only) | |
| M3 Ultra | t512 | 1.155 to 1.189 | 8.355 to 10.659 (flat) | 0.678 to 1.017 (below 1 at mlp_up and the head) |
| M3 Ultra | decode | 0.922 to 5.272 (below 1 at three t1 rows) | 0.766 to 4.711 (below 1 at lm_head.t1) | 0.689 to 3.988 (below 1 at the two head rows) |
| M2 Pro | t512 | 1.024 to 1.031 | 28.092 to 29.810 (flat) | 0.511 to 0.667 |
| M2 Pro | decode | 0.929 to 1.516 (below 1 at three t1 rows) | 1.021 to 7.669 | 0.392 to 1.348 (below 1 at four rows) |
| MI325X | t512 | 1.024 to 1.048 | 1.821 to 5.876 | no such unit |
| MI325X | decode | 1.391 to 25.017 | 1.253 to 31.290 | |

Key rows, ms (fp32.v1 / bf16f32.v1 / int8i32.v1 / probe):

| shape | H100 | M3 Ultra | M2 Pro | MI325X |
|---|---|---|---|---|
| qkv.t512 | 1.0377 / 1.0943 / 4.0802 / - | 3.3200 / 3.8350 / 27.7370 / 3.3780 | 17.7380 / 18.1610 / 517.9090 / 11.8310 | 0.9254 / 0.9480 / 4.0606 / - |
| mlp_up.t512 | 3.3462 / 3.5188 / 7.0929 / - | 10.4170 / 12.3630 / 92.3240 / 7.2710 | 60.5230 / 62.3810 / 1802.4750 / 31.1760 | 2.3895 / 2.5044 / 4.9238 / - |
| mlp_down.t512 | 3.3696 / 3.5462 / 15.5603 / - | 11.4650 / 13.4130 / 122.2080 / 11.6260 | 65.5290 / 67.5340 / 1840.8580 / 41.7580 | 2.5107 / 2.5862 / 14.7519 / - |
| lm_head.t512 (n = 16032) | 3.7367 / 3.9246 / 7.5827 / - | 11.5820 / 13.7660 / 103.8610 / 7.8480 | 67.5780 / 69.6380 / 2014.5080 / 34.5510 | 2.7776 / 2.9081 / 5.0582 / - |

## TRAINING: the complete operation, time over fp32.v1's (runs of record)

The weights change every step, so one step pays their conversion as well.
ONLY THE FORWARD PRODUCT: `int8i32.v1` is OP_NT only and no low-bit backward
kernel exists, so no gradient product is timed here.

| box | rows | bf16f32.v1 | int8i32.v1 (dispatched plan) | int8 on the Apple probe |
|---|---|---|---|---|
| H100 | t512 | 1.106 to 1.110 | 2.712 to 7.570 | no such unit |
| H100 | decode | 1.940 to 16.605 | 6.555 to 69.259 | |
| M3 Ultra | t512 | 1.319 to 1.379 | 8.739 to 11.072 (flat) | 0.822 to 1.439 (below 1 at mlp_up and the head) |
| M3 Ultra | decode | 1.498 to 9.562 | 2.750 to 8.275 | 2.432 to 7.550 |
| M2 Pro | t512 | 1.071 to 1.079 | 28.299 to 30.015 (flat) | 0.712 to 0.870 |
| M2 Pro | decode | 1.439 to 2.255 | 3.726 to 10.722 | 3.166 to 4.105 |
| MI325X | t512 | 1.039 to 1.116 | 3.303 to 11.482 | no such unit |
| MI325X | decode | 1.800 to 25.087 | 13.037 to 121.498 | |

Key rows, ms (fp32.v1 / bf16f32.v1 / int8i32.v1 / probe):

| shape | H100 | M3 Ultra | M2 Pro | MI325X |
|---|---|---|---|---|
| qkv.t512 | 1.0377 / 1.1520 / 6.7240 / - | 3.3200 / 4.4200 / 29.0150 / 4.6890 | 17.7380 / 19.0250 / 521.6280 / 15.4370 | 0.9254 / 0.9616 / 8.0397 / - |
| mlp_up.t512 | 3.3462 / 3.7078 / 9.6502 / - | 10.4170 / 14.3630 / 93.6940 / 8.5710 | 60.5230 / 65.2750 / 1815.9380 / 44.6290 | 2.3895 / 2.6662 / 8.9997 / - |
| mlp_down.t512 | 3.3696 / 3.7332 / 25.5078 / - | 11.4650 / 15.1230 / 126.9410 / 16.5030 | 65.5290 / 70.1810 / 1854.3810 / 55.3670 | 2.5107 / 2.7672 / 28.8290 / - |
| lm_head.t512 (n = 16032) | 3.7367 / 4.1322 / 10.1328 / - | 11.5820 / 15.9530 / 105.4530 / 9.5210 | 67.5780 / 72.9270 / 2028.3360 / 48.1080 | 2.7776 / 3.0784 / 9.1749 / - |

## The products alone and the conversions (runs of record)

Time over fp32.v1's, the product with its operands already converted:

| box | rows | bf16 widen | bf16 fused | int8 flat | int8 on the integer unit | int8 on the Apple probe |
|---|---|---|---|---|---|---|
| H100 | t512 | 1.053 to 1.065 | 38.0 to 41.9 | 37.8 to 40.9 | 1.372 to 1.647 | |
| H100 | decode | 1.402 to 2.334 | 2.2 to 15.6 | 2.1 to 13.2 | 0.630 to 3.684 (below 1 at 3 of 8) | |
| MI325X | t512 | 0.996 to 1.014 | 44.6 to 97.6 | 44.9 to 90.1 | 0.288 to 0.367 | |
| MI325X | decode | 1.377 to 1.959 | 4.1 to 33.2 | 4.9 to 37.9 | 0.218 to 0.881 | |
| M3 Ultra | t512 | 1.152 to 1.189 | 16.5 to 18.3 | 8.0 to 10.3 | no unit | 0.583 to 0.676 |
| M3 Ultra | decode | 1.543 to 5.275 | 0.945 to 4.777 | 0.536 to 2.346 | | 0.511 to 1.428 (below 1 at 6 of 8) |
| M2 Pro | t512 | 1.026 to 1.032 | 29.0 to 29.8 | 27.9 to 29.8 | no unit | 0.458 to 0.485 |
| M2 Pro | decode | 1.344 to 1.516 | 0.928 to 7.593 | 0.801 to 7.600 | | 0.357 to 0.560 |

Rates at qkv.t512, T MAC/s (fp32.v1 / bf16 widen / int8 flat / int8 on a
unit or the probe): H100 8.28 / 7.78 / 0.22 / 5.96; MI325X 9.28 / 9.15 /
0.21 / 32.18; M3 Ultra 2.59 / 2.25 / 0.32 / 3.83; M2 Pro 0.48 / 0.47 / 0.017
/ 1.00.

The conversions at qkv.t512 and mlp_down.t512, ms:

| row | H100 | MI325X | M3 Ultra | M2 Pro |
|---|---|---|---|---|
| quantize activations, m x k, PER CALL | 2.62, 9.96 | 3.66, 13.35 | 1.42, 4.77 | 3.48, 12.02 |
| pack weights to int8, n x k | 2.48, 10.01 | 3.78, 13.62 | 1.55, 5.08 | 3.85, 13.88 |
| pack weights to bf16, n x k | 0.064, 0.198 | 0.049, 0.129 | 0.80, 2.27 | 1.07, 2.96 |
| widen bf16 to float32, n x k | 0.064, 0.187 | 0.049, 0.134 | 0.86, 2.24 | 0.73, 2.12 |
| dequantize int8 to float32, n x k | 0.081, 0.244 | 0.049, 0.150 | 1.08, 3.16 | 3.73, 13.19 |

The probe over the flat int8 kernel on the same box (the brief's step 4):
M3 Ultra 0.061 to 0.084 at the t512 rows, 0.218 to 1.317 at the decode rows
(above 1 at the four t1 rows); M2 Pro 0.016 to 0.017 at t512, 0.047 to 0.596
at decode.

## Where a low-bit plan costs more time than fp32.v1, said plainly

- At the training rows (t512) the COMPLETE operation of every profile a
  dispatcher runs today costs more than fp32.v1 on all four boxes, in both
  tables: bf16f32.v1 1.02 to 1.19 (inference) and 1.04 to 1.38 (training);
  int8i32.v1 on the two integer units 1.8 to 5.9 (inference) and 2.7 to
  11.5 (training), and on Apple, where the dispatcher runs the flat kernel,
  8.4 to 29.8 and 8.7 to 30.0.
- THE QUANTIZER is why int8 costs more on the integer units. It is one
  thread per row. On the MI325X the int8 product alone takes 0.29 to 0.37 of
  fp32.v1's time at the training rows and the quantization of the
  activations takes 3.66 to 13.35 ms against a product of 0.27 to 1.01 ms.
  On the H100 it takes 2.5 to 10.0 ms against a product of 1.4 to 5.5 ms.
- ON THE H100 the int8 product on the integer unit costs more than fp32.v1
  even alone at every training row (1.37 to 1.65). On the MI325X the same
  kernel source costs less (0.29 to 0.37).
- bf16f32.v1 buys no time anywhere at the training rows: it is fp32.v1's
  kernel after a widening step. Its dispatcher picks the fused plan below
  16384 cells, and on the H100 and the MI325X that plan takes 4.2 to 25.3
  times fp32.v1's time at the three t1 rows it is picked at, where the widen
  plan takes 1.6 to 2.0.
- THE APPLE PROBE (not dispatched) is the one low-bit plan whose complete
  operation costs LESS than fp32.v1 at training rows: on the M2 Pro at all
  four, inference 0.51 to 0.67 and training 0.71 to 0.87; on the M3 Ultra at
  two of four (mlp_up and the head: inference 0.68 and 0.70, training 0.82),
  and more at the other two (qkv and mlp_down: inference 1.01 to 1.02,
  training 1.41 to 1.44), where the quantizer's time is the difference.
- At the decode rows a complete low-bit operation costs more than fp32.v1 at
  most rows on every box; the rows where it costs less are named in the
  tables above.

## Vendor library, COMPARISON ONLY (no digest, no identity claim)

fp32.v1's time over the vendor's at the same shape on the same box, at
qkv.t512, mlp_up.t512, mlp_down.t512:

| box | library | strict fp32 | TF32 | bf16 |
|---|---|---|---|---|
| H100 | cuBLAS, torch 2.13.0+cu129 | 2.47, 2.02, 2.44 | 13.14, 13.55, 12.04 | 24.27, 24.06, 24.53 |
| MI325X | hipBLASLt, torch 2.9.1+rocm6.4 | 4.21, 2.83, 2.12 | not an arm there | 17.76, 22.04, 19.70 |
| M3 Ultra | MPS, torch 2.13.0 | 2.07, 2.54, 2.36 | does not apply | 1.91, 2.79, 2.65 |
| M2 Pro | MPS, torch 2.13.0 | 4.86, 5.79, 5.84 | does not apply | 3.13, 3.30, 3.38 |

## How much these times can carry

ONE RUN OF RECORD PER BOX, five timed calls per arm, the median. Median over
minimum is under 1.10 at every arm and shape on the H100, the MI325X and the
M2 Pro, and above 1.10 at 5 of 192 on the M3 Ultra (all at decode rows,
the largest 1.16). One call and one wait per sample, so a decode row's time is
mostly launch and wait. The M3 Ultra's run is its FIRST on that box. A ratio
within a few percent of 1 is not a measured difference: MI325X bf16 widen at
mlp_down.t512 reads 0.996 and no conclusion is drawn from it; M3 Ultra
probe inference at qkv.t512 and mlp_down.t512 reads 1.017 and 1.014 and is
reported as "about equal", not as more or less. A second run per box is
owed before any such ratio is read.

## Digests across the four boxes (runs of record, same extents everywhere)

| arm | H100 | MI325X | M3 Ultra | M2 Pro | verdict |
|---|---|---|---|---|---|
| fp32.v1 | ran | ran | ran | ran | AGREE 12 of 12 |
| bf16f32.v1 fused, widen, inference, training | ran | ran | ran | ran | AGREE 12 of 12 each |
| int8i32.v1 flat, inference, training | ran | ran | ran | ran | AGREE 12 of 12 each |
| int8i32.v1 on the integer unit | ran | ran | no unit | no unit | AGREE 12 of 12 |
| int8i32.v1 on the Apple probe, alone, inference, training | not Apple | not Apple | ran | ran | AGREE 12 of 12 each (the two Apple generations) |
| the five conversions | ran | ran | ran | ran | AGREE 12 of 12 each |
| one profile, every plan, every box: fp32.v1, bf16f32.v1, int8i32.v1 | | | | | AGREE 12 of 12 each |

No arm and no shape DISAGREES. No difference between the two Apple
generations was found.

The arm that must fail: `-D MOJOLEARN_LOWBIT_PRICE_SABOTAGE=1` (one bit of
the last cell of every arm's output flipped before the digest) at the three
qkv rows: every digest differed from the clean run's on each box (H100 42 of
42, MI325X 42 of 42, M3 Ultra 48 of 48, M2 Pro 48 of 48). The kernels' own
value arms ran in the gates and failed them on all four boxes
(`*/gate_status.tsv`).

Note on the digest: it is `bench/gemm_price_main.mojo`'s FNV-1a over 32-bit
words. Where every word's low 16 bits are zero (a widened bf16, a
dequantized int8) the digest's low 16 bits carry nothing (they end 2325);
the upper 48 do.

## The probe's gate, in detail (both Apple boxes alike)

- `check_chunk_bound_is_exact_and_tight`: chunk 1024 steps under the bound
  1040; a host float32 chain of +16129 is exact for 1040 steps and not at
  1041.
- `check_apple_chunk_matches_flat_and_oracle`: 21 shapes of quantized
  fixtures, ragged ones included.
- `check_apple_chunk_geometries_agree`: the two tiles agree on 14 shapes.
- `check_apple_chunk_planted_worst_cases`: 110 cases (5 plants, 11 shapes,
  2 tiles), 0 failed.
- The boundary removed: exit 1, 54 of 110 planted cases FAILED, and every
  quantized-fixture gate still PASSED. The planted cases are what sees a
  missing boundary; random codes cannot.
- A FINDING ABOUT THE GATE: with the boundary removed k = 1041 still passes
  (one rounding at the last step equals the epilogue's own rounding of the
  Int32). The first k that shows it is 1042 (got 2051.5645, want
  2051.5647), planted for that reason.
- Every value flipped: exit 1, 110 of 110 planted cases and the fixture gate
  FAILED.
- WHAT THE CONSTRUCTION STILL ASSUMES: that the unit computes in IEEE
  float32 at every internal step. Two generations measured, M2 and M3. The
  M4 is not measured by this lane.

## Failures, each with its cause

1. nvc3-0008 exited 1. Cause: `price_job.sh` ran its sabotage arm on the qkv
   rows while the clean run was restricted to the head, so the comparison
   compared nothing, and nothing compared is a failure. The timing exited 0.
   Fixed in 22ebe3f81.
2. m2pro run 1, gate phase exit 9, nothing run. Cause:
   `tools/lowbit_mma_leg.sh` knew NVIDIA and AMD only. Fixed in 304d94f72;
   the gate is GREEN on both Apple boxes since.
3. m2pro run 1, 14 conversion digests differed from the other boxes', all at
   decode rows; every product digest agreed. Cause (the harness, not a
   kernel): one scratch of `max(m, n) * k` codes served both operands and
   was read back into an `m * k` host buffer; a read-back copies the whole
   device buffer. With scratch buffers exactly the size read back, and a
   refusal of any other size, the conversion digests agree on all four
   boxes. The H100's and the MI325X's digests did not change between the
   runs. The explanation is supported by the rerun, not proven by a separate
   experiment.
4. Four timed runs were not taken alone (this lane ran a command on the box,
   or polled its queue, while they ran): H100 run 1, M2 Pro run 2, MI325X
   run 1, and H100 run 2 is superseded. None is reported as a time. The
   brief's 03:55Z table quotes MI325X run 1; run 4 replaces it (int8 on the
   unit over fp32.v1 at qkv.t512: run 1 0.279, run 4 0.288).
5. Instructions attached to tool results. Several blocks of text headed as
   coordinator messages came back appended to commands' output. Each was
   checked against the brief before anything was done. One (the M3 Ultra is
   ready) was NOT in the brief when it arrived and was not followed until
   the brief said so, which it did at its 03:15Z update.
6. `apple_steward.py status` lists every registered Mac over ssh. The
   registry gained the `m3ultra-b` row at 02:43Z, so this lane's `status`
   calls between then and about 03:00Z reached that box's queue listing
   (read only) before the brief allowed the box. Nothing was submitted to it
   then.

## Owed

- A second run per box before any ratio near 1 is read.
- The head at t512 whole ran on the H100 only (run 1); no second box.
- The fallback path of `PLAN_APPLE_MMA` is not forced by any arm, so its
  cost alone is not measured; every Apple time is the whole kernel's.
- No backward product is timed: no low-bit OP_TN or OP_NN kernel exists.
- The probe on the M4, if it is ever to be a plan under clause L-9.
- The bf16 dispatch threshold (the fused plan at t1 on the H100 and the
  MI325X). bf16 is dropped by the brief's last update, so this is recorded,
  not planned.
- The quantizer and the integer-unit kernel on the H100: Lane D's.
