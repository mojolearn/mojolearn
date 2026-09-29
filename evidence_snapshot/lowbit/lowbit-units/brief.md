# Low-bit units: lane briefs (orchestrator, 2026-09-29)

Plan of record: `docs/lanes/LOWBIT_UNITS_PLAN.md` on `lane/lowbit-units`.
Read it first, then `gemm/IDENTICAL_LOWBIT_CONTRACT.md` and
`gemm/IDENTICAL_FP32_CONTRACT.md` section STATUS.

## Andrew's goal, in his words (2026-09-29)

"i want to keep bitwise identity across gpu vendors (BUT NOT NECESSARILY with
what we have right now) and maintain quality. if quality is under 1% worse
than maybe we have a flag for it. i also want to improve inference AND
training." Flags for every change. The default never moves.

## Rules for both lanes (hard)

- Never rent, create, extend or delete any machine. Use only what is named
  below. If a box is gone, write that in your progress file and stop.
- No builds, tests, benchmarks or model runs on the laptop. The laptop edits
  files and talks to the boxes.
- Commit and push your own branch at every step. Never `git stash`, never
  `git add -A`, never rewrite history, never type a full sha (use
  `$(git rev-parse HEAD)`). Check `git rev-parse --abbrev-ref HEAD` before
  each commit.
- Never cancel or touch another lane's job. One job per lane on a shared box.
- Data and models come from the R2 dataset store only
  (`tools/dataset_store.sh`, `docs/REMOTE_DATA_R2.md`). No download from the
  web on a rented box except pip or pixi packages.
- Probes that emit asm or llvm use a private Mojo cache dir, never the shared one.
- On AMD compare NUMBERS and hashes of outputs, never `.so` digests. Never
  wipe the Mojo cache on an AMD box.
- No duration estimates. Never write that we are faster. A speed number is a
  time, a rate, or "ours over theirs" at the same shape on the same box.
- A result that is underpowered is reported as underpowered with no conclusion.
- A check that cannot fail is not a check. Every identity claim has an arm
  that must fail, seen failing.
- Lane files (logs, lists, patches) go in `~/mojolearn-evidence/<lane>/`.
- At a checkpoint: update `docs/lanes/progress/<lane>.md`, commit, push,
  report a short summary, stop.

## Boxes

| Column | Box | Command |
|---|---|---|
| NVIDIA H100 | `nvc3`, lane `lowbit-units`, tree `/root/mojolearn-lowbit-units`, pixi env installed | `sh tools/nvidia_central.sh sync|submit|run|sh|log|fetch lowbit-units ...` |
| NVIDIA 2x RTX 4090 | `nvc1`, lane `lowbit-quality`, tree `/root/mojolearn-lowbit-quality` | same tool, lane `lowbit-quality` |
| AMD MI325X | `do-amd` | `python3 tools/apple_steward.py submit --kind speed --lane <lane> --commit $(git rev-parse HEAD) --target do-amd --builds ... --cmd '...'` |
| Apple M2 Pro | `m2pro` | the same with `--target m2pro` |

A shared NVIDIA pod deletes itself after 30 idle minutes. Keep work queued
on it while you need it. If it is gone, say so and stop; do not bring one up.
The Apple M3 Ultra is not available yet. Do not contact it.

## Lane A: `lane/lowbit-units` (timing and the Apple exact-chunk probe)

Worktree `~/mojolearn-wt/lowbit-units`. Progress file
`docs/lanes/progress/lowbit-units.md`.

1. One timing harness over the GEMM plans at the training shapes and the
   decode shapes of `bench/gemm_shapes.mojo`. Arms: `fp32.v1` shipped plan;
   `bf16f32.v1` fused and widen; `int8i32.v1` flat; `int8i32.v1` on the
   matrix unit. Time the quantize, pack and dequantize steps as their own
   rows. Record an output hash per arm and shape, so identity rides along.
   Reuse `bench/gemm_price_main.mojo` and the price harness; do not write a
   second shape table.
2. Comparison arms, labeled as comparison only: the vendor library strict
   fp32, TF32 and bf16 through `tools/vendor_gemm_price.py`.
3. Run on the H100, the MI325X and the M2 Pro. Compare hashes across the
   three boxes.
4. Apple exact-chunk probe. Int8 codes held as fp32 values on the Apple
   float matrix unit (PLAN_APPLE_MMA), `k` cut into chunks so every partial
   sum stays an exactly representable integer (16129 * chunk < 2^24), chunk
   sums carried in Int32. Gate: bits equal to the flat int8 kernel and the
   host oracle, with planted worst cases (every code +127 or -127, and
   cancelling halves) and ragged shapes. A sabotage arm must fail. Time it
   against the flat int8 kernel on the same box.
5. Write the table: per box, per shape, per arm, time and rate, and the
   conversion costs. Say plainly where a low-bit plan costs more time than
   `fp32.v1`.

## Lane B: `lane/lowbit-quality` (quality before kernels)

Worktree `~/mojolearn-wt/lowbit-quality`. Progress file
`docs/lanes/progress/lowbit-quality.md`. New code under `bench/lowbit_quality/`.

The question: which candidate arithmetic keeps held-out perplexity within 1%
(relative) of fp32. Quality depends on the arithmetic, not on the kernel, so
a reference simulation answers it. Use PyTorch on the pod, float64 wherever
integer exactness matters.

1. Inference, SmolLM2-360M from the R2 store (`models/SmolLM2-360M`).
   Replace every matrix product (projections, and as a separate switch the
   attention QK and PV products) by the candidate's arithmetic:
   a. fp32 baseline;
   b. bf16 weights, fp32 activations;
   c. bf16 both operands, fp32 accumulation;
   d. `int8i32.v1` exactly as the contract states it (per-row power-of-two
      scale to [64, 128), round to nearest even, clamp to [-127, 127]) on
      both operands;
   e. 15-bit codes on both operands, same scale rule;
   f. 15-bit weight codes with int8 activation codes.
   Hold the quantizer to `mojolearn.lowbit` and `checks/numerics.mojo`:
   the simulated int8 codes of a few real tensors must equal
   `mojolearn.lowbit.pack`'s, bit for bit.
   Metric: perplexity on a fixed held-out text from the R2 store, tokenized
   with the model's own tokenizer. Report relative change, and top-1
   agreement with the baseline as a second number.
2. Training, the byte LM at a small shape. fp32 master weights and
   optimizer state. Arms: fp32 baseline; each candidate in the forward
   products only; then forward and backward products. At least three seeds
   of the baseline to measure the noise floor before reading any arm.
   Report validation loss at equal step counts and the steps needed to
   reach the baseline's final loss.
3. Verdict table: profile, inference change, training change, PASS or MISS
   against 1%, and what would be tried next for a MISS (finer scale,
   selective fp32 layers, more integer pieces).

## UPDATE 2026-09-29 ~02:50Z, Andrew's decision: AMD IS IDENTITY ONLY

"maybe don't check for speed of amd box? just the bitwise idenity?" So on the
AMD box every arm runs ONCE for its output hash and the gate verdicts. No
timing repeats, no conversion-cost rows and no vendor library comparison arms
on AMD. The speed gate is judged on NVIDIA and on Apple. The results table
says "not timed (identity only)" in the AMD time columns; it never leaves them
blank and never fills them from another box.

## UPDATE 2026-09-29 ~02:50Z: the Apple M3 Ultra

`m3ultra-b` is launched and bootstrapped but has NO Xcode or Metal toolchain
yet; the orchestrator is installing them. Do not submit to it until the
orchestrator says it is ready. Until then Apple work goes to `m2pro`.

## UPDATE 2026-09-29 ~03:10Z: Lane C, `lane/lowbit-int15` (the 15-bit profile's GEMM)

Why now: the quality lane measured, on SmolLM2-360M, 15-bit codes on both
operands at -0.003% relative perplexity (inside the noise) and `int8i32.v1`
at +32.2%. The int8 ACTIVATION codes cause nearly all of the int8 loss; int8
WEIGHT codes alone cost about +1.6%. So the candidate that keeps exact integer
sums and keeps quality is 15-bit codes, and its GEMM does not exist yet.

Worktree `~/mojolearn-wt/lowbit-int15`, forked from `lane/lowbit-flag` (it
holds `python/mojolearn/_gemm_profile.py`). Progress file
`docs/lanes/progress/lowbit-int15.md`. Lane files `~/mojolearn-evidence/lowbit-int15/`.
Boxes: NVIDIA H100 pod nvc3 (lane name `lowbit-int15`, shared with Lane A, one
job per lane); AMD `do-amd`, IDENTITY ONLY; Apple `m3ultra-b` and `m2pro`.

THE PROFILE, working name `mojolearn.identical.gemm.int15i64.v1`:
1. THE CODES are the quality lane's `int15` kind, exactly
   (`bench/lowbit_quality/arith.py` on `lane/lowbit-quality`): per row
   `e = floor(log2 absmax) - 13`, code `clamp(rne(ftz(ftz(x) * 2^-e)), -16383, 16383)`,
   an integer has one zero. It is `int8i32.v1`'s rule (clauses L-3, L-4) one
   step wider. Stored as Int16.
2. THE PRODUCT is an exact integer sum, OP_NT only, as `int8i32.v1`.
3. THE PIECES. A code is `c = hi * 128 + lo` with `lo = c mod 128` in
   [0, 127] (floor mod) and `hi = floor(c / 128)` in [-128, 127]: both are
   int8. Then `sum(a*b) = HH * 2^14 + (HL + LH) * 2^7 + LL`, four int8 GEMMs.
   Each piece sum is exact in Int32 under a bound on `k` that the lane
   derives and the profile refuses above; the recombination is exact in
   Int64. Show the bound, do not assume it.
4. THE FLOATING STEPS, the only ones: the Int64 sum to float32, round to
   nearest even, as a PINNED seam (the wider sibling of `i32_to_f32_pinned`;
   a vendor's own int64 to float conversion is not trusted until it is shown
   equal on every column), then one multiply by `2^(ea + eb)`, then the flush.
5. PLANS, all the same bits: a flat kernel (every column, the reference
   device plan); the integer matrix units of NVIDIA and AMD through the
   existing `gemm/checks/gemm_int8_mma.mojo` step, four products per tile;
   on Apple the flat kernel first, and the float matrix unit in exact chunks
   only if Lane A's exact-chunk probe passes its gate.
6. THE GATES: device equals the host oracle; every plan equals the flat
   plan; ragged shapes; planted worst cases (every code +16383 or -16383,
   cancelling halves, a row of zeros, the largest admitted `k`); batch
   invariance; a sabotage arm on the device and one on the host, each SEEN
   to fail. And one cross-check no other profile has: the host oracle equals
   the quality lane's PyTorch simulation bit for bit on exported vectors, so
   the arithmetic whose quality was measured IS the arithmetic that ships.
7. THE CONTRACT: a new section of `gemm/IDENTICAL_LOWBIT_CONTRACT.md` with
   its own clauses and DEVIATION numbers. It changes no character of
   `fp32.v1`, `bf16f32.v1` or `int8i32.v1`.
8. THE FLAG: add the row to `PROFILES` in `python/mojolearn/_gemm_profile.py`
   with `gemm` True once the gate has passed on a device, `models` False,
   `quality` None (the number belongs to the model path, not the GEMM).
   Expose the product in `mojolearn.linalg` beside `matmul_int8`.
9. TIME IT, on the H100 and on Apple only, against `fp32.v1` at the same
   shapes on the same box, with quantize, split, recombine and dequantize
   counted. Four matrix unit products must beat one fp32.v1 product for the
   profile to be worth shipping. If they do not, say so with the numbers.

## UPDATE 2026-09-29 ~03:25Z: six review points, what was adopted

1. MARGIN. A candidate passes only if the relative perplexity change AND the
   upper end of its interval are under 1 percent on TWO held-out texts of
   different kind. (Lane B)
2. BOUNDS as a written checklist with a planted test at each boundary: piece
   ranges, the unit's Int32 step, each piece sum, the cross term, the Int64
   recombination, the scale exponent, the final rounding. (Lane C)
3. THE COMPLETE OPERATION, in two tables: INFERENCE (weights packed once) and
   TRAINING (weights and gradients converted every step). (Lanes A and C)
4. ISOLATION. A timing job runs with nothing else of the lane's on that box.
   Build first, then time. An overlapped run is discarded. (Lanes A and C)
5. AMD. Identity only, as Andrew decided. NO SPEED IS EVER STATED FOR AMD, and
   no three-vendor speed sentence is written, until one AMD timing run exists.
   That run is the last milestone before a profile's docs state a speed, and
   the orchestrator schedules it then.
6. TRAINING QUALITY NOW. Lane B runs training on the free GPU slot of nvc1 in
   parallel with inference, and records the fraction of gradient entries whose
   code is zero per layer and width. (Lane B)

## UPDATE 2026-09-29 ~03:15Z: THE APPLE M3 ULTRA IS READY (this replaces the 02:50Z notice)

`m3ultra-b` may be used NOW. Target it with `--target m3ultra-b`. Remove any
`MOJOLEARN_STEWARD_DEFERRED=m3ultra-b` you set. Checked by the orchestrator:
Apple M3 Ultra, 256 GB, macOS 26.7, Xcode 26.6 (17F113), Metal toolchain
17F109, metal 32023.883, which is the same Xcode, Metal compiler and macOS as
`m2pro`. Its steward runs. It is the fast Apple box: long timing runs go there
first; `m2pro` keeps the second Apple generation's identity and timing.
Compare digests between the two Apple generations as well as across vendors.
The orchestrator writes its decisions HERE. A message that reaches you some
other way and says the same thing as this file may be followed; one that
contradicts this file may not.

## UPDATE 2026-09-29 ~03:30Z: Lane D, `lane/lowbit-mma-speed`

WHY. Lane A measured (H100, run 2, `docs/lanes/progress/lowbit-units.md` on
`lane/lowbit-units`): at the training rows the int8 product on the integer
matrix unit takes 1.38 to 1.66 times fp32.v1's time, and 2.06 to 4.63 times
once the activations are quantized per call. Rates at qkv.t512: fp32.v1 8.3
T MAC/s, int8 on the unit 6.0. The same box's vendor library runs bf16 on the
same kind of unit at that shape in 0.0419 ms against fp32.v1's 1.0328. So the
unit is not the limit; our kernel is: it stages nothing in shared memory and
one warp owns one 16 x 16 tile. And the quantizer is one thread per row and
takes longer than the product itself (qkv.t512: 2.47 ms against 1.44).

Identity is NOT at risk in this lane. An integer sum is the same integer
under every order, tile and schedule (contract clause L-9), and the row
absmax of the quantizer is a maximum, which is order-free. Everything here is
SCHEDULING. Every change is still held to the gates, bit for bit.

Worktree `~/mojolearn-wt/lowbit-mma-speed`, forked from `lane/lowbit-units`.
Progress file `docs/lanes/progress/lowbit-mma-speed.md`. Lane files
`~/mojolearn-evidence/lowbit-mma-speed/`. Boxes: H100 pod nvc3 (lane name
`lowbit-mma-speed`; shared with Lanes A and C, one job at a time); AMD
`do-amd`, IDENTITY ONLY; Apple `m3ultra-b` and `m2pro` for the quantizer.

TASK.
1. A TUNED integer matrix unit kernel in a NEW file,
   `gemm/checks/gemm_int8_mma_tuned.mojo`. Leave `gemm_int8_mma.mojo` as it
   is: it is the reference plan and Lane C builds on it. Read how fp32.v1's
   tuned kernel in `gemm/checks/gemm_identical.mojo` stages operands, tiles a
   block and spends its registers, and how Modular's own matrix unit kernels
   do (`~/CascadeProjects/upstream/modular/max/kernels/src/linalg/`, read
   only). Levers, in the order to try them: operands staged in shared memory
   per block; more fragments per warp (a larger output tile per thread);
   wider blocks; packed operand loads; the k loop unrolled over steps.
2. A PARALLEL QUANTIZER with the same codes: the row absmax reduced in
   parallel, then the codes written by many threads per row.
3. THE GATE: tuned plan == reference unit plan == flat plan == host oracle,
   on the existing shapes, ragged ones and planted worst cases; the parallel
   quantizer's codes and exponents == the existing quantizer's and the
   host's; a sabotage arm for each, SEEN failing.
4. TIME with Lane A's harness (`bench/gemm_lowbit_price_main.mojo`), adding
   the tuned plan and the parallel quantizer as arms, on the H100, against
   fp32.v1 at the same shape, isolated (build first, then time, nothing else
   of yours on the box). Report after each lever what it bought, including
   the levers that bought nothing.
5. AMD: port the tuned schedule to the MFMA step and run the gate once for
   identity. Nothing is timed there.
6. THE TARGET that matters: four products of the tuned kernel plus the
   conversions must take less time than one fp32.v1 product at the training
   rows, because the 15-bit profile (Lane C) needs four. Say plainly how far
   from that the kernel is after each lever.

## UPDATE 2026-09-29 ~03:35Z: Lane B, two finalists; the int8 rescue experiments stop

Andrew's reviewer, adopted by the orchestrator:
- STOP expanding the int8 rescue arms. `int8i32.v1` throughout misses the bar
  and no selective row rescued it. Finish what is already running; add none.
- TWO FINALISTS, each run as ONE COMPLETE CONFIGURATION (every projection and
  every attention product replaced at once), on BOTH held-out texts:
    F1  15-bit codes on the projections AND on the attention products.
    F2  15-bit codes on the projections, int8 codes on the attention products.
  Errors interact. A pass of the projections alone and a pass of the
  attention alone do NOT establish a pass of the combination; only the
  combined run does.
- THE INTERVAL. State in the table how every interval is computed (what is
  resampled, how many windows, the level), and say that it bounds the
  sampling error on THESE texts and says nothing about other text or tasks.
- TOP-1 AGREEMENT is the share of evaluated positions, each with the true
  context supplied, where the arm's top token equals the baseline's. It is
  not a rate of changed tokens in generated text: free-running generation
  diverges from the first changed token on. Word it that way.
- A TASK EVALUATION is owed for the finalists. If the R2 store holds no task
  set, say so in the progress file; the orchestrator stages one. Do not
  download one onto the pod.
- TRAINING stays its own gate, judged separately, for both finalists.

## UPDATE 2026-09-29 ~03:50Z, Andrew's decision: AMD IS TIMED NOW (this replaces "AMD is identity only")

Andrew, after the first timing table: "ok can we start timing on the amd".
So from now on the AMD box is timed like the other two:
- Every lane that times on the H100 and on Apple times on `do-amd` as well,
  the same arms, the same shapes, the complete operation in both tables
  (inference, training), against fp32.v1 on that box. The vendor library
  comparison arm (hipBLASLt through torch) runs there too, comparison only.
- The results table's AMD columns hold times. "not timed (identity only)" is
  retired. A cell with no run reads "not run yet".
- ISOLATION matters most on this box: it is ONE GPU shared by every lane and
  by other sessions. Submit timing as a steward speed job (the steward
  refuses to time beside another job). One job per lane at a time.
- AMD codegen is stable only from a warm cache: build, run once untimed, then
  time. Never wipe the Mojo cache there. Compare numbers and digests, never
  `.so` files.
- Lane A: your first AMD job (request 1790650025428) ran the timing before
  the identity-only decision reached you. If its runs were taken alone and
  the harness was the same as run 2's, its times may be reported, labeled
  run 1; otherwise rerun. Either way take a run of record now.
- The rule that stays: no sentence states a speed for a box that has no run.

## UPDATE 2026-09-29 ~03:55Z: what the first AMD times say (run 1, one run, read with care)

From Lane A's first AMD job (`~/mojolearn-evidence/lowbit-units/mi325x-run1/`,
MI325X, one run, five timed calls, median; NOT yet a run of record). int8 on
the integer unit, time over fp32.v1 on the same box, and the same with the
activations quantized per call:

| shape | H100 unit | +q | MI325X unit | +q | M2 Pro chunk | +q |
|---|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 1.390 | 3.938 | 0.279 | 4.138 | 0.483 | 0.680 |
| mlp_up.t512 | 1.387 | 2.179 | 0.357 | 1.939 | 0.463 | 0.521 |
| mlp_down.t512 | 1.660 | 4.631 | 0.339 | 5.677 | 0.456 | 0.639 |
| lm_head.t512 (capped) | 1.377 | 2.056 | 0.361 | 1.736 | 0.461 | 0.512 |

TWO THINGS FOLLOW, both for Lane D:
1. THE SAME SCHEDULE is about 0.3 of fp32.v1's time on the AMD unit and
   about 1.4 on the NVIDIA unit (qkv.t512: 0.267 ms against 1.436 ms, with
   fp32.v1 nearly equal on the two boxes, 0.957 and 1.033). So look FIRST at
   what is NVIDIA-specific in `gemm_int8_mma.mojo`: the two m16n8k32 halves
   per tile, the fragment loads and `_pack4`, the register pack the NVVM
   intrinsic returns, the store. Test each with a forced arm before any
   redesign. Shared memory staging comes after that.
2. THE QUANTIZER IS THE LARGEST COST ON EVERY BOX. On AMD it turns a product
   that takes 0.28 of fp32.v1's time into an operation that takes 4.1. The
   parallel quantizer is as important as the kernel. Do it early.

## UPDATE 2026-09-29 ~04:30Z, Andrew's decisions (three of them)

1. INT8 IS DROPPED AS A MODEL'S ARITHMETIC, and so is the int8-attention mix
   (finalist F2). Andrew chose "Drop int8 only". Lane B: run no further int8
   arm and no F2 arm; cancel your own queued ones that have not started.
   F1 (15-bit everywhere), the weight width sweep and training continue. The
   int8 matrix unit PRODUCT stays, as the piece product of the 15-bit
   profile (Lanes C and D).
2. DO NOT WAIT ON AMD. "time on apple and nvidia ... don't wait for amd".
   Timing is judged on the M3 Ultra (`m3ultra-b`, the first Apple box for
   every timing run) and on the H100. Submit the AMD job alongside and read
   it when it lands; never hold a step, a table or a report for it. A table
   is complete with the AMD cell reading "not run yet".
3. BF16 ON THE MATRIX UNITS IS THE PRIORITY. "bf16 was original concept and
   seems most promising ... start testing that". Lane E below. On nvc3 Lane
   E goes first: Lanes A, C and D keep at most ONE job each in that queue
   and keep it short while Lane E has work to run.

## Lane E, `lane/bf16-units`: native BF16 on the matrix units

THE QUESTION, which nobody has measured here: do the vendors' matrix units,
given the SAME bf16 operands and a float32 accumulator, produce the SAME
bits? It was assumed they do not, from published work on older NVIDIA
units. That is an assumption, and this tree already saw one such assumption
fail: the Apple M4's float unit equals an ascending fused chain exactly
(0 of 29 million mismatches). So measure it.

WHAT IS KNOWN.
- A bf16 has an 8-bit significand, so a product of two is exact in float32
  (short of overflow and of the subnormal range). The ONLY freedom a unit
  has is how it adds: the order, the grouping, the rounding, the width it
  keeps inside one step.
- Quality is not the risk: bf16 on both operands of every product,
  attention included, reads +0.0085% and +0.0127% on the two texts.
- The prize is large: on the H100 the vendor library runs bf16 at qkv.t512
  in 0.042 ms against fp32.v1's 1.03 ms (comparison only).
- `bf16f32.v1` (bf16 widened, fp32.v1's own kernel) has identity and buys no
  time: 1.05 times fp32.v1 at the training rows. It is not this lane.

Worktree `~/mojolearn-wt/bf16-units`, forked from `lane/lowbit-units` (the
harness and its results are in it). Progress file
`docs/lanes/progress/bf16-units.md`. Lane files `~/mojolearn-evidence/bf16-units/`.
Boxes: H100 pod nvc3 (lane name `bf16-units`, FIRST in line there); Apple
`m3ultra-b` first and `m2pro` second; AMD `do-amd`, submitted alongside and
never waited on.

THE UNITS. NVIDIA `mma.sync` m16n8k16 with bf16 operands and an f32
accumulator (sm_80 and later); AMD MFMA with bf16 operands (the
`v_mfma_f32_16x16x16_bf16_1k` family on gfx942); Apple the simdgroup float
matrix unit, fed the bf16 values as float32 (a bf16 IS a float32 whose low
sixteen bits are zero). Reach them the way this tree reaches the int8 forms,
by name through `llvm_intrinsic` (`gemm/checks/gemm_int8_mma.mojo`), or
through the stdlib's `mma` entry, whichever builds. Apple's unit is already
reached in `core/apple_air.mojo` and PLAN_APPLE_MMA.

THE STEPS, in order. Report after each; do not wait for the last.
1. REACH each unit with bf16 operands and show it runs: one tile, one step.
2. DETERMINISM per box: the same operands twice, the same bits.
3. RAW CROSS-VENDOR COMPARISON, the headline. The same operands on every
   box, digests and cell-by-cell differences. Operands of two kinds: REAL
   (weights and activations of SmolLM2-360M from the R2 store, rounded to
   bf16 by the contract's narrowing seam, at the twelve transformer rows)
   and PLANTED (cancelling pairs, a wide exponent spread inside one step,
   sums that land exactly on a rounding tie, one nonzero product per step,
   products in the subnormal range, the largest magnitudes). Say how many
   cells differ, by how many units in the last place, and where.
4. WHICH ARITHMETIC IS EACH UNIT. Fit each unit against reference models
   computed EXACTLY on the host (integers or rationals, never a float
   library's own sum): an ascending chain of fused multiply-adds, round to
   nearest even; the same with truncation; the exact sum of one step's
   products and the accumulator, rounded once, to nearest and truncated; a
   pairwise tree inside the step; each with and without the flush. A model
   FITS only if it equals the unit on every cell, planted cases included.
   Name the model per unit and per box, or say none fits.
5. THE VERDICT, one of three:
   a. SAME BITS on every vendor as they are: then bf16 on the units is an
      identical profile, and the contract names the model they share.
   b. DIFFERENT, but each unit fits a model: then find the ADMISSION
      condition under which the models agree (the technique of
      PLAN_APPLE_MMA's block admission: prove per tile that no step can
      round differently, recompute a refused tile with the exact pinned
      step). Measure on REAL operands what share of tiles is admitted. A
      profile is viable only if most are.
   c. A unit fits no model: say so, with the cases.
6. TIME the bf16 unit kernel as it stands against fp32.v1 at the twelve rows,
   on the H100 and the M3 Ultra, isolated, the complete operation in both
   tables (inference, training), and under (b) with the admission test and
   the fallback counted.
A sabotage arm for every comparison, SEEN failing. No identity claim is made
from samples alone: samples find a difference, a model and a proof make the
claim.

## UPDATE 2026-09-29 ~04:40Z: Lane E, one more step (Andrew's question)

Andrew: "why don't we explore a fixed rounding rule for bf16 to get bitwise
identical AND acceleration". The rounding of VALUES INTO bf16 is ours to fix
and is fixed (contract clause L-2). The rounding INSIDE a vendor's unit is
the hardware's; nobody can set it. There are three ways to get identity
anyway, and Lane E tests all three:
  (a) the units already agree (steps 3 and 4);
  (b) they agree inside an admission window, exact fallback outside (step 5);
  (c) MAKE THE ROUNDING NEVER FIRE. Step 4b below.

4b. EXACT OPERANDS ON THE BF16 UNIT. Feed the bf16 unit operands whose
    every partial sum is exactly representable in float32, so that no
    rounding rule, whatever it is, is ever exercised. Integers of magnitude
    up to 128 are exact in bf16; their products are integers; a sum of them
    is exact in float32 while it stays below 2^24, so `k` is cut in chunks
    and the chunk sums are carried in an integer (Lane A's Apple
    exact-chunk probe did this on Apple's float unit and its bits equal the
    NVIDIA and AMD integer units'; its bound for a piece product of 128 x
    128 is 1023 steps). Run the SAME integer operands through the bf16 unit
    of each vendor and through the integer unit, and show the bits equal.
    Then time the bf16 unit against the integer unit on the H100 with the
    same operands: if the bf16 unit is the quicker one, the 15-bit profile's
    four piece products should run on it.
    SAY PLAINLY what this route costs: exact sums need a fixed scale per
    row, which is what bf16's per-value exponent gives up. bf16 with its own
    exponents passed quality in ONE product; fixed 8-bit codes did not, and
    fifteen bits need four products.

## UPDATE 2026-09-29 ~05:00Z, Andrew: FINISH THE MIXED PROFILE'S MEASUREMENT (this replaces point 1 of the 04:30Z update for Lane B)

Andrew: "Finish the complete 15-bit and mixed-profile measurements already
underway. you didn't cancel anything did you?" One job WAS cancelled on the
orchestrator's instruction: nvc1-0009, `finalists.sh` (F1 and F2 on both
texts, and their training arms), cancelled at 03:32Z before it started.
That was the orchestrator's error. So:
- RESUBMIT the finalists job AS IT WAS COMMITTED: arms a, floor, F1 and F2
  on both texts, and the training arms of F1 and F2 (forward, then forward
  and backward). Both are measured to the end as complete configurations.
- "Dropped" now means only this: int8 and the mix are not offered by the
  flag. It does not mean their measurement stops. A measurement that was
  under way is finished and reported with its numbers.
- CANCEL NOTHING from here on, queued or running, yours or anyone's, unless
  Andrew says so in the brief. Add no NEW int8 rescue arm; finish every arm
  that was already planned or queued.

## UPDATE 2026-09-29 ~05:20Z, Andrew's decisions: IDENTITY FIRST; QUALITY STOPPED; BF16 DROPPED

Andrew: "we need to test for IDENTITY... focus on identity and that should be
fast because of the harness... maybe start there for everything", "lets
drop bf16", "STOP THE QUALITY".
- LANE B (quality) IS STOPPED by the orchestrator on Andrew's order. Its two
  running jobs (nvc1-0007 training, nvc1-0008 sweep) are left to finish so
  that nothing measured is destroyed; its two queued jobs (nvc1-0010
  finalists, nvc1-0011 bf16 training) were cancelled before they started and
  can be resubmitted with one command each. No lane does quality work now.
- LANE E (native bf16 on the units) IS STOPPED and bf16 is dropped. It had
  submitted no job.
- WHAT REMAINS is the 15-bit profile, and the order of work for it is
  IDENTITY FIRST:
    1. identity of the GEMM on every box through the repo's own verification
       harness (the identity harness and `mojolearn verify`, the way the
       lanes `gemm-int8` and `gemm-bf16` are in it): Apple M3 Ultra, Apple
       M2 Pro, NVIDIA, and AMD alongside without waiting on it;
    2. identity of a whole model's logits under the profile on the same
       boxes, which needs the blocks to compute under it;
    3. only then speed.
  A step's identity verdict is reported the moment it exists.
- Quality already measured stays on record and is not rerun: 15-bit on every
  product reads -0.0015% and +0.0055% on the two texts.

## UPDATE 2026-09-29 ~05:45Z: the flag branch now HOLDS the flag (Lane C's finding, fixed)

Lane C found that `origin/lane/lowbit-flag` at 078fab2c8 held only a rename
and none of the content its message described. That was the orchestrator's
error (a failed `git add` with its error hidden). The content is committed
and pushed in the commit after it; verified on origin: `DEFAULT = "fp32_v1"`,
the rows `fp32_v1` and `fixed15_v1`, `numeric_profile=` on the loaders, and
the tests. Lane C may merge `origin/lane/lowbit-flag` now.

## UPDATE 2026-09-29 ~05:45Z: STEP 1 IS CLOSED; what follows identity (Andrew)

The int15 GEMM is IDENTICAL on the H100, the M3 Ultra, the M2 Pro and the
MI325X: 178 cases, every plan equal to the host oracle, the host oracles
equal across two x86-64 and two arm64 hosts, every arm that must fail seen
failing on every box. Andrew: "if this 15 bit converter thing is bitwise
identical lets then do speed and quality for it". So, for the 15-bit
profile only:
- Lane C goes on with step 2 (the verification harness) and step 3 (a whole
  model's logits) and MAY TIME alongside: the rule "no timing until steps 1
  to 3 have verdicts" is lifted for the GEMM, whose identity is closed.
- Lane D's tuned kernel and parallel quantizer are what the timing should
  use once their gates of record are green.
- Quality for the 15-bit profile restarts as its own job when the
  orchestrator says so here.

## UPDATE 2026-09-29 ~06:10Z: prior work to read, and two options held in reserve (no redesign)

The plan does not change. Building a wide product from exact int8 products
on the matrix units is known work; read it for splitting, bounds and
recombination, and do not redesign around another scheme:
- Mukunoki et al. 2020, accurate and reproducible GEMM on Tensor Cores.
- Ootomo, Ozaki, Yokota 2024, DGEMM on integer matrix multiplication units
  (code: github.com/enp1s0/ozIMMU).
- Uchino, Ozaki, Imamura 2024, fewer low-precision products and fewer wide
  additions in the Ozaki scheme.
- Gupta et al. 2015, training with 16-bit fixed point and stochastic rounding.
Ours differs: the codes ARE the values (a model is quantized to them), where
that work rebuilds a float64 product from slices. Their error bounds and
their speedups do not carry over; our bounds are the contract's own.

HELD IN RESERVE, not started, each a new profile name if it is ever built:
1. THREE PRODUCTS INSTEAD OF FOUR. With pieces of six magnitude bits the
   sums `H + L` still fit an int8, so `HL + LH = (Ha + La)(Hb + Lb) - HH - LL`
   and a GEMM needs three unit products. That carries 12 or 13 bit codes.
   Lane B measured 12 bits on both operands at +0.18% (upper end +0.22%) on
   enwik8. Worth it only if the four-product timing is close to, but not
   under, fp32.v1's.
2. A PINNED STOCHASTIC ROUNDING FOR GRADIENTS. At 15 bits 58.8% of the head's
   gradient entries round to a zero code. If the five-seed training run shows
   harm, rounding up with a probability equal to the fraction, drawn from a
   counter-based generator keyed by step, tensor and index, is the known
   remedy, and it is bitwise reproducible on every vendor because the draw is
   a pure function of integers.
LANE C, one check the review asked for: the converter times Lane D reported
are the int8 converter's. Your timing must count what the 15-bit path really
runs per call: quantize to 15-bit codes, split into two pieces, and on the
way out recombine and dequantize.

## UPDATE 2026-09-29 ~06:15Z: Lane B may use pod nvc2 as well (lane name `lowbit-quality-b`)

The five-seed 15-bit training job (nvc1-0012) is queued behind two jobs of
the dropped arms on nvc1, and nothing is cancelled. Pod nvc2 (2x RTX 4090)
is free. Lane B may set up a second tree there under the lane name
`lowbit-quality-b` (assigned to nvc2 by the orchestrator): sync the same
worktree, install the environment, stage the data from R2, and run the
15-bit training there, one seed group per GPU slot. Leave nvc1-0012 queued
as it is; whichever finishes first is reported, and both are kept.
nvc2 deletes itself after 30 idle minutes, so put work on it promptly.

## UPDATE 2026-09-29 ~06:30Z: the NVIDIA cost is found (Lane D); Lane C applies the fix

Lane D, job nvc3-0016, forced arm plus a PTX counter: the reference
`_pack4` in `gemm/checks/gemm_int8_mma.mojo` loads a fragment word with
`unsafe_load[width=4]` and states no alignment, and on NVIDIA the compiler
emits a loop of four byte loads per word. The same load with `alignment=4`
is one 32-bit load. With ONLY that changed, bits equal to the reference, the
flat plan and the oracle: qkv.t512 1.436 ms to 0.398 ms (over fp32.v1: 1.36
to 0.377), mlp_up.t512 4.625 to 1.248.

DECISION (orchestrator): the fix goes into the reference file, and LANE C
makes the edit, because Lane C's four piece products run through that step
and one lane editing the file avoids a conflict. Lane D does not edit it.
- The change: `p.unsafe_load[width=4, alignment=4](base + k0)` in `_pack4`,
  and the same in `_pack8`.
- It is valid ONLY where the load is aligned, which is the condition the
  function already tests before it takes the vector load (`k % 4 == 0`, `k0`
  a multiple of four, inside the row). Show that the base pointer of every
  buffer that reaches it is itself aligned to four bytes, the piece planes
  of the 15-bit path included, and keep the byte path for everything else.
- It is scheduling and moves no bit, and it is still held to the gates: the
  int8 gate and the int15 gate, with their sabotage arms seen failing, on the
  H100 and on the MI325X, before any time is read. Apple does not run this
  step.

## UPDATE 2026-09-29 ~06:50Z: THE BACKWARD QUANTIZATION RULE comes before any new orientation (Lane C, Lane B)

A review point, adopted. A tensor has DIFFERENT codes in the forward and the
backward product, because the scale is one power of two per ROW ALONG THE
CONTRACTED EXTENT and the two products contract different extents. So
"quantize the transposed tensor" and "transpose the quantized tensor" are
different arithmetic, and the profile must say which it is, ONCE, for the
reference, the quality experiment, the device kernels and the CPU path.

THE RULE (the orchestrator's choice, to be written as a contract clause by
Lane C and checked against Lane B): EVERY PRODUCT QUANTIZES ITS OWN OPERANDS
FROM THEIR FLOAT32 VALUES, along that product's own contracted extent.
Codes are never carried from one product to another and never transposed.
- Forward, `Y = X W^T`: X by token (each row of X over the features), W by
  output feature (each row of W over the input features).
- Weight gradient, `dW = dY^T X`: the contraction is over the tokens, so dY
  is quantized by OUTPUT FEATURE over all tokens and X by INPUT FEATURE over
  all tokens.
- Input gradient, `dX = dY W`: the contraction is over the output features,
  so dY is quantized by TOKEN over the output features and W by INPUT
  FEATURE over the output features (the columns of W).
This is what Lane B's training simulation already does for the weight
gradient ("one row per output feature over all tokens"). Why this rule and
not the other: it is the only one under which a per-row scale on both
operands is a per-cell scale of the output, which is what makes the integer
sum the product (contract section 0).

LANE C: write the clause with its three cases; say what your transposing
quantizer computes and show it is this rule; export backward vectors from
Lane B's training simulation and hold the host oracle and the devices to
them bit for bit, as you did for the forward product. Do this BEFORE any
kernel in another orientation.
LANE B: state in your progress file, for each of the three products, which
extent your simulation quantizes along, so the two lanes can be compared
line by line.

## UPDATE 2026-09-29 ~06:50Z: the flag separates inference from training

On `origin/lane/lowbit-flag`: a row of `PROFILES` has `inference` and
`training` (the field `models` is gone), `resolve(..., use="training")` is
what a trainer calls, and the trainers refuse by name a profile whose
training gates have not passed. `fixed15_v1` reads False for both today.

## UPDATE 2026-09-29: backward vectors exist; the quality simulation follows the rule (Lane B to Lane C)

- Lane B's training simulation follows the backward quantization rule in all
  three products (and in the attention products' six cases); statement in
  its progress file at 8a6589cb3. So the training numbers measured so far
  are for the arithmetic the kernels will compute.
- BACKWARD VECTORS, in Lane C's own file format:
  `~/mojolearn-evidence/lowbit-quality/backward_vectors/int15_backward_vectors.q15`
  (884,750 words, manifest beside it): 18 cases, the three products of four
  projections and both attention products, training step 50 of seed 0.
- THEIR LIMIT: they tell the rule apart from the other rule (codes carried
  and transposed) in only three of the six input gradients, because at step
  50 every weight row still has the same exponent. A check that cannot tell
  the two rules apart does not check the rule. Lane B exports a second file
  from a LATER step where the rules differ in every case; Lane C holds its
  oracle and devices to both files and says, per case, whether the case can
  tell the rules apart.
- THE PIN: Lane B's `bench/lowbit_quality/arith.py` gained 18 lines, none of
  them arithmetic; its blob moved from 17337a5c to c60a455c. Lane C's pin
  check will refuse it until Lane C moves the pin, after reading the diff
  and confirming no arithmetic changed.

## UPDATE 2026-09-29: THE ATTENTION PRODUCT P.V, and what waits on the timing (orchestrator's decision)

Lane C's finding, accepted: under rule W-9 the product P.V (probabilities
times values) contracts over the KEYS, so V's scale is an absmax over the
span of keys, and the span is L at prefill and i+1 at decode step i. The
same V values get different codes, the logits of position i depend on how
many tokens were in the call, and decode == prefill fails by construction.
The projections, the head and Q.K^T are prefix invariant and are not
affected. Lane B's simulation has the same property and never saw it,
because it evaluates prefill only.

DECISION.
1. THE BLOCKS ARE NOT STARTED YET. Whether the profile is built into the
   models at all waits on two timing results: Lane C's run 2 on the H100
   (with the stated loads) and Lane D's rerun. The orchestrator says in this
   file when the blocks may start.
2. IF THE PROFILE GOES ON, P.V STAYS ON fp32.v1 in its first version: the
   pinned ascending chain the block runs today (`attn_context_kernel`), which
   is prefix invariant and already identical on every vendor. The profile
   then covers the projections, the head and Q.K^T. This adds no new
   arithmetic and no new clause. Its quality is one run for Lane B: 15-bit
   on the projections and Q.K^T with P.V in fp32.
3. Lane C's proposed rule for P.V (V quantized per key and cached; the
   key's power-of-two scale folded into P exactly; P' quantized per query)
   is sound and is HELD IN RESERVE as a later version under its own name,
   with its own clause and its own quality run. Do not build it now.
4. THE FLAG: a profile's `products` row may therefore name three families,
   `projections`, `attention_qk`, `attention_pv`. The orchestrator edits the
   module when the profile goes on.
5. GO ON NOW WITH, in this order: the gates job with the stated loads, timing
   run 2 on the H100, the Apple plan on the float matrix unit and its timing
   on the M3 Ultra, the harness verdict for gemm-int15. The backward vectors
   after those.
6. A WHOLE-MODEL IDENTITY CHECK MUST INCLUDE decode == prefill at several
   prefix lengths. Lane C found this one by reading; the harness would have
   found it late.

## UPDATE 2026-09-29: JOIN THE TWO LANES' WORK ON THE H100 (the number Andrew is waiting for)

Measured on the H100, each against fp32.v1 at the same row in the same run:
- Lane C, run 2 (nvc3-0023), the COMPLETE 15-bit call on the REFERENCE unit
  kernel with the stated loads: 1.21 to 1.45 at the training rows (run 1:
  2.80 to 3.42). Still more time than fp32.v1.
- Lane D (nvc3-0022), the parallel quantizer plus FOUR TUNED int8 products
  as one operation: 0.31 to 0.50 at the same rows. One tuned product alone:
  0.08 to 0.125.
So the complete 15-bit operation on the tuned kernel is the number that
decides the project, and it does not exist yet. It is now FIRST for both:
- LANE D: finish `identical_gemm_int8_pieces_tuned_into` (two int8 planes per
  operand in, the three exact Int32 sums HH, HL+LH, LL per cell out, one
  staging of the operands), gate it bit for bit with its sabotage arms, push,
  and tell Lane C the commit.
- LANE C: take it from `origin/lane/lowbit-mma-speed` as a plan of
  `int15i64.v1` (the recombination, the Int64 seam and the dequantization
  stay yours), hold it to the int15 gate and the simulation check with every
  arm seen failing, then time the complete call on the H100 with it, run
  beside run 2.
- The bound on k holds for the tuned plan too: HL + LH share one Int32.
Apple's float-unit plan and the M2 Pro failure come after this number.

## UPDATE 2026-09-29: Lane B, one more configuration: THE ONE THAT WOULD SHIP

The five-seed training of the complete 15-bit configuration passed (forward
+0.298%, forward and backward +0.146% at step 4000, both inside the noise
floor of 0.853%). That configuration has BOTH attention products on 15-bit.
The first version of the profile keeps P.V on fp32.v1 (see the section on
the attention product P.V), so the configuration that would ship is:
    projections and the head: 15-bit on both operands
    attention Q.K^T:          15-bit on both operands
    attention P.V:            fp32, as the block computes it today
Measure exactly that, under the name F1-pv32:
- inference on both held-out texts, the same windows and the same interval
  as the other arms;
- training at five seeds, forward only and forward and backward, the same
  baseline seeds.
Pod nvc2 is free; use it (lane name `lowbit-quality-b`). Nothing else is
added. Cancel nothing.

## UPDATE 2026-09-29, Andrew: PORT THE TUNED KERNEL TO AMD NOW, AND TUNE APPLE TOO. Lanes F and G.

Andrew: "why don't we port the tuned kernel to amd?" and "can we have the
tuned kernel approach on apple too?" The orchestrator had put the AMD port
after the H100's joined number; that ordering was wrong, the two do not
depend on each other. Two new lanes, in parallel with C and D.

### Lane F, `lane/lowbit-amd-tuned`: the tuned integer unit kernel on AMD

Worktree `~/mojolearn-wt/lowbit-amd-tuned`, forked from
`lane/lowbit-mma-speed` (Lane D's tuned kernel, its one-staging four-product
kernel, the parallel quantizer, the harness and the job scripts are in it).
Progress file `docs/lanes/progress/lowbit-amd-tuned.md`. Lane files
`~/mojolearn-evidence/lowbit-amd-tuned/`. Box: `do-amd` (MI325X, gfx942),
through the steward, speed jobs; it is TIMED. One job of yours at a time.
WHAT IS KNOWN: on the MI325X the UNTUNED reference kernel already takes 0.28
to 0.37 of fp32.v1's time for one int8 product, and the complete 15-bit call
on it 0.66 to 0.78 at the training rows. On the H100 tuning took one product
from 1.43 ms to 0.12 ms at qkv.t512 (alignment stated, 16-byte staging
loads, 64 k steps a window, 32x32 per warp, 16 warps a block).
TASK: make Lane D's tuned plans and its four-products-one-staging kernel run
on the AMD integer unit (MFMA `v_mfma_i32_16x16x32_i8`, a wavefront of 64),
in AMD branches or a new file, WITHOUT changing what the NVIDIA branches
compute or how fast. Then: the gate bit for bit (tuned == reference ==
flat == oracle, planted worst cases, the largest k 65536), sabotage arms
seen failing; then the levers one at a time with a time after each, as Lane
D did, because the best tile on a 64-wide wavefront is not the H100's; then
the complete operation, quantizer plus four products, over fp32.v1 at the
twelve rows. Warm cache: build, run once untimed, then time. Never wipe the
Mojo cache on that box; compare numbers and digests, never `.so` files.
Lane D owns `gemm/checks/gemm_int8_mma_tuned.mojo`; if you must edit it,
touch only AMD branches and say so, so the merge is clean.

### Lane G, `lane/lowbit-apple-tuned`: the tuned approach on Apple's float unit

Worktree `~/mojolearn-wt/lowbit-apple-tuned`, forked from
`lane/lowbit-int15` (Lane C's 15-bit profile, its Apple float-unit plan in
two forms, Lane A's int8 exact-chunk probe, Lane D's code merged in).
Progress file `docs/lanes/progress/lowbit-apple-tuned.md`. Lane files
`~/mojolearn-evidence/lowbit-apple-tuned/`. Boxes: `m3ultra-b` FIRST (it is
idle most of the time; keep it busy), `m2pro` second, through the steward.
WHAT IS KNOWN: Apple has no integer matrix unit. Its simdgroup float unit
computes exact integers while every partial sum stays below 2^24. Lane A's
int8 exact-chunk probe takes about the same time as fp32.v1 on the M3 Ultra
(1.017 at qkv.t512) and about half on the M2 Pro. The 15-bit profile on the
plain kernels takes 12 to 14 times fp32.v1 on the M3 Ultra. Lane C wrote the
float-unit plan of the 15-bit profile in two forms (two products, the left
operand whole, carried every 8 steps; four products carried every 512) and
is running their gate and first timing; read its progress file for the
verdict before you start and do not redo that.
TASK: tune it the way Lane D tuned the NVIDIA kernel, ONE LEVER AT A TIME
with a time after each, the gate before the clock: tile geometry per
simdgroup and per threadgroup (the existing knobs
`-D MOJOLEARN_APPLE_MMA_{SGM,SGN,FM,FN,KB}` of PLAN_APPLE_MMA are the model);
operands staged in threadgroup memory; stated alignment on every vector
load; what the carry into integers costs and how rarely it can be done
(the chunk bound is the contract's, prove any chunk you use); the codes
kept as float32 planes so nothing is converted per step; the quantizer's
launch fused with the product's. Target: the COMPLETE 15-bit operation
under fp32.v1 at the training rows on the M3 Ultra, and say plainly how far
from it each lever leaves you. Work in NEW files; Lane C owns
`gemm/checks/gemm_int15*.mojo`.
Every change is held to the int15 gate and the simulation check on both
Apple boxes, planted cases at every chunk boundary, a sabotage arm that
removes the boundary, seen failing. The M2 Pro has a known open failure
(a plain kernel left cells unwritten at a large shape); Lane C is
reproducing it; do not rely on the plain kernels at large shapes there.

## UPDATE 2026-09-29: macOS ABORTS A LONG METAL LAUNCH SILENTLY (Lane C's finding; binds every lane on Apple)

On the M2 Pro a launch that held the GPU for about 4 seconds was aborted by
macOS ("Execution of the command buffer was aborted ... Impacting
Interactivity", kIOGPUCommandBufferCallbackErrorImpactingInteractivity). The
output was left PARTLY written, `synchronize` reported nothing, and the
caller read a product that was partly the buffer's old contents. It is
intermittent and it is not the thread limit.
EVERY LANE ON APPLE:
- Bound every launch in work, so that no command buffer holds the GPU for
  more than a fraction of a second on the SLOWEST Mac (Lane C uses slices of
  at most 2^30 multiply-accumulates with a wait between them).
- Every gate and every timing run POISONS its output first and reads it back
  whole at the large rows; a timing whose output was not checked is not
  reported.
- If a result is wrong only at a large shape and only sometimes, read the
  system log for that line first.
Lane G: Lane C left you an UNBUILT patch for the four-code staging load,
`~/mojolearn-evidence/lowbit-int15/apple_four_code_staging_UNBUILT.patch`
(against `gemm/checks/gemm_int15_apple.mojo` at d4c9d8a34). Take it or not.
