# Resume here

Written 2026-08-19 at the end of a long session. Read this, then `UNWIRED.md`,
then `PORTED_MAP.tsv`.

## What this repository is

A port of **CatBoost's GPU oblivious (symmetric) tree learner**
into Mojo, targeting Metal first. Nothing from mojotrees. `gbdt/` mirrors
CatBoost's own paths file for file (their constant `catboost/cuda/` prefix
dropped, and `cuda_util` renamed `gpu_util` because CUDA is not our
vocabulary). `mojo_only/` is what CatBoost never had to write.

**The one rule: COPY, DO NOT IMPROVE.** Every deviation is a confound when a
number finally arrives, and every deviation is recorded in `PORTING.md`.

## State: what works

- All three packing policies: binary 32 features per `UInt32`, half-byte 8,
  one-byte 4. Verified on device.
- Six histogram kernels: direct and gather for each policy. The gather
  variant is verified against the direct one under a REVERSED index.
- Scan, sibling subtraction, zero, copy, `WriteReducesHistograms`, score and
  argmax, split flags, segmented stable partition, gathers, partition update,
  per-leaf stats, leaf values. All verified against hand calculations.
- Uniform-binary trees grow correctly at depths 1 to 8, rows conserved, 46 of
  64 leaves populated at depth 6.
- The compressed-index layout builder, per-policy feature blocks, and
  bin-feature to (feature, bin) resolution: all verified on a mixed dataset.
- CatBoost's option names with `check()` refusing every unported option BY
  NAME, plus a three-level `determinism` ladder that has no CatBoost
  counterpart.
- Harness moved in from mojotrees: interleaved arms, machine lock, box
  record. Plus `tools/remote_gpu.sh` for the nvidia and amd columns.

## State: what is broken

Nothing known. Mixed-width trees now grow, split and conserve every row:
depth 4 gives 16 of 16 non-empty leaves, depth 6 gives 44 of 64 against the
uniform binary tree's 46. `UNWIRED.md` has the full trail of nine exclusions
and seven real bugs.

The cause was `TPointHistOneByte::Reduce`, which is TWO stages in CatBoost
(`hist_one_byte.cu:177-230`) and had been collapsed into one strided fold.
The second defect was the test data: `check_mixed_tree` planted bins as
`x % folds[f]`, but `Folds` is the border count and a feature takes bins
`0..Folds`, so every binary feature was the constant 0.

**READ THIS BEFORE WRITING ANOTHER CHECK.** Both defects were invisible to
the checks written to catch them, for the same reason. A single strided fold
gives the RIGHT answer whenever every cell holds the same value, and
`check_one_byte_bits` assigned bins `(r + f) % n_folds`, which makes every
cell equal. Two separate exclusions therefore reported the kernel correct at
exactly the parameters that were failing. Same kernel, same parameters, only
the bin pattern changed:

    uniform   (r + f) % 64      0 wrong of 512
    scattered hashed          490 wrong of 512

A histogram check whose expected value is the same in every cell verifies the
total and nothing about placement. Plant SCATTERED bins and compare per cell
against a host tally. Likewise, conservation cannot see a tree that never
splits, so both tree checks now require `nonempty >= 2`.

## State: what is next

Every timing number in this port was measured on 32 uniform binary features,
which is one policy, one launch per level and a 64-cell histogram. That is
the shape CatBoost's design is LEAST suited to and the one it was least
written for. The mixed path is what needed fixing before those numbers meant
anything, and it now works, so the whole timing table should be re-measured
on a wide mixed dataset before any of it is quoted.

## The method that has worked, and the one that has not

Five hypotheses have been killed on this bug. **Every one was killed by a
measurement in a single attempt. None were killed by reasoning.** Three
inferences each found a REAL bug without finding the one being chased:

1. compressed-index base conflated with the feature-block stride
2. `WriteReducesHistograms` never ported (CatBoost keeps TWO histogram
   layouts and this port only had one)
3. the block-to-flat bridge duplicated, advancing the offset twice

When stuck: **read the bytes.** `mojo_only/mixed_hist_probe.mojo` localized in
one run what three rounds of inference could not.

## Rules earned here, do not relearn them

- A kernel is not ported until it has been **enqueued**. Compiling is not
  evidence; `--emit=object` targets the host.
- Kernel params must be `Int32`, pointers `MutPointer[T, MutAnyOrigin]`.
- `enqueue_copy` is host<->device ONLY. Passing a device pointer as
  `src_ptr` silently does nothing.
- One host staging buffer per `enqueue_copy`; they are asynchronous.
- Only interleaved arms inside ONE process compare. This box drifts 2-3x.
  The harness overturned a committed claim the first time it ran.
- A configuration that cannot be varied inside one process cannot be
  measured here, so make it a parameter.

## Measurements that stand

- Replication pays **8.56x on a wide histogram** (1024 cells) and is
  INDISTINGUISHABLE on a narrow one (64 cells). `replicas_for` keys on output
  size. Fitted to two points; the interval between them is unmeasured.
- **32 binary features is the WORST shape for CatBoost's design.** Every
  timing conclusion in this repo comes from it and should be re-taken on a
  wide dataset once mixed trees work.
- Depth-8 uniform-binary tree, 500k x 32: roughly 45-70 ms depending on
  thermal window. Not comparable to anything without interleaving.

## The comparison, MEASURED (Aug 19 2026)

CatBoost 1.2.10 was installed and run. `tools/catboost_reference.py` is the
script; quantization is moved out of the timed region with `pool.quantize()`
because our port consumes a compressed index and never builds borders, so
leaving it in charges CatBoost for work we do not do (it cost them 10 ms/tree).

800k rows x 100 one-byte features, depth 6, symmetric:

    our GPU port        50.0 ms/tree
    CatBoost CPU        30.1 ms/tree   (10 cores)

**1.66x behind, and this figure is SUPERSEDED -- see
`bench/results/WINDOW_2026-08-20_interleaved-254.md`, which reads 35.35
ms/tree against their 30.06 on the same data (1.18x).** Two sentences that
stood here are now false and are deleted rather than annotated: this 50 ms
was tree growth alone with the gradients handed to it, and there was no
boosting loop to charge. The loop was ported later the same night (see "And
then the boosting loop gave back 17 ms/tree" below), so the interleaved
harness times `fit` -- gradients, tree, model update -- and the
flatters-us caveat does not apply to any interleaved number in this file.

CatBoost's GPU arm cannot run on this box at all, so this is our GPU against
their CPU.

### Why we are slower, and it is not the kernels

Row sweep at 100 one-byte features, depth 6:

    100k  37.2 ms      400k  43.3 ms
    200k  37.6 ms      800k  50.0 ms

**8x the rows costs 1.34x the time.** Extrapolated to zero rows, ~34 of the
50 ms is FIXED per tree and independent of the data. Depth 1 at 800k is
5.4 ms and 6 x 5.4 = 32, which lands on the same floor.

`run_tree_layout` runs **9 `ctx.synchronize()` and ~16 kernel launches per
level**, so 54 host round trips and ~96 launches per depth-6 tree.

`mojo_only/sync_price.mojo` prices them, and it CORRECTED the obvious guess:

    54 bare drains        0.76 ms     (0.014 ms each, nearly free)
    54 copy + drain      13.0  ms     (0.24 ms each)

The barrier is not the cost; the HOST TRANSFER attached to it is. 13 ms is
about 38% of the fixed 34, and the rest is launch count. Five of the copies
per level exist only to broadcast one split descriptor (offset, shift, mask,
one-hot flag, bin) into five separate buffers.

### What this means for porting

The paragraph that stood here said the deficit was the CONTROL PLANE. **A
2026-08-19 measurement falsified it and it is deleted rather than
annotated.** `sync_price.mojo` priced the drains and the copies but never
the kernel time between them. Counters on the live fit read 77 launches and
12 drains per depth-6 tree (12 = CatBoost's own 2/level discipline, so the
scheduling already matches theirs), and at this device's measured dispatch
prices (191 us per launch+drain, 23 us per undrained launch) ALL dispatch
comes to 3.8 ms of the 41.7 ms fixed cost. Nine percent. The remaining
~38 ms is kernel time on the row-independent histogram footprint: 64 leaves
x 100 features x 255 bins x 2 stats = 3.26M cells zeroed, flushed, scanned,
subtracted and scored per level. Fixing dispatch cannot return more than
3.8 ms; the fixed-cost work is in the KERNELS' footprint, and the per-row
work is separately 2.1x CatBoost CPU (103 vs 48.6 us per 1000 rows).

The same day's per-kernel itemization, at 800k x 100 depth 6 (ms/tree):
histogram accumulate ~68, zero ~8, fixed-to-float ~7, bridge ~2, reorder
~12, score ~11, partition stats ~5.4, split+sequence ~3.5, the rest under
2. Three findings out of it, all against their source:

0. **`ctx.get_attribute` costs 1.26 ms PER CALL on Metal** (measured: 100
   calls, 126 ms). `compute_partition_stats` and `launch_reorder_in_leaves`
   each queried MULTIPROCESSOR_COUNT once per level, ~16 ms/tree combined
   on covtype. Their `TArchProps::SMCount()` is a CACHED static read once
   at init, so threading one queried value through is THEIR behavior, not
   a deviation. After caching: covtype partition stats 10.5 -> 2.0 ms,
   reorder 12.3 -> 4.4 ms per tree. The reorder-is-at-the-machine-ceiling
   conclusion drawn earlier the same day was WRONG for this reason: the
   gather probe priced the kernels right (~4 ms), and the other 8 ms was
   this host call hiding inside the phase.
1. **The score kernel ran at grid (1,1,1).** Theirs is
   `argmaxBlockCount = min(ceil(binFeatureCount/256), 64)` blocks with one
   `TBestSplitProperties` per block and a host reduce
   (`greedy_search_helper.cpp:439`, `:513-529`). Fixed; 11 ms became 1.9,
   oracle still 48 of 48.
2. **Every gather-arm histogram launch divides `2 * maxActiveBlocks`**
   (`hist_one_byte.cu:356`); this port used the direct-load number for
   both arms. Fixed (no effect at this shape, where the other grid axes
   already fill the machine).
3. **The `hist_2` fused-two-stat family is PORTED AND IS THE DISPATCH'S
   PATH for every `maxBins <= 128`** (`hist_one_byte.cu:314-328`:
   `HIST2_PASS(5/6/7)`, the PASS family only at 129-255 bins), matching
   their ladder including at CatBoost's GPU default border_count of 128.
   `hist_2_one_byte_base.mojo` + the `_5bit/_6bit/_7bit` files mirror
   `hist_2_one_byte_base.cuh` + `hist_2_one_byte_{5,6,7}bit.cu`;
   `launch_hist2_one_byte` in `greedy_search_helper.mojo` is their
   `HIST2_PASS`, odd-numStats one-stat prelude included. It fuses both
   stats in one pass (half the bin reads, half the pass serialization).
   `pixi run check-hist2` proves exact cell-for-cell agreement with the
   PASS family and the host tally at all three bit widths, direct and
   gather, and FINGERPRINTS which family the dispatch launched (a planted
   bin `(1 << bits) + 1` that hist_2 folds to fold 1 and PASS drops).

A streaming probe on plain device buffers measured 90 GB/s at 20 blocks
(M4 base, 10 GPU cores), so the substrate is not the problem; the
histogram accumulate runs at ~8 GB/s and ~256 threads per core because its
32-floats-per-thread shared scratch fills Apple's 32KB threadgroup budget
at 256 threads (CatBoost runs the same kernel at 384 threads in NVIDIA's
48KB).

Two separate tracks, and they are not the same work:

1. **To make the comparison honest**: port the boosting loop, gradient
   computation and model update. This makes our number WORSE, not better,
   and it is required before quoting 1.66x as a like-for-like result.
2. **To make it faster**: cut the per-level host round trips and launches.
   Pack the five split-descriptor copies into one. Keep the split resolution
   on the device so the argmax never returns to the host.

### Where the numbers stand, end of 2026-08-19 (superseded by the block below)

All same-window, interleaved where two arms are named. Synthetic 800k x 100
depth 6 at 254 borders: 105 ms/tree in the morning, **91 after the evening
round** (score dispatch, memset, FAST-skip, sm_count caching); at
CatBoost's GPU-DEFAULT 128 borders the same tree is **~50 ms through the
newly ported hist_2 family** (interleaved A/B: 0.52-0.57x of the 254 arm).
**covtype 581k x 53, interleaved against CatBoost CPU on the same M4: ours
37.5 ms/tree vs theirs 10.5 at 254 borders, 36.6 vs 10.1 at 128 — 3.5x
behind, down from 4.9x at the morning baseline.** Phase split on covtype
after the fixes: hist ~10.5, reorder ~4.4, splitseq ~4.5, pstats ~2.0,
score 1.6, everything else under 2. The box's GPU window drifted ~4.5x for
about an hour mid-evening (another session's GPU benchmark); every
cross-window comparison taken in that hole was discarded.

### And then the boosting loop gave back 17 ms/tree (later the same night)

`fit` was spending ~17 ms/tree OUTSIDE tree growth, none of it CatBoost's
design: a 6.4 MB gradient readback plus an 800k-row host abs-scan feeding
only the fixed-point scale (comptime-dead under FAST now), a 3.2 MB host
identity upload per tree (their `MakeSequence` builds it on the device,
`fill.cu:47` -> `gbdt/gpu_util/kernel/fill.mojo`), and an 800k-row host
SSE loop (their `functionValue` is a block reduce + one float atomicAdd
inside the SAME gradient kernel, `pointwise_targets.cu:309-317`, now
ported; the reported loss moved in its 7th decimal, the model did not).

**Standings after, same-window, interleaved, same data both arms:**

    800k x 100 @ 128 borders (their GPU default): ours ~48 vs CPU ~32 -> 1.4x
    800k x 100 @ 254 borders:                     ours ~68 vs CPU ~31 -> 2.2x
    covtype 581k x 53 @ 128:                      ours ~24 vs CPU ~10 -> 2.4x
    covtype 581k x 53 @ 254:                      ours ~29 vs CPU ~10 -> 2.8x

The morning started at 21x fixed / 2.1x variable and covtype 4.9x.

### PARITY, end of 2026-08-19, and how to reproduce it

After `482661e` (the Apple 2-warp-shared Int32 histogram arm), interleaved
in one process, same rows and grid both arms, mse printed and matched to
~7 decimals every rep:

    800k x 100 @ 128 borders (their GPU default):
        CatBoost CPU 31.0-31.4 ms/tree, ours 32.0-32.9 -> 1.02-1.05x
    800k x 100 @ 254:  ~31.4 vs ~65.7 -> 2.1x  (254 family not yet shared-i32)
    covtype   @ 128:  ~9.8  vs ~21.3 -> 2.2x
    covtype   @ 254:  ~9.8  vs ~26.6 -> 2.7x

Reproduce:
    pixi run -e bench python tools/interleaved_prep.py <dir> synth
    pixi run -e bench mojo run -I . \
        bench/interleaved/catboost_interleaved.mojo <dir> synth 800000

The Apple arm is deterministic run-to-run (associative integer
accumulation), so this table is deterministic-vs-deterministic; CatBoost's
CPU was measured bitwise stable across thread counts the same night, and
its GPU (which cannot run here) is non_deterministic by their own tag.

The day opened at 21x fixed / 2.1x variable and covtype 4.9x.

### 2026-08-20: the 129-255-bin family, and the resolution trap

`d72bd97`: the per-tree histogram zero-fill became `enqueue_memset`
(their `FillBuffer` is a device-side fill; ours staged 1.6-3.2M host
stores per tree). The "per-tree workspace reuse" lever was REPRICED by
probe and is OFF the queue: the whole ~40-buffer create/retire prologue
measures 0.17 ms/tree -- MAX's allocator pools -- and the host zero
loops (0.6+ ms, growing with border count) were the real cost hiding
under that estimate.

`992aa86`: the PASS family (`TPointHistOneByte`, the 129-255-bin range
their ladder never sends to hist_2) got the same 2-warp-shared Int32
arm as hist_2 -- one `hist_smem_mode_for` row for the whole one-byte
family. It tripped a trap the oracles cannot see: with the blanket
2^28 scale, dither noise flipped near-tied splits at 254 borders on
synth (train mse 0.14375 vs CatBoost's 0.14145; covtype unaffected;
all oracles green -- 4096 rows never builds the tie density). The
harness's per-rep mse column caught it, the second live catch in two
days. Fix: `choose_scale(mag, row_count)` spends the blanket headroom
exactly (`mag*scale + row_count <= 2^30 - 1`), 4x finer, and synth@254
returned to CatBoost's mse at 7 decimals. New coverage that gates the
range: `bench/oracle254.txt` (48/48) and `check-hist2`'s bits-8
both-modes section.

Quiet-window standings after both commits (interleaved, one process,
CPU-arm canary at its usual 31 / 9.7-10.7 ms/tree, mse printed both
arms every rep):

    800k x 100 @ 128 (their GPU default):  0.97-1.05x -- two of three
        reps FASTER than CatBoost, mse matched to 7 decimals
    800k x 100 @ 254:  1.54-1.59x  (was 2.1x), mse 7 decimals
    covtype   @ 128:  1.88-2.34x, mse 6-7 decimals
    covtype   @ 254:  2.29-2.61x  (was 2.7x), mse 6 decimals

WHERE COVTYPE'S GAP LIVES, derived from measured constants rather than
assumed: scaling synth@128's 31 ms/tree by covtype's rows (x0.73) and
features (x0.53) predicts ~12 ms of variable cost, yet we run ~21 --
about 8 ms/tree does not scale with data. That matches the fixed
control plane at THIS BOX's measured prices (launch 21-23 us, drain
191 us, ~30-40 launches per level x 6 levels + 12 drains): CatBoost's
CPU arm pays no launch floor at all, which is why ITS covtype cost
scales down to 9.7 and ours cannot follow. The lever list below
attacks exactly this; the decisive per-phase attribution is a
level_bench run on the covtype shape.

### 2026-08-20, later: THE SCALE WIN -- first decisive victories

The covtype attribution said the gap was a FIXED ~9-11 ms/tree floor
(scratchpad `fixedfloor_probe.mojo`: 581k/290k/145k rows at the covtype
shape -> 19.7 ms/tree with a 9.3-11.5 ms intercept; the floor alone
costs more than CatBoost CPU's whole covtype tree). Floors amortize, so
the prediction was: bigger data, we win. MEASURED at 4M x 100
(interleaved, same process, mse matched both arms every rep, 16 GB box
at 78% free after the run, no thrash signature in the rep variance):

    4M x 100 @ 128 (their GPU default):  0.47-0.56x
        -- OURS IS 1.8-2.1x FASTER THAN CATBOOST CPU
    4M x 100 @ 254:                      0.82-0.88x -- ours faster

    (ours ~100 ms/tree at 5x the rows of the 800k parity run's ~31 --
    sublinear, the floor amortizing; CatBoost CPU went 31 -> 184-210,
    slightly superlinear.)

Fixtures: tools/interleaved_prep.py <dir> synth 4000000 100. The
resolution-aware `choose_scale(mag, n_rows)` holds at 4M rows: mse
matches CatBoost to 6-7 decimals at both border counts.

THE STANDINGS STORY IN ONE LINE: we now BEAT CatBoost's CPU wherever
the data is big enough to fill the GPU (>= ~1M rows at their GPU
default config), tie at 800k, and lose only where the per-tree
launch/drain floor dominates (covtype-sized data) -- and their GPU arm
cannot run on this machine at all.

### 2026-08-20, latest: device-side split resolution landed (3a5280d)

A level now blocks the host ONCE where theirs blocks twice (marked
deviation, `kernel/split_resolve.mojo`): the winner is reduced on the
device in the host loop's exact order, descriptors pack from a
once-per-tree bin-feature table, gates run post-drain with a one-level
rollback that `mojo_only/early_stop_check.mojo` forces on purpose (no
oracle fixture ever stops early, so the rollback needed its own reach
check; building it also established that a constant-bin tree does NOT
root-stop -- the full/empty split scores the parent's own score and
stops one level later via the repeat rule).

Quiet-window standings after the fold (interleaved, mse matched every
rep):

    800k x 100 @ 128:  0.89-0.98x -- ALL THREE REPS FASTER (was
        0.97-1.05x; the parity row became a clean win)
    800k x 100 @ 254:  1.41-1.53x   (was 1.54-1.59x)
    covtype   @ 128:  1.80-2.11x, best rep 17.9 ms/tree (was
        1.88-2.34x / 20.2) -- the predicted ~2 ms of drain fold, banked
    covtype   @ 254:  1.97-2.33x   (was 2.29-2.61x)
    4M x 100  @ 128:  0.47-0.51x -- 2.0-2.1x faster than CatBoost
    4M x 100  @ 254:  0.76-0.88x

THE DEVICE-RESIDENT LEVEL PLAN IS PRICED AND DECLINED (2026-08-20
night). A working-tree probe removed ALL six remaining per-level drains
(approximate balanced sizes fed to the host planner, one drain per
tree): the covtype-shape tree moved 19.7 -> 19.4 ms. The probe's fixed
floor dropped 9.3-11.5 -> 6.7-7.0 ms, but its crude size approximation
inflated the variable cost (wrong-sibling subtraction builds the larger
child half the time), so the honest ceiling for a REAL device-resident
plan -- exact sizes, no wrong siblings -- is ~1-2 ms/tree on covtype.
The residual ~6.7 ms floor is LAUNCH + enqueue + copy cost, not drains,
so the next floor attack is launch-count fusion, not the plan port.
Declined at that price; the lever stays on the list behind fusion.

### 2026-08-20/21: THE INFERENCE PATH (74fdd44)

Their model_evaluation_speed methodology mirrored
(bench/interleaved/predict_interleaved.mojo): raw pool outside the timed
region, quantization inside predict, CPU arms at full threads and one
thread. The arc, all in one bite:

    tree-wise apply (their training-side kernel, our drains/allocs
        deleted): 2.9x BEHIND their CPU evaluator -- 100 trees re-stream
        the cindex from DRAM per tree.
    their OWN GPU evaluator ported (gbdt/models/cuda/evaluator.mojo,
        from libs/model/cuda/evaluator.cu): eval kernel 2.7 ms for
        100 trees x 800k -- the cost was ALL in their linear-scan
        quantize (27-30 ms, the base M4's ALU floor; V100-class cards
        absorb it). A binary search measured 3.5x WORSE (dependent
        divergent loads vs pipelined compare-adds). The shipped Apple
        arm is a TWO-LEVEL scan; all three arms and their prices live
        in kernel_matrix.quantize_search_for.

STANDINGS (interleaved, 100 trees depth 6 @ 128 borders, evaluator arm
asserted equal to the tree-wise apply and to the fit's train mse every
rep):

    800k x 100: parity band -- ratios 0.89-1.55 across windows (median
        ~1.2, best window 0.76-0.90), theirs ~18 ms / ~44 M docs/s
    4M x 100:  ALL EIGHT REPS FASTER, 0.75-0.88x -- ours 52-62 M
        docs/s vs their 46-50, stable

Also landed on the way: `binarize_float_feature_kernel` (their
BinarizeFloatFeatureImpl -- the missing raw-float device quantization),
the batched ensemble pack in `predict`, and the adaptive
ExtTreeBlockWidth (their 128 idles 7/8 tree sub-blocks below ~1000
trees; runtime width, their shape at their scale).

COVTYPE INFERENCE (2026-08-21): ALL EIGHT REPS FASTER, 0.31-0.84x
(median ~0.44) -- ours 50-74 M docs/s vs their 23-26. On the real
dataset where TRAINING trails 2x behind their CPU, inference wins
outright: a model application is three launches, so the launch floor
that decides small-data training does not exist at predict time.

LAUNCH FUSION IS PRICED AND DECLINED (2026-08-21). Counted, the
per-level chain is ~20 launches and every one of them is CatBoost's own
kernel granularity except `fixed_to_float` (ours, 1/level ~ 0.15
ms/tree if fused into the bridge). The covtype training floor is
METAL'S LAUNCH PRICE (21-23 us vs CUDA's ~5) on THEIR launch count --
a platform constant, not fusable waste. The honest paper sentence:
small-data GPU training on Apple pays ~4x CUDA's control-plane tax,
the training crossover sits near ~1M rows, and inference clears it
everywhere because it launches three kernels, not two hundred.

THE 8000-TREE ROW (2026-08-21), their notebook's exact model size:

    800k x 100, 8000 trees depth 6 @ 128 borders, 8 reps interleaved:
    CatBoost CPU full-threads 1.84-2.34 s/predict (1-thread 5.4-5.8 s),
    ours 177-196 ms = 0.082-0.100x -- TEN TO TWELVE TIMES FASTER, every
    rep. Their predict scales linearly with trees; ours amortizes the
    fixed quantize and the eval kernel's tree-parallelism engages fully
    at the scale their own design targets.

    (The first run died on the harness's own cross-arm guard: a fixed
    1e-4 bound calibrated at 100 trees, tripped by 1.32e-4 of sqrt(N)
    float reorder -- exactly the 9.5e-6 x sqrt(80) the 100-tree
    measurement predicts. The bound now scales with the accumulation
    length; both measured points live in the comment.)

THE INFERENCE TABLE, complete (all interleaved, per-rep accuracy
asserted): 100 trees -- covtype 0.31-0.84x, 4M 0.75-0.88x, 800k parity
band 0.89-1.55; 8000 trees -- 800k 0.082-0.100x. Inference beats their
CPU evaluator everywhere except the smallest config, where it ties.

### 2026-08-21: the feature-debt ledger, re-scored from their source

* BAYESIAN BOOTSTRAP: PORTED (b4152f3, their GPU default; MVS is
  Y_ASSERTed away in their GPU oblivious searcher). Stochastic gate set
  in mojo_only/bootstrap_check.mojo; harness 'bayesian' mode shows the
  parity band holds with sampling on and the mse bands overlap.
* RANDOM_STRENGTH: CLOSED WITH NO CODE. On their GPU symmetric path the
  noise CANCELS BY CONSTRUCTION: `NextFeature` sets FeatureId, then
  `beforeSplitCalcer = calcer` copies it, both calcers draw the
  IDENTICAL normal (same GlobalSeed + FeatureId, score_calcers.cuh:
  160-168), and the argmax compares `gain = score - scoreBefore`
  (compute_scores.cu:131-142) where the draw subtracts out exactly.
  Their default 1.0 changes nothing on their own GPU; we mirror by not
  porting it.
* RSM: CLOSED WITH NO CODE. `Rsm` appears in all of catboost/cuda ONLY
  under pairwise_oblivious_trees; the plain GPU oblivious learner has
  no feature sampling at all. Our rsm=1.0 pin IS their GPU behavior.
* BOOST_FROM_AVERAGE / MODEL_SHRINK / LEAF_ESTIMATION: already at their
  GPU-path values (see fit's cursor note; their options check refuses
  boost_from_average on this path).

So the pinned-settings comparison tables were never a handicap match:
with Bayesian now ported, the ONLY feature CatBoost's GPU learner has
that this port lacks is CATEGORICALS/CTRs -- the one genuinely large
remaining block -- plus a holdout-mse quality gate for the stochastic
modes.

THE HOLDOUT GATE LANDED (bench/interleaved/holdout_bayesian.mojo):
both arms train on 80% at Bayesian T=1 on the same grid, fresh seed per
rep per arm, scored on the untouched 20%. Ours 0.0595-0.0610 test mse
vs theirs 0.0605-0.0625 -- at or slightly below theirs every rep. The
stochastic mode's quality claim rests on holdout now, not train mse.

### 2026-08-21, late (e726e11): one-hot, TRUE bit-determinism, the snap

CTR RECON STEP 1 LANDED: one-hot categorical splits end to end
(build_layout flags -> scan skip -> equality split -> predict's
takeEqual), gated analytically -- a 3-category y=(cat==1) fixture that
ONLY equality separates at depth 1 (6.4e-6 vs 0.167).

AND THE DETERMINISM STORY CLOSED: a flaky bootstrap gate exposed that
functionValue and the fixed-scale magnitudes ended in FLOAT ATOMICS
(the historical last-bit loss jitter, and same-seed fits could differ).
Fixed-order per-block folds replaced them; the corrected magnitude
(exact vs float64) shifted the scale 1.3e-6 and re-rolled the dither,
which moved a model 2.7% -- so choose_scale now SNAPS DOWN TO A POWER
OF TWO, making the scale a step function immune to last-bit magnitude
noise. Result: every rep's loss is BIT-IDENTICAL and the models match
CatBoost closer than ever (synth@128 9 sig figs, @254 8, covtype 7-8;
the halved resolution margin at 254 held).

CATEGORICAL ROUND 1, RUN AND REPORTED (2026-08-21): host-side
FeatureFreq CTRs (their exact GPU formula count/(n+1); tools/ctr_prep.py +
bench/interleaved/ctr_quality.mojo)
on AMAZON, 26215/6554 holdout, 100 trees depth 6, both arms restricted
to frequency information: ours 0.05078 test mse vs CatBoost 0.05064 --
0.3%% apart. Attribution: the CPU arm uses `Counter` because CatBoost's
OWN CPU refuses FeatureFreq ("not implemented on CPU yet",
catboost_options.cpp:509 -- more evidence the GPU learner is its own
system); same information class, slightly different normalization, so
this is a quality-band row, not a bitwise one. ADULT: prep bug diagnosed
(float NaN inside object columns; fillna before astype), handed to the
harness stream -- to be RUN AND REPORTED once fixed, whatever it shows.

(That run's ctr binarization was recorded here as "their Uniform-15",
which was wrong and the phrase is deleted rather than annotated. The run
used MinEntropy 15, which IS their FeatureFreq grid; `Uniform 15` is the
grid a **Borders** column gets. Both are real defaults on real paths and
they are different -- see RECON_CTRS.md and PORTING.md 51.)

### 2026-08-20: CTRs, the block itself (this session)

`gbdt/ctrs/` exists, mirroring `catboost/cuda/ctrs/`: the CTR option
surface with `GetDefaultPriors`' THREE-prior fan-out (a Borders cat
feature is THREE columns, not one, plus a FeatureFreq column -- four per
categorical feature under their GPU defaults), target binarization at
MinEntropy-1, `TWeightedBinFreqCalcer`, `THistoryBasedCtrCalcer`, and all
ten elementwise kernels of `ctrs/kernel/ctr_calcers.cu` enqueued and gated
cell by cell. `train(cat_features=...)` now has a categorical path, so the
FeatureFreq half ships in the LIBRARY and not only in the benchmark prep.

Three things worth carrying forward:

* **The kernel-matrix prediction held.** Not one warp intrinsic, not one
  byte of shared memory, not one atomic in the ten CTR kernels.
* **Borders was not wired into `train()`**, and the reason was a seam
  nobody had listed: it needs the CTR ESTIMATION PERMUTATION as well as the
  device sort and scan. Closed 2026-08-21, below.
* **A shipped accuracy bug fell out of the cardinality sweep**: `train()`
  and `_build_cindex_from_floats` derived DIFFERENT fold counts for a
  one-hot feature, so the compressed index was written and read under
  different packing policies. A 16-category one-hot feature was silently
  unlearnable, and the fit/predict consistency assertion could not see it
  because both sides read the same wrong layout. Fixed;
  `mojo_only/one_hot_cardinality_check.mojo` sweeps every policy boundary.

### 2026-08-21, the 254 lever closed out (9b2b64e + 8010b2f)

The Int32 arms dropped the collision serialization their atomics make
redundant (pass loops, write turns): 800k@254 1.25-1.36x (from
1.4-1.5x), covtype@254 1.87-2.02x, every mse column bit-unchanged. Then
the fused two-stat 8-bit arm (the >128 fusion their ladder cannot do
warp-private) landed BIT-IDENTICAL with half the launches -- and FLAT
speed, because the walk it removes was latency-hidden on this box (the
traffic-model lesson, again); kept for the launch count and for
bandwidth-tighter machines. The remaining 254-over-128 delta is 2x
histogram cells through bridge/score/convert -- structural, priced.
ON-BOX TRAINING LEVERS ARE NOW EXHAUSTED: what remains is Pro/Max
silicon, NVIDIA validation, and the CTR device kernels.

### 2026-08-20: A CATEGORICAL MODEL NOW SCORES RAW DATA

Three things blocked it and all three landed, gated by
`pixi run check-ctr-apply`:

1. **The CTR tables are in the file.** `gbdt/models/ctr_value_table.mojo`
   is their `TCtrValueTable` plus the `TModelCtr` fields that turn its
   counts into a value, written as `ctr_table` / `ctr_entry` records --
   their `ctr_data.hash_map`, which for FeatureFreq stores ONE INTEGER per
   category, not a value. The value is formed at apply time by their
   `Calc` (`online_ctr.h:289-292`), and an unseen category takes their
   `emptyVal = Calc(0, denominator)` rather than a neighbour. Because
   FeatureFreq is permutation-independent, the apply-time table reproduces
   the LEARN column bit for bit, and that identity is asserted per row on
   4096 learn rows rather than assumed.
2. **`TBinarySplit` carries `EBinSplitType`**, set by their `ToSplit` rule
   (`cuda/methods/helpers.cpp:164-170`). The apply now reads the predicate
   off the MODEL and cross-checks it against the layout, which is their
   `CB_ENSURE(dataSet.IsOneHot(...))`.
3. **The device evaluator has the one-hot arm its `XorMask` slot
   reserved.** Their GPU evaluator refuses categorical models outright, so
   the predicate came from their CPU evaluator instead of being invented
   (PORTING.md 55). Reach proved by flipping the predicate in a loaded
   model: 405 of 512 rows move.

`predict_floats` no longer refuses a CTR model; what it refuses now is a
model whose tables are MISSING, which is the safety property
`ctr_column_count` was carried for. A FLOAT-ONLY model's file is
BYTE-IDENTICAL before and after (8022 bytes, same SHA-256), which is what
made the trailing-token format choices worth making.

The cardinality sweep at 1, 2, 3, 15, 16, 17, 31, 32, 254 and 255 runs on
both the one-hot and the CTR feature and compares host and device against
an independent host tally PER ROW. It also caught the check being wrong
rather than the code: at k = 2 their `one_hot_max_size` dispatch sends a
`cat_features` column to ONE-HOT and gives it no CTR at all
(`binarizations_manager.cpp:106-109`), so the sweep prints which arm each
row took.

Next known levers, in order: CTR steps 3 and 6 (RECON_CTRS.md: the
device radix-sort/segmented-scan swap, then the hash that tree CTRs
force), the CTR estimation permutation in front of `Borders`, the
epsilon dataset.
Beyond this box: a Pro/Max chip multiplies OUR arms by 2-4x and theirs
by ~1.5x.

### 2026-08-21: BORDERS, END TO END -- the ordered target statistic runs

The last parity item in the CTR block. `train(cat_features=...,
cat_feature_params=[TCatFeatureParams.default()])` now produces FOUR columns
per categorical feature -- three `Borders` priors and one `FeatureFreq`,
CatBoost's own GPU `simple_ctr` default -- and trains. Three things landed
together and the middle one was the actual blocker.

**THE CTR ESTIMATION PERMUTATION** (`gbdt/data/permutation.mojo`): their
`TDataPermutation` over `Shuffle`, `TRandom` and MT19937-64, ported bit for
bit. There is no live oracle for it -- their GPU learner will not run on
Apple and their CPU learner never exposes an order -- so
`tools/permutation_oracle/` compiles CatBoost's OWN
`util/random/mersenne64.{h,cpp}` under a C++ transcription of the four short
functions around it, and `pixi run check-permutation` compares the raw
64-bit stream, `Uniform(t)` and whole orders cell by cell. Green on the
first run.

**THE DEVICE PRIMITIVES, WIRED.** `TCtrBinBuilderGpu` and
`THistoryBasedCtrCalcerGpu` run `ReorderBins`, both calls to
`SegmentedScanAndScatterNonNegativeVector`, `GatherWithMask`,
`ScatterWithMask` and `ScanVector<ui32>` on the device. The last three had
to be ported on the way (`gpu_util/kernel/transform.mojo` and
`gpu_util/kernel/scan.mojo`): they are `cuda_util` functions the bin builder
needs and nothing had written them. `PORTING.md` 52 shrinks to the
FeatureFreq calcer alone. `launch_segmented_scan_vector` stays unwired --
its only call site in all of `catboost/cuda` is `VisitFloatFeatureMeanCtrs`,
and `FloatTargetMeanValue` has no calcer here.

**WHY ROW ORDER IS NOT AN OPTION, MEASURED RATHER THAN ASSERTED.** Their
permutation 0 IS the identity, and that is safe on their side only because
`ShuffleLearnDataIfNeeded` shuffles the whole learn pool at load whenever
the data has a categorical feature (`private/libs/algo/preprocess.cpp:183`).
This port has no such stage. On a 4001-row target-sorted fixture whose
category carries no information at all:

    leak = mean(ctr | target bin 1) - mean(ctr | target bin 0)
        row order (their id 0):        0.303
        the shipped permutation (3):  -0.0027

    train loss, categorical column as the ONLY feature, 30 trees depth 4:
        under the permutation:  0.1234
        in row order:           0.0606

Row order fits a feature containing nothing twice as well, because the
statistic reads the label it is estimating. `PORTING.md` 55.

ONE THING STILL OWED: this port builds ONE CTR column set where their loop
builds `permutation_count` of them (deviation 55a), which is the
ordered-boosting loop and is separate work.

WHAT A BORDERS APPLY-TIME TABLE ACTUALLY HOLDS is now written down in
`RECON_CTRS.md`, read while it was cheap, and the headline is that it is
NOT ordered: `TCtrValueTable` carries a per-category histogram over target
classes computed on the whole learn set, and the permutation is a
training-time device that never reaches the model file.

### 2026-08-20: THE Borders APPLY-TIME TABLE, AND THE DEFAULT FLIPPED

The last gap in the categorical block. A `Borders` model TRAINED and could
not SCORE, because `build_ctr_tables` (then
`build_feature_freq_tables`) had no arm for it. It has one now, and the
claim `RECON_CTRS.md` recorded held on contact: **the apply-time table is
NOT ordered.** `TCtrValueTable` gained their `TargetClassesCount` axis and
`TModelCtr::TargetBorderIdx`; the blob is their
`int[uniqueCategories * TargetClassesCount]` filled by
`++elem[targetClass[z]]` (`private/libs/algo/online_ctr.cpp:927-930`) over
the whole learn set, with no permutation, scan or sort anywhere in it. The
histogram is computed ONCE per categorical feature and shared by the three
Borders priors, which are three `TModelCtr` over one table in their model
too.

`value_for` and `empty_value` now dispatch on `ctr_type` the way
`TStaticCtrProvider::CalcCtrs` does, and the branch that matters is the
UNSEEN category: `Borders` takes `Calc(0, 0)` -- the prior alone -- where
`FeatureFreq` takes `Calc(0, denominator)`. At the {1, 1} prior of their
own fan-out those are 1.0 and 1/(n+1), which on a 4096-row pool is 4097x
apart, and a held-out set is full of exactly those rows.

**AND `train()`'S IMPLICIT FALLBACK FLIPPED TO
`TCatFeatureParams.default()`**, CatBoost's own GPU `simple_ctr`: a
default categorical fit now emits FOUR columns per feature (three Borders
priors plus FeatureFreq) where it emitted one. The switch was opt-in for
exactly one reason -- Borders could not score -- and a switch that
outlives its reason is a defect.

One thing that follows and is worth stating because it will surprise
someone: a `Borders` model's applied predictions do NOT reproduce its own
fit's loss on the learn rows. The column it TRAINED on is the ordered
statistic over the estimation permutation; the column an applied model
carries is the full-learn-set histogram. That gap is CatBoost's design,
not a defect here, and it is why `FeatureFreq`'s bit-identity gate could
not simply be pointed at Borders.

Next known levers, in order: tree CTRs (RECON_CTRS.md step 6, which forces
the category hash), then the epsilon dataset.

### 2026-08-21: the category-hash stack, and step 6 re-scoped from source

The recon read for tree CTRs (RECON_CTRS.md step 6) falsified this file's
own next-lever line: "forces the category hash" was WRONG for training.
`TFeatureTensorTracker::AddFeatureTensor` builds a combination's bins by
feeding dense compressed bins into `TCtrBinBuilder::AddCompressedBins`
(`gpu_data/oblivious_tree_bin_builder.cpp:123-186`) -- the primitive
`TCtrBinBuilderGpu` already ports -- so NO hash exists anywhere on their
training path and our dense codes stay harmless through tree-CTR training.
The hash is the MODEL/APPLY key: `CalcHash(acc, element)` folded over
sign-extended category hashes and bare 0/1 split arms
(`libs/model/ctr_provider.h:94-122`).

That stack is now PORTED AND GATED (`pixi run check-cityhash`):
`gbdt/digest/city.mojo` (their CityHash **1.0**, whose results their own
header says DIFFER from mainline CityHash -- public vectors are the wrong
oracle), `gbdt/cat_feature/cat_feature.mojo` (`CalcCatFeatureHash`, the
low 32 bits), `gbdt/models/hash.mojo` (`CalcHash` + the `(ui64)(int)`
widening). The oracle is their own `city.cpp` compiled byte-for-byte by
`tools/cityhash_oracle/` into `bench/cityhash_oracle.txt`: 43 strings
covering every length branch, 5 combination chains. Unwired by design
(UNWIRED.md): training needs no hash; it goes live with tree-CTR model
tables or raw-string scoring.

THE CHECK EARNED ITS KEEP ON FIRST RUN, and the defect joins the Mojo
numeric-trap family: a SIMD cast chain `uint32 -> int32 -> int64 ->
uint64` ZERO-extends where C++ `(ui64)(int)` sign-extends. All 43 string
rows passed while every chain row containing a hash >= 2^31 failed in
exactly the high 32 bits; the fix is an explicit branch
(`cat_hash_chain_element`), not a cast chain. Assume stdlib numerics are
approximate until measured -- now including casts.

What remains of step 6 is the machinery, not the primitives: the
`TTreeCtrDataSetsHelper` / dataset / visitor system (`methods/
tree_ctrs.cpp` 534 lines, `tree_ctrs_dataset.h` 210,
`tree_ctr_datasets_visitor.cpp` 168, the memory estimator 156) --
per-split tensor extension, per-device dataset packs, lazy compressed
indices, score-calcer reuse, and the visitor's cross-device best-split
race. Every device primitive it stands on is already ported.

### 2026-08-21: ADULT categorical quality row -- run and reported

The row owed since the prep-bug diagnosis, run whatever it showed. The bug
was real float NaN among adult's object-column strings (workclass 1836,
occupation 1843, native-country 583 of 32561 rows): `.astype(str)` alone
does not homogenize them, and `catboost.Pool` REFUSES a NaN cat cell
outright, so the `fillna("nan")` fix landed on BOTH arms --
`tools/ctr_prep.py` at the np.unique site and
`catboost_arm.fit_and_test_mse_cat` before the Pool.

26048/6513 holdout, 100 trees depth 6, frequency-information-only both
arms (the amazon design; 8 freq-ctr + 6 numeric columns, cards 2-42):

    ours 0.0892802 test mse  vs  CatBoost(Counter) 0.0908374 -- all
    three reps bit-identical per arm, ours 1.7% BELOW theirs

Attribution, same as amazon's row: same information CLASS, not same bits
(their CPU has no FeatureFreq; Counter normalizes the same counts
differently). One arm asymmetry worth naming: adult carries a
cardinality-2 feature, which our fixture freq-encodes while their arm
one-hots it at `one_hot_max_size=2` -- a quality-band row, not a parity
row. The GPU box was shared with a concurrent session during the run;
quality rows are immune to the drift that would disqualify a timing row.
One operational note: rep 2 stalled ~30 minutes inside
`catboost.datasets`' per-call re-download (sock_connect) before
recovering, so `catboost_arm` now downloads each dataset once per
process.

### 2026-08-21: EPSILON -- the widest margin yet, on their own dataset

The pre-registered next lever ran (fixtures from CatBoost's own
md5-verified tarball, `tools/epsilon_prep.py`; full record
`bench/results/EPSILON_2026-08-21_interleaved.md`). 400k x 2000 dense,
20 trees depth 6, interleaved, 3 reps per border:

    254 borders: speedups 2.67 / 3.42 / 3.32x  (mse 8 sig figs)
    128 borders: speedups 3.41 / 3.67 / 3.30x  (mse 8 sig figs)

Ranges fully disjoint (ours 186-272 ms/tree, theirs 504-903). At
800k x 100 the 254 arm still loses 1.25-1.36x; at 10x the
rows-x-features product the same code wins 3.3x at the same border
count -- feature width amortizes the fixed floor as effectively as row
count. The scale story is now three measured points: parity near
800k x 100, 1.8-2.1x at 4M x 100, 3.3-3.4x at 400k x 2000.

### 2026-08-21, later: the epsilon tree decomposed -- 83% accumulate, and the density cliff

`mojo_only/shape_sweep.mojo` over prefix-slices of the epsilon fixtures
(full record `bench/results/SHAPE_SWEEP_2026-08-21_epsilon.md`), three
findings, each measured:

1. **t = a + b*R + c*F + d*R*F at the full shape: floor 6%, rows 1%,
   feature-scaled cell passes 10%, ACCUMULATE 83%** (~26.5 GB/s effective
   against the 90 GB/s streaming ceiling).
2. **Sibling subtraction is alive and worth exactly 1.96x** at this shape
   (OFF: 240-252 ms/tree vs ON: 124-128), with BIT-IDENTICAL mse both
   arms -- on the Int32 accumulate the subtraction is exact, so the
   equality is the correctness proof. Their policy mirrored:
   build-smaller-and-subtract, `split_properties_helper.cpp:1283-1355`.
3. **The density cliff**: depth differencing (30.7 / 67.9 / 123.4 ms/tree
   at depths 2/4/6) gives per-level cost 11.5 -> 18.6 -> 27.8 ms while
   built rows per level stay ~constant -- cache-line utilization collapses
   as leaves fragment the indexed reads. Their `GatherBins` arm is a
   checked dead end at 2 stats (same amplified read plus a dense
   write+re-read; their own statCount > 2 threshold encodes exactly this),
   so the cliff is unaddressed in their design and merely tolerable on
   V100-class bandwidth. A density-triggered level compaction was
   priced at 15-30% and then REFUTED the same day by
   `mojo_only/density_probe.mojo`: indexed loads beat the gather arm at
   EVERY density (0.29-0.75x of its cost) and a dense 800MB/level copy
   exceeds depth-6 amplification losses, so the cliff is STRUCTURAL on
   this box and the on-box speed hunt at this shape closes with a
   measured answer at every door. It reopens only on other memory
   systems (re-run the depth differencing there first) or at much
   greater depths.

### 2026-08-21, evening: the quiet-window epsilon rerun, and deviation 52 closed

**EPSILON CONFIRMED QUIET**: with no concurrent session, theirs
339-378 ms/tree, ours 129-156 -- speedups 2.28-2.78x (medians 2.40x @254,
2.77x @128), mse identical to the loaded window. The loaded window's
3.3-3.4x was partly their arm's memory-pressure sensitivity; the standing
claim is now 2.3-3.7x across both windows, ranges disjoint in all 12
reps, quiet medians quotable (EPSILON_2026-08-21_interleaved.md).

**DEVIATION 52's PORT HALF IS CLOSED**: `TWeightedBinFreqCalcerGpu` runs
their `VisitEqualUpToPriorFreqCtrs` launch for launch on the device, over
two pieces ported on the way -- `UpdatePartitionOffsets`
(`gpu_util/kernel/partitions.mojo`, both dispatcher arms including the
`partCount == size` FillBuffer path) and the Sum arm of their cub
`SegmentedReduceVector` (`gpu_util/kernel/segmented_reduce.mojo`,
hand-written; vendor check recorded: MAX ships no segmented reduce).
`pixi run check-freq-ctr-device` gates it BIT-EQUAL against the host
driver and an independent tally (trivial weights make every float sum
exact, so equality is the contract, not a hope), across the
packing-policy cardinality boundaries, both priors, and a sabotage that
must land exactly on the affected categories -- green first run, 950
moved cells all in the right rows. WHAT REMAINS IS ONE LINE in
`gbdt/train.mojo` (the other lane's file): swap the independent-half
call to `compute_simple_ctrs_device(ctx, ...)`; handoff note in the
driver's docstring, UNWIRED.md carries the row until it lands.

### 2026-08-21, night: the new chassis benchmarked -- Logloss and Bayesian epsilon rows

Both from a CLEAN WORKTREE at HEAD (`cf750b0`), so the numbers benchmark
the commits while the feature lane's working files stay out of the
binary. Full record `bench/results/EPSILON_LOGLOSS_2026-08-21.md`.

**LOGLOSS (epsilon's native task; first benchmark of the loss chassis):**
Newton-10 estimation on BOTH arms (their Logloss GPU default -- pinning 1
would be the priced cheat): medians 2.30x @254 / 2.69x @128, losses
matched to 7 significant figures under 200 Newton steps of exposure, our
arm bit-identical rep to rep. The estimator bill at 400k rows is the few
ms/tree the 0.50 ms/iteration pricing predicted; their arm pays its own
Newton-10, so the ratio barely moves from the RMSE rows.

**BAYESIAN (their GPU-default sampling): 3.26-4.41x, the widest training
margin in the record** -- their CPU arm's per-tree weight resampling
costs it 1.5-2x, our sampling costs NOTHING MEASURABLE
(`bootstrap_delta_probe`: det 125.5-128.1 vs bayes 125.4-128.8 ms/tree
interleaved one process; the ~30-50 ms this section first attributed to
our bootstrap was a cross-window thermal artifact, deleted per rule 7),
and our mse BAND sits at or below theirs at both borders. The pinned
deterministic rows therefore UNDERSTATE epsilon: at the configuration
their GPU would actually run, the margin is 3.3-4.4x.

New files: `bench/interleaved/logloss_interleaved.mojo` +
`tools/catboost_logloss_arm.py` (separate from catboost_arm.py so the
NVIDIA lane's uncommitted work cannot collide). Not run on purpose: the
other nine new losses -- epsilon poses no quantile/Poisson task, and the
dataset for those gets chosen BEFORE any number exists.

### 2026-08-21, late night: depth 8 -- the margin WIDENS, the cliff loses to their depth bill

gbm-bench's pinned depth, run through
`bench/interleaved/rmse_depth_interleaved.mojo` (the committed harness
with DEPTH as argv; separate file only while the standard harness is
another lane's working file). Full record
`bench/results/EPSILON_DEPTH8_2026-08-21.md`:

    254: 2.78-3.68x    128: 3.20 / 3.64 / 3.57x    mse 8 sig figs

Depth 6 -> 8 multiplies ours ~2.2x and theirs ~3x: their CPU pays more
for two extra levels than our density cliff costs us, so the depth-6
medians (2.40/2.77x) WIDEN at depth 8. The compaction deferral's
"reopens at depth 8+" condition is hereby weakened comparatively --
depth 8 is a BETTER shape for us. Depth joins rows and features as an
axis that amortizes our floor and multiplies their bill.

### 2026-08-21, later still: MultiClass covtype -- near-parity on our worst dataset, and a dispatch defect caught

First benchmark of the MultiClass chassis, at covtype's NATIVE task
(every incumbent suite binarizes it). Full record
`bench/results/COVTYPE_MULTICLASS_2026-08-21.md`:

    254: 0.92-0.94x    128: 0.94-0.97x    loss 7 sig figs both borders

Covtype RMSE trails 1.8-2.1x; MultiClass's 7 stat planes multiply
per-level work ~3.5x on BOTH arms, amortizing our launch floor while
their CPU scales with the work -- the 2x deficit closes to 3-8%.
CLASSES are the fourth axis (after rows, features, depth) that amortizes
our floor and multiplies their bill.

AND THE RUN CAUGHT A REAL DEFECT: the first 254-border multi-stat fit
ever attempted hit the fused 8-bit arm's two-stat guard instead of a
histogram -- the fused arm (8010b2f) had replaced the >128 shared-Int32
route unconditionally, and MultiClass arrived later in another lane.
Fixed in the same commit: multi-stat shapes route to the PASS family's
z-axis stat pairs; reach proved by the raise-then-run, correctness by
the 7-figure loss column against CatBoost's own output.
