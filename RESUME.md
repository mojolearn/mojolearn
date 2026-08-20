# Resume here

Written 2026-08-19 at the end of a long session. Read this, then `UNWIRED.md`,
then `PORTED_MAP.tsv`.

## What this repository is

A port of **CatBoost's GPU oblivious (symmetric) tree learner**
into Mojo, targeting Metal first. Nothing from mojotrees. `ported/` mirrors
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

**1.66x behind, and the comparison FLATTERS US.** CatBoost's 30 ms is a whole
boosting iteration: gradients, tree, model update. Our 50 ms is tree growth
alone with the gradients handed to it. We have no boosting loop to charge.

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
`fill.cu:47` -> `ported/gpu_util/kernel/fill.mojo`), and an 800k-row host
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

Standings from the last quiet window (a peer bench loaded the box
before the final rerun; re-verify the CPU-arm canary before quoting):
synth@254 1.45-1.72x (from 2.1x), covtype@254 ~2.4x (from 2.7x),
128-border rows unchanged (parity at synth, 2.0-2.2x covtype).

Next known levers, in order: device-side split resolution (~2 ms, a
marked deviation through the matrix), the predict/inference path
(unmeasured; their model_evaluation_speed suite is the arena plan).
Beyond this box: a Pro/Max chip multiplies OUR arm by 2-4x and theirs
by ~1.5x.
