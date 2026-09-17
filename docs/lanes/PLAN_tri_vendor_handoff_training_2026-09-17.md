# Plan: tri-vendor handoff training for a GPT-3 Small shape model

Written 2026-09-17 at Andrew's direction, revised the same day after he asked
how two routes can possibly run concurrently when learning depends on the prior
checkpoint, and asked to keep the cloud Mac to one segment.

Nothing here has run. Every number not marked measured is derived, and says so.

## 1. The headline

**A GPT-3 Small shape model, trained by handing the run off between an Apple
Mac, an AMD box and an NVIDIA box, twice by different routes, ending at the
same bits.**

162,147,840 parameters. d_model 768, 12 layers, 12 heads of head_dim 64,
intermediate 2048, vocab 50257, sequence length 2048. Corpus FineWeb-Edu,
which is the data, not a milestone; the shape is the story.

## 2. Why two routes, and the constraint that makes it work

The claim we can make today (`docs/LM_TRAINING_CLAIM_PLAN.md`) is **sampled**:
at the checkpoints we replayed, three vendors agreed. A divergence in a window
nobody replayed is invisible.

Two routes removes the sampling. Split the run into segments and assign vendors
so that **every segment is run by a DIFFERENT vendor in each route**. If both
routes end at bitwise equal final weights, no vendor diverged at any step,
because a divergence at step k was produced on different hardware in each route
and the finals could not match.

**THE CONSTRAINT IS PER SEGMENT, NOT PER ROUTE.** Two routes that merely look
different are not enough. Andrew's assignment satisfies it:

    route A:  Apple -> AMD     -> NVIDIA
    route B:  AMD   -> NVIDIA  -> AMD
              ----     ----       ----
              differ   differ     differ

Every segment is covered by two distinct vendors, and Apple appears exactly
once in the whole experiment, which is also the cost goal. (The variant
`apple -> nvidia -> amd` against `amd -> nvidia -> amd` does NOT work: segment
2 is NVIDIA in both and segment 3 is AMD in both, so two thirds of the run gets
no cross-vendor coverage. The distinction is worth keeping in writing because
the two assignments look equally reasonable at a glance.)

## 3. More handoffs is better, and nearly free

A handoff is not overhead, it is evidence. Each one is an instance of the
cross-vendor resume claim: stop on vendor X, move the bytes, continue on vendor
Y, and the trajectory does not move. That claim already passed bidirectionally
at 34,944 parameters; every additional handoff at 162M is another instance of
it at a serious size, and more segments localize a divergence more tightly.

The cost is small. A checkpoint at this shape is parameters plus both Adam
moments, 3 x 162,147,840 x 4 bytes, about **1.95 GB**, which moves through R2
in well under a minute against segments measured in hours. Box spin-up is the
real per-segment cost, and the cloud Mac's minimum billing window is the one
that matters, which is another reason to keep Apple to a single segment.

Six to twelve segments looks right. Generalizing Andrew's assignment:

| segment | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **route A** | **Apple** | AMD | NVIDIA | AMD | NVIDIA | AMD |
| **route B** | AMD | NVIDIA | AMD | NVIDIA | AMD | NVIDIA |
| distinct? | yes | yes | yes | yes | yes | yes |

Segments need not be equal length. Make Apple's long enough to be a real
training window and no longer.

## 4. The two routes are independent, so they simply run at the same time

CORRECTED 2026-09-17. An earlier draft of this file proposed starting each of
route B's segments from route A's checkpoint so they could be parallelized, and
argued by induction that this proves what a chained route B would have. The
induction was valid and the whole construction was pointless, because **the two
routes never needed each other's checkpoints in the first place.**

Both routes start from the same initialization at t=0 and are independent
trajectories. Route A runs its segments sequentially on its boxes; route B runs
its segments sequentially on its boxes; neither waits for the other. Two boxes
busy throughout, 2x GPU-hours, and both routes are real artifacts that actually
existed rather than an argument that one would have.

    t:       0 ................................................ T
    route A   [Apple seg1][AMD seg2   ][NVIDIA seg3]
    route B   [AMD   seg1][NVIDIA seg2][AMD    seg3]
                        ^compare A1,B1  ^compare A2,B2  ^compare A3,B3

**Compare at every segment boundary as it happens.** Both routes reach
boundary k at the same STEP INDEX, so both have a checkpoint there and the two
must be bit equal. That gives divergence detection at the first bad boundary
rather than at the end, which was the only real benefit the discarded pipeline
scheme had.

**Wall clock is bounded by the SLOWER route, not by 1x.** Per-segment cost
differs by vendor, and Apple's per-step cost at this shape has never been
measured. Route A carries the Apple segment, so route A is likely the
bottleneck and the experiment finishes when it does. Do not quote a wall-clock
number until `lane/lm-training-shakedown` and a cloud Mac trial have measured
per-vendor step cost at this shape.

## 5. What blocks the Apple leg

**The local M4 cannot hold this shape.** Measured: 16.0 GiB unified memory
shared with macOS. The target shape's measured device peak on an H100 was
**12.14 GB**, with 21.93 GB host RSS on the eager path
(`docs/lanes/BRIEF_lm_step_memory_2026-09-10.md`, run 1). It does not fit, and
`docs/NEURAL_TRAINING_STATUS.md` has recorded this Mac at about 137 MiB free.

**And it must not run here anyway.** Apple is the one scarce column: one Mac,
one GPU, one job at a time, unrentable, a full pass measured over seven hours
during which every other GPU need serializes behind it.

**So the Apple segment goes to a bare-metal cloud Mac.** Two requirements:

1. **Bare metal, never a hosted VM.** Measured 2026-09-13: hosted macOS runners
   build zero AIR/metallib markers and fail with `mojo: error: failed to run
   the pass manager` on "Apple M1 (Virtual)". Worse, `--target-accelerator
   metal:1` makes the compile *succeed while embedding no kernels*. AWS EC2 Mac
   and MacStadium are dedicated bare metal and should work, but this must be
   PROVEN by compiling one binding and running one lane before scheduling any
   training on them.
2. **At least 32 GiB unified**, preferably 64, so a 12 GB device peak plus host
   mirrors is not fighting the OS.

AWS EC2 Mac dedicated hosts bill with a 24-hour minimum allocation, which
materially changes the arithmetic. Price before committing. Holding Apple to
one segment (section 3) is partly a response to exactly this.

## 6. Cost, from measured cells

REVISED AGAIN the same evening: `lane/attention-speed` then landed DEVIATION
2900 (`_bswz`, a causal block-index swizzle, no arithmetic touched), geomean
0.9592, and the lean step is now **0.1979 s** measured on one H100 in one heat
window. THE STEP HAS MOVED THREE TIMES IN ONE DAY; treat any figure here as
perishable and re-read the lane records before quoting one.

REVISED 2026-09-17 evening, after `lane/attention-speed` measured the current
default. The earlier figures in this file used 0.2326 s from
`bench/OPPONENT_REFERENCE.md` (2026-09-12 13:45Z, commit bb679f19). **Four
flips have landed since** (DEVIATIONS 2597, 2650, 2651, 2657, plus the GEMM
`_hg` flip), and the untimed lean step at commit 07707794 is **0.2106 s**, 9.5
percent faster. That is 9,725 tokens/s on an H100 at batch 1, length 2048.

AMD and Apple per-step costs at this shape are **NOT MEASURED**. The figures
below assume H100-equivalent throughput on all legs, which is certainly wrong
for Apple. Treat as a floor, and see section 4 on wall clock.

| token budget | one route | two routes | cost at $2.00 to $2.69/h |
|---|---:|---:|---:|
| 25B, one twelfth of GPT-3 Small's 300B | 672 h | 1,343 h | $2,686 to $3,613 |

**THE ATTENTION UPSIDE IS MUCH SMALLER THAN THIS FILE FIRST CLAIMED.** An
earlier draft said attention was 61 percent of the step and that a 4x to 8x
there would take the two-route run toward $1,500. Measured at the current
default, attention is **61.9 ms of a 231.5 ms timed envelope, 26.7 percent**.
That is a ceiling on what the attention lane can ever return:

| attention gets | step | one route | two routes | saved |
|---|---:|---:|---:|---:|
| 2x faster | 0.1824 s | 619 h | $2,475 to $3,328 | 13.4% |
| 4x faster | 0.1684 s | 571 h | $2,284 to $3,071 | 20.1% |
| 8x faster | 0.1613 s | 547 h | $2,188 to $2,943 | 23.4% |
| **free** | 0.1543 s | 523 h | $2,093 to $2,815 | **26.7%** |

Even a free attention saves 26.7 percent. **The other 169.6 ms, 73.3 percent
of the envelope, is where the remaining money is and it is not itemized.** The
share is measured against the timed envelope, where every tick waits, so it is
a breakdown and not a price; the untimed step is 210.6 ms, and the 21 ms
difference is instrumentation. Before any further kernel work is scheduled,
that 73.3 percent needs the same treatment attention got.

Batch is 1 in every figure. If `lane/lm-training-shakedown` finds a larger
batch fits, both the hours and the wall clock fall, and that is a larger lever
than anything in the table above.

## 7. Phase order

**Phase 0, blocking, running now.** `lane/lm-training-shakedown`: what batch
fits on one H100, does a 10,000-step run survive, is checkpoint and resume
still bit-exact at 162M, can the loader stream from R2. Nothing in this
repository has ever trained more than 128 steps at any size. No money is
committed until it reports.

**Phase 1, the protocol at the control shape, cheap.** The control shape is
20,453,376 parameters (d384, ff1024, 8 layers, v8192, L2048), already measured
on H100 (0.199 s lean, 10,305 tokens/s) and on Apple (the 6.8 s pilot), and
**it fits the local M4**, so phase 1 needs no cloud Mac. Run the full six
segment two-route handoff here over a small budget. Tens of GPU-hours. This
proves the entire protocol: segment boundaries, three-vendor checkpoint
portability, the B-from-A pipeline, the induction argument end to end, the
final bit comparison, and the negative controls. If the protocol is broken we
learn it here for nearly nothing.

**Phase 2, cloud Mac qualification.** Compile a binding and run one lane on the
candidate bare-metal Mac. Prove Metal AOT emits kernels and the memory fits.
Price it including the minimum allocation window.

**Phase 3, the real run at GPT-3 Small shape.**

## 8. Controls, without which none of this proves anything

- **Sabotage one segment and watch the routes diverge.** Build one segment
  against a binding with one arithmetic step altered and confirm the two routes
  end at different bits. A comparison that cannot fail is not a comparison,
  and this one must be seen to fail before any pass is believed.
- **The missing-moments negative control.** A resume that omits optimizer
  moments must FAIL. Reuse the shape already built for the 34,944 parameter
  campaign.
- **Name the column that is alone.** If the routes differ, print the checkpoint
  hash from every vendor at every segment boundary and name the vendor that
  stands alone before attributing anything. Four AMD legs once chased an
  "MI300X divergence" that was the Apple column being wrong.
- **AMD cross-device bytes go through `transfer_bytes` host staging.** On
  RunPod 2x MI300X a device-1 kernel can read a cross-device copy's destination
  before it is written.
- **Every dataset stages from R2**, never a download on a box.
- **Opponents are measured once** and read from `bench/OPPONENT_REFERENCE.md`.
  This plan measures no opponent.

## 9. What this does NOT prove

One shape, one architecture, one optimizer, one sequence length, one data
order. Apple covers one segment, not the run. It says nothing about other
shapes, the FAST tier, architectures outside the wheel, or speed. It is a
statement about the training trajectory of one model across three vendors, and
that is all it should ever be written as.
