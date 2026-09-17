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

Two routes removes the sampling. Split the run into segments and assign
vendors so that **every segment is run by a DIFFERENT vendor in each route**.
If both routes end at bitwise equal final weights, no vendor diverged at any
step, because a divergence at step k was produced on different hardware in each
route and the finals could not match.

**THE CONSTRAINT IS PER SEGMENT, NOT PER ROUTE.** Andrew proposed

    route A:  Apple -> NVIDIA -> AMD
    route B:  AMD   -> NVIDIA -> AMD

That does not work. Segment 2 is NVIDIA in both routes and segment 3 is AMD in
both, so two of the three segments get no cross-vendor coverage at all. Routes
that merely *look* different are not enough; the two routes must differ in
**every position**.

## 3. The assignment, with the Mac held to one segment

Andrew: "maybe we do mac cloud for only 1/3 of 1 run because it is more
expensive." That is compatible with full coverage. Apple does not need to
appear in both routes, it only needs to be the one that differs somewhere.
With six segments:

| segment | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **route A** | **Apple** | NVIDIA | AMD | NVIDIA | AMD | NVIDIA |
| **route B** | NVIDIA | AMD | NVIDIA | AMD | NVIDIA | AMD |
| distinct? | yes | yes | yes | yes | yes | yes |

Every segment is covered by two different vendors. **Apple appears exactly
once, in 1 of 12 segment-runs.** Segments need not be equal length either, so
Apple's can be the short one; make it long enough to be a real training window
and no longer.

What is lost by holding Apple to one segment: we prove Apple agrees with
NVIDIA over segment 1, not over the whole run. That is honest, it is still a
three-vendor training run, and it sits alongside the per-release full Apple
verification column which covers the inference and fixture claims separately.

## 4. How the two routes can be concurrent when learning is sequential

Andrew: "i don't understand. how can they be concurrent.... doesn't learning
need to depend on prio checkpoint?"

It does, and route A is strictly sequential. There is no way around that.
A0 -> Apple -> A1 -> NVIDIA -> A2 -> ... Each segment needs the one before it.

The move is in route B. Run naively, B chains too:
A0 -> NVIDIA -> B1 -> AMD -> B2 -> ..., and the claim is B1 == A1, B2 == A2,
and so on to the final.

**Instead, start each of B's segments from the corresponding A checkpoint:**

    B segment 1:  A0 -> NVIDIA -> B1'     check B1' == A1
    B segment 2:  A1 -> AMD     -> B2'     check B2' == A2
    B segment 3:  A2 -> NVIDIA  -> B3'     check B3' == A3
    ...

**Why this proves the chained claim, by induction.** Suppose B1' == A1. Then
feeding A1 into B's segment 2 is feeding *the identical bytes* that chained
route B would have fed it. Same input, same code, same hardware, therefore the
same output: B2' == B2. And we checked B2' == A2, so B2 == A2. The argument
carries to the final segment. If any check fails, the chained claim is false
anyway, and we have localized the failure to one segment on one vendor instead
of bisecting two different final checkpoints over millions of steps.

**Why it costs no extra wall clock.** B's segment k needs A's checkpoint k-1,
which A produces at time (k-1)T/N. B's segment k then takes T/N and finishes at
kT/N. So B's last segment finishes at exactly T, the same moment A does. It is
a pipeline, not a race: two boxes busy at all times, 2x GPU-hours, **1x wall
clock**.

    t:      0 ......... T/3 ......... 2T/3 ......... T
    route A  [Apple seg1][NVIDIA seg2][AMD    seg3]
    route B  [NVIDIA s1'][AMD    s2' ][NVIDIA s3' ]
              ^both from A0  ^from A1    ^from A2

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

Base cell: our IDENTICAL step at this shape is **0.2326 s** on an H100
(`bench/OPPONENT_REFERENCE.md`, 2026-09-12 13:45Z, commit bb679f19, batch 1,
length 2048). That is 8,805 tokens/s.

AMD and Apple per-step costs at this shape are **NOT MEASURED**. The figures
below assume H100-equivalent throughput on all legs, which is certainly wrong
for Apple. Treat as a floor.

| token budget | one route | two routes | cost at $2.00 to $2.69/h | wall clock |
|---|---:|---:|---:|---:|
| 25B, one twelfth of GPT-3's 300B | 789 h | 1,577 h | $3,155 to $4,243 | about 33 days at 1 box per route |

Batch is 1. If `lane/lm-training-shakedown` finds a larger batch fits, both the
hours and the wall clock fall. If `lane/attention-speed` lands its 2x to 8x,
the two-route figure falls toward $1,500.

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
