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

## 6. Cost, MEASURED, and the verdict is NOT READY

SUPERSEDES every earlier figure in this file. `lane/lm-training-shakedown` ran
two H100 legs at this exact shape and the answer is **NOT READY**, with four
numbers that decide it.

**1. Only batch 1 survives a long run.** Batch 4 died with
`CUDA_ERROR_OUT_OF_MEMORY` at EXACTLY 210 completed steps, at 71.49 GB. The
three-step sweep (16.95 / 26.61 / 45.94 / 84.33 GB, throughput saturating at
batch 4) describes the FIRST 210 STEPS AND NOTHING LONGER. The onset does not
move with batch; only the consequence does. Batch 2 is the one untested batch
that arithmetic says should fit.

**2. A batch-1 run costs 2.1x after about step 480**, replicated on two
different physical H100s (0.440630 s against 0.440543 s median). Three phases:
flat at 0.207 s and 16.95 GB to step ~210, climbing to ~480, then a PLATEAU at
0.44 s and 34.40 GB that holds through step 2,000. It is a one-time regime
change, not a leak. Ruled out by counters rather than assertion: clocks
1980/1980 MHz, 42 C, 356 W of 700, PSI `full avg10=0.00`, host RSS identical to
the byte at steps 1 and 2,000.

**3. THE CAUSE IS NOW KNOWN AND IT IS A BUG, NOT A COST.** `lane/lm-step-memory-build`
ran a 700-step witness at this shape on the same enwik8 and named it: the eager
attention fallback, the ten `[B, n_heads, L, S]` arrays growing ONE LAYER AT A
TIME and never released. It registered `eager_bytes = 17,314,086,912` BEFORE the
leg and the witness read 17,314,086,912 — equal, not close. Every increment is
1,442,840,540 B, exactly one layer, and the device agrees independently at
1,376 MB per grown layer. The falsifier was watched on NVIDIA: fused 0/8 layers
at 288 B, eager 8/8 at 5,905,580,032 B, identical losses.

On a non-`FUSED_RAN` status the fused kernel has ALREADY run and been paid for
before the eager core runs (`modeling_llama.mojo:3588-3592`), which is why it
costs 2.13x rather than the eager path's own speed.

So the table below has two rows, and the second is what this is worth fixing for:

| 25B tokens | s/step | one route | two routes | cost at $2.00 to $2.69/h |
|---|---:|---:|---:|---:|
| with the fallback | 0.4406 | 1,494 h | 2,988 h | $5,976 to $8,038 |
| **fallback fixed** | 0.2057 | 697 h | **1,395 h** | **$2,790 to $3,753** |

Fixing it is worth **$3,200 to $4,300** on the two-route run.

**THE TRIGGER IS DATA DEPENDENT, AND THAT IS ITSELF A FINDING.** The transition
ran steps 69 to 309 on the witness leg and ~210 to 480 on the shakedown leg.
Same mechanism, different step. **The step at which a run doubles its memory and
halves its speed is NOT a property of the shape**, so no fixed step budget is
safe and no three-step cell can see it.

NOT MEASURED, and not guessed at: WHICH BRANCH asks for the eager path.
`regime_product_ok` cannot be it. `FUSED_CORNER` (`fused_attention.mojo:1908-1910`)
is the only candidate and this run does not show it; the witness says the arrays
grew, not who asked. `stages.attn_materialized` on the report separates them and
is owed.

**4. Session recycling does not fix it.** Rebuilding from `export_state()`
every 250 steps gives 0.440323 s against 0.440543 s straight, indistinguishable,
plus 26 s per rebuild. So there is no optimistic column, and the cause survives
a full session teardown inside the same process.

| token budget | one route | two routes | cost at $2.00 to $2.69/h |
|---|---:|---:|---:|
| FineWeb-Edu 10BT | 597.5 h | 1,195 h | $2,390 to $3,215 |
| 25B tokens | **1,493.7 h** | **2,987 h** | **$5,975 to $8,036** |

That is roughly DOUBLE this file's earlier 315 and 789 hour figures, which were
extrapolated from a fast step that only holds for the first 480.

**5. THE DATA PIPELINE DOES NOT EXIST, and it is the real blocker.** There is
no tokenizer on the training path at all. `CorpusBatches` casts raw bytes to
ids, so with a 50257 vocabulary **77,194,752 parameters, 47.6 percent of the
model** (the embedding and lm_head rows 256 to 50,256), would receive no
gradient and train on nothing. A real run needs a vocabulary, and mojolearn
deliberately ships none (the GPT-2 table was removed under the no-third-party-
data rule). **That is a policy decision, not an engineering task, and it gates
everything else in this file.**

### What DOES work at 162M

- **Resume is still bit-exact**, every witness, every step, in a fresh process,
  with the missing-moments control separating exactly where the arithmetic
  requires. All 8 checks printed by name.
- **`logical_shards` runs at 162M**, K up to 64, no refusal, flat throughput.
  At K=64 a 25B run is 190,735 optimizer steps against the 999,999 cap, so the
  step-cap blocker is solvable without touching any of its fourteen guards.

### Premises of this file that were WRONG

- Batch 8 DOES fit, at 84.33 GB, against the capacity json's 161.75 GiB.
- The 21.9 GB host RSS was the stateless path; the resident trainer holds 8.3 GB
  flat.
- The prior record at 162M was **4 steps**, not 128. The 128-step campaign was
  at 34,944 parameters.

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
