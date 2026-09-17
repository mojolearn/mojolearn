# Plan: tri-vendor handoff training for a GPT-3 Small shape model

Written 2026-09-17 at Andrew's direction: "i think we should train some on
macbook.... some on nvidia and some on amd.... eg we handoff loads.... maybe we
do 2? mac cloud, amd, nvidia, then nvidia, mac, amd? that stronger?"

Yes, it is stronger, and this file says exactly how much stronger, what it
costs, what blocks it today and the order to do it in. Nothing here has run.
Every number that is not marked measured is derived, and says so.

## 1. The claim this design buys

The claim we can make today, from the 128-step byte-LM campaign
(`docs/LM_TRAINING_CLAIM_PLAN.md`), is a **sampled** one: at the points we
checked, three vendors agreed. Sampling is the weakness. A divergence in a
window nobody replayed is invisible.

Two routes through different vendors removes the sampling.

    route A:  Apple -> AMD -> NVIDIA
    route B:  NVIDIA -> Apple -> AMD

Split the run into segments. In route A segment 2 runs on AMD; in route B the
same step range runs on Apple. If the two routes end at **bitwise equal final
weights**, then no vendor diverged at any step of any segment, because a
divergence at step k would have been produced on a different vendor in each
route and the two finals could not match. That is full coverage with no
sampling, and it is a single legible sentence:

> The same model, trained twice by different routes through Apple, AMD and
> NVIDIA hardware, ends at the same bits.

## 2. The optimization that makes it affordable in wall time

Run naively, two routes is 2x GPU-hours **and** 2x wall clock, because route B
is sequential. It does not have to be.

**Route B's segments can all run concurrently, each starting from route A's
checkpoint.** The argument is self-supporting. If the claim is true, route B's
own checkpoint at the end of segment k-1 is bit-equal to route A's, so
starting segment k from A's checkpoint is the same computation as starting it
from B's. If the claim is false, some segment's end checkpoint will not match
and the chained claim is false anyway, which is the thing we wanted to learn.

So:

- **GPU-hours: 2x.** Unavoidable, and correct: every step is executed on two
  different vendors.
- **Wall clock: about 1x**, because route B fans out across as many boxes as
  there are segments.
- **A divergence is localized immediately** to one segment on one vendor,
  instead of appearing as two different final checkpoints that then have to be
  bisected over millions of steps.

This is strictly better than running route B sequentially, and it gives the
same final statement.

## 3. What blocks the Apple leg today

**The local M4 cannot hold this shape.** Measured: 16.0 GiB unified memory,
shared with the OS and every application. The target shape's measured device
peak on an H100 was **12.14 GB**, with 21.93 GB host RSS on the eager path
(`docs/lanes/BRIEF_lm_step_memory_2026-09-10.md`, run 1). Even after the lean
step landed, a 12 GB GPU allocation on a 16 GiB machine that is also running
macOS is not viable, and `docs/NEURAL_TRAINING_STATUS.md` has already recorded
this Mac sitting at about 137 MiB free.

**And even if it fit, it must not run here.** Apple is the one scarce column:
one Mac, one GPU, one job at a time, unrentable, and a full Apple pass has
measured over seven hours during which every other GPU need on this machine
serializes behind it. A multi-day Apple training segment on this laptop would
block the whole program. The standing rule is that the Apple column is recorded
once per PyPI release.

**Therefore the Apple leg goes to a cloud Mac**, which is what Andrew's "mac
cloud" already assumed. Two hard requirements:

1. **Bare metal, not a hosted VM.** Measured 2026-09-13: GitHub-hosted macOS
   runners build zero AIR/metallib markers and fail with `mojo: error: failed
   to run the pass manager` on "Apple M1 (Virtual)". `--target-accelerator
   metal:1` makes the compile succeed while embedding no kernels, which is
   worse than failing. AWS EC2 Mac instances and MacStadium hosts are dedicated
   bare-metal Macs and should be fine, but this must be PROVEN by compiling one
   binding and running one lane before any training is scheduled on them.
2. **At least 32 GiB unified memory**, and preferably 64 GiB, so the 12 GB
   device peak plus host mirrors is not fighting the OS.

Pricing a cloud Mac is an open item; AWS EC2 Mac instances bill with a 24-hour
minimum dedicated-host allocation, which materially changes the arithmetic and
must be checked before committing.

## 4. Cost, from measured cells

Base cell: our IDENTICAL step at the target shape is **0.2326 s** on an H100
(`bench/OPPONENT_REFERENCE.md`, 2026-09-12 13:45Z table, commit bb679f19,
batch 1, length 2048, d_model 768, 12 layers, 12 heads, ff 2048, vocab 50257,
162,147,840 parameters). That is 8,805 tokens/s.

AMD and Apple per-step costs at this shape are **NOT MEASURED**. Every figure
below assumes H100-equivalent throughput on all three legs, which is certainly
wrong for Apple and unknown for AMD. Treat them as a floor.

| token budget | single route | two routes (GPU-hours) | two routes, cost at $2.00 to $2.69/h |
|---|---:|---:|---:|
| FineWeb-Edu 10BT (already in R2) | 315 h | 631 h | $1,262 to $1,697 |
| 25B, one twelfth of GPT-3's 300B | 789 h | 1,577 h | $3,155 to $4,243 |

Each vendor carries about a third of each route, so about 105 h per vendor per
route at 10B tokens, 210 h per vendor across both routes.

If `lane/attention-speed` lands its 2x to 8x on the fused attention (attention
was 61 percent of the step at the older default; its share today is unmeasured
and that lane measures it first), the 25B two-route figure falls toward $1,500.

## 5. Phasing. Do NOT start at 162M

The protocol is the risky part, not the model. Prove the protocol where it is
nearly free, then scale it.

**Phase 0, blocking.** `lane/lm-training-shakedown` answers: what batch fits on
one H100, does a 10,000-step run survive, is checkpoint and resume still
bit-exact at 162M, and can the loader stream 10B tokens from R2. Nothing in
this repository has ever trained more than 128 steps at any size. No money is
committed until phase 0 reports.

**Phase 1, the protocol at the control shape.** The control shape is 20,453,376
parameters (d_model 384, ff 1024, 8 layers, vocab 8192, length 2048) and is
already measured on both H100 (0.199 s lean, 10,305 tokens/s) and on Apple (the
6.8 s pilot, pre-lean). **It fits the local M4.** Run the full two-route
handoff at this shape over a small budget. This costs tens of GPU-hours and
proves the entire protocol: segment boundaries, checkpoint portability across
three vendors, route B fan-out, the final bit comparison, and the negative
control. If the protocol is broken, we learn it here for almost nothing.

**Phase 2, cloud Mac qualification.** Compile a binding and run one lane on the
candidate bare-metal cloud Mac. Prove Metal AOT works and the memory fits.
Price it including any minimum allocation window.

**Phase 3, the real run.** FineWeb-Edu 10BT first, because it is already staged
in R2 as 14 parquet shards and needs no new data work. 25B only after 10B has
completed clean end to end.

## 6. Controls, without which none of this proves anything

- **The negative control is mandatory.** A resume that omits optimizer moments
  must FAIL the final comparison. Run it and watch it fail before trusting any
  pass. The effective missing-moments control already exists for the 34,944
  parameter campaign; reuse its shape.
- **A sabotaged segment must break the route.** Build one segment against a
  binding with one arithmetic step altered and confirm the two routes end at
  different bits. A comparison that cannot fail is not a comparison.
- **Name the column that is alone.** If the two routes differ, print the cell
  hash from every vendor at every segment boundary and name the vendor that
  stands alone, before attributing anything. Four AMD legs once chased an
  "MI300X divergence" that was the Apple column being wrong.
- **AMD cross-device bytes go through host staging.** On RunPod 2x MI300X a
  device-1 kernel can read a cross-device copy's destination before it is
  written. All AMD cross-device traffic uses `transfer_bytes`.
- **Every dataset stages from R2**, never a download on a box.
- **Opponents are measured once** and read from `bench/OPPONENT_REFERENCE.md`.
  This plan measures no opponent; the torch columns for this exact shape are
  already in that file.

## 7. What this does NOT prove

One shape, one architecture, one optimizer, one sequence length, one data
order. It says nothing about other shapes, about the FAST tier, about
architectures outside the wheel, or about speed. It is a statement about the
training trajectory of one model on three vendors, and that is all it should
ever be written as.
