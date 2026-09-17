# LM training shakedown: what an H100 actually did

Lane `lane/lm-training-shakedown`. Leg 1 ran 2026-09-17 20:07Z on a RunPod
NVIDIA H100 80GB HBM3 (sm_90a), commit 025910137, driver and card in the leg
directory. Corpus enwik8, staged from R2, 100,000,000 bytes, sha256
`2b49720e...`. Every arm is the IDENTICAL shipped default; nothing here is an
opponent ratio or a default gate.

Target shape throughout: batch B, length 2048, d_model 768, 12 heads, 12 KV,
head_dim 64, intermediate 2048, 12 layers, vocab 50257, 162,147,840
parameters. Resident session, `step_result='lean'`, no per-step witnesses,
three steps per cell with the first (which carries setup) excluded from the
median.

## 1. The batch sweep, and it is nothing like the analytic bound

| batch | tokens/step | median s/step | tokens/s | device peak (this process) | host RSS | first call |
|---|---|---|---|---|---|---|
| 1 | 2,048 | 0.20708 | 9,890.0 | 16.95 GB | 8.32 GB | 13.51 s |
| 2 | 4,096 | 0.37504 | 10,921.6 | 26.61 GB | 8.27 GB | 12.31 s |
| 4 | 8,192 | 0.65732 | 12,462.7 | 45.94 GB | 8.30 GB | 13.03 s |
| 8 | 16,384 | 1.31258 | 12,482.3 | 84.33 GB | 8.27 GB | 13.67 s |

**All four fit.** Device peak is polled `nvidia-smi --query-compute-apps` for
this pid; the device-wide figure is within 12 MB of it in every row, so
nothing else was on the card.

**`bench/results/lm_capacity_2026-09-10/b8-l2048.json` is wrong by 2.06x and
should be read as retired.** It puts batch 8 at 161.75 GiB in the attention
subset alone and records `fit_admitted: false`. The measurement is 84.33 GB =
78.5 GiB for the WHOLE step, and it ran. That file labels itself "host
arithmetic only; not measured memory or throughput", which is exactly right;
the number inside it was never an OOM observation and must not be quoted as
one. The eight per-head attention matrices per layer it sums are not all live
at once.

**Memory is linear in batch with a fixed floor.** Least squares over the four
cells:

    device_GB = 7.367 + 9.624 * batch      (residual under 0.3 GB in every row)

The 9.624 GB per batch unit is the activation and attention working set at
L2048; the 7.367 GB floor is close to the 2.594 GB of parameters, gradient and
the two AdamW moments plus the fixed workspaces. On this card (81,559 MiB =
85.52 GB) that fit admits **batch 8 and refuses batch 9** (93.9 GB predicted).
Batch 8 ran at 98.6% of the device, which is a real run but not a margin
anyone should plan a 200-hour job around.

**Throughput saturates at batch 4.** 9,890 to 10,922 to 12,463 tokens/s is a
26% gain, and batch 8 adds 0.15% on top of batch 4 for 83% more memory. Batch
4 is the operating point: the same tokens per second as batch 8 at 45.94 GB
instead of 84.33 GB.

**Host RSS does not move with batch**: 8.27 to 8.32 GB across a 5x change in
device memory. The 21.9 GB host RSS recorded in the 2026-09-10 probe runs was
the STATELESS (`resident=False`) path, which copies the whole state per call;
the resident path costs 8.3 GB and is the one to use. That is a resolved
question, not an open risk.

## 2. Bitwise identity still holds at 162M, five days and one card later

The three-step witness at batch 1 on enwik8 reproduces the pinned
2026-09-12 record (`bench/results/e1g/2026-09-12_133007-nvidia-h100-owed-rest/
remote/attention-step/.../lm_summary.tsv`, step 3) hash for hash on a
different physical H100 at a different commit:

    gradients  80502bcbaa38a7f5c40a0f9886a7300b13ea31aaef29ac93a241adc4fe601b0d
    parameters cb6cfb17e14fc800c38b68bb37a6453a782a8d13de0107dd077aa5a2759ac808
    m          62a57517ac0891e078a2d17199044887b3e460bf5e42af74859ccf584be9a6eb
    v          eca204720d5c1c692d2ae808d79fb51bf90a502961202b0450615b7f0e4c2d6d
    flags      360d579dbd14759b41afdf7fb5e80c0101e15150ae401d59f92a1e32d129f7cb

This was not an arm of this lane; it fell out of the batch-1 cell and is
recorded because it is evidence.

## 3. The step is 11% faster than the number everyone is quoting

`bench/OPPONENT_REFERENCE.md` carries 0.2326 s for our IDENTICAL arm on
enwik8, measured at commit bb679f19 on 2026-09-12. The same probe, same
shape, same corpus, same arm, on this card at 025910137 gives **0.20708 s**,
which is 11.0% faster. Opponent columns are measured once per tuple and are
NOT re-measured here, so no ratio in that file is restated: our cell moved,
theirs was not observed, and the two cannot be divided across pods
(`bench/OPPONENT_REFERENCE.md` says so itself about `compile_bf16`).

## 4. Recosting from the measured throughput

| corpus | batch 1 | batch 2 | batch 4 | batch 8 |
|---|---|---|---|---|
| FineWeb-Edu 10BT, H100-hours | 280.9 | 254.3 | 222.9 | 222.5 |
| FineWeb-Edu 10BT, optimizer steps | 4,882,812 | 2,441,406 | 1,220,703 | 610,352 |
| 25B tokens, H100-hours | 702.2 | 635.8 | 557.2 | 556.3 |
| 25B tokens, optimizer steps | 12,207,031 | 6,103,516 | 3,051,758 | 1,525,879 |

Against the 999,999-step ceiling (SOURCE_BLOCKERS.md blocker A), **only batch
8 can reach 10B tokens at all**, and **no batch can reach 25B**: batch 13
would be needed and the card refuses batch 9.

At the $2.2 to $2.5 per H100-hour the original estimate was costed with,
10BT at batch 4 is about **$490 to $560** against the $700 quoted at batch 1,
and 25B is about **$1,230 to $1,390** against $2,000. The bill is 21% lower
than the batch-1 estimate and remains the least interesting number here.

## 5. Bit-exact resume DOES still hold at 162,147,840 parameters

`export_checkpoint` refuses this shape (blocker B), so the arm went through the
substitute its own docstring names: `export_state()` arrays as the checkpoint,
restored by constructing a trainer from the saved parameters, optimizer
configuration, shape and data schedule and then calling `load_state_dict` --
the same two calls `from_checkpoint_bytes` makes, minus the size-bounded JSON
envelope. `tools/lm_shakedown_resume.py`. Three arms, three FRESH processes,
the broken one before the one being trusted.

Control shape first (20,453,376 parameters, warmup 4, tail 3): PASS, save
1.30 s, restore 1.00 s. Then the target shape, warmup 8, tail 4:

    saved_at_step                     8
    save_seconds                      11.19
    restore_seconds                   7.63
    control_missing_moments_differs   true
    resume_bitwise_equal              true
    verdict                           PASS

**Every witness of every tail step matched the uninterrupted run**: loss,
flat gradients, parameters, m, v and flags, by sha256, printed by name rather
than counted.

**The control is the part that makes that mean anything.** Zeroing m and v
leaves the parameters untouched, so the first resumed step must read the same
weights and produce the same loss and gradient; the optimizer state can only
show up in what the update does. All eight checks landed on the side the
arithmetic requires:

    CHECK pass: first tail step loss matches (the parameter restore is sound)
    CHECK pass: first tail step gradients matches (the parameter restore is sound)
    CHECK pass: first tail step m differs (the zeroed moments reached the update)
    CHECK pass: first tail step v differs (the zeroed moments reached the update)
    CHECK pass: first tail step parameters differs (the zeroed moments reached the update)
    CHECK pass: last tail step loss differs (the witnesses can see the optimizer)
    CHECK pass: last tail step gradients differs (the witnesses can see the optimizer)
    CHECK pass: last tail step parameters differs (the witnesses can see the optimizer)

A control that matched everywhere would have meant the witnesses cannot see the
optimizer and the resume arm proved nothing. It did not match, and the two
places it was REQUIRED to match are the proof that the parameter half of the
restore is sound and that the moments are the only thing the control changed.

This is ONE GPU and ONE shape. It is not a cross-vendor resume claim; the
NVIDIA/AMD bidirectional result on record is the 34,944-parameter model and
this does not extend it.

At 11.19 s to save and 7.63 s to restore, checkpointing is not a cost worth
thinking about: every 10 minutes of a 223-hour run is a 1.9% tax, every hour
is 0.3%.

## 6. THE LONG RUN: it survives, and it costs 2.1x after about step 480

2,000 consecutive steps at batch 1, target shape, one resident session, no
per-step witnesses. **This is 15.6x the longest run anything in this repository
had ever done at any size, and 500x the longest at this shape.**

Nothing crashed, nothing went NaN, the loss fell and kept falling, and host
memory never moved. But the run does not hold its speed, and what it does is
invisible in three steps.

| step | s/step | device peak | loss |
|---|---|---|---|
| 1 | 12.994 (setup) | 15.607 GB | 10.9033 |
| 51 | 0.20694 | 16.949 GB | |
| 101 | 0.20676 | 16.949 GB | |
| 151 | 0.20972 | 16.949 GB | |
| 201 | 0.20666 | 16.949 GB | |
| 251 | 0.20753 | 19.902 GB | |
| 301 | 0.25892 | 24.197 GB | |
| 351 | 0.28469 | 25.808 GB | |
| 401 | 0.33653 | 27.150 GB | |
| 451 | 0.25890 | 31.445 GB | |
| 501 | 0.31069 | 34.397 GB | |
| 651 | 0.44045 | 34.397 GB | 2.0090 |
| 801 | 0.44015 | 34.397 GB | 1.4451 |
| 951 | 0.46608 | 34.397 GB | 1.8832 |
| 1101 | 0.44075 | 34.397 GB | 1.7804 |

**Three phases, and the third is a PLATEAU, not a runaway.**

  * **Phase 1, steps 1 to about 210.** 0.207 s a step, 16.949 GB. This is the
    regime every previously recorded run lived in, because every previously
    recorded run was three or four steps long.
  * **Phase 2, steps about 210 to about 480.** Device memory climbs from
    16.949 GB to 34.397 GB and step time climbs from 0.207 s to about 0.44 s.
  * **Phase 3, from about step 480 onward.** BOTH SETTLE. Device memory sits at
    34.397 GB and does not move again through step 1,100 and beyond. Step time
    sits between 0.440 and 0.466 and does not move again.

At step 650 this looked like an unbounded leak and it is not one. It is a
one-time transition to a second steady state that costs **2.03x the device
memory and 2.13x the seconds per step**. Saying "leak" here would be wrong.

**What it is not.** Sampled at step 776 while the run was in phase 3
(`drift_diagnostics.txt` in the leg directory, collected by hand over ssh):

  * not clock: `clocks.sm` 1980 MHz of `clocks.max.sm` 1980 MHz;
  * not thermal: 42 C, `HW Thermal Slowdown` and `SW Thermal Slowdown` both
    Not Active;
  * not power: 356 W of a 700 W limit, `SW Power Cap` Not Active, and every
    other clocks-event reason Not Active;
  * not host CPU contention: the container has a 2210000/100000 cgroup quota,
    that is 22.1 CPUs of the host's 208, and `/proc/pressure/cpu` reports
    `full avg10=0.00` with `some avg10=0.09`. The host load average of 29 is
    14% of 208 cores and is not reaching us. This one was checked
    specifically because a shared RunPod host is the obvious confound, and the
    pressure counters rule it out rather than an assertion doing so;
  * not host memory: `ru_maxrss` 8.193 GB at step 1 and 8.193 GB at step
    1,100, unchanged to the byte.

So the cost is on the device, inside the step, at full clocks. **What causes
it is NOT established by this leg and is not guessed at here.**

**The loss is healthy.** 10.9033 at step 1 is ln(50257) = 10.8249 plus
noise, which is exactly an untrained uniform prediction. It spikes early
(57.92 at step 3 in the batch-1 sweep cell), recovers, and is bouncing between
1.4 and 2.3 by step 1,100 on byte-level enwik8. It is not flat and it is not
NaN. The early spike at lr 1e-3 on a fresh 162M model is worth a learning-rate
warmup in a real run, and it is not a defect.

## 7. What section 4's recosting is worth, given section 6

**The batch sweep in section 1 is a PHASE 1 MEASUREMENT.** Every cell is three
steps and every cell therefore lives inside the first 210 steps, where the
step is fast and the memory is small. If the same transition happens at every
batch, then every throughput number in section 1 and every hour in section 4
is optimistic by up to 2.13x, and every device peak is optimistic by up to
2.03x, which at batch 8 would be 68 GB over a card that has 85.5.

The honest recosting is a range, not a number:

| corpus | if phase 1 can be held | if phase 3 is the truth |
|---|---|---|
| FineWeb-Edu 10BT at batch 1 | 280.9 H100-hours | 597.5 H100-hours |
| 25B tokens at batch 1 | 702.2 H100-hours | 1,493.7 H100-hours |

The original estimate this lane was asked to check (315 h and 789 h) sits at
the optimistic end. **The pessimistic end is roughly double it.**

Which end is real depends on one question that leg 2 asks: does tearing the
resident session down and rebuilding it from `export_state()` put the run back
into phase 1? Section 5 already proved that rebuild is bit-exact at this
shape, so if it also resets the clock, then a production run recycles its
session every couple of hundred steps, holds phase 1, and the section 4 table
stands. If it does not reset, phase 3 is the price and every estimate doubles.

**Do not quote section 4 without section 7.**

## 8. The completed 2,000-step run, end to end

The run finished all 2,000 steps in 840 s (`long exit=0 secs=840`,
`finished=2026-09-17T20:30:45Z`). Full curve from
`.../remote/lm-shakedown/long/events.jsonl.gz`:

| what | value |
|---|---|
| steps completed | 2,000 of 2,000, `limited` false |
| step 1 (setup included) | 12.994 s |
| fastest timed step | 0.20650 s |
| median timed step | 0.44063 s |
| slowest timed step | 0.51849 s |
| slowest / median | **1.177x** |
| steps slower than 1.5x the median | **0** |
| median of the first 199 timed steps | 0.20686 s |
| median of the last 199 | 0.41503 s |
| drift across the run | **+100.63%** |
| host `ru_maxrss` first / last | 8.193 GB / 8.193 GB, **delta 0 bytes** |
| host `VmRSS` first / last | 2.361 GB / 2.396 GB, delta +34.7 MB over 2,000 steps |
| device peak first / last / max | 15.607 / 34.397 / 34.397 GB |
| loss first / min / last | 10.903326 / 0.605758 / 1.717035 |
| NaN or inf steps | **0** |
| loss mean, first 200 vs last 200 | 6.18672 vs 1.70408 |
| distinct rounded losses in the last 200 steps | **200** (1 would mean flat) |

Two things in that table matter as much as the drift.

**There are no stalls.** The slowest step of 2,000 is 1.177x the median and
exactly zero steps exceed 1.5x it. Whatever the transition is, it is a smooth
change of regime and not an intermittent hitch, a swap, a page fault storm or
a garbage collection. A 610,352-step run would not be punctuated by surprises;
it would simply run at the phase-3 rate.

**Host memory is not growing.** `ru_maxrss` is identical to the byte at step 1
and step 2,000. `VmRSS` moved 34.7 MB across 2,000 steps, which is 17 KB a
step and is the harness's own accumulating list of step records, not the
trainer. The 21.9 GB host RSS that prompted this question was the stateless
path and does not occur here.

The loss is doing what training looks like: 10.903 at step 1 (ln(50257) =
10.825, an untrained uniform model), a minimum of 0.606, 200 distinct values in
the last 200 steps, and not one NaN or inf in 2,000 optimizer steps.
