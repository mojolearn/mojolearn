# GPT-3 Small under the IDENTICAL contract: NOT READY, and here is the list

Lane `lane/lm-training-shakedown`, 2026-09-17. Evidence in this directory;
leg 1 ran on a RunPod NVIDIA H100 80GB HBM3 at commit 025910137.

**The verdict is NOT READY, and the bill is still not the reason.** Four
things stand between the current stack and a 10B-token run, one of them
absolute, one of them a decision nobody has made, one of them newly measured
and not understood, and one of them merely awkward. None is about money.

Read with: `SOURCE_BLOCKERS.md` (what the code refuses), `MEASURED.md` (what
the H100 did), `DATA_PIPELINE.md` (whether it can be fed).

## The four questions, answered

### 1. What batch fits, and what does throughput do? ANSWERED.

Every batch from 1 to 8 fits on an 80 GiB H100 at the target shape.

| batch | tokens/step | s/step | tokens/s | device peak |
|---|---|---|---|---|
| 1 | 2,048 | 0.20708 | 9,890 | 16.95 GB |
| 2 | 4,096 | 0.37504 | 10,922 | 26.61 GB |
| 4 | 8,192 | 0.65732 | 12,463 | 45.94 GB |
| 8 | 16,384 | 1.31258 | 12,482 | 84.33 GB |

Memory is linear, `device_GB = 7.367 + 9.624 * batch`, which admits batch 8
and refuses batch 9. Throughput saturates at batch 4: batch 8 buys 0.15% more
for 83% more memory and runs at 98.6% of the card. **Batch 4 is the operating
point.** The analytic file that said batch 8 needs 161.75 GiB and
`fit_admitted: false` was arithmetic, never a measurement, and is corrected in
`bench/results/lm_capacity_2026-09-10/SUPERSEDED.md`.

Caveat that matters: **these are three-step cells, so they are all phase-1
numbers.** See question 2.

### 2. Does a long run survive? PARTLY, AND IT GETS 2.1x SLOWER.

2,000 consecutive steps, 15.6x anything this repository had run at any size
and 500x anything at this shape. Nothing crashed, nothing went NaN, the loss
fell from 10.9033 (which is ln(50257), i.e. an untrained model) toward 1.4 to
2.3, and host RSS was 8.193 GB at step 1 and 8.193 GB at step 1,100,
unchanged to the byte.

But the run changes regime once, around steps 210 to 480, and then settles
into a second steady state costing **2.03x the device memory** (16.949 GB to
34.397 GB) and **2.13x the seconds per step** (0.207 to 0.44). Both then
plateau and hold. It is not a leak and it is not a runaway; it is a step
change that no three-step run can see.

It is not clocks (1980 of 1980 MHz), not thermal (42 C, no slowdown), not
power (356 W of 700, no cap), not host CPU contention (PSI `full` exactly
0.00 against a 22.1-CPU quota) and not host memory. **The cause is not
established and is not guessed at here.**

10,000 consecutive steps was NOT run: at the post-transition rate they do not
fit inside a 60-minute lease, which is itself a finding about how a
610,352-step production run would have to be operated.

### 3. Is resume still bit-exact at 162M? YES.

Every witness of every resumed step matched the uninterrupted run: loss, flat
gradients, parameters, m, v and flags, by sha256, in a fresh process. Save
11.19 s, restore 7.63 s.

The missing-moments control separated, and it separated where the arithmetic
requires: the first resumed step's loss and gradient MATCH (proving the
parameter half of the restore) while m, v and the updated parameters differ,
and by the last step the loss and gradient differ too. All eight checks are
printed by name in the leg directory. A control that matched everywhere would
have meant the check could not fail.

**But the checkpoint FILE path is structurally unavailable at this size.**
`export_checkpoint` refuses above 87,381 parameters, so `from_checkpoint`
cannot be reached at 162,147,840. The arm went through `export_state()`
arrays, which the docstring names as the substitute and which no harness in
the repository implemented until this lane.

### 4. Can the pipeline feed 10B tokens? NO.

FineWeb-Edu 10BT is in R2 as 14 pinned parquet shards and staging works fine
(100 MB in 7 s). Everything between the shards and `train_step` is missing:
the loader reads one whole file into host RAM, nothing reads parquet, and
`CorpusBatches` does not tokenize at all -- it casts raw bytes to int32 ids.

The consequence that is easy to miss: at vocab 50,257 the embedding and
lm_head are 77,194,752 of the 162,147,840 parameters, **47.6% of the model**,
and byte ids touch rows 0 to 255 only. Nearly half the model would train on
nothing. A real run needs BPE ids, which needs a vocabulary, and mojolearn
deliberately ships none. Details and scoping in `DATA_PIPELINE.md`.

## The blockers, ranked

1. **The step counter is capped at 999,999**, in the host, the binding and the
   Mojo trainer alike (fourteen guards across five files, listed in
   `SOURCE_BLOCKERS.md`). At batch 8
   that is 16.4B tokens, so 10BT fits and **25B does not, at any batch the
   card admits**. The way past it is `ParallelByteLanguageModelTrainer` with
   `logical_shards` and `devices=(0,)`, which sums K microbatches into one
   optimizer step; it has never run above a toy shape.
2. **The data pipeline does not exist**, and the vocabulary at the front of it
   is an unmade decision, not a task.
3. **The phase-2 transition**, cause unknown, worth 2.1x on every hour
   estimate and 2.03x on every device peak. If it is proportional at every
   batch, batch 4's 45.94 GB becomes about 93 GB and OOMs mid-run, which would
   delete the operating point chosen in question 1.
4. **No checkpoint file above 87,381 parameters.** Small, contained, and this
   lane already wrote the substitute.

## Recosting, as a range

| | if phase 1 holds | if phase 3 is the truth |
|---|---|---|
| FineWeb-Edu 10BT, batch 4 | 222.9 H100-hours | ~475 H100-hours |
| FineWeb-Edu 10BT, batch 1 | 280.9 H100-hours | 597.5 H100-hours |
| 25B tokens, batch 4 | 557.2 H100-hours | ~1,188 H100-hours |

The estimate this lane was asked to check (315 h for 10BT, 789 h for 25B at
batch 1) sits near the optimistic end. At roughly $2.2 to $2.5 per H100-hour
the whole span is **$490 to $3,000**. The money was never the risk and it
still is not.

## What could not be tested, and is owed

  * **whether the phase-2 transition happens above batch 1.** This decides
    whether the batch sweep is usable at all and is the single most important
    owed measurement.
  * **whether recycling the resident session resets it.** Question 3 proved
    the rebuild is bit-exact, so if it also resets the clock the mitigation is
    free and the optimistic column stands.
  * **`logical_shards` at 162M**: throughput, device peak, and whether it runs
    at this shape at all.
  * **10,000 or more consecutive steps.** 2,000 is what fit.
  * **anything on AMD or Apple at this shape.** The cross-vendor resume result
    on record is the 34,944-parameter model and nothing here extends it.
  * **any quality claim.** The loss falls on byte-level enwik8; that is a
    health signal, not a benchmark, and with 47.6% of the model untrained it
    could not be one.

## The recommendation

Do not book 200 hours yet. The two cheap experiments that change the answer
are the phase-transition-versus-batch question and the session-recycle
question, both of which are one rented hour. After those, the order is:
decide the vocabulary, build the pre-tokenization pass, and measure
`logical_shards` at this shape. The 999,999 cap can be left alone if 10BT is
the target and batch 8 is acceptable; it must be raised, or shards must work,
for 25B.
