# Plan: a compute-optimal GPT-3 Small shape model, trained in six segments across NVIDIA, AMD and Apple, twice, ending at the same bits

Written 2026-09-22 at Andrew's direction. Supersedes the 2026-09-17 tri-vendor
plan (deleted in the 2026-09-19 lane cleanup, last revision `da9f7bc52`),
which was costed at 25B tokens and read NOT READY. Three things changed since
then and each one is measured. The eager attention fallback that doubled the
step after step 480 is repaired and a 2,000-step batch-4 run held a flat
12,390 tokens/s and 38,959 MiB on an H100
(`bench/results/lm_attention_fallback_2026-09-18/endurance/verdict.log`).
The run's own vocabulary exists and is pinned in R2
(`vocab/mojolearn-bpe-50257-v1`, trained on FineWeb-Edu shard 000, commit
`5aedd1c94`). One model was trained live on an Apple M4, an H100 and an
MI300X at once through `mojolearn.cross_vendor`, bit-identical after every
step (`bench/results/live_xvendor/2026-09-21/README.md`, commit `eec83b539`).

Nothing in this file has run at the target shape and budget. Every number is
marked measured or derived. Derived numbers are floors until a test run
replaces them.

## 1. The headline

**A GPT-3 Small shape model trained on a compute-optimal token budget, in six
segments handed between NVIDIA, AMD and Apple, one segment trained by NVIDIA
and AMD together, the whole run performed twice by different hardware routes,
with every optimizer step of both runs ending at the same bits.**

Shape, unchanged from the shakedown. batch 4 per shard, length 2048, d_model
768, 12 layers, 12 heads of head_dim 64, intermediate 2048, vocabulary 50,257,
**162,147,840 parameters** (untied embedding and head). Corpus FineWeb-Edu
sample-10BT, 14 parquet shards already pinned in R2. Optimizer AdamW with
weight decay, no clipping (the shipped optimizer contract has no clipping
path, `training/IDENTICAL_OPTIMIZER_CONTRACT.md`).

## 2. Compute-optimal means about 2.6B tokens, not 25B

The 2026-09-17 plan used one twelfth of GPT-3's 300B tokens. A compute-optimal
budget for this parameter count is about 20 tokens per parameter. GPT-3 Small
is conventionally counted at 124M (tied embedding), so the budget is

| item | value | how |
|---|---:|---|
| optimizer steps | 5,000 | chosen |
| shards per step (K) | 64 | recipe, fixed on every device count |
| sequences per step | 256 | 64 shards of batch 4 |
| tokens per optimizer step | 524,288 | GPT-3 Small's own 0.5M batch |
| **tokens total** | **2,621,440,000** | 5,000 x 524,288, 21 per parameter at 124M, 16 at 162M |
| corpus needed | about 5 of the 14 shards plus shard 013 held out | derived |

That is one tenth of the 25B run and it is the whole reason the bill in
section 8 is under $1,000 instead of $6,000 to $8,000.

**Schedule.** Linear warmup over the first 250 steps to a peak, cosine decay
to one tenth of the peak at step 5,000. The trainer holds `lr` as one float32
scalar and nothing varies it per step today. The per-step scalar is computed on
the host in float32 by one function, so every box gets the same bits, and the
value at each step is part of the recorded data schedule. This is engineering
item E2 below. Shard gradients are SUMMED in the fixed fold, not averaged
(`python/mojolearn/parallel_training.py`), so the peak learning rate is set for
a summed gradient of 64 shards and written into the recipe.

## 3. Six segments, five handoffs, two routes, Apple once, one multi-vendor segment

The constraint that makes two routes into a proof is per segment, not per
route. Every segment must be run by different hardware in route A and route B.
Then if both routes end at the same bits, no vendor diverged at any step,
because a divergence at step t was produced on different hardware in each route
and the finals could not agree.

| segment | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| optimizer steps | 1,000 | 1,000 | 400 | 1,500 | 1,000 | 100 |
| **route A** | NVIDIA | AMD | **NVIDIA + AMD** (NVIDIA folds first) | NVIDIA | AMD | **Apple** |
| **route B** | AMD | NVIDIA | **AMD + NVIDIA** (AMD folds first) | AMD | NVIDIA | NVIDIA |
| differs between routes | yes | yes | yes, ownership and fold order swapped | yes | yes | yes |

**Segment 3 is the multi-cluster segment and it is in both routes**, as
Andrew asked. It is the same recipe in both, one NVIDIA box and one AMD box
training the same step together through `mojolearn.cross_vendor`, and it still
satisfies the per-segment constraint because the two routes swap which vendor
owns which shards and which vendor computes the first half of the ordered
fold. Section 5 says how it works at this size.

**Apple appears exactly once, in segment 6 of route A**, the shortest segment
and the last one. Last for a reason. In the pipeline of section 4, route B's
segment k+1 waits for route A's checkpoint k, so an Apple segment in the middle
would stall route B for as long as the Mac takes. Placed last, the Mac is on
nobody's critical path. What is lost is that Apple agrees with NVIDIA over
100 steps and not over the run. The final inference column in section 6 puts
Apple on the final weights as well, at a cost of minutes.

Segment lengths are unequal on purpose. Apple's is sized to what a rented Mac
does in a day. The multi-vendor segment is sized to the wide-area bandwidth
bill in section 5. The rest are long enough that a defect that appears after a
few hundred steps, like the one the shakedown found, is inside a segment.

## 4. Two routes at one wall clock

Route A is strictly sequential, A0 to A1 to A2 and so on. Route B does not
chain. **Route B's segment k starts from route A's checkpoint k-1** and its
result is compared to A's checkpoint k. If B1 equals A1 then feeding A1 into
B's segment 2 is feeding the identical bytes chained route B would have fed
it, so B2 equals what chained B would have produced, and the induction carries
to the final. Every segment of B finishes one segment after the matching
segment of A, so two routes cost two times the GPU hours at about one times the
wall clock, and a failure is localized to one segment on one vendor instead of
being bisected across the run.

Because B's segment k starts from the same bytes as A's, **every per-step
state hash of B's segment is comparable to A's**, not only the boundary. That
is where most of the evidence in section 6 comes from.

## 5. How the bits agree, and what the multi-vendor segment needs

**Within a box.** A step is K=64 logical shards. Each shard's gradient is
computed by the IDENTICAL kernels a one-GPU step uses, so it does not depend on
which GPU or vendor computed it. The shards are combined by the ordered left
fold `total = ftz(fma(1, ftz(total), ftz(g_k)))` for k in shard order, never a
collective, and AdamW is elementwise. So one GPU doing 64 shards in a loop, or
eight GPUs sharing them, produce the same bits. Measured at this shape, 1 and 2
L40S agree after 3 steps (`bench/results/2026-09-20_gpt3_small_multigpu`), and
at the small shape 1, 2 and 4 devices on 2x H100 and 2x MI300X agree
(`bench/results/par_lm_xvendor/2026-09-21`). Device count divides wall clock;
it never changes bits.

**Across vendors, live.** `mojolearn.cross_vendor` (commit `5b7c82abe`) has a
coordinator that runs the same fold on the host and workers that compute their
own shards and apply the total, with a full-state sha256 compared across
workers after every step. It ran for real on three vendors at once. Its limit
is bandwidth. **Every step it moves every shard's whole gradient to the
coordinator**, 4 bytes per parameter, 649 MB per shard at 162M, which at K=64
is 41 GB per step. That is fine for the 2-block model it ran on and impossible
over a wide-area link at this shape.

**The chained fold fixes it without touching a kernel.** The fold is a left
fold, so the owner of shards 0 to j-1 can compute the prefix `fold(g0..g_{j-1})`
on its own box, ship ONE gradient (649 MB) to the owner of shards j to 63, who
continues `fold(prefix, g_j, ..., g_63)` and ships the total back. Two
transfers per step, 1.3 GB, and the result is bit for bit the flat fold a
single GPU computes. The second owner computes its shard gradients while the
first is working and holds them on device (20 shards is 13 GB, fine on an
MI300X). Route A has NVIDIA folding first and AMD finishing; route B swaps them,
so the fold itself is covered by both vendors in both roles. This is
engineering item E5. Shard counts per vendor are a balancing knob, not part of
the recipe; K and the fold order are the recipe.

Derived cost of one multi-vendor step, to be measured in T1c. Compute is the
slower vendor's shards (AMD, about 20 shards at 1.5 to 1.9 s each, 30 to 37 s)
overlapped with NVIDIA's (44 shards at 0.66 s, 29 s), plus two 649 MB transfers
at 50 to 100 MB/s (13 to 26 s). About 45 to 65 s per step, 400 steps in 5 to 7
hours, both boxes busy, about 520 GB moved.

**AMD cross-device bytes stay host-staged.** Any multi-GPU AMD box moves bytes
between devices only through `transfer_bytes`, after the SR-IOV stale-read
finding on 2x MI300X.

## 6. Evidence, and there is a lot of it by construction

The point of the design is that evidence is produced by the training itself,
every step, on every box, not by a separate campaign afterward.

1. **A per-step hash chain on every box.** After every optimizer step the box
   records the sha256 of the complete state (parameters, m, v, flags), the
   sha256 of the summed gradient, every shard's loss, the learning rate used,
   the token offsets of every shard, and the previous line's hash, so the log
   is a chain. Two routes times 5,000 steps is **10,000 full-state hashes**,
   and every B hash must equal the A hash of the same step. That is 5,000
   cross-hardware agreements, not five.
2. **Checkpoints.** Full raw state (about 1.95 GB at this shape, save 11 s,
   restore 8 s measured) at every segment boundary and every 250 steps inside a
   segment, pushed to R2 as they are written with size and sha256 in a
   manifest. About 60 checkpoints, about 120 GB. Nothing lives only on a pod.
3. **Arrival replay at every handoff.** Before a box continues from a
   checkpoint it verifies the sha256 against the manifest, loads, hashes the
   state and matches the sender's final hash, then replays the last two
   optimizer steps of the previous segment from the step-minus-two checkpoint
   on ITS OWN hardware and must reproduce the boundary hash. Every boundary is
   then witnessed by three pieces of hardware, the sender's, the receiver's and
   the other route's, before training goes on.
4. **Record windows.** Two steps either side of every boundary, and steps 1 to
   3, run with the full per-step witness (every shard's gradient hash, the
   fold's intermediate hashes). These are what the CPU and Apple witnesses are
   compared to.
5. **CPU witness column.** A rented CPU pod (about $0.24 an hour) replays one
   shard's gradient from every boundary checkpoint with the host binding and
   must match the recorded shard-gradient hash. The rented CPU column has been
   bit-equal to Metal on every lane measured, so this is also an Apple-class
   witness that costs nothing scarce.
6. **Held-out loss at every boundary**, on shard 013, computed on every vendor
   present, and equal bit for bit.
7. **Negative controls that must be seen to fail, at the real shape, once.**
   One ulp changed in one element of one shard's gradient at one step (the
   coordinator already refuses this, seen on the M4). A resume with zeroed
   moments (differs at every hash, seen at the small shape). A replay at K=63.
   A fold with two shards swapped. A binding with one arithmetic change in the
   attention tail. Each is recorded as EXPECTED FAIL with the hash that
   differed. A comparison that has never been seen to fail is not a comparison.
8. **Environment receipts per box.** GPU model and UUID, driver, CUDA or ROCm
   or Metal version, commit, binding sha256, source inventory hash, pod or
   droplet id, lease, the on-box dead-man, and the 404 that proves the box is
   gone.
9. **Data provenance.** sha256 of every token shard and of the vocabulary,
   and the data schedule as a pure function of the step index, so any step's
   batch can be rebuilt by anyone from the manifest.
10. **The final comparison.** A6 (Apple) and B6 (NVIDIA) equal bit for bit.
    Then an inference column, the twenty-prompt harness of `bench/model/` on
    the final weights on NVIDIA, AMD, the local M4 (650 MB of weights fits
    easily) and CPU, logits and generated ids hashed, all equal.
11. **A real model, not only a determinism exhibit.** The loss curve, held-out
    perplexity on shard 013, and one or two public zero-shot evaluations, so
    the result reads as a trained language model. No speed claim is made
    anywhere, no opponent is measured, and nothing is described as a cost of
    determinism.

## 7. What must be built first (owed engineering)

Ranked. None is large; the point is that it is seven things and not one.

- **E1. A segment runner** (`tools/lm_segment.py`). Start from a checkpoint or
  the seed, run S optimizer steps of K shards on N devices of one box with
  `ParallelByteLanguageModelTrainer`, write the hash chain every step, push a
  checkpoint every 250 steps and at the end, resume after any interruption,
  refuse a checkpoint whose recipe differs. `tools/lm_train.py` already has
  `--hash-every`, `--checkpoint-every`, `--resume`, `--record-window` and
  `--lean` for the one-shard trainer (commit `a13c1d228`); the runner lifts
  those onto the K-shard trainer.
- **E2. The learning-rate schedule** as a per-step float32 scalar computed on
  the host and recorded in the schedule. The trainer takes `lr` as a config
  scalar; the state validator must accept it changing between steps.
- **E3. Pre-tokenized FineWeb-Edu in R2.** parquet to text
  (`tools/fineweb_text.py`) to ids with the pinned vocabulary through the
  compiled tokenizer, written as int32 shards with sha256, pushed to R2. About
  10.5 GB for 2.6B tokens. `mojolearn.lm_corpus` reads its corpus whole into
  RAM; a memmap over the token file is a contained change. Tokenizer
  throughput is unmeasured; T0 measures it on one shard.
- **E4. A long lease.** Every runner refuses a lease over 60 minutes by name,
  and rightly for verification legs. A segment is 6 to 40 hours. Either chain
  60-minute legs through R2 checkpoints (about 3 minutes of setup plus 20 s of
  save and restore per hour, and a fresh pod-availability roll every hour), or
  add a segment lease with an on-box dead-man at the segment's budget plus 20
  percent and a dollar cap. The second is recommended; the checkpoint cadence
  bounds the loss from a dead pod to 250 steps.
- **E5. The chained fold in `cross_vendor`.** A worker that folds from a
  received prefix and ships one gradient, the coordinator co-located on the
  first folder's box, and the 162M payload sizes. Bandwidth measured in T1c.
- **E6. A qualified cloud Mac.** The local M4 has 16 GiB unified memory and
  batch 1 alone measured 16.95 GB device peak on the H100, so the Apple segment
  cannot run here, and it should not, since this Mac is the one scarce column.
  Rent bare metal (hosted macOS VMs compile zero Metal kernels and can succeed
  silently with none embedded, measured 2026-09-13), at least 48 GB unified,
  prove it by compiling one binding and running one step at the target shape,
  and record tokens/s and peak memory. Price the 24-hour minimum before
  committing. This is the only unpriced line in section 8.
- **E7. AMD and Apple on the repaired attention path.** The exact masked-tail
  replay that removed the fallback was enabled and endured on NVIDIA; AMD
  matched it cell by cell at the kernel level
  (`native_cross_vendor/verdict.log`), but a 2,000-step batch-4 endurance run
  on AMD has not been taken. T1b takes it.

## 8. Cost and wall clock, from measured cells

Measured. H100 batch 4 step 0.66 s, 12,390 tokens/s, flat over 2,000 steps
after the repair. MI325X batch 1 repaired step 0.550 s against the H100's
0.197 s, 2.8x, taken 2026-09-18 BEFORE the 2026-09-20 AMD projection GEMM
tuning that cut those GEMMs 33 to 45 percent
(`bench/evidence/2026-09-20_gpt3_small_amd_gemm_dispatch.md`); GEMM was 81
percent of the AMD step, so the AMD full step is now somewhere between 2.0x
and 2.8x the H100's and is unmeasured at batch 4. Apple is unmeasured at this
shape. Prices on record: RunPod H100 SXM $2.69 to $3.49 an hour, DigitalOcean
MI325X $3.80, RunPod MI300X about $2.50 when available, RunPod CPU $0.24.

Derived from those. One optimizer step of 64 shards is 42 s on one H100 and 84
to 118 s on one MI300X or MI325X. The per-segment constraint fixes each vendor's
share whatever the arrangement.

| | optimizer steps | GPU hours | cost |
|---|---:|---:|---:|
| NVIDIA, both routes, segments 1 to 6 less its share of 3 | 4,200 | 49 | $130 to $170 |
| AMD, both routes, segments 1 to 5 less its share of 3 | 4,100 | 96 to 134 | $240 to $510 |
| segment 3, both routes, both boxes | 800 | about 24 box hours | $60 to $80 |
| Apple, route A segment 6, 52M tokens | 100 | unmeasured, a day at 1,500 to 2,500 tokens/s | 24-hour minimum, unpriced |
| CPU witnesses, arrival replays, negative controls | | | about $50 |
| test runs T1 and T2 (section 9) | | | about $120 to $180 |
| **total** | **10,000** | **about 200** | **about $600 to $1,000 plus the Mac** |

Wall clock is route A's critical path. At one GPU per segment about 90 to 110
hours, four to five days. NVIDIA and AMD segments on 4-GPU boxes bring that to
about two days, because device count divides wall clock and never changes bits.
Ordering the AMD segments onto 8x MI300X boxes when RunPod has them is the
cheapest lever on wall clock. Two GPU-hours for every one in the table is the
price of the second route, and it is what turns a sampled claim into a
per-step one.

## 9. Test runs, cheapest first, each one gating the next

**T0, local, no rental, hours.** The whole protocol at the 2-block d32 v512
recipe `tools/par_lm_xvendor.py` already uses. Six segments, two routes, the
B-from-A pipeline, the chained fold with two worker processes on the M4 and
CPU, boundary checkpoints through R2, arrival replays, per-step hash chain,
every negative control, and the final comparison tool. Also the tokenizer
throughput on one FineWeb-Edu shard. This proves the harness and the evidence
tooling before any box is rented. Shrunken cells are blind to shape defects,
so T0 proves the protocol and nothing about the model.

**T1, rented, four one-hour legs, about $20.** At the target shape.
(a) H100, K=64 batch 4, three optimizer steps, per-step hashing cost,
checkpoint push to R2 timed. (b) MI300X or MI325X, the same, plus a 2,000-step
batch-4 endurance to confirm the repaired attention holds on AMD (E7).
(c) H100 plus MI300X through the chained fold, three optimizer steps, bytes on
the wire and seconds per step measured. (d) 1 versus 4 GPUs on one box at
K=64, bits equal, time. Every arm rehearsed on CPU at the small shape first.

**T2, rented, the dress rehearsal, four to six hours, about $100 to $150.**
The full protocol at the target shape with a 60-step budget, ten steps per
segment, both routes, the multi-vendor segment, R2 checkpoints, arrival
replays, the CPU witness, the negative controls at the real shape, the final
bitwise comparison and the inference column. Apple's segment runs on the
qualified cloud Mac if E6 is done, otherwise T2 records it as pending and the
Mac is qualified on its own. T2 is the run to show a reviewer; it is the real
run in miniature with every evidence artifact present.

**T3, the run.** Section 3 as written. No box is rented for T3 until T2 has
produced every artifact in section 6 and every negative control has been seen
to fail.

## 10. What this does not prove

One shape, one architecture, one optimizer without clipping, one sequence
length, one data order, one vocabulary. Apple covers one segment and the final
inference, not the run. Nothing here is a speed claim, no opponent is measured,
and no ratio between our own modes is a cost of determinism. It is a statement
about the training trajectory of one model across three vendors and two
routes, and that is all it should ever be written as.
