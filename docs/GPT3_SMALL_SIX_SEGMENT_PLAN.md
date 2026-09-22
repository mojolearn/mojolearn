# Plan: a compute-optimal GPT-3 Small shape model, trained in six segments across NVIDIA, AMD and Apple, twice, ending at the same bits

Written 2026-09-22 at Andrew's direction, revised the same day after his
questions (route B's last two segments were both NVIDIA, checkpoint cadence,
what an arrival replay is, why the multi-vendor segment is bandwidth bound,
whether the Mac can be this laptop). Supersedes the 2026-09-17 tri-vendor
plan (deleted in the 2026-09-19 lane cleanup, last revision `da9f7bc52`),
which was costed at 25B tokens and read NOT READY. Three things changed since
then and each one is measured. The eager attention fallback that doubled the
step after step 480 is repaired and a 2,000-step batch-4 run held a flat
12,390 tokens/s and 38,959 MiB on an H100
(`bench/results/lm_attention_fallback_2026-09-18/endurance/verdict.log`).
The run's own vocabulary exists and is pinned in R2
(`vocab/mojolearn-bpe-fineweb-edu-50257-v1`, trained on FineWeb-Edu shard
000, commit `5aedd1c94`). One model was trained live on an Apple M4, an H100
and an MI300X at once through `mojolearn.cross_vendor`, bit-identical after
every step (`bench/results/live_xvendor/2026-09-21/README.md`, commit
`eec83b539`).

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

**There are two runs and only two.** Route A and route B. Everything else in
this file (arrival replays, CPU witnesses, negative controls) is a replay of a
few steps from a checkpoint, never a third training run.

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
section 9 is under $1,000 instead of $6,000 to $8,000.

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
and the finals could not agree. A second constraint, added in the revision,
is that **every one of the five handoffs changes vendor in both routes**, so
each handoff is a real cross-hardware checkpoint continuation and never a
same-box resume. The first draft had route B on NVIDIA for segments 5 and 6.

| segment | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| optimizer steps | 1,000 | 1,000 | 400 | 1,500 | 1,000 | 100 |
| **route A** | NVIDIA | AMD | **NVIDIA + AMD** (NVIDIA folds first) | NVIDIA | AMD | **Apple** |
| **route B** | AMD | NVIDIA | **AMD + NVIDIA** (AMD folds first) | AMD | NVIDIA | AMD |
| differs between routes | yes | yes | yes, ownership and fold order swapped | yes | yes | yes |
| handoff into it changes vendor, A / B | seed | yes / yes | yes / yes | yes / yes | yes / yes | yes / yes |

**Segment 3 is the multi-cluster segment and it is in both routes**, as
Andrew asked. It is the same recipe in both, one NVIDIA box and one AMD box
training the same step together through `mojolearn.cross_vendor`, and it still
satisfies the per-segment constraint because the two routes swap which vendor
owns which shards and which vendor computes the first half of the ordered
fold. Section 5 says how it works at this size.

**Apple appears exactly once, in segment 6 of route A**, the shortest segment
and the last one. Section 7 argues last over first. What is lost is that Apple
agrees with AMD over 100 steps and not over the run. The final inference
column in section 6 puts Apple on the final weights as well, at a cost of
minutes.

Segment lengths are unequal on purpose. Apple's is sized to what a rented Mac
does inside its one-day minimum. The multi-vendor segment is sized to the
bandwidth cost in section 5. The rest are long enough that a defect that
appears after a few hundred steps, like the one the shakedown found, is inside
a segment.

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

## 5. How the bits agree, and why the multi-vendor segment is bandwidth bound

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
workers after every step. It ran for real on three vendors at once.

**Why bandwidth, and what code can and cannot do about it.** Data parallel
training means every replica must hold the same total gradient before it can
take the step, and the total is 4 bytes per parameter, 649 MB at this shape,
in exact float32 that cannot be compressed or rounded. The NVIDIA box and the
AMD box live in different data centers (RunPod or DigitalOcean or Hot Aisle),
so at least one gradient-sized message must cross the public internet in each
direction on every optimizer step, and no arithmetic trick removes it. That is
the bound. What code does control is HOW MANY such messages cross.

- As shipped, the coordinator receives every shard's whole gradient. At K=64
  that is 41 GB per step, fine for the 2-block model it ran on and impossible
  here.
- **The chained fold** cuts it to two messages with no kernel change. The
  fold is a left fold, so the owner of shards 0 to j-1 computes the prefix
  `fold(g0..g_{j-1})` on its own box and ships that ONE gradient to the owner
  of shards j to 63, who continues `fold(prefix, g_j, ..., g_63)` and ships the
  total back. Bit for bit the flat fold a single GPU computes. The second
  owner computes its shard gradients while the first is working and holds
  them on device (20 shards is 13 GB, fine on an MI300X). This is item E5.
- **Co-locating both vendors** turns the internet into a data-center link.
  RunPod lists both H100 and MI300X; if both pods land in one region the
  transfer falls from tens of seconds to a few, and the overhead nearly
  disappears. Worth trying first in T1c, not something to rely on, since
  regions and availability are not chosen by us.
- Bigger steps (a larger K) would amortize the transfer, but K is the recipe
  for the whole run, so it cannot be raised for one segment.

Derived cost of one multi-vendor step with the chained fold, to be measured in
T1c. AMD computes its 20 shards (30 to 37 s) while NVIDIA computes its 44
(29 s), then the prefix crosses (649 MB, 7 to 13 s at 50 to 100 MB/s), AMD
finishes the fold on device, and the total crosses back (7 to 13 s). About 45
to 65 s per step against 37 s of pure compute, a 20 to 75 percent overhead on
one segment of 400 steps, 5 to 7 hours, about 520 GB moved. Not a blocker; a
line item.

**AMD cross-device bytes stay host-staged.** Any multi-GPU AMD box moves bytes
between devices only through `transfer_bytes`, after the SR-IOV stale-read
finding on 2x MI300X.

## 6. Evidence, and there is a lot of it by construction

The point of the design is that evidence is produced by the training itself,
every step, on every box, not by a separate campaign afterward. Two different
things are recorded and they should not be confused. A **hash** is 64 hex
characters describing the whole state and is written every step. A
**checkpoint** is the whole state, about 1.95 GB at this shape (parameters,
m, v, flags at 4 bytes each), and is written on a cadence.

1. **A per-step hash chain on every box.** After every optimizer step the box
   records the sha256 of the complete state, the sha256 of the summed
   gradient, every shard's loss, the learning rate used, the token offsets of
   every shard, and the previous line's hash, so the log is a chain. Two
   routes times 5,000 steps is **10,000 full-state hashes**, and every B hash
   must equal the A hash of the same step. That is 5,000 cross-hardware
   agreements, not five. Cost, derived: reading 1.95 GB back and hashing it is
   a few seconds against a 42 s NVIDIA step and an 84 s or longer AMD step,
   under 10 percent. `export_state` was measured at about 16 s of per-element
   Python at this shape on 2026-09-17, so the hash must go through the binary
   stream export, not the array export; T1a measures the real cost, and if it
   is over 10 percent the cadence drops to every 10 steps with every step
   inside the record windows.
2. **Checkpoints every 100 steps**, plus one exactly two steps before every
   segment boundary and one at the boundary, pushed to R2 as they are written
   with size and sha256 in a manifest. Every step is overkill and it is not
   close. A checkpoint is 1.95 GB, save 11 s and a push of 10 to 20 s, so
   every step would cost 50 to 75 percent of the run and 5,000 x 2 x 1.95 GB
   is 19.5 TB. Every 100 steps is about 110 checkpoints, 215 GB, under 1
   percent of the run, a few dollars a month in R2, and a dead pod loses at
   most 100 steps (70 minutes on NVIDIA, about 3 hours on AMD). Nothing lives
   only on a pod.
3. **Arrival replay at every handoff.** Plainly. When a box receives the
   checkpoint that ends segment k, it does not start segment k+1 on trust. It
   verifies the file's sha256 against the manifest, loads it, hashes the
   state and matches the sender's final hash line. Then it fetches the
   checkpoint from two steps before the boundary, runs those two steps ON ITS
   OWN HARDWARE, and must land on the boundary hash bit for bit. Only then
   does it continue. So the last two steps of every segment are computed by
   the sender's hardware, by the receiver's hardware, and by the other route's
   hardware, before anything depends on them, and a divergence is caught at
   the boundary it belongs to, in two steps, instead of surfacing at the end.
   It is the bidirectional resume check of the 34,944-parameter campaign,
   made automatic and applied at all five handoffs of both routes.
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
10. **The final comparison.** A6 (Apple) and B6 (AMD) equal bit for bit.
    Then an inference column, the twenty-prompt harness of `bench/model/` on
    the final weights on NVIDIA, AMD, the local M4 (650 MB of weights fits
    easily) and CPU, logits and generated ids hashed, all equal.
11. **A real model, not only a determinism exhibit.** The loss curve, held-out
    perplexity on shard 013, and one or two public zero-shot evaluations, so
    the result reads as a trained language model. No speed claim is made
    anywhere, no opponent is measured, and nothing is described as a cost of
    determinism.

## 7. Risk, and why Apple goes last

Andrew ranks Apple the highest risk and the multi-vendor segment the next.
Agreed, and the order of the segments is chosen so that either failure still
leaves a complete result.

**Apple.** The risks are memory, Metal ahead-of-time compilation on a rented
machine, unmeasured throughput, and the one-day minimum. None of them is
discovered by running the segment first; all of them are discovered by the
qualification in E6, which is one day of a rented Mac spent compiling one
binding and running a few steps at the target shape, before the run exists.
Running Apple first would put the slowest and least known box at the head of
route A, where route B's segment 2 waits for it, and would spend the Mac's day
before anything else has been proven. Running it last puts the Mac on nobody's
critical path, and **if the Apple segment fails, the fallback is to run route
A's segment 6 on NVIDIA**, which keeps every constraint in the table (AMD in
route B, NVIDIA in route A, a vendor change at the handoff) and yields a
complete two-vendor result with Apple on the final inference column only. So
last is safe and first is not better.

**Can the Apple leg be this laptop overnight? No.** This M4 has 16 GiB unified
memory. Batch 1 alone at this shape measured 16.95 GB of device memory on the
H100 (7.4 GB floor plus 9.6 GB per sequence), and the recipe's shard is batch
4, so the working set is about 39 GB. Cutting the shard to batch 1 would change
K and therefore the bits of the whole run, and still would not fit. Activation
recomputation would cut the per-sequence memory and would be bit-exact by
construction (the same kernels recompute the same forward), but it is Mojo
work in the modeling stack with its own qualification, and even then this GPU
is about a fifteenth of an H100, so 100 steps is a day or two of the laptop
during which no other Metal work can run. The scarce column stays scarce.
Recomputation is worth a lane later, for its own reasons, not for this run.

**Why the cloud Mac costs a day.** Apple's macOS software license requires a
leased Mac in a data center to be leased for a minimum of 24 hours, so every
provider (AWS EC2 Mac dedicated hosts, MacStadium, Scaleway) bills at least a
day. It is Apple's rule, not the provider's, and it cannot be avoided. The
right response is to make the day count. Qualify, then run the 100-step
segment, then the final inference column, in the same day if the timing allows,
and pick the largest GPU the budget permits so the segment is hours and not
the day. Prices below are from memory and are to be checked before renting.

**Multi-vendor.** The risk is engineering (the chained fold) and connectivity
between two providers, both retired by T1c for about $10 before the run
exists. **If the live segment fails during the run, the fallback is to
complete segment 3 on one vendor per route** (NVIDIA in A, AMD in B), which
keeps the per-segment constraint and loses only the live multi-vendor claim
for that segment. The run still completes.

## 8. What must be built first (owed engineering)

Ranked. None is large; the point is that it is seven things and not one.
E1 to E5 are Python-side and are proven locally in T0 before any rental.

- **E1. A segment runner** (`tools/lm_segment.py`). Start from a checkpoint or
  the seed, run S optimizer steps of K shards on N devices of one box with
  `ParallelByteLanguageModelTrainer`, write the hash chain every step, push a
  checkpoint every 100 steps, two before the boundary and at the end, resume
  after any interruption, refuse a checkpoint whose recipe differs, and do the
  arrival replay before continuing. `tools/lm_train.py` already has
  `--hash-every`, `--checkpoint-every`, `--resume`, `--record-window` and
  `--lean` for the one-shard trainer (commit `a13c1d228`); the runner lifts
  those onto the K-shard trainer.
- **E2. The learning-rate schedule** as a per-step float32 scalar computed on
  the host and recorded in the schedule. The trainer takes `lr` as a config
  scalar; the state validator must accept it changing between steps.
- **E3. Pre-tokenized FineWeb-Edu in R2.** The vocabulary IS in R2 already
  (`vocab/mojolearn-bpe-fineweb-edu-50257-v1`, ranks and tokenizer.json,
  pinned in the manifest). The parquet corpus IS in R2 (14 shards). What is
  NOT there is the token stream: only enwik8 has one
  (`corpus/enwik8/tokens/...`). E3 is parquet to text (`tools/fineweb_text.py`)
  to ids with the pinned vocabulary through the compiled tokenizer, written as
  int32 shards with sha256, pushed to R2. About 10.5 GB for 2.6B tokens.
  `mojolearn.lm_corpus` reads its corpus whole into RAM; a memmap over the
  token file is a contained change. Tokenizer throughput is unmeasured; T0
  measures it on one shard, and a RunPod CPU pod does the rest in parallel.
- **E4. A long lease.** Every runner refuses a lease over 60 minutes by name,
  and rightly for verification legs. A segment is 6 to 40 hours. Either chain
  60-minute legs through R2 checkpoints (about 3 minutes of setup plus 20 s of
  save and restore per hour, and a fresh pod-availability roll every hour), or
  add a segment lease with an on-box dead-man at the segment's budget plus 20
  percent and a dollar cap. The second is recommended; the 100-step checkpoint
  cadence bounds the loss from a dead pod.
- **E5. The chained fold in `cross_vendor`.** A worker that folds from a
  received prefix and ships one gradient, the coordinator co-located on the
  first folder's box, and the 162M payload sizes. Bandwidth measured in T1c.
- **E6. A qualified cloud Mac.** Rent bare metal (hosted macOS VMs compile
  zero Metal kernels and can succeed silently with none embedded, measured
  2026-09-13), at least 48 GB unified so a 39 GB working set is not fighting
  the OS, prove it by compiling one binding and running a few steps at the
  target shape, and record tokens/s and peak memory. Candidates, prices from
  memory to be checked: AWS EC2 Mac dedicated hosts (M2 Pro 32 GB is too
  small; M1 Ultra 128 GB with the 64-core GPU is the fastest listed and about
  $60 a day), MacStadium and Scaleway M4 Pro 64 GB hosts. AWS is bare metal by
  construction and is the first to try.
- **E7. AMD on the repaired attention path, endured.** The exact masked-tail
  replay that removed the fallback was enabled and endured on NVIDIA; AMD
  matched it cell by cell at the kernel level
  (`native_cross_vendor/verdict.log`), but a 2,000-step batch-4 endurance run
  on AMD has not been taken. T1b takes it.

## 9. Cost and wall clock, from measured cells

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
share whatever the arrangement; the revision moved 100 steps from NVIDIA to
AMD.

| | optimizer steps | GPU hours | cost |
|---|---:|---:|---:|
| NVIDIA, both routes, less its share of segment 3 | 4,100 | 48 | $130 to $170 |
| AMD, both routes, less its share of segment 3 | 4,200 | 98 to 138 | $250 to $520 |
| segment 3, both routes, both boxes | 800 | about 24 box hours | $60 to $80 |
| Apple, route A segment 6, 52M tokens | 100 | unmeasured, hours on an M1 Ultra | one day, about $60 |
| CPU witnesses, arrival replays, negative controls | | | about $50 |
| test runs T1 and T2 (section 10) | | | about $120 to $180 |
| **total** | **10,000** | **about 200** | **about $650 to $1,050** |

Wall clock is route A's critical path. At one GPU per segment about 90 to 110
hours, four to five days. NVIDIA and AMD segments on 4-GPU boxes bring that to
about two days, because device count divides wall clock and never changes bits.
Ordering the AMD segments onto 8x MI300X boxes when RunPod has them is the
cheapest lever on wall clock. Two GPU-hours for every one in the table is the
price of the second route, and it is what turns a sampled claim into a
per-step one.

## 10. Test runs, cheapest first, each one gating the next

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
the wire and seconds per step measured, both pods on RunPod in one region if
it offers that. (d) 1 versus 4 GPUs on one box at K=64, bits equal, time.
Every arm rehearsed on CPU at the small shape first.

**T-Mac, one rented day, about $60.** E6. Compile, a few steps at the target
shape, tokens/s, peak memory. Scheduled so that the same day can also host the
real segment 6 if T2 has passed by then; otherwise it is a qualification day
and the segment buys a second.

**T2, rented, the dress rehearsal, four to six hours, about $100 to $150.**
The full protocol at the target shape with a 60-step budget, ten steps per
segment, both routes, the multi-vendor segment, R2 checkpoints, arrival
replays, the CPU witness, the negative controls at the real shape, the final
bitwise comparison and the inference column. Apple's segment runs on the
qualified cloud Mac if T-Mac is done, otherwise T2 records it as pending. T2
is the run to show a reviewer; it is the real run in miniature with every
evidence artifact present.

**T3, the run.** Section 3 as written. No box is rented for T3 until T2 has
produced every artifact in section 6 and every negative control has been seen
to fail.

## 11. What this does not prove

One shape, one architecture, one optimizer without clipping, one sequence
length, one data order, one vocabulary. Apple covers one segment and the final
inference, not the run. Nothing here is a speed claim, no opponent is measured,
and no ratio between our own modes is a cost of determinism. It is a statement
about the training trajectory of one model across three vendors and two
routes, and that is all it should ever be written as.
