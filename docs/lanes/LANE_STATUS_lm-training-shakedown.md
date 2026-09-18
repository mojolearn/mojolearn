# Lane status: lm-training-shakedown

Branch `lane/lm-training-shakedown`. Written 2026-09-17 for a session with NO
context. Two H100 legs are complete and both are committed; no pod is out.

## THE VERDICT: NOT READY

Andrew asked whether a GPT-3 Small shape (batch 1, L2048, d_model 768, 12
layers, 12 heads, head_dim 64, intermediate 2048, vocab 50257, **162,147,840
parameters**) can be trained under the IDENTICAL contract for 10B or 25B
tokens. The costing was fine. The stack is not.

**Lead with these four numbers.**

1. **Only batch 1 survives a long run.** Batch 4 OOMs at **step 211**. The
   three-step batch sweep that made batch 4 look like the operating point is a
   measurement of the first 210 steps and nothing else.
2. **A long run at batch 1 costs 2.1x after about step 480**, and that is
   replicated on two different physical H100s. 0.207 s a step becomes 0.44 s.
3. **Tearing the resident session down and rebuilding it does NOT fix it.**
   Median step with a rebuild every 250 steps: 0.44032 s. Without: 0.44054 s.
   The mitigation is dead and it costs 26 s per rebuild on top.
4. **The data pipeline does not exist**, and the vocabulary at the front of it
   is a decision nobody has made.

Recosted at the only rate proven to survive (batch 1, phase 3, 4,649 tokens/s):
**FineWeb-Edu 10BT is 597.5 H100-hours and 25B tokens is 1,493.7**, roughly
double the 315 h / 789 h the request was costed with. The money is still not
the risk.

## What was measured, question by question

### Q1. What batch fits, and what does throughput do?

H100 80GB HBM3, commit 025910137, enwik8 from R2, resident + lean, THREE steps
a cell:

| batch | tokens/step | s/step | tokens/s | device peak |
|---|---|---|---|---|
| 1 | 2,048 | 0.20708 | 9,890.0 | 16.95 GB |
| 2 | 4,096 | 0.37504 | 10,921.6 | 26.61 GB |
| 4 | 8,192 | 0.65732 | 12,462.7 | 45.94 GB |
| 8 | 16,384 | 1.31258 | 12,482.3 | 84.33 GB |

All four FIT in three steps. Memory is linear: `device_GB = 7.367 + 9.624 *
batch`, which admits batch 8 (98.6% of the card) and refuses batch 9.
Throughput saturates at batch 4.

**`bench/results/lm_capacity_2026-09-10/b8-l2048.json` is wrong by 2.06x** and
is corrected by `SUPERSEDED.md` beside it: its 161.75 GiB and `fit_admitted:
false` were arithmetic, never an OOM observation.

**THEN LEG 2 KILLED THE CONCLUSION.** A 700-step batch-4 run died with
`CUDA_ERROR_OUT_OF_MEMORY` inside `byte_lm_session_step` after exactly **210
completed steps**, at 71.49 GB in use. That is the same step index at which
batch 1's memory starts climbing. So the batch sweep is usable for nothing
longer than 210 steps, and **batch 1 is the only batch proven to survive**.
Batch 2 would need about 54 GB after the transition and is UNTESTED.

### Q2. Does a long run survive?

**2,000 consecutive steps at batch 1 completed** (840 s). That is 15.6x the
longest run anything in this repository had done at any size and 500x the
longest at this shape; the previous record at 162M was FOUR steps.

Good news: no crash, no NaN or inf in 2,000 steps, loss 10.903326 to a minimum
of 0.605758 with 200 distinct values in the last 200 steps, host `ru_maxrss`
identical to the byte at step 1 and step 2,000, the slowest step only 1.177x
the median and ZERO steps above 1.5x it (so no stalls, no hitches).

Bad news, in three phases:

  * steps 1 to ~210: 0.207 s, 16.949 GB;
  * steps ~210 to ~480: memory 16.949 -> 34.397 GB, time 0.207 -> 0.44 s;
  * from ~480: BOTH PLATEAU and hold through step 2,000.

It is a one-time transition to a second steady state costing **2.03x memory
and 2.13x time**, not a leak and not a runaway. Drift across the run is
+100.63% (first 199 timed steps median 0.20686, last 199 median 0.41503).

**Ruled out, with counters rather than assertions**: clocks (1980 of 1980 MHz),
thermal (42 C, no slowdown active), power (356 W of 700, no cap active), host
CPU contention (`/proc/pressure/cpu` `full avg10=0.00` against a 22.1-CPU
cgroup quota on a 208-core host) and host memory. Evidence:
`drift_diagnostics.txt` in both leg directories.

**Replicated.** Leg 2's straight arm on a DIFFERENT physical H100: 1,500 steps,
median 0.440543 s, fastest decile 0.206852. Leg 1: median 0.440630, fastest
0.206500.

**The session-recycle mitigation FAILED.** Rebuilding the resident session from
`export_state()` every 250 steps gave median 0.440323 s over 1,808 steps,
indistinguishable from the straight arm, plus 26.03 s median per rebuild
(182.9 s over 7). So the cause survives a full session teardown and rebuild
inside the same process, which is itself a clue.

**I DID NOT run 10,000 consecutive steps.** At the post-transition rate they do
not fit in a 60-minute lease.

### Q3. Is checkpoint/resume bit-exact at 162M?

**YES.** Every witness of every resumed step matched the uninterrupted run in a
fresh process: loss, flat gradients, parameters, m, v and flags, by sha256,
printed by name. Save 11.19 s, restore 7.63 s.

The missing-moments control separated, where the arithmetic requires: the first
resumed step's loss and gradient MATCH (proving the parameter half of the
restore) while m, v and the updated parameters differ, and by the last step the
loss and gradient differ too. All eight checks pass, in
`.../remote/lm-shakedown/resume.log`. A control-shape (20.45M) smoke passed
first.

**But `export_checkpoint` refuses this model.** `_CHECKPOINT_LIMIT` is 2 MiB and
the guard is `n_total * 24 + n_tensors * 8 > limit`, so the largest model the
checkpoint FILE path accepts is **87,381 parameters**; 162,147,840 is 1,855x
over and `from_checkpoint` is unreachable. The arm went through `export_state()`
arrays, which the docstring names as the substitute and which nothing in the
repository implemented until `tools/lm_shakedown_resume.py`.

This is ONE GPU and ONE shape. It does not extend the NVIDIA/AMD cross-vendor
resume result, which is the 34,944-parameter model.

### Q4. Can the loader stream 10B tokens from R2?

**NO.** FineWeb-Edu 10BT IS in R2 (14 pinned parquet shards, 28,518,193,415
bytes, group key `corpus/fineweb-edu-10BT`) and staging is the one part that
works: leg 1 staged enwik8's 100 MB in 7 s. Everything between the shards and
`train_step` is missing.

  * `CorpusBatches` (`tools/lm_step_memory_probe.py:229`) does
    `raw = self.path.read_bytes()` -- one whole file, resident in host RAM.
    There is no mmap, no generator, no chunked token reader anywhere on the LM
    data path, and `train_step(ids)` takes a materialized `int32[B, L+1]` with
    no seam to hook one into.
  * Nothing reads parquet into the trainer. Checked repo-wide: thirteen files
    mention parquet outside `bench/results` and all thirteen are taxi, Criteo or
    gbm-bench fetchers.
  * **There is no tokenizer on the training path at all.** `CorpusBatches` casts
    RAW BYTES to int32 ids. At vocab 50,257 the embedding and lm_head are
    77,194,752 of the 162,147,840 parameters, **47.6% of the model**, and byte
    ids touch rows 0-255 only, so nearly half the model would train on nothing.
  * A real run therefore needs a VOCABULARY, and mojolearn deliberately ships
    none (no-third-party-data rule removed the GPT-2 table). Training our own is
    the option that fits the IDENTICAL contract and
    `python/mojolearn/_bpe_trainer.py` is already deterministic by
    construction. **Nobody has decided this and it gates everything else.**
  * `CorpusBatches` wraps modulo the WHOLE file and ignores the `train_range` /
    `validation_range` / `test_range` enwik8's own manifest declares. Harmless
    at 2,000 steps, silent test-set training at 610,352.

Scoped work, in dependency order: decide the vocabulary; a pre-tokenization
pass (parquet -> text -> `GPT2Tokenizer` -> uint16 shards, 20 GB, one-time,
CPU-only, a RunPod CPU pod is the right box); an `np.memmap` batch reader with
its own corpus manifest schema; and a held-out split that is actually held out.

## The blockers

1. **`completed_steps` is capped at 999,999**, enforced by **fourteen guards
   across five files** (`_byte_lm_impl.py`, `_byte_lm_trainer_host.py`,
   `bindings/_mojolearn_byte_lm.mojo`, `training/byte_lm.mojo`), found by a
   repo-wide grep on the guards' own error text. At batch 1 that is 2.048B
   tokens, full stop.
   **BUT `logical_shards` DOES work at 162M and clears it.** Leg 2 measured
   `ParallelByteLanguageModelTrainer(devices=(0,), logical_shards=K)` at the
   target shape, K = 1, 4, 16, 64, NO refusals:
   | K | tokens per optimizer step | tokens/s |
   |---|---|---|
   | 1 | 2,048 | 9,400.3 |
   | 4 | 8,192 | 9,981.7 |
   | 16 | 32,768 | 10,042.2 |
   | 64 | 131,072 | 10,042.6 |
   Throughput is flat in K (expected: `step.optimizer` is 1.7 ms of a 236 ms
   step). At K=64, 25B tokens is 190,735 optimizer steps, comfortably under the
   cap. **This is the way past blocker 1 and it was the lane's best find.**
   Caveat: three steps per arm, so phase 1 again, and the device peak came back
   `unavailable` in that run.
2. **The phase-2 transition**, cause unknown, 2.1x on time, 2.03x on memory,
   fatal at batch 4, not fixed by a session rebuild.
3. **No data pipeline and no vocabulary decision.**
4. **No checkpoint file above 87,381 parameters** (this lane wrote the
   substitute).

## Pods

**None out.** Both were verified deleted through the API (HTTP 404):
`687sdxsdcqs23u` (leg 1) and `pibjrl033kw6oj` (leg 2). A third refusal,
`2026-09-17_203153-...`, rented nothing.

## Where the evidence lives

  * `bench/results/lm_training_shakedown_2026-09-17/GO_NO_GO.md` — the verdict.
    **Written before leg 2 and now UNDERSTATES the case**; fix it first.
  * `.../SOURCE_BLOCKERS.md` — what the code refuses, with line numbers.
  * `.../MEASURED.md` — leg 1's numbers. Sections 1 and 4 are phase-1 only and
    say so in section 7.
  * `.../DATA_PIPELINE.md` — question 4, fully scoped.
  * `bench/results/e1g/2026-09-17_200504-nvidia-h100-lm-shakedown/` — leg 1
    (batch sweep, both resume arms, the 2,000-step run; big files gzipped).
  * `bench/results/e1g/2026-09-17_203231-nvidia-h100-lm-shakedown-drift/` —
    leg 2 (recycle, straight, batch4-long OOM, shards).
  * `bench/results/lm_capacity_2026-09-10/SUPERSEDED.md` — the correction.

Tools this lane added: `tools/lm_shakedown_resume.py` (162M resume + control),
`tools/lm_recycle_probe.py` (straight vs session-recycle), `tools/lm_shards_probe.py`
(logical shards), and the two leg bodies `tools/lm_shakedown_body.sh` and
`tools/lm_shakedown_long_body.sh`.

## The exact next command

`lane/lm-step-memory-build` has a HYPOTHESIS for blocker 2, stated as one: the
ten `[B,H,L,S]` attention arrays start at one element, grow on demand and are
never released, triggered by `FUSED_CORNER` at `fused_attention.mojo:1908-1910`,
which is data dependent. Their arithmetic lands within ~0.8% of the measured
16.949 -> 34.397 GB step. They built `attention_stage_report()` as a witness so
ONE RUN settles it. **COORDINATE WITH THEM AND DO NOT BOTH CHASE IT.** That
hypothesis also explains why a session rebuild did not help and why the onset is
at the same step index at every batch, both of which this lane measured.

If this lane runs the next leg rather than they do, it is
`tools/lm_shakedown_long_body.sh` with the batch-4 arm replaced by a batch-2
endurance arm (the one untested batch that arithmetic says survives) plus
`attention_stage_report()` around the transition:

```sh
cd ~/mojolearn-wt/lm-training-shakedown
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_shakedown_long_body.sh \
MOJOLEARN_GEMM_LEG_GPU_NVIDIA="NVIDIA H100 80GB HBM3" \
MOJOLEARN_GEMM_LEG_LOCAL_CARD=bench/results/e1g/2026-09-13_221244-nvidia-h100-feature-freq-2710/local/apple.card \
MOJOLEARN_STAGE_KEYS="corpus/enwik8/input.txt" MOJOLEARN_STAGE_STRICT=1 \
MOJOLEARN_GEMM_LEG_WORK_TIMEOUT=3000 \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-lm-shakedown-stage \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --allow-concurrent \
    --minutes 60 --gpu "NVIDIA H100 80GB HBM3"
```

`--allow-concurrent` is needed whenever another lane has a pod up; leg 2 was
refused once for exactly that and rented nothing.

## What contradicted the brief

  * The brief said the `b8-l2048.json` capacity file was "a loose analytic upper
    bound". It is looser than that: **batch 8 fits**, at 84.33 GB against its
    161.75 GiB, and every batch up to 8 runs for three steps.
  * The brief said "run 1 of the step-memory probe saw 21.9 GB host RSS". That
    was the STATELESS path. The resident path this lane used holds 8.3 GB flat
    and does not grow. Not an open risk.
  * The brief quoted 0.2326 s a step from `bench/OPPONENT_REFERENCE.md`. The
    same probe, shape, corpus and arm now measures **0.20708 s**, 11% faster, on
    a newer commit. NO opponent ratio in that file has been restated, because
    their columns were not re-measured and the two cannot be divided across
    pods.
  * The brief said the largest real training run on record was 128 steps. True
    for the 34,944-parameter model; **at the 162M target shape the record was
    FOUR steps** across 413 recorded runs, and every LM run on record at any
    shape was batch 1.
  * The brief's framing that the risk was "nothing has trained long" was right,
    but the failure is not the one expected: it is not instability or a leak.
    The run is numerically healthy for 2,000 steps and simply gets twice as
    slow and twice as hungry, which kills every batch above 1.
  * This lane briefly wrote that there is NO gradient accumulation. That was
    wrong and is corrected in `SOURCE_BLOCKERS.md` blocker D:
    `ParallelByteLanguageModelTrainer` has it, and leg 2 proved it runs at 162M.
