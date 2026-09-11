# Prompt: neural training performance, IDENTICAL only, NVIDIA vs the cached opponent

You are the orchestrator for one lane of mojolearn: make the IDENTICAL
byte LM training step faster on NVIDIA without changing one output bit.
Work in the main thread. Read this file in full, then read the files it
names before you touch code. Do not spawn subagents unless Andrew asks.

## 0. The rules that bind this lane (do not relitigate them)

- IDENTICAL only. Neural ships one numeric mode. Every change must keep
  every chain's terms and their order; only what is recomputed, which
  thread holds which chain, how operands reach shared memory, and what is
  staged or cached may change. No tensor cores, no TF32, no reassociation.
  The arithmetic boundary is docs/lanes/HANDOFF_speed_gemm_2026-09-10.md.
- The only number that counts is OUR IDENTICAL step against the
  opponent's FAST arm on the SAME GPU, driver, version, shape and corpus.
  Opponents are measured ONCE and cached in bench/OPPONENT_REFERENCE.md.
  Rounds afterward run ours alone. Never say "we are faster"; report the
  ratio and its provenance.
- Two ORDINARY corpora, different in kind, before a number is a result
  (ENGINEERING_RULES.md section 9): English text
  training/corpus/tinyshakespeare and source code
  training/corpus/cpython312_lib (fetched by
  tools/fetch_corpus_cpython312_lib.sh, pinned by manifest and sha256).
  Real activations from those corpora are the only timing input for a
  kernel microbenchmark; hashed and heavy-tailed fixtures are for bits
  and reach only.
- The shape is fixed: 12 layers, DM768, FF2048, V50257, L2048, batch 1,
  12 heads of head_dim 64 (`TARGET_SHAPE` in tools/lm_step_memory_probe.py).
  Changing the shape is not a same-workload improvement. Do not measure
  at a small shape and call it the target.
- A switch flips in the SAME session as its bit-identical measured win,
  and only then. A win on one corpus is not a win. A win on the M4 is not
  a win; the M4 is for smoke and bit checks.
- Real compute on rented GPUs. On the Mac, one light thing at a time under
  `nice -n 19`. Never run a target-shape step on the Mac.
- Before EVERY rental, run every command line the leg script will run, on
  the Mac, at the control shape (`CONTROL_SHAPE` in the probe). On Sep 11
  two H100 hours were burned on a probe option that had only been unit
  tested; the third leg ran after the full local path passed. Smoke the
  harness, not a piece of the new code.
- Rentals: `sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent
  --minutes 60 --gpu "NVIDIA H100 80GB HBM3" --local-card
  bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card`
  with `MOJOLEARN_GPU_ARCHS=sm_90a`, `MOJOLEARN_GEMM_LEG_EXTRA=<your sh>`
  and `MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-<lane>`.
  The tool refuses a dirty tree, so launch from `git worktree add --detach`.
  Output comes home under `<leg>/remote/`. The key is
  `~/.mojolearn_runpod_key` (0600, never exported). The lease is one hour
  and self-terminates. Pods named `samba-train-*` or `mojolearn-trees-*`
  belong to other sessions; never touch them. Delete `remote/tools-venv`
  before committing evidence; no blobs in bench/results.
- Git: explicit-path commits only, never `git add -A`; merge, never
  rebase, when the push is non-fast-forward (other lanes share main);
  never rewrite history; report commits as `%h parent %p`.
- Never call a mechanism located before the experiment that isolates it.
  Run candidate arms separately. Prove reach by sabotage (a one-ulp flip
  in the candidate path must move the output; restore must land on the
  baseline bits).
- Writing: American English, no em-dashes, no HANDOFF files or RUN OWED
  essays unless Andrew asks; commit and push instead. One-line factual
  corrections to stale docs are always allowed.
- Mojo traps: `ref` is a reserved word; `&+` is bitwise AND; a buffer is
  freed at its last use, hold it past synchronize; `String` is not
  indexable.

## 1. Where the step's time goes (H100, measured Sep 11, main 3b81dc2e)

Evidence: bench/results/e1g/2026-09-11_004220-nvidia/remote/lm-step-memory/target-lean-timing/result.json.
Untimed lean step median 0.563 s (3,637 tokens per second). The timed
breakdown (every tick waits, so parts are a breakdown, not a price):

| phase | ms | share |
|---|---:|---:|
| native call | 564 | 100% |
| bwd.attention (fused zdot, dq, dkdv, regime scans, flag) | 262.5 | 46.5% |
| attn.core forward (fused forward, regime scans, flag) | 81.3 | 14.4% |
| head forward + backward (2048 x 768 x 50257 GEMMs) | 46.3 | 8.2% |
| bwd.after_attention (o_proj, qkv backward GEMMs) | 41.0 | 7.3% |
| block.mlp_and_residuals | 26.6 | 4.7% |
| attn.qkv_proj | 10.1 | 1.8% |
| norm1, o_proj, rope, embedding, scans, optimizer, CE, pack, upload | under 4 each | |

Attention is 61 percent of the step at 1.9 and 1.3 TFLOP/s executed,
against 11 TFLOP/s for the IDENTICAL GEMM on the same board and 67
nominal. The gap is latency structure, not arithmetic: four query rows
per 256-thread block in the backward, a serial per-row fold on four
threads while 252 wait, one exposed staging round trip per 32 keys, four
barriers per 32 keys, and in the forward the same Q tile re-staged 396
times per block. The full reading is docs/lanes/BRIEF_attention_step_2026-09-11.md
section 3; read it before proposing anything in attention.

The GEMM side: 255 registers per thread, 4,144 bytes of stack, 44 bytes
of spills, one block per SM, 12.5 percent theoretical occupancy on the
128 x 128 staged configuration. The scalar-load trial was NEGATIVE; do not
repeat it without a new mechanism (HANDOFF_speed_gemm_2026-09-10.md).

## 2. What is already in flight (read first, do not redo)

DEVIATIONs 2525 to 2527 (brief above) ship three opt-in attention arms
behind `-D MOJOLEARN_ATTN_ARM_TRIAL=1` and `MOJOLEARN_ATTN_ARM`:
`bwd_stash` (materialize y, dy, dcell once instead of six score
recomputes), `fwd_sstash` (keep the score and exp between the three
forward passes), `bwd_stash_tiled` (64 x 64 register-blocked folds over
the stash), composed as `stash` and `stash_tiled`. The harness is
bench/attention_step_price_main.mojo, the leg script
tools/attention_step_leg.sh, the arms check
transformer/checks/transformer_attention_arms_check.mojo (15 cases x 3
arms, bit-equal to the eager oracle on the M4 and on the H100).

The H100 leg that prices them on real activations from both corpora is
bench/results/e1g/2026-09-11_113013-nvidia-h100-attention-step. Its
gates are in the brief, section 6. START by reading that leg's
`remote/attention-step/` (status.tsv, price_verdicts.txt,
price_tables.txt, timers_summary.tsv, lm_summary.tsv with
`witnesses_equal_baseline`). If `stash_tiled` is bit-equal on both
corpora and faster at the target, flip `ATTN_ARM_DEFAULT` in
transformer/impl/llama/fused_attention.mojo, make the arm compile in the
shipped build, rerun the fused check, and record the before and after in
the brief. If it lost, say where (the timers tell you which kernel) and
move to section 3. An incidental M4 reading at the control shape, not a
gate: stash_tiled was bit-equal and the lean step went from 3.8 s to
2.7 s.

## 3. Ideas to try, in the order I would run them

Each idea names the mechanism, the expected gain from the counted model,
and its gate. One arm per leg run; alternate old and new in one process;
seven rounds and two warmups at the target; a 300 second deadline per
process. Every arm is opt-in until it flips.

### 3.1 Measure the torch opponent at the target shape, once (blocking)

bench/OPPONENT_REFERENCE.md lists "torch byte-LM training step time on
H100" as an OWED row. Without it every step number is an internal before
and after. Measure torch 2.4.1+cu124 eager FP32 (the FAST arm the rule
names; also record TF32 and torch.compile as extra columns, labeled
nondeterministic), same architecture (RMSNorm, RoPE, SwiGLU, untied head,
162,147,840 parameters), same shape, same two corpora, same step boundary
(forward, backward, AdamW update, loss on host). One row per corpus.
Provenance: GPU, driver, container, torch version, commit, evidence path.
Do this on the first leg, alongside 3.2, so no later leg needs torch.

### 3.2 Persistent attention scratch (DEVIATION 2529, proposed in the brief)

The stash arms allocate 2 x 201 MB (backward) and 201 MB (forward) per
call and free after; the timers report it as `attn.bwd_scratch_alloc` and
`attn.fwd_scratch_alloc`. If either is 1 ms or more per layer, hold the
scratch in the resident session (the DEVIATION 2514 owned state) and reuse
it across layers and steps. Pure execution plan; zero bit risk; gate is
the timers and the lean step median. Expected: up to 12 x (alloc time)
per direction.

### 3.3 Backward latency structure: more rows per block, fewer barriers

The backward tiles are FOUR rows or keys per 256-thread block
(`fused_rows_per_block(64) == 4`), grids of 6,144 blocks, eight warps
each, and the z fold runs on lane 0 of each row while 252 threads wait.
Order-preserving changes to try, one at a time:

- z fold width. Give one block 64 rows so 64 lanes each fold their own
  row's z chain (keys ascending, unchanged per row) concurrently. The
  per-row order does not change; only how many rows are in flight. This
  attacks the longest counted phase (1,200 dependent cycles per key block
  on 1.6 percent of the block).
- Double-buffered staging. Stage key block j+1 into the second shared
  page while the dots run on block j; the `_step` chains read shared
  memory, so staging order is invisible to them. Removes one exposed
  round trip per 32 keys. Cost: 2 x 8 KB more shared per block.
- Barrier count. Four barriers per 32 keys in dq and dkdv; with the stash
  in place (3.1 of the brief) the y and dy phases are gone, so two of the
  four barriers go with them. Count the remaining ones and remove any that
  only ordered a recompute.
- Stack-resident `q` and `dctx` vectors. Each thread holds 64 floats in a
  `stack_allocation` indexed by comptime `p`; the resource leg must answer
  whether the compiler promoted them to registers. If ptxas shows local
  memory, restructure so the index is provably comptime (unrolled) or split
  the 64 into two 32-lane halves.

Gate for each: arms check bit-equal, sabotage reach, price on both
corpora's real activations, then the lean step on both corpora
bit-equal to the baseline witnesses and faster.

### 3.4 Forward: keep Q in registers, raise the block count

- Q residency. Each thread's four rows x 64 of Q are re-staged 396 times
  per block. Stage the 64 x 64 Q tile once per block into shared memory
  (8 KB) or hold each thread's Q slice in registers across the three
  passes and all key blocks. The chain reads the same flushed values in
  the same order.
- Grid. 12 x 32 = 384 blocks on 132 SMs leaves under three blocks per SM
  in one wave. Try 32 query rows per block (768 blocks) with the same
  per-row chains, or two heads per block. This is a grid choice, not an
  arithmetic one.
- Denominator fold on 64 of 256 threads is serial per row (keys
  ascending); as in 3.3, widen the rows in flight rather than the fold.

Expected from the count: the forward's 800 thread-instructions per
visible cell at 9 percent of issue capacity means staging and barriers
dominate; a 2x on the forward's 81 ms is plausible after `fwd_sstash`
removes two of the three dots.

### 3.5 Launcher round trips: one regime scan per step, deferred readback

Both attention launchers run three (forward) or four (backward)
`device_absmax` scans per layer, each with an allocation, a launch, two
synchronizes and a host readback, then read the corner flag afterward:
about 84 host round trips per step from attention alone, plus whatever the
GEMM and MLP launchers do. The scans exist for numerical admission and
must stay, but their RESULT can be consumed on device or read back once.
Try: one fused scan kernel over all operands of a layer writing to a
device-side flag word; a single readback per step (or per direction)
that checks all flags; keep the character-equal refusal messages. The
timers (`MOJOLEARN_TRANSFORMER_TIMING=1`, `timing attn.<kernel>`) tell
you what the scans cost today; measure before attributing. Bit risk is
zero if the kernels are unchanged; the gate is the refusal fixtures still
refusing with the same message.

### 3.6 The GEMMs: occupancy through live storage, then the head shape

Head forward and backward (46 ms) plus after-attention (41 ms) plus MLP
(27 ms) plus qkv (10 ms) are about 124 ms of GEMM at roughly 11 TFLOP/s.
The handoff's next justified step is to reduce live storage on the
128 x 128 staged configuration enough to change the block-per-SM count
(255 registers, 4,144 bytes of stack): halve the per-thread accumulator
block (a 64 x 128 tile or 8 x 4 accumulators), keep the K-step and the
fold order per output cell, and read the register and spill counts back
from ptxas BEFORE timing. A tile change moves which thread holds which
chain, not the chain. Then the head GEMM specifically: 2048 x 768 x 50257
is K-short and N-huge; a tile that spans more N per block with the same
K order may lift it separately. Gate: BITS MATCH on the GEMM check
including ragged controls, then the step.

### 3.7 Small fusions on the residual path (last)

norm1, o_proj, rope, residual adds and the optimizer are each under 4 ms
but there are many launches per step. Count launches per step first
(the resident session can count them). Fuse only where the fused kernel
applies the same operations in the same order per element (norm then
residual add, rope then cache write). Expect a few ms; do it only after
3.2 to 3.6 have moved the big numbers.

### 3.8 Not this lane

- Shape, batch, sequence packing, mixed precision, TF32, tensor cores,
  reassociated softmax, flash-style online normalization with a different
  fold order. Each changes bits or the workload.
- Mamba-2/3 chunked scan and the transformer admission failures are
  separate lanes; note them if you trip over them, do not start them.
- Apple and AMD timing. After an NVIDIA flip, the fused check must still
  pass on the M4 (it caught two 32 KB shared-memory bugs on Sep 9) and
  Apple and AMD timing is owed, but speed is decided on NVIDIA.

## 4. Deliverables per round

1. The opponent row (3.1) in bench/OPPONENT_REFERENCE.md with provenance,
   once.
2. For each arm: bit verdicts on both corpora, reach by sabotage, the
   price table at the target, the per-kernel timers, the lean step median
   and witnesses on both corpora, the decision (flipped, negative,
   neutral) with the number that decided it, in
   docs/lanes/BRIEF_attention_step_2026-09-11.md (attention) or a new
   BRIEF_<lane>_<date>.md.
3. Evidence under bench/results/e1g/<leg>/ with `remote/tools-venv`
   deleted; commit with explicit paths; push; merge on non-fast-forward.
4. Every flip lands the same session with the fused check rerun. Every
   negative result is recorded with its DEVIATION number so nobody repeats
   it.
5. At the end, one paragraph to Andrew: step median before and after on
   both corpora, the ratio against the cached torch row, what flipped,
   what lost, and what is still owed. No essay.
