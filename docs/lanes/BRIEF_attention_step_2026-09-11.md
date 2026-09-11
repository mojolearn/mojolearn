# Attention step lane, the fused attention at the LM target shape (DEVIATIONS 2525 to 2527)

Source-only lane, September 11, 2026, IDENTICAL only. Nothing here ran on
the Mac or on a GPU; every number below that is not copied from the
evidence path is derived from the source by counting, and says so. The
shipped default is untouched. Every arm is opt-in behind a build define
and an environment read. UPDATE 2026-09-11 11:44Z: the H100 leg ran, every
gate in section 6 held on both corpora, and `ATTN_ARM_DEFAULT` is now
`stash_tiled` (section 10); the shipped build compiles that arm's clean
kernels and runs them at head_dim 64.

Parent, [HANDOFF_ai_classical_identical_next_2026-09-10.md](HANDOFF_ai_classical_identical_next_2026-09-10.md)
section 3 ("Attention. Profile backward and repeated operand traffic;
reuse/fuse work only with identical arithmetic and reachable fallback
checks. Measure forward and forward+backward separately and then the full
training step."). Arithmetic boundary,
[HANDOFF_speed_gemm_2026-09-10.md](HANDOFF_speed_gemm_2026-09-10.md)
(IDENTICAL FP32, no tensor cores, no TF32, the RN-FMA then FTZ-multiply
seam kept, leaf boundaries and fold topology kept). The step this lane
attacks, [DESIGN_lm_device_owned_step_2026-09-11.md](DESIGN_lm_device_owned_step_2026-09-11.md)
and the tail of [BRIEF_lm_step_memory_2026-09-10.md](BRIEF_lm_step_memory_2026-09-10.md)
("Gate G5 result, PASSED; the default flipped").

## 1. Purpose

The lean IDENTICAL training step at the target shape (12 layers, DM768,
FF2048, V50257, L2048, batch 1, 12 heads of head_dim 64, 12 kv heads; the
shape is `TARGET_SHAPE` in `tools/lm_step_memory_probe.py` and the head
geometry is `LlamaDims(768, 12, 12, 64, 2048)` in `training/byte_lm.mojo`)
is 564 ms of native call on an H100, and 344 ms of it is the fused
attention core (forward 81 ms, backward 262 ms). This lane reads the four
fused kernels, names where their time goes, and ships three opt-in arms
that remove recomputation and re-staging without changing the operations
any output element sees or the order it sees them in, plus the
instruments to price them, per-kernel timers that land in the LM step's
existing `result.json`, a target-shape microbenchmark with bit gates and
reach by sabotage, and an on-box leg script.

## 2. The measured baseline

Evidence, `bench/results/e1g/2026-09-11_004220-nvidia/remote/lm-step-memory/target-lean-timing/result.json`
(H100 80GB HBM3 sm_90a, main 3b81dc2e, 2026-09-11 04:44Z to 04:52Z; one
lean step under `MOJOLEARN_TRANSFORMER_TIMING=1`, every tick waits, so the
parts are a breakdown and not a price; the untimed lean step median was
0.563 s in the same leg, `target-lean/result.json`).

| phase (`component_timing_ms`, summed over the 12 layers where per-layer) | ms | lines |
|---|---:|---:|
| envelope.native_call | 564.2 | 1 |
| envelope.blocks_backward | 373.5 | 1 |
| bwd.attention (fused zdot, dq, dkdv, plus the regime scans and the flag) | 262.5 | 12 |
| envelope.blocks_forward | 126.9 | 1 |
| block.attention_total | 96.4 | 12 |
| attn.core (fused forward, plus the regime scans and the flag) | 81.3 | 12 |
| bwd.after_attention | 41.0 | 12 |
| block.mlp_and_residuals | 26.6 | 12 |
| step.head_backward_da + step.head_backward_db + step.head_forward | 18.4 + 14.0 + 13.9 | 3 |
| attn.qkv_proj | 10.1 | 12 |
| block.norm1 | 3.8 | 12 |
| attn.o_proj | 3.6 | 12 |
| attn.rope_and_cache | 1.4 | 12 |
| everything else (embedding, shadow copy, scans, optimizer, CE, pack, upload) | under 3 each | |

Attention core, 81.3 + 262.5 = 343.8 ms, 61 percent of the native call.

Throughput as HANDOFF_speed_gemm defines it (useful flops over the
milliseconds), applied to attention over the VISIBLE cells. Per layer
12 heads x 2048 x 2049 / 2 = 25,178,112 visible cells; forward useful
`4 * cells * 64` = 6.45 GFLOP, backward useful `10 * cells * 64` (the
score recompute, dP, dV, dQ, dK) = 16.1 GFLOP. Over 12 layers, forward
77.3 GFLOP / 81.3 ms = 0.95 TFLOP/s useful; backward 193 GFLOP / 262.5 ms
= 0.74 TFLOP/s useful. The shipped kernels execute more than that (three
score dots per cell in the forward, six in the backward, section 3), so the
executed rates are 1.9 TFLOP/s forward and 1.3 TFLOP/s backward, against the
11 TFLOP/s the IDENTICAL GEMM reaches on the same board and the 67
TFLOP/s nominal FP32 peak. The gap is not arithmetic.

## 3. The kernel reading (`transformer/impl/llama/fused_attention.mojo`, from the source; nothing measured)

Geometry at the target is `hd == 64`, `s == l == 2048`, `n_rep == 1`,
window 0, so every row `t` sees keys `0 .. t`.

### 3.1 Forward, `fused_attn_forward_regblocked_kernel[64]`

- Block. 64 query rows of one head, 256 threads as 16 x 16 (`tr`, `tc`);
  thread holds 4 rows x 2 keys of score accumulators (`dots`, 8
  registers) and 4 rows x 4 columns of context accumulators (`cacc`, 16
  registers). Grid 12 x 32 = 384 blocks on 132 SMs, under three blocks
  per SM in one wave, so occupancy is bounded by the grid as much as by
  registers.
- Shared. `stg` 8,192 B (the Q/K staging page, reused for V), `tile`
  8,448 B (64 x 33), `stats` 512 B, 17,152 B in all.
- Three passes over the row tile's key range (`kb_lo .. kb_hi`, 32 keys
  per block iteration; for row tile `i` that is `2i + 2` blocks, 33 on
  average), each pass recomputing the 64-term score chain. Per (key
  block, 16-wide p window) the block stages 1,536 floats (1,024 of Q and
  512 of K, that is six global loads, six `ftz`, six shared stores per thread)
  behind two barriers, then 16 p-steps of 6 shared loads, 8 RN-FMA and 8
  FTZ-multiplies per thread. Per key block per pass that is 64 p-steps, 384
  shared loads, 512 FMA, 512 multiplies, 24 staging loads, 8 barriers.
  Shared loads per FMA 0.75 in the dot; the H100's shared path serves 32
  words per clock per SM against 128 FMA lanes, so the dot alone cannot
  pass about a third of FMA peak.
- The SAME 64 x 64 Q tile is re-staged from global memory on every (key
  block, p window) of every pass, about 33 x 4 x 3 = 396 reloads of 16 KB
  per block, 6.3 MB per block, 2.4 GB per layer of L1/L2 reads, two
  thirds of all staged floats.
- Pass 1 folds the row maximum; pass 2 recomputes the scores, takes
  `identical_exp` per cell, folds the denominator serially on 64 of 256
  threads; pass 3 recomputes the scores again, exp again, divides, then
  stages V (8 loads per thread) and folds the context chain on all 256
  threads (16 accumulators per thread, 40 instructions per key).
- Per visible cell, counted, 3 dots (3 x (64 FMA + 64 multiplies + 48
  shared loads)) + 2 exp (about 30 each) + 1 div + the context step (about
  170 per cell across the block) + masks, about 800 thread-instructions.
  Per layer 25.2 M cells x 800 = 20 G; over 12 layers 242 G in 81.3 ms =
  3.0 T thread-instructions/s against about 33 T/s of issue capacity (128
  lanes x 132 SMs x 1.98 GHz), 9 percent. The rest is exposed staging
  latency (one staging round trip per p window, no double buffer), the
  barrier count, and the grid.

### 3.2 Backward, three kernels at `TQ == BJ == 4`

`fused_rows_per_block(64) == 4`, so the backward tiles are FOUR query rows
(zdot, dq) or FOUR keys (dkdv) per 256-thread block, `HD == 64` lanes per
row. Grids are 12 x 512 = 6,144 blocks each. Shared per block 17,696 B (two
32 x 65 operand pages plus two 4 x 33 tiles). Each thread holds its row's
`q` or `dctx` (64 floats, `vec`, `stack_allocation` indexed by comptime
`p`; register-resident if the compiler promotes it, local otherwise; the
resource leg answers which).

- `fused_bwd_zdot_kernel`. Per 32-key block iteration, stages K AND V
  (2,048 floats each, 16 global loads, 16 `ftz`, 16 shared stores per
  thread), barrier; lanes 0 .. 31 each compute one row's `y` for one key
  (64-term dot, exp, div), lanes 32 .. 63 one `dy` (64-term dot), barrier;
  then LANE 0 OF EACH ROW folds the z chain over the 32 keys (32 dependent
  steps of two shared loads, one FMA, one multiply) while 252 threads
  wait; barrier. Counted per key block, the dot phase is about 64 x 8 =
  512 dependent cycles, the four-thread z phase about 32 x (30 + 8) =
  1,200, so the serial fold on 1.6 percent of the block is the longer phase.
- `fused_bwd_dq_kernel`. The same staging and the same two dots (y and dy
  recomputed a second time), then stages 19 to 21 (`ds`, `dcell`) per
  cell, then the dq chain on all 256 threads (one accumulator, two shared
  loads per FMA). Four barriers per key block.
- `fused_bwd_dkdv_kernel`. One block per four keys; per 32-query block
  iteration stages Q and dctx (16 loads per thread), lanes 0 .. 31 compute
  `y` for one query against the block's key (the THIRD recompute of the
  score chain, plus per-cell GLOBAL gathers of `amax[row]` and
  `denom[row]`), lanes 32 .. 63 `dy` (third recompute), then `ds` per cell
  with a global gather of `zdot[row]`, then the dk and dv chains (two
  accumulators, four shared loads per two FMA). Four barriers per query
  block.
- Per visible cell, counted, 6 dots (6 x 128 arithmetic + loads) + 3 exp
  + 3 div + 3 gathers + 2 x `ds` + 3 chain steps over the 64 columns,
  about 1,300 thread-instructions; 25.2 M x 1,300 x 12 = 393 G in 262.5 ms
  = 1.5 T/s, 4.5 percent of issue capacity. The backward is not
  arithmetic-bound and not bandwidth-bound (each block re-reads its head's
  512 KB K and V from L2; 6,144 blocks x 0.5 MB x 2 kernels is 6 GB per
  layer at L2 rates, about 1 ms); it is bound by the four-thread serial
  fold, one exposed staging round trip per 32 keys, four barriers per 32
  keys, and eight warps per block.
- Traffic per layer, counted. zdot and dq each 3.2 GB of L2 reads of K and
  V (every block stages the whole causal range for four rows); dkdv 3.2 GB
  of Q and dctx; plus 3 x 25.2 M x 4 B = 302 MB of scalar gathers.

### 3.3 What the launchers add

Both launchers run three (forward) or four (backward) `device_absmax`
scans before the kernel, each with an allocation, a launch, two
synchronizes and a host readback, and read the corner flag afterwards
(another allocation and readback), about ten host round trips per layer
per direction. They are inside `attn.core` and `bwd.attention`; the new
timers (section 5.1) put a number on them (`attn.fwd_regime_scan`,
`attn.bwd_regime_scan`, `attn.*_corner_flag`).

## 4. The arms, with the identity argument before the code

All three arms keep every chain's terms and their order and change only
what is recomputed and which thread holds which chain. They instantiate at
`head_dim == 64` (`ATTN_STASH_HD`); every other head dim takes the
shipped kernels under every arm. They exist on a
`-D MOJOLEARN_ATTN_ARM_TRIAL=1` build only; the shipped build compiles
none of them and reads no environment. Selection is by `MOJOLEARN_ATTN_ARM`
(`baseline`, `bwd_stash`, `fwd_sstash`, `bwd_stash_tiled`, `stash`,
`stash_tiled`), read by the launchers through `fused_attention_arm_from_env`
(the callers' signatures are fixed), or passed explicitly by the harness
through `fused_forward_launch_arm` / `fused_backward_launch_arm`.
`MOJOLEARN_ATTN_ARM_SABOTAGE=1` selects the arm's sabotage instantiation.

### 4.1 DEVIATION 2525, `bwd_stash`, materialize y, dy and dcell once

Identity argument. `y[t, j]` is `ftz(div(ftz(exp(ftz(ftz(masked) - m))),
d))` with `masked = ftz(pmul(dot, scale) + 0.0)` and `dot` the 64-term
`_step` chain over `p` ascending from `+0.0`; `dy[t, j]` is the 64-term
`_step` chain of `dctx[t, p] * v[j, p]` over `p` ascending. The three
shipped backward kernels compute both from the same staged (and
`ftz`-flushed) operands with the same code and land on the same bits;
writing the Float32 to memory and reading it back is the same value.
`dcell = pmul(ftz(pmul(y, ftz(ftz(dy) - ftz(z)))), scale)` is the same
three operations the shipped dq AND dkdv kernels each apply; computing it
once in dq and reading it in dkdv changes nothing. The folds are the
shipped loops, z over keys ascending on lane 0 (`fused_bwd_zdot_stash_kernel`
IS the shipped zdot plus two stores), dq over keys ascending
(`fused_bwd_dq_stash_kernel` reads `y`, `dy` from the stash instead of
recomputing, computes `dcell`, stores it over `dy` in its own cell, folds
as before), dk and dv over (head ascending, query ascending)
(`fused_bwd_dkdv_stash_kernel` stages the `y` and `dcell` tiles for its four
keys beside Q and dctx and folds as before). Only cells the shipped
kernels compute are stored or read, since the visibility tests are copied
(`_row_range` on the write side, `_key_query_range` on the dkdv read side,
the same predicate solved for the other variable). The corner rule is the
shipped one on the same chains.

Cost model, counted. Per visible cell 2 dots instead of 6, one exp and
one div instead of three, no gathers; scratch 2 x `B * nh * L * S` floats
= 2 x 201 MB at the target, allocated by the launcher per call and freed
after the kernels (the allocation is timed as `attn.bwd_scratch_alloc`);
traffic 201 MB x 2 written by zdot, 402 MB read plus 201 MB written by
dq, 402 MB read by dkdv, 1.4 GB per layer, about 0.6 ms at HBM rates. The
block shapes (TQ 4, BJ 4), the barriers and the four-thread z fold are
unchanged, so this arm removes arithmetic but not the latency structure, so
the expected gain is bounded by how much of the 262 ms is the dots.

Sabotage (reach). dq_stash flips one ulp of the `dcell` it stages for its
own fold (dq moves, dk does not); dkdv_stash flips one ulp of the staged
`y` tile (dv moves, dk does not). The flip of a zero value (an
underflowed weight) is `2^-100`, so `ftz` cannot launder it.

### 4.2 DEVIATION 2526, `fwd_sstash`, keep the score and the exp between passes

Identity argument. The three passes are the contract; recomputing the
score in passes 2 and 3 is not. Pass 1 of
`fused_attn_forward_regblocked_sstash_kernel` computes each visible cell's
`masked` exactly as the shipped pass 1 does (same staging, same
`_step_preflushed` chain over `p` ascending, same `_pmul`, same mask add)
and stores it to a `[B, nh, L, S]` scratch. Pass 2 reads it back, computes
`e = ftz(exp(ftz(ftz(masked) - ftz(m))))` with the pass-1 row maximum
(`stats[r]`, written once at the end of pass 1 and never again, so it is
the value the shipped pass 3 also reads), stores `e` over the score (its
own cell) and into the tile, and folds the denominator as before. Pass 3
reads `e` back and divides. The shipped pass 3 recomputes `e` from the
same `masked` bits and the same `m` and gets the same `e`. The V staging
and the context chain are the shipped code. Only visible cells are
stored or read.

Cost model, counted. Per visible cell 1 dot instead of 3, 1 exp instead
of 2; passes 2 and 3 stage nothing for the dots (the Q re-staging of
section 3.1 drops from 396 to 132 reloads per block); scratch 201 MB per
call (timed as `attn.fwd_scratch_alloc`); traffic 201 MB written, 402 MB
read, 201 MB written, 0.8 GB per layer, about 0.3 ms.

Sabotage (reach). The pass-1 store flips one ulp of the score, so
`denom` and `ctx` move while `amax` (from the register value) does not.

### 4.3 DEVIATION 2527, `bwd_stash_tiled`, register-blocked folds over the stash

Identity argument. With `y`, `dy` and `dcell` materialized, dq, dk and dv
are three K-serial contractions, each output cell one chain from `+0.0`
stepped by `_step` on two operands. `dq[t, d]` over `j` ascending of
`dcell[t, j] * k[j, d]`; `dk[j, d]` over (head in the kv group ascending,
`t` ascending) of `dcell[t, j] * q[t, d]`; `dv[j, d]` the same order of
`y[t, j] * dctx[t, d]`. `fused_bwd_dq_tiled_kernel` gives one block 64
rows x 64 columns and each thread 4 x 4 accumulators, stages a 16-key K
tile and the 16-key `dcell` tile (computed by the staging thread from the
stash exactly as 4.1 computes it, and written back over `dy` for the dkdv
kernel), and steps every visible (row, column) for each key of the tile
in ascending order; key tiles ascend, keys within a tile ascend, so every
row's chain sees `j` ascending. `fused_bwd_dkdv_tiled_kernel` gives one
block 64 keys x 64 columns, 16 + 16 accumulators per thread, stages 16
queries of Q, dctx, `y` and `dcell`, and steps each visible (key, column)
for each query ascending, heads of the kv group ascending outside. Masked
cells are skipped by the copied visibility tests (`_row_range` per row in
dq, `_key_query_range` per key in dkdv), so each chain holds exactly the
shipped terms. Which thread holds which chain, how many chains a thread
holds and how operands reach shared memory are execution-plan choices the
contract does not read. The corner rule is applied per chain as before.
zdot is 4.1's `fused_bwd_zdot_stash_kernel`.

Cost model, counted. Shared loads per FMA 0.5 (dq) and 0.25 (dk and dv)
against 2 in the shipped and 4.1 kernels; one staging round trip and two
barriers per 16 keys (or queries) for 16 x 16 (or 16 x 32) FMA per thread;
grids 384 blocks each; shared 8.4 KB (dq) and 16 KB (dkdv); no gathers.
This is the GEMM lane's shape at K-serial order, so the dq and dkdv
kernels are expected to approach the IDENTICAL GEMM's rate, leaving the
zdot kernel (two dots, the four-thread fold, unchanged) as the backward's
floor; section 8 names the next arm for it.

Sabotage (reach). dq_tiled flips one ulp of the `dcell` it stages (dq
moves, dk does not); dkdv_tiled flips one ulp of the staged `y` tile (dv
moves, dk does not).

### 4.4 Composed arms

`stash` = 2526 + 2525, `stash_tiled` = 2526 + 2527. The forward and
backward launchers read their own bits; the LM probe runs the composed
arms (section 6, `lm-stash`, `lm-stash_tiled`).

### 4.5 Column notes (GPU-agnostic by construction)

No arm names a vendor. Shared pages are as follows. The sstash forward claims the
shipped 17,152 B; the stash backward kernels claim the shipped 17,696 B;
the tiled kernels 8,448 B (dq, 4,096 + 4,096 + 256) and 16,384 B (dkdv);
every page is under the 32 KB column limit, so no kernel-matrix row was
needed and none was added. The scratch is device memory the launcher
allocates per call; on a column where that allocation is slow the timer
line says so (`attn.*_scratch_alloc`), and the persistent-scratch
follow-on (section 8) is the answer, not a vendor branch.

## 5. Instruments

### 5.1 Per-kernel timers (`-D MOJOLEARN_ATTN_PHASE_TIMERS=1`)

Compiled into both launchers; switched on at run time by the block
timers' own switch (`MOJOLEARN_TRANSFORMER_TIMING=1`), so the one LM step
that prints `block.*` and `bwd.*` also prints `timing attn.<kernel> <ms>
ms` per launch on fd 1, in the line shape `tools/lm_step_memory_probe.py`
`parse_timing_lines` already sums by name into `component_timing_ms`
(the `attn.` prefix is already excluded from its total as a sub-phase of
`attn.core` and `bwd.attention`). The names are `attn.fwd_regime_scan`,
`attn.fwd_kernel` (or `attn.fwd_sstash_kernel` and
`attn.fwd_scratch_alloc`), `attn.fwd_corner_flag`, `attn.bwd_regime_scan`,
`attn.bwd_zdot`, `attn.bwd_dq`, `attn.bwd_dkdv` (or `attn.bwd_scratch_alloc`,
`attn.bwd_zdot_stash`, `attn.bwd_dq_stash` / `attn.bwd_dq_tiled`,
`attn.bwd_dkdv_stash` / `attn.bwd_dkdv_tiled`), `attn.bwd_corner_flag`. A
timer build synchronizes around every launch, so it serializes the queue and
its numbers are a breakdown, never a request price; the probe's untimed
steps do not print (the run-time switch is off) and stay unserialized.

### 5.2 The microbenchmark (`bench/attention_step_price_main.mojo`) and the two corpora

Rule change during this lane (Andrew, 2026-09-11, ENGINEERING_RULES
section 9 at 2eddc489). The two data kinds of any neural timing or
promotion claim are two ORDINARY training corpora that differ in what
they are, never an adversarial or heavy-tailed fixture standing in as
the second kind ("we build our software to handle GENERAL NORMAL CASES").
So the harness's timing inputs are activations the REAL training path
produced on two normal corpora, and its generator kinds are a bit
equality smoke and a correctness fixture only.

The two corpora, both pinned by manifest and sha256 (schema
`mojolearn.byte-lm.corpus.v1`).

- English text, `training/corpus/tinyshakespeare/input.txt` (committed;
  1,115,394 bytes, sha256 `86c4e6aa...`, `manifest.json` beside it).
- Source code, `training/corpus/cpython312_lib/input.txt`, the 163
  top-level `Lib/*.py` files of the CPython 3.12.0 release tarball sorted
  bytewise and concatenated, 4,522,096 bytes, sha256
  `f08d783cac53829be0da6def7ac74947f3e339915dfee7ca8d8db5ee344fc956`,
  computed on the Mac 2026-09-11 from the tarball (sha256 `51412956...`).
  The bytes are NOT committed (a directory `.gitignore` refuses them);
  `manifest.json`, `README.md` and `tools/fetch_corpus_cpython312_lib.sh`
  (download, tarball hash, selection rule, corpus hash, refuses a
  mismatch) are the pinned artifact, and the leg runs the fetch on the
  box.

The real training path on a corpus. `tools/lm_step_memory_probe.py`
gains `--corpus <input.txt>` (additive; the coordinator's rule change
put this file in scope). The manifest beside the file is checked (schema,
sha256, length), and step `k`, row `b` reads bytes
`[(k * batch * length + b * length) mod (bytes - length - 1), + length + 1]`
as int32 ids, targets shifted one byte, the byte LM's next-byte schedule
at this shape; `result.json` records `corpus` (path, sha256, manifest
sha256, schedule) and `attention_arm`. Byte ids are below 256 and valid in
the target's 50,257-row vocabulary; the embedding is read on its first 256
rows, which is the same real path.

The operand capture. `-D MOJOLEARN_ATTN_OPERAND_DUMP=1` (never shipped;
`ATTN_OPERAND_DUMP` in `fused_attention.mojo`) makes the backward
launcher's FIRST call in a process (the last layer of the first step)
write `q.bin`, `k.bin`, `v.bin`, `dctx.bin` and `meta.txt` (geometry and
the scale's bits) to `MOJOLEARN_ATTN_OPERAND_DUMP_DIR`, in the launcher's
own layouts, before its regime scans; nothing is written when the
directory is unset or already holds `meta.txt`. This is the small hook
the rule change allowed for; it lives behind a define in this lane's
file and touches no other path.

The harness kinds (`MOJOLEARN_ATTN_KINDS`).

- `file:<dir>`, the real-activation kind and the ONLY timing input, the
  dump above, one directory per corpus (`operands-shakespeare`,
  `operands-cpython` on the leg); the shape is the dump's.
- `hashed`, a cheap bit-equality SMOKE, the k-NN gate's `hashed_block`
  profile in Mojo (log-uniform magnitudes, per-column octave scales,
  twelve cluster offsets), every cell a distinct 53-bit hash, never
  uniform (uniform data hides permutation bugs); its `PRICE` lines are
  not a claim.
- `heavytail`, an adversarial profile (six e-folds, one cell in 64 boosted
  eight times, a diagonal content term) for the CORRECTNESS sections only;
  the leg runs it with `MOJOLEARN_ATTN_TIMING=0`.

Per kind the harness runs the shipped kernels against the eager stage
kernels (the profile's spelling, as `transformer_fused_check` runs them,
here at the target shape), the candidate against the baseline on ctx,
amax, denom, zdot, dq, dk, dv, reach by sabotage with the clean arm
restoring the bits, and (timing on) two warmups and seven rounds with the
arms alternated inside each round (A B, B A), `PRICE` lines per sample,
`TABLE` lines with medians, derived backward time and useful TFLOP/s per
section 2's definition. Outputs are poisoned with NaN between arms so an
arm that wrote nothing cannot inherit the previous bits. On a build
without the trial hook the reach section fails and says why.

### 5.3 The small-shape gate (`transformer/checks/transformer_attention_arms_check.mojo`)

Additive beside `transformer_fused_check.mojo`, every arm on the fused
check's 16 cases (many row tiles and key blocks, windows starting inside
a key block, decode steps, ring gathers, hd 16/24/64/128, n_rep 1 and 2,
the underflow corner, the regime refusals), statuses as the cases
expect, RAN buffers bit-equal to eager, reach at hd 64. Light enough for
the M4.

### 5.4 The leg (`tools/attention_step_leg.sh`)

The `MOJOLEARN_GEMM_LEG_EXTRA` hook of `tools/gemm_remote_leg.sh`, in the
style of `tools/knn_selection_gate.sh`. In order, every item under
`timeout 300` with its exit code in `status.tsv`, all under
`/root/gemm_leg_out/attention-step/`.

1. three harness builds (price, timers, arms check) and the small-shape
   gate;
2. `smoke-<arm>` on `hashed,heavytail`, bits and reach only;
3. the base and byte LM bindings with the trial hook, the timers and the
   operand dump riding `MOJOLEARN_BUILD_EXTRA_DEFINES` (as
   `tools/lm_step_memory_probe.sh` builds them, which takes no flags; the
   leg calls the python probe directly with the wrapper's arguments);
4. `corpus-tinyshakespeare` (hash check of the committed corpus) and
   `corpus-cpython312-lib` (the fetch script);
5. `dump-<corpus>`, one lean target step per corpus with the dump
   directory set, giving `operands-shakespeare/` and `operands-cpython/`
   (each with `sha256.txt` of the four files);
6. `price-<arm>` on `file:operands-shakespeare,file:operands-cpython`
   per arm (eager oracle on the first arm), then `timers-<arm>` on the
   same activations, summed into `timers_summary.tsv`;
7. `lm-<arm>-<corpus>` (`--target --resident-lean --witness-every-step
   --corpus`, 3 steps) and `lmtiming-<arm>-<corpus>` (`--component-timing
   --steps 1`) for `baseline` and `stash_tiled` on both corpora
   (`MOJOLEARN_ATTN_LEG_LM_ARMS` adds `stash` when the lease allows), and
   `lm_summary.tsv` with the lean medians, the per-step witnesses, a
   `witnesses_equal_baseline` verdict per arm and corpus, and the
   attention phase lines.

## 6. Gates

An arm is a candidate for the default only when all of these hold on
the same leg.

1. `arms-check` exit 0 (every case, every arm, statuses and bits).
2. `smoke-<arm>` exit 0 (bits equal and reach proven on the hashed and
   adversarial fixtures; correctness only).
3. `price-<arm>` on BOTH corpora's activations, exit 0, with
   `BITS file:operands-<corpus> baseline_vs_eager * MATCH` (the shipped
   kernels equal the profile at the target shape on real activations),
   `BITS ... <arm>_vs_baseline * MATCH` for all seven buffers, and
   `REACH ... sabotage_flipped_cells > 0 clean_restored=True`.
4. `TABLE` fwd+bwd ratio above 1.0 on BOTH corpora, with the arm's
   variance across the seven rounds inside the win (read the `PRICE`
   lines, not only the medians). A win on one corpus and not the other is
   reported as exactly that and is not flipped.
5. `lm-<arm>-<corpus>/result.json` for both corpora has `limited: false`,
   `steady_median_seconds` below `lm-baseline-<corpus>`'s of the same leg,
   and `step_witnesses[k].sha256` equal to `lm-baseline-<corpus>`'s for
   every k (`witnesses_equal_baseline=True` in `lm_summary.tsv`). The
   baseline's own witnesses on the shakespeare corpus are new evidence
   (run 2 used synthetic ids and cannot be compared byte for byte); the
   baseline's synthetic-id step remains reproducible through
   `tools/lm_step_memory_probe.sh` if a second witness of the shipped path
   is wanted.
6. `lmtiming-<arm>-<corpus>/result.json` `component_timing_ms` has the
   `attn.*` lines present, `bwd.attention` and `attn.core` below the
   baseline's on both corpora.

A `timeout` (124) or a `limited: true` is a recorded limitation, not a
pass. The default flip is a one-line edit of `ATTN_ARM_DEFAULT` plus this
brief, in the same session as the evidence filing.

## 7. What could not be resolved from the source

- Whether `vec` (the 64-float per-thread operand of the backward kernels)
  is register-resident or local memory, and the register counts and
  occupancy of the seven kernels. The GEMM lane's `tools/gemm_cuda_resources.py`
  path (PTX to ptxas) answers it; not run here.
- The cost of a 201 MB `enqueue_create_buffer` per call on the H100
  (whether MAX pools device allocations). The `attn.*_scratch_alloc`
  timer line answers it; if it is a millisecond or more per call the
  persistent scratch (section 8) comes before any default flip.
- How much of the 262 ms is the dots versus the latency structure
  (section 3.2). The `timers-*` and `lmtiming-*` splits between
  `bwd_zdot`, `bwd_dq`, `bwd_dkdv` under `baseline`, `bwd_stash` and
  `bwd_stash_tiled` answer it directly, since 2525 removes dots only, 2527
  removes the structure for dq and dkdv only.

## 8. Follow-ons named, not implemented

- DEVIATION 2528 (proposed). zdot as a register-blocked 64-row kernel
  (the forward's tile shape) computing the y and dy dots with 4 x 2 + 4 x 2
  accumulators per thread over staged Q, dctx, K and V pages, writing the
  stash, and folding z on 64 of 256 threads per row tile (the forward's
  pass-2 pattern) instead of 4 of 256. Same chains, same orders. It is
  the backward's floor once 2527 lands.
- DEVIATION 2529 (proposed). A persistent scratch owned by the caller's
  stage struct (`LlamaDeviceStages` / `LlamaBackwardStages`, outside this
  lane's files) instead of a per-call allocation, and the forward writing
  `y` into it so the backward's zdot needs the `dy` dot only.
- The regime scans and the corner flag, ten host round trips per layer
  per direction (section 3.3), removable by one fused scan kernel and a
  deferred flag read; sized by the new timer lines first.

## 9. Files

- `transformer/impl/llama/fused_attention.mojo`, the arm hook, the
  timers, the operand dump hook, five new kernels, the `_arm` launchers;
  the shipped kernels and their launch geometry are byte-for-byte what
  they were, reached through `_launch_bwd_shipped` and the `not ran_arm`
  branch.
- `bench/attention_step_price_main.mojo`, the microbenchmark.
- `transformer/checks/transformer_attention_arms_check.mojo`, the
  small-shape gate.
- `tools/attention_step_leg.sh`, the leg.
- `tools/lm_step_memory_probe.py`, the `--corpus` option (additive, per
  the rule change).
- `training/corpus/cpython312_lib/manifest.json`, `README.md`,
  `.gitignore` and `tools/fetch_corpus_cpython312_lib.sh`, the pinned
  source-code corpus (bytes not committed).
- This brief.

## 10. Leg result and the flip (2026-09-11, H100 80GB HBM3 sm_90a, commit e6b1350a)

Evidence, `bench/results/e1g/2026-09-11_113013-nvidia-h100-attention-step/remote/attention-step/`
(status.tsv, arms-check.log, smoke-*.log, price-*.log, price_tables.txt,
timers_summary.tsv, lm-*/result.json, lmtiming-*/result.json,
lm_summary.tsv). Two earlier legs the same morning
(`2026-09-11_070550-...` and `2026-09-11_111701-...`) died in the
probe's new `--corpus` option before any measurement (a NameError, then a
duplicate keyword; both mine, both fixed on main, ce67991f and e6b1350a);
their builds, arms check and smokes passed and are kept as evidence of
that. The real activations (q, k, v, dctx at the target shape, 25 MB per
corpus) are not in the repository; they are at
`~/mojolearn-evidence/attention-step-2026-09-11_113013/operands-<corpus>/`
on Andrew's Mac and their sha256s are in each `operands-<corpus>/sha256.txt`
here.

Gates (section 6), all six held on both corpora: arms check PASS (15 x 3);
four smokes PASS; every `BITS ... _vs_baseline` and `baseline_vs_eager`
line MATCH on both corpora's activations (52, 38, 30, 30 lines across the
four price runs, zero mismatches), REACH proven with clean restore in all
eight; the lean step `limited: false` with every step witness equal to the
baseline's on both corpora (`witnesses_equal_baseline=True`, three steps,
six hashes each); the timers present and lower.

Price at the target shape, real activations, medians of seven rounds
(spread across rounds under 0.1 ms on both arms):

| arm | fwd ms | fwd+bwd ms | fwd ratio | fwd+bwd ratio | corpus |
|---|---:|---:|---:|---:|---|
| baseline | 6.79 | 28.46 | 1.00 | 1.00 | shakespeare |
| bwd_stash | 6.76 | 20.91 | 1.00 | 1.36 | shakespeare |
| fwd_sstash | 3.16 | 24.81 | 2.14 | 1.15 | shakespeare |
| bwd_stash_tiled | 6.80 | 17.17 | 0.99 | 1.66 | shakespeare |
| stash_tiled | 3.17 | 13.57 | 2.14 | 2.10 | shakespeare |
| baseline | 6.78 | 28.52 | 1.00 | 1.00 | cpython |
| stash_tiled | 3.16 | 13.57 | 2.14 | 2.10 | cpython |

The lean target step (162,147,840 parameters, L2048, batch 1, three
steps, median of the two steady steps):

| corpus | baseline s | stash_tiled s | ratio | witnesses |
|---|---:|---:|---:|---|
| shakespeare | 0.5619 | 0.3829 | 1.47 | equal, 3 steps x 6 hashes |
| cpython | 0.5593 | 0.3804 | 1.47 | equal, 3 steps x 6 hashes |

Per-kernel timers in the lean step (serialized, a breakdown and not a
price; cpython corpus, shakespeare within 0.5 ms of every line):

| line | baseline ms | stash_tiled ms |
|---|---:|---:|
| envelope.native_call | 560.8 | 381.9 |
| bwd.attention | 261.4 | 125.5 |
| attn.bwd_zdot / attn.bwd_zdot_stash | 52.2 | 89.4 |
| attn.bwd_dq / attn.bwd_dq_tiled | 90.7 | 15.5 |
| attn.bwd_dkdv / attn.bwd_dkdv_tiled | 116.4 | 18.4 |
| attn.core | 81.0 | 38.4 |
| attn.fwd_kernel / attn.fwd_sstash_kernel | 79.5 | 36.8 |
| attn.fwd_scratch_alloc + attn.bwd_scratch_alloc | | 0.14 + 0.14 |
| attn.fwd_regime_scan + attn.bwd_regime_scan + both corner flags | 3.4 | 3.3 |

Reading. The two tiled folds took dq and dkdv from 207 ms to 34 ms. The
stash zdot is now 89 ms, 71 percent of the backward: it is the shipped
zdot (four rows per block, the z fold on lane 0 of each row) plus the
stash stores, so DEVIATION 2528 (a 64-row register-blocked zdot with the
fold on 64 threads) is the backward's next floor. The scratch allocation
is 0.14 ms per direction per step, so DEVIATION 2529 (persistent scratch)
is not worth a lane at this shape. The regime scans and corner flags are
3.3 ms per step; not a lane either.

The flip. `ATTN_ARM_DEFAULT = ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH |
ATTN_ARM_BWD_TILED`; `ATTN_ARM_COMPILED` (trial build, or the default is
an arm) gates the arm launch code, and every sabotage instantiation sits
under `comptime if ATTN_ARM_TRIAL` so a shipped build compiles only the
default arm's clean kernels. Read back on the M4 after the edit, one at a
time under `nice 19`: the shipped `transformer_fused_check` (no trial
define) PASS, 15 cases, the six head_dim 64 cases RAN through the arm and
every buffer bit-identical to eager; `transformer_attention_arms_check`
PASS 15 x 3 with reach; the shipped bindings (no extra defines) ran the
control shape on tinyshakespeare for three steps with every witness equal
to the baseline arm's run of the same morning (Apple column, control
shape only: 3.77 s to 2.79 s per step, not a claim).

Not a claim against anyone: the torch step at the target shape is an
OWED opponent row (bench/OPPONENT_REFERENCE.md); until it exists these
are internal before and after numbers.

## 11. Second round, WIP (2026-09-11, wound down by Andrew before any code): the AMD reading and DEVIATIONS 2528, 2530, 2531, 2533

STATUS: reading and design only. NOTHING BELOW IS BUILT. No kernel, check
case, harness change or leg exists for this section. The deciding speed
column moved to AMD (Instinct MI325X, DigitalOcean, HIP gfx942) during the
lane; NVIDIA is the confirmation column.

### 11.1 How the fused attention sizes blocks on AMD today (from the source)

- `fused_attention.mojo` reads two column rows only:
  `lib_hardware_ftz_fma_for` (NVIDIA only) and `lib_smem_page_fits_for`.
  Every block is `FUSED_THREADS = 256` threads on every column, and every
  grid is the same as on the H100. On AMD `_step` is the software seam
  `ftz(fma(ftz(a), ftz(b), acc))` (`std.math.fma`, then three bit-test
  flushes). `_step_preflushed` drops the two operand flushes.
- `column_shared_limit(AMD)` is 64 KB, so every page fits. The hardware
  matrix marks AMD shared memory as statically partitioned per CU
  (`smem_statically_partitioned_for`, `smem_per_core_for` 64 KB), with 2048
  thread slots per CU and a 64-lane wavefront. Resident blocks per CU are
  therefore `min(2048 // 256 = 8, 65536 // page_bytes)`. The page, not the
  thread count, is the divisor for every shipped attention kernel. The
  rows are UNVALIDATED transcriptions pinned to the MI250X (110 CUs), and
  the MI325X count is not in the repository.
- Shipped default pages and resident blocks per CU on AMD: forward sstash
  17,152 B, 3 (grid 384); zdot_stash 17,696 B, 3 (grid 6,144); dq_tiled
  8,448 B, 7 (grid 384); dkdv_tiled 16,384 B, 4 (grid 384). On the H100
  (164 KB per SM) the same pages allow 9 or more, so the H100 reading
  (latency and registers bound, grid under three blocks per SM) does not
  transfer. On AMD, page bytes are a first-order occupancy term.
- Consequence for geometry: any page-size choice that differs by vendor
  must be a kernel-matrix SCHEDULING row (for example
  `attn_zdot_rows_per_block_for[column]`, `attn_fwd_rows_per_block_for[column]`),
  never an inline branch.

### 11.2 Order of attack on AMD

0. FIRST: price `stash_tiled` against `baseline` on AMD on both corpora
   (the flip was H100 evidence only). The existing leg body
   `tools/attention_step_leg.sh` is NVIDIA-shaped (nvidia-smi, the
   driver/ptxas block); a vendor-agnostic body is owed.
1. DEVIATION 2528 (zdot tiled), with an AMD-sized page.
2. DEVIATION 2531 (forward grid, 32 rows per block): on AMD it also shrinks
   the page (12,736 B, 5 blocks per CU, grid 768).
3. DEVIATION 2533 (preflushed seams in the tiled folds, the forward
   context chain and the new zdot): fewer instructions per term on every
   column.
4. DEVIATION 2530 (forward Q residency) only at 32 rows: at 64 rows its
   page is 27,904 B, which drops AMD from 3 resident blocks to 2.
5. DEVIATION 2532 is not buildable inside this lane's files (below).

### 11.3 DEVIATION 2528, zdot as a register-blocked y/dy kernel plus a row z fold

Design. Kernel A (`ydy_tiled`): block = 64 query rows of one (batch, head);
thread `(tr, tc)` holds rows `tr + 16u` (u < 4) and keys `tc + 16v` (v < 4)
of a 64-key block iteration, so 16 y dot chains and 16 dy dot chains per
thread. Per 16-wide p window it stages Q, dctx (64 rows each), K and V (64
keys each), flushed, stride KS + 4 as the forward pads, then 16 p-steps of
`_step_preflushed`. After the window loop, per visible cell it computes
`masked`, `e` and `y` with the row's `ftz(amax)` and `ftz(denom)` (loaded once
per thread into registers) and stores `y` and `ftz(dy)` to the stash. No
shared tile is needed for z. Kernel B (`zfold`): one row per thread, 256 rows
per block, no shared memory, no barrier. `z = _step(dy_st[j], y_st[j], z)`
runs over the row's visible keys ascending from `+0.0`, then `ftz`, the
corner test (`-0.0` with `hi < s - 1`) and the zdot store. dq_tiled and
dkdv_tiled follow unchanged.

Identity argument. Each `y[t, j]` and `dy[t, j]` is the same 64-term chain
over p ascending from `+0.0` on the same flushed operands (`_step_preflushed`
on `ftz`-staged values equals `_step`, since `ftz` is idempotent), followed by
the shipped pmul, mask add, exp against `ftz(amax[row])` and division by
`ftz(denom[row])`. Only the thread holding the chain and the staging page
change. The z chain is the shipped chain, the same operands in the same
order (dy then y, j ascending over the row's visible range from
`_row_range`), read from the stash the shipped zdot_stash also writes.
Sabotage: kernel A flips one ulp of the `y` it stores (zdot, dq, dk and dv
move; the forward buffers do not).

Page. 256 staged rows x 20 = 5,120 floats = 20,480 B: 3 blocks per CU on
AMD. The AMD variant to price is 32 rows per block (192 rows x 20 = 15,360
B, 4 per CU, grid 768), or KS 8 (half the page, twice the staging round
trips). This is a kernel-matrix row.

Counted expectation (H100 analogy, not a measurement). Per visible cell,
kernel A executes 2 x 64 RN-FMA with 0.5 shared loads per FMA and one
staging round trip per 16 steps, the same count and cadence as dkdv_tiled
(18.4 ms per step), plus one exp and one div per cell. Estimate about 25 ms
against zdot_stash's 89.4 ms, so the lean step goes from about 0.38 s to
about 0.32 s on the H100. AMD is not estimated: no AMD attention timing
exists.

### 11.4 DEVIATION 2531, forward grid (32 rows per block)

`fused_attn_forward_regblocked_sstash_kernel` with `TQ` a comptime
parameter (the hd-128 path already uses 16). RPT = 2, 4 dots and 8 context
accumulators per thread, grid 768 blocks. Identity argument: every chain
(score over p, denominator over keys, context over keys) keeps its terms
and order. The row maximum is an `identical_fmax` fold over values that are
never `-0.0` (the `+ 0.0` mask add) and never NaN (the regime), so its
grouping is free (contract 5.1). Sabotage for the new instantiation: flip
the pass-3 weight it stores in the tile (ctx moves; amax, denom and the
backward do not), which distinguishes it from the shipped sstash sabotage
(denom moves). Page 12,736 B.

### 11.5 DEVIATION 2530, forward Q residency

Stage the `[TQ][64]` Q page once per block. Per window, stage K only (2
slots per thread instead of 6). Reuse the Q page for V in pass 3, since Q
is not read after pass 1. Identity argument: the dots read the same flushed
Q values in the same p order; only the staging schedule changes. Pages:
27,904 B at 64 rows, 15,232 B at 32 rows. Counted expectation on the H100:
small, since staging is about 5 percent of pass-1 instructions.

### 11.6 DEVIATION 2533 (forced by the reading), preflushed seams

`_step` applies software `ftz` to both operands on every column (the NVIDIA
path too, before `fma.rn`). In dq_tiled, dkdv_tiled, the forward context
chain and the zdot fold, both operands are already flushed: staged through
`ftz`, or produced by `_pmul` or `ftz(div)`. A one-ulp sabotage flip of a
normal value stays normal, and the flip of zero is `2^-100`. So
`_step_preflushed` gives the same bits and saves two flushes per term. It
would be built as new instantiations beside the shipped kernels, never as
an edit of them.

### 11.7 DEVIATION 2532 (not buildable here) and its memory

The forward's pass 3 could store `y` over the sstash cell so the backward
needs the dy dot only. That buffer must live from each layer's forward to
its backward, so the callers' stage structs (outside this lane's files)
would own it: 12 x B*nh*L*S floats = 12 x 50,331,648 x 4 B = 2.42 GB of
device memory held across the step at the target shape.

### 11.8 Mechanism notes for whoever resumes

- Arm bits: `BWD_ZTILED = 8`, `FWD_QRES = 32`, `FWD_GRID = 64`, plus a
  `SABOTAGE_NEW = 128` that flips only the new kernels. The shipped kernels'
  sabotage stays under the old bit, so reach on top of `stash_tiled` is
  specific. `fused_attention_arm_name` currently masks with
  `ATTN_ARM_SABOTAGE - 1`, which would drop bits above 16; the parser should
  be the inverse of the name function.
- Keep the shipped path byte for byte. Put a trial-only branch before
  `if want_stash:` / `if want_sstash:` and change only those two conditions
  to `... and not ran_arm`. Make the launch helpers generic, so the shipped
  build instantiates none of the new kernels.
- Arms check: add `stash_tiled` itself, so that new arm = eager and
  default = eager hold in one run. Require forward reach for forward bits
  and backward reach for 2528. At head dims other than 64, assert that
  `SABOTAGE_NEW` moves nothing (the shipped kernels ran).
- Leg body: vendor from the runner's environment (or `rocm-smi` versus
  `nvidia-smi`), arch from `MOJOLEARN_GPU_ARCHS`, and no CUDA-only
  commands on AMD. The AMD pool settings the byte LM legs used
  (`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_*`) belong to the runner.
  Baseline `stash_tiled` for the price (`MOJOLEARN_ATTN_BASELINE`) and for
  `witnesses_equal_baseline`. The item before any arm is `baseline` against
  `stash_tiled` on AMD.
