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
case, harness change or leg exists for this section. (Section 12 records
what a later lane built from it: DEVIATION 2528 only, source only.) The deciding speed
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

## 12. Second round, source built (2026-09-11, worktree lane, nothing run)

STATUS: source only. Nothing in this section was compiled or run. Every
claim that needs a run is marked RUN OWED with its command in 12.6 (the
M4 checks) or 12.7 (the AMD legs). The
shipped default is still `stash_tiled`, and a shipped build compiles none
of the kernels below.

### 12.1 Arm bits and names (the parser and the name function are inverses)

| bit | constant | meaning |
|---:|---|---|
| 8 | `ATTN_ARM_BWD_ZTILED` | DEVIATION 2528, needs `BWD_STASH` and `BWD_TILED` |
| 16 | `ATTN_ARM_SABOTAGE` | the 2525 to 2527 kernels' sabotage (unchanged) |
| 32 | reserved `FWD_QRES` | DEVIATION 2530, not built |
| 64 | reserved `FWD_GRID` | DEVIATION 2531, not built |
| 128 | `ATTN_ARM_SABOTAGE_NEW` | flips only the second-round kernels |
| 256 | `ATTN_ARM_ZROWS32` | trial geometry knob, 2528 at 32 rows per block |
| 512 | `ATTN_ARM_ZROWS64` | trial geometry knob, 2528 at 64 rows per block |

A name is the first-round base (`baseline`, `bwd_stash`, `fwd_sstash`,
`bwd_stash_tiled`, `stash`, `stash_tiled`), then one token per
second-round bit in a fixed order (`_ztiled`, then the geometry token
`_r32` or `_r64`), then `+sabotage` and `+sabotage_new` for the two
sabotage bits. `fused_attention_arm_parse` strips the tokens in reverse
order and raises on an unknown base, a token whose prerequisite is
missing, or a rows token without `_ztiled`. `fused_attention_arm_name`
builds the name from the bits it knows and spells any other bit as
`_bits<N>`, which the parser refuses, so the two functions are inverses
over the valid arms. The first-round `fused_attention_arm_name` masked
with `ATTN_ARM_SABOTAGE - 1` and would have named every second-round arm
by its low bits; that is the defect 11.8 pointed at, and it is fixed.
Names carry no `-` (the leg's `lm_summary` splits run names on it) and no
character outside `tools/do_extra_leg.sh`'s value set.

`stash_tiled_ztiled` takes its rows per block from the new kernel-matrix
SCHEDULING row `attn_zdot_rows_per_block_for[column]` (32 on AMD, 64
elsewhere, UNMEASURED). `stash_tiled_ztiled_r32` and
`stash_tiled_ztiled_r64` force the geometry on any column so one leg
prices both.

### 12.2 DEVIATION 2528, identity argument (written before the code)

Kernels. `fused_bwd_ydy_tiled_kernel[HD, TQ, SABOTAGE]` (kernel A) and
`fused_bwd_zfold_kernel[TZ]` (kernel B), launched by the trial-only
generic `_launch_bwd_ztiled[HD, TQZ, ZSAB]`, which then launches the
shipped `fused_bwd_dq_tiled_kernel` and `fused_bwd_dkdv_tiled_kernel`
instantiations unchanged.

Kernel A. One block holds `TQ` (64 or 32) query rows of one (batch,
head). Thread `(tr, tc)` (`tid // 16`, `tid % 16`) holds rows `tr + 16u`
(`u < TQ // 16`) and keys `tc + 16v` (`v < 4`) of a 64-key block
iteration, so `4 * TQ / 16` y dot chains and as many dy dot chains. Per
16-wide p window it stages Q and dctx for its `TQ` rows and K and V for
the 64 keys, each value through `ftz`, at stride 20 (the forward's
`KS + 4`), behind one barrier, then runs 16 p steps of
`_step_preflushed`. After the four windows, for each visible cell it
computes `masked`, `e` and `y` against the row's `ftz(amax)` and
`ftz(denom)` (loaded once per thread) and stores `y` and `ftz(dy)` to
the two stashes. Kernel B. One row per thread, 256 rows per block, no
shared memory, no barrier; `z = _step(dy_st[j], y_st[j], z)` over the
block's key range ascending, stepping only where `j` is in the row's
`_row_range`; then `ftz`, the corner test and the zdot store.

Why no bit moves.

1. `y[t, j]`. The shipped `fused_bwd_zdot_stash_kernel` computes
   `dot = _step(ftz(q[t, p]), ftz(k[j, p]), dot)` for `p` ascending from
   `+0.0`. `ftz` only maps a subnormal to a signed zero, so it is
   idempotent and leaves every other value alone. `_step(a, b, acc)` is
   `_step_preflushed(ftz(a), ftz(b), acc)` on every column (NVIDIA
   spells both as `fma.rn` then `mul.rn.ftz`; the others as
   `ftz(fma(...))`), so kernel A's chain over the staged `ftz(q)` and
   `ftz(k)` with `_step_preflushed`, q first and k second, p ascending
   from `+0.0` across the four windows, is the same chain on the same
   operands. The cell arithmetic after the dot is the shipped spelling
   term for term: `ftz(_pmul(dot, scale) + 0.0)`, then
   `ftz(identical_exp(ftz(ftz(masked) - ftz(amax[row]))))`, then
   `ftz(identical_div(ftz(e), ftz(denom[row])))`, with `amax` and
   `denom` read from the same buffers.
2. `dy[t, j]`. Likewise `_step(ftz(dctx[t, p]), ftz(v[j, p]), dy)` for
   `p` ascending, dctx first, then `ftz`.
3. `z[t]`. The shipped chain is `z = _step(dys[j], ys[j], z)` over the
   row's visible `j` ascending from `+0.0`, where `dys[j]` and `ys[j]`
   hold exactly the values the stash holds. Kernel B steps the same
   operands in the same order; a key outside the row's range is skipped,
   as the shipped `j >= j_lo and j <= j_hi` test skips it. The iteration
   over the block's union key range (instead of the row's own range) is
   a control-flow choice that keeps the trip count block-uniform; it
   adds no term. The corner test is the shipped one (`ftz(z)` is `-0.0`
   and `j_hi < s - 1`).
4. dq, dk and dv. The shipped `fused_bwd_dq_tiled_kernel` and
   `fused_bwd_dkdv_tiled_kernel` read `y_st`, `dy_st` and `zdot` only at
   visible cells, and kernels A and B write every visible cell with the
   bits `fused_bwd_zdot_stash_kernel` writes. Their instantiations are
   the shipped ones.

Which thread holds a chain, how many chains one thread holds, the page
layout and the barrier count are execution-plan choices the contract does
not read.

Sabotage (reach). Kernel A under `SABOTAGE_NEW` flips one ulp of every `y`
it stores (`_flip_ulp`, a zero becomes `2^-100`). zdot, dq, dk and dv move;
ctx, amax and denom do not, and the arms check and the harness assert
both halves.

Page. `(2 * TQ + 128) * 20` floats, 20,480 B at 64 rows and 15,360 B at
32 rows; both fit every buildable column (`lib_smem_page_fits_for` is
still asked, and a column where the forced page does not fit runs the
first-round stash kernels). Grid `B * nh * ceil(L / TQ)` for kernel A
(384 at 64 rows, 768 at 32) and `B * nh * ceil(L / 256)` for kernel B
(96).

Correction to 11.3. The page-only occupancy count in 11.1 does not favor
32 rows on AMD. Resident blocks per CU are `min(8, 65536 // page)`, so 64
rows give 3 blocks and 192 rows in flight per CU, while 32 rows give 4
blocks and 128 rows. The case for 32 rows would have to come from register
counts or staging latency, not from the page, so the leg prices both
(`_r32`, `_r64`) and the matrix row's AMD value of 32 is a placeholder
until it does. The same count applies to 2531 (17,152 B at 64 rows, 3
blocks, 192 rows; 12,672 B at 32 rows, 5 blocks, 160 rows), and 11.4's
12,736 B is 12,672 B by the kernel's own allocations (8,192 + 4,224 +
256).

### 12.3 What was built, per deviation

- DEVIATION 2528, BUILT (source, not compiled). In
  `transformer/impl/llama/fused_attention.mojo`: the two kernels of
  12.2; the generic `_launch_bwd_ztiled[HD, TQZ, ZSAB]`; in
  `fused_backward_launch_arm` a `comptime if ATTN_ARM_TRIAL` branch
  before the first-round branch, whose condition became
  `if want_stash and not ran_arm:` (the only edit to shipped lines; on a
  shipped build the trial branch does not exist, so `ran_arm` is `False`
  there and the branch runs exactly as before); the arm bits, names and
  parser of 12.1; `fused_attention_arm_reach_bit`;
  `fused_attention_zdot_rows` (the knob, else the matrix row, else 0 when
  the page does not fit). In `checks/kernel_matrix.mojo`:
  `attn_zdot_rows_per_block_for[column]`. The forward launcher is
  untouched. New timer lines under `MOJOLEARN_ATTN_PHASE_TIMERS`:
  `attn.bwd_ydy_tiled`, `attn.bwd_zfold` (then the existing
  `attn.bwd_dq_tiled`, `attn.bwd_dkdv_tiled`).
- (Section 14 later built 2533, 2531 and 2530 as trial arms, source only,
  on the orchestrator's instruction after section 13's H100 reading; the
  three bullets below are this round's record.)
- DEVIATION 2531, NOT BUILT. 2528's `_r32` and `_r64` price on the MI325X
  answers the question 2531 depends on (whether 32-row blocks pay on
  that column), and the page-only count in 12.2 does not favor them;
  building a second 32-row kernel before that number would be building to
  a reading the counts contradict. Its bit stays reserved.
- DEVIATION 2533, NOT BUILT (order 11.2: after 2531). Reading confirmed
  for when it is built: every operand of the tiled dq and dk/dv steps,
  the forward context step and kernel B's step is already flushed
  (staged through `ftz`, or produced by `_pmul` or `ftz(div)`; a
  sabotage flip of a normal stays normal and of a zero is `2^-100`), so
  `_step_preflushed` gives the same bits. It needs new instantiations
  (a seam parameter on copies), never an edit of the shipped kernels.
- DEVIATION 2530, NOT BUILT (order 11.2: after 2533, 32 rows only).

### 12.4 Checks, harness and leg

- `transformer/checks/transformer_attention_arms_check.mojo`: a host-only
  NAMES section (13 spellings round-trip, the six arms times four
  sabotage combinations round-trip as values, eight invalid spellings
  refused), then six arms on the 15 cases: the first three, `stash_tiled`
  itself, `stash_tiled_ztiled_r64` and `stash_tiled_ztiled_r32`. Reach is
  per branch: a first-round forward bit must move the forward, a
  first-round backward bit or 2528 under `sabotage_new` must move the
  backward (zdot now counted too), a `sabotage_new` run of an arm with no
  second-round forward bit (`ATTN_ARM_NEW_FWD_BITS`, 0 today) must move
  no forward cell, and at head dims other than 64 a sabotage must move
  nothing.
- `bench/attention_step_price_main.mojo`: names through
  `fused_attention_arm_parse` (and a round-trip assert), two `PATH` lines
  (arm, resolved `zdot_rows`, `reach_bit`), reach with the arm's reach
  bit, and a failure when a backward-only second-round sabotage moves a
  forward cell. `REACH` lines keep their first fields and add
  `reach_bit`, `forward_moved`, `backward_moved`.
  `MOJOLEARN_ATTN_BASELINE` already existed and is unchanged.
- `tools/attention_step_leg.sh` (additions only, the vendor-agnostic body
  and the enwik8 and pile_github corpora of 7c0a5af4 kept): `BASE` from
  `MOJOLEARN_ATTN_BASELINE`, passed explicitly to every smoke, price and
  timer run; the timer loop runs `$BASE` instead of `baseline`;
  `lm_summary.tsv` compares witnesses with `lm-$BASE-<corpus>` and prints
  the reference; `gate.txt` records `baseline_arm`; the header names the
  new arms and the AMD env.

### 12.5 Risks not resolvable without a build

1. Nothing was compiled. The likeliest compile faults are in the generic
   kernel's comptime arithmetic (`NR * STRIDE`, `NR * KS // 256`) and the
   `mut` DeviceBuffer pass-through of the launch helper; the M4 arms
   check build (12.6 step 2) finds them in minutes.
2. Kernel B reads two global floats per step on 256 threads with no
   staging, so it may be bandwidth-bound rather than serial-bound. The
   `attn.bwd_zfold` timer line sizes it; a row-tiled staging of y and dy
   is the follow-on if it is.
3. The trial bindings on the leg now instantiate four more kernel A
   pipelines (32 and 64 rows, clean and sabotage) and kernel B, so the
   binding builds take longer (they are not under the leg's `timeout`).
4. The AMD value of `attn_zdot_rows_per_block_for` (32) is a placeholder;
   only the forced `_r32` and `_r64` names are evidence-bearing until the
   leg flips the row.
5. Whether kernel A's 32 register accumulators plus 24 register loads
   per thread (at 64 rows) stay register-resident on each column is not
   known from the source (the GEMM lane's resource read answers it on
   NVIDIA).

### 12.6 RUN OWED, in order (M4 light checks, one at a time)

1. The shipped path still builds and is still bit-identical (no trial
   define):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check` (expect PASS, 15 cases, the
   head_dim 64 cases RAN through `stash_tiled`).
2. The arms gate, with the new arms and the names section:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check` (expect
   `PASS, names inverse, 15 cases x 6 arms`).
3. The harness builds and runs 2528 against stash_tiled on the generator
   kinds at a small shape (correctness only, never a timing):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`
   then
   `MOJOLEARN_ATTN_BASELINE=stash_tiled MOJOLEARN_ATTN_ARM=stash_tiled_ztiled_r32 MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`
   and the same with `MOJOLEARN_ATTN_ARM=stash_tiled_ztiled_r64 MOJOLEARN_ATTN_ORACLE=0`
   (expect every `BITS ... _vs_stash_tiled` MATCH,
   `REACH ... clean_restored=True reach_bit=...+sabotage_new forward_moved=0`).

### 12.7 The AMD legs (MI325X, DigitalOcean, from a detached worktree at the lane's commit)

First, 11.2 item 0 (stash_tiled against baseline, no second-round arm),
the example already in the leg header:

    MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
    MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh \
    MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_ATTN_LEG_ARMS=stash_tiled MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1" \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-attention-step \
    bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates

Then DEVIATION 2528 at both geometries against the shipped default, with
the lean step witnesses of all three arms on both corpora:

    MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
    MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh \
    MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_ATTN_BASELINE=stash_tiled MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_ztiled_r64,stash_tiled_ztiled_r32 MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled,stash_tiled_ztiled_r64,stash_tiled_ztiled_r32 MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1" \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-attention-ztiled \
    bash tools/do_extra_leg.sh amd --minutes 90 --skip-gates

Every word is inside the runner's value set (letters, digits, `_.,:/=-`).
The second leg runs twelve probes (three arms, two corpora, lean and
timing); if the lease must be 60 minutes, drop the slower geometry from
`MOJOLEARN_ATTN_LEG_LM_ARMS` after the price lines name it, never
`stash_tiled` (it is the witness reference). Gates are section 6 with
`stash_tiled` in place of `baseline`, and the flip rule is
ENGINEERING_RULES 9 (the geometric mean of the enwik8 and pile_github
ratios below 1, loss not worse on either); a flip also writes
`attn_zdot_rows_per_block_for`'s AMD value from the same leg.

## 13. H100 leg on the benchmark corpora (2026-09-11, RunPod, commit cd086f67): 2528 NO FLIP on NVIDIA

Run on NVIDIA first because the shared DigitalOcean GPU was held by the trees
lane (Andrew: "use runpod then and just do nvidia for now"). The AMD legs in
12.7 are still owed; this is the confirmation column, not the deciding one.

Evidence: bench/results/e1g/2026-09-11_133041-nvidia-h100-attention-torch
(`NVIDIA H100 80GB HBM3`, pod quf7729vxu5q66 terminated and verified, 24
minutes on the pod). Operand dumps (48 MB) are outside the repo at
~/mojolearn-evidence/attention-step-2026-09-11_133041-nvidia-h100-attention-torch/,
sha256 list beside the evidence. `tools/gemm_remote_leg.sh` has no extra-env
plumbing, so the settings live in the body the leg copied to `extra_body.sh`:
`MOJOLEARN_ATTN_LEG_ARMS=stash_tiled,stash_tiled_ztiled_r64,stash_tiled_ztiled_r32`,
`MOJOLEARN_ATTN_LEG_LM_ARMS=baseline,stash_tiled,stash_tiled_ztiled_r64`,
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`, `MOJOLEARN_COMPILE_JOBS=8`, then the torch
opponent leg on the same box (bench/OPPONENT_REFERENCE.md).

- `status.tsv`: all 27 items exit 0. `arms-check`: PASS, names inverse, 15
  cases x 6 arms, reach proven per branch on the H100.
- Price on real activations (`price_tables.txt`, fwd+bwd median ms, enwik8 /
  Pile GitHub; each arm ran against its own baseline run):

  | arm | fwd+bwd | bwd (derived) | fwd+bwd ratio vs baseline |
  |---|---|---|---|
  | baseline | 28.57 to 28.87 | 21.82 to 21.97 | 1.0 |
  | stash_tiled (shipped) | 13.62 / 13.64 | 10.46 / 10.49 | 2.102 / 2.094 |
  | stash_tiled_ztiled_r64 | 16.33 / 16.33 | 13.15 / 13.14 | 1.759 / 1.768 |
  | stash_tiled_ztiled_r32 | 16.79 / 16.78 | 13.60 / 13.59 | 1.702 / 1.708 |

  Every `REACH` line: `clean_restored=True`, `sabotage_new` moves only the
  backward (`forward_moved=0`) on hashed, heavytail and both dump kinds.
- Lean LM step (`lm_summary.tsv`, steady median seconds, enwik8 / Pile
  GitHub), witnesses equal to baseline on every step for every arm:
  baseline 0.5625 / 0.5628; stash_tiled 0.3835 / 0.3833 (0.682 / 0.681 of
  baseline, the default flip holds on the benchmark corpora);
  stash_tiled_ztiled_r64 0.4157 / 0.4154, which is **1.084 / 1.084 of
  stash_tiled, geomean 1.084: NO FLIP.** r32 was not probed; its price is
  worse than r64's.
- Timers: `attn.bwd_zdot_stash` 89.9 ms (stash_tiled) against
  `attn.bwd_ydy_tiled` 115.5 ms plus `attn.bwd_zfold` 6.6 ms (r64). The
  register-blocked y/dy kernel costs 32 ms more per step than the stash
  read it replaces, which is the whole step difference.

Reading: on the H100, 2528 at either row count is slower than the shipped
`stash_tiled` backward, and 32 rows is slower than 64. It stays a trial arm.
Whether the MI325X prices it differently (its kernel matrix row is 32) is the
12.7 second leg; 2531 still waits on that AMD reading.

## 14. Third round, source built (2026-09-11, worktree lane, nothing run): DEVIATIONS 2533, 2531 and 2530 as trial arms

STATUS: source only. Nothing in this section was compiled or run (no build on
the Mac, by rule; the orchestrator's M4 build in 14.8 is the first compile).
The shipped default is still `stash_tiled`. A build without
`-D MOJOLEARN_ATTN_ARM_TRIAL=1` compiles none of the kernels below and runs
what it ran before. Order of work, per 11.2 as section 13 adjusts it: 2533
first, because section 13 puts the NVIDIA floor in the backward and 2533
applies to the shipped stash_tiled backward folds with or without 2528 at
either row count; then 2531; then 2530 on 2531's 32-row geometry.

### 14.1 Arm bits and names (extends 12.1; the parser and the name function stay inverses)

| bit | constant | name token | meaning |
|---:|---|---|---|
| 8 | `ATTN_ARM_BWD_ZTILED` | `_ztiled` | DEVIATION 2528 (section 12) |
| 16 | `ATTN_ARM_SABOTAGE` | `+sabotage` | flips in the first-round kernel instantiations the arm runs |
| 32 | `ATTN_ARM_FWD_QRES` | `_qres` | DEVIATION 2530, needs `_fgrid_r32` |
| 64 | `ATTN_ARM_FWD_GRID` | `_fgrid` | DEVIATION 2531, needs the forward score stash |
| 128 | `ATTN_ARM_SABOTAGE_NEW` | `+sabotage_new` | flips in the second-round kernels only |
| 256, 512 | `ATTN_ARM_ZROWS32`, `ATTN_ARM_ZROWS64` | `_ztiled_r32`, `_ztiled_r64` | 2528 geometry |
| 1024 | `ATTN_ARM_PREFLUSH` | `_pf` | DEVIATION 2533 |
| 2048, 4096 | `ATTN_ARM_FROWS32`, `ATTN_ARM_FROWS64` | `_fgrid_r32`, `_fgrid_r64` | 2531 geometry |

Grammar: base, `_ztiled[_r32|_r64]`, `_fgrid[_r32|_r64]`, `_qres`, `_pf`,
`+sabotage`, `+sabotage_new`. A rows token belongs to the group token right
before it, so section 12's names are unchanged. The parser refuses an unknown
base, a token out of order (`stash_tiled_pf_fgrid_r32`), `_ztiled` without the
tiled stash backward, `_fgrid` without `fwd_sstash` (`bwd_stash_tiled_fgrid`),
`_qres` without `_fgrid_r32` (`stash_tiled_fgrid_qres`,
`stash_tiled_fgrid_r64_qres`), `_pf` with no stash kernel (`baseline_pf`) and
`_pf` with the 2525 non-tiled stash backward (`stash_pf`, `bwd_stash_pf`),
which has no preflushed copy. The name function spells a rows knob without its
group bit as `_bits<N>`. Without that, an invalid arm such as
`_ztiled` plus the forward 32-row knob would have named itself
`..._ztiled_r32` and read back as a different, valid arm.

`_pf` acts on whichever stash kernels the arm runs: `fwd_sstash_pf` is forward
only, `bwd_stash_tiled_pf` backward only, `stash_tiled_pf` both.
`_fgrid` without a rows token reads the new kernel-matrix SCHEDULING row
`attn_fwd_rows_per_block_for[column]` (32 on AMD, 64 elsewhere, the shipped
value; UNMEASURED); `_fgrid_r32` and `_fgrid_r64` force the geometry on any
column. `_pf` without `_fgrid` runs the second-round forward at 64 rows.

`+sabotage` is narrowed in wording, not in behavior for any earlier arm: it
flips only first-round kernel instantiations that the arm runs, and a
second-round copy that replaces a first-round kernel carries no first-round
flip. So `stash_tiled_fgrid_r32+sabotage` flips the backward only, and
`stash_tiled_ztiled_r64_pf+sabotage` flips nothing (every backward kernel it
runs is a second-round one). Without `_pf`, 2528 still honors `+sabotage` in
the tiled folds as section 12 built it. Reach for every second-round arm uses
`+sabotage_new`, as before.

### 14.2 DEVIATION 2533, preflushed seams: identity argument (written before the code)

Claim: at every rewritten step, `_step(a, b, acc)` and
`_step_preflushed(a, b, acc)` return the same bits.

1. By the two bodies, `_step(a, b, acc)` is `_step_preflushed(ftz(a), ftz(b),
   acc)` on every column (NVIDIA: `fma.rn` then `mul.rn.ftz` by 1.0 in both;
   the others: `ftz(identical_mul_add(...))` in both).
2. `ftz` (`checks/numerics.mojo`) changes a value only when its exponent field
   is zero and its mantissa is not, a nonzero subnormal, which it maps to the
   zero of its sign. Zeros, normals, infinities and NaNs pass unchanged. So
   when neither operand is a nonzero subnormal, `ftz(a) == a` and
   `ftz(b) == b` bitwise, and the two spellings agree.
3. No rewritten operand is a nonzero subnormal. Each is the output of `ftz`,
   of a seam whose last operation is a flush (`_step`, `_step_preflushed`,
   `_pmul`: `mul.rn.ftz` or `ftz`), or of `_flip_ulp` of such a value (bit 0 of
   a normal leaves its exponent field nonzero; a zero becomes 2^-100, a
   normal). Per site:
   - zdot stash, y dot: `vec` holds `ftz(q)`, `ks` holds staged `ftz(k)`.
     dy dot: `vec` holds `ftz(dctx)`, `vs` staged `ftz(v)`. z fold: `dys` is
     `ftz(dy)`, `ys` is `ftz(identical_div(...))`.
   - 2528 kernel B, z fold: `dy_st` is kernel A's `ftz(ddots)`, `y_st` its
     `ftz(identical_div(...))` or that value's sabotage flip.
   - tiled dq fold: `dcell` is `_pmul(ftz(ds), scale)` (or 0.0 at a masked
     cell, which the fold never steps); `ka` is staged `ftz(k)`.
   - tiled dk and dv folds: the staged `dcell` is the value the dq kernel wrote
     over `dy_st` at that visible cell (a `_pmul` output); the staged `y` is the
     zdot kernel's (or kernel A's) `ftz(identical_div(...))` or its flip; `qa`
     and `da` are staged `ftz(q)` and `ftz(dctx)`.
   - forward context chain: `w` is `ftz(identical_div(ftz(e), ftz(denom)))`;
     `va` is staged `ftz(v)`.
4. A masked cell never reaches a rewritten step (the visibility tests are
   copied), so the zeros that stand in for masked cells do not matter.
5. Everything else is the shipped code: the same staging, chain order, corner
   tests and stores. The rewritten steps live in copies
   (`fused_bwd_zdot_stash_pf_kernel[HD, TQ, SABN]`,
   `fused_bwd_dq_tiled_pf_kernel[HD]`, `fused_bwd_dkdv_tiled_pf_kernel[HD]`),
   in the PF parameter of the trial-only 2528 kernel B
   (`fused_bwd_zfold_kernel[TZ, PF]`) and in the PF parameter of the
   second-round forward (14.3). The shipped kernels are not edited.

Not rewritten: the per-cell `_pmul` (one seam per cell against 64 or 128 per
cell in the chains) and the forward denominator `ftz(ftz(dacc) + ftz(e))`,
which is not an fma chain.

Counted expectation (no number): each rewritten term drops two software `ftz`
calls (a bit test and a select each). Per visible cell that is 128 terms plus
the z fold in the zdot stash kernel (89.9 ms of section 13's step), 64 in the
dq fold, 128 in the dk/dv folds and 64 in the forward context chain. If the
backend already removed those flushes, 2533 prices at 1.00.

Sabotage (reach, `+sabotage_new`). Backward: the zdot stash copy stores each
row's zdot flipped one ulp, so zdot moves (dq and dk move through z) and dv
does not (dv reads y and dctx only). The dq and dk/dv copies have no flip of
their own: `_launch_bwd_stash_tiled_pf[HD, ZSAB]` launches them
unconditionally in the same comptime instantiation, so the zdot flip proves
all three ran (the 2528 kernel B precedent). Forward: 14.3's PF site.

### 14.3 DEVIATION 2531, forward grid: identity argument (written before the code)

Kernel `fused_attn_forward_r2_kernel[HD, TQ, QRES, PF, SABN]`, a generic copy
of `fused_attn_forward_regblocked_sstash_kernel`, instantiated at HD 64 with TQ
64 or 32. At TQ 32 thread `(tr, tc)` holds rows `tr + 16u` (u < 2), keys
`tc + 16v` (v < 2) of each 32-key block iteration and context columns
`tc + 16v` (v < 4): 4 dot and 8 context accumulators per thread, grid
`B * nh * ceil(L / 32)`, launched by the trial-only `_launch_fwd_r2`.

1. Scores. Each cell's dot is `_step_preflushed(ftz(q[t, p]), ftz(k[j, p]),
   dot)` over p ascending from +0.0 across the four 16-wide windows, on the
   same staged values the shipped kernel stages. Only the thread holding it
   changes.
2. Row maximum. `mpart[u]` folds one row's cells for two keys per key block;
   16 slots per row are then folded by threads `tid < TQ`. Keys per thread are
   unchanged, so the grouping is in fact the shipped one. Either way it is an
   `identical_fmax` fold over values that are never -0.0 (the `+ 0.0` mask
   add) and never NaN (the regime), so its grouping is free (contract 5.1).
3. Stash, exp, weight. The shipped sstash code on the same stash cell
   `stbase + t * s + j` and the same stats slots (`stats[r]` the maximum,
   `stats[TQ + r]` the denominator, r the row inside the block, written once
   each before they are read).
4. Denominator. Thread `tid < TQ` folds row `t0 + tid` over key blocks
   ascending and keys ascending from +0.0: the shipped serial chain.
5. Context. `cacc[u * 4 + v]` folds `(row tr + 16u, column tc + 16v)` over key
   blocks ascending and keys ascending with `_step(w, v, acc)`, or
   `_step_preflushed` under PF by 14.2: the shipped chain.
6. Corner. The shipped test per (row, column) chain.
7. Rows per block is a partition of consecutive rows. Every row lies in
   exactly one block, the loop bounds are block-uniform, and every thread
   reaches every barrier.

Page: 12,672 B at 32 rows, 17,152 B at 64 (the shipped page). At TQ 64 with
QRES and PF off the copy runs the shipped sstash arithmetic, so
`stash_tiled_fgrid_r64` prices the copy itself against `stash_tiled`.

Sabotage (`+sabotage_new`, not QRES, not PF): the pass-3 weight stored in the
tile is flipped one ulp, so ctx moves and amax and denom hold. The first-round
sstash sabotage, by contrast, flips the pass-1 score and moves denom.

### 14.4 DEVIATION 2530, forward Q residency: identity argument (written before the code)

QRES, at TQ 32 only (the parser requires `_fgrid_r32`, and
`fused_attention_fwd_rows` returns 0 otherwise). Before pass 1 the block stages
its query rows once, coalesced `[32][64]` through `ftz` (8 slots per thread,
one barrier). Each p window of pass 1 stages K only, `[32][16]` at stride 20,
at offset 2,048 of the same page (2 slots per thread instead of 4). Pass 3
stages V into `[0, 2048)`, over the Q rows.

1. The dots read `stg[(tr + 16u) * 64 + pw * 16 + p]`, which is
   `ftz(q[t, pw * 16 + p])`: the value the per-window staging stores, since
   `ftz` of a global value is the same whenever it is computed. The K
   operands, the p order and the chain are unchanged.
2. Q is read in pass 1 only; passes 2 and 3 read the stash. V overwriting the
   Q rows in pass 3 therefore changes no value a later read sees, and V's
   range `[0, 2048)` does not reach the K rows at 2,048.
3. Everything else is 14.3.

Page: 15,232 B (2,048 Q/V + 640 K + 1,056 tile + 64 stats floats). Counted:
two staging loads per thread per window instead of four, plus one staging
round trip and one barrier per block. Section 11.5 expected a small effect on
the H100.

Sabotage (`+sabotage_new`): every staged Q value is flipped one ulp (a zero
becomes 2^-100), so the scores move and amax or denom move. That distinguishes
it from the 2531 and 2533 forward flips, which hold both. In a `_qres_pf` arm
only the Q flip is compiled.

### 14.5 Reach and attribution (the arms check and the harness)

| second-round flip | moves | must hold |
|---|---|---|
| 2528 kernel A, stored y | zdot, dq, dk, dv | forward |
| 2533 backward, stored zdot | zdot (dq, dk through z) | dv, forward |
| 2533 forward, ctx of context lane 0 | ctx columns 0 to 15 | amax, denom, ctx columns 16 to 63, backward |
| 2531, pass-3 weight in the tile | ctx | amax, denom, backward |
| 2530, every staged Q value | amax or denom, and ctx | nothing in the forward |

Per branch, under `+sabotage_new`: an arm with a second-round forward kernel
(`fused_attention_arm_new_forward`) must move the forward, and one without
must move no forward cell. The same holds for the backward
(`fused_attention_arm_new_backward`). In the arms check the backward reads the
eager amax and denom, so both directions are independent. In the harness the
backward reads the sabotaged forward's amax and denom, so backward-must-hold
and the 2533 backward attribution are asserted only when amax and denom held.
Attribution follows the table: `_qres` must move amax or denom; a non-QRES
second-round forward must hold them; the 2533 forward flip must hold ctx
columns 16 and up; the 2533 backward flip must move zdot and hold dv when
2528 is not in the arm. A `_ztiled..._pf` arm is attributed per direction
only, because 2528's y flip moves every backward buffer. The geometry (32 or
64 rows) is not attributable from the outputs, the same as 2528's. The
harness prints it in its `PATH` line (`fwd_rows=`, `preflush=`, beside
`zdot_rows=`).

### 14.6 What was built, per file

- `transformer/impl/llama/fused_attention.mojo`:
  - the bits, parser and name function of 14.1;
  - `fused_attention_arm_new_forward` and `fused_attention_arm_new_backward`;
  - `_fwd_r2_page_bytes`, `fused_attention_fwd_rows` (the knob, else the
    matrix row, else 0 when the page does not fit);
  - the kernels of 14.2 to 14.4;
  - the generic launch helpers `_launch_fwd_r2[HD, TQ, QRES, PF, SABN]` and
    `_launch_bwd_stash_tiled_pf[HD, ZSAB]`, and
    `_launch_bwd_ztiled[HD, TQZ, ZSAB, PF]`.

  In `fused_forward_launch_arm`, a `comptime if ATTN_ARM_TRIAL` branch
  precedes the first-round branch, whose condition became
  `if want_sstash and not ran_arm:`. That is the only edit to a shipped
  line, the same pattern 2528 used in the backward; on a shipped build
  `ran_arm` is `False` there. In `fused_backward_launch_arm`, the trial
  branch dispatches 2528 with PF, then `_pf` alone on the tiled stash
  backward. New timer lines (under `MOJOLEARN_ATTN_PHASE_TIMERS`):
  `attn.fwd_r2_kernel`, `attn.bwd_zdot_stash_pf`, `attn.bwd_dq_tiled_pf`,
  `attn.bwd_dkdv_tiled_pf`, `attn.bwd_zfold_pf`.
- `checks/kernel_matrix.mojo`: `attn_fwd_rows_per_block_for[column]`.
- `transformer/checks/transformer_attention_arms_check.mojo`: 13 arms
  (section 12's six, plus `stash_tiled_ztiled_r64_pf`, `stash_tiled_pf`,
  `stash_tiled_fgrid_r64`, `stash_tiled_fgrid_r32`,
  `stash_tiled_fgrid_r32_pf`, `stash_tiled_fgrid_r32_qres`,
  `stash_tiled_fgrid_r32_qres_pf`). Together they run every second-round
  forward instantiation (rows 64 and 32, with and without Q residency and
  preflush) and both second-round backward launches. The names section
  has 24 spellings round-tripping and 18 invalid spellings refused; reach is
  checked both directions per branch and attributed as in 14.5. The hd 16,
  24 and 128 cases assert that every arm's sabotage moves nothing.
- `bench/attention_step_price_main.mojo`: the `PATH` fields, the `REACH`
  fields `amax_denom_moved`, `ctx_moved_columns_16_up`, `zdot_moved`,
  `dv_moved`, and the per-branch and attribution failures of 14.5.
- `tools/attention_step_leg.sh`: header and `gate.txt` only (names pass
  through).
- `tools/attention_round3_leg.sh`: the leg body wrapper (14.9).

A trial build now instantiates 18 more kernel pipelines: 12 forward (six
geometry and flag combinations times the sabotage flag), two zdot stash
copies, one dq copy, one dk/dv copy and two kernel B variants.

### 14.7 Risks only a build or a box can settle

1. Nothing was compiled. The likeliest faults are in the generic forward
   kernel's comptime expressions (`SPG` and `KOFF` as conditional
   expressions, `comptime if SABN and PF and not QRES and v == 0` on a
   comptime-for variable), the `comptime if PF` blocks that bind kernel
   aliases in host helpers (the early `return` inside one in
   `_launch_bwd_ztiled`), and the widened helper signatures.
2. Build time: every trial build (arms check, harness, both leg bindings)
   compiles 18 more pipelines; the binding builds are not under the leg's
   `timeout`.
3. 2533 may price at 1.00 if the backend already removed the operand flushes;
   only the leg says.
4. 2531 at 32 rows doubles the blocks and the per-block fixed work (row
   ranges, the reductions' barriers). On the H100, 2528's r32 was slower than
   r64, so the same is plausible here. Whether AMD's partitioned shared memory
   reverses that is the AMD leg's question.
5. 2530 adds a round trip and a barrier per block and halves per-window
   staging loads. 11.5 expected a small effect on the H100.
6. Attribution assumes a one-ulp weight flip moves some ctx cell and a one-ulp
   Q flip moves some amax or denom cell. Flips of this kind held for the
   first-round and 2528 sabotages on the H100, and a flip that moves nothing
   fails reach loudly, never silently.
7. `attn_fwd_rows_per_block_for`'s AMD value (32) is a placeholder; only the
   forced `_fgrid_r32` and `_fgrid_r64` names are evidence-bearing until a leg
   flips the row.
8. The Hot Aisle runner (`tools/hotaisle_leg.sh`) is a skeleton that refuses
   every mode; the AMD command in 14.9 waits for it. The MI300X arch is taken
   as gfx942 (docs/MLP_AMD_NEXT_RUN.md), and the runner's design refuses a
   mismatch with the box.

### 14.8 RUN OWED, in order (M4 light checks, one at a time)

1. The shipped path still builds and is still bit-identical (no trial
   define):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check` (expect
   `transformer_fused_check: PASS, 15 cases`, the head_dim 64 cases RAN
   through `stash_tiled`).
2. The arms gate:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check` (expect
   `names: 24 spellings and 52 arm values round-trip, 18 invalid spellings refused`
   and `transformer_attention_arms_check: PASS, names inverse, 15 cases x 13 arms`,
   with per-branch `REACH` lines at head_dim 64: `forward_flipped_cells`,
   `backward_flipped_cells`, `forward_cells_that_must_hold_moved=0`,
   `backward_cells_that_must_hold_moved=0`, `amax_denom_moved`,
   `ctx_moved_columns_16_up=0` for the `_pf` forward arms, `zdot_moved` and
   `dv_moved=0` for `stash_tiled_pf` and the `_fgrid..._pf` arms).
3. The harness at a small shape, correctness only
   (`MOJOLEARN_ATTN_TIMING=0`), against stash_tiled:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`
   then
   `MOJOLEARN_ATTN_BASELINE=stash_tiled MOJOLEARN_ATTN_ARM=stash_tiled_pf MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`
   and the same with `MOJOLEARN_ATTN_ORACLE=0` for
   `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32`, then
   `stash_tiled_fgrid_r32_qres_pf`, then `stash_tiled_ztiled_r64_pf`. Expect
   `PATH candidate ... fwd_rows=64 preflush=True` (`fwd_rows=32` for the
   `_fgrid_r32` arms), every `BITS ... _vs_stash_tiled` MATCH,
   `REACH ... clean_restored=True reach_bit=...+sabotage_new`, and
   `attention_step_price: PASS`.

### 14.9 The legs

The body is `tools/attention_round3_leg.sh`. It exports
`MOJOLEARN_ATTN_BASELINE=stash_tiled`,
`MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_pf,stash_tiled_fgrid_r64,stash_tiled_fgrid_r32,stash_tiled_fgrid_r32_qres,stash_tiled_fgrid_r32_qres_pf`,
`MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled,stash_tiled_pf,stash_tiled_fgrid_r32,stash_tiled_fgrid_r32_qres_pf`,
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1` and `MOJOLEARN_COMPILE_JOBS=8`, each only
when unset, refuses LM arms without the baseline, and runs
`sh tools/attention_step_leg.sh`. That is five smokes, five prices on both
corpora's activations and 16 LM probes, against section 13's 27 items in 24
pod minutes.

NVIDIA, RunPod (the measurement column while ENGINEERING_RULES 10's NVIDIA
clause is in force), from a `git worktree add --detach` checkout at the lane's
merge commit:

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GPU_ARCHS=sm_90a \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_round3_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-round3 \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
        --gpu "NVIDIA H100 80GB HBM3"

AMD, Hot Aisle MI300X, once `tools/hotaisle_leg.sh` is built (its interface
mirrors `tools/do_extra_leg.sh`; the extra env overrides the wrapper's
defaults by name):

    MOJOLEARN_HOTAISLE_TEAM=<team handle> \
    MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_round3_leg.sh \
    MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_ATTN_BASELINE=stash_tiled MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_pf,stash_tiled_fgrid_r64,stash_tiled_fgrid_r32,stash_tiled_fgrid_r32_qres,stash_tiled_fgrid_r32_qres_pf MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled,stash_tiled_pf,stash_tiled_fgrid_r32,stash_tiled_fgrid_r32_qres_pf MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 MOJOLEARN_COMPILE_JOBS=8" \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-attention-round3 \
    bash tools/hotaisle_leg.sh --rent --minutes 90 --skip-gates --gpu MI300X

Every value is inside the runners' value set (letters, digits, `_.,:/=-`).
Gates: section 6 with `stash_tiled` in place of `baseline` (arms check exit 0,
smokes exit 0, every `BITS ... _vs_stash_tiled` MATCH on both corpora's
activations with reach proven, lean steps `limited: false`). Flip rule,
ENGINEERING_RULES 9: for an arm, the geometric mean of its enwik8 and
pilegithub lean step ratios (`steady_median_seconds` of `lm-<arm>-<corpus>`
over `lm-stash_tiled-<corpus>`, same leg) below 1, and
`witnesses_equal_baseline=True` for every step on both corpora. A flip edits
`ATTN_ARM_DEFAULT` (and, for `_fgrid`, `attn_fwd_rows_per_block_for`'s row for
that column from the same leg) in the same session as the evidence filing. If
the lease runs short, drop LM arms after the price lines rank them, never
`stash_tiled` (the witness reference).

## 15. Round 3 on the H100 and the NVIDIA flip (2026-09-11): DEVIATION 2534, source built, nothing run

STATUS: 15.1 is a leg's reading; everything from 15.2 on is source only.
Nothing in 15.2 or 15.3 was compiled or run (no build on the Mac, by rule).
The orchestrator's M4 commands in 15.4 are the first compile.

### 15.1 The H100 leg (RunPod, commit 5bcfa71d)

Evidence
`bench/results/e1g/2026-09-11_154257-nvidia-h100-80gb-hbm3-attention-round3/remote/attention-step/`
(filed on main by the orchestrator; the figures below are the orchestrator's
reading of it). One pod, `NVIDIA H100 80GB HBM3`, SM clock 1980 MHz, body
`tools/attention_round3_leg.sh` (14.9), every arm against the shipped
`stash_tiled`, lean LM step on enwik8 and Pile GitHub, every step witness
equal to stash_tiled's on both corpora.

| arm | enwik8 s | Pile GitHub s | ratio to stash_tiled (enwik8 / Pile GitHub) |
|---|---:|---:|---|
| stash_tiled (shipped) | 0.3845 | 0.3819 | 1.000 / 1.000 |
| stash_tiled_fgrid_r32 | 0.3718 | 0.3716 | 0.967 / 0.973 |
| stash_tiled_pf | 0.3488 | 0.3473 | 0.907 / 0.909 |
| stash_tiled_fgrid_r32_qres_pf | 0.3346 | 0.3340 | 0.870 / 0.875 |

Price on real activations, fwd+bwd, stash_tiled over the arm (the TABLE
ratio): `fgrid_r32` 1.07, `fgrid_r32_qres` 1.08, `pf` 1.27,
`fgrid_r32_qres_pf` 1.41, `fgrid_r64` 1.00. Lean step timers, stash_tiled to
`stash_tiled_fgrid_r32_qres_pf`: backward zdot stash 89.9 to 66.2 ms,
forward kernel 37.0 to 20.9 ms, dq tiled 15.5 to 12.5 ms, dk/dv tiled 18.5
to 12.4 ms.

Reading. On the H100 all three third-round deviations lower the step, and
the composed arm is the lowest on both corpora. `fgrid_r64` prices at 1.00,
so the forward copy at the shipped geometry costs nothing, and the 32-row
geometry is what pays. The winner's geometric mean ratio is 0.872, below 1,
with witnesses equal: ENGINEERING_RULES 9 flips it in this session. The
MI300X verdict is still owed, so AMD does not flip.

### 15.2 The flip (DEVIATION 2534)

`checks/kernel_matrix.mojo`, next to the attention rows (not at the end of
the file; the GEMM `ksplit` flip lane, DEVIATION 2595, appends a row too):

- `attn_default_arm_for[column]`, a ROUTING row returning an arm word:
  NVIDIA `ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF` (3175, measured,
  15.1); AMD `ATTN_DEFAULT_WORD_STASH_TILED` (7) with the comment that the
  MI300X leg decides it; Apple and every other column 7. The matrix cannot
  import `fused_attention.mojo` (that file imports the matrix), so the words
  are literals there and `fused_attention.mojo` asserts at build time that
  each equals its own composition (`ATTN_ARM_STASH_TILED`,
  `ATTN_ARM_R3_DEFAULT`).
- `-D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1` returns the NVIDIA word on
  every column: a check knob (the `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL`
  pattern) so a no-trial build on the M4 compiles and runs the shipped
  round 3 branches. Never a shipped build.
- `attn_fwd_rows_per_block_for`: NVIDIA 64 to 32 (measured on the H100,
  15.1); AMD stays 32 (placeholder until the MI300X leg); others 64. The
  default forces `_fgrid_r32`, so this row moves only the bare `_fgrid`
  token, never a shipped path.

`transformer/impl/llama/fused_attention.mojo` (no vendor branch; the file
reads the row through `TARGET_COLUMN` like every other row):

- `ATTN_ARM_DEFAULT = attn_default_arm_for[TARGET_COLUMN]()`. Build-time
  asserts in `fused_attention_arm_from_env` (every launcher calls it): the
  two literal words equal this file's bits; the default carries no sabotage
  bit and no DEVIATION 2528 bit (`ATTN_ARM_DEFAULT_REFUSED_BITS`; 2528's
  kernels are not compiled on a shipped build and did not flip, section 13);
  a default with a second-round forward has a page that fits the column.
- A shipped build compiles, beyond the first-round clean kernels it already
  compiled, exactly the clean second-round instantiations its column's
  default needs, one per direction: `ATTN_SHIPPED_FWD_R2` launches
  `_launch_fwd_r2[64, ATTN_DEFAULT_FWD_ROWS, ATTN_DEFAULT_FWD_QRES,
  ATTN_DEFAULT_FWD_PF, False]` (on NVIDIA `[64, 32, True, True, False]`)
  when `_attn_fwd_r2_key(arm)` equals the default's key, and
  `ATTN_SHIPPED_BWD_PF` launches `_launch_bwd_stash_tiled_pf[64, False]`
  when the arm has `_pf` on the tiled stash backward. Each branch sits after
  the trial-only branch and before the first-round one, whose condition is
  already `and not ran_arm`. On Apple and AMD neither constant is true and
  the shipped build compiles what it compiled before.
- `fused_forward_launch_ran` and `fused_backward_launch_ran` carry the old
  launcher bodies plus `mut ran: Int`, the arm word of the kernels that
  launched, written inside the launching branch (0 for the shipped kernels
  or a refusal; `fwd_sstash`, `bwd_stash`, `bwd_stash_tiled`; the
  second-round words with rows resolved; never a sabotage bit).
  `fused_forward_launch_arm` and `fused_backward_launch_arm` keep their
  signatures and call them. `fused_attention_arm_forward_resolved` and
  `fused_attention_arm_backward_resolved` say what each launcher must report
  at head_dim 64 on this build.

Identity. No kernel was written or edited. On NVIDIA the shipped path now
launches the instantiations whose bit equality with eager sections 14.2 to
14.4 argue, the M4 arms check (15 cases x 13 arms) and the H100 leg's BITS
lines (every one MATCH against stash_tiled and the oracle) measured, through
the same generic launch helpers the trial tree calls. Which kernel runs is a
schedule; the contract reads the bits.

How stash_tiled stays reachable. A trial build (`-D
MOJOLEARN_ATTN_ARM_TRIAL=1`) still compiles every arm and reads
`MOJOLEARN_ATTN_ARM` per launcher call: `stash_tiled` and every other name
run as before, unset or empty runs the column default, and the harness's
`MOJOLEARN_ATTN_BASELINE=stash_tiled` prices the new default against it. On
a shipped build an explicit stash_tiled word passed to `fused_*_launch_arm`
still runs stash_tiled (its forward key is 0, not the default's, and it has
no `_pf`), because the first-round kernels are compiled on every shipped
build.

A later flip edits the matrix row (and, for a bare `_fgrid`,
`attn_fwd_rows_per_block_for`) and this brief, no longer `ATTN_ARM_DEFAULT`
itself (14.9's last paragraph predates the row).

### 15.3 Labels: baseline, stash_tiled and the default cannot be confused

- `transformer/checks/transformer_fused_check.mojo` (no trial define): a
  `DEFAULT column=<column> arm=<name> word=<n>` line; the default word must
  name itself and resolve to itself at head_dim 64 on the build (else FAIL:
  the row names an arm the column cannot run as named); an
  `ARM this_run=... is_default=... forward_hd64=... backward_hd64=...` line;
  every direct launch prints `ran <name>` and FAILS unless it ran the arm's
  resolved half at head_dim 64 and the shipped kernels elsewhere or on a
  refusal; the PASS line names the column, the arm, the default and how many
  launches ran it. `expected_ran` is shared with the arms check.
- `transformer/checks/transformer_attention_arms_check.mojo`: the same ran
  assertion for all 13 arms, clean and sabotaged, per direction.
- `bench/attention_step_price_main.mojo`: `default` is an alias for the
  column's arm, replaced by the explicit name before anything prints; a
  `DEFAULT column=... arm=... baseline_is_default=... candidate_is_default=...
  baseline_requested=... candidate_requested=...` line; `PATH` lines gain
  `is_default=` and `resolved_hd64=`; every correctness run prints
  `RAN <kind> <arm> forward=<name> backward=<name>` and raises at head_dim
  64 when they are not the arm's resolved kernels; TABLE headers carry
  `column=` and `default_arm=`.
- `bindings/_mojolearn_byte_lm.mojo`: `byte_lm_attention_arm()` returns
  `[arm, default, trial_build, resolved_hd64]` (constants and the
  environment only). `python/mojolearn/_byte_lm_impl.py`:
  `run_metadata()['native_attention_arm']` (None for an older binding).
- `tools/lm_step_memory_probe.py`: `attention_arm` in the setup event and
  result.json is now the binding's name for the arm that ran (it was the raw
  `MOJOLEARN_ATTN_ARM`, None when unset), beside `attention_arm_source`,
  `attention_arm_requested`, `attention_arm_default`,
  `attention_arm_is_default`, `attention_arm_resolved_hd64` and
  `attention_arm_trial_build`.
- `tools/attention_step_leg.sh`: `MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1` builds
  and runs the no-trial fused check (`build-shipped-fused-check`,
  `shipped-fused-check`) and copies its DEFAULT, ARM and PASS lines into
  gate.txt; the first smoke log's `DEFAULT` line goes to gate.txt; an LM arm
  `default` runs with `MOJOLEARN_ATTN_ARM` empty (`lm-default-<corpus>`);
  `lm_summary.tsv` prints `arm=`, `arm_requested=`, `arm_default=`,
  `arm_is_default=`, `arm_resolved_hd64=`.
- `tools/attention_flip_r3_leg.sh`: the H100 confirmation body (15.5).

### 15.4 RUN OWED on the M4 (the orchestrator's light commands, one at a time)

1. The shipped path, no trial define, Apple's row (stash_tiled):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check`. Expect
   `DEFAULT column=apple arm=stash_tiled word=7`,
   `DEFAULT resolved_hd64=stash_tiled`,
   `ARM this_run=stash_tiled is_default=True forward_hd64=fwd_sstash backward_hd64=bwd_stash_tiled`
   and `transformer_fused_check: PASS, 15 cases, ...` naming `17 direct
   launches RAN fwd_sstash / bwd_stash_tiled at head_dim 64` (nine forward
   and eight backward launches at head_dim 64 are not refused by the case
   list).
2. Optional but the only M4 reach of the shipped round 3 branches, still no
   trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check-r3`
   then `nice -n 19 /tmp/fused-check-r3`. Expect
   `DEFAULT column=apple arm=stash_tiled_fgrid_r32_qres_pf word=3175`, the
   same resolved name, and PASS with
   `RAN fwd_sstash_fgrid_r32_qres_pf / bwd_stash_tiled_pf` (the Q residency
   page, 15,232 B, fits Metal's 32 KB).
3. The arms gate, trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check`. Expect the 14.8 lines unchanged
   (`names: 24 spellings and 52 arm values round-trip, 18 invalid spellings refused`,
   `transformer_attention_arms_check: PASS, names inverse, 15 cases x 13 arms`),
   every status line now ending `ran <name>`, and no FAIL line containing
   `RAN`.
4. The price harness at L 512 against stash_tiled, correctness only:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`
   then
   `MOJOLEARN_ATTN_BASELINE=stash_tiled MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`.
   Expect
   `DEFAULT column=apple arm=stash_tiled source=kernel_matrix.attn_default_arm_for baseline_is_default=True candidate_is_default=False ...`,
   `PATH baseline arm=stash_tiled is_default=True resolved_hd64=stash_tiled ...`,
   `PATH candidate arm=stash_tiled_fgrid_r32_qres_pf is_default=False resolved_hd64=stash_tiled_fgrid_r32_qres_pf ... fwd_rows=32 preflush=True ...`,
   `RAN hashed stash_tiled forward=fwd_sstash backward=bwd_stash_tiled`,
   `RAN hashed stash_tiled_fgrid_r32_qres_pf forward=fwd_sstash_fgrid_r32_qres_pf backward=bwd_stash_tiled_pf`,
   every `BITS ... _vs_stash_tiled` MATCH,
   `REACH ... clean_restored=True reach_bit=stash_tiled_fgrid_r32_qres_pf+sabotage_new`,
   and `attention_step_price: PASS (stash_tiled_fgrid_r32_qres_pf vs stash_tiled)`.
5. The byte LM binding build only (no step):
   `nice -n 19 env MOJOLEARN_NUMERIC_MODE=identical sh bindings/build_byte_lm.sh`
   (expect the `built ...` line; the new export is `byte_lm_attention_arm`).

### 15.5 The confirmation legs

NVIDIA, H100 on RunPod, from a `git worktree add --detach` checkout at the
lane's merge commit. The body `tools/attention_flip_r3_leg.sh` sets
`MOJOLEARN_ATTN_BASELINE=stash_tiled`,
`MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_fgrid_r32_qres_pf`,
`MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled,stash_tiled_fgrid_r32_qres_pf`,
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`, `MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1` and
`MOJOLEARN_COMPILE_JOBS=8`, then runs `tools/attention_step_leg.sh`:

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GPU_ARCHS=sm_90a \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_flip_r3_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-flip-r3 \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
        --gpu "NVIDIA H100 80GB HBM3"

Gates: section 6 with stash_tiled in place of baseline; `shipped-fused-check`
exit 0 with `shipped_check: DEFAULT column=nvidia arm=stash_tiled_fgrid_r32_qres_pf`
and its PASS line in gate.txt; `harness: DEFAULT column=nvidia arm=stash_tiled_fgrid_r32_qres_pf`;
in `lm_summary.tsv` the stash_tiled rows `arm_is_default=False` and the
default's rows `arm=stash_tiled_fgrid_r32_qres_pf arm_is_default=True`, with
`witnesses_equal_baseline=True` on both corpora. The flip holds when the
geometric mean of the two lean step ratios stays below 1.

AMD, MI300X on Hot Aisle, where the shipped default is still stash_tiled and
this leg decides it (the 14.9 arms against stash_tiled):

    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_round3_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-attention-round3 \
    bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates

The runner reads the arch from rocminfo when `MOJOLEARN_GPU_ARCHS` is unset
(the MI300X is gfx942). Its smoke logs must say
`DEFAULT column=amd arm=stash_tiled`. The runner caps the lease at 60
minutes, and 14.9's five prices and 16 LM probes were sized for 90; if the
price lines already rank the arms, `MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled,stash_tiled_fgrid_r32_qres_pf"`
keeps the witness reference and the NVIDIA winner, and
`MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1` in the same variable adds the no-trial
check. An AMD flip edits `attn_default_arm_for`'s AMD line and
`attn_fwd_rows_per_block_for`'s AMD value from the same leg's `_fgrid_r32`
and `_fgrid_r64` prices.

### 15.6 Risks only a build or a box can settle

1. Nothing was compiled. The likeliest compile faults: module-scope
   comptime evaluation of `fused_attention_fwd_rows(ATTN_ARM_DEFAULT)`,
   `_attn_fwd_r2_key` and `fused_attention_arm_new_backward`; `comptime
   assert` inside a non-generic function (the repository's precedents are in
   a generic matrix row and a check function); comptime globals as
   parameters of `_launch_fwd_r2`; code after a `comptime if` that returns in
   the two resolved functions; `mut ran: Int` passed through the
   `_arm` wrappers.
2. The shipped round 3 branches compile only on a no-trial build whose
   column default needs them: NVIDIA, or the M4 under
   `MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN`. The leg's LM probes use trial
   bindings, which never compile those branches; the leg's shipped fused
   check reaches them at small shapes only. A shipped NVIDIA binding's LM
   step is argued, not measured, to match the trial run of the same name,
   because both launch `_launch_fwd_r2[64, 32, True, True, False]` and
   `_launch_bwd_stash_tiled_pf[64, False]` through the same helpers.
3. A shipped NVIDIA build now compiles four more GPU kernels (the forward
   copy, the preflushed zdot stash, dq and dk/dv folds). Build time grows.
4. `ran` proves which branch launched, not what the kernel computed; reach
   of the kernel content stays the trial sabotage (arms check, harness).
5. `attention_arm` in result.json changed meaning from the raw request to
   the binding's name. Readers of older evidence still see the request, and
   a trial binding now raises on an invalid `MOJOLEARN_ATTN_ARM` at
   `run_metadata()`, before the first step instead of during it.
6. AMD: the default and `attn_fwd_rows_per_block_for`'s 32 are placeholders
   until the MI300X leg reads them.

## 16. The dk/dv fold on AMD (2026-09-11, worktree lane, source built, nothing run): DEVIATIONS 2596 and 2597

STATUS: 16.1 reads legs already filed; everything from 16.2 on is source
only. Nothing here was compiled or run (no build on the Mac, by rule). The
orchestrator's M4 commands in 16.7 are the first compile. The shipped path
is unchanged: a build without `-D MOJOLEARN_ATTN_ARM_TRIAL=1` compiles none
of the kernels below and dispatches exactly as before.

### 16.1 The measurement (filed legs, lean LM step component timers, ms per step)

AMD's shipped default is `stash_tiled_fgrid_r32_qres_pf` since the round 3
AMD legs (kernel matrix `attn_default_arm_for`; lean step geomean 0.9297 of
stash_tiled on the MI325X and 0.9325 on the MI300X, witnesses equal). Its
backward is `_launch_bwd_stash_tiled_pf`: the preflushed zdot stash, dq tiled
and dk/dv tiled kernels.

| box, arm, corpus | dkdv tiled | dq tiled | zdot stash | bwd.attention | envelope |
|---|---:|---:|---:|---:|---:|
| MI325X DO, stash_tiled, enwik8 | 705.0 | 31.9 | 51.0 | 798.3 | 3406.6 |
| MI325X DO, stash_tiled, Pile GitHub | 703.8 | 32.0 | 51.0 | 798.2 | 3344.5 |
| MI325X DO, default (`_pf` kernels), enwik8 | 511.2 | 27.7 | 34.5 | 584.6 | 3102.0 |
| MI325X DO, default, Pile GitHub | 519.3 | 27.9 | 34.5 | 592.7 | 3102.0 |
| MI325X DO, stash_tiled_pf, enwik8 / Pile GitHub | 515.0 / 516.2 | 27.9 / 27.7 | 34.4 / 34.2 | 690.4 / 589.2 | |
| MI300X RunPod, stash_tiled, enwik8 / Pile GitHub | 626.1 / 736.8 | 34.7 / 34.8 | 51.4 / 51.3 | 717.8 / 829.6 | |
| MI300X RunPod, default, enwik8 / Pile GitHub | 646.0 / 733.2 | 30.4 / 30.4 | 34.7 / 34.7 | 716.7 / 804.0 | |
| H100, stash_tiled, enwik8 | 18.5 | 15.5 | 89.9 | 126.2 | 384.5 |
| H100, default (`_pf` kernels), enwik8 | 12.4 | 12.5 | 66.2 | 93.4 | 334.5 |

AMD over H100, same arm: dk/dv 38x (stash_tiled) and 41x (default); dq
2.1x and 2.2x; zdot stash 0.57x and 0.52x. Under the AMD default dk/dv is 87
percent of `bwd.attention` and 16.5 percent of the MI325X timing step.

Two further readings of the same files, both measured and neither a cause.

1. The harness and the LM step disagree on AMD only. `price_tables.txt`
   prices one layer at the LM shape (B 1, L 2048, 12 heads, 12 kv heads,
   hd 64) on that corpus's real activations; the LM step runs 12 such
   layers. Twelve times the harness's derived backward against the LM
   step's `bwd.attention`: H100 stash_tiled 12 x 10.47 = 125.7 against
   126.2, default 12 x 7.78 = 93.3 against 93.4 (equal to a percent). MI325X
   stash_tiled 12 x 40.07 = 480.8 against 798.3 (1.66x), default 12 x 31.54
   = 378.5 against 584.6 (1.54x); MI300X stash_tiled 12 x 41.56 = 498.7
   against 717.8 (1.44x). Under the default the LM step's dk/dv alone (511.2
   / 12 = 42.6 ms per layer) exceeds the harness's whole backward per layer
   (31.5 ms). Subtracting the LM's zdot, dq, scans and corner reads per
   layer from the harness backward leaves at most about 32 ms per layer for
   dk/dv under stash_tiled and about 26 under the default, still about 20
   to 25 times the H100's 1.5 and 1.0. So the kernel is anomalous on its own
   and the LM step's context adds more on AMD. The harness's per-kernel
   timers (`timers-<arm>`) were skipped on both AMD legs
   (`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`), so the split between the two is not
   measured.
2. The AMD timers are one timing step per arm and corpus and move by about
   100 ms outside dk/dv between corpora on the same box (MI325X `attn.core`
   166.4 against 60.6, `attn.o_proj` 128.4 against 26.1,
   `attn.bwd_corner_flag` 107.1 against 5.8), and dk/dv itself reads 424 to
   737 ms on the MI300X. Per-kernel AMD ratios are read to a factor, not a
   percent.

### 16.2 The source reading: the tiled dq fold beside the tiled dk/dv fold

`fused_bwd_dq_tiled_pf_kernel` and `fused_bwd_dkdv_tiled_pf_kernel`
(`transformer/impl/llama/fused_attention.mojo`; the first-round
`fused_bwd_dq_tiled_kernel` and `fused_bwd_dkdv_tiled_kernel` differ only in
`_step` for `_step_preflushed` and a sabotage flag). Counts at the target
shape, causal, `n_rep == 1`.

| | dq tiled | dk/dv tiled |
|---|---|---|
| block | 64 query rows of one (batch, head) | 64 keys of one (batch, kv head) |
| grid | 12 x 32 = 384 | 12 x 32 = 384 |
| tile | 16 keys | 16 queries, per head of the kv group |
| tiles per head, summed over blocks | 4 x (1 + ... + 32) = 1,984 | 4 x (32 + ... + 1) = 1,984 |
| barriers per tile | 2 (plus one per block for `zs`) | 2 |
| shared allocations, page | `kst` 1,024, `dst` 1,024, `zs` 64 floats: 3, 8,448 B | `qs`, `dcs`, `ys`, `dss` 1,024 floats each: 4, 16,384 B |
| resident blocks per CU by 11.1 (`min(2048 // 256, 65536 // page)`) | 7 | 4 |
| staging per thread per tile | 4 K slots (1 global load, 1 shared store) and 4 dcell slots (row range, 2 stash loads, 1 shared load, 2 `_pmul`, 1 stash store, 1 shared store): 12 global loads, 4 global stores, 8 shared stores | 4 Q/dctx slots (2 global loads, 2 shared stores) and 4 y/dcell slots (key range, 2 stash loads, 2 shared stores): 16 global loads, 16 shared stores |
| fold per thread per tile | 16 keys x (4 column loads + 4 x (1 cell load + 4 steps)): 128 shared loads, 256 steps | 16 queries x (8 column loads + 4 x (2 cell loads + 8 steps)): 256 shared loads, 512 steps |
| thread state | 16 accumulators, 4 column operands, 1 cell operand, 8 Int32 range bounds | 32 accumulators, 8 column operands, 2 cell operands, 8 Int32 range bounds, the corner Bool |
| stash access per tile | 64 rows x 16 adjacent keys; the next tile reads the same 64 rows, 16 keys on | 16 rows x 64 adjacent keys; the next tile reads 16 new rows, 16 x 2,048 floats (131 KB) further on |

By count dk/dv does exactly twice dq's fold work and twice its shared
stores per tile, over the same tile and barrier count. Counting cannot make
a 20x difference. What dk/dv carries and dq does not is a footprint that
could cross a vendor threshold: twice the thread state, twice the page in
one more allocation, and a column-major walk of the row-major
`[B, n_heads, L, S]` stashes.

### 16.3 Competing explanations, and what the box must show to separate them

None of these is measured. Each names the readback, timer or price that
would confirm or refute it.

- C1, thread state over the HIP register budget. If 32 accumulators, 10
  operand registers and 8 range bounds cannot stay register-resident, the
  compiler spills to thread-local memory, and every step pays local loads
  and stores. If local memory is partitioned per CU the way shared memory
  is (11.1), the spill can also lower resident blocks. Consistent with the
  filed readings: dq (16 accumulators) costs 2x the H100 and the zdot stash
  (two scalar chains) 0.5x, and the 32-row forward (12 accumulators against
  24) gained 1.72x on the MI325X against 1.36x on the H100, though its page
  also shrank. Confirms it: `RESOURCES label=dkdv_tiled_pf regs=` above
  dq_tiled_pf's with `local=` above 0 for dk/dv only, and both `_kvsplit`
  and `_kvgrid_r32` (16 accumulators each) well below the default in the
  dk/dv timer lines, with `_kvgrid_r64` (the copy at the shipped geometry)
  at the default's.
- C2, page occupancy. 16,384 B leaves 4 resident blocks per CU against dq's
  7 by the 11.1 formula. At 110 CUs (an UNVALIDATED MI250X transcription)
  that is 440 blocks, above the 384-block grid, so the formula says the
  grid still fits one wave. C2 needs the formula to be wrong on these boxes
  (16,384 B is exactly a quarter of 64 KB, so any per-allocation padding
  gives 3, and a 32 KB partition gives 2). The zdot stash's larger 17,696 B
  page over a 6,144-block grid runs at 0.5x the H100, which argues against a
  general AMD page penalty. Confirms it: `blocks_per_sm_256=` for dk/dv
  below dq's (at or below 3), and `_kvsplit` (8,192 B) clearly below
  `_kvgrid_r32` (12,288 B) though both hold 16 accumulators.
- C3, the stash walk. The volume of stash reads is identical and the
  locality is not: a dq block walks 64 rows along the key axis; a dk/dv
  block walks 2,048 rows down the query axis, 16 new rows per tile. If HIP's
  global memory path is miss- or translation-bound, dk/dv pays per tile
  what dq pays per 32 tiles. Confirms it: resources for dk/dv equal to dq's
  (C1, C2 and C4 refuted) and every arm below near 1.00 of the default in
  the dk/dv timers (none changes the walk), `_kvsplit` possibly above 1.00
  (two walks). The build that attacks it is a transposed copy of the two
  stashes (a copy kernel, no arithmetic) that the fold then reads row-major.
- C4, a serial fallback. A runtime that cannot place the kernel's resources
  might run fewer threads per block or one block per CU. Confirms it:
  `max_threads=` below 256 or `blocks_per_sm_256=1` for dk/dv only.
- C5, the LM step's context (16.1 item 1). Confirms it: the harness's
  per-kernel breakdown (`timers-<arm>`, `MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0`)
  on the same dumps. If its `attn.bwd_dkdv_tiled_pf` per call is about the
  LM line over 12, the cost is the kernel's; if it is about half, the LM
  step adds the rest (memory pool, buffer placement, the timing step's
  synchronizes), and no kernel arm reaches that half. The LM backward's dk
  and dv are zeroed stage buffers like dq's
  (`transformer/checks/transformer_backward.mojo`, `LlamaBackwardStages`),
  so first touch of the outputs is not a dk/dv-only term.

Order of likelihood from the source: C1 first (it is the only footprint
term consistent with both AMD-cheap kernels), then C2, then C3. C4 and C5
are read by the same leg at no extra build.

### 16.4 Identity argument (written before the code)

Kernels. `fused_bwd_dkdv_r2_kernel[HD, BJ, SAB]` (DEVIATION 2597): a
generic copy of `fused_bwd_dkdv_tiled_pf_kernel` with `BJ` keys per block
(64 or 32). `fused_bwd_kvfold_r2_kernel[HD, BJ, SAB]` (DEVIATION 2596): one
fold, `dst[j, d]` over (head in the kv group ascending, query tiles
ascending, queries ascending in a tile) of
`_step_preflushed(cell[t, j], col[t, d], acc)`, launched twice by one
instantiation: dk with `cell` the dy stash (which holds dcell after dq) and
`col` q_rope, then dv with `cell` the y stash and `col` dctx. Both are
launched by the trial-only generic `_launch_bwd_stash_tiled_kv[HD, BJ,
SPLIT]`, which first launches the shipped `_pf` zdot stash and dq tiled
instantiations exactly as `_launch_bwd_stash_tiled_pf` does.

Claim: every dk, dv, zdot and dq bit, and the corner flag, equal the default
arm's (and so the eager oracle's, which the arms check and the H100 and AMD
legs measured for the default).

1. zdot and dq. The same `fused_bwd_zdot_stash_pf_kernel[64, 4, ZSAB]` and
   `fused_bwd_dq_tiled_pf_kernel[64]` instantiations on the same buffers, in
   the same order, with the same grids. The y stash and the dcell values dq
   writes over the dy stash are therefore bit-equal to the default's.
2. One dk chain. For key `j` and column `d` the shipped chain starts at
   +0.0 and steps `_step_preflushed(dcell[t, j], ftz(q[t, d]), acc)` for
   each head `h` of the kv group ascending, then each query `t` ascending
   over the queries that see `j` (`_key_query_range`), skipping the rest.
   In both new kernels the accumulator for (j, d) starts at +0.0 per block,
   the head loop and the query-tile loop run ascending, the queries inside
   a tile run ascending, and the step is taken exactly where the copied
   test `t < l and lo[u] <= t <= hi[u]` holds, with `lo[u], hi[u]` the
   key's `_key_query_range`. The operands are staged as the shipped kernel
   stages them: `ftz(q)` in the column tile, the stash value as stored in
   the cell tile, and the step reads them in the shipped order (cell first,
   column second). So the chain is the same terms in the same order.
3. One dv chain. Likewise with `y[t, j]` and `ftz(dctx[t, d])`.
4. Keys per block. The blocks partition the keys `0 .. S - 1` into runs of
   `BJ`, and inside a block thread `(tr, tc)` (`tid // 16`, `tid % 16`)
   holds keys `j0 + tr + 16u` for `u < BJ // 16` and columns `tc + 16v` for
   `v < HD // 16`, so every (j, d) chain is held by exactly one thread. The
   query-tile loop runs the block's union range, from the first key's first
   visible query to the last key's last visible query (both bounds increase
   with `j`), and steps only visible queries, so a narrower block skips
   fewer tiles and adds no term. Which thread holds a chain and how many
   chains a thread holds are execution-plan choices the contract does not
   read.
5. Two kernels. dk and dv share no accumulator and no stored value in the
   shipped kernel; their chains read disjoint buffers and write disjoint
   outputs. Folding them in two launches changes no chain. The dk launch
   and the dv launch read only buffers no kernel writes after dq, so their
   order changes nothing.
6. The corner flag. The shipped kernel sets `corner` to 1.0 when some
   accumulator of the thread holds -0.0 at the end of some head's run and
   the thread holds at least one key below `S`. The dk launch sets it under
   that condition over its dk accumulators, the dv launch over its dv
   accumulators; the flag is written only with 1.0 from 0.0, so the value
   after both launches is the OR, which is the shipped condition. dq and
   zdot write it as before. The launcher's status is therefore the
   default's.
7. The seam. Every step is `_step_preflushed` on operands that are not
   nonzero subnormals (14.2 item 3: staged `ftz`, a `_pmul` output, an
   `ftz(identical_div)` output), so it equals `_step`.

Sabotage (reach, `+sabotage_kv`, `MOJOLEARN_ATTN_ARM_SABOTAGE=kv`). Each new
kernel flips one ulp of every staged cell operand (the joint kernel both the
y and the dcell tiles; the fold its cell tile, so the dk launch flips dcell
and the dv launch flips y). dk and dv move; zdot, dq and the forward hold,
because the flip is in a staging page no other kernel reads. The arms check
and the harness assert both halves. Under `+sabotage_new` the same arms flip
the zdot stash copy as the default does, and the new kernels run clean, so
14.5's 2533 attribution (zdot moves, dv holds) still reads the backward.

### 16.5 What was built, per file

Arm bits and names (extends 14.1; the parser and the name function stay
inverses):

| bit | constant | token | meaning |
|---:|---|---|---|
| 8192 | `ATTN_ARM_BWD_KVSPLIT` | `_kvsplit` | DEVIATION 2596 |
| 16384 | `ATTN_ARM_BWD_KVGRID` | `_kvgrid` | DEVIATION 2597, keys from `attn_dkdv_keys_per_block_for` |
| 32768, 65536 | `ATTN_ARM_KVROWS32`, `ATTN_ARM_KVROWS64` | `_kvgrid_r32`, `_kvgrid_r64` | 2597 geometry |
| 131072 | `ATTN_ARM_SABOTAGE_KV` | `+sabotage_kv` | flips in the 2596 / 2597 kernels only |

Grammar: base, `_ztiled[_r32|_r64]`, `_fgrid[_r32|_r64]`, `_qres`, `_pf`,
`_kvgrid[_r32|_r64]`, `_kvsplit`, `+sabotage`, `+sabotage_new`,
`+sabotage_kv`. The parser refuses `_kvgrid` or `_kvsplit` without `_pf` on
the tiled stash backward (`stash_tiled_kvsplit`, `fwd_sstash_pf_kvgrid`),
with `_ztiled` (2528 launches its own folds), and out of order.

| arm | keys per block | dk/dv accumulators per thread | operand registers | page per launch |
|---|---:|---:|---:|---:|
| `stash_tiled_fgrid_r32_qres_pf` (AMD default) | 64 | 32 | 10 | 16,384 B, 4 allocations |
| `..._pf_kvgrid_r64` (control, the copy) | 64 | 32 | 10 | 16,384 B, 4 |
| `..._pf_kvgrid_r32` (2597) | 32 | 16 | 10 | 12,288 B, 4 |
| `..._pf_kvsplit` (2596) | 64 | 16 per launch | 5 | 8,192 B, 2 |
| `..._pf_kvgrid_r32_kvsplit` | 32 | 8 per launch | 5 | 6,144 B, 2 |

- `transformer/impl/llama/fused_attention.mojo`: the bits, parser, name
  function and `MOJOLEARN_ATTN_ARM_SABOTAGE=kv`; `_kv_r2_page_bytes`,
  `fused_attention_arm_kv`, `fused_attention_kv_keys` (the knob, else the
  matrix row, else 64 for `_kvsplit` alone; 0 when the page does not fit,
  and then the plain `_pf` backward runs) and `_attn_kv_ran_bits`; the two
  kernels of 16.4; `_launch_bwd_stash_tiled_kv[HD, BJ, SPLIT]` (runtime
  `zsab` and `ksab` pick the sabotage instantiations); in
  `fused_backward_launch_ran` a trial-only branch before the plain `_pf`
  branch; `fused_attention_arm_backward_resolved` reports the kv word; the
  new bits join `ATTN_ARM_DEFAULT_REFUSED_BITS` (a shipped build compiles
  none of these kernels, so no default may name them). New timer lines
  under `MOJOLEARN_ATTN_PHASE_TIMERS`: `attn.bwd_kvgrid_dkdv_pf`, or
  `attn.bwd_kvsplit_dk_pf` then `attn.bwd_kvsplit_dv_pf` (beside the
  default's `attn.bwd_dkdv_tiled_pf`). Stale AMD sentences in the header
  and in `ATTN_ARM_DEFAULT`'s docstring now say AMD runs the round 3 arm.
- `checks/kernel_matrix.mojo`: `attn_dkdv_keys_per_block_for[column]` (AMD
  32, others 64, UNMEASURED; the forced `_kvgrid_r32` / `_kvgrid_r64` names
  are the evidence-bearing ones). `attn_default_arm_for`'s docstring no
  longer says AMD waits on the MI300X leg.
- `transformer/checks/transformer_attention_arms_check.mojo`: 17 arms (14.6's
  13 plus the four in the table above); names: 31 spellings round-trip, 17
  arms times 8 sabotage combinations (three sabotage bits) round-trip as
  values, 25 invalid spellings refused; a third run per kv arm under
  `+sabotage_kv` that must move dk and dv and hold zdot, dq and the forward
  at head_dim 64, and move nothing at other head dims.
- `bench/attention_step_price_main.mojo`: `PATH` gains `kv_keys=` and
  `kv_split=`; a `REACH_KV` line per kind for a kv arm (a `+sabotage_kv` run
  between the `+sabotage_new` run and the clean restore) with `dk_moved`,
  `dv_moved`, `zdot_moved`, `dq_moved`, `forward_moved`, failing unless dk
  and dv move and the rest hold; and `MOJOLEARN_ATTN_RESOURCES` (default 1):
  before the kinds, `DeviceContext.compile_function` on eight kernels at
  head_dim 64 (`zdot_stash_pf`, `dq_tiled_pf`, `dkdv_tiled`,
  `dkdv_tiled_pf`, `kvgrid_r64`, `kvgrid_r32`, `kvsplit_r64_fold`,
  `kvsplit_r32_fold`), each a `RESOURCES_BEGIN` line with the source counts
  (geometry, accumulators, operand registers, thread-local floats, page
  bytes, allocations) and one `RESOURCES` line per runtime attribute
  (`regs`, `local`, `shared`, `const`, `max_threads`, `blocks_per_sm_256`),
  each kernel in its own try (`RESOURCES_ERROR`), as
  `bench/gemm_step_resources_main.mojo` reads them. It launches nothing.
- `tools/attention_step_leg.sh`: smokes and timer runs pass
  `MOJOLEARN_ATTN_RESOURCES=0`, the first price run 1 (one readback per
  leg), `resources.txt` collects the lines, `gate.txt` names 2596 and 2597.
- `tools/attention_dkdv_leg.sh`: the body wrapper (16.8).

A trial build instantiates 8 more kernel pipelines (two kernels times two
geometries times the sabotage flag) and the harness compiles 8 kernels for
the readback.

### 16.6 Risks only a build or a box can settle

1. Nothing was compiled. Likeliest faults: one comptime kernel alias
   launched twice with different buffers in `_launch_bwd_stash_tiled_kv`;
   runtime `if ksab:` holding comptime aliases inside `comptime if SPLIT:`;
   `ctx.compile_function` on the generic kernels in the harness; the
   `continue` in the arms check's sabotage loop.
2. A one-ulp flip of every staged dcell or y is assumed to move some dk and
   some dv cell on every head_dim 64 case, as the first-round dq and dk/dv
   flips did. A flip that moves nothing fails reach loudly.
3. If C3 or C5 is the cause, no arm here reaches it; the leg's resources,
   the dk/dv timer lines and (with `MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0`) the
   standalone breakdown say which, and the transposed stash is the next
   build.
4. `_kvsplit` pays two launches, two staging walks of the query tiles and
   two sets of barriers. If the thread state is not the cause it prices
   above 1.00; on the H100, where dk/dv is 12 ms of a 334 ms step, it may
   price above 1.00 regardless.
5. `_kvgrid_r32` doubles the dk/dv grid (768 blocks) and the per-block fixed
   work (the key ranges, the per-block tile bounds).
6. The compile readback may raise on AMD for some attributes (the
   occupancy query is the likeliest); each prints or raises by itself. It
   also adds eight pipeline compiles to the first price run.
7. The kernel-matrix row's AMD value (32) is a placeholder, as 12.5 item 4.
8. The AMD timers are one sample each and noisy (16.1 item 2); the flip
   reads the lean step medians, never the timers.

### 16.7 RUN OWED on the M4 (the orchestrator's light commands, one at a time)

1. The shipped path, no trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check`. Expect 15.4 item 1 unchanged:
   `DEFAULT column=apple arm=stash_tiled word=7`,
   `ARM this_run=stash_tiled is_default=True forward_hd64=fwd_sstash backward_hd64=bwd_stash_tiled`,
   `transformer_fused_check: PASS, 15 cases, ...`.
2. The arms gate, trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check`. Expect
   `names: 31 spellings and 136 arm values round-trip, 25 invalid spellings refused`,
   status lines such as
   `stash_tiled_fgrid_r32_qres_pf_kvsplit backward status: RAN  ran bwd_stash_tiled_pf_kvsplit`
   and `... ran bwd_stash_tiled_pf_kvgrid_r32_kvsplit`, at head_dim 64
   `REACH stash_tiled_fgrid_r32_qres_pf_kvsplit+sabotage_kv dk_moved=<n> dv_moved=<n> zdot_moved=0 dq_moved=0 forward_moved=0`
   with both counts above 0 for each of the four kv arms, and
   `transformer_attention_arms_check: PASS, names inverse, 15 cases x 17 arms`.
3. The harness at L 512 against the column default (on the M4 `default`
   resolves to stash_tiled), correctness only:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`
   then
   `MOJOLEARN_ATTN_BASELINE=default MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvsplit MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_RESOURCES=1 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`,
   then the same with `MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_RESOURCES=0`
   for `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`, then
   `stash_tiled_fgrid_r32_qres_pf_kvgrid_r64`, then
   `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_kvsplit`. Expect
   `DEFAULT column=apple arm=stash_tiled ...`,
   `PATH candidate arm=stash_tiled_fgrid_r32_qres_pf_kvsplit is_default=False resolved_hd64=stash_tiled_fgrid_r32_qres_pf_kvsplit ... kv_keys=64 kv_split=True`
   (`kv_keys=32` for the `_kvgrid_r32` arms),
   `RAN hashed stash_tiled_fgrid_r32_qres_pf_kvsplit forward=fwd_sstash_fgrid_r32_qres_pf backward=bwd_stash_tiled_pf_kvsplit`,
   every `BITS ... _vs_stash_tiled` MATCH,
   `REACH ... clean_restored=True reach_bit=stash_tiled_fgrid_r32_qres_pf_kvsplit+sabotage_new`,
   `REACH_KV ... dk_moved=<above 0> dv_moved=<above 0> zdot_moved=0 dq_moved=0 forward_moved=0`,
   eight `RESOURCES_BEGIN` labels on the first run, each followed by
   `RESOURCES` lines or a `RESOURCES_ERROR` (Metal may refuse an attribute;
   that is a reading, not a failure), and
   `attention_step_price: PASS (stash_tiled_fgrid_r32_qres_pf_kvsplit vs stash_tiled)`.

### 16.8 The AMD leg

The body is `tools/attention_dkdv_leg.sh`. Each setting only when unset:
`MOJOLEARN_ATTN_BASELINE=stash_tiled_fgrid_r32_qres_pf` (the shipped AMD
default),
`MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_kvgrid_r64,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_kvsplit`,
`MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled_fgrid_r32_qres_pf,stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_kvsplit`,
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`, `MOJOLEARN_COMPILE_JOBS=8` and
`MOJOLEARN_GPU_ARCHS=gfx942` (tools/gemm_remote_leg.sh does not export the
arch; the Hot Aisle and DigitalOcean runners do, and theirs wins). It
refuses LM arms without the baseline, then runs
`sh tools/attention_step_leg.sh`: four smokes, four prices on both corpora's
activations (the first with the resource readback) and 16 LM probes, the
load of 14.9's round 3 leg.

First `tools/pick_box.sh --need amd`, then the runner it names, from a
`git worktree add --detach` checkout at the lane's merge commit:

    # hotaisle
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_dkdv_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-attention-dkdv \
    bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates

    # do
    MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
    MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_dkdv_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-do-attention-dkdv \
    bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates

    # runpod-amd (as the round 3 MI300X leg ran, its leg.txt)
    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_dkdv_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-runpod-attention-dkdv \
    sh tools/gemm_remote_leg.sh amd --payload gemm --rent --minutes 60 \
        --gpu "AMD Instinct MI300X OAM"

`MOJOLEARN_HOTAISLE_EXTRA_ENV` or `MOJOLEARN_DO_EXTRA_ENV` with
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0` adds the standalone per-kernel breakdown
that separates C5, when the lease allows; if the lease runs short, drop LM
arms after the price lines rank them, never the baseline.

Gates: section 6 with the AMD default in place of `baseline` (arms check
exit 0, smokes exit 0, every `BITS ... _vs_stash_tiled_fgrid_r32_qres_pf`
MATCH on both corpora's activations, `REACH` and `REACH_KV` proven, lean
steps `limited: false`). Flip rule, ENGINEERING_RULES 9: for an arm, the
geometric mean of its enwik8 and pilegithub lean step ratios
(`steady_median_seconds` of `lm-<arm>-<corpus>` over
`lm-stash_tiled_fgrid_r32_qres_pf-<corpus>`, same leg) below 1, and
`witnesses_equal_baseline=True` for every step on both corpora. A flip
edits `attn_default_arm_for`'s AMD word (after its lane adds the shipped
branch and takes the bits out of `ATTN_ARM_DEFAULT_REFUSED_BITS`) and, for
`_kvgrid`, `attn_dkdv_keys_per_block_for`'s AMD value from the same leg.

Reading the leg against 16.3: `resources.txt` (regs, local, occupancy for
dq against dk/dv and the new kernels) and the `lmtiming-*` dk/dv lines
(`attn.bwd_dkdv_tiled_pf` against `attn.bwd_kvgrid_dkdv_pf` or the sum of
`attn.bwd_kvsplit_dk_pf` and `attn.bwd_kvsplit_dv_pf`) say which cause
holds; the lean step medians decide the flip.
