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

## 16. The dk/dv backward on AMD: the step against the harness (2026-09-11, worktree lane, source built, nothing run): DEVIATIONS 2596 and 2597

STATUS: 16.1 reads legs already filed; everything from 16.2 on is source
only. Nothing here was compiled or run (no build on the Mac, by rule). The
orchestrator's M4 commands in 16.7 are the first compile. The shipped path is
unchanged: a build without `-D MOJOLEARN_ATTN_ARM_TRIAL=1` compiles none of
the launches below and dispatches exactly as before. AMD's shipped default is
`baseline` again (fc742e62); the AMD ratio for these arms is against
`baseline`.

### 16.1 The measurement

The deciding evidence is one RunPod MI300X pod running all three arms
(bench/results/e1g/2026-09-11_171959-amd-mi300x-runpod-attention-three, commit
a1a22f3f, GEMM ksplit default on, every step witness equal to baseline's).

Lean LM step, steady median seconds (enwik8 / Pile GitHub): baseline 2.192 /
2.166; stash_tiled 2.543 / 2.536 (geomean 1.166 of baseline);
stash_tiled_fgrid_r32_qres_pf, the round 3 arm, 2.213 / 2.325 (1.041).

Step component timers, ms per step (enwik8 / Pile GitHub):

| arm | bwd.attention | dk/dv | dq | zdot | forward kernel | attn.core |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 169.2 / 168.8 | `bwd_dkdv` 63.6 / 63.3 | 51.1 / 51.3 | 50.0 / 49.8 | 134.1 / 240.6 | 137.5 / 244.1 |
| stash_tiled | 818.5 / 836.4 | `bwd_dkdv_tiled` 626.7 / 746.6 | 34.5 / 34.4 | 152.1 / 50.1 | 58.6 / 58.2 | 62.2 / 61.8 |
| round 3 arm | 594.6 / 590.9 | `bwd_dkdv_tiled_pf` 525.2 / 521.2 | 30.0 / 29.9 | 34.2 / 34.3 | 24.5 / 24.5 | 28.1 / 28.3 |

The price harness on the same pod, on each corpus's real last-layer
activations at the step's shape (B 1, L 2048, 12 heads, 12 kv heads, hd 64),
ms per layer, medians (`price_tables.txt`):

| arm | fwd | fwd+bwd | bwd (derived) |
|---|---:|---:|---:|
| baseline | 11.04 to 11.49 | 25.00 to 25.32 | 13.74 to 14.02 |
| stash_tiled | 5.17 / 5.17 | 47.12 / 46.72 | 41.95 / 41.55 |
| round 3 arm | 2.29 / 2.30 | 33.66 / 33.78 | 31.37 / 31.48 |

A correction to the reading that opened this section of the lane: 11.2 to
11.5, 5.17 and 2.29 ms are the harness's FORWARD medians. Its backward already
ranks the tiled backward above baseline's on AMD, 3.0x (stash_tiled) and 2.3x
(the round 3 arm), where the H100 harness has the stash backward at 0.48x of
baseline's (10.47 against 21.9 ms, section 13).

The step and the harness, 12 layers against one:

| arm | forward: 12 x harness against `attn.core` | backward: 12 x harness against `bwd.attention` |
|---|---|---|
| baseline | 138 against 137.5 (enwik8; Pile GitHub 136 against 244, a 100 ms swing the same box shows elsewhere) | 165 to 168 against 169.2 / 168.8: 1.00x to 1.03x |
| stash_tiled | 62 against 62.2 / 61.8 | 503 / 499 against 818.5 / 836.4: 1.63x / 1.68x |
| round 3 arm | 27.5 against 28.1 / 28.3 | 376 / 378 against 594.6 / 590.9: 1.58x / 1.56x |

So there are two gaps, and they are different in kind.

1. In the harness, with no step around it, the tiled backward costs 2.3x to
   3.0x baseline's backward on AMD. Baseline's whole backward per layer (13.9
   ms) is less than the round 3 arm's dk/dv alone can be: in the step the
   round 3 arm's zdot and dq cost 5.4 ms per layer, so its harness dk/dv is at
   most about 26 ms per layer against baseline's 5.3 in the step.
2. In the step, the stash backward costs about 1.6x what twelve harness calls
   cost, while baseline's backward and every arm's forward agree with the
   harness. The excess is 213 to 338 ms per step. If it all sits in dk/dv, the
   round 3 arm's in-harness dk/dv would be about 305 ms per step equivalent
   and the in-step extra about 220.

The two earlier AMD legs show the same second gap (MI325X DO stash_tiled 1.66x,
round 3 arm 1.54x; MI300X RunPod stash_tiled 1.44x) and the H100 does not
(1.00x for both stash arms). The harness's per-kernel breakdown
(`timers-<arm>`) was skipped on all three AMD legs
(`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`), so neither gap is split by kernel inside
the harness.

### 16.2 The source reading, centered on the step against the harness

What the step does around the stash backward that the harness does not,
read from `transformer/impl/llama/fused_attention.mojo`,
`transformer/checks/transformer_backward.mojo` and
`bench/attention_step_price_main.mojo`:

- Scratch per call. `_launch_bwd_stash_tiled_pf` enqueues two
  `[B, n_heads, L, S]` stashes (50,331,648 floats, 201 MB each) at the start
  of every backward call and frees them after the dk/dv launch;
  `_launch_fwd_r2` (and the first-round sstash branch) allocates a third per
  forward call. Baseline allocates none: it recomputes y and dy inside each
  kernel. In the harness these are the only large allocations between frees,
  so every call can land on the region the previous call released; in the
  step they interleave with 12 layers of stage buffers, the GEMM ksplit
  workspaces and the optimizer state. This is the one difference that exists
  for exactly the arms that show gap 2 and not for baseline.
- Who touches the stash pages, in which order. The zdot stash kernel writes
  both stashes at every visible cell (the first touch of those pages in the
  call); the tiled dq kernel reads y and rewrites dy with dcell, row-major
  (64 rows, 16 adjacent keys per tile, the same rows tile after tile); the
  tiled dk/dv kernel reads both column-major (16 rows x 64 keys per tile,
  16 new rows every tile, 2,048 rows per block). Page granularity, translation
  and cache reach therefore cost dk/dv far more than dq whenever the stash
  pages are cold or scattered, and nothing when they are warm and contiguous.
- Timers and synchronizes. Every step timer synchronizes before it reads the
  clock; the harness synchronizes once per call. Not the gap: baseline's step
  timers agree with its harness, and the lean step medians (no timers) rank
  the arms as the timers do.
- Pipelines. Compiled once per process; the component-timing step runs after
  the probe's steady steps, as for baseline. Not the gap by the same test.
- Grid and block count. The harness reads the step's own last-layer shape
  from the dump's `meta.txt`; both run the same kernels at the same grids
  (384 dk/dv blocks), and the step's timed kernel runs alone on the device.
  Not a difference.
- Output buffers. The step's dk, dv and dq are zeroed stage buffers
  (`LlamaBackwardStages`, `_zeros`) exactly as the harness's are uploaded;
  first touch of the outputs is not a dk/dv-only term.

The tiled dq fold beside the tiled dk/dv fold (the source behind gap 1), at
the target shape, causal, `n_rep == 1`:

| | dq tiled | dk/dv tiled |
|---|---|---|
| block | 64 query rows of one (batch, head) | 64 keys of one (batch, kv head) |
| grid | 12 x 32 = 384 | 12 x 32 = 384 |
| tiles per head, summed over blocks | 4 x (1 + ... + 32) = 1,984 | 4 x (32 + ... + 1) = 1,984 |
| barriers per tile | 2 | 2 |
| shared allocations, page | 3, 8,448 B | 4, 16,384 B |
| resident blocks per CU by 11.1 | 7 | 4 |
| staging per thread per tile | 12 global loads, 4 global stores, 8 shared stores | 16 global loads, 16 shared stores |
| fold per thread per tile | 128 shared loads, 256 steps | 256 shared loads, 512 steps |
| thread state | 16 accumulators, 5 operand registers, 8 range bounds | 32 accumulators, 10 operand registers, 8 range bounds |
| stash walk | row-major, the same 64 rows tile after tile | column-major, 16 new rows every tile |

By count dk/dv does exactly twice dq's fold work per tile over the same tiles
and barriers, which cannot make the observed gap against dq by itself. The
recompute dk/dv kernel baseline runs (`fused_bwd_dkdv_kernel[64, 4]`) reads no
stash: it stages K and V per key, recomputes y and dy per cell, then folds dk
and dv with two accumulators per thread and a 17,696-byte page.

### 16.3 Competing explanations, and what the box must show to separate them

None of these is measured.

Gap 2 (the step's extra cost of the stash backward):

- S1, stash placement and residency. Per-call 201 MB stashes land on cold or
  scattered device pages in the step and on the just-released warm region in
  the harness; dk/dv's column-major walk pays for it most. Separates: the
  `_kvrecompute` arm (16.4 A) frees both stashes before dk/dv and runs the
  recompute kernel, so its `attn.bwd_kvre_dkdv` step line should read about
  baseline's `attn.bwd_dkdv` (63.6 ms) if S1 holds; `attn.bwd_kvre_stash_free`
  prices the free. The harness breakdown (`timers-<arm>`,
  `MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0`) puts dk/dv per call beside the step line
  over 12.
- S2, allocator state (pending frees, pool growth). Freed stashes not reusable
  until a later synchronize, so each call allocates fresh device memory in the
  step. Separates: `attn.bwd_scratch_alloc` and `attn.fwd_scratch_alloc`
  stayed below 1.6 ms on every AMD leg, so an allocation-time cost is already
  small; S2 survives only as a placement effect, read like S1.
- S3, the step's other work contending for the same CUs. Timers synchronize
  around every kernel, so nothing else is queued during the dk/dv tick; S3 is
  unlikely by the source, and the `_kvrecompute` line reads it as S1 does.

Gap 1 (the harness's own cost of the tiled dk/dv fold on AMD):

- C1, thread state over the HIP register budget: 32 accumulators and 10
  operand registers per thread spill to thread-local memory. Separates:
  `RESOURCES label=dkdv_tiled_pf regs=` above `dq_tiled_pf`'s with `local=`
  above 0, and `_kvsplit` and `_kvgrid_r32` (16 accumulators each, 16.4 B)
  clearly below the round 3 arm in the dk/dv lines, `_kvgrid_r64` (the copy
  at the shipped geometry) at its level.
- C2, page occupancy: 16,384 B leaves 4 resident blocks per CU by the 11.1
  formula (above the 384-block grid at 110 CUs, so it needs the formula to be
  wrong on this box). Separates: `blocks_per_sm_256=` for dk/dv below dq's,
  and `_kvsplit` (8,192 B) clearly below `_kvgrid_r32` (12,288 B).
- C3, the column-major walk itself, cold or warm. Separates: resources for
  dk/dv equal to dq's and every 2597 arm near the round 3 arm's dk/dv. The
  build that would attack it is a transposed copy of the two stashes read
  row-major.
- C4, a serial fallback: `max_threads=` below 256 or `blocks_per_sm_256=1`
  for dk/dv only.

Order of likelihood from the source: for gap 2, S1 (it is the only step-only
difference that exists for exactly the stash arms); for gap 1, C1, then C3,
then C2. `_kvrecompute` removes both gaps from dk/dv if S1 and C1 or C3 hold;
the 2597 arms and the readback say whether the tiled fold is worth keeping.

### 16.4 Identity arguments (written before the code)

A. DEVIATION 2596, `_kvrecompute`. Launch `_launch_bwd_stash_tiled_kvre[HD]`:
the two stashes, the preflushed zdot stash kernel and the preflushed tiled dq
kernel (`_launch_bwd_stash_zdq_pf`, the same instantiations, order, grids and
buffers `_launch_bwd_stash_tiled_pf` launches), a synchronize, both stashes
freed, then the shipped recompute kernel `fused_bwd_dkdv_kernel[64, 4]` at its
shipped geometry (grid `B * n_kv * ceil(S / 4)`, 256 threads).

1. zdot and dq are the round 3 arm's kernels on the same inputs, so their
   bits are the round 3 arm's, which the arms check, the harness and every leg
   measured equal to baseline's and to eager.
2. `fused_bwd_dkdv_kernel` is the kernel baseline runs, unedited. Its outputs
   are a function of q, dctx, k, v, amax, denom and zdot only (it reads no
   stash). q, dctx, k and v are the call's operands; amax and denom come from
   whichever forward ran, bit-equal to baseline's by the forward arms' own
   identity (sections 12 to 15); zdot is bit-equal by item 1. Same kernel,
   same inputs, same geometry: dk and dv are baseline's bits.
3. The corner flag. zdot and dq set it for their chains exactly as in the
   round 3 arm; the recompute kernel sets it for the dk and dv chains exactly
   as in baseline. The flag is written only with 1.0, so the value is the OR
   of the same conditions, and the launcher's status is baseline's.
4. Freeing the stashes after a synchronize and before dk/dv changes no value
   any kernel reads: no kernel after dq reads them.

Sabotage (`+sabotage_kv`): the launch copies `denom` into a scratch with every
row flipped one ulp (`_attn_flip_copy_kernel`, `_flip_ulp`) and hands that to
the recompute kernel. Its y moves, so dv moves and dcell, then dk, move; zdot,
dq and the forward are untouched (the copy is a scratch). If the tiled fold
ran instead it would read the stash, not `denom`, and dk and dv would hold, so
the check also attributes the launch.

B. DEVIATION 2597, `_kvgrid[_r32|_r64]` and `_kvsplit`. Kernels
`fused_bwd_dkdv_r2_kernel[HD, BJ, SAB]` (a generic copy of
`fused_bwd_dkdv_tiled_pf_kernel` with `BJ` keys per block, 64 or 32) and
`fused_bwd_kvfold_r2_kernel[HD, BJ, SAB]` (one fold, `dst[j, d]` over (head in
the kv group ascending, query tiles ascending, queries ascending in a tile) of
`_step_preflushed(cell[t, j], col[t, d], acc)`, launched twice by one
instantiation: dk with `cell` the dy stash holding dcell and `col` q_rope,
then dv with `cell` the y stash and `col` dctx), by
`_launch_bwd_stash_tiled_kv[HD, BJ, SPLIT]` after `_launch_bwd_stash_zdq_pf`.

1. zdot and dq: item A.1.
2. One dk chain. For key `j` and column `d` the shipped chain starts at +0.0
   and steps `_step_preflushed(dcell[t, j], ftz(q[t, d]), acc)` for each head
   of the kv group ascending, then each query `t` ascending over the queries
   that see `j`, skipping the rest. In both kernels the accumulator for
   (j, d) starts at +0.0 per block, the head and query-tile loops run
   ascending, queries inside a tile run ascending, and the step is taken
   exactly where the copied test `t < l and lo[u] <= t <= hi[u]` holds (the
   key's `_key_query_range`). The operands are staged as the shipped kernel
   stages them (`ftz(q)` in the column tile, the stash value as stored in the
   cell tile) and stepped in the shipped order (cell, then column).
3. One dv chain. Likewise with `y[t, j]` and `ftz(dctx[t, d])`.
4. Keys per block partition the keys into runs of `BJ`; thread `(tr, tc)`
   holds keys `j0 + tr + 16u` (`u < BJ // 16`) and columns `tc + 16v`, so each
   (j, d) chain has exactly one thread. The query-tile loop runs the block's
   union range (both range bounds increase with `j`) and steps only visible
   queries, so a narrower block skips fewer tiles and adds no term.
5. Two kernels. dk and dv share no accumulator and write disjoint outputs;
   both launches read only buffers nothing writes after dq, so their order
   changes nothing.
6. The corner flag: the dk launch sets it over dk chains, the dv launch over
   dv chains, each under the shipped per-thread condition; the OR is the joint
   kernel's condition.
7. The seam: every step's operands are staged `ftz` values, `_pmul` outputs or
   `ftz(identical_div)` outputs, never nonzero subnormals, so
   `_step_preflushed` equals `_step` (14.2).

Sabotage (`+sabotage_kv`): each kernel flips one ulp of every staged cell
operand (the joint kernel the y and dcell tiles; the fold its cell tile); dk
and dv move, zdot, dq and the forward hold.

Under `+sabotage_new` every kv arm flips the zdot stash copy as the round 3
arm does and runs its dk/dv launch clean, so 14.5's attribution (zdot moves, dv
holds) still reads the backward.

### 16.5 What was built, per file

| bit | constant | token | meaning |
|---:|---|---|---|
| 8192 | `ATTN_ARM_BWD_KVSPLIT` | `_kvsplit` | DEVIATION 2597, two single-fold launches |
| 16384 | `ATTN_ARM_BWD_KVGRID` | `_kvgrid` | DEVIATION 2597, keys from `attn_dkdv_keys_per_block_for` |
| 32768, 65536 | `ATTN_ARM_KVROWS32`, `ATTN_ARM_KVROWS64` | `_kvgrid_r32`, `_kvgrid_r64` | 2597 geometry |
| 131072 | `ATTN_ARM_SABOTAGE_KV` | `+sabotage_kv` | flips in the 2596 / 2597 dk/dv launch only |
| 262144 | `ATTN_ARM_BWD_KVRECOMPUTE` | `_kvrecompute` | DEVIATION 2596 |

Grammar: base, `_ztiled[_r32|_r64]`, `_fgrid[_r32|_r64]`, `_qres`, `_pf`,
`_kvrecompute`, `_kvgrid[_r32|_r64]`, `_kvsplit`, `+sabotage`,
`+sabotage_new`, `+sabotage_kv`. The parser refuses any of the three kv tokens
without `_pf` on the tiled stash backward, with `_ztiled`, `_kvrecompute`
together with `_kvgrid` or `_kvsplit`, and any token out of order.

| arm (all on `stash_tiled_fgrid_r32_qres_pf`) | dk/dv kernel | keys per block | dk/dv accumulators per thread | page per launch | stashes during dk/dv |
|---|---|---:|---:|---:|---|
| (the round 3 arm) | `fused_bwd_dkdv_tiled_pf_kernel` | 64 | 32 | 16,384 B | held |
| `_kvrecompute` (2596) | `fused_bwd_dkdv_kernel[64, 4]` | 4 | 2 | 17,696 B | freed |
| `_kvgrid_r64` (2597, control) | `fused_bwd_dkdv_r2_kernel` | 64 | 32 | 16,384 B | held |
| `_kvgrid_r32` (2597) | `fused_bwd_dkdv_r2_kernel` | 32 | 16 | 12,288 B | held |
| `_kvsplit` (2597) | `fused_bwd_kvfold_r2_kernel` x 2 | 64 | 16 | 8,192 B | held |
| `_kvgrid_r32_kvsplit` (2597) | `fused_bwd_kvfold_r2_kernel` x 2 | 32 | 8 | 6,144 B | held |

- `transformer/impl/llama/fused_attention.mojo`: the bits, parser, name
  function and `MOJOLEARN_ATTN_ARM_SABOTAGE=kv`; `_kv_r2_page_bytes`,
  `fused_attention_arm_kv`, `fused_attention_kv_keys` (4 for `_kvrecompute`;
  the knob, else the matrix row, else 64 for the 2597 tokens; 0 when a 2597
  page does not fit, and then the plain `_pf` backward runs) and
  `_attn_kv_ran_bits`; the two 2597 kernels; `_attn_flip_copy_kernel`;
  `_launch_bwd_stash_zdq_pf[HD]`, `_launch_bwd_stash_tiled_kvre[HD]` and
  `_launch_bwd_stash_tiled_kv[HD, BJ, SPLIT]`; in `fused_backward_launch_ran`
  a trial-only branch before the plain `_pf` branch;
  `fused_attention_arm_backward_resolved` reports the kv word; the new bits
  join `ATTN_ARM_DEFAULT_REFUSED_BITS`. New timer lines under
  `MOJOLEARN_ATTN_PHASE_TIMERS`: `attn.bwd_kvre_stash_free` and
  `attn.bwd_kvre_dkdv` (2596); `attn.bwd_kvgrid_dkdv_pf`, or
  `attn.bwd_kvsplit_dk_pf` then `attn.bwd_kvsplit_dv_pf` (2597). Stale AMD
  default sentences corrected.
- `checks/kernel_matrix.mojo`: `attn_dkdv_keys_per_block_for[column]` (AMD 32,
  others 64, UNMEASURED; the forced `_kvgrid_r32` / `_kvgrid_r64` names are the
  evidence-bearing ones); `attn_default_arm_for`'s docstring says AMD is
  `baseline`.
- `transformer/checks/transformer_attention_arms_check.mojo`: 18 arms (14.6's
  13 plus the five above); names: 33 spellings round-trip, 18 arms times 8
  sabotage combinations round-trip as values, 27 invalid spellings refused; a
  third run per kv arm under `+sabotage_kv` that must move dk and dv and hold
  zdot, dq and the forward at head_dim 64, and move nothing at other head
  dims.
- `bench/attention_step_price_main.mojo`: `PATH` gains `kv_keys=` and
  `kv_split=`; a `REACH_KV` run and line per kind for a kv arm, failing unless
  dk and dv move and the rest hold; `MOJOLEARN_ATTN_RESOURCES` (default 1):
  before the kinds, `DeviceContext.compile_function` on nine backward kernels
  at head_dim 64 (`zdot_stash_pf`, `dq_tiled_pf`, `dkdv_recompute`,
  `dkdv_tiled`, `dkdv_tiled_pf`, `kvgrid_r64`, `kvgrid_r32`,
  `kvsplit_r64_fold`, `kvsplit_r32_fold`), each a `RESOURCES_BEGIN` line with
  the source counts and one `RESOURCES` line per attribute (`regs`, `local`,
  `shared`, `const`, `max_threads`, `blocks_per_sm_256`), each in its own try,
  as `bench/gemm_step_resources_main.mojo` reads them. It launches nothing.
- `tools/attention_step_leg.sh`: smokes and timer runs pass
  `MOJOLEARN_ATTN_RESOURCES=0`, the first price run 1; `resources.txt`
  collects the lines; `gate.txt` names 2596 and 2597.
- `tools/attention_dkdv_leg.sh`: the body wrapper (16.8).

A trial build instantiates 8 more kernel pipelines for 2597 and one flip-copy
kernel for 2596; the harness compiles nine kernels for the readback.

### 16.6 Risks only a build or a box can settle

1. Nothing was compiled. Likeliest faults: the `mut DeviceBuffer` stashes
   passed through `_launch_bwd_stash_zdq_pf`; one comptime kernel alias
   launched twice in `_launch_bwd_stash_tiled_kv`; runtime `if ksab:` holding
   comptime aliases inside `comptime if SPLIT:`; `ctx.enqueue_function` on the
   non-generic `_attn_flip_copy_kernel`; `ctx.compile_function` on the generic
   kernels in the harness; the `continue` in the arms check's sabotage loop.
2. `_kvrecompute` adds one host synchronize per backward call (after dq, so
   the stashes can be freed before dk/dv). It is inside the priced path.
3. If gap 2 comes from the forward stash (`_launch_fwd_r2`'s per-call sstash)
   or from the stash kernels' effect on device memory state that outlives the
   free, `attn.bwd_kvre_dkdv` reads above 63.6 ms and the recompute kernel
   inherits part of the gap; the free tick and the harness breakdown say
   which.
4. The denom flip may, on a rare case, trip the corner flag and turn a
   sabotage run's status into FUSED_CORNER; the harness then fails that run
   loudly. A one-ulp flip that moves no dk or dv cell also fails loudly.
5. If C3 is gap 1's cause, no 2597 arm reaches it; the transposed stash copy
   is the next build.
6. `_kvsplit` pays two launches and two staging walks; `_kvgrid_r32` doubles
   the dk/dv grid. Either can price above the round 3 arm if C1 and C2 are not
   the cause.
7. The compile readback may raise on AMD for some attributes; each prints or
   raises by itself. It adds nine pipeline compiles to the first price run.
8. `attn_dkdv_keys_per_block_for`'s AMD value (32) is a placeholder.
9. AMD step timers are one sample each and swing by about 100 ms outside the
   kernel of interest (16.1); the flip reads the lean step medians only.

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
   `names: 33 spellings and 144 arm values round-trip, 27 invalid spellings refused`,
   status lines ending `ran bwd_stash_tiled_pf_kvrecompute`,
   `ran bwd_stash_tiled_pf_kvsplit`, `ran bwd_stash_tiled_pf_kvgrid_r32_kvsplit`
   and so on, at head_dim 64 for each of the five kv arms
   `REACH <arm>+sabotage_kv dk_moved=<n> dv_moved=<n> zdot_moved=0 dq_moved=0 forward_moved=0`
   with both counts above 0, and
   `transformer_attention_arms_check: PASS, names inverse, 15 cases x 18 arms`.
3. The harness at L 512 against the column default (on the M4 `default`
   resolves to stash_tiled), correctness only:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`
   then
   `MOJOLEARN_ATTN_BASELINE=default MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvrecompute MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_RESOURCES=1 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`,
   then the same with `MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_RESOURCES=0` for
   `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvsplit`, then
   `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`, then
   `stash_tiled_fgrid_r32_qres_pf_kvgrid_r64`, then
   `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_kvsplit`. Expect
   `DEFAULT column=apple arm=stash_tiled ...`,
   `PATH candidate arm=stash_tiled_fgrid_r32_qres_pf_kvrecompute is_default=False resolved_hd64=stash_tiled_fgrid_r32_qres_pf_kvrecompute ... kv_keys=4 kv_split=False`
   (`kv_keys=64 kv_split=True` for `_kvsplit`, `kv_keys=32` for the
   `_kvgrid_r32` arms),
   `RAN hashed stash_tiled_fgrid_r32_qres_pf_kvrecompute forward=fwd_sstash_fgrid_r32_qres_pf backward=bwd_stash_tiled_pf_kvrecompute`,
   every `BITS ... _vs_stash_tiled` MATCH,
   `REACH ... clean_restored=True reach_bit=<arm>+sabotage_new`,
   `REACH_KV ... dk_moved=<above 0> dv_moved=<above 0> zdot_moved=0 dq_moved=0 forward_moved=0`,
   nine `RESOURCES_BEGIN` labels on the first run, each followed by
   `RESOURCES` lines or a `RESOURCES_ERROR` (Metal may refuse an attribute;
   that is a reading, not a failure), and
   `attention_step_price: PASS (<arm> vs stash_tiled)`.

### 16.8 The AMD leg

The body is `tools/attention_dkdv_leg.sh`. Each setting only when unset:
`MOJOLEARN_ATTN_BASELINE=baseline` (the shipped AMD default);
`MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_fgrid_r32_qres_pf_kvrecompute,stash_tiled_fgrid_r32_qres_pf,stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_kvgrid_r64,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_kvsplit`;
`MOJOLEARN_ATTN_LEG_LM_ARMS=baseline,stash_tiled_fgrid_r32_qres_pf_kvrecompute,stash_tiled_fgrid_r32_qres_pf,stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`;
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`; `MOJOLEARN_COMPILE_JOBS=8`;
`MOJOLEARN_GPU_ARCHS=gfx942` (tools/gemm_remote_leg.sh does not export the
arch; the Hot Aisle and DigitalOcean runners export theirs, which wins). It
refuses LM arms without the baseline, then runs
`sh tools/attention_step_leg.sh`: six smokes, six prices on both corpora's
activations (the first with the resource readback) and 20 LM probes. The
three-arm pod leg in 16.1 ran its 12 probes and builds in 16 minutes.

First `tools/pick_box.sh --need amd`, then the runner it names, from a
`git worktree add --detach` checkout at the lane's merge commit:

    # hotaisle
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_dkdv_leg.sh \
    MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0" \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-attention-dkdv \
    bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates

    # do
    MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
    MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_dkdv_leg.sh \
    MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0" \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-do-attention-dkdv \
    bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates

    # runpod-amd (as the 16.1 pod leg ran; no extra-env plumbing, so the
    # wrapper's SKIP_TIMERS=1 stands)
    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_dkdv_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-runpod-attention-dkdv \
    sh tools/gemm_remote_leg.sh amd --payload gemm --rent --minutes 60 \
        --gpu "AMD Instinct MI300X OAM"

`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0` adds the harness's per-kernel breakdown,
the direct read of gap 2; it costs one more build and seven one-round runs.
If the lease runs short, drop LM arms after the price lines rank them, never
`baseline`.

Gates: section 6 against `baseline` (arms check exit 0, smokes exit 0, every
`BITS ... _vs_baseline` MATCH on both corpora's activations, `REACH` and
`REACH_KV` proven, lean steps `limited: false`). Flip rule, ENGINEERING_RULES
9 (one default per switch; this switch reaches only the fused attention
backward): for an arm, the geometric mean of its enwik8 and pilegithub lean
step ratios (`steady_median_seconds` of `lm-<arm>-<corpus>` over
`lm-baseline-<corpus>`, same leg) below 1, and `witnesses_equal_baseline=True`
for every step on both corpora. A flip edits `attn_default_arm_for`'s AMD word
after its lane adds the shipped branch and takes the arm's bits out of
`ATTN_ARM_DEFAULT_REFUSED_BITS`, and for `_kvgrid` sets
`attn_dkdv_keys_per_block_for`'s AMD value from the same leg.

The lane's target, from the coordinator: the in-step dk/dv at or below
baseline's 63.6 ms with the round 3 forward and zdot gains kept. Counted, not
measured: if `_kvrecompute`'s `attn.bwd_kvre_dkdv` reads about 64 ms, its
backward is about 34 (zdot) + 30 (dq) + 64 (dk/dv) + the free and the scans,
about 135 to 140 ms against baseline's 169, and its forward kernel about 24.5
against baseline's 134, so the attention part of the step would lose roughly
140 ms against baseline's.

Reading the leg against 16.3: `lmtiming-*` (`attn.bwd_kvre_dkdv` against
baseline's `attn.bwd_dkdv`; `attn.bwd_dkdv_tiled_pf` against
`attn.bwd_kvgrid_dkdv_pf` or the sum of `attn.bwd_kvsplit_dk_pf` and
`attn.bwd_kvsplit_dv_pf`), `resources.txt`, and with timers on the
`timers_summary.tsv` per-kernel lines against the step lines over 12. The lean
step medians decide the flip.

## 17. The zdot kernel on the H100 (2026-09-11, worktree lane, source built, nothing run): DEVIATION 2598

STATUS. 17.1 copies filed evidence. Everything from 17.2 on is source only.
Nothing here was compiled or run (no build on the Mac, by rule), and the
orchestrator's M4 commands in 17.8 are the first compile. A build without
`-D MOJOLEARN_ATTN_ARM_TRIAL=1` compiles none of the kernels or launches
below and dispatches exactly as before. This lane measures and tunes NVIDIA
only; sections 15 and 16 put the round 3 default behind `baseline` on AMD,
and nothing here is sized for AMD.

### 17.1 The measurement

Evidence
`bench/results/e1g/2026-09-11_164101-nvidia-h100-80gb-hbm3-new-defaults-torch/remote/attention-step/`
(H100 80GB HBM3, commit e629434d, one pod). Lean step, steady median seconds
(enwik8 / Pile GitHub), every step witness equal: the shipped NVIDIA default
`stash_tiled_fgrid_r32_qres_pf` 0.2950 / 0.2949, `stash_tiled` 0.3414 /
0.3409. Component timers of one serialized step (`lmtiming-*`, a breakdown
and never a price), ms:

| line | stash_tiled (enwik8 / Pile GitHub) | shipped NVIDIA default (enwik8 / Pile GitHub) |
|---|---:|---:|
| envelope.native_call | 341.7 / 341.4 | 293.4 / 295.4 |
| bwd.attention | 125.2 / 125.1 | 93.2 / 93.9 |
| attn.bwd_zdot_stash, attn.bwd_zdot_stash_pf | 89.3 / 89.3 | 66.2 / 66.6 |
| attn.bwd_dq_tiled, attn.bwd_dq_tiled_pf | 15.4 / 15.4 | 12.5 / 12.6 |
| attn.bwd_dkdv_tiled, attn.bwd_dkdv_tiled_pf | 18.5 / 18.3 | 12.4 / 12.6 |
| attn.fwd_sstash_kernel, attn.fwd_r2_kernel | 36.8 / 36.8 | 20.8 / 21.0 |

The same pod's price on each corpus's last-layer activations
(`price_tables.txt`) has the default's backward at 7.72 / 7.70 ms per layer
(derived), and 12 x 7.72 = 92.6 ms against the step's 93.2. On the H100 the
harness and the step agree on the backward (section 16.1's AMD gap is not
there).

Two older lines bear on the reading. Section 10 timed, on one pod and one
corpus, the recompute zdot `attn.bwd_zdot` at 52.2 ms and
`attn.bwd_zdot_stash` at 89.4 ms. Section 13 timed 2528's register-blocked
y/dy kernel at 115.5 ms plus its row z fold at 6.6 ms.

### 17.2 The source reading: where the 66 ms goes (counted, not measured)

The kernel is `fused_bwd_zdot_stash_pf_kernel[64, 4, False]`, launched once
per layer by `_launch_bwd_stash_tiled_pf`, with a synchronize before it (the
scratch tick) and after it (its own tick).

Geometry at the target (B 1, L = S = 2048, 12 heads, n_rep 1, window 0).

- A block is 4 query rows of one head on 256 threads. Thread `(tr, lane)`
  holds row `4 tb + tr`, and the y chain of key `32 kb + lane` when
  `lane < 32` or the dy chain of key `32 kb + lane - 32` otherwise. Grid
  12 x 512 = 6,144 blocks per layer, 73,728 per step.
- Block `tb` runs `(4 tb + 3) // 32 + 1 = tb // 8 + 1` key-block
  iterations of 32 keys, so 8 x (1 + ... + 64) = 16,640 per head, 199,680
  per layer and 2,396,160 per step (32.5 per block).
- One iteration is three phases, each ending in a barrier. Phase 1 stages K
  and V for the 32 keys (8 slots per thread, 16 global loads, 16 `ftz`, 16
  shared stores, 4,096 floats per block). Phase 2 runs one 64-term
  `_step_preflushed` chain per thread; a y lane then applies `_pmul`, the mask
  add, `identical_exp` (`portable_expf`, eight FMA, a floor, two multiplies
  and three compares) and `identical_div`; every visible thread then writes its
  value TWICE, to its shared slot (`ys` or `dys`) and to the global stash
  (`y_st` or `dy_st`, cell `row * S + j`). Phase 3 has lane 0 of each of the
  4 rows fold z over the iteration's 32 keys (two shared loads and one step
  per key) while 252 threads wait.
- Counted per step: 302,137,344 visible cells; 38.7 G dot terms; 302 M exp
  and 302 M div; 604 M global stash stores (2.42 GB); 302 M z steps; 7.19 M
  barriers; 2.40 M staging round trips carrying 9.8 G floats (39.3 GB) of K
  and V; per block, 64 global loads per thread for the row's q or dctx.

How zdot is consumed. `fused_bwd_dq_tiled_pf_kernel` loads `zdot` once per
row into its `zs` page (64 threads, one barrier per block) and uses it in
`ds = _pmul(y, ftz(ftz(dy) - ftz(z)))` at every visible cell of the row,
then writes `dcell` over `dy_st`. `fused_bwd_dkdv_tiled_pf_kernel` never
reads zdot: dk folds the dcell dq wrote, dv folds the y the zdot kernel
wrote. So zdot reaches dq directly and dk through dcell, and dv not at all.

Competing explanations, with the timer or price line that separates each.
None of them is measured.

- S, the stash stores in the dot phase. The only source difference between
  the two kernels section 10 timed on one pod (`fused_bwd_zdot_kernel` 52.2
  ms, `fused_bwd_zdot_stash_kernel` 89.4 ms, both on `_step` seams) is the two
  global stores per visible cell. The staging, the dots, the shared slots and
  the z phase are the same statements. The stores are issued at the END of
  each thread's dependent chain, so every iteration carries a second global
  round trip after the dots, beside the staging round trip at its start.
  Bandwidth alone reads badly as the cause: the stores are 2.4 GB per step,
  against 39 GB of staging loads that the 52.2 ms recompute kernel also
  carries. If 2533's seams left the store cost alone, the stores are about 37
  of the 66.2 ms; if their cost shrank in proportion, about 27. Line: a copy
  that moves the stores into the NEXT iteration's staging round trip
  (`_zdefer`, 17.3) reads well below 66 ms in `attn.bwd_zdot_zdefer_pf`, and
  its harness `bwd (derived)` falls by the saving over 12. If the cost is
  bandwidth and not placement, `_zdefer` reads about 66.
- Z, the z phase. A third barrier per iteration and a 32-step serial fold on
  4 of 256 threads (section 3.2 counted this phase longer than the dot phase,
  before 2533 shortened both). Line: `_zlag` (the fold moved into the staging
  phase) reads below `_zdefer`; without Z the two read the same.
- D, the dot and exp latency. A y lane runs 64 terms, exp and div before its
  barrier, a dy lane 64 terms. Line: both copies stay near 66 ms minus what S
  and Z take; nothing built here moves D.
- K, the K and V staging (every 4 rows re-stage the whole causal range, 39 GB
  per step). Line: both copies stay near 66 ms; the recompute kernel's 52.2
  ms bounds staging plus dots plus z on `_step` seams.
- G, grid and per-block fixed work (73,728 blocks per step, four
  `_row_range` calls and 64 global loads per thread per block). Same lines as
  K.
- Launches and synchronizes. 12 launches and 24 synchronizes per step around
  this kernel; `attn.bwd_scratch_alloc`, a synchronize plus two allocations,
  reads 0.14 ms. Not the 66 ms by the source.
- Shared work with dq. None: dq reads the stash and one zdot per row, and
  recomputes nothing the zdot kernel computed.
- Shared work with the forward. The y half: the backward's y dot is the
  forward's pass-1 score chain and y is the forward's pass-3 weight, on the
  same operands. Keeping y across the step is DEVIATION 2532 (11.7, 2.42 GB
  held in the callers' stage structs, outside these files); no line separates
  it until that arm exists.

Why 2528 does not answer S or Z. It removed both (the stores went into kernel
A's cell arithmetic after four p windows, z went to a separate kernel) and ALSO
replaced the dot geometry with 16 + 16 register chains per thread and four
staging round trips per 64-key iteration; that kernel read 115.5 ms. The loss
prices the register-blocked dot geometry on the H100. It says nothing about the
stores or the z phase in the one-chain-per-thread geometry, which is what the
copies below keep.

Order of likelihood from the source: S (the only difference between two
kernels timed 37 ms apart on one pod), then Z, then D, K and G.

### 17.3 The arms (DEVIATION 2598)

`fused_bwd_zdot_sched_pf_kernel[HD, TQ, LAG, SABN]` is a copy of
`fused_bwd_zdot_stash_pf_kernel` at the shipped geometry (TQ 4, 32 keys per
iteration, one chain per thread, the same four allocations and 17,696-byte
page). It changes WHEN two things happen, never what is computed.

- `_zdefer` (LAG False). Phase 2 writes the shared slot only. In phase 1 of
  the next iteration each thread writes its slot's value to the stash, beside
  its K and V loads, and a tail phase after the loop writes the last
  iteration's. One global round trip per iteration instead of two. The z
  phase and its barrier are unchanged. Attacks S.
- `_zlag` (LAG True). The same stores, and lane 0 folds z over the previous
  iteration's keys in that same phase 1 (the last iteration's in the tail).
  Two barriers per iteration instead of three and one global round trip.
  Attacks S and Z.

Both are launched through `_launch_bwd_stash_zdq_pf[HD]` (the helper
DEVIATIONS 2596 and 2597 share), which now takes the zdot schedule as a
runtime word and launches the shipped copy when the word is 0, then the shipped
preflushed dq instantiation. The plain arm then launches the shipped
preflushed dk/dv instantiation (`_launch_bwd_stash_tiled_zsched_pf`), and the
tokens compose with `_kvrecompute`, `_kvgrid` and `_kvsplit` with their kernels
untouched. No geometry differs by vendor, so no kernel-matrix row; the page is
the shipped copy's, so it fits wherever that one does.

### 17.4 Identity argument (written before the code)

Claim. After the new kernel, `zdot`, `y_st` and `dy_st` hold at every visible
row and cell the bits `fused_bwd_zdot_stash_pf_kernel[64, 4, False]` writes,
and the kernels after it are the shipped instantiations, so dq, dk and dv are
the shipped bits.

1. The shared slots. Phase 2 of iteration kb on thread `(tr, lane)` is the
   shipped phase 2 statement for statement (the same staged `ftz(k)` or
   `ftz(v)`, the same `vec`, `_step_preflushed` over p ascending from +0.0,
   `_pmul`, the mask add, `identical_exp` against `ftz(amax[row])`,
   `identical_div` by `ftz(denom[row])`, or `ftz(dy)`), and it stores the
   value to the same slot `ys[33 tr + kj]` or `dys[33 tr + kj]` under the same
   visibility test. The shipped copy stores the same variable to that slot and
   to the stash.
2. The stash. Thread `(tr, lane)` writes `y_st[row * S + j]` (or `dy_st`)
   for `j = 32 (kb - 1) + kj` in phase 1 of iteration kb, and for
   `j = 32 kb_hi + kj` in the tail, under the same test (`valid`, `active`,
   `j_lo <= j <= j_hi`), with the value it loads from its own slot. That slot
   was written in phase 2 of the iteration that holds key j (the same test, so
   the write happened) and is not written again before the load: slots are
   written in phase 2 only, and this phase 1 (or the tail) lies between that
   phase 2 and the next. Loading a stored Float32 returns its bits; no
   arithmetic touches it. Every visible cell of the block's rows lies in some
   iteration from kb_lo to kb_hi, because both ends of `_row_range` are
   nondecreasing in t. So every cell the shipped copy writes is written with the
   same bits, and no masked cell is written by either. The dq kernel reads the
   stash after the launcher's synchronize, when the launch is complete.
3. z. The shipped chain starts at +0.0 and, after each iteration's phase 2,
   steps `_step_preflushed(dys[33 tr + jj], ys[33 tr + jj], z)` for jj
   ascending over that iteration's visible keys, kb ascending. `_zdefer` keeps
   that phase verbatim. `_zlag` takes the same steps for iteration kb - 1 in
   phase 1 of iteration kb, and for kb_hi in the tail, on slots that still hold
   iteration kb - 1's values by item 2 (lane 0 reads the slots of the other
   threads of its row, and no thread writes a slot in phase 1 or the tail).
   Same terms, same order (kb_lo's keys ascending, then kb_lo + 1's, up to
   kb_hi's), same seam, on operands 14.2 showed flushed.
4. The corner test and the zdot store are the shipped statements after the loop
   (after the tail fold under `_zlag`).
5. Barriers. Every thread of a block runs the same kb range and reaches every
   barrier. The added statements sit between existing barriers or after the
   last one, never around one. Under `_zlag` the dropped barrier closed a phase
   only lane 0 used.
6. No data race. In phase 1 each thread writes its own staging slots
   `tid + 256 si` and reads `ys` and `dys`; in phase 2 it reads `ks` and `vs`
   and writes its own `ys` or `dys` slot; the tail reads slots only. Global
   writes go to distinct cells `row * S + j`.
7. The consumers. `fused_bwd_dq_tiled_pf_kernel[64]` and
   `fused_bwd_dkdv_tiled_pf_kernel[64]` (or, composed, the unedited 2596 and
   2597 launches) read zdot, `y_st` and `dy_st` only at visible rows and cells,
   so by 2 to 4 their inputs are the shipped bits.

Sabotage (reach, `+sabotage_new`). The copy stores zdot flipped one ulp
(`_flip_ulp`) at ODD flat rows `(bb * nh + h) * L + t` only. zdot moves at odd
rows and holds at even rows, dq and dk move through it, dv and the forward
hold, and nothing else in the kernel is flipped. 2533's flip moves zdot at
every row, so `zdot_moved_even_rows=0` with `zdot_moved_odd_rows` above 0
names this copy. `_zdefer` and `_zlag` share the flip, so which schedule ran
is not attributable from the outputs (the 2528 geometry precedent); the `ran`
word names it. Every head_dim 64 case of the fused check has an odd row.

### 17.5 What was built, per file

| bit | constant | token | meaning |
|---:|---|---|---|
| 524288 | `ATTN_ARM_BWD_ZDEFER` | `_zdefer` | DEVIATION 2598, the stash stores deferred into the next staging round trip |
| 1048576 | `ATTN_ARM_BWD_ZLAG` | `_zlag` | DEVIATION 2598, the stores deferred and the z fold lagged into the same phase |

Grammar: base, `_ztiled[_r32|_r64]`, `_fgrid[_r32|_r64]`, `_qres`, `_pf`,
`_zdefer` or `_zlag`, `_kvrecompute`, `_kvgrid[_r32|_r64]`, `_kvsplit`,
`+sabotage`, `+sabotage_new`, `+sabotage_kv`. The parser refuses `_zdefer` or
`_zlag` without `_pf` on the tiled stash backward, with `_ztiled`, the two
together, and out of order.

- `transformer/impl/llama/fused_attention.mojo`: the bits (both in
  `ATTN_ARM_NEW_BITS` and `ATTN_ARM_DEFAULT_REFUSED_BITS`), parser, name
  function and `fused_attention_arm_zsched`; the kernel of 17.3;
  `_launch_bwd_stash_zdq_pf[HD]` with a trailing `zsched` word (0 launches the
  shipped copy exactly as before), passed through
  `_launch_bwd_stash_tiled_kvre` and `_launch_bwd_stash_tiled_kv`;
  `_launch_bwd_stash_tiled_zsched_pf[HD]`; in `fused_backward_launch_ran` the
  kv branch passes the word and reports it, and a new trial-only branch before
  the plain `_pf` branch launches the zsched helper;
  `fused_attention_arm_backward_resolved` reports the word. Timer lines under
  `MOJOLEARN_ATTN_PHASE_TIMERS`: `attn.bwd_zdot_zdefer_pf` and
  `attn.bwd_zdot_zlag_pf` in place of `attn.bwd_zdot_stash_pf`, then the
  existing dq and dk/dv lines. The shipped `_launch_bwd_stash_tiled_pf` and
  every shipped branch are untouched.
- `transformer/checks/transformer_attention_arms_check.mojo`: 21 arms (16.5's
  18 plus `stash_tiled_fgrid_r32_qres_pf_zdefer`,
  `stash_tiled_fgrid_r32_qres_pf_zlag` and
  `stash_tiled_fgrid_r32_qres_pf_zlag_kvgrid_r32`); names 39 spellings
  round-trip, 21 arms times 8 sabotage combinations round-trip as values, 35
  invalid spellings refused; under `+sabotage_new` at head_dim 64 a zsched arm
  must also move zdot at odd rows and hold it at even rows.
- `bench/attention_step_price_main.mojo`: `PATH` gains `zsched=`; the `REACH`
  line gains `zdot_moved_even_rows=` and `zdot_moved_odd_rows=`; a zsched
  candidate gets a `REACH_Z` run per kind with the forward clean and the
  backward under `+sabotage_new` (so the attribution holds on the round 3
  arms, whose Q residency flip moves amax and denom in the joint run), failing
  unless zdot moves at odd rows only and dv and the forward hold; the
  `RESOURCES` readback gains `zdot_zdefer_pf` and `zdot_zlag_pf`.
- `tools/attention_step_leg.sh`: header and `gate.txt` only (names pass
  through).
- `tools/attention_zdot_leg.sh`: the leg body wrapper (17.9).

A trial build instantiates four more kernel pipelines (two schedules times the
sabotage flag); the harness compiles two more for the readback.

### 17.6 Risks only a build or a box can settle

1. Nothing was compiled. Likeliest faults: `barrier()` inside `comptime if not
   LAG:` inside the runtime key loop (the r2 forward's `comptime if phase ==
   0:` blocks inside its key loop are the precedent); four runtime branches
   holding comptime kernel aliases in `_launch_bwd_stash_zdq_pf`; the new
   trailing argument at the two kv helpers and their five call sites;
   `Case.run_pair` in the harness.
2. If S is bandwidth, or the device already merges a thread's late write with
   its next load, `_zdefer` prices at 1.00 and `_zlag` gets only Z.
3. `_zlag` lengthens phase 1 by lane 0's fold. If the staging round trip is
   shorter than 32 z steps, phase 1 becomes as long as the old phase 3, and
   the dropped barrier is all it gains.
4. The odd-row flip needs an odd visible row; a case without one fails loudly.
5. The lease. The leg runs five smokes, five prices and 20 LM probes, the size
   of section 16.8's leg. `_zdefer` is price-only (its `bwd (derived)` beside
   `_zlag`'s separates S from Z); if its price ranks below `_zlag`'s, its LM
   step is a follow-on leg.
6. The kv arms on NVIDIA (the coordinator's addition, 17.9). dk/dv is 12.4 ms
   of the H100 step, so their H100 ratio is bounded by that line; they ride
   this pod to get NVIDIA numbers, not to be tuned here.

### 17.7 The flip rule for this leg

ENGINEERING_RULES 9. For an arm, the geometric mean of its enwik8 and
pilegithub lean step ratios (`steady_median_seconds` of `lm-<arm>-<corpus>`
over `lm-stash_tiled_fgrid_r32_qres_pf-<corpus>`, same pod) below 1, and
`witnesses_equal_baseline=True` for every step on both corpora. A flip edits
only `attn_default_arm_for`'s NVIDIA word (a new literal beside
`ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF`, asserted against this file's
bits), takes the arm's bits out of `ATTN_ARM_DEFAULT_REFUSED_BITS`, and adds the
clean shipped branch its lane owes (a shipped build compiles none of these
kernels today).

### 17.8 RUN OWED on the M4 (the orchestrator's light commands, one at a time)

1. The shipped path, no trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check`. Expect 15.4 item 1 unchanged:
   `DEFAULT column=apple arm=stash_tiled word=7`,
   `ARM this_run=stash_tiled is_default=True forward_hd64=fwd_sstash backward_hd64=bwd_stash_tiled`,
   `transformer_fused_check: PASS, 15 cases, ...`.
2. The arms gate, trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check`. Expect
   `names: 39 spellings and 168 arm values round-trip, 35 invalid spellings refused`;
   status lines ending `ran bwd_stash_tiled_pf_zdefer`,
   `ran bwd_stash_tiled_pf_zlag` and `ran bwd_stash_tiled_pf_zlag_kvgrid_r32`;
   at head_dim 64 for each of those three arms
   `REACH <arm>+sabotage_new zdot_moved=<n> dv_moved=0 zdot_moved_even_rows=0 zdot_moved_odd_rows=<n>`
   with n above 0 (and `REACH ..._zlag_kvgrid_r32+sabotage_kv dk_moved=<n> dv_moved=<n> zdot_moved=0 dq_moved=0 forward_moved=0`);
   and `transformer_attention_arms_check: PASS, names inverse, 15 cases x 21 arms`.
3. The price harness at L 512 against the column default (on the M4
   `default` resolves to stash_tiled), correctness only:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`
   then
   `MOJOLEARN_ATTN_BASELINE=default MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_zlag MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_RESOURCES=1 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`,
   then the same with `MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_RESOURCES=0` for
   `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_zdefer`, then
   `stash_tiled_fgrid_r32_qres_pf_zlag_kvgrid_r32`. Expect
   `DEFAULT column=apple arm=stash_tiled ...`,
   `PATH candidate arm=stash_tiled_fgrid_r32_qres_pf_zlag is_default=False resolved_hd64=stash_tiled_fgrid_r32_qres_pf_zlag ... zsched=zlag`,
   `RAN hashed stash_tiled_fgrid_r32_qres_pf_zlag forward=fwd_sstash_fgrid_r32_qres_pf backward=bwd_stash_tiled_pf_zlag`,
   every `BITS ... _vs_stash_tiled` MATCH,
   `REACH ... clean_restored=True reach_bit=stash_tiled_fgrid_r32_qres_pf_zlag+sabotage_new`,
   `REACH_Z ... forward_moved=0 zdot_moved_even_rows=0 zdot_moved_odd_rows=<above 0> dv_moved=0`,
   `RESOURCES_BEGIN label=zdot_zdefer_pf` and `label=zdot_zlag_pf` on the
   first run (each followed by `RESOURCES` lines or a `RESOURCES_ERROR`), a
   `REACH_KV` line on the composed arm, and
   `attention_step_price: PASS (<arm> vs stash_tiled)`.

### 17.9 The H100 leg

The body is `tools/attention_zdot_leg.sh`. Each setting only when unset:
`MOJOLEARN_ATTN_BASELINE=stash_tiled_fgrid_r32_qres_pf` (the shipped NVIDIA
default, by name);
`MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_fgrid_r32_qres_pf_zlag,stash_tiled_fgrid_r32_qres_pf_zdefer,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_zlag_kvgrid_r32`;
`MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled_fgrid_r32_qres_pf,stash_tiled_fgrid_r32_qres_pf_zlag,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_zlag_kvgrid_r32`;
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`; `MOJOLEARN_COMPILE_JOBS=8`. It refuses LM
arms without the baseline, then runs `sh tools/attention_step_leg.sh`. The
`_kvgrid_r32` and `_kvsplit` arms are the coordinator's addition (their
DigitalOcean MI325X leg,
bench/results/e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv, scored
geomeans 0.844 and 0.855 against AMD `baseline`); here they are priced against
the NVIDIA default on the same pod, kernels unchanged, and
`_zlag_kvgrid_r32` is the composed name.

NVIDIA, H100 on RunPod, from a `git worktree add --detach` checkout at the
lane's merge commit (the leg ships the committed tree). The card is supplied,
not generated, so the Mac runs no device arm:

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GPU_ARCHS=sm_90a \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_zdot_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-zdot \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
        --gpu "NVIDIA H100 80GB HBM3" \
        --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card

Gates: section 6 with `stash_tiled_fgrid_r32_qres_pf` in place of `baseline`
(arms check exit 0; smokes exit 0; every `BITS ... _vs_stash_tiled_fgrid_r32_qres_pf`
MATCH on both corpora's activations; `REACH` with `clean_restored=True`,
`REACH_Z` for the zsched arms and `REACH_KV` for the kv arms proven; lean steps
`limited: false`; `harness: DEFAULT column=nvidia arm=stash_tiled_fgrid_r32_qres_pf`
in gate.txt). Flip: 17.7. Reading against 17.2: the price `bwd (derived)` of
`_zdefer` and `_zlag` against the default's 7.7 ms per layer, and
`attn.bwd_zdot_zlag_pf` in `lmtiming-*` against the default's
`attn.bwd_zdot_stash_pf` (66.2 ms on the last pod). If the lease runs short,
drop LM arms after the price lines rank them, never the default (the witness
reference).

### 17.10 Merge note (branch `lane/attention-zdot-h100-merged`, origin/main at afba564c, source only, nothing built)

This lane was merged with section 18 (the AMD default
`stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`, word 52327). Main made the 2597
bits shippable and compiles one clean
`_launch_bwd_stash_tiled_kv[64, ATTN_DEFAULT_KV_KEYS, ATTN_DEFAULT_KV_SPLIT]`
on a shipped build whose default carries them. That helper calls
`_launch_bwd_stash_zdq_pf`, which this lane had given four runtime
branches holding the zdot schedule kernels. So the merge puts those four
branches under `comptime if ATTN_ARM_TRIAL`: a shipped build raises on a
nonzero `zsched` and instantiates no DEVIATION 2598 kernel, and the shipped
call site passes `zsched` 0 (a sixth call site beside 17.6 item 1's five). A
trial build compiles and launches what this lane did. `ATTN_ARM_DEFAULT_REFUSED_BITS`
is main's set plus `ATTN_ARM_ZSCHED_BITS`. The sabotage copies stay trial-only
(section 18.2). The arms check is this lane's 21 arms (section 18's 18 plus
the three 2598 arms), so 18.5 item 4's counts read as 17.8 item 2's on this
branch. The fused check's `DKDV` and `DEFAULT` lines are main's. RUN OWED on
the merged branch, the orchestrator's M4 light commands, one at a time:

1. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check && /tmp/fused-check`.
   Expect PASS, `DEFAULT column=apple arm=stash_tiled word=7`, `DKDV ... shipped_kv_branch=False default_kv_keys=0` (18.5 item 1).
2. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check && /tmp/fused-check`.
   Expect PASS, `DEFAULT column=apple arm=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 word=52327`, `DKDV ... shipped_kv_branch=True default_kv_keys=32` (18.5 item 2). This is the build that proves the shipped kv branch compiles with the zdot gating.
3. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check && /tmp/fused-check`.
   Expect PASS (18.5 item 3).
4. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check && /tmp/arms-check`.
   Expect `names: 39 spellings and 168 arm values round-trip, 35 invalid spellings refused`, the `REACH <arm>+sabotage_new` lines with `zdot_moved_even_rows=0` and `zdot_moved_odd_rows` above 0 for the three zdot arms only, and `transformer_attention_arms_check: PASS, names inverse, 15 cases x 21 arms` (17.8 item 2).
5. The price harness at L 512, correctness only, against `default` (on the M4, stash_tiled):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`,
   then `MOJOLEARN_ATTN_BASELINE=default MOJOLEARN_ATTN_ARM=<arm> MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`
   for `<arm>` = `stash_tiled_fgrid_r32_qres_pf_zlag` (with `MOJOLEARN_ATTN_RESOURCES=1`), then `stash_tiled_fgrid_r32_qres_pf_zdefer` and `stash_tiled_fgrid_r32_qres_pf_zlag_kvgrid_r32` (with `MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_RESOURCES=0`).
   Expect every `BITS` MATCH, `REACH_Z ... forward_moved=0 zdot_moved_even_rows=0 zdot_moved_odd_rows=<above 0> dv_moved=0`, a `REACH_KV` line on the composed arm, and `attention_step_price: PASS (<arm> vs stash_tiled)` (17.8 item 3).
6. `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`.
   Expect its PASS line unchanged (18.5 item 5).

## 18. The AMD dk/dv flip wired as the shipped default (2026-09-11, worktree lane `lane/attention-kv-default`, source built, nothing run)

STATUS: 18.1 reads a leg already filed; everything from 18.2 on is source
only. Nothing was compiled or run (no build on the Mac, by rule). The
orchestrator's M4 commands in 18.5 are the first compile. No new deviation
number: this is DEVIATION 2534's default mechanism extended to the
DEVIATION 2597 arms.

### 18.1 The AMD evidence

`bench/results/e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv` (filed on
main at db8a92ed). DigitalOcean `gpu-mi325x1-256gb`, AMD Instinct MI325X,
commit 5cc3b8df, the 16.8 body (`tools/attention_dkdv_leg.sh`) against the
then shipped AMD default `baseline`, every step witness equal on both
corpora, every price verdict PASS against baseline.

| arm | lean step enwik8 s | lean step Pile GitHub s | geomean over baseline | verdict |
|---|---:|---:|---:|---|
| baseline (shipped then) | 1.623 | 1.633 | 1.000 | |
| stash_tiled_fgrid_r32_qres_pf | 1.846 | 1.865 | 1.1400 | NO FLIP |
| `_kvrecompute` | 1.396 | 1.420 | 0.8652 | FLIP |
| `_kvsplit` | 1.383 | 1.401 | 0.8552 | FLIP |
| `_kvgrid_r32` | 1.376 | 1.370 | 0.8436 | FLIP, the winner |

In-step dk/dv: baseline 169.8 ms, the round 3 tiled fold 406.2 ms,
`_kvgrid_r32` 21.2 ms, `_kvsplit` 24.8 + 24.8 ms, `_kvrecompute` 170.6 ms.
The verdict line:

    verdict stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 FLIP geomean=0.8436 enwik8=0.8478 pilegithub=0.8394 (vs baseline, DO MI325X, witnesses equal)

The leg's smoke on that box printed `PATH candidate
arm=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 ...
resolved_hd64=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 ... fwd_rows=32
preflush=True ... kv_keys=32 kv_split=False`: on the AMD column the forward
Q residency page (15,232 B) and the joint 2597 page at 32 keys (12,288 B)
both fit. By ENGINEERING_RULES 9 the AMD default becomes the winner. It could
not until now because `ATTN_ARM_DEFAULT_REFUSED_BITS` refused every 2596 and
2597 bit: a shipped build compiled none of their kernels.

### 18.2 The change

`checks/kernel_matrix.mojo`:

- `ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32 = 52327`
  (3175 | 16384 `_kvgrid` | 32768 keys 32).
- `attn_default_arm_for`: the AMD row returns that word, with a comment citing
  the evidence path and the verdict line. NVIDIA stays
  `stash_tiled_fgrid_r32_qres_pf`, Apple and every other column
  `stash_tiled`.
- `-D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1` returns the AMD word on
  every column (a check knob, the `MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN`
  pattern; never a shipped build). A `comptime assert` refuses both knobs in
  one build.
- `attn_dkdv_keys_per_block_for`: AMD stays 32 and is now MEASURED (the
  winning arm's `_r32`); every other column stays 64. The AMD default forces
  32 with `_kvgrid_r32`, so the row and the default agree on AMD and a bare
  `_kvgrid` resolves to the default's instantiation there.

`transformer/impl/llama/fused_attention.mojo` (no vendor branch; no
environment read on a shipped build; no kernel edited):

- `ATTN_ARM_R3_KVGRID_R32_DEFAULT = ATTN_ARM_R3_DEFAULT | ATTN_ARM_BWD_KVGRID
  | ATTN_ARM_KVROWS32`.
- `ATTN_ARM_DEFAULT_REFUSED_BITS` loses `ATTN_ARM_BWD_KVGRID`,
  `ATTN_ARM_KVROWS32`, `ATTN_ARM_KVROWS64` and `ATTN_ARM_BWD_KVSPLIT`. It keeps
  the sabotages (`ATTN_ARM_SABOTAGE_KV` included), DEVIATION 2528 and
  `ATTN_ARM_BWD_KVRECOMPUTE`. `_kvrecompute` was not free to include: it has
  its own launcher (`_launch_bwd_stash_tiled_kvre`, an extra synchronize and a
  stash free) and a sabotage that launches `_attn_flip_copy_kernel`, and it
  priced third.
- `_attn_kv_key(arm)`: which 2597 dk/dv instantiation an arm resolves to, as
  `keys * 2 + split`, 0 when none (no 2597 token, `_kvrecompute`, or a page
  that does not fit). The analogue of `_attn_fwd_r2_key`.
- Module constants beside `ATTN_SHIPPED_BWD_PF`: `ATTN_DEFAULT_KV_KEYS =
  fused_attention_kv_keys(ATTN_ARM_DEFAULT)`, `ATTN_DEFAULT_KV_SPLIT`,
  `ATTN_DEFAULT_KV_KEY = _attn_kv_key(ATTN_ARM_DEFAULT)` and
  `ATTN_SHIPPED_BWD_KV = ATTN_SHIPPED_BWD_PF and ATTN_DEFAULT_KV_KEY != 0`.
- `fused_backward_launch_ran`: inside `comptime if ATTN_SHIPPED_BWD_PF`, a
  `comptime if ATTN_SHIPPED_BWD_KV` branch before the plain `_pf` launch. When
  the arm has the stash, the tiled fold, `_pf`, head_dim 64 and
  `_attn_kv_key(arm) == ATTN_DEFAULT_KV_KEY`, it launches
  `_launch_bwd_stash_tiled_kv[64, ATTN_DEFAULT_KV_KEYS, ATTN_DEFAULT_KV_SPLIT]`
  with both sabotage flags False (on AMD `[64, 32, False]`), and reports
  `bwd_stash_tiled_pf` plus `_attn_kv_ran_bits(arm, keys)`. Any other `_pf`
  arm takes the plain `_pf` launch as before. This is the same helper the
  trial tree calls.
- The sabotage instantiations stay trial-only. `_launch_bwd_stash_zdq_pf` and
  `_launch_bwd_stash_tiled_kv` chose them with a runtime `if zsab` / `if ksab`,
  which would have instantiated both copies in any build that called them.
  Each sabotage arm now sits under `comptime if ATTN_ARM_TRIAL` and raises on a
  build without the define. The shipped caller passes False, and a trial
  build compiles and launches exactly what it did before.
- `fused_attention_arm_backward_resolved`: on a shipped build with
  `ATTN_SHIPPED_BWD_KV`, an arm with `_pf` whose `_attn_kv_key` equals the
  default's resolves to `bwd_stash_tiled_pf` plus the kv bits, the same
  condition the launcher uses.
- Build-time asserts in `fused_attention_arm_from_env` (every launcher calls
  it), beside 2534's: the new literal word equals
  `ATTN_ARM_R3_KVGRID_R32_DEFAULT`; a default with a 2597 token carries the
  stash, the tiled fold and `_pf` (the parser's rule); a keys knob comes with
  `_kvgrid` and never both knobs together; the default's 2597 page fits the
  column (`ATTN_DEFAULT_KV_KEY != 0`); and a no-trial build with such a default
  compiles the branch (`ATTN_SHIPPED_BWD_KV`). The forward half of the AMD
  word is NVIDIA's and meets 2534's page assert. `_kvrecompute`, `_ztiled` and
  the sabotages stay refused.

How a shipped build decides which dk/dv kernels to compile: from the column
default alone, at comptime. `ATTN_ARM_DEFAULT` comes from the matrix row.
`fused_attention_kv_keys` resolves its keys (the `_kvgrid_r32` knob, else the
row), and `_attn_kv_key` turns them into one instantiation key. If that key
is not 0 and the build has no trial define, the backward launcher compiles
that one clean `_launch_bwd_stash_tiled_kv` instantiation, which pulls in one
2597 kernel (`fused_bwd_dkdv_r2_kernel[64, 32, False]` on AMD), plus the clean
preflushed zdot stash and dq it already compiled through `_pf`. On NVIDIA and
Apple the key is 0 and the shipped build compiles what it compiled before.

`transformer/checks/transformer_fused_check.mojo`: a `DKDV default_hd64=<kernel>
this_run_hd64=<kernel> shipped_kv_branch=<bool> default_kv_keys=<n>` line
after the `ARM` line (the kernel is named from the resolved backward word by
`dkdv_kernel_name`). Direct backward launches are counted (`bwd_launches`), and
so are those that reported a 2596 / 2597 token (`kv_launches`). The check
FAILS when the arm's resolved backward carries a kv token and no launch
reported it, or when a launch reported one the arm does not resolve to.
`require_ran` already fails on any mismatch per launch. The PASS line ends
`; dk/dv <kernel> in <n> backward launches`. `run_case` gained two `mut`
counters; the arms check does not import it.

Also: `tools/attention_step_leg.sh` copies the `DKDV` line into gate.txt with
the shipped check's `DEFAULT` and `ARM` lines. Stale "AMD baseline" comments
there and in `bench/attention_step_price_main.mojo` now name the new default.

### 18.3 Why no bit can move

No kernel was written or edited. 16.4 B (the 2597 identity argument: every dk
and dv chain from +0.0 with the shipped terms in the shipped order, keys per
block a partition, two launches sharing no accumulator, the corner flag an OR
of the same conditions, `_step_preflushed` equal to `_step` on these operands)
is unchanged. On AMD the shipped path now launches the instantiation whose bit
equality with eager 16.4 B argues. The MI325X leg measured it on the box: its
arms check printed `transformer_attention_arms_check: PASS, names inverse, 15
cases x 18 arms, every RAN buffer bit-identical to eager, ... reach proven per
branch at head_dim 64`, its price run printed `BITS ...
stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_vs_baseline ... MATCH` on the dumped
activations, and every step witness was equal. All of it went through the same
generic launch helper the trial tree calls, with the same clean kernel
instantiation. The M4 run of the 18-arm arms check is 16.7 item 2 and 18.5
item 4. The only change is
which kernels a shipped build compiles, and which of the bit-equal schedules it
launches. The sabotage gating adds a `comptime if` around launches a shipped
build never took, and changes nothing a trial build compiles or launches.

### 18.4 Risks only a build or a box can settle

1. Nothing was compiled. Likeliest faults: module-scope comptime evaluation of
   `fused_attention_kv_keys(ATTN_ARM_DEFAULT)` and `_attn_kv_key` (the first
   comptime use of `fused_attention_kv_keys`, which also names
   `fused_rows_per_block`); `comptime if ATTN_ARM_TRIAL: ... else: raise`
   inside a runtime `if` in the two launchers; a comptime global as a
   parameter of `_launch_bwd_stash_tiled_kv`; `comptime assert` with
   `is_defined` inside `attn_default_arm_for`.
2. A shipped AMD build now compiles the round 3 forward copy, the preflushed
   zdot stash and dq, and `fused_bwd_dkdv_r2_kernel[64, 32, False]`, where
   before (`baseline`) it compiled no arm kernel. Build time grows on AMD.
3. The trial LM probes on the MI325X used trial bindings. A shipped AMD binding's
   step is argued, not measured, to match the trial run of the same name: both
   launch `_launch_fwd_r2[64, 32, True, True, False]` and
   `_launch_bwd_stash_tiled_kv[64, 32, False]` with clean flags.
4. `ran` proves which branch launched, not what the kernel computed. Reach of
   the kernel content stays the trial `+sabotage_kv` (arms check, harness).
5. On a shipped build, a sabotage bit passed explicitly to
   `fused_backward_launch_arm` on a kv arm now raises instead of silently
   launching a sabotage kernel the shipped build compiled. No shipped caller
   passes one.
6. The winner was measured on one MI325X leg. No MI300X leg has priced the 2597
   arms against baseline.

### 18.5 RUN OWED on the M4 (the orchestrator's light commands, one at a time)

1. The shipped path, no trial define, Apple's row (still stash_tiled):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check`. Expect
   `DEFAULT column=apple arm=stash_tiled word=7`,
   `ARM this_run=stash_tiled is_default=True forward_hd64=fwd_sstash backward_hd64=bwd_stash_tiled`,
   `DKDV default_hd64=fused_bwd_dkdv_tiled_kernel[64] this_run_hd64=fused_bwd_dkdv_tiled_kernel[64] shipped_kv_branch=False default_kv_keys=0`
   and `transformer_fused_check: PASS, 15 cases, ... 17 direct launches RAN fwd_sstash / bwd_stash_tiled at head_dim 64; dk/dv fused_bwd_dkdv_tiled_kernel[64] in 8 backward launches`.
2. The shipped 2597 dk/dv branch on the M4, still no trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check-kv`
   then `nice -n 19 /tmp/fused-check-kv`. Expect
   `DEFAULT column=apple arm=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 word=52327`,
   `DEFAULT resolved_hd64=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`,
   `ARM this_run=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 is_default=True forward_hd64=fwd_sstash_fgrid_r32_qres_pf backward_hd64=bwd_stash_tiled_pf_kvgrid_r32`,
   `DKDV default_hd64=fused_bwd_dkdv_r2_kernel[64,32] this_run_hd64=fused_bwd_dkdv_r2_kernel[64,32] shipped_kv_branch=True default_kv_keys=32`,
   every backward status line at head_dim 64 ending `ran bwd_stash_tiled_pf_kvgrid_r32`,
   every buffer bit-identical, and
   `transformer_fused_check: PASS, 15 cases, ... 17 direct launches RAN fwd_sstash_fgrid_r32_qres_pf / bwd_stash_tiled_pf_kvgrid_r32 at head_dim 64; dk/dv fused_bwd_dkdv_r2_kernel[64,32] in 8 backward launches`
   (the 12,288 B page fits Metal's 32 KB).
3. The round 3 knob still reaches its branch:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check-r3`
   then `nice -n 19 /tmp/fused-check-r3`. Expect 15.4 item 2's lines, plus
   `DKDV default_hd64=fused_bwd_dkdv_tiled_pf_kernel[64] ... shipped_kv_branch=False default_kv_keys=0`
   and a PASS naming `RAN fwd_sstash_fgrid_r32_qres_pf / bwd_stash_tiled_pf`
   and `dk/dv fused_bwd_dkdv_tiled_pf_kernel[64] in 8 backward launches`.
4. The arms gate, trial define, unchanged from 16.7 item 2:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check`. Expect
   `names: 33 spellings and 144 arm values round-trip, 27 invalid spellings refused`,
   the five `REACH <arm>+sabotage_kv ...` lines with dk and dv moved and
   zdot, dq and the forward held, and
   `transformer_attention_arms_check: PASS, names inverse, 15 cases x 18 arms`.
5. `checks/kernel_matrix.mojo` changed, so the GEMM device check must stay
   green (the spelling in BRIEF_gemm_long_k_2026-09-11.md):
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check`
   then `nice -n 19 /tmp/gemm-device-check`. Expect its PASS line unchanged.

### 18.6 The AMD confirmation leg (owed, waiting for Andrew's go)

A shipped-default confirmation on AMD is owed: the no-trial fused check on
the box (`MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1`, whose gate.txt must show
`shipped_check: DEFAULT column=amd arm=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`,
`shipped_check: DKDV default_hd64=fused_bwd_dkdv_r2_kernel[64,32] ... shipped_kv_branch=True default_kv_keys=32`
and its PASS line), and an LM arm `default` beside `baseline` on both corpora,
whose `lm_summary.tsv` rows must read
`arm=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 arm_is_default=True` with
witnesses equal. Andrew has moved new legs to RunPod NVIDIA, so this leg waits
for his go. Nothing in this lane rents.

## 19. The zdot H100 leg and the NVIDIA dk/dv flip (2026-09-11, measured)

Leg `bench/results/e1g/2026-09-11_185833-nvidia-h100-80gb-hbm3-attention-zdot`,
commit 6d4bd867, one RunPod H100 80GB HBM3 pod at 1980 MHz, card IDENTICAL,
every phase exit 0, every LM step witness equal to the reference (the previous
NVIDIA default `stash_tiled_fgrid_r32_qres_pf`). Operand dumps are kept outside
the repository in `~/mojolearn-evidence/` under the same leg name.

Lean LM step, steady medians in seconds, ratio to the previous default:

| arm | enwik8 | Pile GitHub | geomean | verdict |
|---|---|---|---|---|
| `stash_tiled_fgrid_r32_qres_pf` (previous default) | 0.2929 | 0.2926 | 1.0000 | reference |
| `_kvgrid_r32` | 0.2900 (0.9901) | 0.2901 (0.9915) | 0.9908 | FLIP |
| `_zlag_kvgrid_r32` | 0.2895 (0.9881) | 0.2903 (0.9922) | 0.9902 | FLIP, trial only |
| `_zlag` | 0.2917 (0.9958) | 0.2919 (0.9978) | 0.9968 | FLIP, trial only |
| `_kvsplit` | 0.2978 (1.0166) | 0.2954 (1.0096) | 1.0131 | NO FLIP |

Real-activation price, fwd+bwd ratio to the previous default (above 1 is
cheaper): `_kvgrid_r32` 1.021 on both corpora, `_zlag` 1.007 and 1.008,
`_zdefer` 1.006 and 1.009, `_zlag_kvgrid_r32` in the same band, `_kvsplit`
0.978 and 0.975. In the step timers the zdot phase reads 5.5 ms per call under
both the stash copy and `_zlag`, so the 2598 schedule does not move the zdot
kernel on NVIDIA; the dk/dv phase moves from 1.0 ms (`bwd_dkdv_tiled_pf`) to
0.8 ms (`bwd_kvgrid_dkdv_pf`) per call.

Decision. `_kvgrid_r32` flips the NVIDIA row of `attn_default_arm_for` to
`ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32`, the word AMD
already ships, so the shipped NVIDIA build now compiles the same 2597 dk/dv
kernel as AMD and nothing new. `_zlag_kvgrid_r32` measured 0.9902, which is
inside the noise of `_kvgrid_r32` (0.9908), and its DEVIATION 2598 zdot
schedule is compiled on trial builds only, so shipping it would add a kernel
to every shipped build for no separable gain; it stays a trial arm. A
same-pod confirmation leg (the shipped check naming the new default, both
defaults' lean steps, and every torch column) follows the flip.

What is left in the step. The step breakdown leg on the same day
(`bench/results/e1g/2026-09-11_190725-nvidia-h100-80gb-hbm3-step-breakdown`,
bits identical with timers compiled in and switched on) attributes 145 ms of
the 300 ms native call to GEMM (48%) and 116 ms to attention kernels (39%);
nothing else exceeds 7 ms. The next NVIDIA target is GEMM
(`docs/lanes/BRIEF_gemm_kernel_2026-09-11.md`), not attention.

## 20. The final attention pass on the H100: the backward reads the forward's exp stash (2026-09-11, worktree lane `lane/attention-final-h100`, then built by the neural session on `merge/attention-final`): DEVIATIONS 2650, 2651 and 2652

STATUS. The lane wound down on Andrew's call after 20.1 to 20.5 (the
reading, the mechanism, the arithmetic and the identity argument, written
before any code). The neural session (mojolearn-d1, about 21:15Z the same
day) then built 20.6 as written, with the differences 20.6 notes at its end,
and ran 20.8 on the Apple M4; the results are in 20.11, the H100 leg's in
20.12. A build without `-D MOJOLEARN_ATTN_ARM_TRIAL=1` compiles none of the
kernels or launches below, reads no new environment, and dispatches exactly
as before (the `attn_estash_cells` host field exists on every build and
reads 0 there). Andrew called this the last optimization pass, so 20.2
first asks whether any arm can save 1 percent of the step (2.9 ms of 290
ms) at all.

### 20.1 The ranking, ms per step on the current NVIDIA default

Default `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (word 52327). Step timers
of one serialized step, `lmtiming-stash_tiled_fgrid_r32_qres_pf_kvgrid_r32-*`
in bench/results/e1g/2026-09-11_185833-nvidia-h100-80gb-hbm3-attention-zdot
(enwik8 / Pile GitHub, 12 layers, a breakdown and never a price; envelope
290.3 / 290.4 ms):

| rank | phase | ms per step | ms per call | movable by |
|---:|---|---:|---:|---|
| 1 | zdot, `attn.bwd_zdot_stash_pf` | 66.37 / 66.35 | 5.53 | this section |
| 2 | forward kernel, `attn.fwd_r2_kernel` | 20.70 / 20.74 | 1.73 | tuned in round 3 (2530, 2531, 2533) |
| 3 | dq, `attn.bwd_dq_tiled_pf` | 12.49 / 12.49 | 1.04 | a new fold kernel, at most a few ms |
| 4 | dk/dv, `attn.bwd_kvgrid_dkdv_pf` | 9.93 / 10.00 | 0.83 | flipped in section 19 |
| 5 | regime scans, forward and backward | 2.59 / 2.61 | 0.22 | seven scans per layer |
| 6 | corner flags | 0.54 / 0.55 | 0.05 | |
| 7 | scratch allocations | 0.26 / 0.26 | 0.02 | |

Launches and synchronizes (rows 5 to 7) are 3.4 ms per step in all. Fusing
the seven scans into one launch per direction and deferring both flag reads
saves at most about half of that, 1.7 ms, under the bar; not built. The
forward kernel was the round 3 target and no source reading here names a
geometry it has not priced. dq and dk/dv together are 22.5 ms and already run
2 and 4 blocks per SM (20.2). The largest movable phase is the zdot kernel.

### 20.2 Why the zdot kernel is slow on the H100: one block per SM (the readback, and a reading)

The sm_90 occupancy rule `tools/knn_selector_kernel_stats.py` computes (65,536
registers per SM, registers per thread padded to 8) gives a 256-thread block
`floor(256 / pad8(regs))` slots per SM. The zdot leg's `resources.txt` (H100,
commit 6d4bd867) against that rule, every line consistent:

| kernel | regs | padded | 256 / padded | `blocks_per_sm_256` read back |
|---|---:|---:|---:|---:|
| `zdot_stash_pf` (shipped zdot) | 134 | 136 | 1.88 | 1 |
| `zdot_zdefer_pf` (2598) | 152 | 152 | 1.68 | 1 |
| `zdot_zlag_pf` (2598) | 165 | 168 | 1.52 | 1 |
| `dkdv_recompute` | 135 | 136 | 1.88 | 1 |
| `dq_tiled_pf` | 118 | 120 | 2.13 | 2 |
| `dkdv_tiled_pf`, `kvgrid_r64` | 106 | 112 | 2.29 | 2 |
| `kvsplit_r64_fold` | 79 | 80 | 3.20 | 3 |
| `kvgrid_r32` | 63 | 64 | 4.00 | 4 |
| `kvsplit_r32_fold` | 48 | 48 | 5.33 | 5 |

The shipped zdot kernel misses two blocks per SM by 6 registers, and it is the
only attention kernel in the step at one. Its grid is 6,144 blocks per layer
on about 132 SMs, so the SM count, not the grid, bounds how many run at once.

The history reads the same way. It is a reading, not a measurement, because
no register count was ever read back for the recompute kernel:

- Section 10 timed `fused_bwd_zdot_kernel` (recompute, no stash) at 52.2 ms
  and `fused_bwd_zdot_stash_kernel` at 89.4 ms on one pod. The two sources
  differ only by two global stores per visible cell (17.2, S). Two stores
  cannot be 37 ms of bandwidth (2.4 GB per step against the 39 GB of K and V
  staging both kernels carry), but they can be the registers that take a
  kernel from at most 128 to above 128, from two blocks per SM to one:
  89.4 / 52.2 = 1.71.
- DEVIATION 2598 (`_zdefer`, `_zlag`) moved the stores and the z fold between
  phases. Registers rose to 152 and 165, occupancy stayed at 1, and the zdot
  phase stayed at 5.5 ms per call (section 19).
- DEVIATION 2528 (`_ztiled`) replaced the geometry with 16 + 16 register
  chains per thread and four staging round trips per 64-key iteration, and
  read 115.5 ms (section 13). That prices register-blocked dots, not
  occupancy.
- `_kvgrid_r32` took dk/dv from 106 regs and 2 blocks per SM to 63 and 4, and
  dk/dv moved from 1.0 to 0.8 ms per call (section 19), the one attention
  change of the day that altered occupancy and moved the step.

So a zdot arm that only reschedules work inside the same thread (tiling,
deferral) cannot move the kernel's rate: the SM still runs one 256-thread
block of four rows. What moves it is more rows per block-iteration (less
block time per row), fewer registers per thread (more blocks per SM), or
both.

### 20.3 The mechanism: take y from the forward

The backward's y at cell `(t, j)` is `ftz(identical_div(ftz(e), ftz(denom[t])))`
with `e = ftz(identical_exp(ftz(ftz(masked) - ftz(amax[t]))))`, and `masked`
is the score chain. The shipped forward `fused_attn_forward_r2_kernel` writes
exactly that `e` into its `[B, n_heads, L, S]` score scratch at every visible
cell in its pass 2, reads it in pass 3, and the launcher frees it. The
backward's zdot kernel then recomputes the whole of it: a K stage, a 64-term
score chain, `identical_exp`. Section 11.7 named keeping y across the step
(DEVIATION 2532) and stopped at the callers' stage structs and 2.42 GB.
Keeping `e` instead of y needs no forward change at all: the same forward
instantiation writes into a buffer the caller keeps instead of one the
launcher frees.

DEVIATION 2650, `_estash`. `fused_bwd_zdot_estash_kernel[64, 8, DRES=False,
SABN]` is the zdot stash copy with the y lane gone:

- A 256-thread block is EIGHT query rows of one head times 32 keys (thread
  `(tr, kj)` holds row `8 tb + tr` and key `32 kb + kj`), not four rows times
  64 lanes. Grid 12 x 256 = 3,072 blocks per layer.
- Phase 1 stages V only for the iteration's 32 keys (8 slots per thread, the
  shipped V slot count, K gone).
- Phase 2, every visible thread: the dy chain (the shipped statements) and,
  beside it, one global load of the kept `e` and the shipped division, then
  the shipped four stores (the two shared slots, `y_st`, `dy_st`).
- Phase 3, lane 0 of each of the 8 rows folds z over the 32 keys (the shipped
  fold), and the zdot store and corner test are the shipped statements.
- It reads `e_st`, `dctx`, `v_cache` and `denom`; not q, k, amax or scale.

dq and dk/dv are the shipped clean `fused_bwd_dq_tiled_pf_kernel[64]` and
`fused_bwd_dkdv_r2_kernel[64, keys, False]` on the stashes it wrote.

DEVIATION 2651, `_estash_dres`. The same kernel with `DRES=True`: the eight
rows' dctx (512 floats) are staged once per block into the shared page (2
slots per thread, one barrier), the QRES pattern of DEVIATION 2530, so no
thread holds the 64-float `vec` (64 registers of the shipped 134, `local=0`
in the readback). The dy chain then loads both operands from shared memory.

DEVIATION 2652, the kept stash in the LM step. `LlamaDeviceStages` gains one
host field, `attn_estash_cells` (0 means nothing kept). On a trial build only,
`eager_attention_forward` resets it, and when the path is fused with no eager
stages needed and the arm carries `_estash`, it grows `stages.aexp` to the
call's `[B, n_heads, L, S]` (a lean struct holds it at one element) and runs
`fused_forward_launch_estash_ran` with `stages.aexp` as the score scratch;
on FUSED_RAN it records the cell count. `llama_decoder_layer_backward_device`
passes `fwd.aexp` and `fwd.attn_estash_cells` to
`fused_backward_launch_estash_ran`. `attention_eager_core`, the only writer
of `aexp`, clears the field (on every build: a host integer nothing on a
shipped build reads), so an eager rewrite can never be read as the fused
forward's `e`. The shipped calls sit in the `else` branches unchanged.
`aexp` is the natural home: the kept values are the exp stage at the visible
cells, and `attn_materialized` stays False, as it already says `aexp` does
not hold the eager stage.

Why the rate moves, where 2528 and 2598 did not (counted at the target,
causal, `n_rep` 1; nothing measured):

| per step | shipped zdot | `_estash` |
|---|---:|---:|
| blocks | 73,728 | 36,864 |
| block iterations | 2,396,160 | 1,198,080 |
| barriers | 7.19 M | 3.59 M |
| staged floats | 9.81 G (K and V) | 2.45 G (V) |
| score chain terms | 38.7 G | 19.3 G |
| `identical_exp` | 302 M | 0 |
| `identical_div` | 302 M | 302 M |
| global stash stores | 604 M | 604 M |
| global loads of `e` | 0 | 302 M |
| z steps | 302 M | 302 M |

Block `tb` runs `tb // 4 + 1` iterations at 8 rows (4 x (1 + ... + 64) =
8,320 per head) against `tb // 8 + 1` at 4 rows (16,640). Per iteration the
dependent latency is not longer: phase 2 is one 64-term chain per thread as
before (the y lane's exp and division tail is gone, the division now runs
beside the chain), phase 3 is the same 32-step fold, and phase 1 stages half
the floats. `_estash` changes the rows per block-iteration at the same
threads per block, and drops the y path from every thread's code; 2528 and
2598 did neither.

### 20.4 Expected saving, with the arithmetic

The bar is 1 percent of 290 ms, 2.9 ms per step.

- Occupancy and latency reading (20.2). zdot time scales with block
  iterations over blocks per SM. At one block per SM, 66.4 x 1,198,080 /
  2,396,160 = 33.2 ms, a saving of 33 ms (11 percent of the step). If the
  thread code falls to at most 128 registers (the y path, `m_row`, the K
  slots and the exp temporaries removed; unknown until the readback), two
  blocks per SM give 16.6 ms, a saving of 50 ms.
- Pessimistic reading (17.2's S: the 604 M stash stores cost 27 to 37 ms and
  nothing here changes them). The other 29 to 39 ms halves, so zdot reads 44
  to 52 ms, a saving of 15 to 20 ms. The added 302 M global loads of `e` are
  bounded above by the dq kernel, which does 604 M stash loads, 302 M stores
  and its K staging in 12.5 ms: at most about 4 ms. Net at least 11 ms, 3.6
  percent.
- `_estash_dres`. If `vec`'s 64 registers leave the thread and the count
  lands at 80 or below, three blocks per SM give about 11 ms (a saving near
  55 ms). The dy chain then does two shared loads per term instead of one;
  if the H100's shared path is the bound (section 3.1's 32 words per clock
  per SM), it prices at or above `_estash`. The pair's price and RESOURCES
  lines separate registers from shared bandwidth.

Every reading clears the bar. The one cost outside time is memory: the step
keeps 12 x 201 MB = 2.42 GB of exp stash resident on the device (the 2.42 GB
section 11.7 named for 2532), held in `stages.aexp` across steps.

### 20.5 Identity argument (written before the code)

Claim. On a call where the kept stash is valid (`attn_estash_cells` equals
`B * n_heads * L * S` and was set by this call's forward), `zdot`, `y_st` and
`dy_st` after `fused_bwd_zdot_estash_kernel` hold at every visible row and cell
the bits `fused_bwd_zdot_stash_pf_kernel[64, 4, False]` writes, and dq, dk
and dv are the shipped clean instantiations on those stashes, so every
backward output is the shipped bits. On any other call the launcher runs the
shipped backward, unchanged.

1. The kept `e`. The forward that set the field ran
   `fused_attn_forward_r2_kernel[64, 32, True, True, False]` (the default
   word's forward) into the kept buffer. Its pass 1 chain is `_step_preflushed`
   over `p` ascending from +0.0 on `ftz(q[t, p])` (the Q page) and
   `ftz(k[j, p])` (the K page); the backward's y chain is `_step_preflushed`
   over `p` ascending from +0.0 on `vec[p] = ftz(q[t, p])` and
   `ks[kj, p] = ftz(k[j, p])`, the same operands in the same argument order,
   so `dot` has the same bits. `masked = ftz(_pmul(dot, scale) + 0.0)` is the
   same statement in both. The forward stores `masked`, and in pass 2 loads it
   and computes `e = ftz(identical_exp(ftz(ftz(masked) - ftz(stats[r]))))`,
   where `stats[r]` is the row maximum `m` it also stored as `amax[t]`. The
   shipped backward computes `ftz(identical_exp(ftz(ftz(masked) - m_row)))`
   with `m_row = ftz(amax[t])`. Storing and loading a Float32 returns its bits,
   so both `e` are equal. The forward stores `e` at cell `(bb nh + h) L S + t S
   + j`, the backward's `row * S + j`, the same index, and pass 3 only reads it.
2. Coverage. The forward writes `e` at every visible `(t, j)` of every row
   of its blocks: its thread `(tr, tc)` covers rows `tr + 16u` and keys
   `tc + 16v` of each 32-key iteration from `kb_lo` to `kb_hi`, and both ends
   of `_row_range` are nondecreasing in `t`. The estash kernel loads `e` only
   under the copied visibility test (`valid`, `j_lo <= j <= j_hi`), so it never
   reads an unwritten cell.
3. amax and denom. The backward's y needs the row maximum of the forward that
   wrote `e`, and `d_row = ftz(denom[t])`. In the LM step both come from the
   same `LlamaDeviceStages` the forward wrote (the backward's docstring: `fwd`
   is the saved stages of this same call). In the arms check and the price
   harness the forward and the backward run on one case in sequence, and the
   check compares the forward's amax and denom with eager's in the same run.
4. y. The estash kernel computes `yv = ftz(identical_div(ftz(e_st[row S + j]),
   d_row))`, the shipped y statement on equal operands (`ftz` of a stored
   `ftz` output is itself).
5. dy. The shipped statements: `vec[p] = ftz(dctx[t, p])` (under `DRES` the
   shared page cell `(t - t0) * 64 + p`, staged through `ftz` from the same
   index), `vs[kj, p] = ftz(v[j, p])`, `_step_preflushed(dctx, v, dy)` over `p`
   ascending from +0.0, `dyv = ftz(dy)`.
6. Partition. Row `t` lies in block `t // 8`; key `j` in iteration `j // 32`,
   which lies in `[kb_lo, kb_hi]` of the block by item 2's monotonicity; so
   every visible cell has exactly one thread `(t - 8 tb, j - 32 kb)` and one
   iteration. Every thread of a block runs the same `kb` range and reaches
   every barrier; rows at or past `L` compute nothing and only meet barriers.
7. z. The shipped chain starts at +0.0 and, after each iteration's phase 2,
   steps `_step_preflushed(dys[jj], ys[jj], z)` for `jj` ascending over that
   iteration's visible keys, iterations ascending. The estash kernel takes
   the same steps in the same phase on slots written in that phase 2 by
   threads `(tr, jj)` of the same row, from the values item 4 and 5 argued.
   The corner test and the zdot store are the shipped statements.
8. Stores and races. `y_st` and `dy_st` are written at the same visible cells
   with the same values as the shipped kernel. Each thread writes its own two
   slots and two stash cells; the global writes go to distinct cells; the
   kept `e` buffer is read only (the backward allocates its own `y_st`, so a
   second backward on the same forward, as the checks run, reads the same
   `e`).
9. The consumers. `fused_bwd_dq_tiled_pf_kernel[64]` and
   `fused_bwd_dkdv_r2_kernel[64, keys, False]` are the shipped clean
   instantiations the default launches, on stashes and a zdot with the
   shipped bits, in the shipped order.
10. Staleness. The field is set only after a fused forward returned FUSED_RAN
    into `aexp` on this call, reset at the start of every
    `eager_attention_forward` on a trial build, and cleared by every
    `attention_eager_core` (the only writer of `aexp`). A backward whose
    forward took any other path, or refused, or hit the corner, or whose
    stages were rematerialized, reads 0 and runs the shipped backward.

`_estash_dres` changes where dctx is read from (item 5) and nothing else.

Sabotage (reach, `+sabotage_new`). The estash kernel stores zdot flipped one
ulp (`_flip_ulp`) at EVEN flat rows only. zdot moves at even rows and holds at
odd rows, dq and dk move through it, dv and the forward hold. 2533's flip moves
every row and 2598's odd rows only, so `zdot_moved_odd_rows=0` with
`zdot_moved_even_rows` above 0 names this kernel. Row 0 of head 0 is even, so
every head_dim 64 case has one. The forward under `+sabotage_new` flips its Q
page (2530) and so moves `e`; the checks therefore run the estash backward's
reach on a clean forward (as REACH_Z did for 2598). Second reach, the read:
after a clean forward, the check overwrites the kept buffer with NaN and runs
the clean backward. A kernel that reads `e` moves y, so dv must move
(`REACH_E`); a kernel that recomputes y would not.

### 20.6 What was built, per file

| bit | constant | token | meaning |
|---:|---|---|---|
| 2097152 | `ATTN_ARM_BWD_ESTASH` | `_estash` | DEVIATION 2650, zdot from the forward's kept exp stash, 8 rows per block |
| 4194304 | `ATTN_ARM_ESTASH_DRES` | `_estash_dres` (with the bit above) | DEVIATION 2651, dctx rows in the shared page |

Grammar: base, `_ztiled[_r32|_r64]`, `_fgrid[_r32|_r64]`, `_qres`, `_pf`,
one of `_zdefer`, `_zlag`, `_estash`, `_estash_dres`, then `_kvrecompute`,
`_kvgrid[_r32|_r64]`, `_kvsplit`, then the sabotages. `_estash` needs
stash_tiled with `_fgrid_r32`, `_qres`, `_pf` and `_kvgrid` (any keys), and
refuses `_ztiled`, `_zdefer`, `_zlag`, `_kvrecompute` and `_kvsplit`. Both
bits are in `ATTN_ARM_NEW_BITS` and `ATTN_ARM_DEFAULT_REFUSED_BITS`.

- `transformer/impl/llama/fused_attention.mojo`: the bits, parser, name
  function, `fused_attention_arm_estash_runs`, the resolved backward word (the
  estash bits beside the kv bits on a trial build); the kernel of 20.3 with
  its page constants; `_launch_fwd_r2_keep` (the shipped forward
  instantiation into a caller's buffer), `_launch_bwd_estash[HD, DRES, BJ]`
  (two stashes, the estash zdot, a wait, the shipped dq and dk/dv, a wait);
  `fused_forward_launch_estash_ran` and `fused_backward_launch_estash_ran`,
  which run the estash path when the arm carries it and the kept stash is
  valid, and otherwise return the shipped launchers' result unchanged. Timer
  lines `attn.fwd_r2_keep_kernel`, `attn.bwd_zdot_estash_pf` and
  `attn.bwd_zdot_estash_dres_pf`, then the existing dq and dk/dv lines. Every
  launch has `step_count_launch` before it and every synchronize
  `step_count_sync`. No shipped branch or kernel is edited.
- `transformer/impl/llama/modeling_llama.mojo`: the field, its reset in
  `attention_eager_core`, the trial branch in `eager_attention_forward`
  (DEVIATION 2652).
- `transformer/checks/transformer_backward.mojo`: the trial branch at the
  fused backward call (DEVIATION 2652).
- `transformer/checks/transformer_attention_arms_check.mojo`: 23 arms (the 21
  of 17.5 plus `stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32` and
  `stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`), launched through
  the estash launchers with a kept buffer per case; under `+sabotage_new` an
  estash arm's backward runs on a clean forward and must move zdot at even
  rows only and hold dv; a fourth pass poisons the kept buffer and must move
  dv (`REACH_E`).
- `transformer/checks/transformer_fused_check.mojo`: the backward launch goes
  through `fused_backward_launch_estash_ran` with the wrapper's `stages.aexp`
  and field (the wrapper runs again right before it, because the eager
  backward reference rematerializes the stages), and a `WRAPPER` line prints
  the kept cell count. On a shipped build the call returns the shipped
  launcher's result, so the check reads as before.
- `bench/attention_step_price_main.mojo`: `Case` holds a kept buffer and cell
  count and launches through the estash launchers; `PATH` gains `estash=`; a
  `REACH_ES` run (clean forward, `+sabotage_new` backward) and a `REACH_E`
  run for estash candidates; `RESOURCES` gains `zdot_estash_pf` and
  `zdot_estash_dres_pf`.
- `tools/attention_final_leg.sh`: the leg body (20.9); `tools/attention_step_leg.sh`
  header and `gate.txt` only.

A trial build instantiates four more kernel pipelines (DRES times SABN); the
forward and the dq and dk/dv instantiations are ones a trial build already
compiles.

As built (differences from the list above, none of them numeric). The
launcher is `_launch_bwd_estash[HD, DRES, SABN]` with the dk/dv keys and the
sabotage_kv choice as runtime arguments (`_estash_dkdv_launch[HD, BJ]`
launches the joint 2597 fold, clean or sabotage_kv). The two entry points
`fused_forward_launch_estash_ran` and `fused_backward_launch_estash_ran` take
the caller's kept buffer and count; the forward one resets the count on entry
and sets it after its kernel ran to the end (FUSED_RAN or FUSED_CORNER; on
the corner the wrapper takes the eager path, which clears it), the backward
one runs the estash path only when the count equals this call's cells and
otherwise returns the plain launcher's result. `fused_attention_arm_estash_runs`
is the build-and-arm test both use (trial build, the bit, the forward
resolving to 32 rows with `_qres` and `_pf`, a dk/dv keys count, the page
fits); `fused_attention_arm_backward_resolved` adds the estash bits to the
word under the same test. `eager_attention_forward` resets the field on
every build and keeps the stash only when no eager stage is needed;
`attention_eager_core` clears it. The arms check launches EVERY arm through
the two entry points (a kept buffer per case) and adds a fourth pass for the
estash arms (`+poisoned_estash`, the REACH_E line); the fused check does the
same and prints `WRAPPER status=... estash_cells=... cells=...`. The harness
`Case` carries the kept buffer, `poison_kept`, the `REACH_ES` and `REACH_E`
runs, `PATH ... estash=`, and the `zdot_estash_pf` / `zdot_estash_dres_pf`
RESOURCES rows (trial builds only). `tools/attention_step_leg.sh` is not
edited.

### 20.7 The flip rule for this leg

ENGINEERING_RULES 9. For an arm, the geometric mean of its enwik8 and
pilegithub lean step ratios (`steady_median_seconds` of `lm-<arm>-<corpus>`
over `lm-stash_tiled_fgrid_r32_qres_pf_kvgrid_r32-<corpus>`, same pod) below
1, with `witnesses_equal_baseline=True` for every step on both corpora. A flip
needs the lane that takes it to add the shipped branch (the kernel, the two
launch helpers and the DEVIATION 2652 glue out of `comptime if ATTN_ARM_TRIAL`),
move the bits out of `ATTN_ARM_DEFAULT_REFUSED_BITS`, and edit only the
NVIDIA word of `attn_default_arm_for`.

### 20.8 RUN OWED on the M4 (the orchestrator's light commands, one at a time)

1. The shipped path, no trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check`
   then `nice -n 19 /tmp/fused-check`. Expect 18.5 item 1's `DEFAULT`, `ARM`
   and `DKDV` lines, `WRAPPER ... estash_cells=0` on every case, and
   `transformer_fused_check: PASS, 15 cases`.
2. The NVIDIA and AMD default word on the Mac, still no trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check-kv`
   then `nice -n 19 /tmp/fused-check-kv`. Expect 18.5 item 2's lines and PASS.
3. The arms gate, trial define:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_attention_arms_check.mojo -o /tmp/arms-check`
   then `nice -n 19 /tmp/arms-check`. Expect
   `names: 43 spellings and 184 arm values round-trip, 48 invalid spellings refused`;
   status lines ending `ran stash_tiled_pf_estash_kvgrid_r32` and
   `ran stash_tiled_pf_estash_dres_kvgrid_r32` at head_dim 64 (backward);
   for both arms `REACH <arm>+sabotage_new zdot_moved=<n> dv_moved=0 zdot_moved_even_rows=<n> zdot_moved_odd_rows=0`
   and `REACH_E <arm> dv_moved=<n> zdot_moved=<n>` with n above 0; and
   `transformer_attention_arms_check: PASS, names inverse, 15 cases x 23 arms`.
4. The wrapper glue, trial define, one estash arm per run:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_fused_check.mojo -o /tmp/fused-check-trial`
   then `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32 nice -n 19 /tmp/fused-check-trial`
   and the same with `stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`.
   Expect `ARM this_run=<arm> ... backward_hd64=stash_tiled_pf_estash[_dres]_kvgrid_r32`,
   `WRAPPER ... estash_cells=<B*nh*L*S>` above 0 on the head_dim 64 cases
   whose forward RAN, `fused backward status: RAN  ran <that word>` there,
   and PASS.
5. The price harness at L 512, correctness only, against the NVIDIA default
   by name:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . bench/attention_step_price_main.mojo -o /tmp/attn-price`
   then
   `MOJOLEARN_ATTN_BASELINE=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32 MOJOLEARN_ATTN_KINDS=hashed,heavytail MOJOLEARN_ATTN_TIMING=0 MOJOLEARN_ATTN_RESOURCES=1 MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 nice -n 19 /tmp/attn-price`,
   then the `_estash_dres_` arm with `MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_RESOURCES=0`.
   Expect `PATH candidate ... estash=estash` (or `estash_dres`), every `BITS`
   MATCH, `REACH_ES ... forward_moved=0 zdot_moved_even_rows=<above 0> zdot_moved_odd_rows=0 dv_moved=0`,
   `REACH_E ... dv_moved=<above 0>`, `REACH ... clean_restored=True`, a
   `REACH_KV` line, `RESOURCES_BEGIN label=zdot_estash_pf` and
   `label=zdot_estash_dres_pf` (Metal may answer `RESOURCES_ERROR`), and
   `attention_step_price: PASS`.
6. Compile proof of the backward glue (the byte LM bindings cannot build on
   the Mac), build only, both builds:
   `nice -n 19 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . transformer/checks/transformer_backward_check.mojo -o /tmp/bwd-check-trial`
   and the same without the trial define to `/tmp/bwd-check`. Expect both to
   build.

### 20.9 The H100 leg

Body `tools/attention_final_leg.sh`. Each setting only when unset:
`MOJOLEARN_ATTN_BASELINE=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (the
shipped NVIDIA default, by name);
`MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`;
`MOJOLEARN_ATTN_LEG_LM_ARMS` the baseline and both arms;
`MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1`; `MOJOLEARN_COMPILE_JOBS=8`. From a
`git worktree add --detach` checkout at the lane's merge commit:

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GPU_ARCHS=sm_90a \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_final_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-final \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
        --gpu "NVIDIA H100 80GB HBM3" \
        --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card

Expected about 20 to 25 minutes on the pod: the zdot leg's body took 22.5
minutes with 20 LM probes; this one runs 12 (three arms, two corpora, a lean
and a timing probe each, about 97 s per pair) after the same builds (about 4
minutes), dumps and prices.

Gates: section 6 with the NVIDIA default in place of `baseline` (arms check
exit 0; smokes exit 0; every `BITS ... _vs_stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`
MATCH on both corpora's activations; `REACH` with `clean_restored=True`,
`REACH_ES`, `REACH_E` and `REACH_KV` proven; lean steps `limited: false`).
Flip: 20.7. Reading: `RESOURCES label=zdot_estash_pf regs=` and
`blocks_per_sm_256=` against 134 and 1; `attn.bwd_zdot_estash_pf` in
`lmtiming-*` against 66.4 ms; `envelope.native_call` against 290.3.

### 20.10 Risks only a build or a box can settle

1. Nothing was compiled. Likeliest faults: the new `var` field and its uses
   in two modules; `len()` on a `DeviceBuffer` (the lean capacity helper is
   the precedent); `stages.aexp` passed `mut` beside other `stages` fields
   (the shipped forward call is the precedent); a comptime-sized shared
   allocation that depends on `DRES`; `~` on the arm mask in the checks.
2. The register count of the estash kernels. If it stays above 128 the arm
   gets the rows-per-block half of 20.4 only (about 33 ms); `_estash_dres`
   exists to cover that.
3. Shared bandwidth under `_estash_dres` (two shared loads per chain term).
4. The global load of `e` in the dot phase. If a global load's latency is
   longer than the 64-term chain, phase 2 lengthens; the pessimistic bound in
   20.4 already charges 4 ms.
5. Memory. 2.42 GB of kept stash on the H100 (80 GB) for the whole run, and a
   201 MB fill per layer in the first step (the probe's warmup step).
6. The lean probe's memory lines will show the kept stash; they are not the
   price.
7. The byte LM bindings first compile the DEVIATION 2652 glue on the box;
   M4 item 6 compiles the same functions in a check first.

### 20.11 The H100 leg, the flip, and the shipped branch (2026-09-11, measured): DEVIATION 2657

Pod 43v3euoz80r9zy, one RunPod NVIDIA H100 80GB HBM3, commit 8dc33f00, one
heat window, baseline `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (the NVIDIA
default of section 19). Evidence
`bench/results/e1g/2026-09-11_215636-nvidia-h100-80gb-hbm3-attention-estash`
(the operand `.bin` dumps of that directory are 6 MB each and live outside the
repo in `~/mojolearn-evidence/`; their `meta.txt` and `sha256.txt` are kept).

Lean step, `steady_median_seconds`, enwik8 / Pile GitHub:

| arm | enwik8 s | pilegithub s | ratios | geomean |
|---|---:|---:|---|---:|
| baseline `..._kvgrid_r32` | 0.29217 | 0.29157 | 1, 1 | 1 |
| `..._estash_kvgrid_r32` | 0.24374 | 0.24358 | 0.8342, 0.8354 | 0.8348 |
| `..._estash_dres_kvgrid_r32` | 0.24033 | 0.23881 | 0.8226, 0.8190 | **0.8207** |

`witnesses_equal_baseline=True` for every step on both corpora and both arms,
every leg stage exited 0, the byte LM binding built on the box, and the Apple
vs NVIDIA identity trace is IDENTICAL over 60 matched stages. ENGINEERING
RULES 9 flips the winner, so the NVIDIA row of `attn_default_arm_for` becomes
`stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32` (word 6343783).

The register lens of 20.2 predicted this and the readback confirms it. Risk 2
of 20.10 (the count staying above 128) did not happen:

| kernel | regs | padded | 256 / padded | `blocks_per_sm_256` |
|---|---:|---:|---:|---:|
| `zdot_stash_pf` (the shipped zdot) | 134 | 136 | 1.88 | 1 |
| `zdot_estash_pf` | 125 | 128 | 2.00 | 2 |
| `zdot_estash_dres_pf` | 64 | 64 | 4.00 | 4 |

`_estash` clears the 128 cliff the shipped kernel missed by 6 registers, and
`_estash_dres` (dctx rows in the shared page, so no thread holds the 64-float
vector) reaches 4 blocks per SM at a 12,480 byte page. In-step zdot falls
66.4 -> 14.9 ms per step; on real activations the backward is 7.52 -> 3.23 ms
(fwd+bwd 1.85x) with the forward unchanged at 1.00x, which is the shape the
mechanism predicts: 20.3 moves work out of the backward only.

THE SHIPPED BRANCH (what the flip owed, per 20.7). The estash kernels, their
two launch helpers and the DEVIATION 2652 glue were compiled under
`-D MOJOLEARN_ATTN_ARM_TRIAL=1` only, and `fused_attention_arm_estash_runs`
opened with `comptime if not ATTN_ARM_TRIAL: return False`, so the default
word alone would have run the old backward silently. DEVIATION 2657 adds:

- `ATTN_SHIPPED_BWD_ESTASH` (a shipped build whose column default carries the
  `_estash` bit and whose page fits) and `ATTN_DEFAULT_ESTASH_DRES`, beside
  `ATTN_SHIPPED_BWD_KV`, which they mirror.
- `fused_attention_arm_estash_runs` now admits a shipped build whose default
  carries the same estash bits, the way the 2597 test compares
  `_attn_kv_key(arm)` with `ATTN_DEFAULT_KV_KEY`.
- Both entry points are gated `ATTN_ARM_TRIAL or ATTN_SHIPPED_BWD_ESTASH`;
  inside them the `+sabotage_new` instantiations stay behind
  `comptime if ATTN_ARM_TRIAL`, and a shipped build instantiates exactly
  `_launch_fwd_r2_keep[64, 32, True, True, False]` and
  `_launch_bwd_estash[64, ATTN_DEFAULT_ESTASH_DRES, False]`.
- `ATTN_ARM_ESTASH_BITS` leaves `ATTN_ARM_DEFAULT_REFUSED_BITS`, and
  `fused_attention_arm_backward_resolved` reports the estash bits on the
  shipped path too.
- The DEVIATION 2652 glue in `modeling_llama.mojo` and
  `transformer_backward.mojo` takes the same two-part gate.
- `checks/kernel_matrix.mojo` gains the word constant (asserted equal to this
  file's `ATTN_ARM_R3_KVGRID_R32_ESTASH_DRES_DEFAULT` at build time) and a
  third check knob, `MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN`, with the
  mutual exclusion assert widened to three. The knob is how the M4 gates a
  branch Apple's own default does not carry; never a shipped build.

## 21. AMD flips estash too (2026-09-12, measured): DEVIATION 2657 on gfx942

Section 20 flipped `_estash_dres` on NVIDIA and left AMD on
`stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`, the dk/dv winner of section 18,
with the estash price on AMD unmeasured. It is measured now, on a Hot Aisle
MI300X (gfx942), and it wins there too, so `attn_default_arm_for` returns the
estash word for `COLUMN_AMD` as well and the two vendor columns agree on one
arm name for the first time since section 15.

Evidence `bench/results/e1g/2026-09-12_133010-amd-mi300x-hotaisle-attn-estash`,
commit bb679f19, one VM, one heat window, against the previous AMD default:

    verdict stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 FLIP
    geomean=0.9716 enwik8=0.9724 pilegithub=0.9709
    witnesses_equal_baseline=True on both corpora

Lean step 0.7573 / 0.7593 to 0.7363 / 0.7372 s (enwik8 / Pile GitHub).

### 21.1 The register lens on gfx942, and why the gain is a third of NVIDIA's

The same `resources.txt` readback section 20.11 introduced, on this VM, at
head_dim 64 and 256 threads per block:

| kernel | regs | shared B | blocks per CU |
|---|---:|---:|---:|
| `zdot_stash_pf` (the shipped one) | 118 | 17,696 | 3 |
| `zdot_zdefer_pf` | 116 | 17,696 | 3 |
| `zdot_zlag_pf` | 117 | 17,696 | 3 |
| `zdot_estash_pf` | 116 | 10,432 | 4 |
| `zdot_estash_dres_pf` (the winner) | 60 | 12,480 | 5 |

THIS IS THE WHOLE EXPLANATION OF THE DIFFERENCE BETWEEN THE COLUMNS. On the
H100 the shipped zdot kernel was 134 registers at ONE block per SM and the
flip took it to 64 registers at FOUR, which is why NVIDIA read 0.8207. Here
the shipped kernel already fits THREE blocks per CU and the flip takes it to
five. AMD was never as starved, so it has less to win back, and 0.9716 is
what "less to win back" looks like in a number. The register counts themselves
are nearly identical across the two vendors (118 against 134 shipped, 60
against 64 flipped), which is the reassuring part: the same kernel compiles to
about the same thread state on both, and only the occupancy arithmetic differs.

### 21.2 What this section does not claim

The leg measured a TRIAL arm against the old shipped default. A shipped gfx942
build had never compiled `ATTN_SHIPPED_BWD_ESTASH`, because until the routing
change the AMD column carried no estash bit; the leg's own shipped fused check
ran and passed, but it ran the OLD default and so is not evidence for the new
one. That branch is gated separately and the flip does not reach main without
it. `ATTN_ES_FITS` is safe by inspection at least: `column_shared_limit`
gives AMD 64 KB against the 12,480 byte `_estash_dres` page, and this VM ran
that kernel today.

No Apple estash measurement exists and none is owed: Apple's default carries
no estash bit, so a shipped Apple build compiles none of this branch, which
M4 gate C confirmed by still resolving `stash_tiled` with no knob.
