# Handoff: fused attention lane (2026-09-09)

Branch `lane/fused-attention` (worktree `.claude/worktrees/agent-a94c0adce7c14179f`),
on top of main `71d2ba71`. Every build, gate and timing below ran on rented
NVIDIA L40S pods (Mojo 1.0.0 ed45d567, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, torch
2.4.1+cu124); nothing was built or run on the Mac. Both pods are
terminated and verified gone (`mmzcdqdbg27c6w` expired on its lease
while the session was rate-limited; `n5h6et424b5oj6` DELETE 204, GET 404).

## Commits (`%h parent %p`)

- `74cffc7c parent 71d2ba71` transformer: fused attention (three-pass
  recomputation, bits unchanged) with the eager fallback, the fused check,
  lean stages, the pod driver
- `b3bbfb2d parent 74cffc7c` the gates keep the eager stages materialized
  (their device dumps read them); the regime case crosses the bound
- `16430a44 parent b3bbfb2d` surface: memcpy marshaling instead of element
  loops; `MOJOLEARN_TRANSFORMER_TIMING` phase breakdown
- `161e0874 parent 16430a44` memcpy takes keyword arguments; the first
  L40S round's logs and cards
- `8b996d6d parent 161e0874` the input refusals scan on the device; one
  memcpy per upload and download
- (this file, `bench/OPPONENT_REFERENCE.md` rows, and
  `bench/results/attnlane_fused_2026-09-09/round2_8b996d6d/` follow)

Files: `transformer/impl/transformers/models/llama/fused_attention.mojo`
(new), `modeling_llama.mojo`, `transformer/checks/transformer_backward.mojo`,
`transformer_backward_check.mojo`, `transformer_fused_check.mojo` (new),
`bindings/_mojolearn_transformer.mojo`, `transformer/bench_window_timing.py`,
`tools/attn_pod.sh` (new), `pixi.toml` (task `check-transformer-fused`).

## What changed and why

1. **The fused attention path, IDENTICAL tier only, same bits.** Forward:
   three sweeps over each row's VISIBLE key range with K/V tiles staged
   through threadgroup memory: the row max (`identical_fmax`, any order,
   contract 5.1), the denominator (scores recomputed, `exp(s - m)`, the
   ascending chain from `+0.0`), then the context (scores recomputed,
   `e / denom` one division each, the ascending fma chain). The score is
   the gemm profile's one-leaf chain (`head_dim <= 128`, `P == 1`), spelled
   with the tuned GEMM's per-step seam (`fma.rn.ftz.f32` on the NVIDIA
   column). No online softmax anywhere: the fold is the contract. Backward:
   `zdot`, `dq`, and `dk`/`dv` kernels that recompute `y` and `dy` per cell
   (stages 17-21 never materialized) and fold in the eager chains' order
   (`h` in the kv group ascending, `t` ascending).
2. **Masked cells are skipped only where that is provably exact.** Host
   regime bound before the launch (`head_dim * max|q| * max|k| < 2^100`,
   likewise `dctx`/`v`, all operands finite); a kernel-side flag when a
   chain ends its visible run holding `-0.0` (a flushed subnormal product;
   the masked tail could launder it). Either refusal runs the eager path,
   which is the profile. `transformer_fused_check.mojo` builds the
   underflow case and asserts the corner fires; the regime cases assert the
   refusal.
3. **The gate structure.** With the trace on (the cards), or through the
   planted entry point (every check's device dump), the eager kernels still
   run and record S11-S18 and stages 17-21; the fused kernels then run and
   `attn.ctx` / stages 22-24 are recorded FROM THE FUSED OUTPUT, so the
   shipped cards gate the fused bits at every fixture and every downstream
   stage. The plain entry point (`llama_decoder_layer_forward`) and the
   binding run one path only; stage buffers are allocated lean (one
   element) and grown on demand by the eager fallback.
4. **The surface was the other half of the time.** Element-by-element
   List marshaling (16.7 M appends per activation) became one memcpy per
   buffer; the per-call download-and-walk of the block input (65 ms) and
   the host walk of the weights (38 ms) became one device kernel that finds
   the first non-finite index, with the same message at the same index.
5. **Default.** Under IDENTICAL the fused path IS the default
   (`attention_path_choice` returns auto, which is fused);
   `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager` is the A/B arm. FAST and
   DETERMINISTIC builds compile to the eager path unconditionally and were
   neither built, run nor timed. Sabotage builds and planted calls take the
   eager path.

## Bit evidence (L40S, pod n5h6et424b5oj6, tree 8b996d6d,
`bench/results/attnlane_fused_2026-09-09/round2_8b996d6d/`)

- `check-transformer` identical, clause (a): PASS, 17 cases, 30/30 stages,
  349206 cells; clause (d) window 0/4/5/20 and the three split-prefill
  arms PASS (`check_transformer.log`). `transformer.identical.card` md5
  `8ce661b469681b18fb5cf4d566ad78ff`, BYTE-EQUAL (cmp, on the pod and again
  here) to `bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/transformer.identical.card`.
- `check-transformer-backward` identical, clause (a): PASS, 17 cases,
  37/37 stages, 412172 cells. `transformer-backward.identical.card` md5
  `7eee8da90ecb4dba1d154991e2e67e30`, BYTE-EQUAL to
  `bench/results/apple_cards_2026-09-03/transformer-backward.identical.card`.
- `check-transformer-fused`: PASS, 12 cases (L up to 700, window 0/7/20/50/
  64/96/2048, decode at pos0 200, a ring gather with key_lo 137, head_dim
  16/24/64/128, n_rep 1/2/4), 102 compared buffers bit-identical (forward
  ctx/amax/denom direct and through the wrapper; backward zdot/dq/dk/dv);
  the underflow case reports CORNER on both directions, the two regime
  cases are refused (`fused_check.log`).
- Surface test identical: 116 checks, 0 failed (`surface_identical.log`),
  decode == prefill, split prefill, ring snapshot, backward vs the float64
  oracle at window 0 and 3, two-call bit repeat, all through the fused path.
- The same gates were green at `b3bbfb2d` on the first pod
  (`bench/results/attnlane_fused_2026-09-09/recheck.log`, cards there too).

## Timing, ours IDENTICAL through the Python surface vs the torch rows

Medians of 3 rounds after 1 warm-up, host copies included, d_model 1024,
16 heads, 4 kv heads, head_dim 64, intermediate 4096, window 2048, batch 4.
The torch rows are READ from `bench/OPPONENT_REFERENCE.md` (seq 4096: the
samba lane's row, driver 580.126.09; seq 1024 and 16384 added by this lane
ONCE, logs named there).

| seq | arm | forward | forward+backward | log |
|---|---|---|---|---|
| 4096 | ours, parent 71d2ba71 (eager, old surface) | 514.8 | 1111.9 | `attnlane_fused_2026-09-09/base_timing_4096.log` |
| 4096 | ours, b3bbfb2d fused, old surface | 480.8 | 812.5 | `attnlane_fused_2026-09-09/timing_4096_fused.log` |
| 4096 | ours, 8b996d6d fused (default) | **249.2** | **525.8** | `round2_8b996d6d/timing_4096_fused.log` |
| 4096 | ours, 8b996d6d forced eager (A/B) | 279.9 | 847.3 | `round2_8b996d6d/timing_4096_eager.log` |
| 4096 | torch eager fp32 SDPA | 33.6 | 106.9 | reference row |
| 4096 | torch.compile | 34.7 | 93.6 | reference row |
| 1024 | ours, 8b996d6d fused | 46.2 | 93.4 | `round2_8b996d6d/timing_1024_ours.log` |
| 1024 | torch eager fp32 SDPA | 5.7 | 16.8 | reference row (this lane) |
| 16384 | ours | REFUSED (8192 position ceiling, DEVIATION 812) | | `attnlane_fused_2026-09-09/timing_16384_ours.console` |
| 16384 | torch eager fp32 SDPA | 291.6 | 931.6 | reference row (this lane) |

Ratios at seq 4096: forward 7.4x torch SDPA (target 5x, NOT MET);
forward+backward 4.9x (target 5x, met). At seq 1024: 8.1x / 5.6x.

Where the 249 ms goes (`round2_8b996d6d/timing_4096_break.console`,
synchronized host timers, one round): weights up 18, cache/x up 13,
norm1 0.7, q/k/v proj 3.7, rope+cache 0.4, **fused attention core 177**,
o_proj 2.5, mlp+residuals 30.7, outputs down 19.6. Backward call 255 ms:
before the attention 96 (mlp/norm GEMM backwards), **fused attention
backward 137**, after 22.

The first pod's surface measurement of the torch seq 4096 row was
re-run by a scripting mistake (`--seq 1024` omitted); those two logs were
renamed `mistaken_seq4096_*_rerun.log` on that pod and were lost with it.
The reference row is unchanged and is the one quoted.

## What is unfinished and why

1. **The forward target (5x) is missed because the fused kernels are
   shared-memory bound.** Thread `(row, lane)` computes one 64-term dot per
   key with one scalar LDS per fma (1:1); the LDS unit issues one warp
   instruction per cycle against four fma pipes, so the score passes run
   at a quarter of the fma rate at best, three passes deep. The fix is the
   tuned GEMM's shape: a `TQ x BK` score tile per block with each thread
   owning an `RPT x CPT` register tile (both operands staged, 8 LDS per 16
   fma at 4x4), `TQ` of 64 rows so each staged K/V tile serves 64 rows (16x
   less L2 traffic than today's 4), the row chains (denominator, context,
   zdot, dq) reading `e`/`w`/`dcell` from a `[TQ, BK]` smem tile and the
   context chain as `(row-group of 4, d-quad)` threads with float4 V loads.
   The arithmetic and fold order do not change (the tile only decides which
   thread holds which independent dot), so the gates above are the whole
   verification. Same restructuring for `fused_bwd_dkdv_kernel`
   (`BJ x TT` tiles). Expected: the 177 ms core under 60 ms.
2. **The weight upload per call (18 ms) and the output download (20 ms)**
   are the surface's statelessness; a device-resident weight cache keyed on
   the caller's buffers would remove the first. Not started.
3. **The backward's non-attention GEMMs (96 ms)** were not looked at.
4. **The Apple column of everything** (RUN OWED below). The AMD column.
5. `head_dim` outside {16, 24, 64, 128} takes the eager path; the leaf tree
   for `head_dim > 128` is not spelled in the fused kernels. ORCHESTRATOR,
   Apple RUN OWED 2026-09-09: `head_dim` 128 ALSO takes the eager path on
   any column whose shared limit is under 35,600 bytes (Apple, 32 KB): Metal
   refused the pipeline in `check-transformer-fused` at `hd128_win20_l70`.
   `fused_supported_head_dim` now reads `lib_smem_page_fits_for`; the check
   expects the column's answer. Fusing hd 128 on Apple needs a smaller key
   tile (`BK` 32 for the forward) and is part of the register-blocked
   rewrite in item 1.

## RUN OWED on the Apple M4 (the orchestrator runs these, one at a time)

    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
    MOJOLEARN_IDENTITY_TRACE=/tmp/attn.card MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D=1 \
        tools/with_identical_mode.sh pixi run check-transformer
    cmp /tmp/attn.card bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/transformer.identical.card
    MOJOLEARN_IDENTITY_TRACE=/tmp/attnb.card tools/with_identical_mode.sh pixi run check-transformer-backward
    cmp /tmp/attnb.card bench/results/apple_cards_2026-09-03/transformer-backward.identical.card
    tools/with_identical_mode.sh pixi run check-transformer-fused
    cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_transformer_surface

Expected: both cards byte-equal, fused check 12 cases PASS with every
status as on the L40S (Apple keeps the software ftz seam; the bits must
still agree), 116 checks 0 failed.

## Next commands for a fresh agent, in order

1. `git checkout lane/fused-attention`; read this file and
   `transformer/impl/transformers/models/llama/fused_attention.mojo`'s header.
2. Rent one L40S: `MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key sh tools/attn_pod.sh up "NVIDIA L40S"`
   (creates, arms `tools/runpod_guard.sh` for 60 minutes, installs pixi);
   `sh tools/attn_pod.sh ship <sha> /root/mojolearn`; put the two reference
   cards under `/root/cards/`; `sh tools/attn_pod.sh run full <job.sh>` with
   the job in this lane's session scratch (builds the three check binaries
   with `tools/with_identical_mode.sh pixi run mojo build -I . <check>.mojo -o ...`,
   the binding with `MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh`,
   runs the gates against `/root/cards`, then
   `PYTHONPATH=/root/mojolearn/python python3 transformer/bench_window_timing.py --skip-torch`
   from `/root/mojolearn/python`); `sh tools/attn_pod.sh wait full`;
   `fetch /root/jobs <dir>`; `MOJOLEARN_RUNPOD_KEY_FILE=... sh tools/attn_pod.sh down`.
3. Rewrite `fused_attn_forward_kernel` as item 1 above; gate with
   `/root/bin/fused_check` first (seconds), then the two cards, then time
   with `MOJOLEARN_TRANSFORMER_TIMING=1 ... --rounds 1` for the breakdown.
4. Merge after the Apple RUN OWED list is green.

## Wound down 2026-09-09 (round 3, register-blocked lane, Andrew's order)

Branch `lane/fused-attention-regblock` (worktree
`.claude/worktrees/agent-afe1520062beaa327`), created on main `bdc110a0`.
The wind-down order arrived while the design was being turned into code.

**What exists.** No kernel code was written to the tree. The default path
on the branch is exactly main's (the landed three-pass kernels); nothing
is behind a define because nothing was added. **No pod was rented** (no
`attn_pod.sh up`, no state file, 0.0 pod hours). **No gate ran** and no
timing was taken. The only product of this round is the design below,
which was worked out against the landed kernels, the tuned GEMM's
register tile (`gemm_identical.mojo::identical_gemm_tuned_kernel`,
`_tuned_g2r`), and the kernel matrix's `lib_smem_page_fits_for`.

### The register-blocked design as worked out (not built)

Every kernel keeps 256 threads. The rows-per-block kernels (forward,
zdot, dq) own `TQ = 64` query rows of one `(batch, head)`; dkdv owns
`BJ = 64` keys of one `(batch, kv head)`. The score tile of one key (or
query) block is `64 x BK` with thread `(tr = tid // 16, tc = tid % 16)`
holding rows `tr + u * 16` (u < 4) and columns `tc + v * 16` (v < BK/16),
which is the tuned GEMM's `accrow`/`acccol` mapping and its 4x4 register
tile at `BK = 64`. Both operands are staged per `KS`-wide window of
`head_dim` as `[rows][KS + 4]` shared pages (the GEMM's `SSTRIDE`, float4
loads along `p`, bank-conflict free), flushed once at staging
(`ftz_simd`), and the dot is the same ascending `_step` chain over `p`
(windows ascending, `p` ascending inside each), so the score bits cannot
move. `KS = 16` or `32` where `head_dim % KS == 0`, else `KS = head_dim`
(hd 24).

Row chains. Pass 1 (max) folds each thread's cells into per-row register
partials, then 16 partials per row combine through shared memory (any
order, contract 5.1). Pass 2 (denominator) stores `e` into a row-major
`[64][BK + 1]` shared tile and threads `tid < 64` walk their row's cells
ascending with `ftz(ftz(acc) + ftz(e))`, skipping masked cells by the
row's `[j_lo, j_hi]` predicate exactly as today. Pass 3 (context) stores
`w` TRANSPOSED as `[BK][64 + 4]` so the chain thread `(row group of 4,
d-quad)` reads one float4 of `w` (4 rows) and one float4 of `v` (4 d's)
per key: 2 LDS.128 per 16 fma, with the visible-range predicate per row
per key (a compare against a comptime column index) and `-0.0` corner
check at the end exactly as today (`acc == -0.0 and j_hi < s - 1`).
Head dims above 64 give each thread `ceil((64/4) * (hd/4) / 256)` chain
slots (2 at hd 128); hd 16 and 24 leave chain threads idle. The V tile
`[BK][hd]` (no pad; every chain thread reads the same key row at once)
is staged into the SAME buffer as the q/k windows after the score phase
(a union, one extra barrier per key tile), which is what makes hd 128 fit
a 32 KB column. zdot keeps two row-major tiles (`y`, `dy`) and a 64-thread
chain; dq stores one transposed `dcell` tile and stages the full K tile
into the window buffer for a `(row group, d-quad)` chain; dkdv's score
tile is `[64 keys] x [TT queries]`, the per-query scalars `amax/denom/
zdot` are loaded per column, `y` and `dcell` are stored `[TT][64 + 4]`,
and the full q and dctx tiles `[TT][hd]` share the window buffer for the
`(key group, d-quad)` chain over queries ascending inside heads ascending,
with the per-head `-0.0` check as today.

Page bytes (floats x 4; `stg` is the union buffer, `tile` the e/w tile
which is `max(64 * (BK + 1), BK * 68, 2 * 64 * 16)` floats, plus 64 floats
of row scalars in the forward):

| kernel | hd | plan (BK or TT, KS) | bytes | 48 KB (NVIDIA) | 32 KB (Apple) |
|---|---|---|---|---|---|
| forward | 64 | BK 64, KS 32 | 36,096 | fits | no |
| forward | 64 | BK 32, KS 32 | 22,784 | fits | fits |
| forward | 128 | BK 64, any KS | 50,432 | no | no |
| forward | 128 | BK 32, KS 32 | 25,344 | fits | fits |
| forward | 16 | BK 64, KS 16 | 27,904 | fits | fits |
| forward | 24 | BK 64, KS 24 | 32,000 | fits | fits (768 B spare) |
| zdot | 64 | BK 64, KS 32 | 51,712 | no | no |
| zdot | 64 | BK 64, KS 16 | 43,520 | fits | no |
| zdot | 64 | BK 32, KS 32 | 30,720 | fits | fits |
| dq | 64 | BK 64, KS 32 | 35,840 | fits | no |
| dq | 64 | BK 32, KS 32 | 22,528 | fits | fits |
| dkdv | 64 | TT 32, KS 32 | 33,792 | fits | no |
| dkdv | 64 | TT 16, KS 32 | 20,224 | fits | fits |
| dkdv | 128 | TT 32, KS 32 | 50,176 | no | no |
| dkdv | 128 | TT 16, KS 32 | 25,088 | fits | fits |

The intended resolver is one comptime scan per (kernel kind, head dim)
over the candidate list `(64,32), (64,16), (32,32), (32,16), (16,16)`
(dkdv over `(32,32), (32,16), (16,32), (16,16)`), taking the first plan
whose bytes pass `lib_smem_page_fits_for[TARGET_COLUMN, bytes]()`, with
the plan encoded as `bk * 1000 + ks` in a comptime Int and
`fused_supported_head_dim` reporting whether the resolved plans of all
four kinds fit. No new kernel-matrix row is needed. The kernel matrix's
`lib_smem_page_fits_for` docstring still says hd 128 claims 35,600 bytes
and takes the eager path on Apple; that sentence becomes stale the day
the BK 32 plan lands (the orchestrator owns that file).

Why the masked cells are handled by predicate and not by folding
`+0.0` weights. Folding them would be exact for the denominator (the
chain is never `-0.0`) and would reproduce the eager tail laundering for
the context, zdot and dq chains, which would make the `underflow` case
report RAN instead of CORNER; the round-3 brief asks for the corner
semantics unchanged, so the chains skip by predicate (2 integer compares
and a predicated fma per cell, second order next to the 64-term dots).

### Where the 177 ms forward core goes, as estimated (not measured)

The landed kernel stages one 16 KB K tile per 4 query rows per pass
(about 100 GB of L2 to SM traffic at the Samba shape), runs the
denominator chain on 4 of 256 threads per block, and runs the context
chain as a runtime loop with two scalar LDS and a branch per fma. The
tiled design cuts the staging traffic about 8x (64 rows per staged tile,
q re-staged per window), runs 64 chains per block, and unrolls the chains
over comptime column indices. The `identical_exp` (about 25 fp32 ops)
and `identical_div` (one hardware division) seams are cheap; they are
not where the time is. This paragraph is an estimate; the next agent
should measure the phase breakdown before and after
(`MOJOLEARN_TRANSFORMER_TIMING=1 ... --rounds 1`).

### RUN OWED on the Apple M4

Nothing new. The branch's code is main's; the round-2 list above stands.

### Next commands for a fresh agent, in order

1. `git checkout lane/fused-attention-regblock`; read this section, the
   file header of `fused_attention.mojo`, and the tuned GEMM's
   `identical_gemm_tuned_kernel` accumulate loop and `_tuned_g2r`.
2. Write the four kernels as above in `fused_attention.mojo` with the
   comptime plan resolver; keep the launcher signatures (raw buffers) so
   `modeling_llama.mojo` and `transformer_backward.mojo` do not change.
   A shared-memory pointer crosses a helper boundary with
   `[origin: MutOrigin, //]` and
   `MutPointer[Float32, origin, address_space = AddressSpace.SHARED]`
   (`checks/shared_pointer_probe.mojo`).
3. Add fused-check cases for hd 128 at the BK 32 plan and for a visible
   run length that is not a multiple of 64 or of BK (for example L 150
   with window 45), and print the resolved plans in the check's header.
4. Rent one L40S (`tools/attn_pod.sh up "NVIDIA L40S"`), ship the commit
   ONCE (the archive is about 10 MB at 30 KB/s), then iterate with
   `attn_pod.sh put` on the single changed file; build with
   `tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_fused_check.mojo -o /root/bin/fused_check`
   and run it first (seconds), then the two cards, the surface test, and
   the timing per the round-2 job log
   `bench/results/attnlane_fused_2026-09-09/round2_8b996d6d/full.log`.

## Resumed 2026-09-09: register-blocked hd64 forward

The hd64 forward now uses 64 query rows per block, a 32-key tile, and
16-wide contracted windows. Each of 256 threads holds a 4x2 score tile
in registers; the staged operands feed eight independent ascending dot
chains. The denominator still folds keys ascending on one thread per row,
and context threads hold sixteen independent output chains, each folding
keys ascending. The three passes, scalar seams, visible predicates,
regime refusal and signed-zero corner fallback are unchanged. Backward
and other head dimensions retain their previous implementation.

Shared allocation is 17,152 bytes: 2,048 floats reused for Q/K windows and
V, 64x33 floats for scores/weights, and 128 row scalars. This fits Apple’s
32 KB limit; the existing conservative hd64 support check claims 19,744
bytes and remains sufficient. No kernel-matrix row changed. The existing
hd128 refusal on Apple remains.

### NVIDIA results

Dedicated pod `m4gh2e66ikwihq`, L40S, driver 580.159.03, torch 2.4.1+cu124.
Base source `497e517c9b308c24320b383c0d80b0ec54482ca7`; candidate is that
source with the two attention files changed. Only our IDENTICAL arm ran.
Evidence: `bench/results/attnlane_regblock_2026-09-09/jobs/`.

Same pod, same driver, same shape and timing harness for all rows below:
d_model 1024, heads 16, kv heads 4, head_dim 64, intermediate 4096,
window 2048, sequence 4096, batch 4; one warmup, median of three rounds.
The Python surface includes our host transfers, as in the original table.
Torch SDPA was measured once for this previously absent driver tuple.

| Arm | Forward ms | Forward + backward ms | Source log |
|---|---:|---:|---|
| Before (landed fused kernels) | 252.0 | 532.5 | `baseline_timing.json` (text) |
| Register-blocked BK32, retained | 154.7 | 444.9 | `candidate_timing.log` |
| Torch eager FP32 SDPA | 35.9 | 114.3 | `torch_reference.log` |
| BK64 experiment, rejected | 209.5 | 453.4 | `bk64_timing.log` |

Retained candidate / opponent: **4.31x forward, 3.89x forward + backward**.
Forward decreased 38.6%; forward + backward decreased 16.5% from the
same-pod baseline. Synchronized phase diagnostics put the attention core
at about 75 ms, down from 177 ms; backward remains about 138 ms.

BK64 (4x4 score registers, 33,536 shared bytes, matrix fit selecting BK32
on Apple) passed its NVIDIA bit gate but increased core time to about
88 ms. Its implementation is preserved only as
`rejected_bk64.mojo.txt` in the results directory; it is not product code.
No timing claim uses the earlier driver’s opponent row.

Validation on NVIDIA, retained BK32:

- `check-transformer-fused`: 13 cases, every compared buffer bit identical,
  all expected statuses. Added L150/window45, crossing query and key tile
  boundaries; existing decode, ring-span, underflow and regime cases pass.
- Forward check with clause (d): PASS; card byte equal by `cmp` to the
  shipped Apple forward card.
- Backward check: PASS; card byte equal by `cmp` to the shipped Apple
  backward card.
- IDENTICAL transformer binding rebuilt; surface: 116 checks, 0 failed.
- `full.rc` is 0, including both card comparisons and surface validation.

### Apple RUN OWED

The orchestrator already ran the new 13-case fused check: PASS, every
compared buffer bit identical (`/tmp/mojolearn-attention-next-apple.log`).
The final source remains BK32; the rejected BK64 experiment only ran on
the isolated pod. The original full Apple list above (binding, forward
card with clause d, backward card, surface) is still owed to this change
until the orchestrator records its completion. No agent ran Mac builds.

Backward register blocking, hd128 Apple fusion, persistent weights and
AMD validation remain unfinished. The existing 8192-position ceiling is
unchanged.

Pod cleanup: DELETE HTTP 204 at 21:06:02 EDT, GET HTTP 404 at 21:06:07;
verified gone. Approximately 0.18 pod hours, lease never extended.

### Completed Apple integration, 2026-09-09

The final BK32 source passed all 13 fused cases, both original forward and
backward cards compared byte-for-byte equal, and the rebuilt IDENTICAL
transformer binding passed 116/116 surface checks. Logs are in
`bench/results/attnlane_regblock_2026-09-09/apple/`. No Apple run remains
owed for this change; AMD remains unmeasured.
