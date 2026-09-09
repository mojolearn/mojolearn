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
