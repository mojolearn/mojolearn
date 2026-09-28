# neural-apple: progress (Apple Metal speed, neural family)

Branch `lane/neural-apple` (off origin/main, then merged origin/lane/merged at
1c355dfd4). BEFORE arm = `lane/neural-apple-base` (b11745d8e): origin/lane/merged
plus `tools/apple_speed_neural/profile.sh` only. Home Mac for speed jobs:
m3ultra-b first, then **m4pro-b** (the orchestrator moved this lane there on
2026-09-28; m3ultra-b runs the M3 identity sweep). Every before/after pair below
ran on m4pro-b (Apple M4 Pro, 48 GB) through `tools/apple_steward.py submit
--kind speed`. No verification runs in this lane (Andrew, 2026-09-28): bits are
shown by digests the timing runs print.

## Tools (tools/apple_speed_neural/)

- `profile.sh`: every GPU lane of `tools/bench_board_neural.py` (public API, `ours`
  arm, median of 5 rounds after a warm-up, output digest per round), then the byte
  LM lean step at the T3 shard (B4 L2048 d768 12L h12 ff2048 V50257) with component
  timing; `NEURAL_CENSUS=1` adds a `-D MOJOLEARN_STEP_PHASE_TIMERS=1` byte LM copy
  and its per-step launch/sync counts.
- `attn_arms.sh`: the byte LM step under each no-trial attention default knob, with
  final witnesses.
- `attn_trial.sh`: one `MOJOLEARN_ATTN_ARM_TRIAL` build, the step under named arms.
- `gemm_geom.sh`: the twelve T3 GEMM calls (bench/gemm_excp_ab_main.mojo) under
  Apple matrix-tile geometries, hash and median per call.
- `ab.sh`: alternating A/B of public lanes between commits (one worktree per
  commit, only the named bindings built).

## Profile (BEFORE, m4pro-b)

Byte LM T3 shard step 10.07 s (813 tok/s). Uninstrumented component split:
blocks backward 6.49 s (attention backward 2.85 s, after-attention 1.15 s), blocks
forward 2.01 s (attention core 0.92 s), head GEMMs 1.25 s, CE 0.22 s. Census: 804
launches, 610 waits per step.
Rough rates: block GEMMs about 1.3-2.2 TF/s (30-50% of the M4 Pro's fp32 rate),
attention at about 0.3 TF/s (7%). The weight-gradient GEMMs (`dB`, k = 8192
tokens, output 768 x 768 or 768 x 2048) run 2-3x slower than the forward GEMM of
the same FLOPs (census: proj_dB 1037 ms vs proj_fwd 357 ms, gateup_dB 962 vs 356).
The matrix plan gives them 144-384 blocks against 1536 for the forward.

## IDENTICAL changes (all bit-inert by construction; digests below)

| commit | change |
|---|---|
| a64433591 | fused attention: regime scans in one wait (`device_absmax4`, the same `absmax_partial_kernel` launches into one partial buffer); `_read_flags` one wait; `_zero_flag` no wait |
| 727a12f04 | Apple forward/backward attention launchers: no host wait between in-order launches (scratch allocation, after zdot) |
| 1ba4e0a98 | samba ops (`training/samba_ops.mojo`): one wait per op |
| 3149fe96d | mamba3 forward binding: uploads and downloads in flight, one wait |
| f0259f772 | mamba1/mamba2 forward bindings: downloads straight into the caller's arrays, one wait |
| 30497d57e | transformer forward binding: uploads without a wait (the block's own first wait covers them), one wait after the downloads |

### Before -> after, IDENTICAL, m4pro-b (b11745d8e -> 30497d57e)

| lane (bench_board_neural, `full` shape) | before ms | after ms | digest |
|---|---|---|---|
| lm-train-step (B1 L2048 d384 8L V8192) | 1143.3 | 1150.0 | train lane (no digest) |
| lm-forward | 396.7 | 391.2 | equal |
| gemm 4096^3 | 122.4 | 122.1 | equal |
| transformer-forward | 41.4 | 39.4 | equal |
| mamba1-forward | 24.4 | 36.4 (?) | equal |
| mamba2-forward | 43.1 | 42.0 | equal |
| mamba3-forward | 33.2 | 31.9 | equal |
| samba-train-step | 1693.6 | 1680.5 | losses equal |
| samba-forward | 81.3 | 77.2 | equal |
| mlp-train-step | 6.10 | 6.13 | losses equal |
| byte LM T3 shard step (s) | 10.067 | 10.065 | final witness equal |

Census waits per step 610 -> 214 (launches 804 unchanged), final witness
(gradients, parameters, m, v, flags) equal. On the M4 Pro those waits were not the
cost: the step time did not move. mamba1-forward's 24 -> 36 ms is under an
alternating A/B (`ab.sh`) before anything is concluded.

## Attention default word (IDENTICAL, byte LM B1 L2048 d768 12L, m4pro-b, 30497d57e)

| word | steady s/step | final witness |
|---|---|---|
| Apple default `stash_tiled_fgrid_r32_qres_pf` | 3.354 | e295c543 (params) |
| `..._kvgrid_r32` (AMD's) | 3.303 | equal |
| `..._estash_dres_kvgrid_r32` (NVIDIA pre-bswz) | **2.918** | equal |
| `..._estash_dres_kvgrid_r32_bswz` (NVIDIA's) | 2.955 | equal |

The estash word is 13% faster per step on the M4 Pro with every witness equal. It
keeps a `[B, nh, L, S]` exp stash per layer, which ran out of memory at 12 layers on
the 16 GB M4 (Sep 26), so it is not flipped yet.

### Estash word by memory (16b924fe1, orchestrator's rule)

The Apple row of `attn_default_arm_for` is now NVIDIA's pre-bswz estash word
`stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`, GATED AT RUN TIME:
- a shipped Apple build compiles both the estash word and the round 3 word;
- `fused_attention_arm_from_env` returns the estash word only after a trainer's
  `attention_estash_memory_grant`. The byte LM trainer calls it once, after its
  persistent buffers exist. The grant requires every layer's kept exp stash plus
  one layer's y/dy scratches (4 bytes x b x nh x l x s each) to fit under 60% of
  `DeviceContext.get_memory_info()` free memory;
- a refusal is sticky for the process;
- with no grant (TransformerBlock, Samba, any process without a byte LM trainer,
  a Mac without the memory) it runs the round 3 word, exactly the old path.

Bits: every final witness is equal across the words (table above). Gain at
B1 12L on the M4 Pro: 13% per step. `-D MOJOLEARN_ATTN_APPLE_R3_ONLY=1` restores
the old build default.
