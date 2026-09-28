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

### The gate's measured boundary (8baa1d382)

On the 48 GB M4 Pro, the trial build's default word (estash) at the T3 shard (B4,
12 layers, V 50,257) trained at **7.2 s a step**; the shipped build runs 10.07 s.
That run also had the small GEMM tile below 512 tiles. The process footprint
reached 41 GB, and after the final witness a Metal command buffer never
completed. The worker was killed; the GPU stayed wedged (the killed process sat
in exit state, and every new Metal job blocked). **This Mac needs a reboot.** It
was reported to the orchestrator; nothing was done on the Mac beyond this
lane's own processes.
The gate now counts every layer's stash, one layer's y/dy and the logits with
their gradient (2 x 4 x B x L x V bytes), and grants only under 35% of free
device memory:
- that shape needs 14.6 GB and is refused;
- B1 x 12L needs 3.6 GB and is granted;
- a 256 GB M3 Ultra would grant B4.
`byte_lm_attention_estash_gate` reads back [gated, granted, denied, need, free],
and `tools/lm_step_memory_probe.py` records it as `attention_estash_gate`.

## Weight-gradient GEMMs: small tile (4b2f22927, 82cfbc690)

Below 2048 default tiles, the Apple matrix plan now runs the same kernel at
FM = FN = 2 (a quarter tile, four times the blocks). T3 shard calls on the M4 Pro
(bench/gemm_excp_ab_main.mojo, ordinary operands, ms). Every hash is equal
across the six geometries (default, no small, small < 2048, small FN 4,
FM = FN = 2 everywhere, SGM 4 x SGN 2):

| call | no small | small < 512 | small < 2048 |
|---|---|---|---|
| proj_fwd | 7.2 | 7.3 | 6.1 |
| proj_dA | 7.4 | 7.4 | 6.1 |
| proj_dB | 21.5 | 10.5 | 10.4 |
| gateup_fwd | 14.4 | 15.1 | 14.4 |
| gateup_dA | 19.0 | 19.1 | 16.5 |
| gateup_dB | 40.9 | 20.3 | 20.5 |
| down_fwd | 18.6 | 18.8 | 16.4 |
| down_dA | 14.4 | 14.4 | 14.9 |
| down_dB | 40.9 | 20.6 | 20.1 |
| head_fwd | 292.6 | 291.1 | 291.5 |
| head_dA | 604.5 | 592.5 | 517.4 |
| head_dB | 343.6 | 345.1 | 345.3 |

Summed over one step's calls (48 proj x 3, 24 gateup x 3, 12 down x 3, 1 head x
3), that is about 1.5 s less GEMM time per T3 shard step. Operands with
non-admitted windows ("mixed", "sparse") are 5-7x slower on every geometry; the
real step's head timings match the "ordinary" kind. This tile change reaches
every Apple IDENTICAL GEMM with fewer than 2048 default tiles, classical users
included; their bits are unchanged by construction.
