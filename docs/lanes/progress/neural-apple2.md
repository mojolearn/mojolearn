# neural-apple2: progress (Apple speed round 2, neural family)

Branch `lane/neural-apple2`, forked from `lane/apple-merged` at 037daa353.
Family: transformer, Mamba-1/2/3, Samba, MLP, embedding, byte LM, training.
Brief: ~/mojolearn-evidence/apple2_speed_brief.md. Round 1:
docs/lanes/progress/neural-apple.md.

Rule for every A/B: IDENTICAL bits must not move (GPT-3 route B on AMD
depends on them). Every row below names its digests (bench lane output
digests, byte LM final witness and loss digests).

## Jobs

| steward id | Mac | commit | what |
|---|---|---|---|
| 1790611567835 | m4pro-a | a2c1420ef | A/B base=35d08f9ca vs new=a2c1420ef (samba-train-step, samba-forward, mamba3-forward, lm-train-step, lm-forward, transformer-forward, gemm; 2 alternations) + byte LM T3 step A/B (2 alternations, witness + loss digests) + GEMM small-tile window check on the M4 Pro (default, SMALL_KB 16, no small tile) |
| 1790603073363 | m3ultra-b | 35d08f9ca | profile (every bench lane, T3 shard step, census with attention per-kernel timers; the M3 Ultra grants the estash word at T3, so the recompute change below cannot run there), Samba train step cProfile, attention arms at T3 (granted, FORCE_DENY + NO_ERECOMP = the old denied path, FORCE_DENY = recompute), GEMM geometry sweep (default, KB_WIDE 32, GROUP_M 16, DB, KB_WIDE 32 x SGM 4, no small tile, SPLIT 2048, SPLIT 4096, SPLIT 2048 x KB_WIDE 32), mamba1/mamba2/transformer forward A/B b11745d8e vs 30497d57e vs ca692c7e2 |

## Changes

| commit | change | default? | status |
|---|---|---|---|
| ca692c7e2 | Apple matrix GEMM: double-buffered shared pages (`-D MOJOLEARN_APPLE_MMA_DB`) | opt-in arm | REJECTED: 1.5-2x slower on every T3 call (M3 Ultra); stays an arm |
| 0d5d08027 | Apple matrix GEMM: per-leaf wide window (`-D MOJOLEARN_APPLE_MMA_KB_WIDE=32`) | opt-in arm | faster on small-tile calls, slower on default-tile calls (M3 Ultra): see 26f39f2cd |
| 26f39f2cd | Apple matrix GEMM: the SMALL tile walks k in 32-step windows when the leaf allows (`APPLE_MMA_SMALL_KB`, `-D MOJOLEARN_APPLE_MMA_SMALL_KB=16` reverts) | DEFAULT | M3 Ultra: every hash equal, small-tile calls 16-30% faster; M4 check + step A/B in job 2 |
| 35d08f9ca | Apple matrix GEMM: leaf-group split (`-D MOJOLEARN_APPLE_MMA_SPLIT_BLOCKS=<n>`; power-of-two leaf groups aligned at leaf 0, group nodes folded by `_ksplit_fold_launch`) | opt-in arm | hashes equal; head_dA 179 -> 124 ms but no better than the small-tile KB 32 default overall; stays an arm |
| 68077a9d9 | Mamba-1/2/3 backward bindings: the binding's process-lifetime context (`neural_ctx`) instead of a fresh `DeviceContext()` per call (a new Metal queue and pipeline compiles on every call) | DEFAULT | measuring (job 2) |
| 0678acea6 | Mamba-3 backward: scratch allocations without a wait each | DEFAULT | measuring (job 2) |
| 815db5956 | Apple attention: a process DENIED the kept exp stashes recomputes one layer's stash in the backward (estash forward into scratch + estash backward) instead of the round 3 zdot | opt-in since 07b658ab3 (`-D MOJOLEARN_ATTN_APPLE_ERECOMP`) | REJECTED as a default: witness equal but 3.619 s vs 3.537 s (round 3 backward) at T3 on the M3 Ultra |

Shared code note: the GEMM arms touch `gemm/checks/gemm_identical.mojo`
(every Apple IDENTICAL GEMM, classical families included) but only under
their defines; the recompute touches `fused_attention.mojo` and
`transformer_backward.mojo` (byte LM trainer only: the grant is asked by the
byte LM trainer alone).

## Job 1 findings (m3ultra-b, Apple M3 Ultra 256 GB, 035d08f9ca build = 037daa353 default path)

BEFORE numbers on the M3 Ultra (bench_board_neural `full`, median of 5, ms):
lm-train-step 278.2, lm-forward 222.4 (digest 4a8e781b0739a038), gemm 4096^3
80.8 (535b4c27bd9313d1), transformer-forward 23.7 (d5a2b289afdb5709),
mamba1-forward 20.1 (dfe79ab628aa17cf), mamba2-forward 29.2
(00da58895303c580), mamba3-forward 25.0 (481c50cae2dd749e), samba-train-step
889.9, samba-forward 54.2 (ddce61948b8456e0), mlp-train-step 8.1.

Byte LM T3 shard (B4 L2048 d768 12L V50257), estash GRANTED here (need 14.6 GB,
free 213 GB): 2.917 s a step (2808 tok/s). Final witness gradients ab96db5b...,
parameters ecaba3f7... = round 1's M4 Pro BEFORE (so the IDENTICAL bits at the
T3 shape are unchanged by apple-merged and by this lane's default path);
loss digests steps 1-4 676298da, afc46227, 34fe4c49, 50902bed.

Census (timers on, 3.48 s): attention backward zdot (the Apple matrix estash
zdot kernel) 733 ms, estash forward 274 ms, dk/dv 149 ms, dq 121 ms; GEMMs
about 1.6 s in all (head_dA 179, gateup_dB 177, proj_dB 160, gateup_dA 157).

Samba train step (cProfile, 3 steps): 2.17 s of 2.66 s in
`_mojolearn_mamba.mamba3_backward` (362 ms a call; the transformer backward is
28 ms a call). Cause: `mamba3_prefill_backward` (and the mamba1/mamba2 ones)
built a fresh `DeviceContext()` per call. Fixed in 68077a9d9.

Attention arms at T3 (5 steps each, same build): granted estash 2.923 s;
FORCE_DENY + NO_ERECOMP (the old denied path, the round 3 word) 3.537 s;
final witness digests (gradients 8a12ffdb..., parameters daea9cb6...) and
loss digests EQUAL across the two words.

GEMM geometry sweep (M3 Ultra, T3 calls, ordinary operands, ms; every hash equal
across all geometries; the `kbw32sg4` arm did not build):

| call | default | KB_WIDE 32 | GROUP_M 16 | DB | no small tile | SPLIT 2048 | SPLIT 4096 |
|---|---|---|---|---|---|---|---|
| proj_fwd | 2.31 | 1.96 | 2.33 | 3.68 | 1.97 | 2.47 | 2.38 |
| proj_dA | 2.30 | 1.93 | 2.35 | 3.58 | 2.02 | 2.48 | 2.47 |
| proj_dB | 3.03 | 2.27 | 2.98 | 5.62 | 2.53 | 2.17 | 2.04 |
| gateup_fwd | 4.36 | 5.27 | 4.62 | 5.72 | 4.65 | 4.56 | 4.46 |
| gateup_dA | 6.25 | 4.77 | 6.04 | 10.42 | 4.83 | 5.09 | 4.90 |
| gateup_dB | 7.11 | 5.10 | 6.94 | 12.69 | 5.20 | 4.90 | 4.93 |
| down_fwd | 5.92 | 4.78 | 6.19 | 10.38 | 5.04 | 4.89 | 5.16 |
| down_dA | 4.53 | 5.42 | 4.59 | 5.89 | 4.56 | 4.76 | 4.65 |
| down_dB | 7.07 | 4.94 | 6.98 | 12.61 | 5.31 | 4.81 | 4.80 |
| head_fwd | 102.4 | 116.7 | 102.5 | 130.7 | 104.9 | 102.5 | 102.4 |
| head_dA | 179.3 | 128.0 | 179.2 | 357.5 | 148.6 | 124.3 | 124.4 |
| head_dB | 115.6 | 137.5 | 116.7 | 156.3 | 115.3 | 115.9 | 114.9 |

mamba1-forward A/B (the round 1 open item), M3 Ultra, alternating, 3 reps,
ms: b11745d8e 20.25 / 20.50 / 20.29; 30497d57e 21.87 / 19.79 / 19.92;
35d08f9ca 20.45 / 19.52 / 19.95; digest dfe79ab628aa17cf in every race.
RESOLVED: no regression. The 24.4 -> 36.4 ms on the M4 Pro was one run on a
box that wedged in the same job. mamba2-forward and transformer-forward also
flat, digests equal (00da58895303c580, d5a2b289afdb5709).
