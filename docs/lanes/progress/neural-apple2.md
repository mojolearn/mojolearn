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
| 1790603073363 | m3ultra-b | 815db5956 | profile (every bench lane, T3 shard step, census with attention per-kernel timers; the M3 Ultra grants the estash word at T3, so the recompute change below cannot run there), Samba train step cProfile, attention arms at T3 (granted, FORCE_DENY + NO_ERECOMP = the old denied path, FORCE_DENY = recompute), GEMM geometry sweep (default, KB_WIDE 32, GROUP_M 16, DB, KB_WIDE 32 x SGM 4, no small tile), mamba1/mamba2/transformer forward A/B b11745d8e vs 30497d57e vs ca692c7e2 |

## Changes

| commit | change | default? | status |
|---|---|---|---|
| ca692c7e2 | Apple matrix GEMM: double-buffered shared pages (`-D MOJOLEARN_APPLE_MMA_DB`) | opt-in arm | measuring (job 1) |
| 0d5d08027 | Apple matrix GEMM: per-leaf wide window (`-D MOJOLEARN_APPLE_MMA_KB_WIDE=32`) | opt-in arm | measuring (job 1) |
| 815db5956 | Apple attention: a process DENIED the kept exp stashes recomputes one layer's stash in the backward (estash forward into scratch + estash backward) instead of the round 3 zdot | DEFAULT ON (denied processes only; `-D MOJOLEARN_ATTN_APPLE_NO_ERECOMP` opts out) | measuring (job 1); revert if the witness moves or no gain |

Shared code note: the GEMM arms touch `gemm/checks/gemm_identical.mojo`
(every Apple IDENTICAL GEMM, classical families included) but only under
their defines; the recompute touches `fused_attention.mojo` and
`transformer_backward.mojo` (byte LM trainer only: the grant is asked by the
byte LM trainer alone).
