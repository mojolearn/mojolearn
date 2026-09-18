# LANE STATUS: lane/attention-replay-vendors

FOR A READER WITH NO CONTEXT. Task (Andrew, 2026-09-18): turn on the exact
masked-tail attention replay (`attn_masked_tail_replay_for`) and bounded
eager storage (`byte_lm_release_eager_for`) on AMD and Apple; NVIDIA already
has them (docs/lanes/LANE_STATUS_lm-attention-fallback.md). Worktree
`~/mojolearn-wt/attention-replay-vendors`, branch cut from origin/main
ede6f6243.

## State

- WIP: both rows in `checks/kernel_matrix.mojo` now return True for NVIDIA,
  AMD and Apple. NOT QUALIFIED YET; do not merge until the sections below
  say PASS.
- Pods out: none.

## Findings before any run (from source, not prose)

The replay lives in exactly two kernels: `fused_bwd_dq_tiled_pf_kernel`
(dQ) and `fused_bwd_zdot_estash_kernel` (zdot). The exact dk/dv tail guard
lives in `fused_bwd_dkdv_r2_kernel`. Which of those a column's TRAINING
reaches is decided by its default arm word (`attn_default_arm_for`):

| column | default arm | dQ kernel | zdot kernel | dk/dv kernel |
|---|---|---|---|---|
| NVIDIA | ..._estash_dres_kvgrid_r32_bswz | dq_tiled_pf (REPLAY) | zdot_estash (REPLAY) | dkdv_r2 (exact guard) |
| AMD | stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 | dq_tiled_pf (REPLAY) | zdot_stash_pf (refuses, no replay) | dkdv_r2 (exact guard) |
| Apple | stash_tiled | dq_tiled (refuses, no replay) | zdot_stash (refuses) | dkdv_tiled (old guard) |

Consequences, registered before running anything:

1. AMD gets the dQ replay (93.3% of the NVIDIA enwik8 refusals, 3449/3697)
   and the dk/dv exact guard (the other 248). Its zdot kernel has no replay;
   NVIDIA training saw ZERO zdot sites, and the chain values are the same
   arithmetic, so AMD is predicted to see zero zdot refusals as well. Any
   AMD zdot CORNER falsifies that and is reported, not hidden.
2. AMD's non-estash backward goes through `fused_backward_launch_ran`, which
   read only the corner flag and discarded the replay bit, so
   `backward_repair_sites` would read 0 on AMD even while dQ replay ran:
   an instrumentation blind spot. Plumbed through (metadata only).
3. APPLE: the replay row is INERT on Apple's shipped training path,
   because Apple's default arm (`stash_tiled`) reaches none of the three
   kernels that carry it. Only the release-eager row acts on Apple's default.
   The Metal evidence therefore: native gates calling the replay kernels
   directly, plus a training witness under the NVIDIA schedule define
   (`MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN=1`) that reaches the replay.
   Making replay act in Apple's default training would need Apple's arm
   word changed: CANDIDATE, not opened.

## Plan

- AMD: Hot Aisle MI300X, one VM per corpus, legacy
  (`-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1`)
  vs default, 700 steps, `tools/lm_ce_alias_probe.py`, seed 20260917,
  target shape; compare to NVIDIA witnesses in
  bench/results/lm_attention_fallback_2026-09-18/default/.
- Apple: native gates + sabotage through `mac_slot.sh metal`; reduced
  HD64 training witness equal to NVIDIA/AMD at the same shape.
- identity_break byte-LM/transformer lanes before/after on NVIDIA, AMD, CPU.

## Candidates (not opened)

- Apple default arm word does not reach any replay kernel (finding 3).
