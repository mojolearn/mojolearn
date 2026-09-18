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
- Pods out (14:10Z), all MINE, each self-deletes at its lease:
  RunPod ur4bbe8u4pe9wr (H100, extra body mode gates -> nvidia_gates/),
  RunPod uywfr8o08f4qeh (H100, mode identity -> nvidia_identity/),
  RunPod rf1ruxuaac3q5i (CPU pod, identity before/after -> cpu_identity/).
  NOT MINE, never touch: 8dqkcbscpgbqu2 (another session's GEMM pod).
  Hot Aisle: two legs queued/waiting for stock (amd_enwik8, amd_pile_github);
  slot 1 taken by attn-replay-enwik8 at 14:06Z, VM not yet created.
  RunPod MI300X create returned "no instances currently available" twice
  (failed_runpod_no_stock/, nothing rented). A first Hot Aisle launch was
  refused locally for a dirty tree (refused_dirty_tree/, nothing rented).
- Apple: native gate binaries built (one worker; ~/mojolearn-evidence/
  attention-replay-vendors/apple-native/bin), bswz/legacy bindings BUILT (exit 0),
  then ONE Metal job queued through mac_slot.sh metal behind the release
  runner and two apple-seam tickets (scripts in apple_scripts/).
- Local: Apple byte-LM binding with the flip + plumbing compiled, one worker,
  nice 19 (local/apple_compile_default.log, 212 s). Compile only.

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

## AMD predictions, registered before the first rental (2026-09-18 ~14:30Z)

RunPod MI300X (Hot Aisle full: both team slots held by lane gemm-single-leaf),
`tools/gemm_remote_leg.sh amd`, strict R2 staging of the one corpus key.
Bodies: `tools/lm_attention_vendor_train_{enwik8,pile_github}.sh` ->
`tools/lm_attention_vendor_train_body.sh`. enwik8 adds a release-only arm
(released_legacy), Pile GitHub a dk/dv-guard-only arm (guarded).
Host verdict: `tools/lm_attention_vendor_compare.py <amd dir> <nvidia dir>`
(its 11 corruption controls per arm fail first on the NVIDIA record itself).

The arithmetic is claimed identical across columns, so the routing is
predicted to match NVIDIA per step AND per layer, not only in total:

| | enwik8 | Pile GitHub |
|---|---|---|
| legacy backward | RAN 4703 / CORNER 3697 | RAN 6632 / CORNER 1768 |
| repaired backward | RAN 8400 / CORNER 0 | RAN 8400 / CORNER 0 |
| repaired replay sites | dQ 3449, zdot 0 | dQ 1612, zdot 0 |
| forward | RAN 8400 both arms | RAN 8400 both arms |

- All 700 losses and six hashes at steps 0/699 equal between AMD arms and
  equal to NVIDIA's; one differing bit rejects the AMD flip.
- repaired eager_bytes 432 every step, aexp 2415919104.
- Guarded (Pile): CORNER 1768 - 156 = 1612 (NVIDIA's dQ count), if the
  dk/dv guard alone removes the same 156 there as the arithmetic predicts.
- Late (last 50 steps) median: AMD legacy ~0.72-0.86 s (the gemm-class
  pure 501-700 medians), repaired 0.40-0.65 s; speedup > 1.1x on EACH
  corpus. <= 1.1x on either falsifies the speed prediction (bits still decide
  the flip; speed is reported either way).
- An AMD zdot CORNER or zdot site anywhere falsifies prediction 1 above.

## Candidates (not opened)

- Apple default arm word does not reach any replay kernel (finding 3).
