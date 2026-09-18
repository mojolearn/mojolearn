# LANE STATUS: lane/attention-replay-vendors

FOR A READER WITH NO CONTEXT. Task (Andrew, 2026-09-18): turn on the exact
masked-tail attention replay (`attn_masked_tail_replay_for`) and bounded
eager storage (`byte_lm_release_eager_for`) on AMD and Apple; NVIDIA already
has them (docs/lanes/LANE_STATUS_lm-attention-fallback.md). Worktree
`~/mojolearn-wt/attention-replay-vendors`, branch cut from origin/main
ede6f6243.

## State (15:50Z): BOTH FLIPS QUALIFIED AND MERGED TO MAIN

- `attn_masked_tail_replay_for` and `byte_lm_release_eager_for` answer True
  for NVIDIA, AMD and Apple (checks/kernel_matrix.mojo, docstrings carry the
  numbers). AMD: two-corpus 700-step pairs equal NVIDIA bit for bit and per
  layer, refusals 3697/1768 -> 0, late step 1.3214x / 1.2268x (geomean
  1.2733x). Apple: Metal corner fixtures and reduced HD64 witness equal
  NVIDIA; the replay row is INERT in Apple's DEFAULT training (its arm reaches
  no replay kernel), the release row is active. identity_break before/after
  on NVIDIA, AMD and CPU: nothing moved.
- Pods out: none (see "Pods" at the end).

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

## NVIDIA native gates on the flipped source: PASS (pod ur4bbe8u4pe9wr, 404 verified)

Commit a9584a477, H100. `nvidia_gates/remote/attn-replay-extra/gates_verdict.txt`:
row-off control (LEGACY_CORNER) fails "masked-tail repair disabled: INERT";
SAB_Z / SAB_DQ fail with fused=0x80000000 eager=0x00000000; preserve
corruptions fail with fused=0x00000000 eager=0x80000000; dk/dv tail sabotage
fails "no_tail accepted dk differs at 0 fused=0x00000000 eager=0x80000000".
Then clean masked-tail, tail-guard and broad fused gates PASS.

Reduced HD64 witness [1,256,128,2,2,64,256,2,256], seed 20260917, enwik8,
100 steps: default == legacy on all 100 losses and 5 witnessed steps. Every
forward/backward status RAN, zero replay sites: the replay is INERT at this
shape (no corner occurs). It is a cross-column fused-HD64 witness, not a
replay witness; the replay's own evidence is the native fixtures.

## CPU column identity_break before/after: nothing moved (pod rf1ruxuaac3q5i, 404 verified, $0.05)

`cpu_identity/remote/leg_out/`: after = branch, before = main's source rebuilt
at the same path (before_apply.log prints the two reverted rows). 14 neural
lanes x 9 fixtures: train column IDENTICAL=99, REFUSED=27 in BOTH (the three
par-byte-lm lanes need the GPU byte-LM binding; a CPU pod has none: "rebuild
bindings/build_byte_lm.sh for parallel training"); infer/model IDENTICAL=153,
batch IDENTICAL=90, rlpair IDENTICAL=72; zero DIVERGENT/MOVED. INERT by
construction: both rows answer False for COLUMN_CPU before and after.

## NVIDIA identity_break before/after: nothing moved (pod uywfr8o08f4qeh, 404 verified)

`nvidia_identity/remote/attn-replay-extra/`: 9 neural lanes x 9 fixtures,
train IDENTICAL=81, infer/model IDENTICAL=117 (N/A=45), batch IDENTICAL=72,
rlpair IDENTICAL=72; zero DIVERGENT/MOVED. The GPU bindings really were
rebuilt from main's source: `_mojolearn_byte_lm.so` 8483a59c -> ab42c4c8 and
`_mojolearn_transformer.so` ed88d286 -> 9d5b9046 (the plumbing is compiled
into both; NVIDIA's rows are unchanged). DEFECT IN THAT LEG, stated: the
before-side HOST builds refused ("output already exists") because the body
cleared only python/mojolearn/identical, so the host lanes' "before" cells ran
the after host bindings. Host lanes are CPU arithmetic whose rows answer
False either way, and the CPU pod above rebuilt them properly. Body fixed
(clears python/mojolearn/host too) before the AMD identity leg.

## APPLE native corner fixtures on Metal: PASS, every control watched FAIL

One Metal job through mac_slot.sh metal (admitted after 1085 s queue, 5.5 s
run). `apple_native/gates_verdict.txt`, default Apple build (no REPAIR define:
the matrix row itself turns replay on):
- row-off control (LEGACY_CORNER): "masked-tail repair disabled: INERT".
- zdot, chain at -0 at its visible-run end, masked tail launders to +0:
  replay disabled (SAB_Z) -> "zdot repair differs: fused=0x80000000
  eager=0x00000000"; clean -> "MATCH zdot fused=0x00000000 eager=0x00000000".
- dQ, same corner, all 64 cells: SAB_DQ -> "dq repair differs:
  fused=0x80000000 eager=0x00000000"; clean -> 64 x fused=eager=0x00000000.
- preservation (the -0 must survive a negative masked term): corrupting the
  fused output to +0 fails for zdot and dQ; clean -> fused=eager=0x80000000.
- joint dk/dv (r2): no_tail accepted with dk=-0 == eager; masked_suffix and
  next_head_prefix REFUSE (corner=True) and "masked cells change dk" (the
  eager path is what runs there); tail sabotage fails "no_tail accepted dk
  differs at 0 fused=0x00000000 eager=0x80000000".
- broad transformer_fused_check on Apple's default arm: PASS.
All 130 masked-tail MATCH lines and all tail-guard lines are byte-equal to
the NVIDIA leg's (`tools/lm_attention_cross_vendor_check.py`, its two
corruption controls fail first: apple_native/cross_vendor_nvidia_apple.log).

Apple reduced training witness: attempt 1 ran NO training (no core
`_mojolearn.so` in the worktree: "base binding has no all_finite_f32").
Attempt 2 (core built, one worker, 38 s; Metal run 21 s total, ~0.06 s/step):
shape [1,256,128,2,2,64,256,2,256], seed 20260917, enwik8, 100 steps,
witnesses at 0/25/50/75/99. apple_default, apple_bswz (NVIDIA schedule
define, reaches dq_tiled_pf/zdot_estash/dkdv_r2 on Metal) and apple_legacy
ALL EQUAL nvidia_default on every loss and every witness hash
(apple_native/reduced_vs_nvidia.log; 6 corruption controls and the blind
head_dim-8 control fail first). Step 99: loss eb0baf9c..., gradients
7eacbaf2..., parameters 5c4f10b6..., m e268e153..., v 92db0481..., flags
5b6fb58e.... Every forward/backward status RAN, zero replay sites: at this
shape no corner occurs, so the witness is INERT for the replay itself (it
covers the fused HD64 path and the flipped build); the replay's Metal
evidence is the corner fixtures above.

## AMD Pile GitHub: PASS, bits and routing equal to NVIDIA at every step and layer

Hot Aisle MI300X VM d89c0dc6 (13core), source dac7749a1, strict R2 staging
("R2 STAGED 1 key(s)"; the body's --check also passed). DELETE 204, verified
gone (GET 404, not listed) 14:52:43Z. Host verdict
`amd_pile_github_vs_nvidia.log` (22 corruption controls failed first):
- all 700 losses, six hashes at steps 0 and 699, and per-step per-layer
  forward/backward status and replay-site vectors EQUAL to NVIDIA's, for
  legacy and for repaired; legacy == repaired bits on AMD. Step 699 hashes:
  loss 8d4315c3..., gradients 1e0b647b..., parameters 749594fd..., m
  781348d9..., v 50dd0b46..., flags 360d579d....
- legacy backward RAN 6632 / CORNER 1768; repaired RAN 8400 / CORNER 0;
  replay sites dQ 1612, zdot 0 (INERT, as predicted); forward RAN 8400.
- guarded (dk/dv exact guard only): CORNER 1612 (predicted 1612); per
  layer-step, 156 legacy CORNERs vanish under the dk/dv guard alone and all
  1612 remaining are dQ replay sites; 0 unexplained.
- repaired eager_bytes 432, aexp 2415919104 every step.
- late (last 50) median: legacy 0.773939 s, repaired 0.630843 s ->
  **1.2268x**. Head (steps 1-50) 0.625 / 0.627 s: the repaired late step is
  back at the head step, i.e. the whole eager-fallback cost is removed. The
  speed prediction (repaired 0.40-0.65 s, >1.1x) passed; my legacy range
  (0.72-0.86 s) held. Smaller than NVIDIA's 1.916x because AMD's non-attention
  step (~0.62 s) is ~3x NVIDIA's, and Pile's corner count is the low one.
- device memory: rocm-smi reports 183625 MiB used device-wide in EVERY arm
  and step: the allocator reservation, not live arrays. Not a measurement of
  the storage bound; eager_bytes is.

## AMD enwik8: PASS, bits and routing equal to NVIDIA at every step and layer

Hot Aisle MI300X VM 1e44908b, source 608e2fa95 (numerical source identical
to dac7749a1; later commits are tools/docs), strict R2 staging ("R2 STAGED 1
key(s)"). Verified gone (GET 404, not listed) 15:34:52Z. Host verdict
`amd_enwik8_vs_nvidia.log` (22 corruption controls failed first):
- all 700 losses, six hashes at 0 and 699, per-step per-layer statuses and
  replay sites EQUAL to NVIDIA's for both arms; AMD legacy == repaired bits.
  Step 699: loss 1d383a22..., gradients 978787f7..., parameters 745dabc4...,
  m 91dc9edd..., v b0c5fc9b..., flags 360d579d....
- legacy backward RAN 4703 / CORNER 3697; repaired RAN 8400 / CORNER 0;
  replay sites dQ 3449, zdot 0 (INERT); forward RAN 8400. All as predicted.
- eager_bytes 17314086912 -> 432; aexp 2415919104.
- late median legacy 0.835384 s -> repaired 0.632183 s: **1.3214x**
  (head 0.625 / 0.627 s).
- release-only control (released_legacy, legacy arithmetic + bounded
  storage): every loss, hash and routing vector equals legacy
  (`amd_enwik8_release.log`, 12 controls fail first); eager_bytes 432 after
  every step, release active on 499 steps; tail 0.837907 s (storage, not speed).

**AMD two-corpus late-step geomean: 1.2733x** (enwik8 1.3214x, Pile GitHub
1.2268x). Both > 1.1x as registered.

## AMD native gates, reduced witness, identity_break before/after: PASS

DigitalOcean MI325X droplet 601730505 (gfx942, the AMD column; Hot Aisle had
no stock), source 2e33978dd, R2 staged, DELETE 204 then GET 404 at 15:41Z.
`amd_extra/remote/attn-replay-extra/`:
- gates on the SHIPPED AMD default build: row-off control "INERT", SAB_Z /
  SAB_DQ fused=0x80000000 eager=0x00000000, preserve corruptions
  fused=0x00000000 eager=0x80000000, tail sabotage fails; clean masked-tail,
  tail-guard and broad fused gates PASS. All 130 sites equal NVIDIA's
  (amd_extra_sites_vs_nvidia.log, controls fail first).
- reduced HD64 witness: amd_default and amd_legacy equal nvidia_default,
  apple_default and apple_bswz on every loss and witness hash
  (reduced_three_columns.log). Replay INERT at that shape (no corner).
- identity_break, 9 neural lanes x 9 fixtures, before (main's source rebuilt
  at the same path, host bindings cleared this time) vs after: train
  IDENTICAL=81, infer/model 117 (N/A 45), batch 72, rlpair 72; zero
  DIVERGENT/MOVED. GPU bindings really differ (byte_lm 69d1e715 -> 31b128c3,
  transformer 910efe2f -> 15d4a2cd). These tiny-shape lanes do not reach the
  HD64 fused kernels' corner, so for the replay they are INERT: they show the
  flip moved no bit of the lanes that exist, not that the replay is right.

## Post-merge AMD confirmation, registered before running (15:44Z)

Main dbc7bd4a3 merged this lane with main's independently qualified AMD GEMM
operand staging (`lib_gemm_stage_ftz_for`, 1a22490ed/112108c52), which my AMD
runs did not include. Each change was proven identical to NVIDIA on its own;
the combination was not run. One enwik8 legacy/repaired 700-step pair on the
merged source (DigitalOcean MI325X, gfx942). Prediction: every loss, the six
hashes and per-layer routing equal NVIDIA's again (3697 -> 0, dQ 3449);
the repaired late step drops below 0.632 s because the GEMM share shrinks,
and the replay speedup rises above 1.3214x. Any differing bit is a defect in
the combination and reverts the AMD row until explained.

### Result: PASS (droplet 601734705, 404 verified 16:05Z)

Source 74141e520 (contains main dbc7bd4a3), DigitalOcean MI325X, strict R2.
`amd_postmerge_enwik8_vs_nvidia.log` (22 controls fail first): all 700
losses, six hashes at 0/699 and per-layer routing equal NVIDIA's in both
arms; 3697 -> 0 refusals, dQ 3449 sites, zdot 0. Late median 0.762693 ->
0.550012 s = **1.3867x** (head 0.547 / 0.549 s). Predictions passed. This is
an MI325X figure on the merged source, so it is not directly comparable with
the MI300X 1.3214x. rocm-smi on this droplet reports real usage: device
33981 -> 18069 MiB late in the run (the MI300X VM reported a flat 183625 MiB
reservation).

## Merged

Main a6751f04c (first merge dbc7bd4a3). `tools/lm_attention_replay_vendors_before.json`
records the lane's own hunks against ede6f6243 and is valid only for the
trees the recorded legs shipped; it refuses (by design) to apply to a later tree.

## Candidates (not opened)

- Apple's default arm (`stash_tiled`) reaches no replay kernel, so the replay
  row is INERT in Apple's default training. Moving Apple to a `_pf` arm (the
  reduced witness under the NVIDIA schedule define already equals NVIDIA on
  Metal) would make it act; that is a routing change with its own
  qualification, not opened.
- AMD's zdot kernel (zdot_stash_pf) has no replay; zero zdot corners occurred
  in 8400 backward layer-steps per corpus on AMD, so nothing is lost today.
- The 700-step witnesses compare step outputs, not d_q_rope/d_k/d_v buffers
  (stages 22-24); carried over from lane/attention-corner-predicate.

## Pods (15:45Z)

- NONE. Every box this lane rented is verified gone: RunPod ur4bbe8u4pe9wr,
  uywfr8o08f4qeh, rf1ruxuaac3q5i (404); Hot Aisle d89c0dc6, 1e44908b (404,
  not listed); DigitalOcean 601730505 (404).
