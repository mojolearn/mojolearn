# RESUME: lane/amd-step-time (AMD optimizer step time at the GPT-3 Small shape)

Read this first after a restart. Branch `lane/amd-step-time` (from origin/main
5a3804a45), worktree
`/Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/agent-a66ed12de5c578ebc`.
Never merge or push main from this lane. Never touch the live T3 run
(`~/mojolearn-wt/gpt3-tooling`, `~/mojolearn-evidence/gpt3-run/t3/`, R2
`runs/t3/2026-09-22/`, its MI325X droplet and its RunPod H100).

## Goal

One optimizer step at the T3 shape (162,147,840 parameters, batch 4, length
2048, vocabulary 50,257, K=64 shards) takes 139 s on one MI325X against 39.9 s
on one H100. Target under 60 s with identical bits (per shard 2.17 s to under
0.94 s).

## State (update every session)

- 2026-09-24 13:49 UTC: branch created. No AMD box: Hot Aisle stock 0 (balance
  $44.65), RunPod MI300X stock None, DigitalOcean GPU lock held by the run
  (`extra:A-3-amd`) until about 22:00 UTC.
- Kernel map: `kernel_map.md` in this directory (written before any rental).
- 14:26 UTC: Hot Aisle 1x MI300X leg 1 (60 min cap, $2.99/h), body
  `tools/amd_step_time_leg1.sh`, worked interactively through
  `docker exec mojolearn-leg` (scratch helper hx.sh). Evidence lands in
  `legs/<stamp>-hotaisle-mi300x-leg1/remote/amd-step-time/`.

## HEADLINE (update me)

MI300X, same host type: 141.0 s (main) -> 49.1 s a step (branch head), every
replay PASS against the H100 chain, 181/201 GEMM lanes IDENTICAL (0
divergent). Changes: GEMM launch bound (spills), AMD leaf split, class ftz in
AMD device code, AMD attention `_bswz`. README.md is the write-up.

IN PROGRESS (leg 6, 2026-09-24 ~18:30 UTC): a matrix-core IDENTICAL GEMM.
Facts so far: `v_mfma_f32_32x32x1f32` == VALU fma exactly (8.65M elements),
ignores MODE flush. Plan: MFMA for the FMA, the flush as `acc * one` (one =
1.0 from a kernel argument, unfoldable) with the wave's MODE set to flush f32
outputs (a product by one is exact, so flush-before-round cannot bite); probe 2
checks layout, modes 0..3, the multiply flush and class under each mode.
If it holds: new kernel behind an AMD row, A/B hashes, device check,
replay. MI325X confirmation on DigitalOcean after ~22:00 UTC
(`tools/amd_step_time_leg_mi325x.sh`).

## RESULTS SO FAR (MI300X, leg 1)

1. ROOT CAUSE: VGPR SPILLS. Without a launch bound the gfx942 backend budgets
   for 1,024-thread blocks (128 VGPRs) and SPILLS the 128x128 register tile:
   `.vgpr_spill_count` tuned128 630, kpack 396/390. Fix: declare
   `MAX_THREADS_PER_BLOCK_METADATA` = 256 (the real launch size) on the two
   GEMM kernels (`GEMM_LAUNCH_BOUND`, revert `-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND`)
   -> spills 0. T3-shape GEMM A/B (`bench/gemm_excp_ab_main.mojo`): 28 of 28
   output hashes IDENTICAL, every call about 3x faster (proj_fwd 3.71 -> 1.28 ms,
   head_fwd 170 -> 56.6, head_dA 249 -> 80.5, head_dB 170 -> 55.5).
   Timed B4 shard (random init): 2205 ms -> 988 ms; GEMM 1892 -> 678 ms.
2. EXCP seam DEAD: TRAPSTS reads 0x80000000 always; no sticky EXCP bits on
   this device (probe `gemm/checks/amd_excp_probe.mojo`).
3. DETECT seam (software subnormal witness): bit-identical but SLOWER than the
   plain seam once spills are gone; row off by default (trial define).
4. Same launch bound on the 7 shipped attention kernels: SLOWER (fwd r2 73.4 ->
   77.4, dq 66.2 -> 74.1 ms); removed (attention file equals main).
5. DETECT v2 (bitwise OR; hot loop VALU 3.6 -> 1.5 slots per product step):
   24/24 hashes identical but still slower than the plain seam (proj_fwd 1.28 ->
   1.64 ms): the kernel is not VALU-issue bound at 1 wave per SIMD. Off.
6. CHAIN PROOF: the branch binding (GEMM launch bound only) replayed steps
   101..103 from ckpt 100 (A-1 chain) and 1999..2000 from ckpt 1998 (A-2
   chain) on the MI300X: PASS, every state/gradient hash equal to the H100
   chain; 62.7 s a step (+7.1 s host hashing) against 139 s before (MI325X,
   T1). Same-box baseline owed (leg 2).
7. VGPRs now 260-290 per lane (accum offset 256): still 1 wave per SIMD.

Leg 1 cost $2.24 (balance 44.65 -> 42.41). VM verified gone 15:14:41Z.

LEG 2 (15:19-16:05 UTC, $2.19): same-box baseline 141.0 s; launch bound
62.7 s; + AMD leaf split (`lib_gemm_leaf_split_for`, the branch HEAD) 57.5 s a
step; all replays PASS; 19/20 identity lanes VERIFIED (0 divergent). Full
write-up: README.md in this directory.

NEXT: leg 3 (`tools/amd_step_time_leg3.sh`, gates ON): all device bindings,
gemm backward/workspace checks, verify over the 201 GEMM-reaching non-par
lanes. Then the MI325X confirmation on DigitalOcean after ~22:00 UTC
(leg 2 body works there too: `tools/do_extra_leg.sh amd --segment-lease N
--dollar-cap USD`, push /root/urls + /root/amd_in after the droplet is up).

(old) NEXT (leg 2, `tools/amd_step_time_leg2.sh`; push /root/urls with
`tools/amd_step_time_urls.py` and /root/amd_in/{A-1.chain.partial.jsonl,
A-2.chain.jsonl} after the VM is up): GEMM arm prices (trial arms, k768
dispatch arms), same-box baseline replay, identity lanes, then GEMM occupancy.

## Findings so far (from the repository and its history, no new measurement)

- Per shard the step is one `byte_gradient_device` (forward, head, cross
  entropy, backward) plus a device fold add; 64 of them plus one AdamW per
  optimizer step. The MI325X shard is 2.17 s, the H100 shard about 0.62 s.
- The last AMD itemization (MI300X, B1/L2048, 2026-09-18, recovered from git
  history `2e7d346a5^:bench/results/e1g/2026-09-18_013257-amd-mi300x-hotaisle-step-breakdown`)
  put GEMM at 81 percent of the AMD step at 2.72 TFLOP/s, FLAT across shapes
  (1.13x spread), against 12.84 TFLOP/s and a 1.61x spread on the H100.
- Every GEMM of the step runs one of two kernels: `identical_gemm_tuned_kernel`
  (AMD sends k=768 calls with m>=4096 there) and `identical_gemm_kpack_kernel`
  (everything else, group mode for long k). Both do `_tuned_step` per product:
  on AMD `v_fmac` plus the post-round class flush (5 issue slots, 2026-09-18)
  where NVIDIA issues 2.
- The hardware MODE flush is a DEFECT (flush before round), closed on
  2026-09-17. Not used here.

## Plan

1. Profile on the box: B4 itemization (timers build, component timing), GEMM
   price per call kind with the DIAG=1 floor (bare FMA), resources (VGPR,
   LDS, scratch, occupancy) of the two GEMM kernels.
2. Candidate (GEMM, AMD only, bit-exact by argument): EXACT ADMISSION. If the
   smallest nonzero exponent fields over a thread's A lines and B lines satisfy
   Ea + Eb >= 174, every product and every accumulator of those cells is a
   multiple of 2^-126 or zero, so no FMA result can be subnormal and
   `ftz(fma(a,b,acc)) == fma(a,b,acc)` exactly; the class flush can be skipped
   and the FMA packed (`v_pk_fma_f32`). A thread that fails admission keeps the
   shipped seam. Same operands, same order, same rounding.
3. Then attention, then the rest, by measured seconds.

## Owed / open

- Chain replay (steps 101..103 from ckpt 100, and 1999..2000 from ckpt 1998
  against A/2's chain, local copy of the chain in the scratchpad) on the
  branch binding; the identity lanes on AMD; the MI325X before/after step.
- NVIDIA re-proof: the launch-bound metadata is in shared source (it emits
  `.maxntid 256` on NVIDIA); NVIDIA bits and speed must be re-proven before a
  release (not rented in this lane).

## Costs

- Leg 1 Hot Aisle MI300X 13core: $2.24.
- Leg 2 Hot Aisle MI300X 13core: $2.19 (balance $40.07).
