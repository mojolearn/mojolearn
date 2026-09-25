# RESUME: lane/amd-step-time (AMD optimizer step time at the GPT-3 Small shape)

Read this first after a restart. Branch `lane/amd-step-time` (from origin/main
5a3804a45), worktree
`/Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/agent-a66ed12de5c578ebc`.
Never merge or push main from this lane. Never touch the live T3 run
(`~/mojolearn-wt/gpt3-tooling`, `~/mojolearn-evidence/gpt3-run/t3/`, R2
`runs/t3/2026-09-22/`, its MI325X droplet and its RunPod H100).

## PASS 2 (branch lane/amd-step-time-2, from origin/main 34c86fc66 = 0.8.18)

Worktree `/Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/agent-aff3e16a28427aacd`.
Never merge or push main. Never touch T3 (running A/4 on the only
DigitalOcean MI325X from 02:07 UTC Sep 25, about 30 h), never rent on
DigitalOcean or NVIDIA. Hot Aisle only for GPU minutes (balance $27.71 at the
start, $5 floor); compile checks and gfx942 asm on a RunPod CPU pod
(`tools/amd_step_time2_cpu_check.sh` through `tools/runpod_cpu_leg.sh
--cmd-file`, about $0.02 each), never on the Mac.

### 2026-09-25 09:45 UTC: LEG 1 RAN. Admission PROVEN; attention trials NOT

Branch `lane/amd-step-time-2-proof` (from lane/amd-step-time-2 79ff2d56f).
Hot Aisle 1x MI300X, 08:54:34 to 09:37:42 UTC (VM verified gone, 404),
$2.14 (balance $27.71 -> $25.57 after the last billing tick). Evidence
`legs/2026-09-25_085416-hotaisle-mi300x-pass2-leg1/`, write-up and table in
README.md, "Pass 2 leg 1".

PROVEN on the device (admit, the branch default):
- GEMM A/B: valu, noadmit and admit hash files byte-identical, 64 lines over
  all six kinds (ordinary, tiny, mixed, skew, border, sparse), every rehash
  equal.
- Lean B4 witnesses (loss, gradients, parameters, m, v, flags at 3 steps)
  equal across admit, noadmit, dq, dqkv, attn3 and equal to leg 7's.
- Replays PASS against the H100 chain: 101 abc8b816b5c3fb15, 102
  a9421f91b947f82c, 103 fcdb48b8ab51f2ef, 1999 dcb05e4e668a81e1, 2000
  0e39ed2bfe9bcbae.
- gemm device (8 gates), backward (10 gates) and workspace (4608 cells)
  checks green; 201 GEMM lanes: 181 VERIFIED, 0 DIVERGENT, 20 REFUSED (the
  same 20 as leg 7).
- Step: 29.7 s a step steady (leg 7's 0.8.18 build: 32.3 s); lean shard
  0.503 -> 0.459 s on the same VM; kernel time 496.0 -> 453.3 ms a shard.

NOT PROVEN (the attention trial kernels, off by default):
- Their witnesses equal admit's and dqkv (101 to 103) and attn3 (101 to 103,
  1999 to 2000) replays PASS with the same digests; attn3 is 26.4 s a step,
  kernel time 403.9 ms a shard. BUT `amd_mfma_probe3` misses its own pass
  line: under MODE 2 the raw 16x16x1 MFMA differs from host fma on 14,080 of
  4,194,304 words (0 under MODE 3; the GEMM's 32x32x1 is 0 under MODE 2).
  The product by one still equals ftz(fma) on every word (0 mismatches), and
  that is the value the kernels carry, but the difference is unexplained, so
  the argument is open. The dq-only build was not replayed, and no lane or
  gemm check covers the attention trial kernels.

WHAT A RELEASE MAY TAKE FROM THIS BRANCH: the GEMM exact admission (default
on, AMD only), and nothing else from pass 2. The attention defines stay off.
The NVIDIA re-proof owed from pass 1 (`GEMM_LAUNCH_BOUND`, shared source)
still stands. Admission's code is AMD-only.

Fixed: the leg 1 body now installs `libdw1` (rocprofv3 needs `libdw.so.1`;
the scripted trace failed and was rerun by hand in the container).

NEXT: explain the probe3 MODE 2 difference (dump the differing words: is it
the subnormal result rounding, or the sign of a flushed zero?) before any
attention trial counts; then lever 2 (packed flush on steps that are not
admitted: skew, mixed and sparse operands ran 4 to 8 % slower with admission
than without).

### 2026-09-25 02:30 UTC: the ranking at 32.3 s (before renting)

Source: the first pass's leg 8 (2026-09-24 20:49, Hot Aisle MI300X, branch
head = 0.8.18's AMD code), rocprofv3 kernel trace of two lean B4 steps
(`legs/2026-09-24_204925-hotaisle-mi300x-leg8/remote/amd-step-time/prof/lean_kernel_stats.csv`,
copied into this branch; the leg itself timed out after the trace, $2.69).
Lean step 0.503 s a shard; kernel time 0.493 s a shard. Per shard:

| rank | what | ms a shard | share | note |
|---|---|---|---|---|
| 1 | matrix-core GEMM, all calls but the two whole-leaf head calls | 238.8 | 48 % | `identical_gemm_mfma_kernel`, 253 launches |
| 2 | attention (4 kernels x 12 layers) | 124.0 | 25 % | forward `fwd_r2` 37.0, dq 32.7, dk/dv 32.2, zdot 22.1; VALU chains, `_step_preflushed` |
| 3 | head GEMM forward and dB (whole-leaf MFMA launches) | 59.2 | 12 % | head dA is inside rank 1 (group launch) |
| 4 | GEMM group folds + the one VALU GEMM call | 19.4 | 4 % | `_ksplit_fold`, 8.0 ms VALU call |
| 5 | everything else | ~51 | 10 % | norms 7.1, embedding backward 7.9, nonfinite scans 6.7, buffer fills 4.2, cross entropy 5.2, copies, AdamW 1.5 (once a step in T3) |
| - | host (hashing outside the step; AdamW + validation once a step) | ~0.1 s a step | | the step is 64 x shard |

Counters on the matrix-core GEMM (leg 8 pmc): 33 VALU instructions per MFMA
(head forward: 10.2e9 VALU, 309e6 MFMA). Each `v_mfma_f32_32x32x1f32`
(64 cycles on the matrix pipe) is followed by 32 products by one on the VALU
(the flush, 128 cycles unpacked). So the kernel is VALU bound at about twice
the matrix-core time.

### 2026-09-25 03:00 UTC: built, compile-checked, waiting for Hot Aisle stock

- Exact admission in `identical_gemm_mfma_kernel` (default ON, AMD only,
  `-D MOJOLEARN_GEMM_MFMA_NO_ADMIT=1` reverts). CPU-box asm
  (`cpu/admit1/`): the admitted loop is back-to-back
  `v_mfma_f32_32x32x1_2b_f32` with no VALU between steps; the shipped path
  issues the pair, `s_nop 15`, then 14 `v_mul_f32` + 25 `v_pk_mul_f32` per
  pair, so the flush already packs but is serialized behind the MFMA pair.
  No spills (290 VGPRs).
- Attention on the matrix cores, three TRIAL kernels (AMD only, opt-in
  defines, off by default until proven): dq (`MOJOLEARN_ATTN_DQ_MFMA`),
  dk/dv (`MOJOLEARN_ATTN_DKDV_MFMA`), forward context (`MOJOLEARN_ATTN_FWD_MFMA`),
  each `v_mfma_f32_16x16x1f32` + product by one under MODE 2 set only
  around the chain, masked keys on the VALU for exactly the visible cells.
  Compile and asm on the CPU box (`cpu/dq1/`, `cpu/dkdv1/`, `cpu/fwd1/`):
  dq 16 MFMA per 16-key tile, no spills (the shipped VALU dq kernel
  spills 39 VGPRs); `gemm/checks/amd_mfma_probe3.mojo` measures the 16x16x1
  step against host fma and the flush (not yet run).
- Leg 1 body `tools/amd_step_time2_leg1.sh` (GEMM A/B over 6 operand kinds
  for valu/noadmit/admit, lean witnesses for admit/noadmit/dq/dqkv/attn3,
  replays, kernel trace, all bindings + 201 lanes). Launcher waits for
  13core stock (0 since 02:10 UTC; only the 2x MI300X offering at $5.98/h).

### 2026-09-25 07:20 UTC: NO GPU LEG RAN. Hot Aisle had no stock for 5 hours

Hot Aisle listed no 1x MI300X (13core) from 02:10 to 07:17 UTC; from about
03:20 it listed no offering at all (the 2x MI300X also went). The launcher
was stopped at 07:18 UTC so nothing rents unattended. Balance unchanged,
$27.71. RunPod CPU pods for compile checks: 6 created, 2 of them never
reached ssh (refused by the runner, deleted, verified gone), about $0.10 total.

TO RUN LEG 1 (nothing else is needed; about 55 minutes, about $2.99):
1. `python3 tools/amd_step_time_urls.py <dir>/urls runs/t3/2026-09-22/A/1/ckpt_00000100.blm runs/t3/2026-09-22/A/2/ckpt_00001998.blm`
   (the presigned URLs last 12 h), and put `A-1.chain.partial.jsonl`
   (= `~/mojolearn-evidence/gpt3-run/witness/A-1.chain.partial.jsonl`),
   `A-2.chain.jsonl` (= the H100 A/2 leg's `segment/chain.jsonl`) and
   `lanes_gemm_nonpar.txt` in `<dir>/amd_in`.
2. `MOJOLEARN_GEMM_LEG_EXTRA=tools/amd_step_time2_leg1.sh MOJOLEARN_GEMM_LEG_OUT=bench/results/amd_step_time_2026-09-24/legs/<stamp>-hotaisle-mi300x-pass2-leg1 MOJOLEARN_STAGE_KEYS="" MOJOLEARN_HOTAISLE_LANE=amd-step-time-2 MOJOLEARN_GPU_ARCHS=gfx942 bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates`
3. Once `vm_details.json` shows the address: `tar czf - -C <dir> urls amd_in | ssh hotaisle@<ip> 'sudo tar xzf - -C /root'`
   (the body waits for these files before the step half).
4. Read `remote/amd-step-time/session.txt`: `ab <tag> ... hashes_vs_valu=`
   must read IDENTICAL for noadmit and admit on every line (six kinds);
   lean witnesses of admit, noadmit, dq, dqkv, attn3 must be equal; every
   replay PASS; `gemm_*_check` green; verify chunks 0 DIVERGENT. Any
   difference: revert that lever (admission: its define; the attention
   kernels are already off by default).

STATUS OF THE BRANCH DEFAULTS (superseded 09:45 UTC: leg 1 proved admission): exact admission is ON by default in the AMD
GEMM on this branch and was NOT yet proven on a device (the proof is the
argument in the source plus the leg 1 checks above). Do not merge before
leg 1 reads IDENTICAL. The three attention kernels are OFF by default.

### The plan (levers in order, each one leg, bits proven after each)

1. GEMM EXACT ADMISSION (AMD only, `identical_gemm_mfma_kernel`, trial
   define `-D MOJOLEARN_GEMM_MFMA_NO_ADMIT=1` reverts). When every product of
   a leaf is a multiple of 2^-126 (smallest nonzero exponent fields of the
   block's staged A and B words sum to at least 174, tested per window),
   the flush after each step is provably the identity on every word (the
   argument is in the source comment), so the MFMA step alone is the
   contract step and the 32 VALU products are not issued. A window that
   fails switches the rest of its leaf to the shipped step. Expected: up to
   2x on rank 1 and 3 (about 150 ms a shard, 9 s a step) on data that admits.
   Proof: the 28 shipped A/B hashes plus three new kinds (skew: subnormal
   products that a one-sided test would wrongly admit; border: every window
   admitted at the bound; sparse: leaves switching) equal to the VALU build
   and to the MFMA build without admission; replays; lanes.
2. GEMM: the packed flush (`v_pk_mul_f32`, 16 instead of 32 VALU per MFMA)
   on the steps that are not admitted, and four independent accumulators
   per wave if the asm shows the MFMA dependency is then the limit.
3. ATTENTION on the matrix cores: the score chains (q.k over 64 head
   dimensions, from +0.0, `_step_preflushed`) and the context / gradient
   chains are the same `ftz(fma_rn(a, b, acc))` ascending chain, so
   `v_mfma_f32_32x32x1f32` plus the product by one (and exact admission)
   apply. Masked cells are skipped by the chains, never stepped with a zero
   (a zero step can turn -0.0 into +0.0), so only full key blocks go to the
   matrix cores; the diagonal block keeps the VALU step on the same
   accumulator registers. Largest single piece of work; after 1 and 2.
4. The rest (norms, embedding backward, scans, fills): about 50 ms a shard,
   not attempted unless 1 to 3 finish.

Reach: all of the above is AMD-only code (the MFMA kernel is only
instantiated under `lib_gemm_mfma_for` = AMD; attention MFMA would be behind
an AMD kernel-matrix row). NVIDIA and Apple compile the same files; the CPU
check builds the A/B harness for sm_90a to show they still compile.

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

MI300X: 141.0 s (main) -> 32.3 s a step (branch head, leg 7), ahead of one
H100 (39.9 s). Every replay PASS against the H100 chain; 181/201 GEMM lanes
IDENTICAL, 0 divergent, 20 refused for unbuilt bindings; GEMM device,
backward, workspace checks green. Changes (all AMD-only rows except the GEMM
launch bound): launch bound (spills), leaf split, class ftz, attention
`_bswz`, matrix-core GEMM (`lib_gemm_mfma_for`) with its group launch
(`GEMM_MFMA_MIN_GROUP_LEAVES` = 4). README.md is the write-up.

MERGED: origin/main fast-forwarded to the branch at b573f2b40 (2026-09-24
~20:10 UTC, at Andrew's request), after the cross-target compile check
(`xtarget/`: NVIDIA sm_90a builds with `.maxntid 256`, Apple M4 compiles and
carries no bound). The worktree at ~/mojolearn-wt/release-0814 has `main`
checked out at the old tip; it was not touched.

NEXT: MI325X confirmation on DigitalOcean after ~22:00 UTC
(`tools/amd_step_time_leg_mi325x.sh` with `tools/do_extra_leg.sh amd`; push
/root/urls and /root/amd_in after the droplet is up; the scratchpad helper
hx.sh speaks `docker exec` for Hot Aisle, a DO droplet is plain root ssh).
Then (optional) attention on the matrix cores (its dot chains take the same
MFMA + product-by-one spelling).

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
