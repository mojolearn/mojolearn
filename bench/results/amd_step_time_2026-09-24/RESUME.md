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

- (nothing yet)

## Costs

- (nothing rented yet)
