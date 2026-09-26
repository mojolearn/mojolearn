# Apple IDENTICAL neural lane: the matrix-path GEMM and three-vendor agreement (2026-09-26)

Branch `lane/apple-identical-neural`. Evidence only; nothing here is run by `mojolearn verify`.

## The claim and how it was checked

1. **Apple's fp32 simdgroup multiply-accumulate is the contract's step chain.**
   `gemm/checks/apple_simdgroup_probe.mojo` on the M4: 0 mismatches against the ascending
   `identical_mul_add` chain over 7 operand kinds x 4,194,304 cells (`apple-m4/sg_probe_m4.txt`).
2. **`PLAN_APPLE_MMA` gives the shipped bits.** The T3 shard GEMM harness
   (`bench/gemm_excp_ab_main.mojo`, 12 calls x 6 operand kinds, 64 cases) prints the same 64 hashes
   on the M4 before (`apple-m4/gemm_ab_base.*`) and after (`apple-m4/gemm_ab_vec.*`) the change.
3. **The three vendors agree.** The same 64 hashes on an NVIDIA H100 80GB HBM3 (RunPod, this branch at
   2f6141ac4, `nvidia-h100/ab.*.log`) and an AMD MI300X (Hot Aisle, `amd-mi300x-hotaisle/remote/ab.*.log`):
   64 of 64 equal on each. The device card (60 stages of `gemm_device_check`) is IDENTICAL Apple vs
   NVIDIA (`nvidia-h100/diff_apple_vs_nvidia.txt`); gemm_device/backward/workspace checks green on AMD,
   backward/workspace green on NVIDIA.
4. **The whole training step agrees.** Byte LM, 1 x 2048, d768, 12 heads, FF 2048, 2 layers,
   V 50,257, four resident lean steps witnessed every step (sha256 of loss, gradients, parameters,
   m, v): the H100 on this branch (`nvidia-h100/step/result.json`, f8f5c5f7b) equals the M4 on this
   branch (`apple-m4/step_new`) equals the M4 on origin/main 06b4ec239 (`apple-m4/step_main`), all
   four steps.

## Speed (Apple M4, 10-core GPU, fanless; alternating builds)

- T3 shard GEMMs, ordinary operands: head_fwd 3724 -> 514 ms (1230 GF/s), head_dA 3494 -> 675,
  head_dB 3735 -> 590, gateup_fwd 102 -> 21.5, down_fwd 107 -> 21.5, proj 41-55 -> 10-12 ms.
- Byte LM step at the shape above, consecutive steps: 3.09 s -> about 1.7 s with the first matrix
  kernel (before the staging rewrite). The emb/head views save 309 MB and no measurable time.
