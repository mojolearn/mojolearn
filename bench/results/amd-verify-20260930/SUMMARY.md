# AMD check of everything merged 2026-09-30 (DigitalOcean MI325X, gfx942), part 1

Job `tools/amd_verify_job.sh` on lane/amd-verify-20260930 (main through PR #7), started 22:45 UTC after the
board's AMD part finished. Parts 2 and 3 (classical pass, priority pass) still running when this was saved.

| check | AMD result | equals NVIDIA? |
|---|---|---|
| Kernel PCA output sha256, 256 / 1000 / 10000 rows | b92023de7a6e / 45d3b6071431 / 1b442b0e601b | yes (= L40S, L4, Apple M4, CPU) |
| Mamba-3 y + 10 gradients, S16 arms regs2, regs, shared, smem48, naive and the angle-naive build, board shape | all one digest | yes (= L40S and L4) |
| same, default shape (B8 L512 d768) | all one digest | yes |
| gemm-int8 board cell digest | 9b16c7064e10cecd | yes |
| matmul, 21 shapes and transposes | all equal | yes (= L4) |
| fixed15 price harness, 422 DIGEST lines, transposed tile on and off | 0 differ | yes (= L40S) |
| fused small-MLP step vs per-operation, 64 steps, 256 and 32 rows | PASS, every byte equal (1.41x / 1.43x) | — |
| small-MLP GPU gate, fused and per-operation | 26 / 26 each | — |

## Part 2: classical pass on AMD (tools/classical_pass_run.py amd)

Old path vs new path on AMD: same digest for all five (LU 23.4x, SGD-reg 10.1x, SGD-clf 10.4x, LARS 1.28x, IVF 2.71x
at the digest shapes). Neural toggle sweep on AMD: every experiment's digests equal to baseline.

**Cross-vendor: THREE CASES DIFFER FROM NVIDIA, old and new paths alike (so not caused by the fixes).**

| case | NVIDIA L40S (= L4 0.8.31 wheel GPU = CPU) | AMD (this job, cold gfx942 build) |
|---|---|---|
| LU 1024 / 8192 | b53aeffc5bf56d98 / 2a296d1d25da0a2c | same / same |
| SGD-clf 20000 / 1000000 | a6cb75bf0e2f93da / bcac4b45fd82f48b | same / same |
| **SGD-reg 20000 / 1000000** | 431ad9f1214c29a2 / f07701df3903e412 | **f2571b6b6c94680d / 308d00bac4bac078** |
| **LARS 200000** / 1000000 | 356afc301a0d55c1 / 88fba1aba928096e | **97a2c17f10a7f5ee** / same |
| **IVF 40000** / 400000 | 5bf7822421de6cd5 / 26a769d35c913838 | **deec079c73e23671** / same |

The released 0.8.31 wheel on the L4 gives the NVIDIA digests on both its GPU and CPU columns, so the CPU
reference agrees with NVIDIA. Next: the released 0.8.31 wheel (cached gfx942 binaries) on AMD, GPU and CPU
columns, to tell a cold-build artifact from a shipped divergence.

## The divergence, isolated with the RELEASED 0.8.31 wheel (no overlay)

| case | L4 box GPU | L4 box CPU (Zen 2 EPYC 7542) | MI325X GPU | MI325X box CPU (Zen 5 EPYC 9575F, `MOJOLEARN_VENDOR=cpu`) |
|---|---|---|---|---|
| SGD-reg 20000 | 431ad9f1214c29a2 | 431ad9f1214c29a2 | f2571b6b6c94680d | **f2571b6b6c94680d** |
| LARS 200000 | 356afc301a0d55c1 | 356afc301a0d55c1 | 97a2c17f10a7f5ee | **97a2c17f10a7f5ee** |
| IVF 40000 | 5bf7822421de6cd5 | 5bf7822421de6cd5 | deec079c73e23671 | 5bf7822421de6cd5 |

The harness inputs (X, y) are byte-identical on both hosts. So:
- **SGD-reg and LARS follow the HOST CPU, not the GPU vendor**: the identity path depends on something
  computed on the host whose result varies between x86 CPUs (suspect: a numpy/BLAS reduction in the Python
  wrapper, whose kernel and summation order OpenBLAS picks per CPU, AVX2 vs AVX-512).
- **IVF 40000 is a true AMD GPU divergence** (AMD CPU = NVIDIA = reference); the 400000 shape agrees.
