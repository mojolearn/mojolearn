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
