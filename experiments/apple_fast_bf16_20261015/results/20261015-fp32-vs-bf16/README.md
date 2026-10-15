# FP32 vs BF16 public matmul, fast build, Apple M4 (2026-10-15)

Binary: fast linalg binding with BF16 on (sha256 1f9e5e1eb119...), the build retained in
apple-fast-bf16-default. One warmup and one scored call per shape, one process per arm. Inputs are float32
standard normal (seed 7); the BF16 arms carry their rounding error against the FP64 oracle.
`linalg.matmul` (FP32) runs the identical GEMM profile even in the fast build.

| M, N, K | FP32 ms | BF16 (bits ready) ms | BF16 (cast in clock) ms |
| --- | ---: | ---: | ---: |
| 2048, 2048, 2048 | 39.1 | 27.8 | 46.3 |
| 4096, 4096, 4096 | 119.4 | 152.5 | 283.6 |
| 3000, 5000, 2500 | 1094.5 | 152.1 | 286.6 |
| 3072, 5120, 2560 | 66.8 | 95.6 | - |
| 4096, 5000, 4096 | 239.0 | 234.6 | - |
| 4096, 4096, 2500 | 1210.1 | 112.0 | - |

Relative RMS error: FP32 about 1e-6, BF16 2.35e-3.

Verdict: do not switch FP32 models to BF16. BF16 is not faster at large aligned shapes and its error is about
2000 times larger. The ragged-K FP32 rows are a cliff in the identical FP32 GEMM, fixed in lane/identical-gemm-ktail.
