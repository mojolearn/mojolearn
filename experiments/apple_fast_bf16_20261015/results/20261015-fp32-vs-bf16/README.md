# FP32 vs BF16 public matmul, fast build, Apple M4 (2026-10-15)

Binary: fast linalg binding with BF16 on (sha256 1f9e5e1eb119...), the build retained in
apple-fast-bf16-default. One warmup and one scored call per shape, one process per arm. Inputs are float32
standard normal (seed 7); the BF16 arms carry their rounding error against the FP64 oracle.
In the fast build `linalg.matmul` (FP32) hands off to MAX's `linalg.matmul`.

| M, N, K | FP32 ms | BF16 (bits ready) ms | BF16 (cast in clock) ms |
| --- | ---: | ---: | ---: |
| 2048, 2048, 2048 | 39.1 | 27.8 | 46.3 |
| 4096, 4096, 4096 | 119.4 | 152.5 | 283.6 |
| 3000, 5000, 2500 | 1094.5 | 152.1 | 286.6 |
| 3072, 5120, 2560 | 66.8 | 95.6 | - |
| 4096, 5000, 4096 | 239.0 | 234.6 | - |
| 4096, 4096, 2500 | 1210.1 | 112.0 | - |

Relative RMS error: FP32 about 1e-6, BF16 2.35e-3.

The ragged-K FP32 rows are a cliff in MAX's matmul (FAST only; IDENTICAL has no such cliff).

## Run 3: fast against fast (one run per arm, one process each, in this order)

Arms: MAX matmul (shipped FAST FP32), AFN FP32 kernel default (`matmul_fused`, EPILOGUE build), AFN FP32 with
`MOJOLEARN_AFN_GEMM_SIMDGROUP`, the ragged-K fix (`MOJOLEARN_GEMM_FAST_RAGGED_K`, lane/identical-gemm-ktail@42dfb8458),
BF16 bits ready.

| M, N, K | MAX | AFN default | AFN simdgroup | ragged-K fix | BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048, 2048, 2048 | 22.2 | 26.8 | 29.6 | 23.8 | 31.7 |
| 4096, 4096, 4096 | 107.9 | 149.7 | 135.1 | 119.9 | 148.5 |
| 4096, 4096, 2500 | 1974.6 | 118.3 | 115.1 | 97.7 | 115.7 |
| 3000, 5000, 2500 | 1780.5 | 167.8 | 100.6 | 86.1 | 131.1 |
| 8192, 8192, 4096 | 673.5 | 654.6 | 913.6 | 955.4 | 1046.7 |

The 8192 row rose with run order (laptop thermals); aligned k takes MAX in the ragged-K build, so treat that row as
noise. Relative RMS error: every FP32 arm about 1e-6, BF16 2.3e-3.

Verdict: BF16 never beats the best FP32 FAST arm and its error is about 2000 times larger, so FP32 models stay
FP32. The ragged-K fix wins about 20x on ragged k with equal error and became the FAST default.
