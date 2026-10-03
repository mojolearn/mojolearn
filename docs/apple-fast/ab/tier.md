# lane/apple-fast-tier: FAST slower than IDENTICAL (PLAN.md item 3)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles the old code.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_APPLE_FAST_GEMM_NT_TILED=1` | env, read at dispatch | `core/gemm.mojo` gemm_nt, gemm_nt_gram | the threadgroup-tiled Apple kernel (IDENTICAL's geometry, plain `fma` chains, `pinned=False`) instead of MAX's matmul |
| `MOJOLEARN_APPLE_FAST_GEMM_TN_V1=1` | env | `core/gemm.mojo` gemm_tn past split-K (m == n > 128: pca/ols Istella d=220) | IDENTICAL's `gemm_tn_identical_v1` arm (pinned OP_TN plan) instead of transpose + MAX matmul |
| `MOJOLEARN_APPLE_FAST_GEMM_PINNED=1` | env | `gemm/checks/gemm_identical.mojo` identical_gemm / identical_gemm_into | skips `_fast_vendor_gemm`, so FAST takes IDENTICAL's shipped plan (rbf-sampler / nystroem transform GEMMs, GP, kernel methods) |
| `-D MOJOLEARN_SEQ_FAST_FMA=1` | build define | `sequence/ops.mojo` fma3 | fused `fma` instead of FAST's unfused `a * b + c` on every sequence chain (theta's one-thread-per-series Nelder-Mead loop) |
| `-D MOJOLEARN_SEQ_THETA_REG=1` | build define | `sequence/theta.mojo` theta_run_reg / theta_forecast_reg | the theta recurrence in registers: no per-step state row and error written to and read back from device scratch (64 threads, nothing to hide the latency behind); only row n-1 is stored; same operations, same order |

Why: under FAST, `core/gemm.mojo` and `identical_gemm_into` hand every GEMM to MAX's `linalg.matmul`,
which on Apple has no split-K and no tuned fp32 tile; IDENTICAL's hand kernels are what the board's
IDENTICAL column ran faster with (pca Istella 814 vs 988 ms; rbf-sampler, theta likewise).
Theta's FAST/IDENTICAL kernels differ only in `fma3`, `mul` and `div`, so the fma arm is the one
cheap hypothesis; the two baselines (`tier-theta-ident`, `tier-theta-fast`) measure the gap at head first.

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality
stays within FAST's run-to-run spread; then the env read goes and the arm is the code.
`tools/afc_ab_def.sh` is new: afc_ab.sh for a build define (two FAST builds of one binding, alternated).
