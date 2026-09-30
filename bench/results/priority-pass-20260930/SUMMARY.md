# Priority-list pass, NVIDIA L40S, 2026-09-30

lane/neural-net-experiment 525272b4a plus two compile fixes (restored Mamba-3 prefill session bindings;
`mut` operands on the device-resident backward), built once (seven IDENTICAL bindings, sm_89) over the
mojolearn 0.8.31 wheel. Transformer session checks (reuse, refusals, lifetime, budget) and the fresh-prefill
check PASS. Each row: the new route vs its restore env, same box, same input; board algos races, ours only.

| row | new | old | output hash | verdict |
|---|---:|---:|---|---|
| Mamba-3 backward session (vs MAMBA3_LEGACY_SETUP=1) | samba-train 135.9 ms, mamba3-fwd 2.62 | 141.6, 2.93 | same (digests, loss series) | keep |
| Mamba-3 stage reuse (vs RETAIN_STAGES=0) | 135.9 | 137.5 | same | keep (small) |
| LM-step norm dW on the block workspace | lm-train 44.49 | 44.88 | same | keep (small) |
| gemm-int8 32x32 tile (vs INT8_MMA_REFERENCE build) | 2455 ms | 2454 | same | neutral; the 2.4 s is outside the kernel |
| SVD Istella / taxi | 47.9 s / 1.29 s | 54.1 / 1.56 | same | keep |
| QR Istella / taxi | 26.6 s / 1.26 s | 26.6 / 1.26 | same | neutral |
| Cholesky (board lane is the public class, untouched) | 0.91 s | 0.84 | same | neutral |
| lr-warmup-cosine | 0.73 s | 332.5 s (0.8.25 AMD board) | same as the old route's AMD hash 61e4ce1e09ca405e | keep; old arm stopped here after 20 min |
| Adafactor | 0.66 s | 6.59 s | same, and = 0.8.25 AMD board ff4a5ff93e72424e | keep |
| clip-grad-norm | 7.8 ms | 47.2 ms | same (bea62536a1718eb8) | keep; note: 0.8.25 AMD board read 973aad3bdd470a4e, pre-existing, to check |

eigh: defaults unchanged by design; not raced (the default route runs for hours at n = 4096).
results.json's per-row fields read None: its parser looked for `cells`; the numbers above are from each race's raw JSON.
AMD and Apple not run.
