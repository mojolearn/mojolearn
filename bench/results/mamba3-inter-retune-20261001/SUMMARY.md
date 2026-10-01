# Mamba-3 inter-chunk split retune (PR #29, lane/neural-pass25), 2026-10-01

Every run on both hosts: sha 94bd23787c50a4da (L 512) / 62e3661f915836ba (L 2048), main and branch, default rule and
MOJOLEARN_M3_INTER_PBLOCKS=8 / 1, one thread = policy.
| inter_chunk at the policy (ms) | Zen 5 main -> branch | Zen 4 (128 threads) main -> branch |
|---|---|---|
| L 512 | 1.00 -> 0.92 | 2.73 -> 1.93 |
| L 2048 | 5.07 -> 3.30 | 7.42 -> 7.63 |
| mamba3-infer cell | 8.21 -> 7.01 ms | 25.41 -> 28.40 ms (this host's run-to-run spread) |
