# Mamba-3 kscale work estimate and inter-chunk state-row blocks (PR #26, lane/neural-pass20 ba1f20ac3), 2026-10-01

Fixture-input ab tool; every run on both hosts: sha 94bd23787c50a4da (L 512) and 62e3661f915836ba (L 2048), main and
branch, default rule and MOJOLEARN_M3_INTER_PBLOCKS=8 and =1, one thread = policy.

| stage wall at the policy (ms) | Zen 5 main -> branch (pblocks 8 / 1) | Zen 4 main -> branch (pblocks 8 / 1) |
|---|---|---|
| kscale L 512 | 1.26 -> 0.38 | 3.64 -> 0.59 |
| kscale L 2048 | 2.91 -> 0.51 | 6.90 -> 1.28 |
| inter_chunk L 512 | 0.89 -> 1.00 (0.89 / 1.02) | 3.39 -> 2.55 (2.16 / 5.60) |
| inter_chunk L 2048 | 3.20 -> 3.89 (3.00 / 3.08) | 6.60 -> 7.09 (5.97 / 13.56) |
| mamba3-infer cell | 9.75 -> 8.68 | 26.90 -> 28.14 |

kscale: clear gain. The inter-chunk split's default rule is slower at L 2048 on both hosts; net per call is still a gain.
samba-infer cells: not measured (the job did not build the byte LM host binding that cell needs).
