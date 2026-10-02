# PRs #88 / #89 / #90: Bayes and ARD device kernels

Each PR was measured on its own branch against main, with the same board datasets and the GPU route on every box:
- NVIDIA: L40S (nvc3)
- AMD: MI325X (DigitalOcean)
- Apple: M2 Pro, Metal

Digests are identical in every tree on every vendor:
- bayesian-ridge: taxi d8493e07013aa66a, istella 63079ff7f9c8c96d
- ard: taxi 7eef23ebe21118d2, istella 04d5abbbf4bf875b

## Median ms (main → lane)

| race | NVIDIA | AMD | M2 Pro |
|---|---|---|---|
| bayes istella, #88 Jacobi | 70232 → 56978 | 185196 → 153948 | 250180 → 217669 |
| bayes istella, #90 grid Gram | 70232 → 55793 | 185196 → 126173 | 250180 → 238645 |
| ard istella, #89 sigma | 15095 → 3326 | 70970 → 13858 | 94249 → 22926 |
| ard istella, #90 | 15095 → 14440 | 70970 → 67517 | 94249 → 92962 |
| ard taxi, #89 | 32.6 → 36.7 | 210.7 → 206.2 | 86.5 → 84.5 |
| bayes taxi, #90 | n/a | 1067 → 1109 | 745 → 779 |

sklearn, on M2 Pro's CPU: bayes istella 10.4 s, taxi 93 ms; ard istella 13.3 s, taxi 23.6 ms.

## Status

- #88: merged (53468fd83).
- #89 and #90 conflict with #88 in x_linear/bayes.mojo and tops.mojo. They go back to the writer to rebase, then get one digest recheck per vendor.

## Raw data

R2, under measurements/2026-10-02/:
- bayes88-amd-{main,p85,p86,p87}
- pr88-metal
- NVIDIA j88 output
