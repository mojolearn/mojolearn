# PR #67 pooled logits output (lane/neural-pass61), 2026-10-01

Arms: main vs lane/neural-pass61 (main merged in), released 0.8.33 venv with the branch builds overlaid.
Bulk logs in R2: measurements/2026-10-01/ab-pr67-nvidia.tar.gz (row in r2-index.tsv).

## NVIDIA L40S (nvc1), stage runs x2 interleaved + timers build + board race

| row | main | pass61 |
|---|---|---|
| lm-forward stage (ms) | 51.7 / 51.2 | 17.2 / 17.0 |
| lm-train-step stage (ms) | 41.8 / 41.5 | 41.6 / 41.7 |
| lm-forward race (ms) | 54.7 | 16.8 |
| lm-forward digest | 4a8e781b0739a038 | 4a8e781b0739a038 |
| losses begin | 9.018733024597168, 8.418445587158203 | same |

Timers (branch): `python.logits_out` 29.5 ms fresh on the first two calls, then 0.007 to 0.017 ms pooled;
`python.logits_binding` 16.7 ms steady.

AMD MI325X (DO pass8) and M3 Ultra (~/pr67_mac.sh) pending; merge only when all three agree on bits and speed.

Also tracked in the same evidence commit: bench/results/pr65-attention-stack-20261001/nvidia-raw.txt (#65 NVIDIA raw).
