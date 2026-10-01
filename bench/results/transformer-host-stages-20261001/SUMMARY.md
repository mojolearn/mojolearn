# Transformer forward serial stages over host tasks (PR #25, lane/neural-pass21), 2026-10-01

Fixture-built inputs (splitmix64, no numpy sampling) in tools/host_threads_ab_check.py, so the shas compare across hosts.

| sha (one thread = policy) | main and branch, Zen 5 / MI325X box | Zen 4 / L40S pod | author's M4 |
|---|---|---|---|
| transformer L 512 / 2048 | 0209f9ff0597cfd4 / 93b0da9f81be0dc5 | same | same |
| mamba3 L 512 / 2048 | 94bd23787c50a4da / 62e3661f915836ba | same | same |

GPU: check-transformer clause (a) 30/30 stages bit-identical on all 349,206 cells, and transformer_options_check PASS, on the MI325X and the L40S.

CPU cells, main vs branch, same box, 5 rounds:
| cell | Zen 5 | Zen 4 |
|---|---|---|
| transformer-infer | 20.69 -> 12.46 ms (1.66x) | 37.97 -> 25.80 ms (1.47x) |
| samba-infer (mean_nll 5.635909657868805 both) | 101.21 -> 70.23 ms (1.44x) | 220.04 -> 202.06 ms |
