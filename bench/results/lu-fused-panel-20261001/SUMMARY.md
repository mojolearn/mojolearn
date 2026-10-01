# Fused LU panel kernel (PR #40, lane/neural-pass36 a604980aa), 2026-10-01: NOT MERGED (not faster)

lu-factor / lu-solve (n 8192), digest 43e06346de01739b in every run on all three vendors. "restore" =
MOJOLEARN_XD_LU_FUSED=0 (the blocked route already on main, PR #37).
| | before (0.8.32 / main) | fused (branch) | blocked (restore) |
|---|---|---|---|
| AMD MI325X lu-factor | 10,692.7 ms (0.8.32) | 10,583.9 | 10,059.2 |
| NVIDIA L40S lu-factor | 6,267.7 ms (0.8.32) | 4,704.3 | 4,496.1 |
| Apple M3 Ultra lu-factor | 14,499.3 ms (main) | 15,572.7 | 14,464.5 |
The fused panel is 4-8% slower than the blocked route on every vendor; the remaining LU time is not the launch count.
