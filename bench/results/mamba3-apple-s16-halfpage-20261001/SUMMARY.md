# Mamba-3 backward, Apple half-page S16 arms (PR #44, lane/neural-apple3), 2026-10-01

`tools/strides_digest.py 2 512 384` (Mamba3Block forward + backward, y and every gradient, two calls): the digest
JSON (md5 26abf7d32984a13d6441a7222132eae3) is byte-identical for every arm (default, naive, regs2h, regsh) on the
NVIDIA L40S, the AMD MI325X and the Apple M3 Ultra, and equals main's on the M3.

Sum of the Mamba-3 backward stages (`tools/mamba3_backward_timing.py`, same shape, ms):

| arm | M3 Ultra | MI325X | L40S |
|---|---:|---:|---:|
| main | 421.7 | | |
| branch default | 170.5 | 71.0 | 13.3 |
| naive | 416.6 | 71.7 | 46.7 |
| regs2h | 170.2 | 56.9 | 16.6 |
| regsh | 382.3 | 63.9 | 19.9 |

Metal: the new default is 2.5x faster than main. NVIDIA and AMD keep their defaults (regs, regs2); on AMD the opt-in
regs2h is 20% under regs2 (follow-up). The M3 neural_experiments samba-train-step rows failed only because the job did
not build the training binding; mamba3-forward ran.
Also here: PR #40 on the M3 before main was merged into it (m3-ultra/races.txt; branch 45 commits behind main, not a
valid comparison). PR #40 with main merged: even with main by default, fused arm 46% slower; not merged.
