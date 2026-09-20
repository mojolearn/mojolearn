# FAST NVIDIA Extra Trees TPB screen — 2026-09-20

- Source: `89d745efd`
- Device: NVIDIA L40S, CUDA target `sm_89`
- Public workload: Taxi classification, 1,000,000 x 16, 100 trees, depth 16
- Mode: FAST only; IDENTICAL and DETERMINISTIC were unchanged.

All baseline, TPB-256, and TPB-512 fits produced prediction/model hash
`9d151e0449a610ab`, logloss `0.527541`, AUC `0.608084`, 1,252,834 nodes,
626,467 leaves, and maximum depth 16.

TPB-256 rotated sequence (`base, candidate, candidate, base, base,
candidate`) measured baseline 1024.181 / 1012.695 / 955.437 ms and candidate
999.264 / 1028.738 / 948.913 ms. The aggregate medians suggest 1.33%, but
the adjacent comparisons have mixed signs (+2.4%, -1.6%, +0.7%). That is not
stable evidence of a gain.

TPB-512 measured baseline 1005.203 / 1038.086 / 1053.764 ms and candidate
1016.942 / 1018.954 / 1125.471 ms. Adjacent comparisons again have mixed
signs (-1.2%, +1.8%, -6.8%).

Disposition: retain NVIDIA's 128-thread default. Neither wider occupancy arm
has reproducible evidence across the rotated public-fit comparison. This is
a stability rejection, not a fixed percentage-threshold rejection.

The guarded RunPod `frl0dgmbtedkxs` was terminated at 2026-09-20 13:57:43
America/New_York: DELETE returned HTTP 204, followed by verified GET HTTP 404
at 13:57:44. No machine address, lease state, or credential is retained here.
