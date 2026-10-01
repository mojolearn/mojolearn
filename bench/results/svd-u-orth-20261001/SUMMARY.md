# SVD U from the sliced orth (PR #23, lane/neural-pass17, main merged in), 2026-10-01

Board svd cell, ours, 3 rounds. AMD MI325X and NVIDIA L40S: released 0.8.32 (before) vs 0.8.32 + main + #23
(after) vs after with MOJOLEARN_LINALG_SVD_U=householder (restore = main's route). M3 Ultra: main vs main + #23 vs
householder. Raw lines in races.txt.

| cell | AMD restore -> after | L40S restore -> after | M3 main -> branch | digest after (all three) |
|---|---|---|---|---|
| taxi | 267 -> 132 ms | 320 -> 136 ms | 230 -> 49 ms | 64f2344451a1d423 (was f07891b9aaf42a98) |
| istella | 10,336 -> 10,206 ms | 16,093 -> 16,112 ms | 15,950 -> 15,829 ms | 92c189cb2337dc73 (unchanged) |

U's bits change by design (the new route is more orthonormal, see PR #23); the new bits agree on AMD, NVIDIA and
Apple. Merged under the standing rule (e990bf3bd): same bits across hardware within a release.
