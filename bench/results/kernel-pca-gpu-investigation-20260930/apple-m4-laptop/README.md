# Apple M4 (laptop, Metal) check of the opt-in Lanczos KernelPCA, 2026-09-30

mojolearn 0.8.31 macOS arm64 wheel; `_expansion_neighbors.py` replaced by the
candidate's (main's copy equals the wheel's, so the only change is the
candidate's 31 lines). `tools/kernel_pca_trial_check.py`, same board-stride
taxi input as the L40S run (input sha256 prefixes 6eadcafd451b, 3f6f8569cc6c,
62050adcc4e8 equal NVIDIA's).

| rows | passed | repeat identical | output bits vs NVIDIA L40S | fit_transform ms (1st, 2nd) |
|---|---|---|---|---|
| 256 | yes | yes | SAME | 501, 94 |
| 1000 | yes | yes | SAME | 177, 105 |
| 10000 | yes | yes | SAME | 2939, 2278 |

Old path (full host eigensolve, default today) on the same M4, same input:
256 rows 338 ms, 1000 rows 17444 ms. A 3000-row old-path fit was still running
after 9 minutes and was stopped.

AMD identity still pending (no AMD box free outside the board machines).
