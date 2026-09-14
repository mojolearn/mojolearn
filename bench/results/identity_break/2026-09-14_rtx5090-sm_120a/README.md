# RTX 5090 (sm_120a), the first untested architecture leg (2026-09-14)

`tools/identity_three_columns_leg.sh` on a RunPod NVIDIA GeForce RTX 5090 (32 GB, Blackwell consumer,
`sm_120a` resolved from compute capability 12.0), commit 6796ceff9, all 25 bindings built, the 46
lanes on the nine fixtures, two fits per cell (`bench/results/e1g/2026-09-14_093508-nvidia-rtx5090-identity-46-lanes`).

Result: `cells=414 stable=410 moved=0 refused=4`. Diffed as a fourth column beside the 2026-09-14
Apple M4, H100 and MI300X record: `summary: IDENTICAL=414` and `summary (infer/model): IDENTICAL=459`;
every cell the 5090 produced carries the same bits as the three vendors (the 41 ONE-COLUMN model
cells are the save/load the newer commit added to logistic, ols, ridge, tsvd, pca, which the
older columns record as no-save).

The four refusals are all on the `odd` fixture (17 columns) and all through the device Jacobi
eigen solver on the 17 x 17 Gram or covariance matrix: pca, tsvd, ols (`lstsq_eig`) and ridge
(`svdEig`) report "the device Jacobi did not converge in 15 sweeps" with off-diagonal ratios
0.0027, 0.0012, 0.505 and 0.0027 against a tolerance of 1e-07. On the M4, the H100 and the MI300X
the same four cells converge and agree. Under IDENTICAL the sweep is pinned arithmetic, so a
different convergence on one architecture is a contract violation on that architecture, not a
tolerance matter; it is DEVIATION 2711, open, diagnosis lane `lane/sm120a-jacobi`.
