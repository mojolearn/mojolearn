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

## The diagnosis probe (lane/sm120a-jacobi, 2026-09-14)

`jacobi_probe.apple-m4.txt` is `decomposition/checks/jacobi_sm120a_probe.mojo` run on the Apple M4
(IDENTICAL, Mojo 1.0.0 ed45d567, the same toolchain the 5090 leg recorded) on the same `odd` bytes
(sha256 595dda3a45cf8a3e...): the four refusing lanes' matrices built by the fits' own kernels, each
one bitwise symmetric, repeatable and within 5e-7 of a Float64 host product, then the shipped Jacobi
and a per-sweep copy of it held bit-equal to the shipped kernel. Every case converges in 5 sweeps
from an off-diagonal ratio of 0.0343. The reading the Mac lines add to the finding: `ols.equilibrated`
and `tsvd.gram` are the same matrix up to one power-of-two scale (all 17 scales are `0x3c000000`) and
the Jacobi is exactly invariant to that scale (same `v_hash` and the same host ratio at every sweep),
so a deterministic violation on the 5090 could not have reported 0.505 for ols and 0.0012 for tsvd.
The 5090 run of the probe is owed: brief `docs/lanes/BRIEF_sm120a_jacobi_2026-09-14.md`, wrapper body
`tools/jacobi_sm120a_probe_leg.sh`.
