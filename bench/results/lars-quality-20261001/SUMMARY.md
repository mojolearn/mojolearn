# Lars on the Istella board cell (lane/lars-quality, 2026-10-01)

Board cell: algos/lars on istella, NVIDIA L40S, 0.8.25. Ours scored held-out R2 -1.36, cuML 0.328, and the scikit-learn arm -4.2e13 (raw/algos/rows-full/lars-istella.json). Every arm fits on the same 1,000,000 standardized fit rows and is scored on the same 100,000 held-out rows. The parameters match (n_nonzero_coefs=500, fit_intercept, eps). The board is not the cause.

## Cause

scikit-learn's `method='lar'`: when an active coefficient crosses zero (z_pos < gamma), it flips `sign_active` and adds no feature on the next step. In LAR an active variable's correlation keeps its sign; only the coefficient changes sign. The flip therefore breaks the equiangular step: the active correlations stop being equal, and the path diverges. x_linear/lars.mojo copied that behavior. The fix (DEVIATION 5010) keeps the zero-crossing step for LassoLars only.

The numpy port (port.py) with the flip reproduces scikit-learn to the last digit. Without the flip it ends at least squares.

Smallest reproducer: n=60, d=5, seed 58 (lars_inputs.npz, `python -m mojolearn.tests.test_x_linear_sanity lars-crossing`). Train R2 values: least squares 0.9266, scikit-learn Lars (float64) -25.27, ours 0.8.32 the same as scikit-learn, ours fixed 0.9266.

## Istella (mkblock.py + fit.py, the board's block rebuilt bit for bit)

| fit | held-out R2 | train R2 | active | coef digest |
|---|---|---|---|---|
| least squares (float64) | 0.3296 | 0.3385 | | |
| scikit-learn 1.9.1 Lars, float32 X | -1.05e22 | | 201 | |
| scikit-learn Lars, float64 X | -7e121 | | 201 | |
| ours 0.8.32: x86 CPU, AMD MI325X, NVIDIA L40S | -1.3603 | -1.1059 | 83 | 9e39e78ea5e0c6fc on all three |
| ours fixed: x86 CPU, AMD MI325X, NVIDIA L40S | 0.3090 | 0.3131 | 87 | 3a39bddfaccc293d on all three |

## Identity (digest.py on lars_inputs.npz)

- 0.8.32: DO x86 CPU == AMD GPU == NVIDIA pod CPU == NVIDIA L40S (digests-0832.txt).
- Fixed: Apple M4 CPU == Metal == DO x86 CPU == AMD GPU == NVIDIA CPU == L40S (digests-new.txt). FAST on Metal also passes the sanity case.
- Rows that did not change: all ten no-crossing rows (Lars and LassoLars) and every LassoLars row. Rows that changed: the three crossing Lars rows, as intended.

## Remaining gap (not fixed here): float32 Gram chains

The fixed fit scores 0.309, against 0.328 for cuML and 0.330 for least squares. The cause is the centered Gram and X'y. Each is one sequential float32 chain over 1M rows (`centered_gram`, `t_centered_gram`, `xg_gram_kernel`). On Istella the max relative error is 1.35% for the Gram and 0.8% for X'y (chain.py). Near-collinear features then fail the Cholesky pivot and are skipped: 111 of 201 nonconstant features. chain2.py runs the same fixed path on the alternatives. With a Gram in 1024-row blocks, the error is 2e-5 and R2 is 0.307. With a compensated (Kahan) chain, the error is 1.2e-7 and R2 is 0.327. That change would alter the bits of every Lars/LassoLars, BayesianRidge, ARD and RidgeCV/RidgeClassifier Gram fit, so it needs its own lane.
