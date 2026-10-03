# BayesianRidge istella quality under FAST (lane/apple-fast-bayesq, 2026-10-03)

## Measured (M3, tools/bayesq_probe.sh istella, tag bayesq-probe-istella, head f1518fe87)

| arm | alpha_ | lambda_ | n_iter_ | held-out r2 | rmse |
|---|---|---|---|---|---|
| scikit-learn float64 | 2.147 | 65.68 | 10 | 0.3287 | 0.6844 |
| scikit-learn float32 (the board's sklearn-cpu arm) | 1.45e-3 | 6.5e-11 | 300 | -890.2 | 24.94 |
| ours FAST (main) | 4.5e-5 | 1.1e-8 | 300 | -32316 | 150.2 |
| ours FAST at sklearn-f32's alpha/lambda, max_iter=0 | | | 0 | -1.0e12 | 8.4e5 |

istella fit rows: 19 of 220 columns constant; the float64 centered Gram has 35 eigenvalues below 1e-12 max ev
(max ev 3.03e7) and 37 below 1e-7 max ev. Projecting each arm's coef onto the float64 eigenbasis, ours carries its
held-out error in REAL directions of ev 300-8000 (z ~1e3 where float64 has ~0.1), not in the null ones.

## Cause

`x_linear/bayes.mojo:bayes_eig_prep` (the Jacobi of the float32 centered Gram, then `tmp = max(0, ev)` and
`vty = V'X'y`) resolves eigenvalues only to about eps * max ev (~4 units here). The 16 exactly collinear directions
(beyond the 19 zero columns, which stay exact zeros) come back as noise eigenvalues with noise V'X'y, so
`bayes_coef_one` (`x_linear/bayes.mojo`, z_k = vty_k / (ev_k + lam/alpha)) puts huge weight on them, and their
eigenvectors mix about noise/gap (1e-2) into the real directions of ev 300-8000. The evidence update
(`bayes_step`) then sees a huge |w|^2 and small sse and walks lambda to 1e-8: an unregularized fit along noise.
scikit-learn's float32 SVD of X has the same failure at a smaller scale (r2 -890); float64 has none.
Not the cause: centering, the alpha/lambda formulas, tol, n_iter, the guarded Gram sse (the max_iter=0 solve alone is
already 1e12 off at sklearn-f32's hyperparameters).

## Fix (FAST + Apple, `BAYES_FAST_Q`, `-D MOJOLEARN_BAYES_FAST_Q_OFF` is the A/B arm)

`bayes_eig_prep` sets ev and V'X'y to 0 for every direction with ev <= d * eps32 * max ev (numpy's pinv/matrix_rank
cut). Those directions get z_k = 0 at every lam/alpha and add nothing to gamma, which is what float64 gives them
(z ~1e-5, gamma term ~0). O(d) on the lead after the team Jacobi; the grid driver, the one-block fit, the guarded
and plain Gram sse all read the zeroed tmp/vty (dz = 0 there, so the delta form and its bound are unchanged).
taxi (well conditioned) has no direction under the cut: same bits expected.

## IDENTICAL

Fixed in lane/fix-bayes-null (2026-10-03): the cut now runs unconditionally in `bayes_eig_prep` on every vendor, every
mode and the host column; `BAYES_FAST_Q` and `-D MOJOLEARN_BAYES_FAST_Q_OFF` are gone (FAST and IDENTICAL share the code
path). IDENTICAL bits change on every vendor and the host column together. The history below is kept as measured.

IDENTICAL runs the same float32 Gram + Jacobi with no cut (archived M3 board 2026-09-29: IDENTICAL istella r2 -41689,
rmse 170.6): the same bug. Not changed here (the orchestrator owns IDENTICAL); flagged in
~/mojolearn-evidence/apple-fast/FLAGS-for-orchestrator-2026-10-03.md. The cut would change IDENTICAL bits on every
vendor and the host column together.

## A/B (M3, afc_ab_def x_linear, one run per arm; arm A `-D MOJOLEARN_BAYES_FAST_Q_OFF`, arm B the cut)

| tag | arm A | arm B |
|---|---|---|
| bayesq-istella | 2839 ms, r2 -32316.3, rmse 150.17, digest e924b832 | 494 ms, r2 0.32840, rmse 0.68456, digest 360b6f8a |
| bayesq-taxi | 19.2 ms, r2 0.908983, rmse 4.80505, digest 3ce1ea7b | 23.3 ms, same r2/rmse, same digest 3ce1ea7b (bits unchanged; timing noise, n=1) |

istella now matches scikit-learn float64 (0.3287) and beats the board's float32 sklearn arm (-890). Faster too: the
evidence iteration converges instead of running 300 iterations.
