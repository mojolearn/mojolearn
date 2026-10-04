# Apple Kalman time-parallel experiments

Status: **SOURCE ONLY — NOT COMPILED, NOT RUN, NOT MEASURED.** No speedup,
correctness, forecast-quality, or numerical-equivalence claim is made. No
production dispatch, candidate search, optimizer, or build target is changed.
These are Metal shader candidates, not a callable AutoARIMA backend yet.

## Variants

| ID | Implemented source candidate | Hypothesis / activation recipe |
| --- | --- | --- |
| K1 | Gaussian associative prefix scan | Encode `kalman_scan_leaves`; ping-pong `kalman_scan_step` for strides 1,2,4,... less than nobs; encode innovations, likelihood and final state. Replaces the long covariance and mean recurrence with logarithmic parallel depth. |
| K2-B8 / B16 / B32 | Blocked Gaussian prefix scan | Leaves; `kalman_scan_blocks` with block_size 8,16,32; scan ceil(nobs/B) totals; `kalman_scan_carry`; same outputs as K1. Trades short local serial chains for fewer expensive matrix compositions and less device-memory traffic. |
| K3 | Scalar exact-observation specialization | Only rd=1, n_diff=0, Z=[1]. Encode scalar innovations, common likelihood reduction, scalar final state. Every innovation is independent after t=0; avoids both the recurrence and scan workspaces. Covers eligible low-order candidate fits, not general ARIMA. |

The implementation follows the Gaussian filtering monoid in
[Särkkä and García-Fernández, Temporal Parallelization of Bayesian Smoothers](https://arxiv.org/abs/1905.13002).
The publication metadata/abstract was located; equation-by-equation verification
against its full text is still owed (the repository PDF fetch failed).
Each record contains a conditional affine Gaussian and an observation
information factor. Composition eliminates the intermediate state; chronological
order matters. Inclusive prefixes supply filtered means/covariances, which the
output kernel converts back to the next-step predictive convention.

## Explicit activation boundary

There is deliberately no environment variable that silently switches the product.
Future harness work must compile this source explicitly and supply one `Model`
per dispatch. Do not run that work as part of this source-only task. Encoding the
sequence above is the intended activation interface. Metal `Stage.count` is
nobs for K1 or the number of block totals for K2; `Stage.stride` doubles each pass.
Every input/output scan pair must use distinct buffers. Track the final ping-pong
buffer, including the zero-pass case for nobs=1 or one block. Ensure buffer
visibility between passes; never attempt an in-place device-wide scan. Separate
candidate models may be encoded into one command buffer after independent buffer
ranges have been assigned; do not share scratch across simultaneous fits.

Inputs come from `arima/impl/batched_kalman.mojo` initialization, not a new
stationarity or approximate steady-state initializer:

- Convert production column-major T, RQR and P0 to fixed-stride row-major Mat;
  copy alpha0 and Z. Set rd in [1,8], nobs>n_diff, and drift[n_diff]=mu with
  all other entries zero. P0 must remain the original pre-filter covariance.
- `Model.Q` is RQR including sigma2. Rank deficiency is allowed; neither Q nor
  posterior covariance is inverted. Leaves require Z Q Z' > 0 after t=0;
  models failing that extra eligibility condition must use the existing kernel.
- Missing/nonfinite observations are refused. Do not fill gaps with zero or
  silently shorten the likelihood. Production already refuses missing values.
  Validate all model matrices and vectors as finite before dispatch.
- For exogenous inputs, subtract the existing observation intercept before
  dispatch. Residuals/likelihood then refer to this adjusted series; add the
  original intercept back to reported predictions and future forecasts.
- Validate `block_size>0`, scan counts, lengths and rd on the host before any
  kernel. Shader checks are diagnostic, not a substitute for buffer bounds.
- Read and honor all status outputs before accepting loglike, final state or
  forecasts. On error those outputs can be unwritten; they must never feed the
  optimizer. Likelihood uses first-error signed step codes for diffuse/summed
  observations; an invalid likelihood length uses INT_MAX.

## Semantics and limitations

Initialization uses the supplied finite diffuse P0 (including the existing
kappa=1e6 convention). The first n_diff observations still update the filter but
contribute no likelihood term. The common reduction retains the existing
`-.5 * (sum(log(F)) + N * (sum(v*v/F)/N + log(2*pi)))` expression and ascending
reduction order. No extra log of the quadratic sum is introduced.

The full scan does **not** preserve production Float32 operation order, ftz
behavior, or its stepwise covariance symmetrization/absolute-diagonal operation.
K3 also changes arithmetic despite being equivalent in real arithmetic. These
are opt-in Apple fast-path candidates, never substitutes for IDENTICAL. An
unstable/poorly conditioned factor solve refuses rather than adding jitter.
Fixed 8x8 temporaries and two pivoted solves per compose can cause register spills;
K1 does more work than a serial Kalman filter and may lose at this state size.
K2 exists specifically to explore that tradeoff. A future rd-specialized template
or square-root implementation needs its own validation; neither is claimed here.

Likelihood reduction remains one short serial sum; it is a deliberate control to
avoid conflating scan arithmetic and reduction changes. Existing forecast logic
can consume returned predictive alpha/P, but no forecast adapter is wired here.
No confidence intervals, missing-value support, gradients, model grouping,
optimizer changes, or search expansion are added.

## Deferred correctness gates (NOT RUN)

Before any future performance evaluation, compare per-step predictions,
innovations, F, final predictive state, and likelihood against the existing
sequential implementation. Include nobs=1, odd sizes, every block remainder,
rd=1..8, diffuse orders, nonzero mu, exogenous intercepts, near-unit roots,
rank-one Q, nonpositive F, nonfinite data, and ill-conditioned compositions.
Compare K1 and every K2 block size, chronological composition and initialization;
verify first-error refusal and ping-pong parity. Keep derivative perturbations and
candidate grouping fixed while checking optimizer acceptance and selected order.

Taxi forecast quality is a separate acceptance gate. Hold the search space,
initial parameters, optimizer budget, split and forecast horizon fixed. A faster
likelihood that selects worse forecasts is not a successful experiment. Longer
search is intentionally excluded. No new measurement scripts or automated runs
were added.
