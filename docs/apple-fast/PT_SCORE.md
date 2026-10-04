# PowerTransformer score precision experiment (2026-10-04)

Baseline: freshly fetched origin/main `61ea51757`. Branch:
`lane/apple-fast-pt-precision`. Opt-in `MOJOLEARN_PT_SCORE`; Apple FAST only.
No default change. Pending manager build, quality and M3 timing.

## Diagnosis from source (not a measured attribution)

Prior COLBATCH changed the Welford/Chan fold order and produced max relative
lambda shift 9.5e-3; sklearn-f64 error worsened 5.7e-3 to 6.3e-3. See
EXPERIMENTS.md PT_COLBATCH. Adding SPEC was already slower; do not repeat it.

`transform.mojo:pt_finish` computes and stores the absolute negative
log-likelihood in float32. Near its minimum, objective differences are
quadratic in lambda error; even exact moments rounded back to this f32
objective cannot discriminate sufficiently close candidates. COLBATCH also
rounds each Welford update and Chan merge in float32. `power_from_log` uses
(exp(z)-1)/lambda, which loses precision around lambda=0 (or YJ lambda=2).
These are demonstrable precision mechanisms, but their individual numeric
contributions on board fixtures remain unmeasured.

The experiment uses the analytic NLL score n*Cov(y,y')/Var(y)-sum(J), keeping
centered variance/covariance and Jacobian in existing x_linear.ff paired
float32 arithmetic. Its sign is computed without dividing by variance.
The parallel GPU tiles and parallel GPU reduction remain; the existing
staged schedule supplies 50 bounded bisection steps in [-8,8]. Score zero
is first order in lambda error. A small-z series avoids transform/derivative
cancellation at lambda 0/2. Final transform remains the existing path.
Speculation is disabled by this define, including with PTIMPUTE_ALL, since
its objective-value search tree cannot consume derivative signs.

Limits: logs/exponentials and individual transformed values remain float32;
this is not full float64 emulation. The bounded interval is unchanged from
main; the score method assumes a single minimum within it. Extreme overflow
is not repaired. No accuracy or speed claim until the manager runs gates.

## Manager build / queue proposals

Required binding: x_prep only. Build arms at current main and this branch;
FAST baseline has no added define, candidate has only
`-D MOJOLEARN_PT_SCORE`. Use the manager semaphore/build wrapper:

```
MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_MOJO_BUILD_FLAGS='-D MOJOLEARN_PT_SCORE' bash bindings/build_x_prep.sh
```

Quality CMD after ensure_so installs the corresponding arm, in its checkout:

```
PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=fast ~/board-0834/cache/venv/bin/python tools/pt_score_quality.py dump ~/pt-score-main.npz --reference
PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=fast ~/board-0834/cache/venv/bin/python tools/pt_score_quality.py dump ~/pt-score-candidate.npz
~/board-0834/cache/venv/bin/python tools/pt_score_quality.py compare ~/pt-score-main.npz ~/pt-score-candidate.npz
```

The first command uses baseline binding (script may be copied from this
branch to the baseline checkout); the second uses candidate binding. The
reference is fitted once for quality only, never opponent timing. Checker
saves fitted lambdas, actual transformed outputs, float64 NLL per sample,
and squared skewness plus excess-kurtosis normality diagnostics. It compares
EVERY column to the same float64 sklearn reference and fails any regression
beyond explicit noise (lambda 1e-5, standardized transform RMS 1e-5,
NLL/sample 1e-7, normalized normality-reference error 1e-4). Finite/shape
checks cannot alone pass. Four fixtures: 100k x 11 / 220 prior distribution
mix; Box-Cox lognormal; near-zero/two lambda and near-constant stress.

Manager: one M3 timing per arm for power-transformer on istella and taxi,
against current main; no opponent reruns. No local builds/tests/SSH were
run by this lane. No queue was submitted; manager owns machines. Keep held
unless both speed and real quality pass, then obtain manager merge approval.

## First score trial failed quality

`gap26-pt-score-quality`, compiled source `bc112b172`: stress fixture failed
lambda (worst regression .01330737 vs1e-5), per-observation NLL (4.083e-7
vs1e-7), transform RMS (9.606e-5 vs1e-5), and nonfinite/shape normality
gate. Box-Cox improves/passes; that does not override stress failure. Both
timings were correctly skipped. No acceptance or measured speed claim.

Continuation merges current main `a72da2e4a`; prior label-only x_prep
changes did not change PT numerics. Thresholds remain unchanged. New
`diagnose MAIN.npz CAND.npz --fixture stress` reads saved arrays only and
prints per-column lambdas, objective, normality, output range/std and
reference RMS. Compare now reports failing column indices and which arm
or reference has nonfinite diagnostics, rather than one ambiguous string.

### Actual saved-artifact diagnosis (no new fit)

Inputs fetched by manager:
`~/mojolearn-evidence/apple-fast/pt-score-stress/{A-stress,B-stress}.npz`.
Column diagnostics: same directory `diagnosis.jsonl`; generated only by
reading the saved outputs with the new `diagnose` command. The first six
columns are mixed-sign Yeo-Johnson stress, the last two are
`10 + Normal(0,.001)` rounded to float32.

| Column | main lambda | candidate lambda | float64 reference lambda | main / candidate reference RMS |
|---|---:|---:|---:|---:|
| 0–5 | errors ~1e-4 to6e-4 | within2.2e-7 of reference | -0.24 or2.24 | ~3.66e-5–2.04e-4 /1.51e-7–1.80e-7 |
| 6 | 5.67288685 | 4.97281361 | 52.60793413 | .003213635 /.003309695 |
| 7 | 4.67475605 | 4.08560848 | -63.38517006 | .999999834 /1.000000605 |

Column6 explains all reported worst lambda/NLL/transform regressions:
candidate NLL -6.90966747244 versus main -6.90966788074; reference
-6.90968146625. On column7 candidate NLL actually improves over main:
-6.90464413525 versus -6.90464357745; reference -6.90467946463.

The normality failure is **the reference column7**, not either GPU output.
Saved sklearn float64 standardized output is the constant
-2.110811525568579e-14 (std0), hence skew/kurtosis diagnostic NaN. All GPU
outputs are finite and std approximately1. Its huge negative fitted lambda
makes conventional `expm1(lambda*log1p(x))/lambda` numerically constant
before standardization. Comparing normalized outputs against that constant
also makes column7's existing RMS metric unreliable. This is a separate
oracle-conditioning defect, not authorization to ignore a failed gate.

The runtime limitation is also real: paired-f32 accumulation receives
already-rounded f32 `log`, `y` and `dy`. Near-constant columns produce tiny
centered moments; derivative covariance and Jacobian terms nearly cancel.
Compensating only their accumulation cannot recover per-row precision.
Also both true reference optima lie far outside main's unchanged[-8,8]
search interval, independently limiting attainable lambda agreement.

No speculative kernel repair was applied: fixing only the moment reduction
or increasing the iteration count cannot repair these causes. Extending
the search alone can drive the existing final transform into cancellation
(as seen in the reference). A defensible next repair needs a stable
centered/scaled transform shared by score evaluation AND standardized
output, plus a mathematically stable reference transform at the same fitted
lambda; bracket handling then needs explicit validation. That is broader
than the present score-only change. Keep this candidate held, all stress
columns retained, tolerances unchanged, timings skipped. No fallback to
serial/CPU fitting or unsupported width/variance gate was introduced.
