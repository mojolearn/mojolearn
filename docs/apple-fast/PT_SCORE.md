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

## Verified compiled-arm queue support

Compiled source: `bc112b1726f03cf058f71f781575f8659534b7ba` (both M2 arms rc0,
manager evidence). Helper branch `lane/apple-fast-pt-queue` changes tools/docs
only. Manager stages `manifest.json`, `A.so`, `B.so` under
`~/mq/verified-arms/bc112b1726f03cf058f71f781575f8659534b7ba/x_prep/` on M3,
and provisions FAST base/board venv before running the following CMDs.

First, quality only (100000 rows per fixture):

```sh
MOJOLEARN_NUMERIC_MODE=fast "$HOME/board-0834/cache/venv/bin/python" tools/pt_score_verified.py quality gap26-pt-score
```

Then two conditional one-run-per-arm timing CMDs, on the same helper branch:

```sh
MOJOLEARN_NUMERIC_MODE=fast "$HOME/board-0834/cache/venv/bin/python" tools/pt_score_verified.py timing gap26-pt-score gap26-pt-score-taxi taxi
MOJOLEARN_NUMERIC_MODE=fast "$HOME/board-0834/cache/venv/bin/python" tools/pt_score_verified.py timing gap26-pt-score gap26-pt-score-istella istella
```

Each timing refuses to run unless the quality PASS matches the exact compiled
source manifest, checker, fixture helper, row count, and saved quality artifacts.
It then invokes the manager's `verified_arms.py` with `afc_ab_def.sh`, which
validates both hashes, source scope and defines, sets `AFC_SKIP_BUILD=1`, and
refuses to replay an existing scored race. No kernel recompilation is needed.
A quality failure creates no PASS marker; do not bypass it to obtain timings.

Quality evidence:
`~/mq/out/gap26-pt-score-quality/{A.log,B.log,A.npz,B.npz,compare.log,PASS.json}`.
Timing evidence:
`~/mq/out/race-gap26-pt-score-{taxi,istella}/race.log`.
No new quality thresholds: the original per-column fitted lambda, transformed
values, NLL and reference-normality comparisons decide the quality gate.
The baseline computes the sklearn oracle once for untimed quality only.
