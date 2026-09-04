# Upstream is retiring what some of our lanes mirror

Found 2026-08-23 night by the holtwinters lane, then swept across the pinned
tree. Recorded here because it spans lanes and the decision is Andrew's, not
any single lane's.

## The finding

**The whole `cuml.tsa` module is deprecated in cuML 26.08 and removed in
26.12.** Not one estimator. All of it. Read in the pinned tree,
`/Users/andrewhendel/CascadeProjects/upstream/cuml-v26.08.00`:

| upstream | notice | our lane |
|---|---|---|
| `python/cuml/cuml/tsa/arima.pyx:139` | `cuml.tsa.ARIMA` and `cuml.ARIMA` removed in 26.12 | `arima/` |
| `python/cuml/cuml/tsa/holtwinters.pyx:60` | `cuml.tsa.ExponentialSmoothing` removed in 26.12 | `holtwinters/` |
| `python/cuml/cuml/tsa/auto_arima.pyx:95` | `cuml.tsa.auto_arima.AutoARIMA` removed in 26.12 | `tsa/` (auto_arima's d loop, DEVIATION 672, landed a87c529) |
| `python/cuml/cuml/tsa/stationarity.pyx:49` | `cuml.tsa.stationarity.kpss_test` removed in 26.12 | `tsa/` (KPSS, DEVIATION 671, landed a87c529) |

**NOT the same thing, do not conflate.** `metrics/`, `ensemble/` and
`cluster/kmeans` also show `.. deprecated:: 26.08` lines, but those are
INDENTED under parameter documentation (`convert_dtype` and friends). Those are
parameter deprecations, not algorithm removals, and they do not threaten those
lanes. The four rows above are CLASS level.

## What it does and does not change

It changes NOTHING about correctness. v26.08.00 is the pinned target, the ports
are faithful mirrors of it, and a mirror of a pinned commit does not rot when
upstream moves. The deprecation warning itself is NOT ported, because mirroring
it would be porting their release policy rather than their algorithm.

What it changes is one OWED item with a deadline. After 26.12 there is no
upstream to re-sync against and no way to run a real cuML arm for comparison.
Every "check our numbers against a live cuML run" task in these three lanes has
an expiry.

## The open question, which is Andrew's

Three lanes mirror an algorithm family its own authors are retiring. The
options, stated without a recommendation attached to any of them:

1. Finish and freeze them as mirrors of a pinned commit, which is exactly what
   they are, and say so in the READMEs. Their value does not depend on upstream
   keeping the code.
2. Do any cross-check against a live cuML BEFORE 26.12, and treat that as a
   dated task rather than an open one.
3. Deprioritize the family relative to lanes whose upstream is not retiring.

Nothing here is acted on. See `arima/README.md`, `holtwinters/README.md` (UNPORTED row 20) and `tsa/README.md`.
