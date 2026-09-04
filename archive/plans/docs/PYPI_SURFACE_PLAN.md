# Exposing the whole library on PyPI: what is there, what is missing, what it costs

Written 2026-08-24, answering: *"expose everything in mojolearn to pypi ...
that may be clean separation?"*

**Short answer to the separation question: yes, and it is already the shape
this project has.** The paper repository (`~/CascadeProjects/mlsys`) holds a
manuscript that quotes result artifacts; this repository holds the system;
PyPI holds the thing a reader can install. The paper does not ship code and
the library does not carry claims. What is NOT yet true is the "everything"
half: the released surface is a fraction of what is built and certified.

**One constraint the separation runs into.** MLSys is double-blind and
anonymization failures are desk rejected, so the main-track submission may not
name the package, link it, or cite it. The artifact paragraph in the paper is
written for the camera-ready. **The industrial track's rules permit product
names and URLs under an anonymized byline**, and that is the one decision
worth making deliberately rather than defaulting past: on the industrial track
`pip install mojolearn` can appear in the submitted paper.

## Where the surface is today

Published on PyPI, macOS arm64 wheel, every numeric mode as its own compiled
binary set in one wheel. **Read `python/mojolearn/__init__.py`'s `__all__`, not
a table here** -- the count in this file went 11 -> 16 -> 24 -> 25 in eight
days and every written-down number was false the day after. At 2026-09-01 it
is 25 estimators plus the `metrics` and `linalg` submodules and the
`kpss_test` / `select_d` functions.

**`SVR` was the cheapest of these and is worth recording as a shape.** It
needed NO new compiled extension: it rides `_mojolearn_svm.so`, which
already carried `SVC` and the isolation forest, so `_backend.py`'s
`_MODULES`, `pyproject.toml` and `build_release_wheel.sh`'s `EXT_NAMES` were
all unchanged. The whole cost was two host entry points, two bindings, one
class and one boundary gate, against a solver that was already 44 of 44. The
ordering rule below is what found it: highest surface per unit of work over
a lane that is already gated.

## What is built and still NOT reachable from Python

This table had ten rows on 2026-08-24. Eight of them shipped: `holtwinters`
(`ExponentialSmoothing`), `solver` (`Lasso`, `ElasticNet`), `hierarchy`
(`AgglomerativeClustering`), `metrics`, `isolation_forest` (`IsolationForest`),
`svm` (`SVC`, and `SVR` since 2026-09-01), `gemm` (the `linalg` submodule)
and spectral clustering (`SpectralClustering`). Those rows are deleted
rather than annotated. What is left:

| family | what exists | what is missing | size |
|---|---|---|---|
| `spectral/` | cuVS spectral embedding as well as cuML spectral clustering, its own identity contract | `SpectralEmbedding` -- the CLUSTERING half shipped, the EMBEDDING half did not | **M** |
| `arima/`, `tsa/` | **`ARIMA` SHIPPED 2026-09-01** (`fit`/`predict`/`forecast`, batched, commit `cc269dca`; `estimate_x0` and an own-written batched L-BFGS closed the no-fit gap the "what is missing" cell used to record), on top of `kpss_test` / `select_d`; arima's KERNEL card is byte-identical on THREE vendors at one commit, `221aa141`, 139 stages (`bench/results/e1/CERT_2026-08-31.md`), the FIT is one Apple box, and the Python surface gate (`check-arima-surface`) RAN GREEN IN BOTH TIERS on that same Apple M4 on 2026-09-02, 88 checks and 0 failed in each process, **APPLE ONLY** with a second and third vendor through the surface still open (`arima/README.md`) | `AutoARIMA` -- the `p/q/P/Q/k` search and its information-criterion arms are NOT PORTED (`arima/NOT_IMPLEMENTED.tsv`); exogenous regressors, confidence intervals, CSS and missing observations refused by name | **M** |
| `mamba/` | one Mamba-1 block, its own contract | not scikit-learn shaped; belongs behind a separate module or not at all | **defer** |

The ordering rule the shipped eight followed, and the reason it worked: take
the family with the highest surface per unit of work and an existing
three-vendor card, so the identity table grows WITH the surface rather than
after it.

## What this does NOT need

- No change to the wheel's structure: it already carries every numeric mode
  as its own compiled binary set. (This read "both numeric modes ... selects
  on `MOJOLEARN_NUMERIC_MODE` at import" while there were two and the
  environment was the only selector. Since 2026-08-29 there are THREE --
  `fast`, `deterministic`, `identical` -- and the mode is a runtime parameter:
  `mojolearn.set_numeric_mode()` or `numeric_mode=` on an estimator, with the
  environment variable still setting the starting default.)
- No change to the identity machinery: each family above already has its
  ledger rows and, for six of them, three-vendor cards.
- No new claim in the paper. **The paper's certificate is per training stage
  and already includes families that are not on the Python surface** (the
  classical lanes are Mojo-only paths reached by the matrix). Exposing them
  makes the artifact match the paper; it does not change what the paper says.

## What a release run costs, and the standing constraint

`docs/PYPI_RELEASE.md` is the procedure: clean-worktree build at the M1 ISA
floor, smoke fits on real Apple GPU hardware, then upload. Two standing rules
bound how this gets done: **no heavy compute on the laptop** (builds go to the
ephemeral runner, one at a time), and **publishing is outward-facing and needs
Andrew's word each time** -- a version on PyPI cannot be recalled, only
yanked.

## The three-tier matrix: every binding now COMPILES in every tier, 2026-09-01

`packaging/macos/build_release_wheel.sh` defaults to
`MODES="fast deterministic identical"` and drives all twelve build scripts
under each, so a release cut attempts **36 binding-tier combinations**. Until
2026-09-01, **three of those thirty-six had never been attempted at all**:

    _mojolearn_arima     identical        never built
    _mojolearn_arima     deterministic    never built
    _mojolearn_training  deterministic    never built

On-disk evidence at the time: `python/mojolearn/` carried 12 `.so`,
`identical/` carried 11 and `deterministic/` carried 10. All three now build.
Coverage is **12 / 12 / 12**.

**WHY A GREEN LANE DID NOT COVER THIS, because the asymmetry is the whole
lesson.** arima's ARITHMETIC is certified identical on three vendors -- 139
card stages byte-identical at `221aa141`, `bench/results/e1/CERT_2026-08-31.md`
-- so every status line about that lane said identical was fine. But that
certification ran the CHECK binary. A BINDING is a different compilation unit
with a different entry point and its own build script, and nothing had ever
compiled it under `-D MOJOLEARN_NUMERIC_IDENTICAL=1`. **A green card is not
evidence that the `.so` builds.** The same distinction applies to every lane
whose identity is certified: the card and the wheel are different artifacts.

**WHAT IS PROVEN AND WHAT IS NOT.** These three COMPILE and produce a `.so`.
They are NOT launch-verified, because `build_arima.sh` and `build_training.sh`
both set `MOJOLEARN_SKIP_BUILD_GATE=1` for any non-fast mode -- the
build-time smoke gate imports the FAST package, so it cannot run against an
upper-tier artifact by construction. So the claim here is exactly "the release
matrix has no unbuildable cell", and NOT "the upper-tier arima and training
binaries have been launched". `python/mojolearn/tests/test_training_surface.py`
does launch the IDENTICAL training binary and passes 69 checks, which covers
one of the three; arima-identical, arima-deterministic and
training-deterministic are compiled and unlaunched.

**OWED**: a launch of each of those three, which needs a smoke path that
imports the tier under test rather than the fast package. That is the gap in
the build scripts, not in these artifacts.

### The wheel smoke launches 5 of the 12 bindings, MEASURED

Chasing the line above turned up a larger hole and it is not tier-specific.
`verify_wheel.sh` installs the wheel into a clean venv under every interpreter
and loops `packaging/macos/smoke.py` over all three tiers, which is the right
shape. But `smoke.py` exercises nine ESTIMATORS that between them reach only
FIVE BINDINGS. Measured 2026-09-01 by wrapping `_backend.binding` and
recording every name it was asked for, rather than by reading estimator names
and inferring:

    LOADED (5)       _mojolearn, _mojolearn_estimators, _mojolearn_gbdt,
                     _mojolearn_rf, _mojolearn_trees
    NEVER LOADED (7) _mojolearn_arima, _mojolearn_linalg, _mojolearn_metrics,
                     _mojolearn_solver, _mojolearn_svm, _mojolearn_training,
                     _mojolearn_tsa

**ALL TWELVE ARE USER-REACHABLE**, so this is untested shipped surface and not
dead weight: `_arima_impl.py`, `_linalg_impl.py`, `_svm_impl.py`,
`_training_impl.py` and `_tsa_impl.py` each own one, `_hierarchy_impl.py:291`
calls `_mojolearn_solver.linkage_fit`, and `_metrics_impl.py` loads
`_mojolearn_metrics` through a loader of its own.

Across three tiers that is **21 of 36 shipped artifacts that wheel
verification never launches**. A binding among those seven could fail to
import, or import and die at its first kernel launch -- the exact
`MACOSX_DEPLOYMENT_TARGET` failure this tree has already shipped once -- and
`verify_wheel.sh` would pass the wheel.

PCA and TruncatedSVD look like they should cover `_mojolearn_linalg` and do
not: `decomposition.py:155` and `:382` both bind `_mojolearn_estimators`.
`LinearRegression` likewise does not reach `_mojolearn_solver`. Guessing
coverage from estimator names gives the wrong answer, which is why the number
above was measured.

TWO INSTRUMENT CAVEATS, stated so the number is not over-read. The probe
watches `_backend.binding`, so (a) it cannot see `_metrics_impl.py`, which
deliberately bypasses that function with its own loader and carries a
standing note that the workaround should be deleted; and (b) a binding loaded
at import time before the wrapper was installed would be missed, though all
five recorded loads happened at fit time.

**OWED**: extend `smoke.py` to touch at least one entry point per binding, so
that "the wheel verifies" means all twelve. That is the single highest-value
item in the release lane and it is cheap -- one fit or one call each.

### `packaging/macos/smoke_estimators.py` IS NOT WIRED IN

It is a strict subset of `smoke.py` -- PCA, TruncatedSVD, DBSCAN,
LinearRegression -- and neither `verify_wheel.sh` nor
`build_release_wheel.sh` references it. It sits in the packaging directory
looking like coverage and contributes none. Either wire it or delete it; do
not count it.

**WHY THIS MATTERED ENOUGH TO CHECK.** 0.3.0 shipped broken because a build
path was never exercised before publishing (`docs/LINUX_WHEEL.md`); the
standing memory is `verify-on-a-box-that-did-not-build-it`. An unattempted
cell in the release matrix is that shape one step earlier, and it fails during
a release cut rather than before one.

## GaussianProcessRegressor: EXPOSED 2026-09-01 (was: NOT reachable, and NOT for the usual reason)

Every other entry on this page is absent because it lacks a binding. This one
had everything it needed at the lane level and was held back deliberately;
the history of that withholding is kept below because it is the record.

THE STATED REASON WAS WRONG AND IS WITHDRAWN, 2026-09-01. This paragraph read
"its IDENTICAL card diverges Apple against AMD on 8 of 3494 stages, starting
at `gp.kernel`". Those eight lines all sit inside ONE block, and that block is
the SABOTAGED half of the `GP_SAB_STD_EXP` clean-then-sabotaged pair -- an arm
whose entire purpose is that a device `exp` is a vendor choice in its last
bit. The shipped path is byte-identical across the other 3,486 lines. The
cards were diffed, which no leg had done, and then the diff was read without
noticing that the sweep writes sabotaged fits into the same card.

SUPERSEDED, LATER THE SAME DAY: IT IS REACHABLE. The paragraph that stood
here read "It remains NOT reachable, because exposing an estimator is
Andrew's call and not a consequence of correcting a misreading." Andrew
delegated the call and the orchestrator took it: with the sole blocker
withdrawn at `9835094e`, the withholding text contradicted the evidence, so
the surface was written -- `python/mojolearn/_gp_impl.py`
(`GaussianProcessRegressor` plus the `RBF`, `Matern`, `ConstantKernel` and
`WhiteKernel` spec classes), the `_NOT_YET` entry deleted the way its three
predecessors were, `python/mojolearn/tests/test_gp_surface.py` as the
surface gate, and `_mojolearn_gp` registered in `_backend.py`'s `_MODULES`
and `_build_script` (DEVIATION 869's both-places rule).

WHAT THIS ENTRY STILL LACKS, unlike every sibling above it: the binding
itself. `bindings/_mojolearn_gp.mojo` and `bindings/build_gp.sh` do not
exist yet -- the surface resolves its binding on first use and raises BY
NAME with the build command until they do, which is `_backend.py`'s designed
partial-build state. Also still owed: the regenerated card showing the new
`sabotage=` header field naming that block, on Apple and then on AMD. See
`bench/results/e1/GP_CROSS_VENDOR_DIVERGENCE.md` (corrected in place) and
`_gp_impl.py`'s header, which carries the history the `_NOT_YET` entry used
to.
