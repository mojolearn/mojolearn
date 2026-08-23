"""The Python wrapper's `max_features` REACHES the forest, for BOTH estimators.

DEVIATION 458's gate. `bindings/_mojolearn_trees.mojo` once applied
`for_regression()` AFTER reading the 21 params, so its `max_features_spec =
ALL` default overwrote slots 8-9 and every `ExtraTreesRegressor` fit sampled
all columns whatever the caller asked; the classifier, which takes the
classifier defaults, honoured every form. No check in this lane crossed the
Python boundary, so the regression went unseen until the E2 matrix hashed
five forms to one forest. This check crosses the boundary and gates REACH,
not output (the standing rule): the forests for max_features 1.0 / 0.5 /
'sqrt' / 3 must be pairwise DIFFERENT and `max_features_` must equal the
count sklearn would resolve, on the regressor and the classifier, on the
device arm and the host arm. An identical pair is a path that ran and was
not gated.

DEVIATIONS 459 / 460 (2026-08-23) add the second table: `criterion` and
`bootstrap` / `max_samples` must REACH the forest from the Python side.
entropy must build a different forest than gini (a 4-class target -- on a
binary separable target the two criteria pick the same feature at every
node, measured, so that fixture cannot see the switch), `log_loss` must be
entropy's alias (same digest), bootstrap=True must differ from False and
repeat at one seed, max_samples must move the forest and be reported back
as `_n_samples_bootstrap`. And each is SABOTAGED from this side of the
boundary: the params list is rebuilt with the criterion slot forced to
gini's code / the bootstrap slot forced to 0 / the max_samples slot forced
to 0, and the digest must COLLAPSE to the unsabotaged one -- which proves it
is that slot, and nothing else on the list, that carries the knob.

Run (the extension is built for the pkg/gbmbench python):

    PYTHONPATH=python pixi run -e gbmbench python extratrees/tools/wrapper_reach_check.py
"""
import hashlib
import math
import sys

import numpy as np

try:
    from mojolearn.extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    from mojolearn import extratrees as _et_mod
except ImportError as exc:
    # the package imports every binding (cluster, trees, rf, gbdt); on a
    # fresh clone or a remote box before phase 3 of tools/e1_bootstrap.sh
    # builds them this check has nothing to run against. SKIP, by name --
    # a missing binary is not a reach failure (E2 Mac reference run,
    # 2026-08-23, read as a PHASE2-FINDING until this line existed)
    print(f"SKIP wrapper_reach_check: mojolearn does not import here ({exc})")
    sys.exit(0)

N_ROWS, N_FEATURES = 4096, 16
FORMS = [1.0, 0.5, "sqrt", 3]


def resolved(mf, n):
    """sklearn's resolution of the four forms above."""
    if mf == "sqrt":
        return max(1, int(math.sqrt(n)))
    if isinstance(mf, int):
        return mf
    return max(1, int(mf * n))


def forest_digest(m):
    h = hashlib.sha256()
    for a in (m._offsets, m._colid, m._quesval, m._left_child, m._leaves):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _fit_with_slots(cls, X, y, slot_overrides, **kw):
    """Fit through the wrapper with named params-list slots OVERRIDDEN after
    the wrapper built the list -- the sabotage arm. Slot numbers are
    `bindings/_mojolearn_trees.mojo::N_FIT_PARAMS`'s."""
    real = _et_mod._fit_params

    def sabotaged(*args):
        p = real(*args)
        for slot, value in slot_overrides.items():
            p[slot] = value
        return p

    _et_mod._fit_params = sabotaged
    try:
        return cls(**kw).fit(X, y)
    finally:
        _et_mod._fit_params = real


SLOT_BOOTSTRAP = 11
SLOT_MAX_SAMPLES = 18
SLOT_CRITERION = 21


def check_criterion_and_bootstrap(X, y_clf4, y_reg):
    """The DEVIATION 459 / 460 table, with one sabotage per knob."""
    failed = False
    kw = dict(n_estimators=4, max_depth=6, random_state=7)
    for device in ("gpu", "cpu"):
        gini = ExtraTreesClassifier(device=device, **kw).fit(X, y_clf4)
        ent = ExtraTreesClassifier(device=device, criterion="entropy",
                                   **kw).fit(X, y_clf4)
        ll = ExtraTreesClassifier(device=device, criterion="log_loss",
                                  **kw).fit(X, y_clf4)
        dg, de, dl = forest_digest(gini), forest_digest(ent), forest_digest(ll)
        if de == dg:
            print(f"FAIL clf/{device}: criterion='entropy' built the gini"
                  f" forest ({dg}); the criterion did not reach the scorer")
            failed = True
        if dl != de:
            print(f"FAIL clf/{device}: log_loss ({dl}) is not entropy's alias"
                  f" ({de})")
            failed = True
        # SABOTAGE: the wrapper says entropy, slot 21 says gini -> gini's
        # forest. That proves slot 21 is the carrier.
        sab = _fit_with_slots(ExtraTreesClassifier, X, y_clf4,
                              {SLOT_CRITERION: 0}, device=device,
                              criterion="entropy", **kw)
        if forest_digest(sab) != dg:
            print(f"FAIL clf/{device}: forcing slot {SLOT_CRITERION} to gini"
                  " did not reproduce the gini forest -- something other than"
                  " the slot carries the criterion")
            failed = True
        print(f"clf/{device}: gini={dg} entropy={de} log_loss={dl}"
              f" sabotaged-entropy={forest_digest(sab)}")

        boot = ExtraTreesClassifier(device=device, bootstrap=True,
                                    **kw).fit(X, y_clf4)
        boot2 = ExtraTreesClassifier(device=device, bootstrap=True,
                                     **kw).fit(X, y_clf4)
        half = ExtraTreesClassifier(device=device, bootstrap=True,
                                    max_samples=0.5, **kw).fit(X, y_clf4)
        db, db2, dh = forest_digest(boot), forest_digest(boot2), forest_digest(half)
        if db == dg:
            print(f"FAIL clf/{device}: bootstrap=True built the no-bootstrap"
                  f" forest ({dg})")
            failed = True
        if db != db2:
            print(f"FAIL clf/{device}: bootstrap=True at random_state=7 gave"
                  f" two forests ({db}, {db2})")
            failed = True
        if dh == db:
            print(f"FAIL clf/{device}: max_samples=0.5 built the max_samples"
                  f"=None forest ({db})")
            failed = True
        if half._n_samples_bootstrap != N_ROWS // 2:
            print(f"FAIL clf/{device}: max_samples=0.5 reported"
                  f" {half._n_samples_bootstrap}, want {N_ROWS // 2}")
            failed = True
        if boot._n_samples_bootstrap != N_ROWS:
            print(f"FAIL clf/{device}: max_samples=None reported"
                  f" {boot._n_samples_bootstrap}, want {N_ROWS}")
            failed = True
        if gini._n_samples_bootstrap is not None:
            print(f"FAIL clf/{device}: no bootstrap reported a sample count")
            failed = True
        # SABOTAGES: slot 11 forced to 0 under bootstrap=True -> the
        # no-bootstrap forest; slot 18 forced to 0 under max_samples=0.5 ->
        # the max_samples=None bootstrap forest.
        sab_b = _fit_with_slots(ExtraTreesClassifier, X, y_clf4,
                                {SLOT_BOOTSTRAP: 0}, device=device,
                                bootstrap=True, **kw)
        if forest_digest(sab_b) != dg:
            print(f"FAIL clf/{device}: forcing slot {SLOT_BOOTSTRAP} to 0 did"
                  " not reproduce the no-bootstrap forest")
            failed = True
        sab_m = _fit_with_slots(ExtraTreesClassifier, X, y_clf4,
                                {SLOT_MAX_SAMPLES: 0}, device=device,
                                bootstrap=True, max_samples=0.5, **kw)
        if forest_digest(sab_m) != db:
            print(f"FAIL clf/{device}: forcing slot {SLOT_MAX_SAMPLES} to 0"
                  " did not reproduce the max_samples=None forest")
            failed = True
        print(f"clf/{device}: bootstrap={db} again={db2} max_samples=0.5={dh}"
              f" (n={half._n_samples_bootstrap}) sabotaged-bootstrap="
              f"{forest_digest(sab_b)} sabotaged-max_samples="
              f"{forest_digest(sab_m)}")

        # the regressor's bootstrap, same three facts
        r = ExtraTreesRegressor(device=device, **kw).fit(X, y_reg)
        rb = ExtraTreesRegressor(device=device, bootstrap=True,
                                 **kw).fit(X, y_reg)
        rb2 = ExtraTreesRegressor(device=device, bootstrap=True,
                                  **kw).fit(X, y_reg)
        dr, drb, drb2 = forest_digest(r), forest_digest(rb), forest_digest(rb2)
        if drb == dr:
            print(f"FAIL reg/{device}: bootstrap=True built the no-bootstrap"
                  " forest")
            failed = True
        if drb != drb2:
            print(f"FAIL reg/{device}: bootstrap=True at random_state=7 gave"
                  " two forests")
            failed = True
        print(f"reg/{device}: plain={dr} bootstrap={drb} again={drb2}")
    # refusals that must survive: max_samples without bootstrap, oob_score
    for bad in (dict(max_samples=0.5), dict(oob_score=True)):
        try:
            ExtraTreesClassifier(**bad).fit(X, y_clf4)
        except Exception as exc:
            if list(bad)[0] not in str(exc):
                print(f"FAIL: refusal for {bad} does not name it: {exc}")
                failed = True
        else:
            print(f"FAIL: {bad} was accepted")
            failed = True
    return failed


def main():
    # Hashed values, not a uniform grid: the standing rule that uniform data
    # hides a permutation applies to a column subset as much as to a row one.
    rng = np.random.default_rng(458)
    X = rng.random((N_ROWS, N_FEATURES), dtype=np.float32)
    y_reg = (X @ rng.integers(-5, 6, N_FEATURES).astype(np.float32)).astype(
        np.float32
    )
    y_clf = (y_reg > np.median(y_reg)).astype(np.int64)
    # a 4-class target for the criterion table (DEVIATION 459)
    y_clf4 = np.digitize(y_reg, np.quantile(y_reg, [0.3, 0.6, 0.8]))
    kw = dict(n_estimators=4, max_depth=6, random_state=7)
    failed = False
    for name, cls, y, devices in (
        ("regressor", ExtraTreesRegressor, y_reg, ("gpu", "cpu")),
        ("classifier", ExtraTreesClassifier, y_clf, ("gpu",)),
    ):
        for device in devices:
            digests = {}
            for mf in FORMS:
                m = cls(device=device, max_features=mf, **kw).fit(X, y)
                want = resolved(mf, N_FEATURES)
                if m.max_features_ != want:
                    print(
                        f"FAIL {name}/{device} max_features={mf!r}:"
                        f" max_features_ is {m.max_features_}, want {want}"
                    )
                    failed = True
                digests[mf] = forest_digest(m)
            seen = {}
            for mf, d in digests.items():
                if d in seen:
                    print(
                        f"FAIL {name}/{device}: max_features={mf!r} and"
                        f" {seen[d]!r} built the SAME forest ({d}); the knob"
                        " did not reach the sampler"
                    )
                    failed = True
                seen[d] = mf
            print(
                f"{name}/{device}: "
                + " ".join(f"{mf!r}={d}" for mf, d in digests.items())
            )
    if check_criterion_and_bootstrap(X, y_clf4, y_reg):
        failed = True
    if failed:
        print("wrapper_reach_check: FAIL")
        return 1
    print("wrapper_reach_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
