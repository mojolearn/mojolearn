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

Run (the extension is built for the pkg/gbmbench python):

    PYTHONPATH=python pixi run -e gbmbench python extratrees/tools/wrapper_reach_check.py
"""
import hashlib
import math
import sys

import numpy as np

from mojolearn.extratrees import ExtraTreesClassifier, ExtraTreesRegressor

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


def main():
    # Hashed values, not a uniform grid: the standing rule that uniform data
    # hides a permutation applies to a column subset as much as to a row one.
    rng = np.random.default_rng(458)
    X = rng.random((N_ROWS, N_FEATURES), dtype=np.float32)
    y_reg = (X @ rng.integers(-5, 6, N_FEATURES).astype(np.float32)).astype(
        np.float32
    )
    y_clf = (y_reg > np.median(y_reg)).astype(np.int64)
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
    if failed:
        print("wrapper_reach_check: FAIL")
        return 1
    print("wrapper_reach_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
