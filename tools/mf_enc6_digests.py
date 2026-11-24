"""Before/after digests of the OrdinalEncoder / OneHotEncoder member forms lane mf-enc6 added (2026-11-23), host
column. One line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a
refusal is digested by its exception type and words; a warning by its category and words. Base: origin/lane/member-fits-6
b613b99d34 (each form below made `_ordinal_admit` / `_onehot_admit` answer False and the class's own door ran it),
head lane/mf-enc6. Test tool, never product runtime.

Rows covered (see ~/mojolearn-evidence/briefs-20261122/mf-enc6-work/rows.tsv): given categories, the infrequent
grouping (min_frequency, max_categories), a drop given per column (with and without the grouping), handle_unknown
'warn' (and 'ignore' / 'infrequent_if_exist' with a drop), the FAST packed category route (the `fast_*` units, on an
install with a fast set; otherwise they digest the skip words on both sides).

Each configuration runs the estimator's own fit (`door_*`) and its GridSearchCV member refit (`member_*`: one given
split, train = every row, a one-point grid): categories_, infrequent_categories_, drop_idx_, the private
`_grouping` / `_infrequent` / `_drop_after` / `_missing`, transform and inverse_transform where defined.

EXPECTED: every `door_*` line equal base vs head, except the door's own words for an infrequent-drop refusal
("Unable to drop category 4.0 ..."; the category now printed by Mojo's float formatting, the same text for these
values) and `_missing`, which the door's general OneHotEncoder fit now also packs (head only; absent on base). Every
`member_*` line on head equal to its `door_*` line (on base the member line equals the door line because the door
ran it). The `warn_*` lines: the warnings of transform on unknown rows, equal door vs member, base vs head. No bits
change: the same programs on the same routes. Configurations are independent: one process each across the host
cores.

    python tools/mf_enc6_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np


def _digest(value):
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        h = hashlib.sha256()
        for item in value:
            h.update(b"None" if item is None else _digest(item).encode())
        return h.hexdigest()[:24]
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(n=60, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=n, p=[0.4, 0.3, 0.2, 0.06, 0.04])
    b = rng.choice([10.0, 20.0, 30.0], size=n, p=[0.6, 0.35, 0.05])
    c = rng.choice([-1.0, 5.0], size=n)
    return np.stack([a, b, c], axis=1).astype(np.float32)


GIVEN = [[0.0, 1.0, 2.0, 3.0, 4.0, 7.0], [10.0, 20.0, 30.0], [-1.0, 5.0]]

CONFIGS = {
    "ord_given": ("OrdinalEncoder", {"categories": GIVEN}),
    "ord_given_unknown": ("OrdinalEncoder", {"categories": [[0.0, 1.0], GIVEN[1], GIVEN[2]]}),
    "ord_given_encoded": ("OrdinalEncoder", {"categories": [[0.0, 1.0], GIVEN[1], GIVEN[2]],
                                             "handle_unknown": "use_encoded_value", "unknown_value": -1}),
    "ord_mf4": ("OrdinalEncoder", {"min_frequency": 4}),
    "ord_mf_frac": ("OrdinalEncoder", {"min_frequency": 0.1}),
    "ord_maxc2": ("OrdinalEncoder", {"max_categories": 2}),
    "oh_given": ("OneHotEncoder", {"categories": GIVEN}),
    "oh_mf4": ("OneHotEncoder", {"min_frequency": 4}),
    "oh_mf4_infrequent": ("OneHotEncoder", {"min_frequency": 4, "handle_unknown": "infrequent_if_exist"}),
    "oh_drop_list": ("OneHotEncoder", {"drop": [1.0, 20.0, -1.0]}),
    "oh_drop_list_mf4": ("OneHotEncoder", {"drop": [0.0, 10.0, 5.0], "min_frequency": 4}),
    "oh_drop_first_mf4": ("OneHotEncoder", {"drop": "first", "min_frequency": 4}),
    "oh_drop_missing": ("OneHotEncoder", {"drop": [9.0, 10.0, 5.0]}),
    "oh_drop_infrequent": ("OneHotEncoder", {"drop": [4.0, 10.0, 5.0], "min_frequency": 4}),
    "oh_warn": ("OneHotEncoder", {"handle_unknown": "warn"}),
    "oh_ignore_first": ("OneHotEncoder", {"handle_unknown": "ignore", "drop": "first"}),
    "fast_ord": ("OrdinalEncoder", {}),
    "fast_oh": ("OneHotEncoder", {}),
}

FIELDS = ("categories_", "infrequent_categories_", "drop_idx_", "_grouping", "_infrequent", "_drop_after", "_missing")


def _member(est, X):
    from mojolearn.model_selection import GridSearchCV
    every = np.arange(len(X))
    params = est.get_params()
    name = sorted(params)[0]
    # scoring=None needs the class's own `score` (scikit-learn 1.9 metrics/_scorer.py `check_scoring`, a TypeError
    # before any fit; providers/python/member_ensemble_boundary_door.mojo `_no_score_refusal`). A class without one
    # is searched by a scorer whose response it lacks: every score is error_score (nan) and the best is refitted
    # through the member, as scikit-learn does (lane w3f2-cv).
    scoring = None if getattr(est, "score", None) is not None else "r2"
    search = GridSearchCV(est, {name: [params[name]]}, cv=[(every, every)], scoring=scoring, refit=True)
    search.fit(X)
    return search.best_estimator_


def _side(name, side):
    import mojolearn
    cls, kwargs = CONFIGS[name]
    if cls == "OneHotEncoder":
        kwargs = dict(kwargs, sparse_output=False)
    X = _data()
    est = getattr(mojolearn, cls)(**kwargs)
    fitted = est.fit(X) if side == "door" else _member(est, X)
    out = []
    for field in FIELDS:
        if hasattr(fitted, field):
            out.append((field, getattr(fitted, field)))
    xt = np.asarray(fitted.transform(X))
    out.append(("transform", xt))
    try:
        out.append(("inverse", fitted.inverse_transform(xt)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append(("inverse_raised", f"{type(e).__name__}: {e}"))
    unknown = X[:4].copy()
    unknown[:, 0] = 99.0
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            out.append(("unknown", fitted.transform(unknown)))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append(("unknown_raised", f"{type(e).__name__}: {e}"))
    out.append(("warn", [f"{w.category.__name__}: {w.message}" for w in seen]))
    return out


def _unit(name):
    lines = []
    previous = None
    for side in ("door", "member"):
        try:
            if name.startswith("fast_"):
                import mojolearn
                previous = mojolearn.set_numeric_mode("fast")
            for suffix, value in _side(name, side):
                text = value if isinstance(value, str) else (
                    "|".join(value) if suffix == "warn" else _digest(value))
                lines.append((f"{side}_{name}_{suffix}", text))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            lines.append((f"{side}_{name}_raised", f"{type(e).__name__}: {e}"))
        finally:
            if previous is not None:
                import mojolearn
                mojolearn.set_numeric_mode(previous)
                previous = None
    return lines


def main():
    units = list(CONFIGS)
    with ProcessPoolExecutor(max_workers=min(len(units), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, units):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
