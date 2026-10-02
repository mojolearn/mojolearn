#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""lane/py-misc-msel: before == after equality and before/after timing of the
model_selection routes moved onto the core helpers. The before arm is the
definition in the same build (MOJOLEARN_MSEL_BEFORE=1, read per call).

    check.py equal            every split's rows, scorer inputs, permutation
                              scores and check_cv decisions, byte for byte
    check.py time [--n N]     each changed path, before then after, one process
"""
import argparse, hashlib, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np  # noqa: E402
import mojolearn as ml  # noqa: E402
from mojolearn import model_selection as ms  # noqa: E402
from mojolearn._array import Array  # noqa: E402

BEFORE = "MOJOLEARN_MSEL_BEFORE"


def arm(before):
    if before:
        os.environ[BEFORE] = "1"
    else:
        os.environ.pop(BEFORE, None)


def rows_digest(pairs):
    h = hashlib.sha256()
    count = 0
    for tr, te in pairs:
        for side in (tr, te):
            a = np.asarray(side)
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        count += 1
    return h.hexdigest()[:16], count


class HashEstimator:
    """A classifier whose score is a digest of what it was fit on and scored
    on: any change in a fold's rows, their order, or y's words moves it."""
    _estimator_type = "classifier"

    def __init__(self, salt=0):
        self.salt = salt

    def get_params(self, deep=False):
        return {"salt": self.salt}

    def set_params(self, **p):
        self.salt = p.get("salt", self.salt)
        return self

    def fit(self, X, y):
        a = np.asarray(y)
        self.seen_ = hashlib.blake2b(str(a.dtype).encode() + a.tobytes() + np.asarray(X).tobytes()[:4096],
                                     digest_size=8).digest()
        return self

    def score(self, X, y):
        a = np.asarray(y)
        d = hashlib.blake2b(self.seen_ + a.tobytes(), digest_size=6).digest()
        return int.from_bytes(d, "little") / float(1 << 48)


class Proba:
    def __init__(self, P):
        self.P = P

    def predict_proba(self, X):
        return self.P


def splitters(n, groups_m):
    return [
        ("kfold", ms.KFold(5)), ("kfold7", ms.KFold(7)),
        ("gkf", ms.GroupKFold(4)), ("gkf_shuffle", ms.GroupKFold(4, shuffle=True, random_state=5)),
        ("sgkf", ms.StratifiedGroupKFold(3)), ("sgkf_shuffle", ms.StratifiedGroupKFold(3, shuffle=True, random_state=2)),
        ("sss", ms.StratifiedShuffleSplit(4, test_size=0.3, random_state=8)),
        ("sss_int", ms.StratifiedShuffleSplit(3, test_size=37, train_size=101, random_state=1)),
        ("gss", ms.GroupShuffleSplit(3, test_size=0.3, random_state=9)),
        ("gss_both", ms.GroupShuffleSplit(2, test_size=3, train_size=5, random_state=4)),
        ("logo", ms.LeaveOneGroupOut()), ("lpgo", ms.LeavePGroupsOut(2)),
        ("rkf", ms.RepeatedKFold(n_splits=3, n_repeats=2, random_state=4)),
        ("predef", ms.PredefinedSplit((np.arange(n) % 5) - 1)),
        ("predef_sparse", ms.PredefinedSplit(np.where(np.arange(n) % 3 == 0, 7, np.where(np.arange(n) % 3 == 1, -1, 100)))),
    ]


def equal():
    bad = 0
    total = 0
    for n, gm in ((600, 23), (5000, 97), (300, 11)):
        X = np.random.default_rng(n).standard_normal((n, 4)).astype(np.float32)
        yv = (np.arange(n) * 7 % 3).astype(np.int32)
        g = ((np.arange(n) // 7) % gm).astype(np.int32)
        inputs = [("np_i32", yv, g), ("np_i64_list", yv.astype(np.int64), g.tolist()),
                  ("str_groups", [str(v) for v in yv], [f"g{v:03d}" for v in g]),
                  ("float_y", yv.astype(np.float64), g.astype(np.float64))]
        for tag, y, gg in inputs:
            for name, cv in splitters(n, gm):
                res = {}
                for before in (True, False):
                    arm(before)
                    try:
                        res[before] = rows_digest(cv.split(X, y, gg))
                    except Exception as e:  # the same error on both sides is equal
                        res[before] = ("ERR", type(e).__name__, str(e))
                total += 1
                if res[True] != res[False]:
                    bad += 1
                    print(f"DIFF n={n} {tag} {name}: before {res[True]} after {res[False]}")
        # iterable cv of numpy pairs
        pairs = [(np.arange(0, n // 2, dtype=d), np.arange(n // 2, n, dtype=d)) for d in (np.int64, np.int32)]
        res = {}
        for before in (True, False):
            arm(before)
            res[before] = rows_digest(ms.check_cv(pairs).split(X))
        total += 1
        if res[True] != res[False]:
            bad += 1
            print(f"DIFF n={n} iterable: {res}")
        # check_cv stratify decisions
        for yy in (yv, yv.astype(np.float32), yv.astype(np.float64) + 0.5, np.where(np.arange(n) == 3, np.inf, 1.0),
                   yv.astype(np.uint8)):
            got = {}
            for before in (True, False):
                arm(before)
                got[before] = type(ms.check_cv(5, Array.from_buffer(np.ascontiguousarray(yy)),
                                               classifier=True)).__name__
            total += 1
            if got[True] != got[False]:
                bad += 1
                print(f"DIFF check_cv {yy.dtype}: {got}")
        # scorer column
        P = np.random.default_rng(n + 1).random((n, 2))
        P[5, 1] = np.nan
        yb = (np.arange(n) % 2).astype(np.int32)
        for Pv in (P.astype(np.float32), P, np.asfortranarray(P.astype(np.float32))):
            for sname in ("roc_auc", "average_precision", "neg_brier_score", "neg_log_loss"):
                sc = ms.get_scorer(sname)
                vals = {}
                for before in (True, False):
                    arm(before)
                    try:
                        vals[before] = repr(sc(Proba(Pv), X, yb))
                    except Exception as e:
                        vals[before] = ("ERR", type(e).__name__, str(e))
                total += 1
                if vals[True] != vals[False]:
                    bad += 1
                    print(f"DIFF scorer {sname} {Pv.dtype} {Pv.flags['C_CONTIGUOUS']}: {vals}")
        # permutation_test_score
        for tag, y, gg, cv in (("int_cv5", yv, None, 5), ("kfold", yv, None, ms.KFold(4)),
                               ("kfold_shuf", yv.astype(np.float64), None, ms.KFold(3, shuffle=True, random_state=2)),
                               ("groups_gkf", yv, g, ms.GroupKFold(3)), ("groups_int", yv.tolist(), g.tolist(), 3),
                               ("groups_str", yv, [f"g{v:03d}" for v in g], ms.KFold(3)),
                               ("list_float", [float(v) for v in yv], None, 4)):
            out = {}
            for before in (True, False):
                arm(before)
                s0, perm, p = ms.permutation_test_score(HashEstimator(), X, y, groups=gg, cv=cv, n_permutations=7,
                                                        random_state=3)
                out[before] = (repr(s0), np.asarray(perm).tobytes().hex(), repr(p))
            total += 1
            if out[True] != out[False]:
                bad += 1
                print(f"DIFF permutation n={n} {tag}: before {out[True][0]} {out[True][2]} after {out[False][0]} {out[False][2]}")
        # with a real estimator, small
        est = ml.GaussianNB() if hasattr(ml, "GaussianNB") else None
        if est is not None and n <= 600:
            out = {}
            for before in (True, False):
                arm(before)
                s0, perm, p = ms.permutation_test_score(est, X, yv, groups=g, cv=ms.GroupKFold(3), n_permutations=3,
                                                        random_state=1, scoring="accuracy")
                out[before] = (repr(s0), np.asarray(perm).tobytes().hex(), repr(p))
            total += 1
            if out[True] != out[False]:
                bad += 1
                print(f"DIFF permutation GaussianNB n={n}: {out}")
    arm(False)
    print(f"EQUAL {'PASS' if bad == 0 else 'FAIL'}: {total - bad} of {total} cases byte-equal")
    return 1 if bad else 0


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def time_all(n):
    rng = np.random.default_rng(0)
    X = np.zeros((n, 2), np.float32)
    y = (rng.integers(0, 3, n)).astype(np.int32)
    g1000 = (np.arange(n) * 1000 // n).astype(np.int32)
    g30 = (np.arange(n) * 30 // n).astype(np.int32)
    g100 = (np.arange(n) * 100 // n).astype(np.int32)
    P = rng.random((n, 2)).astype(np.float32)
    yb = (np.arange(n) % 2).astype(np.int32)
    pairs = [(np.arange(0, n // 2, dtype=np.int64), np.arange(n // 2, n, dtype=np.int64)) for _ in range(5)]
    cases = [
        ("LeaveOneGroupOut 1000 groups", lambda: rows_digest(ms.LeaveOneGroupOut().split(X, y, g1000))),
        ("LeavePGroupsOut(2) 30 groups", lambda: rows_digest(ms.LeavePGroupsOut(2).split(X, y, g30))),
        ("GroupKFold(5) 1000 groups", lambda: rows_digest(ms.GroupKFold(5).split(X, y, g1000))),
        ("GroupKFold(5, shuffle) 1000 groups", lambda: rows_digest(ms.GroupKFold(5, shuffle=True, random_state=1).split(X, y, g1000))),
        ("StratifiedGroupKFold(5) 100 groups", lambda: rows_digest(ms.StratifiedGroupKFold(5).split(X, y, g100))),
        ("GroupShuffleSplit(10) 1000 groups", lambda: rows_digest(ms.GroupShuffleSplit(10, random_state=1).split(X, y, g1000))),
        ("StratifiedShuffleSplit(10)", lambda: rows_digest(ms.StratifiedShuffleSplit(10, random_state=1).split(X, y))),
        ("PredefinedSplit 5 folds", lambda: rows_digest(ms.PredefinedSplit(np.arange(n) % 5).split(X))),
        ("KFold(5) unshuffled", lambda: rows_digest(ms.KFold(5).split(X))),
        ("iterable cv 5 pairs", lambda: rows_digest(ms.check_cv(pairs).split(X))),
        ("check_cv stratify (float y)", lambda: type(ms.check_cv(5, Array.from_buffer(y.astype(np.float32)), classifier=True)).__name__),
        ("scorer roc_auc column (n,2) f4", lambda: ms.get_scorer("roc_auc")(Proba(P), X, yb)),
        ("permutation_test_score 10 perms, KFold(5)", lambda: ms.permutation_test_score(
            HashEstimator(), X, y, cv=ms.KFold(5), n_permutations=10, random_state=0)[1].tobytes()),
        ("permutation_test_score 10 perms, cv=5 (stratified)", lambda: ms.permutation_test_score(
            HashEstimator(), X, y, cv=5, n_permutations=10, random_state=0)[1].tobytes()),
        ("permutation_test_score 3 perms, 1000 groups, GroupKFold(5)", lambda: ms.permutation_test_score(
            HashEstimator(), X, y, groups=g1000, cv=ms.GroupKFold(5), n_permutations=3, random_state=0)[1].tobytes()),
    ]
    print(f"TIME n={n} vendor={os.environ.get('MOJOLEARN_VENDOR', 'gpu')} host={os.uname().nodename}")
    print(f"{'case':62s} {'before s':>10s} {'after s':>10s} {'x':>7s} same")
    for name, fn in cases:
        arm(False)
        fn()  # warm (bindings, programs)
        arm(True)
        tb, rb = timed(fn)
        arm(False)
        ta, ra = timed(fn)
        print(f"{name:62s} {tb:10.3f} {ta:10.3f} {tb / max(ta, 1e-9):7.1f} {'yes' if rb == ra else 'NO'}", flush=True)
    arm(False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("equal", "time"))
    ap.add_argument("--n", type=int, default=1_000_000)
    a = ap.parse_args()
    if a.cmd == "time" and (os.environ.get("MOJOLEARN_VENDOR", "").strip().lower() == "cpu"
                            or ml.vendor() == "cpu"):
        sys.exit("check.py time: refused on the host column: our CPU is never timed (Andrew, Oct 2 2026)")
    sys.exit(equal() if a.cmd == "equal" else time_all(a.n) or 0)
