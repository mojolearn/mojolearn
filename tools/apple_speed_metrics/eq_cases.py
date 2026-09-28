# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane metrics-apple2: one digest per case of the metrics family's public
outputs, at many shapes. The A/B job (bench/x_metrics_apple_ab.sh) runs this
in the base tree and in the head tree and requires every line equal: the
IDENTICAL promise (bits never move) checked beyond the board's 29 cases.
Lines: `EQ <case> <digest>`."""
import hashlib
import os
import sys

import numpy as np

# --tree <dir>: the checkout whose python/ package is measured (the A/B job
# runs this one file against the base and the head trees)
_tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_tree, "python"))


def dig(v):
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, (list, tuple)):
            h.update(b"[%d" % len(x))
            for e in x:
                walk(e)
            return
        if isinstance(x, float):
            h.update(np.float64(x).tobytes())
            return
        if isinstance(x, (int, str, type(None))):
            h.update(repr(x).encode())
            return
        a = np.asarray(x.tolist() if hasattr(x, "tolist") else x)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    walk(v)
    return h.hexdigest()[:16]


def splits(cv, X, y=None):
    return [(tr, te) for tr, te in cv.split(X, y)]


def main():
    import mojolearn.metrics as M
    import mojolearn.model_selection as S
    rs = np.random.RandomState(7)
    cases = []
    for n in (2, 3, 5, 17, 1000, 4099, 70001):
        X = np.empty((n, 1))
        for k in (2, 3, 5, 7):
            if k > n:
                continue
            for seed in (0, 11):
                cases.append(("kfold_sh_%d_%d_%d" % (n, k, seed), lambda X=X, k=k, s=seed: splits(S.KFold(k, shuffle=True, random_state=s), X)))
            cases.append(("kfold_%d_%d" % (n, k), lambda X=X, k=k: splits(S.KFold(k), X)))
            for m in (2, 3, 6):
                y = rs.randint(0, m, n)
                if np.bincount(y, minlength=m).min() < 1:
                    continue
                for sh in (False, True):
                    cases.append(("skfold_%d_%d_%d_%d" % (n, k, m, sh),
                                  lambda X=X, y=y, k=k, sh=sh: splits(S.StratifiedKFold(k, shuffle=sh, random_state=3 if sh else None), X, y)))
                ys = np.array(["c%d" % v for v in y])
                cases.append(("skfold_str_%d_%d_%d" % (n, k, m), lambda X=X, y=ys, k=k: splits(S.StratifiedKFold(k, shuffle=True, random_state=5), X, y)))
        for ts in (0.1, 0.25, 0.5):
            if n >= 5:
                cases.append(("ss_%d_%s" % (n, ts), lambda X=X, ts=ts: splits(S.ShuffleSplit(4, test_size=ts, random_state=9), X)))
                v = rs.rand(n).astype(np.float32)
                cases.append(("tts_%d_%s" % (n, ts), lambda v=v, ts=ts: S.train_test_split(v, list(range(len(v))), test_size=ts, random_state=1)))
        if n >= 5:
            cases.append(("rkfold_%d" % n, lambda X=X: splits(S.RepeatedKFold(n_splits=2, n_repeats=3, random_state=4), X)))
    # the binary and multiclass curves (ties, few distinct scores, weights)
    for n in (3, 50, 1000, 20011, 200003):
        for distinct in (4, 300, 0):
            s = rs.rand(n)
            if distinct:
                s = np.floor(s * distinct) / distinct
            s = s.astype(np.float32)
            y = (rs.rand(n) < 0.35).astype(np.int64)
            if y.min() == y.max():
                y[0] = 1 - y[0]
            w = (rs.randint(0, 4, n) * 0.25).astype(np.float32)
            tag = "%d_%d" % (n, distinct)
            cases.append(("roc_" + tag, lambda y=y, s=s: M.roc_curve(y, s)))
            cases.append(("roc_nodrop_" + tag, lambda y=y, s=s: M.roc_curve(y, s, drop_intermediate=False)))
            cases.append(("roc_w_" + tag, lambda y=y, s=s, w=w: M.roc_curve(y, s, sample_weight=w)))
            cases.append(("auc_" + tag, lambda y=y, s=s: M.roc_auc_score(y, s)))
            cases.append(("auc_w_" + tag, lambda y=y, s=s, w=w: M.roc_auc_score(y, s, sample_weight=w)))
            cases.append(("auc_mf_" + tag, lambda y=y, s=s: M.roc_auc_score(y, s, max_fpr=0.3)))
            cases.append(("ap_" + tag, lambda y=y, s=s: M.average_precision_score(y, s)))
            cases.append(("ap_w_" + tag, lambda y=y, s=s, w=w: M.average_precision_score(y, s, sample_weight=w)))
            cases.append(("prc_" + tag, lambda y=y, s=s: M.precision_recall_curve(y, s)))
            cases.append(("det_" + tag, lambda y=y, s=s: M.det_curve(y, s)))
            if n >= 50:
                yc = rs.randint(0, 4, n)
                P = rs.rand(n, 4).astype(np.float64)
                if distinct:
                    P = np.floor(P * distinct) + 1
                P = (P / P.sum(axis=1, keepdims=True)).astype(np.float32)
                for avg in ("macro", "weighted"):
                    cases.append(("ovr_%s_" % avg + tag, lambda yc=yc, P=P, a=avg: M.roc_auc_score(yc, P, multi_class="ovr", average=a)))
                cases.append(("ovr_w_" + tag, lambda yc=yc, P=P, w=w: M.roc_auc_score(yc, P, multi_class="ovr", sample_weight=w)))
                Pb = P.copy()
                Pb[n // 2, 0] += np.float32(2e-5)
                cases.append(("ovr_badsum_" + tag, lambda yc=yc, P=Pb: M.roc_auc_score(yc, P, multi_class="ovr")))
                Pc = P.copy()
                Pc[n // 3, 1] += np.float32(5e-6)
                cases.append(("ovr_nearsum_" + tag, lambda yc=yc, P=Pc: M.roc_auc_score(yc, P, multi_class="ovr")))
                cases.append(("ovo_" + tag, lambda yc=yc, P=P: M.roc_auc_score(yc, P, multi_class="ovo")))
                cases.append(("ap_ovr_" + tag, lambda yc=yc, P=P: M.average_precision_score(np.eye(4)[yc].astype(np.int64), P)))
    # the expected MI (adjusted_mutual_info_score): balanced, skewed, many classes
    for n in (7, 100, 5000, 100003, 1000000):
        for ka, kb in ((2, 2), (5, 5), (3, 20), (40, 7)):
            a = rs.randint(0, ka, n)
            b = (a + rs.randint(0, 2, n) * rs.randint(0, kb, n)) % kb
            sk = (rs.rand(n) ** 3 * ka).astype(np.int64)
            for avg in ("arithmetic", "geometric", "max"):
                cases.append(("ami_%d_%d_%d_%s" % (n, ka, kb, avg), lambda a=a, b=b, m=avg: M.adjusted_mutual_info_score(a, b, average_method=m)))
            cases.append(("ami_skew_%d_%d_%d" % (n, ka, kb), lambda a=sk, b=b: M.adjusted_mutual_info_score(a, b)))
    for name, fn in cases:
        try:
            print("EQ %s %s" % (name, dig(fn())), flush=True)
        except Exception as e:
            print("EQ %s RAISED %s %s" % (name, type(e).__name__, str(e)[:80]), flush=True)


if __name__ == "__main__":
    main()
