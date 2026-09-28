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
    import warnings
    warnings.simplefilter("ignore")
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
        if n >= 17:
            yr = rs.randint(0, 3, n)
            yr[:2] = 7                     # a class rarer than n_splits (the warning path)
            for sh in (False, True):
                cases.append(("skfold_rare_%d_%d" % (n, sh), lambda X=X, y=yr, sh=sh: splits(S.StratifiedKFold(5, shuffle=sh, random_state=2 if sh else None), X, y)))
            cases.append(("rskfold_%d" % n, lambda X=X, y=yr: splits(S.RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=8), X, y)))
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
                if n <= 20011:
                    cases.append(("ovr_micro_" + tag, lambda yc=yc, P=P: M.roc_auc_score(np.eye(4)[yc].astype(np.int64), P, average="micro")))
                    cases.append(("ovr_none_" + tag, lambda yc=yc, P=P: M.roc_auc_score(yc, P, multi_class="ovr", average=None)))
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
    # lane metrics-apple3: score rows of very different magnitudes (their
    # binary64 row sum is not exact, so the row-sum check takes fsum itself),
    # rows that are exactly representable, and sizes on both sides of the
    # host-task threshold; NMI / AMI with many classes (many short cells)
    for n in (300, 40000, 200003, 1000000):
        yc = rs.randint(0, 5, n)
        Z = rs.randn(n, 5) * 30.0
        Z = Z - Z.max(axis=1, keepdims=True)
        W = np.exp(Z)
        W = (W / W.sum(axis=1, keepdims=True)).astype(np.float32)
        cases.append(("ovr_wide_%d" % n, lambda yc=yc, P=W: M.roc_auc_score(yc, P, multi_class="ovr")))
        cases.append(("logloss_wide_%d" % n, lambda yc=yc, P=W: M.log_loss(yc, P)))
        Q = (rs.randint(1, 9, (n, 5)) / 8.0)
        Q = (Q / Q.sum(axis=1, keepdims=True)).astype(np.float32)
        cases.append(("ovr_exact_%d" % n, lambda yc=yc, P=Q: M.roc_auc_score(yc, P, multi_class="ovr")))
        cases.append(("topk_exact_%d" % n, lambda yc=yc, P=Q: M.top_k_accuracy_score(yc, P, k=2)))
    # one-vs-one and the micro averages (native row selection, strided flags)
    for n in (300, 5000, 120001):
        for k in (3, 6):
            yc = rs.randint(0, k, n)
            P = rs.rand(n, k) + 0.05
            P = (P / P.sum(axis=1, keepdims=True)).astype(np.float32)
            w = (rs.randint(1, 5, n) * 0.25).astype(np.float32)
            t = "%d_%d" % (n, k)
            oh = np.eye(k)[yc].astype(np.int64)
            cases.append(("ovo3_" + t, lambda yc=yc, P=P: M.roc_auc_score(yc, P, multi_class="ovo")))
            cases.append(("ovo3_weighted_" + t, lambda yc=yc, P=P: M.roc_auc_score(yc, P, multi_class="ovo", average="weighted")))
            cases.append(("ovo3_labels_" + t, lambda yc=yc, P=P, k=k: M.roc_auc_score(yc, P, multi_class="ovo", labels=list(range(k)))))
            cases.append(("ovo3_str_" + t, lambda yc=yc, P=P: M.roc_auc_score(np.array(["c%d" % v for v in yc]), P, multi_class="ovo")))
            cases.append(("ovr3_micro_" + t, lambda oh=oh, P=P: M.roc_auc_score(oh, P, average="micro")))
            cases.append(("ovr3_micro_mc_" + t, lambda yc=yc, P=P: M.roc_auc_score(yc, P, multi_class="ovr", average="micro")))
            cases.append(("ovr3_micro_w_" + t, lambda yc=yc, P=P, w=w: M.roc_auc_score(yc, P, multi_class="ovr", average="micro", sample_weight=w)))
            cases.append(("ovr3_macro_w_" + t, lambda yc=yc, P=P, w=w: M.roc_auc_score(yc, P, multi_class="ovr", sample_weight=w)))
            cases.append(("ap3_micro_" + t, lambda yc=yc, P=P: M.average_precision_score(yc, P, average="micro")))
            cases.append(("ap3_micro_w_" + t, lambda yc=yc, P=P, w=w: M.average_precision_score(yc, P, average="micro", sample_weight=w)))
            cases.append(("ap3_macro_" + t, lambda yc=yc, P=P: M.average_precision_score(yc, P)))
    # classification counts through the small-span label encoder: negative
    # labels, labels far from zero, a span too wide for it, uint8 and int32
    for n in (300, 70001, 300000):
        a = rs.randint(0, 7, n)
        b = (a + (rs.rand(n) < 0.3) * rs.randint(0, 7, n)) % 7
        for name, f in (("neg", lambda v: v - 3), ("far", lambda v: v * 9000 + 10 ** 9), ("wide", lambda v: v * 100000),
                        ("u8", lambda v: v.astype(np.uint8)), ("i32", lambda v: v.astype(np.int32))):
            ya, yb = f(a), f(b)
            t = "%s_%d" % (name, n)
            cases.append(("bacc3_" + t, lambda ya=ya, yb=yb: M.balanced_accuracy_score(ya, yb)))
            cases.append(("mcc3_" + t, lambda ya=ya, yb=yb: M.matthews_corrcoef(ya, yb)))
            cases.append(("prfs3_" + t, lambda ya=ya, yb=yb: M.precision_recall_fscore_support(ya, yb, average=None)))
            cases.append(("ari3_" + t, lambda ya=ya, yb=yb: M.adjusted_rand_score(ya, yb)))
        X = np.empty((n, 1))
        for name, f in (("neg", lambda v: v - 3), ("far", lambda v: v * 9000 + 10 ** 9), ("i32", lambda v: v.astype(np.int32))):
            ya = f(a[::-1].copy())
            cases.append(("skfold3_%s_%d" % (name, n), lambda X=X, y=ya: splits(S.StratifiedKFold(5, shuffle=True, random_state=4), X, y)))
            cases.append(("skfold3_plain_%s_%d" % (name, n), lambda X=X, y=ya: splits(S.StratifiedKFold(4), X, y)))
    for n in (70001, 400000):
        for ka, kb in ((60, 60), (200, 3), (2, 500)):
            a = rs.randint(0, ka, n)
            b = (a * 7 + rs.randint(0, 3, n)) % kb
            cases.append(("ami3_%d_%d_%d" % (n, ka, kb), lambda a=a, b=b: M.adjusted_mutual_info_score(a, b)))
            cases.append(("nmi3_%d_%d_%d" % (n, ka, kb), lambda a=a, b=b: M.normalized_mutual_info_score(a, b)))
    for name, fn in cases:
        try:
            print("EQ %s %s" % (name, dig(fn())), flush=True)
        except Exception as e:
            print("EQ %s RAISED %s %s" % (name, type(e).__name__, str(e)[:80]), flush=True)


if __name__ == "__main__":
    main()
