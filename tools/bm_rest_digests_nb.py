#!/usr/bin/env python3
"""Output digests of the naive Bayes operations whose programs moved to typed builders (lane bm-rest-nb,
2026-11-23): GaussianNB and the discrete kinds' partial_fit, CategoricalNB fit and partial_fit. A testing tool,
never product execution.

One line per configuration, `<name> sha256=<digest of every output's bytes>` or `<name> error <message>`: the
fitted state (every attribute the operation publishes, in a fixed order) and predict_proba. Run it on two builds of
the CPU column (before and after a change) and diff the files: equal lines mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/bm_rest_digests_nb.py > out.txt

Every configuration runs on the default route and with MOJOLEARN_XPREP_BLOCKED=0 (the `_b0` lines); the variable is
read at each call, so it is set per configuration inside the worker.

Parallel plan: the configurations are independent fits, one per worker process (at most the allocated cores, at most
the configuration count).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

NB_ATTRS = ("class_count_", "theta_", "var_", "class_prior_", "epsilon_", "_const", "_raw_var", "feature_count_",
            "class_log_prior_", "feature_log_prob_", "_w", "_bias", "_cc", "_flp", "_cmax", "n_categories_",
            "category_count_", "classes_")


def _data(seed=0, n=600, d=5):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5)
    Xc = np.clip(np.floor(np.abs(X) * 2), 0, 4).astype(np.float32)
    w = (np.abs(X[:, 2]) + 0.5).astype(np.float32)
    return X, y, Xc, w


# name -> (estimator, data kind, keywords, plan). Plans: "fit"; "pf3" three batches (the first without class 2,
# the last weighted); "pf1" one partial_fit; "pfstr" three batches with string labels (the label walk);
# "pfgrow" CategoricalNB batches whose category range grows; "pfbad" a second batch with an unknown label;
# "neg" a negative category; "badmin" a min_categories of the wrong shape.
def _cases():
    c = {}
    for nm, kind in (("GaussianNB", "x"), ("MultinomialNB", "abs"), ("ComplementNB", "abs"),
                     ("BernoulliNB", "x"), ("CategoricalNB", "cat")):
        t = nm.lower()
        c[f"{t}_pf3"] = (nm, kind, {}, "pf3")
        c[f"{t}_pf1"] = (nm, kind, {}, "pf1")
        c[f"{t}_pfstr"] = (nm, kind, {}, "pfstr")
        c[f"{t}_pfbad"] = (nm, kind, {}, "pfbad")
    c["gaussiannb_pf3_priors"] = ("GaussianNB", "x", dict(priors=[0.2, 0.3, 0.5]), "pf3")
    c["gaussiannb_pf3_smooth"] = ("GaussianNB", "x", dict(var_smoothing=1e-3), "pf3")
    c["multinomialnb_pf3_opts"] = ("MultinomialNB", "abs", dict(alpha=0.3, fit_prior=False), "pf3")
    c["multinomialnb_pf3_prior"] = ("MultinomialNB", "abs", dict(class_prior=[0.5, 0.25, 0.25]), "pf3")
    c["complementnb_pf3_norm"] = ("ComplementNB", "abs", dict(alpha=0.4, norm=True), "pf3")
    c["bernoullinb_pf3_bin"] = ("BernoulliNB", "x", dict(binarize=0.5, alpha=2.0), "pf3")
    c["bernoullinb_pf3_nobin"] = ("BernoulliNB", "x", dict(binarize=None), "pf3")
    c["multinomialnb_pf_neg"] = ("MultinomialNB", "x", {}, "pf1")
    c["categoricalnb_fit"] = ("CategoricalNB", "cat", {}, "fit")
    c["categoricalnb_fit_w"] = ("CategoricalNB", "cat", {}, "fitw")
    c["categoricalnb_fit_min6"] = ("CategoricalNB", "cat", dict(min_categories=6), "fitw")
    c["categoricalnb_fit_minlist"] = ("CategoricalNB", "cat", dict(min_categories=[2, 7, 5, 3, 6]), "fitw")
    c["categoricalnb_fit_alpha"] = ("CategoricalNB", "cat", dict(alpha=0.5, fit_prior=False), "fit")
    c["categoricalnb_fit_prior"] = ("CategoricalNB", "cat", dict(class_prior=[0.2, 0.3, 0.5]), "fit")
    c["categoricalnb_fit_badmin"] = ("CategoricalNB", "cat", dict(min_categories=[2, 3]), "fit")
    c["categoricalnb_fit_neg"] = ("CategoricalNB", "x", {}, "fit")
    c["categoricalnb_pfgrow"] = ("CategoricalNB", "cat", {}, "pfgrow")
    c["categoricalnb_pfgrow_min"] = ("CategoricalNB", "cat", dict(min_categories=3), "pfgrow")
    out = {}
    for name, case in c.items():
        out[name] = case + ("1",)
        out[name + "_b0"] = case + ("0",)
    return out


def _digest(model, X):
    import numpy as np
    h = hashlib.sha256()
    for a in NB_ATTRS:
        if not hasattr(model, a):
            continue
        v = getattr(model, a)
        h.update(a.encode())
        if isinstance(v, (list, tuple)):
            for b in v:
                h.update(np.ascontiguousarray(np.asarray(b)).tobytes())
        else:
            h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(model.predict_proba(X))).tobytes())
    return h.hexdigest()


def _run(name):
    import numpy as np
    cls, kind, kw, plan, blocked = _cases()[name]
    os.environ["MOJOLEARN_XPREP_BLOCKED"] = blocked
    import mojolearn as ml
    X, y, Xc, w = _data()
    data = {"x": X, "abs": np.abs(X), "cat": Xc}[kind]
    try:
        m = getattr(ml, cls)(**kw)
        if plan == "fit":
            m.fit(data, y)
        elif plan == "fitw":
            m.fit(data, y, sample_weight=w.astype(np.float64))
        elif plan == "pf1":
            m.partial_fit(data, y, classes=[0, 1, 2])
        elif plan in ("pf3", "pfstr"):
            labels = np.array(["a", "b", "c"])[y] if plan == "pfstr" else y
            classes = ["a", "b", "c"] if plan == "pfstr" else [0, 1, 2]
            sel = np.flatnonzero(y[:200] != 2)
            m.partial_fit(data[sel], labels[sel], classes=classes)
            m.partial_fit(data[200:450], labels[200:450])
            m.partial_fit(data[450:], labels[450:], sample_weight=w[450:])
        elif plan == "pfgrow":
            m.partial_fit(np.minimum(data[:200], 1), y[:200], classes=[0, 1, 2])
            m.partial_fit(np.minimum(data[200:400], 3), y[200:400])
            m.partial_fit(data[400:], y[400:], sample_weight=w[400:])
        elif plan == "pfbad":
            m.partial_fit(data[:300], y[:300], classes=[0, 1, 2])
            bad = y[300:].copy()
            bad[5] = 7
            m.partial_fit(data[300:], bad)
        return f"{name} sha256={_digest(m, data)}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    names = sorted(_cases())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("BM_REST_NB_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
