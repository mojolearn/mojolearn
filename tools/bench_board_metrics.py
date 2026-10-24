# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ALGOS FAMILY'S METRIC LANES: every public `mojolearn.metrics` function against its peer.

measure-coverage (2026-10-24, Andrew 2026-10-22: every exposed API "measured if appropriate with
apples-to-apples comparison against peers"). One lane per function, `metric-<name>`, kind 'extra',
registered through tools/bench_board_extra.py (which hands `build`, `quality` and `input_arrays` here).

    ours         mojolearn.metrics.<name>(*inputs, **params)     host arrays in, the result on the host
    sklearn-cpu  sklearn.metrics.<name>(*inputs, **params)       (sklearn.metrics.cluster for
                                                                  contingency_matrix / pair_confusion_matrix)
    scipy-cpu    entropy: scipy.stats.entropy(bincount(labels), base); kl_divergence: scipy.special.rel_entr
                 (P, Q).sum() (no renormalization, cuML's and ours' definition); sklearn has neither
    cuml-gpu     cuml.metrics.<name> on NVIDIA where cuML 26.8 has the function with the same parameters
                 (CUML below); its inputs go to the device before its clock (cuML's kernel-only clock,
                 CLAUDE.md OPPONENT CLOCKS); on NVIDIA the CPU arms then do not race
                 (bench_board.gpu_opponents_first)

No lane passes None: the parameter check reads None on both sides as a library default, not a matched value
(bench_board_params.check), so every setting a lane passes is a concrete value (cohen_kappa_score
weights='linear', confusion_matrix normalize='true', entropy base 2).

THE SAME EXPLICIT PARAMETERS ON EVERY ARM: each lane's `params` are passed by keyword to every arm, and every
arm's record is those params, the input-array digests and (opponents) the library function it calls, so the
parameter check (tools/bench_board_params.py) compares equal records. Settings a cuML function does not take
are not in that lane's params on any arm (CUML_PARAMS keeps the lane's params to the shared ones).

THE INPUTS (built before the clock, seeded 7, the same arrays on every arm) come from the board's existing
held-out rows (bench_board_algos cls / reg blocks: Xq, yq of taxi and Istella; no new shapes):
    label    y_true = yq (binary), y_pred = (score > 0.5) with score = sigmoid(2 (2 y - 1) + N(0, 1))
    score    y_true = yq (binary), y_score = that score (a probability)
    reg      y_true = yq (real), y_pred = y_true + N(0, (std y / 2)^2)
    regpos   both shifted to be positive: |y| + 1 (the deviance / log / percentage errors' domain)
    cluster  labels_true = yq in 10 quantile bins, labels_pred = labels_true with 20% replaced by seeded draws
    multi    y_true = the cluster labels (10 classes), y_score = one-hot + N(0, 1) (top_k_accuracy_score)
    rank     y_true = (Xq[:, :L] > 0) as relevance, y_score = Xq[:, :L] + N(0, 1), L = min(16, d) columns
    internal X = Xq (20,000 stride rows; silhouette is quadratic), labels = Xq[:, 0] in 8 quantile bins
    embed    X = Xq (5,000 stride rows; trustworthiness ranks every pair), X_embedded = Xq[:, :2] + N(0, 0.1)
    curve    x = sorted seeded uniform draws, y = sqrt(x) (auc)
    dist     P, Q = two positive columns of Xq normalized to sum 1 (kl_divergence)
The class, label and column counts (10, 8, 16) are workload choices, not dispatch rules: no route of ours or of
any opponent keys on them.
"""

#: (name, input kind, params, sklearn module or None, binding) -> one lane each. `binding` is the device
#: binding the function's module loads (python/mojolearn/_metrics_impl.py: _mojolearn_metrics;
#: _expansion_metrics.py: _mojolearn_x_metrics), read back for the numeric mode and named in the BUILDS list.
_M, _X = "_mojolearn_metrics", "_mojolearn_x_metrics"
_SK = "sklearn.metrics"
_SKC = "sklearn.metrics.cluster"
_BIN = dict(average="binary", pos_label=1, zero_division=0.0)
METRICS = (
    # classification labels
    ("accuracy_score", "label", dict(normalize=True), _SK, _M),
    ("balanced_accuracy_score", "label", dict(adjusted=False), _SK, _X),
    ("cohen_kappa_score", "label", dict(weights="linear"), _SK, _X),
    ("confusion_matrix", "label", dict(labels=[0, 1], normalize="true"), _SK, _M),
    ("multilabel_confusion_matrix", "label", dict(labels=[0, 1], samplewise=False), _SK, _X),
    ("f1_score", "label", dict(_BIN), _SK, _M),
    ("precision_score", "label", dict(_BIN), _SK, _M),
    ("recall_score", "label", dict(_BIN), _SK, _M),
    ("jaccard_score", "label", dict(_BIN), _SK, _X),
    ("fbeta_score", "label", dict(_BIN, beta=0.5), _SK, _X),
    ("precision_recall_fscore_support", "label", dict(_BIN, beta=1.0), _SK, _X),
    ("classification_report", "label", dict(labels=[0, 1], digits=4, output_dict=True, zero_division=0.0),
     _SK, _X),
    ("class_likelihood_ratios", "label", dict(labels=[0, 1]), _SK, _X),
    ("hamming_loss", "label", {}, _SK, _X),
    ("matthews_corrcoef", "label", {}, _SK, _X),
    ("zero_one_loss", "label", dict(normalize=True), _SK, _X),
    # classification scores
    ("roc_auc_score", "score", {}, _SK, _M),
    ("average_precision_score", "score", dict(average="macro", pos_label=1), _SK, _X),
    ("roc_curve", "score", dict(pos_label=1, drop_intermediate=True), _SK, _X),
    ("precision_recall_curve", "score", dict(pos_label=1, drop_intermediate=False), _SK, _M),
    # det_curve / brier_score_loss: drop_intermediate and scale_by_half are newer than the pinned
    # scikit-learn 1.7.2 surface ours follows; both libraries' defaults are the same rule (no point dropped; the
    # binary Brier score halved), so only pos_label is passed
    ("det_curve", "score", dict(pos_label=1), _SK, _X),
    ("brier_score_loss", "score", dict(pos_label=1), _SK, _X),
    ("d2_brier_score", "score", dict(pos_label=1), _SK, _X),
    ("log_loss", "score", dict(normalize=True, labels=[0, 1]), _SK, _M),
    ("d2_log_loss_score", "score", dict(labels=[0, 1]), _SK, _X),
    ("hinge_loss", "decision", {}, _SK, _X),
    ("top_k_accuracy_score", "multi", dict(k=3, normalize=True, labels=list(range(10))), _SK, _X),
    # regression
    ("mean_squared_error", "reg", dict(multioutput="uniform_average"), _SK, _M),
    ("root_mean_squared_error", "reg", dict(multioutput="uniform_average"), _SK, _M),
    ("mean_absolute_error", "reg", dict(multioutput="uniform_average"), _SK, _M),
    ("median_absolute_error", "reg", dict(multioutput="uniform_average"), _SK, _X),
    ("max_error", "reg", {}, _SK, _X),
    ("explained_variance_score", "reg", dict(multioutput="uniform_average", force_finite=True), _SK, _X),
    ("r2_score", "reg", {}, _SK, _M),
    ("d2_absolute_error_score", "reg", dict(multioutput="uniform_average"), _SK, _X),
    ("d2_pinball_score", "reg", dict(alpha=0.5, multioutput="uniform_average"), _SK, _X),
    ("mean_pinball_loss", "reg", dict(alpha=0.5, multioutput="uniform_average"), _SK, _X),
    ("mean_absolute_percentage_error", "regpos", dict(multioutput="uniform_average"), _SK, _X),
    ("mean_squared_log_error", "regpos", dict(multioutput="uniform_average"), _SK, _X),
    ("root_mean_squared_log_error", "regpos", dict(multioutput="uniform_average"), _SK, _X),
    ("mean_poisson_deviance", "regpos", {}, _SK, _X),
    ("mean_gamma_deviance", "regpos", {}, _SK, _X),
    ("mean_tweedie_deviance", "regpos", dict(power=1.5), _SK, _X),
    ("d2_tweedie_score", "regpos", dict(power=1.5), _SK, _X),
    # clustering against labels
    ("adjusted_rand_score", "cluster", {}, _SK, _M),
    ("rand_score", "cluster", {}, _SK, _M),
    ("adjusted_mutual_info_score", "cluster", dict(average_method="arithmetic"), _SK, _X),
    ("normalized_mutual_info_score", "cluster", dict(average_method="arithmetic"), _SK, _X),
    ("mutual_info_score", "cluster", {}, _SK, _M),
    ("homogeneity_score", "cluster", {}, _SK, _M),
    ("completeness_score", "cluster", {}, _SK, _M),
    ("v_measure_score", "cluster", dict(beta=1.0), _SK, _M),
    ("homogeneity_completeness_v_measure", "cluster", dict(beta=1.0), _SK, _M),
    ("fowlkes_mallows_score", "cluster", {}, _SK, _M),
    ("contingency_matrix", "cluster", dict(sparse=False), _SKC, _X),
    ("pair_confusion_matrix", "cluster", {}, _SKC, _X),
    ("entropy", "labels", dict(base=2.0), None, _M),
    # clustering, internal
    ("silhouette_score", "internal", dict(metric="euclidean"), _SK, _M),
    ("silhouette_samples", "internal", dict(metric="euclidean"), _SK, _M),
    ("calinski_harabasz_score", "internal", {}, _SK, _X),
    ("davies_bouldin_score", "internal", {}, _SK, _X),
    # ranking
    ("coverage_error", "rank", {}, _SK, _X),
    ("label_ranking_average_precision_score", "rank", {}, _SK, _X),
    ("label_ranking_loss", "rank", {}, _SK, _X),
    ("dcg_score", "rank", dict(k=10, log_base=2, ignore_ties=False), _SK, _X),
    ("ndcg_score", "rank", dict(k=10, ignore_ties=False), _SK, _X),
    # other
    ("auc", "curve", {}, _SK, _X),
    ("trustworthiness", "embed", dict(n_neighbors=5, metric="euclidean"), "sklearn.manifold", _M),
    ("kl_divergence", "dist", {}, None, _M),
)

#: The functions cuML 26.8 (cuml.metrics, cuml.metrics.cluster) has with this lane's parameters (positional
#: inputs, the lane's keywords). Every other metric has no GPU peer and races the CPU library.
CUML = {
    "accuracy_score": "cuml.metrics", "roc_auc_score": "cuml.metrics", "r2_score": "cuml.metrics",
    "mean_squared_error": "cuml.metrics", "mean_absolute_error": "cuml.metrics",
    "mean_squared_log_error": "cuml.metrics", "kl_divergence": "cuml.metrics",
    "trustworthiness": "cuml.metrics", "adjusted_rand_score": "cuml.metrics.cluster",
    "homogeneity_score": "cuml.metrics.cluster", "completeness_score": "cuml.metrics.cluster",
    "v_measure_score": "cuml.metrics.cluster", "mutual_info_score": "cuml.metrics.cluster",
    "entropy": "cuml.metrics.cluster", "silhouette_score": "cuml.metrics.cluster",
}

#: input kind -> (board block, stride subsets of the held-out rows)
KIND_BLOCK = {"label": ("cls", {}), "score": ("cls", {}), "decision": ("cls", {}), "multi": ("reg", {}),
              "reg": ("reg", {}), "regpos": ("reg", {}), "cluster": ("reg", {}), "labels": ("reg", {}),
              "internal": ("reg", {"Xq": 20_000}), "embed": ("reg", {"Xq": 5_000}), "rank": ("reg", {}),
              "curve": ("reg", {}), "dist": ("reg", {})}
N_CLASSES, N_CLUSTERS, RANK_COLUMNS = 10, 8, 16
TAB = ("taxi", "istella")


def slug(name):
    return "metric-" + name.replace("_", "-")


def spec_of(name):
    return next(m for m in METRICS if m[0] == name)


def register(add):
    """Add one lane per metric through bench_board_algos._add."""
    for name, kind, params, sk, binding in METRICS:
        block, sub = KIND_BLOCK[kind]
        other = {}
        if sk is not None:
            other["sklearn-cpu"] = "sklearn"
        else:
            other["scipy-cpu"] = "scipy"
        if name in CUML:
            other["cuml-gpu"] = "cuml"
        add(slug(name), xlane="metrics", ours=("metrics." + name,), kind="extra", task="metric",
            metric=name, metric_inputs=kind, block=block, datasets=TAB, params=dict(params), sub=dict(sub),
            binding=binding, quality="x-metric",
            fit_text="metrics.%s on the %s inputs (%s rows), host in, result on the host" % (
                name, kind, "20,000 stride held-out" if kind == "internal" else
                "5,000 stride held-out" if kind == "embed" else "the held-out"),
            other=other,
            notes=["inputs: " + INPUT_TEXT[kind], "the same keyword parameters on every arm: %r" % (params,)],
            mism=(["cuML's inputs are on the device before its clock (kernel-only); ours and the CPU "
                   "libraries take host arrays inside theirs"] if name in CUML else []))


INPUT_TEXT = {
    "label": "y_true = held-out binary target, y_pred = (sigmoid(2(2y-1) + N(0,1)) > 0.5), seed 7",
    "score": "y_true = held-out binary target, y_score = sigmoid(2(2y-1) + N(0,1)), seed 7",
    "decision": "y_true = held-out binary target, pred_decision = 2 sigmoid(2(2y-1) + N(0,1)) - 1, seed 7",
    "multi": "y_true = held-out target in 10 quantile bins, y_score = one-hot + N(0,1), seed 7",
    "reg": "y_true = held-out real target, y_pred = y_true + N(0, (std/2)^2), seed 7",
    "regpos": "|y| + 1 of the reg inputs (positive, the deviances' domain), seed 7",
    "cluster": "labels_true = held-out target in 10 quantile bins, labels_pred = 20% redrawn, seed 7",
    "labels": "labels = held-out target in 10 quantile bins",
    "internal": "X = 20,000 stride held-out rows, labels = column 0 in 8 quantile bins",
    "embed": "X = 5,000 stride held-out rows, X_embedded = X[:, :2] + N(0, 0.01), seed 7",
    "rank": "y_true = (Xq[:, :16] > 0), y_score = Xq[:, :16] + N(0,1), seed 7",
    "curve": "x = sorted uniform draws over the held-out row count, y = sqrt(x), seed 7",
    "dist": "P, Q = |Xq[:, 0]| + 1e-3 and |Xq[:, 1]| + 1e-3, each normalized to sum 1",
}

QUALITY_TEXT = {"x-metric": "max relative difference of every returned number vs ours (and the shapes); "
                            "the reference CPU library's value is the opponent's own"}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def _bins(np, v, k):
    edges = np.quantile(np.asarray(v, dtype=np.float64), np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, v, side="right").astype(np.int32)


def inputs(np, kind, D, seed):
    """[(name, array)] the arms of a metric lane are handed positionally, in order (seeded, built before the
    clock, the same arrays on every arm)."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    yq = np.ascontiguousarray(D["yq"], dtype=np.float64)
    if kind in ("label", "score", "decision"):
        y = (yq > 0.5).astype(np.int32)
        score = 1.0 / (1.0 + np.exp(-(2.0 * (2.0 * y - 1.0) + rng.standard_normal(y.shape[0]))))
        if kind == "label":
            return [("y_true", y), ("y_pred", (score > 0.5).astype(np.int32))]
        if kind == "score":
            return [("y_true", y), ("y_score", score.astype(f32))]
        return [("y_true", y), ("pred_decision", (2.0 * score - 1.0).astype(f32))]
    if kind in ("reg", "regpos"):
        pred = yq + rng.standard_normal(yq.shape[0]) * (0.5 * float(yq.std()) or 1.0)
        if kind == "regpos":
            return [("y_true", (np.abs(yq) + 1.0).astype(f32)), ("y_pred", (np.abs(pred) + 1.0).astype(f32))]
        return [("y_true", yq.astype(f32)), ("y_pred", pred.astype(f32))]
    if kind in ("cluster", "labels", "multi"):
        lt = _bins(np, yq, N_CLASSES)
        if kind == "labels":
            return [("labels", lt)]
        if kind == "multi":
            onehot = np.eye(N_CLASSES, dtype=np.float64)[lt]
            return [("y_true", lt), ("y_score", (onehot + rng.standard_normal(onehot.shape)).astype(f32))]
        lp = lt.copy()
        redraw = rng.random(lt.shape[0]) < 0.2
        lp[redraw] = rng.integers(0, N_CLASSES, int(redraw.sum())).astype(np.int32)
        return [("labels_true", lt), ("labels_pred", lp)]
    Xq = np.ascontiguousarray(D["Xq"], dtype=f32)
    if kind == "internal":
        return [("X", Xq), ("labels", _bins(np, Xq[:, 0], N_CLUSTERS))]
    if kind == "embed":
        emb = Xq[:, :2] + 0.1 * rng.standard_normal((Xq.shape[0], 2))
        return [("X", Xq), ("X_embedded", emb.astype(f32))]
    if kind == "rank":
        L = min(RANK_COLUMNS, Xq.shape[1])
        rel = (Xq[:, :L] > 0).astype(np.int32)
        return [("y_true", rel), ("y_score", (Xq[:, :L] + rng.standard_normal((Xq.shape[0], L))).astype(f32))]
    if kind == "curve":
        x = np.sort(rng.random(yq.shape[0]))
        return [("x", x.astype(f32)), ("y", np.sqrt(x).astype(f32))]
    if kind == "dist":
        P = np.abs(Xq[:, 0].astype(np.float64)) + 1e-3
        Q = np.abs(Xq[:, 1].astype(np.float64)) + 1e-3
        return [("P", (P / P.sum()).astype(f32)), ("Q", (Q / Q.sum()).astype(f32))]
    raise ValueError("no metric inputs for kind %r" % (kind,))


def _staged_inputs(np, kind, D, seed, A):
    """inputs(); inside the smoke worker once per staged block (smoke_inproc.memo_on), the same arrays for every
    arm."""
    SI = A["_staging"]()
    if SI is None:
        return inputs(np, kind, D, seed)
    return SI.memo_on("metric-inputs-" + kind, (D["yq"], D["Xq"]), lambda: inputs(np, kind, D, seed))


def _flat(np, r, out, key="out"):
    """Every number a metric returns as {path: float64 array} (dicts by sorted key, tuples by index)."""
    if isinstance(r, dict):
        for k in sorted(r, key=str):
            _flat(np, r[k], out, "%s.%s" % (key, k))
        return out
    if isinstance(r, (tuple, list)) and not (len(r) and all(isinstance(v, (int, float)) for v in r)):
        for i, v in enumerate(r):
            _flat(np, v, out, "%s.%d" % (key, i))
        return out
    if hasattr(r, "toarray"):
        r = r.toarray()
    if hasattr(r, "get") and not isinstance(r, dict):          # a cupy array
        r = r.get()
    out[key] = np.asarray(r, dtype=np.float64).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def opponent_function(name, arm):
    """The library callable an opponent arm of metric `name` calls (its __implementation__)."""
    import importlib
    if arm == "cuml-gpu":
        return getattr(importlib.import_module(CUML[name]), name)
    if arm == "scipy-cpu":
        if name == "entropy":
            from scipy import stats
            return stats.entropy
        from scipy import special
        return special.rel_entr
    sk = spec_of(name)[3]
    return getattr(importlib.import_module(sk), name)


def build(lane, arm, D, A, ours, fn, info, rec, Runner, S, np):
    """One arm of a metric lane (bench_board_extra.build hands every 'metric' task here)."""
    s = A["LANES"][lane]
    name, kind, p = s["metric"], s["metric_inputs"], s["params"]
    args = _staged_inputs(np, kind, D, A["SEED"], A)
    rec.update(p)
    rec["input_arrays"] = A["_input_array_records"](args)
    host = [a for _, a in args]
    if ours:
        info["config"] = "mojolearn.metrics.%s(%s, %s)" % (name, ", ".join(n for n, _ in args), ", ".join(
            "%s=%r" % kv for kv in sorted(p.items())))

        def fit():
            S["o"] = _flat(np, fn(*host, **p), {})
        return Runner(info, fit, lambda: S["o"], record=rec)
    impl = opponent_function(name, arm)
    rec["__implementation__"] = impl
    if arm == "cuml-gpu":
        import cuml
        info.update(device="gpu", library="cuml", version=cuml.__version__)
        up, cinfo, sync = A["_cuml_up"]({n: a for n, a in args})
        info.update({k: v for k, v in cinfo.items() if k not in info})
        dev = [up[n] for n, _ in args]
        info["config"] = "%s.%s(%s, %s), inputs on the device before the clock" % (
            CUML[name], name, ", ".join(n for n, _ in args), ", ".join("%s=%r" % kv for kv in sorted(p.items())))

        def fit():
            S["o"] = _flat(np, impl(*dev, **p), {})
        return Runner(info, fit, lambda: S["o"], sync=sync, record=rec)
    if arm == "scipy-cpu":
        import scipy
        info.update(library="scipy", version=scipy.__version__)
        if name == "entropy":
            info["config"] = "scipy.stats.entropy(np.bincount(labels), base=%r)" % (p["base"],)

            def fit():
                S["o"] = _flat(np, impl(np.bincount(host[0]), base=p["base"]), {})
        else:
            info["config"] = "scipy.special.rel_entr(P, Q).sum() (no renormalization)"

            def fit():
                S["o"] = _flat(np, impl(host[0].astype(np.float64), host[1].astype(np.float64)).sum(), {})
        return Runner(info, fit, lambda: S["o"], record=rec)
    info["config"] = "%s.%s(%s, %s)" % (spec_of(name)[3], name, ", ".join(n for n, _ in args), ", ".join(
        "%s=%r" % kv for kv in sorted(p.items())))

    def fit():
        S["o"] = _flat(np, impl(*host, **p), {})
    return Runner(info, fit, lambda: S["o"], record=rec)


def input_arrays(lane, D, A):
    np = A["_np"]()
    s = A["LANES"][lane]
    return A["_input_array_records"](_staged_inputs(np, s["metric_inputs"], D, A["SEED"], A))


def quality(np, o, ref):
    """{max_rel_diff_vs_ours, outputs_equal_shape} of one arm's outputs against ours'."""
    e = {"outputs": len(o)}
    if ref is None:
        return e
    worst, same = 0.0, True
    for k, r in ref.items():
        v = o.get(k)
        if v is None or v.shape != r.shape:
            same = False
            continue
        if r.size:
            worst = max(worst, float(np.abs(v - r).max() / max(float(np.abs(r).max()), 1e-30)))
    e["outputs_equal_shape_vs_ours"] = bool(same and set(o) == set(ref))
    e["max_rel_diff_vs_ours"] = worst
    return e
