# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ALGOS FAMILY'S `extra` LANES: public algorithms COVERAGE.md listed as
not raced without a good reason (Andrew, 2026-09-29: "why are 18 algos
missing? good reason?"), raced through tools/bench_board_algos.py's worker
protocol, seed and parameter rules, and parameter check.

    qn-reg            QNRegressor(loss='squared_error')   cuml.solvers.QN(loss='l2') (NVIDIA),
                                                           scikit-learn LinearRegression (CPU)
    kpss              kpss_test(Y, d=0)                    statsmodels kpss(regression='c',
                                                           nlags=ours' ceil(12 (n/100)^0.25))
    select-d          select_d(Y, D=0)                     the same kpss over d = 0, 1
    clip-grad-norm    clip_grad_norm_(grads, 1.0)          torch.nn.utils.clip_grad_norm_
    cross-entropy     cross_entropy(logits, t, grad)       F.cross_entropy forward + backward
    bpe-encode        BpeTokenizer.encode_batch            Hugging Face tokenizers encode_batch,
                                                           the SAME vocabulary (ours, rendered
                                                           as tokenizer.json)
    bpe-train         BpeVocabularyTrainer.train           tokenizers BpeTrainer, byte level
    lr-step, lr-exponential, lr-onecycle, lr-constant, lr-warmup-linear, lr-warmup-cosine
                      100,000 steps of the schedule        torch.optim.lr_scheduler, same
                                                           parameters
    jl-min-dim        johnson_lindenstrauss_min_dim        scikit-learn's, over one grid

tools/bench_board_algos.py registers these lanes (kind 'extra') and hands
`build` and `quality` to this module. Standard library only at import.
"""
import math
import os

SCHED_STEPS = 100_000
TEXT_BYTES = 4 * 1024 * 1024
TRAIN_BYTES = 1024 * 1024
DOC_CHARS = 2048
BPE_VOCAB = 4096
GRAD_TENSORS = 8
GRAD_SIZE = 2_097_152             # 8 x 2M = 16,777,216 values, the optimizer lanes' size
CE_ROWS, CE_CLASSES = 8192, 8192
QN = dict(loss="squared_error", fit_intercept=True, l1_strength=0.0, l2_strength=0.0,
          max_iter=1000, tol=1e-4, linesearch_max_iter=50, lbfgs_memory=5,
          penalty_normalized=True)
SCHED = {
    "lr-step": ("StepLR", dict(base_lr=0.1, step_size=1000, gamma=0.5)),
    "lr-exponential": ("ExponentialLR", dict(base_lr=0.1, gamma=0.9999)),
    "lr-onecycle": ("OneCycleLR", dict(max_lr=0.1, total_steps=SCHED_STEPS, pct_start=0.3,
                                       anneal_strategy="cos", div_factor=25.0,
                                       final_div_factor=1e4, three_phase=False)),
    "lr-constant": ("training.ConstantLR", dict(peak_lr=1e-3, warmup_steps=1000)),
    "lr-warmup-linear": ("training.WarmupLinearLR", dict(peak_lr=1e-3, warmup_steps=1000,
                                                         total_steps=SCHED_STEPS, min_lr=1e-5)),
    "lr-warmup-cosine": ("training.WarmupCosineLR", dict(peak_lr=1e-3, warmup_steps=1000,
                                                         total_steps=SCHED_STEPS, min_lr=1e-5)),
}
TORCH_SCHED_TEXT = {
    "lr-step": "StepLR(SGD(lr=0.1), step_size=1000, gamma=0.5)",
    "lr-exponential": "ExponentialLR(SGD(lr=0.1), gamma=0.9999)",
    "lr-onecycle": "OneCycleLR(SGD(lr=0.1), max_lr=0.1, total_steps=100000, pct_start=0.3, "
                   "anneal_strategy='cos', div_factor=25, final_div_factor=1e4, "
                   "three_phase=False, cycle_momentum=False)",
    "lr-constant": "LinearLR(SGD(lr=1e-3), start_factor=1/1000, end_factor=1, total_iters=999) "
                   "(warmup to peak, then constant)",
    "lr-warmup-linear": "SequentialLR([LinearLR(1/1000, 1, 999), LinearLR(1, 1e-5/1e-3, 99000)], "
                        "milestones=[1000])",
    "lr-warmup-cosine": "SequentialLR([LinearLR(1/1000, 1, 999), CosineAnnealingLR(T_max=99000, "
                        "eta_min=1e-5)], milestones=[1000])",
}
_TS = ("taxi-hourly", "synthetic")
_TAB = ("taxi", "istella")


def register(add):
    """Add the extra lanes to the algos table through its `_add`."""
    add("qn-reg", xlane="linear", ours="QNRegressor", kind="extra", task="qn", block="reg",
        datasets=_TAB, params=dict(QN), quality="x-qn",
        fit_text="fit(X, y)", other={"sklearn-cpu": "sklearn", "cuml-gpu": "cuml-qn"},
        notes=["ours and cuML: the quasi-Newton solver on the squared loss, no penalty "
               "(cuml.solvers.QN(loss='l2'), present in the pinned cuML 26.08); scikit-learn "
               "LinearRegression is the CPU estimator with the same loss and penalty"],
        mism=["scikit-learn LinearRegression solves the same least-squares problem in closed "
              "form (scipy lstsq); it has no max_iter, tol or L-BFGS settings"])
    add("kpss", xlane="sequence", ours="kpss_test", kind="extra", task="kpss", block="ts",
        datasets=_TS, params=dict(d=0, D=0, s=0, pval_threshold=0.05),
        quality="x-kpss",
        fit_text="the KPSS test of every series", other={"statsmodels-cpu": "statsmodels"},
        notes=["statsmodels.tsa.stattools.kpss(y, regression='c', nlags=ceil(12 (n/100)^0.25)): "
               "ours' lag count passed explicitly; one call per series"],
        mism=["statsmodels interpolates the p-value in its table and computes in float64; ours "
              "decides against cuML's table in float32"])
    add("select-d", xlane="sequence", ours="select_d", kind="extra", task="selectd", block="ts",
        datasets=_TS, params=dict(D=0, s=0, pval_threshold=0.05),
        quality="x-selectd",
        fit_text="the differencing order of every series (KPSS at d = 0, 1)",
        other={"statsmodels-cpu": "statsmodels"},
        notes=["statsmodels: kpss (regression='c', ours' lag count for the differenced length) "
               "on np.diff(y, d) for d = 0, 1; the first stationary d, else 2 (cuML's rule)"])
    add("clip-grad-norm", xlane="training", ours=("clip_grad_norm_", "training.clip_grad_norm_"),
        kind="extra", task="clip", block="tensor", datasets=("synthetic",),
        params=dict(max_norm=1.0, norm_type=2.0, error_if_nonfinite=True),
        quality="x-clip",
        fit_text="clip_grad_norm_ over 8 fp32 gradients of 2,097,152 values", torch_fp32=True,
        notes=["gradients default_rng(7) standard normal; every round clips the same arrays "
               "again, on every arm, so each arm walks the same sequence"])
    add("cross-entropy", xlane="training", ours=("cross_entropy", "training.cross_entropy"),
        kind="extra", task="xent", block="tensor", datasets=("synthetic",),
        params=dict(reduction="mean", label_smoothing=0.0, ignore_index=-100),
        quality="x-xent",
        fit_text="cross_entropy(logits (8192 x 8192), targets) forward and dlogits",
        torch_fp32=True,
        notes=["logits default_rng(7) standard normal, targets integers in [0, 8192)"])
    for slug, what in (("bpe-encode", "encode_batch of 2,048 documents"),
                       ("bpe-train", "train a 4,096-token byte-level vocabulary")):
        add(slug, xlane="prep", ours=("BpeTokenizer",) if slug == "bpe-encode" else
            ("tokenizer.BpeVocabularyTrainer",), kind="extra",
            task=slug.replace("-", ""), block="corpus", datasets=("enwik8",),
            params=dict(vocab_size=BPE_VOCAB, min_frequency=2),
            quality="x-" + slug.replace("-", ""),
            fit_text=what, other={"hf-tokenizers-cpu": "tokenizers"},
            notes=["the first 4 MiB of enwik8 (UTF-8, invalid bytes replaced), 2,048-character "
                   "documents; the vocabulary is trained on the first 1 MiB",
                   "bpe-encode: both arms use OUR trained vocabulary (Hugging Face loads it "
                   "from render_tokenizer_json()), trained before the clock"],
            mism=["bpe-train: each library breaks count ties by its own rule"])
    for slug, (cls, p) in SCHED.items():
        add(slug, xlane="sequence" if "." not in cls else "training", ours=(cls,), kind="extra",
            task="sched", block="tensor", datasets=("synthetic",), params=dict(p),
            quality="x-sched",
            fit_text="the learning rate of steps 1 .. %d" % SCHED_STEPS,
            other={"torch-cpu": "torch"}, notes=["torch: " + TORCH_SCHED_TEXT[slug]])
    add("jl-min-dim", xlane="decomp", ours="johnson_lindenstrauss_min_dim", kind="extra",
        task="jl", block="tensor", datasets=("synthetic",), params={},
        quality="x-jl",
        fit_text="johnson_lindenstrauss_min_dim over 200 n_samples x 50 eps (one scalar call "
                 "per point on both arms)",
        other={"sklearn-cpu": "sklearn"},
        notes=["n_samples = unique(int(logspace(1, 9, 200))), eps = linspace(0.05, 0.95, 50)"])


QUALITY_TEXT = {
    "x-qn": "held-out R2 and RMSE",
    "x-kpss": "agreement of the stationarity flags; max relative difference of the statistic vs "
              "statsmodels",
    "x-selectd": "agreement of the chosen d per series with statsmodels'",
    "x-clip": "relative difference of the returned total norm vs ours",
    "x-xent": "loss relative difference vs float64; max relative difference of dlogits (first "
              "256 rows) vs ours",
    "x-bpeencode": "fraction of documents whose ids equal ours",
    "x-bpetrain": "Jaccard of the learned token byte strings against ours",
    "x-sched": "max relative difference of the learning rates vs ours",
    "x-jl": "fraction of grid points where the integer equals scikit-learn's",
}


def opponents(vendor, lane, spec, torch_gpu):
    if spec.get("torch_fp32"):
        return ("torch-eager-fp32", "torch-compile-fp32")
    return tuple(a for a in spec["other"] if a != "cuml-gpu" or vendor == "nvidia")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def kpss_lags(n):
    """Ours' (and cuML's) lag count: ceil(12 (n / 100)^(1/4)) (Schwert 1989)."""
    return int(math.ceil(12.0 * (n / 100.0) ** 0.25))


def corpus_docs(A):
    p = A["corpus_path"](A["CORPUS_KEYS"]["enwik8"])
    if not os.path.isfile(p):
        raise SystemExit("REFUSING: %s missing (R2 key %s)" % (p, A["CORPUS_KEYS"]["enwik8"]))
    with open(p, "rb") as fh:
        text = fh.read(TEXT_BYTES).decode("utf-8", "replace")
    docs = [text[i:i + DOC_CHARS] for i in range(0, len(text), DOC_CHARS)]
    train = [text[i:i + DOC_CHARS] for i in range(0, min(len(text), TRAIN_BYTES), DOC_CHARS)]
    return docs, train


def jl_grid(np):
    n = np.unique(np.logspace(1, 9, 200).astype(np.int64))
    eps = np.linspace(0.05, 0.95, 50)
    return [(int(a), float(b)) for a in n for b in eps]


def _grads(np, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(GRAD_SIZE).astype(np.float32) for _ in range(GRAD_TENSORS)]


def _ce_inputs(np, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((CE_ROWS, CE_CLASSES)).astype(np.float32)
    t = rng.integers(0, CE_CLASSES, CE_ROWS).astype(np.int64)
    return x, t


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def build(lane, arm, D, A):
    """One arm of an extra lane. `A` is tools/bench_board_algos.py's globals()."""
    np = A["_np"]()
    s = A["LANES"][lane]
    t, p, SEED = s["task"], s["params"], A["SEED"]
    Runner = A["Runner"]
    S = {}
    ours = arm in A["OURS_ARMS"]
    if ours:
        name, fn = A["_ours_class"](lane)
        info = A["_ours_info"](lane)
        info["config"] = "mojolearn.%s(%s)" % (name, ", ".join("%s=%r" % kv for kv in sorted(p.items())))
        rec = dict(p, __library__="mojolearn")
    else:
        info = A["_tool"]("bench_board_more")._sk_info()
        info.update(library=s["other"].get(arm, "torch"), device="cpu", pre_clock_fit=False,
                    input_home="host")
        rec = {"__library__": s["other"].get(arm, "torch")}

    # ---- QN regression
    if t == "qn":
        X, y, Xq = D["X"], np.ascontiguousarray(D["y"], dtype=np.float32), D["Xq"]
        if ours:
            def fit():
                m = fn(**p).fit(X, y)
                S["pred"] = A["_arr"](m.predict(Xq), np.float32)
            return Runner(info, fit, lambda: {"pred": S["pred"]}, record=rec)
        if arm == "sklearn-cpu":
            from sklearn.linear_model import LinearRegression
            est = LinearRegression(fit_intercept=True)
            info["config"] = "sklearn.linear_model.LinearRegression(fit_intercept=True)"

            def fit():
                m = LinearRegression(fit_intercept=True).fit(X, y)
                S["pred"] = m.predict(Xq).astype(np.float32)
            return Runner(info, fit, lambda: {"pred": S["pred"]}, record=A["_BP"]().arm_record(est))
        from cuml.solvers import QN as CumlQN
        kw = dict(p, loss="l2")
        est = CumlQN(**kw)
        info.update(device="gpu", library="cuml")
        info["config"] = "cuml.solvers.QN(%s)" % ", ".join("%s=%r" % kv for kv in sorted(kw.items()))
        up, cinfo, cupy_sync = A["_cuml_up"]({"X": X, "y": y})
        info.update({k: v for k, v in cinfo.items() if k not in info})

        def fit():
            m = CumlQN(**kw).fit(up["X"], up["y"])
            w = np.asarray(A["_host"](m.coef_), dtype=np.float64).reshape(-1)[:X.shape[1]]
            b = float(np.asarray(A["_host"](m.intercept_), dtype=np.float64).reshape(-1)[0])
            S["pred"] = (Xq.astype(np.float64) @ w + b).astype(np.float32)
        return Runner(info, fit, lambda: {"pred": S["pred"]}, sync=cupy_sync,
                      record=A["_BP"]().arm_record(est))

    # ---- KPSS and select_d
    if t in ("kpss", "selectd"):
        Y = np.ascontiguousarray(D["Yfit"], dtype=np.float32)       # (series, n)
        n = Y.shape[1]
        if ours:
            def fit():
                if t == "kpss":
                    st, stat = fn(np.ascontiguousarray(Y.T), return_statistic=True, **p)
                    S["o"] = {"flag": A["_arr"](st, np.int64), "stat": A["_arr"](stat, np.float64)}
                else:
                    S["o"] = {"d": A["_arr"](fn(np.ascontiguousarray(Y.T), **p), np.int64)}
            return Runner(info, fit, lambda: S["o"], record=rec)
        import warnings
        import statsmodels
        from statsmodels.tsa.stattools import kpss
        info["version"] = statsmodels.__version__
        Y64 = Y.astype(np.float64)
        info["config"] = ("statsmodels kpss(y, regression='c', nlags=ceil(12 (n/100)^0.25)) per "
                          "series%s" % ("" if t == "kpss" else ", d = 0, 1"))

        def test(y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, pval, _lags, _crit = kpss(y, regression="c", nlags=kpss_lags(y.shape[0]))
            return stat, pval

        def fit():
            if t == "kpss":
                out = [test(y) for y in Y64]
                S["o"] = {"flag": np.array([int(pv > p["pval_threshold"]) for _, pv in out]),
                          "stat": np.array([st for st, _ in out])}
            else:
                ds = []
                for y in Y64:
                    d = 2
                    for k in (0, 1):
                        if test(np.diff(y, k) if k else y)[1] > p["pval_threshold"]:
                            d = k
                            break
                    ds.append(d)
                S["o"] = {"d": np.array(ds, dtype=np.int64)}
        rec.update(regression="c", nlags=kpss_lags(n), pval_threshold=p["pval_threshold"])
        return Runner(info, fit, lambda: S["o"], record=rec)

    # ---- clip_grad_norm_ and cross_entropy (torch at fp32, eager and compile)
    if t in ("clip", "xent"):
        if ours:
            if t == "clip":
                G = _grads(np, SEED)

                def fit():
                    S["norm"] = float(fn(G, p["max_norm"], norm_type=p["norm_type"],
                                         error_if_nonfinite=p["error_if_nonfinite"]))
                return Runner(info, fit, lambda: {"norm": np.array([S["norm"]])}, record=rec)
            x, tg = _ce_inputs(np, SEED)
            tg32 = tg.astype(np.int32)

            def fit():
                loss, g = fn(x, tg32, reduction=p["reduction"], label_smoothing=p["label_smoothing"],
                             ignore_index=p["ignore_index"], return_grad=True)
                S["o"] = {"loss": np.array([float(loss)]),
                          "grad": A["_arr"](g, np.float32)[:256].copy()}
            return Runner(info, fit, lambda: S["o"], record=rec)
        import torch
        dev = A["_torch_device"](torch)
        mode = arm.split("-")[1]
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        info.update(library="torch", version=torch.__version__, device="gpu" if dev != "cpu" else "cpu",
                    setting="%s-fp32" % mode, input_home="device")
        sync = (lambda: A["_torch_sync"](torch, dev))
        if t == "clip":
            G = _grads(np, SEED)
            params = [torch.nn.Parameter(torch.zeros(GRAD_SIZE, device=dev)) for _ in G]
            for q, g in zip(params, G):
                q.grad = torch.from_numpy(g).to(dev)

            def clip():
                return torch.nn.utils.clip_grad_norm_(params, p["max_norm"], norm_type=p["norm_type"],
                                                      error_if_nonfinite=p["error_if_nonfinite"])
            run = torch.compile(clip) if mode == "compile" else clip
            info["config"] = "torch.nn.utils.clip_grad_norm_(8 grads, 1.0, 2.0) on %s, %s" % (dev, mode)

            def fit():
                S["norm"] = float(run())
            return Runner(info, fit, lambda: {"norm": np.array([S["norm"]])}, sync=sync,
                          record=dict(rec, seed=SEED, **p))
        x_np, tg = _ce_inputs(np, SEED)
        xt = torch.from_numpy(x_np).to(dev).requires_grad_(True)
        tt = torch.from_numpy(tg).to(dev)
        F = torch.nn.functional

        def ce(a, b):
            return F.cross_entropy(a, b, reduction=p["reduction"], label_smoothing=p["label_smoothing"],
                                   ignore_index=p["ignore_index"])
        run = torch.compile(ce) if mode == "compile" else ce
        info["config"] = "F.cross_entropy(logits, targets).backward() on %s, %s" % (dev, mode)

        def fit():
            xt.grad = None
            loss = run(xt, tt)
            loss.backward()
            S["o"] = {"loss": np.array([float(loss.item())]),
                      "grad": xt.grad[:256].float().cpu().numpy()}
        return Runner(info, fit, lambda: S["o"], sync=sync, record=dict(rec, seed=SEED, **p))

    # ---- BPE encode and train
    if t in ("bpeencode", "bpetrain"):
        docs, train = corpus_docs(A)
        if t == "bpeencode":
            import mojolearn as ml
            vocab = ml.tokenizer.BpeVocabularyTrainer(vocab_size=p["vocab_size"],
                                                      min_frequency=p["min_frequency"]).train(train)
            info["vocabulary"] = "ours, trained before the clock on %d documents (%d tokens)" % (
                len(train), vocab.n_tokens)
            if ours:
                tok = vocab.tokenizer()

                def fit():
                    S["ids"] = tok.encode_batch(docs)
            else:
                import tokenizers
                info["version"] = tokenizers.__version__
                tok = tokenizers.Tokenizer.from_str(vocab.render_tokenizer_json())
                info["config"] = "tokenizers.Tokenizer.from_str(ours' tokenizer.json).encode_batch"

                def fit():
                    S["ids"] = [e.ids for e in tok.encode_batch(docs, add_special_tokens=False)]
            def out():
                lens = np.array([len(x) for x in S["ids"]], dtype=np.int64)
                flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in S["ids"]])
                return {"lens": lens, "ids": flat}
            return Runner(info, fit, out, record=rec)
        if ours:
            def fit():
                v = fn(vocab_size=p["vocab_size"], min_frequency=p["min_frequency"]).train(train)
                S["tok"] = [bytes(x) for x in v.tokens]
        else:
            import tokenizers
            from tokenizers import decoders, models, pre_tokenizers, trainers
            info["version"] = tokenizers.__version__
            info["config"] = ("Tokenizer(BPE()), ByteLevel(add_prefix_space=False) pre-tokenizer, "
                              "BpeTrainer(vocab_size=%d, min_frequency=%d, initial_alphabet="
                              "ByteLevel.alphabet(), special_tokens=[])" % (p["vocab_size"],
                                                                            p["min_frequency"]))
            inv = {c: b for b, c in _bytes_to_unicode().items()}

            def fit():
                tk = tokenizers.Tokenizer(models.BPE())
                tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                tk.decoder = decoders.ByteLevel()
                tr = trainers.BpeTrainer(vocab_size=p["vocab_size"], min_frequency=p["min_frequency"],
                                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                         special_tokens=[], show_progress=False)
                tk.train_from_iterator(train, tr)
                S["tok"] = [bytes(inv[c] for c in tok) for tok in tk.get_vocab()]

        def out():
            joined = b"\x00".join(sorted(S["tok"]))
            return {"tokens": np.frombuffer(joined, dtype=np.uint8).copy(),
                    "n": np.array([len(S["tok"])])}
        return Runner(info, fit, out, record=rec)

    # ---- learning-rate schedules
    if t == "sched":
        if ours:
            sched = fn(**p)

            def fit():
                S["lr"] = np.array([sched.lr_at(k) for k in range(1, SCHED_STEPS + 1)])
            return Runner(info, fit, lambda: {"lr": S["lr"]}, record=rec)
        import warnings
        import torch
        info.update(library="torch", version=torch.__version__, config=TORCH_SCHED_TEXT[lane])
        O = torch.optim
        L = O.lr_scheduler

        def make():
            base = p.get("base_lr", p.get("max_lr", p.get("peak_lr")))
            opt = O.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base)
            if lane == "lr-step":
                return L.StepLR(opt, step_size=p["step_size"], gamma=p["gamma"])
            if lane == "lr-exponential":
                return L.ExponentialLR(opt, gamma=p["gamma"])
            if lane == "lr-onecycle":
                return L.OneCycleLR(opt, max_lr=p["max_lr"], total_steps=p["total_steps"],
                                    pct_start=p["pct_start"], anneal_strategy=p["anneal_strategy"],
                                    div_factor=p["div_factor"], final_div_factor=p["final_div_factor"],
                                    three_phase=p["three_phase"], cycle_momentum=False)
            w = p["warmup_steps"]
            warm = L.LinearLR(opt, start_factor=1.0 / w, end_factor=1.0, total_iters=w - 1)
            if lane == "lr-constant":
                return warm
            span = p["total_steps"] - w
            if lane == "lr-warmup-linear":
                tail = L.LinearLR(opt, start_factor=1.0, end_factor=p["min_lr"] / p["peak_lr"],
                                  total_iters=span)
            else:
                tail = L.CosineAnnealingLR(opt, T_max=span, eta_min=p["min_lr"])
            return L.SequentialLR(opt, [warm, tail], milestones=[w])

        def fit():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sc = make()
                out = np.empty(SCHED_STEPS)
                for k in range(SCHED_STEPS):
                    out[k] = sc.get_last_lr()[0]
                    if k + 1 < SCHED_STEPS:
                        sc.step()
            S["lr"] = out
        return Runner(info, fit, lambda: {"lr": S["lr"]}, record=dict(rec, **p))

    # ---- Johnson-Lindenstrauss bound
    if t == "jl":
        grid = jl_grid(np)
        if ours:
            def fit():
                S["k"] = np.array([fn(n, eps=e) for n, e in grid], dtype=np.int64)
            return Runner(info, fit, lambda: {"k": S["k"]}, record=rec)
        from sklearn.random_projection import johnson_lindenstrauss_min_dim as jl
        info["config"] = "sklearn.random_projection.johnson_lindenstrauss_min_dim(n, eps=e) per point"

        def fit():
            S["k"] = np.array([int(jl(n, eps=e)) for n, e in grid], dtype=np.int64)
        return Runner(info, fit, lambda: {"k": S["k"]}, record=rec)
    raise SystemExit("bench_board_extra: no arm for lane %r task %r" % (lane, t))


def _bytes_to_unicode():
    """GPT-2's byte to printable-character table (the byte-level alphabet
    Hugging Face tokenizers' ByteLevel uses)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# ---------------------------------------------------------------------------
# quality (the conductor)
# ---------------------------------------------------------------------------

def quality(lane, D, outs, A):
    np = A["_np"]()
    s = A["LANES"][lane]
    t = s["task"]
    q = {arm: {} for arm in outs}
    ref = outs.get("ours")
    for arm, o in outs.items():
        e = q[arm]
        try:
            if t == "qn":
                e.update(A["_reg"](D["yq"], o["pred"].reshape(-1)))
            elif t == "kpss":
                sm = outs.get("statsmodels-cpu")
                if sm is not None:
                    e["flag_agreement_vs_statsmodels"] = float((o["flag"] == sm["flag"]).mean())
                    st = sm["stat"].astype(np.float64)
                    e["stat_max_rel_diff_vs_statsmodels"] = float(
                        (np.abs(o["stat"] - st) / np.maximum(np.abs(st), 1e-12)).max())
                e["stationary_fraction"] = float(o["flag"].mean())
            elif t == "selectd":
                sm = outs.get("statsmodels-cpu")
                if sm is not None:
                    e["d_agreement_vs_statsmodels"] = float((o["d"] == sm["d"]).mean())
            elif t == "clip" and ref is not None:
                e["norm"] = float(o["norm"][0])
                e["norm_rel_diff_vs_ours"] = abs(float(o["norm"][0]) - float(ref["norm"][0])) / max(
                    abs(float(ref["norm"][0])), 1e-30)
            elif t == "xent":
                x, tg = _ce_inputs(np, A["SEED"])
                x64 = x.astype(np.float64)
                mx = x64.max(axis=1, keepdims=True)
                lse = mx[:, 0] + np.log(np.exp(x64 - mx).sum(axis=1))
                exact = float((lse - x64[np.arange(x.shape[0]), tg]).mean())
                e["loss_rel_err_vs_fp64"] = abs(float(o["loss"][0]) - exact) / abs(exact)
                if ref is not None and arm != "ours":
                    g, rg = o["grad"].astype(np.float64), ref["grad"].astype(np.float64)
                    e["grad_max_rel_diff_vs_ours"] = float(np.abs(g - rg).max() / (np.abs(rg).max() or 1.0))
            elif t == "bpeencode" and ref is not None:
                same = 0
                oi, ol = np.split(o["ids"], np.cumsum(o["lens"])[:-1]), o["lens"]
                ri = np.split(ref["ids"], np.cumsum(ref["lens"])[:-1])
                for a, b in zip(oi, ri):
                    same += int(a.shape == b.shape and bool((a == b).all()))
                e["documents_equal_to_ours"] = same / float(len(ol))
                e["tokens"] = int(ol.sum())
            elif t == "bpetrain":
                mine = set(bytes(o["tokens"]).split(b"\x00"))
                e["n_tokens"] = int(o["n"][0])
                if ref is not None:
                    theirs = set(bytes(ref["tokens"]).split(b"\x00"))
                    e["jaccard_vs_ours"] = len(mine & theirs) / float(len(mine | theirs))
            elif t == "sched" and ref is not None:
                r = ref["lr"].astype(np.float64)
                e["max_rel_diff_vs_ours"] = float((np.abs(o["lr"] - r) / np.maximum(np.abs(r), 1e-30)).max())
            elif t == "jl":
                sk = outs.get("sklearn-cpu")
                if sk is not None:
                    e["equal_fraction_vs_sklearn"] = float((o["k"] == sk["k"]).mean())
        except Exception as exc:  # noqa: BLE001
            e["quality_error"] = repr(exc)[:300]
    return q
