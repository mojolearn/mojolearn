# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ALGOS FAMILY'S `extra` LANES: public algorithms COVERAGE.md listed as
not raced without a good reason (2026-09-29: 18 algorithms were missing),
raced through tools/bench_board_algos.py's worker
protocol, seed and parameter rules, and parameter check.

    qn-reg            QNRegressor(loss='squared_error')   cuml.solvers.QN(loss='l2') (NVIDIA),
                                                           scikit-learn TweedieRegressor(power=0,
                                                           identity link, alpha=0, 'lbfgs') (CPU)
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
    lowbit-*          mojolearn.linalg quantize_int8, dequantize_int8, quantize_int15,
                      dequantize_int15, matmul_int15, to_bf16, from_bf16
                                                           torch twins of the same contract
                                                           (gemm/IDENTICAL_LOWBIT_CONTRACT.md)
    train-*           mojolearn.training rms_norm_forward/backward, linear_forward/backward,
                      embedding_forward/backward, residual_dropout(_backward),
                      chunked_lm_head_loss, accumulate_grads
                                                           torch functional + autograd

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
# measure-coverage (2026-10-22): the low-bit and training-function lanes. Sizes are the board's existing
# workloads, not new targets: the elementwise conversions read one 4096 x 4096 float32 matrix (16,777,216
# values, the optimizer and clip lanes' size); matmul_int15 is the gemm lanes' 4096^3 product; the training
# layers take the LM lanes' kind of shape, 8,192 token rows of width 1,024, a 4,096-wide linear and a
# 32,768-row vocabulary.
LB_ROWS = LB_COLS = 4096
TR_ROWS, TR_WIDTH, TR_OUT, TR_VOCAB = 8192, 1024, 4096, 32768
TR_EPS = 1e-6
TR_DROPOUT_P = 0.1
LM_HEAD_CHUNK = 1024              # torch's chunked CE: 1,024 rows of logits at a time (8 x 128 MB)
ACC_MICROBATCHES = 8
# smoke-fixes (2026-10-14; algos/qn-reg: "delta, warm_start missing from ours read-back"): both
# pinned on ours and cuML. delta: the objective-change stop; None meant tol * 0.01 on both (cuML
# 26.08 python/cuml/cuml/solvers/qn.pyx:113 change_tol = delta if not None else tol * 0.01; ours
# python/mojolearn/linear_model.py QNRegressor.fit, the same rule), now that value explicitly.
# warm_start=False: cuML starts from zero unless warm_start and a prior coef_ (qn.pyx:552); ours
# takes only False (QNRegressor.__init__).
QN = dict(loss="squared_error", fit_intercept=True, l1_strength=0.0, l2_strength=0.0,
          max_iter=1000, tol=1e-4, delta=1e-4 * 0.01, linesearch_max_iter=50, lbfgs_memory=5,
          warm_start=False, penalty_normalized=True)
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
        # FAIRNESS FIX (2026-10-13 audit): the CPU arm was scikit-learn LinearRegression, a
        # closed-form lstsq, raced against ours' quasi-Newton iterations: a different algorithm.
        # It is now scikit-learn's own L-BFGS on the same loss, TweedieRegressor(power=0,
        # link='identity', alpha=0, solver='lbfgs'), with max_iter, tol and fit_intercept pinned
        # to ours'. Results raced before this change were wrong for this lane; apply extra
        # scrutiny.
        notes=["ours and cuML: the quasi-Newton solver on the squared loss, no penalty "
               "(cuml.solvers.QN(loss='l2'), present in the pinned cuML 26.08); scikit-learn: "
               "TweedieRegressor(power=0, link='identity', alpha=0, solver='lbfgs'), its L-BFGS "
               "(scipy L-BFGS-B) on the same half squared error"],
        mism=["L-BFGS memory: ours and cuML 5; scikit-learn's scipy L-BFGS-B 10 (maxcor, not a "
              "scikit-learn argument); line search at most 50 steps on every arm (scikit-learn "
              "maxls=50)", "stopping: each library's own rule reads tol=1e-4 (scikit-learn "
              "gtol on the projected gradient)"])
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
        # FAIRNESS FIX (2026-10-13 audit): every round clipped the already-clipped arrays again
        # (torch always multiplies by the clamped coefficient; ours scales only above max_norm),
        # so the per-round work was not shown equal. Every round now clips a fresh copy of the
        # unclipped gradients on every arm, put back after the round, outside the clock. Results
        # raced before this change were wrong for this lane; apply extra scrutiny.
        notes=["gradients default_rng(7) standard normal; every round clips the unclipped "
               "gradients (restored after each round, outside the clock), on every arm, so each "
               "arm scales the same arrays by the same coefficient every round"])
    add("cross-entropy", xlane="training", ours=("cross_entropy", "training.cross_entropy"),
        kind="extra", task="xent", block="tensor", datasets=("synthetic",),
        params=dict(reduction="mean", label_smoothing=0.0, ignore_index=-100),
        quality="x-xent",
        fit_text="cross_entropy(logits (8192 x 8192), targets) forward and dlogits",
        torch_fp32=True,
        # FAIRNESS FIX (2026-10-13 audit): ours' clock returned the whole (8192, 8192) dlogits to
        # the host while torch's copied 256 rows; torch's clock now copies the whole gradient to
        # the host too. Results raced before this change were wrong for this lane; apply extra
        # scrutiny.
        notes=["logits default_rng(7) standard normal, targets integers in [0, 8192)",
               "both arms return the loss and the whole dlogits on the host inside the clock "
               "(ours' cross_entropy(..., return_grad=True) returns a host array; torch copies "
               "xt.grad to the host)"])
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
    # measure-coverage (2026-10-22, Andrew: every exposed API measured against its peer): the low-bit linalg
    # functions against torch twins of the SAME contract (gemm/IDENTICAL_LOWBIT_CONTRACT.md L-1..L-6,
    # W-1..W-7), explicit in torch ops: per-row power-of-two exponent read from the absmax's exponent field
    # (torch.frexp), subnormals flushed, round half to even (torch.round), clamp, int8 / int15 codes; bf16 by
    # torch's own float32 -> bfloat16 cast (round to nearest even) after the flush. Every arm is handed host
    # arrays and returns host arrays inside its clock (ours' functions take and return host buffers), so the
    # copies are inside both clocks.
    lb_note = ("one seed-7 standard normal 4096 x 4096 float32 matrix (codes / bits / exponents for the "
               "dequantize and widen lanes drawn seeded in their ranges); host in, host out on every arm")
    for slug, ours_name, what in (
            ("lowbit-quantize-int8", "linalg.quantize_int8", "quantize_int8(x): codes and row exponents"),
            ("lowbit-dequantize-int8", "linalg.dequantize_int8", "dequantize_int8(codes, exponents)"),
            ("lowbit-quantize-int15", "linalg.quantize_int15", "quantize_int15(x): hi, lo planes, exponents"),
            ("lowbit-dequantize-int15", "linalg.dequantize_int15", "dequantize_int15(hi, lo, exponents)"),
            ("lowbit-to-bf16", "linalg.to_bf16", "to_bf16(x): bfloat16 bits"),
            ("lowbit-from-bf16", "linalg.from_bf16", "from_bf16(bits): float32")):
        add(slug, xlane="linalg", ours=(ours_name,), kind="extra", task=slug.replace("-", "_"),
            block="tensor", datasets=("synthetic",), params=dict(rows=LB_ROWS, cols=LB_COLS),
            binding="_mojolearn_linalg",
            quality="x-lowbit", fit_text=what + " of a 4096 x 4096 matrix, host to host",
            torch_fp32=True, notes=[lb_note],
            mism=["torch has no function of this contract; the torch arms are the contract written in "
                  "torch ops (frexp, ldexp, round half to even, clamp), eager and torch.compile"])
    add("lowbit-matmul-int15", xlane="linalg", ours=("linalg.matmul_int15",), kind="extra",
        task="lowbit_matmul_int15", block="tensor", datasets=("synthetic",), binding="_mojolearn_linalg",
        params=dict(m=LB_ROWS, n=LB_ROWS, k=LB_COLS), quality="x-lowbit",
        fit_text="matmul_int15((hi, lo, e)_a, (hi, lo, e)_b): a @ b.T of 4096^3, host to host",
        torch_arms={"nvidia": ("torch-eager-fp64", "torch-eager-int8"), "amd": ("torch-eager-fp64",),
                    "apple": ()},
        notes=["a and b: seed-7 standard normal 4096 x 4096 float32, quantized by the profile (W-1..W-3) "
               "before the clock; every arm is handed the same planes and exponents",
               "torch-eager-fp64: the codes as float64, one torch matmul (every partial sum is an integer "
               "below 2^53, so the float64 product is the exact integer sum in any order), rounded to "
               "float32 (nearest even) and scaled by 2^(ea + eb): the contract's value exactly",
               "torch-eager-int8 (NVIDIA): torch._int_mm on the int8 planes, HH, HL, LH and LL as four "
               "exact int32 products, combined in int64 (code = hi * 128 + lo), then the same rounding "
               "and scale"],
        mism=["torch has no fifteen-bit product; both torch arms compute the contract's exact integer sum "
              "with torch's own kernels (float64 matmul, int8 matmul)",
              "torch._int_mm is CUDA only: AMD races the float64 arm alone"])
    # the training-function layers against torch functional (+ autograd for the backward ones), fp32, TF32
    # off, eager and torch.compile; host in, host out on every arm (ours' functions take and return host
    # arrays, as the cross-entropy lane)
    tr_note = ("seed-7 standard normal float32 inputs: %d rows of width %d, a %d x %d linear weight, a "
               "%d x %d table with %d seeded ids" % (TR_ROWS, TR_WIDTH, TR_OUT, TR_WIDTH, TR_VOCAB, TR_WIDTH,
                                                       TR_ROWS))
    for slug, ours_name, task, what, torch_text in (
            ("train-rms-norm-forward", "training.rms_norm_forward", "tr_rms_fwd",
             "rms_norm_forward(x, weight, eps)", "F.rms_norm(x, (D,), weight, eps)"),
            ("train-rms-norm-backward", "training.rms_norm_backward", "tr_rms_bwd",
             "rms_norm_backward(dy, x, weight, eps)", "F.rms_norm forward + autograd backward of dy"),
            ("train-linear-forward", "training.linear_forward", "tr_lin_fwd",
             "linear_forward(a, weight)", "F.linear(a, weight)"),
            ("train-linear-backward", "training.linear_backward", "tr_lin_bwd",
             "linear_backward(dc, a, weight)", "F.linear forward + autograd backward of dc"),
            ("train-embedding-forward", "training.embedding_forward", "tr_emb_fwd",
             "embedding_forward(weight, ids)", "F.embedding(ids, weight)"),
            ("train-embedding-backward", "training.embedding_backward", "tr_emb_bwd",
             "embedding_backward(dy, ids, vocab)", "F.embedding forward + autograd backward of dy"),
            ("train-residual-dropout", "training.residual_dropout", "tr_rdrop_fwd",
             "residual_dropout(values, residual, p=0.1)", "residual + F.dropout(values, 0.1)"),
            ("train-residual-dropout-backward", "training.residual_dropout_backward", "tr_rdrop_bwd",
             "residual_dropout_backward(gradient, p=0.1)", "autograd backward of residual + F.dropout"),
            ("train-chunked-lm-head-loss", "training.chunked_lm_head_loss", "tr_lmhead",
             "chunked_lm_head_loss(hidden, weight, targets, return_grad=True)",
             "per 1,024-row chunk: F.linear + F.cross_entropy(reduction='sum') + backward, d_hidden and "
             "d_weight accumulated, divided by the rows"),
            ("train-accumulate-grads", "training.accumulate_grads", "tr_accum",
             "accumulate_grads(8 microbatch gradients of 2,097,152 values, tokens=None)",
             "grad.add_(g) over the 8 microbatch gradients")):
        mism = []
        if task in ("tr_rdrop_fwd", "tr_rdrop_bwd"):
            mism.append("each library draws its dropout mask from its own generator (ours: the Philox "
                        "coordinates (seed 7, stream 0, offset); torch: torch.manual_seed(7)), so the dropped "
                        "elements differ; quality checks the deterministic part (kept = residual + "
                        "values / (1 - p), dropped = residual) and the dropped fraction against p")
        if task == "tr_lmhead":
            mism.append("torch materializes each 1,024-row chunk of logits (no fused linear cross-entropy "
                        "is pinned for the board); ours never holds the logits")
        if task == "tr_accum":
            mism.append("ours adds by a balanced tree in ascending microbatch order; torch's add_ is a "
                        "running sum; quality is against the float64 sum")
        add(slug, xlane="training", ours=(ours_name, ours_name.split(".", 1)[1]), kind="extra", task=task,
            block="tensor", datasets=("synthetic",),
            params=dict(rows=TR_ROWS, width=TR_WIDTH, out=TR_OUT, vocab=TR_VOCAB, eps=TR_EPS,
                        p=TR_DROPOUT_P, seed=7, microbatches=ACC_MICROBATCHES),
            quality="x-train", fit_text=what + ", host to host", torch_fp32=True,
            notes=[tr_note, "torch: " + torch_text], mism=mism)
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
    "x-lowbit": "fraction of output words equal to the contract's numpy reference (exact functions), "
                "max relative difference vs it",
    "x-train": "max relative difference of every output vs the float64 numpy reference (dropout: the "
               "kept/dropped contract error and the dropped fraction)",
}


def opponents(vendor, lane, spec, torch_gpu):
    if spec.get("torch_arms"):
        return tuple(spec["torch_arms"].get(vendor, ()))
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


def _text_arrays(np, docs):
    """[(name, array)]: the documents an arm is handed, as arrays the input record can hash: their
    UTF-8 bytes back to back and each document's byte length (the same lists every arm reads)."""
    raw = [d.encode("utf-8") for d in docs]
    return [("docs_utf8", np.frombuffer(b"".join(raw), dtype=np.uint8)),
            ("doc_bytes", np.array([len(b) for b in raw], dtype=np.int64))]


def input_arrays(lane, D, A, arrays=None):
    """The input-array record (bench_board_algos._input_array_records: name, shape, dtype, sha256) of the
    arrays lane `lane`'s arms are handed, the same names in the same order on every arm, so the board's
    measured-shape evidence (bench_board_evidence.measured_shape) and the parameter fairness check read
    them. The synthetic `tensor` and `corpus` blocks stage no dataset arrays, so without this record
    their cells carried no measured input shape (0.8.49 AMD board: algos/clip-grad-norm stopped on
    "incomplete worker evidence"). `arrays` passes arrays the arm already built (the clip gradients,
    the corpus documents); the rest are rebuilt by the same seeded functions the arms use. Hashed
    while the arm is built, before any clock."""
    np = A["_np"]()
    t = A["LANES"][lane]["task"]
    if arrays is None:
        if t == "qn":
            arrays = [("X", D["X"]), ("y", np.ascontiguousarray(D["y"], dtype=np.float32)), ("Xq", D["Xq"])]
        elif t in ("kpss", "selectd"):
            arrays = [("Y", np.ascontiguousarray(D["Yfit"], dtype=np.float32))]
        elif t == "xent":
            x, tg = _ce_inputs_once(np, A["SEED"], A)
            arrays = [("logits", x), ("targets", tg)]
        elif t == "sched":
            # the steps every arm evaluates the schedule at (1 .. SCHED_STEPS)
            arrays = [("steps", np.arange(1, SCHED_STEPS + 1, dtype=np.int64))]
        elif t == "jl":
            grid = jl_grid(np)
            arrays = [("n_samples", np.array([n for n, _ in grid], dtype=np.int64)),
                      ("eps", np.array([e for _, e in grid], dtype=np.float64))]
        else:
            raise SystemExit("bench_board_extra: no input record for lane %r task %r" % (lane, t))
    return A["_input_array_records"](arrays)


def _ce_inputs_once(np, seed, A):
    """_ce_inputs(np, seed). Inside the smoke worker (smoke-fixes r14, 2026-10-14; cross-entropy STARTED at 5 s
    in this 8192 x 8192 draw, once per arm) it is drawn once per process by the same function
    (smoke_inproc.staged, keyed by seed and size) and every arm reads the same arrays (ours passes x to the
    loss; the torch arms copy it to the device)."""
    SI = A["_staging"]()
    if SI is None:
        return _ce_inputs(np, seed)
    return SI.staged(("extra-ce-inputs", seed, CE_ROWS, CE_CLASSES), lambda: _ce_inputs(np, seed))




# ---------------------------------------------------------------------------
# measure-coverage (2026-10-22): the low-bit and training-function lanes
# ---------------------------------------------------------------------------

def _np_ftz(np, x):
    """Flush float32 subnormals to signed zero (the contract's ftz)."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(np.abs(x) < np.float32(2.0 ** -126), np.float32(0.0) * x, x).astype(np.float32)


def np_row_exponent(np, x, shift):
    """floor(log2 absmax) - shift per row, read from the exponent field; an all-zero row takes 0
    (contract L-3 / W-1)."""
    am = np.abs(_np_ftz(np, x)).max(axis=1)
    e = np.frexp(am)[1].astype(np.int64) - 1 - shift
    return np.where(am > 0, e, 0).astype(np.int32)


def np_quantize(np, x, shift, lim):
    """The contract's codes (L-4 / W-2), float32 math, round half to even, clamp; and the exponents."""
    xf = _np_ftz(np, x)
    e = np_row_exponent(np, xf, shift)
    scaled = _np_ftz(np, np.ldexp(xf, -e[:, None].astype(np.int64)).astype(np.float32))
    return np.clip(np.rint(scaled), -lim, lim).astype(np.int32), e


def lowbit_inputs(np, t, seed):
    """{name: array} every arm of a low-bit lane is handed (seeded; the same values on every arm)."""
    rng = np.random.default_rng(seed)
    if t in ("lowbit_quantize_int8", "lowbit_quantize_int15", "lowbit_to_bf16"):
        return {"x": rng.standard_normal((LB_ROWS, LB_COLS)).astype(np.float32)}
    if t == "lowbit_dequantize_int8":
        return {"codes": rng.integers(-127, 128, (LB_ROWS, LB_COLS)).astype(np.int8),
                "exponents": rng.integers(-12, 1, LB_ROWS).astype(np.int32)}
    if t == "lowbit_dequantize_int15":
        return {"hi": rng.integers(-128, 128, (LB_ROWS, LB_COLS)).astype(np.int8),
                "lo": rng.integers(0, 128, (LB_ROWS, LB_COLS)).astype(np.int8),
                "exponents": rng.integers(-20, -5, LB_ROWS).astype(np.int32)}
    if t == "lowbit_from_bf16":
        # the upper 16 bits of normal float32 values: finite bf16 bits
        x = rng.standard_normal((LB_ROWS, LB_COLS)).astype(np.float32)
        return {"bits": (x.view(np.uint32) >> 16).astype(np.uint16)}
    if t == "lowbit_matmul_int15":
        out = {}
        for side in ("a", "b"):
            c, e = np_quantize(np, rng.standard_normal((LB_ROWS, LB_COLS)).astype(np.float32), 13, 16383)
            out[side + "_hi"] = (c >> 7).astype(np.int8)
            out[side + "_lo"] = (c & 127).astype(np.int8)
            out[side + "_exponents"] = e
        return out
    raise ValueError("no low-bit inputs for %r" % (t,))


def train_inputs(np, t, seed):
    """{name: array} every arm of a training-function lane is handed (seeded)."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    if t in ("tr_rms_fwd", "tr_rms_bwd"):
        out = {"x": rng.standard_normal((TR_ROWS, TR_WIDTH)).astype(f32),
               "weight": (1.0 + 0.1 * rng.standard_normal(TR_WIDTH)).astype(f32)}
        if t == "tr_rms_bwd":
            out["dy"] = rng.standard_normal((TR_ROWS, TR_WIDTH)).astype(f32)
        return out
    if t in ("tr_lin_fwd", "tr_lin_bwd"):
        out = {"a": rng.standard_normal((TR_ROWS, TR_WIDTH)).astype(f32),
               "weight": (rng.standard_normal((TR_OUT, TR_WIDTH)) / np.sqrt(TR_WIDTH)).astype(f32)}
        if t == "tr_lin_bwd":
            out["dc"] = rng.standard_normal((TR_ROWS, TR_OUT)).astype(f32)
        return out
    if t in ("tr_emb_fwd", "tr_emb_bwd"):
        out = {"ids": rng.integers(0, TR_VOCAB, TR_ROWS).astype(np.int64)}
        if t == "tr_emb_fwd":
            out["weight"] = rng.standard_normal((TR_VOCAB, TR_WIDTH)).astype(f32)
        else:
            out["dy"] = rng.standard_normal((TR_ROWS, TR_WIDTH)).astype(f32)
        return out
    if t == "tr_rdrop_fwd":
        return {"values": rng.standard_normal((LB_ROWS, LB_COLS)).astype(f32),
                "residual": rng.standard_normal((LB_ROWS, LB_COLS)).astype(f32)}
    if t == "tr_rdrop_bwd":
        return {"gradient": rng.standard_normal((LB_ROWS, LB_COLS)).astype(f32)}
    if t == "tr_lmhead":
        return {"hidden": rng.standard_normal((TR_ROWS, TR_WIDTH)).astype(f32),
                "weight": (rng.standard_normal((TR_VOCAB, TR_WIDTH)) / np.sqrt(TR_WIDTH)).astype(f32),
                "targets": rng.integers(0, TR_VOCAB, TR_ROWS).astype(np.int64)}
    if t == "tr_accum":
        return {"grad%d" % i: rng.standard_normal(GRAD_SIZE).astype(f32) for i in range(ACC_MICROBATCHES)}
    raise ValueError("no training inputs for %r" % (t,))


def _build_lowbit_train(lane, arm, D, A, t, p, ours, fn, info, rec, Runner, S, np):
    """One arm of a low-bit or training-function lane: host arrays in, host arrays out, inside the clock."""
    SEED = A["SEED"]
    X = lowbit_inputs(np, t, SEED) if t.startswith("lowbit_") else train_inputs(np, t, SEED)
    rec["input_arrays"] = input_arrays(lane, D, A, list(X.items()))
    arr = A["_arr"]
    if ours:
        # the same ids / targets as int32 (ours' integer inputs; torch reads them as int64), cast before the clock
        I32 = {k: X[k].astype(np.int32) for k in ("ids", "targets") if k in X}

        def call():
            if t in ("lowbit_quantize_int8", "lowbit_to_bf16"):
                r = fn(X["x"])
            elif t == "lowbit_quantize_int15":
                r = fn(X["x"])
            elif t == "lowbit_dequantize_int8":
                r = fn(X["codes"], X["exponents"])
            elif t == "lowbit_dequantize_int15":
                r = fn(X["hi"], X["lo"], X["exponents"])
            elif t == "lowbit_from_bf16":
                r = fn(X["bits"])
            elif t == "lowbit_matmul_int15":
                r = fn((X["a_hi"], X["a_lo"], X["a_exponents"]), (X["b_hi"], X["b_lo"], X["b_exponents"]))
            elif t == "tr_rms_fwd":
                r = fn(X["x"], X["weight"], p["eps"])
            elif t == "tr_rms_bwd":
                r = fn(X["dy"], X["x"], X["weight"], p["eps"])
            elif t == "tr_lin_fwd":
                r = fn(X["a"], X["weight"])
            elif t == "tr_lin_bwd":
                r = fn(X["dc"], X["a"], X["weight"])
            elif t == "tr_emb_fwd":
                r = fn(X["weight"], I32["ids"])
            elif t == "tr_emb_bwd":
                r = fn(X["dy"], I32["ids"], p["vocab"])
            elif t == "tr_rdrop_fwd":
                r = fn(X["values"], X["residual"], p=p["p"], seed=p["seed"], stream=0, offset=0)
            elif t == "tr_rdrop_bwd":
                r = fn(X["gradient"], p=p["p"], seed=p["seed"], stream=0, offset=0)
            elif t == "tr_lmhead":
                r = fn(X["hidden"], X["weight"], I32["targets"], return_grad=True)
            elif t == "tr_accum":
                r = fn([X["grad%d" % i] for i in range(ACC_MICROBATCHES)], None)
            else:
                raise ValueError(t)
            return r

        def fit():
            r = call()
            if t == "tr_lmhead":
                S["o"] = {"loss": np.array([float(r[0])]), "d_hidden": arr(r[1], np.float32),
                          "d_weight": arr(r[2], np.float32)}
            elif isinstance(r, tuple):
                S["o"] = {"out%d" % i: np.asarray(arr(v)) for i, v in enumerate(r)}
            else:
                S["o"] = {"out0": np.asarray(arr(r))}
        return Runner(info, fit, lambda: S["o"], record=rec)

    import torch
    dev = A["_torch_device"](torch)
    mode = arm.split("-")[1]
    setting = arm[len("torch-"):]
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info.update(library="torch", version=torch.__version__, device="gpu" if dev != "cpu" else "cpu",
                setting=setting, input_home="host",
                inside_clock="host-to-device copies, the operation, device-to-host copies")
    sync = (lambda: A["_torch_sync"](torch, dev))
    F = torch.nn.functional
    tiny = 2.0 ** -126

    def up(a):
        if a.dtype == np.uint16:          # bf16 bits: torch reads them as int16 words (the same 16 bits)
            a = a.view(np.int16)
        return torch.from_numpy(a).to(dev)

    def ftz(x):
        return torch.where(x.abs() < tiny, x * 0.0, x)

    def row_exp(x, shift):
        am = ftz(x).abs().amax(dim=1)
        e = torch.frexp(am)[1].to(torch.int32) - 1 - shift
        return torch.where(am > 0, e, torch.zeros_like(e))

    def quant(x, shift, lim):
        xf = ftz(x)
        e = row_exp(xf, shift)
        scaled = ftz(torch.ldexp(xf, (-e).to(torch.float32)[:, None]))
        return torch.clamp(torch.round(scaled), -lim, lim).to(torch.int32), e

    def op(**T):
        if t == "lowbit_quantize_int8":
            c, e = quant(T["x"], 6, 127)
            return c.to(torch.int8), e
        if t == "lowbit_quantize_int15":
            c, e = quant(T["x"], 13, 16383)
            return (c >> 7).to(torch.int8), (c & 127).to(torch.int8), e
        if t == "lowbit_dequantize_int8":
            return ftz(torch.ldexp(T["codes"].to(torch.float32), T["exponents"].to(torch.float32)[:, None]))
        if t == "lowbit_dequantize_int15":
            c = T["hi"].to(torch.int32) * 128 + T["lo"].to(torch.int32)
            return ftz(torch.ldexp(c.to(torch.float32), T["exponents"].to(torch.float32)[:, None]))
        if t == "lowbit_to_bf16":
            return ftz(T["x"]).to(torch.bfloat16).view(torch.int16)
        if t == "lowbit_from_bf16":
            return T["bits"].view(torch.bfloat16).to(torch.float32)
        if t == "lowbit_matmul_int15":
            if setting == "eager-fp64":
                ca = (T["a_hi"].to(torch.float64) * 128 + T["a_lo"].to(torch.float64))
                cb = (T["b_hi"].to(torch.float64) * 128 + T["b_lo"].to(torch.float64))
                exact = ca @ cb.t()
            else:
                mm = torch._int_mm
                ah, al, bh, bl = T["a_hi"], T["a_lo"], T["b_hi"].t(), T["b_lo"].t()
                exact = ((mm(ah, bh).to(torch.int64) << 14)
                         + ((mm(ah, bl).to(torch.int64) + mm(al, bh).to(torch.int64)) << 7)
                         + mm(al, bl).to(torch.int64)).to(torch.float64)
            ee = (T["a_exponents"][:, None] + T["b_exponents"][None, :]).to(torch.float32)
            return ftz(torch.ldexp(exact.to(torch.float32), ee))
        if t == "tr_rms_fwd":
            return F.rms_norm(T["x"], (T["x"].shape[1],), T["weight"], p["eps"])
        if t == "tr_rms_bwd":
            x = T["x"].requires_grad_(True)
            w = T["weight"].requires_grad_(True)
            y = F.rms_norm(x, (x.shape[1],), w, p["eps"])
            dx, dw = torch.autograd.grad(y, (x, w), T["dy"])
            return dx, dw
        if t == "tr_lin_fwd":
            return F.linear(T["a"], T["weight"])
        if t == "tr_lin_bwd":
            a = T["a"].requires_grad_(True)
            w = T["weight"].requires_grad_(True)
            da, dw = torch.autograd.grad(F.linear(a, w), (a, w), T["dc"])
            return da, dw
        if t == "tr_emb_fwd":
            return F.embedding(T["ids"], T["weight"])
        if t == "tr_emb_bwd":
            w = torch.zeros((p["vocab"], T["dy"].shape[1]), device=dev, requires_grad=True)
            (dw,) = torch.autograd.grad(F.embedding(T["ids"], w), (w,), T["dy"])
            return dw
        if t == "tr_rdrop_fwd":
            return T["residual"] + F.dropout(T["values"], p=p["p"], training=True)
        if t == "tr_rdrop_bwd":
            v = torch.zeros_like(T["gradient"], requires_grad=True)
            r = torch.zeros_like(T["gradient"], requires_grad=True)
            dv, dr = torch.autograd.grad(r + F.dropout(v, p=p["p"], training=True), (v, r), T["gradient"])
            return dv, dr
        if t == "tr_lmhead":
            h, w, tg = T["hidden"], T["weight"], T["targets"]
            dh = torch.empty_like(h)
            dw = torch.zeros_like(w)
            loss = torch.zeros((), device=dev, dtype=torch.float32)
            n = h.shape[0]
            for st in range(0, n, LM_HEAD_CHUNK):   # driver-loop: over torch's logit chunks
                hc = h[st:st + LM_HEAD_CHUNK].detach().requires_grad_(True)
                wc = w.detach().requires_grad_(True)
                lc = F.cross_entropy(F.linear(hc, wc), tg[st:st + LM_HEAD_CHUNK], reduction="sum")
                ghc, gwc = torch.autograd.grad(lc, (hc, wc))
                dh[st:st + LM_HEAD_CHUNK] = ghc / n
                dw += gwc / n
                loss = loss + lc.detach()
            return loss / n, dh, dw
        if t == "tr_accum":
            g = T["grad0"].clone()
            for i in range(1, ACC_MICROBATCHES):   # driver-loop: over the microbatches
                g.add_(T["grad%d" % i])
            return g
        raise ValueError(t)

    if t == "lowbit_matmul_int15" and setting == "eager-int8" and not hasattr(torch, "_int_mm"):
        raise RuntimeError("REFUSED: torch %s has no torch._int_mm" % torch.__version__)
    run = torch.compile(op) if mode == "compile" else op
    info["config"] = "torch twin of %s on %s, %s (TF32 off)" % (s_name(A, lane), dev, setting)

    def fit():
        T = {k: up(v) for k, v in X.items()}
        r = run(**T)
        r = r if isinstance(r, tuple) else (r,)
        host = [v.detach().cpu().numpy() for v in r]
        if t == "tr_lmhead":
            S["o"] = {"loss": np.array([float(host[0])]), "d_hidden": host[1], "d_weight": host[2]}
        else:
            if t == "lowbit_to_bf16":
                host[0] = host[0].view(np.uint16)
            S["o"] = {"out%d" % i: v for i, v in enumerate(host)}
    return Runner(info, fit, lambda: S["o"], sync=sync, record=dict(
        rec, __implementation__=getattr(F, {"tr_rms_fwd": "rms_norm", "tr_rms_bwd": "rms_norm",
                                             "tr_lin_fwd": "linear", "tr_lin_bwd": "linear",
                                             "tr_emb_fwd": "embedding", "tr_emb_bwd": "embedding",
                                             "tr_rdrop_fwd": "dropout", "tr_rdrop_bwd": "dropout",
                                             "tr_lmhead": "cross_entropy"}.get(t, "linear")) if t.startswith("tr_")
        and t != "tr_accum" else (torch.Tensor.add_ if t == "tr_accum" else
                                  (torch._int_mm if setting == "eager-int8" else torch.ldexp)),
        seed=SEED, **p))


def s_name(A, lane):
    return A["LANES"][lane]["ours"][0]


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

    # ---- measure-coverage (2026-10-22): low-bit linalg and training-function lanes
    if t.startswith("lowbit_") or t.startswith("tr_"):
        return _build_lowbit_train(lane, arm, D, A, t, p, ours, fn if ours else None, info, rec, Runner, S, np)

    # ---- QN regression
    if t == "qn":
        X, y, Xq = D["X"], np.ascontiguousarray(D["y"], dtype=np.float32), D["Xq"]
        ins = input_arrays(lane, D, A)
        if ours:
            rec["input_arrays"] = ins
            def fit():
                m = fn(**p).fit(X, y)
                S["pred"] = A["_arr"](m.predict(Xq), np.float32)
            return Runner(info, fit, lambda: {"pred": S["pred"]}, record=rec)
        if arm == "sklearn-cpu":
            # FAIRNESS FIX (2026-10-13 audit): scikit-learn's L-BFGS on the squared loss, not the
            # closed-form LinearRegression (see the lane's notes). Results raced before this change
            # were wrong for this lane; apply extra scrutiny.
            from sklearn.linear_model import TweedieRegressor
            kw = dict(power=0.0, link="identity", alpha=0.0, solver="lbfgs",
                      max_iter=p["max_iter"], tol=p["tol"], fit_intercept=p["fit_intercept"],
                      warm_start=False)
            est = TweedieRegressor(**kw)
            info["config"] = "sklearn.linear_model.TweedieRegressor(%s)" % ", ".join(
                "%s=%r" % kv for kv in sorted(kw.items()))

            def fit():
                m = TweedieRegressor(**kw).fit(X, y)
                S["pred"] = m.predict(Xq).astype(np.float32)
            r = A["_BP"]().arm_record(est)
            r["params"]["input_arrays"] = ins
            return Runner(info, fit, lambda: {"pred": S["pred"]}, record=r)
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
        r = A["_BP"]().arm_record(est)
        r["params"]["input_arrays"] = ins
        return Runner(info, fit, lambda: {"pred": S["pred"]}, sync=cupy_sync, record=r)

    # ---- KPSS and select_d
    if t in ("kpss", "selectd"):
        Y = np.ascontiguousarray(D["Yfit"], dtype=np.float32)       # (series, n)
        n = Y.shape[1]
        rec["input_arrays"] = input_arrays(lane, D, A)
        # measure-coverage (2026-10-24): one explicit record on both arms. Ours' KPSS is level-stationarity
        # (regression 'c') with the ceil(12 (n/100)^0.25) lag count, the values the statsmodels arm is
        # given; the statsmodels arm tests the undifferenced series (d = D = s = 0, kpss) or d = 0, 1 in
        # order (select_d), the values ours is given. Before, ours read back d, D, s and statsmodels
        # regression, nlags, so the parameter check refused the pair on names present on one side only.
        rec.update(regression="c", nlags=kpss_lags(n))
        if t == "selectd":
            rec["d_candidates"] = [0, 1]
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
        rec.update(p, __implementation__=kpss)
        return Runner(info, fit, lambda: S["o"], record=rec)

    # ---- clip_grad_norm_ and cross_entropy (torch at fp32, eager and compile)
    if t in ("clip", "xent"):
        if ours:
            if t == "clip":
                G = _grads(np, SEED)
                G0 = [g.copy() for g in G]          # FAIRNESS FIX (2026-10-13 audit): the unclipped
                rec["input_arrays"] = input_arrays(lane, D, A, [("grad%d" % i, g) for i, g in enumerate(G)])

                def fit():
                    S["norm"] = float(fn(G, p["max_norm"], norm_type=p["norm_type"],
                                         error_if_nonfinite=p["error_if_nonfinite"]))

                def outputs():
                    # after the round's clocks: put the unclipped gradients back for the next round
                    for g, g0 in zip(G, G0):
                        np.copyto(g, g0)
                    return {"norm": np.array([S["norm"]])}
                return Runner(info, fit, outputs, record=rec)
            x, tg = _ce_inputs_once(np, SEED, A)
            tg32 = tg.astype(np.int32)
            # the drawn logits and int64 targets, as on the torch arms (ours hands the same values as int32)
            rec["input_arrays"] = input_arrays(lane, D, A)

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
            rec["input_arrays"] = input_arrays(lane, D, A, [("grad%d" % i, g) for i, g in enumerate(G)])
            params = [torch.nn.Parameter(torch.zeros(GRAD_SIZE, device=dev)) for _ in G]
            for q, g in zip(params, G):
                q.grad = torch.from_numpy(g).to(dev)
            # FAIRNESS FIX (2026-10-13 audit): the unclipped gradients, put back after each round
            # (outside the clock) so every round clips the same unclipped arrays. Results raced
            # before this change were wrong for this lane; apply extra scrutiny.
            G0_dev = [q.grad.detach().clone() for q in params]

            def clip():
                return torch.nn.utils.clip_grad_norm_(params, p["max_norm"], norm_type=p["norm_type"],
                                                      error_if_nonfinite=p["error_if_nonfinite"])
            run = torch.compile(clip) if mode == "compile" else clip
            info["config"] = "torch.nn.utils.clip_grad_norm_(8 grads, 1.0, 2.0) on %s, %s" % (dev, mode)

            def fit():
                S["norm"] = float(run())

            def outputs():
                with torch.no_grad():
                    for q, g0 in zip(params, G0_dev):
                        q.grad.copy_(g0)
                sync()
                return {"norm": np.array([S["norm"]])}
            # smoke-fixes (2026-10-14; algos/clip-grad-norm: "missing real library implementation read-back:
            # torch-*"): the record names the torch function the arm really calls
            return Runner(info, fit, outputs, sync=sync, record=dict(
                rec, __implementation__=torch.nn.utils.clip_grad_norm_, seed=SEED, **p))
        x_np, tg = _ce_inputs_once(np, SEED, A)
        rec["input_arrays"] = input_arrays(lane, D, A)
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
            # FAIRNESS FIX (2026-10-13 audit): the whole gradient comes to the host inside the
            # clock, as ours' host return does (it copied 256 rows). Results raced before this
            # change were wrong for this lane; apply extra scrutiny.
            g_host = xt.grad.float().cpu().numpy()
            S["o"] = {"loss": np.array([float(loss.item())]), "grad": g_host[:256].copy()}
        # smoke-fixes (2026-10-14; algos/cross-entropy: "missing real library implementation read-back:
        # torch-*"): the record names the torch function the arm really calls
        return Runner(info, fit, lambda: S["o"], sync=sync, record=dict(
            rec, __implementation__=F.cross_entropy, seed=SEED, **p))

    # ---- BPE encode and train
    if t in ("bpeencode", "bpetrain"):
        docs, train = corpus_docs(A)
        # the documents the timed call reads: encode the 2,048-character documents, train the first 1 MiB
        rec["input_arrays"] = input_arrays(lane, D, A, _text_arrays(np, docs if t == "bpeencode" else train))
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
                # measure-coverage (2026-10-24): the same vocabulary settings (ours' vocabulary, trained
                # before the clock, is the one both arms encode with) and the library call, read back
                rec.update(p, __implementation__=tokenizers.Tokenizer.encode_batch)

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
            # measure-coverage (2026-10-24): BpeTrainer takes the lane's vocab_size and min_frequency
            # explicitly (above); read back with the library class
            rec.update(p, __implementation__=trainers.BpeTrainer)

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
        rec["input_arrays"] = input_arrays(lane, D, A)
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
        # measure-coverage (2026-10-24): the torch scheduler class, for the real-implementation check
        impl = {"lr-step": L.StepLR, "lr-exponential": L.ExponentialLR, "lr-onecycle": L.OneCycleLR,
                "lr-constant": L.LinearLR}.get(lane, L.SequentialLR)
        return Runner(info, fit, lambda: {"lr": S["lr"]}, record=dict(rec, __implementation__=impl, **p))

    # ---- Johnson-Lindenstrauss bound
    if t == "jl":
        grid = jl_grid(np)
        rec["input_arrays"] = input_arrays(lane, D, A)
        if ours:
            def fit():
                S["k"] = np.array([fn(n, eps=e) for n, e in grid], dtype=np.int64)
            return Runner(info, fit, lambda: {"k": S["k"]}, record=rec)
        from sklearn.random_projection import johnson_lindenstrauss_min_dim as jl
        info["config"] = "sklearn.random_projection.johnson_lindenstrauss_min_dim(n, eps=e) per point"

        def fit():
            S["k"] = np.array([int(jl(n, eps=e)) for n, e in grid], dtype=np.int64)
        # measure-coverage (2026-10-24): the library function, for the real-implementation check
        return Runner(info, fit, lambda: {"k": S["k"]}, record=dict(rec, __implementation__=jl))
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
            elif t.startswith("lowbit_") or t.startswith("tr_"):
                e.update(_lowbit_train_quality(np, t, s["params"], o, A["SEED"]))
            elif t == "jl":
                sk = outs.get("sklearn-cpu")
                if sk is not None:
                    e["equal_fraction_vs_sklearn"] = float((o["k"] == sk["k"]).mean())
        except Exception as exc:  # noqa: BLE001
            e["quality_error"] = repr(exc)[:300]
    return q


def _relmax(np, a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-30))


def _lowbit_train_quality(np, t, p, o, seed):
    """Every output of one arm against the contract's numpy reference (low-bit: exact words; training:
    float64)."""
    e = {}
    if t.startswith("lowbit_"):
        X = lowbit_inputs(np, t, seed)
        if t == "lowbit_quantize_int8":
            c, ex = np_quantize(np, X["x"], 6, 127)
            ref = [c.astype(np.int8), ex]
        elif t == "lowbit_quantize_int15":
            c, ex = np_quantize(np, X["x"], 13, 16383)
            ref = [(c >> 7).astype(np.int8), (c & 127).astype(np.int8), ex]
        elif t == "lowbit_dequantize_int8":
            ref = [np.ldexp(X["codes"].astype(np.float32), X["exponents"][:, None]).astype(np.float32)]
        elif t == "lowbit_dequantize_int15":
            c = X["hi"].astype(np.int32) * 128 + X["lo"].astype(np.int32)
            ref = [np.ldexp(c.astype(np.float32), X["exponents"][:, None]).astype(np.float32)]
        elif t == "lowbit_to_bf16":
            u = _np_ftz(np, X["x"]).view(np.uint32).astype(np.uint64)
            r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16          # round to nearest even on the low 16 bits
            ref = [r.astype(np.uint16)]
        elif t == "lowbit_from_bf16":
            ref = [(X["bits"].astype(np.uint32) << 16).view(np.float32)]
        else:
            ca = X["a_hi"].astype(np.float64) * 128 + X["a_lo"].astype(np.float64)
            cb = X["b_hi"].astype(np.float64) * 128 + X["b_lo"].astype(np.float64)
            exact = (ca @ cb.T).astype(np.float32)
            ee = X["a_exponents"][:, None].astype(np.int64) + X["b_exponents"][None, :].astype(np.int64)
            ref = [_np_ftz(np, np.ldexp(exact, ee).astype(np.float32))]
        eq = []
        for i, r in enumerate(ref):
            got = np.ascontiguousarray(o["out%d" % i])
            r = np.ascontiguousarray(r)
            eq.append(float((got.reshape(-1).view(np.uint8).reshape(got.size, -1)
                             == np.asarray(r, dtype=got.dtype).reshape(-1).view(np.uint8).reshape(got.size, -1))
                            .all(1).mean()) if got.shape == r.shape else 0.0)
            if np.issubdtype(np.asarray(r).dtype, np.floating):
                e["max_rel_diff_vs_contract_out%d" % i] = _relmax(np, got, r)
        e["equal_word_fraction_vs_contract"] = min(eq)
        return e
    X = train_inputs(np, t, seed)
    f64 = np.float64
    if t == "tr_rms_fwd":
        x, w = X["x"].astype(f64), X["weight"].astype(f64)
        r = np.sqrt((x * x).mean(1, keepdims=True) + p["eps"])
        e["max_rel_diff_vs_fp64"] = _relmax(np, o["out0"], w * x / r)
    elif t == "tr_rms_bwd":
        x, w, dy = X["x"].astype(f64), X["weight"].astype(f64), X["dy"].astype(f64)
        r = np.sqrt((x * x).mean(1, keepdims=True) + p["eps"])
        xn = x / r
        g = dy * w
        dx = (g - xn * (g * xn).mean(1, keepdims=True)) / r
        dw = (dy * xn).sum(0)
        e["dx_max_rel_diff_vs_fp64"] = _relmax(np, o["out0"], dx)
        e["dweight_max_rel_diff_vs_fp64"] = _relmax(np, o["out1"], dw)
    elif t == "tr_lin_fwd":
        e["max_rel_diff_vs_fp64"] = _relmax(np, o["out0"], X["a"].astype(f64) @ X["weight"].astype(f64).T)
    elif t == "tr_lin_bwd":
        dc = X["dc"].astype(f64)
        e["da_max_rel_diff_vs_fp64"] = _relmax(np, o["out0"], dc @ X["weight"].astype(f64))
        e["dweight_max_rel_diff_vs_fp64"] = _relmax(np, o["out1"], dc.T @ X["a"].astype(f64))
    elif t == "tr_emb_fwd":
        e["max_rel_diff_vs_fp64"] = _relmax(np, o["out0"], X["weight"][X["ids"]])
    elif t == "tr_emb_bwd":
        ref = np.zeros((p["vocab"], X["dy"].shape[1]))
        np.add.at(ref, X["ids"], X["dy"].astype(f64))
        e["max_rel_diff_vs_fp64"] = _relmax(np, o["out0"], ref)
    elif t in ("tr_rdrop_fwd", "tr_rdrop_bwd"):
        v = X["values"] if t == "tr_rdrop_fwd" else X["gradient"]
        base = X["residual"] if t == "tr_rdrop_fwd" else np.zeros_like(v)
        dropped_part = np.asarray(o["out0"], dtype=f64) - base.astype(f64)
        kept = dropped_part != 0
        scale = 1.0 / (1.0 - p["p"])
        err = np.abs(dropped_part[kept] - v.astype(f64)[kept] * scale)
        e["kept_contract_max_rel_err"] = float(err.max() / max(float(np.abs(v).max() * scale), 1e-30)) if kept.any() else 0.0
        e["dropped_fraction"] = float(1.0 - kept.mean())
        e["dropped_fraction_minus_p"] = float((1.0 - kept.mean()) - p["p"])
        if t == "tr_rdrop_bwd":
            e["d_residual_max_rel_diff"] = _relmax(np, o["out1"], X["gradient"])
    elif t == "tr_lmhead":
        h, w, tg = X["hidden"].astype(f64), X["weight"].astype(f64), X["targets"]
        n = h.shape[0]
        loss = 0.0
        dh = np.empty_like(h)
        dw = np.zeros_like(w)
        for st in range(0, n, LM_HEAD_CHUNK):
            z = h[st:st + LM_HEAD_CHUNK] @ w.T
            z -= z.max(1, keepdims=True)
            pz = np.exp(z)
            pz /= pz.sum(1, keepdims=True)
            idx = np.arange(z.shape[0])
            loss -= np.log(pz[idx, tg[st:st + LM_HEAD_CHUNK]]).sum()
            pz[idx, tg[st:st + LM_HEAD_CHUNK]] -= 1.0
            pz /= n
            dh[st:st + LM_HEAD_CHUNK] = pz @ w
            dw += pz.T @ h[st:st + LM_HEAD_CHUNK]
        e["loss_rel_err_vs_fp64"] = abs(float(o["loss"][0]) - loss / n) / abs(loss / n)
        e["d_hidden_max_rel_diff_vs_fp64"] = _relmax(np, o["d_hidden"], dh)
        e["d_weight_max_rel_diff_vs_fp64"] = _relmax(np, o["d_weight"], dw)
    elif t == "tr_accum":
        ref = sum(X["grad%d" % i].astype(f64) for i in range(ACC_MICROBATCHES))
        e["max_rel_diff_vs_fp64"] = _relmax(np, o["out0"], ref)
    return e
