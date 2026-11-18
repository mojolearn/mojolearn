#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/measure_owed_driver.py -- the owed rows the board cannot express (bench/results/paper_compare/
owed-pre0867.json, kind "driver"), OUR ARM ONLY, GPU only, one warm-up and one timed call per case.

    <arm venv>/bin/python tools/measure_owed_driver.py --owed JSON --row ID --arm candidate|prior --out FILE.jsonl

Run by tools/ops/paper_compare_job.sh with each arm's interpreter (the candidate and the published prior ours-only
venvs) and MOJOLEARN_NUMERIC_MODE=identical in the environment; it appends one JSON line per case:
{row, arm, case, ms, status, version, numeric_mode}. A case that raises is recorded REFUSED with the error and the
next case runs. Drivers (the row's "driver" field):

  while_loop    Program.while_loop of a tanh(x @ w) body for `iterations` steps (the condition never ends the loop),
                once per check_every value the row's arms name (None = the loop's own choice, 1, 8)
  pandas_input  each estimator's fit on the same float32 data as an ndarray and as a pandas DataFrame
  mlp_fit       MLPClassifier.fit on a non-board shape
  kde_widths    KernelDensity fit + score_samples on synthetic rows at each width the row names (the KDE_QTILE
                cost rule's neighbouring and wide non-board shapes)
  mamba_backward  Mamba1Block.backward and Mamba2Block.backward (the zero-state block VJP, one public call) on
                synthetic weights at the row's (batch, length, d_model); an arm may be a B wheel set the row names
  samba_train_step  SambaStack.train_step (forward, loss, tape backward, AdamW; one public call) on a stack
                initialized from a seeded Generator, once per case the row's shape names (batch, length, d_model,
                layers): neighbouring and non-board model sizes for a Mamba-3 backward change; each line carries
                the step's loss beside its time; an arm may be a B wheel set the row names
  samba_step    SambaStack.train_step, then the first SambaStack.forward after a step (the weights it left), on a
                stack the library initializes itself, at each (batch, length, d_model, layers) case the row
                names: the neighbouring and non-board shapes of the Samba step arms; an arm may be a B wheel set
  cnn_widths    CNNClassifier.fit on synthetic images at each conv_channels tuple the row names; an arm may be a
                B wheel set the row names
  f16_widen     models.safetensors.widen_f16 of synthetic float16 bits at each element count the row names; an
                arm may be a B wheel set the row names
  gemm_shapes   linalg.matmul(a, b) (fp32, A [m, k] @ B [k, n], seeded standard normal) at each (m, n, k) case the
                row names: the neighbouring and non-board shapes of a GEMM kernel arm; an arm may be a B wheel set

Runs nothing at import; standard library + numpy + pandas + mojolearn in the arm's venv only (it never imports the
repo's tools). Serial by nature: the cases are timed calls on one GPU and must not share it.
"""
import argparse
import json
import os
import sys
import time


def _ml():
    import mojolearn as ml
    return ml


def _numeric_mode(ml):
    fn = getattr(ml, "numeric_mode", None)
    try:
        return fn() if callable(fn) else os.environ.get("MOJOLEARN_NUMERIC_MODE")
    except Exception as exc:  # noqa: BLE001 -- recorded, not fatal
        return "unknown (%s)" % exc


def timed(call):
    """(ms of the second call, None) after one warm-up, or (None, 'REFUSED: ...')."""
    try:
        call()
        t0 = time.perf_counter()
        call()
        return (time.perf_counter() - t0) * 1000.0, None
    except Exception as exc:  # noqa: BLE001 -- a refusal is a result
        return None, "REFUSED: %s: %s" % (type(exc).__name__, str(exc)[:400])


def case_while_loop(row, arm_cases):
    import numpy as np
    from mojolearn import ops
    sh = row["shape"]
    n, d, iters = int(sh["rows"]), int(sh["features"]), int(sh["iterations"])
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((n, d)).astype(np.float32)
    w0 = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = []
    with ops.Session() as s:
        x = s.tensor(x0)
        w = s.tensor(w0)
        body = s.capture(x, s.tanh(s.matmul(x, w)))
        # tanh keeps |x| <= 1, so the mean stays below 2: the condition holds and max_iterations ends the loop
        cond = s.capture(x, s.less(s.mean(x), s.full((1, 1), 2.0)))
        for case in arm_cases:
            every = None if case in ("None", "none", "") else int(case)
            # check_every only when the case names one: the prior wheel (0.8.66) has no such keyword, and its
            # arm runs the loop as that version does (lane candidate-build)
            kw = {} if every is None else {"check_every": every}
            ms, why = timed(lambda: body.while_loop(x0, condition=cond, max_iterations=iters, **kw))
            out.append(("check_every=%s" % case, ms, why))
    return out


def case_pandas_input(row, arm_cases):
    import numpy as np
    ml = _ml()
    sh = row["shape"]
    n, d = int(sh["rows"]), int(sh["features"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y_reg = (X[:, 0] * 2.0 + X[:, 1]).astype(np.float32)
    y_clf = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(np.int32)
    frames = {}

    def data(kind):
        if kind == "ndarray":
            return X
        if "df" not in frames:
            import pandas as pd
            frames["df"] = pd.DataFrame(X, columns=["f%d" % i for i in range(d)])
        return frames["df"]
    make = {
        "Ridge": (lambda: ml.Ridge(alpha=1.0), y_reg),
        "LogisticRegression": (lambda: ml.LogisticRegression(max_iter=100), y_clf),
        "KMeans": (lambda: ml.KMeans(n_clusters=8, n_init=1, random_state=0), None),
        "StandardScaler": (lambda: ml.StandardScaler(), None),
        "RandomForestClassifier": (lambda: ml.RandomForestClassifier(n_estimators=32, max_depth=8,
                                                                     random_state=0), y_clf),
        "KNeighborsClassifier": (lambda: ml.KNeighborsClassifier(n_neighbors=5), y_clf),
    }
    out = []
    for est in row.get("estimators") or sorted(make):
        if est not in make:
            out.append((est, None, "REFUSED: no driver case for %s" % est))
            continue
        ctor, y = make[est]
        for kind in arm_cases:
            def call(ctor=ctor, y=y, kind=kind):
                m = ctor()
                return m.fit(data(kind)) if y is None else m.fit(data(kind), y)
            ms, why = timed(call)
            out.append(("%s %s" % (est, kind), ms, why))
    return out


def case_mlp_fit(row, arm_cases):
    import numpy as np
    ml = _ml()
    sh = row["shape"]
    n, d, k = int(sh["rows"]), int(sh["features"]), int(sh["classes"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (np.abs(X[:, :k]).argmax(axis=1)).astype(np.int32)
    hidden = tuple(int(h) for h in sh["hidden"])
    ms, why = timed(lambda: ml.MLPClassifier(hidden_layer_sizes=hidden, max_iter=int(sh["max_iter"]),
                                             random_state=0).fit(X, y))
    return [("MLPClassifier hidden=%s n=%d d=%d" % (list(hidden), n, d), ms, why)]


def case_kde_widths(row, arm_cases):
    import numpy as np
    ml = _ml()
    sh = row["shape"]
    n, q = int(sh["rows"]), int(sh["queries"])
    out = []
    for d in sh["features"]:  # serial by nature: timed calls on one GPU
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, int(d))).astype(np.float32)
        Q = X[:q].copy()
        ms, why = timed(lambda: ml.KernelDensity(bandwidth=1.0).fit(X).score_samples(Q))
        out.append(("KernelDensity fit+score_samples n=%d q=%d d=%d" % (n, q, int(d)), ms, why))
    return out


def case_mamba_backward(row, arm_cases):
    """The public Mamba-1 / Mamba-2 block backward (the block step op's VJP) on synthetic float32 weights in the
    blocks' documented shapes, x and dy of shape (batch, length, d_model)."""
    import numpy as np
    ml = _ml()
    sh = row["shape"]
    b, l, dm = int(sh["batch"]), int(sh["length"]), int(sh["d_model"])
    rng = np.random.default_rng(0)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    x = (rng.standard_normal((b, l, dm)) * 0.5).astype(np.float32)
    dy = (rng.standard_normal((b, l, dm)) * 0.5).astype(np.float32)
    out = []
    di, r = 2 * dm, -(-dm // 16)
    m1 = {"norm.weight": np.ones((dm,), np.float32), "in_proj.weight": w(2 * di, dm),
          "conv1d.weight": w(di, 1, 4), "conv1d.bias": w(di), "x_proj.weight": w(r + 32, di),
          "dt_proj.weight": w(di, r), "dt_proj.bias": w(di),
          "A_log": np.log(np.tile(np.arange(1, 17, dtype=np.float32), (di, 1))), "D": np.ones((di,), np.float32),
          "out_proj.weight": w(dm, di)}
    block1 = ml.Mamba1Block(m1, numeric_mode="identical")
    ms, why = timed(lambda: block1.backward(x, dy))
    out.append(("Mamba1Block.backward b=%d l=%d d_model=%d" % (b, l, dm), ms, why))
    nh, cd = di // 64, di + 256
    m2 = {"block_norm.weight": np.ones((dm,), np.float32), "in_proj.weight": w(2 * di + 256 + nh, dm),
          "conv1d.weight": w(cd, 1, 4), "conv1d.bias": w(cd), "dt_bias": w(nh),
          "A_log": np.log(np.arange(1, nh + 1, dtype=np.float32)), "D": np.ones((nh,), np.float32),
          "norm.weight": np.ones((di,), np.float32), "out_proj.weight": w(dm, di)}
    block2 = ml.Mamba2Block(m2, numeric_mode="identical")
    ms, why = timed(lambda: block2.backward(x, dy))
    out.append(("Mamba2Block.backward b=%d l=%d d_model=%d" % (b, l, dm), ms, why))
    return out


def case_samba_train_step(row, arm_cases):
    """SambaStack.train_step at each case of the row's shape: {"name", "batch", "length", "d_model", "layers",
    and for attention layers "n_heads" and "intermediate"; optional "vocab" (256)}. The stack is initialized from
    Generator(7) (the same parameters on every arm), the ids from default_rng(0). The loss recorded is the timed
    (second) step's, so two arms that differ only in a summation order show it beside the time."""
    import numpy as np
    from mojolearn.training import Generator
    ml = _ml()
    out = []
    for spec in row["shape"]["cases"]:  # serial by nature: timed calls on one GPU
        b, l, dm = int(spec["batch"]), int(spec["length"]), int(spec["d_model"])
        vocab = int(spec.get("vocab", 256))
        layers = tuple(str(v) for v in spec["layers"])
        label = "SambaStack.train_step %s b=%d l=%d d_model=%d layers=%d" % (spec["name"], b, l, dm, len(layers))
        seen = {}
        try:
            cfg = ml.SambaConfig(vocab=vocab, d_model=dm, layers=layers, n_heads=spec.get("n_heads"),
                                 intermediate=spec.get("intermediate"))
            stack = ml.SambaStack(cfg, generator=Generator(7), lr=1e-3, numeric_mode="identical")
            ids = np.random.default_rng(0).integers(0, vocab, (b, l + 1)).astype(np.int32)
            x, y = np.ascontiguousarray(ids[:, :-1]), np.ascontiguousarray(ids[:, 1:])
        except Exception as exc:  # noqa: BLE001 -- a refusal is a result
            out.append((label, None, "REFUSED: %s: %s" % (type(exc).__name__, str(exc)[:400])))
            continue

        def call(stack=stack, x=x, y=y, seen=seen):
            seen["loss"] = float(stack.train_step(x, y)["loss"])
        ms, why = timed(call)
        out.append((label, ms, why, {"loss": seen.get("loss")}))
    return out


def case_cnn_widths(row, arm_cases):
    """CNNClassifier.fit on synthetic (C, H, W) images at each conv_channels tuple the row names (the NI12 fill
    rule's neighbouring, non-board channel widths: x_cnn/device.mojo `ni12_implicit_layer`)."""
    import numpy as np
    ml = _ml()
    sh = row["shape"]
    n, k = int(sh["rows"]), int(sh["classes"])
    c, h, w = (int(v) for v in sh["input_shape"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, c * h * w)).astype(np.float32)
    y = (np.arange(n) % k).astype(np.int32)
    out = []
    for chans in sh["conv_channels"]:  # serial by nature: timed calls on one GPU
        chans = tuple(int(v) for v in chans)
        ms, why = timed(lambda: ml.CNNClassifier(input_shape=(c, h, w), conv_channels=chans, kernel_size=3,
                                                 pool_size=2, batch_size=int(sh["batch_size"]),
                                                 max_iter=int(sh["max_iter"]), random_state=0).fit(X, y))
        out.append(("CNNClassifier.fit conv_channels=%s n=%d" % (list(chans), n), ms, why))
    return out


def case_f16_widen(row, arm_cases):
    """`mojolearn.models.safetensors.widen_f16` (float16 bits -> float32 Array) at each element count the row
    names, on uint16 bits covering every half pattern in turn (A: the linalg from_f16 device round trip; B
    f16-widen-host: the safetensors reader's host-parallel widening). The counts are sizes, not any model's."""
    import numpy as np
    from mojolearn.models.safetensors import widen_f16
    out = []
    for n in row["shape"]["counts"]:  # serial by nature: timed calls on one host and GPU
        bits = (np.arange(int(n), dtype=np.uint64) % 65536).astype(np.uint16)
        ms, why = timed(lambda: widen_f16(bits))
        out.append(("widen_f16 n=%d" % int(n), ms, why))
    return out


def case_samba_step(row, arm_cases):
    """SambaStack.train_step and the forward that follows it, per case of the row's shape.cases. The stack draws
    its own weights (Generator(7)); ids are uniform bytes from default_rng(0). The shapes are sizes next to and
    away from the board's, never a dataset."""
    import numpy as np
    ml = _ml()
    from mojolearn.training import Generator
    out = []
    for c in row["shape"]["cases"]:  # serial by nature: timed calls on one GPU
        b, l, dm = int(c["batch"]), int(c["length"]), int(c["d_model"])
        layers = tuple(c["layers"])
        tag = "b=%d l=%d d_model=%d layers=%s" % (b, l, dm, "/".join(x[0] for x in layers))
        try:
            vocab = int(c.get("vocab", 256))
            cfg = ml.SambaConfig(vocab=vocab, d_model=dm, layers=layers, n_heads=c.get("n_heads"),
                                 intermediate=c.get("intermediate"))
            stack = ml.SambaStack(cfg, generator=Generator(7), numeric_mode="identical")
            ids = np.random.default_rng(0).integers(0, vocab, (b, l + 1)).astype(np.int32)
            x, y = np.ascontiguousarray(ids[:, :-1]), np.ascontiguousarray(ids[:, 1:])
        except Exception as exc:  # noqa: BLE001 -- a refusal is a result
            out.append(("SambaStack " + tag, None, "REFUSED: %s: %s" % (type(exc).__name__, str(exc)[:400])))
            continue
        ms, why = timed(lambda: stack.train_step(x, y))
        out.append(("SambaStack.train_step " + tag, ms, why))
        # the FIRST forward after a step is the case (whether the step left the device registry current): one
        # warm-up forward, one more untimed step, then the one timed forward
        try:
            stack.forward(x)
            stack.train_step(x, y)
            t0 = time.perf_counter()
            stack.forward(x)
            ms, why = (time.perf_counter() - t0) * 1000.0, None
        except Exception as exc:  # noqa: BLE001 -- a refusal is a result
            ms, why = None, "REFUSED: %s: %s" % (type(exc).__name__, str(exc)[:400])
        out.append(("SambaStack.forward first after a step " + tag, ms, why))
    return out


def case_gemm_shapes(row, arm_cases):
    """`mojolearn.linalg.matmul(a, b)` (the board gemm lane's call) at each (m, n, k) the row names: A is [m, k],
    B is [k, n], float32 standard normal from a seeded Generator. The shapes are sizes beside and away from the
    board's, never the board's own (no dimension targeting)."""
    import numpy as np
    from mojolearn import linalg
    out = []
    for case in row["shape"]["cases"]:  # serial by nature: timed calls on one GPU must not share it
        m, n, k = int(case["m"]), int(case["n"]), int(case["k"])
        rng = np.random.default_rng(0)
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        ms, why = timed(lambda: linalg.matmul(a, b))
        out.append(("linalg.matmul m=%d n=%d k=%d" % (m, n, k), ms, why))
    return out


DRIVERS = {"gemm_shapes": case_gemm_shapes, "f16_widen": case_f16_widen, "while_loop": case_while_loop, "pandas_input": case_pandas_input, "mlp_fit": case_mlp_fit,
           "kde_widths": case_kde_widths, "mamba_backward": case_mamba_backward, "cnn_widths": case_cnn_widths,
           "samba_train_step": case_samba_train_step,
           "samba_step": case_samba_step}


def arm_cases(row, arm):
    """The row's cases for one arm: 'candidate:check_every=8' -> '8'; an arm named alone has one case ''."""
    cases = []
    for spec in row.get("arms") or []:
        name, _, case = spec.partition(":")
        if name == arm:
            cases.append(case.split("=", 1)[-1] if case else "")
    return cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--owed", required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    with open(a.owed) as fh:
        doc = json.load(fh)
    row = next((r for r in doc["rows"] if r["id"] == a.row and r["kind"] == "driver"), None)
    if row is None:
        raise SystemExit("measure_owed_driver: no driver row %s" % a.row)
    cases = arm_cases(row, a.arm)
    if not cases:
        print("OWED-DRIVER row=%s arm=%s: not an arm of this row" % (a.row, a.arm))
        return 0
    ml = _ml()
    base = {"row": a.row, "arm": a.arm, "version": getattr(ml, "__version__", None), "numeric_mode": _numeric_mode(ml)}
    try:
        results = DRIVERS[row["driver"]](row, cases)
    except Exception as exc:  # noqa: BLE001 -- the driver's setup refused: one REFUSED line
        results = [(row["driver"], None, "REFUSED: %s: %s" % (type(exc).__name__, str(exc)[:400]))]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # one append of whole lines (2026-11-14: a line cut by a full disk crashed the owed report; the report now also
    # reads such a line as UNREADABLE)
    # a driver may return a fourth element: extra fields for its line (a quality value shown beside the time)
    body = "".join(json.dumps(dict(base, case=res[0], ms=res[1], status=res[2] or "ok", **(res[3] if len(res) > 3 else {})),
                              sort_keys=True) + "\n"
                   for res in results)
    with open(a.out, "a") as fh:
        fh.write(body)
        fh.flush()
        os.fsync(fh.fileno())
    for res in results:
        case, ms, why = res[0], res[1], res[2]
        print("OWED-DRIVER row=%s arm=%s case=%s ms=%s status=%s%s"
              % (a.row, a.arm, case, "%.3f" % ms if ms is not None else "-", why or "ok",
                 "".join(" %s=%r" % kv for kv in sorted((res[3] if len(res) > 3 else {}).items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
